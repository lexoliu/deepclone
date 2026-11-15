use super::{
    Error, Weights,
    audio_transformer::{AttentionConfig, AudioTransformer, Gelu},
    layers::{Causal, CausalEdge, Conv, LayerNorm},
};
use burn::tensor::{Int, Tensor, activation, module};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct EncoderConfig {
    #[serde(flatten)]
    pub attention: AttentionConfig,
    pub norm_eps: f32,
    pub upsampling_ratios: Vec<usize>,
    pub num_residual_layers: usize,
    pub dilation_growth_rate: usize,
    pub use_conv_shortcut: bool,
    pub use_causal_conv: bool,
    pub pad_mode: String,
    pub num_semantic_quantizers: usize,
}
struct Residual {
    first: Conv<Causal>,
    last: Conv<Causal>,
    shortcut: Option<Conv<Causal>>,
}
impl Residual {
    fn load(w: &Weights<'_>, p: &str, dilation: usize, shortcut: bool) -> Result<Self, Error> {
        Ok(Self {
            first: Conv::load(w, &format!("{p}.block.1.conv"), 1, dilation, 1, Causal)?,
            last: Conv::load(w, &format!("{p}.block.3.conv"), 1, 1, 1, Causal)?,
            shortcut: if shortcut {
                Some(Conv::load(
                    w,
                    &format!("{p}.shortcut.conv"),
                    1,
                    1,
                    1,
                    Causal,
                )?)
            } else {
                None
            },
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = self.last.forward(activation::elu(
            self.first.forward(activation::elu(x.clone(), 1.0)),
            1.0,
        ));
        y + match &self.shortcut {
            Some(conv) => conv.forward(x),
            None => x,
        }
    }
}
struct Stage {
    residuals: Vec<Residual>,
    down: Conv<Causal>,
}
struct Quantizer {
    projection: Conv<Causal>,
    codebooks: Vec<Tensor<2>>,
}
impl Quantizer {
    fn load(w: &Weights<'_>, p: &str, groups: usize) -> Result<Self, Error> {
        let codebooks = (0..groups)
            .map(|i| {
                let p = format!("{p}.layers.{i}.codebook");
                let sum = w.tensor::<2>(&format!("{p}.embed_sum"))?;
                // MimiEuclideanCodebook normalizes centroids with epsilon 1e-5.
                let count = w
                    .tensor::<1>(&format!("{p}.cluster_usage"))?
                    .clamp_min(1e-5)
                    .unsqueeze_dim(1);
                Ok(sum / count)
            })
            .collect::<Result<_, Error>>()?;
        Ok(Self {
            projection: Conv::load(w, &format!("{p}.input_proj"), 1, 1, 1, Causal)?,
            codebooks,
        })
    }
    fn encode(&self, x: Tensor<3>) -> Vec<Tensor<3, Int>> {
        let mut residual = self.projection.forward(x).swap_dims(1, 2);
        let [b, t, d] = residual.dims();
        let mut codes = Vec::with_capacity(self.codebooks.len());
        for book in &self.codebooks {
            let flat = residual.clone().reshape([b * t, d]);
            let xnorm = (flat.clone() * flat.clone()).sum_dim(1);
            let wnorm = (book.clone() * book.clone()).sum_dim(1).transpose();
            let distances = (xnorm + wnorm - flat.matmul(book.clone().transpose()) * 2.0)
                .clamp_min(0.0)
                .sqrt();
            let indices = distances.argmin(1).reshape([b, t]);
            residual = residual - module::embedding(book.clone(), indices.clone());
            codes.push(indices.unsqueeze_dim(1));
        }
        codes
    }
}
/// FP32 Mimi encoder and split residual vector quantization.
pub struct CodecEncoder {
    first: Conv<Causal>,
    stages: Vec<Stage>,
    last: Conv<Causal>,
    transformer: AudioTransformer<LayerNorm, Gelu>,
    downsample: Conv<CausalEdge>,
    semantic: Quantizer,
    acoustic: Quantizer,
}
impl CodecEncoder {
    pub fn load(
        w: &Weights<'_>,
        c: &EncoderConfig,
        groups: usize,
        downsample_rate: usize,
    ) -> Result<Self, Error> {
        let convolution_rate = c.upsampling_ratios.iter().product::<usize>();
        if !c.use_causal_conv
            || c.pad_mode != "constant"
            || convolution_rate == 0
            || !downsample_rate.is_multiple_of(convolution_rate)
            || groups <= c.num_semantic_quantizers
        {
            return Err(Error::Tensor {
                name: "audio encoder".into(),
                reason: "unsupported causal codec configuration".into(),
            });
        }
        let mut index = 1;
        let mut stages = Vec::new();
        for &ratio in c.upsampling_ratios.iter().rev() {
            let mut residuals = Vec::new();
            for j in 0..c.num_residual_layers {
                residuals.push(Residual::load(
                    w,
                    &format!("encoder.encoder.layers.{index}"),
                    c.dilation_growth_rate.pow(j as u32),
                    c.use_conv_shortcut,
                )?);
                index += 1;
            }
            index += 1; // ELU has no stored parameters.
            let down = Conv::load(
                w,
                &format!("encoder.encoder.layers.{index}.conv"),
                ratio,
                1,
                1,
                Causal,
            )?;
            index += 1;
            stages.push(Stage { residuals, down });
        }
        index += 1;
        Ok(Self {
            first: Conv::load(w, "encoder.encoder.layers.0.conv", 1, 1, 1, Causal)?,
            stages,
            last: Conv::load(
                w,
                &format!("encoder.encoder.layers.{index}.conv"),
                1,
                1,
                1,
                Causal,
            )?,
            transformer: AudioTransformer::load(
                w,
                "encoder.encoder_transformer",
                c.attention.clone(),
                c.norm_eps,
                None,
            )?,
            downsample: Conv::load(
                w,
                "encoder.downsample.conv",
                downsample_rate / convolution_rate,
                1,
                1,
                CausalEdge,
            )?,
            semantic: Quantizer::load(
                w,
                "encoder.quantizer.semantic_residual_vector_quantizer",
                c.num_semantic_quantizers,
            )?,
            acoustic: Quantizer::load(
                w,
                "encoder.quantizer.acoustic_residual_vector_quantizer",
                groups - c.num_semantic_quantizers,
            )?,
        })
    }
    fn features(&self, audio: Tensor<3>) -> Tensor<3> {
        let mut x = self.first.forward(audio);
        for stage in &self.stages {
            for residual in &stage.residuals {
                x = residual.forward(x);
            }
            x = stage.down.forward(activation::elu(x, 1.0));
        }
        x = self.last.forward(activation::elu(x, 1.0));
        x = self.transformer.forward(x.swap_dims(1, 2)).swap_dims(1, 2);
        self.downsample.forward(x)
    }
    pub fn forward(&self, audio: Tensor<3>) -> Tensor<3, Int> {
        let x = self.features(audio);
        let mut codes = self.semantic.encode(x.clone());
        codes.extend(self.acoustic.encode(x));
        Tensor::cat(codes, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device};
    #[test]
    #[ignore = "requires pinned audio fixtures and model weights"]
    fn encoder_features_match_reference() {
        futures_lite::future::block_on(async {
            let source =
                std::path::PathBuf::from(std::env::var_os("DEEPCLONE_BURN_SOURCE").unwrap());
            let fixture =
                std::path::PathBuf::from(std::env::var_os("DEEPCLONE_BURN_FIXTURES").unwrap());
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(source.join("speech_tokenizer/config.json")).unwrap(),
            )
            .unwrap();
            let device = Device::wgpu_options().init_async().await.unwrap();
            let bytes = std::fs::read(source.join("speech_tokenizer/model.safetensors")).unwrap();
            let w = Weights::new(&bytes, device.clone(), DType::F32).unwrap();
            let bytes = std::fs::read(fixture.join("audio-fixtures.safetensors")).unwrap();
            let data = Weights::new(&bytes, device, DType::F32).unwrap();
            let c = serde_json::from_value(config["encoder_config"].clone()).unwrap();
            let encoder = CodecEncoder::load(
                &w,
                &c,
                config["encoder_valid_num_quantizers"].as_u64().unwrap() as usize,
                config["encode_downsample_rate"].as_u64().unwrap() as usize,
            )
            .unwrap();
            let actual = super::super::read(encoder.features(data.tensor("audio").unwrap()))
                .await
                .unwrap();
            let expected = super::super::read(data.tensor::<3>("encoded").unwrap())
                .await
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            let (index, error) = actual
                .iter()
                .zip(expected)
                .enumerate()
                .map(|(i, (a, b))| (i, (a - b).abs()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            assert!(error < 1e-3, "encoded feature max error {error} at {index}");
        });
    }
}
