use super::{
    Error, WeightSource,
    audio_transformer::{AttentionConfig, AudioTransformer, Gated},
    layers::{Causal, Conv, LayerNorm, Linear, RmsNorm, TransposeConv},
};
use burn::tensor::{Int, Tensor, activation, module};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct CodecDecoderConfig {
    #[serde(flatten)]
    pub attention: AttentionConfig,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub latent_dim: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
}
struct Quantizer {
    codebooks: Vec<Tensor<2>>,
    output: Conv<Causal>,
}
impl Quantizer {
    fn load(w: &impl WeightSource, p: &str, groups: usize) -> Result<Self, Error> {
        let codebooks = (0..groups)
            .map(|i| {
                let p = format!("{p}.vq.layers.{i}._codebook");
                let sum = w.tensor::<2>(&format!("{p}.embedding_sum"))?;
                // EuclideanCodebook uses a 1e-5 floor for unused centroids.
                let count = w
                    .tensor::<1>(&format!("{p}.cluster_usage"))?
                    .clamp_min(1e-5)
                    .unsqueeze_dim(1);
                Ok(sum / count)
            })
            .collect::<Result<_, Error>>()?;
        Ok(Self {
            codebooks,
            output: Conv::load(w, &format!("{p}.output_proj"), 1, 1, 1, Causal)?,
        })
    }
    fn decode(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        let [b, _, t] = codes.dims();
        let mut decoded = Vec::with_capacity(self.codebooks.len());
        for (i, book) in self.codebooks.iter().enumerate() {
            let ids = codes.clone().slice([0..b, i..i + 1, 0..t]).reshape([b, t]);
            decoded.push(module::embedding(book.clone(), ids));
        }
        // Qwen adds residual books in order, preserving FP16 rounding order.
        let mut decoded = decoded.into_iter();
        let mut sum = decoded.next().expect("validated nonempty quantizer");
        for x in decoded {
            sum = sum + x;
        }
        self.output.forward(sum.swap_dims(1, 2))
    }
}
struct Snake {
    alpha: Tensor<3>,
    inverse_beta: Tensor<3>,
}
impl Snake {
    fn load(w: &impl WeightSource, p: &str) -> Result<Self, Error> {
        let alpha = w.tensor::<1>(&format!("{p}.alpha"))?.exp();
        let channels = alpha.dims()[0];
        // SnakeBeta specifies a 1e-9 denominator stabilizer.
        let inverse_beta = (w.tensor::<1>(&format!("{p}.beta"))?.exp() + 1e-9).recip();
        Ok(Self {
            alpha: alpha.reshape([1, channels, 1]),
            inverse_beta: inverse_beta.reshape([1, channels, 1]),
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let periodic = (x.clone() * self.alpha.clone()).sin();
        x + periodic.clone() * periodic * self.inverse_beta.clone()
    }
}
struct Residual {
    act1: Snake,
    conv1: Conv<Causal>,
    act2: Snake,
    conv2: Conv<Causal>,
}
impl Residual {
    fn load(w: &impl WeightSource, p: &str, dilation: usize) -> Result<Self, Error> {
        Ok(Self {
            act1: Snake::load(w, &format!("{p}.act1"))?,
            conv1: Conv::load(w, &format!("{p}.conv1.conv"), 1, dilation, 1, Causal)?,
            act2: Snake::load(w, &format!("{p}.act2"))?,
            conv2: Conv::load(w, &format!("{p}.conv2.conv"), 1, 1, 1, Causal)?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        x.clone()
            + self
                .conv2
                .forward(self.act2.forward(self.conv1.forward(self.act1.forward(x))))
    }
}
struct Upsample {
    activation: Snake,
    conv: TransposeConv,
    residuals: Vec<Residual>,
}
impl Upsample {
    fn load(w: &impl WeightSource, p: &str, stride: usize) -> Result<Self, Error> {
        Ok(Self {
            activation: Snake::load(w, &format!("{p}.block.0"))?,
            conv: TransposeConv::load(w, &format!("{p}.block.1.conv"), stride)?,
            // Qwen's decoder block fixes its three residual dilations to 1, 3, 9.
            residuals: [1, 3, 9]
                .into_iter()
                .enumerate()
                .map(|(i, d)| Residual::load(w, &format!("{p}.block.{}", i + 2), d))
                .collect::<Result<_, _>>()?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mut x = self.conv.forward(self.activation.forward(x));
        for residual in &self.residuals {
            x = residual.forward(x);
        }
        x
    }
}
struct ConvNext {
    depthwise: Conv<Causal>,
    norm: LayerNorm,
    first: Linear,
    last: Linear,
    gamma: Tensor<1>,
}
impl ConvNext {
    fn load(w: &impl WeightSource, p: &str, channels: usize) -> Result<Self, Error> {
        Ok(Self {
            depthwise: Conv::load(w, &format!("{p}.dwconv.conv"), 1, 1, channels, Causal)?,
            // Qwen3TTSTokenizerV2ConvNeXtBlock fixes LayerNorm epsilon to 1e-6.
            norm: LayerNorm::load(w, &format!("{p}.norm"), 1e-6)?,
            first: Linear::load(w, &format!("{p}.pwconv1"))?,
            last: Linear::load(w, &format!("{p}.pwconv2"))?,
            gamma: w.tensor(&format!("{p}.gamma"))?,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = self
            .norm
            .forward(self.depthwise.forward(x.clone()).swap_dims(1, 2));
        let y = self.last.forward(activation::gelu(self.first.forward(y)))
            * self.gamma.clone().unsqueeze();
        x + y.swap_dims(1, 2)
    }
}
struct PreUpsample {
    conv: TransposeConv,
    block: ConvNext,
}
/// Causal codec reconstruction. Chunk boundaries and left context are owned by
/// the shared synthesis session; this component decodes the supplied code span.
pub struct CodecDecoder {
    semantic: Quantizer,
    acoustic: Quantizer,
    semantic_groups: usize,
    preconv: Conv<Causal>,
    input: Linear,
    transformer: AudioTransformer<RmsNorm, Gated>,
    norm: RmsNorm,
    output: Linear,
    preupsample: Vec<PreUpsample>,
    first: Conv<Causal>,
    upsample: Vec<Upsample>,
    last_activation: Snake,
    last: Conv<Causal>,
}
impl CodecDecoder {
    pub fn load(w: &impl WeightSource, c: &CodecDecoderConfig) -> Result<Self, Error> {
        if c.num_semantic_quantizers == 0 || c.num_quantizers <= c.num_semantic_quantizers {
            return Err(Error::Tensor {
                name: "codec quantizer".into(),
                reason: "expected nonempty semantic and acoustic groups".into(),
            });
        }
        let p = "decoder.pre_transformer";
        let end = c.upsample_rates.len() + 1;
        Ok(Self {
            semantic: Quantizer::load(w, "decoder.quantizer.rvq_first", c.num_semantic_quantizers)?,
            acoustic: Quantizer::load(
                w,
                "decoder.quantizer.rvq_rest",
                c.num_quantizers - c.num_semantic_quantizers,
            )?,
            semantic_groups: c.num_semantic_quantizers,
            preconv: Conv::load(w, "decoder.pre_conv.conv", 1, 1, 1, Causal)?,
            input: Linear::load(w, &format!("{p}.input_proj"))?,
            transformer: AudioTransformer::load(
                w,
                p,
                c.attention.clone(),
                c.rms_norm_eps,
                Some(c.sliding_window),
            )?,
            norm: RmsNorm::load(w, &format!("{p}.norm"), c.rms_norm_eps)?,
            output: Linear::load(w, &format!("{p}.output_proj"))?,
            preupsample: c
                .upsampling_ratios
                .iter()
                .enumerate()
                .map(|(i, &stride)| {
                    Ok(PreUpsample {
                        conv: TransposeConv::load(
                            w,
                            &format!("decoder.upsample.{i}.0.conv"),
                            stride,
                        )?,
                        block: ConvNext::load(w, &format!("decoder.upsample.{i}.1"), c.latent_dim)?,
                    })
                })
                .collect::<Result<_, Error>>()?,
            first: Conv::load(w, "decoder.decoder.0.conv", 1, 1, 1, Causal)?,
            upsample: c
                .upsample_rates
                .iter()
                .enumerate()
                .map(|(i, &stride)| {
                    Upsample::load(w, &format!("decoder.decoder.{}", i + 1), stride)
                })
                .collect::<Result<_, _>>()?,
            last_activation: Snake::load(w, &format!("decoder.decoder.{end}"))?,
            last: Conv::load(
                w,
                &format!("decoder.decoder.{}.conv", end + 1),
                1,
                1,
                1,
                Causal,
            )?,
        })
    }
    pub fn forward(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        let [b, g, t] = codes.dims();
        let x = self
            .semantic
            .decode(codes.clone().slice([0..b, 0..self.semantic_groups, 0..t]))
            + self
                .acoustic
                .decode(codes.slice([0..b, self.semantic_groups..g, 0..t]));
        let x = self.input.forward(self.preconv.forward(x).swap_dims(1, 2));
        let mut x = self
            .output
            .forward(self.norm.forward(self.transformer.forward(x)))
            .swap_dims(1, 2);
        for stage in &self.preupsample {
            x = stage.block.forward(stage.conv.forward(x));
        }
        x = self.first.forward(x);
        for stage in &self.upsample {
            x = stage.forward(x);
        }
        self.last
            .forward(self.last_activation.forward(x))
            .clamp(-1.0, 1.0)
    }
}
