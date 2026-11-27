use super::{
    Error, WeightSource,
    audio_transformer::{AttentionConfig, AudioHistory, AudioTransformer, Gated},
    layers::{Causal, Conv, ConvHistory, LayerNorm, Linear, RmsNorm, TransposeConv},
};
use burn::tensor::{DType, Int, Tensor, activation, module};
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
    fn load(w: &mut impl WeightSource, p: &str, groups: usize) -> Result<Self, Error> {
        let codebooks = (0..groups)
            .map(|i| {
                let p = format!("{p}.vq.layers.{i}._codebook");
                // EuclideanCodebook uses a 1e-5 floor for unused centroids.
                w.normalized_codebook(
                    &format!("{p}.embedding_sum"),
                    &format!("{p}.cluster_usage"),
                    1e-5,
                )
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
    alpha: Tensor<1>,
    inverse_beta: Tensor<1>,
}
impl Snake {
    fn load(w: &mut impl WeightSource, p: &str) -> Result<Self, Error> {
        let raw_alpha = w.tensor::<1>(&format!("{p}.alpha"))?;
        let dtype = raw_alpha.dtype();
        let alpha = raw_alpha.cast(DType::F32).exp().cast(dtype);
        // SnakeBeta specifies a 1e-9 denominator stabilizer.
        let inverse_beta = (w.tensor::<1>(&format!("{p}.beta"))?.cast(DType::F32).exp() + 1e-9)
            .recip()
            .cast(dtype);
        Ok(Self {
            alpha,
            inverse_beta,
        })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        super::conv::snake(x, self.alpha.clone(), self.inverse_beta.clone())
    }
}
struct Residual {
    act1: Snake,
    conv1: Conv<Causal>,
    act2: Snake,
    conv2: Conv<Causal>,
}
impl Residual {
    fn load(w: &mut impl WeightSource, p: &str, dilation: usize) -> Result<Self, Error> {
        Ok(Self {
            act1: Snake::load(w, &format!("{p}.act1"))?,
            conv1: Conv::load(w, &format!("{p}.conv1.conv"), 1, dilation, 1, Causal)?,
            act2: Snake::load(w, &format!("{p}.act2"))?,
            conv2: Conv::load(w, &format!("{p}.conv2.conv"), 1, 1, 1, Causal)?,
        })
    }
    fn stream(&self, x: Tensor<3>, history: &mut ResidualHistory) -> Tensor<3> {
        let y = self
            .conv1
            .stream(self.act1.forward(x.clone()), &mut history.first);
        self.conv2
            .stream_add(self.act2.forward(y), &mut history.last, x)
    }
}
#[derive(Clone, Default)]
struct ResidualHistory {
    first: ConvHistory,
    last: ConvHistory,
}
struct Upsample {
    activation: Snake,
    conv: TransposeConv,
    residuals: Vec<Residual>,
}
impl Upsample {
    fn load(w: &mut impl WeightSource, p: &str, stride: usize) -> Result<Self, Error> {
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
    fn history(&self) -> UpsampleHistory {
        UpsampleHistory {
            conv: ConvHistory::default(),
            residuals: self
                .residuals
                .iter()
                .map(|_| ResidualHistory::default())
                .collect(),
        }
    }
    fn stream(&self, x: Tensor<3>, history: &mut UpsampleHistory) -> Tensor<3> {
        let mut x = self
            .conv
            .stream(self.activation.forward(x), &mut history.conv);
        for (residual, past) in self.residuals.iter().zip(&mut history.residuals) {
            x = residual.stream(x, past);
        }
        x
    }
}
#[derive(Clone)]
struct UpsampleHistory {
    conv: ConvHistory,
    residuals: Vec<ResidualHistory>,
}
struct ConvNext {
    depthwise: Conv<Causal>,
    norm: LayerNorm,
    first: Linear,
    last: Linear,
    gamma: Tensor<1>,
}
impl ConvNext {
    fn load(w: &mut impl WeightSource, p: &str, channels: usize) -> Result<Self, Error> {
        Ok(Self {
            depthwise: Conv::load(w, &format!("{p}.dwconv.conv"), 1, 1, channels, Causal)?,
            // Qwen3TTSTokenizerV2ConvNeXtBlock fixes LayerNorm epsilon to 1e-6.
            norm: LayerNorm::load(w, &format!("{p}.norm"), 1e-6)?,
            first: Linear::load(w, &format!("{p}.pwconv1"))?,
            last: Linear::load(w, &format!("{p}.pwconv2"))?,
            gamma: w.tensor(&format!("{p}.gamma"))?,
        })
    }
    fn stream(&self, x: Tensor<3>, history: &mut ConvHistory) -> Tensor<3> {
        let y = self
            .norm
            .forward(self.depthwise.stream(x.clone(), history).swap_dims(1, 2));
        let y = self.last.forward(activation::gelu(self.first.forward(y)))
            * self.gamma.clone().unsqueeze();
        x + y.swap_dims(1, 2)
    }
}
struct PreUpsample {
    conv: TransposeConv,
    block: ConvNext,
}
#[derive(Clone, Default)]
struct PreUpsampleHistory {
    conv: ConvHistory,
    block: ConvHistory,
}
#[derive(Clone)]
struct CodecHistory {
    preconv: ConvHistory,
    transformer: AudioHistory,
    preupsample: Vec<PreUpsampleHistory>,
    first: ConvHistory,
    upsample: Vec<UpsampleHistory>,
    last: ConvHistory,
}
/// The supplied prefix identifies the block, including its official left context.
#[derive(Clone)]
pub(crate) struct CodecCache {
    packed: Vec<i32>,
    frames: usize,
    samples: Vec<f32>,
    history: CodecHistory,
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
    pub fn load(w: &mut impl WeightSource, c: &CodecDecoderConfig) -> Result<Self, Error> {
        if c.num_semantic_quantizers == 0 || c.num_quantizers <= c.num_semantic_quantizers {
            return Err(Error::Tensor {
                name: "codec quantizer".into(),
                reason: "expected nonempty semantic and acoustic groups".into(),
            });
        }
        let p = "decoder.pre_transformer";
        let end = c.upsample_rates.len() + 1;
        let semantic =
            Quantizer::load(w, "decoder.quantizer.rvq_first", c.num_semantic_quantizers)?;
        let acoustic = Quantizer::load(
            w,
            "decoder.quantizer.rvq_rest",
            c.num_quantizers - c.num_semantic_quantizers,
        )?;
        w.device()
            .flush()
            .map_err(|error| Error::Execution(error.to_string()))?;
        let preconv = Conv::load(w, "decoder.pre_conv.conv", 1, 1, 1, Causal)?;
        let input = Linear::load(w, &format!("{p}.input_proj"))?;
        let transformer = AudioTransformer::load(
            w,
            p,
            c.attention.clone(),
            c.rms_norm_eps,
            Some(c.sliding_window),
        )?;
        let norm = RmsNorm::load(w, &format!("{p}.norm"), c.rms_norm_eps)?;
        let output = Linear::load(w, &format!("{p}.output_proj"))?;
        let mut preupsample = Vec::with_capacity(c.upsampling_ratios.len());
        for (i, &stride) in c.upsampling_ratios.iter().enumerate() {
            preupsample.push(PreUpsample {
                conv: TransposeConv::load(w, &format!("decoder.upsample.{i}.0.conv"), stride)?,
                block: ConvNext::load(w, &format!("decoder.upsample.{i}.1"), c.latent_dim)?,
            });
            w.device()
                .flush()
                .map_err(|error| Error::Execution(error.to_string()))?;
        }
        let first = Conv::load(w, "decoder.decoder.0.conv", 1, 1, 1, Causal)?;
        let mut upsample = Vec::with_capacity(c.upsample_rates.len());
        for (i, &stride) in c.upsample_rates.iter().enumerate() {
            upsample.push(Upsample::load(
                w,
                &format!("decoder.decoder.{}", i + 1),
                stride,
            )?);
            w.device()
                .flush()
                .map_err(|error| Error::Execution(error.to_string()))?;
        }
        let last_activation = Snake::load(w, &format!("decoder.decoder.{end}"))?;
        let last = Conv::load(
            w,
            &format!("decoder.decoder.{}.conv", end + 1),
            1,
            1,
            1,
            Causal,
        )?;
        Ok(Self {
            semantic,
            acoustic,
            semantic_groups: c.num_semantic_quantizers,
            preconv,
            input,
            transformer,
            norm,
            output,
            preupsample,
            first,
            upsample,
            last_activation,
            last,
        })
    }
    pub(crate) fn cache(&self) -> CodecCache {
        CodecCache {
            packed: Vec::new(),
            frames: 0,
            samples: Vec::new(),
            history: self.history(),
        }
    }
    pub(crate) async fn prefix(
        &self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
        cache: &mut CodecCache,
        device: &burn::tensor::Device,
    ) -> Result<Vec<f32>, Error> {
        let extends = frames >= cache.frames
            && (0..groups).all(|group| {
                packed[group * frames..group * frames + cache.frames]
                    == cache.packed[group * cache.frames..(group + 1) * cache.frames]
            });
        if !extends {
            *cache = self.cache();
        }
        let mut offset = cache.frames;
        if offset < frames {
            // Bound causal decode working memory to one streaming chunk (12 frames) so
            // priming a long reference clip reuses the same small activation buffers as
            // steady-state streaming synthesis instead of allocating full-clip buffers.
            const MAX_STREAM_CHUNK_FRAMES: usize = 12;
            while offset < frames {
                let end = (offset + MAX_STREAM_CHUNK_FRAMES).min(frames);
                let chunk_frames = end - offset;
                let suffix: Vec<_> = (0..groups)
                    .flat_map(|group| {
                        packed[group * frames + offset..group * frames + end]
                            .iter()
                            .copied()
                    })
                    .collect();
                let codes = Tensor::<3, Int>::from_data(
                    burn::tensor::TensorData::new(suffix, [1, groups, chunk_frames]),
                    device,
                );
                let samples = match super::read(self.stream(codes, &mut cache.history)).await {
                    Ok(samples) => samples,
                    Err(error) => {
                        *cache = self.cache();
                        return Err(error);
                    }
                };
                cache.samples.extend(samples);
                offset = end;
            }
            cache.frames = frames;
            cache.packed = packed;
        }
        Ok(cache.samples.clone())
    }
    fn history(&self) -> CodecHistory {
        CodecHistory {
            preconv: ConvHistory::default(),
            transformer: self.transformer.history(),
            preupsample: self
                .preupsample
                .iter()
                .map(|_| PreUpsampleHistory::default())
                .collect(),
            first: ConvHistory::default(),
            upsample: self.upsample.iter().map(Upsample::history).collect(),
            last: ConvHistory::default(),
        }
    }
    pub fn forward(&self, codes: Tensor<3, Int>) -> Tensor<3> {
        self.stream(codes, &mut self.history())
    }
    fn stream(&self, codes: Tensor<3, Int>, history: &mut CodecHistory) -> Tensor<3> {
        let [b, g, t] = codes.dims();
        let x = self
            .semantic
            .decode(codes.clone().slice([0..b, 0..self.semantic_groups, 0..t]))
            + self
                .acoustic
                .decode(codes.slice([0..b, self.semantic_groups..g, 0..t]));
        let x = self
            .input
            .forward(self.preconv.stream(x, &mut history.preconv).swap_dims(1, 2));
        let mut x = self
            .output
            .forward(
                self.norm
                    .forward(self.transformer.stream(x, &mut history.transformer)),
            )
            .swap_dims(1, 2);
        for (stage, past) in self.preupsample.iter().zip(&mut history.preupsample) {
            x = stage
                .block
                .stream(stage.conv.stream(x, &mut past.conv), &mut past.block);
        }
        x = self.first.stream(x, &mut history.first);
        for (stage, past) in self.upsample.iter().zip(&mut history.upsample) {
            x = stage.stream(x, past);
        }
        self.last
            .stream(self.last_activation.forward(x), &mut history.last)
            .clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::{Weights, read};
    use burn::tensor::{DType, Device};
    use std::path::PathBuf;

    fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: shape");
        assert!(
            actual.iter().chain(expected).all(|v| v.is_finite()),
            "{label}: nonfinite samples"
        );
        let error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(error < 1e-3, "{label}: maximum error {error}");
    }

    #[test]
    #[ignore = "requires real codec fixtures, pinned official weights and a GPU adapter"]
    fn cached_codec_prefixes_match_full_decoding_and_official_waveforms() {
        futures_lite::future::block_on(async {
            let source = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_SOURCE").unwrap());
            let fixture = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_FIXTURES").unwrap());
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(source.join("speech_tokenizer/config.json")).unwrap(),
            )
            .unwrap();
            let config: CodecDecoderConfig =
                serde_json::from_value(config["decoder_config"].clone()).unwrap();
            let device = Device::wgpu_options().init_async().await.unwrap();
            let bytes = std::fs::read(source.join("speech_tokenizer/model.safetensors")).unwrap();
            let mut weights = Weights::new(&bytes, device.clone(), DType::F32).unwrap();
            let decoder = CodecDecoder::load(&mut weights, &config).unwrap();
            let bytes = std::fs::read(fixture.join("audio-fixtures.safetensors")).unwrap();
            let data = Weights::new(&bytes, device.clone(), DType::F32).unwrap();
            let codes = data.indices::<3>("codes.80").unwrap();
            let [_, groups, frames] = codes.dims();
            let packed = codes
                .into_data_async()
                .await
                .unwrap()
                .try_into_vec::<i32>()
                .unwrap();
            let mut cache = decoder.cache();
            // Uneven spans cross the 72-frame sliding attention window; repeated
            // and shorter prefixes exercise reuse and block/session invalidation.
            for stop in [1, 8, 20, 51, 80, 80, 8, 80] {
                let prefix: Vec<_> = (0..groups)
                    .flat_map(|g| packed[g * frames..g * frames + stop].iter().copied())
                    .collect();
                let full = read(decoder.forward(Tensor::<3, Int>::from_data(
                    burn::tensor::TensorData::new(prefix.clone(), [1, groups, stop]),
                    &device,
                )))
                .await
                .unwrap();
                let actual = decoder
                    .prefix(prefix, groups, stop, &mut cache, &device)
                    .await
                    .unwrap();
                assert_close(&actual, &full, &format!("{stop}-frame cached/full"));
                if stop == 80 {
                    let expected = read(data.tensor::<3>("waveform.80").unwrap())
                        .await
                        .unwrap();
                    assert_close(&actual, &expected, "cached/official");
                }
            }
            // A different real-code span with the same shape cannot reuse state.
            let mut cache = decoder.cache();
            for offset in [0, 1] {
                let prefix: Vec<_> = (0..groups)
                    .flat_map(|g| {
                        packed[g * frames + offset..g * frames + offset + 8]
                            .iter()
                            .copied()
                    })
                    .collect();
                let full = read(decoder.forward(Tensor::<3, Int>::from_data(
                    burn::tensor::TensorData::new(prefix.clone(), [1, groups, 8]),
                    &device,
                )))
                .await
                .unwrap();
                let actual = decoder
                    .prefix(prefix, groups, 8, &mut cache, &device)
                    .await
                    .unwrap();
                assert_close(&actual, &full, "changed-prefix");
            }
            let prefix: Vec<_> = (0..groups)
                .flat_map(|g| packed[g * frames..g * frames + 8].iter().copied())
                .collect();
            let mut checkpoint = decoder.cache();
            decoder
                .prefix(prefix, groups, 8, &mut checkpoint, &device)
                .await
                .unwrap();
            // Independent utterances share the reference tensors but never mutate
            // their checkpoint, even when the generated suffix differs.
            for offset in [0, 1] {
                let prefix: Vec<_> = (0..groups)
                    .flat_map(|g| {
                        (0..frames).map({
                            let packed = &packed;
                            move |f| {
                                packed[g * frames + if f < 8 { f } else { (f + offset) % frames }]
                            }
                        })
                    })
                    .collect();
                let full = read(decoder.forward(Tensor::<3, Int>::from_data(
                    burn::tensor::TensorData::new(prefix.clone(), [1, groups, frames]),
                    &device,
                )))
                .await
                .unwrap();
                let mut branch = checkpoint.clone();
                let actual = decoder
                    .prefix(prefix, groups, frames, &mut branch, &device)
                    .await
                    .unwrap();
                assert_close(&actual, &full, "reference-checkpoint branch");
                assert_eq!(checkpoint.frames, 8);
            }
        });
    }

    #[test]
    #[ignore = "requires real codec fixtures, pinned official weights and a GPU adapter"]
    fn fp16_codec_matches_official_waveforms() {
        futures_lite::future::block_on(async {
            let source = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_SOURCE").unwrap());
            let fixture = PathBuf::from(std::env::var_os("DEEPCLONE_BURN_FIXTURES").unwrap());
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(source.join("speech_tokenizer/config.json")).unwrap(),
            )
            .unwrap();
            let config: CodecDecoderConfig =
                serde_json::from_value(config["decoder_config"].clone()).unwrap();
            let device = Device::wgpu_options().init_async().await.unwrap();
            let bytes = std::fs::read(source.join("speech_tokenizer/model.safetensors")).unwrap();
            let mut weights = Weights::new(&bytes, device.clone(), DType::F16).unwrap();
            let decoder = CodecDecoder::load(&mut weights, &config).unwrap();
            let bytes = std::fs::read(fixture.join("audio-fixtures.safetensors")).unwrap();
            let data = Weights::new(&bytes, device.clone(), DType::F32).unwrap();
            let codes = data.indices::<3>("codes.80").unwrap();
            let [_, groups, frames] = codes.dims();
            let packed = codes
                .into_data_async()
                .await
                .unwrap()
                .try_into_vec::<i32>()
                .unwrap();
            let mut cache = decoder.cache();
            for stop in [12, 24, 60, 80] {
                let prefix: Vec<_> = (0..groups)
                    .flat_map(|g| packed[g * frames..g * frames + stop].iter().copied())
                    .collect();
                let _ = decoder
                    .prefix(prefix, groups, stop, &mut cache, &device)
                    .await
                    .unwrap();
            }
            let expected = read(data.tensor::<3>("waveform.80").unwrap())
                .await
                .unwrap();
            let actual = &cache.samples;
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual.iter().all(|v| v.is_finite()),
                "nonfinite FP16 samples"
            );
            let diff_sq: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| f64::from(a - b).powi(2))
                .sum();
            let ref_sq: f64 = expected.iter().map(|&b| f64::from(b).powi(2)).sum();
            let rel_l2 = (diff_sq / ref_sq).sqrt();
            let max_abs = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "FP16 CodecDecoder vs FP32 official (80 frames): rel_l2={rel_l2:.5}, max_abs={max_abs:.5}"
            );
            assert!(
                rel_l2 < 0.02,
                "FP16 CodecDecoder relative L2 error too high: {rel_l2}"
            );
        });
    }
}
