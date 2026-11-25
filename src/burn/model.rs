//! Complete Qwen model operations for the shared cloning session.
use super::{
    CodecDecoder, CodecDecoderConfig, CodecEncoder, Decoder, DecoderConfig, EncoderConfig, Speaker,
    SpeakerConfig, WeightSource, layers::Linear,
};
use crate::{
    generation::{Codec, Model},
    inference::Error,
};
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        cubecl::dtype_to_storage_type,
        ops::FloatTensorOps,
        tensor::{FloatTensor, IntTensor},
        wgpu::CubeBackend,
    },
    cubecl::{ir::ElemType, prelude::*},
    tensor::{DType, Device, Int, Shape, Tensor as BurnTensor, TensorData, activation},
};
use serde::Deserialize;
use std::num::NonZeroUsize;

const EMBED_LANES: usize = 256;

#[backend_extension(Cube, Fusion)]
trait FrameEmbeddings: Backend {
    #[fusion(dtype = table, shape = Shape::from([1, 1, table[1]]))]
    fn embed_device_token(
        table: FloatTensor<Self>,
        token: IntTensor<Self>,
        row_offset: usize,
    ) -> FloatTensor<Self>;

    #[fusion(dtype = text, shape = Shape::from([1, 1, table[1]]))]
    fn sum_frame_embeddings(
        table: FloatTensor<Self>,
        rows: IntTensor<Self>,
        text: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

impl FrameEmbeddings for CubeBackend {
    fn embed_device_token(
        table: FloatTensor<Self>,
        token: IntTensor<Self>,
        row_offset: usize,
    ) -> FloatTensor<Self> {
        let width = table.shape()[1];
        let output = Self::float_empty(
            Shape::from([1, 1, width]),
            &table.device,
            table.dtype.into(),
        );
        embed_device_token_kernel::launch(
            &table.client,
            CubeCount::Static(width.div_ceil(EMBED_LANES).max(1) as u32, 1, 1),
            CubeDim::new_1d(EMBED_LANES as u32),
            table.clone().into_tensor_arg(),
            token.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            row_offset,
            dtype_to_storage_type(table.dtype),
        );
        output
    }

    fn sum_frame_embeddings(
        table: FloatTensor<Self>,
        rows: IntTensor<Self>,
        text: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let width = table.shape()[1];
        let output = Self::float_empty(Shape::from([1, 1, width]), &text.device, text.dtype.into());
        sum_frame_embeddings_kernel::launch(
            &table.client,
            CubeCount::Static(width.div_ceil(EMBED_LANES).max(1) as u32, 1, 1),
            CubeDim::new_1d(EMBED_LANES as u32),
            table.clone().into_tensor_arg(),
            rows.into_tensor_arg(),
            text.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dtype_to_storage_type(table.dtype),
        );
        output
    }
}

#[cube(launch)]
fn embed_device_token_kernel<F: Float>(
    table: &Tensor<F>,
    token: &Tensor<i32>,
    output: &mut Tensor<F>,
    row_offset: usize,
    #[define(F)] _dtype: ElemType,
) {
    let col = ABSOLUTE_POS;
    if col >= table.shape(1) {
        terminate!();
    }
    // Failed selections carry negative status codes; clamp to row 0 until read_tokens surfaces them.
    let raw = token[0];
    let index = if raw < 0i32 { 0usize } else { raw as usize };
    let row = row_offset + index;
    output[col] = table[row * table.stride(0) + col * table.stride(1)];
}

#[cube(launch)]
fn sum_frame_embeddings_kernel<F: Float>(
    table: &Tensor<F>,
    rows: &Tensor<i32>,
    text: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: ElemType,
) {
    let col = ABSOLUTE_POS;
    if col >= table.shape(1) {
        terminate!();
    }
    let count = rows.shape(0);
    let t_s0 = table.stride(0);
    let t_s1 = table.stride(1);
    let r_s0 = rows.stride(0);
    let mut acc = table[(rows[0] as usize) * t_s0 + col * t_s1];
    for g in 1..count {
        let val = table[(rows[g * r_s0] as usize) * t_s0 + col * t_s1];
        acc = F::cast_from(f32::cast_from(acc) + f32::cast_from(val));
    }
    output[col] = F::cast_from(f32::cast_from(acc) + f32::cast_from(text[col * text.stride(2)]));
}

#[cfg(all(feature = "browser", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = debug)]
    fn codec_timing(marker: &str, frames: usize, elapsed_ms: f64);
}

#[derive(Deserialize)]
pub struct SpeechConfig {
    pub talker_config: TalkerConfig,
    pub speaker_encoder_config: SpeakerConfig,
}
#[derive(Deserialize)]
pub struct TalkerConfig {
    #[serde(flatten)]
    pub decoder: DecoderConfig,
    pub code_predictor_config: DecoderConfig,
    pub num_code_groups: usize,
    pub hidden_act: String,
}
#[derive(Deserialize)]
pub struct AudioConfig {
    pub encoder_config: EncoderConfig,
    pub decoder_config: CodecDecoderConfig,
    pub encoder_valid_num_quantizers: usize,
    pub encode_downsample_rate: usize,
    pub decode_upsample_rate: usize,
    pub input_sample_rate: u32,
    pub output_sample_rate: u32,
}

pub struct Burn {
    talker: Decoder,
    predictor: Decoder,
    text: BurnTensor<2>,
    text_first: Linear,
    text_last: Linear,
    code_table: BurnTensor<2>,
    code_offsets: Vec<usize>,
    code_vocabs: Vec<usize>,
    talker_head: Linear,
    predictor_heads: Vec<Linear>,
    projection: Option<Linear>,
    speaker: Speaker,
    encoder: CodecEncoder,
    device: Device,
    dtype: DType,
}
pub struct BurnCodec {
    decoder: CodecDecoder,
    decoder_cache: super::codec_decoder::CodecCache,
    reference_decoder_cache: Option<super::codec_decoder::CodecCache>,
    device: Device,
}
impl Burn {
    pub fn load(
        speech: &impl WeightSource,
        conditioning: &impl WeightSource,
        audio: &impl WeightSource,
        config: SpeechConfig,
        codec: AudioConfig,
        capacity: NonZeroUsize,
    ) -> Result<(Self, BurnCodec), Error> {
        let c = config.talker_config;
        if c.hidden_act != "silu" {
            return Err(Error::Contract("unsupported text projection activation"));
        }
        let mut codes: Vec<BurnTensor<2>> =
            vec![speech.tensor("talker.model.codec_embedding.weight")?];
        for group in 0..c.num_code_groups - 1 {
            codes.push(speech.tensor(&format!(
                "talker.code_predictor.model.codec_embedding.{group}.weight"
            ))?);
        }
        let width = codes[0].dims()[1];
        let mut code_offsets = Vec::with_capacity(codes.len());
        let mut code_vocabs = Vec::with_capacity(codes.len());
        let mut offset = 0usize;
        for table in &codes {
            let [vocab, w] = table.dims();
            if w != width {
                return Err(Error::Contract("mismatched codec embedding widths"));
            }
            code_offsets.push(offset);
            code_vocabs.push(vocab);
            offset += vocab;
        }
        let code_table = BurnTensor::cat(codes, 0);
        let projection = if c.decoder.hidden_size != c.code_predictor_config.hidden_size {
            Some(Linear::load(
                speech,
                "talker.code_predictor.small_to_mtp_projection",
            )?)
        } else {
            None
        };
        let decoder = CodecDecoder::load(audio, &codec.decoder_config)?;
        let decoder_cache = decoder.cache();
        Ok((
            Self {
                talker: Decoder::load(speech, "talker.model", c.decoder, capacity)?,
                predictor: Decoder::load(
                    speech,
                    "talker.code_predictor.model",
                    c.code_predictor_config,
                    NonZeroUsize::new(c.num_code_groups + 1)
                        .ok_or(Error::Contract("empty code groups"))?,
                )?,
                text: speech.tensor("talker.model.text_embedding.weight")?,
                text_first: Linear::load(speech, "talker.text_projection.linear_fc1")?,
                text_last: Linear::load(speech, "talker.text_projection.linear_fc2")?,
                code_table,
                code_offsets,
                code_vocabs,
                talker_head: Linear::load(speech, "talker.codec_head")?,
                predictor_heads: (0..c.num_code_groups - 1)
                    .map(|i| Linear::load(speech, &format!("talker.code_predictor.lm_head.{i}")))
                    .collect::<Result<_, _>>()?,
                projection,
                speaker: Speaker::load(conditioning, &config.speaker_encoder_config)?,
                encoder: CodecEncoder::load(
                    audio,
                    &codec.encoder_config,
                    codec.encoder_valid_num_quantizers,
                    codec.encode_downsample_rate,
                )?,
                device: speech.device().clone(),
                dtype: speech.dtype(),
            },
            BurnCodec {
                decoder,
                decoder_cache,
                reference_decoder_cache: None,
                device: speech.device().clone(),
            },
        ))
    }
    fn embed(&self, table: &BurnTensor<2>, token: u32) -> Result<BurnTensor<3>, Error> {
        let [vocabulary, width] = table.dims();
        let row = token as usize;
        if row >= vocabulary {
            return Err(Error::Contract("token outside embedding vocabulary"));
        }
        // A known row needs no uploaded integer index or embedding gather kernel.
        Ok(table
            .clone()
            .slice([row..row + 1, 0..width])
            .reshape([1, 1, width]))
    }
}
impl Model for Burn {
    type Hidden = BurnTensor<3>;
    type Token = super::sampling::DeviceToken;
    type Logits = super::sampling::DeviceLogits;
    type Codec = BurnCodec;
    fn code_token(&mut self, group: usize, token: &Self::Token) -> Result<Self::Hidden, Error> {
        let &offset = self
            .code_offsets
            .get(group)
            .ok_or(Error::Contract("code group out of range"))?;
        Ok(BurnTensor::from_dispatch(Dispatch::embed_device_token(
            self.code_table.clone().into_dispatch(),
            token.raw().into_dispatch(),
            offset,
        )))
    }
    async fn read_tokens(&self, tokens: &[Self::Token]) -> Result<Vec<u32>, Error> {
        Ok(super::sampling::read_tokens(tokens).await?)
    }
    fn groups(&self) -> usize {
        self.code_vocabs.len()
    }
    fn context_capacity(&self) -> usize {
        self.talker.capacity()
    }
    fn reset_generation(&mut self) {
        self.talker.reset();
        self.predictor.reset();
    }
    fn reset_predictor(&mut self) {
        self.predictor.reset();
    }
    fn text(&mut self, token: u32) -> Result<Self::Hidden, Error> {
        Ok(self.text_last.forward(activation::silu(
            self.text_first.forward(self.embed(&self.text, token)?),
        )))
    }
    fn code(&mut self, group: usize, token: u32) -> Result<Self::Hidden, Error> {
        let &vocab = self
            .code_vocabs
            .get(group)
            .ok_or(Error::Contract("code group out of range"))?;
        let index = token as usize;
        if index >= vocab {
            return Err(Error::Contract("token outside embedding vocabulary"));
        }
        let row = self.code_offsets[group] + index;
        let width = self.code_table.dims()[1];
        Ok(self
            .code_table
            .clone()
            .slice([row..row + 1, 0..width])
            .reshape([1, 1, width]))
    }
    fn frame(&mut self, codes: &[u32], text: &Self::Hidden) -> Result<Self::Hidden, Error> {
        if codes.len() != self.code_vocabs.len() {
            return Err(Error::Contract("wrong frame code groups"));
        }
        let mut rows = Vec::with_capacity(codes.len());
        for (group, &token) in codes.iter().enumerate() {
            let index = token as usize;
            if index >= self.code_vocabs[group] {
                return Err(Error::Contract("token outside embedding vocabulary"));
            }
            rows.push((self.code_offsets[group] + index) as i32);
        }
        let rows =
            BurnTensor::<1, Int>::from_data(TensorData::new(rows, [codes.len()]), &self.device);
        Ok(BurnTensor::from_dispatch(Dispatch::sum_frame_embeddings(
            self.code_table.clone().into_dispatch(),
            rows.into_dispatch(),
            text.clone().into_dispatch(),
        )))
    }
    fn add(a: &Self::Hidden, b: &Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(a.clone() + b.clone())
    }
    fn talker(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(self.talker.step(hidden)?)
    }
    fn prefill(
        &mut self,
        first: Self::Hidden,
        remaining: impl IntoIterator<Item = Self::Hidden>,
    ) -> Result<Self::Hidden, Error> {
        let mut prefix = vec![first];
        prefix.extend(remaining);
        let hidden = self.talker.step(BurnTensor::cat(prefix, 1))?;
        let [b, t, h] = hidden.dims();
        Ok(hidden.slice([0..b, t - 1..t, 0..h]))
    }
    async fn talker_head(
        &mut self,
        hidden: Self::Hidden,
    ) -> Result<(Self::Hidden, Self::Logits), Error> {
        let hidden = self.talker.normalize(hidden);
        let logits =
            super::sampling::DeviceLogits(self.talker_head.forward(hidden.clone()).flatten(0, 2));
        Ok((hidden, logits))
    }
    fn project(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(match &self.projection {
            Some(p) => p.forward(hidden),
            None => hidden,
        })
    }
    fn predictor(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(self.predictor.step_unflushed(hidden)?)
    }
    async fn predictor_head(
        &mut self,
        group: usize,
        hidden: Self::Hidden,
    ) -> Result<Self::Logits, Error> {
        Ok(super::sampling::DeviceLogits(
            self.predictor_heads[group]
                .forward(self.predictor.normalize(hidden))
                .flatten(0, 2),
        ))
    }
    async fn condition(
        &mut self,
        samples: &[f32],
        mel: &[f32],
        channels: usize,
    ) -> Result<(Self::Hidden, Vec<Vec<u32>>), Error> {
        let mel = BurnTensor::<3>::from_data(
            TensorData::new(mel.to_vec(), [1, mel.len() / channels, channels]),
            &self.device,
        );
        let speaker = self.speaker.forward(mel).swap_dims(1, 2).cast(self.dtype);
        let audio = BurnTensor::<3>::from_data(
            TensorData::new(samples.to_vec(), [1, 1, samples.len()]),
            &self.device,
        );
        let codes = self.encoder.forward(audio);
        let [_, groups, frames] = codes.dims();
        let values = codes
            .into_data_async()
            .await
            .map_err(|e| super::Error::Execution(e.to_string()))?
            .try_into_vec::<i32>()
            .map_err(|e| super::Error::Execution(e.to_string()))?;
        let codes = (0..frames)
            .map(|f| {
                (0..groups)
                    .map(|g| {
                        u32::try_from(values[g * frames + f])
                            .map_err(|_| Error::Contract("negative reference code"))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((speaker, codes))
    }
}
impl Codec for BurnCodec {
    fn reset(&mut self) {
        self.decoder_cache = self
            .reference_decoder_cache
            .clone()
            .unwrap_or_else(|| self.decoder.cache());
    }
    async fn prepare_reference(&mut self, codes: &[Vec<u32>]) -> Result<(), Error> {
        let groups = codes
            .first()
            .ok_or(Error::Contract("reference contains no codec frames"))?
            .len();
        let context = crate::inference::codec_context_start(codes.len());
        let packed = (0..groups)
            .flat_map(|group| {
                codes[context..]
                    .iter()
                    .map(move |frame| frame[group] as i32)
            })
            .collect();
        let mut checkpoint = self.decoder.cache();
        self.decoder
            .prefix(
                packed,
                groups,
                codes.len() - context,
                &mut checkpoint,
                &self.device,
            )
            .await?;
        self.reference_decoder_cache = Some(checkpoint);
        Ok(())
    }
    async fn decode_prefix(
        &mut self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
        #[cfg(all(feature = "browser", target_arch = "wasm32"))]
        let started = js_sys::Date::now();
        let samples = self
            .decoder
            .prefix(
                packed,
                groups,
                frames,
                &mut self.decoder_cache,
                &self.device,
            )
            .await?;
        #[cfg(all(feature = "browser", target_arch = "wasm32"))]
        codec_timing("Speech codec timing", frames, js_sys::Date::now() - started);
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn fused_frame_embeddings_match_sequential_slice_and_add() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F32, DType::F16] {
                let width = 300usize;
                let vocabs = [11usize, 9, 9, 9];
                let total_rows: usize = vocabs.iter().sum();
                let mut offsets = Vec::new();
                let mut acc = 0usize;
                for &v in &vocabs {
                    offsets.push(acc);
                    acc += v;
                }
                let table = BurnTensor::<2>::from_data(
                    TensorData::new(
                        (0..total_rows * width)
                            .map(|i| (((i * 19 + 5) % 97) as f32 - 48.0) * 0.03125)
                            .collect::<Vec<_>>(),
                        [total_rows, width],
                    ),
                    (&device, dtype),
                );
                let text = BurnTensor::<3>::from_data(
                    TensorData::new(
                        (0..width)
                            .map(|i| (((i * 13 + 7) % 61) as f32 - 30.0) * 0.03125)
                            .collect::<Vec<_>>(),
                        [1, 1, width],
                    ),
                    (&device, dtype),
                );

                // Check device-token row lookup, including negative status clamping.
                for (group, raw) in [(0usize, 7i32), (2, 4), (3, -2)] {
                    let token =
                        BurnTensor::<1, Int>::from_data(TensorData::new(vec![raw], [1]), &device);
                    let actual = BurnTensor::<3>::from_dispatch(Dispatch::embed_device_token(
                        table.clone().into_dispatch(),
                        token.into_dispatch(),
                        offsets[group],
                    ));
                    let row = offsets[group] + raw.max(0) as usize;
                    let expected = table
                        .clone()
                        .slice([row..row + 1, 0..width])
                        .reshape([1, 1, width]);
                    let (actual, expected) = futures_lite::future::zip(
                        super::super::read(actual),
                        super::super::read(expected),
                    )
                    .await;
                    assert_eq!(
                        actual.unwrap(),
                        expected.unwrap(),
                        "{dtype:?} group {group} raw {raw}"
                    );
                }

                // Check fused multi-group frame embedding sum against sequential adds.
                let codes = [3u32, 8, 0, 5];
                let rows: Vec<i32> = codes
                    .iter()
                    .enumerate()
                    .map(|(g, &c)| (offsets[g] + c as usize) as i32)
                    .collect();
                let rows_tensor =
                    BurnTensor::<1, Int>::from_data(TensorData::new(rows, [codes.len()]), &device);
                let actual = BurnTensor::<3>::from_dispatch(Dispatch::sum_frame_embeddings(
                    table.clone().into_dispatch(),
                    rows_tensor.into_dispatch(),
                    text.clone().into_dispatch(),
                ));
                let mut expected = table
                    .clone()
                    .slice([
                        offsets[0] + codes[0] as usize..offsets[0] + codes[0] as usize + 1,
                        0..width,
                    ])
                    .reshape([1, 1, width]);
                for (g, &c) in codes.iter().enumerate().skip(1) {
                    let row = offsets[g] + c as usize;
                    let part = table
                        .clone()
                        .slice([row..row + 1, 0..width])
                        .reshape([1, 1, width]);
                    expected = expected + part;
                }
                expected = expected + text;
                let (actual, expected) = futures_lite::future::zip(
                    super::super::read(actual),
                    super::super::read(expected),
                )
                .await;
                assert_eq!(actual.unwrap(), expected.unwrap(), "{dtype:?} frame sum");
            }
        });
    }
}
