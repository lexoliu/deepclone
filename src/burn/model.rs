//! Complete Qwen model operations for the shared cloning session.
use super::{
    CodecDecoder, CodecDecoderConfig, CodecEncoder, Decoder, DecoderConfig, EncoderConfig, Speaker,
    SpeakerConfig, WeightSource, layers::Linear, read,
};
use crate::{generation::Model, inference::Error};
use burn::tensor::{DType, Device, Int, Tensor, TensorData, activation};
use serde::Deserialize;
use std::num::NonZeroUsize;

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
    text: Tensor<2>,
    text_first: Linear,
    text_last: Linear,
    codes: Vec<Tensor<2>>,
    talker_head: Linear,
    predictor_heads: Vec<Linear>,
    projection: Option<Linear>,
    speaker: Speaker,
    encoder: CodecEncoder,
    decoder: CodecDecoder,
    device: Device,
    dtype: DType,
}
impl Burn {
    pub fn load(
        speech: &impl WeightSource,
        conditioning: &impl WeightSource,
        audio: &impl WeightSource,
        config: SpeechConfig,
        codec: AudioConfig,
        capacity: NonZeroUsize,
    ) -> Result<Self, Error> {
        let c = config.talker_config;
        if c.hidden_act != "silu" {
            return Err(Error::Contract("unsupported text projection activation"));
        }
        let mut codes = vec![speech.tensor("talker.model.codec_embedding.weight")?];
        for group in 0..c.num_code_groups - 1 {
            codes.push(speech.tensor(&format!(
                "talker.code_predictor.model.codec_embedding.{group}.weight"
            ))?);
        }
        let projection = if c.decoder.hidden_size != c.code_predictor_config.hidden_size {
            Some(Linear::load(
                speech,
                "talker.code_predictor.small_to_mtp_projection",
            )?)
        } else {
            None
        };
        Ok(Self {
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
            codes,
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
            decoder: CodecDecoder::load(audio, &codec.decoder_config)?,
            device: speech.device().clone(),
            dtype: speech.dtype(),
        })
    }
    fn embed(&self, table: &Tensor<2>, token: u32) -> Result<Tensor<3>, Error> {
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
    type Hidden = Tensor<3>;
    fn waveform_frames(&self, valid: usize, _capacity: usize) -> usize {
        valid
    }
    fn groups(&self) -> usize {
        self.codes.len()
    }
    fn reset(&mut self) {
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
        self.embed(&self.codes[group], token)
    }
    fn frame(&mut self, codes: &[u32], text: &Self::Hidden) -> Result<Self::Hidden, Error> {
        if codes.len() != self.codes.len() {
            return Err(Error::Contract("wrong frame code groups"));
        }
        let mut sum = self.code(0, codes[0])?;
        for (group, &token) in codes.iter().enumerate().skip(1) {
            sum = sum + self.code(group, token)?;
        }
        Ok(sum + text.clone())
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
        let hidden = self.talker.step(Tensor::cat(prefix, 1))?;
        let [b, t, h] = hidden.dims();
        Ok(hidden.slice([0..b, t - 1..t, 0..h]))
    }
    async fn talker_head(
        &mut self,
        hidden: Self::Hidden,
    ) -> Result<(Self::Hidden, Vec<f32>), Error> {
        let hidden = self.talker.normalize(hidden);
        let logits = read(self.talker_head.forward(hidden.clone())).await?;
        Ok((hidden, logits))
    }
    fn project(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(match &self.projection {
            Some(p) => p.forward(hidden),
            None => hidden,
        })
    }
    fn predictor(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(self.predictor.step(hidden)?)
    }
    async fn predictor_head(
        &mut self,
        group: usize,
        hidden: Self::Hidden,
    ) -> Result<Vec<f32>, Error> {
        Ok(read(self.predictor_heads[group].forward(self.predictor.normalize(hidden))).await?)
    }
    async fn condition(
        &mut self,
        samples: &[f32],
        mel: &[f32],
        channels: usize,
    ) -> Result<(Self::Hidden, Vec<Vec<u32>>), Error> {
        let mel = Tensor::<3>::from_data(
            TensorData::new(mel.to_vec(), [1, mel.len() / channels, channels]),
            &self.device,
        );
        let speaker = self.speaker.forward(mel).swap_dims(1, 2).cast(self.dtype);
        let audio = Tensor::<3>::from_data(
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
    async fn waveform(
        &mut self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
        let codes =
            Tensor::<3, Int>::from_data(TensorData::new(packed, [1, groups, frames]), &self.device);
        Ok(read(self.decoder.forward(codes)).await?)
    }
}
