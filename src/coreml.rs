//! Core ML model loading and native tensor execution.
use crate::{
    decoder::Decoder,
    generation::Model as Backend,
    inference::{Assets, AudioContract, Config, Engine, Error, Sampling},
    sampling::TokenSampler,
};
use ondevice_coreml::{ComputeUnits, Function, Inputs, Model, Tensor, f16};
use serde::Deserialize;
use std::{fs::File, io::BufReader, path::Path};
#[derive(Deserialize)]
struct Manifest {
    weight_bits: u8,
    decoders: Decoders,
    text_embedding: String,
    codec_embeddings: Vec<String>,
    codec_frame: String,
    predictor_projection: Option<String>,
}
#[derive(Deserialize)]
struct Decoders {
    talker: DecoderFiles,
    predictor: DecoderFiles,
}
#[derive(Deserialize)]
struct DecoderFiles {
    chunks: Vec<String>,
    heads: Vec<String>,
    rope_theta: f32,
}

pub struct CoreMl {
    talker: Decoder,
    predictor: Decoder,
    talker_head: Function,
    predictor_heads: Vec<Function>,
    predictor_projection: Option<Function>,
    codec_embeddings: Vec<Function>,
    codec_frame: Function,
    text_embedding: Function,
    speaker: Function,
    encoder: Function,
    waveform: Function,
}
fn json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, Error> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}
fn open(path: impl AsRef<Path>, units: ComputeUnits) -> Result<Function, Error> {
    Ok(Model::open(path)?.function(None, units)?)
}
impl<T: TokenSampler, S: TokenSampler> Engine<CoreMl, T, S> {
    pub fn open(
        transformers: &Path,
        audio: &Path,
        units: ComputeUnits,
        sampling: Sampling<T, S>,
    ) -> Result<Self, Error> {
        let manifest: Manifest = json(transformers.join("transformers.json"))?;
        let config: Config = json(transformers.join("config.json"))?;
        let encoder_contract: AudioContract = json(audio.join("encoder.contract.json"))?;
        let decoder_contract: AudioContract = json(audio.join("decoder.contract.json"))?;
        if encoder_contract.code_groups != config.talker_config.num_code_groups
            || decoder_contract.code_groups != encoder_contract.code_groups
        {
            return Err(Error::Contract(
                "audio and speech model code groups disagree",
            ));
        }
        let filters = json(audio.join("mel.json"))?;
        let tokenizer = tokenizers::Tokenizer::from_file(transformers.join("tokenizer.json"))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let load_decoder = |files: &DecoderFiles| {
            Decoder::open(
                files.chunks.iter().map(|p| transformers.join(p)),
                files.rope_theta,
                units,
            )
        };
        if manifest.decoders.talker.heads.len() != 1 {
            return Err(Error::Contract("Talker requires exactly one output head"));
        }
        let backend = CoreMl {
            talker: load_decoder(&manifest.decoders.talker)?,
            talker_head: open(transformers.join(&manifest.decoders.talker.heads[0]), units)?,
            predictor: load_decoder(&manifest.decoders.predictor)?,
            predictor_heads: manifest
                .decoders
                .predictor
                .heads
                .iter()
                .map(|p| open(transformers.join(p), units))
                .collect::<Result<_, _>>()?,
            predictor_projection: manifest
                .predictor_projection
                .map(|p| open(transformers.join(p), units))
                .transpose()?,
            codec_embeddings: manifest
                .codec_embeddings
                .iter()
                .map(|p| open(transformers.join(p), units))
                .collect::<Result<_, _>>()?,
            codec_frame: open(transformers.join(manifest.codec_frame), units)?,
            text_embedding: open(transformers.join(manifest.text_embedding), units)?,
            speaker: open(audio.join("speaker.mlmodelc"), units)?,
            encoder: open(audio.join("encoder.mlmodelc"), units)?,
            waveform: open(audio.join("decoder.mlmodelc"), units)?,
        };
        Self::new(
            backend,
            Assets {
                weight_bits: manifest.weight_bits,
                config,
                encoder_contract,
                decoder_contract,
                filters,
                tokenizer,
            },
            sampling,
        )
    }
}
impl Backend for CoreMl {
    type Hidden = Tensor<f16>;
    type Token = u32;
    type Logits = Vec<f32>;
    fn code_token(&mut self, group: usize, token: &u32) -> Result<Self::Hidden, Error> {
        self.code(group, *token)
    }
    async fn read_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>, Error> {
        Ok(tokens.to_vec())
    }
    fn groups(&self) -> usize {
        self.codec_embeddings.len()
    }
    fn context_capacity(&self) -> usize {
        self.talker.capacity()
    }
    fn reset(&mut self) {
        self.talker.reset();
        self.predictor.reset();
    }
    fn reset_predictor(&mut self) {
        self.predictor.reset();
    }
    fn text(&mut self, token: u32) -> Result<Self::Hidden, Error> {
        embed(&mut self.text_embedding, token)
    }
    fn code(&mut self, group: usize, token: u32) -> Result<Self::Hidden, Error> {
        embed(&mut self.codec_embeddings[group], token)
    }
    fn frame(&mut self, codes: &[u32], text: &Self::Hidden) -> Result<Self::Hidden, Error> {
        frame(&mut self.codec_frame, codes, text)
    }
    fn add(a: &Self::Hidden, b: &Self::Hidden) -> Result<Self::Hidden, Error> {
        add(a, b)
    }
    fn talker(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(self.talker.step(hidden)?)
    }
    async fn talker_head(
        &mut self,
        hidden: Self::Hidden,
    ) -> Result<(Self::Hidden, Vec<f32>), Error> {
        head(&mut self.talker_head, &hidden)
    }
    fn project(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        if let Some(function) = &mut self.predictor_projection {
            let mut inputs = Inputs::new();
            inputs.insert("hidden", &hidden)?;
            Ok(function.predict(&inputs)?.tensor("hidden_out")?)
        } else {
            Ok(hidden)
        }
    }
    fn predictor(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error> {
        Ok(self.predictor.step(hidden)?)
    }
    async fn predictor_head(
        &mut self,
        group: usize,
        hidden: Self::Hidden,
    ) -> Result<Vec<f32>, Error> {
        Ok(head(&mut self.predictor_heads[group], &hidden)?.1)
    }
    async fn condition(
        &mut self,
        samples: &[f32],
        mel: &[f32],
        channels: usize,
    ) -> Result<(Self::Hidden, Vec<Vec<u32>>), Error> {
        let mel = Tensor::from_slice(&[1, mel.len() / channels, channels], mel)?;
        let mut inputs = Inputs::new();
        inputs.insert("mel", &mel)?;
        let speaker = self
            .speaker
            .predict(&inputs)?
            .tensor::<f16>("speaker")?
            .read()
            .into_data();
        if speaker.len() != self.talker.hidden_size() {
            return Err(Error::Contract(
                "speaker embedding dimension differs from Talker",
            ));
        }
        let speaker = Tensor::from_slice(&[1, speaker.len(), 1, 1], &speaker)?;
        let waveform = Tensor::from_slice(&[1, 1, samples.len()], samples)?;
        let mut inputs = Inputs::new();
        inputs.insert("audio", &waveform)?;
        let prediction = self
            .encoder
            .predict(&inputs)?
            .tensor::<i32>("codes")?
            .read();
        let shape = prediction.shape();
        let groups = self.codec_embeddings.len();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != groups {
            return Err(Error::Contract("audio encoder returned invalid code shape"));
        }
        let frames = shape[2];
        let mut codes = Vec::with_capacity(frames);
        for frame in 0..frames {
            codes.push(
                (0..groups)
                    .map(|group| {
                        u32::try_from(prediction.data()[group * frames + frame])
                            .map_err(|_| Error::Contract("negative audio code"))
                    })
                    .collect::<Result<_, _>>()?,
            );
        }
        Ok((speaker, codes))
    }
    async fn waveform(
        &mut self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error> {
        let tensor = Tensor::from_slice(&[1, groups, frames], &packed)?;
        let mut inputs = Inputs::new();
        inputs.insert("codes", &tensor)?;
        Ok(self
            .waveform
            .predict(&inputs)?
            .tensor::<f32>("audio")?
            .read()
            .into_data())
    }
}

fn frame(function: &mut Function, codes: &[u32], text: &Tensor<f16>) -> Result<Tensor<f16>, Error> {
    let ids = codes
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| Error::Contract("codec token exceeds int32")))
        .collect::<Result<Vec<_>, _>>()?;
    let ids = Tensor::from_slice(&[1, ids.len()], &ids)?;
    let mut inputs = Inputs::new();
    inputs.insert("codes", &ids)?;
    inputs.insert("text", text)?;
    Ok(function
        .predict(&inputs)
        .map_err(Error::CoreMl)?
        .tensor("hidden")?)
}

pub fn embed(function: &mut Function, token: u32) -> Result<Tensor<f16>, Error> {
    let token = i32::try_from(token).map_err(|_| Error::Contract("token exceeds int32"))?;
    let tensor = Tensor::from_slice(&[1, 1], &[token])?;
    let mut inputs = Inputs::new();
    inputs.insert("token", &tensor)?;
    Ok(function
        .predict(&inputs)
        .map_err(Error::CoreMl)?
        .tensor("hidden")?)
}

fn head(function: &mut Function, hidden: &Tensor<f16>) -> Result<(Tensor<f16>, Vec<f32>), Error> {
    let mut inputs = Inputs::new();
    inputs.insert("hidden", hidden)?;
    let prediction = function.predict(&inputs).map_err(Error::CoreMl)?;
    let logits = prediction
        .tensor::<f16>("logits")?
        .read()
        .into_data()
        .into_iter()
        .map(f16::to_f32)
        .collect();
    Ok((prediction.tensor("normalized")?, logits))
}

fn add(a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, Error> {
    if a.shape() != b.shape() {
        return Err(Error::Contract("prompt embedding shapes disagree"));
    }
    let values = a
        .read()
        .data()
        .iter()
        .zip(b.read().data())
        .map(|(x, y)| f16::from_f32(x.to_f32() + y.to_f32()))
        .collect::<Vec<_>>();
    Ok(Tensor::from_slice(a.shape(), &values)?)
}
