//! End-to-end cloning on an application-owned inference worker.
use crate::{
    decoder::Decoder,
    generation::{self, Generator, Networks, PredictorProjection, Selection},
    mel::Spectrogram,
    sampling::{RepetitionPenalty, TokenSampler},
};
use ondevice_coreml::{ComputeUnits, Function, Inputs, Model, Tensor, f16};
use serde::Deserialize;
use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    CoreMl(#[from] ondevice_coreml::Error),
    #[error(transparent)]
    Generation(#[from] generation::Error),
    #[error(transparent)]
    Decoder(#[from] crate::decoder::Error),
    #[error(transparent)]
    Mel(#[from] crate::mel::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("model contract: {0}")]
    Contract(&'static str),
    #[error("unsupported language: {0}")]
    Language(String),
}

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
#[derive(Deserialize)]
struct Config {
    tts_model_size: String,
    tts_bos_token_id: u32,
    tts_eos_token_id: u32,
    tts_pad_token_id: u32,
    talker_config: TalkerConfig,
}
#[derive(Deserialize)]
struct TalkerConfig {
    hidden_size: usize,
    vocab_size: u32,
    num_code_groups: usize,
    codec_bos_id: u32,
    codec_pad_id: u32,
    codec_eos_token_id: u32,
    codec_think_id: u32,
    codec_nothink_id: u32,
    codec_think_bos_id: u32,
    codec_think_eos_id: u32,
    codec_language_id: BTreeMap<String, u32>,
}
#[derive(Deserialize)]
struct AudioContract {
    capacity: usize,
    sample_rate: u32,
    samples_per_frame: usize,
    code_groups: usize,
}

pub struct PredictorInput(Option<Function>);
impl PredictorProjection for PredictorInput {
    fn project(&mut self, hidden: Tensor<f16>) -> Result<Tensor<f16>, generation::Error> {
        match &mut self.0 {
            Some(function) => function.project(hidden),
            None => Ok(hidden),
        }
    }
}

pub struct Sampling<T, S> {
    pub talker: T,
    pub predictor: S,
    pub repetition: RepetitionPenalty,
    pub minimum_frames: usize,
}

/// Reusable conditioning from one reference recording. It remains local to
/// the caller and carries no global voice registry or persistent side effects.
pub struct Reference {
    codes: Vec<Vec<u32>>,
    speaker: Tensor<f16>,
    text: Vec<u32>,
}

pub struct Synthesis {
    pub model_size: String,
    pub transformer_weight_bits: u8,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub generated_frames: usize,
    pub reached_eos: bool,
}

pub struct Engine<T, S> {
    weight_bits: u8,
    generator: Generator<PredictorInput, T, S>,
    text_embedding: Function,
    speaker: Function,
    encoder: Function,
    waveform: Function,
    spectrogram: Spectrogram,
    tokenizer: tokenizers::Tokenizer,
    config: Config,
    encoder_contract: AudioContract,
    decoder_contract: AudioContract,
}

fn json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, Error> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}
fn open(path: impl AsRef<Path>, units: ComputeUnits) -> Result<Function, Error> {
    Ok(Model::open(path)?.function(None, units)?)
}

impl<T: TokenSampler, S: TokenSampler> Engine<T, S> {
    pub fn open(
        transformers: &Path,
        audio: &Path,
        units: ComputeUnits,
        sampling: Sampling<T, S>,
    ) -> Result<Self, Error> {
        let manifest: Manifest = json(transformers.join("transformers.json"))?;
        let config: Config = json(transformers.join("config.json"))?;
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
        let networks = Networks {
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
            predictor_projection: PredictorInput(
                manifest
                    .predictor_projection
                    .map(|p| open(transformers.join(p), units))
                    .transpose()?,
            ),
            codec_embeddings: manifest
                .codec_embeddings
                .iter()
                .map(|p| open(transformers.join(p), units))
                .collect::<Result<_, _>>()?,
            codec_frame: open(transformers.join(manifest.codec_frame), units)?,
        };
        // The published Qwen generator reserves the final 1024 codec IDs.
        let reserved_start =
            config
                .talker_config
                .vocab_size
                .checked_sub(1024)
                .ok_or(Error::Contract(
                    "codec vocabulary is smaller than its reserved range",
                ))?;
        let selection = Selection {
            talker: sampling.talker,
            predictor: sampling.predictor,
            repetition: sampling.repetition,
            eos: config.talker_config.codec_eos_token_id,
            reserved_start,
            minimum_frames: sampling.minimum_frames,
        };
        let encoder_contract: AudioContract = json(audio.join("encoder.contract.json"))?;
        let decoder_contract: AudioContract = json(audio.join("decoder.contract.json"))?;
        if encoder_contract.code_groups != config.talker_config.num_code_groups
            || decoder_contract.code_groups != encoder_contract.code_groups
        {
            return Err(Error::Contract(
                "audio and speech model code groups disagree",
            ));
        }
        Ok(Self {
            weight_bits: manifest.weight_bits,
            generator: Generator::new(networks, selection)?,
            text_embedding: open(transformers.join(manifest.text_embedding), units)?,
            speaker: open(audio.join("speaker.mlmodelc"), units)?,
            encoder: open(audio.join("encoder.mlmodelc"), units)?,
            waveform: open(audio.join("decoder.mlmodelc"), units)?,
            spectrogram: Spectrogram::new(json(audio.join("mel.json"))?)?,
            tokenizer: tokenizers::Tokenizer::from_file(transformers.join("tokenizer.json"))
                .map_err(|e| Error::Tokenizer(e.to_string()))?,
            config,
            encoder_contract,
            decoder_contract,
        })
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec())
    }

    pub fn reference(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        transcript: &str,
    ) -> Result<Reference, Error> {
        if sample_rate != self.spectrogram.sample_rate()
            || sample_rate != self.encoder_contract.sample_rate
        {
            return Err(Error::Contract("reference sample rate differs from model"));
        }
        if samples.len() > self.encoder_contract.capacity || transcript.is_empty() {
            return Err(Error::Contract(
                "reference exceeds encoder capacity or lacks a transcript",
            ));
        }
        let mel = self.spectrogram.compute(samples)?;
        let channels = self.spectrogram.channels();
        let mel = Tensor::from_slice(&[1, mel.len() / channels, channels], &mel)?;
        let mut inputs = Inputs::new();
        inputs.insert("mel", &mel)?;
        let speaker = self
            .speaker
            .predict(&inputs)?
            .tensor::<f16>("speaker")?
            .read()
            .into_data();
        if speaker.len() != self.config.talker_config.hidden_size {
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
        let groups = self.encoder_contract.code_groups;
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
        Ok(Reference {
            codes,
            speaker,
            text: self.tokenize(transcript)?,
        })
    }

    pub fn synthesize(
        &mut self,
        reference: &Reference,
        text: &str,
        language: &str,
        maximum_frames: usize,
    ) -> Result<Synthesis, Error> {
        if maximum_frames == 0 || text.is_empty() {
            return Err(Error::Contract(
                "text and generation budget must be nonempty",
            ));
        }
        let mut prefill = Vec::new();
        for id in self.tokenize("<|im_start|>assistant\n")? {
            prefill.push(generation::embed(&mut self.text_embedding, id)?);
        }
        let padding = generation::embed(&mut self.text_embedding, self.config.tts_pad_token_id)?;
        let bos = generation::embed(&mut self.text_embedding, self.config.tts_bos_token_id)?;
        let cfg = &self.config.talker_config;
        let language = language.to_lowercase();
        let tags = if language == "auto" {
            vec![
                cfg.codec_nothink_id,
                cfg.codec_think_bos_id,
                cfg.codec_think_eos_id,
            ]
        } else {
            let id = *cfg
                .codec_language_id
                .get(&language)
                .ok_or_else(|| Error::Language(language.clone()))?;
            vec![
                cfg.codec_think_id,
                cfg.codec_think_bos_id,
                id,
                cfg.codec_think_eos_id,
            ]
        };
        for id in tags {
            prefill.push(add(&padding, &self.generator.embed_code(0, id)?)?);
        }
        prefill.push(add(&padding, &reference.speaker)?);
        prefill.push(add(&bos, &self.generator.embed_code(0, cfg.codec_pad_id)?)?);
        let codec_bos = cfg.codec_bos_id;
        let mut ids = reference.text.clone();
        ids.extend(self.tokenize(text)?);
        ids.push(self.config.tts_eos_token_id);
        let mut text_embeddings = ids
            .into_iter()
            .map(|id| generation::embed(&mut self.text_embedding, id))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();
        let first = text_embeddings
            .next()
            .ok_or(Error::Contract("ICL text is empty"))?;
        prefill.push(add(&first, &self.generator.embed_code(0, codec_bos)?)?);
        for codes in &reference.codes {
            let text = text_embeddings.next().unwrap_or_else(|| padding.clone());
            prefill.push(self.generator.embed_frame(codes, &text)?);
        }
        let mut sequence = self
            .generator
            .start(prefill, text_embeddings.collect(), padding)?;
        let mut frames = Vec::new();
        let mut reached_eos = false;
        for _ in 0..maximum_frames {
            match sequence.next_frame()? {
                Some(frame) => frames.push(frame),
                None => {
                    reached_eos = true;
                    break;
                }
            }
        }
        let generated_frames = frames.len();
        let mut codes = reference.codes.clone();
        codes.extend(frames);
        let samples = self.decode(&codes)?;
        let reference_samples = reference.codes.len() * self.decoder_contract.samples_per_frame;
        Ok(Synthesis {
            model_size: self.config.tts_model_size.clone(),
            transformer_weight_bits: self.weight_bits,
            samples: samples[reference_samples..].to_vec(),
            sample_rate: self.decoder_contract.sample_rate,
            generated_frames,
            reached_eos,
        })
    }

    fn decode(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>, Error> {
        // Qwen's published chunked_decode uses 300 frames plus 25 left-context
        // frames. Preserve those boundaries, including the first chunk.
        let (chunk_size, left_context) = (300, 25);
        let capacity = self.decoder_contract.capacity;
        if capacity < chunk_size + left_context {
            return Err(Error::Contract(
                "waveform decoder capacity must cover 300 frames and 25 context frames",
            ));
        }
        let groups = self.decoder_contract.code_groups;
        let samples_per_frame = self.decoder_contract.samples_per_frame;
        let mut result = Vec::with_capacity(codes.len() * samples_per_frame);
        for start in (0..codes.len()).step_by(chunk_size) {
            let end = (start + chunk_size).min(codes.len());
            let context = start.min(left_context);
            let mut packed = vec![0i32; groups * capacity];
            for (frame, values) in codes[start - context..end].iter().enumerate() {
                if values.len() != groups {
                    return Err(Error::Contract("code frame group count mismatch"));
                }
                for (group, &value) in values.iter().enumerate() {
                    packed[group * capacity + frame] =
                        i32::try_from(value).map_err(|_| Error::Contract("code exceeds int32"))?;
                }
            }
            let tensor = Tensor::from_slice(&[1, groups, capacity], &packed)?;
            let mut inputs = Inputs::new();
            inputs.insert("codes", &tensor)?;
            let output = self
                .waveform
                .predict(&inputs)?
                .tensor::<f32>("audio")?
                .read();
            let output = output.data();
            let valid = (context + end - start) * samples_per_frame;
            if output.len() < valid {
                return Err(Error::Contract(
                    "decoder waveform is shorter than its code frames",
                ));
            }
            result.extend_from_slice(&output[context * samples_per_frame..valid]);
        }
        if result.iter().any(|v| !v.is_finite()) {
            return Err(Error::Contract("waveform contains nonfinite values"));
        }
        Ok(result)
    }
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
