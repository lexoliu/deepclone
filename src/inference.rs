//! End-to-end cloning on an application-owned inference worker.
use crate::{
    decoder::Decoder,
    generation::{self, Generator, Networks, PredictorProjection, Selection},
    mel::Spectrogram,
    sampling::{RepetitionPenalty, TokenSampler},
};
use ondevice_coreml::{ComputeUnits, Function, Inputs, Model, Tensor, f16};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    io::BufReader,
    num::NonZeroUsize,
    path::Path,
};

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
    #[error("text has no stable token boundary within the configured {0}-byte buffer")]
    TextCapacity(usize),
    #[error("generation exceeded the configured budget of {0} frames")]
    FrameBudget(usize),
    #[error("model emitted audio EOS before consuming the complete text input")]
    PrematureEos,
    #[error("inference worker stopped without completing the session")]
    WorkerStopped,
    #[error("session cancelled")]
    Cancelled,
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

/// Frame-major acoustic codes, including the conditioning prefix needed by
/// the published chunked decoder. Construction validates matrix boundaries.
#[derive(Debug, Serialize)]
pub struct AcousticCodes {
    groups: NonZeroUsize,
    values: Vec<u32>,
    reference_frames: usize,
}

impl AcousticCodes {
    pub fn new(
        groups: NonZeroUsize,
        values: Vec<u32>,
        reference_frames: usize,
    ) -> Result<Self, Error> {
        if !values.len().is_multiple_of(groups.get()) || reference_frames > values.len() / groups {
            return Err(Error::Contract(
                "acoustic code matrix or reference boundary is invalid",
            ));
        }
        Ok(Self {
            groups,
            values,
            reference_frames,
        })
    }

    pub fn groups(&self) -> NonZeroUsize {
        self.groups
    }
    pub fn values(&self) -> &[u32] {
        &self.values
    }
    pub fn reference_frames(&self) -> usize {
        self.reference_frames
    }
    pub fn frames(&self) -> usize {
        self.values.len() / self.groups
    }
    pub fn generated_frames(&self) -> usize {
        self.frames() - self.reference_frames
    }
}

#[derive(Debug, Serialize)]
pub struct Generation {
    pub model_size: String,
    pub transformer_weight_bits: u8,
    pub codes: AcousticCodes,
}

pub struct Waveform {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct Synthesis {
    pub generation: Generation,
    pub waveform: Waveform,
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
        let encoder_contract: AudioContract = json(audio.join("encoder.contract.json"))?;
        let decoder_contract: AudioContract = json(audio.join("decoder.contract.json"))?;
        if encoder_contract.code_groups != config.talker_config.num_code_groups
            || decoder_contract.code_groups != encoder_contract.code_groups
        {
            return Err(Error::Contract(
                "audio and speech model code groups disagree",
            ));
        }
        let spectrogram = Spectrogram::new(json(audio.join("mel.json"))?)?;
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
        Ok(Self {
            weight_bits: manifest.weight_bits,
            generator: Generator::new(networks, selection)?,
            text_embedding: open(transformers.join(manifest.text_embedding), units)?,
            speaker: open(audio.join("speaker.mlmodelc"), units)?,
            encoder: open(audio.join("encoder.mlmodelc"), units)?,
            waveform: open(audio.join("decoder.mlmodelc"), units)?,
            spectrogram,
            tokenizer,
            config,
            encoder_contract,
            decoder_contract,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.decoder_contract.sample_rate
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

    /// Collect the same incremental session used by the streaming interface.
    pub fn synthesize(
        &mut self,
        reference: &Reference,
        text: &str,
        language: &str,
        maximum_frames: usize,
    ) -> Result<Synthesis, Error> {
        let mut session = self.session(reference, language, maximum_frames, usize::MAX)?;
        session.push_text(text)?;
        session.finish_input()?;
        let mut samples = Vec::new();
        while !session.finished {
            if let Some(chunk) = session.next_chunk(12, || false)? {
                samples.extend(chunk.samples);
            }
        }
        let sample_rate = session.decoder_contract.sample_rate;
        let generation = session.into_generation()?;
        Ok(Synthesis {
            generation,
            waveform: Waveform {
                samples,
                sample_rate,
            },
        })
    }

    pub fn generate(
        &mut self,
        reference: &Reference,
        text: &str,
        language: &str,
        maximum_frames: usize,
    ) -> Result<Generation, Error> {
        let mut session = self.session(reference, language, maximum_frames, usize::MAX)?;
        session.push_text(text)?;
        session.finish_input()?;
        while !session.finished {
            session.advance(&|| false)?;
        }
        session.into_generation()
    }

    pub(crate) fn session<'a>(
        &'a mut self,
        reference: &'a Reference,
        language: &str,
        maximum_frames: usize,
        text_capacity: usize,
    ) -> Result<CloningSession<'a, T, S>, Error> {
        if maximum_frames == 0 {
            return Err(Error::Contract("generation budget must be nonzero"));
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
        let (&first, remaining) = reference
            .text
            .split_first()
            .ok_or(Error::Contract("reference text is empty"))?;
        let first = generation::embed(&mut self.text_embedding, first)?;
        prefill.push(add(&first, &self.generator.embed_code(0, codec_bos)?)?);
        let sequence = self.generator.start(prefill)?;
        let text = crate::text::TextBuffer::new(&self.tokenizer, text_capacity);
        let groups = NonZeroUsize::new(self.encoder_contract.code_groups)
            .ok_or(Error::Contract("codec must have at least one code group"))?;
        let codes = AcousticCodes::new(
            groups,
            reference.codes.iter().flatten().copied().collect(),
            reference.codes.len(),
        )?;
        Ok(CloningSession {
            text_embedding: &mut self.text_embedding,
            waveform: &mut self.waveform,
            decoder_contract: &self.decoder_contract,
            config: &self.config,
            weight_bits: self.weight_bits,
            reference,
            sequence,
            padding,
            text,
            tokens: remaining.iter().copied().collect(),
            prefix: 0,
            input_finished: false,
            has_text: false,
            finished: false,
            maximum_frames,
            emitted: reference.codes.len(),
            codes,
        })
    }

    /// Decode the conditioning prefix but return only generated audio.
    pub fn decode(&mut self, codes: &AcousticCodes) -> Result<Waveform, Error> {
        let samples = decode_range(
            &mut self.waveform,
            &self.decoder_contract,
            codes,
            codes.reference_frames,
            codes.frames(),
        )?;
        Ok(Waveform {
            samples,
            sample_rate: self.decoder_contract.sample_rate,
        })
    }
}

fn decode_range(
    waveform: &mut Function,
    decoder_contract: &AudioContract,
    codes: &AcousticCodes,
    mut start: usize,
    end: usize,
) -> Result<Vec<f32>, Error> {
    if codes.groups.get() != decoder_contract.code_groups {
        return Err(Error::Contract(
            "code group count differs from waveform decoder",
        ));
    }
    // Qwen's chunked_decode resets at 300 frames with 25 frames of left
    // context. Incremental output preserves these absolute boundaries.
    let (block_frames, left_context) = (300, 25);
    let capacity = decoder_contract.capacity;
    if capacity < block_frames + left_context {
        return Err(Error::Contract(
            "waveform decoder capacity must cover 300 frames and 25 context frames",
        ));
    }
    let groups = codes.groups.get();
    let samples_per_frame = decoder_contract.samples_per_frame;
    let mut result = Vec::with_capacity((end - start) * samples_per_frame);
    while start < end {
        let block = start / block_frames * block_frames;
        let context_start = block.saturating_sub(left_context);
        let stop = end.min(block + block_frames);
        let mut packed = vec![0i32; groups * capacity];
        for (frame, values) in codes.values[context_start * groups..stop * groups]
            .chunks_exact(groups)
            .enumerate()
        {
            for (group, &value) in values.iter().enumerate() {
                packed[group * capacity + frame] =
                    i32::try_from(value).map_err(|_| Error::Contract("code exceeds int32"))?;
            }
        }
        let tensor = Tensor::from_slice(&[1, groups, capacity], &packed)?;
        let mut inputs = Inputs::new();
        inputs.insert("codes", &tensor)?;
        let output = waveform.predict(&inputs)?.tensor::<f32>("audio")?.read();
        let valid = (stop - context_start) * samples_per_frame;
        if output.data().len() < valid {
            return Err(Error::Contract(
                "decoder waveform is shorter than its code frames",
            ));
        }
        result
            .extend_from_slice(&output.data()[(start - context_start) * samples_per_frame..valid]);
        start = stop;
    }
    if result.iter().any(|v| !v.is_finite()) {
        return Err(Error::Contract("waveform contains nonfinite values"));
    }
    Ok(result)
}

/// Contiguous mono PCM with an absolute position in this utterance.
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub start_sample: usize,
}

pub(crate) struct CloningSession<'a, T, S> {
    text_embedding: &'a mut Function,
    waveform: &'a mut Function,
    decoder_contract: &'a AudioContract,
    config: &'a Config,
    weight_bits: u8,
    reference: &'a Reference,
    sequence: generation::Sequence<'a, PredictorInput, T, S>,
    padding: Tensor<f16>,
    text: crate::text::TextBuffer<'a>,
    tokens: VecDeque<u32>,
    prefix: usize,
    input_finished: bool,
    has_text: bool,
    pub(crate) finished: bool,
    maximum_frames: usize,
    emitted: usize,
    codes: AcousticCodes,
}

impl<T: TokenSampler, S: TokenSampler> CloningSession<'_, T, S> {
    pub(crate) fn push_text(&mut self, text: &str) -> Result<(), Error> {
        if self.input_finished {
            return Err(Error::Contract("text input is already finished"));
        }
        self.has_text |= !text.is_empty();
        self.tokens.extend(self.text.push(text)?);
        Ok(())
    }

    pub(crate) fn finish_input(&mut self) -> Result<(), Error> {
        if self.input_finished {
            return Err(Error::Contract("text input is already finished"));
        }
        if !self.has_text {
            return Err(Error::Contract("text must be nonempty"));
        }
        self.tokens.extend(self.text.finish()?);
        self.tokens.push_back(self.config.tts_eos_token_id);
        self.input_finished = true;
        Ok(())
    }

    fn text_embedding(&mut self) -> Result<Option<Tensor<f16>>, Error> {
        if let Some(id) = self.tokens.pop_front() {
            Ok(Some(generation::embed(self.text_embedding, id)?))
        } else if self.input_finished {
            Ok(Some(self.padding.clone()))
        } else {
            Ok(None)
        }
    }

    /// False means text starvation; no padding or EOS is fabricated.
    fn advance(&mut self, cancelled: &impl Fn() -> bool) -> Result<bool, Error> {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        while self.prefix < self.reference.codes.len() {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let Some(text) = self.text_embedding()? else {
                return Ok(false);
            };
            let embedded = self
                .sequence
                .embed_frame(&self.reference.codes[self.prefix], &text)?;
            self.sequence.prefill(embedded)?;
            self.prefix += 1;
        }
        let Some(text) = self.text_embedding()? else {
            return Ok(false);
        };
        match self.sequence.next_frame(&text)? {
            Some(frame) => {
                if self.codes.generated_frames() == self.maximum_frames {
                    return Err(Error::FrameBudget(self.maximum_frames));
                }
                self.codes.values.extend(frame);
            }
            None => {
                if !self.input_finished || !self.tokens.is_empty() {
                    return Err(Error::PrematureEos);
                }
                self.finished = true;
            }
        }
        Ok(true)
    }

    pub(crate) fn next_chunk(
        &mut self,
        frames: usize,
        cancelled: impl Fn() -> bool,
    ) -> Result<Option<AudioChunk>, Error> {
        while !self.finished && self.codes.frames() - self.emitted < frames {
            if !self.advance(&cancelled)? {
                break;
            }
        }
        if self.codes.frames() == self.emitted {
            return Ok(None);
        }
        let start_sample =
            (self.emitted - self.codes.reference_frames) * self.decoder_contract.samples_per_frame;
        let samples = decode_range(
            self.waveform,
            self.decoder_contract,
            &self.codes,
            self.emitted,
            self.codes.frames(),
        )?;
        self.emitted = self.codes.frames();
        Ok(Some(AudioChunk {
            samples,
            sample_rate: self.decoder_contract.sample_rate,
            start_sample,
        }))
    }

    pub(crate) fn into_generation(self) -> Result<Generation, Error> {
        if !self.finished {
            return Err(Error::Contract("session has not completed"));
        }
        Ok(Generation {
            model_size: self.config.tts_model_size.clone(),
            transformer_weight_bits: self.weight_bits,
            codes: self.codes,
        })
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
