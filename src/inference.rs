//! End-to-end cloning on an application-owned inference worker.
use crate::{
    generation::{self, Generator, Model, Selection},
    mel::Spectrogram,
    sampling::{RepetitionPenalty, TokenSampler},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, VecDeque},
    num::NonZeroUsize,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("audio consumer: {0}")]
    Callback(String),
    #[cfg(feature = "burn")]
    #[error(transparent)]
    Burn(#[from] crate::burn::Error),
    #[error(transparent)]
    Sampling(#[from] crate::sampling::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
    #[error(transparent)]
    CoreMl(#[from] ondevice_coreml::Error),
    #[cfg(all(feature = "coreml", target_vendor = "apple"))]
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
    #[error("generation reached the configured budget of {limit} frames before EOS")]
    FrameBudget {
        limit: usize,
        /// Valid acoustic codes up to the limit, including reference conditioning.
        codes: AcousticCodes,
    },
    #[error("model emitted audio EOS before consuming the complete text input")]
    PrematureEos,
    #[error("inference worker stopped without completing the session")]
    WorkerStopped,
    #[error("text forwarding stopped without EOF")]
    InputStopped,
    #[error("session cancelled")]
    Cancelled,
}

#[derive(Deserialize)]
pub struct Config {
    pub tts_model_size: String,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub talker_config: TalkerConfig,
}
#[derive(Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub vocab_size: u32,
    pub num_code_groups: usize,
    pub codec_bos_id: u32,
    pub codec_pad_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_language_id: BTreeMap<String, u32>,
}
#[derive(Deserialize)]
pub struct AudioContract {
    pub capacity: usize,
    pub sample_rate: u32,
    pub samples_per_frame: usize,
    pub code_groups: usize,
}

pub struct Sampling<T, S> {
    pub talker: T,
    pub predictor: S,
    pub repetition: RepetitionPenalty,
    pub minimum_frames: usize,
}

/// Reusable conditioning from one reference recording. It remains local to
/// the caller and carries no global voice registry or persistent side effects.
pub struct Reference<H> {
    codes: Vec<Vec<u32>>,
    speaker: H,
    text: Vec<u32>,
}

/// Frame-major acoustic codes, including the conditioning prefix needed by
/// the published chunked decoder. Construction validates matrix boundaries.
#[derive(Clone, Debug, Serialize)]
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

pub struct Engine<B: Model, T, S> {
    weight_bits: u8,
    generator: Generator<B, T, S>,
    spectrogram: Spectrogram,
    tokenizer: tokenizers::Tokenizer,
    config: Config,
    encoder_contract: AudioContract,
    decoder_contract: AudioContract,
}

impl<B: Model, T: TokenSampler, S: TokenSampler> Engine<B, T, S> {
    pub fn new(backend: B, assets: Assets, sampling: Sampling<T, S>) -> Result<Self, Error> {
        let Assets {
            weight_bits,
            config,
            encoder_contract,
            decoder_contract,
            filters,
            tokenizer,
        } = assets;
        if backend.groups() != config.talker_config.num_code_groups
            || encoder_contract.code_groups != backend.groups()
            || decoder_contract.code_groups != backend.groups()
        {
            return Err(Error::Contract("model code groups disagree"));
        }
        let reserved_start =
            config
                .talker_config
                .vocab_size
                .checked_sub(1024)
                .ok_or(Error::Contract(
                    "codec vocabulary is smaller than reserved range",
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
            weight_bits,
            generator: Generator::new(backend, selection)?,
            spectrogram: Spectrogram::new(filters)?,
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

    pub async fn reference(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        transcript: &str,
    ) -> Result<Reference<B::Hidden>, Error> {
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
        let (speaker, codes) = self
            .generator
            .networks
            .condition(samples, &mel, channels)
            .await?;
        Ok(Reference {
            codes,
            speaker,
            text: self.tokenize(transcript)?,
        })
    }

    /// Collect the same incremental session used by the streaming interface.
    pub async fn synthesize(
        &mut self,
        reference: &Reference<B::Hidden>,
        text: &str,
        language: &str,
        maximum_frames: usize,
    ) -> Result<Synthesis, Error> {
        let mut session = self.session(reference, language, maximum_frames, usize::MAX)?;
        session.push_text(text)?;
        session.finish_input()?;
        let mut samples = Vec::new();
        while !session.finished {
            if let Some(chunk) = session.next_chunk(12, || false).await? {
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

    pub async fn generate(
        &mut self,
        reference: &Reference<B::Hidden>,
        text: &str,
        language: &str,
        maximum_frames: usize,
    ) -> Result<Generation, Error> {
        let mut session = self.session(reference, language, maximum_frames, usize::MAX)?;
        session.push_text(text)?;
        session.finish_input()?;
        while !session.finished {
            session.advance(&|| false).await?;
        }
        session.into_generation()
    }

    pub fn session<'a>(
        &'a mut self,
        reference: &'a Reference<B::Hidden>,
        language: &str,
        maximum_frames: usize,
        text_capacity: usize,
    ) -> Result<CloningSession<'a, B, T, S>, Error> {
        if maximum_frames == 0 {
            return Err(Error::Contract("generation budget must be nonzero"));
        }
        let mut prefill = Vec::new();
        for id in self.tokenize("<|im_start|>assistant\n")? {
            prefill.push(self.generator.networks.text(id)?);
        }
        let padding = self.generator.networks.text(self.config.tts_pad_token_id)?;
        let bos = self.generator.networks.text(self.config.tts_bos_token_id)?;
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
            prefill.push(B::add(&padding, &self.generator.embed_code(0, id)?)?);
        }
        prefill.push(B::add(&padding, &reference.speaker)?);
        prefill.push(B::add(
            &bos,
            &self.generator.embed_code(0, cfg.codec_pad_id)?,
        )?);
        let codec_bos = cfg.codec_bos_id;
        let (&first, remaining) = reference
            .text
            .split_first()
            .ok_or(Error::Contract("reference text is empty"))?;
        let first = self.generator.networks.text(first)?;
        prefill.push(B::add(&first, &self.generator.embed_code(0, codec_bos)?)?);
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
    pub async fn decode(&mut self, codes: &AcousticCodes) -> Result<Waveform, Error> {
        let samples = decode_range(
            &mut self.generator.networks,
            &self.decoder_contract,
            codes,
            codes.reference_frames,
            codes.frames(),
        )
        .await?;
        Ok(Waveform {
            samples,
            sample_rate: self.decoder_contract.sample_rate,
        })
    }
}

// Qwen's official chunked decoder resets every 300 codec frames and
// includes 25 frames from the preceding block as causal left context.
const CODEC_BLOCK_FRAMES: usize = 300;
const CODEC_LEFT_CONTEXT: usize = 25;
pub(crate) const CODEC_CAPACITY: usize = CODEC_BLOCK_FRAMES + CODEC_LEFT_CONTEXT;
pub(crate) fn codec_context_start(frame: usize) -> usize {
    (frame / CODEC_BLOCK_FRAMES * CODEC_BLOCK_FRAMES).saturating_sub(CODEC_LEFT_CONTEXT)
}

async fn decode_range<B: Model>(
    waveform: &mut B,
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
    let capacity = decoder_contract.capacity;
    if capacity < CODEC_CAPACITY {
        return Err(Error::Contract(
            "waveform decoder capacity must cover 300 frames and 25 context frames",
        ));
    }
    let groups = codes.groups.get();
    let samples_per_frame = decoder_contract.samples_per_frame;
    let mut result = Vec::with_capacity((end - start) * samples_per_frame);
    while start < end {
        let block = start / CODEC_BLOCK_FRAMES * CODEC_BLOCK_FRAMES;
        let context_start = codec_context_start(start);
        let stop = end.min(block + CODEC_BLOCK_FRAMES);
        let frames = waveform.waveform_frames(stop - context_start, capacity);
        let mut packed = vec![0i32; groups * frames];
        for (frame, values) in codes.values[context_start * groups..stop * groups]
            .chunks_exact(groups)
            .enumerate()
        {
            for (group, &value) in values.iter().enumerate() {
                packed[group * frames + frame] =
                    i32::try_from(value).map_err(|_| Error::Contract("code exceeds int32"))?;
            }
        }
        let output = waveform.waveform(packed, groups, frames).await?;
        let valid = (stop - context_start) * samples_per_frame;
        if output.as_slice().len() < valid {
            return Err(Error::Contract(
                "decoder waveform is shorter than its code frames",
            ));
        }
        result.extend_from_slice(
            &output.as_slice()[(start - context_start) * samples_per_frame..valid],
        );
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

pub struct CloningSession<'a, B: Model, T, S> {
    decoder_contract: &'a AudioContract,
    config: &'a Config,
    weight_bits: u8,
    reference: &'a Reference<B::Hidden>,
    sequence: generation::Sequence<'a, B, T, S>,
    padding: B::Hidden,
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

impl<B: Model, T: TokenSampler, S: TokenSampler> CloningSession<'_, B, T, S> {
    pub fn push_text(&mut self, text: &str) -> Result<(), Error> {
        if self.input_finished {
            return Err(Error::Contract("text input is already finished"));
        }
        self.has_text |= !text.is_empty();
        self.tokens.extend(self.text.push(text)?);
        Ok(())
    }

    pub fn finish_input(&mut self) -> Result<(), Error> {
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

    fn text_embedding(&mut self) -> Result<Option<B::Hidden>, Error> {
        if let Some(id) = self.tokens.pop_front() {
            Ok(Some(self.sequence.generator.networks.text(id)?))
        } else if self.input_finished {
            Ok(Some(self.padding.clone()))
        } else {
            Ok(None)
        }
    }

    /// False means text starvation; no padding or EOS is fabricated.
    async fn advance(&mut self, cancelled: &impl Fn() -> bool) -> Result<bool, Error> {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let mut prefix = Vec::new();
        while self.prefix < self.reference.codes.len() {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let Some(text) = self.text_embedding()? else {
                break;
            };
            let embedded = self
                .sequence
                .embed_frame(&self.reference.codes[self.prefix], &text)?;
            prefix.push(embedded);
            self.prefix += 1;
        }
        let mut prefix = prefix.into_iter();
        if let Some(first) = prefix.next() {
            self.sequence.prefill(first, prefix)?;
        }
        if self.prefix < self.reference.codes.len() {
            return Ok(false);
        }
        if self.codes.generated_frames() == self.maximum_frames {
            return Err(Error::FrameBudget {
                limit: self.maximum_frames,
                codes: self.codes.clone(),
            });
        }
        let Some(text) = self.text_embedding()? else {
            return Ok(false);
        };
        match self.sequence.next_frame(&text).await? {
            Some(frame) => {
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

    pub async fn next_chunk(
        &mut self,
        frames: usize,
        cancelled: impl Fn() -> bool,
    ) -> Result<Option<AudioChunk>, Error> {
        while !self.finished && self.codes.frames() - self.emitted < frames {
            if !self.advance(&cancelled).await? {
                break;
            }
        }
        if self.codes.frames() == self.emitted {
            return Ok(None);
        }
        let start_sample =
            (self.emitted - self.codes.reference_frames) * self.decoder_contract.samples_per_frame;
        let samples = decode_range(
            &mut self.sequence.generator.networks,
            self.decoder_contract,
            &self.codes,
            self.emitted,
            self.codes.frames(),
        )
        .await?;
        self.emitted = self.codes.frames();
        Ok(Some(AudioChunk {
            samples,
            sample_rate: self.decoder_contract.sample_rate,
            start_sample,
        }))
    }

    pub fn into_generation(self) -> Result<Generation, Error> {
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

/// Model metadata and text/audio frontend shared by every execution backend.
pub struct Assets {
    pub weight_bits: u8,
    pub config: Config,
    pub encoder_contract: AudioContract,
    pub decoder_contract: AudioContract,
    pub filters: crate::mel::Filters,
    pub tokenizer: tokenizers::Tokenizer,
}
