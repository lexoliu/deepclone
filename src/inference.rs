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
    #[error("Core ML {operation}: {source}")]
    CoreMlOperation {
        operation: &'static str,
        #[source]
        source: ondevice_coreml::Error,
    },
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
    #[error("a single text grapheme exceeds the {0}-token utterance capacity")]
    SegmentCapacity(usize),
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

/// Per-utterance safety and text scheduling; neither bounds the total stream.
#[derive(Clone, Debug)]
pub struct SynthesisOptions {
    pub maximum_frames: NonZeroUsize,
    pub text_segment_tokens: NonZeroUsize,
    pub unstable_text_bytes: NonZeroUsize,
}
impl SynthesisOptions {
    pub fn new(maximum_frames: NonZeroUsize) -> Self {
        Self {
            maximum_frames,
            // Scheduling policy, independently adjustable by the caller.
            text_segment_tokens: NonZeroUsize::new(128).unwrap(),
            unstable_text_bytes: NonZeroUsize::new(65536).unwrap(),
        }
    }
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
    pub segments: Vec<SpeechSegment>,
}
impl Generation {
    pub fn generated_frames(&self) -> usize {
        self.codes.generated_frames()
    }
}

/// A naturally completed generation span in the continuous acoustic stream.
/// Text tokens are paired with precisely this span when conditioning its successor.
#[derive(Debug, Serialize)]
pub struct SpeechSegment {
    pub start_frame: usize,
    pub end_frame: usize,
    pub text_tokens: Vec<u32>,
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

    /// Synthesize continuous PCM while rolling generation contexts internally.
    pub async fn synthesize(
        &mut self,
        reference: &Reference<B::Hidden>,
        text: &str,
        language: &str,
        options: SynthesisOptions,
    ) -> Result<Synthesis, Error> {
        let sample_rate = self.sample_rate();
        let mut session = self.session(reference, language, options)?;
        session.push_text(text)?;
        session.finish_input()?;
        let mut samples = Vec::new();
        while !session.finished {
            if let Some(chunk) = session.next_chunk(12, || false).await? {
                samples.extend(chunk.samples);
            }
        }
        Ok(Synthesis {
            generation: session.into_generation()?,
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
        options: SynthesisOptions,
    ) -> Result<Generation, Error> {
        let mut session = self.session(reference, language, options)?;
        session.push_text(text)?;
        session.finish_input()?;
        while !session.finished {
            session.advance(&|| false).await?;
        }
        session.into_generation()
    }

    fn tags(&self, language: &str) -> Result<Vec<u32>, Error> {
        let cfg = &self.config.talker_config;
        let language = language.to_lowercase();
        if language == "auto" {
            Ok(vec![
                cfg.codec_nothink_id,
                cfg.codec_think_bos_id,
                cfg.codec_think_eos_id,
            ])
        } else {
            let id = *cfg
                .codec_language_id
                .get(&language)
                .ok_or_else(|| Error::Language(language.clone()))?;
            Ok(vec![
                cfg.codec_think_id,
                cfg.codec_think_bos_id,
                id,
                cfg.codec_think_eos_id,
            ])
        }
    }

    pub fn session<'a>(
        &'a mut self,
        reference: &'a Reference<B::Hidden>,
        language: &str,
        options: SynthesisOptions,
    ) -> Result<Session<'a, B, T, S>, Error> {
        let tags = self.tags(language)?;
        let assistant = self.tokenize("<|im_start|>assistant\n")?;
        let groups = NonZeroUsize::new(self.encoder_contract.code_groups)
            .ok_or(Error::Contract("codec must have at least one code group"))?;
        let codes = AcousticCodes::new(
            groups,
            reference.codes.iter().flatten().copied().collect(),
            reference.codes.len(),
        )?;
        self.generator.networks.reset_waveform();
        let emitted = codes.frames();
        let splitter = crate::utterances::Splitter::new(
            options.text_segment_tokens.get(),
            options.unstable_text_bytes.get(),
        );
        let text_capacity = options.unstable_text_bytes.get();
        Ok(Session {
            generator: &mut self.generator,
            config: &self.config,
            tokenizer: &self.tokenizer,
            decoder_contract: &self.decoder_contract,
            weight_bits: self.weight_bits,
            speaker: &reference.speaker,
            initial_text: &reference.text,
            assistant,
            tags,
            options,
            splitter,
            text: crate::text::TextBuffer::new(&self.tokenizer, text_capacity),
            pending: VecDeque::new(),
            active: None,
            input_finished: false,
            finished: false,
            emitted,
            codes,
            segments: Vec::new(),
            pending_error: None,
        })
    }

    /// Decode the continuous stream with the same absolute codec block boundaries.
    pub async fn decode(&mut self, codes: &AcousticCodes) -> Result<Waveform, Error> {
        self.generator.networks.reset_waveform();
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
            sample_rate: self.sample_rate(),
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

/// Contiguous mono PCM with an absolute position in the whole output stream.
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub start_sample: usize,
}

/// One text stream, one continuous codec timeline, and bounded generation contexts.
pub struct Session<'a, B: Model, T, S> {
    generator: &'a mut Generator<B, T, S>,
    config: &'a Config,
    tokenizer: &'a tokenizers::Tokenizer,
    decoder_contract: &'a AudioContract,
    weight_bits: u8,
    speaker: &'a B::Hidden,
    initial_text: &'a [u32],
    assistant: Vec<u32>,
    tags: Vec<u32>,
    options: SynthesisOptions,
    splitter: crate::utterances::Splitter,
    text: crate::text::TextBuffer<'a>,
    pending: VecDeque<String>,
    active: Option<Active<B::Hidden>>,
    input_finished: bool,
    pub(crate) finished: bool,
    emitted: usize,
    codes: AcousticCodes,
    segments: Vec<SpeechSegment>,
    pending_error: Option<Error>,
}
struct Active<H> {
    sequence: generation::Sequence<H>,
    padding: H,
    tokens: VecDeque<u32>,
    target_tokens: Vec<u32>,
    text: String,
    prompt_end: usize,
    prefix: usize,
    generated_start: usize,
    maximum_frames: usize,
    text_capacity: usize,
    input_finished: bool,
}
impl<B: Model, T: TokenSampler, S: TokenSampler> Session<'_, B, T, S> {
    pub fn push_text(&mut self, text: &str) -> Result<(), Error> {
        if self.input_finished {
            return Err(Error::Contract("text input is already finished"));
        }
        self.pending
            .extend(self.splitter.push(text, self.tokenizer)?);
        Ok(())
    }
    pub fn finish_input(&mut self) -> Result<(), Error> {
        if self.input_finished {
            return Err(Error::Contract("text input is already finished"));
        }
        self.pending.extend(self.splitter.finish(self.tokenizer)?);
        self.input_finished = true;
        Ok(())
    }
    fn open(&mut self) -> Result<(), Error> {
        let (prompt_start, prompt_end, prompt_text) = self.segments.last().map_or(
            (0, self.codes.reference_frames, self.initial_text),
            |span| {
                (
                    span.start_frame,
                    span.end_frame,
                    span.text_tokens.as_slice(),
                )
            },
        );
        let (&first, remaining) = prompt_text
            .split_first()
            .ok_or(Error::Contract("conditioning text is empty"))?;
        // Published streaming ICL prefix: assistant, language tags, speaker,
        // text BOS/codec PAD, first reference token/codec BOS, reference frames.
        let prefix = self.assistant.len() + self.tags.len() + 3;
        let prompt_frames = prompt_end - prompt_start;
        let maximum_frames = self
            .generator
            .networks
            .context_capacity()
            .checked_sub(prefix + prompt_frames)
            .filter(|&capacity| capacity > 0)
            .ok_or(Error::Contract("conditioning exhausts the Talker context"))?
            .min(self.options.maximum_frames.get());
        let text_capacity = (maximum_frames + prompt_frames)
            .checked_sub(prompt_text.len())
            .filter(|&capacity| capacity > 0)
            .ok_or(Error::Contract(
                "conditioning text exhausts the generation budget",
            ))?
            .min(self.options.text_segment_tokens.get());
        let padding = self.generator.networks.text(self.config.tts_pad_token_id)?;
        let mut prefill = Vec::new();
        for &id in &self.assistant {
            prefill.push(self.generator.networks.text(id)?);
        }
        for &id in &self.tags {
            prefill.push(B::add(&padding, &self.generator.embed_code(0, id)?)?);
        }
        prefill.push(B::add(&padding, self.speaker)?);
        let bos = self.generator.networks.text(self.config.tts_bos_token_id)?;
        prefill.push(B::add(
            &bos,
            &self
                .generator
                .embed_code(0, self.config.talker_config.codec_pad_id)?,
        )?);
        let first = self.generator.networks.text(first)?;
        prefill.push(B::add(
            &first,
            &self
                .generator
                .embed_code(0, self.config.talker_config.codec_bos_id)?,
        )?);
        let sequence = self.generator.start(prefill)?;
        self.active = Some(Active {
            sequence,
            padding,
            tokens: remaining.iter().copied().collect(),
            target_tokens: Vec::new(),
            text: String::new(),
            prompt_end,
            prefix: prompt_start,
            generated_start: self.codes.frames(),
            maximum_frames,
            text_capacity,
            input_finished: false,
        });
        Ok(())
    }
    fn close_input(&mut self) -> Result<(), Error> {
        let active = self
            .active
            .as_mut()
            .ok_or(Error::Contract("no active context"))?;
        let tail = self.text.finish()?;
        active.target_tokens.extend_from_slice(&tail);
        active.tokens.extend(tail);
        active.tokens.push_back(self.config.tts_eos_token_id);
        active.input_finished = true;
        Ok(())
    }
    fn schedule(&mut self) -> Result<(), Error> {
        if self.active.is_none() {
            while self.pending.front().is_some_and(|s| s.trim().is_empty()) {
                self.pending.pop_front();
            }
            if self.pending.is_empty() {
                self.finished = self.input_finished;
                return Ok(());
            }
            self.open()?;
        }
        while !self.active.as_ref().unwrap().input_finished {
            let Some(next) = self.pending.front() else {
                if self.input_finished {
                    self.close_input()?;
                }
                break;
            };
            let active = self.active.as_ref().unwrap();
            let mut combined = active.text.clone();
            combined.push_str(next);
            let count = self
                .tokenizer
                .encode(combined.as_str(), false)
                .map_err(|e| Error::Tokenizer(e.to_string()))?
                .len();
            if count > active.text_capacity {
                if !active.text.is_empty() {
                    self.close_input()?;
                    break;
                }
                // A previous generated prompt can leave less room than the
                // initial recording. Repartition only this unconsumed sentence.
                let mut split = crate::utterances::Splitter::new(
                    active.text_capacity,
                    self.options.unstable_text_bytes.get(),
                );
                let next = self.pending.pop_front().unwrap();
                let mut pieces = split.push(&next, self.tokenizer)?;
                pieces.extend(split.finish(self.tokenizer)?);
                for piece in pieces.into_iter().rev() {
                    self.pending.push_front(piece);
                }
                continue;
            }
            let next = self.pending.pop_front().unwrap();
            let paragraph = combined.contains("\n\n")
                || combined.contains("\r\n\r\n")
                || combined.contains('\u{2029}');
            let ids = self.text.push(&next)?;
            let active = self.active.as_mut().unwrap();
            active.text.push_str(&next);
            active.target_tokens.extend_from_slice(&ids);
            active.tokens.extend(ids);
            if paragraph {
                self.close_input()?;
            }
        }
        Ok(())
    }
    /// False means text starvation; neither padding nor EOS is fabricated.
    async fn advance(&mut self, cancelled: &impl Fn() -> bool) -> Result<bool, Error> {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        self.schedule()?;
        let Some(active) = self.active.as_mut() else {
            return Ok(false);
        };
        let mut prefix = Vec::new();
        while active.prefix < active.prompt_end {
            if cancelled() {
                return Err(Error::Cancelled);
            }
            let text = match active.tokens.pop_front() {
                Some(id) => self.generator.networks.text(id)?,
                None if active.input_finished => active.padding.clone(),
                None => break,
            };
            let group = self.codes.groups.get();
            prefix.push(self.generator.embed_frame(
                &self.codes.values[active.prefix * group..(active.prefix + 1) * group],
                &text,
            )?);
            active.prefix += 1;
        }
        let mut prefix = prefix.into_iter();
        if let Some(first) = prefix.next() {
            self.generator
                .prefill(&mut active.sequence, first, prefix)?;
        }
        if active.prefix < active.prompt_end {
            return Ok(false);
        }
        let text = match active.tokens.pop_front() {
            Some(id) => self.generator.networks.text(id)?,
            None if active.input_finished => active.padding.clone(),
            None => return Ok(false),
        };
        match self
            .generator
            .next_frame(&mut active.sequence, &text, active.maximum_frames)
            .await?
        {
            generation::Frame::Codes(frame) => self.codes.values.extend(frame),
            generation::Frame::Eos => {
                if !active.input_finished || !active.tokens.is_empty() {
                    return Err(Error::PrematureEos);
                }
                if self.codes.frames() == active.generated_start {
                    return Err(Error::Contract("generation completed without audio"));
                }
                self.segments.push(SpeechSegment {
                    start_frame: active.generated_start,
                    end_frame: self.codes.frames(),
                    text_tokens: std::mem::take(&mut active.target_tokens),
                });
                self.active = None;
            }
            generation::Frame::Limit => {
                return Err(Error::FrameBudget {
                    limit: active.maximum_frames,
                    codes: self.codes.clone(),
                });
            }
        }
        Ok(true)
    }
    pub async fn next_chunk(
        &mut self,
        frames: usize,
        cancelled: impl Fn() -> bool,
    ) -> Result<Option<AudioChunk>, Error> {
        if frames == 0 {
            return Err(Error::Contract("audio chunk frames must be nonzero"));
        }
        if let Some(error) = self.pending_error.take() {
            return Err(error);
        }
        while !self.finished && self.codes.frames() - self.emitted < frames {
            match self.advance(&cancelled).await {
                Ok(true) => {}
                Ok(false) => break,
                Err(error)
                    if !matches!(error, Error::Cancelled) && self.codes.frames() > self.emitted =>
                {
                    self.pending_error = Some(error);
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        if self.codes.frames() == self.emitted {
            return Ok(None);
        }
        let start_sample =
            (self.emitted - self.codes.reference_frames) * self.decoder_contract.samples_per_frame;
        let samples = decode_range(
            &mut self.generator.networks,
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
        if self.segments.is_empty() {
            return Err(Error::Contract("text must be nonempty"));
        }
        Ok(Generation {
            model_size: self.config.tts_model_size.clone(),
            transformer_weight_bits: self.weight_bits,
            codes: self.codes,
            segments: self.segments,
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

#[cfg(test)]
mod tests;
