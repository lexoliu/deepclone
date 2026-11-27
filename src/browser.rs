//! Web Worker bindings. Reference audio and generated PCM never leave the worker.
use crate::{
    burn::{AudioConfig, Burn, BurnCodec, Conditioning, SpeechConfig, UploadedWeights},
    inference::{Assets, AudioContract, Engine, Error, Reference, Sampling, SynthesisOptions},
    sampling::{Multinomial, Parameters, RepetitionPenalty},
    streaming::{self, Clock, Options},
};
use burn::tensor::{DType, Device, Tensor};
use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect, Uint32Array};
use rand::{SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};
use std::{cell::Cell, fmt::Write as _, num::NonZeroUsize, rc::Rc, time::Duration};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

// This module is called repeatedly through wasm-bindgen, including externref helpers.
// Explicit initialization prevents wasm-ld's command-style constructor wrappers from
// rerunning inventory/Pliron registration on every exported call.
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn report_panic(message: &str);
}

#[wasm_bindgen(start)]
pub fn initialize() {
    std::panic::set_hook(Box::new(|info| report_panic(&info.to_string())));
    unsafe extern "C" {
        fn __wasm_call_ctors();
    }
    // SAFETY: wasm-bindgen invokes its start entry once after instantiation, before
    // exposing any model operations. The function is supplied by the Wasm linker.
    unsafe { __wasm_call_ctors() };
}

type Sampler = Multinomial<StdRng>;
type Cloner = Engine<Burn, Sampler, Sampler>;
fn error(value: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&value.to_string()).into()
}

#[wasm_bindgen]
pub struct Sha256Hasher {
    hasher: Sha256,
}
impl Default for Sha256Hasher {
    fn default() -> Self {
        Self::new()
    }
}
#[wasm_bindgen]
impl Sha256Hasher {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            hasher: Sha256::new(),
        }
    }
    pub fn update(&mut self, chunk: &[u8]) {
        self.hasher.update(chunk);
    }
    pub fn finalize(self) -> String {
        let digest = self.hasher.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            let _ = write!(&mut hex, "{byte:02x}");
        }
        hex
    }
}

#[wasm_bindgen]
pub struct BrowserRuntime {
    device: Device,
}
#[wasm_bindgen]
impl BrowserRuntime {
    pub async fn create(configuration: &str) -> Result<BrowserRuntime, JsValue> {
        use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
        let configuration = serde_json::from_str(configuration).map_err(error)?;
        if !CubeClRuntimeConfig::try_set(configuration) {
            return Err(error(
                "GPU configuration must be set before runtime initialization",
            ));
        }
        let device = Device::wgpu_options().init_async().await.map_err(error)?;
        Ok(Self { device })
    }
    pub fn builder(&self, fp16: bool) -> ModelBuilder {
        let precision = if fp16 { DType::F16 } else { DType::F32 };
        ModelBuilder {
            weights: UploadedWeights::new(self.device.clone(), precision),
            precision,
        }
    }
    pub fn reclaim_memory(&self) {
        let _ = self.device.flush();
        self.device.memory_cleanup();
    }
}
#[wasm_bindgen]
pub struct ModelBuilder {
    weights: UploadedWeights,
    precision: DType,
}
impl ModelBuilder {
    #[allow(clippy::too_many_arguments)]
    fn build_engine(
        backend: Burn,
        codec: BurnCodec,
        weight_bits: u8,
        config: &str,
        audio: &AudioConfig,
        mel: &str,
        tokenizer: &[u8],
        seed: u64,
    ) -> Result<Cloner, JsValue> {
        let encoder_contract = AudioContract {
            capacity: usize::MAX,
            sample_rate: audio.input_sample_rate,
            samples_per_frame: audio.encode_downsample_rate,
            code_groups: audio.encoder_valid_num_quantizers,
        };
        // Official chunked decoding needs 300 frames plus 25 frames of context.
        let decoder_contract = AudioContract {
            capacity: crate::inference::CODEC_CAPACITY,
            sample_rate: audio.output_sample_rate,
            samples_per_frame: audio.decode_upsample_rate,
            code_groups: audio.encoder_valid_num_quantizers,
        };
        let parameters = Parameters::new(0.9, NonZeroUsize::new(50), 1.0).map_err(error)?;
        let sampling = Sampling {
            talker: Multinomial::new(StdRng::seed_from_u64(seed), parameters),
            predictor: Multinomial::new(StdRng::seed_from_u64(seed.wrapping_add(1)), parameters),
            repetition: RepetitionPenalty::new(1.05).map_err(error)?,
            minimum_frames: 2,
        };
        Engine::new(
            backend,
            codec,
            Assets {
                weight_bits,
                config: serde_json::from_str(config).map_err(error)?,
                encoder_contract,
                decoder_contract,
                filters: serde_json::from_str(mel).map_err(error)?,
                tokenizer: tokenizers::Tokenizer::from_bytes(tokenizer).map_err(error)?,
            },
            sampling,
        )
        .map_err(error)
    }
}
#[wasm_bindgen]
impl ModelBuilder {
    pub fn upload(&mut self, bytes: &[u8], conditioning: bool) -> Result<(), JsValue> {
        self.weights
            .upload(
                bytes,
                if conditioning {
                    DType::F32
                } else {
                    self.precision
                },
            )
            .map_err(error)
    }
    pub fn upload_tensor(
        &mut self,
        name: String,
        dtype: &str,
        shape: Vec<usize>,
        bytes: Vec<u8>,
        conditioning: bool,
    ) -> Result<(), JsValue> {
        let source_dtype = crate::burn::parse_safetensors_dtype(dtype).map_err(error)?;
        self.weights
            .upload_tensor(
                name,
                source_dtype,
                shape,
                bytes,
                if conditioning {
                    DType::F32
                } else {
                    self.precision
                },
            )
            .map_err(error)
    }
    pub fn finish(
        mut self,
        config: &str,
        codec: &str,
        mel: &str,
        tokenizer: &[u8],
        context: usize,
        seed: u64,
    ) -> Result<BrowserVoice, JsValue> {
        let speech: SpeechConfig = serde_json::from_str(config).map_err(error)?;
        let audio: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let weight_bits = self.weights.transformer_weight_bits();
        let audio_meta: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let (backend, codec_runner) = Burn::load_unified(
            &mut self.weights,
            speech,
            audio,
            NonZeroUsize::new(context).ok_or_else(|| error("context must be nonzero"))?,
        )
        .map_err(error)?;
        let engine = Self::build_engine(
            backend,
            codec_runner,
            weight_bits,
            config,
            &audio_meta,
            mel,
            tokenizer,
            seed,
        )?;
        Ok(BrowserVoice {
            engine,
            reference: None,
        })
    }
    pub fn finish_conditioning(
        mut self,
        config: &str,
        codec: &str,
        mel: &str,
    ) -> Result<BrowserConditioner, JsValue> {
        let speech: SpeechConfig = serde_json::from_str(config).map_err(error)?;
        let audio: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let filters: crate::mel::Filters = serde_json::from_str(mel).map_err(error)?;
        let spectrogram = crate::mel::Spectrogram::new(filters).map_err(error)?;
        if audio.input_sample_rate != spectrogram.sample_rate() {
            return Err(error("mel filter sample rate differs from audio encoder"));
        }
        let conditioning = Conditioning::load(&mut self.weights, &speech, &audio).map_err(error)?;
        Ok(BrowserConditioner {
            conditioning,
            spectrogram,
            sample_rate: audio.input_sample_rate,
        })
    }
    pub fn finish_synthesis(
        mut self,
        config: &str,
        codec: &str,
        mel: &str,
        tokenizer: &[u8],
        context: usize,
        seed: u64,
    ) -> Result<BrowserVoice, JsValue> {
        let speech: SpeechConfig = serde_json::from_str(config).map_err(error)?;
        let audio: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let audio_meta: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let weight_bits = self.weights.transformer_weight_bits();
        let (backend, codec_runner) = Burn::load_synthesis(
            &mut self.weights,
            speech,
            audio,
            NonZeroUsize::new(context).ok_or_else(|| error("context must be nonzero"))?,
        )
        .map_err(error)?;
        let engine = Self::build_engine(
            backend,
            codec_runner,
            weight_bits,
            config,
            &audio_meta,
            mel,
            tokenizer,
            seed,
        )?;
        Ok(BrowserVoice {
            engine,
            reference: None,
        })
    }
}

#[wasm_bindgen]
pub struct BrowserConditioner {
    conditioning: Conditioning,
    spectrogram: crate::mel::Spectrogram,
    sample_rate: u32,
}
#[wasm_bindgen]
impl BrowserConditioner {
    pub async fn extract(&self, samples: Vec<f32>, sample_rate: u32) -> Result<JsValue, JsValue> {
        if sample_rate != self.sample_rate {
            return Err(error("reference sample rate differs from model"));
        }
        let mel = self.spectrogram.compute(&samples).map_err(error)?;
        let channels = self.spectrogram.channels();
        let (speaker, speaker_shape, codes) = self
            .conditioning
            .extract(&samples, &mel, channels)
            .await
            .map_err(error)?;
        let frames = codes.len();
        let groups = codes.first().map_or(0, Vec::len);
        let flat_codes: Vec<u32> = codes.into_iter().flatten().collect();
        let result = Object::new();
        Reflect::set(
            &result,
            &JsValue::from_str("speaker"),
            &Float32Array::from(speaker.as_slice()),
        )?;
        let shape = Array::new();
        for dim in speaker_shape {
            shape.push(&JsValue::from(dim as u32));
        }
        Reflect::set(&result, &JsValue::from_str("speakerShape"), &shape)?;
        Reflect::set(
            &result,
            &JsValue::from_str("codes"),
            &Uint32Array::from(flat_codes.as_slice()),
        )?;
        Reflect::set(
            &result,
            &JsValue::from_str("groups"),
            &JsValue::from(groups as u32),
        )?;
        Reflect::set(
            &result,
            &JsValue::from_str("frames"),
            &JsValue::from(frames as u32),
        )?;
        Ok(result.into())
    }
}

#[wasm_bindgen]
pub struct TextInput {
    sender: async_channel::Sender<Result<String, Error>>,
    receiver: async_channel::Receiver<Result<String, Error>>,
    cancelled: Rc<Cell<bool>>,
}
#[wasm_bindgen]
impl TextInput {
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Result<TextInput, JsValue> {
        let capacity =
            NonZeroUsize::new(capacity).ok_or_else(|| error("input capacity must be nonzero"))?;
        let (sender, receiver) = async_channel::bounded(capacity.get());
        Ok(Self {
            sender,
            receiver,
            cancelled: Rc::new(Cell::new(false)),
        })
    }
    pub async fn push(&self, text: String) -> Result<(), JsValue> {
        self.sender.send(Ok(text)).await.map_err(error)
    }
    pub fn finish(&self) {
        self.sender.close();
    }
    pub fn cancel(&self) {
        self.cancelled.set(true);
        self.sender.close();
    }
}
#[wasm_bindgen]
pub struct BrowserVoice {
    engine: Cloner,
    reference: Option<Reference<Tensor<3>>>,
}
#[wasm_bindgen]
impl BrowserVoice {
    pub async fn reference(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
        transcript: String,
    ) -> Result<(), JsValue> {
        self.reference = Some(
            self.engine
                .reference(&samples, sample_rate, &transcript)
                .await
                .map_err(error)?,
        );
        Ok(())
    }
    pub async fn set_reference_conditioning(
        &mut self,
        speaker: Vec<f32>,
        speaker_shape: Vec<usize>,
        codes: Vec<u32>,
        groups: usize,
        transcript: String,
    ) -> Result<(), JsValue> {
        let shape: [usize; 3] = speaker_shape
            .try_into()
            .map_err(|_| error("speaker shape must have rank 3"))?;
        if groups == 0 || !codes.len().is_multiple_of(groups) {
            return Err(error("conditioning codes shape is invalid"));
        }
        let frames: Vec<Vec<u32>> = codes.chunks_exact(groups).map(|row| row.to_vec()).collect();
        let speaker = self
            .engine
            .backend_mut()
            .speaker_tensor(speaker, shape)
            .map_err(error)?;
        self.reference = Some(
            self.engine
                .reference_from_parts(speaker, frames, &transcript)
                .await
                .map_err(error)?,
        );
        Ok(())
    }
    pub fn required_text_tokens(
        &self,
        text: &str,
        maximum_frames: usize,
    ) -> Result<Uint32Array, JsValue> {
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| error("prepare reference audio first"))?;
        let options = SynthesisOptions::new(
            NonZeroUsize::new(maximum_frames)
                .ok_or_else(|| error("frame budget must be nonzero"))?,
        );
        let tokens = self
            .engine
            .required_text_tokens(reference, text, &options)
            .map_err(error)?;
        Ok(Uint32Array::from(tokens.as_slice()))
    }
    pub fn set_text_embeddings(
        &mut self,
        token_ids: Vec<u32>,
        dtype: &str,
        bytes: Vec<u8>,
        width: usize,
    ) -> Result<(), JsValue> {
        let source_dtype = crate::burn::parse_safetensors_dtype(dtype).map_err(error)?;
        self.engine
            .backend_mut()
            .set_text_embeddings(&token_ids, source_dtype, bytes, width)
            .map_err(error)
    }
    pub async fn run(
        &mut self,
        input: &TextInput,
        language: String,
        maximum_frames: usize,
        chunk_frames: usize,
        on_audio: Function,
    ) -> Result<(), JsValue> {
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| error("prepare reference audio first"))?;
        let mut options = Options::new(
            NonZeroUsize::new(maximum_frames)
                .ok_or_else(|| error("frame budget must be nonzero"))?,
        );
        options.audio_chunk_frames =
            NonZeroUsize::new(chunk_frames).ok_or_else(|| error("chunk frames must be nonzero"))?;
        let clock = BrowserClock(web_time::Instant::now());
        streaming::run(
            &mut self.engine,
            reference,
            input.receiver.clone(),
            &language,
            options,
            streaming::Execution {
                clock,
                cancelled: || input.cancelled.get(),
            },
            async |chunk| {
                let samples = Float32Array::from(chunk.samples.as_slice());
                let result = on_audio
                    .call2(&JsValue::NULL, &samples, &JsValue::from(chunk.sample_rate))
                    .map_err(|e| Error::Callback(format!("{e:?}")))?;
                JsFuture::from(Promise::resolve(&result))
                    .await
                    .map_err(|e| Error::Callback(format!("{e:?}")))?;
                Ok(())
            },
        )
        .await
        .map_err(error)?;
        Ok(())
    }
}
struct BrowserClock(web_time::Instant);
impl Clock for BrowserClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
    async fn sleep(&self, duration: Duration) {
        // gloo's timer accepts u32 milliseconds; split larger caller durations.
        let mut millis = duration.as_millis();
        while millis > 0 {
            let part = millis.min(u32::MAX as u128) as u32;
            gloo_timers::future::TimeoutFuture::new(part).await;
            millis -= part as u128;
        }
    }
}
