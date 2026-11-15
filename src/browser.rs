//! Web Worker bindings. Reference audio and generated PCM never leave the worker.
use crate::{
    burn::{AudioConfig, Burn, SpeechConfig, UploadedWeights},
    inference::{Assets, AudioContract, Engine, Error, Reference, Sampling},
    sampling::{Multinomial, Parameters, RepetitionPenalty},
    streaming::{self, Clock, Options},
};
use burn::tensor::{DType, Device, Tensor};
use js_sys::{Float32Array, Function, Promise};
use rand::{SeedableRng, rngs::StdRng};
use std::{cell::Cell, num::NonZeroUsize, rc::Rc, time::Duration};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

type Sampler = Multinomial<StdRng>;
type Cloner = Engine<Burn, Sampler, Sampler>;
fn error(value: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&value.to_string()).into()
}

#[wasm_bindgen]
pub struct ModelBuilder {
    weights: UploadedWeights,
    precision: DType,
}
#[wasm_bindgen]
impl ModelBuilder {
    pub async fn create(fp16: bool) -> Result<ModelBuilder, JsValue> {
        let device = Device::wgpu_options().init_async().await.map_err(error)?;
        let precision = if fp16 { DType::F16 } else { DType::F32 };
        Ok(Self {
            weights: UploadedWeights::new(device, precision),
            precision,
        })
    }
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
    pub fn finish(
        self,
        config: &str,
        codec: &str,
        mel: &str,
        tokenizer: &[u8],
        context: usize,
        seed: u64,
    ) -> Result<BrowserVoice, JsValue> {
        let speech: SpeechConfig = serde_json::from_str(config).map_err(error)?;
        let audio: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let encoder_contract = AudioContract {
            capacity: usize::MAX,
            sample_rate: audio.input_sample_rate,
            samples_per_frame: audio.encode_downsample_rate,
            code_groups: audio.encoder_valid_num_quantizers,
        };
        // Official chunked decoding needs 300 frames plus 25 frames of context.
        let decoder_contract = AudioContract {
            capacity: 325,
            sample_rate: audio.output_sample_rate,
            samples_per_frame: audio.decode_upsample_rate,
            code_groups: audio.encoder_valid_num_quantizers,
        };
        let backend = Burn::load(
            &self.weights,
            &self.weights,
            &self.weights,
            speech,
            audio,
            NonZeroUsize::new(context).ok_or_else(|| error("context must be nonzero"))?,
        )
        .map_err(error)?;
        let parameters = Parameters::new(0.9, NonZeroUsize::new(50), 1.0).map_err(error)?;
        let sampling = Sampling {
            talker: Multinomial::new(StdRng::seed_from_u64(seed), parameters),
            predictor: Multinomial::new(StdRng::seed_from_u64(seed.wrapping_add(1)), parameters),
            repetition: RepetitionPenalty::new(1.05).map_err(error)?,
            minimum_frames: 2,
        };
        let engine = Engine::new(
            backend,
            Assets {
                weight_bits: if self.precision == DType::F16 { 16 } else { 32 },
                config: serde_json::from_str(config).map_err(error)?,
                encoder_contract,
                decoder_contract,
                filters: serde_json::from_str(mel).map_err(error)?,
                tokenizer: tokenizers::Tokenizer::from_bytes(tokenizer).map_err(error)?,
            },
            sampling,
        )
        .map_err(error)?;
        Ok(BrowserVoice {
            engine,
            reference: None,
        })
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
