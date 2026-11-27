//! Web Worker bindings. Reference audio and generated PCM never leave the worker.
use crate::{
    burn::{
        AudioConfig, Burn, BurnCodec, Conditioning, SpeechConfig, WeightSource,
        burn_float_element_bytes, normalize_codebook_bytes, parse_safetensors_dtype,
        permute_conv_rows, raw_element_bytes, raw_to_burn_dtype, upload_persistent,
        upload_persistent_int,
    },
    inference::{Assets, AudioContract, Engine, Error, Reference, Sampling, SynthesisOptions},
    sampling::{Multinomial, Parameters, RepetitionPenalty},
    streaming::{self, Clock, Options},
};
use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use js_sys::{Array, Float32Array, Function, Object, Promise, Reflect, Uint8Array, Uint32Array};
use rand::{SeedableRng, rngs::StdRng};
use safetensors::Dtype;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    cell::Cell, collections::HashMap, fmt::Write as _, num::NonZeroUsize, rc::Rc, time::Duration,
};
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

#[derive(Deserialize)]
struct IndexedTensor {
    shard: u32,
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    length: u64,
}

const CONV_STREAM_CHUNK_BYTES: usize = 2 * 1024 * 1024;

/// OPFS-backed weight reader that pre-packs convolutions, joined projections,
/// and codebooks on CPU before uploading one persistent GPU buffer per weight.
struct StreamingWeights {
    device: Device,
    dtype: DType,
    tensors: HashMap<String, IndexedTensor>,
    read_into: Function,
    packed_bits: Option<u8>,
}

impl StreamingWeights {
    fn new(
        device: Device,
        dtype: DType,
        index_json: &str,
        read_into: Function,
    ) -> Result<Self, crate::burn::Error> {
        let tensors: HashMap<String, IndexedTensor> =
            serde_json::from_str(index_json).map_err(|err| crate::burn::Error::Tensor {
                name: "index".into(),
                reason: err.to_string(),
            })?;
        let mut packed_bits = None;
        for (name, entry) in &tensors {
            if entry.dtype == "U32" {
                if name.ends_with(".qweight_i4") {
                    packed_bits = Some(4);
                } else if name.ends_with(".qweight_i8") && packed_bits != Some(4) {
                    packed_bits = Some(8);
                }
            }
        }
        Ok(Self {
            device,
            dtype,
            tensors,
            read_into,
            packed_bits,
        })
    }

    fn transformer_weight_bits(&self) -> u8 {
        if let Some(bits) = self.packed_bits {
            bits
        } else if self.dtype == DType::F16 {
            16
        } else {
            32
        }
    }

    fn entry(&self, name: &str) -> Result<&IndexedTensor, crate::burn::Error> {
        self.tensors
            .get(name)
            .ok_or_else(|| crate::burn::Error::Tensor {
                name: name.into(),
                reason: "missing tensor".into(),
            })
    }

    fn validate_entry(
        &self,
        name: &str,
        entry: &IndexedTensor,
    ) -> Result<(Dtype, usize), crate::burn::Error> {
        let source_dtype = parse_safetensors_dtype(&entry.dtype)?;
        let elem_bytes = raw_element_bytes(source_dtype)?;
        let elements = entry
            .shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| crate::burn::Error::Tensor {
                name: name.into(),
                reason: format!("tensor shape overflow: {:?}", entry.shape),
            })?;
        let expected =
            elements
                .checked_mul(elem_bytes)
                .ok_or_else(|| crate::burn::Error::Tensor {
                    name: name.into(),
                    reason: format!("tensor byte length overflow: {:?}", entry.shape),
                })?;
        if entry.length as usize != expected {
            return Err(crate::burn::Error::Tensor {
                name: name.into(),
                reason: format!(
                    "expected {expected} bytes for shape {:?} ({source_dtype:?}), got {}",
                    entry.shape, entry.length
                ),
            });
        }
        Ok((source_dtype, expected))
    }

    fn read_slice(
        &self,
        shard: u32,
        offset: u64,
        dest: &mut [u8],
    ) -> Result<(), crate::burn::Error> {
        if dest.is_empty() {
            return Ok(());
        }
        // SAFETY: `read_into` only runs synchronous OPFS `SyncAccessHandle.read`
        // into `view` and never re-enters Wasm, so linear memory cannot grow while
        // `view` is alive.
        let view = unsafe { Uint8Array::view_mut_raw(dest.as_mut_ptr(), dest.len()) };
        self.read_into
            .call3(
                &JsValue::NULL,
                &JsValue::from(shard),
                &JsValue::from(offset as f64),
                &view,
            )
            .map_err(|err| {
                crate::burn::Error::Execution(format!("streaming shard read failed: {err:?}"))
            })?;
        Ok(())
    }

    fn read_tensor_bytes(
        &self,
        name: &str,
    ) -> Result<(Dtype, Vec<usize>, Vec<u8>), crate::burn::Error> {
        let entry = self.entry(name)?;
        let (source_dtype, expected) = self.validate_entry(name, entry)?;
        let mut bytes = vec![0u8; expected];
        self.read_slice(entry.shard, entry.start, &mut bytes)?;
        Ok((source_dtype, entry.shape.clone(), bytes))
    }
}

impl WeightSource for StreamingWeights {
    fn device(&self) -> &Device {
        &self.device
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensors.get(name).map(|entry| entry.shape.clone())
    }
    fn tensor<const D: usize>(&mut self, name: &str) -> Result<Tensor<D>, crate::burn::Error> {
        self.tensor_as(name, self.dtype)
    }
    fn tensor_as<const D: usize>(
        &mut self,
        name: &str,
        target: DType,
    ) -> Result<Tensor<D>, crate::burn::Error> {
        let (source_dtype, shape, bytes) = self.read_tensor_bytes(name)?;
        let shape: [usize; D] = shape.try_into().map_err(|_| crate::burn::Error::Tensor {
            name: name.into(),
            reason: format!("expected rank {D}"),
        })?;
        let from_dtype = raw_to_burn_dtype(name, source_dtype)?;
        let data = TensorData::from_bytes_vec(bytes, shape, from_dtype).convert_dtype(target);
        Ok(upload_persistent(&self.device, data, target))
    }
    fn optional<const D: usize>(
        &mut self,
        name: &str,
    ) -> Result<Option<Tensor<D>>, crate::burn::Error> {
        if self.tensors.contains_key(name) {
            self.tensor(name).map(Some)
        } else {
            Ok(None)
        }
    }
    fn optional_packed<const D: usize>(
        &mut self,
        name: &str,
    ) -> Result<Option<Tensor<D, Int>>, crate::burn::Error> {
        if !self.tensors.contains_key(name) {
            return Ok(None);
        }
        let (source_dtype, shape, bytes) = self.read_tensor_bytes(name)?;
        if source_dtype != Dtype::U32 {
            return Err(crate::burn::Error::Tensor {
                name: name.into(),
                reason: format!("expected U32 packed weights, got {source_dtype:?}"),
            });
        }
        let shape: [usize; D] = shape.try_into().map_err(|_| crate::burn::Error::Tensor {
            name: name.into(),
            reason: format!("expected rank {D}"),
        })?;
        let data = TensorData::from_bytes_vec(bytes, shape, DType::I32);
        Ok(Some(upload_persistent_int(&self.device, data)))
    }
    fn cat_axis0(&mut self, names: &[&str]) -> Result<Tensor<2>, crate::burn::Error> {
        if names.is_empty() {
            return Err(crate::burn::Error::Tensor {
                name: "cat".into(),
                reason: "expected at least one tensor".into(),
            });
        }
        let target_elem_bytes = burn_float_element_bytes(self.dtype)?;
        let mut total_rows = 0usize;
        let mut cols = None;
        for &name in names {
            let entry = self.entry(name)?;
            self.validate_entry(name, entry)?;
            if entry.shape.len() != 2 {
                return Err(crate::burn::Error::Tensor {
                    name: name.into(),
                    reason: format!("expected rank 2, got {:?}", entry.shape),
                });
            }
            let width = entry.shape[1];
            if let Some(expected) = cols
                && width != expected
            {
                return Err(crate::burn::Error::Tensor {
                    name: name.into(),
                    reason: format!("expected width {expected}, got {width}"),
                });
            }
            cols = Some(width);
            total_rows += entry.shape[0];
        }
        let cols = cols.unwrap_or(0);
        let mut out = vec![0u8; total_rows * cols * target_elem_bytes];
        let mut cursor = 0usize;
        for &name in names {
            let entry = self.entry(name)?;
            let source_dtype = parse_safetensors_dtype(&entry.dtype)?;
            let from_dtype = raw_to_burn_dtype(name, source_dtype)?;
            let expected = entry.length as usize;
            if from_dtype == self.dtype {
                self.read_slice(
                    entry.shard,
                    entry.start,
                    &mut out[cursor..cursor + expected],
                )?;
                cursor += expected;
            } else {
                let mut raw = vec![0u8; expected];
                self.read_slice(entry.shard, entry.start, &mut raw)?;
                let converted = TensorData::from_bytes_vec(raw, entry.shape.clone(), from_dtype)
                    .convert_dtype(self.dtype);
                let src = converted.as_bytes();
                out[cursor..cursor + src.len()].copy_from_slice(src);
                cursor += src.len();
            }
        }
        let data = TensorData::from_bytes_vec(out, [total_rows, cols], self.dtype);
        Ok(upload_persistent(&self.device, data, self.dtype))
    }
    fn cat_packed_axis0(
        &mut self,
        names: &[&str],
    ) -> Result<Option<Tensor<2, Int>>, crate::burn::Error> {
        if names.is_empty() {
            return Ok(None);
        }
        let present = names
            .iter()
            .filter(|name| self.tensors.contains_key(**name))
            .count();
        if present == 0 {
            return Ok(None);
        }
        if present != names.len() {
            return Err(crate::burn::Error::Tensor {
                name: names[0].into(),
                reason: "cannot join quantized and dense projections".into(),
            });
        }
        let mut total_rows = 0usize;
        let mut cols = None;
        for &name in names {
            let entry = self.entry(name)?;
            let (source_dtype, _) = self.validate_entry(name, entry)?;
            if source_dtype != Dtype::U32 {
                return Err(crate::burn::Error::Tensor {
                    name: name.into(),
                    reason: format!("expected U32 packed weights, got {source_dtype:?}"),
                });
            }
            if entry.shape.len() != 2 {
                return Err(crate::burn::Error::Tensor {
                    name: name.into(),
                    reason: format!("expected rank 2, got {:?}", entry.shape),
                });
            }
            let width = entry.shape[1];
            if let Some(expected) = cols
                && width != expected
            {
                return Err(crate::burn::Error::Tensor {
                    name: name.into(),
                    reason: format!("expected width {expected}, got {width}"),
                });
            }
            cols = Some(width);
            total_rows += entry.shape[0];
        }
        let cols = cols.unwrap_or(0);
        let mut out = vec![0u8; total_rows * cols * 4];
        let mut cursor = 0usize;
        for &name in names {
            let entry = self.entry(name)?;
            let len = entry.length as usize;
            self.read_slice(entry.shard, entry.start, &mut out[cursor..cursor + len])?;
            cursor += len;
        }
        let data = TensorData::from_bytes_vec(out, [total_rows, cols], DType::I32);
        Ok(Some(upload_persistent_int(&self.device, data)))
    }
    fn stack_1d(&mut self, names: &[&str]) -> Result<Tensor<2>, crate::burn::Error> {
        if names.is_empty() {
            return Err(crate::burn::Error::Tensor {
                name: "stack".into(),
                reason: "expected at least one tensor".into(),
            });
        }
        let target_elem_bytes = burn_float_element_bytes(self.dtype)?;
        let mut cols = None;
        for &name in names {
            let entry = self.entry(name)?;
            self.validate_entry(name, entry)?;
            if entry.shape.len() != 1 {
                return Err(crate::burn::Error::Tensor {
                    name: name.into(),
                    reason: format!("expected rank 1, got {:?}", entry.shape),
                });
            }
            let width = entry.shape[0];
            if let Some(expected) = cols
                && width != expected
            {
                return Err(crate::burn::Error::Tensor {
                    name: name.into(),
                    reason: format!("expected width {expected}, got {width}"),
                });
            }
            cols = Some(width);
        }
        let cols = cols.unwrap_or(0);
        let mut out = vec![0u8; names.len() * cols * target_elem_bytes];
        let mut cursor = 0usize;
        for &name in names {
            let entry = self.entry(name)?;
            let source_dtype = parse_safetensors_dtype(&entry.dtype)?;
            let from_dtype = raw_to_burn_dtype(name, source_dtype)?;
            let len = entry.length as usize;
            if from_dtype == self.dtype {
                self.read_slice(entry.shard, entry.start, &mut out[cursor..cursor + len])?;
                cursor += len;
            } else {
                let mut raw = vec![0u8; len];
                self.read_slice(entry.shard, entry.start, &mut raw)?;
                let converted = TensorData::from_bytes_vec(raw, entry.shape.clone(), from_dtype)
                    .convert_dtype(self.dtype);
                let src = converted.as_bytes();
                out[cursor..cursor + src.len()].copy_from_slice(src);
                cursor += src.len();
            }
        }
        let data = TensorData::from_bytes_vec(out, [names.len(), cols], self.dtype);
        Ok(upload_persistent(&self.device, data, self.dtype))
    }
    fn conv_packed(
        &mut self,
        name: &str,
        transpose: bool,
    ) -> Result<Tensor<3>, crate::burn::Error> {
        let entry = self.entry(name)?;
        let (source_dtype, total_bytes) = self.validate_entry(name, entry)?;
        if entry.shape.len() != 3 {
            return Err(crate::burn::Error::Tensor {
                name: name.into(),
                reason: format!("expected rank 3, got {:?}", entry.shape),
            });
        }
        let from_dtype = raw_to_burn_dtype(name, source_dtype)?;
        let elem_bytes = raw_element_bytes(source_dtype)?;
        let [d0, d1, k] = [entry.shape[0], entry.shape[1], entry.shape[2]];
        let (c_out, c_in) = if transpose { (d1, d0) } else { (d0, d1) };
        let mut out = vec![0u8; total_bytes];
        if total_bytes > 0 {
            let row_bytes = d1 * k * elem_bytes;
            let rows_per_chunk = (CONV_STREAM_CHUNK_BYTES / row_bytes.max(1)).max(1).min(d0);
            let mut chunk_buf = vec![0u8; rows_per_chunk * row_bytes];
            let mut outer_start = 0usize;
            while outer_start < d0 {
                let outer_count = (d0 - outer_start).min(rows_per_chunk);
                let chunk_len = outer_count * row_bytes;
                let file_offset = entry.start + (outer_start * row_bytes) as u64;
                self.read_slice(entry.shard, file_offset, &mut chunk_buf[..chunk_len])?;
                permute_conv_rows(
                    &chunk_buf[..chunk_len],
                    &mut out,
                    outer_start,
                    outer_count,
                    d0,
                    d1,
                    k,
                    elem_bytes,
                    transpose,
                );
                outer_start += outer_count;
            }
        }
        let data =
            TensorData::from_bytes_vec(out, [k, c_out, c_in], from_dtype).convert_dtype(self.dtype);
        Ok(upload_persistent(&self.device, data, self.dtype))
    }
    fn normalized_codebook(
        &mut self,
        sum_name: &str,
        usage_name: &str,
        eps: f32,
    ) -> Result<Tensor<2>, crate::burn::Error> {
        let (sum_dtype, sum_shape, sum_bytes) = self.read_tensor_bytes(sum_name)?;
        let (usage_dtype, usage_shape, usage_bytes) = self.read_tensor_bytes(usage_name)?;
        let sum_shape: [usize; 2] =
            sum_shape
                .try_into()
                .map_err(|_| crate::burn::Error::Tensor {
                    name: sum_name.into(),
                    reason: "expected rank 2".into(),
                })?;
        let usage_shape: [usize; 1] =
            usage_shape
                .try_into()
                .map_err(|_| crate::burn::Error::Tensor {
                    name: usage_name.into(),
                    reason: "expected rank 1".into(),
                })?;
        let data = normalize_codebook_bytes(
            sum_name,
            sum_dtype,
            sum_shape,
            sum_bytes,
            usage_name,
            usage_dtype,
            usage_shape,
            usage_bytes,
            eps,
            self.dtype,
        )?;
        Ok(upload_persistent(&self.device, data, self.dtype))
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
    pub fn load_conditioning_streaming(
        &self,
        index_json: &str,
        read_into: Function,
        config: &str,
        codec: &str,
        mel: &str,
    ) -> Result<BrowserConditioner, JsValue> {
        let mut weights =
            StreamingWeights::new(self.device.clone(), DType::F32, index_json, read_into)
                .map_err(error)?;
        let speech: SpeechConfig = serde_json::from_str(config).map_err(error)?;
        let audio: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let filters: crate::mel::Filters = serde_json::from_str(mel).map_err(error)?;
        let spectrogram = crate::mel::Spectrogram::new(filters).map_err(error)?;
        if audio.input_sample_rate != spectrogram.sample_rate() {
            return Err(error("mel filter sample rate differs from audio encoder"));
        }
        let conditioning = Conditioning::load(&mut weights, &speech, &audio).map_err(error)?;
        drop(weights);
        let _ = self.device.flush();
        self.device.memory_cleanup();
        Ok(BrowserConditioner {
            conditioning,
            spectrogram,
            sample_rate: audio.input_sample_rate,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn load_synthesis_streaming(
        &self,
        fp16: bool,
        index_json: &str,
        read_into: Function,
        config: &str,
        codec: &str,
        mel: &str,
        tokenizer: &[u8],
        context: usize,
        seed: u64,
    ) -> Result<BrowserVoice, JsValue> {
        let precision = if fp16 { DType::F16 } else { DType::F32 };
        let mut weights =
            StreamingWeights::new(self.device.clone(), precision, index_json, read_into)
                .map_err(error)?;
        let speech: SpeechConfig = serde_json::from_str(config).map_err(error)?;
        let audio: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let audio_meta: AudioConfig = serde_json::from_str(codec).map_err(error)?;
        let weight_bits = weights.transformer_weight_bits();
        let (backend, codec_runner) = Burn::load_synthesis(
            &mut weights,
            speech,
            audio,
            NonZeroUsize::new(context).ok_or_else(|| error("context must be nonzero"))?,
        )
        .map_err(error)?;
        // Drop the streaming index before parsing tokenizer.json so Wasm reuses its heap.
        drop(weights);
        let _ = self.device.flush();
        self.device.memory_cleanup();
        let engine = build_engine(
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
    pub fn reclaim_memory(&self) {
        let _ = self.device.flush();
        self.device.memory_cleanup();
    }
}

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
