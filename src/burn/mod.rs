//! Qwen tensor execution on Burn's WGPU runtime.
mod attention;
mod codec_decoder;
mod conv;
mod fused;
mod gemv;
mod layers;
mod projection;
mod quant_gemv;
pub mod sampling;
mod sort;
pub use codec_decoder::{CodecDecoder, CodecDecoderConfig};
mod codec_encoder;
pub use codec_encoder::{CodecEncoder, EncoderConfig};
mod audio_transformer;
mod speaker;
mod transformer;
pub use speaker::{Speaker, SpeakerConfig};

use burn::tensor::{DType, Device, Int, Tensor, TensorData};
use safetensors::{Dtype, SafeTensors};
pub use transformer::{Decoder, DecoderConfig};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Weights(#[from] safetensors::SafeTensorError),
    #[error("tensor {name}: {reason}")]
    Tensor { name: String, reason: String },
    #[error("GPU execution: {0}")]
    Execution(String),
    #[error("decoder context exhausted after {0} tokens")]
    Context(usize),
}

pub(crate) fn upload_persistent<const D: usize>(
    device: &Device,
    data: TensorData,
    dtype: DType,
) -> Tensor<D> {
    let slot = std::sync::Mutex::new(Some(data));
    let target = device.clone();
    device.memory_persistent_allocations((), move |()| {
        let data = slot
            .lock()
            .expect("persistent upload lock poisoned")
            .take()
            .expect("persistent upload invoked twice");
        Tensor::from_data(data, (&target, dtype))
    })
}

pub(crate) fn upload_persistent_int<const D: usize>(
    device: &Device,
    data: TensorData,
) -> Tensor<D, Int> {
    let slot = std::sync::Mutex::new(Some(data));
    let target = device.clone();
    device.memory_persistent_allocations((), move |()| {
        let data = slot
            .lock()
            .expect("persistent upload lock poisoned")
            .take()
            .expect("persistent upload invoked twice");
        Tensor::from_data(data, &target)
    })
}

pub(crate) fn zeros_persistent<const D: usize>(
    shape: [usize; D],
    device: &Device,
    dtype: DType,
) -> Tensor<D> {
    let target = device.clone();
    device.memory_persistent_allocations((), move |()| Tensor::zeros(shape, (&target, dtype)))
}

pub(crate) fn raw_element_bytes(dtype: Dtype) -> Result<usize, Error> {
    match dtype {
        Dtype::U32 | Dtype::F32 => Ok(4),
        Dtype::F16 | Dtype::BF16 => Ok(2),
        other => Err(Error::Tensor {
            name: "dtype".into(),
            reason: format!("unsupported dtype {other:?}"),
        }),
    }
}

pub(crate) fn raw_to_burn_dtype(name: &str, dtype: Dtype) -> Result<DType, Error> {
    match dtype {
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        Dtype::F32 => Ok(DType::F32),
        other => Err(Error::Tensor {
            name: name.into(),
            reason: format!("unsupported weight dtype {other:?}"),
        }),
    }
}

pub(crate) fn burn_float_element_bytes(dtype: DType) -> Result<usize, Error> {
    match dtype {
        DType::F16 | DType::BF16 => Ok(2),
        DType::F32 => Ok(4),
        other => Err(Error::Tensor {
            name: "precision".into(),
            reason: format!("unsupported execution dtype {other:?}"),
        }),
    }
}

/// Scatter a contiguous chunk of outer rows from `[d0, d1, k]` into `[k, c_out, c_in]`.
/// When `transpose` is false, `d0 = c_out` and `d1 = c_in` (`permute([2, 0, 1])`).
/// When `transpose` is true, `d0 = c_in` and `d1 = c_out` (`permute([2, 1, 0])`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn permute_conv_rows(
    src: &[u8],
    dst: &mut [u8],
    outer_start: usize,
    outer_count: usize,
    d0: usize,
    d1: usize,
    k: usize,
    elem_bytes: usize,
    transpose: bool,
) {
    let (c_out, c_in) = if transpose { (d1, d0) } else { (d0, d1) };
    for local_outer in 0..outer_count {
        let outer = outer_start + local_outer;
        for inner in 0..d1 {
            let (co, ci) = if transpose {
                (inner, outer)
            } else {
                (outer, inner)
            };
            let src_base = (local_outer * d1 + inner) * k;
            let dst_base = co * c_in + ci;
            let plane = c_out * c_in;
            for ki in 0..k {
                let s = (src_base + ki) * elem_bytes;
                let d = (ki * plane + dst_base) * elem_bytes;
                dst[d..d + elem_bytes].copy_from_slice(&src[s..s + elem_bytes]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_codebook_bytes(
    sum_name: &str,
    sum_dtype: Dtype,
    sum_shape: [usize; 2],
    sum_bytes: Vec<u8>,
    usage_name: &str,
    usage_dtype: Dtype,
    usage_shape: [usize; 1],
    usage_bytes: Vec<u8>,
    eps: f32,
    target_dtype: DType,
) -> Result<TensorData, Error> {
    let [vocab, width] = sum_shape;
    if usage_shape[0] != vocab {
        return Err(Error::Tensor {
            name: usage_name.into(),
            reason: format!(
                "codebook usage length {} does not match {} rows",
                usage_shape[0], vocab
            ),
        });
    }
    let sum_burn = raw_to_burn_dtype(sum_name, sum_dtype)?;
    let usage_burn = raw_to_burn_dtype(usage_name, usage_dtype)?;
    let mut sum = TensorData::from_bytes_vec(sum_bytes, [vocab, width], sum_burn)
        .convert_dtype(DType::F32)
        .try_to_vec::<f32>()
        .map_err(|error| Error::Execution(format!("{sum_name}: {error:?}")))?;
    let usage = TensorData::from_bytes_vec(usage_bytes, [vocab], usage_burn)
        .convert_dtype(DType::F32)
        .try_to_vec::<f32>()
        .map_err(|error| Error::Execution(format!("{usage_name}: {error:?}")))?;
    for (row, &count) in sum.chunks_exact_mut(width).zip(&usage) {
        let denom = count.max(eps);
        for value in row {
            *value /= denom;
        }
    }
    Ok(TensorData::new(sum, [vocab, width]).convert_dtype(target_dtype))
}

/// Borrowed, validated safetensors data. Device tensors own their uploaded storage.
/// The same byte-oriented loader works with native files and browser downloads.
pub struct Weights<'a> {
    tensors: SafeTensors<'a>,
    device: Device,
    dtype: DType,
}
impl<'a> Weights<'a> {
    pub fn new(bytes: &'a [u8], device: Device, dtype: DType) -> Result<Self, Error> {
        if !matches!(dtype, DType::F16 | DType::F32) {
            return Err(Error::Tensor {
                name: "precision".into(),
                reason: "Qwen execution requires F16 or F32".into(),
            });
        }
        Ok(Self {
            tensors: SafeTensors::deserialize(bytes)?,
            device,
            dtype,
        })
    }
    pub fn tensor<const D: usize>(&self, name: &str) -> Result<Tensor<D>, Error> {
        self.tensor_as(name, self.dtype)
    }
    pub fn tensor_as<const D: usize>(&self, name: &str, target: DType) -> Result<Tensor<D>, Error> {
        let source = self.tensors.tensor(name)?;
        if source.shape().len() != D {
            return Err(Error::Tensor {
                name: name.into(),
                reason: format!("expected rank {D}, got {:?}", source.shape()),
            });
        }
        let dtype = raw_to_burn_dtype(name, source.dtype())?;
        let data = TensorData::from_bytes_vec(source.data().to_vec(), source.shape(), dtype)
            .convert_dtype(target);
        Ok(upload_persistent(&self.device, data, target))
    }
    pub fn indices<const D: usize>(&self, name: &str) -> Result<Tensor<D, Int>, Error> {
        let source = self.tensors.tensor(name)?;
        let dtype = match source.dtype() {
            Dtype::I32 => DType::I32,
            Dtype::I64 => DType::I64,
            other => {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("expected integer codes, got {other:?}"),
                });
            }
        };
        if source.shape().len() != D {
            return Err(Error::Tensor {
                name: name.into(),
                reason: format!("expected rank {D}"),
            });
        }
        let data = TensorData::from_bytes_vec(source.data().to_vec(), source.shape(), dtype)
            .convert_dtype(DType::I32);
        Ok(upload_persistent_int(&self.device, data))
    }
    fn optional<const D: usize>(&self, name: &str) -> Result<Option<Tensor<D>>, Error> {
        if self.tensors.names().contains(&name) {
            self.tensor(name).map(Some)
        } else {
            Ok(None)
        }
    }
    fn optional_packed<const D: usize>(&self, name: &str) -> Result<Option<Tensor<D, Int>>, Error> {
        if !self.tensors.names().contains(&name) {
            return Ok(None);
        }
        let source = self.tensors.tensor(name)?;
        if source.dtype() != Dtype::U32 {
            return Err(Error::Tensor {
                name: name.into(),
                reason: format!("expected U32 packed weights, got {:?}", source.dtype()),
            });
        }
        if source.shape().len() != D {
            return Err(Error::Tensor {
                name: name.into(),
                reason: format!("expected rank {D}, got {:?}", source.shape()),
            });
        }
        let data = TensorData::from_bytes_vec(source.data().to_vec(), source.shape(), DType::I32);
        Ok(Some(upload_persistent_int(&self.device, data)))
    }
}

pub async fn read<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f32>, Error> {
    tensor
        .cast(DType::F32)
        .into_data_async()
        .await
        .map_err(|e| Error::Execution(e.to_string()))?
        .try_into_vec::<f32>()
        .map_err(|e| Error::Execution(e.to_string()))
}

mod model;
pub use model::{AudioConfig, Burn, BurnCodec, Conditioning, SpeechConfig};

/// Parse a safetensors header dtype label into its typed representation.
pub fn parse_safetensors_dtype(label: &str) -> Result<Dtype, Error> {
    match label {
        "F32" => Ok(Dtype::F32),
        "F16" => Ok(Dtype::F16),
        "BF16" => Ok(Dtype::BF16),
        "U32" => Ok(Dtype::U32),
        other => Err(Error::Tensor {
            name: "dtype".into(),
            reason: format!("unsupported safetensors dtype {other}"),
        }),
    }
}

/// Weight providers separate model construction from file or network storage.
pub trait WeightSource {
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;
    fn tensor<const D: usize>(&mut self, name: &str) -> Result<Tensor<D>, Error>;
    fn tensor_as<const D: usize>(&mut self, name: &str, dtype: DType) -> Result<Tensor<D>, Error> {
        let tensor = self.tensor::<D>(name)?;
        Ok(if tensor.dtype() == dtype {
            tensor
        } else {
            tensor.cast(dtype)
        })
    }
    fn tensor_shape(&self, _name: &str) -> Option<Vec<usize>> {
        None
    }
    fn optional<const D: usize>(&mut self, name: &str) -> Result<Option<Tensor<D>>, Error>;
    fn optional_packed<const D: usize>(
        &mut self,
        _name: &str,
    ) -> Result<Option<Tensor<D, Int>>, Error> {
        Ok(None)
    }
    fn cat_axis0(&mut self, names: &[&str]) -> Result<Tensor<2>, Error> {
        if names.is_empty() {
            return Err(Error::Tensor {
                name: "cat".into(),
                reason: "expected at least one tensor".into(),
            });
        }
        let tensors = names
            .iter()
            .map(|name| self.tensor::<2>(name))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Tensor::cat(tensors, 0))
    }
    fn cat_packed_axis0(&mut self, names: &[&str]) -> Result<Option<Tensor<2, Int>>, Error> {
        if names.is_empty() {
            return Ok(None);
        }
        let mut present = Vec::with_capacity(names.len());
        for &name in names {
            present.push((name, self.optional_packed::<2>(name)?));
        }
        if present.iter().all(|(_, item)| item.is_none()) {
            return Ok(None);
        }
        let mut tensors = Vec::with_capacity(names.len());
        for (name, item) in present {
            let Some(tensor) = item else {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: "cannot join quantized and dense projections".into(),
                });
            };
            tensors.push(tensor);
        }
        Ok(Some(Tensor::cat(tensors, 0)))
    }
    fn stack_1d(&mut self, names: &[&str]) -> Result<Tensor<2>, Error> {
        if names.is_empty() {
            return Err(Error::Tensor {
                name: "stack".into(),
                reason: "expected at least one tensor".into(),
            });
        }
        let tensors = names
            .iter()
            .map(|name| Ok(self.tensor::<1>(name)?.unsqueeze_dim(0)))
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Tensor::cat(tensors, 0))
    }
    fn conv_packed(&mut self, name: &str, transpose: bool) -> Result<Tensor<3>, Error> {
        let weight = self.tensor::<3>(name)?;
        let order = if transpose { [2, 1, 0] } else { [2, 0, 1] };
        Ok(conv::contiguous(weight.permute(order)))
    }
    fn normalized_codebook(
        &mut self,
        sum_name: &str,
        usage_name: &str,
        eps: f32,
    ) -> Result<Tensor<2>, Error> {
        let raw_sum = self.tensor::<2>(sum_name)?;
        let dtype = raw_sum.dtype();
        let sum = raw_sum.cast(DType::F32);
        let count = self
            .tensor::<1>(usage_name)?
            .cast(DType::F32)
            .clamp_min(eps)
            .unsqueeze_dim(1);
        Ok((sum / count).cast(dtype))
    }
}
impl WeightSource for Weights<'_> {
    fn device(&self) -> &Device {
        &self.device
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn tensor<const D: usize>(&mut self, name: &str) -> Result<Tensor<D>, Error> {
        Weights::tensor(self, name)
    }
    fn tensor_as<const D: usize>(&mut self, name: &str, dtype: DType) -> Result<Tensor<D>, Error> {
        Weights::tensor_as(self, name, dtype)
    }
    fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensors
            .tensor(name)
            .ok()
            .map(|view| view.shape().to_vec())
    }
    fn optional<const D: usize>(&mut self, name: &str) -> Result<Option<Tensor<D>>, Error> {
        Weights::optional(self, name)
    }
    fn optional_packed<const D: usize>(
        &mut self,
        name: &str,
    ) -> Result<Option<Tensor<D, Int>>, Error> {
        Weights::optional_packed(self, name)
    }
    fn cat_axis0(&mut self, names: &[&str]) -> Result<Tensor<2>, Error> {
        if names.is_empty() {
            return Err(Error::Tensor {
                name: "cat".into(),
                reason: "expected at least one tensor".into(),
            });
        }
        let target_elem_bytes = burn_float_element_bytes(self.dtype)?;
        let mut total_rows = 0usize;
        let mut cols = None;
        for &name in names {
            let source = self.tensors.tensor(name)?;
            if source.shape().len() != 2 {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("expected rank 2, got {:?}", source.shape()),
                });
            }
            let width = source.shape()[1];
            if let Some(expected) = cols
                && width != expected
            {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("expected width {expected}, got {width}"),
                });
            }
            cols = Some(width);
            total_rows += source.shape()[0];
        }
        let cols = cols.unwrap_or(0);
        let mut out = Vec::with_capacity(total_rows * cols * target_elem_bytes);
        for &name in names {
            let source = self.tensors.tensor(name)?;
            let from_dtype = raw_to_burn_dtype(name, source.dtype())?;
            if from_dtype == self.dtype {
                out.extend_from_slice(source.data());
            } else {
                let converted =
                    TensorData::from_bytes_vec(source.data().to_vec(), source.shape(), from_dtype)
                        .convert_dtype(self.dtype);
                out.extend_from_slice(converted.as_bytes());
            }
        }
        let data = TensorData::from_bytes_vec(out, [total_rows, cols], self.dtype);
        Ok(upload_persistent(&self.device, data, self.dtype))
    }
    fn cat_packed_axis0(&mut self, names: &[&str]) -> Result<Option<Tensor<2, Int>>, Error> {
        if names.is_empty() {
            return Ok(None);
        }
        let all_names = self.tensors.names();
        let present_count = names
            .iter()
            .filter(|name| all_names.contains(*name))
            .count();
        if present_count == 0 {
            return Ok(None);
        }
        if present_count != names.len() {
            return Err(Error::Tensor {
                name: names[0].into(),
                reason: "cannot join quantized and dense projections".into(),
            });
        }
        let mut total_rows = 0usize;
        let mut cols = None;
        for &name in names {
            let source = self.tensors.tensor(name)?;
            if source.dtype() != Dtype::U32 {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("expected U32 packed weights, got {:?}", source.dtype()),
                });
            }
            if source.shape().len() != 2 {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("expected rank 2, got {:?}", source.shape()),
                });
            }
            let width = source.shape()[1];
            if let Some(expected) = cols
                && width != expected
            {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("expected width {expected}, got {width}"),
                });
            }
            cols = Some(width);
            total_rows += source.shape()[0];
        }
        let cols = cols.unwrap_or(0);
        let mut out = Vec::with_capacity(total_rows * cols * 4);
        for &name in names {
            out.extend_from_slice(self.tensors.tensor(name)?.data());
        }
        let data = TensorData::from_bytes_vec(out, [total_rows, cols], DType::I32);
        Ok(Some(upload_persistent_int(&self.device, data)))
    }
    fn stack_1d(&mut self, names: &[&str]) -> Result<Tensor<2>, Error> {
        if names.is_empty() {
            return Err(Error::Tensor {
                name: "stack".into(),
                reason: "expected at least one tensor".into(),
            });
        }
        let target_elem_bytes = burn_float_element_bytes(self.dtype)?;
        let mut cols = None;
        for &name in names {
            let source = self.tensors.tensor(name)?;
            if source.shape().len() != 1 {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("expected rank 1, got {:?}", source.shape()),
                });
            }
            let width = source.shape()[0];
            if let Some(expected) = cols
                && width != expected
            {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("expected width {expected}, got {width}"),
                });
            }
            cols = Some(width);
        }
        let cols = cols.unwrap_or(0);
        let mut out = Vec::with_capacity(names.len() * cols * target_elem_bytes);
        for &name in names {
            let source = self.tensors.tensor(name)?;
            let from_dtype = raw_to_burn_dtype(name, source.dtype())?;
            if from_dtype == self.dtype {
                out.extend_from_slice(source.data());
            } else {
                let converted =
                    TensorData::from_bytes_vec(source.data().to_vec(), source.shape(), from_dtype)
                        .convert_dtype(self.dtype);
                out.extend_from_slice(converted.as_bytes());
            }
        }
        let data = TensorData::from_bytes_vec(out, [names.len(), cols], self.dtype);
        Ok(upload_persistent(&self.device, data, self.dtype))
    }
    fn conv_packed(&mut self, name: &str, transpose: bool) -> Result<Tensor<3>, Error> {
        let source = self.tensors.tensor(name)?;
        if source.shape().len() != 3 {
            return Err(Error::Tensor {
                name: name.into(),
                reason: format!("expected rank 3, got {:?}", source.shape()),
            });
        }
        let from_dtype = raw_to_burn_dtype(name, source.dtype())?;
        let elem_bytes = raw_element_bytes(source.dtype())?;
        let [d0, d1, k] = [source.shape()[0], source.shape()[1], source.shape()[2]];
        let (c_out, c_in) = if transpose { (d1, d0) } else { (d0, d1) };
        let mut out = vec![0u8; d0 * d1 * k * elem_bytes];
        permute_conv_rows(
            source.data(),
            &mut out,
            0,
            d0,
            d0,
            d1,
            k,
            elem_bytes,
            transpose,
        );
        let data =
            TensorData::from_bytes_vec(out, [k, c_out, c_in], from_dtype).convert_dtype(self.dtype);
        Ok(upload_persistent(&self.device, data, self.dtype))
    }
    fn normalized_codebook(
        &mut self,
        sum_name: &str,
        usage_name: &str,
        eps: f32,
    ) -> Result<Tensor<2>, Error> {
        let sum = self.tensors.tensor(sum_name)?;
        let usage = self.tensors.tensor(usage_name)?;
        let sum_shape: [usize; 2] = sum.shape().try_into().map_err(|_| Error::Tensor {
            name: sum_name.into(),
            reason: format!("expected rank 2, got {:?}", sum.shape()),
        })?;
        let usage_shape: [usize; 1] = usage.shape().try_into().map_err(|_| Error::Tensor {
            name: usage_name.into(),
            reason: format!("expected rank 1, got {:?}", usage.shape()),
        })?;
        let data = normalize_codebook_bytes(
            sum_name,
            sum.dtype(),
            sum_shape,
            sum.data().to_vec(),
            usage_name,
            usage.dtype(),
            usage_shape,
            usage.data().to_vec(),
            eps,
            self.dtype,
        )?;
        Ok(upload_persistent(&self.device, data, self.dtype))
    }
}
/// GPU-resident shards. Each source byte buffer may be released after upload,
/// and each tensor is moved out as its owning layer is constructed.
#[cfg(not(target_arch = "wasm32"))]
pub struct UploadedWeights {
    device: Device,
    dtype: DType,
    packed_bits: Option<u8>,
    tensors: std::collections::BTreeMap<String, (Tensor<1>, Vec<usize>)>,
    packed: std::collections::BTreeMap<String, (Tensor<1, Int>, Vec<usize>)>,
}
#[cfg(not(target_arch = "wasm32"))]
impl UploadedWeights {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            device,
            dtype,
            packed_bits: None,
            tensors: Default::default(),
            packed: Default::default(),
        }
    }
    pub(crate) fn transformer_weight_bits(&self) -> u8 {
        if let Some(bits) = self.packed_bits {
            bits
        } else if self.dtype == DType::F16 {
            16
        } else {
            32
        }
    }
    pub fn upload_tensor(
        &mut self,
        name: String,
        source_dtype: Dtype,
        shape: Vec<usize>,
        bytes: Vec<u8>,
        dtype: DType,
    ) -> Result<(), Error> {
        if self.tensors.contains_key(&name) || self.packed.contains_key(&name) {
            return Err(Error::Tensor {
                name,
                reason: "duplicate tensor in weight shards".into(),
            });
        }
        let elements = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| Error::Tensor {
                name: name.clone(),
                reason: format!("tensor shape overflow: {shape:?}"),
            })?;
        let element_bytes = match source_dtype {
            Dtype::U32 | Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            other => {
                return Err(Error::Tensor {
                    name,
                    reason: format!("unsupported dtype {other:?}"),
                });
            }
        };
        let expected_bytes = elements
            .checked_mul(element_bytes)
            .ok_or_else(|| Error::Tensor {
                name: name.clone(),
                reason: format!("tensor byte length overflow: {shape:?}"),
            })?;
        if bytes.len() != expected_bytes {
            return Err(Error::Tensor {
                name,
                reason: format!(
                    "expected {expected_bytes} bytes for shape {shape:?} ({source_dtype:?}), got {}",
                    bytes.len()
                ),
            });
        }
        if source_dtype == Dtype::U32 {
            if name.ends_with(".qweight_i4") {
                self.packed_bits = Some(4);
            } else if name.ends_with(".qweight_i8") && self.packed_bits != Some(4) {
                self.packed_bits = Some(8);
            }
            // Store raw 32-bit packed words in Burn's native i32 integer storage;
            // kernels reinterpret bits as u32 so WGSL/Metal/Vulkan share one layout.
            let data = TensorData::from_bytes_vec(bytes, [elements], DType::I32);
            self.packed
                .insert(name, (upload_persistent_int(&self.device, data), shape));
            return Ok(());
        }
        let from_dtype = match source_dtype {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            _ => unreachable!("validated above"),
        };
        let data = TensorData::from_bytes_vec(bytes, [elements], from_dtype).convert_dtype(dtype);
        self.tensors
            .insert(name, (upload_persistent(&self.device, data, dtype), shape));
        Ok(())
    }
    pub fn upload(&mut self, bytes: &[u8], dtype: DType) -> Result<(), Error> {
        let shard = SafeTensors::deserialize(bytes)?;
        for (name, source) in shard.tensors() {
            self.upload_tensor(
                name,
                source.dtype(),
                source.shape().to_vec(),
                source.data().to_vec(),
                dtype,
            )?;
        }
        Ok(())
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl WeightSource for UploadedWeights {
    fn device(&self) -> &Device {
        &self.device
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn tensor<const D: usize>(&mut self, name: &str) -> Result<Tensor<D>, Error> {
        let (tensor, shape) = self.tensors.remove(name).ok_or_else(|| Error::Tensor {
            name: name.into(),
            reason: "missing tensor".into(),
        })?;
        let shape: [usize; D] = shape.try_into().map_err(|_| Error::Tensor {
            name: name.into(),
            reason: format!("expected rank {D}"),
        })?;
        Ok(tensor.reshape(shape))
    }
    fn optional<const D: usize>(&mut self, name: &str) -> Result<Option<Tensor<D>>, Error> {
        if self.tensors.contains_key(name) {
            self.tensor(name).map(Some)
        } else {
            Ok(None)
        }
    }
    fn optional_packed<const D: usize>(
        &mut self,
        name: &str,
    ) -> Result<Option<Tensor<D, Int>>, Error> {
        let Some((tensor, shape)) = self.packed.remove(name) else {
            return Ok(None);
        };
        let shape: [usize; D] = shape.try_into().map_err(|_| Error::Tensor {
            name: name.into(),
            reason: format!("expected rank {D}"),
        })?;
        Ok(Some(tensor.reshape(shape)))
    }
}

pub mod package;

#[cfg(test)]
mod prepack_tests {
    use super::*;

    #[test]
    fn permute_conv_rows_matches_chunked_and_full_layouts() {
        for transpose in [false, true] {
            let (d0, d1, k) = if transpose { (5, 3, 4) } else { (3, 5, 4) };
            let (c_out, c_in) = if transpose { (d1, d0) } else { (d0, d1) };
            let src: Vec<u16> = (0..(d0 * d1 * k) as u16).collect();
            let src_bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();

            let mut full = vec![0u8; src_bytes.len()];
            permute_conv_rows(&src_bytes, &mut full, 0, d0, d0, d1, k, 2, transpose);

            let mut chunked = vec![0u8; src_bytes.len()];
            let row_bytes = d1 * k * 2;
            let mut start = 0;
            while start < d0 {
                let count = (d0 - start).min(2);
                let slice = &src_bytes[start * row_bytes..(start + count) * row_bytes];
                permute_conv_rows(slice, &mut chunked, start, count, d0, d1, k, 2, transpose);
                start += count;
            }
            assert_eq!(full, chunked);

            let out: Vec<u16> = full
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            for ki in 0..k {
                for co in 0..c_out {
                    for ci in 0..c_in {
                        let expected = if transpose {
                            ((ci * c_out + co) * k + ki) as u16
                        } else {
                            ((co * c_in + ci) * k + ki) as u16
                        };
                        assert_eq!(out[(ki * c_out + co) * c_in + ci], expected);
                    }
                }
            }
        }
    }

    #[test]
    fn normalize_codebook_bytes_clamps_unused_centroids_and_converts_dtype() {
        let sum: Vec<f32> = vec![2.0, 4.0, 6.0, 1.0, 3.0, 5.0];
        let usage: Vec<f32> = vec![2.0, 0.0];
        let sum_bytes: Vec<u8> = sum.iter().flat_map(|v| v.to_le_bytes()).collect();
        let usage_bytes: Vec<u8> = usage.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = normalize_codebook_bytes(
            "sum",
            Dtype::F32,
            [2, 3],
            sum_bytes,
            "usage",
            Dtype::F32,
            [2],
            usage_bytes,
            1e-5,
            DType::F32,
        )
        .unwrap();
        let values = data.try_to_vec::<f32>().unwrap();
        assert_eq!(values[0..3], [1.0, 2.0, 3.0]);
        assert!((values[3] - 100_000.0).abs() < 1e-1);
        assert!((values[4] - 300_000.0).abs() < 1e-1);
        assert!((values[5] - 500_000.0).abs() < 1e-1);
    }
}
