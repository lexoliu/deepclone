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
        let source = self.tensors.tensor(name)?;
        if source.shape().len() != D {
            return Err(Error::Tensor {
                name: name.into(),
                reason: format!("expected rank {D}, got {:?}", source.shape()),
            });
        }
        let dtype = match source.dtype() {
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            Dtype::F32 => DType::F32,
            other => {
                return Err(Error::Tensor {
                    name: name.into(),
                    reason: format!("unsupported weight dtype {other:?}"),
                });
            }
        };
        let data = TensorData::from_bytes_vec(source.data().to_vec(), source.shape(), dtype)
            .convert_dtype(self.dtype);
        Ok(Tensor::from_data(data, (&self.device, self.dtype)))
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
        Ok(Tensor::from_data(data, &self.device))
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
        Ok(Some(Tensor::from_data(data, &self.device)))
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
pub use model::{AudioConfig, Burn, BurnCodec, SpeechConfig};

/// Weight providers separate model construction from file or network storage.
pub trait WeightSource {
    fn device(&self) -> &Device;
    fn dtype(&self) -> DType;
    fn tensor<const D: usize>(&self, name: &str) -> Result<Tensor<D>, Error>;
    fn optional<const D: usize>(&self, name: &str) -> Result<Option<Tensor<D>>, Error>;
    fn optional_packed<const D: usize>(
        &self,
        _name: &str,
    ) -> Result<Option<Tensor<D, Int>>, Error> {
        Ok(None)
    }
}
impl WeightSource for Weights<'_> {
    fn device(&self) -> &Device {
        &self.device
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn tensor<const D: usize>(&self, name: &str) -> Result<Tensor<D>, Error> {
        self.tensor(name)
    }
    fn optional<const D: usize>(&self, name: &str) -> Result<Option<Tensor<D>>, Error> {
        self.optional(name)
    }
    fn optional_packed<const D: usize>(&self, name: &str) -> Result<Option<Tensor<D, Int>>, Error> {
        self.optional_packed(name)
    }
}
/// GPU-resident shards. Each source byte buffer may be released after upload.
pub struct UploadedWeights {
    device: Device,
    dtype: DType,
    tensors: std::collections::BTreeMap<String, (Tensor<1>, Vec<usize>)>,
    packed: std::collections::BTreeMap<String, (Tensor<1, Int>, Vec<usize>)>,
}
impl UploadedWeights {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            device,
            dtype,
            tensors: Default::default(),
            packed: Default::default(),
        }
    }
    pub(crate) fn transformer_weight_bits(&self) -> u8 {
        if self.packed.keys().any(|name| name.ends_with(".qweight_i4")) {
            4
        } else if self.packed.keys().any(|name| name.ends_with(".qweight_i8")) {
            8
        } else if self.dtype == DType::F16 {
            16
        } else {
            32
        }
    }
    pub fn upload(&mut self, bytes: &[u8], dtype: DType) -> Result<(), Error> {
        let shard = SafeTensors::deserialize(bytes)?;
        for (name, source) in shard.tensors() {
            if self.tensors.contains_key(&name) || self.packed.contains_key(&name) {
                return Err(Error::Tensor {
                    name,
                    reason: "duplicate tensor in weight shards".into(),
                });
            }
            let shape = source.shape().to_vec();
            let elements = shape.iter().product::<usize>();
            if source.dtype() == Dtype::U32 {
                // Store raw 32-bit packed words in Burn's native i32 integer storage;
                // kernels reinterpret bits as u32 so WGSL/Metal/Vulkan share one layout.
                let data =
                    TensorData::from_bytes_vec(source.data().to_vec(), [elements], DType::I32);
                self.packed
                    .insert(name, (Tensor::from_data(data, &self.device), shape));
                continue;
            }
            let source_dtype = match source.dtype() {
                Dtype::F32 => DType::F32,
                Dtype::F16 => DType::F16,
                Dtype::BF16 => DType::BF16,
                other => {
                    return Err(Error::Tensor {
                        name,
                        reason: format!("unsupported dtype {other:?}"),
                    });
                }
            };
            let data = TensorData::from_bytes_vec(source.data().to_vec(), [elements], source_dtype)
                .convert_dtype(dtype);
            self.tensors.insert(
                name,
                (Tensor::from_data(data, (&self.device, dtype)), shape),
            );
        }
        Ok(())
    }
}
impl WeightSource for UploadedWeights {
    fn device(&self) -> &Device {
        &self.device
    }
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn tensor<const D: usize>(&self, name: &str) -> Result<Tensor<D>, Error> {
        let (tensor, shape) = self.tensors.get(name).ok_or_else(|| Error::Tensor {
            name: name.into(),
            reason: "missing tensor".into(),
        })?;
        let shape: [usize; D] = shape.clone().try_into().map_err(|_| Error::Tensor {
            name: name.into(),
            reason: format!("expected rank {D}"),
        })?;
        Ok(tensor.clone().reshape(shape))
    }
    fn optional<const D: usize>(&self, name: &str) -> Result<Option<Tensor<D>>, Error> {
        if self.tensors.contains_key(name) {
            self.tensor(name).map(Some)
        } else {
            Ok(None)
        }
    }
    fn optional_packed<const D: usize>(&self, name: &str) -> Result<Option<Tensor<D, Int>>, Error> {
        let Some((tensor, shape)) = self.packed.get(name) else {
            return Ok(None);
        };
        let shape: [usize; D] = shape.clone().try_into().map_err(|_| Error::Tensor {
            name: name.into(),
            reason: format!("expected rank {D}"),
        })?;
        Ok(Some(tensor.clone().reshape(shape)))
    }
}

pub mod package;
