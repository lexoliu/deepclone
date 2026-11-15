//! Qwen tensor execution on Burn's WGPU runtime.
mod codec_decoder;
mod layers;
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
        Ok(Tensor::from_data(data, &self.device))
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
