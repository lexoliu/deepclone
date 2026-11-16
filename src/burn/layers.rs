use super::{Error, WeightSource};
use burn::tensor::{DType, Tensor, activation};

pub(crate) struct Linear {
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}
impl Linear {
    pub fn load(weights: &impl WeightSource, prefix: &str) -> Result<Self, Error> {
        Ok(Self {
            weight: weights
                .tensor::<2>(&format!("{prefix}.weight"))?
                .transpose(),
            bias: weights.optional(&format!("{prefix}.bias"))?,
        })
    }
    pub fn cast(mut self, dtype: DType) -> Self {
        self.weight = self.weight.cast(dtype);
        self.bias = self.bias.map(|bias| bias.cast(dtype));
        self
    }
    pub fn gated(&self, gate: Tensor<3>, up: Tensor<3>) -> Tensor<3> {
        // Qwen's gated intermediate can exceed FP16 while its projection fits.
        let dtype = gate.dtype();
        let product = activation::silu(gate.cast(DType::F32)) * up.cast(DType::F32);
        self.forward(product).cast(dtype)
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dtype = x.dtype();
        let y = x.matmul(self.weight.clone().cast(dtype).unsqueeze());
        match &self.bias {
            Some(bias) => y + bias.clone().cast(dtype).unsqueeze(),
            None => y,
        }
    }
}

pub(crate) struct RmsNorm {
    weight: Tensor<1>,
    epsilon: f32,
}
impl RmsNorm {
    pub fn load(weights: &impl WeightSource, prefix: &str, epsilon: f32) -> Result<Self, Error> {
        Ok(Self {
            weight: weights.tensor(&format!("{prefix}.weight"))?,
            epsilon,
        })
    }
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        // Qwen accumulates normalization statistics in FP32, then restores
        // the activation dtype before applying its learned scale.
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let variance = (x.clone() * x.clone()).mean_dim(D - 1);
        (x * (variance + self.epsilon).sqrt().recip()).cast(dtype) * self.weight.clone().unsqueeze()
    }
}

pub(crate) trait Padding {
    fn apply(&self, x: Tensor<3>, kernel: usize, stride: usize) -> Tensor<3>;
}
pub(crate) struct Causal;
pub(crate) struct CausalEdge;
fn causal_pad(
    x: Tensor<3>,
    kernel: usize,
    stride: usize,
    mode: burn::tensor::ops::PadMode,
) -> Tensor<3> {
    let time = x.dims()[2];
    x.pad(
        [
            (0, 0),
            (0, 0),
            (kernel - stride, (stride - time % stride) % stride),
        ],
        mode,
    )
}
impl Padding for Causal {
    fn apply(&self, x: Tensor<3>, kernel: usize, stride: usize) -> Tensor<3> {
        causal_pad(x, kernel, stride, burn::tensor::ops::PadMode::Constant(0.0))
    }
}
impl Padding for CausalEdge {
    fn apply(&self, x: Tensor<3>, kernel: usize, stride: usize) -> Tensor<3> {
        causal_pad(x, kernel, stride, burn::tensor::ops::PadMode::Edge)
    }
}
pub(crate) struct ReflectSame;
impl Padding for ReflectSame {
    fn apply(&self, x: Tensor<3>, kernel: usize, _stride: usize) -> Tensor<3> {
        let left = (kernel - 1) / 2;
        x.pad(
            [(0, 0), (0, 0), (left, kernel - 1 - left)],
            burn::tensor::ops::PadMode::Reflect,
        )
    }
}
pub(crate) struct Conv<P> {
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    options: burn::tensor::ops::ConvOptions<1>,
    padding: P,
}
impl<P: Padding> Conv<P> {
    pub fn load(
        weights: &impl WeightSource,
        prefix: &str,
        stride: usize,
        dilation: usize,
        groups: usize,
        padding: P,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: weights.tensor(&format!("{prefix}.weight"))?,
            bias: weights.optional(&format!("{prefix}.bias"))?,
            options: burn::tensor::ops::ConvOptions::new([stride], [0], [dilation], groups),
            padding,
        })
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let kernel = (self.weight.dims()[2] - 1) * self.options.dilation[0] + 1;
        let x = self.padding.apply(x, kernel, self.options.stride[0]);
        burn::tensor::module::conv1d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            self.options.clone(),
        )
    }
}

pub(crate) struct LayerNorm {
    weight: Tensor<1>,
    bias: Option<Tensor<1>>,
    epsilon: f64,
}
impl LayerNorm {
    pub fn load(weights: &impl WeightSource, prefix: &str, epsilon: f32) -> Result<Self, Error> {
        Ok(Self {
            weight: weights.tensor(&format!("{prefix}.weight"))?,
            bias: weights.optional(&format!("{prefix}.bias"))?,
            epsilon: epsilon.into(),
        })
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        burn::tensor::module::layer_norm(x, self.weight.clone(), self.bias.clone(), self.epsilon)
    }
}

pub(crate) struct TransposeConv {
    phases: Conv<Causal>,
    bias: Option<Tensor<1>>,
    stride: std::num::NonZeroUsize,
    channels: usize,
}
impl TransposeConv {
    pub fn load(weights: &impl WeightSource, prefix: &str, stride: usize) -> Result<Self, Error> {
        let stride = std::num::NonZeroUsize::new(stride).ok_or_else(|| Error::Tensor {
            name: prefix.into(),
            reason: "transposed convolution stride must be nonzero".into(),
        })?;
        let weight = weights.tensor::<3>(&format!("{prefix}.weight"))?;
        let [input, channels, kernel] = weight.dims();
        let phases = stride.get();
        let taps = kernel.div_ceil(phases);
        // Each output phase is an ordinary causal convolution over input time.
        // Pad incomplete phases with zero taps, then reverse taps for cross-correlation.
        let weight = weight
            .pad(
                [(0, 0), (0, 0), (0, taps * phases - kernel)],
                burn::tensor::ops::PadMode::Constant(0.0),
            )
            .reshape([input, channels, taps, phases])
            .permute([1, 3, 0, 2])
            .flip([3])
            .reshape([channels * phases, input, taps]);
        Ok(Self {
            phases: Conv {
                weight,
                bias: None,
                options: burn::tensor::ops::ConvOptions::new([1], [0], [1], 1),
                padding: Causal,
            },
            bias: weights.optional(&format!("{prefix}.bias"))?,
            stride,
            channels,
        })
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, _, frames] = x.dims();
        let phases = self.stride.get();
        let y = self
            .phases
            .forward(x)
            .reshape([batch, self.channels, phases, frames])
            .swap_dims(2, 3)
            .reshape([batch, self.channels, frames * phases]);
        match &self.bias {
            Some(bias) => y + bias.clone().reshape([1, self.channels, 1]),
            None => y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn gated_projection_preserves_large_intermediates() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            let projection = Linear {
                weight: Tensor::from_data(
                    TensorData::new(vec![0.125f32], [1, 1]),
                    (&device, DType::F16),
                ),
                bias: None,
            };
            let gate = Tensor::from_data(
                TensorData::new(vec![400f32], [1, 1, 1]),
                (&device, DType::F16),
            );
            let up = Tensor::from_data(
                TensorData::new(vec![200f32], [1, 1, 1]),
                (&device, DType::F16),
            );
            let result = super::super::read(projection.gated(gate, up))
                .await
                .unwrap();
            assert_eq!(result, vec![10000.0]);
        });
    }
}
