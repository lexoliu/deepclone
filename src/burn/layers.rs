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
        let y = if x.dims()[0] == 1 && x.dims()[1] == 1 {
            super::gemv::project(x, self.weight.clone())
        } else {
            super::projection::project(x, self.weight.clone())
        };
        match &self.bias {
            Some(bias) => y + bias.clone().cast(dtype).unsqueeze(),
            None => y,
        }
    }
}

/// Independent projections sharing one input execute as one matrix multiplication.
pub(crate) struct JoinedLinear<const N: usize> {
    projection: Linear,
    widths: [usize; N],
}
impl<const N: usize> JoinedLinear<N> {
    pub fn load(weights: &impl WeightSource, prefixes: [String; N]) -> Result<Self, Error> {
        let layers = prefixes
            .iter()
            .map(|p| Linear::load(weights, p))
            .collect::<Result<Vec<_>, _>>()?;
        let widths = std::array::from_fn(|i| layers[i].weight.dims()[1]);
        let bias = if layers.iter().any(|layer| layer.bias.is_some()) {
            Some(Tensor::cat(
                layers
                    .iter()
                    .map(|layer| {
                        layer.bias.clone().unwrap_or_else(|| {
                            Tensor::<1>::zeros([layer.weight.dims()[1]], weights.device())
                                .cast(weights.dtype())
                        })
                    })
                    .collect(),
                0,
            ))
        } else {
            None
        };
        // Preserve contiguous input features within each output row so decode
        // lanes can cooperate on the reduction without repacking every token.
        let weight = Tensor::cat(
            layers
                .into_iter()
                .map(|layer| layer.weight.transpose())
                .collect(),
            0,
        )
        .transpose();
        Ok(Self {
            projection: Linear { weight, bias },
            widths,
        })
    }
    pub fn forward(&self, input: Tensor<3>) -> [Tensor<3>; N] {
        let output = self.projection.forward(input);
        let [batch, tokens, _] = output.dims();
        let mut offset = 0;
        std::array::from_fn(|i| {
            let end = offset + self.widths[i];
            let part = output.clone().slice([0..batch, 0..tokens, offset..end]);
            offset = end;
            part
        })
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
        self.convolve(x)
    }
    fn convolve(&self, x: Tensor<3>) -> Tensor<3> {
        burn::tensor::module::conv1d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            self.options.clone(),
        )
    }
}

/// Only the causal receptive field survives between codec spans.
#[derive(Clone, Default)]
pub(crate) struct ConvHistory {
    input: Option<Tensor<3>>,
}
impl Conv<Causal> {
    pub fn stream(&self, x: Tensor<3>, history: &mut ConvHistory) -> Tensor<3> {
        // Every convolution in Qwen's waveform decoder has unit stride.
        debug_assert_eq!(self.options.stride[0], 1);
        let context = (self.weight.dims()[2] - 1) * self.options.dilation[0];
        let x = match history.input.take() {
            Some(previous) => Tensor::cat(vec![previous, x], 2),
            None => self.padding.apply(x, context + 1, 1),
        };
        let [b, c, t] = x.dims();
        if context > 0 {
            history.input = Some(x.clone().slice([0..b, 0..c, t - context..t]));
        }
        self.convolve(x)
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
    weight: Tensor<3>,
    bias: Option<Tensor<1>>,
    stride: usize,
}
impl TransposeConv {
    pub fn load(weights: &impl WeightSource, prefix: &str, stride: usize) -> Result<Self, Error> {
        Ok(Self {
            weight: weights.tensor(&format!("{prefix}.weight"))?,
            bias: weights.optional(&format!("{prefix}.bias"))?,
            stride,
        })
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let frames = x.dims()[2] * self.stride;
        let y = burn::tensor::module::conv_transpose1d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            burn::tensor::ops::ConvTransposeOptions::new([self.stride], [0], [0], [1], 1),
        );
        let [b, c, _] = y.dims();
        y.slice([0..b, 0..c, 0..frames])
    }
}

impl TransposeConv {
    pub fn stream(&self, x: Tensor<3>, history: &mut ConvHistory) -> Tensor<3> {
        let context = (self.weight.dims()[2] - self.stride).div_ceil(self.stride);
        let (x, previous) = match history.input.take() {
            Some(previous) => {
                let frames = previous.dims()[2];
                (Tensor::cat(vec![previous, x], 2), frames)
            }
            None => (x, 0),
        };
        let [b, c, t] = x.dims();
        if context > 0 {
            history.input = Some(x.clone().slice([0..b, 0..c, t.saturating_sub(context)..t]));
        }
        let y = self.forward(x);
        let [b, c, t] = y.dims();
        y.slice([0..b, 0..c, previous * self.stride..t])
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
