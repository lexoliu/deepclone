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
}
impl<const N: usize> JoinedLinear<N> {
    pub fn load(weights: &impl WeightSource, prefixes: [String; N]) -> Result<Self, Error> {
        let layers = prefixes
            .iter()
            .map(|p| Linear::load(weights, p))
            .collect::<Result<Vec<_>, _>>()?;
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
        })
    }
    pub fn raw(&self, input: Tensor<3>) -> Tensor<3> {
        self.projection.forward(input)
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
    pub fn weight(&self) -> &Tensor<1> {
        &self.weight
    }
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        super::fused::rms_norm(x, self.weight.clone(), self.epsilon)
    }
    pub fn add_forward(&self, residual: Tensor<3>, delta: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        super::fused::add_rms_norm(residual, delta, self.weight.clone(), self.epsilon)
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
        let [output, input, kernel] = self.weight.dims();
        if kernel == 1 && self.options.groups == 1 && self.options.stride == [1] {
            let projected = super::projection::project(
                x.swap_dims(1, 2),
                self.weight.clone().reshape([output, input]).transpose(),
            )
            .swap_dims(1, 2);
            return match &self.bias {
                Some(bias) => projected + bias.clone().reshape([1, output, 1]),
                None => projected,
            };
        }

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
    fn convolution_matches_reference_across_layouts_and_stream_boundaries() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F16, DType::F32] {
                for [batch, channels, time, outputs] in [[2, 6, 19, 10], [1, 38, 35, 22]] {
                    for transposed in [false, true] {
                        let mut input = Tensor::<3>::from_data(
                            TensorData::new(
                                (0..batch * channels * time)
                                    .map(|i| (i as f32 * 0.017).sin())
                                    .collect::<Vec<_>>(),
                                if transposed {
                                    [batch, time, channels]
                                } else {
                                    [batch, channels, time]
                                },
                            ),
                            (&device, dtype),
                        );
                        if transposed {
                            input = input.swap_dims(1, 2);
                        }
                        for bias in [false, true] {
                            for [kernel, stride, dilation, groups] in
                                [[1, 1, 1, 1], [3, 1, 2, 1], [1, 1, 1, 2], [3, 2, 1, 1]]
                            {
                                let layer = Conv {
                                    weight: Tensor::from_data(
                                        TensorData::new(
                                            (0..outputs * channels / groups * kernel)
                                                .map(|i| (i as f32 * 0.031).cos() / 100.)
                                                .collect::<Vec<_>>(),
                                            [outputs, channels / groups, kernel],
                                        ),
                                        (&device, dtype),
                                    ),
                                    bias: bias.then(|| {
                                        Tensor::from_data(
                                            TensorData::new(
                                                (0..outputs)
                                                    .map(|i| i as f32 / 100.)
                                                    .collect::<Vec<_>>(),
                                                [outputs],
                                            ),
                                            (&device, dtype),
                                        )
                                    }),
                                    options: burn::tensor::ops::ConvOptions::new(
                                        [stride],
                                        [0],
                                        [dilation],
                                        groups,
                                    ),
                                    padding: Causal,
                                };
                                let padded = layer.padding.apply(
                                    input.clone(),
                                    (kernel - 1) * dilation + 1,
                                    stride,
                                );
                                let pointwise = kernel == 1 && groups == 1 && stride == 1;
                                let reference_dtype = if pointwise { DType::F32 } else { dtype };
                                let reference = burn::tensor::module::conv1d(
                                    padded.cast(reference_dtype),
                                    layer.weight.clone().cast(reference_dtype),
                                    layer.bias.clone().map(|v| v.cast(reference_dtype)),
                                    layer.options.clone(),
                                )
                                .cast(dtype);
                                let mut actual = vec![layer.forward(input.clone())];
                                if pointwise {
                                    let mut history = ConvHistory::default();
                                    let mut start = 0;
                                    let spans = [1, 8, time];
                                    let streamed = spans
                                        .into_iter()
                                        .map(|end| {
                                            let part = input.clone().slice([
                                                0..batch,
                                                0..channels,
                                                start..end,
                                            ]);
                                            start = end;
                                            layer.stream(part, &mut history)
                                        })
                                        .collect();
                                    actual.push(Tensor::cat(streamed, 2));
                                }
                                let expected = super::super::read(reference).await.unwrap();
                                for output in actual {
                                    let output = super::super::read(output).await.unwrap();
                                    assert_eq!(output.len(), expected.len());
                                    assert!(output.iter().all(|v| v.is_finite()));
                                    let error = output
                                        .iter()
                                        .zip(&expected)
                                        .map(|(a, b)| (a - b).abs())
                                        .fold(0f32, f32::max);
                                    let tolerance = if dtype == DType::F16 { 0.001 } else { 1e-5 };
                                    assert!(
                                        error < tolerance,
                                        "{dtype:?}, transposed={transposed}, bias={bias}, \
                                         kernel={kernel}, stride={stride}, groups={groups}: {error}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        });
    }

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
