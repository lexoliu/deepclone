use super::{Error, WeightSource};
use burn::tensor::{DType, Int, Tensor};

struct QuantizedWeight {
    qweight: Tensor<2, Int>,
    scales: Tensor<2>,
    bits: u8,
    group_size: usize,
}

enum LinearWeight {
    Dense(Box<Tensor<2>>),
    Quantized(Box<QuantizedWeight>),
}

impl LinearWeight {
    fn out_features(&self) -> usize {
        match self {
            Self::Dense(w) => w.dims()[1],
            Self::Quantized(qw) => qw.qweight.dims()[0],
        }
    }
}

pub(crate) struct Linear {
    weight: LinearWeight,
    bias: Option<Tensor<1>>,
}
impl Linear {
    pub fn load(weights: &impl WeightSource, prefix: &str) -> Result<Self, Error> {
        let bias = weights.optional(&format!("{prefix}.bias"))?;
        if let Some(qweight) = weights.optional_packed::<2>(&format!("{prefix}.qweight_i4"))? {
            return Self::from_quantized(weights, prefix, qweight, 4, bias);
        }
        if let Some(qweight) = weights.optional_packed::<2>(&format!("{prefix}.qweight_i8"))? {
            return Self::from_quantized(weights, prefix, qweight, 8, bias);
        }
        Ok(Self {
            weight: LinearWeight::Dense(Box::new(
                weights
                    .tensor::<2>(&format!("{prefix}.weight"))?
                    .transpose(),
            )),
            bias,
        })
    }
    fn from_quantized(
        weights: &impl WeightSource,
        prefix: &str,
        qweight: Tensor<2, Int>,
        bits: u8,
        bias: Option<Tensor<1>>,
    ) -> Result<Self, Error> {
        let scales = weights.tensor::<2>(&format!("{prefix}.scales"))?;
        let [n, k_words] = qweight.dims();
        let [sn, groups] = scales.dims();
        let pack = 32 / usize::from(bits);
        let k = k_words * pack;
        if sn != n || groups == 0 || !k.is_multiple_of(groups) || !(k / groups).is_multiple_of(pack)
        {
            return Err(Error::Tensor {
                name: format!("{prefix}.scales"),
                reason: format!(
                    "quantized W{bits} shape mismatch: qweight=[{n},{k_words}], scales=[{sn},{groups}]"
                ),
            });
        }
        Ok(Self {
            weight: LinearWeight::Quantized(Box::new(QuantizedWeight {
                qweight,
                scales,
                bits,
                group_size: k / groups,
            })),
            bias,
        })
    }
    pub fn cast(mut self, dtype: DType) -> Self {
        self.weight = match self.weight {
            LinearWeight::Dense(w) => LinearWeight::Dense(Box::new(w.as_ref().clone().cast(dtype))),
            LinearWeight::Quantized(qw) => LinearWeight::Quantized(Box::new(QuantizedWeight {
                qweight: qw.qweight,
                scales: qw.scales.cast(dtype),
                bits: qw.bits,
                group_size: qw.group_size,
            })),
        };
        self.bias = self.bias.map(|bias| bias.cast(dtype));
        self
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let dtype = x.dtype();
        let is_decode = x.dims()[0] == 1 && x.dims()[1] == 1;
        let y = match &self.weight {
            LinearWeight::Dense(weight) => {
                if is_decode {
                    super::gemv::project(x, weight.as_ref().clone())
                } else {
                    super::projection::project(x, weight.as_ref().clone())
                }
            }
            LinearWeight::Quantized(qw) => {
                if is_decode {
                    super::quant_gemv::project_gemv(
                        x,
                        qw.qweight.clone(),
                        qw.scales.clone(),
                        qw.bits,
                        qw.group_size,
                    )
                } else {
                    super::quant_gemv::project_batched(
                        x,
                        qw.qweight.clone(),
                        qw.scales.clone(),
                        qw.bits,
                        qw.group_size,
                    )
                }
            }
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
                            Tensor::<1>::zeros([layer.weight.out_features()], weights.device())
                                .cast(weights.dtype())
                        })
                    })
                    .collect(),
                0,
            ))
        } else {
            None
        };
        let weight = match &layers[0].weight {
            LinearWeight::Dense(_) => {
                // Preserve contiguous input features within each output row so decode
                // lanes can cooperate on the reduction without repacking every token.
                let mut rows = Vec::with_capacity(layers.len());
                for layer in layers {
                    let LinearWeight::Dense(w) = layer.weight else {
                        return Err(Error::Tensor {
                            name: prefixes[0].clone(),
                            reason: "cannot join dense and quantized projections".into(),
                        });
                    };
                    rows.push(w.as_ref().clone().transpose());
                }
                LinearWeight::Dense(Box::new(Tensor::cat(rows, 0).transpose()))
            }
            LinearWeight::Quantized(qw) => {
                let (bits, group_size) = (qw.bits, qw.group_size);
                let mut qweights = Vec::with_capacity(layers.len());
                let mut scales = Vec::with_capacity(layers.len());
                qweights.push(qw.qweight.clone());
                scales.push(qw.scales.clone());

                for layer in layers.into_iter().skip(1) {
                    let LinearWeight::Quantized(qw_next) = layer.weight else {
                        return Err(Error::Tensor {
                            name: prefixes[0].clone(),
                            reason: "cannot join quantized and dense projections".into(),
                        });
                    };
                    if qw_next.bits != bits || qw_next.group_size != group_size {
                        return Err(Error::Tensor {
                            name: prefixes[0].clone(),
                            reason: format!(
                                "mismatched quantization in joined projection: W{}/g{} vs W{bits}/g{group_size}",
                                qw_next.bits, qw_next.group_size
                            ),
                        });
                    }
                    qweights.push(qw_next.qweight);
                    scales.push(qw_next.scales);
                }
                LinearWeight::Quantized(Box::new(QuantizedWeight {
                    qweight: Tensor::cat(qweights, 0),
                    scales: Tensor::cat(scales, 0),
                    bits,
                    group_size,
                }))
            }
        };
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
    const PAD_MODE: usize;
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
    const PAD_MODE: usize = super::conv::PAD_CAUSAL_ZERO;
    fn apply(&self, x: Tensor<3>, kernel: usize, stride: usize) -> Tensor<3> {
        causal_pad(x, kernel, stride, burn::tensor::ops::PadMode::Constant(0.0))
    }
}
impl Padding for CausalEdge {
    const PAD_MODE: usize = super::conv::PAD_CAUSAL_EDGE;
    fn apply(&self, x: Tensor<3>, kernel: usize, stride: usize) -> Tensor<3> {
        causal_pad(x, kernel, stride, burn::tensor::ops::PadMode::Edge)
    }
}
pub(crate) struct ReflectSame;
impl Padding for ReflectSame {
    const PAD_MODE: usize = super::conv::PAD_REFLECT_SAME;
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
    kernel: usize,
    packed: bool,
    padding: P,
}
impl<P: Padding> Conv<P> {
    fn from_parts(
        weight: Tensor<3>,
        bias: Option<Tensor<1>>,
        options: burn::tensor::ops::ConvOptions<1>,
        padding: P,
    ) -> Self {
        let kernel = weight.dims()[2];
        let packed = options.groups == 1
            && (P::PAD_MODE != super::conv::PAD_REFLECT_SAME
                || (options.stride == [1] && kernel > 1));
        // Pack `[C_out, C_in, K]` to contiguous `[K, C_out, C_in]` once at load
        // so tiled channel-first convolutions read coalesced input channels.
        let weight = if packed {
            super::conv::contiguous(weight.permute([2, 0, 1]))
        } else {
            weight
        };
        Self {
            weight,
            bias,
            options,
            kernel,
            packed,
            padding,
        }
    }
    pub fn load(
        weights: &impl WeightSource,
        prefix: &str,
        stride: usize,
        dilation: usize,
        groups: usize,
        padding: P,
    ) -> Result<Self, Error> {
        Ok(Self::from_parts(
            weights.tensor(&format!("{prefix}.weight"))?,
            weights.optional(&format!("{prefix}.bias"))?,
            burn::tensor::ops::ConvOptions::new([stride], [0], [dilation], groups),
            padding,
        ))
    }
    fn is_causal_depthwise(&self) -> bool {
        P::PAD_MODE == super::conv::PAD_CAUSAL_ZERO
            && !self.packed
            && self.options.stride == [1]
            && self.options.groups == self.weight.dims()[0]
            && self.weight.dims()[1] == 1
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_inner(x, None)
    }
    pub fn forward_add(&self, x: Tensor<3>, residual: Tensor<3>) -> Tensor<3> {
        self.forward_inner(x, Some(residual))
    }
    fn forward_inner(&self, x: Tensor<3>, residual: Option<Tensor<3>>) -> Tensor<3> {
        if self.packed {
            let eff_k = (self.kernel - 1) * self.options.dilation[0] + 1;
            let left_pad = (eff_k - 1) / 2;
            let right_pad = eff_k - 1 - left_pad;
            let can_fused = P::PAD_MODE != super::conv::PAD_REFLECT_SAME
                || (x.dims()[2] > left_pad && x.dims()[2] > right_pad);
            if can_fused {
                return match residual {
                    Some(res) => super::conv::conv1d_add(
                        x,
                        None,
                        self.weight.clone(),
                        self.bias.clone(),
                        self.options.stride[0],
                        self.options.dilation[0],
                        P::PAD_MODE,
                        res,
                    ),
                    None => super::conv::conv1d(
                        x,
                        None,
                        self.weight.clone(),
                        self.bias.clone(),
                        self.options.stride[0],
                        self.options.dilation[0],
                        P::PAD_MODE,
                    ),
                };
            }
            let x = self.padding.apply(x, eff_k, self.options.stride[0]);
            let unpacked = self.weight.clone().permute([1, 2, 0]);
            let out =
                burn::tensor::module::conv1d(x, unpacked, self.bias.clone(), self.options.clone());
            return match residual {
                Some(res) => res + out,
                None => out,
            };
        }
        if self.is_causal_depthwise() {
            let out = super::conv::depthwise_causal_conv1d(
                x,
                None,
                self.weight
                    .clone()
                    .reshape([self.options.groups, self.kernel]),
                self.bias.clone(),
                self.options.dilation[0],
            );
            return match residual {
                Some(res) => res + out,
                None => out,
            };
        }
        let kernel = (self.kernel - 1) * self.options.dilation[0] + 1;
        let x = self.padding.apply(x, kernel, self.options.stride[0]);
        let out = self.convolve(x);
        match residual {
            Some(res) => res + out,
            None => out,
        }
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
        self.stream_inner(x, history, None)
    }
    pub fn stream_add(
        &self,
        x: Tensor<3>,
        history: &mut ConvHistory,
        residual: Tensor<3>,
    ) -> Tensor<3> {
        self.stream_inner(x, history, Some(residual))
    }
    fn stream_inner(
        &self,
        x: Tensor<3>,
        history: &mut ConvHistory,
        residual: Option<Tensor<3>>,
    ) -> Tensor<3> {
        // Every convolution in Qwen's waveform decoder has unit stride.
        debug_assert_eq!(self.options.stride[0], 1);
        let context = (self.kernel - 1) * self.options.dilation[0];
        let previous = history.input.take();
        if self.packed || self.is_causal_depthwise() {
            let [b, c, t] = x.dims();
            if context > 0 {
                history.input = Some(if t >= context {
                    x.clone().slice([0..b, 0..c, t - context..t])
                } else {
                    let combined = match &previous {
                        Some(prev) => Tensor::cat(vec![prev.clone(), x.clone()], 2),
                        None => self.padding.apply(x.clone(), context + 1, 1),
                    };
                    let ct = combined.dims()[2];
                    combined.slice([0..b, 0..c, ct - context..ct])
                });
            }
            return if self.packed {
                match residual {
                    Some(res) => super::conv::conv1d_add(
                        x,
                        previous,
                        self.weight.clone(),
                        self.bias.clone(),
                        1,
                        self.options.dilation[0],
                        super::conv::PAD_CAUSAL_ZERO,
                        res,
                    ),
                    None => super::conv::conv1d(
                        x,
                        previous,
                        self.weight.clone(),
                        self.bias.clone(),
                        1,
                        self.options.dilation[0],
                        super::conv::PAD_CAUSAL_ZERO,
                    ),
                }
            } else {
                let out = super::conv::depthwise_causal_conv1d(
                    x,
                    previous,
                    self.weight
                        .clone()
                        .reshape([self.options.groups, self.kernel]),
                    self.bias.clone(),
                    self.options.dilation[0],
                );
                match residual {
                    Some(res) => res + out,
                    None => out,
                }
            };
        }
        let x = match previous {
            Some(previous) => Tensor::cat(vec![previous, x], 2),
            None => self.padding.apply(x, context + 1, 1),
        };
        let [b, c, t] = x.dims();
        if context > 0 {
            history.input = Some(x.clone().slice([0..b, 0..c, t - context..t]));
        }
        let out = self.convolve(x);
        match residual {
            Some(res) => res + out,
            None => out,
        }
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
        let weight = weights.tensor::<3>(&format!("{prefix}.weight"))?;
        // Pack `[C_in, C_out, K]` to contiguous `[K, C_out, C_in]` once at load
        // so phase-grouped transposed convolutions read coalesced input channels.
        Ok(Self {
            weight: super::conv::contiguous(weight.permute([2, 1, 0])),
            bias: weights.optional(&format!("{prefix}.bias"))?,
            stride,
        })
    }
    pub fn stream(&self, x: Tensor<3>, history: &mut ConvHistory) -> Tensor<3> {
        let context = (self.weight.dims()[0] - self.stride).div_ceil(self.stride);
        let previous = history.input.take();
        let [b, c, t] = x.dims();
        if context > 0 {
            history.input = Some(if t >= context {
                x.clone().slice([0..b, 0..c, t - context..t])
            } else {
                let combined = match &previous {
                    Some(prev) => Tensor::cat(vec![prev.clone(), x.clone()], 2),
                    None => x.clone(),
                };
                let ct = combined.dims()[2];
                combined.slice([0..b, 0..c, ct.saturating_sub(context)..ct])
            });
        }
        super::conv::causal_conv_transpose1d(
            x,
            previous,
            self.weight.clone(),
            self.bias.clone(),
            self.stride,
        )
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
                                let raw_weight = Tensor::from_data(
                                    TensorData::new(
                                        (0..outputs * channels / groups * kernel)
                                            .map(|i| (i as f32 * 0.031).cos() / 100.)
                                            .collect::<Vec<_>>(),
                                        [outputs, channels / groups, kernel],
                                    ),
                                    (&device, dtype),
                                );
                                let raw_bias = bias.then(|| {
                                    Tensor::from_data(
                                        TensorData::new(
                                            (0..outputs)
                                                .map(|i| i as f32 / 100.)
                                                .collect::<Vec<_>>(),
                                            [outputs],
                                        ),
                                        (&device, dtype),
                                    )
                                });
                                let layer = Conv::from_parts(
                                    raw_weight.clone(),
                                    raw_bias.clone(),
                                    burn::tensor::ops::ConvOptions::new(
                                        [stride],
                                        [0],
                                        [dilation],
                                        groups,
                                    ),
                                    Causal,
                                );
                                let padded = layer.padding.apply(
                                    input.clone(),
                                    (kernel - 1) * dilation + 1,
                                    stride,
                                );
                                let packed_causal = groups == 1;
                                let reference_dtype =
                                    if packed_causal { DType::F32 } else { dtype };
                                let reference = burn::tensor::module::conv1d(
                                    padded.cast(reference_dtype),
                                    raw_weight.cast(reference_dtype),
                                    raw_bias.map(|v| v.cast(reference_dtype)),
                                    layer.options.clone(),
                                )
                                .cast(dtype);
                                let mut actual = vec![layer.forward(input.clone())];
                                if stride == 1 {
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
                weight: LinearWeight::Dense(Box::new(Tensor::from_data(
                    TensorData::new(vec![0.125f32], [1, 1]),
                    (&device, DType::F16),
                ))),
                bias: None,
            };
            let gate_up = Tensor::from_data(
                TensorData::new(vec![400f32, 200f32], [1, 1, 2]),
                (&device, DType::F16),
            );
            let result = super::super::read(
                projection
                    .forward(super::super::fused::gated_silu(gate_up))
                    .cast(DType::F16),
            )
            .await
            .unwrap();
            assert_eq!(result, vec![10000.0]);
        });
    }
}
