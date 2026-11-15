use super::{Error, WeightSource};
use burn::tensor::{DType, Tensor};

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
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = x.matmul(self.weight.clone().unsqueeze());
        match &self.bias {
            Some(bias) => y + bias.clone().unsqueeze(),
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
