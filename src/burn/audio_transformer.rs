use super::{
    Error, WeightSource,
    layers::{LayerNorm, Linear, RmsNorm},
};
use burn::tensor::{DType, Device, Int, Tensor, TensorData, activation};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
}
pub(crate) trait Norm: Sized {
    fn load(w: &impl WeightSource, p: &str, epsilon: f32) -> Result<Self, Error>;
    fn apply(&self, x: Tensor<3>) -> Tensor<3>;
}
impl Norm for RmsNorm {
    fn load(w: &impl WeightSource, p: &str, e: f32) -> Result<Self, Error> {
        Self::load(w, p, e)
    }
    fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward(x)
    }
}
impl Norm for LayerNorm {
    fn load(w: &impl WeightSource, p: &str, e: f32) -> Result<Self, Error> {
        Self::load(w, p, e)
    }
    fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward(x)
    }
}
pub(crate) trait FeedForward: Sized {
    fn load(w: &impl WeightSource, p: &str) -> Result<Self, Error>;
    fn apply(&self, x: Tensor<3>) -> Tensor<3>;
}
pub(crate) struct Gated {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl FeedForward for Gated {
    fn load(w: &impl WeightSource, p: &str) -> Result<Self, Error> {
        Ok(Self {
            gate: Linear::load(w, &format!("{p}.gate_proj"))?,
            up: Linear::load(w, &format!("{p}.up_proj"))?,
            down: Linear::load(w, &format!("{p}.down_proj"))?,
        })
    }
    fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        self.down
            .forward(activation::silu(self.gate.forward(x.clone())) * self.up.forward(x))
    }
}
pub(crate) struct Gelu {
    first: Linear,
    last: Linear,
}
impl FeedForward for Gelu {
    fn load(w: &impl WeightSource, p: &str) -> Result<Self, Error> {
        Ok(Self {
            first: Linear::load(w, &format!("{p}.fc1"))?,
            last: Linear::load(w, &format!("{p}.fc2"))?,
        })
    }
    fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        self.last.forward(activation::gelu(self.first.forward(x)))
    }
}
struct Layer<N, F> {
    input: N,
    post: N,
    mlp: F,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    attention_scale: Tensor<1>,
    mlp_scale: Tensor<1>,
}
impl<N: Norm, F: FeedForward> Layer<N, F> {
    fn load(w: &impl WeightSource, p: &str, e: f32) -> Result<Self, Error> {
        Ok(Self {
            input: N::load(w, &format!("{p}.input_layernorm"), e)?,
            post: N::load(w, &format!("{p}.post_attention_layernorm"), e)?,
            mlp: F::load(w, &format!("{p}.mlp"))?,
            query: Linear::load(w, &format!("{p}.self_attn.q_proj"))?,
            key: Linear::load(w, &format!("{p}.self_attn.k_proj"))?,
            value: Linear::load(w, &format!("{p}.self_attn.v_proj"))?,
            output: Linear::load(w, &format!("{p}.self_attn.o_proj"))?,
            attention_scale: w.tensor(&format!("{p}.self_attn_layer_scale.scale"))?,
            mlp_scale: w.tensor(&format!("{p}.mlp_layer_scale.scale"))?,
        })
    }
    fn forward(
        &self,
        x: Tensor<3>,
        c: &AttentionConfig,
        cosine: &Tensor<4>,
        sine: &Tensor<4>,
        mask: &Tensor<4, burn::tensor::Bool>,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let n = self.input.apply(x.clone());
        let q = self
            .query
            .forward(n.clone())
            .reshape([b, t, c.num_attention_heads, c.head_dim])
            .swap_dims(1, 2);
        let k = self
            .key
            .forward(n.clone())
            .reshape([b, t, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        let v = self
            .value
            .forward(n)
            .reshape([b, t, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        let q = super::transformer::rotary(q, cosine, sine);
        let k = super::transformer::rotary(k, cosine, sine);
        let expand = |x: Tensor<4>| {
            x.reshape([b, c.num_key_value_heads, 1, t, c.head_dim])
                .repeat_dim(2, c.num_attention_heads / c.num_key_value_heads)
                .reshape([b, c.num_attention_heads, t, c.head_dim])
        };
        let scores = (q * (c.head_dim as f32).sqrt().recip())
            .matmul(expand(k).swap_dims(2, 3))
            .cast(DType::F32)
            .mask_fill(mask.clone(), f32::NEG_INFINITY);
        let mixed = activation::softmax(scores, 3)
            .cast(x.dtype())
            .matmul(expand(v))
            .swap_dims(1, 2)
            .reshape([b, t, c.num_attention_heads * c.head_dim]);
        let x = x + self.output.forward(mixed) * self.attention_scale.clone().unsqueeze();
        let n = self.post.apply(x.clone());
        x + self.mlp.apply(n) * self.mlp_scale.clone().unsqueeze()
    }
}
pub(crate) struct AudioTransformer<N, F> {
    layers: Vec<Layer<N, F>>,
    config: AttentionConfig,
    window: Option<usize>,
}
impl<N: Norm, F: FeedForward> AudioTransformer<N, F> {
    pub fn load(
        w: &impl WeightSource,
        p: &str,
        c: AttentionConfig,
        epsilon: f32,
        window: Option<usize>,
    ) -> Result<Self, Error> {
        if c.head_dim == 0
            || !c.head_dim.is_multiple_of(2)
            || c.num_key_value_heads == 0
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
        {
            return Err(Error::Tensor {
                name: p.into(),
                reason: "invalid audio attention dimensions".into(),
            });
        }
        Ok(Self {
            layers: (0..c.num_hidden_layers)
                .map(|i| Layer::load(w, &format!("{p}.layers.{i}"), epsilon))
                .collect::<Result<_, _>>()?,
            config: c,
            window,
        })
    }
    pub fn forward(&self, mut x: Tensor<3>) -> Tensor<3> {
        let t = x.dims()[1];
        let device = x.device();
        let (cosine, sine) = rotary_tables(
            &device,
            x.dtype(),
            t,
            self.config.head_dim,
            self.config.rope_theta,
        );
        let positions = Tensor::<1, Int>::arange(0..t as i64, &device);
        let distance = positions.clone().reshape([t, 1]) - positions.reshape([1, t]);
        let mut mask = distance.clone().lower_elem(0);
        if let Some(window) = self.window {
            mask = mask.bool_or(distance.greater_equal_elem(window as i64));
        }
        let mask = mask.reshape([1, 1, t, t]);
        for layer in &self.layers {
            x = layer.forward(x, &self.config, &cosine, &sine, &mask);
        }
        x
    }
}
pub(crate) fn rotary_tables(
    device: &Device,
    dtype: DType,
    frames: usize,
    width: usize,
    theta: f32,
) -> (Tensor<4>, Tensor<4>) {
    let mut cosines = Vec::with_capacity(frames * width);
    let mut sines = Vec::with_capacity(frames * width);
    for position in 0..frames {
        for index in 0..width {
            let frequency = theta.powf(-((2 * (index % (width / 2))) as f32) / width as f32);
            let (s, c) = (position as f32 * frequency).sin_cos();
            cosines.push(c);
            sines.push(s);
        }
    }
    let shape = [1, 1, frames, width];
    (
        Tensor::<4>::from_data(TensorData::new(cosines, shape), device).cast(dtype),
        Tensor::<4>::from_data(TensorData::new(sines, shape), device).cast(dtype),
    )
}
