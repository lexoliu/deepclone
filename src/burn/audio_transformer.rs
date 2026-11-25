use super::{
    Error, WeightSource,
    layers::{JoinedLinear, LayerNorm, Linear, RmsNorm},
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
    gate_up: JoinedLinear<2>,
    down: Linear,
}
impl FeedForward for Gated {
    fn load(w: &impl WeightSource, p: &str) -> Result<Self, Error> {
        Ok(Self {
            gate_up: JoinedLinear::load(w, ["gate", "up"].map(|name| format!("{p}.{name}_proj")))?,
            down: Linear::load(w, &format!("{p}.down_proj"))?.cast(DType::F32),
        })
    }
    fn apply(&self, x: Tensor<3>) -> Tensor<3> {
        let dtype = x.dtype();
        self.down
            .forward(super::fused::gated_silu(self.gate_up.raw(x)))
            .cast(dtype)
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
    qkv: JoinedLinear<3>,
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
            qkv: JoinedLinear::load(
                w,
                ["q", "k", "v"].map(|name| format!("{p}.self_attn.{name}_proj")),
            )?,
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
        history: &mut Option<(Tensor<4>, Tensor<4>)>,
    ) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let n = self.input.apply(x.clone());
        let qkv = self.qkv.raw(n);
        let q_width = c.num_attention_heads * c.head_dim;
        let kv_width = c.num_key_value_heads * c.head_dim;
        let q = qkv
            .clone()
            .slice([0..b, 0..t, 0..q_width])
            .reshape([b, t, c.num_attention_heads, c.head_dim])
            .swap_dims(1, 2);
        let k = qkv
            .clone()
            .slice([0..b, 0..t, q_width..q_width + kv_width])
            .reshape([b, t, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        let v = qkv
            .slice([0..b, 0..t, q_width + kv_width..q_width + 2 * kv_width])
            .reshape([b, t, c.num_key_value_heads, c.head_dim])
            .swap_dims(1, 2);
        let q = super::transformer::rotary(q, cosine, sine);
        let k = super::transformer::rotary(k, cosine, sine);
        let (k, v) = match history.take() {
            Some((keys, values)) => (
                Tensor::cat(vec![keys, k], 2),
                Tensor::cat(vec![values, v], 2),
            ),
            None => (k, v),
        };
        *history = Some((k.clone(), v.clone()));
        let keys = k.dims()[2];
        let expand = |x: Tensor<4>| {
            if c.num_attention_heads == c.num_key_value_heads {
                x
            } else {
                x.reshape([b, c.num_key_value_heads, 1, keys, c.head_dim])
                    .repeat_dim(2, c.num_attention_heads / c.num_key_value_heads)
                    .reshape([b, c.num_attention_heads, keys, c.head_dim])
            }
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
    #[allow(clippy::too_many_arguments)]
    fn forward_fused(
        &self,
        x: Tensor<3>,
        c: &AttentionConfig,
        rotary: &Tensor<3>,
        cache_pos: usize,
        rotary_pos: usize,
        window: usize,
        capacity: usize,
        history: &mut Option<(Tensor<4>, Tensor<4>)>,
    ) -> Tensor<3> {
        let (keys, values) = history.take().unwrap_or_else(|| {
            let shape = [1, c.num_key_value_heads, capacity, c.head_dim];
            let device = x.device();
            let dtype = x.dtype();
            (
                Tensor::<4>::zeros(shape, &device).cast(dtype),
                Tensor::<4>::zeros(shape, &device).cast(dtype),
            )
        });
        let n = self.input.apply(x.clone());
        let qkv = self.qkv.raw(n);
        let scale = (c.head_dim as f32).sqrt().recip();
        let (q, keys, values) = super::fused::qkv_rope_cache_no_norm(
            qkv,
            rotary.clone(),
            keys,
            values,
            cache_pos,
            rotary_pos,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            scale,
        );
        *history = Some((keys.clone(), values.clone()));
        let mixed = super::attention::prefill_windowed(q, keys, values, cache_pos, window);
        let x = x + self.output.forward(mixed) * self.attention_scale.clone().unsqueeze();
        let n = self.post.apply(x.clone());
        x + self.mlp.apply(n) * self.mlp_scale.clone().unsqueeze()
    }
}
const AUDIO_ROTARY_PRECOMPUTED: usize = 512;
const AUDIO_CACHE_HEADROOM: usize = 256;

pub(crate) struct AudioTransformer<N, F> {
    layers: Vec<Layer<N, F>>,
    config: AttentionConfig,
    window: Option<usize>,
    rotary: Option<Tensor<3>>,
}
#[derive(Clone)]
pub(crate) struct AudioHistory {
    layers: Vec<Option<(Tensor<4>, Tensor<4>)>>,
    frames: usize,
    cache_len: usize,
    capacity: usize,
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
            || window == Some(0)
        {
            return Err(Error::Tensor {
                name: p.into(),
                reason: "invalid audio attention dimensions".into(),
            });
        }
        let rotary = Some(rotary_packed_at(
            w.device(),
            w.dtype(),
            0,
            AUDIO_ROTARY_PRECOMPUTED,
            c.head_dim,
            c.rope_theta,
        ));
        Ok(Self {
            layers: (0..c.num_hidden_layers)
                .map(|i| Layer::load(w, &format!("{p}.layers.{i}"), epsilon))
                .collect::<Result<_, _>>()?,
            config: c,
            window,
            rotary,
        })
    }
    pub fn history(&self) -> AudioHistory {
        AudioHistory {
            layers: (0..self.layers.len()).map(|_| None).collect(),
            frames: 0,
            cache_len: 0,
            capacity: 0,
        }
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.stream(x, &mut self.history())
    }
    pub fn stream(&self, mut x: Tensor<3>, history: &mut AudioHistory) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let max_span = self
            .window
            .map_or(history.frames + t, |w| w.min(history.frames + t).max(t));
        if b == 1
            && self.config.head_dim.is_multiple_of(2)
            && super::attention::supports_fused(max_span, self.config.head_dim)
        {
            return self.stream_fused(x, history);
        }
        let device = x.device();
        if history.cache_len < history.capacity {
            for past in &mut history.layers {
                if let Some((k, v)) = past.take() {
                    let [kb, kv, _, hd] = k.dims();
                    let sel = [0..kb, 0..kv, 0..history.cache_len, 0..hd];
                    *past = Some((k.slice(sel.clone()), v.slice(sel)));
                }
            }
            history.capacity = history.cache_len;
        }
        let (cosine, sine) = rotary_tables_at(
            &device,
            x.dtype(),
            history.frames,
            t,
            self.config.head_dim,
            self.config.rope_theta,
        );
        let total = history.frames + t;
        let cache_start = history.frames - history.cache_len;
        let key_count = total - cache_start;
        let queries = Tensor::<1, Int>::arange(history.frames as i64..total as i64, &device);
        let keys = Tensor::<1, Int>::arange(cache_start as i64..total as i64, &device);
        let distance = queries.reshape([t, 1]) - keys.reshape([1, key_count]);
        let mut mask = distance.clone().lower_elem(0);
        if let Some(window) = self.window {
            mask = mask.bool_or(distance.greater_equal_elem(window as i64));
        }
        let mask = mask.reshape([1, 1, t, key_count]);
        for (layer, past) in self.layers.iter().zip(&mut history.layers) {
            x = layer.forward(x, &self.config, &cosine, &sine, &mask, past);
        }
        history.frames = total;
        history.cache_len += t;
        history.capacity = history.cache_len;
        x
    }
    fn stream_fused(&self, mut x: Tensor<3>, history: &mut AudioHistory) -> Tensor<3> {
        let t = x.dims()[1];
        let device = x.device();
        let dtype = x.dtype();
        if history.cache_len + t > history.capacity {
            let new_cap = self.window.map_or(history.cache_len + t, |w| {
                (w + AUDIO_CACHE_HEADROOM).max(w + t)
            });
            if history.cache_len > 0 {
                let keep = self
                    .window
                    .map_or(history.cache_len, |w| (w - 1).min(history.cache_len));
                let start = history.cache_len - keep;
                for past in &mut history.layers {
                    if let Some((k, v)) = past.take() {
                        let [_, kv_heads, _, head_dim] = k.dims();
                        let shape = [1, kv_heads, new_cap, head_dim];
                        let mut nk = Tensor::<4>::zeros(shape, &device).cast(dtype);
                        let mut nv = Tensor::<4>::zeros(shape, &device).cast(dtype);
                        if keep > 0 {
                            let src = [0..1, 0..kv_heads, start..history.cache_len, 0..head_dim];
                            let dst = [0..1, 0..kv_heads, 0..keep, 0..head_dim];
                            nk = nk.slice_assign(dst.clone(), k.slice(src.clone()));
                            nv = nv.slice_assign(dst, v.slice(src));
                        }
                        *past = Some((nk, nv));
                    }
                }
                history.cache_len = keep;
            }
            history.capacity = new_cap;
        }
        let (rotary, rotary_pos) = match &self.rotary {
            Some(cached) if history.frames + t <= cached.dims()[1] && cached.dtype() == dtype => {
                (cached.clone(), history.frames)
            }
            _ => (
                rotary_packed_at(
                    &device,
                    dtype,
                    history.frames,
                    t,
                    self.config.head_dim,
                    self.config.rope_theta,
                ),
                0,
            ),
        };
        let window = self.window.unwrap_or(0);
        for (layer, past) in self.layers.iter().zip(&mut history.layers) {
            x = layer.forward_fused(
                x,
                &self.config,
                &rotary,
                history.cache_len,
                rotary_pos,
                window,
                history.capacity,
                past,
            );
        }
        history.cache_len += t;
        history.frames += t;
        x
    }
}
fn rotary_packed_at(
    device: &Device,
    dtype: DType,
    start: usize,
    frames: usize,
    width: usize,
    theta: f32,
) -> Tensor<3> {
    let (cosine, sine) = rotary_tables_at(device, dtype, start, frames, width, theta);
    Tensor::cat(
        vec![
            cosine.reshape([1, frames, width]),
            sine.reshape([1, frames, width]),
        ],
        0,
    )
}
pub(crate) fn rotary_tables(
    device: &Device,
    dtype: DType,
    frames: usize,
    width: usize,
    theta: f32,
) -> (Tensor<4>, Tensor<4>) {
    rotary_tables_at(device, dtype, 0, frames, width, theta)
}
fn rotary_tables_at(
    device: &Device,
    dtype: DType,
    start: usize,
    frames: usize,
    width: usize,
    theta: f32,
) -> (Tensor<4>, Tensor<4>) {
    let mut cosines = Vec::with_capacity(frames * width);
    let mut sines = Vec::with_capacity(frames * width);
    for position in start..start + frames {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::WeightSource;

    struct UnusedWeights;
    impl WeightSource for UnusedWeights {
        fn device(&self) -> &Device {
            unreachable!("validation fails before device access")
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn tensor<const D: usize>(&self, name: &str) -> Result<Tensor<D>, Error> {
            Err(Error::Tensor {
                name: name.into(),
                reason: "unused".into(),
            })
        }
        fn optional<const D: usize>(&self, _name: &str) -> Result<Option<Tensor<D>>, Error> {
            Ok(None)
        }
    }

    #[test]
    fn audio_transformer_rejects_zero_window() {
        let config = AttentionConfig {
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            head_dim: 16,
            rope_theta: 10000.0,
        };
        assert!(
            AudioTransformer::<RmsNorm, Gated>::load(&UnusedWeights, "test", config, 1e-6, Some(0))
                .is_err()
        );
    }
}
