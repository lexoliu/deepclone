use super::{
    Error, Weights,
    layers::{Linear, RmsNorm},
};
use burn::tensor::{Tensor, activation};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct DecoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
}

struct Layer {
    input: RmsNorm,
    post: RmsNorm,
    query_norm: RmsNorm,
    key_norm: RmsNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    keys: Tensor<4>,
    values: Tensor<4>,
}
impl Layer {
    fn load(
        w: &Weights<'_>,
        prefix: &str,
        c: &DecoderConfig,
        capacity: usize,
    ) -> Result<Self, Error> {
        let shape = [1, c.num_key_value_heads, capacity, c.head_dim];
        Ok(Self {
            input: RmsNorm::load(w, &format!("{prefix}.input_layernorm"), c.rms_norm_eps)?,
            post: RmsNorm::load(
                w,
                &format!("{prefix}.post_attention_layernorm"),
                c.rms_norm_eps,
            )?,
            query_norm: RmsNorm::load(w, &format!("{prefix}.self_attn.q_norm"), c.rms_norm_eps)?,
            key_norm: RmsNorm::load(w, &format!("{prefix}.self_attn.k_norm"), c.rms_norm_eps)?,
            query: Linear::load(w, &format!("{prefix}.self_attn.q_proj"))?,
            key: Linear::load(w, &format!("{prefix}.self_attn.k_proj"))?,
            value: Linear::load(w, &format!("{prefix}.self_attn.v_proj"))?,
            output: Linear::load(w, &format!("{prefix}.self_attn.o_proj"))?,
            gate: Linear::load(w, &format!("{prefix}.mlp.gate_proj"))?,
            up: Linear::load(w, &format!("{prefix}.mlp.up_proj"))?,
            down: Linear::load(w, &format!("{prefix}.mlp.down_proj"))?,
            keys: Tensor::<4>::zeros(shape, &w.device).cast(w.dtype),
            values: Tensor::<4>::zeros(shape, &w.device).cast(w.dtype),
        })
    }
    fn forward(
        &mut self,
        x: Tensor<3>,
        position: usize,
        c: &DecoderConfig,
        cosine: &Tensor<4>,
        sine: &Tensor<4>,
    ) -> Tensor<3> {
        let normalized = self.input.forward(x.clone());
        let q = self
            .query_norm
            .forward(self.query.forward(normalized.clone()).reshape([
                1,
                c.num_attention_heads,
                1,
                c.head_dim,
            ]));
        let k = self
            .key_norm
            .forward(self.key.forward(normalized.clone()).reshape([
                1,
                c.num_key_value_heads,
                1,
                c.head_dim,
            ]));
        let v = self
            .value
            .forward(normalized)
            .reshape([1, c.num_key_value_heads, 1, c.head_dim]);
        let q = rotary(q, cosine, sine);
        let k = rotary(k, cosine, sine);
        let selection = [
            0..1,
            0..c.num_key_value_heads,
            position..position + 1,
            0..c.head_dim,
        ];
        // Moving the cache into slice_assign permits device-side in-place
        // updates. No alias of the full allocation survives between steps.
        self.keys
            .inplace(|cache| cache.slice_assign(selection.clone(), k));
        self.values
            .inplace(|cache| cache.slice_assign(selection, v));
        let selection = [
            0..1,
            0..c.num_key_value_heads,
            0..position + 1,
            0..c.head_dim,
        ];
        let keys = expand_kv(self.keys.clone().slice(selection.clone()), c);
        let values = expand_kv(self.values.clone().slice(selection), c);
        let scores = (q * (c.head_dim as f32).sqrt().recip()).matmul(keys.swap_dims(2, 3));
        let mixed = activation::softmax(scores.cast(burn::tensor::DType::F32), 3)
            .cast(x.dtype())
            .matmul(values)
            .reshape([1, 1, c.num_attention_heads * c.head_dim]);
        let x = x + self.output.forward(mixed);
        let normalized = self.post.forward(x.clone());
        let gate = activation::silu(self.gate.forward(normalized.clone()));
        x + self.down.forward(gate * self.up.forward(normalized))
    }
}
pub(crate) fn rotary(x: Tensor<4>, cosine: &Tensor<4>, sine: &Tensor<4>) -> Tensor<4> {
    let [b, h, t, d] = x.dims();
    let first = x.clone().slice([0..b, 0..h, 0..t, 0..d / 2]);
    let second = x.clone().slice([0..b, 0..h, 0..t, d / 2..d]);
    x * cosine.clone() + Tensor::cat(vec![-second, first], 3) * sine.clone()
}
fn expand_kv(x: Tensor<4>, c: &DecoderConfig) -> Tensor<4> {
    let [b, h, t, d] = x.dims();
    x.reshape([b, h, 1, t, d])
        .repeat_dim(2, c.num_attention_heads / c.num_key_value_heads)
        .reshape([b, c.num_attention_heads, t, d])
}

/// One decoder's weights and independently owned KV cache. Resetting invalidates
/// the old prefix without reading or clearing unused device storage.
pub struct Decoder {
    layers: Vec<Layer>,
    norm: RmsNorm,
    config: DecoderConfig,
    capacity: usize,
    position: usize,
    cosine: Tensor<4>,
    sine: Tensor<4>,
}
impl Decoder {
    pub fn load(
        weights: &Weights<'_>,
        prefix: &str,
        config: DecoderConfig,
        capacity: std::num::NonZeroUsize,
    ) -> Result<Self, Error> {
        let capacity = capacity.get();
        if config.head_dim == 0
            || !config.head_dim.is_multiple_of(2)
            || config.num_key_value_heads == 0
            || !config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads)
            || capacity > config.max_position_embeddings
        {
            return Err(Error::Tensor {
                name: prefix.into(),
                reason: "invalid rotary, grouped attention, or context dimensions".into(),
            });
        }
        let (cosine, sine) = super::audio_transformer::rotary_tables(
            &weights.device,
            weights.dtype,
            capacity,
            config.head_dim,
            config.rope_theta,
        );
        Ok(Self {
            layers: (0..config.num_hidden_layers)
                .map(|i| Layer::load(weights, &format!("{prefix}.layers.{i}"), &config, capacity))
                .collect::<Result<_, _>>()?,
            norm: RmsNorm::load(weights, &format!("{prefix}.norm"), config.rms_norm_eps)?,
            cosine,
            sine,
            config,
            capacity,
            position: 0,
        })
    }
    pub fn reset(&mut self) {
        self.position = 0;
    }
    pub fn step(&mut self, mut hidden: Tensor<3>) -> Result<Tensor<3>, Error> {
        if self.position == self.capacity {
            return Err(Error::Context(self.capacity));
        }
        if hidden.dims() != [1, 1, self.config.hidden_size] {
            return Err(Error::Tensor {
                name: "hidden".into(),
                reason: "expected one token with the configured hidden width".into(),
            });
        }
        let slice = [
            0..1,
            0..1,
            self.position..self.position + 1,
            0..self.config.head_dim,
        ];
        let cosine = self.cosine.clone().slice(slice.clone());
        let sine = self.sine.clone().slice(slice);
        for layer in &mut self.layers {
            hidden = layer.forward(hidden, self.position, &self.config, &cosine, &sine);
        }
        self.position += 1;
        Ok(hidden)
    }
    pub fn normalize(&self, hidden: Tensor<3>) -> Tensor<3> {
        self.norm.forward(hidden)
    }
}
