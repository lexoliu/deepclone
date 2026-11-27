use super::{
    Error, WeightSource,
    layers::{JoinedLinear, Linear, RmsNorm},
};
use burn::tensor::{Bool, Int, Tensor, activation};
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
    qk_norm: Tensor<2>,
    qkv: JoinedLinear<3>,
    output: Linear,
    gate_up: JoinedLinear<2>,
    down: Linear,
    kv: Option<(Tensor<4>, Tensor<4>)>,
}
impl Layer {
    fn load(
        w: &mut impl WeightSource,
        prefix: &str,
        c: &DecoderConfig,
        capacity: usize,
    ) -> Result<Self, Error> {
        let shape = [1, c.num_key_value_heads, capacity, c.head_dim];
        let qk_norm = w.stack_1d(&[
            &format!("{prefix}.self_attn.q_norm.weight"),
            &format!("{prefix}.self_attn.k_norm.weight"),
        ])?;
        Ok(Self {
            input: RmsNorm::load(w, &format!("{prefix}.input_layernorm"), c.rms_norm_eps)?,
            post: RmsNorm::load(
                w,
                &format!("{prefix}.post_attention_layernorm"),
                c.rms_norm_eps,
            )?,
            qk_norm,
            qkv: JoinedLinear::load(
                w,
                ["q", "k", "v"].map(|name| format!("{prefix}.self_attn.{name}_proj")),
            )?,
            output: Linear::load(w, &format!("{prefix}.self_attn.o_proj"))?,
            gate_up: JoinedLinear::load(
                w,
                ["gate", "up"].map(|name| format!("{prefix}.mlp.{name}_proj")),
            )?,
            // The gated activation accumulates in FP32 while projection weights
            // retain model storage; GEMV converts each weight as it is read.
            down: Linear::load(w, &format!("{prefix}.mlp.down_proj"))?,
            kv: Some((
                super::zeros_persistent(shape, w.device(), w.dtype()),
                super::zeros_persistent(shape, w.device(), w.dtype()),
            )),
        })
    }
    /// Run one layer given the residual stream `x` and its already-normalized
    /// input `normalized`. Returns the updated residual stream before adding
    /// `down` (still in FP32) so the next layer can fuse the cast, residual
    /// addition, and input normalization into one dispatch.
    fn forward(
        &mut self,
        x: Tensor<3>,
        normalized: Tensor<3>,
        position: usize,
        c: &DecoderConfig,
        rotary: &Tensor<3>,
        mask: Option<&Tensor<4, Bool>>,
    ) -> (Tensor<3>, Tensor<3>) {
        let tokens = x.dims()[1];
        let scale = (c.head_dim as f32).sqrt().recip();
        let qkv = self.qkv.raw(normalized);
        // Moving both caches into the fused kernel permits in-place updates;
        // no alias of the full allocation survives between steps.
        let (keys, values) = self.kv.take().expect("KV cache present");
        let (q, keys, values) = super::fused::qkv_rope_cache(
            qkv,
            self.qk_norm.clone(),
            rotary.clone(),
            keys,
            values,
            position,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.rms_norm_eps,
            scale,
        );
        let mixed = if tokens == 1 {
            super::attention::decode(q, keys.clone(), values.clone(), position + tokens)
        } else if super::attention::supports_fused(position + tokens, c.head_dim) {
            super::attention::prefill(q, keys.clone(), values.clone(), position)
        } else {
            let selection = [
                0..1,
                0..c.num_key_value_heads,
                0..position + tokens,
                0..c.head_dim,
            ];
            let groups = c.num_attention_heads / c.num_key_value_heads;
            // Broadcast shared K/V over query groups instead of materializing repeated heads.
            let q = q.reshape([1, c.num_key_value_heads, groups, tokens, c.head_dim]);
            let keys_view = keys.clone().slice(selection.clone()).reshape([
                1,
                c.num_key_value_heads,
                1,
                position + tokens,
                c.head_dim,
            ]);
            let values_view = values.clone().slice(selection).reshape([
                1,
                c.num_key_value_heads,
                1,
                position + tokens,
                c.head_dim,
            ]);
            let scores = q
                .matmul(keys_view.swap_dims(3, 4))
                .cast(burn::tensor::DType::F32);
            let scores = match mask {
                Some(mask) => scores.mask_fill(mask.clone().unsqueeze(), f32::NEG_INFINITY),
                None => scores,
            };
            activation::softmax(scores, 4)
                .cast(x.dtype())
                .matmul(values_view)
                .reshape([1, c.num_attention_heads, tokens, c.head_dim])
                .swap_dims(1, 2)
                .reshape([1, tokens, c.num_attention_heads * c.head_dim])
        };
        self.kv = Some((keys, values));
        let projected = self.output.forward(mixed);
        let (x, normalized) = self.post.add_forward(x, projected);
        let gate_up = self.gate_up.raw(normalized);
        let down = self.down.forward(super::fused::gated_silu(gate_up));
        (x, down)
    }
}
pub(crate) fn rotary(x: Tensor<4>, cosine: &Tensor<4>, sine: &Tensor<4>) -> Tensor<4> {
    super::fused::rotary(x, cosine.clone(), sine.clone())
}
/// One decoder's weights and independently owned KV cache. Resetting invalidates
/// the old prefix without reading or clearing unused device storage.
pub struct Decoder {
    layers: Vec<Layer>,
    norm: RmsNorm,
    config: DecoderConfig,
    capacity: usize,
    position: usize,
    rotary: Tensor<3>,
}
impl Decoder {
    pub fn load(
        weights: &mut impl WeightSource,
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
        let rotary = super::audio_transformer::rotary_packed_persistent(
            weights.device(),
            weights.dtype(),
            0,
            capacity,
            config.head_dim,
            config.rope_theta,
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Layer::load(
                weights,
                &format!("{prefix}.layers.{i}"),
                &config,
                capacity,
            )?);
            // Submit per-layer concatenations so consumed source buffers return
            // to CubeCL's allocator before the next layer allocates.
            weights
                .device()
                .flush()
                .map_err(|error| Error::Execution(error.to_string()))?;
        }
        Ok(Self {
            layers,
            norm: RmsNorm::load(weights, &format!("{prefix}.norm"), config.rms_norm_eps)?,
            rotary,
            config,
            capacity,
            position: 0,
        })
    }
    pub fn reset(&mut self) {
        self.position = 0;
    }
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn step(&mut self, hidden: Tensor<3>) -> Result<Tensor<3>, Error> {
        self.step_inner(hidden, true)
    }
    pub(crate) fn step_unflushed(&mut self, hidden: Tensor<3>) -> Result<Tensor<3>, Error> {
        self.step_inner(hidden, false)
    }
    fn step_inner(&mut self, mut hidden: Tensor<3>, flush: bool) -> Result<Tensor<3>, Error> {
        let [batch, tokens, width] = hidden.dims();
        if tokens > self.capacity - self.position {
            return Err(Error::Context(self.capacity));
        }
        if batch != 1 || tokens == 0 || width != self.config.hidden_size {
            return Err(Error::Tensor {
                name: "hidden".into(),
                reason: "expected a nonempty token sequence with the configured hidden width"
                    .into(),
            });
        }
        let mask = if tokens > 1
            && !super::attention::supports_fused(self.position + tokens, self.config.head_dim)
        {
            let device = hidden.device();
            let queries = Tensor::<1, Int>::arange(
                self.position as i64..(self.position + tokens) as i64,
                &device,
            )
            .reshape([tokens, 1]);
            let keys = Tensor::<1, Int>::arange(0..(self.position + tokens) as i64, &device)
                .reshape([1, self.position + tokens]);
            Some(
                (queries - keys)
                    .lower_elem(0)
                    .reshape([1, 1, tokens, self.position + tokens]),
            )
        } else {
            None
        };
        if !self.layers.is_empty() {
            let dtype = hidden.dtype();
            let mut normalized = self.layers[0].input.forward(hidden.clone());
            for index in 0..self.layers.len() {
                let (residual, down) = self.layers[index].forward(
                    hidden,
                    normalized,
                    self.position,
                    &self.config,
                    &self.rotary,
                    mask.as_ref(),
                );
                if index + 1 < self.layers.len() {
                    (hidden, normalized) = self.layers[index + 1].input.add_forward(residual, down);
                } else {
                    hidden = residual + down.cast(dtype);
                    break;
                }
            }
        }
        // Submit each Talker step without waiting for GPU completion so long
        // reference prefixes never accumulate one fusion queue. Short Predictor
        // bursts defer flushing until the frame's token readback.
        if flush {
            hidden
                .device()
                .flush()
                .map_err(|error| Error::Execution(error.to_string()))?;
        }
        self.position += tokens;
        Ok(hidden)
    }
    pub fn normalize(&self, hidden: Tensor<3>) -> Tensor<3> {
        self.norm.forward(hidden)
    }
}
