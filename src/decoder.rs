//! Qwen's autoregressive decoder chain. Call from an inference worker.
use crate::coreml_batch::BatchSizes;
use ondevice_coreml::{ComputeUnits, Function, Inputs, Model, Session, Tensor, f16};
#[cfg(feature = "inference")]
use std::num::NonZeroUsize;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    CoreMl(#[from] ondevice_coreml::Error),
    #[error("decoder contract: {0}")]
    Contract(&'static str),
    #[error("decoder context exhausted after {0} tokens")]
    ContextExhausted(usize),
    #[error("decoder sequence failed; reset before continuing")]
    InvalidState,
    #[error("decoder chunk {chunk}, position {position}: {source}")]
    Prediction {
        chunk: usize,
        position: usize,
        source: ondevice_coreml::Error,
    },
}

/// Independently owned chunk states for one request. Output tensors remain in
/// native storage between chunks. Reset starts another independent sequence.
pub struct Decoder {
    chunks: Vec<Session>,
    frequencies: Vec<f32>,
    mask: Vec<f16>,
    hidden_size: usize,
    #[cfg(feature = "inference")]
    sequence_lengths: BatchSizes,
    position: Option<usize>,
}
impl Decoder {
    pub fn open<P: AsRef<Path>>(
        paths: impl IntoIterator<Item = P>,
        rope_theta: f32,
        units: ComputeUnits,
    ) -> Result<Self, Error> {
        Self::open_with_lengths(paths, rope_theta, BatchSizes::default(), units)
    }
    pub(crate) fn open_with_lengths<P: AsRef<Path>>(
        paths: impl IntoIterator<Item = P>,
        rope_theta: f32,
        sequence_lengths: BatchSizes,
        units: ComputeUnits,
    ) -> Result<Self, Error> {
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(Error::Contract("RoPE theta must be finite and positive"));
        }
        let functions = paths
            .into_iter()
            .map(|path| Model::open(path)?.function(None, units))
            .collect::<Result<Vec<Function>, ondevice_coreml::Error>>()?;
        let first = functions
            .first()
            .ok_or(Error::Contract("at least one chunk is required"))?;
        let shape = |name: &str| {
            first
                .description()
                .inputs
                .get(name)
                .map(|feature| feature.default_shape.as_slice())
        };
        let Some([1, hidden_size, 1, 1]) = shape("hidden") else {
            return Err(Error::Contract(
                "hidden must have shape [1, channels, 1, 1]",
            ));
        };
        let Some([1, 1, head_dim, 1]) = shape("cosine") else {
            return Err(Error::Contract(
                "cosine must have shape [1, 1, head_dim, 1]",
            ));
        };
        let Some([1, 1, 1, capacity]) = shape("mask") else {
            return Err(Error::Contract("mask must have shape [1, 1, 1, capacity]"));
        };
        let (hidden_size, head_dim, capacity) = (*hidden_size, *head_dim, *capacity);
        if !sequence_lengths.fits(capacity) {
            return Err(Error::Contract("sequence lengths exceed the context"));
        }
        if head_dim == 0 || head_dim % 2 != 0 || capacity == 0 {
            return Err(Error::Contract(
                "nonzero context and even rotary width are required",
            ));
        }
        for function in &functions {
            let description = function.description();
            for (name, expected) in [
                ("hidden", vec![1, hidden_size, 1, 1]),
                ("cosine", vec![1, 1, head_dim, 1]),
                ("sine", vec![1, 1, head_dim, 1]),
                ("mask", vec![1, 1, 1, capacity]),
            ] {
                if !description.inputs.get(name).is_some_and(|feature| {
                    feature.default_shape == expected
                        && feature.data_type == ondevice_coreml::DataType::Float16
                }) {
                    return Err(Error::Contract("chunk input shapes or dtypes disagree"));
                }
            }
            if description.states.is_empty() {
                return Err(Error::Contract("chunk must expose KV state"));
            }
            if !description
                .outputs
                .get("hidden_out")
                .is_some_and(|feature| {
                    feature.data_type == ondevice_coreml::DataType::Float16
                        && (feature.default_shape == [1, hidden_size, 1, 1]
                            || (sequence_lengths.flexible() && feature.default_shape.is_empty()))
                })
            {
                return Err(Error::Contract("chunk output must match hidden input"));
            }
        }
        let frequencies = (0..head_dim / 2)
            .map(|index| rope_theta.powf(-((2 * index) as f32) / head_dim as f32))
            .collect();
        Ok(Self {
            chunks: functions.into_iter().map(Function::into_session).collect(),
            frequencies,
            mask: vec![f16::MIN; capacity],
            hidden_size,
            #[cfg(feature = "inference")]
            sequence_lengths,
            position: Some(0),
        })
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn position(&self) -> Option<usize> {
        self.position
    }
    pub fn capacity(&self) -> usize {
        self.mask.len()
    }
    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.reset();
        }
        self.mask.fill(f16::MIN);
        self.position = Some(0);
    }
    /// Advance once. An inference failure invalidates this sequence: reset
    /// before retrying, since earlier chunks may already have updated state.
    pub fn step(&mut self, hidden: Tensor<f16>) -> Result<Tensor<f16>, Error> {
        self.span(hidden, 1)
    }
    /// Consume a known prefix in the model's supported causal batches. Single
    /// tokens and outputs passed between layer groups stay in native storage.
    #[cfg(feature = "inference")]
    pub(crate) fn prefill(
        &mut self,
        first: Tensor<f16>,
        remaining: impl IntoIterator<Item = Tensor<f16>>,
    ) -> Result<Tensor<f16>, Error> {
        let tokens: Vec<_> = std::iter::once(first).chain(remaining).collect();
        let position = self.position.ok_or(Error::InvalidState)?;
        if tokens.len() > self.capacity() - position {
            return Err(Error::ContextExhausted(self.capacity()));
        }
        if tokens
            .iter()
            .any(|token| token.shape() != [1, self.hidden_size, 1, 1])
        {
            return Err(Error::Contract("prefix token shape disagrees with decoder"));
        }
        let mut offset = 0;
        let mut last = None;
        while let Some(remaining) = NonZeroUsize::new(tokens.len() - offset) {
            let length = self.sequence_lengths.batch(remaining);
            let batch = &tokens[offset..offset + length];
            let hidden = if length == 1 {
                batch[0].clone()
            } else {
                let mut values = vec![f16::ZERO; self.hidden_size * length];
                for (index, token) in batch.iter().enumerate() {
                    for (channel, &value) in token.read().data().iter().enumerate() {
                        values[channel * length + index] = value;
                    }
                }
                Tensor::from_slice(&[1, self.hidden_size, 1, length], &values)?
            };
            last = Some(self.span(hidden, length)?);
            offset += length;
        }
        let last = last.expect("prefix contains its first token");
        let length = last.shape()[3];
        if length == 1 {
            Ok(last)
        } else {
            let values: Vec<_> = last
                .read()
                .data()
                .chunks_exact(length)
                .map(|channel| channel[length - 1])
                .collect();
            Ok(Tensor::from_slice(&[1, self.hidden_size, 1, 1], &values)?)
        }
    }
    fn span(&mut self, mut hidden: Tensor<f16>, length: usize) -> Result<Tensor<f16>, Error> {
        let position = self.position.ok_or(Error::InvalidState)?;
        if length > self.capacity() - position {
            return Err(Error::ContextExhausted(self.capacity()));
        }
        if hidden.shape() != [1, self.hidden_size, 1, length] {
            return Err(Error::Contract(
                "token embedding shape disagrees with decoder",
            ));
        }
        let mut cosines = Vec::with_capacity(self.frequencies.len() * 2 * length);
        let mut sines = Vec::with_capacity(self.frequencies.len() * 2 * length);
        for frequency in &self.frequencies {
            for index in position..position + length {
                let (sine, cosine) = (frequency * index as f32).sin_cos();
                cosines.push(f16::from_f32(cosine));
                sines.push(f16::from_f32(sine));
            }
        }
        cosines.extend_from_within(..);
        sines.extend_from_within(..);
        let shape = [1, 1, self.frequencies.len() * 2, length];
        let cosine = Tensor::from_slice(&shape, &cosines)?;
        let sine = Tensor::from_slice(&shape, &sines)?;
        let capacity = self.capacity();
        self.mask[capacity - position - length..capacity - position].fill(f16::ZERO);
        let mask = if length == 1 {
            Tensor::from_slice(&[1, 1, 1, capacity], &self.mask)?
        } else {
            let mut values = vec![f16::MIN; length * capacity];
            for index in 0..length {
                values[index * capacity + capacity - position - length
                    ..index * capacity + capacity - length + index + 1]
                    .fill(f16::ZERO);
            }
            Tensor::from_slice(&[1, 1, length, capacity], &values)?
        };
        // Prevent accidental continuation after a partially completed chain.
        let next_position = position + length;
        self.position = None;
        for (index, chunk) in self.chunks.iter_mut().enumerate() {
            let mut inputs = Inputs::new();
            inputs.insert("hidden", &hidden)?;
            inputs.insert("cosine", &cosine)?;
            inputs.insert("sine", &sine)?;
            inputs.insert("mask", &mask)?;
            hidden = chunk
                .predict(&inputs)
                .map_err(|source| Error::Prediction {
                    chunk: index,
                    position,
                    source,
                })?
                .tensor("hidden_out")?;
            if hidden.shape() != [1, self.hidden_size, 1, length] {
                return Err(Error::Contract("chunk output must match hidden input"));
            }
        }
        self.position = Some(next_position);
        Ok(hidden)
    }
}
