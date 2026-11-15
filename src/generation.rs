//! Shared autoregressive generation with backend-owned tensors and KV state.
use crate::inference::Error;
use crate::sampling::{RepetitionPenalty, TokenSampler};

/// Execution operations needed by the shared Qwen generation loop.
/// Readbacks are asynchronous so the same session can execute inside a Web Worker.
#[allow(async_fn_in_trait)]
pub trait Model {
    type Hidden: Clone;
    fn groups(&self) -> usize;
    /// Fixed-shape backends can retain their model capacity; dynamic backends use the valid span.
    fn waveform_frames(&self, valid: usize, capacity: usize) -> usize {
        let _ = valid;
        capacity
    }
    fn reset(&mut self);
    fn reset_predictor(&mut self);
    fn text(&mut self, token: u32) -> Result<Self::Hidden, Error>;
    fn code(&mut self, group: usize, token: u32) -> Result<Self::Hidden, Error>;
    fn frame(&mut self, codes: &[u32], text: &Self::Hidden) -> Result<Self::Hidden, Error>;
    fn add(a: &Self::Hidden, b: &Self::Hidden) -> Result<Self::Hidden, Error>;
    fn talker(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error>;
    async fn talker_head(
        &mut self,
        hidden: Self::Hidden,
    ) -> Result<(Self::Hidden, Vec<f32>), Error>;
    fn project(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error>;
    fn predictor(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error>;
    async fn predictor_head(
        &mut self,
        group: usize,
        hidden: Self::Hidden,
    ) -> Result<Vec<f32>, Error>;
    async fn condition(
        &mut self,
        samples: &[f32],
        mel: &[f32],
        channels: usize,
    ) -> Result<(Self::Hidden, Vec<Vec<u32>>), Error>;
    async fn waveform(
        &mut self,
        packed: Vec<i32>,
        groups: usize,
        frames: usize,
    ) -> Result<Vec<f32>, Error>;
}
pub struct Selection<T, S> {
    pub talker: T,
    pub predictor: S,
    pub repetition: RepetitionPenalty,
    pub eos: u32,
    /// First reserved codec identifier. Tokens from here onward, except EOS,
    /// are excluded, following the model's generation contract.
    pub reserved_start: u32,
    pub minimum_frames: usize,
}

pub struct Generator<B: Model, T, S> {
    pub(crate) networks: B,
    selection: Selection<T, S>,
}

pub struct Sequence<'a, B: Model, T, S> {
    pub(crate) generator: &'a mut Generator<B, T, S>,
    pending: Option<B::Hidden>,
    history: Vec<u32>,
    finished: bool,
}

impl<B: Model, T: TokenSampler, S: TokenSampler> Generator<B, T, S> {
    pub fn embed_code(&mut self, group: usize, token: u32) -> Result<B::Hidden, Error> {
        self.networks.code(group, token)
    }
    pub fn embed_frame(&mut self, codes: &[u32], text: &B::Hidden) -> Result<B::Hidden, Error> {
        self.networks.frame(codes, text)
    }
    pub fn new(networks: B, selection: Selection<T, S>) -> Result<Self, Error> {
        if networks.groups() < 2 {
            return Err(Error::Contract("at least two code groups are required"));
        }
        Ok(Self {
            networks,
            selection,
        })
    }

    pub fn start(
        &mut self,
        prefill: impl IntoIterator<Item = B::Hidden>,
    ) -> Result<Sequence<'_, B, T, S>, Error> {
        self.networks.reset();
        let mut last = None;
        for hidden in prefill {
            last = Some(self.networks.talker(hidden)?);
        }
        let last = last.ok_or(Error::Contract("prefill must not be empty"))?;
        let pending = Some(last);
        Ok(Sequence {
            generator: self,
            pending,
            history: Vec::new(),
            finished: false,
        })
    }
}

impl<B: Model, T: TokenSampler, S: TokenSampler> Sequence<'_, B, T, S> {
    pub fn embed_frame(&mut self, codes: &[u32], text: &B::Hidden) -> Result<B::Hidden, Error> {
        self.generator.embed_frame(codes, text)
    }

    /// Extend the conditioning prefix without resetting the KV state.
    pub fn prefill(&mut self, hidden: B::Hidden) -> Result<(), Error> {
        self.pending = None;
        let hidden = self.generator.networks.talker(hidden)?;
        self.pending = Some(hidden);
        Ok(())
    }

    pub async fn next_frame(&mut self, text: &B::Hidden) -> Result<Option<Vec<u32>>, Error> {
        let Some(hidden) = self.pending.take() else {
            return if self.finished {
                Ok(None)
            } else {
                Err(Error::Contract("sequence failed; start a new request"))
            };
        };
        let (past, mut logits) = self.generator.networks.talker_head(hidden).await?;
        let selection = &mut self.generator.selection;
        if selection.eos as usize >= logits.len()
            || selection.reserved_start as usize > logits.len()
        {
            return Err(Error::Contract(
                "reserved token range is outside the vocabulary",
            ));
        }
        selection.repetition.apply(&mut logits, &self.history)?;
        for (index, value) in logits
            .iter_mut()
            .enumerate()
            .skip(selection.reserved_start as usize)
        {
            if index != selection.eos as usize {
                *value = f32::NEG_INFINITY;
            }
        }
        if self.history.len() < selection.minimum_frames {
            logits[selection.eos as usize] = f32::NEG_INFINITY;
        }
        let first = selection.talker.sample(&mut logits)?;
        if first == selection.eos {
            self.finished = true;
            return Ok(None);
        }
        let networks = &mut self.generator.networks;
        networks.reset_predictor();
        let past = networks.project(past)?;
        networks.predictor(past)?;
        let mut codes = vec![first];
        for index in 0..networks.groups() - 1 {
            let embedded = networks.code(index, codes[index])?;
            let projected = networks.project(embedded)?;
            let hidden = networks.predictor(projected)?;
            let mut logits = networks.predictor_head(index, hidden).await?;
            codes.push(selection.predictor.sample(&mut logits)?);
        }
        let embedded = networks.frame(&codes, text)?;
        let hidden = networks.talker(embedded)?;
        self.pending = Some(hidden);
        self.history.push(first);
        Ok(Some(codes))
    }
}
