//! Shared autoregressive generation with backend-owned tensors and KV state.
use crate::inference::Error;
use crate::sampling::{Logits, RepetitionPenalty, TokenSampler};

/// Execution operations needed by the shared Qwen generation loop.
/// Readbacks are asynchronous so the same session can execute inside a Web Worker.
#[allow(async_fn_in_trait)]
pub trait Model {
    type Hidden: Clone;
    type Token: Clone;
    type Logits: Logits<Token = Self::Token>;
    fn code_token(&mut self, group: usize, token: &Self::Token) -> Result<Self::Hidden, Error>;
    async fn read_tokens(&self, tokens: &[Self::Token]) -> Result<Vec<u32>, Error>;
    fn groups(&self) -> usize;
    /// Available Talker KV positions in this backend's loaded model.
    fn context_capacity(&self) -> usize;
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
    /// Consume an ordered conditioning prefix, retaining only its last output.
    fn prefill(
        &mut self,
        first: Self::Hidden,
        remaining: impl IntoIterator<Item = Self::Hidden>,
    ) -> Result<Self::Hidden, Error> {
        let mut last = self.talker(first)?;
        for hidden in remaining {
            last = self.talker(hidden)?;
        }
        Ok(last)
    }
    async fn talker_head(
        &mut self,
        hidden: Self::Hidden,
    ) -> Result<(Self::Hidden, Self::Logits), Error>;
    fn project(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error>;
    fn predictor(&mut self, hidden: Self::Hidden) -> Result<Self::Hidden, Error>;
    async fn predictor_head(
        &mut self,
        group: usize,
        hidden: Self::Hidden,
    ) -> Result<Self::Logits, Error>;
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

/// EOS is checked before the frame allowance and before detail-code prediction.
pub enum Frame {
    Codes(Vec<u32>),
    Eos,
    Limit,
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
        let mut prefill = prefill.into_iter();
        let first = prefill
            .next()
            .ok_or(Error::Contract("prefill must not be empty"))?;
        let last = self.networks.prefill(first, prefill)?;
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
    pub fn prefill(
        &mut self,
        first: B::Hidden,
        remaining: impl IntoIterator<Item = B::Hidden>,
    ) -> Result<(), Error> {
        self.pending = None;
        let hidden = self.generator.networks.prefill(first, remaining)?;
        self.pending = Some(hidden);
        Ok(())
    }

    pub async fn next_frame(
        &mut self,
        text: &B::Hidden,
        maximum_frames: usize,
    ) -> Result<Frame, Error> {
        let Some(hidden) = self.pending.take() else {
            return if self.finished {
                Ok(Frame::Eos)
            } else {
                Err(Error::Contract("sequence failed; start a new request"))
            };
        };
        let (past, logits) = self.generator.networks.talker_head(hidden).await?;
        let selection = &mut self.generator.selection;
        let logits = logits.constrain(
            selection.repetition,
            &self.history,
            selection.eos,
            selection.reserved_start,
            self.history.len() < selection.minimum_frames,
        )?;
        let first_token = selection.talker.sample(logits).await?;
        let first = self
            .generator
            .networks
            .read_tokens(std::slice::from_ref(&first_token))
            .await?[0];
        if first == selection.eos {
            self.finished = true;
            return Ok(Frame::Eos);
        }
        if self.history.len() == maximum_frames {
            return Ok(Frame::Limit);
        }
        let networks = &mut self.generator.networks;
        networks.reset_predictor();
        let past = networks.project(past)?;
        networks.predictor(past)?;
        let mut tokens = vec![first_token];
        for index in 0..networks.groups() - 1 {
            let embedded = networks.code_token(index, &tokens[index])?;
            let projected = networks.project(embedded)?;
            let hidden = networks.predictor(projected)?;
            let logits = networks.predictor_head(index, hidden).await?;
            tokens.push(selection.predictor.sample(logits).await?);
        }
        let codes = networks.read_tokens(&tokens).await?;
        let embedded = networks.frame(&codes, text)?;
        let hidden = networks.talker(embedded)?;
        self.pending = Some(hidden);
        self.history.push(first);
        Ok(Frame::Codes(codes))
    }
}
