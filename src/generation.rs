//! Worker-owned autoregressive generation with native tensor handoff.
use crate::{
    decoder::Decoder,
    sampling::{RepetitionPenalty, TokenSampler},
};
use ondevice_coreml::{Function, Inputs, Tensor, f16};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    CoreMl(#[from] ondevice_coreml::Error),
    #[error("Talker: {0}")]
    Talker(crate::decoder::Error),
    #[error("code predictor: {0}")]
    Predictor(crate::decoder::Error),
    #[error(transparent)]
    Sampling(#[from] crate::sampling::Error),
    #[error("generation contract: {0}")]
    Contract(&'static str),
    #[error("token embedding {token}: {source}")]
    Embedding {
        token: u32,
        source: ondevice_coreml::Error,
    },
    #[error("codec frame embedding: {0}")]
    Frame(ondevice_coreml::Error),
    #[error("output head: {0}")]
    Head(ondevice_coreml::Error),
}

/// The predictor input mapping is identity for 0.6B and learned for 1.7B.
pub trait PredictorProjection {
    fn project(&mut self, hidden: Tensor<f16>) -> Result<Tensor<f16>, Error>;
}
pub struct IdentityProjection;
impl PredictorProjection for IdentityProjection {
    fn project(&mut self, hidden: Tensor<f16>) -> Result<Tensor<f16>, Error> {
        Ok(hidden)
    }
}
impl PredictorProjection for Function {
    fn project(&mut self, hidden: Tensor<f16>) -> Result<Tensor<f16>, Error> {
        let mut inputs = Inputs::new();
        inputs.insert("hidden", &hidden)?;
        Ok(self.predict(&inputs)?.tensor("hidden_out")?)
    }
}

pub struct Networks<P> {
    pub talker: Decoder,
    pub talker_head: Function,
    pub predictor: Decoder,
    pub predictor_heads: Vec<Function>,
    pub predictor_projection: P,
    pub codec_embeddings: Vec<Function>,
    pub codec_frame: Function,
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

pub struct Generator<P, T, S> {
    networks: Networks<P>,
    selection: Selection<T, S>,
}

pub struct Sequence<'a, P, T, S> {
    generator: &'a mut Generator<P, T, S>,
    pending: Option<(Tensor<f16>, Vec<f32>)>,
    trailing: std::vec::IntoIter<Tensor<f16>>,
    padding: Tensor<f16>,
    history: Vec<u32>,
    finished: bool,
}

impl<P: PredictorProjection, T: TokenSampler, S: TokenSampler> Generator<P, T, S> {
    pub fn embed_code(&mut self, group: usize, token: u32) -> Result<Tensor<f16>, Error> {
        let function = self
            .networks
            .codec_embeddings
            .get_mut(group)
            .ok_or(Error::Contract("code group is outside the model"))?;
        embed(function, token)
    }
    pub fn embed_frame(&mut self, codes: &[u32], text: &Tensor<f16>) -> Result<Tensor<f16>, Error> {
        frame(&mut self.networks.codec_frame, codes, text)
    }
    pub fn new(networks: Networks<P>, selection: Selection<T, S>) -> Result<Self, Error> {
        if networks.predictor_heads.is_empty()
            || networks.codec_embeddings.len() != networks.predictor_heads.len() + 1
        {
            return Err(Error::Contract(
                "one codec embedding per group and one predictor head per residual group are required",
            ));
        }
        Ok(Self {
            networks,
            selection,
        })
    }

    /// Starts an independent request; borrowing prevents two requests from
    /// sharing these native states concurrently. Failure requires a new request.
    pub fn start(
        &mut self,
        prefill: impl IntoIterator<Item = Tensor<f16>>,
        trailing: Vec<Tensor<f16>>,
        padding: Tensor<f16>,
    ) -> Result<Sequence<'_, P, T, S>, Error> {
        self.networks.talker.reset();
        self.networks.predictor.reset();
        let mut last = None;
        for hidden in prefill {
            last = Some(self.networks.talker.step(hidden).map_err(Error::Talker)?);
        }
        let last = last.ok_or(Error::Contract("prefill must not be empty"))?;
        let pending = Some(head(&mut self.networks.talker_head, &last)?);
        Ok(Sequence {
            generator: self,
            pending,
            trailing: trailing.into_iter(),
            padding,
            history: Vec::new(),
            finished: false,
        })
    }
}

impl<P: PredictorProjection, T: TokenSampler, S: TokenSampler> Sequence<'_, P, T, S> {
    pub fn next_frame(&mut self) -> Result<Option<Vec<u32>>, Error> {
        let Some((past, mut logits)) = self.pending.take() else {
            return if self.finished {
                Ok(None)
            } else {
                Err(Error::Contract("sequence failed; start a new request"))
            };
        };
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
        networks.predictor.reset();
        let past = networks.predictor_projection.project(past)?;
        networks.predictor.step(past).map_err(Error::Predictor)?;
        let mut codes = vec![first];
        for index in 0..networks.predictor_heads.len() {
            let embedded = embed(&mut networks.codec_embeddings[index], codes[index])?;
            let projected = networks.predictor_projection.project(embedded)?;
            let hidden = networks
                .predictor
                .step(projected)
                .map_err(Error::Predictor)?;
            let (_, mut logits) = head(&mut networks.predictor_heads[index], &hidden)?;
            codes.push(selection.predictor.sample(&mut logits)?);
        }
        let text = self.trailing.next().unwrap_or_else(|| self.padding.clone());
        let embedded = frame(&mut networks.codec_frame, &codes, &text)?;
        let hidden = networks.talker.step(embedded).map_err(Error::Talker)?;
        self.pending = Some(head(&mut networks.talker_head, &hidden)?);
        self.history.push(first);
        Ok(Some(codes))
    }
}

fn frame(function: &mut Function, codes: &[u32], text: &Tensor<f16>) -> Result<Tensor<f16>, Error> {
    let ids = codes
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| Error::Contract("codec token exceeds int32")))
        .collect::<Result<Vec<_>, _>>()?;
    let ids = Tensor::from_slice(&[1, ids.len()], &ids)?;
    let mut inputs = Inputs::new();
    inputs.insert("codes", &ids)?;
    inputs.insert("text", text)?;
    Ok(function
        .predict(&inputs)
        .map_err(Error::Frame)?
        .tensor("hidden")?)
}

pub fn embed(function: &mut Function, token: u32) -> Result<Tensor<f16>, Error> {
    let identifier = token;
    let token = i32::try_from(token).map_err(|_| Error::Contract("token exceeds int32"))?;
    let tensor = Tensor::from_slice(&[1, 1], &[token])?;
    let mut inputs = Inputs::new();
    inputs.insert("token", &tensor)?;
    Ok(function
        .predict(&inputs)
        .map_err(|source| Error::Embedding {
            token: identifier,
            source,
        })?
        .tensor("hidden")?)
}

fn head(function: &mut Function, hidden: &Tensor<f16>) -> Result<(Tensor<f16>, Vec<f32>), Error> {
    let mut inputs = Inputs::new();
    inputs.insert("hidden", hidden)?;
    let prediction = function.predict(&inputs).map_err(Error::Head)?;
    let logits = prediction
        .tensor::<f16>("logits")?
        .read()
        .into_data()
        .into_iter()
        .map(f16::to_f32)
        .collect();
    Ok((prediction.tensor("normalized")?, logits))
}
