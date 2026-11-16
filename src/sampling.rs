//! Token selection with caller-owned randomness.
use rand::RngExt;
use std::collections::BTreeSet;
use std::num::NonZeroUsize;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0} must be finite and greater than zero")]
    Positive(&'static str),
    #[error("top-p must lie in (0, 1]")]
    Probability,
    #[error("logits contain NaN or positive infinity")]
    InvalidLogits,
    #[error("no token has a finite logit")]
    NoCandidate,
    #[error("device sampling: {0}")]
    Device(String),
    #[error("token identifier cannot be represented as u32")]
    Vocabulary,
    #[error("history token {0} is outside the vocabulary")]
    History(u32),
}

/// Validated sampling parameters. Vocabulary size bounds top-k naturally.
#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    pub(crate) temperature: f32,
    pub(crate) top_k: Option<NonZeroUsize>,
    pub(crate) top_p: f32,
}
impl Parameters {
    pub fn new(temperature: f32, top_k: Option<NonZeroUsize>, top_p: f32) -> Result<Self, Error> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(Error::Positive("temperature"));
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(Error::Probability);
        }
        Ok(Self {
            temperature,
            top_k,
            top_p,
        })
    }
}

/// Apply the sign-sensitive repetition penalty once per historical token.
#[derive(Debug, Clone, Copy)]
pub struct RepetitionPenalty(pub(crate) f32);
impl RepetitionPenalty {
    pub fn new(value: f32) -> Result<Self, Error> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::Positive("repetition penalty"));
        }
        Ok(Self(value))
    }
    pub fn apply(&self, logits: &mut [f32], history: &[u32]) -> Result<(), Error> {
        for token in history.iter().copied().collect::<BTreeSet<_>>() {
            let value = logits
                .get_mut(token as usize)
                .ok_or(Error::History(token))?;
            *value = if *value < 0.0 {
                *value * self.0
            } else {
                *value / self.0
            };
        }
        Ok(())
    }
}

/// Backend-owned scores and selected tokens. Device implementations enqueue
/// sampling without forcing a readback between autoregressive predictions.
#[allow(async_fn_in_trait)]
pub trait Logits: Sized {
    type Token: Clone;
    fn constrain(
        self,
        penalty: RepetitionPenalty,
        history: &[u32],
        eos: u32,
        reserved_start: u32,
        suppress_eos: bool,
    ) -> Result<Self, Error>;
    fn greedy(self) -> Result<Self::Token, Error>;
    fn multinomial<R: rand::Rng>(self, sampler: &mut Multinomial<R>) -> Result<Self::Token, Error>;
    async fn sample_host<S: HostSampler>(self, sampler: &mut S) -> Result<Self::Token, Error>;
}

#[allow(async_fn_in_trait)]
pub trait TokenSampler {
    async fn sample<L: Logits>(&mut self, logits: L) -> Result<L::Token, Error>;
}

/// Custom host algorithms opt into readback explicitly through `Host`.
pub trait HostSampler {
    fn sample(&mut self, logits: &mut [f32]) -> Result<u32, Error>;
}
pub struct Host<S>(pub S);
impl<S: HostSampler> TokenSampler for Host<S> {
    async fn sample<L: Logits>(&mut self, logits: L) -> Result<L::Token, Error> {
        logits.sample_host(&mut self.0).await
    }
}
#[derive(Default)]
pub struct Greedy;
impl TokenSampler for Greedy {
    async fn sample<L: Logits>(&mut self, logits: L) -> Result<L::Token, Error> {
        logits.greedy()
    }
}

pub struct Multinomial<R> {
    pub(crate) random: R,
    pub(crate) parameters: Parameters,
    candidates: Vec<(usize, f64)>,
}
impl<R: rand::Rng> Multinomial<R> {
    pub fn new(random: R, parameters: Parameters) -> Self {
        Self {
            random,
            parameters,
            candidates: Vec::new(),
        }
    }
}
impl<R: rand::Rng> TokenSampler for Multinomial<R> {
    async fn sample<L: Logits>(&mut self, logits: L) -> Result<L::Token, Error> {
        logits.multinomial(self)
    }
}
impl Logits for Vec<f32> {
    type Token = u32;
    fn constrain(
        mut self,
        penalty: RepetitionPenalty,
        history: &[u32],
        eos: u32,
        reserved_start: u32,
        suppress_eos: bool,
    ) -> Result<Self, Error> {
        if eos as usize >= self.len() || reserved_start as usize > self.len() {
            return Err(Error::Vocabulary);
        }
        penalty.apply(&mut self, history)?;
        for (i, value) in self.iter_mut().enumerate().skip(reserved_start as usize) {
            if i != eos as usize {
                *value = f32::NEG_INFINITY;
            }
        }
        if suppress_eos {
            self[eos as usize] = f32::NEG_INFINITY;
        }
        Ok(self)
    }
    fn greedy(self) -> Result<u32, Error> {
        validate(&self)?;
        let index = self
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
            .ok_or(Error::NoCandidate)?
            .0;
        u32::try_from(index).map_err(|_| Error::Vocabulary)
    }
    fn multinomial<R: rand::Rng>(self, sampler: &mut Multinomial<R>) -> Result<u32, Error> {
        validate(&self)?;
        sampler.candidates.clear();
        sampler.candidates.extend(
            self.iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, &v)| (i, f64::from(v) / f64::from(sampler.parameters.temperature))),
        );
        if sampler.candidates.is_empty() {
            return Err(Error::NoCandidate);
        }
        if let Some(k) = sampler.parameters.top_k
            && k.get() < sampler.candidates.len()
        {
            sampler
                .candidates
                .select_nth_unstable_by(k.get() - 1, |a, b| b.1.total_cmp(&a.1));
            // Keep ties at the threshold, matching self-based top-k filtering.
            let cutoff = sampler.candidates[k.get() - 1].1;
            sampler.candidates.retain(|(_, value)| *value >= cutoff);
        }
        sampler
            .candidates
            .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let maximum = sampler.candidates[0].1;
        let mut total = 0.0;
        for (_, value) in &mut sampler.candidates {
            *value = (*value - maximum).exp();
            total += *value;
        }
        let threshold = f64::from(sampler.parameters.top_p) * total;
        let mut retained = 0.0;
        let mut count = 0;
        for (_, weight) in &sampler.candidates {
            retained += weight;
            count += 1;
            if retained >= threshold {
                break;
            }
        }
        sampler.candidates.truncate(count);
        let draw = sampler.random.random::<f64>() * retained;
        let mut cumulative = 0.0;
        for &(index, weight) in &sampler.candidates {
            cumulative += weight;
            if draw < cumulative {
                return u32::try_from(index).map_err(|_| Error::Vocabulary);
            }
        }
        Err(Error::NoCandidate)
    }
    async fn sample_host<S: HostSampler>(mut self, sampler: &mut S) -> Result<u32, Error> {
        sampler.sample(&mut self)
    }
}
fn validate(logits: &[f32]) -> Result<(), Error> {
    if logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        return Err(Error::InvalidLogits);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn greedy_ignores_suppressed_tokens_and_breaks_ties_by_index() {
        assert_eq!(vec![f32::NEG_INFINITY, 2., 2.].greedy().unwrap(), 1);
        assert!(matches!(
            vec![f32::NEG_INFINITY].greedy(),
            Err(Error::NoCandidate)
        ));
    }

    #[test]
    fn nucleus_and_repetition_filter_before_drawing() {
        let params = Parameters::new(1., None, 0.5).unwrap();
        let mut sampler = Multinomial::new(StdRng::seed_from_u64(42), params);
        // Penalizing a repeated token once puts token 1 above the nucleus cutoff.
        for _ in 0..32 {
            let mut logits = [4., 3., -10.];
            RepetitionPenalty::new(4.)
                .unwrap()
                .apply(&mut logits, &[0, 0])
                .unwrap();
            assert_eq!(logits.to_vec().multinomial(&mut sampler).unwrap(), 1);
        }
    }
}
