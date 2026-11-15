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
    #[error("token identifier cannot be represented as u32")]
    Vocabulary,
    #[error("history token {0} is outside the vocabulary")]
    History(u32),
}

/// Validated sampling parameters. Vocabulary size bounds top-k naturally.
#[derive(Debug, Clone, Copy)]
pub struct Parameters {
    temperature: f32,
    top_k: Option<NonZeroUsize>,
    top_p: f32,
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
pub struct RepetitionPenalty(f32);
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

pub trait TokenSampler {
    /// Negative infinity suppresses a token. Selection may transform logits.
    fn sample(&mut self, logits: &mut [f32]) -> Result<u32, Error>;
}

#[derive(Default)]
pub struct Greedy;
impl TokenSampler for Greedy {
    fn sample(&mut self, logits: &mut [f32]) -> Result<u32, Error> {
        validate(logits)?;
        let index = logits
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
            .ok_or(Error::NoCandidate)?
            .0;
        u32::try_from(index).map_err(|_| Error::Vocabulary)
    }
}

pub struct Multinomial<R> {
    random: R,
    parameters: Parameters,
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
    fn sample(&mut self, logits: &mut [f32]) -> Result<u32, Error> {
        validate(logits)?;
        self.candidates.clear();
        self.candidates.extend(
            logits
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, &v)| (i, f64::from(v) / f64::from(self.parameters.temperature))),
        );
        if self.candidates.is_empty() {
            return Err(Error::NoCandidate);
        }
        if let Some(k) = self.parameters.top_k
            && k.get() < self.candidates.len()
        {
            self.candidates
                .select_nth_unstable_by(k.get() - 1, |a, b| b.1.total_cmp(&a.1));
            // Keep ties at the threshold, matching logits-based top-k filtering.
            let cutoff = self.candidates[k.get() - 1].1;
            self.candidates.retain(|(_, value)| *value >= cutoff);
        }
        self.candidates
            .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let maximum = self.candidates[0].1;
        let mut total = 0.0;
        for (_, value) in &mut self.candidates {
            *value = (*value - maximum).exp();
            total += *value;
        }
        let threshold = f64::from(self.parameters.top_p) * total;
        let mut retained = 0.0;
        let mut count = 0;
        for (_, weight) in &self.candidates {
            retained += weight;
            count += 1;
            if retained >= threshold {
                break;
            }
        }
        self.candidates.truncate(count);
        let draw = self.random.random::<f64>() * retained;
        let mut cumulative = 0.0;
        for &(index, weight) in &self.candidates {
            cumulative += weight;
            if draw < cumulative {
                return u32::try_from(index).map_err(|_| Error::Vocabulary);
            }
        }
        Err(Error::NoCandidate)
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
        assert_eq!(Greedy.sample(&mut [f32::NEG_INFINITY, 2., 2.]).unwrap(), 1);
        assert!(matches!(
            Greedy.sample(&mut [f32::NEG_INFINITY]),
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
            assert_eq!(sampler.sample(&mut logits).unwrap(), 1);
        }
    }
}
