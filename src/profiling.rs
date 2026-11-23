//! Test-only measurements of model work and caller-owned sampling.
use crate::sampling::{Error, Logits, TokenSampler};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Default, Serialize)]
pub(crate) struct Counter {
    pub calls: usize,
    pub seconds: f64,
}

pub(crate) struct Measurement<'a> {
    counter: &'a mut Counter,
    started: Instant,
}
impl Counter {
    pub fn measure(&mut self) -> Measurement<'_> {
        Measurement {
            counter: self,
            started: Instant::now(),
        }
    }
}
impl Drop for Measurement<'_> {
    fn drop(&mut self) {
        self.counter.calls += 1;
        self.counter.seconds += self.started.elapsed().as_secs_f64();
    }
}

#[derive(Clone, Default, Serialize)]
pub(crate) struct Profile(BTreeMap<&'static str, Counter>);
impl Profile {
    pub fn measure(&mut self, stage: &'static str) -> Measurement<'_> {
        self.0.entry(stage).or_default().measure()
    }
    pub fn record(&mut self, stage: &'static str, duration: Duration) {
        let counter = self.0.entry(stage).or_default();
        counter.calls += 1;
        counter.seconds += duration.as_secs_f64();
    }
    pub fn with(&self, other: &Self) -> Self {
        Self(
            self.0
                .iter()
                .chain(&other.0)
                .map(|(&name, &counter)| (name, counter))
                .collect(),
        )
    }
    pub fn since(&self, previous: &Self) -> Self {
        Self(
            self.0
                .iter()
                .map(|(&stage, counter)| {
                    let before = previous.0.get(stage).copied().unwrap_or_default();
                    (
                        stage,
                        Counter {
                            calls: counter.calls - before.calls,
                            seconds: counter.seconds - before.seconds,
                        },
                    )
                })
                .collect(),
        )
    }
}

pub(crate) struct TimedSampler<'a, S> {
    pub inner: S,
    pub counter: &'a mut Counter,
}
impl<S: TokenSampler> TokenSampler for TimedSampler<'_, S> {
    async fn sample<L: Logits>(&mut self, logits: L) -> Result<L::Token, Error> {
        let _measurement = self.counter.measure();
        self.inner.sample(logits).await
    }
}
