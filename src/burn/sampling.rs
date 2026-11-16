//! Device-resident token selection with caller-owned randomness.
use crate::sampling::{Error, HostSampler, Logits, Multinomial, RepetitionPenalty};
use burn::tensor::{DType, Int, Tensor, TensorData};
use rand::RngExt;

pub struct DeviceLogits(pub Tensor<1>);

/// Negative indices retain sampling failures until the next mandatory token
/// readback. No waveform is emitted from an invalid frame.
#[derive(Clone)]
pub struct DeviceToken(Tensor<1, Int>);
impl DeviceToken {
    pub(crate) fn index(&self) -> Tensor<1, Int> {
        // Failed selections still pass through queued device operations safely;
        // read_tokens surfaces the original error before the frame is consumed.
        self.0.clone().clamp_min(0)
    }
}
fn execution(error: impl std::fmt::Display) -> Error {
    Error::Device(error.to_string())
}
fn checked(logits: Tensor<1>, index: Tensor<1, Int>) -> DeviceToken {
    let invalid = logits
        .clone()
        .is_nan()
        .bool_or(logits.clone().equal_elem(f32::INFINITY))
        .any();
    let finite = logits
        .clone()
        .greater_elem(f32::NEG_INFINITY)
        .bool_and(logits.lower_elem(f32::INFINITY))
        .any();
    // -1 and -2 are internal status codes, outside the nonnegative token domain.
    DeviceToken(
        index
            .mask_fill(finite.bool_not(), -1)
            .mask_fill(invalid, -2),
    )
}
pub(crate) async fn read_tokens(tokens: &[DeviceToken]) -> Result<Vec<u32>, Error> {
    let values = Tensor::cat(tokens.iter().map(|token| token.0.clone()).collect(), 0)
        .into_data_async()
        .await
        .map_err(execution)?
        .try_into_vec::<i32>()
        .map_err(execution)?;
    values
        .into_iter()
        .map(|index| match index {
            -1 => Err(Error::NoCandidate),
            -2 => Err(Error::InvalidLogits),
            value => u32::try_from(value).map_err(|_| Error::Vocabulary),
        })
        .collect()
}
impl Logits for DeviceLogits {
    type Token = DeviceToken;
    fn constrain(
        self,
        penalty: RepetitionPenalty,
        history: &[u32],
        eos: u32,
        reserved_start: u32,
        suppress_eos: bool,
    ) -> Result<Self, Error> {
        let n = self.0.dims()[0];
        if eos as usize >= n || reserved_start as usize > n {
            return Err(Error::Vocabulary);
        }
        let mut factors = vec![1f32; n];
        for &token in history {
            *factors
                .get_mut(token as usize)
                .ok_or(Error::History(token))? = penalty.0;
        }
        for (index, factor) in factors.iter_mut().enumerate().skip(reserved_start as usize) {
            if index != eos as usize {
                *factor = 0.;
            }
        }
        if suppress_eos {
            factors[eos as usize] = 0.;
        }
        let factor = Tensor::<1>::from_data(TensorData::new(factors, [n]), &self.0.device());
        let values = self.0.cast(DType::F32);
        let negative = values.clone().lower_elem(0.);
        let suppressed = factor.clone().equal_elem(0.);
        Ok(Self(
            (values.clone() / factor.clone())
                .mask_where(negative, values * factor)
                .mask_fill(suppressed, f32::NEG_INFINITY),
        ))
    }
    fn greedy(self) -> Result<DeviceToken, Error> {
        Ok(checked(self.0.clone(), self.0.argmax(0).reshape([1])))
    }
    fn multinomial<R: rand::Rng>(self, sampler: &mut Multinomial<R>) -> Result<DeviceToken, Error> {
        let original = self.0.cast(DType::F32);
        let n = original.dims()[0];
        let (mut sorted, indices) = original.clone().sort_descending_with_indices(0);
        if let Some(k) = sampler.parameters.top_k.filter(|k| k.get() < n) {
            let cutoff = sorted.clone().narrow(0, k.get() - 1, 1);
            let excluded = sorted.clone().lower(cutoff);
            sorted = sorted.mask_fill(excluded, f32::NEG_INFINITY);
        }
        let maximum = sorted.clone().max_dim(0);
        let mut weights = ((sorted - maximum) / sampler.parameters.temperature).exp();
        if sampler.parameters.top_p < 1. {
            let threshold = weights.clone().sum_dim(0) * sampler.parameters.top_p;
            let excluded = (weights.clone().cumsum(0) - weights.clone()).greater_equal(threshold);
            weights = weights.mask_fill(excluded, 0.);
        }
        let cumulative = weights.cumsum(0);
        let total = cumulative.clone().narrow(0, n - 1, 1);
        // Rounding an f64 draw to f32 must not include the upper endpoint.
        let draw = (sampler.random.random::<f64>() as f32).min(f32::from_bits(1f32.to_bits() - 1));
        let rank = cumulative.greater(total * draw).int().argmax(0);
        Ok(checked(original, indices.gather(0, rank).reshape([1])))
    }
    async fn sample_host<S: HostSampler>(self, sampler: &mut S) -> Result<DeviceToken, Error> {
        let device = self.0.device();
        let mut values = super::read(self.0).await.map_err(execution)?;
        let token = sampler.sample(&mut values)?;
        if token as usize >= values.len() {
            return Err(Error::Vocabulary);
        }
        let index = i32::try_from(token).map_err(|_| Error::Vocabulary)?;
        Ok(DeviceToken(Tensor::from_data(
            TensorData::new(vec![index], [1]),
            &device,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::{Greedy, Parameters, TokenSampler};
    use burn::tensor::Device;
    use rand::{SeedableRng, rngs::StdRng};
    use std::num::NonZeroUsize;

    fn scores(device: &Device, values: &[f32]) -> DeviceLogits {
        DeviceLogits(Tensor::from_data(
            TensorData::new(values.to_vec(), [values.len()]),
            device,
        ))
    }
    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn device_selection_matches_host_sampling_and_rejects_invalid_scores() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for parameters in [
                Parameters::new(0.9, NonZeroUsize::new(3), 1.).unwrap(),
                Parameters::new(0.7, None, 0.8).unwrap(),
                Parameters::new(1., NonZeroUsize::new(1), 1.).unwrap(),
            ] {
                for values in [
                    vec![0.1, 2.3, -1.7, 0.9, f32::NEG_INFINITY],
                    vec![2., 2., 0., -1.],
                ] {
                    let mut host = Multinomial::new(StdRng::seed_from_u64(42), parameters);
                    let mut gpu = Multinomial::new(StdRng::seed_from_u64(42), parameters);
                    let mut expected = Vec::new();
                    let mut selected = Vec::new();
                    for _ in 0..64 {
                        expected.push(host.sample(values.clone()).await.unwrap());
                        selected.push(gpu.sample(scores(&device, &values)).await.unwrap());
                    }
                    assert_eq!(read_tokens(&selected).await.unwrap(), expected);
                }
            }
            let token = Greedy
                .sample(scores(&device, &[f32::NEG_INFINITY, 2., 2.]))
                .await
                .unwrap();
            assert_eq!(read_tokens(&[token]).await.unwrap(), [1]);
            for values in [vec![f32::NAN, 1.], vec![f32::INFINITY, 1.]] {
                let token = scores(&device, &values).greedy().unwrap();
                assert!(matches!(
                    read_tokens(&[token]).await,
                    Err(Error::InvalidLogits)
                ));
            }
            let token = scores(&device, &[f32::NEG_INFINITY; 2]).greedy().unwrap();
            assert!(matches!(
                read_tokens(&[token]).await,
                Err(Error::NoCandidate)
            ));
        });
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn device_repetition_and_reserved_tokens_match_host_constraints() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            let values = vec![-2., 3., 2., 1., -1., 4.];
            let penalty = RepetitionPenalty::new(1.05).unwrap();
            for suppress in [false, true] {
                let expected = values
                    .clone()
                    .constrain(penalty, &[0, 0, 1], 4, 3, suppress)
                    .unwrap();
                let result = scores(&device, &values)
                    .constrain(penalty, &[0, 0, 1], 4, 3, suppress)
                    .unwrap();
                let actual = super::super::read(result.0).await.unwrap();
                for (a, b) in actual.into_iter().zip(expected) {
                    assert!(a == b || (a - b).abs() < 1e-6);
                }
            }
        });
    }
}
