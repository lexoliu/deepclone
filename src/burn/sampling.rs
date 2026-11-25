//! Device-resident token selection with caller-owned randomness.
use crate::sampling::{Error, HostSampler, Logits, Multinomial, RepetitionPenalty};
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        cubecl::dtype_to_storage_type,
        ops::{FloatTensorOps, IntTensorOps},
        tensor::{FloatTensor, IntTensor},
        wgpu::CubeBackend,
    },
    cubecl::{ir::ElemType, prelude::*},
    tensor::{DType, FloatDType, Int, IntDType, Shape, Tensor as BurnTensor, TensorData},
};
use rand::RngExt;
use std::num::NonZeroUsize;

const CONSTRAIN_LANES: usize = 256;
const GREEDY_LANES: usize = 256;
const MAX_SAMPLE_LANES: usize = 256;

#[backend_extension(Cube, Fusion)]
trait DeviceSampling: Backend {
    #[fusion(meta = |logits, _factors| burn::backend::fusion::custom::TensorSpec::new(
        logits.shape.clone(),
        DType::F32,
    ))]
    fn constrain_logits(logits: FloatTensor<Self>, factors: FloatTensor<Self>)
    -> FloatTensor<Self>;

    #[fusion(meta = |_logits| burn::backend::fusion::custom::TensorSpec::new(
        Shape::from([1]),
        DType::I32,
    ))]
    fn sample_greedy(logits: FloatTensor<Self>) -> IntTensor<Self>;

    #[fusion(meta = |_logits, _temperature, _top_k, _top_p, _draw| burn::backend::fusion::custom::TensorSpec::new(
        Shape::from([1]),
        DType::I32,
    ))]
    fn sample_multinomial(
        logits: FloatTensor<Self>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        draw: f32,
    ) -> IntTensor<Self>;
}

impl DeviceSampling for CubeBackend {
    fn constrain_logits(
        logits: FloatTensor<Self>,
        factors: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let shape = logits.shape();
        let n = shape.num_elements();
        let output = Self::float_empty(shape, &logits.device, FloatDType::F32);
        constrain_logits_kernel::launch(
            &logits.client,
            CubeCount::Static(n.div_ceil(CONSTRAIN_LANES).max(1) as u32, 1, 1),
            CubeDim::new_1d(CONSTRAIN_LANES as u32),
            logits.clone().into_tensor_arg(),
            factors.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dtype_to_storage_type(logits.dtype),
        );
        output
    }

    fn sample_greedy(logits: FloatTensor<Self>) -> IntTensor<Self> {
        let output = Self::int_empty(Shape::from([1]), &logits.device, IntDType::I32);
        fused_greedy_kernel::launch(
            &logits.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(GREEDY_LANES as u32),
            logits.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            GREEDY_LANES,
            dtype_to_storage_type(logits.dtype),
        );
        output
    }

    fn sample_multinomial(
        logits: FloatTensor<Self>,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        draw: f32,
    ) -> IntTensor<Self> {
        let n = logits.shape().num_elements();
        let padded = n.next_power_of_two();
        let shared_slots =
            logits.client.properties().hardware.max_shared_memory_size / size_of::<u32>();
        let cached_scores = if padded + n <= shared_slots { n } else { 0 };
        let lanes = (padded / 2).clamp(1, MAX_SAMPLE_LANES).next_power_of_two();
        let output = Self::int_empty(Shape::from([1]), &logits.device, IntDType::I32);
        fused_multinomial_kernel::launch(
            &logits.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(lanes as u32),
            logits.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            temperature,
            top_k,
            top_p,
            draw,
            padded,
            cached_scores,
            lanes,
            dtype_to_storage_type(logits.dtype),
        );
        output
    }
}

#[cube(launch)]
fn constrain_logits_kernel<F: Float>(
    logits: &Tensor<F>,
    factors: &Tensor<f32>,
    output: &mut Tensor<f32>,
    #[define(F)] _dtype: ElemType,
) {
    let idx = ABSOLUTE_POS;
    if idx >= output.shape(0) {
        terminate!();
    }
    let f = factors[idx * factors.stride(0)];
    if f == 0f32 {
        // IEEE-754 negative infinity bit pattern avoids WGSL infinity literals.
        output[idx] = f32::reinterpret(0xff80_0000u32);
    } else {
        let v = f32::cast_from(logits[idx * logits.stride(0)]);
        output[idx] = if v < 0f32 { v * f } else { v / f };
    }
}

#[cube(launch)]
fn fused_greedy_kernel<F: Float>(
    logits: &Tensor<F>,
    output: &mut Tensor<i32>,
    #[comptime] lanes: usize,
    #[define(F)] _dtype: ElemType,
) {
    let n = logits.shape(0);
    let stride = logits.stride(0);
    let tid = UNIT_POS as usize;

    let mut val_s = Shared::<[f32]>::new_slice(lanes);
    let mut idx_s = Shared::<[u32]>::new_slice(lanes);
    let mut flag_s = Shared::<[u32]>::new_slice(lanes);

    let mut best_val = 0f32;
    let mut best_idx = 0xffff_ffffu32;
    let mut local_flags = 0u32;

    let mut i = tid;
    while i < n {
        let v = f32::cast_from(logits[i * stride]);
        let bits = u32::reinterpret(v);
        let abs_bits = bits & 0x7fff_ffffu32;
        if abs_bits > 0x7f80_0000u32 || bits == 0x7f80_0000u32 {
            local_flags |= 2u32;
        } else if (bits & 0x7f80_0000u32) != 0x7f80_0000u32 {
            local_flags |= 1u32;
            if best_idx == 0xffff_ffffu32 || v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        i += lanes;
    }

    val_s[tid] = best_val;
    idx_s[tid] = best_idx;
    flag_s[tid] = local_flags;
    sync_cube();

    #[unroll]
    for stage in 0..lanes.ilog2() as usize {
        let distance = comptime!(lanes >> (stage + 1));
        if tid < distance {
            flag_s[tid] |= flag_s[tid + distance];
            let other_idx = idx_s[tid + distance];
            if other_idx != 0xffff_ffffu32 {
                let cur_idx = idx_s[tid];
                let other_val = val_s[tid + distance];
                let cur_val = val_s[tid];
                if cur_idx == 0xffff_ffffu32
                    || other_val > cur_val
                    || (other_val == cur_val && other_idx < cur_idx)
                {
                    val_s[tid] = other_val;
                    idx_s[tid] = other_idx;
                }
            }
        }
        sync_cube();
    }

    if tid == 0 {
        let flags = flag_s[0];
        if (flags & 2u32) != 0u32 {
            output[0] = -2i32;
        } else if (flags & 1u32) == 0u32 {
            output[0] = -1i32;
        } else {
            output[0] = idx_s[0] as i32;
        }
    }
}

#[cube(launch)]
fn fused_multinomial_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<i32>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    draw: f32,
    #[comptime] padded: usize,
    #[comptime] cached_scores: usize,
    #[comptime] lanes: usize,
    #[define(F)] _dtype: ElemType,
) {
    let mut data = Shared::<[u32]>::new_slice(padded + cached_scores);
    let n = input.shape(0);
    let stride = input.stride(0);
    let tid = UNIT_POS as usize;

    let mut local_flags = 0u32;
    let mut i = tid;
    while i < n {
        let v = f32::cast_from(input[i * stride]);
        let bits = u32::reinterpret(v);
        let abs_bits = bits & 0x7fff_ffffu32;
        if abs_bits > 0x7f80_0000u32 || bits == 0x7f80_0000u32 {
            local_flags |= 2u32;
        } else if (bits & 0x7f80_0000u32) != 0x7f80_0000u32 {
            local_flags |= 1u32;
        }
        if cached_scores > 0 {
            data[padded + i] = bits;
        }
        i += lanes;
    }

    data[tid] = local_flags;
    sync_cube();
    #[unroll]
    for stage in 0..lanes.ilog2() as usize {
        let distance = comptime!(lanes >> (stage + 1));
        if tid < distance {
            data[tid] |= data[tid + distance];
        }
        sync_cube();
    }
    let flags = data[0];
    let valid = (flags & 2u32) == 0u32 && (flags & 1u32) != 0u32;
    sync_cube();

    // Defer invalid-logit exits until after the sort and shared barriers so every
    // workgroupBarrier remains in WGSL uniform control flow under browser Dawn/Tint.
    crate::burn::sort::bitonic_sort_indices(input, &mut data, padded, cached_scores);

    let best_orig = data[0] as usize;
    let max_val = if cached_scores > 0 {
        f32::reinterpret(data[padded + best_orig])
    } else {
        f32::cast_from(input[best_orig * stride])
    };
    let has_top_k = top_k > 0usize && top_k < n;
    let cutoff_orig = if has_top_k {
        data[top_k - 1usize] as usize
    } else {
        best_orig
    };
    let cutoff = if cached_scores > 0 {
        f32::reinterpret(data[padded + cutoff_orig])
    } else {
        f32::cast_from(input[cutoff_orig * stride])
    };
    sync_cube();

    if cached_scores > 0 {
        if valid {
            let mut rank = tid;
            while rank < n {
                let orig = data[rank] as usize;
                let s = f32::reinterpret(data[padded + orig]);
                let s_bits = u32::reinterpret(s);
                let w = if s_bits == 0xff80_0000u32 || (has_top_k && s < cutoff) {
                    0f32
                } else {
                    ((s - max_val) / temperature).exp()
                };
                data[padded + orig] = u32::reinterpret(w);
                rank += lanes;
            }
        }
        sync_cube();
    }

    if tid == 0 {
        if (flags & 2u32) != 0u32 {
            output[0] = -2i32;
        } else if (flags & 1u32) == 0u32 {
            output[0] = -1i32;
        } else {
            let mut kept_k = 0usize;
            let mut total = 0f32;
            let mut rank = 0usize;
            while rank < n {
                let orig = data[rank] as usize;
                let w = if cached_scores > 0 {
                    f32::reinterpret(data[padded + orig])
                } else {
                    let s = f32::cast_from(input[orig * stride]);
                    let s_bits = u32::reinterpret(s);
                    if s_bits == 0xff80_0000u32 || (has_top_k && s < cutoff) {
                        0f32
                    } else {
                        ((s - max_val) / temperature).exp()
                    }
                };
                if w == 0f32 {
                    break;
                }
                total += w;
                kept_k = rank + 1usize;
                rank += 1usize;
            }

            let mut kept_p = kept_k;
            let mut retained = total;
            if top_p < 1f32 {
                let threshold = total * top_p;
                let mut prefix = 0f32;
                let mut count = 0usize;
                let mut rank = 0usize;
                while rank < kept_k {
                    let orig = data[rank] as usize;
                    let w = if cached_scores > 0 {
                        f32::reinterpret(data[padded + orig])
                    } else {
                        let s = f32::cast_from(input[orig * stride]);
                        ((s - max_val) / temperature).exp()
                    };
                    prefix += w;
                    count = rank + 1usize;
                    if prefix >= threshold {
                        break;
                    }
                    rank += 1usize;
                }
                kept_p = count;
                retained = prefix;
            }

            let target = retained * draw;
            let mut cumulative = 0f32;
            let mut chosen = data[kept_p - 1usize] as i32;
            let mut rank = 0usize;
            while rank < kept_p {
                let orig = data[rank] as usize;
                let w = if cached_scores > 0 {
                    f32::reinterpret(data[padded + orig])
                } else {
                    let s = f32::cast_from(input[orig * stride]);
                    ((s - max_val) / temperature).exp()
                };
                cumulative += w;
                if cumulative > target {
                    chosen = orig as i32;
                    break;
                }
                rank += 1usize;
            }
            output[0] = chosen;
        }
    }
}

pub struct DeviceLogits(pub BurnTensor<1>);

/// Negative indices retain sampling failures until the next mandatory token
/// readback. No waveform is emitted from an invalid frame.
#[derive(Clone)]
pub struct DeviceToken(BurnTensor<1, Int>);
impl DeviceToken {
    pub(crate) fn raw(&self) -> BurnTensor<1, Int> {
        self.0.clone()
    }
}
fn execution(error: impl std::fmt::Display) -> Error {
    Error::Device(error.to_string())
}
pub(crate) async fn read_tokens(tokens: &[DeviceToken]) -> Result<Vec<u32>, Error> {
    let values = BurnTensor::cat(tokens.iter().map(|token| token.0.clone()).collect(), 0)
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
        let factor = BurnTensor::<1>::from_data(TensorData::new(factors, [n]), &self.0.device());
        Ok(Self(BurnTensor::from_dispatch(Dispatch::constrain_logits(
            self.0.into_dispatch(),
            factor.into_dispatch(),
        ))))
    }
    fn greedy(self) -> Result<DeviceToken, Error> {
        if self.0.dims()[0] == 0 {
            return Err(Error::NoCandidate);
        }
        Ok(DeviceToken(BurnTensor::from_dispatch(
            Dispatch::sample_greedy(self.0.into_dispatch()),
        )))
    }
    fn multinomial<R: rand::Rng>(self, sampler: &mut Multinomial<R>) -> Result<DeviceToken, Error> {
        let n = self.0.dims()[0];
        if n == 0 {
            return Err(Error::NoCandidate);
        }
        let top_k = sampler
            .parameters
            .top_k
            .filter(|k| k.get() < n)
            .map_or(0, NonZeroUsize::get);
        // Rounding an f64 draw to f32 must not include the upper endpoint.
        let draw = (sampler.random.random::<f64>() as f32).min(f32::from_bits(1f32.to_bits() - 1));
        Ok(DeviceToken(BurnTensor::from_dispatch(
            Dispatch::sample_multinomial(
                self.0.into_dispatch(),
                sampler.parameters.temperature,
                top_k,
                sampler.parameters.top_p,
                draw,
            ),
        )))
    }
    async fn sample_host<S: HostSampler>(self, sampler: &mut S) -> Result<DeviceToken, Error> {
        let device = self.0.device();
        let mut values = super::read(self.0).await.map_err(execution)?;
        let token = sampler.sample(&mut values)?;
        if token as usize >= values.len() {
            return Err(Error::Vocabulary);
        }
        let index = i32::try_from(token).map_err(|_| Error::Vocabulary)?;
        Ok(DeviceToken(BurnTensor::from_data(
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

    fn scores(device: &Device, values: &[f32], dtype: DType) -> DeviceLogits {
        DeviceLogits(BurnTensor::from_data(
            TensorData::new(values.to_vec(), [values.len()]),
            (device, dtype),
        ))
    }
    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn device_selection_matches_host_sampling_and_rejects_invalid_scores() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F32, DType::F16] {
                for parameters in [
                    Parameters::new(0.9, NonZeroUsize::new(3), 1.).unwrap(),
                    Parameters::new(0.7, None, 0.8).unwrap(),
                    Parameters::new(1., NonZeroUsize::new(1), 1.).unwrap(),
                ] {
                    for values in [
                        vec![0.125, 2.25, -1.75, 0.875, f32::NEG_INFINITY],
                        vec![2., 2., 0., -1.],
                        vec![1.5],
                    ] {
                        let mut host = Multinomial::new(StdRng::seed_from_u64(42), parameters);
                        let mut gpu = Multinomial::new(StdRng::seed_from_u64(42), parameters);
                        let mut expected = Vec::new();
                        let mut selected = Vec::new();
                        for _ in 0..64 {
                            expected.push(host.sample(values.clone()).await.unwrap());
                            selected
                                .push(gpu.sample(scores(&device, &values, dtype)).await.unwrap());
                        }
                        assert_eq!(read_tokens(&selected).await.unwrap(), expected);
                    }
                }
                let token = Greedy
                    .sample(scores(&device, &[f32::NEG_INFINITY, 2., 2.], dtype))
                    .await
                    .unwrap();
                assert_eq!(read_tokens(&[token]).await.unwrap(), [1]);
                let parameters = Parameters::new(0.9, NonZeroUsize::new(1), 1.).unwrap();
                let mut sampler = Multinomial::new(StdRng::seed_from_u64(42), parameters);
                for values in [vec![f32::NAN, 1.], vec![f32::INFINITY, 1.]] {
                    let token = scores(&device, &values, dtype).greedy().unwrap();
                    assert!(matches!(
                        read_tokens(&[token]).await,
                        Err(Error::InvalidLogits)
                    ));
                    let token = sampler
                        .sample(scores(&device, &values, dtype))
                        .await
                        .unwrap();
                    assert!(matches!(
                        read_tokens(&[token]).await,
                        Err(Error::InvalidLogits)
                    ));
                }
                let token = scores(&device, &[f32::NEG_INFINITY; 2], dtype)
                    .greedy()
                    .unwrap();
                assert!(matches!(
                    read_tokens(&[token]).await,
                    Err(Error::NoCandidate)
                ));
                let token = sampler
                    .sample(scores(&device, &[f32::NEG_INFINITY; 2], dtype))
                    .await
                    .unwrap();
                assert!(matches!(
                    read_tokens(&[token]).await,
                    Err(Error::NoCandidate)
                ));
            }
            // Verify full Qwen3 talker (3072) and predictor (2048) vocabularies.
            let params = Parameters::new(0.9, NonZeroUsize::new(50), 0.95).unwrap();
            for n in [2048usize, 3072usize] {
                let values: Vec<f32> = (0..n)
                    .map(|i| (((i * 37 + 13) % 97) as f32 - 48.0) * 0.125)
                    .collect();
                let mut host = Multinomial::new(StdRng::seed_from_u64(7), params);
                let mut gpu = Multinomial::new(StdRng::seed_from_u64(7), params);
                let mut expected = Vec::new();
                let mut selected = Vec::new();
                for _ in 0..32 {
                    expected.push(host.sample(values.clone()).await.unwrap());
                    selected.push(
                        gpu.sample(scores(&device, &values, DType::F16))
                            .await
                            .unwrap(),
                    );
                }
                assert_eq!(read_tokens(&selected).await.unwrap(), expected);
            }
        });
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn device_repetition_and_reserved_tokens_match_host_constraints() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            let values = vec![-2., 3., 2., 1., -1., 4.];
            let penalty = RepetitionPenalty::new(1.05).unwrap();
            for dtype in [DType::F32, DType::F16] {
                for suppress in [false, true] {
                    let expected = values
                        .clone()
                        .constrain(penalty, &[0, 0, 1], 4, 3, suppress)
                        .unwrap();
                    let result = scores(&device, &values, dtype)
                        .constrain(penalty, &[0, 0, 1], 4, 3, suppress)
                        .unwrap();
                    let actual = super::super::read(result.0).await.unwrap();
                    for (a, b) in actual.into_iter().zip(expected) {
                        assert!(a == b || (a - b).abs() < 1e-6);
                    }
                }
            }
        });
    }
}
