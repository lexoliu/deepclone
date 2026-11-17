//! Final sampling selection and logit validation in one GPU dispatch.
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        ops::IntTensorOps,
        tensor::{FloatTensor, IntTensor},
        wgpu::CubeBackend,
    },
    cubecl::prelude::*,
    tensor::{DType, Int, IntDType, Shape, Tensor as BurnTensor},
};

#[backend_extension(Cube, Fusion)]
trait SampleSelection: Backend {
    #[fusion(meta = |_logits| burn::backend::fusion::custom::TensorSpec::new(Shape::from([1]), DType::I32))]
    fn select_sample(
        _logits: FloatTensor<Self>,
        cumulative: FloatTensor<Self>,
        indices: IntTensor<Self>,
        draw: f32,
    ) -> IntTensor<Self>;
}

pub(super) fn select(
    logits: BurnTensor<1>,
    cumulative: BurnTensor<1>,
    indices: BurnTensor<1, Int>,
    draw: f32,
) -> BurnTensor<1, Int> {
    BurnTensor::from_dispatch(Dispatch::select_sample(
        logits.into_dispatch(),
        cumulative.into_dispatch(),
        indices.into_dispatch(),
        draw,
    ))
}

impl SampleSelection for CubeBackend {
    fn select_sample(
        logits: FloatTensor<Self>,
        cumulative: FloatTensor<Self>,
        indices: IntTensor<Self>,
        draw: f32,
    ) -> IntTensor<Self> {
        let output = Self::int_empty(Shape::from([1]), &logits.device, IntDType::I32);
        let dim = CubeDim::new(&logits.client, logits.shape()[0]);
        select_kernel::launch(
            &logits.client,
            CubeCount::Static(1, 1, 1),
            dim,
            logits.clone().into_tensor_arg(),
            cumulative.into_tensor_arg(),
            indices.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            ScalarArg::new(draw),
            dim.num_elems() as usize,
        );
        output
    }
}

#[cube(launch)]
fn select_kernel(
    logits: &Tensor<f32>,
    cumulative: &Tensor<f32>,
    indices: &Tensor<i32>,
    output: &mut Tensor<i32>,
    draw: f32,
    #[comptime] lanes: usize,
) {
    let n = logits.shape(0);
    let threshold = cumulative[(n - 1) * cumulative.stride(0)] * draw;
    let mut rank = n;
    // Bit 0 records a finite candidate; bit 1 records an invalid logit.
    let mut status = 0u32;
    let mut i = UNIT_POS as usize;
    while i < n {
        let value = logits[i * logits.stride(0)];
        if value.is_nan() || value == f32::INFINITY {
            status |= 2;
        }
        if value > f32::NEG_INFINITY && value < f32::INFINITY {
            status |= 1;
        }
        if cumulative[i * cumulative.stride(0)] > threshold {
            rank = usize::min(rank, i);
        }
        i += CUBE_DIM as usize;
    }
    let mut ranks = Shared::<[usize]>::new_slice(lanes);
    let mut statuses = Shared::<[u32]>::new_slice(lanes);
    let lane = UNIT_POS as usize;
    ranks[lane] = rank;
    statuses[lane] = status;
    sync_cube();
    #[unroll]
    for stage in 0..lanes.next_power_of_two().ilog2() as usize {
        let active = comptime!(lanes.div_ceil(1 << stage));
        let distance = comptime!(active.div_ceil(2));
        if lane < distance && lane + distance < active {
            ranks[lane] = usize::min(ranks[lane], ranks[lane + distance]);
            statuses[lane] |= statuses[lane + distance];
        }
        sync_cube();
    }
    if UNIT_POS == 0 {
        if (statuses[0] & 2) != 0 {
            output[0] = -2;
        } else if (statuses[0] & 1) == 0 {
            output[0] = -1;
        } else {
            // The former boolean argmax selected rank zero if no entry was true.
            let selected = if ranks[0] == n { 0 } else { ranks[0] };
            output[0] = indices[selected * indices.stride(0)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn fused_selection_matches_tensor_selection_at_probability_boundaries() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            for n in [1, 17, 2048, 3072] {
                // Plateaus model candidates removed by top-k/top-p; repeated
                // logits exercise stable index order independently of rank.
                let values: Vec<_> = (0..n).map(|i| (i % 19) as f32 - 9.).collect();
                let logits = BurnTensor::<1>::from_data(TensorData::new(values, [n]), &device);
                let cumulative = BurnTensor::<1>::from_data(
                    TensorData::new((0..n).map(|i| (i / 3) as f32).collect::<Vec<_>>(), [n]),
                    &device,
                );
                let indices = BurnTensor::<1, Int>::from_data(
                    TensorData::new((0..n).rev().map(|i| i as i32).collect::<Vec<_>>(), [n]),
                    &device,
                );
                for draw in [
                    0.,
                    f32::EPSILON,
                    0.25,
                    0.5,
                    0.75,
                    f32::from_bits(1f32.to_bits() - 1),
                ] {
                    let total = cumulative.clone().narrow(0, n - 1, 1);
                    let rank = cumulative.clone().greater(total * draw).int().argmax(0);
                    expected.push(indices.clone().gather(0, rank).reshape([1]));
                    actual.push(select(
                        logits.clone(),
                        cumulative.clone(),
                        indices.clone(),
                        draw,
                    ));
                }
            }
            let (actual, expected) = futures_lite::future::zip(
                BurnTensor::cat(actual, 0).into_data_async(),
                BurnTensor::cat(expected, 0).into_data_async(),
            )
            .await;
            assert_eq!(
                actual.unwrap().try_into_vec::<i32>().unwrap(),
                expected.unwrap().try_into_vec::<i32>().unwrap()
            );
        });
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn fused_selection_retains_invalid_logit_errors_across_all_lanes() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            let n = 3072;
            let cumulative =
                BurnTensor::<1>::from_data(TensorData::new(vec![1f32; n], [n]), &device);
            let indices = BurnTensor::<1, Int>::zeros([n], &device);
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            for (value, status) in [
                (f32::NEG_INFINITY, -1),
                (f32::NAN, -2),
                (f32::INFINITY, -2),
                (0., 0),
            ] {
                for position in [0, 17, n - 1] {
                    let mut values = vec![f32::NEG_INFINITY; n];
                    values[position] = value;
                    let logits = BurnTensor::from_data(TensorData::new(values, [n]), &device);
                    actual.push(select(logits, cumulative.clone(), indices.clone(), 0.5));
                    expected.push(status);
                }
            }
            assert_eq!(
                BurnTensor::cat(actual, 0)
                    .into_data_async()
                    .await
                    .unwrap()
                    .try_into_vec::<i32>()
                    .unwrap(),
                expected
            );
        });
    }
}
