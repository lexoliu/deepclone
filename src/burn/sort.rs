//! Stable device sorting for the small vocabularies used during autoregression.
use burn::cubecl;
use burn::cubecl::prelude::*;

#[cfg(test)]
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        ops::{FloatTensorOps, IntTensorOps},
        tensor::{FloatTensor, IntTensor},
        wgpu::CubeBackend,
    },
    tensor::{DType, FloatDType, Int, IntDType, Tensor as BurnTensor},
};

#[cfg(test)]
#[backend_extension(Cube, Fusion)]
trait VocabularySort: Backend {
    #[fusion(meta = |input| (
        input.clone(),
        burn::backend::fusion::custom::TensorSpec::new(input.shape.clone(), DType::I32),
    ))]
    fn vocabulary_sort(input: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>);
}

#[cfg(test)]
pub(super) fn descending(input: BurnTensor<1>) -> (BurnTensor<1>, BurnTensor<1, Int>) {
    let (values, indices) = Dispatch::vocabulary_sort(input.into_dispatch());
    (
        BurnTensor::from_dispatch(values),
        BurnTensor::from_dispatch(indices),
    )
}

#[cfg(test)]
impl VocabularySort for CubeBackend {
    fn vocabulary_sort(input: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>) {
        let shape = input.shape();
        let len = shape.num_elements();
        let padded = len.next_power_of_two();
        let shared_slots =
            input.client.properties().hardware.max_shared_memory_size / size_of::<u32>();
        let cached_scores = if padded + len <= shared_slots { len } else { 0 };
        let values = Self::float_empty(shape.clone(), &input.device, FloatDType::F32);
        let indices = Self::int_empty(shape, &input.device, IntDType::I32);
        let dim = CubeDim::new(&input.client, (padded / 2).max(1));
        stable_sort::launch(
            &input.client,
            CubeCount::Static(1, 1, 1),
            dim,
            input.clone().into_tensor_arg(),
            values.clone().into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            padded,
            cached_scores,
        );
        (values, indices)
    }
}

#[cube]
pub(super) fn bitonic_sort_indices<F: Float>(
    input: &Tensor<F>,
    data: &mut Shared<[u32]>,
    #[comptime] padded: usize,
    #[comptime] cached_scores: usize,
) {
    let n = input.shape(0);
    let mut i = UNIT_POS as usize;
    while i < padded {
        data[i] = i as u32;
        i += CUBE_DIM as usize;
    }
    sync_cube();
    let mut size = 2usize;
    while size <= padded {
        let mut distance = size / 2;
        while distance > 0 {
            let mut pair = UNIT_POS as usize;
            while pair < padded / 2 {
                // Insert a zero at distance's bit to enumerate only the lower
                // side of each independent compare-exchange pair.
                let low = distance - 1;
                let i = ((pair & !low) << 1) | (pair & low);
                let other = i + distance;
                let a = data[i] as usize;
                let b = data[other] as usize;
                let before = if a < n {
                    if b < n {
                        let av = if cached_scores > 0 {
                            f32::reinterpret(data[padded + a])
                        } else {
                            f32::cast_from(input[a * input.stride(0)])
                        };
                        let bv = if cached_scores > 0 {
                            f32::reinterpret(data[padded + b])
                        } else {
                            f32::cast_from(input[b * input.stride(0)])
                        };
                        av > bv || (av == bv && a < b)
                    } else {
                        true
                    }
                } else {
                    b >= n && a < b
                };
                let descending = (i & size) == 0;
                if descending != before {
                    data[i] = b as u32;
                    data[other] = a as u32;
                }
                pair += CUBE_DIM as usize;
            }
            sync_cube();
            distance /= 2;
        }
        size *= 2;
    }
}

#[cfg(test)]
#[cube(launch)]
fn stable_sort(
    input: &Tensor<f32>,
    values: &mut Tensor<f32>,
    indices: &mut Tensor<i32>,
    #[comptime] padded: usize,
    #[comptime] cached_scores: usize,
) {
    // The first padded slots hold indices. If space permits, the remaining
    // slots cache the original scores without duplicating the padded values.
    let mut data = Shared::<[u32]>::new_slice(padded + cached_scores);
    let n = input.shape(0);
    if cached_scores > 0 {
        let mut i = UNIT_POS as usize;
        while i < n {
            data[padded + i] = u32::reinterpret(input[i * input.stride(0)]);
            i += CUBE_DIM as usize;
        }
    }
    bitonic_sort_indices(input, &mut data, padded, cached_scores);
    let mut i = UNIT_POS as usize;
    while i < n {
        let original = data[i] as usize;
        values[i] = if cached_scores > 0 {
            f32::reinterpret(data[padded + original])
        } else {
            input[original * input.stride(0)]
        };
        indices[i] = original as i32;
        i += CUBE_DIM as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};
    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn device_sort_preserves_full_vocabulary_order_and_ties() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            // Includes the full predictor and talker vocabularies, non-power-of-two
            // padding, and repeated logits whose index order must remain stable.
            for n in [17, 2048, 3072] {
                let input: Vec<f32> = (0..n).map(|i| ((i * 31) % 137) as f32 - 68.).collect();
                let mut expected: Vec<_> = (0..n).collect();
                expected.sort_by(|&a, &b| input[b].total_cmp(&input[a]).then(a.cmp(&b)));
                let (values, indices) = descending(BurnTensor::from_data(
                    TensorData::new(input.clone(), [n]),
                    &device,
                ));
                let (values, indices) = futures_lite::future::zip(
                    super::super::read(values),
                    indices.into_data_async(),
                )
                .await;
                let actual = indices.unwrap().try_into_vec::<i32>().unwrap();
                assert_eq!(
                    actual,
                    expected.iter().map(|&i| i as i32).collect::<Vec<_>>()
                );
                assert_eq!(
                    values.unwrap(),
                    expected.into_iter().map(|i| input[i]).collect::<Vec<_>>()
                );
            }
        });
    }
}
