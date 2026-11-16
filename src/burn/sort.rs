//! Stable device sorting for the small vocabularies used during autoregression.
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        ops::{FloatTensorOps, IntTensorOps},
        tensor::{FloatTensor, IntTensor},
        wgpu::CubeBackend,
    },
    cubecl::prelude::*,
    tensor::{DType, FloatDType, Int, IntDType, Tensor as BurnTensor},
};

#[backend_extension(Cube, Fusion)]
trait VocabularySort: Backend {
    #[fusion(meta = |input| (
        input.clone(),
        burn::backend::fusion::custom::TensorSpec::new(input.shape.clone(), DType::I32),
    ))]
    fn vocabulary_sort(input: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>);
}

pub(super) fn descending(input: BurnTensor<1>) -> (BurnTensor<1>, BurnTensor<1, Int>) {
    let (values, indices) = Dispatch::vocabulary_sort(input.into_dispatch());
    (
        BurnTensor::from_dispatch(values),
        BurnTensor::from_dispatch(indices),
    )
}
impl VocabularySort for CubeBackend {
    fn vocabulary_sort(input: FloatTensor<Self>) -> (FloatTensor<Self>, IntTensor<Self>) {
        let shape = input.shape();
        let padded = shape.num_elements().next_power_of_two();
        let values = Self::float_empty(shape.clone(), &input.device, FloatDType::F32);
        let indices = Self::int_empty(shape, &input.device, IntDType::I32);
        let dim = CubeDim::new(&input.client, padded);
        stable_sort::launch(
            &input.client,
            CubeCount::Static(1, 1, 1),
            dim,
            input.clone().into_tensor_arg(),
            values.clone().into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            padded,
        );
        (values, indices)
    }
}

#[cube(launch)]
fn stable_sort(
    input: &Tensor<f32>,
    values: &mut Tensor<f32>,
    indices: &mut Tensor<i32>,
    #[comptime] padded: usize,
) {
    // Sorting indices needs half the shared storage of sorting (value,index) pairs.
    let mut order = Shared::<[u32]>::new_slice(padded);
    let n = input.shape(0);
    let mut i = UNIT_POS as usize;
    while i < padded {
        order[i] = i as u32;
        i += CUBE_DIM as usize;
    }
    sync_cube();
    let mut size = 2usize;
    while size <= padded {
        let mut distance = size / 2;
        while distance > 0 {
            let mut i = UNIT_POS as usize;
            while i < padded {
                let other = i ^ distance;
                if other > i {
                    let a = order[i];
                    let b = order[other];
                    let before = if (a as usize) < n {
                        if (b as usize) < n {
                            let av = input[a as usize * input.stride(0)];
                            let bv = input[b as usize * input.stride(0)];
                            av > bv || (av == bv && a < b)
                        } else {
                            true
                        }
                    } else {
                        if (b as usize) < n { false } else { a < b }
                    };
                    let descending = (i & size) == 0;
                    if descending != before {
                        order[i] = b;
                        order[other] = a;
                    }
                }
                i += CUBE_DIM as usize;
            }
            sync_cube();
            distance /= 2;
        }
        size *= 2;
    }
    let mut i = UNIT_POS as usize;
    while i < n {
        let original = order[i] as usize;
        values[i] = input[original * input.stride(0)];
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
