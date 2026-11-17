//! Matrix-vector projection without padding a single token into a GEMM tile.
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension, cubecl::dtype_to_storage_type,
        ops::FloatTensorOps, tensor::FloatTensor, wgpu::CubeBackend,
    },
    cubecl::{ir::ElemType, prelude::*},
    tensor::{Shape, Tensor as BurnTensor},
};
#[backend_extension(Cube, Fusion)]
trait VectorProjection: Backend {
    #[fusion(dtype = input, shape = Shape::from([1, 1, weight[1]]))]
    fn vector_projection(input: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self>;
}
pub(super) fn project(input: BurnTensor<3>, weight: BurnTensor<2>) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::vector_projection(
        input.into_dispatch(),
        weight.into_dispatch(),
    ))
}
impl VectorProjection for CubeBackend {
    fn vector_projection(input: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self> {
        let width = weight.shape()[1];
        let output = Self::float_empty(
            Shape::from([1, 1, width]),
            &input.device,
            input.dtype.into(),
        );
        let cooperative = weight.meta.strides()[0] == 1;
        let dim = CubeDim::new(
            &input.client,
            if cooperative { input.shape()[2] } else { width },
        );
        let lanes = dim.num_elems() as usize;
        let groups = if cooperative {
            width
        } else {
            width.div_ceil(lanes)
        };
        let weight_dtype = dtype_to_storage_type(weight.dtype);
        vector_projection_kernel::launch(
            &input.client,
            CubeCount::Static(groups as u32, 1, 1),
            dim,
            input.clone().into_tensor_arg(),
            weight.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            cooperative,
            lanes,
            [dtype_to_storage_type(input.dtype), weight_dtype],
        );
        output
    }
}
#[cube(launch)]
fn vector_projection_kernel<F: Float, W: Float>(
    input: &Tensor<F>,
    weight: &Tensor<W>,
    output: &mut Tensor<F>,
    #[comptime] cooperative: bool,
    #[comptime] lanes: usize,
    #[define(F, W)] _dtypes: [ElemType; 2],
) {
    let k = input.shape(2);
    // Preserve the former weight-to-activation cast at each load, including
    // FP32-to-FP16 rounding, without allocating a converted weight matrix.
    if cooperative {
        // A transposed row-major weight has contiguous reduction elements.
        // Cooperating lanes read adjacent K elements for one output column.
        let column = CUBE_POS;
        let sum = RuntimeCell::<f32>::new(0f32);
        let mut i = UNIT_POS as usize;
        while i < k {
            sum.store(
                sum.read()
                    + f32::cast_from(input[i * input.stride(2)])
                        * f32::cast_from(F::cast_from(
                            weight[i * weight.stride(0) + column * weight.stride(1)],
                        )),
            );
            i += CUBE_DIM as usize;
        }
        let mut partial = Shared::<[f32]>::new_slice(lanes);
        partial[UNIT_POS as usize] = sum.consume();
        sync_cube();
        #[unroll]
        for stage in 0..lanes.ilog2() as usize {
            let distance = comptime!(lanes >> (stage + 1));
            if (UNIT_POS as usize) < distance {
                partial[UNIT_POS as usize] += partial[UNIT_POS as usize + distance];
            }
            sync_cube();
        }
        if UNIT_POS == 0 {
            output[column] = F::cast_from(partial[0]);
        }
    } else {
        // Row-major projections have adjacent output columns. Neighboring lanes
        // read adjacent weights, reusing the input vector through the GPU cache.
        let column = ABSOLUTE_POS;
        if column >= output.shape(2) {
            terminate!();
        }
        let sum = RuntimeCell::<f32>::new(0f32);
        for i in 0..k {
            sum.store(
                sum.read()
                    + f32::cast_from(input[i * input.stride(2)])
                        * f32::cast_from(F::cast_from(
                            weight[i * weight.stride(0) + column * weight.stride(1)],
                        )),
            );
        }
        output[column] = F::cast_from(sum.consume());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, TensorData};
    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn gemv_contiguous_columns_match_reference() {
        check(false);
    }
    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn gemv_contiguous_reduction_matches_reference() {
        check(true);
    }
    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn mixed_storage_retains_weight_cast_rounding_and_reduction() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for input_dtype in [DType::F16, DType::F32] {
                for weight_dtype in [DType::F16, DType::F32] {
                    for transposed in [false, true] {
                        let (k, n) = (3072, 128);
                        let input = BurnTensor::<3>::from_data(
                            TensorData::new(
                                (0..k)
                                    .map(|i| (i as f32 * 0.017).sin() * 3.)
                                    .collect::<Vec<_>>(),
                                [1, 1, k],
                            ),
                            (&device, input_dtype),
                        );
                        let mut weight = BurnTensor::<2>::from_data(
                            TensorData::new(
                                (0..k * n)
                                    .map(|i| (i as f32 * 0.031).cos() / 100.)
                                    .collect::<Vec<_>>(),
                                if transposed { [n, k] } else { [k, n] },
                            ),
                            (&device, weight_dtype),
                        );
                        if transposed {
                            weight = weight.transpose();
                        }
                        let (actual, expected) = futures_lite::future::zip(
                            super::super::read(project(input.clone(), weight.clone())),
                            super::super::read(project(input, weight.cast(input_dtype))),
                        )
                        .await;
                        assert_eq!(
                            actual.unwrap(),
                            expected.unwrap(),
                            "{input_dtype:?}/{weight_dtype:?}, transposed={transposed}"
                        );
                    }
                }
            }
        });
    }

    fn check(transposed: bool) {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F32, DType::F16] {
                for (k, n) in [(37, 11), (1024, 384)] {
                    let input: Vec<f32> = (0..k).map(|i| (i % 17) as f32 / 32. - 0.25).collect();
                    let data: Vec<f32> = (0..k * n).map(|i| (i % 31) as f32 / 32. - 0.5).collect();
                    let mut weight = BurnTensor::<2>::from_data(
                        TensorData::new(data.clone(), if transposed { [n, k] } else { [k, n] }),
                        &device,
                    )
                    .cast(dtype);
                    if transposed {
                        weight = weight.transpose();
                    }
                    let input_tensor = BurnTensor::<3>::from_data(
                        TensorData::new(input.clone(), [1, 1, k]),
                        &device,
                    )
                    .cast(dtype);
                    let actual = super::super::read(project(input_tensor, weight))
                        .await
                        .unwrap();
                    for (column, actual) in actual.into_iter().enumerate() {
                        let expected: f64 = (0..k)
                            .map(|i| {
                                f64::from(input[i])
                                    * f64::from(
                                        data[if transposed {
                                            column * k + i
                                        } else {
                                            i * n + column
                                        }],
                                    )
                            })
                            .sum();
                        let tolerance = if dtype == DType::F16 { 0.005 } else { 1e-5 };
                        assert!(
                            (f64::from(actual) - expected).abs() < tolerance,
                            "{dtype:?}, {k}, {column}: {actual} vs {expected}"
                        );
                    }
                }
            }
        });
    }
}
