//! Matrix-vector projection without padding a single token into a GEMM tile.
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension, cubecl::dtype_to_storage_type,
        ops::FloatTensorOps, tensor::FloatTensor, wgpu::CubeBackend,
    },
    cubecl::{ir::ElemType, prelude::*, tensor_vector_size_parallel},
    tensor::{Shape, Tensor as BurnTensor},
};

const COOP_LANES_X: usize = 32;
const COOP_LANES_Y: usize = 2;
const COOP_ROWS_PER_LANE: usize = 4;
const COOP_OUTPUTS_PER_CUBE: usize = COOP_LANES_Y * COOP_ROWS_PER_LANE;

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
        let dtypes = [
            dtype_to_storage_type(input.dtype),
            dtype_to_storage_type(weight.dtype),
        ];
        if weight.meta.strides()[0] == 1 {
            // Cap at 4 (WebGPU vec4) so mixed-dtype and pre-cast weights choose
            // the same vector width and keep bit-identical reduction order.
            let in_vec = tensor_vector_size_parallel(
                input.client.io_optimized_vector_sizes(input.dtype.size()),
                input.meta.shape(),
                input.meta.strides(),
                2,
            );
            let w_vec = tensor_vector_size_parallel(
                weight.client.io_optimized_vector_sizes(weight.dtype.size()),
                weight.meta.shape(),
                weight.meta.strides(),
                0,
            );
            let vector_size = in_vec.min(w_vec).min(4);
            let groups = width.div_ceil(COOP_OUTPUTS_PER_CUBE).max(1);
            gemv_cooperative_kernel::launch(
                &input.client,
                CubeCount::Static(groups as u32, 1, 1),
                CubeDim::new_2d(COOP_LANES_X as u32, COOP_LANES_Y as u32),
                vector_size,
                input.clone().into_tensor_arg(),
                weight.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                dtypes,
            );
        } else {
            let dim = CubeDim::new(&input.client, width);
            let lanes = dim.num_elems() as usize;
            let groups = width.div_ceil(lanes).max(1);
            vector_projection_rowmajor_kernel::launch(
                &input.client,
                CubeCount::Static(groups as u32, 1, 1),
                dim,
                input.clone().into_tensor_arg(),
                weight.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                dtypes,
            );
        }
        output
    }
}

#[cube]
fn lane_sum<N: Size>(p: Vector<f32, N>) -> f32 {
    let mut d = 0f32;
    #[unroll]
    for lane in 0..p.vector_size() {
        d += p.extract(lane);
    }
    d
}

#[cube(launch)]
fn gemv_cooperative_kernel<F: Float, W: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    weight: &Tensor<Vector<W, N>>,
    output: &mut Tensor<F>,
    #[define(F, W)] _dtypes: [ElemType; 2],
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let col_base = (CUBE_POS_X as usize) * COOP_OUTPUTS_PER_CUBE + ty * COOP_ROWS_PER_LANE;
    let width = output.shape(2);
    let k = input.shape(2);
    let vec_size = input.vector_size();
    let k_vecs = k / vec_size;
    let w_stride = weight.stride(1) / vec_size;
    let in_stride = input.stride(2);

    let col0 = col_base;
    let col1 = col_base + 1usize;
    let col2 = col_base + 2usize;
    let col3 = col_base + 3usize;
    let has0 = col0 < width;
    let has1 = col1 < width;
    let has2 = col2 < width;
    let has3 = col3 < width;

    let s0 = RuntimeCell::<f32>::new(0f32);
    let s1 = RuntimeCell::<f32>::new(0f32);
    let s2 = RuntimeCell::<f32>::new(0f32);
    let s3 = RuntimeCell::<f32>::new(0f32);

    // Preserve the weight-to-activation cast at each load while reusing each
    // vectorized activation read across four adjacent output columns.
    let mut i = tx;
    while i < k_vecs {
        let x = if comptime!(vec_size > 1) {
            Vector::<f32, N>::cast_from(input[i])
        } else {
            Vector::<f32, N>::cast_from(input[i * in_stride])
        };
        if has0 {
            let w =
                Vector::<f32, N>::cast_from(Vector::<F, N>::cast_from(weight[col0 * w_stride + i]));
            s0.store(s0.read() + lane_sum(x * w));
        }
        if has1 {
            let w =
                Vector::<f32, N>::cast_from(Vector::<F, N>::cast_from(weight[col1 * w_stride + i]));
            s1.store(s1.read() + lane_sum(x * w));
        }
        if has2 {
            let w =
                Vector::<f32, N>::cast_from(Vector::<F, N>::cast_from(weight[col2 * w_stride + i]));
            s2.store(s2.read() + lane_sum(x * w));
        }
        if has3 {
            let w =
                Vector::<f32, N>::cast_from(Vector::<F, N>::cast_from(weight[col3 * w_stride + i]));
            s3.store(s3.read() + lane_sum(x * w));
        }
        i += COOP_LANES_X;
    }

    let mut partial = Shared::<[f32]>::new_slice(COOP_OUTPUTS_PER_CUBE * COOP_LANES_X);
    let row_s = ty * (COOP_ROWS_PER_LANE * COOP_LANES_X);
    partial[row_s + tx] = s0.consume();
    partial[row_s + COOP_LANES_X + tx] = s1.consume();
    partial[row_s + COOP_LANES_X * 2usize + tx] = s2.consume();
    partial[row_s + COOP_LANES_X * 3usize + tx] = s3.consume();
    sync_cube();

    #[unroll]
    for stage in 0..COOP_LANES_X.ilog2() as usize {
        let distance = comptime!(COOP_LANES_X >> (stage + 1));
        if tx < distance {
            partial[row_s + tx] += partial[row_s + tx + distance];
            partial[row_s + COOP_LANES_X + tx] += partial[row_s + COOP_LANES_X + tx + distance];
            partial[row_s + COOP_LANES_X * 2usize + tx] +=
                partial[row_s + COOP_LANES_X * 2usize + tx + distance];
            partial[row_s + COOP_LANES_X * 3usize + tx] +=
                partial[row_s + COOP_LANES_X * 3usize + tx + distance];
        }
        sync_cube();
    }

    if tx < COOP_ROWS_PER_LANE {
        let col = col_base + tx;
        if col < width {
            output[col] = F::cast_from(partial[row_s + tx * COOP_LANES_X]);
        }
    }
}

#[cube(launch)]
fn vector_projection_rowmajor_kernel<F: Float, W: Float>(
    input: &Tensor<F>,
    weight: &Tensor<W>,
    output: &mut Tensor<F>,
    #[define(F, W)] _dtypes: [ElemType; 2],
) {
    // Row-major projections have adjacent output columns. Neighboring lanes
    // read adjacent weights, reusing the input vector through the GPU cache.
    let column = ABSOLUTE_POS;
    if column >= output.shape(2) {
        terminate!();
    }
    let k = input.shape(2);
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
                for (k, n) in [(37, 11), (1024, 384), (3072, 1024)] {
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
