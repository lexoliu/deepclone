//! Shared-memory tiles for batched linear projections.
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
trait BatchedProjection: Backend {
    #[fusion(dtype = input, shape = Shape::from([input[0], input[1], weight[1]]))]
    fn batched_projection(input: FloatTensor<Self>, weight: FloatTensor<Self>)
    -> FloatTensor<Self>;
}

pub(super) fn project(input: BurnTensor<3>, weight: BurnTensor<2>) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::batched_projection(
        input.into_dispatch(),
        weight.into_dispatch(),
    ))
}

const TILE_MN: usize = 32;
const TILE_K: usize = 16;

impl BatchedProjection for CubeBackend {
    fn batched_projection(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let batch = input.shape()[0];
        let rows = input.shape()[1];
        let columns = weight.shape()[1];
        let output = Self::float_empty(
            Shape::from([batch, rows, columns]),
            &input.device,
            input.dtype.into(),
        );
        let tile_m = if rows <= TILE_K { TILE_K } else { TILE_MN };
        // A 16 × 16 workgroup computes a 32 × 32 register tile (or 16 × 32 for narrow
        // sequences <= 16 frames) with TILE_K = 16 within WebGPU's 16 KB guarantee.
        projection_kernel::launch(
            &input.client,
            CubeCount::Static(
                columns.div_ceil(TILE_MN) as u32,
                rows.div_ceil(tile_m) as u32,
                batch as u32,
            ),
            CubeDim::new_2d(TILE_K as u32, TILE_K as u32),
            input.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            tile_m,
            weight.meta.strides()[0] == 1,
            input.meta.strides()[1] < input.meta.strides()[2],
            [
                dtype_to_storage_type(input.dtype),
                dtype_to_storage_type(weight.dtype),
            ],
        );
        output
    }
}

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn projection_kernel<F: Float, W: Float>(
    input: &Tensor<F>,
    weight: &Tensor<W>,
    output: &mut Tensor<F>,
    #[comptime] tile_m: usize,
    #[comptime] transposed: bool,
    #[comptime] input_transposed: bool,
    #[define(F, W)] _dtypes: [ElemType; 2],
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let row_base = (CUBE_POS_Y as usize) * tile_m;
    let col_base = (CUBE_POS_X as usize) * TILE_MN;
    let batch = CUBE_POS_Z as usize;
    let k = input.shape(2);
    let rows = output.shape(1);
    let columns = output.shape(2);
    let in_b = batch * input.stride(0);
    let in_s1 = input.stride(1);
    let in_s2 = input.stride(2);
    let w_s0 = weight.stride(0);
    let w_s1 = weight.stride(1);

    let mut left = Shared::<[F]>::new_slice(tile_m * TILE_K);
    let mut right = Shared::<[F]>::new_slice(512usize);

    let acc00 = RuntimeCell::<f32>::new(0f32);
    let acc01 = RuntimeCell::<f32>::new(0f32);
    let acc10 = RuntimeCell::<f32>::new(0f32);
    let acc11 = RuntimeCell::<f32>::new(0f32);

    for block in 0..k.div_ceil(TILE_K) {
        let k_base = block * TILE_K;

        // Channel-first convolutions expose time as the contiguous input axis.
        // Read the contiguous dimension along tx and transpose in shared memory if needed.
        if input_transposed {
            let ik = k_base + ty;
            let r0 = row_base + tx;
            let r1 = r0 + TILE_K;
            let mut a0 = F::new(0_f32);
            let mut a1 = F::new(0_f32);
            if ik < k {
                if r0 < rows {
                    a0 = input[in_b + r0 * in_s1 + ik * in_s2];
                }
                if comptime!(tile_m > TILE_K) && r1 < rows {
                    a1 = input[in_b + r1 * in_s1 + ik * in_s2];
                }
            }
            left[tx * TILE_K + ty] = a0;
            if comptime!(tile_m > TILE_K) {
                left[(tx + TILE_K) * TILE_K + ty] = a1;
            }
        } else {
            let ik = k_base + tx;
            let r0 = row_base + ty;
            let r1 = r0 + TILE_K;
            let mut a0 = F::new(0_f32);
            let mut a1 = F::new(0_f32);
            if ik < k {
                if r0 < rows {
                    a0 = input[in_b + r0 * in_s1 + ik * in_s2];
                }
                if comptime!(tile_m > TILE_K) && r1 < rows {
                    a1 = input[in_b + r1 * in_s1 + ik * in_s2];
                }
            }
            left[ty * TILE_K + tx] = a0;
            if comptime!(tile_m > TILE_K) {
                left[(ty + TILE_K) * TILE_K + tx] = a1;
            }
        }

        // Read the contiguous dimension cooperatively for either weight layout.
        if transposed {
            let wk = k_base + tx;
            let c0 = col_base + ty;
            let c1 = c0 + TILE_K;
            let mut b0 = F::new(0_f32);
            let mut b1 = F::new(0_f32);
            if wk < k {
                if c0 < columns {
                    b0 = F::cast_from(weight[wk * w_s0 + c0 * w_s1]);
                }
                if c1 < columns {
                    b1 = F::cast_from(weight[wk * w_s0 + c1 * w_s1]);
                }
            }
            right[tx * TILE_MN + ty] = b0;
            right[tx * TILE_MN + ty + TILE_K] = b1;
        } else {
            let wk = k_base + ty;
            let c0 = col_base + tx;
            let c1 = c0 + TILE_K;
            let mut b0 = F::new(0_f32);
            let mut b1 = F::new(0_f32);
            if wk < k {
                if c0 < columns {
                    b0 = F::cast_from(weight[wk * w_s0 + c0 * w_s1]);
                }
                if c1 < columns {
                    b1 = F::cast_from(weight[wk * w_s0 + c1 * w_s1]);
                }
            }
            right[ty * TILE_MN + tx] = b0;
            right[ty * TILE_MN + tx + TILE_K] = b1;
        }

        sync_cube();
        #[unroll]
        for i in 0..TILE_K {
            let a0 = f32::cast_from(left[ty * TILE_K + i]);
            let b0 = f32::cast_from(right[i * TILE_MN + tx]);
            let b1 = f32::cast_from(right[i * TILE_MN + tx + TILE_K]);
            acc00.store(acc00.read() + a0 * b0);
            acc01.store(acc01.read() + a0 * b1);
            if comptime!(tile_m > TILE_K) {
                let a1 = f32::cast_from(left[(ty + TILE_K) * TILE_K + i]);
                acc10.store(acc10.read() + a1 * b0);
                acc11.store(acc11.read() + a1 * b1);
            }
        }
        sync_cube();
    }

    let r0 = row_base + ty;
    let r1 = r0 + TILE_K;
    let c0 = col_base + tx;
    let c1 = c0 + TILE_K;
    if r0 < rows {
        if c0 < columns {
            output[(batch * rows + r0) * columns + c0] = F::cast_from(acc00.consume());
        }
        if c1 < columns {
            output[(batch * rows + r0) * columns + c1] = F::cast_from(acc01.consume());
        }
    }
    if comptime!(tile_m > TILE_K) && r1 < rows {
        if c0 < columns {
            output[(batch * rows + r1) * columns + c0] = F::cast_from(acc10.consume());
        }
        if c1 < columns {
            output[(batch * rows + r1) * columns + c1] = F::cast_from(acc11.consume());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, TensorData};

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn batched_projection_matches_matrix_product() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F16, DType::F32] {
                for weight_dtype in [DType::F16, DType::F32] {
                    for (transposed, input_transposed) in
                        [(false, false), (false, true), (true, false), (true, true)]
                    {
                        for [batch, rows, k, columns] in [
                            [2, 19, 37, 23],
                            [1, 9, 1024, 256],
                            [1, 12, 512, 512],
                            [1, 12, 512, 1536],
                        ] {
                            let mut input = BurnTensor::<3>::from_data(
                                TensorData::new(
                                    (0..batch * rows * k)
                                        .map(|i| (i as f32 * 0.017).sin())
                                        .collect::<Vec<_>>(),
                                    if input_transposed {
                                        [batch, k, rows]
                                    } else {
                                        [batch, rows, k]
                                    },
                                ),
                                (&device, dtype),
                            );
                            if input_transposed {
                                input = input.swap_dims(1, 2);
                            }
                            let mut weight = BurnTensor::<2>::from_data(
                                TensorData::new(
                                    (0..k * columns)
                                        .map(|i| (i as f32 * 0.031).cos() / 100.)
                                        .collect::<Vec<_>>(),
                                    if transposed {
                                        [columns, k]
                                    } else {
                                        [k, columns]
                                    },
                                ),
                                (&device, weight_dtype),
                            );
                            if transposed {
                                weight = weight.transpose();
                            }
                            // FP32 reference accumulation retains the activation's weight rounding.
                            let reference = input
                                .clone()
                                .cast(DType::F32)
                                .matmul(weight.clone().cast(dtype).cast(DType::F32).unsqueeze())
                                .cast(dtype);
                            let (actual, expected) = futures_lite::future::zip(
                                super::super::read(project(input, weight)),
                                super::super::read(reference),
                            )
                            .await;
                            let (actual, expected) = (actual.unwrap(), expected.unwrap());
                            assert!(actual.iter().all(|v| v.is_finite()));
                            let maximum = actual
                                .iter()
                                .zip(&expected)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0f32, f32::max);
                            let tolerance = if dtype == DType::F16 { 0.001 } else { 1e-5 };
                            assert!(
                                maximum < tolerance,
                                "{dtype:?}/{weight_dtype:?}, transposed={transposed}: {maximum}"
                            );
                        }
                    }
                }
            }
        });
    }
}
