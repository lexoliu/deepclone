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
        // A 16 × 16 tile uses WebGPU's guaranteed 256 invocation workgroup.
        let tile = 16;
        projection_kernel::launch(
            &input.client,
            CubeCount::Static(
                columns.div_ceil(tile) as u32,
                rows.div_ceil(tile) as u32,
                batch as u32,
            ),
            CubeDim::new_2d(tile as u32, tile as u32),
            input.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            weight.meta.strides()[0] == 1,
            input.meta.strides()[1] < input.meta.strides()[2],
            tile,
            [
                dtype_to_storage_type(input.dtype),
                dtype_to_storage_type(weight.dtype),
            ],
        );
        output
    }
}

#[cube(launch)]
fn projection_kernel<F: Float, W: Float>(
    input: &Tensor<F>,
    weight: &Tensor<W>,
    output: &mut Tensor<F>,
    #[comptime] transposed: bool,
    #[comptime] input_transposed: bool,
    #[comptime] tile: usize,
    #[define(F, W)] _dtypes: [ElemType; 2],
) {
    let local_column = UNIT_POS_X as usize;
    let local_row = UNIT_POS_Y as usize;
    let row = (CUBE_POS_Y as usize) * tile + local_row;
    let column = (CUBE_POS_X as usize) * tile + local_column;
    let batch = CUBE_POS_Z as usize;
    let k = input.shape(2);
    let rows = output.shape(1);
    let columns = output.shape(2);
    let mut left = Shared::<[F]>::new_slice(tile * tile);
    let mut right = Shared::<[F]>::new_slice(tile * tile);
    let sum = RuntimeCell::<f32>::new(0f32);
    for block in 0..k.div_ceil(tile) {
        // Channel-first convolutions expose time as the contiguous input axis.
        // Read that axis cooperatively, transposing only the shared-memory tile.
        let input_row = (CUBE_POS_Y as usize) * tile
            + if input_transposed {
                local_column
            } else {
                local_row
            };
        let input_k = block * tile
            + if input_transposed {
                local_row
            } else {
                local_column
            };
        let mut a = F::new(0_f32);
        if input_row < input.shape(1) && input_k < k {
            a = input
                [batch * input.stride(0) + input_row * input.stride(1) + input_k * input.stride(2)];
        }
        if input_transposed {
            left[local_column * tile + local_row] = a;
        } else {
            left[local_row * tile + local_column] = a;
        }
        // Read the contiguous dimension cooperatively for either weight layout.
        let weight_k = block * tile + if transposed { local_column } else { local_row };
        let weight_n =
            (CUBE_POS_X as usize) * tile + if transposed { local_row } else { local_column };
        let mut b = F::new(0_f32);
        if weight_k < k && weight_n < weight.shape(1) {
            b = F::cast_from(weight[weight_k * weight.stride(0) + weight_n * weight.stride(1)]);
        }
        if transposed {
            right[local_column * tile + local_row] = b;
        } else {
            right[local_row * tile + local_column] = b;
        }
        sync_cube();
        #[unroll]
        for i in 0..tile {
            sum.store(
                sum.read()
                    + f32::cast_from(left[local_row * tile + i])
                        * f32::cast_from(right[i * tile + local_column]),
            );
        }
        sync_cube();
    }
    if row < rows && column < columns {
        output[(batch * rows + row) * columns + column] = F::cast_from(sum.consume());
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
                        for [batch, rows, k, columns] in [[2, 19, 37, 23], [1, 9, 1024, 256]] {
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
