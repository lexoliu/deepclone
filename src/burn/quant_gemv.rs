//! Fused W4A16 and W8A16 per-group dequantizing GEMV and tiled prefill projections.
//!
//! Packed weights store biased unsigned lanes inside little-endian 32-bit words
//! (reinterpreted from Burn's `i32` integer storage for universal WGSL/Metal/Vulkan
//! compatibility):
//! - W4 (`bits = 4`): 8 nibbles per `u32`, `w = (nib - 8.0) * scale`, `qweight: [N, K/8]`.
//! - W8 (`bits = 8`): 4 bytes per `u32`, `w = (byte - 128.0) * scale`, `qweight: [N, K/4]`.
//!
//! Scales have shape `[N, K / group_size]` in `F16` (or `F32` when running FP32).
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        cubecl::dtype_to_storage_type,
        ops::FloatTensorOps,
        tensor::{FloatTensor, IntTensor},
        wgpu::CubeBackend,
    },
    cubecl::{ir::ElemType, prelude::*, tensor_vector_size_parallel},
    tensor::{Int, Shape, Tensor as BurnTensor},
};

const COOP_LANES_X: usize = 32;
const COOP_LANES_Y: usize = 2;
const COOP_ROWS_PER_LANE: usize = 4;
const COOP_OUTPUTS_PER_CUBE: usize = COOP_LANES_Y * COOP_ROWS_PER_LANE;

const TILE_MN: usize = 32;
const TILE_K: usize = 16;

#[backend_extension(Cube, Fusion)]
trait QuantizedProjection: Backend {
    #[fusion(dtype = input, shape = Shape::from([1, 1, qweight[0]]))]
    fn quant_gemv(
        input: FloatTensor<Self>,
        qweight: IntTensor<Self>,
        scales: FloatTensor<Self>,
        bits: u8,
        group_size: usize,
    ) -> FloatTensor<Self>;

    #[fusion(dtype = input, shape = Shape::from([input[0], input[1], qweight[0]]))]
    fn quant_batched_projection(
        input: FloatTensor<Self>,
        qweight: IntTensor<Self>,
        scales: FloatTensor<Self>,
        bits: u8,
        group_size: usize,
    ) -> FloatTensor<Self>;
}

pub(super) fn project_gemv(
    input: BurnTensor<3>,
    qweight: BurnTensor<2, Int>,
    scales: BurnTensor<2>,
    bits: u8,
    group_size: usize,
) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::quant_gemv(
        input.into_dispatch(),
        qweight.into_dispatch(),
        scales.into_dispatch(),
        bits,
        group_size,
    ))
}

pub(super) fn project_batched(
    input: BurnTensor<3>,
    qweight: BurnTensor<2, Int>,
    scales: BurnTensor<2>,
    bits: u8,
    group_size: usize,
) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::quant_batched_projection(
        input.into_dispatch(),
        qweight.into_dispatch(),
        scales.into_dispatch(),
        bits,
        group_size,
    ))
}

impl QuantizedProjection for CubeBackend {
    fn quant_gemv(
        input: FloatTensor<Self>,
        qweight: IntTensor<Self>,
        scales: FloatTensor<Self>,
        bits: u8,
        group_size: usize,
    ) -> FloatTensor<Self> {
        let width = qweight.shape()[0];
        let output = Self::float_empty(
            Shape::from([1, 1, width]),
            &input.device,
            input.dtype.into(),
        );
        let dtypes = [
            dtype_to_storage_type(input.dtype),
            dtype_to_storage_type(scales.dtype),
        ];
        let in_vec = tensor_vector_size_parallel(
            input.client.io_optimized_vector_sizes(input.dtype.size()),
            input.meta.shape(),
            input.meta.strides(),
            2,
        );
        let vector_size = if in_vec >= 4 { 4 } else { 1 };
        let groups = width.div_ceil(COOP_OUTPUTS_PER_CUBE).max(1);
        if bits == 4 {
            quant_gemv_w4_kernel::launch(
                &input.client,
                CubeCount::Static(groups as u32, 1, 1),
                CubeDim::new_2d(COOP_LANES_X as u32, COOP_LANES_Y as u32),
                vector_size,
                input.clone().into_tensor_arg(),
                qweight.into_tensor_arg(),
                scales.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                group_size / 8,
                dtypes,
            );
        } else {
            debug_assert_eq!(bits, 8);
            quant_gemv_w8_kernel::launch(
                &input.client,
                CubeCount::Static(groups as u32, 1, 1),
                CubeDim::new_2d(COOP_LANES_X as u32, COOP_LANES_Y as u32),
                vector_size,
                input.clone().into_tensor_arg(),
                qweight.into_tensor_arg(),
                scales.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                group_size / 4,
                dtypes,
            );
        }
        output
    }

    fn quant_batched_projection(
        input: FloatTensor<Self>,
        qweight: IntTensor<Self>,
        scales: FloatTensor<Self>,
        bits: u8,
        group_size: usize,
    ) -> FloatTensor<Self> {
        let batch = input.shape()[0];
        let rows = input.shape()[1];
        let columns = qweight.shape()[0];
        let output = Self::float_empty(
            Shape::from([batch, rows, columns]),
            &input.device,
            input.dtype.into(),
        );
        let tile_m = if rows <= TILE_K { TILE_K } else { TILE_MN };
        quant_projection_kernel::launch(
            &input.client,
            CubeCount::Static(
                columns.div_ceil(TILE_MN) as u32,
                rows.div_ceil(tile_m) as u32,
                batch as u32,
            ),
            CubeDim::new_2d(TILE_K as u32, TILE_K as u32),
            input.clone().into_tensor_arg(),
            qweight.into_tensor_arg(),
            scales.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            tile_m,
            bits == 4,
            group_size,
            input.meta.strides()[1] < input.meta.strides()[2],
            [
                dtype_to_storage_type(input.dtype),
                dtype_to_storage_type(scales.dtype),
            ],
        );
        output
    }
}

#[cube]
fn unpack_dot_w4(
    w: u32,
    x0: f32,
    x1: f32,
    x2: f32,
    x3: f32,
    x4: f32,
    x5: f32,
    x6: f32,
    x7: f32,
    x_bias: f32,
    scale: f32,
) -> f32 {
    let d0 = f32::cast_from(w & 0xFu32) * x0
        + f32::cast_from((w >> 4u32) & 0xFu32) * x1
        + f32::cast_from((w >> 8u32) & 0xFu32) * x2
        + f32::cast_from((w >> 12u32) & 0xFu32) * x3;
    let d1 = f32::cast_from((w >> 16u32) & 0xFu32) * x4
        + f32::cast_from((w >> 20u32) & 0xFu32) * x5
        + f32::cast_from((w >> 24u32) & 0xFu32) * x6
        + f32::cast_from((w >> 28u32) & 0xFu32) * x7;
    (d0 + d1 - x_bias) * scale
}

#[cube]
fn unpack_dot_w8(w: u32, x0: f32, x1: f32, x2: f32, x3: f32, x_bias: f32, scale: f32) -> f32 {
    let d = f32::cast_from(w & 0xFFu32) * x0
        + f32::cast_from((w >> 8u32) & 0xFFu32) * x1
        + f32::cast_from((w >> 16u32) & 0xFFu32) * x2
        + f32::cast_from((w >> 24u32) & 0xFFu32) * x3;
    (d - x_bias) * scale
}

#[cube(launch)]
fn quant_gemv_w4_kernel<F: Float, S: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    qweight: &Tensor<i32>,
    scales: &Tensor<S>,
    output: &mut Tensor<F>,
    #[comptime] words_per_group: usize,
    #[define(F, S)] _dtypes: [ElemType; 2],
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let row_base = (CUBE_POS_X as usize) * COOP_OUTPUTS_PER_CUBE + ty * COOP_ROWS_PER_LANE;
    let width = qweight.shape(0);
    let k_words = qweight.shape(1);
    let qw_s0 = qweight.stride(0);
    let sc_s0 = scales.stride(0);
    let vec_size = input.vector_size();
    let in_s = input.stride(2);

    let r0 = row_base;
    let r1 = row_base + 1usize;
    let r2 = row_base + 2usize;
    let r3 = row_base + 3usize;
    let has0 = r0 < width;
    let has1 = r1 < width;
    let has2 = r2 < width;
    let has3 = r3 < width;

    let s0 = RuntimeCell::<f32>::new(0f32);
    let s1 = RuntimeCell::<f32>::new(0f32);
    let s2 = RuntimeCell::<f32>::new(0f32);
    let s3 = RuntimeCell::<f32>::new(0f32);

    let w0 = r0 * qw_s0;
    let w1 = r1 * qw_s0;
    let w2 = r2 * qw_s0;
    let w3 = r3 * qw_s0;
    let sc0 = r0 * sc_s0;
    let sc1 = r1 * sc_s0;
    let sc2 = r2 * sc_s0;
    let sc3 = r3 * sc_s0;

    let mut wi = tx;
    if has3 {
        while wi < k_words {
            let (x0, x1, x2, x3, x4, x5, x6, x7) = if comptime!(vec_size == 4) {
                let v0 = Vector::<f32, N>::cast_from(input[wi * 2usize]);
                let v1 = Vector::<f32, N>::cast_from(input[wi * 2usize + 1usize]);
                (
                    v0.extract(0usize),
                    v0.extract(1usize),
                    v0.extract(2usize),
                    v0.extract(3usize),
                    v1.extract(0usize),
                    v1.extract(1usize),
                    v1.extract(2usize),
                    v1.extract(3usize),
                )
            } else {
                let k0 = wi * 8usize;
                (
                    f32::cast_from(input[k0 * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 1usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 2usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 3usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 4usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 5usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 6usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 7usize) * in_s].extract(0usize)),
                )
            };
            let x_bias = 8.0f32 * ((x0 + x1 + x2 + x3) + (x4 + x5 + x6 + x7));
            let g = if comptime!(words_per_group.is_power_of_two()) {
                wi >> comptime!(words_per_group.trailing_zeros() as usize)
            } else {
                wi / words_per_group
            };

            s0.store(
                s0.read()
                    + unpack_dot_w4(
                        u32::reinterpret(qweight[w0 + wi]),
                        x0,
                        x1,
                        x2,
                        x3,
                        x4,
                        x5,
                        x6,
                        x7,
                        x_bias,
                        f32::cast_from(scales[sc0 + g]),
                    ),
            );
            s1.store(
                s1.read()
                    + unpack_dot_w4(
                        u32::reinterpret(qweight[w1 + wi]),
                        x0,
                        x1,
                        x2,
                        x3,
                        x4,
                        x5,
                        x6,
                        x7,
                        x_bias,
                        f32::cast_from(scales[sc1 + g]),
                    ),
            );
            s2.store(
                s2.read()
                    + unpack_dot_w4(
                        u32::reinterpret(qweight[w2 + wi]),
                        x0,
                        x1,
                        x2,
                        x3,
                        x4,
                        x5,
                        x6,
                        x7,
                        x_bias,
                        f32::cast_from(scales[sc2 + g]),
                    ),
            );
            s3.store(
                s3.read()
                    + unpack_dot_w4(
                        u32::reinterpret(qweight[w3 + wi]),
                        x0,
                        x1,
                        x2,
                        x3,
                        x4,
                        x5,
                        x6,
                        x7,
                        x_bias,
                        f32::cast_from(scales[sc3 + g]),
                    ),
            );
            wi += COOP_LANES_X;
        }
    } else {
        while wi < k_words {
            let (x0, x1, x2, x3, x4, x5, x6, x7) = if comptime!(vec_size == 4) {
                let v0 = Vector::<f32, N>::cast_from(input[wi * 2usize]);
                let v1 = Vector::<f32, N>::cast_from(input[wi * 2usize + 1usize]);
                (
                    v0.extract(0usize),
                    v0.extract(1usize),
                    v0.extract(2usize),
                    v0.extract(3usize),
                    v1.extract(0usize),
                    v1.extract(1usize),
                    v1.extract(2usize),
                    v1.extract(3usize),
                )
            } else {
                let k0 = wi * 8usize;
                (
                    f32::cast_from(input[k0 * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 1usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 2usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 3usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 4usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 5usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 6usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 7usize) * in_s].extract(0usize)),
                )
            };
            let x_bias = 8.0f32 * ((x0 + x1 + x2 + x3) + (x4 + x5 + x6 + x7));
            let g = if comptime!(words_per_group.is_power_of_two()) {
                wi >> comptime!(words_per_group.trailing_zeros() as usize)
            } else {
                wi / words_per_group
            };

            if has0 {
                s0.store(
                    s0.read()
                        + unpack_dot_w4(
                            u32::reinterpret(qweight[w0 + wi]),
                            x0,
                            x1,
                            x2,
                            x3,
                            x4,
                            x5,
                            x6,
                            x7,
                            x_bias,
                            f32::cast_from(scales[sc0 + g]),
                        ),
                );
            }
            if has1 {
                s1.store(
                    s1.read()
                        + unpack_dot_w4(
                            u32::reinterpret(qweight[w1 + wi]),
                            x0,
                            x1,
                            x2,
                            x3,
                            x4,
                            x5,
                            x6,
                            x7,
                            x_bias,
                            f32::cast_from(scales[sc1 + g]),
                        ),
                );
            }
            if has2 {
                s2.store(
                    s2.read()
                        + unpack_dot_w4(
                            u32::reinterpret(qweight[w2 + wi]),
                            x0,
                            x1,
                            x2,
                            x3,
                            x4,
                            x5,
                            x6,
                            x7,
                            x_bias,
                            f32::cast_from(scales[sc2 + g]),
                        ),
                );
            }
            wi += COOP_LANES_X;
        }
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
        let col = row_base + tx;
        if col < width {
            output[col] = F::cast_from(partial[row_s + tx * COOP_LANES_X]);
        }
    }
}

#[cube(launch)]
fn quant_gemv_w8_kernel<F: Float, S: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    qweight: &Tensor<i32>,
    scales: &Tensor<S>,
    output: &mut Tensor<F>,
    #[comptime] words_per_group: usize,
    #[define(F, S)] _dtypes: [ElemType; 2],
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let row_base = (CUBE_POS_X as usize) * COOP_OUTPUTS_PER_CUBE + ty * COOP_ROWS_PER_LANE;
    let width = qweight.shape(0);
    let k_words = qweight.shape(1);
    let qw_s0 = qweight.stride(0);
    let sc_s0 = scales.stride(0);
    let vec_size = input.vector_size();
    let in_s = input.stride(2);

    let r0 = row_base;
    let r1 = row_base + 1usize;
    let r2 = row_base + 2usize;
    let r3 = row_base + 3usize;
    let has0 = r0 < width;
    let has1 = r1 < width;
    let has2 = r2 < width;
    let has3 = r3 < width;

    let s0 = RuntimeCell::<f32>::new(0f32);
    let s1 = RuntimeCell::<f32>::new(0f32);
    let s2 = RuntimeCell::<f32>::new(0f32);
    let s3 = RuntimeCell::<f32>::new(0f32);

    let w0 = r0 * qw_s0;
    let w1 = r1 * qw_s0;
    let w2 = r2 * qw_s0;
    let w3 = r3 * qw_s0;
    let sc0 = r0 * sc_s0;
    let sc1 = r1 * sc_s0;
    let sc2 = r2 * sc_s0;
    let sc3 = r3 * sc_s0;

    let mut wi = tx;
    if has3 {
        while wi < k_words {
            let (x0, x1, x2, x3) = if comptime!(vec_size == 4) {
                let v = Vector::<f32, N>::cast_from(input[wi]);
                (
                    v.extract(0usize),
                    v.extract(1usize),
                    v.extract(2usize),
                    v.extract(3usize),
                )
            } else {
                let k0 = wi * 4usize;
                (
                    f32::cast_from(input[k0 * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 1usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 2usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 3usize) * in_s].extract(0usize)),
                )
            };
            let x_bias = 128.0f32 * (x0 + x1 + x2 + x3);
            let g = if comptime!(words_per_group.is_power_of_two()) {
                wi >> comptime!(words_per_group.trailing_zeros() as usize)
            } else {
                wi / words_per_group
            };

            s0.store(
                s0.read()
                    + unpack_dot_w8(
                        u32::reinterpret(qweight[w0 + wi]),
                        x0,
                        x1,
                        x2,
                        x3,
                        x_bias,
                        f32::cast_from(scales[sc0 + g]),
                    ),
            );
            s1.store(
                s1.read()
                    + unpack_dot_w8(
                        u32::reinterpret(qweight[w1 + wi]),
                        x0,
                        x1,
                        x2,
                        x3,
                        x_bias,
                        f32::cast_from(scales[sc1 + g]),
                    ),
            );
            s2.store(
                s2.read()
                    + unpack_dot_w8(
                        u32::reinterpret(qweight[w2 + wi]),
                        x0,
                        x1,
                        x2,
                        x3,
                        x_bias,
                        f32::cast_from(scales[sc2 + g]),
                    ),
            );
            s3.store(
                s3.read()
                    + unpack_dot_w8(
                        u32::reinterpret(qweight[w3 + wi]),
                        x0,
                        x1,
                        x2,
                        x3,
                        x_bias,
                        f32::cast_from(scales[sc3 + g]),
                    ),
            );
            wi += COOP_LANES_X;
        }
    } else {
        while wi < k_words {
            let (x0, x1, x2, x3) = if comptime!(vec_size == 4) {
                let v = Vector::<f32, N>::cast_from(input[wi]);
                (
                    v.extract(0usize),
                    v.extract(1usize),
                    v.extract(2usize),
                    v.extract(3usize),
                )
            } else {
                let k0 = wi * 4usize;
                (
                    f32::cast_from(input[k0 * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 1usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 2usize) * in_s].extract(0usize)),
                    f32::cast_from(input[(k0 + 3usize) * in_s].extract(0usize)),
                )
            };
            let x_bias = 128.0f32 * (x0 + x1 + x2 + x3);
            let g = if comptime!(words_per_group.is_power_of_two()) {
                wi >> comptime!(words_per_group.trailing_zeros() as usize)
            } else {
                wi / words_per_group
            };

            if has0 {
                s0.store(
                    s0.read()
                        + unpack_dot_w8(
                            u32::reinterpret(qweight[w0 + wi]),
                            x0,
                            x1,
                            x2,
                            x3,
                            x_bias,
                            f32::cast_from(scales[sc0 + g]),
                        ),
                );
            }
            if has1 {
                s1.store(
                    s1.read()
                        + unpack_dot_w8(
                            u32::reinterpret(qweight[w1 + wi]),
                            x0,
                            x1,
                            x2,
                            x3,
                            x_bias,
                            f32::cast_from(scales[sc1 + g]),
                        ),
                );
            }
            if has2 {
                s2.store(
                    s2.read()
                        + unpack_dot_w8(
                            u32::reinterpret(qweight[w2 + wi]),
                            x0,
                            x1,
                            x2,
                            x3,
                            x_bias,
                            f32::cast_from(scales[sc2 + g]),
                        ),
                );
            }
            wi += COOP_LANES_X;
        }
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
        let col = row_base + tx;
        if col < width {
            output[col] = F::cast_from(partial[row_s + tx * COOP_LANES_X]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn quant_projection_kernel<F: Float, S: Float>(
    input: &Tensor<F>,
    qweight: &Tensor<i32>,
    scales: &Tensor<S>,
    output: &mut Tensor<F>,
    #[comptime] tile_m: usize,
    #[comptime] is_w4: bool,
    #[comptime] group_size: usize,
    #[comptime] input_transposed: bool,
    #[define(F, S)] _dtypes: [ElemType; 2],
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
    let qw_s0 = qweight.stride(0);
    let sc_s0 = scales.stride(0);

    let mut left = Shared::<[F]>::new_slice(tile_m * TILE_K);
    let mut right = Shared::<[F]>::new_slice(512usize);

    let acc00 = RuntimeCell::<f32>::new(0f32);
    let acc01 = RuntimeCell::<f32>::new(0f32);
    let acc10 = RuntimeCell::<f32>::new(0f32);
    let acc11 = RuntimeCell::<f32>::new(0f32);

    for block in 0..k.div_ceil(TILE_K) {
        let k_base = block * TILE_K;

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

        let wk = k_base + tx;
        let c0 = col_base + ty;
        let c1 = c0 + TILE_K;
        let mut b0 = F::new(0_f32);
        let mut b1 = F::new(0_f32);
        if wk < k {
            let g = wk / group_size;
            if is_w4 {
                let wi = wk / 8usize;
                let shift = ((wk & 7usize) * 4usize) as u32;
                if c0 < columns {
                    let word = u32::reinterpret(qweight[c0 * qw_s0 + wi]);
                    let nib = (word >> shift) & 0xFu32;
                    let sc = f32::cast_from(scales[c0 * sc_s0 + g]);
                    b0 = F::cast_from((f32::cast_from(nib) - 8.0f32) * sc);
                }
                if c1 < columns {
                    let word = u32::reinterpret(qweight[c1 * qw_s0 + wi]);
                    let nib = (word >> shift) & 0xFu32;
                    let sc = f32::cast_from(scales[c1 * sc_s0 + g]);
                    b1 = F::cast_from((f32::cast_from(nib) - 8.0f32) * sc);
                }
            } else {
                let wi = wk / 4usize;
                let shift = ((wk & 3usize) * 8usize) as u32;
                if c0 < columns {
                    let word = u32::reinterpret(qweight[c0 * qw_s0 + wi]);
                    let byte = (word >> shift) & 0xFFu32;
                    let sc = f32::cast_from(scales[c0 * sc_s0 + g]);
                    b0 = F::cast_from((f32::cast_from(byte) - 128.0f32) * sc);
                }
                if c1 < columns {
                    let word = u32::reinterpret(qweight[c1 * qw_s0 + wi]);
                    let byte = (word >> shift) & 0xFFu32;
                    let sc = f32::cast_from(scales[c1 * sc_s0 + g]);
                    b1 = F::cast_from((f32::cast_from(byte) - 128.0f32) * sc);
                }
            }
        }
        right[tx * TILE_MN + ty] = b0;
        right[tx * TILE_MN + ty + TILE_K] = b1;

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

    fn quantize_reference(
        weights: &[f32],
        n: usize,
        k: usize,
        bits: u8,
        group_size: usize,
    ) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
        let pack = 32 / usize::from(bits);
        let qmax = if bits == 4 { 7.0f32 } else { 127.0f32 };
        let bias = if bits == 4 { 8i32 } else { 128i32 };
        let mask = if bits == 4 { 0xFu32 } else { 0xFFu32 };
        let words_per_row = k / pack;
        let groups_per_row = k / group_size;
        let words_per_group = group_size / pack;
        let mut qwords = vec![0i32; n * words_per_row];
        let mut scales = vec![0f32; n * groups_per_row];
        let mut dequant = vec![0f32; n * k];
        for r in 0..n {
            for g in 0..groups_per_row {
                let start = r * k + g * group_size;
                let amax = weights[start..start + group_size]
                    .iter()
                    .fold(0f32, |m, &v| m.max(v.abs()))
                    .max(1e-6);
                let s = burn::tensor::f16::from_f32(amax / qmax).to_f32().max(1e-7);
                scales[r * groups_per_row + g] = s;
                for wi in 0..words_per_group {
                    let mut packed = 0u32;
                    for lane in 0..pack {
                        let v = weights[start + wi * pack + lane];
                        let q = (v / s).round().clamp(-qmax, qmax) as i32;
                        dequant[start + wi * pack + lane] = (q as f32) * s;
                        let u = ((q + bias) as u32) & mask;
                        packed |= u << (lane * usize::from(bits));
                    }
                    qwords[r * words_per_row + g * words_per_group + wi] = packed as i32;
                }
            }
        }
        (qwords, scales, dequant)
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn quantized_gemv_and_batched_projection_match_dequantized_reference() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            let group_size = 32usize;
            for bits in [4u8, 8u8] {
                let pack = 32 / usize::from(bits);
                for dtype in [DType::F32, DType::F16] {
                    for (t, k, n) in [
                        (1, 1024, 384),
                        (1, 3072, 1024),
                        (19, 256, 73),
                        (12, 512, 1024),
                    ] {
                        let input_data: Vec<f32> = (0..t * k)
                            .map(|i| ((i * 17 + 3) % 41) as f32 / 32.0 - 0.5)
                            .collect();
                        let weight_data: Vec<f32> = (0..n * k)
                            .map(|i| ((i * 31 + 7) % 53) as f32 / 64.0 - 0.4)
                            .collect();
                        let (qwords, scales, dequant) =
                            quantize_reference(&weight_data, n, k, bits, group_size);

                        let input = BurnTensor::<3>::from_data(
                            TensorData::new(input_data.clone(), [1, t, k]),
                            (&device, dtype),
                        );
                        let qweight = BurnTensor::<2, Int>::from_data(
                            TensorData::new(qwords, [n, k / pack]),
                            &device,
                        );
                        let scales_t = BurnTensor::<2>::from_data(
                            TensorData::new(scales, [n, k / group_size]),
                            (&device, DType::F16),
                        );

                        let out = if t == 1 {
                            project_gemv(input.clone(), qweight, scales_t, bits, group_size)
                        } else {
                            project_batched(input.clone(), qweight, scales_t, bits, group_size)
                        };
                        let deq_t = BurnTensor::<2>::from_data(
                            TensorData::new(dequant, [n, k]),
                            (&device, DType::F32),
                        )
                        .transpose();
                        let deq_ref = if t == 1 {
                            deq_t
                        } else {
                            deq_t.cast(dtype).cast(DType::F32)
                        };
                        let reference = input
                            .cast(DType::F32)
                            .matmul(deq_ref.unsqueeze())
                            .cast(dtype);

                        let (actual, expected) = futures_lite::future::zip(
                            super::super::read(out),
                            super::super::read(reference),
                        )
                        .await;
                        let (actual, expected) = (actual.unwrap(), expected.unwrap());
                        assert_eq!(actual.len(), expected.len());
                        assert!(actual.iter().all(|v| v.is_finite()));
                        let max_err = actual
                            .iter()
                            .zip(&expected)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        let tol = if dtype == DType::F16 { 0.01 } else { 1e-4 };
                        assert!(
                            max_err < tol,
                            "W{bits} {dtype:?} t={t} k={k} n={n}: max_err={max_err}"
                        );
                    }
                }
            }
        });
    }
}
