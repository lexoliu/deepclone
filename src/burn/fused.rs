//! Fused transformer normalization, rotary KV cache update, and gated activation kernels.
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension,
        cubecl::dtype_to_storage_type,
        fusion::custom::TensorSpec,
        ops::FloatTensorOps,
        tensor::FloatTensor,
        wgpu::{CubeBackend, CubeTensor},
    },
    cubecl::{ir::ElemType, prelude::*},
    tensor::{DType, FloatDType, Shape, Tensor as BurnTensor},
};

pub(super) fn into_contiguous(tensor: CubeTensor) -> CubeTensor {
    if tensor.is_contiguous() {
        return tensor;
    }
    let (client, device, dtype) = (tensor.client.clone(), tensor.device.clone(), tensor.dtype);
    let output = burn::cubecl::std::tensor::into_contiguous(
        &client,
        tensor.binding(),
        dtype_to_storage_type(dtype),
    );
    CubeTensor::new(client, output.handle, *output.metadata, device, dtype)
}

pub(super) fn into_mut_contiguous(tensor: CubeTensor) -> CubeTensor {
    if tensor.can_mut() && tensor.is_contiguous() && tensor.is_nonoverlapping() {
        tensor
    } else if tensor.is_contiguous() {
        tensor.copy()
    } else {
        into_contiguous(tensor)
    }
}

#[backend_extension(Cube, Fusion)]
trait FusedTransformer: Backend {
    #[fusion(dtype = input, shape = input.clone())]
    fn rms_norm(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        epsilon: f32,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |residual, _delta, _weight, _epsilon| (residual.clone(), residual.clone()))]
    fn add_rms_norm(
        residual: FloatTensor<Self>,
        delta: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    #[allow(clippy::too_many_arguments)]
    #[fusion(meta = |qkv, _qk_norm, _rotary, keys, values, _position, _rotary_position, num_q_heads, _num_kv_heads, head_dim, _epsilon, _scale, _has_qk_norm| (
        TensorSpec::new(Shape::from([1, *num_q_heads, qkv.shape[1], *head_dim]), qkv.dtype),
        keys.clone(),
        values.clone(),
    ))]
    fn qkv_rope_cache(
        qkv: FloatTensor<Self>,
        qk_norm: FloatTensor<Self>,
        rotary: FloatTensor<Self>,
        keys: FloatTensor<Self>,
        values: FloatTensor<Self>,
        position: usize,
        rotary_position: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        epsilon: f32,
        scale: f32,
        has_qk_norm: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(dtype = DType::F32, shape = Shape::from([gate_up[0], gate_up[1], gate_up[2] / 2]))]
    fn gated_silu(gate_up: FloatTensor<Self>) -> FloatTensor<Self>;

    #[fusion(dtype = input, shape = input.clone())]
    fn rotary(
        input: FloatTensor<Self>,
        cosine: FloatTensor<Self>,
        sine: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

pub(super) fn rms_norm<const D: usize>(
    input: BurnTensor<D>,
    weight: BurnTensor<1>,
    epsilon: f32,
) -> BurnTensor<D> {
    BurnTensor::from_dispatch(Dispatch::rms_norm(
        input.into_dispatch(),
        weight.into_dispatch(),
        epsilon,
    ))
}

pub(super) fn add_rms_norm(
    residual: BurnTensor<3>,
    delta: BurnTensor<3>,
    weight: BurnTensor<1>,
    epsilon: f32,
) -> (BurnTensor<3>, BurnTensor<3>) {
    let (sum, normed) = Dispatch::add_rms_norm(
        residual.into_dispatch(),
        delta.into_dispatch(),
        weight.into_dispatch(),
        epsilon,
    );
    (
        BurnTensor::from_dispatch(sum),
        BurnTensor::from_dispatch(normed),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qkv_rope_cache(
    qkv: BurnTensor<3>,
    qk_norm: BurnTensor<2>,
    rotary: BurnTensor<3>,
    keys: BurnTensor<4>,
    values: BurnTensor<4>,
    position: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    epsilon: f32,
    scale: f32,
) -> (BurnTensor<4>, BurnTensor<4>, BurnTensor<4>) {
    let (query, keys, values) = Dispatch::qkv_rope_cache(
        qkv.into_dispatch(),
        qk_norm.into_dispatch(),
        rotary.into_dispatch(),
        keys.into_dispatch(),
        values.into_dispatch(),
        position,
        position,
        num_q_heads,
        num_kv_heads,
        head_dim,
        epsilon,
        scale,
        true,
    );
    (
        BurnTensor::from_dispatch(query),
        BurnTensor::from_dispatch(keys),
        BurnTensor::from_dispatch(values),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qkv_rope_cache_no_norm(
    qkv: BurnTensor<3>,
    rotary: BurnTensor<3>,
    keys: BurnTensor<4>,
    values: BurnTensor<4>,
    cache_position: usize,
    rotary_position: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> (BurnTensor<4>, BurnTensor<4>, BurnTensor<4>) {
    let qkv_dispatch = qkv.into_dispatch();
    let (query, keys, values) = Dispatch::qkv_rope_cache(
        qkv_dispatch.clone(),
        qkv_dispatch,
        rotary.into_dispatch(),
        keys.into_dispatch(),
        values.into_dispatch(),
        cache_position,
        rotary_position,
        num_q_heads,
        num_kv_heads,
        head_dim,
        0.0,
        scale,
        false,
    );
    (
        BurnTensor::from_dispatch(query),
        BurnTensor::from_dispatch(keys),
        BurnTensor::from_dispatch(values),
    )
}

pub(super) fn gated_silu(gate_up: BurnTensor<3>) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::gated_silu(gate_up.into_dispatch()))
}

pub(super) fn rotary(
    input: BurnTensor<4>,
    cosine: BurnTensor<4>,
    sine: BurnTensor<4>,
) -> BurnTensor<4> {
    BurnTensor::from_dispatch(Dispatch::rotary(
        input.into_dispatch(),
        cosine.into_dispatch(),
        sine.into_dispatch(),
    ))
}

fn reduction_lanes(client: &burn::cubecl::client::Client, work: usize) -> usize {
    let limit = (client.properties().hardware.max_units_per_cube as usize).min(256);
    let max_pow2 = 1usize << (usize::BITS - 1 - limit.leading_zeros());
    work.max(1).next_power_of_two().min(max_pow2)
}

pub(super) fn row_grid(rows: usize, max_x: u32) -> CubeCount {
    let max_x = max_x as usize;
    if rows <= max_x {
        CubeCount::Static(rows as u32, 1, 1)
    } else {
        let y = rows.div_ceil(max_x);
        CubeCount::Static(max_x as u32, y as u32, 1)
    }
}

impl FusedTransformer for CubeBackend {
    fn rms_norm(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        epsilon: f32,
    ) -> FloatTensor<Self> {
        let input = into_contiguous(input);
        let weight = into_contiguous(weight);
        let shape = input.shape();
        let width = weight.shape()[0];
        let rows = shape.num_elements() / width;
        let output = Self::float_empty(shape, &input.device, input.dtype.into());
        let lanes = reduction_lanes(&input.client, width);
        let max_x = input.client.properties().hardware.max_cube_count.0;
        rms_norm_kernel::launch(
            &input.client,
            row_grid(rows, max_x),
            CubeDim::new_1d(lanes as u32),
            input.clone().into_tensor_arg(),
            weight.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            rows,
            width,
            epsilon,
            lanes,
            dtype_to_storage_type(input.dtype),
        );
        output
    }

    fn add_rms_norm(
        residual: FloatTensor<Self>,
        delta: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        epsilon: f32,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let sum = into_mut_contiguous(residual);
        let delta = into_contiguous(delta);
        let weight = into_contiguous(weight);
        let shape = sum.shape();
        let width = weight.shape()[0];
        let rows = shape.num_elements() / width;
        let normed = Self::float_empty(shape, &sum.device, sum.dtype.into());
        let lanes = reduction_lanes(&sum.client, width);
        let max_x = sum.client.properties().hardware.max_cube_count.0;
        add_rms_norm_kernel::launch(
            &sum.client,
            row_grid(rows, max_x),
            CubeDim::new_1d(lanes as u32),
            sum.clone().into_tensor_arg(),
            delta.clone().into_tensor_arg(),
            weight.into_tensor_arg(),
            normed.clone().into_tensor_arg(),
            rows,
            width,
            epsilon,
            lanes,
            [
                dtype_to_storage_type(sum.dtype),
                dtype_to_storage_type(delta.dtype),
            ],
        );
        (sum, normed)
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_rope_cache(
        qkv: FloatTensor<Self>,
        qk_norm: FloatTensor<Self>,
        rotary: FloatTensor<Self>,
        keys: FloatTensor<Self>,
        values: FloatTensor<Self>,
        position: usize,
        rotary_position: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        epsilon: f32,
        scale: f32,
        has_qk_norm: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        let qkv = into_contiguous(qkv);
        let qk_norm = if has_qk_norm {
            into_contiguous(qk_norm)
        } else {
            qkv.clone()
        };
        let rotary = into_contiguous(rotary);
        let keys = into_mut_contiguous(keys);
        let values = into_mut_contiguous(values);
        let tokens = qkv.shape()[1];
        let capacity = keys.shape()[2];
        let query = Self::float_empty(
            Shape::from([1, num_q_heads, tokens, head_dim]),
            &qkv.device,
            qkv.dtype.into(),
        );
        let half = head_dim / 2;
        let lanes = reduction_lanes(&qkv.client, half);
        let total_heads = num_q_heads + 2 * num_kv_heads;
        qkv_rope_cache_kernel::launch(
            &qkv.client,
            CubeCount::Static(total_heads as u32, tokens as u32, 1),
            CubeDim::new_1d(lanes as u32),
            qkv.clone().into_tensor_arg(),
            qk_norm.into_tensor_arg(),
            rotary.into_tensor_arg(),
            query.clone().into_tensor_arg(),
            keys.clone().into_tensor_arg(),
            values.clone().into_tensor_arg(),
            position,
            rotary_position,
            tokens,
            capacity,
            epsilon,
            scale,
            num_q_heads,
            num_kv_heads,
            head_dim,
            lanes,
            has_qk_norm,
            dtype_to_storage_type(qkv.dtype),
        );
        (query, keys, values)
    }

    fn gated_silu(gate_up: FloatTensor<Self>) -> FloatTensor<Self> {
        let gate_up = into_contiguous(gate_up);
        let shape = gate_up.shape();
        let batch = shape[0];
        let tokens = shape[1];
        let inter = shape[2] / 2;
        let output = Self::float_empty(
            Shape::from([batch, tokens, inter]),
            &gate_up.device,
            FloatDType::F32,
        );
        let total = batch * tokens * inter;
        let dim = CubeDim::new(&gate_up.client, total);
        let groups = total.div_ceil(dim.num_elems() as usize);
        let max_x = gate_up.client.properties().hardware.max_cube_count.0;
        gated_silu_kernel::launch(
            &gate_up.client,
            row_grid(groups, max_x),
            dim,
            gate_up.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            total,
            inter,
            dtype_to_storage_type(gate_up.dtype),
        );
        output
    }

    fn rotary(
        input: FloatTensor<Self>,
        cosine: FloatTensor<Self>,
        sine: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let input = into_contiguous(input);
        let cosine = into_contiguous(cosine);
        let sine = into_contiguous(sine);
        let shape = input.shape();
        let [batch, heads, tokens, width] = [shape[0], shape[1], shape[2], shape[3]];
        let output = Self::float_empty(shape, &input.device, input.dtype.into());
        let half = width / 2;
        let lanes = reduction_lanes(&input.client, half);
        rotary_kernel::launch(
            &input.client,
            CubeCount::Static(heads as u32, tokens as u32, batch as u32),
            CubeDim::new_1d(lanes as u32),
            input.clone().into_tensor_arg(),
            cosine.into_tensor_arg(),
            sine.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            heads,
            tokens,
            width,
            dtype_to_storage_type(input.dtype),
        );
        output
    }
}

#[cube(launch)]
fn rms_norm_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    output: &mut Tensor<F>,
    rows: usize,
    width: usize,
    epsilon: f32,
    #[comptime] lanes: usize,
    #[define(F)] _dtype: ElemType,
) {
    let row = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    if row >= rows {
        terminate!();
    }
    let base = row * width;
    let sum = RuntimeCell::<f32>::new(0f32);
    let mut col = UNIT_POS as usize;
    while col < width {
        let v = f32::cast_from(input[base + col]);
        sum.store(sum.read() + v * v);
        col += lanes;
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
    let inv = (partial[0] / (width as f32) + epsilon).sqrt().recip();
    let mut col = UNIT_POS as usize;
    while col < width {
        let v = f32::cast_from(input[base + col]);
        output[base + col] = F::cast_from(v * inv) * weight[col];
        col += lanes;
    }
}

#[cube(launch)]
fn add_rms_norm_kernel<F: Float, D: Float>(
    sum: &mut Tensor<F>,
    delta: &Tensor<D>,
    weight: &Tensor<F>,
    normed: &mut Tensor<F>,
    rows: usize,
    width: usize,
    epsilon: f32,
    #[comptime] lanes: usize,
    #[define(F, D)] _dtypes: [ElemType; 2],
) {
    let row = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    if row >= rows {
        terminate!();
    }
    let base = row * width;
    let sq = RuntimeCell::<f32>::new(0f32);
    let mut col = UNIT_POS as usize;
    while col < width {
        let s = sum[base + col] + F::cast_from(delta[base + col]);
        sum[base + col] = s;
        let v = f32::cast_from(s);
        sq.store(sq.read() + v * v);
        col += lanes;
    }
    let mut partial = Shared::<[f32]>::new_slice(lanes);
    partial[UNIT_POS as usize] = sq.consume();
    sync_cube();
    #[unroll]
    for stage in 0..lanes.ilog2() as usize {
        let distance = comptime!(lanes >> (stage + 1));
        if (UNIT_POS as usize) < distance {
            partial[UNIT_POS as usize] += partial[UNIT_POS as usize + distance];
        }
        sync_cube();
    }
    let inv = (partial[0] / (width as f32) + epsilon).sqrt().recip();
    let mut col = UNIT_POS as usize;
    while col < width {
        let v = f32::cast_from(sum[base + col]);
        normed[base + col] = F::cast_from(v * inv) * weight[col];
        col += lanes;
    }
}

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn qkv_rope_cache_kernel<F: Float>(
    qkv: &Tensor<F>,
    qk_norm: &Tensor<F>,
    rotary: &Tensor<F>,
    query: &mut Tensor<F>,
    keys: &mut Tensor<F>,
    values: &mut Tensor<F>,
    position: usize,
    rotary_position: usize,
    tokens: usize,
    capacity: usize,
    epsilon: f32,
    scale: f32,
    #[comptime] num_q_heads: usize,
    #[comptime] num_kv_heads: usize,
    #[comptime] head_dim: usize,
    #[comptime] lanes: usize,
    #[comptime] has_qk_norm: bool,
    #[define(F)] _dtype: ElemType,
) {
    let head = CUBE_POS_X as usize;
    let token = CUBE_POS_Y as usize;
    let half = comptime!(head_dim / 2);
    let total_heads = comptime!(num_q_heads + 2 * num_kv_heads);
    let qkv_stride = comptime!(total_heads * head_dim);
    let src_base = token * qkv_stride + head * head_dim;
    let abs_pos = position + token;
    let rot_pos = rotary_position + token;

    if head < comptime!(num_q_heads + num_kv_heads) {
        let mut inv = 1f32;
        if comptime!(has_qk_norm) {
            let sq = RuntimeCell::<f32>::new(0f32);
            let mut d = UNIT_POS as usize;
            while d < half {
                let a = f32::cast_from(qkv[src_base + d]);
                let b = f32::cast_from(qkv[src_base + d + half]);
                sq.store(sq.read() + a * a + b * b);
                d += lanes;
            }
            let mut partial = Shared::<[f32]>::new_slice(lanes);
            partial[UNIT_POS as usize] = sq.consume();
            sync_cube();
            #[unroll]
            for stage in 0..lanes.ilog2() as usize {
                let distance = comptime!(lanes >> (stage + 1));
                if (UNIT_POS as usize) < distance {
                    partial[UNIT_POS as usize] += partial[UNIT_POS as usize + distance];
                }
                sync_cube();
            }
            inv = (partial[0] / (head_dim as f32) + epsilon).sqrt().recip();
        }
        let is_q = head < num_q_heads;
        let norm_base = if is_q { 0usize } else { head_dim };
        let cos_base = rot_pos * head_dim;
        let sin_base = rotary.stride(0) + rot_pos * head_dim;
        let scale_f = F::cast_from(scale);
        let mut d = UNIT_POS as usize;
        while d < half {
            let na = if comptime!(has_qk_norm) {
                let a = f32::cast_from(qkv[src_base + d]);
                F::cast_from(a * inv) * qk_norm[norm_base + d]
            } else {
                qkv[src_base + d]
            };
            let nb = if comptime!(has_qk_norm) {
                let b = f32::cast_from(qkv[src_base + d + half]);
                F::cast_from(b * inv) * qk_norm[norm_base + d + half]
            } else {
                qkv[src_base + d + half]
            };
            let ra = na * rotary[cos_base + d] + (F::new(0_f32) - nb) * rotary[sin_base + d];
            let rb = nb * rotary[cos_base + d + half] + na * rotary[sin_base + d + half];
            if is_q {
                let dst_base = (head * tokens + token) * head_dim;
                query[dst_base + d] = ra * scale_f;
                query[dst_base + d + half] = rb * scale_f;
            } else {
                let kv_head = head - num_q_heads;
                let dst_base = (kv_head * capacity + abs_pos) * head_dim;
                keys[dst_base + d] = ra;
                keys[dst_base + d + half] = rb;
            }
            d += lanes;
        }
    } else {
        let kv_head = head - comptime!(num_q_heads + num_kv_heads);
        let dst_base = (kv_head * capacity + abs_pos) * head_dim;
        let mut d = UNIT_POS as usize;
        while d < half {
            values[dst_base + d] = qkv[src_base + d];
            values[dst_base + d + half] = qkv[src_base + d + half];
            d += lanes;
        }
    }
}

#[cube(launch)]
fn gated_silu_kernel<F: Float>(
    gate_up: &Tensor<F>,
    output: &mut Tensor<f32>,
    total: usize,
    inter: usize,
    #[define(F)] _dtype: ElemType,
) {
    let cube = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    let idx = cube * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if idx >= total {
        terminate!();
    }
    let row = idx / inter;
    let col = idx - row * inter;
    let base = row * (inter * 2);
    let g = f32::cast_from(gate_up[base + col]);
    let u = f32::cast_from(gate_up[base + inter + col]);
    // Match burn::tensor::activation::silu's sigmoid expression so FP16
    // attention-sink rounding boundaries stay aligned between single-token
    // and batched steps.
    output[idx] = g * (0f32 - (1f32 + (0f32 - g).exp()).ln()).exp() * u;
}

#[cube(launch)]
fn rotary_kernel<F: Float>(
    input: &Tensor<F>,
    cosine: &Tensor<F>,
    sine: &Tensor<F>,
    output: &mut Tensor<F>,
    heads: usize,
    tokens: usize,
    width: usize,
    #[define(F)] _dtype: ElemType,
) {
    let head = CUBE_POS_X as usize;
    let token = CUBE_POS_Y as usize;
    let batch = CUBE_POS_Z as usize;
    let half = width / 2;
    let x_base = ((batch * heads + head) * tokens + token) * width;
    let cs_base = token * width;
    let mut col = UNIT_POS as usize;
    while col < half {
        let a = input[x_base + col];
        let b = input[x_base + col + half];
        output[x_base + col] =
            a * cosine[cs_base + col] + (F::new(0_f32) - b) * sine[cs_base + col];
        output[x_base + col + half] =
            b * cosine[cs_base + col + half] + a * sine[cs_base + col + half];
        col += CUBE_DIM as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, TensorData, activation};

    fn unfused_rms_norm<const D: usize>(
        x: BurnTensor<D>,
        weight: BurnTensor<1>,
        epsilon: f32,
    ) -> BurnTensor<D> {
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let variance = (x.clone() * x.clone()).mean_dim(D - 1);
        (x * (variance + epsilon).sqrt().recip()).cast(dtype) * weight.unsqueeze()
    }

    fn unfused_rotary(
        x: BurnTensor<4>,
        cosine: &BurnTensor<4>,
        sine: &BurnTensor<4>,
    ) -> BurnTensor<4> {
        let [b, h, t, d] = x.dims();
        let first = x.clone().slice([0..b, 0..h, 0..t, 0..d / 2]);
        let second = x.clone().slice([0..b, 0..h, 0..t, d / 2..d]);
        x * cosine.clone() + BurnTensor::cat(vec![-second, first], 3) * sine.clone()
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn fused_rms_norm_and_add_rms_norm_match_reference() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            let eps = 1e-6_f32;
            for dtype in [DType::F16, DType::F32] {
                for (tokens, width) in [(1, 37), (1, 1024), (7, 128), (3, 2048)] {
                    let x = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..tokens * width)
                                .map(|i| ((i * 19 + 3) as f32).sin() * 2.5)
                                .collect::<Vec<_>>(),
                            [1, tokens, width],
                        ),
                        (&device, dtype),
                    );
                    let w = BurnTensor::<1>::from_data(
                        TensorData::new(
                            (0..width)
                                .map(|i| 0.5 + ((i * 11) as f32).cos() * 0.5)
                                .collect::<Vec<_>>(),
                            [width],
                        ),
                        (&device, dtype),
                    );
                    let expected = unfused_rms_norm(x.clone(), w.clone(), eps);
                    let actual = rms_norm(x.clone(), w.clone(), eps);
                    let (actual, expected) = futures_lite::future::zip(
                        super::super::read(actual),
                        super::super::read(expected),
                    )
                    .await;
                    let maximum = actual
                        .unwrap()
                        .into_iter()
                        .zip(expected.unwrap())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let tolerance = if dtype == DType::F16 { 0.002 } else { 1e-5 };
                    assert!(
                        maximum < tolerance,
                        "rms_norm {dtype:?} {tokens}x{width}: {maximum}"
                    );

                    for delta_dtype in [dtype, DType::F32] {
                        let delta = BurnTensor::<3>::from_data(
                            TensorData::new(
                                (0..tokens * width)
                                    .map(|i| ((i * 29 + 5) as f32).cos() * 1.25)
                                    .collect::<Vec<_>>(),
                                [1, tokens, width],
                            ),
                            (&device, delta_dtype),
                        );
                        let expected_sum = x.clone() + delta.clone().cast(dtype);
                        let expected_norm = unfused_rms_norm(expected_sum.clone(), w.clone(), eps);
                        // Exercise both aliased (shared) and uniquely owned (in-place) residual paths.
                        let (sum_shared, norm_shared) =
                            add_rms_norm(x.clone(), delta.clone(), w.clone(), eps);
                        let (sum_owned, norm_owned) =
                            add_rms_norm(x.clone() * 1.0, delta, w.clone(), eps);
                        let ((s1, n1), (s2, n2)) = futures_lite::future::zip(
                            futures_lite::future::zip(
                                super::super::read(sum_shared),
                                super::super::read(norm_shared),
                            ),
                            futures_lite::future::zip(
                                super::super::read(sum_owned),
                                super::super::read(norm_owned),
                            ),
                        )
                        .await;
                        let (es, en) = futures_lite::future::zip(
                            super::super::read(expected_sum),
                            super::super::read(expected_norm),
                        )
                        .await;
                        let (es, en) = (es.unwrap(), en.unwrap());
                        for (s, n) in [(s1.unwrap(), n1.unwrap()), (s2.unwrap(), n2.unwrap())] {
                            assert_eq!(s, es, "add_rms_norm sum {dtype:?}/{delta_dtype:?}");
                            let max_err = n
                                .into_iter()
                                .zip(&en)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0f32, f32::max);
                            assert!(
                                max_err < tolerance,
                                "add_rms_norm norm {dtype:?}/{delta_dtype:?}: {max_err}"
                            );
                        }
                    }
                }
            }
        });
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn fused_qkv_rope_cache_and_gated_silu_match_reference() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            let eps = 1e-6_f32;
            let (num_q, num_kv, head_dim, capacity) = (16usize, 8usize, 128usize, 64usize);
            for dtype in [DType::F16, DType::F32] {
                let (cosine, sine) = super::super::audio_transformer::rotary_tables(
                    &device,
                    dtype,
                    capacity,
                    head_dim,
                    1_000_000.0,
                );
                let rotary_packed = BurnTensor::cat(
                    vec![
                        cosine.clone().reshape([1, capacity, head_dim]),
                        sine.clone().reshape([1, capacity, head_dim]),
                    ],
                    0,
                );
                let q_norm_w = BurnTensor::<1>::from_data(
                    TensorData::new(
                        (0..head_dim)
                            .map(|i| 0.8 + (i as f32 * 0.01).sin() * 0.2)
                            .collect(),
                        [head_dim],
                    ),
                    (&device, dtype),
                );
                let k_norm_w = BurnTensor::<1>::from_data(
                    TensorData::new(
                        (0..head_dim)
                            .map(|i| 0.9 + (i as f32 * 0.02).cos() * 0.2)
                            .collect(),
                        [head_dim],
                    ),
                    (&device, dtype),
                );
                let qk_norm = BurnTensor::cat(
                    vec![
                        q_norm_w.clone().unsqueeze_dim(0),
                        k_norm_w.clone().unsqueeze_dim(0),
                    ],
                    0,
                );
                let mut keys_fused =
                    BurnTensor::<4>::zeros([1, num_kv, capacity, head_dim], (&device, dtype));
                let mut vals_fused =
                    BurnTensor::<4>::zeros([1, num_kv, capacity, head_dim], (&device, dtype));
                let mut keys_ref = keys_fused.clone();
                let mut vals_ref = vals_fused.clone();

                let scale = (head_dim as f32).sqrt().recip();
                for (position, tokens) in [(0usize, 13usize), (13, 1), (14, 1), (15, 7)] {
                    let width = (num_q + 2 * num_kv) * head_dim;
                    let qkv = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..tokens * width)
                                .map(|i| ((i * 37 + position * 13) as f32).sin() * 1.5)
                                .collect(),
                            [1, tokens, width],
                        ),
                        (&device, dtype),
                    );
                    let q_raw = qkv
                        .clone()
                        .slice([0..1, 0..tokens, 0..num_q * head_dim])
                        .reshape([1, tokens, num_q, head_dim])
                        .swap_dims(1, 2);
                    let k_raw = qkv
                        .clone()
                        .slice([
                            0..1,
                            0..tokens,
                            num_q * head_dim..(num_q + num_kv) * head_dim,
                        ])
                        .reshape([1, tokens, num_kv, head_dim])
                        .swap_dims(1, 2);
                    let v_raw = qkv
                        .clone()
                        .slice([0..1, 0..tokens, (num_q + num_kv) * head_dim..width])
                        .reshape([1, tokens, num_kv, head_dim])
                        .swap_dims(1, 2);
                    let cs_slice = [0..1, 0..1, position..position + tokens, 0..head_dim];
                    let cos_s = cosine.clone().slice(cs_slice.clone());
                    let sin_s = sine.clone().slice(cs_slice);
                    let q_ref = unfused_rotary(
                        unfused_rms_norm(q_raw, q_norm_w.clone(), eps),
                        &cos_s,
                        &sin_s,
                    ) * scale;
                    let k_ref = unfused_rotary(
                        unfused_rms_norm(k_raw, k_norm_w.clone(), eps),
                        &cos_s,
                        &sin_s,
                    );
                    let sel = [0..1, 0..num_kv, position..position + tokens, 0..head_dim];
                    keys_ref.inplace(|c| c.slice_assign(sel.clone(), k_ref.clone()));
                    vals_ref.inplace(|c| c.slice_assign(sel, v_raw));

                    let (q_act, k_out, v_out) = qkv_rope_cache(
                        qkv,
                        qk_norm.clone(),
                        rotary_packed.clone(),
                        keys_fused,
                        vals_fused,
                        position,
                        num_q,
                        num_kv,
                        head_dim,
                        eps,
                        scale,
                    );
                    keys_fused = k_out;
                    vals_fused = v_out;

                    let ((qa, qr), (ka, kr)) = futures_lite::future::zip(
                        futures_lite::future::zip(
                            super::super::read(q_act),
                            super::super::read(q_ref),
                        ),
                        futures_lite::future::zip(
                            super::super::read(keys_fused.clone()),
                            super::super::read(keys_ref.clone()),
                        ),
                    )
                    .await;
                    let (va, vr) = futures_lite::future::zip(
                        super::super::read(vals_fused.clone()),
                        super::super::read(vals_ref.clone()),
                    )
                    .await;
                    let tol = if dtype == DType::F16 { 0.002 } else { 1e-5 };
                    for (label, a, b) in [
                        ("query", qa.unwrap(), qr.unwrap()),
                        ("keys", ka.unwrap(), kr.unwrap()),
                        ("values", va.unwrap(), vr.unwrap()),
                    ] {
                        let max_err = a
                            .into_iter()
                            .zip(b)
                            .map(|(x, y)| (x - y).abs())
                            .fold(0f32, f32::max);
                        assert!(
                            max_err < tol,
                            "qkv_rope_cache {label} {dtype:?} pos={position} tok={tokens}: {max_err}"
                        );
                    }
                }

                // Gated SiLU including values exceeding FP16 range when multiplied,
                // plus multi-token model dimensions matching [1, 2, 6144].
                let gate_up = BurnTensor::<3>::from_data(
                    TensorData::new(
                        vec![400.0, -2.5, 0.5, 1.25, 200.0, 3.0, -4.0, 2.0],
                        [1, 1, 8],
                    ),
                    (&device, dtype),
                );
                let expected =
                    activation::silu(gate_up.clone().slice([0..1, 0..1, 0..4]).cast(DType::F32))
                        * gate_up.clone().slice([0..1, 0..1, 4..8]).cast(DType::F32);
                let actual = gated_silu(gate_up);
                let (actual, expected) = futures_lite::future::zip(
                    super::super::read(actual),
                    super::super::read(expected),
                )
                .await;
                let max_err = actual
                    .unwrap()
                    .into_iter()
                    .zip(expected.unwrap())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max_err < 1e-2, "gated_silu {dtype:?}: {max_err}");

                let inter = 3072usize;
                let gu2_data: Vec<f32> = (0..2 * 2 * inter)
                    .map(|i| ((i * 17 + 3) as f32).sin() * 8.0)
                    .collect();
                let gu2 = BurnTensor::<3>::from_data(
                    TensorData::new(gu2_data.clone(), [1, 2, 2 * inter]),
                    (&device, dtype),
                );
                let gu1 = BurnTensor::<3>::from_data(
                    TensorData::new(gu2_data[..2 * inter].to_vec(), [1, 1, 2 * inter]),
                    (&device, dtype),
                );
                let exp2 =
                    activation::silu(gu2.clone().slice([0..1, 0..2, 0..inter]).cast(DType::F32))
                        * gu2
                            .clone()
                            .slice([0..1, 0..2, inter..2 * inter])
                            .cast(DType::F32);
                let act2 = gated_silu(gu2);
                let act1 = gated_silu(gu1);
                let ((a2, e2), a1) = futures_lite::future::zip(
                    futures_lite::future::zip(super::super::read(act2), super::super::read(exp2)),
                    super::super::read(act1),
                )
                .await;
                let (a2, e2, a1) = (a2.unwrap(), e2.unwrap(), a1.unwrap());
                assert_eq!(
                    &a2[..inter],
                    &a1[..],
                    "gated_silu row 0 differs between t=1 and t=2 ({dtype:?})"
                );
                let max_err2 = a2
                    .iter()
                    .zip(&e2)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max_err2 < 1e-4, "gated_silu 2x3072 {dtype:?}: {max_err2}");

                // Non-contiguous 4D rms_norm (swap_dims layout used by q_norm/k_norm).
                let nc = BurnTensor::<4>::from_data(
                    TensorData::new(
                        (0..2 * 16 * 128)
                            .map(|i| ((i * 29) as f32).sin() * 2.0)
                            .collect(),
                        [1, 2, 16, 128],
                    ),
                    (&device, dtype),
                )
                .swap_dims(1, 2);
                let nc_exp = unfused_rms_norm(nc.clone(), q_norm_w.clone(), eps);
                let nc_act = rms_norm(nc, q_norm_w.clone(), eps);
                let (nc_a, nc_e) = futures_lite::future::zip(
                    super::super::read(nc_act),
                    super::super::read(nc_exp),
                )
                .await;
                let nc_err = nc_a
                    .unwrap()
                    .into_iter()
                    .zip(nc_e.unwrap())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    nc_err < if dtype == DType::F16 { 0.002 } else { 1e-5 },
                    "noncontig rms_norm {dtype:?}: {nc_err}"
                );

                // Standalone rotary used by AudioTransformer.
                let audio_x = BurnTensor::<4>::from_data(
                    TensorData::new(
                        (0..16 * 9 * 64).map(|i| ((i * 23) as f32).sin()).collect(),
                        [1, 16, 9, 64],
                    ),
                    (&device, dtype),
                );
                let (acos, asin) =
                    super::super::audio_transformer::rotary_tables(&device, dtype, 9, 64, 10000.0);
                let expected = unfused_rotary(audio_x.clone(), &acos, &asin);
                let actual = rotary(audio_x, acos, asin);
                let (actual, expected) = futures_lite::future::zip(
                    super::super::read(actual),
                    super::super::read(expected),
                )
                .await;
                let max_err = actual
                    .unwrap()
                    .into_iter()
                    .zip(expected.unwrap())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let tol = if dtype == DType::F16 { 0.002 } else { 1e-5 };
                assert!(max_err < tol, "rotary {dtype:?}: {max_err}");

                // Standalone qkv_rope_cache_no_norm used by AudioTransformer.
                let (h_q, h_kv, d_head, cap) = (8usize, 8usize, 64usize, 32usize);
                let kf = BurnTensor::<4>::zeros([1, h_kv, cap, d_head], (&device, dtype));
                let vf = BurnTensor::<4>::zeros([1, h_kv, cap, d_head], (&device, dtype));
                let scale = (d_head as f32).sqrt().recip();
                let t = 5usize;
                let w_tot = (h_q + 2 * h_kv) * d_head;
                let qkv_a = BurnTensor::<3>::from_data(
                    TensorData::new(
                        (0..t * w_tot)
                            .map(|i| ((i * 17 + 3) as f32).sin() * 0.7)
                            .collect(),
                        [1, t, w_tot],
                    ),
                    (&device, dtype),
                );
                let (ac, asn) = super::super::audio_transformer::rotary_tables(
                    &device, dtype, 16, d_head, 10000.0,
                );
                let rot_packed = BurnTensor::cat(
                    vec![
                        ac.clone().reshape([1, 16, d_head]),
                        asn.clone().reshape([1, 16, d_head]),
                    ],
                    0,
                );
                let cache_pos = 7usize;
                let rot_pos = 11usize;
                let q_r = unfused_rotary(
                    qkv_a
                        .clone()
                        .slice([0..1, 0..t, 0..h_q * d_head])
                        .reshape([1, t, h_q, d_head])
                        .swap_dims(1, 2),
                    &ac.clone()
                        .slice([0..1, 0..1, rot_pos..rot_pos + t, 0..d_head]),
                    &asn.clone()
                        .slice([0..1, 0..1, rot_pos..rot_pos + t, 0..d_head]),
                ) * scale;
                let k_r = unfused_rotary(
                    qkv_a
                        .clone()
                        .slice([0..1, 0..t, h_q * d_head..(h_q + h_kv) * d_head])
                        .reshape([1, t, h_kv, d_head])
                        .swap_dims(1, 2),
                    &ac.slice([0..1, 0..1, rot_pos..rot_pos + t, 0..d_head]),
                    &asn.slice([0..1, 0..1, rot_pos..rot_pos + t, 0..d_head]),
                );
                let (q_a, kf, _v_out) = qkv_rope_cache_no_norm(
                    qkv_a, rot_packed, kf, vf, cache_pos, rot_pos, h_q, h_kv, d_head, scale,
                );
                let ((qa, qr), (ka, kr)) = futures_lite::future::zip(
                    futures_lite::future::zip(super::super::read(q_a), super::super::read(q_r)),
                    futures_lite::future::zip(
                        super::super::read(kf.slice([
                            0..1,
                            0..h_kv,
                            cache_pos..cache_pos + t,
                            0..d_head,
                        ])),
                        super::super::read(k_r),
                    ),
                )
                .await;
                for (lbl, a, b) in [
                    ("q_no_norm", qa.unwrap(), qr.unwrap()),
                    ("k_no_norm", ka.unwrap(), kr.unwrap()),
                ] {
                    let e = a
                        .into_iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max);
                    assert!(e < tol, "{lbl} {dtype:?}: {e}");
                }
            }
        });
    }
}
