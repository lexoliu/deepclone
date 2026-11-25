//! Causal grouped attention directly over the live KV cache.
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension, cubecl::dtype_to_storage_type,
        ops::FloatTensorOps, tensor::FloatTensor, wgpu::CubeBackend,
    },
    cubecl::{ir::ElemType, prelude::*, tensor_vector_size_parallel},
    tensor::{DType, Shape, Tensor as BurnTensor, activation},
};

use super::gemv::lane_sum;

const SHORT_SCORE_CAP: usize = 256;
const LONG_SCORE_CAP: usize = 2048;
const MAX_SHARED_QUERY: usize = 256;
const SHORT_LANES: usize = 32;
const LONG_LANES: usize = 128;

#[backend_extension(Cube, Fusion)]
trait DecodeAttention: Backend {
    #[fusion(dtype = query, shape = Shape::from([1, query[2], query[1] * values[3]]))]
    fn causal_attention_fused(
        query: FloatTensor<Self>,
        keys: FloatTensor<Self>,
        values: FloatTensor<Self>,
        start_position: usize,
        window: usize,
    ) -> FloatTensor<Self>;

    #[fusion(dtype = query, shape = Shape::from([1, query[1], 1, *length]))]
    fn query_scores(
        query: FloatTensor<Self>,
        keys: FloatTensor<Self>,
        length: usize,
    ) -> FloatTensor<Self>;

    #[fusion(dtype = probabilities, shape = Shape::from([1, probabilities[1], 1, values[3]]))]
    fn weighted_values(
        probabilities: FloatTensor<Self>,
        values: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

pub(super) fn supports_fused(total_length: usize, width: usize) -> bool {
    total_length > 0 && total_length <= LONG_SCORE_CAP && width <= MAX_SHARED_QUERY
}

pub(super) fn decode(
    query: BurnTensor<4>,
    keys: BurnTensor<4>,
    values: BurnTensor<4>,
    length: usize,
) -> BurnTensor<3> {
    let [_, heads, _, width] = query.dims();
    if supports_fused(length, width) {
        return BurnTensor::from_dispatch(Dispatch::causal_attention_fused(
            query.into_dispatch(),
            keys.into_dispatch(),
            values.into_dispatch(),
            length - 1,
            0,
        ));
    }
    decode_unfused(query, keys, values, length).reshape([1, 1, heads * width])
}

pub(super) fn prefill(
    query: BurnTensor<4>,
    keys: BurnTensor<4>,
    values: BurnTensor<4>,
    start_position: usize,
) -> BurnTensor<3> {
    prefill_windowed(query, keys, values, start_position, 0)
}

pub(super) fn prefill_windowed(
    query: BurnTensor<4>,
    keys: BurnTensor<4>,
    values: BurnTensor<4>,
    start_position: usize,
    window: usize,
) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::causal_attention_fused(
        query.into_dispatch(),
        keys.into_dispatch(),
        values.into_dispatch(),
        start_position,
        window,
    ))
}

fn decode_unfused(
    query: BurnTensor<4>,
    keys: BurnTensor<4>,
    values: BurnTensor<4>,
    length: usize,
) -> BurnTensor<4> {
    let dtype = query.dtype();
    let heads = query.dims()[1];
    let kv_heads = keys.dims()[1];
    let scores = BurnTensor::<4>::from_dispatch(Dispatch::query_scores(
        query.into_dispatch(),
        keys.into_dispatch(),
        length,
    ));
    // Keep the established softmax reduction and rounding boundaries. Small
    // probability differences can change an entire autoregressive trajectory.
    let scores = scores.reshape([1, kv_heads, heads / kv_heads, 1, length]);
    let probabilities = activation::softmax(scores.cast(DType::F32), 4)
        .cast(dtype)
        .reshape([1, heads, 1, length]);
    BurnTensor::from_dispatch(Dispatch::weighted_values(
        probabilities.into_dispatch(),
        values.into_dispatch(),
    ))
}

impl DecodeAttention for CubeBackend {
    fn causal_attention_fused(
        query: FloatTensor<Self>,
        keys: FloatTensor<Self>,
        values: FloatTensor<Self>,
        start_position: usize,
        window: usize,
    ) -> FloatTensor<Self> {
        let heads = query.shape()[1];
        let tokens = query.shape()[2];
        let width = values.shape()[3];
        let total_length = start_position + tokens;
        let max_span = if window > 0 {
            total_length.min(window)
        } else {
            total_length
        };
        let output = Self::float_empty(
            Shape::from([1, tokens, heads * width]),
            &query.device,
            query.dtype.into(),
        );
        let score_cap = if max_span <= SHORT_SCORE_CAP {
            SHORT_SCORE_CAP
        } else {
            LONG_SCORE_CAP
        };
        let lanes = if max_span <= SHORT_SCORE_CAP {
            SHORT_LANES
        } else {
            LONG_LANES
        };
        let vec_w = if query.meta.strides()[3] == 1
            && keys.meta.strides()[3] == 1
            && values.meta.strides()[3] == 1
            && output.meta.strides()[2] == 1
        {
            let q_vec = tensor_vector_size_parallel(
                query.client.io_optimized_vector_sizes(query.dtype.size()),
                query.meta.shape(),
                query.meta.strides(),
                3,
            );
            let k_vec = tensor_vector_size_parallel(
                keys.client.io_optimized_vector_sizes(keys.dtype.size()),
                keys.meta.shape(),
                keys.meta.strides(),
                3,
            );
            let v_vec = tensor_vector_size_parallel(
                values.client.io_optimized_vector_sizes(values.dtype.size()),
                values.meta.shape(),
                values.meta.strides(),
                3,
            );
            let o_vec = tensor_vector_size_parallel(
                output.client.io_optimized_vector_sizes(output.dtype.size()),
                output.meta.shape(),
                output.meta.strides(),
                2,
            );
            q_vec.min(k_vec).min(v_vec).min(o_vec).min(4)
        } else {
            1
        };
        causal_attention_fused_kernel::launch(
            &query.client,
            CubeCount::Static(heads as u32, tokens as u32, 1),
            CubeDim::new_1d(lanes as u32),
            vec_w,
            query.clone().into_tensor_arg(),
            keys.into_tensor_arg(),
            values.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            start_position,
            window,
            score_cap,
            lanes,
            dtype_to_storage_type(query.dtype),
        );
        output
    }

    fn query_scores(
        query: FloatTensor<Self>,
        keys: FloatTensor<Self>,
        length: usize,
    ) -> FloatTensor<Self> {
        let shape = query.shape();
        let output = Self::float_empty(
            Shape::from([1, shape[1], 1, length]),
            &query.device,
            query.dtype.into(),
        );
        let dim = CubeDim::new(&query.client, length);
        query_scores_kernel::launch(
            &query.client,
            CubeCount::Static(shape[1] as u32, 1, 1),
            dim,
            query.clone().into_tensor_arg(),
            keys.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dtype_to_storage_type(query.dtype),
        );
        output
    }

    fn weighted_values(
        probabilities: FloatTensor<Self>,
        values: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let shape = probabilities.shape();
        let width = values.shape()[3];
        let output = Self::float_empty(
            Shape::from([1, shape[1], 1, width]),
            &probabilities.device,
            probabilities.dtype.into(),
        );
        let dim = CubeDim::new(&probabilities.client, width);
        weighted_values_kernel::launch(
            &probabilities.client,
            CubeCount::Static(shape[1] as u32, 1, 1),
            dim,
            probabilities.clone().into_tensor_arg(),
            values.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dtype_to_storage_type(probabilities.dtype),
        );
        output
    }
}

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn causal_attention_fused_kernel<F: Float, N: Size>(
    query: &Tensor<Vector<F, N>>,
    keys: &Tensor<Vector<F, N>>,
    values: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    start_position: usize,
    window: usize,
    #[comptime] score_cap: usize,
    #[comptime] lanes: usize,
    #[define(F)] _dtype: ElemType,
) {
    let head = CUBE_POS_X as usize;
    let q_row = CUBE_POS_Y as usize;
    let length = start_position + q_row + 1usize;
    let min_token = if window > 0usize && length > window {
        length - window
    } else {
        0usize
    };
    let span = length - min_token;
    let kv_head = head / (query.shape(1) / keys.shape(1));
    let vec_size = query.vector_size();
    let width_vecs = query.shape(3) / vec_size;
    let tid = UNIT_POS as usize;

    let mut scores = Shared::<[f32]>::new_slice(score_cap);
    let mut q_shared =
        Shared::<[Vector<f32, N>]>::new_slice(comptime!(MAX_SHARED_QUERY / vec_size));
    let mut scratch = Shared::<[f32]>::new_slice(lanes);

    let q_base = (head * query.stride(1) + q_row * query.stride(2)) / vec_size;
    let q_col_stride = query.stride(3);
    let mut c = tid;
    while c < width_vecs {
        let qv = if comptime!(vec_size > 1) {
            query[q_base + c]
        } else {
            query[q_base + c * q_col_stride]
        };
        q_shared[c] = Vector::<f32, N>::cast_from(qv);
        c += lanes;
    }
    sync_cube();

    // Preserve the exact score rounding boundary before FP32 softmax.
    let k_base = (kv_head * keys.stride(1)) / vec_size;
    let k_tok_stride = keys.stride(2) / vec_size;
    let k_col_stride = keys.stride(3);
    let mut token = tid;
    while token < span {
        let dot = RuntimeCell::<f32>::new(0f32);
        let k_row = k_base + (min_token + token) * k_tok_stride;
        for column in 0..width_vecs {
            let k_elem = if comptime!(vec_size > 1) {
                keys[k_row + column]
            } else {
                keys[k_row + column * k_col_stride]
            };
            let kv = Vector::<f32, N>::cast_from(k_elem);
            dot.store(dot.read() + lane_sum(q_shared[column] * kv));
        }
        scores[token] = f32::cast_from(F::cast_from(dot.consume()));
        token += lanes;
    }
    sync_cube();

    let active = if span < lanes { span } else { lanes };
    if tid < active {
        let mut m = scores[tid];
        let mut tok = tid + lanes;
        while tok < span {
            let s = scores[tok];
            if s > m {
                m = s;
            }
            tok += lanes;
        }
        scratch[tid] = m;
    }
    sync_cube();

    #[unroll]
    for stage in 0..lanes.ilog2() as usize {
        let distance = comptime!(lanes >> (stage + 1));
        if tid < distance && tid + distance < active {
            let other = scratch[tid + distance];
            if other > scratch[tid] {
                scratch[tid] = other;
            }
        }
        sync_cube();
    }
    let max_val = scratch[0];
    sync_cube();

    let local_sum = RuntimeCell::<f32>::new(0f32);
    let mut tok = tid;
    while tok < span {
        let e = (scores[tok] - max_val).exp();
        scores[tok] = e;
        local_sum.store(local_sum.read() + e);
        tok += lanes;
    }
    scratch[tid] = local_sum.consume();
    sync_cube();

    #[unroll]
    for stage in 0..lanes.ilog2() as usize {
        let distance = comptime!(lanes >> (stage + 1));
        if tid < distance {
            scratch[tid] += scratch[tid + distance];
        }
        sync_cube();
    }
    let sum_val = scratch[0];

    // Preserve the exact probability cast rounding boundary before value mixing.
    let mut tok = tid;
    while tok < span {
        scores[tok] = f32::cast_from(F::cast_from(scores[tok] / sum_val));
        tok += lanes;
    }
    sync_cube();

    let v_base = (kv_head * values.stride(1)) / vec_size;
    let v_tok_stride = values.stride(2) / vec_size;
    let v_col_stride = values.stride(3);
    let out_base = (q_row * output.stride(1)) / vec_size + head * width_vecs;
    let mut column = tid;
    while column < width_vecs {
        let mixed = RuntimeCell::<Vector<f32, N>>::new(Vector::<f32, N>::new(0f32));
        for tok in 0..span {
            let prob = Vector::<f32, N>::new(scores[tok]);
            let v_elem = if comptime!(vec_size > 1) {
                values[v_base + (min_token + tok) * v_tok_stride + column]
            } else {
                values[v_base + (min_token + tok) * v_tok_stride + column * v_col_stride]
            };
            let v = Vector::<f32, N>::cast_from(v_elem);
            mixed.store(mixed.read() + prob * v);
        }
        output[out_base + column] = Vector::<F, N>::cast_from(mixed.consume());
        column += lanes;
    }
}

#[cube(launch)]
fn query_scores_kernel<F: Float>(
    query: &Tensor<F>,
    keys: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: ElemType,
) {
    let head = CUBE_POS;
    let kv_head = head / (query.shape(1) / keys.shape(1));
    let width = query.shape(3);
    let length = output.shape(3);
    let mut token = UNIT_POS as usize;
    while token < length {
        let dot = RuntimeCell::<f32>::new(0f32);
        for column in 0..width {
            dot.store(
                dot.read()
                    + f32::cast_from(query[head * query.stride(1) + column * query.stride(3)])
                        * f32::cast_from(
                            keys[kv_head * keys.stride(1)
                                + token * keys.stride(2)
                                + column * keys.stride(3)],
                        ),
            );
        }
        let offset = head * output.stride(1) + token;
        output[offset] = F::cast_from(dot.consume());
        token += CUBE_DIM as usize;
    }
}

#[cube(launch)]
fn weighted_values_kernel<F: Float>(
    probabilities: &Tensor<F>,
    values: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: ElemType,
) {
    let head = CUBE_POS;
    let kv_head = head / (probabilities.shape(1) / values.shape(1));
    let width = output.shape(3);
    let length = probabilities.shape(3);
    let mut column = UNIT_POS as usize;
    while column < width {
        let mixed = RuntimeCell::<f32>::new(0f32);
        for token in 0..length {
            mixed.store(
                mixed.read()
                    + f32::cast_from(probabilities[head * probabilities.stride(1) + token])
                        * f32::cast_from(
                            values[kv_head * values.stride(1)
                                + token * values.stride(2)
                                + column * values.stride(3)],
                        ),
            );
        }
        let offset = head * output.stride(1) + column;
        output[offset] = F::cast_from(mixed.consume());
        column += CUBE_DIM as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, TensorData, activation};

    fn unfused(
        query: BurnTensor<4>,
        keys: BurnTensor<4>,
        values: BurnTensor<4>,
        length: usize,
    ) -> BurnTensor<3> {
        let [_, heads, _, width] = query.dims();
        let kv_heads = keys.dims()[1];
        let selection = [0..1, 0..kv_heads, 0..length, 0..width];
        let q = query.reshape([1, kv_heads, heads / kv_heads, 1, width]);
        let k = keys
            .slice(selection.clone())
            .reshape([1, kv_heads, 1, length, width]);
        let v = values
            .slice(selection)
            .reshape([1, kv_heads, 1, length, width]);
        let dtype = q.dtype();
        activation::softmax(q.matmul(k.swap_dims(3, 4)).cast(DType::F32), 4)
            .cast(dtype)
            .matmul(v)
            .reshape([1, 1, heads * width])
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn decode_attention_retains_cache_order_across_growth_and_reset() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F16, DType::F32] {
                let mut keys = BurnTensor::<4>::zeros([1, 8, 256, 128], (&device, dtype));
                let mut values = keys.clone();
                let mut actual = Vec::new();
                let mut expected = Vec::new();
                for (step, length) in [1, 2, 15, 16, 17, 163, 179, 1, 5, 13, 163]
                    .into_iter()
                    .enumerate()
                {
                    let data = BurnTensor::<4>::from_data(
                        TensorData::new(
                            (0..8 * length * 128)
                                .map(|i| ((i * 31 + step * 19) as f32).sin())
                                .collect::<Vec<_>>(),
                            [1, 8, length, 128],
                        ),
                        (&device, dtype),
                    );
                    let selection = [0..1, 0..8, 0..length, 0..128];
                    keys.inplace(|cache| cache.slice_assign(selection.clone(), data.clone()));
                    values.inplace(|cache| cache.slice_assign(selection, data * 0.5));
                    let query = BurnTensor::<4>::from_data(
                        TensorData::new(
                            (0..16 * 128)
                                .map(|i| ((i * 47 + step * 7) as f32).cos() / 128f32.sqrt())
                                .collect::<Vec<_>>(),
                            [1, 16, 1, 128],
                        ),
                        (&device, dtype),
                    );
                    actual.push(decode(query.clone(), keys.clone(), values.clone(), length));
                    expected.push(unfused(query, keys.clone(), values.clone(), length));
                    device.flush().unwrap();
                }
                let (actual, expected) = futures_lite::future::zip(
                    super::super::read(BurnTensor::cat(actual, 1)),
                    super::super::read(BurnTensor::cat(expected, 1)),
                )
                .await;
                let maximum = actual
                    .unwrap()
                    .into_iter()
                    .zip(expected.unwrap())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    maximum < if dtype == DType::F16 { 0.002 } else { 1e-5 },
                    "{dtype:?}: {maximum}"
                );
            }
        });
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn decode_attention_matches_unfused_grouped_attention() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F16, DType::F32] {
                for (heads, kv_heads, width, length, capacity) in [
                    (4, 2, 16, 1, 7),
                    (16, 8, 128, 15, 32),
                    (16, 8, 128, 163, 256),
                    (4, 1, 64, 1027, 2048),
                    (2, 1, 16, 8193, 9000),
                ] {
                    let tensor = |shape: [usize; 4], salt: usize| {
                        let n = shape.iter().product::<usize>();
                        BurnTensor::<4>::from_data(
                            TensorData::new(
                                (0..n)
                                    .map(|i| (((i * salt + 7) % 113) as f32 - 56.) / 64.)
                                    .collect::<Vec<_>>(),
                                shape,
                            ),
                            (&device, dtype),
                        )
                    };
                    let query = tensor([1, heads, 1, width], 19) * (width as f32).sqrt().recip();
                    let keys = tensor([1, kv_heads, capacity, width], 31);
                    let values = tensor([1, kv_heads, capacity, width], 47);
                    let expected = unfused(query.clone(), keys.clone(), values.clone(), length);
                    let (actual, expected) = futures_lite::future::zip(
                        super::super::read(decode(query, keys, values, length)),
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
                    assert!(maximum < tolerance, "{dtype:?} length {length}: {maximum}");
                }
            }
        });
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn fused_decode_attention_matches_unfused_softmax_rounding() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F16, DType::F32] {
                for (heads, kv_heads, width, length, capacity) in [
                    (4, 2, 16, 1, 8),
                    (16, 8, 128, 3, 17),
                    (16, 8, 128, 16, 17),
                    (16, 8, 128, 256, 256),
                    (16, 8, 128, 257, 512),
                    (16, 8, 128, 1027, 2048),
                ] {
                    let tensor = |shape: [usize; 4], salt: usize| {
                        let n = shape.iter().product::<usize>();
                        BurnTensor::<4>::from_data(
                            TensorData::new(
                                (0..n)
                                    .map(|i| (((i * salt + 7) % 113) as f32 - 56.) / 64.)
                                    .collect::<Vec<_>>(),
                                shape,
                            ),
                            (&device, dtype),
                        )
                    };
                    let query = tensor([1, heads, 1, width], 19) * (width as f32).sqrt().recip();
                    let keys = tensor([1, kv_heads, capacity, width], 31);
                    let values = tensor([1, kv_heads, capacity, width], 47);
                    let expected =
                        decode_unfused(query.clone(), keys.clone(), values.clone(), length)
                            .reshape([1, 1, heads * width]);
                    let (actual, expected) = futures_lite::future::zip(
                        super::super::read(decode(query, keys, values, length)),
                        super::super::read(expected),
                    )
                    .await;
                    let maximum = actual
                        .unwrap()
                        .into_iter()
                        .zip(expected.unwrap())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let tolerance = if dtype == DType::F16 { 1e-3 } else { 1e-6 };
                    assert!(
                        maximum < tolerance,
                        "fused vs unfused {dtype:?} length {length}: {maximum}"
                    );
                }
            }
        });
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn fused_prefill_causal_attention_matches_unfused_reference() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F16, DType::F32] {
                for (heads, kv_heads, width, start_pos, tokens, capacity, window) in [
                    (4, 2, 16, 0, 7, 16, 0usize),
                    (16, 8, 128, 0, 19, 32, 0),
                    (16, 8, 128, 12, 19, 64, 0),
                    (16, 8, 128, 0, 151, 256, 0),
                    (16, 8, 128, 120, 140, 512, 0),
                    (8, 8, 64, 151, 12, 256, 64),
                    (8, 2, 64, 245, 12, 300, 250),
                ] {
                    let tensor = |shape: [usize; 4], salt: usize| {
                        let n = shape.iter().product::<usize>();
                        BurnTensor::<4>::from_data(
                            TensorData::new(
                                (0..n)
                                    .map(|i| (((i * salt + 7) % 113) as f32 - 56.) / 64.)
                                    .collect::<Vec<_>>(),
                                shape,
                            ),
                            (&device, dtype),
                        )
                    };
                    let total = start_pos + tokens;
                    let query =
                        tensor([1, heads, tokens, width], 19) * (width as f32).sqrt().recip();
                    let keys = tensor([1, kv_heads, capacity, width], 31);
                    let values = tensor([1, kv_heads, capacity, width], 47);

                    let queries = BurnTensor::<1, burn::tensor::Int>::arange(
                        start_pos as i64..total as i64,
                        &device,
                    )
                    .reshape([tokens, 1]);
                    let k_idx =
                        BurnTensor::<1, burn::tensor::Int>::arange(0..total as i64, &device)
                            .reshape([1, total]);
                    let distance = queries - k_idx;
                    let mut mask = distance.clone().lower_elem(0);
                    if window > 0 {
                        mask = mask.bool_or(distance.greater_equal_elem(window as i64));
                    }
                    let mask = mask.reshape([1, 1, 1, tokens, total]);
                    let groups = heads / kv_heads;
                    let sel = [0..1, 0..kv_heads, 0..total, 0..width];
                    let k_view = keys
                        .clone()
                        .slice(sel.clone())
                        .reshape([1, kv_heads, 1, total, width]);
                    let v_view = values
                        .clone()
                        .slice(sel)
                        .reshape([1, kv_heads, 1, total, width]);
                    let scores = query
                        .clone()
                        .reshape([1, kv_heads, groups, tokens, width])
                        .matmul(k_view.swap_dims(3, 4))
                        .cast(DType::F32)
                        .mask_fill(mask, f32::NEG_INFINITY);
                    let expected = activation::softmax(scores, 4)
                        .cast(dtype)
                        .matmul(v_view)
                        .reshape([1, heads, tokens, width])
                        .swap_dims(1, 2)
                        .reshape([1, tokens, heads * width]);

                    let (actual, expected) = futures_lite::future::zip(
                        super::super::read(prefill_windowed(
                            query, keys, values, start_pos, window,
                        )),
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
                        "prefill {dtype:?} start={start_pos} tokens={tokens} window={window}: {maximum}"
                    );
                }
            }
        });
    }
}
