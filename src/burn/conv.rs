//! Channel-first 1D causal convolutions, transposed convolutions, and Snake activation.
use super::fused::{into_contiguous, into_mut_contiguous, row_grid};
use burn::cubecl;
use burn::{
    backend::{
        Backend, Dispatch, TensorMetadata, backend_extension, cubecl::dtype_to_storage_type,
        ops::FloatTensorOps, tensor::FloatTensor, wgpu::CubeBackend,
    },
    cubecl::{ir::ElemType, prelude::*},
    tensor::{Shape, Tensor as BurnTensor},
};

pub(super) const PAD_CAUSAL_ZERO: usize = 0;
pub(super) const PAD_CAUSAL_EDGE: usize = 1;
pub(super) const PAD_REFLECT_SAME: usize = 2;

#[backend_extension(Cube, Fusion)]
trait CodecConv: Backend {
    #[fusion(dtype = input, shape = input.clone())]
    fn contiguous(input: FloatTensor<Self>) -> FloatTensor<Self>;

    #[fusion(dtype = x, shape = x.clone())]
    fn snake(
        x: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        inverse_beta: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    #[allow(clippy::too_many_arguments)]
    #[fusion(dtype = x, shape = Shape::from([x[0], weight[1], x[2].div_ceil(*stride)]))]
    fn causal_conv1d(
        x: FloatTensor<Self>,
        history: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        stride: usize,
        dilation: usize,
        pad_mode: usize,
        has_history: bool,
        has_bias: bool,
        has_residual: bool,
    ) -> FloatTensor<Self>;

    #[allow(clippy::too_many_arguments)]
    #[fusion(dtype = x, shape = x.clone())]
    fn depthwise_causal_conv1d(
        x: FloatTensor<Self>,
        history: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        dilation: usize,
        has_history: bool,
        has_bias: bool,
    ) -> FloatTensor<Self>;

    #[allow(clippy::too_many_arguments)]
    #[fusion(dtype = x, shape = Shape::from([x[0], weight[1], x[2] * *stride]))]
    fn causal_conv_transpose1d(
        x: FloatTensor<Self>,
        history: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        stride: usize,
        has_history: bool,
        has_bias: bool,
    ) -> FloatTensor<Self>;
}

pub(super) fn contiguous<const D: usize>(input: BurnTensor<D>) -> BurnTensor<D> {
    BurnTensor::from_dispatch(Dispatch::contiguous(input.into_dispatch()))
}

pub(super) fn snake(
    x: BurnTensor<3>,
    alpha: BurnTensor<1>,
    inverse_beta: BurnTensor<1>,
) -> BurnTensor<3> {
    BurnTensor::from_dispatch(Dispatch::snake(
        x.into_dispatch(),
        alpha.into_dispatch(),
        inverse_beta.into_dispatch(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn conv1d(
    x: BurnTensor<3>,
    history: Option<BurnTensor<3>>,
    weight: BurnTensor<3>,
    bias: Option<BurnTensor<1>>,
    stride: usize,
    dilation: usize,
    pad_mode: usize,
) -> BurnTensor<3> {
    conv1d_inner(x, history, weight, bias, stride, dilation, pad_mode, None)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn conv1d_add(
    x: BurnTensor<3>,
    history: Option<BurnTensor<3>>,
    weight: BurnTensor<3>,
    bias: Option<BurnTensor<1>>,
    stride: usize,
    dilation: usize,
    pad_mode: usize,
    residual: BurnTensor<3>,
) -> BurnTensor<3> {
    conv1d_inner(
        x,
        history,
        weight,
        bias,
        stride,
        dilation,
        pad_mode,
        Some(residual),
    )
}

#[allow(clippy::too_many_arguments)]
fn conv1d_inner(
    x: BurnTensor<3>,
    history: Option<BurnTensor<3>>,
    weight: BurnTensor<3>,
    bias: Option<BurnTensor<1>>,
    stride: usize,
    dilation: usize,
    pad_mode: usize,
    residual: Option<BurnTensor<3>>,
) -> BurnTensor<3> {
    let has_history = history.is_some();
    let history = history.unwrap_or_else(|| x.clone());
    let has_bias = bias.is_some();
    let bias_dispatch = match bias {
        Some(bias) => bias.into_dispatch(),
        None => x.clone().into_dispatch(),
    };
    let has_residual = residual.is_some();
    let residual_dispatch = match residual {
        Some(residual) => residual.into_dispatch(),
        None => x.clone().into_dispatch(),
    };
    BurnTensor::from_dispatch(Dispatch::causal_conv1d(
        x.into_dispatch(),
        history.into_dispatch(),
        weight.into_dispatch(),
        bias_dispatch,
        residual_dispatch,
        stride,
        dilation,
        pad_mode,
        has_history,
        has_bias,
        has_residual,
    ))
}

pub(super) fn depthwise_causal_conv1d(
    x: BurnTensor<3>,
    history: Option<BurnTensor<3>>,
    weight: BurnTensor<2>,
    bias: Option<BurnTensor<1>>,
    dilation: usize,
) -> BurnTensor<3> {
    let has_history = history.is_some();
    let history = history.unwrap_or_else(|| x.clone());
    let has_bias = bias.is_some();
    let bias_dispatch = match bias {
        Some(bias) => bias.into_dispatch(),
        None => x.clone().into_dispatch(),
    };
    BurnTensor::from_dispatch(Dispatch::depthwise_causal_conv1d(
        x.into_dispatch(),
        history.into_dispatch(),
        weight.into_dispatch(),
        bias_dispatch,
        dilation,
        has_history,
        has_bias,
    ))
}

pub(super) fn causal_conv_transpose1d(
    x: BurnTensor<3>,
    history: Option<BurnTensor<3>>,
    weight: BurnTensor<3>,
    bias: Option<BurnTensor<1>>,
    stride: usize,
) -> BurnTensor<3> {
    let has_history = history.is_some();
    let history = history.unwrap_or_else(|| x.clone());
    let has_bias = bias.is_some();
    let bias_dispatch = match bias {
        Some(bias) => bias.into_dispatch(),
        None => x.clone().into_dispatch(),
    };
    BurnTensor::from_dispatch(Dispatch::causal_conv_transpose1d(
        x.into_dispatch(),
        history.into_dispatch(),
        weight.into_dispatch(),
        bias_dispatch,
        stride,
        has_history,
        has_bias,
    ))
}

impl CodecConv for CubeBackend {
    fn contiguous(input: FloatTensor<Self>) -> FloatTensor<Self> {
        if !input.is_contiguous() {
            into_contiguous(input)
        } else if input.handle.offset_start.is_some() || input.handle.offset_end.is_some() {
            // Detach sub-slice views so retaining a small tail (e.g. ConvHistory) cannot
            // pin a much larger parent activation buffer in GPU memory.
            input.copy()
        } else {
            input
        }
    }

    fn snake(
        x: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        inverse_beta: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let alpha = into_contiguous(alpha);
        let inverse_beta = into_contiguous(inverse_beta);
        let can_mutate = x.can_mut() && x.is_contiguous() && x.is_nonoverlapping();
        let x = if can_mutate {
            into_mut_contiguous(x)
        } else {
            into_contiguous(x)
        };
        let shape = x.shape();
        let [batch, channels, time] = [shape[0], shape[1], shape[2]];
        let par_vec = cubecl::tensor_vector_size_parallel(
            x.client.io_optimized_vector_sizes(x.dtype.size()),
            x.meta.shape(),
            x.meta.strides(),
            2,
        );
        let vec_t = if par_vec >= 4 && time.is_multiple_of(4) {
            4
        } else {
            1
        };
        let t_vecs = time / vec_t;
        let max_count = x.client.properties().hardware.max_cube_count;
        if t_vecs >= 64 && channels <= max_count.1 as usize && batch <= max_count.2 as usize {
            let dim = CubeDim::new_1d(t_vecs.next_power_of_two().min(256) as u32);
            let t_groups = t_vecs
                .div_ceil(dim.num_elems() as usize)
                .max(1)
                .min(max_count.0 as usize);
            let count = CubeCount::Static(t_groups as u32, channels as u32, batch as u32);
            if can_mutate {
                snake_inplace_kernel::launch(
                    &x.client,
                    count,
                    dim,
                    vec_t,
                    x.clone().into_tensor_arg(),
                    alpha.into_tensor_arg(),
                    inverse_beta.into_tensor_arg(),
                    channels,
                    t_vecs,
                    dtype_to_storage_type(x.dtype),
                );
                return x;
            }
            let output = Self::float_empty(shape, &x.device, x.dtype.into());
            snake_out_kernel::launch(
                &x.client,
                count,
                dim,
                vec_t,
                x.clone().into_tensor_arg(),
                alpha.into_tensor_arg(),
                inverse_beta.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                channels,
                t_vecs,
                dtype_to_storage_type(x.dtype),
            );
            return output;
        }
        let total_vecs = batch * channels * t_vecs;
        let dim = CubeDim::new(&x.client, total_vecs);
        let groups = total_vecs.div_ceil(dim.num_elems() as usize);
        if can_mutate {
            snake_inplace_1d_kernel::launch(
                &x.client,
                row_grid(groups, max_count.0),
                dim,
                vec_t,
                x.clone().into_tensor_arg(),
                alpha.into_tensor_arg(),
                inverse_beta.into_tensor_arg(),
                total_vecs,
                channels,
                t_vecs,
                dtype_to_storage_type(x.dtype),
            );
            x
        } else {
            let output = Self::float_empty(shape, &x.device, x.dtype.into());
            snake_out_1d_kernel::launch(
                &x.client,
                row_grid(groups, max_count.0),
                dim,
                vec_t,
                x.clone().into_tensor_arg(),
                alpha.into_tensor_arg(),
                inverse_beta.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                total_vecs,
                channels,
                t_vecs,
                dtype_to_storage_type(x.dtype),
            );
            output
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn causal_conv1d(
        x: FloatTensor<Self>,
        history: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        residual: FloatTensor<Self>,
        stride: usize,
        dilation: usize,
        pad_mode: usize,
        has_history: bool,
        has_bias: bool,
        has_residual: bool,
    ) -> FloatTensor<Self> {
        let x = into_contiguous(x);
        let history = if has_history {
            into_contiguous(history)
        } else {
            x.clone()
        };
        let weight = into_contiguous(weight);
        let bias = if has_bias {
            into_contiguous(bias)
        } else {
            x.clone()
        };
        let residual = if has_residual {
            into_contiguous(residual)
        } else {
            x.clone()
        };
        let [batch, c_in, t_in] = [x.shape()[0], x.shape()[1], x.shape()[2]];
        let [kernel, c_out, _] = [weight.shape()[0], weight.shape()[1], weight.shape()[2]];
        let t_out = t_in.div_ceil(stride);
        let hist_len = if has_history { history.shape()[2] } else { 0 };
        let eff_k = (kernel - 1) * dilation + 1;
        let left_pad = if pad_mode == PAD_REFLECT_SAME {
            (eff_k - 1) / 2
        } else {
            eff_k - stride
        };
        let output = Self::float_empty(
            Shape::from([batch, c_out, t_out]),
            &x.device,
            x.dtype.into(),
        );
        if c_out <= 4 {
            let total = batch * c_out * t_out;
            let dim = CubeDim::new(&x.client, total);
            let groups = total.div_ceil(dim.num_elems() as usize);
            let max_x = x.client.properties().hardware.max_cube_count.0;
            let weight_elems = kernel * c_out * c_in;
            let cache_weights = if weight_elems <= 1024 {
                weight_elems
            } else {
                0
            };
            causal_conv1d_small_out_kernel::launch(
                &x.client,
                row_grid(groups, max_x),
                dim,
                x.clone().into_tensor_arg(),
                history.into_tensor_arg(),
                weight.into_tensor_arg(),
                bias.into_tensor_arg(),
                residual.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                total,
                c_in,
                c_out,
                t_in,
                t_out,
                hist_len,
                kernel,
                stride,
                dilation,
                left_pad,
                stride == 1,
                pad_mode,
                has_bias,
                has_residual,
                cache_weights,
                dtype_to_storage_type(x.dtype),
            );
        } else {
            causal_conv1d_tiled_kernel::launch(
                &x.client,
                CubeCount::Static(
                    t_out.div_ceil(32) as u32,
                    c_out.div_ceil(32) as u32,
                    batch as u32,
                ),
                CubeDim::new_2d(16, 16),
                x.clone().into_tensor_arg(),
                history.into_tensor_arg(),
                weight.into_tensor_arg(),
                bias.into_tensor_arg(),
                residual.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                c_in,
                c_out,
                t_in,
                t_out,
                hist_len,
                kernel,
                stride,
                dilation,
                left_pad,
                stride == 1,
                pad_mode,
                has_bias,
                has_residual,
                dtype_to_storage_type(x.dtype),
            );
        }
        output
    }

    fn depthwise_causal_conv1d(
        x: FloatTensor<Self>,
        history: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        dilation: usize,
        has_history: bool,
        has_bias: bool,
    ) -> FloatTensor<Self> {
        let x = into_contiguous(x);
        let history = if has_history {
            into_contiguous(history)
        } else {
            x.clone()
        };
        let weight = into_contiguous(weight);
        let bias = if has_bias {
            into_contiguous(bias)
        } else {
            x.clone()
        };
        let shape = x.shape();
        let [batch, channels, t_out] = [shape[0], shape[1], shape[2]];
        let kernel = weight.shape()[1];
        let hist_len = if has_history { history.shape()[2] } else { 0 };
        let bias_flag = u32::from(has_bias);
        let output = Self::float_empty(shape, &x.device, x.dtype.into());
        let total = batch * channels * t_out;
        let dim = CubeDim::new(&x.client, total);
        let groups = total.div_ceil(dim.num_elems() as usize);
        let max_x = x.client.properties().hardware.max_cube_count.0;
        depthwise_causal_conv1d_kernel::launch(
            &x.client,
            row_grid(groups, max_x),
            dim,
            x.clone().into_tensor_arg(),
            history.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            total,
            channels,
            t_out,
            hist_len,
            kernel,
            dilation,
            bias_flag,
            dtype_to_storage_type(x.dtype),
        );
        output
    }

    fn causal_conv_transpose1d(
        x: FloatTensor<Self>,
        history: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        stride: usize,
        has_history: bool,
        has_bias: bool,
    ) -> FloatTensor<Self> {
        let x = into_contiguous(x);
        let history = if has_history {
            into_contiguous(history)
        } else {
            x.clone()
        };
        let weight = into_contiguous(weight);
        let bias = if has_bias {
            into_contiguous(bias)
        } else {
            x.clone()
        };
        let [batch, c_in, t_in] = [x.shape()[0], x.shape()[1], x.shape()[2]];
        let [kernel, c_out, _] = [weight.shape()[0], weight.shape()[1], weight.shape()[2]];
        let hist_len = if has_history { history.shape()[2] } else { 0 };
        let output = Self::float_empty(
            Shape::from([batch, c_out, t_in * stride]),
            &x.device,
            x.dtype.into(),
        );
        causal_conv_transpose1d_tiled_kernel::launch(
            &x.client,
            CubeCount::Static(
                t_in.div_ceil(32) as u32,
                c_out.div_ceil(32) as u32,
                (batch * stride) as u32,
            ),
            CubeDim::new_2d(16, 16),
            x.clone().into_tensor_arg(),
            history.into_tensor_arg(),
            weight.into_tensor_arg(),
            bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            c_in,
            c_out,
            t_in,
            hist_len,
            kernel,
            stride,
            has_bias,
            dtype_to_storage_type(x.dtype),
        );
        output
    }
}

#[cube]
fn snake_eval<N: Size>(v: Vector<f32, N>, alpha: f32, inv_beta: f32) -> Vector<f32, N> {
    let a = Vector::<f32, N>::new(alpha);
    let ib = Vector::<f32, N>::new(inv_beta);
    let periodic = (v * a).sin();
    v + periodic * periodic * ib
}

#[cube(launch)]
fn snake_inplace_kernel<F: Float, N: Size>(
    x: &mut Tensor<Vector<F, N>>,
    alpha: &Tensor<F>,
    inverse_beta: &Tensor<F>,
    channels: usize,
    t_vecs: usize,
    #[define(F)] _dtype: ElemType,
) {
    let c = CUBE_POS_Y as usize;
    let row = ((CUBE_POS_Z as usize) * channels + c) * t_vecs;
    let a = f32::cast_from(alpha[c]);
    let ib = f32::cast_from(inverse_beta[c]);
    let grid_stride = (CUBE_COUNT_X as usize) * (CUBE_DIM as usize);
    let mut tv = (CUBE_POS_X as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    while tv < t_vecs {
        let idx = row + tv;
        let v = Vector::<f32, N>::cast_from(x[idx]);
        x[idx] = Vector::<F, N>::cast_from(snake_eval(v, a, ib));
        tv += grid_stride;
    }
}

#[cube(launch)]
fn snake_out_kernel<F: Float, N: Size>(
    x: &Tensor<Vector<F, N>>,
    alpha: &Tensor<F>,
    inverse_beta: &Tensor<F>,
    output: &mut Tensor<Vector<F, N>>,
    channels: usize,
    t_vecs: usize,
    #[define(F)] _dtype: ElemType,
) {
    let c = CUBE_POS_Y as usize;
    let row = ((CUBE_POS_Z as usize) * channels + c) * t_vecs;
    let a = f32::cast_from(alpha[c]);
    let ib = f32::cast_from(inverse_beta[c]);
    let grid_stride = (CUBE_COUNT_X as usize) * (CUBE_DIM as usize);
    let mut tv = (CUBE_POS_X as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    while tv < t_vecs {
        let idx = row + tv;
        let v = Vector::<f32, N>::cast_from(x[idx]);
        output[idx] = Vector::<F, N>::cast_from(snake_eval(v, a, ib));
        tv += grid_stride;
    }
}

#[cube(launch)]
fn snake_inplace_1d_kernel<F: Float, N: Size>(
    x: &mut Tensor<Vector<F, N>>,
    alpha: &Tensor<F>,
    inverse_beta: &Tensor<F>,
    total_vecs: usize,
    channels: usize,
    t_vecs: usize,
    #[define(F)] _dtype: ElemType,
) {
    let cube = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    let idx = cube * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if idx >= total_vecs {
        terminate!();
    }
    let bc = idx / t_vecs;
    let c = bc - (bc / channels) * channels;
    let v = Vector::<f32, N>::cast_from(x[idx]);
    x[idx] = Vector::<F, N>::cast_from(snake_eval(
        v,
        f32::cast_from(alpha[c]),
        f32::cast_from(inverse_beta[c]),
    ));
}

#[cube(launch)]
fn snake_out_1d_kernel<F: Float, N: Size>(
    x: &Tensor<Vector<F, N>>,
    alpha: &Tensor<F>,
    inverse_beta: &Tensor<F>,
    output: &mut Tensor<Vector<F, N>>,
    total_vecs: usize,
    channels: usize,
    t_vecs: usize,
    #[define(F)] _dtype: ElemType,
) {
    let cube = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    let idx = cube * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if idx >= total_vecs {
        terminate!();
    }
    let bc = idx / t_vecs;
    let c = bc - (bc / channels) * channels;
    let v = Vector::<f32, N>::cast_from(x[idx]);
    output[idx] = Vector::<F, N>::cast_from(snake_eval(
        v,
        f32::cast_from(alpha[c]),
        f32::cast_from(inverse_beta[c]),
    ));
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn load_conv_sample<F: Float>(
    x: &Tensor<F>,
    history: &Tensor<F>,
    x_row: usize,
    h_row: usize,
    t_in: usize,
    hist_len: usize,
    t: usize,
    tap_offset: usize,
    left_pad: usize,
    stride: usize,
    #[comptime] pad_mode: usize,
) -> f32 {
    let pos = t * stride + tap_offset;
    let mut val = 0f32;
    if pos < left_pad {
        let deficit = left_pad - pos;
        if comptime!(pad_mode == PAD_CAUSAL_ZERO) {
            if hist_len > 0usize && deficit <= hist_len {
                val = f32::cast_from(history[h_row + (hist_len - deficit)]);
            }
        } else if comptime!(pad_mode == PAD_CAUSAL_EDGE) {
            val = f32::cast_from(x[x_row]);
        } else {
            val = f32::cast_from(x[x_row + deficit]);
        }
    } else {
        let raw = pos - left_pad;
        if raw < t_in {
            val = f32::cast_from(x[x_row + raw]);
        } else if comptime!(pad_mode == PAD_CAUSAL_EDGE) {
            val = f32::cast_from(x[x_row + (t_in - 1usize)]);
        } else if comptime!(pad_mode == PAD_REFLECT_SAME) {
            let excess = raw - (t_in - 1usize);
            val = f32::cast_from(x[x_row + (t_in - 1usize - excess)]);
        }
    }
    val
}

/// Register-tiled 32 × 32 (2 × 2 outputs/thread) 1D convolution over contiguous `[B, C, T]`
/// tensors and packed `[K, C_out, C_in]` weights. Uses 4 KB shared memory per 16 × 16 workgroup
/// with a workgroup-uniform interior fast path that skips padding and bounds checks.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn causal_conv1d_tiled_kernel<F: Float>(
    x: &Tensor<F>,
    history: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    residual: &Tensor<F>,
    output: &mut Tensor<F>,
    c_in: usize,
    c_out: usize,
    t_in: usize,
    t_out: usize,
    hist_len: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    left_pad: usize,
    #[comptime] unit_stride: bool,
    #[comptime] pad_mode: usize,
    #[comptime] has_bias: bool,
    #[comptime] has_residual: bool,
    #[define(F)] _dtype: ElemType,
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let t_base = (CUBE_POS_X as usize) * 32usize;
    let oc_base = (CUBE_POS_Y as usize) * 32usize;
    let batch = CUBE_POS_Z as usize;

    let mut w_tile = Shared::<[f32]>::new_slice(512usize);
    let mut x_tile = Shared::<[f32]>::new_slice(512usize);

    let acc00 = RuntimeCell::<f32>::new(0f32);
    let acc01 = RuntimeCell::<f32>::new(0f32);
    let acc10 = RuntimeCell::<f32>::new(0f32);
    let acc11 = RuntimeCell::<f32>::new(0f32);

    let ic_blocks = c_in.div_ceil(16usize);
    let full_oc = oc_base + 32usize <= c_out;
    let oc0 = oc_base + ty;
    let oc1 = oc0 + 16usize;
    let t0 = t_base + tx;
    let t1 = t0 + 16usize;

    for tap in 0..kernel {
        let tap_offset = tap * dilation;
        let w_tap_base = tap * c_out;
        let pos_min = t_base * stride + tap_offset;
        let pos_max = (t_base + 31usize) * stride + tap_offset;
        let mut x_interior = false;
        if t_base + 32usize <= t_out && pos_min >= left_pad {
            let raw_max = pos_max - left_pad;
            x_interior = raw_max < t_in;
        }

        for ic_block in 0..ic_blocks {
            let ic_base = ic_block * 16usize;
            let full_ic = ic_base + 16usize <= c_in;
            let ic_w = ic_base + tx;

            if full_oc && full_ic {
                w_tile[ty * 16usize + tx] =
                    f32::cast_from(weight[(w_tap_base + oc0) * c_in + ic_w]);
                w_tile[(ty + 16usize) * 16usize + tx] =
                    f32::cast_from(weight[(w_tap_base + oc1) * c_in + ic_w]);
            } else {
                let mut w0 = 0f32;
                let mut w1 = 0f32;
                if ic_w < c_in {
                    if oc0 < c_out {
                        w0 = f32::cast_from(weight[(w_tap_base + oc0) * c_in + ic_w]);
                    }
                    if oc1 < c_out {
                        w1 = f32::cast_from(weight[(w_tap_base + oc1) * c_in + ic_w]);
                    }
                }
                w_tile[ty * 16usize + tx] = w0;
                w_tile[(ty + 16usize) * 16usize + tx] = w1;
            }

            let ic_x = ic_base + ty;
            let x_row = (batch * c_in + ic_x) * t_in;
            if x_interior && full_ic {
                let src_base = pos_min - left_pad;
                if comptime!(unit_stride) {
                    x_tile[ty * 32usize + tx] = f32::cast_from(x[x_row + src_base + tx]);
                    x_tile[ty * 32usize + tx + 16usize] =
                        f32::cast_from(x[x_row + src_base + tx + 16usize]);
                } else {
                    let s0 = src_base + tx * stride;
                    x_tile[ty * 32usize + tx] = f32::cast_from(x[x_row + s0]);
                    x_tile[ty * 32usize + tx + 16usize] =
                        f32::cast_from(x[x_row + s0 + 16usize * stride]);
                }
            } else {
                let h_row = (batch * c_in + ic_x) * hist_len;
                let mut v0 = 0f32;
                let mut v1 = 0f32;
                if ic_x < c_in {
                    if t0 < t_out {
                        v0 = load_conv_sample(
                            x, history, x_row, h_row, t_in, hist_len, t0, tap_offset, left_pad,
                            stride, pad_mode,
                        );
                    }
                    if t1 < t_out {
                        v1 = load_conv_sample(
                            x, history, x_row, h_row, t_in, hist_len, t1, tap_offset, left_pad,
                            stride, pad_mode,
                        );
                    }
                }
                x_tile[ty * 32usize + tx] = v0;
                x_tile[ty * 32usize + tx + 16usize] = v1;
            }

            sync_cube();
            #[unroll]
            for k in 0..16usize {
                let wk0 = w_tile[ty * 16usize + k];
                let wk1 = w_tile[(ty + 16usize) * 16usize + k];
                let xk0 = x_tile[k * 32usize + tx];
                let xk1 = x_tile[k * 32usize + tx + 16usize];
                acc00.store(acc00.read() + wk0 * xk0);
                acc01.store(acc01.read() + wk0 * xk1);
                acc10.store(acc10.read() + wk1 * xk0);
                acc11.store(acc11.read() + wk1 * xk1);
            }
            sync_cube();
        }
    }

    if oc0 < c_out {
        let mut b0 = 0f32;
        if comptime!(has_bias) {
            b0 = f32::cast_from(bias[oc0]);
        }
        let out_row = (batch * c_out + oc0) * t_out;
        if t0 < t_out {
            let v = if comptime!(has_residual) {
                acc00.consume() + b0 + f32::cast_from(residual[out_row + t0])
            } else {
                acc00.consume() + b0
            };
            output[out_row + t0] = F::cast_from(v);
        }
        if t1 < t_out {
            let v = if comptime!(has_residual) {
                acc01.consume() + b0 + f32::cast_from(residual[out_row + t1])
            } else {
                acc01.consume() + b0
            };
            output[out_row + t1] = F::cast_from(v);
        }
    }
    if oc1 < c_out {
        let mut b1 = 0f32;
        if comptime!(has_bias) {
            b1 = f32::cast_from(bias[oc1]);
        }
        let out_row = (batch * c_out + oc1) * t_out;
        if t0 < t_out {
            let v = if comptime!(has_residual) {
                acc10.consume() + b1 + f32::cast_from(residual[out_row + t0])
            } else {
                acc10.consume() + b1
            };
            output[out_row + t0] = F::cast_from(v);
        }
        if t1 < t_out {
            let v = if comptime!(has_residual) {
                acc11.consume() + b1 + f32::cast_from(residual[out_row + t1])
            } else {
                acc11.consume() + b1
            };
            output[out_row + t1] = F::cast_from(v);
        }
    }
}

/// Time-parallel 1D convolution for narrow outputs such as the final mono waveform head.
/// Caches small packed weight tables once per workgroup in shared memory.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn causal_conv1d_small_out_kernel<F: Float>(
    x: &Tensor<F>,
    history: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    residual: &Tensor<F>,
    output: &mut Tensor<F>,
    total: usize,
    c_in: usize,
    c_out: usize,
    t_in: usize,
    t_out: usize,
    hist_len: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    left_pad: usize,
    #[comptime] unit_stride: bool,
    #[comptime] pad_mode: usize,
    #[comptime] has_bias: bool,
    #[comptime] has_residual: bool,
    #[comptime] cache_weights: usize,
    #[define(F)] _dtype: ElemType,
) {
    let mut w_shared = Shared::<[f32]>::new_slice(comptime!(if cache_weights > 0 {
        cache_weights
    } else {
        1usize
    }));
    if comptime!(cache_weights > 0) {
        let mut wi = UNIT_POS as usize;
        while wi < comptime!(cache_weights) {
            w_shared[wi] = f32::cast_from(weight[wi]);
            wi += CUBE_DIM as usize;
        }
        sync_cube();
    }

    let cube = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    let idx = cube * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if idx >= total {
        terminate!();
    }
    let t = idx % t_out;
    let bo = idx / t_out;
    let oc = bo % c_out;
    let batch = bo / c_out;

    let mut init = 0f32;
    if comptime!(has_bias) {
        init = f32::cast_from(bias[oc]);
    }
    let acc = RuntimeCell::<f32>::new(init);
    for tap in 0..kernel {
        let tap_offset = tap * dilation;
        let w_base = (tap * c_out + oc) * c_in;
        let pos = t * stride + tap_offset;
        if comptime!(unit_stride && pad_mode == PAD_CAUSAL_ZERO) && pos >= left_pad {
            let src_t = pos - left_pad;
            for ic in 0..c_in {
                let xv = f32::cast_from(x[(batch * c_in + ic) * t_in + src_t]);
                let wv = if comptime!(cache_weights > 0) {
                    w_shared[w_base + ic]
                } else {
                    f32::cast_from(weight[w_base + ic])
                };
                acc.store(acc.read() + xv * wv);
            }
        } else {
            for ic in 0..c_in {
                let x_row = (batch * c_in + ic) * t_in;
                let h_row = (batch * c_in + ic) * hist_len;
                let xv = load_conv_sample(
                    x, history, x_row, h_row, t_in, hist_len, t, tap_offset, left_pad, stride,
                    pad_mode,
                );
                let wv = if comptime!(cache_weights > 0) {
                    w_shared[w_base + ic]
                } else {
                    f32::cast_from(weight[w_base + ic])
                };
                acc.store(acc.read() + xv * wv);
            }
        }
    }
    let sum = if comptime!(has_residual) {
        acc.consume() + f32::cast_from(residual[idx])
    } else {
        acc.consume()
    };
    output[idx] = F::cast_from(sum);
}

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn depthwise_causal_conv1d_kernel<F: Float>(
    x: &Tensor<F>,
    history: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    total: usize,
    channels: usize,
    t_out: usize,
    hist_len: usize,
    kernel: usize,
    dilation: usize,
    has_bias: u32,
    #[define(F)] _dtype: ElemType,
) {
    let cube = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    let idx = cube * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if idx >= total {
        terminate!();
    }
    let t = idx % t_out;
    let bc = idx / t_out;
    let c = bc % channels;
    let x_row = bc * t_out;
    let h_row = bc * hist_len;
    let w_row = c * kernel;

    let acc = RuntimeCell::<f32>::new(if has_bias != 0 {
        f32::cast_from(bias[c])
    } else {
        0f32
    });
    for tap in 0..kernel {
        let lag = (kernel - 1 - tap) * dilation;
        let wv = f32::cast_from(weight[w_row + tap]);
        if t >= lag {
            acc.store(acc.read() + f32::cast_from(x[x_row + (t - lag)]) * wv);
        } else if hist_len > 0 {
            let deficit = lag - t;
            if deficit <= hist_len {
                acc.store(acc.read() + f32::cast_from(history[h_row + (hist_len - deficit)]) * wv);
            }
        }
    }
    output[idx] = F::cast_from(acc.consume());
}

#[allow(clippy::too_many_arguments)]
#[cube]
fn load_transpose_sample<F: Float>(
    x: &Tensor<F>,
    history: &Tensor<F>,
    x_row: usize,
    h_row: usize,
    hist_len: usize,
    ti: usize,
    m: usize,
) -> f32 {
    if ti >= m {
        f32::cast_from(x[x_row + (ti - m)])
    } else if hist_len > 0usize {
        let deficit = m - ti;
        if deficit <= hist_len {
            f32::cast_from(history[h_row + (hist_len - deficit)])
        } else {
            0f32
        }
    } else {
        0f32
    }
}

/// Register-tiled 32 × 32 (2 × 2 outputs/thread) causal 1D transposed convolution grouped by
/// output phase `t_out = ti * stride + phase`. Uses 4 KB shared memory per 16 × 16 workgroup
/// with a workgroup-uniform interior fast path that skips history and bounds checks.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn causal_conv_transpose1d_tiled_kernel<F: Float>(
    x: &Tensor<F>,
    history: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    c_in: usize,
    c_out: usize,
    t_in: usize,
    hist_len: usize,
    kernel: usize,
    stride: usize,
    #[comptime] has_bias: bool,
    #[define(F)] _dtype: ElemType,
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let t_base = (CUBE_POS_X as usize) * 32usize;
    let oc_base = (CUBE_POS_Y as usize) * 32usize;
    let bz = CUBE_POS_Z as usize;
    let batch = bz / stride;
    let phase = bz - batch * stride;

    let mut w_tile = Shared::<[f32]>::new_slice(512usize);
    let mut x_tile = Shared::<[f32]>::new_slice(512usize);

    let acc00 = RuntimeCell::<f32>::new(0f32);
    let acc01 = RuntimeCell::<f32>::new(0f32);
    let acc10 = RuntimeCell::<f32>::new(0f32);
    let acc11 = RuntimeCell::<f32>::new(0f32);

    let ic_blocks = c_in.div_ceil(16usize);
    let full_oc = oc_base + 32usize <= c_out;
    let oc0 = oc_base + ty;
    let oc1 = oc0 + 16usize;
    let ti0 = t_base + tx;
    let ti1 = ti0 + 16usize;

    let mut m = 0usize;
    let mut k_idx = phase;
    while k_idx < kernel {
        let w_k_base = k_idx * c_out;
        let x_interior = t_base + 32usize <= t_in && t_base >= m;
        for ic_block in 0..ic_blocks {
            let ic_base = ic_block * 16usize;
            let full_ic = ic_base + 16usize <= c_in;
            let ic_w = ic_base + tx;

            if full_oc && full_ic {
                w_tile[ty * 16usize + tx] = f32::cast_from(weight[(w_k_base + oc0) * c_in + ic_w]);
                w_tile[(ty + 16usize) * 16usize + tx] =
                    f32::cast_from(weight[(w_k_base + oc1) * c_in + ic_w]);
            } else {
                let mut w0 = 0f32;
                let mut w1 = 0f32;
                if ic_w < c_in {
                    if oc0 < c_out {
                        w0 = f32::cast_from(weight[(w_k_base + oc0) * c_in + ic_w]);
                    }
                    if oc1 < c_out {
                        w1 = f32::cast_from(weight[(w_k_base + oc1) * c_in + ic_w]);
                    }
                }
                w_tile[ty * 16usize + tx] = w0;
                w_tile[(ty + 16usize) * 16usize + tx] = w1;
            }

            let ic_x = ic_base + ty;
            let x_row = (batch * c_in + ic_x) * t_in;
            if x_interior && full_ic {
                let src_base = t_base - m;
                x_tile[ty * 32usize + tx] = f32::cast_from(x[x_row + src_base + tx]);
                x_tile[ty * 32usize + tx + 16usize] =
                    f32::cast_from(x[x_row + src_base + tx + 16usize]);
            } else {
                let h_row = (batch * c_in + ic_x) * hist_len;
                let mut v0 = 0f32;
                let mut v1 = 0f32;
                if ic_x < c_in {
                    if ti0 < t_in {
                        v0 = load_transpose_sample(x, history, x_row, h_row, hist_len, ti0, m);
                    }
                    if ti1 < t_in {
                        v1 = load_transpose_sample(x, history, x_row, h_row, hist_len, ti1, m);
                    }
                }
                x_tile[ty * 32usize + tx] = v0;
                x_tile[ty * 32usize + tx + 16usize] = v1;
            }

            sync_cube();
            #[unroll]
            for k in 0..16usize {
                let wk0 = w_tile[ty * 16usize + k];
                let wk1 = w_tile[(ty + 16usize) * 16usize + k];
                let xk0 = x_tile[k * 32usize + tx];
                let xk1 = x_tile[k * 32usize + tx + 16usize];
                acc00.store(acc00.read() + wk0 * xk0);
                acc01.store(acc01.read() + wk0 * xk1);
                acc10.store(acc10.read() + wk1 * xk0);
                acc11.store(acc11.read() + wk1 * xk1);
            }
            sync_cube();
        }
        m += 1;
        k_idx += stride;
    }

    let t_out = t_in * stride;
    if oc0 < c_out {
        let mut b0 = 0f32;
        if comptime!(has_bias) {
            b0 = f32::cast_from(bias[oc0]);
        }
        let out_row = (batch * c_out + oc0) * t_out;
        if ti0 < t_in {
            output[out_row + ti0 * stride + phase] = F::cast_from(acc00.consume() + b0);
        }
        if ti1 < t_in {
            output[out_row + ti1 * stride + phase] = F::cast_from(acc01.consume() + b0);
        }
    }
    if oc1 < c_out {
        let mut b1 = 0f32;
        if comptime!(has_bias) {
            b1 = f32::cast_from(bias[oc1]);
        }
        let out_row = (batch * c_out + oc1) * t_out;
        if ti0 < t_in {
            output[out_row + ti0 * stride + phase] = F::cast_from(acc10.consume() + b1);
        }
        if ti1 < t_in {
            output[out_row + ti1 * stride + phase] = F::cast_from(acc11.consume() + b1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{DType, Device, TensorData};

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn snake_matches_reference_shared_and_inplace() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F32, DType::F16] {
                for [batch, channels, time] in [[2, 7, 19], [1, 96, 385], [1, 96, 384], [2, 8, 96]]
                {
                    let x = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..batch * channels * time)
                                .map(|i| ((i * 19 + 5) as f32).sin() * 2.0)
                                .collect::<Vec<_>>(),
                            [batch, channels, time],
                        ),
                        (&device, dtype),
                    );
                    let alpha = BurnTensor::<1>::from_data(
                        TensorData::new(
                            (0..channels)
                                .map(|i| 0.5 + ((i * 7) as f32).cos() * 0.4)
                                .collect::<Vec<_>>(),
                            [channels],
                        ),
                        (&device, dtype),
                    );
                    let inv_beta = BurnTensor::<1>::from_data(
                        TensorData::new(
                            (0..channels)
                                .map(|i| 0.8 + ((i * 13) as f32).sin() * 0.3)
                                .collect::<Vec<_>>(),
                            [channels],
                        ),
                        (&device, dtype),
                    );
                    let periodic = (x.clone() * alpha.clone().reshape([1, channels, 1])).sin();
                    let expected = x.clone()
                        + periodic.clone() * periodic * inv_beta.clone().reshape([1, channels, 1]);
                    let shared = snake(x.clone(), alpha.clone(), inv_beta.clone());
                    let owned = snake(x * 1.0, alpha, inv_beta);
                    let ((s, o), e) = futures_lite::future::zip(
                        futures_lite::future::zip(
                            super::super::read(shared),
                            super::super::read(owned),
                        ),
                        super::super::read(expected),
                    )
                    .await;
                    let (s, o, e) = (s.unwrap(), o.unwrap(), e.unwrap());
                    let tol = if dtype == DType::F16 { 0.005 } else { 1e-5 };
                    for actual in [s, o] {
                        let err = actual
                            .iter()
                            .zip(&e)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        assert!(
                            err < tol,
                            "snake {dtype:?} {batch}x{channels}x{time}: {err}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    #[ignore = "requires a WebGPU adapter"]
    fn causal_conv1d_and_transpose_match_burn_reference() {
        futures_lite::future::block_on(async {
            let device = Device::wgpu_options().init_async().await.unwrap();
            for dtype in [DType::F32, DType::F16] {
                let tol = if dtype == DType::F16 { 0.005 } else { 1e-4 };
                // Full causal conv1d: covers small-out with shared weight cache (c_out=1),
                // partial boundary tiles and interior fast-path tiles in 32x32 tiled conv1d,
                // pointwise (k=1), and dilated (d=9, d=11).
                for [batch, c_in, c_out, time, kernel, dilation] in [
                    [1, 96, 1, 73, 7, 1],
                    [2, 19, 37, 41, 7, 3],
                    [1, 48, 35, 65, 1, 1],
                    [1, 32, 32, 80, 7, 9],
                    [1, 96, 96, 128, 7, 9],
                    [1, 16, 32, 72, 7, 11],
                ] {
                    let input = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..batch * c_in * time)
                                .map(|i| ((i * 17 + 3) as f32).sin() * 0.5)
                                .collect::<Vec<_>>(),
                            [batch, c_in, time],
                        ),
                        (&device, dtype),
                    );
                    let weight = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..c_out * c_in * kernel)
                                .map(|i| ((i * 31 + 1) as f32).cos() * 0.05)
                                .collect::<Vec<_>>(),
                            [c_out, c_in, kernel],
                        ),
                        (&device, dtype),
                    );
                    let packed = contiguous(weight.clone().permute([2, 0, 1]));
                    for use_bias in [false, true] {
                        let bias = use_bias.then(|| {
                            BurnTensor::<1>::from_data(
                                TensorData::new(
                                    (0..c_out)
                                        .map(|i| (i as f32) * 0.01 - 0.1)
                                        .collect::<Vec<_>>(),
                                    [c_out],
                                ),
                                (&device, dtype),
                            )
                        });
                        let pad_left = (kernel - 1) * dilation;
                        let padded = input.clone().pad(
                            [(0, 0), (0, 0), (pad_left, 0)],
                            burn::tensor::ops::PadMode::Constant(0.0),
                        );
                        let reference = burn::tensor::module::conv1d(
                            padded.cast(DType::F32),
                            weight.clone().cast(DType::F32),
                            bias.clone().map(|b| b.cast(DType::F32)),
                            burn::tensor::ops::ConvOptions::new([1], [0], [dilation], 1),
                        )
                        .cast(dtype);
                        let full = conv1d(
                            input.clone(),
                            None,
                            packed.clone(),
                            bias.clone(),
                            1,
                            dilation,
                            PAD_CAUSAL_ZERO,
                        );
                        // Stream in uneven spans including t < context and t >= context.
                        let mut parts = Vec::new();
                        let mut hist: Option<BurnTensor<3>> = None;
                        let mut start = 0;
                        for end in [1, 4, 21, time] {
                            let part = input.clone().slice([0..batch, 0..c_in, start..end]);
                            parts.push(conv1d(
                                part.clone(),
                                hist.clone(),
                                packed.clone(),
                                bias.clone(),
                                1,
                                dilation,
                                PAD_CAUSAL_ZERO,
                            ));
                            if pad_left > 0 {
                                let t = end - start;
                                hist = if t >= pad_left {
                                    Some(part.slice([0..batch, 0..c_in, t - pad_left..t]))
                                } else {
                                    let combined = match hist.take() {
                                        Some(prev) => BurnTensor::cat(vec![prev, part], 2),
                                        None => part.pad(
                                            [(0, 0), (0, 0), (pad_left, 0)],
                                            burn::tensor::ops::PadMode::Constant(0.0),
                                        ),
                                    };
                                    let ct = combined.dims()[2];
                                    Some(combined.slice([0..batch, 0..c_in, ct - pad_left..ct]))
                                };
                            }
                            start = end;
                        }
                        let streamed = BurnTensor::cat(parts, 2);
                        let residual = BurnTensor::<3>::from_data(
                            TensorData::new(
                                (0..batch * c_out * time)
                                    .map(|i| ((i * 11 + 5) as f32).cos() * 0.25)
                                    .collect::<Vec<_>>(),
                                [batch, c_out, time],
                            ),
                            (&device, dtype),
                        );
                        let with_res = conv1d_add(
                            input.clone(),
                            None,
                            packed.clone(),
                            bias.clone(),
                            1,
                            dilation,
                            PAD_CAUSAL_ZERO,
                            residual.clone(),
                        );
                        let ref_res = (reference.clone().cast(DType::F32)
                            + residual.cast(DType::F32))
                        .cast(dtype);
                        let ((f, s), (e, (wr, er))) = futures_lite::future::zip(
                            futures_lite::future::zip(
                                super::super::read(full),
                                super::super::read(streamed),
                            ),
                            futures_lite::future::zip(
                                super::super::read(reference),
                                futures_lite::future::zip(
                                    super::super::read(with_res),
                                    super::super::read(ref_res),
                                ),
                            ),
                        )
                        .await;
                        let (f, s, e, wr, er) =
                            (f.unwrap(), s.unwrap(), e.unwrap(), wr.unwrap(), er.unwrap());
                        for (kind, out, exp) in
                            [("full", f, &e), ("stream", s, &e), ("residual", wr, &er)]
                        {
                            let err = out
                                .iter()
                                .zip(exp)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0f32, f32::max);
                            assert!(
                                err < tol,
                                "causal_conv1d {kind} {dtype:?} c={c_in}->{c_out} k={kernel} d={dilation} bias={use_bias}: {err}"
                            );
                        }
                    }
                }

                // Depthwise causal conv1d.
                let (batch, c, time, kernel, dilation) = (2usize, 37usize, 43usize, 7usize, 2usize);
                let input = BurnTensor::<3>::from_data(
                    TensorData::new(
                        (0..batch * c * time)
                            .map(|i| ((i * 23) as f32).sin() * 0.5)
                            .collect::<Vec<_>>(),
                        [batch, c, time],
                    ),
                    (&device, dtype),
                );
                let weight = BurnTensor::<3>::from_data(
                    TensorData::new(
                        (0..c * kernel)
                            .map(|i| ((i * 19) as f32).cos() * 0.1)
                            .collect::<Vec<_>>(),
                        [c, 1, kernel],
                    ),
                    (&device, dtype),
                );
                let bias = BurnTensor::<1>::from_data(
                    TensorData::new((0..c).map(|i| i as f32 * 0.01).collect::<Vec<_>>(), [c]),
                    (&device, dtype),
                );
                let pad_left = (kernel - 1) * dilation;
                let reference = burn::tensor::module::conv1d(
                    input
                        .clone()
                        .pad(
                            [(0, 0), (0, 0), (pad_left, 0)],
                            burn::tensor::ops::PadMode::Constant(0.0),
                        )
                        .cast(DType::F32),
                    weight.clone().cast(DType::F32),
                    Some(bias.clone().cast(DType::F32)),
                    burn::tensor::ops::ConvOptions::new([1], [0], [dilation], c),
                )
                .cast(dtype);
                let actual = depthwise_causal_conv1d(
                    input,
                    None,
                    weight.reshape([c, kernel]),
                    Some(bias),
                    dilation,
                );
                let (a, e) = futures_lite::future::zip(
                    super::super::read(actual),
                    super::super::read(reference),
                )
                .await;
                let err = a
                    .unwrap()
                    .iter()
                    .zip(&e.unwrap())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                assert!(err < tol, "depthwise {dtype:?}: {err}");

                // Strided causal, causal-edge (CodecEncoder), and reflect-same (Speaker) conv1d.
                for [c_in, c_out, time, kernel, stride, dilation, pad_mode] in [
                    [32, 64, 300, 8, 4, 1, PAD_CAUSAL_ZERO],
                    [64, 64, 313, 10, 5, 1, PAD_CAUSAL_ZERO],
                    [64, 64, 151, 4, 2, 1, PAD_CAUSAL_EDGE],
                    [64, 64, 137, 5, 1, 1, PAD_REFLECT_SAME],
                    [64, 64, 137, 3, 1, 4, PAD_REFLECT_SAME],
                    [35, 37, 51, 3, 1, 2, PAD_REFLECT_SAME],
                ] {
                    let input = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..c_in * time)
                                .map(|i| ((i * 13 + 7) as f32).sin() * 0.5)
                                .collect::<Vec<_>>(),
                            [1, c_in, time],
                        ),
                        (&device, dtype),
                    );
                    let weight = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..c_out * c_in * kernel)
                                .map(|i| ((i * 29 + 3) as f32).cos() * 0.05)
                                .collect::<Vec<_>>(),
                            [c_out, c_in, kernel],
                        ),
                        (&device, dtype),
                    );
                    let bias = BurnTensor::<1>::from_data(
                        TensorData::new(
                            (0..c_out)
                                .map(|i| (i as f32) * 0.01 - 0.05)
                                .collect::<Vec<_>>(),
                            [c_out],
                        ),
                        (&device, dtype),
                    );
                    let eff_k = (kernel - 1) * dilation + 1;
                    let padded = if pad_mode == PAD_REFLECT_SAME {
                        let left = (eff_k - 1) / 2;
                        input.clone().pad(
                            [(0, 0), (0, 0), (left, eff_k - 1 - left)],
                            burn::tensor::ops::PadMode::Reflect,
                        )
                    } else {
                        let mode = if pad_mode == PAD_CAUSAL_EDGE {
                            burn::tensor::ops::PadMode::Edge
                        } else {
                            burn::tensor::ops::PadMode::Constant(0.0)
                        };
                        input.clone().pad(
                            [
                                (0, 0),
                                (0, 0),
                                (eff_k - stride, (stride - time % stride) % stride),
                            ],
                            mode,
                        )
                    };
                    let reference = burn::tensor::module::conv1d(
                        padded.cast(DType::F32),
                        weight.clone().cast(DType::F32),
                        Some(bias.clone().cast(DType::F32)),
                        burn::tensor::ops::ConvOptions::new([stride], [0], [dilation], 1),
                    )
                    .cast(dtype);
                    let packed = contiguous(weight.permute([2, 0, 1]));
                    let actual =
                        conv1d(input, None, packed, Some(bias), stride, dilation, pad_mode);
                    let (a, e) = futures_lite::future::zip(
                        super::super::read(actual),
                        super::super::read(reference),
                    )
                    .await;
                    let err = a
                        .unwrap()
                        .iter()
                        .zip(&e.unwrap())
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        err < tol,
                        "conv1d pad={pad_mode} s={stride} k={kernel} d={dilation} {dtype:?}: {err}"
                    );
                }

                // Causal transposed conv1d across all codec upsample (stride, kernel) shapes,
                // covering both boundary tiles and interior fast-path tiles in the 32x32 kernel.
                for [batch, c_in, c_out, time, stride, kernel] in [
                    [1, 35, 37, 25, 2, 2],
                    [2, 24, 19, 17, 3, 6],
                    [1, 48, 36, 19, 5, 10],
                    [1, 32, 32, 12, 8, 16],
                    [1, 64, 64, 73, 4, 8],
                    [1, 32, 96, 128, 3, 6],
                ] {
                    let input = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..batch * c_in * time)
                                .map(|i| ((i * 29 + 7) as f32).sin() * 0.5)
                                .collect::<Vec<_>>(),
                            [batch, c_in, time],
                        ),
                        (&device, dtype),
                    );
                    let weight = BurnTensor::<3>::from_data(
                        TensorData::new(
                            (0..c_in * c_out * kernel)
                                .map(|i| ((i * 13 + 3) as f32).cos() * 0.05)
                                .collect::<Vec<_>>(),
                            [c_in, c_out, kernel],
                        ),
                        (&device, dtype),
                    );
                    let packed = contiguous(weight.clone().permute([2, 1, 0]));
                    for use_bias in [false, true] {
                        let bias = use_bias.then(|| {
                            BurnTensor::<1>::from_data(
                                TensorData::new(
                                    (0..c_out).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
                                    [c_out],
                                ),
                                (&device, dtype),
                            )
                        });
                        let raw = burn::tensor::module::conv_transpose1d(
                            input.clone().cast(DType::F32),
                            weight.clone().cast(DType::F32),
                            bias.clone().map(|b| b.cast(DType::F32)),
                            burn::tensor::ops::ConvTransposeOptions::new(
                                [stride],
                                [0],
                                [0],
                                [1],
                                1,
                            ),
                        )
                        .cast(dtype);
                        let reference = raw.slice([0..batch, 0..c_out, 0..time * stride]);
                        let full = causal_conv_transpose1d(
                            input.clone(),
                            None,
                            packed.clone(),
                            bias.clone(),
                            stride,
                        );
                        let context = (kernel - stride).div_ceil(stride);
                        let mut parts = Vec::new();
                        let mut hist: Option<BurnTensor<3>> = None;
                        let mut start = 0;
                        for end in [1, 5, time] {
                            let part = input.clone().slice([0..batch, 0..c_in, start..end]);
                            parts.push(causal_conv_transpose1d(
                                part.clone(),
                                hist.clone(),
                                packed.clone(),
                                bias.clone(),
                                stride,
                            ));
                            if context > 0 {
                                let t = end - start;
                                hist = Some(part.slice([0..batch, 0..c_in, t - context..t]));
                            }
                            start = end;
                        }
                        let streamed = BurnTensor::cat(parts, 2);
                        let ((f, s), e) = futures_lite::future::zip(
                            futures_lite::future::zip(
                                super::super::read(full),
                                super::super::read(streamed),
                            ),
                            super::super::read(reference),
                        )
                        .await;
                        let (f, s, e) = (f.unwrap(), s.unwrap(), e.unwrap());
                        for (kind, out) in [("full", f), ("stream", s)] {
                            let err = out
                                .iter()
                                .zip(&e)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0f32, f32::max);
                            assert!(
                                err < tol,
                                "transpose {kind} {dtype:?} s={stride} k={kernel} bias={use_bias}: {err}"
                            );
                        }
                    }
                }
            }
        });
    }
}
