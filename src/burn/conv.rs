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
    #[fusion(dtype = x, shape = Shape::from([x[0], weight[1], x[2]]))]
    fn causal_conv1d(
        x: FloatTensor<Self>,
        history: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: FloatTensor<Self>,
        dilation: usize,
        has_history: bool,
        has_bias: bool,
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

pub(super) fn causal_conv1d(
    x: BurnTensor<3>,
    history: Option<BurnTensor<3>>,
    weight: BurnTensor<3>,
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
    BurnTensor::from_dispatch(Dispatch::causal_conv1d(
        x.into_dispatch(),
        history.into_dispatch(),
        weight.into_dispatch(),
        bias_dispatch,
        dilation,
        has_history,
        has_bias,
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
        into_contiguous(input)
    }

    fn snake(
        x: FloatTensor<Self>,
        alpha: FloatTensor<Self>,
        inverse_beta: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let alpha = into_contiguous(alpha);
        let inverse_beta = into_contiguous(inverse_beta);
        let shape = x.shape();
        let [batch, channels, time] = [shape[0], shape[1], shape[2]];
        let total = batch * channels * time;
        let dim = CubeDim::new(&x.client, total);
        let groups = total.div_ceil(dim.num_elems() as usize);
        let max_x = x.client.properties().hardware.max_cube_count.0;
        if x.can_mut() && x.is_contiguous() && x.is_nonoverlapping() {
            let x = into_mut_contiguous(x);
            snake_inplace_kernel::launch(
                &x.client,
                row_grid(groups, max_x),
                dim,
                x.clone().into_tensor_arg(),
                alpha.into_tensor_arg(),
                inverse_beta.into_tensor_arg(),
                total,
                channels,
                time,
                dtype_to_storage_type(x.dtype),
            );
            x
        } else {
            let x = into_contiguous(x);
            let output = Self::float_empty(shape, &x.device, x.dtype.into());
            snake_out_kernel::launch(
                &x.client,
                row_grid(groups, max_x),
                dim,
                x.clone().into_tensor_arg(),
                alpha.into_tensor_arg(),
                inverse_beta.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                total,
                channels,
                time,
                dtype_to_storage_type(x.dtype),
            );
            output
        }
    }

    fn causal_conv1d(
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
        let [batch, c_in, t_out] = [x.shape()[0], x.shape()[1], x.shape()[2]];
        let [kernel, c_out, _] = [weight.shape()[0], weight.shape()[1], weight.shape()[2]];
        let hist_len = if has_history { history.shape()[2] } else { 0 };
        let bias_flag = u32::from(has_bias);
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
            causal_conv1d_small_out_kernel::launch(
                &x.client,
                row_grid(groups, max_x),
                dim,
                x.clone().into_tensor_arg(),
                history.into_tensor_arg(),
                weight.into_tensor_arg(),
                bias.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                total,
                c_in,
                c_out,
                t_out,
                hist_len,
                kernel,
                dilation,
                bias_flag,
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
                output.clone().into_tensor_arg(),
                c_in,
                c_out,
                t_out,
                hist_len,
                kernel,
                dilation,
                bias_flag,
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
        let bias_flag = u32::from(has_bias);
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
            bias_flag,
            dtype_to_storage_type(x.dtype),
        );
        output
    }
}

#[cube(launch)]
fn snake_inplace_kernel<F: Float>(
    x: &mut Tensor<F>,
    alpha: &Tensor<F>,
    inverse_beta: &Tensor<F>,
    total: usize,
    channels: usize,
    time: usize,
    #[define(F)] _dtype: ElemType,
) {
    let cube = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    let idx = cube * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if idx >= total {
        terminate!();
    }
    let bc = idx / time;
    let c = bc - (bc / channels) * channels;
    let v = f32::cast_from(x[idx]);
    let periodic = (v * f32::cast_from(alpha[c])).sin();
    x[idx] = F::cast_from(v + periodic * periodic * f32::cast_from(inverse_beta[c]));
}

#[cube(launch)]
fn snake_out_kernel<F: Float>(
    x: &Tensor<F>,
    alpha: &Tensor<F>,
    inverse_beta: &Tensor<F>,
    output: &mut Tensor<F>,
    total: usize,
    channels: usize,
    time: usize,
    #[define(F)] _dtype: ElemType,
) {
    let cube = (CUBE_POS_X as usize) + (CUBE_POS_Y as usize) * (CUBE_COUNT_X as usize);
    let idx = cube * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if idx >= total {
        terminate!();
    }
    let bc = idx / time;
    let c = bc - (bc / channels) * channels;
    let v = f32::cast_from(x[idx]);
    let periodic = (v * f32::cast_from(alpha[c])).sin();
    output[idx] = F::cast_from(v + periodic * periodic * f32::cast_from(inverse_beta[c]));
}

/// Register-tiled 32 × 32 causal 1D convolution over contiguous `[B, C, T]` tensors
/// and packed `[K, C_out, C_in]` weights. Uses 4 KB shared memory per 16 × 16 workgroup.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn causal_conv1d_tiled_kernel<F: Float>(
    x: &Tensor<F>,
    history: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    c_in: usize,
    c_out: usize,
    t_out: usize,
    hist_len: usize,
    kernel: usize,
    dilation: usize,
    has_bias: u32,
    #[define(F)] _dtype: ElemType,
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let t_base = (CUBE_POS_X as usize) * 32;
    let oc_base = (CUBE_POS_Y as usize) * 32;
    let batch = CUBE_POS_Z as usize;

    let mut w_tile = Shared::<[f32]>::new_slice(512usize);
    let mut x_tile = Shared::<[f32]>::new_slice(512usize);

    let acc00 = RuntimeCell::<f32>::new(0f32);
    let acc01 = RuntimeCell::<f32>::new(0f32);
    let acc10 = RuntimeCell::<f32>::new(0f32);
    let acc11 = RuntimeCell::<f32>::new(0f32);

    let ic_blocks = c_in.div_ceil(16usize);
    for tap in 0..kernel {
        let lag = (kernel - 1 - tap) * dilation;
        let w_tap_base = tap * c_out;
        for ic_block in 0..ic_blocks {
            let ic_base = ic_block * 16;

            let ic_w = ic_base + tx;
            let oc0 = oc_base + ty;
            let oc1 = oc0 + 16;
            let mut w0 = 0f32;
            if oc0 < c_out && ic_w < c_in {
                w0 = f32::cast_from(weight[(w_tap_base + oc0) * c_in + ic_w]);
            }
            w_tile[ty * 16 + tx] = w0;

            let mut w1 = 0f32;
            if oc1 < c_out && ic_w < c_in {
                w1 = f32::cast_from(weight[(w_tap_base + oc1) * c_in + ic_w]);
            }
            w_tile[(ty + 16) * 16 + tx] = w1;

            let ic_x = ic_base + ty;
            let x_row = (batch * c_in + ic_x) * t_out;
            let h_row = (batch * c_in + ic_x) * hist_len;

            let t0 = t_base + tx;
            let mut v0 = 0f32;
            if ic_x < c_in && t0 < t_out {
                if t0 >= lag {
                    v0 = f32::cast_from(x[x_row + (t0 - lag)]);
                } else if hist_len > 0 {
                    let deficit = lag - t0;
                    if deficit <= hist_len {
                        v0 = f32::cast_from(history[h_row + (hist_len - deficit)]);
                    }
                }
            }
            x_tile[ty * 32 + tx] = v0;

            let t1 = t0 + 16;
            let mut v1 = 0f32;
            if ic_x < c_in && t1 < t_out {
                if t1 >= lag {
                    v1 = f32::cast_from(x[x_row + (t1 - lag)]);
                } else if hist_len > 0 {
                    let deficit = lag - t1;
                    if deficit <= hist_len {
                        v1 = f32::cast_from(history[h_row + (hist_len - deficit)]);
                    }
                }
            }
            x_tile[ty * 32 + tx + 16] = v1;

            sync_cube();
            #[unroll]
            for k in 0..16usize {
                let wk0 = w_tile[ty * 16 + k];
                let wk1 = w_tile[(ty + 16) * 16 + k];
                let xk0 = x_tile[k * 32 + tx];
                let xk1 = x_tile[k * 32 + tx + 16];
                acc00.store(acc00.read() + wk0 * xk0);
                acc01.store(acc01.read() + wk0 * xk1);
                acc10.store(acc10.read() + wk1 * xk0);
                acc11.store(acc11.read() + wk1 * xk1);
            }
            sync_cube();
        }
    }

    let oc0 = oc_base + ty;
    let oc1 = oc0 + 16;
    let t0 = t_base + tx;
    let t1 = t0 + 16;
    let b0 = if has_bias != 0 && oc0 < c_out {
        f32::cast_from(bias[oc0])
    } else {
        0f32
    };
    let b1 = if has_bias != 0 && oc1 < c_out {
        f32::cast_from(bias[oc1])
    } else {
        0f32
    };
    if oc0 < c_out {
        let out_row = (batch * c_out + oc0) * t_out;
        if t0 < t_out {
            output[out_row + t0] = F::cast_from(acc00.consume() + b0);
        }
        if t1 < t_out {
            output[out_row + t1] = F::cast_from(acc01.consume() + b0);
        }
    }
    if oc1 < c_out {
        let out_row = (batch * c_out + oc1) * t_out;
        if t0 < t_out {
            output[out_row + t0] = F::cast_from(acc10.consume() + b1);
        }
        if t1 < t_out {
            output[out_row + t1] = F::cast_from(acc11.consume() + b1);
        }
    }
}

/// Time-parallel causal 1D convolution for narrow outputs such as the final mono waveform head.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
fn causal_conv1d_small_out_kernel<F: Float>(
    x: &Tensor<F>,
    history: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    total: usize,
    c_in: usize,
    c_out: usize,
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
    let bo = idx / t_out;
    let oc = bo % c_out;
    let batch = bo / c_out;

    let acc = RuntimeCell::<f32>::new(if has_bias != 0 {
        f32::cast_from(bias[oc])
    } else {
        0f32
    });
    for tap in 0..kernel {
        let lag = (kernel - 1 - tap) * dilation;
        let w_base = (tap * c_out + oc) * c_in;
        if t >= lag {
            let src_t = t - lag;
            for ic in 0..c_in {
                let xv = f32::cast_from(x[(batch * c_in + ic) * t_out + src_t]);
                let wv = f32::cast_from(weight[w_base + ic]);
                acc.store(acc.read() + xv * wv);
            }
        } else if hist_len > 0 {
            let deficit = lag - t;
            if deficit <= hist_len {
                let src_h = hist_len - deficit;
                for ic in 0..c_in {
                    let xv = f32::cast_from(history[(batch * c_in + ic) * hist_len + src_h]);
                    let wv = f32::cast_from(weight[w_base + ic]);
                    acc.store(acc.read() + xv * wv);
                }
            }
        }
    }
    output[idx] = F::cast_from(acc.consume());
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

/// Register-tiled 32 × 32 causal 1D transposed convolution grouped by output phase
/// `t_out = ti * stride + phase`, reading packed `[K, C_out, C_in]` weights.
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
    has_bias: u32,
    #[define(F)] _dtype: ElemType,
) {
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let t_base = (CUBE_POS_X as usize) * 32;
    let oc_base = (CUBE_POS_Y as usize) * 32;
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
    let mut m = 0usize;
    let mut k_idx = phase;
    while k_idx < kernel {
        let w_k_base = k_idx * c_out;
        for ic_block in 0..ic_blocks {
            let ic_base = ic_block * 16;

            let ic_w = ic_base + tx;
            let oc0 = oc_base + ty;
            let oc1 = oc0 + 16;
            let mut w0 = 0f32;
            if oc0 < c_out && ic_w < c_in {
                w0 = f32::cast_from(weight[(w_k_base + oc0) * c_in + ic_w]);
            }
            w_tile[ty * 16 + tx] = w0;

            let mut w1 = 0f32;
            if oc1 < c_out && ic_w < c_in {
                w1 = f32::cast_from(weight[(w_k_base + oc1) * c_in + ic_w]);
            }
            w_tile[(ty + 16) * 16 + tx] = w1;

            let ic_x = ic_base + ty;
            let x_row = (batch * c_in + ic_x) * t_in;
            let h_row = (batch * c_in + ic_x) * hist_len;

            let ti0 = t_base + tx;
            let mut v0 = 0f32;
            if ic_x < c_in && ti0 < t_in {
                if ti0 >= m {
                    v0 = f32::cast_from(x[x_row + (ti0 - m)]);
                } else if hist_len > 0 {
                    let deficit = m - ti0;
                    if deficit <= hist_len {
                        v0 = f32::cast_from(history[h_row + (hist_len - deficit)]);
                    }
                }
            }
            x_tile[ty * 32 + tx] = v0;

            let ti1 = ti0 + 16;
            let mut v1 = 0f32;
            if ic_x < c_in && ti1 < t_in {
                if ti1 >= m {
                    v1 = f32::cast_from(x[x_row + (ti1 - m)]);
                } else if hist_len > 0 {
                    let deficit = m - ti1;
                    if deficit <= hist_len {
                        v1 = f32::cast_from(history[h_row + (hist_len - deficit)]);
                    }
                }
            }
            x_tile[ty * 32 + tx + 16] = v1;

            sync_cube();
            #[unroll]
            for k in 0..16usize {
                let wk0 = w_tile[ty * 16 + k];
                let wk1 = w_tile[(ty + 16) * 16 + k];
                let xk0 = x_tile[k * 32 + tx];
                let xk1 = x_tile[k * 32 + tx + 16];
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

    let oc0 = oc_base + ty;
    let oc1 = oc0 + 16;
    let ti0 = t_base + tx;
    let ti1 = ti0 + 16;
    let t_out = t_in * stride;
    let b0 = if has_bias != 0 && oc0 < c_out {
        f32::cast_from(bias[oc0])
    } else {
        0f32
    };
    let b1 = if has_bias != 0 && oc1 < c_out {
        f32::cast_from(bias[oc1])
    } else {
        0f32
    };
    if oc0 < c_out {
        let out_row = (batch * c_out + oc0) * t_out;
        if ti0 < t_in {
            output[out_row + ti0 * stride + phase] = F::cast_from(acc00.consume() + b0);
        }
        if ti1 < t_in {
            output[out_row + ti1 * stride + phase] = F::cast_from(acc01.consume() + b0);
        }
    }
    if oc1 < c_out {
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
                for [batch, channels, time] in [[2, 7, 19], [1, 96, 385]] {
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
                // Full causal conv1d: covers small-out (c_out=1), tiled (c_out=37), pointwise (k=1), dilated (d=9).
                for [batch, c_in, c_out, time, kernel, dilation] in [
                    [1, 96, 1, 73, 7, 1],
                    [2, 19, 37, 41, 7, 3],
                    [1, 48, 35, 65, 1, 1],
                    [1, 32, 32, 80, 7, 9],
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
                        let full = causal_conv1d(
                            input.clone(),
                            None,
                            packed.clone(),
                            bias.clone(),
                            dilation,
                        );
                        // Stream in uneven spans including t < context and t >= context.
                        let mut parts = Vec::new();
                        let mut hist: Option<BurnTensor<3>> = None;
                        let mut start = 0;
                        for end in [1, 4, 21, time] {
                            let part = input.clone().slice([0..batch, 0..c_in, start..end]);
                            parts.push(causal_conv1d(
                                part.clone(),
                                hist.clone(),
                                packed.clone(),
                                bias.clone(),
                                dilation,
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

                // Causal transposed conv1d across all codec upsample (stride, kernel) shapes.
                for [batch, c_in, c_out, time, stride, kernel] in [
                    [1, 35, 37, 25, 2, 2],
                    [2, 24, 19, 17, 3, 6],
                    [1, 48, 36, 19, 5, 10],
                    [1, 32, 32, 12, 8, 16],
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
