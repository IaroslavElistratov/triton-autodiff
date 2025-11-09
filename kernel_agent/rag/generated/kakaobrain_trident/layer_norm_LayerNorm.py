# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/layer_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/layer_norm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def dtype(input):
    if input == torch.float32:
        return tl.float32
    elif input == torch.float16:
        return tl.float16
    elif input == torch.bfloat16:
        return tl.bfloat16
    elif input == torch.int64:
        return tl.int64
    else:
        raise ValueError(f"Unable to convert the given input: '{input}'.")


def pop_trace():
    if config.use_trace:
        nvtx.pop_range(domain='Trident')


def push_trace(message: str):
    if config.use_trace:
        nvtx.push_range(message, color='green', domain='Trident')


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@staticmethod
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, rstd_ptr: tl.tensor, mean_ptr: tl.tensor,
    input_ptr: tl.tensor, y_size: tl.int32, x_size: tl.int32, weight_ptr:
    tl.tensor, bias_ptr: tl.tensor, eps: tl.float32, dtype: tl.constexpr,
    x_block_size: tl.constexpr, require_x_boundary_check: tl.constexpr):
    y_offset = tl.program_id(0)
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(y_size, x_size),
        strides=(x_size, 1), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    rstd_block_ptr = tl.make_block_ptr(rstd_ptr, shape=(y_size,), strides=(
        1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    mean_block_ptr = tl.make_block_ptr(mean_ptr, shape=(y_size,), strides=(
        1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(y_size, x_size),
        strides=(x_size, 1), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    if require_x_boundary_check:
        input = tl.load(input_block_ptr, boundary_check=(1,),
            padding_option='zero')
        mean = tl.sum(input / x_size, 1)
        condition = tl.arange(0, x_block_size) < x_size
        centered_mean = tl.where(condition, input - mean, 0)
    else:
        input = tl.load(input_block_ptr)
        mean = tl.sum(input / x_size, 1)
        centered_mean = input - mean
    var = tl.sum(centered_mean * centered_mean / x_size, 1)
    rstd = tl.math.rsqrt(var + eps)
    output = centered_mean * rstd
    if weight_ptr is not None:
        weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(x_size,),
            strides=(1,), offsets=(0,), block_shape=(x_block_size,), order=(0,)
            )
        if require_x_boundary_check:
            weight = tl.load(weight_block_ptr, boundary_check=(0,))
        else:
            weight = tl.load(weight_block_ptr)
        output *= weight
    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(bias_ptr, shape=(x_size,),
            strides=(1,), offsets=(0,), block_shape=(x_block_size,), order=(0,)
            )
        if require_x_boundary_check:
            bias = tl.load(bias_block_ptr, boundary_check=(0,))
        else:
            bias = tl.load(bias_block_ptr)
        output += bias
    if require_x_boundary_check:
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
    else:
        tl.store(output_block_ptr, output.to(dtype))
    tl.store(rstd_block_ptr, rstd.to(dtype))
    tl.store(mean_block_ptr, mean.to(dtype))


# Forward method (kernel launch code)
def _LayerNorm_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, normalized_shape, weight, bias, eps = args
    util.push_trace('LayerNorm.__forward')
    output, rstd, mean = LayerNorm.__forward(input, normalized_shape,
        weight, bias, eps)
    util.pop_trace()
    ctx.save_for_backward(input, weight, bias, rstd, mean)
    ctx.normalized_shape = normalized_shape
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_weight_staging_ptr: tl.tensor,
    grad_output_ptr: tl.tensor, input_ptr: tl.tensor, y_size: tl.int32,
    x_size: tl.int32, weight_ptr: tl.tensor, rstd_ptr: tl.tensor, mean_ptr:
    tl.tensor, dtype: tl.constexpr, x_block_size: tl.constexpr,
    require_x_boundary_check: tl.constexpr):
    y_offset = tl.program_id(0)
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(y_size,
        x_size), strides=(x_size, 1), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        y_size, x_size), strides=(x_size, 1), offsets=(y_offset, 0),
        block_shape=(1, x_block_size), order=(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(y_size, x_size),
        strides=(x_size, 1), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    rstd_block_ptr = tl.make_block_ptr(rstd_ptr, shape=(y_size,), strides=(
        1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    mean_block_ptr = tl.make_block_ptr(mean_ptr, shape=(y_size,), strides=(
        1,), offsets=(y_offset,), block_shape=(1,), order=(0,))
    if require_x_boundary_check:
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,),
            padding_option='zero')
        input = tl.load(input_block_ptr, boundary_check=(1,),
            padding_option='zero')
    else:
        grad_output = tl.load(grad_output_block_ptr)
        input = tl.load(input_block_ptr)
    rstd = tl.load(rstd_block_ptr)
    mean = tl.load(mean_block_ptr)
    centered_mean = input - mean
    if weight_ptr is not None:
        weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(1, x_size),
            strides=(x_size, 1), offsets=(0, 0), block_shape=(1,
            x_block_size), order=(1, 0))
        if require_x_boundary_check:
            weight = tl.load(weight_block_ptr, boundary_check=(1,))
        else:
            weight = tl.load(weight_block_ptr)
        grad_norm = weight * grad_output
    else:
        grad_norm = grad_output
    grad_std = tl.sum(grad_norm * centered_mean, 1)
    grad_var = grad_std * -(0.5 * rstd * rstd * rstd) / x_size
    grad_distance = 2 * centered_mean * grad_var
    grad_centered_mean = grad_norm * rstd + grad_distance
    grad_mean = -tl.sum(grad_centered_mean, 1) / x_size
    grad_input = grad_centered_mean + grad_mean
    if require_x_boundary_check:
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check
            =(1,))
    else:
        tl.store(grad_input_block_ptr, grad_input.to(dtype))
    if grad_weight_staging_ptr is not None:
        grad_weight_staging_block_ptr = tl.make_block_ptr(
            grad_weight_staging_ptr, shape=(y_size, x_size), strides=(
            x_size, 1), offsets=(y_offset, 0), block_shape=(1, x_block_size
            ), order=(1, 0))
        norm = centered_mean * rstd
        grad_weight = norm * grad_output
        if require_x_boundary_check:
            tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype),
                boundary_check=(1,))
        else:
            tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype))


# Backward method (kernel launch code)
def _LayerNorm_backward(ctx: Any, *grad_outputs: Any):
    grad_output, = grad_outputs
    input, weight, bias, rstd, mean = ctx.saved_tensors
    util.push_trace('LayerNorm.__backward')
    grad_input, grad_weight, grad_bias = LayerNorm.__backward(grad_output,
        input, ctx.normalized_shape, weight, bias, rstd, mean)
    util.pop_trace()
    return grad_input, None, grad_weight, grad_bias, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, normalized_shape, weight, bias, eps = args
        util.push_trace('LayerNorm.__forward')
        output, rstd, mean = LayerNorm.__forward(input, normalized_shape,
            weight, bias, eps)
        util.pop_trace()
        ctx.save_for_backward(input, weight, bias, rstd, mean)
        ctx.normalized_shape = normalized_shape
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, = grad_outputs
        input, weight, bias, rstd, mean = ctx.saved_tensors
        util.push_trace('LayerNorm.__backward')
        grad_input, grad_weight, grad_bias = LayerNorm.__backward(grad_output,
            input, ctx.normalized_shape, weight, bias, rstd, mean)
        util.pop_trace()
        return grad_input, None, grad_weight, grad_bias, None, None, None, None

    @staticmethod
    def __forward(input, normalized_shape, weight, bias, eps):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        x_size = functools.reduce(lambda x, y: x * y, normalized_shape)
        y_size = input.numel() // x_size
        output = torch.empty_like(input)
        rstd = torch.empty(y_size, **factory_kwargs)
        mean = torch.empty(y_size, **factory_kwargs)

        def grid(meta):
            return y_size,
        util.push_trace('kernel.LayerNorm.forward')
        kernel.LayerNorm.forward[grid](output, rstd, mean, input, y_size,
            x_size, weight, bias, eps, util.dtype(input.dtype), triton.
            next_power_of_2(x_size))
        util.pop_trace()
        return output, rstd, mean

    @staticmethod
    def __backward(grad_output, input, normalized_shape, weight, bias, rstd,
        mean):
        factory_kwargs = {'device': grad_output.device, 'dtype':
            grad_output.dtype}
        x_size = functools.reduce(lambda x, y: x * y, normalized_shape)
        y_size = grad_output.numel() // x_size
        grad_input = torch.empty_like(input)
        if weight is not None:
            grad_weight_staging = torch.empty(y_size, x_size, **factory_kwargs)
        else:
            grad_weight_staging = None

        def grid(meta):
            return y_size,
        util.push_trace('kernel.LayerNorm.backward')
        kernel.LayerNorm.backward[grid](grad_input, grad_weight_staging,
            grad_output, input, y_size, x_size, weight, rstd, mean, util.
            dtype(grad_output.dtype), triton.next_power_of_2(x_size))
        util.pop_trace()
        if weight is not None:
            util.push_trace('torch.sum')
            grad_weight = torch.sum(grad_weight_staging, 0)
            util.pop_trace()
        else:
            grad_weight = None
        if bias is not None:
            util.push_trace('torch.sum')
            grad_bias = torch.sum(grad_output, 0)
            util.pop_trace()
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias
