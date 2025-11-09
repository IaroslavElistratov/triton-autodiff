# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/rms_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/rms_norm.py
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


def size_and_stride(input: torch.Tensor, dim: int):
    if input.dim() == 2:
        if dim == 0:
            x_size, y_size = input.shape
            y_stride = input.stride(1)
            x_stride = input.stride(0)
        else:
            y_size, x_size = input.shape
            y_stride = input.stride(0)
            x_stride = input.stride(1)
        return y_size, x_size, y_stride, x_stride
    elif input.dim() == 3:
        if dim == 0:
            z_size, y_size, x_size = input.shape[0], input.shape[1
                ], input.shape[2]
            z_stride, y_stride, x_stride = input.stride(0), input.stride(1
                ), input.stride(2)
        elif dim == 1:
            z_size, y_size, x_size = input.shape[1], input.shape[0
                ], input.shape[2]
            z_stride, y_stride, x_stride = input.stride(1), input.stride(0
                ), input.stride(2)
        else:
            z_size, y_size, x_size = input.shape[2], input.shape[0
                ], input.shape[1]
            z_stride, y_stride, x_stride = input.stride(2), input.stride(0
                ), input.stride(1)
        return z_size, y_size, x_size, z_stride, y_stride, x_stride
    else:
        raise ValueError(f'{dim} is not supported.')


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@staticmethod
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, rms_ptr: tl.tensor, input_ptr: tl.tensor,
    y_size: tl.int32, x_size: tl.int32, y_stride: tl.int32, x_stride: tl.
    int32, partial_size: tl.constexpr, weight_ptr: tl.tensor, bias_ptr: tl.
    tensor, eps: tl.float32, dtype: tl.constexpr, x_block_size: tl.
    constexpr, require_x_boundary_check: tl.constexpr):
    y_offset = tl.program_id(0)
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    rms_block_ptr = tl.make_block_ptr(rms_ptr, shape=(y_size,), strides=(1,
        ), offsets=(y_offset,), block_shape=(1,), order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(x_size,),
        strides=(1,), offsets=(0,), block_shape=(x_block_size,), order=(0,))
    if require_x_boundary_check:
        input = tl.load(input_block_ptr, boundary_check=(1,))
    else:
        input = tl.load(input_block_ptr)
    if x_block_size != partial_size:
        condition = tl.arange(0, x_block_size) < partial_size
        partial_input = tl.where(condition, input, 0)
    else:
        partial_input = input
    rms = tl.math.sqrt(tl.sum(partial_input * partial_input / partial_size, 1))
    norm = input / (rms + eps)
    if require_x_boundary_check:
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
    else:
        weight = tl.load(weight_block_ptr)
    output = norm * weight
    if bias_ptr is not None:
        bias_block_ptr = tl.make_block_ptr(bias_ptr, shape=(1, x_size),
            strides=(x_stride, 1), offsets=(0, 0), block_shape=(1,
            x_block_size), order=(1, 0))
        if require_x_boundary_check:
            bias = tl.load(bias_block_ptr, boundary_check=(1,))
        else:
            bias = tl.load(bias_block_ptr)
        output += bias
    tl.store(rms_block_ptr, rms.to(dtype))
    if require_x_boundary_check:
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
    else:
        tl.store(output_block_ptr, output.to(dtype))


# Forward method (kernel launch code)
def _RMSNorm_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, p, weight, bias, eps = args
    util.push_trace('RMSNorm.__forward')
    output, rms = RMSNorm.__forward(input, p, weight, bias, eps)
    util.pop_trace()
    ctx.save_for_backward(input, rms, weight, bias)
    ctx.p = p
    ctx.eps = eps
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_weight_staging: tl.tensor,
    grad_output_ptr: tl.tensor, input_ptr: tl.tensor, y_size: tl.int32,
    x_size: tl.int32, y_stride: tl.int32, x_stride: tl.int32, rms_ptr: tl.
    tensor, partial_size: tl.constexpr, weight_ptr: tl.tensor, eps: tl.
    float32, dtype: tl.constexpr, x_block_size: tl.constexpr,
    require_x_boundary_check: tl.constexpr):
    y_offset = tl.program_id(0)
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(y_size,
        x_size), strides=(y_stride, x_stride), offsets=(y_offset, 0),
        block_shape=(1, x_block_size), order=(1, 0))
    grad_weight_staging_block_ptr = tl.make_block_ptr(grad_weight_staging,
        shape=(y_size, x_size), strides=(y_stride, x_stride), offsets=(
        y_offset, 0), block_shape=(1, x_block_size), order=(1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        y_size, x_size), strides=(y_stride, x_stride), offsets=(y_offset, 0
        ), block_shape=(1, x_block_size), order=(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(y_size, x_size),
        strides=(y_stride, x_stride), offsets=(y_offset, 0), block_shape=(1,
        x_block_size), order=(1, 0))
    rms_block_ptr = tl.make_block_ptr(rms_ptr, shape=(y_size,), strides=(1,
        ), offsets=(y_offset,), block_shape=(1,), order=(0,))
    weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(x_size,),
        strides=(1,), offsets=(0,), block_shape=(x_block_size,), order=(0,))
    if require_x_boundary_check:
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
        input = tl.load(input_block_ptr, boundary_check=(1,))
    else:
        grad_output = tl.load(grad_output_block_ptr)
        input = tl.load(input_block_ptr)
    rms = tl.load(rms_block_ptr)
    if require_x_boundary_check:
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
    else:
        weight = tl.load(weight_block_ptr)
    grad_norm = grad_output * weight
    norm = input / (rms + eps)
    grad_weight = grad_output * norm
    if require_x_boundary_check:
        tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype),
            boundary_check=(1,))
    else:
        tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype))
    grad_rms = grad_norm * -input / (rms * rms + eps)
    if require_x_boundary_check:
        condition = tl.arange(0, x_block_size) < x_size
        grad_rms = tl.where(condition, grad_rms, 0.0)
    grad_rms = tl.sum(grad_rms, 1)
    grad_mean_square = grad_rms / (2 * rms)
    grad_partial_input = 2 * input * grad_mean_square / partial_size
    if x_block_size != partial_size:
        condition = tl.arange(0, x_block_size) < partial_size
        grad_partial_input = tl.where(condition, grad_partial_input, 0)
    grad_input = grad_norm / (rms + eps) + grad_partial_input
    if require_x_boundary_check:
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check
            =(1,))
    else:
        tl.store(grad_input_block_ptr, grad_input.to(dtype))


# Backward method (kernel launch code)
def _RMSNorm_backward(ctx: Any, *grad_outputs: Any):
    grad_output, = grad_outputs
    input, rms, weight, bias = ctx.saved_tensors
    util.push_trace('RMSNorm.__backward')
    grad_input, grad_weight, grad_bias = RMSNorm.__backward(grad_output,
        input, ctx.p, rms, weight, bias, ctx.eps)
    util.pop_trace()
    return grad_input, None, grad_weight, grad_bias, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, p, weight, bias, eps = args
        util.push_trace('RMSNorm.__forward')
        output, rms = RMSNorm.__forward(input, p, weight, bias, eps)
        util.pop_trace()
        ctx.save_for_backward(input, rms, weight, bias)
        ctx.p = p
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, = grad_outputs
        input, rms, weight, bias = ctx.saved_tensors
        util.push_trace('RMSNorm.__backward')
        grad_input, grad_weight, grad_bias = RMSNorm.__backward(grad_output,
            input, ctx.p, rms, weight, bias, ctx.eps)
        util.pop_trace()
        return grad_input, None, grad_weight, grad_bias, None

    @staticmethod
    def __forward(input: torch.Tensor, p: float, weight: torch.Tensor, bias:
        torch.Tensor, eps: float):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, 1)
        output = torch.empty_like(input)
        rms = torch.empty(y_size, **factory_kwargs)

        def grid(meta):
            return y_size,
        util.push_trace('kernel.RMSNorm.forward')
        kernel.RMSNorm.forward[grid](output, rms, input, y_size, x_size,
            y_stride, x_stride, x_size if p < 0.0 or p > 1.0 else x_size *
            p, weight, bias, eps, util.dtype(input.dtype), triton.
            next_power_of_2(x_size))
        util.pop_trace()
        return output, rms

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, p: float,
        rms: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
        ):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        y_size, x_size, y_stride, x_stride = util.size_and_stride(input, 1)
        grad_input = torch.empty_like(grad_output)
        grad_weight_staging = torch.empty((y_size, x_size), **factory_kwargs)

        def grid(meta):
            return y_size,
        util.push_trace('kernel.RMSNorm.backward')
        kernel.RMSNorm.backward[grid](grad_input, grad_weight_staging,
            grad_output, input, y_size, x_size, y_stride, x_stride, rms, 
            x_size if p < 0.0 or p > 1.0 else x_size * p, weight, eps, util
            .dtype(input.dtype), triton.next_power_of_2(x_size))
        util.pop_trace()
        util.push_trace('torch.sum')
        grad_weight = torch.sum(grad_weight_staging, 0)
        util.pop_trace()
        if bias is not None:
            util.push_trace('torch.sum')
            grad_bias = function.sum(grad_output, 0)
            util.pop_trace()
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias
