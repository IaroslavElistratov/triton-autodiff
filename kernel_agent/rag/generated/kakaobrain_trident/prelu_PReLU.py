# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/prelu.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/prelu.py
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
@util.autotune(prelu_configs(), ['x_size'])
@triton.heuristics({'require_y_boundary_check': lambda args: args['y_size'] %
    args['y_block_size'], 'require_x_boundary_check': lambda args: args[
    'x_size'] % args['x_block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, input_ptr: tl.tensor, weight_ptr: tl.
    tensor, num_batches: tl.int32, y_size: tl.int32, x_size: tl.int32,
    batch_stride: tl.int32, y_stride: tl.int32, x_stride: tl.int32, dtype:
    tl.constexpr, y_block_size: tl.constexpr, x_block_size: tl.constexpr,
    require_y_boundary_check: tl.constexpr, require_x_boundary_check: tl.
    constexpr):
    pid = tl.program_id(0)
    num_y_blocks = tl.cdiv(y_size, y_block_size)
    num_x_blocks = tl.cdiv(x_size, x_block_size)
    num_blocks = num_y_blocks * num_x_blocks
    batch_offset = pid // num_blocks
    block = pid % num_blocks
    y_block = block // num_x_blocks
    x_block = block % num_x_blocks
    y_offset = y_block * y_block_size
    x_offset = x_block * x_block_size
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(num_batches,
        y_size, x_size), strides=(batch_stride, y_stride, x_stride),
        offsets=(batch_offset, y_offset, x_offset), block_shape=(1,
        y_block_size, x_block_size), order=(2, 1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(num_batches,
        y_size, x_size), strides=(batch_stride, y_stride, x_stride),
        offsets=(batch_offset, y_offset, x_offset), block_shape=(1,
        y_block_size, x_block_size), order=(2, 1, 0))
    weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(y_size, 1),
        strides=(1, 0), offsets=(y_offset, 0), block_shape=(y_block_size, 1
        ), order=(1, 0))
    if require_y_boundary_check | require_x_boundary_check:
        input = tl.load(input_block_ptr, boundary_check=(1, 2))
    else:
        input = tl.load(input_block_ptr)
    if require_y_boundary_check:
        weight = tl.load(weight_block_ptr, boundary_check=(0,))
    else:
        weight = tl.load(weight_block_ptr)
    output = language.math.LeakyReLU.forward(input, weight)
    if require_y_boundary_check | require_x_boundary_check:
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(1, 2))
    else:
        tl.store(output_block_ptr, output.to(dtype))


# Forward method (kernel launch code)
def _PReLU_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, weight = args
    util.push_trace('PReLU.__forward')
    output = PReLU.__forward(input.view(PReLU.__shape(input)), weight)
    util.pop_trace()
    ctx.save_for_backward(input, weight)
    return output.view(input.shape)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(prelu_configs(), ['x_size'])
@triton.heuristics({'require_y_boundary_check': lambda args: args['y_size'] %
    args['y_block_size'], 'require_x_boundary_check': lambda args: args[
    'x_size'] % args['x_block_size']})
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_weight_staging_ptr: tl.tensor,
    grad_output_ptr: tl.tensor, input_ptr: tl.tensor, weight_ptr: tl.tensor,
    num_batches: tl.int32, y_size: tl.int32, x_size: tl.int32, batch_stride:
    tl.int32, y_stride: tl.int32, x_stride: tl.int32, dtype: tl.constexpr,
    y_block_size: tl.constexpr, x_block_size: tl.constexpr,
    require_y_boundary_check: tl.constexpr, require_x_boundary_check: tl.
    constexpr):
    pid = tl.program_id(0)
    num_y_blocks = tl.cdiv(y_size, y_block_size)
    num_x_blocks = tl.cdiv(x_size, x_block_size)
    num_blocks = num_y_blocks * num_x_blocks
    batch_offset = pid // num_blocks
    block = pid % num_blocks
    y_block = block // num_x_blocks
    x_block = block % num_x_blocks
    y_offset = y_block * y_block_size
    x_offset = x_block * x_block_size
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(
        num_batches, y_size, x_size), strides=(batch_stride, y_stride,
        x_stride), offsets=(batch_offset, y_offset, x_offset), block_shape=
        (1, y_block_size, x_block_size), order=(2, 1, 0))
    grad_weight_staging_block_ptr = tl.make_block_ptr(grad_weight_staging_ptr,
        shape=(num_batches, y_size, x_size), strides=(batch_stride,
        y_stride, x_stride), offsets=(batch_offset, y_offset, x_offset),
        block_shape=(1, y_block_size, x_block_size), order=(2, 1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        num_batches, y_size, x_size), strides=(batch_stride, y_stride,
        x_stride), offsets=(batch_offset, y_offset, x_offset), block_shape=
        (1, y_block_size, x_block_size), order=(2, 1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(num_batches,
        y_size, x_size), strides=(batch_stride, y_stride, x_stride),
        offsets=(batch_offset, y_offset, x_offset), block_shape=(1,
        y_block_size, x_block_size), order=(2, 1, 0))
    weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(y_size, 1),
        strides=(1, 0), offsets=(y_offset, 0), block_shape=(y_block_size, 1
        ), order=(1, 0))
    if require_y_boundary_check | require_x_boundary_check:
        input = tl.load(input_block_ptr, boundary_check=(1, 2))
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(1, 2))
    else:
        input = tl.load(input_block_ptr)
        grad_output = tl.load(grad_output_block_ptr)
    weight = tl.load(weight_block_ptr)
    grad_input = language.math.LeakyReLU.backward(grad_output, input, weight)
    grad_weight = grad_output * tl.where(input > 0, 0, input)
    if require_y_boundary_check | require_x_boundary_check:
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check
            =(1, 2))
        tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype),
            boundary_check=(1, 2))
    else:
        tl.store(grad_input_block_ptr, grad_input.to(dtype))
        tl.store(grad_weight_staging_block_ptr, grad_weight.to(dtype))


# Backward method (kernel launch code)
def _PReLU_backward(ctx: Any, *grad_outputs: Any):
    grad_output = grad_outputs[0]
    input, weight = ctx.saved_tensors
    util.push_trace('PReLU.__backward')
    grad_input, grad_weight = PReLU.__backward(grad_output, input.view(
        PReLU.__shape(input)), weight)
    util.pop_trace()
    return grad_input.view(input.shape), grad_weight, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class PReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, weight = args
        util.push_trace('PReLU.__forward')
        output = PReLU.__forward(input.view(PReLU.__shape(input)), weight)
        util.pop_trace()
        ctx.save_for_backward(input, weight)
        return output.view(input.shape)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output = grad_outputs[0]
        input, weight = ctx.saved_tensors
        util.push_trace('PReLU.__backward')
        grad_input, grad_weight = PReLU.__backward(grad_output, input.view(
            PReLU.__shape(input)), weight)
        util.pop_trace()
        return grad_input.view(input.shape), grad_weight, None

    @staticmethod
    def __forward(input: torch.Tensor, weight: torch.Tensor):
        num_batches, y_size, x_size = input.shape
        output = torch.empty_like(input)

        def grid(meta):
            num_y_blocks = triton.cdiv(y_size, meta['y_block_size'])
            num_x_blocks = triton.cdiv(x_size, meta['x_block_size'])
            return num_batches * num_y_blocks * num_x_blocks,
        util.push_trace('kernel.PReLU.forward')
        kernel.PReLU.forward[grid](output, input, weight, num_batches,
            y_size, x_size, input.stride(0), input.stride(1), input.stride(
            2), util.dtype(output.dtype))
        util.pop_trace()
        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, weight:
        torch.Tensor):
        num_batches, y_size, x_size = input.shape
        grad_input = torch.empty_like(input)
        grad_weight_staging = torch.empty_like(input)

        def grid(meta):
            num_y_blocks = triton.cdiv(y_size, meta['y_block_size'])
            num_x_blocks = triton.cdiv(x_size, meta['x_block_size'])
            return num_batches * num_y_blocks * num_x_blocks,
        util.push_trace('kernel.PReLU.backward')
        kernel.PReLU.backward[grid](grad_input, grad_weight_staging,
            grad_output, input, weight, num_batches, y_size, x_size,
            grad_input.stride(0), grad_input.stride(1), grad_input.stride(2
            ), util.dtype(grad_input.dtype))
        util.pop_trace()
        if grad_weight_staging.dim() < 3:
            grad_weight = grad_weight_staging
        else:
            util.push_trace('torch.sum')
            grad_weight = torch.sum(grad_weight_staging, 2)
            util.pop_trace()
        return grad_input, grad_weight

    @staticmethod
    def __shape(input: torch.Tensor):
        if input.dim() == 1:
            return 1, 1, -1
        elif input.dim() == 2:
            return *input.shape, 1
        else:
            return *input.shape[0:2], -1
