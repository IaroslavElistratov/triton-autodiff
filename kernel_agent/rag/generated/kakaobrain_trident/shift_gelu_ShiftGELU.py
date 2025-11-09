# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/shift_gelu.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/shift_gelu.py
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
@util.autotune(shift_gelu_configs(), ['x_size'])
@triton.jit
def forward(output_ptr: tl.tensor, shift_ptr: tl.tensor, input_ptr: tl.
    tensor, y_size: tl.int32, x_size: tl.int32, bias_ptr: tl.tensor, dtype:
    tl.constexpr, x_block_size: tl.constexpr):
    pid = tl.program_id(0)
    num_x_blocks = tl.cdiv(x_size, x_block_size)
    y_offset = pid // num_x_blocks
    x = pid % num_x_blocks
    x_offset = x * x_block_size
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(y_size, x_size),
        strides=(x_size, 1), offsets=(y_offset, x_offset), block_shape=(1,
        x_block_size), order=(1, 0))
    shift_block_ptr = tl.make_block_ptr(shift_ptr, shape=(y_size, x_size),
        strides=(x_size, 1), offsets=(y_offset, x_offset), block_shape=(1,
        x_block_size), order=(1, 0))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(y_size, x_size),
        strides=(x_size, 1), offsets=(y_offset, x_offset), block_shape=(1,
        x_block_size), order=(1, 0))
    bias_block_ptr = tl.make_block_ptr(bias_ptr, shape=(1, x_size), strides
        =(x_size, 1), offsets=(0, x_offset), block_shape=(1, x_block_size),
        order=(1, 0))
    input = tl.load(input_block_ptr, boundary_check=(1,))
    bias = tl.load(bias_block_ptr, boundary_check=(1,))
    shift = input + bias
    output = language.math.GELU.forward(shift)
    tl.store(output_block_ptr, output.to(dtype), boundary_check=(1,))
    tl.store(shift_block_ptr, shift.to(dtype), boundary_check=(1,))


# Forward method (kernel launch code)
def _ShiftGELU_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, bias = args
    util.push_trace('ShiftGELU.__forward')
    output, shift = ShiftGELU.__forward(input, bias)
    util.pop_trace()
    ctx.save_for_backward(input, shift)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(shift_gelu_configs(), ['x_size'])
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    shift_ptr: tl.tensor, y_size: tl.int32, x_size: tl.int32, dtype: tl.
    constexpr, x_block_size: tl.constexpr):
    pid = tl.program_id(0)
    num_x_blocks = tl.cdiv(x_size, x_block_size)
    y_offset = pid // num_x_blocks
    x = pid % num_x_blocks
    x_offset = x * x_block_size
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(y_size,
        x_size), strides=(x_size, 1), offsets=(y_offset, x_offset),
        block_shape=(1, x_block_size), order=(1, 0))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        y_size, x_size), strides=(x_size, 1), offsets=(y_offset, x_offset),
        block_shape=(1, x_block_size), order=(1, 0))
    shift_block_ptr = tl.make_block_ptr(shift_ptr, shape=(y_size, x_size),
        strides=(x_size, 1), offsets=(y_offset, x_offset), block_shape=(1,
        x_block_size), order=(1, 0))
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(1,))
    shift = tl.load(shift_block_ptr, boundary_check=(1,))
    grad_input = language.math.GELU.backward(grad_output, shift)
    tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check=(1,))


# Backward method (kernel launch code)
def _ShiftGELU_backward(ctx: Any, *grad_outputs: Any):
    input, shift = ctx.saved_tensors
    grad_output = grad_outputs[0]
    util.push_trace('ShiftGELU.__backward')
    grad_input, grad_bias = ShiftGELU.__backward(grad_output, input, shift)
    util.pop_trace()
    return grad_input, grad_bias


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ShiftGELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, bias = args
        util.push_trace('ShiftGELU.__forward')
        output, shift = ShiftGELU.__forward(input, bias)
        util.pop_trace()
        ctx.save_for_backward(input, shift)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        input, shift = ctx.saved_tensors
        grad_output = grad_outputs[0]
        util.push_trace('ShiftGELU.__backward')
        grad_input, grad_bias = ShiftGELU.__backward(grad_output, input, shift)
        util.pop_trace()
        return grad_input, grad_bias

    @staticmethod
    def __forward(input: torch.Tensor, bias: torch.Tensor):
        y_size, x_size = input.shape
        output = torch.empty_like(input)
        shift = torch.empty_like(input)

        def grid(meta):
            return y_size * triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.ShiftGELU.forward')
        kernel.ShiftGELU.forward[grid](output, shift, input, y_size, x_size,
            bias, util.dtype(input.dtype))
        util.pop_trace()
        return output, shift

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, shift:
        torch.Tensor):
        y_size, x_size = input.shape
        grad_input = torch.empty_like(input)

        def grid(meta):
            return y_size * triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.ShiftGELU.backward')
        kernel.ShiftGELU.backward[grid](grad_input, grad_output, shift,
            y_size, x_size, util.dtype(input.dtype))
        util.pop_trace()
        util.push_trace('torch.sum')
        grad_bias = torch.sum(grad_input, 0)
        util.pop_trace()
        return grad_input, grad_bias
