# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/dropout.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/dropout.py
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
@util.autotune(dropout_configs(), ['x_size'])
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, input_ptr: tl.tensor, x_size: tl.int32,
    p: tl.float32, seed: tl.int32, dtype: tl.constexpr, x_block_size: tl.
    constexpr, require_x_boundary_check: tl.constexpr):
    pid = tl.program_id(0)
    x_offset = pid * x_block_size
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(x_size,),
        strides=(1,), offsets=(x_offset,), block_shape=(x_block_size,),
        order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(x_size,), strides
        =(1,), offsets=(x_offset,), block_shape=(x_block_size,), order=(0,))
    if require_x_boundary_check:
        input = tl.load(input_block_ptr, boundary_check=(0,))
    else:
        input = tl.load(input_block_ptr)
    condition = tl.rand(seed, tl.arange(0, x_block_size) + x_offset) > p
    output = tl.where(condition, input / (1.0 - p + language.eps), 0.0)
    if require_x_boundary_check:
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(0,))
    else:
        tl.store(output_block_ptr, output.to(dtype))


# Forward method (kernel launch code)
def _Dropout_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, p = args
    util.push_trace('Dropout.__forward')
    output = Dropout.__forward(input, p)
    util.pop_trace()
    ctx.save_for_backward(input, output)
    ctx.p = p
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(dropout_configs(), ['x_size'])
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    output_ptr: tl.tensor, x_size: tl.int32, p: tl.float32, dtype: tl.
    constexpr, x_block_size: tl.constexpr, require_x_boundary_check: tl.
    constexpr):
    pid = tl.program_id(0)
    x_offset = pid * x_block_size
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(x_size,
        ), strides=(1,), offsets=(x_offset,), block_shape=(x_block_size,),
        order=(0,))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        x_size,), strides=(1,), offsets=(x_offset,), block_shape=(
        x_block_size,), order=(0,))
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(x_size,),
        strides=(1,), offsets=(x_offset,), block_shape=(x_block_size,),
        order=(0,))
    if require_x_boundary_check:
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,))
        output = tl.load(output_block_ptr, boundary_check=(0,))
    else:
        grad_output = tl.load(grad_output_block_ptr)
        output = tl.load(output_block_ptr)
    condition = (p == 0.0) | (output > 0.0)
    grad_input = tl.where(condition, grad_output * (1.0 - p + language.eps),
        0.0)
    if require_x_boundary_check:
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check
            =(0,))
    else:
        tl.store(grad_input_block_ptr, grad_input.to(dtype))


# Backward method (kernel launch code)
def _Dropout_backward(ctx: Any, *grad_outputs: Any):
    grad_output, = grad_outputs
    input, output = ctx.saved_tensors
    util.push_trace('Dropout.__backward')
    grad_input = Dropout.__backward(grad_output, input, output, ctx.p)
    util.pop_trace()
    return grad_input, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Dropout(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, p = args
        util.push_trace('Dropout.__forward')
        output = Dropout.__forward(input, p)
        util.pop_trace()
        ctx.save_for_backward(input, output)
        ctx.p = p
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, = grad_outputs
        input, output = ctx.saved_tensors
        util.push_trace('Dropout.__backward')
        grad_input = Dropout.__backward(grad_output, input, output, ctx.p)
        util.pop_trace()
        return grad_input, None, None

    @staticmethod
    def __forward(input: torch.Tensor, p: torch.float32):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        x_size = input.numel()
        output = torch.empty(x_size, **factory_kwargs)

        def grid(meta):
            return triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.Dropout.forward')
        kernel.Dropout.forward[grid](output, input, x_size, p, torch.random
            .seed(), util.dtype(output.dtype))
        util.pop_trace()
        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor, output:
        torch.Tensor, p: torch.float32):
        x_size = input.numel()
        grad_input = torch.empty_like(input)

        def grid(meta):
            return triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.Dropout.backward')
        kernel.Dropout.backward[grid](grad_input, grad_output, output,
            x_size, p, util.dtype(grad_input.dtype))
        util.pop_trace()
        return grad_input
