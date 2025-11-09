# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/kakaobrain/trident
# Source-Files: trident/operation/silu.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rl_ay0ux/trident-main/trident/operation/silu.py
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
@util.autotune(silu_configs(), ['x_size'])
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def forward(output_ptr: tl.tensor, input_ptr: tl.tensor, x_size: tl.int32,
    dtype: tl.constexpr, x_block_size: tl.constexpr,
    require_x_boundary_check: tl.constexpr):
    x_offset = tl.program_id(0) * x_block_size
    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(x_size,),
        strides=(1,), offsets=(x_offset,), block_shape=(x_block_size,),
        order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(x_size,), strides
        =(1,), offsets=(x_offset,), block_shape=(x_block_size,), order=(0,))
    if require_x_boundary_check:
        input = tl.load(input_block_ptr, boundary_check=(0,))
    else:
        input = tl.load(input_block_ptr)
    sigma = 1 / (1 + tl.math.fast_expf(-input.to(tl.float32)))
    output = input * sigma
    if require_x_boundary_check:
        tl.store(output_block_ptr, output.to(dtype), boundary_check=(0,))
    else:
        tl.store(output_block_ptr, output.to(dtype))


# Forward method (kernel launch code)
def _SiLU_forward(ctx: Any, *args: Any, **kwargs: Any):
    input, = args
    util.push_trace('SiLU.__forward')
    output = SiLU.__forward(input.view(-1, input.shape[-1]))
    util.pop_trace()
    ctx.save_for_backward(input)
    return output.view(input.shape)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@staticmethod
@util.autotune(silu_configs(), ['x_size'])
@triton.heuristics({'require_x_boundary_check': lambda args: args['x_size'] %
    args['x_block_size']})
@triton.jit
def backward(grad_input_ptr: tl.tensor, grad_output_ptr: tl.tensor,
    input_ptr: tl.tensor, x_size: tl.int32, dtype: tl.constexpr,
    x_block_size: tl.constexpr, require_x_boundary_check: tl.constexpr):
    x_offset = tl.program_id(0) * x_block_size
    grad_input_block_ptr = tl.make_block_ptr(grad_input_ptr, shape=(x_size,
        ), strides=(1,), offsets=(x_offset,), block_shape=(x_block_size,),
        order=(0,))
    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(
        x_size,), strides=(1,), offsets=(x_offset,), block_shape=(
        x_block_size,), order=(0,))
    input_block_ptr = tl.make_block_ptr(input_ptr, shape=(x_size,), strides
        =(1,), offsets=(x_offset,), block_shape=(x_block_size,), order=(0,))
    if require_x_boundary_check:
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,))
        input = tl.load(input_block_ptr, boundary_check=(0,))
    else:
        grad_output = tl.load(grad_output_block_ptr)
        input = tl.load(input_block_ptr)
    sigma = 1 / (1 + tl.math.fast_expf(-input.to(tl.float32)))
    grad_input = grad_output * (sigma + input * sigma * (1 - sigma))
    if require_x_boundary_check:
        tl.store(grad_input_block_ptr, grad_input.to(dtype), boundary_check
            =(0,))
    else:
        tl.store(grad_input_block_ptr, grad_input.to(dtype))


# Backward method (kernel launch code)
def _SiLU_backward(ctx: Any, *grad_outputs: Any):
    grad_output, = grad_outputs
    input, = ctx.saved_tensors
    util.push_trace('SiLU.__backward')
    grad_input = SiLU.__backward(grad_output, input)
    util.pop_trace()
    return grad_input, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SiLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, = args
        util.push_trace('SiLU.__forward')
        output = SiLU.__forward(input.view(-1, input.shape[-1]))
        util.pop_trace()
        ctx.save_for_backward(input)
        return output.view(input.shape)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        grad_output, = grad_outputs
        input, = ctx.saved_tensors
        util.push_trace('SiLU.__backward')
        grad_input = SiLU.__backward(grad_output, input)
        util.pop_trace()
        return grad_input, None

    @staticmethod
    def __forward(input: torch.Tensor):
        factory_kwargs = {'device': input.device, 'dtype': input.dtype}
        x_size = input.numel()
        output = torch.empty(x_size, **factory_kwargs)

        def grid(meta):
            return triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.SiLU.forward')
        kernel.SiLU.forward[grid](output, input, x_size, util.dtype(output.
            dtype))
        util.pop_trace()
        return output

    @staticmethod
    def __backward(grad_output: torch.Tensor, input: torch.Tensor):
        x_size = input.numel()
        grad_input = torch.empty_like(input)

        def grid(meta):
            return triton.cdiv(x_size, meta['x_block_size']),
        util.push_trace('kernel.SiLU.backward')
        kernel.SiLU.backward[grid](grad_input, grad_output, input, x_size,
            util.dtype(grad_input.dtype))
        util.pop_trace()
        return grad_input
