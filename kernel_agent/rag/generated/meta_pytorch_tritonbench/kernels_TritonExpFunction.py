# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/tritonbench
# Source-Files: tritonbench/operators/vector_exp/kernels.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ct7v_342/tritonbench-main/tritonbench/operators/vector_exp/kernels.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

# Common helper imports
from triton import cdiv
from math import exp

@triton.jit
def time(_semantic=None):
    if IS_HIP:
        return tl.inline_asm_elementwise(
            """
            s_memrealtime $0
            s_waitcnt vmcnt(0)
            """
            , '=r', [], dtype=tl.int64, is_pure=False, pack=1)
    elif IS_CUDA:
        return tl.extra.cuda.globaltimer()
    else:
        tl.static_assert(False, 'Unsupported platform')


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def triton_exp_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.
    constexpr, profile_mem=None):
    if profile_mem is not None:
        start = time()
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.exp(x)
    tl.store(output_ptr + offsets, output, mask=mask)
    if profile_mem is not None:
        end = time()
        tl.store(profile_mem + pid, end - start)


# Forward method (kernel launch code)
def _TritonExpFunction_forward(ctx, x: torch.Tensor, block_size: int=1024,
    profile_mem: torch.Tensor=None):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, block_size),)
    triton_exp_kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size,
        profile_mem=profile_mem)
    ctx.save_for_backward(output)
    ctx.block_size = block_size
    ctx.profile_mem = profile_mem
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def triton_exp_backward_kernel(grad_output_ptr, output_ptr, grad_input_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr, profile_mem=None):
    if profile_mem is not None:
        start = time()
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    grad_output = tl.load(grad_output_ptr + offsets, mask=mask)
    output = tl.load(output_ptr + offsets, mask=mask)
    grad_input = grad_output * output
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)
    if profile_mem is not None:
        end = time()
        tl.store(profile_mem + pid, end - start)


# Backward method (kernel launch code)
def _TritonExpFunction_backward(ctx, grad_output: torch.Tensor):
    output, = ctx.saved_tensors
    grad_input = torch.empty_like(grad_output)
    n_elements = grad_output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, ctx.block_size),)
    triton_exp_backward_kernel[grid](grad_output, output, grad_input,
        n_elements, BLOCK_SIZE=ctx.block_size, profile_mem=ctx.profile_mem)
    return grad_input, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class TritonExpFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, block_size: int=1024, profile_mem:
        torch.Tensor=None):
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, block_size),)
        triton_exp_kernel[grid](x, output, n_elements, BLOCK_SIZE=
            block_size, profile_mem=profile_mem)
        ctx.save_for_backward(output)
        ctx.block_size = block_size
        ctx.profile_mem = profile_mem
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        output, = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)
        n_elements = grad_output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, ctx.block_size),)
        triton_exp_backward_kernel[grid](grad_output, output, grad_input,
            n_elements, BLOCK_SIZE=ctx.block_size, profile_mem=ctx.profile_mem)
        return grad_input, None, None
