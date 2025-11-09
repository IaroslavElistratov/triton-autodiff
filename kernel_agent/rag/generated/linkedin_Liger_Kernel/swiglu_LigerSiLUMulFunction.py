# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/swiglu.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/swiglu.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f'Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}.'
            )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def is_hip() ->bool:
    return torch.version.hip is not None


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: tl.
    constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


def swiglu_forward(a, b):
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    _swiglu_forward_kernel[n_rows,](a, b, c, c.stride(-2), n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return a, b, c.view(*ori_shape)


# Forward method (kernel launch code)
@ensure_contiguous
def _LigerSiLUMulFunction_forward(ctx, a, b):
    a, b, c = swiglu_forward(a, b)
    ctx.save_for_backward(a, b)
    return c


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.
    constexpr, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


def swiglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    _swiglu_backward_kernel[n_rows,](dc, a, b, dc.stride(-2), n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return a.view(*ori_shape), b.view(*ori_shape)


# Backward method (kernel launch code)
@ensure_contiguous
def _LigerSiLUMulFunction_backward(ctx, dc):
    a, b = ctx.saved_tensors
    a, b = swiglu_backward(a, b, dc)
    return a, b


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerSiLUMulFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = swiglu_backward(a, b, dc)
        return a, b
