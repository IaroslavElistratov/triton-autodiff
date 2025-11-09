# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/zhixuan-lin/forgetting-transformer
# Source-Files: src/forgetting_transformer/model/forgetting_transformer/token_shift.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_o52u21n_/forgetting-transformer-main/src/forgetting_transformer/model/forgetting_transformer/token_shift.py
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

def maybe_contiguous(x):
    return x.contiguous() if x.stride(-1) != 1 else x


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BLOCK_T': block_t}, num_warps=
    num_warps) for block_t in [32, 64, 128] for num_warps in [2, 4, 8]],
    key=['T', 'D'])
@triton.jit
def shift_fwd_kernel(X_PTR, PREV_WEIGHT_PTR, CURR_WEIGHT_PTR, OUT_PTR,
    stride_x_b, stride_x_t, stride_x_h, stride_x_d, stride_weight_b,
    stride_weight_t, stride_weight_h, T: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr):
    """
        everything is (B, T, D)
    """
    b_offset = tl.program_id(axis=2).to(tl.int64)
    t_offset = tl.program_id(axis=1).to(tl.int64) * BLOCK_T
    h_offset = tl.program_id(axis=0).to(tl.int64)
    x_ptr_offset = (b_offset * stride_x_b + t_offset * stride_x_t + 
        h_offset * stride_x_h)
    X_PTR += x_ptr_offset
    OUT_PTR += x_ptr_offset
    weight_ptr_offset = (b_offset * stride_weight_b + t_offset *
        stride_weight_t + h_offset * stride_weight_h)
    CURR_WEIGHT_PTR += weight_ptr_offset
    PREV_WEIGHT_PTR += weight_ptr_offset
    x_ptr = X_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(
        0, D)[None, :] * stride_x_d
    t_offset_block = t_offset + tl.arange(0, BLOCK_T)[:, None]
    x_mask = t_offset_block < T
    x_prev_ptr = x_ptr - stride_x_t
    t_prev_offset_block = t_offset_block - 1
    x_prev_mask = (t_prev_offset_block < T) & (t_prev_offset_block >= 0)
    curr_weight_ptr = CURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None
        ] * stride_weight_t
    prev_weight_ptr = PREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None
        ] * stride_weight_t
    x = tl.load(x_ptr, mask=x_mask, other=0.0)
    x_prev = tl.load(x_prev_ptr, mask=x_prev_mask, other=0.0)
    curr_weight = tl.load(curr_weight_ptr, mask=x_mask, other=0.0)
    prev_weight = tl.load(prev_weight_ptr, mask=x_mask, other=0.0)
    result = x * curr_weight.to(tl.float32) + x_prev * prev_weight.to(tl.
        float32)
    result = result.to(x.dtype)
    out_ptr = OUT_PTR + tl.arange(0, BLOCK_T)[:, None
        ] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    tl.store(out_ptr, result, mask=x_mask)


# Forward method (kernel launch code)
def _TokenShift_forward(ctx, x: torch.Tensor, prev_weight: torch.Tensor,
    curr_weight: torch.Tensor):
    B, T, H, D = x.size()
    assert D in {16, 32, 64, 128}
    assert prev_weight.size() == curr_weight.size() == (B, T, H)
    assert prev_weight.stride() == curr_weight.stride()
    x = maybe_contiguous(x)
    out = torch.empty_like(x)
    assert x.stride() == out.stride()
    grid = lambda meta: (H, triton.cdiv(T, meta['BLOCK_T']), B)
    shift_fwd_kernel[grid](x, prev_weight, curr_weight, out, *x.stride(), *
        curr_weight.stride(), T=T, D=D)
    ctx.save_for_backward(x, prev_weight, curr_weight)
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BLOCK_T': block_t}, num_warps=
    num_warps) for block_t in [32, 64, 128] for num_warps in [2, 4, 8]],
    key=['T', 'D'])
@triton.jit
def shift_bwd_kernel(X_PTR, PREV_WEIGHT_PTR, CURR_WEIGHT_PTR, DOUT_PTR,
    DX_PTR, DPREV_WEIGHT_PTR, DCURR_WEIGHT_PTR, stride_x_b, stride_x_t,
    stride_x_h, stride_x_d, stride_weight_b, stride_weight_t,
    stride_weight_h, T: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr):
    """
        everything is (B, T, D)
    """
    b_offset = tl.program_id(axis=2).to(tl.int64)
    t_offset = tl.program_id(axis=1).to(tl.int64) * BLOCK_T
    h_offset = tl.program_id(axis=0).to(tl.int64)
    x_ptr_offset = (b_offset * stride_x_b + t_offset * stride_x_t + 
        h_offset * stride_x_h)
    X_PTR += x_ptr_offset
    DX_PTR += x_ptr_offset
    DOUT_PTR += x_ptr_offset
    weight_ptr_offset = (b_offset * stride_weight_b + t_offset *
        stride_weight_t + h_offset * stride_weight_h)
    CURR_WEIGHT_PTR += weight_ptr_offset
    PREV_WEIGHT_PTR += weight_ptr_offset
    DCURR_WEIGHT_PTR += weight_ptr_offset
    DPREV_WEIGHT_PTR += weight_ptr_offset
    x_ptr = X_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(
        0, D)[None, :] * stride_x_d
    t_offset_block = t_offset + tl.arange(0, BLOCK_T)[:, None]
    x_mask = t_offset_block < T
    dout_ptr = DOUT_PTR + tl.arange(0, BLOCK_T)[:, None
        ] * stride_x_t + tl.arange(0, D)[None, :] * stride_x_d
    dout_next_ptr = dout_ptr + stride_x_t
    t_next_offset_block = t_offset_block + 1
    x_next_mask = t_next_offset_block < T
    x_prev_ptr = x_ptr - stride_x_t
    t_prev_offset_block = t_offset_block - 1
    x_prev_mask = (t_prev_offset_block < T) & (t_prev_offset_block >= 0)
    curr_weight_ptr = CURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None
        ] * stride_weight_t
    prev_weight_ptr = PREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None
        ] * stride_weight_t
    next_prev_weight_ptr = prev_weight_ptr + stride_weight_t
    x = tl.load(x_ptr, mask=x_mask, other=0.0)
    x_prev = tl.load(x_prev_ptr, mask=x_prev_mask, other=0.0)
    dout = tl.load(dout_ptr, mask=x_mask, other=0.0)
    dout_next = tl.load(dout_next_ptr, mask=x_next_mask, other=0.0)
    curr_weight = tl.load(curr_weight_ptr, mask=x_mask, other=0.0)
    next_prev_weight = tl.load(next_prev_weight_ptr, mask=x_next_mask,
        other=0.0)
    dx = dout * curr_weight.to(tl.float32) + dout_next * next_prev_weight.to(tl
        .float32)
    dx = dx.to(x.dtype)
    dcurr_weight = tl.sum(dout.to(tl.float32) * x, axis=1, keep_dims=True)
    dprev_weight = tl.sum(dout.to(tl.float32) * x_prev, axis=1, keep_dims=True)
    dx_ptr = DX_PTR + tl.arange(0, BLOCK_T)[:, None] * stride_x_t + tl.arange(
        0, D)[None, :] * stride_x_d
    tl.store(dx_ptr, dx, mask=x_mask)
    dcurr_weight_ptr = DCURR_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None
        ] * stride_weight_t
    tl.store(dcurr_weight_ptr, dcurr_weight, mask=x_mask)
    dprev_weight_ptr = DPREV_WEIGHT_PTR + tl.arange(0, BLOCK_T)[:, None
        ] * stride_weight_t
    tl.store(dprev_weight_ptr, dprev_weight, mask=x_mask)


# Backward method (kernel launch code)
def _TokenShift_backward(ctx, dout: torch.Tensor):
    x, prev_weight, curr_weight = ctx.saved_tensors
    B, T, H, D = x.size()
    assert D in {16, 32, 64, 128}
    assert prev_weight.size() == curr_weight.size() == (B, T, H)
    x = maybe_contiguous(x)
    dx = torch.empty_like(x)
    dcurr_weight = torch.empty_like(curr_weight)
    dprev_weight = torch.empty_like(prev_weight)
    assert prev_weight.stride() == curr_weight.stride() == dcurr_weight.stride(
        ) == dprev_weight.stride()
    assert dout.stride() == x.stride() == dx.stride()
    grid = lambda meta: (H, triton.cdiv(T, meta['BLOCK_T']), B)
    shift_bwd_kernel[grid](x, prev_weight, curr_weight, dout, dx,
        dprev_weight, dcurr_weight, *x.stride(), *curr_weight.stride(), T=T,
        D=D)
    return dx, dprev_weight, dcurr_weight


# ============================================================
# autograd.Function Class Definition
# ============================================================

class TokenShift(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, prev_weight: torch.Tensor,
        curr_weight: torch.Tensor):
        B, T, H, D = x.size()
        assert D in {16, 32, 64, 128}
        assert prev_weight.size() == curr_weight.size() == (B, T, H)
        assert prev_weight.stride() == curr_weight.stride()
        x = maybe_contiguous(x)
        out = torch.empty_like(x)
        assert x.stride() == out.stride()
        grid = lambda meta: (H, triton.cdiv(T, meta['BLOCK_T']), B)
        shift_fwd_kernel[grid](x, prev_weight, curr_weight, out, *x.stride(
            ), *curr_weight.stride(), T=T, D=D)
        ctx.save_for_backward(x, prev_weight, curr_weight)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        x, prev_weight, curr_weight = ctx.saved_tensors
        B, T, H, D = x.size()
        assert D in {16, 32, 64, 128}
        assert prev_weight.size() == curr_weight.size() == (B, T, H)
        x = maybe_contiguous(x)
        dx = torch.empty_like(x)
        dcurr_weight = torch.empty_like(curr_weight)
        dprev_weight = torch.empty_like(prev_weight)
        assert prev_weight.stride() == curr_weight.stride(
            ) == dcurr_weight.stride() == dprev_weight.stride()
        assert dout.stride() == x.stride() == dx.stride()
        grid = lambda meta: (H, triton.cdiv(T, meta['BLOCK_T']), B)
        shift_bwd_kernel[grid](x, prev_weight, curr_weight, dout, dx,
            dprev_weight, dcurr_weight, *x.stride(), *curr_weight.stride(),
            T=T, D=D)
        return dx, dprev_weight, dcurr_weight
