# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/OpenNLPLab/lightning-attention
# Source-Files: lightning_attn/ops/triton/srmsnorm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_dllrsnna/lightning-attention-main/lightning_attn/ops/triton/srmsnorm.py
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
import time

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def srms_norm_fw(X, Y, V, stride, N, eps, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_zm = tl.where(mask, x, 0.0)
    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)
    y = x_zm * rstd
    tl.store(V + row, rstd)
    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)


# Forward method (kernel launch code)
def __SrmsNorm_forward(ctx, x, eps):
    if x.dtype == torch.float16:
        eps = max(eps, 1.6e-05)
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    if not x_arg.is_contiguous() or not y.is_contiguous():
        x_arg = x_arg.contiguous()
        y = y.contiguous()
    num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)
    srms_norm_fw[M,](x_arg, y, rstd, x_arg.stride(0), N, eps, num_warps=
        num_warps, BLOCK_SIZE_N=BLOCK_SIZE_N)
    ctx.save_for_backward(x, rstd)
    ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
    ctx.num_warps = num_warps
    return y.reshape_as(x)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def srms_norm_bwd_dx_fused(DX, DY, X, V, stride, N, BLOCK_SIZE_N: tl.constexpr
    ):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    rstd = tl.load(V + row)
    xhat = x * rstd
    wdy = dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * mean1) * rstd
    mask = cols < N
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)


# Backward method (kernel launch code)
def __SrmsNorm_backward(ctx, dy):
    x, rstd = ctx.saved_tensors
    x = x.reshape(-1, x.size(-1))
    M, N = x.size()
    GROUP_SIZE_M = 32
    if N <= 8192:
        GROUP_SIZE_M = 64
    if N <= 4096:
        GROUP_SIZE_M = 96
    if N <= 2048:
        GROUP_SIZE_M = 128
    if N <= 1024:
        GROUP_SIZE_M = 256
    if dy.dtype == torch.float32:
        GROUP_SIZE_M = GROUP_SIZE_M // 2
    dy = dy.contiguous()
    dx = torch.empty_like(dy)
    assert dy.numel() == x.numel(
        ), 'Something is wrong in the backward graph, possibly because of an inplace operation after the layernorm'
    num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)
    srms_norm_bwd_dx_fused[M,](dx, dy, x, rstd, x.stride(0), N,
        BLOCK_SIZE_N=ctx.BLOCK_SIZE_N, num_warps=num_warps)
    dx = dx.reshape_as(dy)
    return dx, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _SrmsNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, eps):
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-05)
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        if not x_arg.is_contiguous() or not y.is_contiguous():
            x_arg = x_arg.contiguous()
            y = y.contiguous()
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)
        srms_norm_fw[M,](x_arg, y, rstd, x_arg.stride(0), N, eps, num_warps
            =num_warps, BLOCK_SIZE_N=BLOCK_SIZE_N)
        ctx.save_for_backward(x, rstd)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps
        return y.reshape_as(x)

    @staticmethod
    def backward(ctx, dy):
        x, rstd = ctx.saved_tensors
        x = x.reshape(-1, x.size(-1))
        M, N = x.size()
        GROUP_SIZE_M = 32
        if N <= 8192:
            GROUP_SIZE_M = 64
        if N <= 4096:
            GROUP_SIZE_M = 96
        if N <= 2048:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        if dy.dtype == torch.float32:
            GROUP_SIZE_M = GROUP_SIZE_M // 2
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        assert dy.numel() == x.numel(
            ), 'Something is wrong in the backward graph, possibly because of an inplace operation after the layernorm'
        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)
        srms_norm_bwd_dx_fused[M,](dx, dy, x, rstd, x.stride(0), N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N, num_warps=num_warps)
        dx = dx.reshape_as(dy)
        return dx, None, None
