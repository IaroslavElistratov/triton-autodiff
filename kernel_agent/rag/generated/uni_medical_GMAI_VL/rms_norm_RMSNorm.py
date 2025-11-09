# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/uni-medical/GMAI-VL
# Source-Files: xtuner/model/modules/dispatch/triton_kernels/rms_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_vekqljbq/GMAI-VL-main/xtuner/model/modules/dispatch/triton_kernels/rms_norm.py
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
import time

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _rms_norm_fwd_fused(X, Y, W, Rstd, stride, N, eps, BLOCK_SIZE: tl.constexpr
    ):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)


# Forward method (kernel launch code)
def _RMSNorm_forward(ctx, x, weight, eps):
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This rms norm doesn't support feature dim >= 64KB."
            )
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _rms_norm_fwd_fused[M,](x_arg, y, weight, rstd, x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    ctx.save_for_backward(x, weight, rstd)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.eps = eps
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _rms_norm_bwd_dwdb(DW, FINAL_DW, M, N, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)


@triton.jit
def _rms_norm_bwd_dx_fused(DX, DY, DW, X, W, Rstd, Lock, stride, N, eps,
    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    rstd = tl.load(Rstd + row)
    xhat = x * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * c1) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * xhat).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.atomic_xchg(Lock, 0)


# Backward method (kernel launch code)
def _RMSNorm_backward(ctx, dy):
    x, w, v = ctx.saved_tensors
    N = w.shape[0]
    GROUP_SIZE_M = 64
    if N <= 8192:
        GROUP_SIZE_M = 96
    if N <= 4096:
        GROUP_SIZE_M = 128
    if N <= 1024:
        GROUP_SIZE_M = 256
    locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
    _dw = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.
        device)
    dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(dy)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    _rms_norm_bwd_dx_fused[M,](dx, dy, _dw, x, w, v, locks, x_arg.stride(0),
        N, ctx.eps, BLOCK_SIZE_N=ctx.BLOCK_SIZE, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=ctx.num_warps)

    def grid(meta):
        return [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
    _rms_norm_bwd_dwdb[grid](_dw, dw, GROUP_SIZE_M, N, BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128)
    return dx, dw, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError(
                "This rms norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _rms_norm_fwd_fused[M,](x_arg, y, weight, rstd, x_arg.stride(0), N,
            eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, v = ctx.saved_tensors
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device='cuda')
        _dw = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device
            =w.device)
        dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _rms_norm_bwd_dx_fused[M,](dx, dy, _dw, x, w, v, locks, x_arg.
            stride(0), N, ctx.eps, BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M, num_warps=ctx.num_warps)

        def grid(meta):
            return [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _rms_norm_bwd_dwdb[grid](_dw, dw, GROUP_SIZE_M, N, BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128)
        return dx, dw, None
