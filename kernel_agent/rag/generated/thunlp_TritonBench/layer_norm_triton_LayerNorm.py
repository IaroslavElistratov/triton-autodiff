# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/thunlp/TritonBench
# Source-Files: data/TritonBench_G_v1/layer_norm_triton.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_chw4t45n/TritonBench-main/data/TritonBench_G_v1/layer_norm_triton.py
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
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps,
    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


# Forward method (kernel launch code)
def _LayerNorm_forward(ctx, x, normalized_shape, weight, bias, eps):
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _layer_norm_fwd_fused[M,](x_arg, y, weight, bias, mean, rstd, x_arg.
        stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        num_ctas=1)
    ctx.save_for_backward(x, weight, bias, mean, rstd)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.eps = eps
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M: tl
    .constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, Mean, Rstd, Lock, stride,
    N, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
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
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = dy.to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)


# Backward method (kernel launch code)
def _LayerNorm_backward(ctx, dy):
    x, w, b, m, v = ctx.saved_tensors
    N = w.shape[0]
    GROUP_SIZE_M = 64
    if N <= 8192:
        GROUP_SIZE_M = 96
    if N <= 4096:
        GROUP_SIZE_M = 128
    if N <= 1024:
        GROUP_SIZE_M = 256
    locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
    _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
    _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
    dw = torch.empty((N,), dtype=w.dtype, device=w.device)
    db = torch.empty((N,), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(dy)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    _layer_norm_bwd_dx_fused[M,](dx, dy, _dw, _db, x, w, m, v, locks, x_arg
        .stride(0), N, BLOCK_SIZE_N=ctx.BLOCK_SIZE, GROUP_SIZE_M=
        GROUP_SIZE_M, num_warps=ctx.num_warps)
    grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
    _layer_norm_bwd_dwdb[grid](_dw, _db, dw, db, min(GROUP_SIZE_M, M), N,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_ctas=1)
    return dx, None, dw, db, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _layer_norm_fwd_fused[M,](x_arg, y, weight, bias, mean, rstd, x_arg
            .stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.
            device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        db = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[M,](dx, dy, _dw, _db, x, w, m, v, locks,
            x_arg.stride(0), N, BLOCK_SIZE_N=ctx.BLOCK_SIZE, GROUP_SIZE_M=
            GROUP_SIZE_M, num_warps=ctx.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layer_norm_bwd_dwdb[grid](_dw, _db, dw, db, min(GROUP_SIZE_M, M),
            N, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None
