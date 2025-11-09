# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/hpcaitech/FastFold
# Source-Files: fastfold/model/fastnn/kernel/triton/layer_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_hurhnsff/FastFold-main/fastfold/model/fastnn/kernel/triton/layer_norm.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _layer_norm_fwd_fused(Out, A, Weight, Bias, Mean, Rstd, stride, N, eps,
    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Out += row * stride
    A += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0.0).to(tl.float32)
        a = tl.where(cols < N, a - mean, 0.0)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        weight = tl.load(Weight + cols, mask=mask)
        bias = tl.load(Bias + cols, mask=mask)
        a = tl.load(A + cols, mask=mask, other=0.0).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        tl.store(Out + cols, out, mask=mask)


# Forward method (kernel launch code)
def _LayerNormTritonFunc_forward(ctx, a_raw, normalized_shape, weight, bias,
    eps):
    a = a_raw.contiguous()
    out = torch.empty_like(a)
    a_arg = a.reshape(-1, a.shape[-1])
    M, N = a_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device='cuda')
    rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
    MAX_FUSED_SIZE = 65536 // a.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _layer_norm_fwd_fused[M,](out, a_arg, weight, bias, mean, rstd, a_arg.
        stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    ctx.save_for_backward(a, weight, bias, mean, rstd)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.eps = eps
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _layer_norm_bwd_dwdb(A, DOut, Mean, Var, DW, DB, M, N, BLOCK_SIZE_M: tl
    .constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    UNROLL: tl.constexpr = 4
    for i in range(0, M, BLOCK_SIZE_M * UNROLL):
        for j in range(UNROLL):
            rows = i + j * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            a = tl.load(A + offs, mask=mask, other=0.0).to(tl.float32)
            dout = tl.load(DOut + offs, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(Mean + rows, mask=rows < M, other=0.0)
            rstd = tl.load(Var + rows, mask=rows < M, other=0.0)
            a_hat = (a - mean[:, None]) * rstd[:, None]
            dw += dout * a_hat
            db += dout
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(DW + cols, sum_dw, mask=cols < N)
    tl.store(DB + cols, sum_db, mask=cols < N)


@triton.jit
def _layer_norm_bwd_dx_fused(_DA, _DOut, _A, Weight, Mean, Rstd, stride,
    NumRows, NumCols, eps, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    row = pid
    A = _A + row * stride
    DOut = _DOut + row * stride
    DA = _DA + row * stride
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    _mean1 = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    _mean2 = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, NumCols, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < NumCols
        a = tl.load(A + cols, mask=mask, other=0).to(tl.float32)
        dout = tl.load(DOut + cols, mask=mask, other=0).to(tl.float32)
        weight = tl.load(Weight + cols, mask=mask, other=0).to(tl.float32)
        a_hat = (a - mean) * rstd
        wdout = weight * dout
        _mean1 += a_hat * wdout
        _mean2 += wdout
    mean1 = tl.sum(_mean1, axis=0) / NumCols
    mean2 = 0.0
    mean2 = tl.sum(_mean2, axis=0) / NumCols
    for off in range(0, NumCols, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < NumCols
        a = tl.load(A + cols, mask=mask, other=0).to(tl.float32)
        dout = tl.load(DOut + cols, mask=mask, other=0).to(tl.float32)
        weight = tl.load(Weight + cols, mask=mask, other=0).to(tl.float32)
        a_hat = (a - mean) * rstd
        wdout = weight * dout
        da = (wdout - (a_hat * mean1 + mean2)) * rstd
        tl.store(DA + cols, da, mask=mask)


# Backward method (kernel launch code)
def _LayerNormTritonFunc_backward(ctx, dout):
    assert dout.is_contiguous()
    a, weight, bias, mean, var = ctx.saved_tensors
    N = weight.shape[0]
    da = torch.empty_like(dout)
    x_arg = a.reshape(-1, a.shape[-1])
    M, N = x_arg.shape
    dweight = torch.empty((weight.shape[0],), dtype=weight.dtype, device=
        weight.device)
    dbias = torch.empty((weight.shape[0],), dtype=weight.dtype, device=
        weight.device)
    _layer_norm_bwd_dx_fused[M,](da, dout, a, weight, mean, var, x_arg.
        stride(0), M, N, ctx.eps, BLOCK_SIZE_N=ctx.BLOCK_SIZE, num_warps=
        ctx.num_warps)
    if N > 10240:
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_M = 32
        num_warps = 4
    if N > 384:
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_M = 16
        num_warps = 8
    else:
        BLOCK_SIZE_N = 4
        BLOCK_SIZE_M = 256
        num_warps = 8
    grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
    _layer_norm_bwd_dwdb[grid](a, dout, mean, var, dweight, dbias, M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps=
        num_warps)
    return da, None, dweight, dbias, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNormTritonFunc(torch.autograd.Function):

    def forward(ctx, a_raw, normalized_shape, weight, bias, eps):
        a = a_raw.contiguous()
        out = torch.empty_like(a)
        a_arg = a.reshape(-1, a.shape[-1])
        M, N = a_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
        MAX_FUSED_SIZE = 65536 // a.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        BLOCK_SIZE = max(BLOCK_SIZE, 128)
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _layer_norm_fwd_fused[M,](out, a_arg, weight, bias, mean, rstd,
            a_arg.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
            )
        ctx.save_for_backward(a, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        assert dout.is_contiguous()
        a, weight, bias, mean, var = ctx.saved_tensors
        N = weight.shape[0]
        da = torch.empty_like(dout)
        x_arg = a.reshape(-1, a.shape[-1])
        M, N = x_arg.shape
        dweight = torch.empty((weight.shape[0],), dtype=weight.dtype,
            device=weight.device)
        dbias = torch.empty((weight.shape[0],), dtype=weight.dtype, device=
            weight.device)
        _layer_norm_bwd_dx_fused[M,](da, dout, a, weight, mean, var, x_arg.
            stride(0), M, N, ctx.eps, BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps)
        if N > 10240:
            BLOCK_SIZE_N = 128
            BLOCK_SIZE_M = 32
            num_warps = 4
        if N > 384:
            BLOCK_SIZE_N = 16
            BLOCK_SIZE_M = 16
            num_warps = 8
        else:
            BLOCK_SIZE_N = 4
            BLOCK_SIZE_M = 256
            num_warps = 8
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layer_norm_bwd_dwdb[grid](a, dout, mean, var, dweight, dbias, M, N,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, num_warps
            =num_warps)
        return da, None, dweight, dbias, None
