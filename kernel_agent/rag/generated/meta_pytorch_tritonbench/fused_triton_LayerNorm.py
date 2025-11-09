# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/tritonbench
# Source-Files: tritonbench/operators/layer_norm/fused_triton.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ct7v_342/tritonbench-main/tritonbench/operators/layer_norm/fused_triton.py
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
from math import ceil
import time

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _layer_norm_fwd_fused_no_bias(X, Y, W, Mean, Rstd, stride, N, eps,
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
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)


# Forward method (kernel launch code)
def _LayerNorm_forward(ctx, x, normalized_shape, weight, bias, eps):
    y = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device='cuda')
    rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _layer_norm_fwd_fused_no_bias[M,](x_arg, y, weight, mean, rstd, x_arg.
        stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    ctx.save_for_backward(x, weight, mean, rstd)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.eps = eps
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'M_INCREMENT': M_INCREMENT},
    num_warps=w) for M_INCREMENT in [1, 2, 4, 8, 16] for w in [2, 4, 8]],
    key=['N'])
@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, X, W, Mean, Rstd, stride, N, M,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, M_INCREMENT: tl
    .constexpr, N_POW_2: tl.constexpr):
    pid = tl.program_id(0)
    start_row = pid * BLOCK_SIZE_M
    grad_w = tl.full([BLOCK_SIZE_N], 0, tl.float32)
    cols = tl.arange(0, BLOCK_SIZE_N)
    if N_POW_2:
        col_mask = None
    else:
        col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)[None, :]
    for cur_row in tl.range(0, BLOCK_SIZE_M, M_INCREMENT):
        rows = start_row + cur_row + tl.arange(0, M_INCREMENT)
        row_indices = rows * stride
        row_mask = rows < M
        mean = tl.load(Mean + rows, mask=row_mask).to(tl.float32)[:, None]
        rstd = tl.load(Rstd + rows, mask=row_mask).to(tl.float32)[:, None]
        if N_POW_2:
            index_mask = row_mask[:, None]
        else:
            index_mask = row_mask[:, None] & col_mask[None, :]
        indices = row_indices[:, None] + cols[None, :]
        x = tl.load(X + indices, mask=index_mask, other=0)
        x_dtype = x.dtype
        x_f32 = x.to(tl.float32)
        dy = tl.load(DY + indices, mask=index_mask, other=0).to(tl.float32)
        xhat = (x_f32 - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=1) / N
        c2 = tl.sum(wdy, axis=1) / N
        dx = (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd
        tl.store(DX + indices, dx.to(x_dtype), mask=index_mask)
        dw = dy * xhat
        partial_dw = tl.sum(dw, axis=0)
        grad_w += partial_dw
    tl.store(DW + pid * N + cols, grad_w, mask=col_mask)


# Backward method (kernel launch code)
def _LayerNorm_backward(ctx, dy):
    x, w, m, v = ctx.saved_tensors
    x_arg = x.reshape(-1, x.shape[-1])
    N = w.shape[0]
    dw = torch.empty((N,), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(dy)
    M, N = x_arg.shape
    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
    BLOCK_SIZE_M = min(2048, triton.next_power_of_2(M // (8 * NUM_SMS)))
    PARTIAL_SIZE = math.ceil(M / BLOCK_SIZE_M)
    _dw = torch.empty((PARTIAL_SIZE, N), dtype=torch.float32, device=w.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    assert ctx.BLOCK_SIZE <= MAX_FUSED_SIZE, "This layer norm doesn't support feature dim >= 64KB."
    _layer_norm_bwd_dx_fused[PARTIAL_SIZE,](dx, dy, _dw, x_arg, w, m, v,
        x_arg.stride(0), N, M, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=ctx.
        BLOCK_SIZE, N_POW_2=N % ctx.BLOCK_SIZE == 0)
    dw = torch.sum(_dw, dim=0)
    return dx, None, dw, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device='cuda')
        rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _layer_norm_fwd_fused_no_bias[M,](x_arg, y, weight, mean, rstd,
            x_arg.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
            )
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, m, v = ctx.saved_tensors
        x_arg = x.reshape(-1, x.shape[-1])
        N = w.shape[0]
        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        M, N = x_arg.shape
        NUM_SMS = torch.cuda.get_device_properties('cuda'
            ).multi_processor_count
        BLOCK_SIZE_M = min(2048, triton.next_power_of_2(M // (8 * NUM_SMS)))
        PARTIAL_SIZE = math.ceil(M / BLOCK_SIZE_M)
        _dw = torch.empty((PARTIAL_SIZE, N), dtype=torch.float32, device=w.
            device)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        assert ctx.BLOCK_SIZE <= MAX_FUSED_SIZE, "This layer norm doesn't support feature dim >= 64KB."
        _layer_norm_bwd_dx_fused[PARTIAL_SIZE,](dx, dy, _dw, x_arg, w, m, v,
            x_arg.stride(0), N, M, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=
            ctx.BLOCK_SIZE, N_POW_2=N % ctx.BLOCK_SIZE == 0)
        dw = torch.sum(_dw, dim=0)
        return dx, None, dw, None, None
