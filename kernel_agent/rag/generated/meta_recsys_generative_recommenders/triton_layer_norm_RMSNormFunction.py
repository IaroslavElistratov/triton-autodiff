# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/triton/triton_layer_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/triton/triton_layer_norm.py
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
def _weighted_rms_norm_fwd(X, Y, W, Rstd, D, eps, stride_x, stride_y,
    BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    y = y * w
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


# Forward method (kernel launch code)
def _RMSNormFunction_forward(ctx, x: torch.Tensor, weight: torch.Tensor,
    eps: float) ->torch.Tensor:
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1
    assert weight.numel() == D
    y = torch.empty_like(x)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_D // 256, 1), 8)
    _weighted_rms_norm_fwd[N,](x, y, weight, rstd, D, eps, x.stride(0), y.
        stride(0), BLOCK_D=BLOCK_D, num_warps=num_warps)
    ctx.save_for_backward(x, weight, rstd)
    ctx.BLOCK_D = BLOCK_D
    ctx.num_warps = num_warps
    ctx.eps = eps
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton_autotune(configs=_get_bwd_dwdb_configs(), key=['D'])
@triton.jit
def _rms_norm_bwd_dwdb(DW, FINAL_DW, N, D, BLOCK_N: tl.constexpr, BLOCK_D:
    tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=
        cols < D)


@triton.jit
def _weighted_rms_norm_bwd_dx(DX, DY, DW, X, W, Rstd, Lock, stride_dx,
    stride_dy, stride_x, D, eps, GROUP_N, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd + row)
    xhat = x * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / D
    dx = (wdy - xhat * c1) * rstd
    tl.store(DX + cols, dx, mask=mask)
    lock_id = row % GROUP_N
    Lock += lock_id
    Count = Lock + GROUP_N
    DW = DW + lock_id * D + cols
    partial_dw = dy * xhat
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
def _RMSNormFunction_backward(ctx, dy: torch.Tensor) ->Tuple[torch.Tensor,
    Optional[torch.Tensor], None]:
    x, weight, rstd = ctx.saved_tensors
    N, D = x.shape
    dx = torch.empty_like(x)
    if D <= 1024:
        GROUP_N = 256 * 8
    elif D <= 4096:
        GROUP_N = 128 * 8
    elif D <= 8192:
        GROUP_N = 96 * 8
    else:
        GROUP_N = 64 * 8
    GROUP_N = N if GROUP_N > N else GROUP_N
    locks = torch.zeros(2 * GROUP_N, dtype=torch.int32, device=x.device)
    _dweight = torch.empty((GROUP_N, D), dtype=torch.float32, device=x.device)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    _weighted_rms_norm_bwd_dx[N,](dx, dy, _dweight, x, weight, rstd, locks,
        dx.stride(0), dy.stride(0), x.stride(0), D, ctx.eps, GROUP_N=
        GROUP_N, BLOCK_D=ctx.BLOCK_D, num_warps=ctx.num_warps)

    def grid(META):
        return triton.cdiv(D, META['BLOCK_D']),
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    blocks = triton.next_power_of_2(sms * 4)
    BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
    BLOCK_D = min(max(BLOCK_D, 4), 128)
    _rms_norm_bwd_dwdb[grid](_dweight, dweight, GROUP_N, D, BLOCK_D=BLOCK_D)
    return dx, dweight, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float
        ) ->torch.Tensor:
        assert x.dim() == 2
        x = switch_to_contiguous_if_needed(x)
        N, D = x.shape
        assert weight.dim() == 1
        assert weight.numel() == D
        y = torch.empty_like(x)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
        if D > BLOCK_D:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_D // 256, 1), 8)
        _weighted_rms_norm_fwd[N,](x, y, weight, rstd, D, eps, x.stride(0),
            y.stride(0), BLOCK_D=BLOCK_D, num_warps=num_warps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) ->Tuple[torch.Tensor, Optional[
        torch.Tensor], None]:
        x, weight, rstd = ctx.saved_tensors
        N, D = x.shape
        dx = torch.empty_like(x)
        if D <= 1024:
            GROUP_N = 256 * 8
        elif D <= 4096:
            GROUP_N = 128 * 8
        elif D <= 8192:
            GROUP_N = 96 * 8
        else:
            GROUP_N = 64 * 8
        GROUP_N = N if GROUP_N > N else GROUP_N
        locks = torch.zeros(2 * GROUP_N, dtype=torch.int32, device=x.device)
        _dweight = torch.empty((GROUP_N, D), dtype=torch.float32, device=x.
            device)
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        _weighted_rms_norm_bwd_dx[N,](dx, dy, _dweight, x, weight, rstd,
            locks, dx.stride(0), dy.stride(0), x.stride(0), D, ctx.eps,
            GROUP_N=GROUP_N, BLOCK_D=ctx.BLOCK_D, num_warps=ctx.num_warps)

        def grid(META):
            return triton.cdiv(D, META['BLOCK_D']),
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _rms_norm_bwd_dwdb[grid](_dweight, dweight, GROUP_N, D, BLOCK_D=BLOCK_D
            )
        return dx, dweight, None
