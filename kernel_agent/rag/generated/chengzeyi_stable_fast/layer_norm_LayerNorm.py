# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/chengzeyi/stable-fast
# Source-Files: src/sfast/triton/ops/layer_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_4r10uf69/stable-fast-main/src/sfast/triton/ops/layer_norm.py
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
from einops import reduce
import time

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride: tl.constexpr, N:
    tl.constexpr, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N).to(tl.float32)
        m2_ = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        weight_ = (cols < N).to(tl.float32)
        _mean, _m2, _weight = x, m2_, weight_
    else:
        _mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        _m2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        _weight = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N).to(tl.float32)
            m2_ = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            weight_ = (cols < N).to(tl.float32)
            if off == 0:
                _mean, _m2, _weight = x, m2_, weight_
            else:
                _mean, _m2, _weight = welford_combine(_mean, _m2, _weight,
                    x, m2_, weight_)
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1 / tl.sqrt(var + eps)
    mean = mean.to(x.dtype)
    rstd = rstd.to(x.dtype)
    if Mean is not None:
        tl.store(Mean + row, mean)
    if Rstd is not None:
        tl.store(Rstd + row, rstd)
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if W is None:
            w = tl.full((BLOCK_SIZE,), 1.0, dtype=x.dtype)
        else:
            w = tl.load(W + cols, mask=mask)
        if B is None:
            b = tl.zeros((BLOCK_SIZE,), dtype=x.dtype)
        else:
            b = tl.load(B + cols, mask=mask)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
    else:
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            if W is None:
                w = tl.full((BLOCK_SIZE,), 1.0, dtype=x.dtype)
            else:
                w = tl.load(W + cols, mask=mask)
            if B is None:
                b = tl.zeros((BLOCK_SIZE,), dtype=x.dtype)
            else:
                b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            tl.store(Y + cols, y, mask=mask)


@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (mean_1 + delta * w2_over_w, m2_1 + m2_2 + delta * delta *
        weight_1 * w2_over_w, new_weight)


# Forward method (kernel launch code)
def _LayerNorm_forward(ctx, x, normalized_shape, weight, bias, eps):
    x = x.contiguous()
    weight = weight.contiguous() if weight is not None else None
    bias = bias.contiguous() if bias is not None else None
    y = torch.empty_like(x)
    N = functools.reduce(operator.mul, normalized_shape, 1)
    x_arg = x.reshape(-1, N)
    M, N = x_arg.shape
    needs_backward = any(x is not None and x.requires_grad for x in [x,
        weight, bias])
    if needs_backward:
        mean = torch.empty((M,), dtype=x.dtype, device=x.device)
        rstd = torch.empty((M,), dtype=x.dtype, device=x.device)
    else:
        mean, rstd = None, None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 16)
    _layer_norm_fwd_fused[M,](x_arg, y, weight, bias, mean, rstd, x_arg.
        stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    ctx.save_for_backward(x, weight, bias, mean, rstd)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.eps = eps
    ctx.normalized_shape = normalized_shape
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _LayerNorm_backward(ctx, dy):
    dy.contiguous()
    x, w, b, m, v = ctx.saved_tensors
    x = x.contiguous()
    w = w.contiguous() if w is not None else None
    b = b.contiguous() if b is not None else None
    m = m.contiguous()
    v = v.contiguous()
    grad_input_mask = ctx.needs_input_grad[0], ctx.needs_input_grad[2
        ], ctx.needs_input_grad[3]
    grad_inputs = aten.native_layer_norm_backward(dy, x, ctx.
        normalized_shape, m, v, w, b, grad_input_mask)
    dx, dw, db = grad_inputs
    return dx, None, dw, db, None
    M = m.numel()
    N = x.numel() // M
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
    _db = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.
        device)
    dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
    db = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(dy)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    _layer_norm_bwd_dx_fused[M,](dx, dy, _dw, _db, x, w, b, m, v, locks,
        x_arg.stride(0), N, ctx.eps, BLOCK_SIZE_N=ctx.BLOCK_SIZE,
        GROUP_SIZE_M=GROUP_SIZE_M, num_warps=ctx.num_warps)

    def grid(meta):
        return [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
    _layer_norm_bwd_dwdb[grid](_dw, _db, dw, db, GROUP_SIZE_M, N,
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=128)
    return dx, None, dw, db, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        x = x.contiguous()
        weight = weight.contiguous() if weight is not None else None
        bias = bias.contiguous() if bias is not None else None
        y = torch.empty_like(x)
        N = functools.reduce(operator.mul, normalized_shape, 1)
        x_arg = x.reshape(-1, N)
        M, N = x_arg.shape
        needs_backward = any(x is not None and x.requires_grad for x in [x,
            weight, bias])
        if needs_backward:
            mean = torch.empty((M,), dtype=x.dtype, device=x.device)
            rstd = torch.empty((M,), dtype=x.dtype, device=x.device)
        else:
            mean, rstd = None, None
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError(
                "This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 16)
        _layer_norm_fwd_fused[M,](x_arg, y, weight, bias, mean, rstd, x_arg
            .stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.normalized_shape = normalized_shape
        return y

    @staticmethod
    def backward(ctx, dy):
        dy.contiguous()
        x, w, b, m, v = ctx.saved_tensors
        x = x.contiguous()
        w = w.contiguous() if w is not None else None
        b = b.contiguous() if b is not None else None
        m = m.contiguous()
        v = v.contiguous()
        grad_input_mask = ctx.needs_input_grad[0], ctx.needs_input_grad[2
            ], ctx.needs_input_grad[3]
        grad_inputs = aten.native_layer_norm_backward(dy, x, ctx.
            normalized_shape, m, v, w, b, grad_input_mask)
        dx, dw, db = grad_inputs
        return dx, None, dw, db, None
        M = m.numel()
        N = x.numel() // M
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
        _db = torch.empty((GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device
            =w.device)
        dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        db = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[M,](dx, dy, _dw, _db, x, w, b, m, v, locks,
            x_arg.stride(0), N, ctx.eps, BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M, num_warps=ctx.num_warps)

        def grid(meta):
            return [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layer_norm_bwd_dwdb[grid](_dw, _db, dw, db, GROUP_SIZE_M, N,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128)
        return dx, None, dw, db, None
