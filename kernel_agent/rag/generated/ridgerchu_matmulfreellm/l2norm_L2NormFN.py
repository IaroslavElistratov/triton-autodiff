# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/ridgerchu/matmulfreellm
# Source-Files: mmfreelm/modules/l2norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_vy7sa4r5/matmulfreellm-master/mmfreelm/modules/l2norm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N'])
@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: tl.constexpr
    ):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y, mask=mask)


def _l2_norm_fwd(x, eps=1e-06):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
        M, N = x.shape
    assert x.stride(-1) == 1
    y = torch.empty_like(x)
    assert y.stride(-1) == 1
    N = x.shape[-1]
    M = x.shape[0]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _l2_norm_fwd_1pass_kernel[M,](x, y, x.stride(0), N, eps, BLOCK_N)
    return y.reshape(x_shape_og)


# Forward method (kernel launch code)
def _L2NormFN_forward(ctx, x, eps=1e-06):
    y = _l2_norm_fwd(x, eps)
    ctx.x_shape_og = x_shape_og
    ctx.eps = eps
    ctx.x_dtype = x.dtype
    ctx.save_for_backward(x)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N'])
@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, stride_x_row, N, eps, BLOCK_N: tl.constexpr
    ):
    row = tl.program_id(0)
    X += row * stride_x_row
    DX += row * stride_x_row
    DY += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)


def _l2_norm_bwd(x, dy, eps=1e-05):
    x_shape_og = x.shape
    x = x.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    assert dy.shape == x.shape
    dx = torch.empty_like(x)
    N = x.shape[-1]
    M = x.shape[0]
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _l2_norm_bwd_kernel[M,](x, dy, dx, x.stride(0), N, eps, BLOCK_N)
    return dx.reshape(x_shape_og)


# Backward method (kernel launch code)
def _L2NormFN_backward(ctx, dy, *args):
    x, = ctx.saved_tensors
    dx = _l2_norm_bwd(x, dy, ctx.eps)
    return dx, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class L2NormFN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, eps=1e-06):
        y = _l2_norm_fwd(x, eps)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy, *args):
        x, = ctx.saved_tensors
        dx = _l2_norm_bwd(x, dy, ctx.eps)
        return dx, None
