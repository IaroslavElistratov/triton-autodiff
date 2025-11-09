# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/softmax.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/softmax.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _softmax_multi_block_forward_kernel(Y_ptr, Y_row_stride, X_ptr,
    X_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    m = tl.float32(-float('inf'))
    d = tl.float32(0.0)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask,
            other=-float('inf'), cache_modifier='.ca')
        blk_max = tl.max(xblk, axis=0)
        new_m = tl.max(m, blk_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(xblk - new_m), axis=0)
        m = new_m
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        xblk = tl.load(X_ptr + row_id * X_row_stride + idx, mask=mask,
            other=-float('inf'), cache_modifier='.ca')
        yblk = tl.exp(xblk - m) / d
        tl.store(Y_ptr + row_id * Y_row_stride + idx, yblk, mask=mask,
            cache_modifier='.cs')


@triton.jit
def _softmax_single_block_forward_kernel(Y_ptr, Y_row_stride, X_ptr,
    X_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(X_ptr + row_id * X_row_stride + offs, mask=mask, other=-
        float('inf'), cache_modifier='.ca')
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    d = tl.sum(e, axis=0)
    y = e / d
    tl.store(Y_ptr + row_id * Y_row_stride + offs, y, mask=mask,
        cache_modifier='.cs')


def _softmax_forward(x: torch.Tensor) ->Tuple[torch.Tensor, int, int, bool]:
    *batch, n_cols = x.shape
    x2d = x.contiguous().view(-1, n_cols)
    n_rows = x2d.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    y2d = torch.empty_like(x2d)
    if n_cols <= BLOCK_SIZE:
        _softmax_single_block_forward_kernel[n_rows,](y2d, y2d.stride(0),
            x2d, x2d.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=
            num_warps)
        multi_block_launch = False
    else:
        _softmax_multi_block_forward_kernel[n_rows,](y2d, y2d.stride(0),
            x2d, x2d.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=
            num_warps)
        multi_block_launch = True
    return y2d.view(*batch, n_cols), BLOCK_SIZE, num_warps, multi_block_launch


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


# Forward method (kernel launch code)
@ensure_contiguous
def _LigerSoftmaxFunction_forward(ctx, input_: torch.Tensor):
    y, BLOCK_SIZE, num_warps, multi_block_launch = _softmax_forward(input_)
    ctx.save_for_backward(y)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.multi_block_launch = multi_block_launch
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _softmax_multi_block_backward_kernel(dy_ptr, dy_stride, y_ptr, y_stride,
    dx_ptr, dx_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    acc = tl.float32(0.0)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask,
            other=0.0)
        y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=
            0.0, cache_modifier='.ca')
        acc += tl.sum(dy_blk * y_blk, axis=0)
    for start in tl.range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        dy_blk = tl.load(dy_ptr + row_id * dy_stride + idx, mask=mask,
            other=0.0)
        y_blk = tl.load(y_ptr + row_id * y_stride + idx, mask=mask, other=
            0.0, cache_modifier='.ca')
        dx_blk = y_blk * (dy_blk - acc)
        tl.store(dx_ptr + row_id * dx_stride + idx, dx_blk, mask=mask,
            cache_modifier='.wb')


@triton.jit
def _softmax_single_block_backward_kernel(dy_ptr, dy_stride, y_ptr,
    y_stride, dx_ptr, dx_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    dy = tl.load(dy_ptr + row_id * dy_stride + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + row_id * y_stride + offs, mask=mask, other=0.0,
        cache_modifier='.ca')
    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)
    tl.store(dx_ptr + row_id * dx_stride + offs, dx, mask=mask,
        cache_modifier='.wb')


def _softmax_backward(dy: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int,
    num_warps: int, multi_block_launch: bool) ->torch.Tensor:
    *batch, n_cols = dy.shape
    dy2d = dy.contiguous().view(-1, n_cols)
    y2d = y.contiguous().view(-1, n_cols)
    n_rows = dy2d.shape[0]
    dx2d = torch.empty_like(dy2d)
    if not multi_block_launch and n_cols <= BLOCK_SIZE:
        _softmax_single_block_backward_kernel[n_rows,](dy2d, dy2d.stride(0),
            y2d, y2d.stride(0), dx2d, dx2d.stride(0), n_cols, BLOCK_SIZE=
            BLOCK_SIZE, num_warps=num_warps)
    else:
        _softmax_multi_block_backward_kernel[n_rows,](dy2d, dy2d.stride(0),
            y2d, y2d.stride(0), dx2d, dx2d.stride(0), n_cols, BLOCK_SIZE=
            BLOCK_SIZE, num_warps=num_warps)
    return dx2d.view(*batch, n_cols)


# Backward method (kernel launch code)
@ensure_contiguous
def _LigerSoftmaxFunction_backward(ctx, grad_output):
    y, = ctx.saved_tensors
    dx = _softmax_backward(grad_output, y, ctx.BLOCK_SIZE, ctx.num_warps,
        ctx.multi_block_launch)
    return dx


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerSoftmaxFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, input_: torch.Tensor):
        y, BLOCK_SIZE, num_warps, multi_block_launch = _softmax_forward(input_)
        ctx.save_for_backward(y)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.multi_block_launch = multi_block_launch
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        dx = _softmax_backward(grad_output, y, ctx.BLOCK_SIZE, ctx.
            num_warps, ctx.multi_block_launch)
        return dx
