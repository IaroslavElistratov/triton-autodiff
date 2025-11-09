# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/multi_token_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/multi_token_attention.py
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


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _mask_fwd_kernel(scores_ptr, out_ptr, stride_b, stride_m, stride_n, L,
    mask_val: tl.constexpr, BLOCK: tl.constexpr, num_warps: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)
    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)
    base = scores_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    future = col_idx[None, :] > row_idx[:, None]
    mask_load = in_bounds & ~future
    out = tl.load(base + offs, mask=mask_load, other=mask_val,
        cache_modifier='.ca')
    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds,
        cache_modifier='.cs')


def _mask_inf_forward(scores: torch.Tensor) ->torch.Tensor:
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.view(N, L, L)
    out = torch.empty_like(scores_f)
    sb, sm, sn = scores_f.stride(0), scores_f.stride(1), scores_f.stride(2)
    BLOCK_SIZE, num_warps = calculate_settings(L)
    grid = triton.cdiv(L, BLOCK_SIZE), triton.cdiv(L, BLOCK_SIZE), N
    _mask_fwd_kernel[grid](scores_f, out, sb, sm, sn, L, mask_val=-
        1000000000.0, BLOCK=BLOCK_SIZE, num_warps=num_warps)
    return out.view(*batch, L, L)


def _mask_zero_forward(scores: torch.Tensor) ->torch.Tensor:
    *batch, L, _ = scores.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    scores_f = scores.view(N, L, L)
    out = torch.empty_like(scores_f)
    sb, sm, sn = scores_f.stride(0), scores_f.stride(1), scores_f.stride(2)
    BLOCK_SIZE, num_warps = calculate_settings(L)
    grid = triton.cdiv(L, BLOCK_SIZE), triton.cdiv(L, BLOCK_SIZE), N
    _mask_fwd_kernel[grid](scores_f, out, sb, sm, sn, L, mask_val=0.0,
        BLOCK=BLOCK_SIZE, num_warps=num_warps)
    return out.view(*batch, L, L)


# Forward method (kernel launch code)
@ensure_contiguous
def _LigerMultiTokenAttentionFunction_forward(ctx, scores, weight, bias=
    None, stride=1, padding=0, dilation=1, groups=1, sparse=False):
    scores_inf = _mask_inf_forward(scores)
    out_flat_sparse = None
    activation_output = None
    ctx.sparse = sparse
    if sparse:
        if scores_inf.dtype != torch.float32:
            raise RuntimeError(
                'Liger sparse multi-token attention currently only supports fp32 input scores'
                )
        probs_sparse, out_flat_sparse = _sparsemax_forward(scores_inf, dim=-1)
        activation_output = probs_sparse
        ctx.save_for_backward(scores_inf, activation_output,
            out_flat_sparse, weight, bias)
        ctx.out_flat_sparse_saved = True
    else:
        probs_softmax, _, _, _ = _softmax_forward(scores_inf)
        activation_output = probs_softmax
        ctx.save_for_backward(scores_inf, activation_output, weight, bias)
        ctx.out_flat_sparse_saved = False
    out_conv = F.conv2d(activation_output, weight, bias, stride=stride,
        padding=padding, dilation=dilation, groups=groups)
    out = _mask_zero_forward(out_conv)
    ctx.stride = _pair(stride)
    ctx.padding = _pair(padding)
    ctx.dilation = _pair(dilation)
    ctx.groups = groups
    ctx.dim = -1
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _mask_bwd_kernel(grad_in_ptr, out_ptr, stride_b, stride_m, stride_n, L,
    BLOCK: tl.constexpr, num_warps: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch_id = tl.program_id(2)
    row_idx = row_block * BLOCK + tl.arange(0, BLOCK)
    col_idx = col_block * BLOCK + tl.arange(0, BLOCK)
    in_bounds = (row_idx[:, None] < L) & (col_idx[None, :] < L)
    base = grad_in_ptr + batch_id * stride_b
    offs = row_idx[:, None] * stride_m + col_idx[None, :] * stride_n
    grad_vals = tl.load(base + offs, mask=in_bounds, other=0.0,
        cache_modifier='.ca')
    future = col_idx[None, :] > row_idx[:, None]
    zero = tl.zeros(grad_vals.shape, dtype=grad_vals.dtype)
    out = tl.where(future, zero, grad_vals)
    tl.store(out_ptr + batch_id * stride_b + offs, out, mask=in_bounds,
        cache_modifier='.wb')


def _mask_inf_backward(grad: torch.Tensor) ->torch.Tensor:
    *batch, L, _ = grad.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    grad_f = grad.view(N, L, L)
    out = torch.empty_like(grad_f)
    sb, sm, sn = grad_f.stride(0), grad_f.stride(1), grad_f.stride(2)
    BLOCK_SIZE, num_warps = calculate_settings(L)
    grid = triton.cdiv(L, BLOCK_SIZE), triton.cdiv(L, BLOCK_SIZE), N
    _mask_bwd_kernel[grid](grad_f, out, sb, sm, sn, L, BLOCK=BLOCK_SIZE,
        num_warps=num_warps)
    return out.view(*batch, L, L)


def _mask_zero_backward(grad: torch.Tensor) ->torch.Tensor:
    *batch, L, _ = grad.shape
    N = int(torch.prod(torch.tensor(batch))) if batch else 1
    grad_f = grad.view(N, L, L)
    out = torch.empty_like(grad_f)
    sb, sm, sn = grad_f.stride(0), grad_f.stride(1), grad_f.stride(2)
    BLOCK_SIZE, num_warps = calculate_settings(L)
    grid = triton.cdiv(L, BLOCK_SIZE), triton.cdiv(L, BLOCK_SIZE), N
    _mask_bwd_kernel[grid](grad_f, out, sb, sm, sn, L, BLOCK=BLOCK_SIZE,
        num_warps=num_warps)
    return out.view(*batch, L, L)


# Backward method (kernel launch code)
@ensure_contiguous
def _LigerMultiTokenAttentionFunction_backward(ctx, grad_out):
    if ctx.out_flat_sparse_saved:
        scores_inf, activation_output, out_flat_sparse, weight, bias = (ctx
            .saved_tensors)
    else:
        scores_inf, activation_output, weight, bias = ctx.saved_tensors
        out_flat_sparse = None
    use_sparsemax = ctx.sparse
    dim = ctx.dim
    stride, padding, dilation, groups = (ctx.stride, ctx.padding, ctx.
        dilation, ctx.groups)
    grad_conv = _mask_zero_backward(grad_out)
    grad_probs = F.conv_transpose2d(grad_conv, weight, None, stride=stride,
        padding=padding, dilation=dilation, groups=groups)
    grad_weight = torch.nn.grad.conv2d_weight(input=activation_output,
        weight_size=weight.shape, grad_output=grad_conv, stride=stride,
        padding=padding, dilation=dilation, groups=groups)
    grad_bias = None
    if bias is not None:
        grad_bias = grad_conv.sum(dim=(0, 2, 3))
    grad_scores_inf = None
    if use_sparsemax:
        if not ctx.out_flat_sparse_saved or out_flat_sparse is None:
            raise RuntimeError(
                'Internal error: Sparse flag is set but sparse tensor was not saved.'
                )
        grad_scores_inf = _sparsemax_backward(grad_probs, out_flat_sparse,
            dim=dim)
    else:
        grad_probs_cont = grad_probs
        probs_cont = activation_output
        dot = (grad_probs_cont * probs_cont).sum(dim=-1, keepdim=True)
        grad_scores_inf = probs_cont * (grad_probs_cont - dot)
    grad_scores = _mask_inf_backward(grad_scores_inf)
    return grad_scores, grad_weight, grad_bias, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerMultiTokenAttentionFunction(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, scores, weight, bias=None, stride=1, padding=0,
        dilation=1, groups=1, sparse=False):
        scores_inf = _mask_inf_forward(scores)
        out_flat_sparse = None
        activation_output = None
        ctx.sparse = sparse
        if sparse:
            if scores_inf.dtype != torch.float32:
                raise RuntimeError(
                    'Liger sparse multi-token attention currently only supports fp32 input scores'
                    )
            probs_sparse, out_flat_sparse = _sparsemax_forward(scores_inf,
                dim=-1)
            activation_output = probs_sparse
            ctx.save_for_backward(scores_inf, activation_output,
                out_flat_sparse, weight, bias)
            ctx.out_flat_sparse_saved = True
        else:
            probs_softmax, _, _, _ = _softmax_forward(scores_inf)
            activation_output = probs_softmax
            ctx.save_for_backward(scores_inf, activation_output, weight, bias)
            ctx.out_flat_sparse_saved = False
        out_conv = F.conv2d(activation_output, weight, bias, stride=stride,
            padding=padding, dilation=dilation, groups=groups)
        out = _mask_zero_forward(out_conv)
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.dim = -1
        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out):
        if ctx.out_flat_sparse_saved:
            (scores_inf, activation_output, out_flat_sparse, weight, bias
                ) = ctx.saved_tensors
        else:
            scores_inf, activation_output, weight, bias = ctx.saved_tensors
            out_flat_sparse = None
        use_sparsemax = ctx.sparse
        dim = ctx.dim
        stride, padding, dilation, groups = (ctx.stride, ctx.padding, ctx.
            dilation, ctx.groups)
        grad_conv = _mask_zero_backward(grad_out)
        grad_probs = F.conv_transpose2d(grad_conv, weight, None, stride=
            stride, padding=padding, dilation=dilation, groups=groups)
        grad_weight = torch.nn.grad.conv2d_weight(input=activation_output,
            weight_size=weight.shape, grad_output=grad_conv, stride=stride,
            padding=padding, dilation=dilation, groups=groups)
        grad_bias = None
        if bias is not None:
            grad_bias = grad_conv.sum(dim=(0, 2, 3))
        grad_scores_inf = None
        if use_sparsemax:
            if not ctx.out_flat_sparse_saved or out_flat_sparse is None:
                raise RuntimeError(
                    'Internal error: Sparse flag is set but sparse tensor was not saved.'
                    )
            grad_scores_inf = _sparsemax_backward(grad_probs,
                out_flat_sparse, dim=dim)
        else:
            grad_probs_cont = grad_probs
            probs_cont = activation_output
            dot = (grad_probs_cont * probs_cont).sum(dim=-1, keepdim=True)
            grad_scores_inf = probs_cont * (grad_probs_cont - dot)
        grad_scores = _mask_inf_backward(grad_scores_inf)
        return (grad_scores, grad_weight, grad_bias, None, None, None, None,
            None)
