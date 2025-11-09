# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/JonasGeiping/linear_cross_entropy_loss
# Source-Files: variants/malek_xent.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_0e62yj5n/linear_cross_entropy_loss-main/variants/malek_xent.py
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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def fused_cross_entropy_fwd_bwd_kernel(output_loss_ptr,
    output_logit_grad_ptr, input_logit_ptr, input_targ_ptr,
    input_divisor_ptr, output_loss_stride, output_logit_grad_stride,
    input_logit_stride, input_targ_stride, n_cols, ignore_index: tl.
    constexpr, requires_grad: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    logit_grad_row_start_ptr = (output_logit_grad_ptr + row_idx *
        output_logit_grad_stride)
    logit_row_start_ptr = input_logit_ptr + row_idx * input_logit_stride
    targ_ptr = input_targ_ptr + row_idx * input_targ_stride
    loss_ptr = output_loss_ptr + row_idx * output_loss_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    logit_row_ptrs = logit_row_start_ptr + col_offsets
    logit_grad_row_ptrs = logit_grad_row_start_ptr + col_offsets
    logit_row_unnormalized = tl.load(logit_row_ptrs, mask=col_offsets <
        n_cols, other=float('-Inf'))
    targ = tl.load(targ_ptr)
    divisor = tl.load(input_divisor_ptr)
    logit_row = logit_row_unnormalized - tl.max(logit_row_unnormalized, axis=0)
    exp_logit_row = tl.exp(logit_row)
    sum_exp_logit_row = tl.sum(exp_logit_row, axis=0)
    log_sum_exp_logit_row = tl.log(sum_exp_logit_row)
    logit_gt_logit = tl.sum(tl.where(targ == col_offsets, logit_row, 0.0))
    loss = log_sum_exp_logit_row - logit_gt_logit
    loss = loss / divisor
    loss = tl.where(targ == ignore_index, 0.0, loss)
    tl.store(loss_ptr, loss)
    if requires_grad:
        targ_one_hot = tl.where(targ == col_offsets, 1.0, 0.0)
        grad = exp_logit_row / sum_exp_logit_row - targ_one_hot
        grad = grad / divisor
        grad = tl.where(targ == ignore_index, 0.0, grad)
        tl.store(logit_grad_row_ptrs, grad, mask=col_offsets < n_cols)


# Forward method (kernel launch code)
def _FusedCrossEntropyLossFunction_forward(ctx, in_feat: torch.Tensor,
    proj_weight: torch.Tensor, targ: torch.Tensor, n_loop_iters: int,
    ignore_index: int, reduction: str):
    n_tokens = in_feat.shape[0]
    n_classes = proj_weight.shape[0]
    assert in_feat.ndim == 2, in_feat.ndim
    assert proj_weight.ndim == 2, proj_weight.ndim
    assert targ.ndim == 1, targ.shape
    assert in_feat.shape[0] == targ.shape[0
        ], f'Number of tokens in in_feat and targ is not equal: (in_feat.shape, targ.shape) = {in_feat.shape, targ.shape!r}'
    assert reduction in ('mean', 'sum'), reduction
    assert n_loop_iters > 0, n_loop_iters
    assert n_tokens % n_loop_iters == 0, (n_tokens, n_loop_iters)
    NUM_WARPS = 16
    BLOCK_SIZE = triton.next_power_of_2(n_classes)
    loss = torch.empty(n_tokens, dtype=in_feat.dtype, device=in_feat.device)
    dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled(
        ) else in_feat.dtype
    if proj_weight.requires_grad:
        grad_proj_weight = torch.zeros_like(proj_weight, dtype=dtype)
    else:
        grad_proj_weight = None
    if in_feat.requires_grad:
        grad_in_feat = torch.zeros_like(in_feat)
    else:
        grad_in_feat = None
    divisor = (targ != ignore_index).sum().to(dtype
        ) if reduction == 'mean' else torch.ones(1, dtype=dtype, device=
        in_feat.device)
    proj_weight_cast = proj_weight.to(dtype)
    loop_chunk_size = triton.cdiv(n_tokens, n_loop_iters)
    logits_chunk_cast = torch.zeros((loop_chunk_size, n_classes), dtype=
        dtype, device=in_feat.device)
    for i, in_feat_chunk in enumerate(torch.split(in_feat, loop_chunk_size)):
        token_start_idx = i * loop_chunk_size
        token_end_idx = (i + 1) * loop_chunk_size
        in_feat_chunk = in_feat_chunk.to(dtype)
        torch.matmul(in_feat_chunk, proj_weight_cast.T, out=logits_chunk_cast)
        logits_chunk = logits_chunk_cast.float()
        loss_chunk = loss[token_start_idx:token_end_idx]
        targ_chunk = targ[token_start_idx:token_end_idx]
        n_tokens_chunk = logits_chunk.shape[0]
        grad_logits_chunk = logits_chunk
        fused_cross_entropy_fwd_bwd_kernel[n_tokens_chunk,](loss_chunk,
            grad_logits_chunk, logits_chunk, targ_chunk, divisor,
            loss_chunk.stride(0), grad_logits_chunk.stride(0), logits_chunk
            .stride(0), targ_chunk.stride(0), n_classes, ignore_index,
            requires_grad=in_feat.requires_grad or proj_weight.
            requires_grad, num_warps=NUM_WARPS, BLOCK_SIZE=BLOCK_SIZE)
        grad_logits_chunk = grad_logits_chunk.to(dtype)
        if in_feat.requires_grad:
            grad_in_feat[token_start_idx:token_end_idx
                ] = grad_logits_chunk @ proj_weight_cast
        if proj_weight.requires_grad:
            torch.addmm(grad_proj_weight, grad_logits_chunk.T,
                in_feat_chunk, out=grad_proj_weight)
    loss = loss.sum()
    ctx.in_feat_requires_grad = in_feat.requires_grad
    ctx.proj_weight_requires_grad = proj_weight.requires_grad
    if proj_weight.requires_grad and in_feat.requires_grad:
        ctx.save_for_backward(grad_in_feat, grad_proj_weight)
    elif proj_weight.requires_grad and not in_feat.requires_grad:
        ctx.save_for_backward(grad_proj_weight)
    elif not proj_weight.requires_grad and in_feat.requires_grad:
        ctx.save_for_backward(grad_in_feat)
    return loss


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _FusedCrossEntropyLossFunction_backward(ctx, grad_output):
    if ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
        grad_in_feat, grad_proj_weight = ctx.saved_tensors
    elif not ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
        grad_proj_weight, = ctx.saved_tensors
    elif ctx.in_feat_requires_grad and not ctx.proj_weight_requires_grad:
        grad_in_feat, = ctx.saved_tensors
    assert grad_output.shape == tuple(), grad_output.shape
    grad_in_feat *= grad_output
    grad_proj_weight *= grad_output
    return grad_in_feat, grad_proj_weight, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedCrossEntropyLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in_feat: torch.Tensor, proj_weight: torch.Tensor, targ:
        torch.Tensor, n_loop_iters: int, ignore_index: int, reduction: str):
        n_tokens = in_feat.shape[0]
        n_classes = proj_weight.shape[0]
        assert in_feat.ndim == 2, in_feat.ndim
        assert proj_weight.ndim == 2, proj_weight.ndim
        assert targ.ndim == 1, targ.shape
        assert in_feat.shape[0] == targ.shape[0
            ], f'Number of tokens in in_feat and targ is not equal: (in_feat.shape, targ.shape) = {in_feat.shape, targ.shape!r}'
        assert reduction in ('mean', 'sum'), reduction
        assert n_loop_iters > 0, n_loop_iters
        assert n_tokens % n_loop_iters == 0, (n_tokens, n_loop_iters)
        NUM_WARPS = 16
        BLOCK_SIZE = triton.next_power_of_2(n_classes)
        loss = torch.empty(n_tokens, dtype=in_feat.dtype, device=in_feat.device
            )
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled(
            ) else in_feat.dtype
        if proj_weight.requires_grad:
            grad_proj_weight = torch.zeros_like(proj_weight, dtype=dtype)
        else:
            grad_proj_weight = None
        if in_feat.requires_grad:
            grad_in_feat = torch.zeros_like(in_feat)
        else:
            grad_in_feat = None
        divisor = (targ != ignore_index).sum().to(dtype
            ) if reduction == 'mean' else torch.ones(1, dtype=dtype, device
            =in_feat.device)
        proj_weight_cast = proj_weight.to(dtype)
        loop_chunk_size = triton.cdiv(n_tokens, n_loop_iters)
        logits_chunk_cast = torch.zeros((loop_chunk_size, n_classes), dtype
            =dtype, device=in_feat.device)
        for i, in_feat_chunk in enumerate(torch.split(in_feat, loop_chunk_size)
            ):
            token_start_idx = i * loop_chunk_size
            token_end_idx = (i + 1) * loop_chunk_size
            in_feat_chunk = in_feat_chunk.to(dtype)
            torch.matmul(in_feat_chunk, proj_weight_cast.T, out=
                logits_chunk_cast)
            logits_chunk = logits_chunk_cast.float()
            loss_chunk = loss[token_start_idx:token_end_idx]
            targ_chunk = targ[token_start_idx:token_end_idx]
            n_tokens_chunk = logits_chunk.shape[0]
            grad_logits_chunk = logits_chunk
            fused_cross_entropy_fwd_bwd_kernel[n_tokens_chunk,](loss_chunk,
                grad_logits_chunk, logits_chunk, targ_chunk, divisor,
                loss_chunk.stride(0), grad_logits_chunk.stride(0),
                logits_chunk.stride(0), targ_chunk.stride(0), n_classes,
                ignore_index, requires_grad=in_feat.requires_grad or
                proj_weight.requires_grad, num_warps=NUM_WARPS, BLOCK_SIZE=
                BLOCK_SIZE)
            grad_logits_chunk = grad_logits_chunk.to(dtype)
            if in_feat.requires_grad:
                grad_in_feat[token_start_idx:token_end_idx
                    ] = grad_logits_chunk @ proj_weight_cast
            if proj_weight.requires_grad:
                torch.addmm(grad_proj_weight, grad_logits_chunk.T,
                    in_feat_chunk, out=grad_proj_weight)
        loss = loss.sum()
        ctx.in_feat_requires_grad = in_feat.requires_grad
        ctx.proj_weight_requires_grad = proj_weight.requires_grad
        if proj_weight.requires_grad and in_feat.requires_grad:
            ctx.save_for_backward(grad_in_feat, grad_proj_weight)
        elif proj_weight.requires_grad and not in_feat.requires_grad:
            ctx.save_for_backward(grad_proj_weight)
        elif not proj_weight.requires_grad and in_feat.requires_grad:
            ctx.save_for_backward(grad_in_feat)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
            grad_in_feat, grad_proj_weight = ctx.saved_tensors
        elif not ctx.in_feat_requires_grad and ctx.proj_weight_requires_grad:
            grad_proj_weight, = ctx.saved_tensors
        elif ctx.in_feat_requires_grad and not ctx.proj_weight_requires_grad:
            grad_in_feat, = ctx.saved_tensors
        assert grad_output.shape == tuple(), grad_output.shape
        grad_in_feat *= grad_output
        grad_proj_weight *= grad_output
        return grad_in_feat, grad_proj_weight, None, None, None, None
