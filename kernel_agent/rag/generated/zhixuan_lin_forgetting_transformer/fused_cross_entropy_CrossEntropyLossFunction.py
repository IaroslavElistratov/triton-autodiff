# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/zhixuan-lin/forgetting-transformer
# Source-Files: src/forgetting_transformer/model/forgetting_transformer/fused_cross_entropy.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_o52u21n_/forgetting-transformer-main/src/forgetting_transformer/model/forgetting_transformer/fused_cross_entropy.py
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

@triton.heuristics({'HAS_SMOOTHING': lambda args: args['label_smoothing'] >
    0.0})
@triton.jit
def cross_entropy_fwd_kernel(loss_ptr, lse_ptr, z_loss_ptr, logits_ptr,
    labels_ptr, label_smoothing, logit_scale, lse_square_scale,
    ignore_index, total_classes, class_start_idx, n_cols, n_rows,
    logits_row_stride, BLOCK_SIZE: tl.constexpr, HAS_SMOOTHING: tl.
    constexpr, SPLIT: tl.constexpr):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols,
        other=-float('inf'))
    logits = logits.to(tl.float32) * logit_scale
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols,
            (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (lse if not SPLIT else 0.0
                    ) - label_smoothing * sum_logits / total_classes - (1 -
                    label_smoothing) * logits_label
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        elif HAS_SMOOTHING:
            loss = label_smoothing * ((lse if not SPLIT else 0.0) - 
                sum_logits / total_classes)
        else:
            loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)


def fused_cross_entropy_forward(logits: torch.Tensor, target: torch.Tensor,
    label_smoothing: float=0.0, logit_scale: float=1.0, lse_square_scale:
    float=0.0, ignore_index: int=-100, process_group=None):
    n_rows, n_cols = logits.shape
    assert target.shape == (n_rows,)
    world_size = (1 if process_group is None else torch.distributed.
        get_world_size(process_group))
    total_classes = world_size * n_cols
    rank = 0 if process_group is None else torch.distributed.get_rank(
        process_group)
    class_start_idx = rank * n_cols
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    MAX_BLOCK_SIZE = 64 * 1024
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
    num_warps = (4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if
        BLOCK_SIZE < 128 * 1024 else 32)
    split = world_size > 1 or n_cols > MAX_BLOCK_SIZE
    n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
    losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
    z_losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device
        )
    with torch.cuda.device(logits.device.index):
        cross_entropy_fwd_kernel[n_rows, n_splits](losses, lse, z_losses,
            logits, target, label_smoothing, logit_scale, lse_square_scale,
            ignore_index, total_classes, class_start_idx, n_cols, n_rows,
            logits.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            SPLIT=split)
    if split:
        if n_splits > 1:
            lse = torch.logsumexp(lse, dim=0)
            losses = losses.sum(dim=0)
        if world_size > 1:
            lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype,
                device=lse.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse,
                group=process_group)
            handle_losses = torch.distributed.all_reduce(losses, op=torch.
                distributed.ReduceOp.SUM, group=process_group, async_op=True)
            lse = torch.logsumexp(lse_allgather, dim=0)
            handle_losses.wait()
        losses += lse
        if lse_square_scale != 0.0:
            z_losses = lse_square_scale * lse.square()
            z_losses.masked_fill_(target == ignore_index, 0.0)
            losses += z_losses
        else:
            z_losses = torch.zeros_like(losses)
        losses.masked_fill_(target == ignore_index, 0.0)
    return losses, z_losses, lse, total_classes, class_start_idx


# Forward method (kernel launch code)
def _CrossEntropyLossFunction_forward(ctx, logits, target, label_smoothing=
    0.0, logit_scale=1.0, lse_square_scale=0.0, ignore_index=-100,
    inplace_backward=False, process_group=None):
    losses, z_losses, lse, total_classes, class_start_idx = (
        fused_cross_entropy_forward(logits, target, label_smoothing,
        logit_scale, lse_square_scale, ignore_index, process_group))
    ctx.save_for_backward(logits, lse, target)
    ctx.mark_non_differentiable(z_losses)
    ctx.label_smoothing = label_smoothing
    ctx.logit_scale = logit_scale
    ctx.lse_square_scale = lse_square_scale
    ctx.ignore_index = ignore_index
    ctx.total_classes = total_classes
    ctx.class_start_idx = class_start_idx
    ctx.inplace_backward = inplace_backward
    return losses, z_losses


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'HAS_SMOOTHING': lambda args: args['label_smoothing'] >
    0.0})
@triton.jit
def cross_entropy_bwd_kernel(dlogits_ptr, dloss_ptr, logits_ptr, lse_ptr,
    labels_ptr, label_smoothing, logit_scale, lse_square_scale,
    ignore_index, total_classes, class_start_idx, n_cols, logits_row_stride,
    dlogits_row_stride, dloss_row_stride, BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignore_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols,
        other=-float('inf')).to(tl.float32) * logit_scale
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = label_smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 -
            label_smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, dloss * logit_scale * probs, mask=
        col_offsets < n_cols)


# Backward method (kernel launch code)
def _CrossEntropyLossFunction_backward(ctx, grad_losses, grad_z_losses):
    del grad_z_losses
    logits, lse, target = ctx.saved_tensors
    dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
    n_rows, n_cols = logits.shape
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16

    def grid(META):
        return n_rows, triton.cdiv(n_cols, META['BLOCK_SIZE'])
    with torch.cuda.device(logits.device.index):
        cross_entropy_bwd_kernel[grid](dlogits, grad_losses, logits, lse,
            target, ctx.label_smoothing, ctx.logit_scale, ctx.
            lse_square_scale, ctx.ignore_index, ctx.total_classes, ctx.
            class_start_idx, n_cols, logits.stride(0), dlogits.stride(0),
            grad_losses.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return dlogits, None, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CrossEntropyLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, target, label_smoothing=0.0, logit_scale=1.0,
        lse_square_scale=0.0, ignore_index=-100, inplace_backward=False,
        process_group=None):
        losses, z_losses, lse, total_classes, class_start_idx = (
            fused_cross_entropy_forward(logits, target, label_smoothing,
            logit_scale, lse_square_scale, ignore_index, process_group))
        ctx.save_for_backward(logits, lse, target)
        ctx.mark_non_differentiable(z_losses)
        ctx.label_smoothing = label_smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignore_index = ignore_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward
        return losses, z_losses

    @staticmethod
    def backward(ctx, grad_losses, grad_z_losses):
        del grad_z_losses
        logits, lse, target = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16

        def grid(META):
            return n_rows, triton.cdiv(n_cols, META['BLOCK_SIZE'])
        with torch.cuda.device(logits.device.index):
            cross_entropy_bwd_kernel[grid](dlogits, grad_losses, logits,
                lse, target, ctx.label_smoothing, ctx.logit_scale, ctx.
                lse_square_scale, ctx.ignore_index, ctx.total_classes, ctx.
                class_start_idx, n_cols, logits.stride(0), dlogits.stride(0
                ), grad_losses.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=
                num_warps)
        return dlogits, None, None, None, None, None, None, None, None
