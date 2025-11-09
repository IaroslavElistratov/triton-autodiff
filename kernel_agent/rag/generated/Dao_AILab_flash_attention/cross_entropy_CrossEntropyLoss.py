# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Dao-AILab/flash-attention
# Source-Files: flash_attn/ops/triton/cross_entropy.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_aqv8vcou/flash-attention-main/flash_attn/ops/triton/cross_entropy.py
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
from math import exp
from math import log
from einops import reduce

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'HAS_SMOOTHING': lambda args: args['smoothing'] > 0.0})
@triton.jit
def cross_entropy_fwd_kernel(loss_ptr, lse_ptr, z_loss_ptr, logits_ptr,
    labels_ptr, smoothing, logit_scale, lse_square_scale, ignore_index,
    total_classes, class_start_idx, n_cols, logits_row_stride, BLOCK_SIZE:
    tl.constexpr, HAS_SMOOTHING: tl.constexpr, SPLIT: tl.constexpr,
    PRECOMPUTED_LSE: tl.constexpr):
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    sum_logits = 0.0
    if not PRECOMPUTED_LSE:
        m_i = -float('inf')
        l_i = 0.0
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            logits = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-
                float('inf')).to(tl.float32) * logit_scale
            if HAS_SMOOTHING:
                sum_logits += tl.sum(tl.where(cols < n_cols, logits, 0.0))
            m_i_new = tl.maximum(m_i, tl.max(logits))
            l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(tl.exp(logits - m_i_new)
                )
            m_i = m_i_new
        lse = tl.log(l_i) + m_i
        tl.store(lse_ptr + row_idx, lse)
    else:
        lse = tl.load(lse_ptr + row_idx)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= 0 and label_idx < n_cols:
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (lse if not SPLIT else 0.0
                    ) - smoothing * sum_logits / total_classes - (1 - smoothing
                    ) * logits_label
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        elif HAS_SMOOTHING:
            loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits /
                total_classes)
        else:
            loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + row_idx, z_loss)


# Forward method (kernel launch code)
def _CrossEntropyLoss_forward(ctx, logits, labels, precomputed_lse=None,
    smoothing=0.0, logit_scale=1.0, lse_square_scale=0.0, ignore_index=-100,
    inplace_backward=False, process_group=None):
    if labels.dtype == torch.long and labels.data_ptr() % 16 != 0:
        labels = F.pad(labels, (0, 1))[..., :-1]
        assert labels.data_ptr() % 16 == 0
    assert logit_scale > 0.0
    n_rows, n_cols = logits.shape
    assert labels.shape == (n_rows,)
    world_size = (1 if process_group is None else torch.distributed.
        get_world_size(process_group))
    total_classes = world_size * n_cols
    rank = 0 if process_group is None else torch.distributed.get_rank(
        process_group)
    class_start_idx = rank * n_cols
    use_precomputed_lse = (precomputed_lse is not None and logit_scale == 
        1.0 and smoothing == 0.0)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    MAX_BLOCK_SIZE = 16 * 1024
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
    num_warps = (4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if
        BLOCK_SIZE < 128 * 1024 else 32)
    losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    if use_precomputed_lse:
        assert precomputed_lse.shape == (n_rows,)
        lse = precomputed_lse.contiguous()
    else:
        lse = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    z_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    with torch.cuda.device(logits.device.index):
        cross_entropy_fwd_kernel[n_rows,](losses, lse, z_losses, logits,
            labels, smoothing, logit_scale, lse_square_scale, ignore_index,
            total_classes, class_start_idx, n_cols, logits.stride(0),
            BLOCK_SIZE=BLOCK_SIZE, SPLIT=world_size > 1, PRECOMPUTED_LSE=
            use_precomputed_lse, num_warps=num_warps)
    if world_size > 1:
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
            z_losses.masked_fill_(labels == ignore_index, 0.0)
            losses += z_losses
        else:
            z_losses = torch.zeros_like(losses)
        losses.masked_fill_(labels == ignore_index, 0.0)
    ctx.save_for_backward(logits, lse, labels)
    ctx.mark_non_differentiable(z_losses)
    ctx.smoothing = smoothing
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

@triton.heuristics({'HAS_SMOOTHING': lambda args: args['smoothing'] > 0.0})
@triton.jit
def cross_entropy_bwd_kernel(dlogits_ptr, dloss_ptr, logits_ptr, lse_ptr,
    labels_ptr, smoothing, logit_scale, lse_square_scale, ignore_index,
    total_classes, class_start_idx, n_cols, logits_row_stride,
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
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - smooth_positive,
            probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, dloss * logit_scale * probs, mask=
        col_offsets < n_cols)


# Backward method (kernel launch code)
def _CrossEntropyLoss_backward(ctx, grad_losses, grad_z_losses):
    del grad_z_losses
    logits, lse, labels = ctx.saved_tensors
    dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
    n_rows, n_cols = logits.shape
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16
    grid = lambda META: (n_rows, triton.cdiv(n_cols, META['BLOCK_SIZE']))
    with torch.cuda.device(logits.device.index):
        cross_entropy_bwd_kernel[grid](dlogits, grad_losses, logits, lse,
            labels, ctx.smoothing, ctx.logit_scale, ctx.lse_square_scale,
            ctx.ignore_index, ctx.total_classes, ctx.class_start_idx,
            n_cols, logits.stride(0), dlogits.stride(0), grad_losses.stride
            (0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return dlogits, None, None, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, precomputed_lse=None, smoothing=0.0,
        logit_scale=1.0, lse_square_scale=0.0, ignore_index=-100,
        inplace_backward=False, process_group=None):
        if labels.dtype == torch.long and labels.data_ptr() % 16 != 0:
            labels = F.pad(labels, (0, 1))[..., :-1]
            assert labels.data_ptr() % 16 == 0
        assert logit_scale > 0.0
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        world_size = (1 if process_group is None else torch.distributed.
            get_world_size(process_group))
        total_classes = world_size * n_cols
        rank = 0 if process_group is None else torch.distributed.get_rank(
            process_group)
        class_start_idx = rank * n_cols
        use_precomputed_lse = (precomputed_lse is not None and logit_scale ==
            1.0 and smoothing == 0.0)
        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        MAX_BLOCK_SIZE = 16 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
        num_warps = (4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else
            16 if BLOCK_SIZE < 128 * 1024 else 32)
        losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        if use_precomputed_lse:
            assert precomputed_lse.shape == (n_rows,)
            lse = precomputed_lse.contiguous()
        else:
            lse = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        z_losses = torch.empty(n_rows, dtype=torch.float, device=logits.device)
        with torch.cuda.device(logits.device.index):
            cross_entropy_fwd_kernel[n_rows,](losses, lse, z_losses, logits,
                labels, smoothing, logit_scale, lse_square_scale,
                ignore_index, total_classes, class_start_idx, n_cols,
                logits.stride(0), BLOCK_SIZE=BLOCK_SIZE, SPLIT=world_size >
                1, PRECOMPUTED_LSE=use_precomputed_lse, num_warps=num_warps)
        if world_size > 1:
            if world_size > 1:
                lse_allgather = torch.empty(world_size, n_rows, dtype=lse.
                    dtype, device=lse.device)
                torch.distributed.all_gather_into_tensor(lse_allgather, lse,
                    group=process_group)
                handle_losses = torch.distributed.all_reduce(losses, op=
                    torch.distributed.ReduceOp.SUM, group=process_group,
                    async_op=True)
                lse = torch.logsumexp(lse_allgather, dim=0)
                handle_losses.wait()
            losses += lse
            if lse_square_scale != 0.0:
                z_losses = lse_square_scale * lse.square()
                z_losses.masked_fill_(labels == ignore_index, 0.0)
                losses += z_losses
            else:
                z_losses = torch.zeros_like(losses)
            losses.masked_fill_(labels == ignore_index, 0.0)
        ctx.save_for_backward(logits, lse, labels)
        ctx.mark_non_differentiable(z_losses)
        ctx.smoothing = smoothing
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
        logits, lse, labels = ctx.saved_tensors
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16
        grid = lambda META: (n_rows, triton.cdiv(n_cols, META['BLOCK_SIZE']))
        with torch.cuda.device(logits.device.index):
            cross_entropy_bwd_kernel[grid](dlogits, grad_losses, logits,
                lse, labels, ctx.smoothing, ctx.logit_scale, ctx.
                lse_square_scale, ctx.ignore_index, ctx.total_classes, ctx.
                class_start_idx, n_cols, logits.stride(0), dlogits.stride(0
                ), grad_losses.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=
                num_warps)
        return dlogits, None, None, None, None, None, None, None, None, None
