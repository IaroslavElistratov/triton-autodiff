# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Lightning-AI/lightning-thunder
# Source-Files: thunder/executors/triton_crossentropy_impl.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_g83ylltk/lightning-thunder-main/thunder/executors/triton_crossentropy_impl.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _class_indices_forward(LOGITS, PROBS, IDX, LOSS, weight, N,
    WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS: tl.constexpr,
    CLASS_INDICES: tl.constexpr, LABEL_SMOOTHING: tl.constexpr,
    IGNORE_INDEX: tl.constexpr, BUFFER_DTYPE: tl.constexpr, BLOCK: tl.constexpr
    ):
    buffer_dtype = BUFFER_DTYPE
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    logit_start_ptrs = LOGITS + row * N
    logit_ptrs = logit_start_ptrs + cols
    m_prev = -float('inf')
    l_prev = 0.0
    m_prev = m_prev.to(buffer_dtype)
    l_prev = l_prev.to(buffer_dtype)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK,
            other=-float('inf')).to(buffer_dtype)
        m_curr = tl.maximum(tl.max(row_logits, 0), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(row_logits - m_curr)
        l_curr = tl.sum(p, 0) + l_prev
        l_prev = l_curr
        m_prev = m_curr
        logit_ptrs += BLOCK
    logit_ptrs = logit_start_ptrs + cols
    WRIT_PROBS = PROBS + row * N + cols
    if LABEL_SMOOTHING:
        sum_total = 0.0
        sum_total = sum_total.to(buffer_dtype)
        weights_total = 0.0
        weights_total = weights_total.to(buffer_dtype)
        if WEIGHTS:
            weight_ptr = weight + cols
    l_prev_log = tl.log(l_prev)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK,
            other=l_prev_log + m_prev).to(buffer_dtype)
        if LABEL_SMOOTHING and WEIGHTS:
            full_weights_val = tl.load(weight_ptr, mask=cols < N - start_n *
                BLOCK, other=0.0)
            weights_total += tl.sum(full_weights_val, 0)
        row_minus_max = row_logits - m_prev
        log_softmax = l_prev_log - row_minus_max
        if LABEL_SMOOTHING and WEIGHTS:
            log_softmax *= full_weights_val
        if LABEL_SMOOTHING:
            sum_total += tl.sum(log_softmax, 0)
        tl.store(WRIT_PROBS, log_softmax, mask=cols < N - start_n * BLOCK)
        logit_ptrs += BLOCK
        WRIT_PROBS += BLOCK
        if LABEL_SMOOTHING and WEIGHTS:
            weight_ptr += BLOCK
    idx = tl.load(IDX + row)
    use_class = 0.0
    if IGNORE_INDEX >= 0:
        use_class = idx == IGNORE_INDEX
    READ_PROBS = PROBS + row * N + idx
    tl.debug_barrier()
    probs = tl.load(READ_PROBS)
    if WEIGHTS and not LABEL_SMOOTHING:
        weight_ptr = weight + idx
        weights_val = tl.load(weight_ptr)
        probs = weights_val * probs
    if LABEL_SMOOTHING:
        tl.store(WEIGHT_BUFFER + row, weights_total)
        probs = (1 - smoothing_factor
            ) * probs + smoothing_factor * sum_total / N
    probs = probs * (1.0 - use_class)
    tl.store(LOSS + row, probs)


@triton.jit
def _class_probs_forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER,
    smoothing_factor, log_size_logits, WEIGHTS: tl.constexpr, CLASS_INDICES:
    tl.constexpr, LABEL_SMOOTHING: tl.constexpr, IGNORE_INDEX: tl.constexpr,
    BUFFER_DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    buffer_dtype = BUFFER_DTYPE
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    logit_start_ptrs = LOGITS + row * N
    logit_ptrs = logit_start_ptrs + cols
    m_prev = -float('inf')
    l_prev = 0.0
    m_prev = m_prev.to(buffer_dtype)
    l_prev = l_prev.to(buffer_dtype)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK,
            other=-float('inf')).to(buffer_dtype)
        m_curr = tl.maximum(tl.max(row_logits, 0), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(row_logits - m_curr)
        l_curr = tl.sum(p, 0) + l_prev
        l_prev = l_curr
        m_prev = m_curr
        logit_ptrs += BLOCK
    logit_ptrs = logit_start_ptrs + cols
    WRIT_PROBS = PROBS + row * N + cols
    sum_total = 0.0
    weights_total = 0.0
    sum_total = sum_total.to(buffer_dtype)
    weights_total = weights_total.to(buffer_dtype)
    idx_ptr = IDX + row * N + cols
    if WEIGHTS:
        weight_ptr = weight + cols
    l_prev_log = tl.log(l_prev)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK,
            other=l_prev_log + m_prev).to(buffer_dtype)
        idx = tl.load(idx_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
        full_weights_val = (1.0 - smoothing_factor
            ) * idx + smoothing_factor / N
        if WEIGHTS:
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n *
                BLOCK, other=0.0)
            full_weights_val = weights_val * full_weights_val
        else:
            full_weights_val = tl.where(cols < N - start_n * BLOCK,
                full_weights_val, 0.0)
        weights_total += tl.sum(full_weights_val, 0)
        row_minus_max = row_logits - m_prev
        log_softmax = l_prev_log - row_minus_max
        log_softmax *= full_weights_val
        sum_total += tl.sum(log_softmax, 0)
        tl.store(WRIT_PROBS, log_softmax, mask=cols < N - start_n * BLOCK)
        logit_ptrs += BLOCK
        WRIT_PROBS += BLOCK
        idx_ptr += BLOCK
        if WEIGHTS:
            weight_ptr += BLOCK
    tl.store(WEIGHT_BUFFER + row, weights_total)
    probs = sum_total
    tl.store(LOSS + row, probs)


@triton.autotune(configs=[triton.Config({'BLOCK': 1024}, num_stages=
    FORWARD_NUM_STAGES, num_warps=1), triton.Config({'BLOCK': 2048},
    num_stages=FORWARD_NUM_STAGES, num_warps=8), triton.Config({'BLOCK': 
    4096}, num_stages=FORWARD_NUM_STAGES, num_warps=8), triton.Config({
    'BLOCK': 8192}, num_stages=FORWARD_NUM_STAGES, num_warps=16), triton.
    Config({'BLOCK': 16384}, num_stages=FORWARD_NUM_STAGES, num_warps=16)],
    key=['N', 'CLASS_INDICES', 'log_size_logits', 'BUFFER_DTYPE'])
@triton.jit
def _forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER,
    smoothing_factor, log_size_logits, WEIGHTS: tl.constexpr, CLASS_INDICES:
    tl.constexpr, LABEL_SMOOTHING: tl.constexpr, IGNORE_INDEX: tl.constexpr,
    BUFFER_DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    if CLASS_INDICES:
        _class_indices_forward(LOGITS, PROBS, IDX, LOSS, weight, N,
            WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS,
            CLASS_INDICES, LABEL_SMOOTHING, IGNORE_INDEX, BUFFER_DTYPE, BLOCK)
    else:
        _class_probs_forward(LOGITS, PROBS, IDX, LOSS, weight, N,
            WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS,
            CLASS_INDICES, LABEL_SMOOTHING, IGNORE_INDEX, BUFFER_DTYPE, BLOCK)


# Forward method (kernel launch code)
def _CrossEntropy_forward(ctx, logits, indices, weight, ignore_index,
    reduction, label_smoothing):
    buffer_dtype = None
    assert weight is None or len(weight.shape) == 1 and weight.shape[0
        ] == logits.shape[-1]
    if buffer_dtype is None:
        if logits.dtype in [torch.bfloat16, torch.float16]:
            buffer_dtype = torch.float32
        else:
            buffer_dtype = logits.dtype
    buffer_dtype_enum = _TORCH2DTYPE[buffer_dtype]
    device, dtype = logits.device, logits.dtype
    n_cols = logits.shape[-1]
    result = torch.empty((logits.shape[0],), dtype=dtype, device=device)
    neg_logprobs = torch.empty_like(logits, dtype=buffer_dtype, device=device)
    weights_buffer = torch.empty_like(result, dtype=buffer_dtype)
    grid = lambda opt: (logits.numel() // n_cols,)
    log_size_logits = int(math.log(math.prod(logits.shape) / n_cols))
    _forward[grid](logits, neg_logprobs, indices, result, weight, n_cols,
        weights_buffer, label_smoothing, log_size_logits, WEIGHTS=weight is not
        None, CLASS_INDICES=indices.dtype == torch.int64, LABEL_SMOOTHING=
        label_smoothing > 0.0, IGNORE_INDEX=ignore_index, BUFFER_DTYPE=
        _DTYPE2TRITON[buffer_dtype_enum])
    ctx.save_for_backward(neg_logprobs, indices, weights_buffer)
    ctx.WEIGHT = weight
    ctx.label_smoothing = label_smoothing
    ctx.ignore_index = ignore_index
    ctx.reduction = reduction
    ctx.buffer_dtype = buffer_dtype_enum
    if reduction == 'none':
        return result
    elif reduction == 'sum':
        return result.sum(dim=0)
    elif reduction == 'mean':
        if indices.dtype == torch.int64:
            denom = (indices != ignore_index).float()
            if weight is not None:
                class_weights = weight[indices]
                denom *= class_weights
            denom = denom.sum()
        else:
            denom = indices.shape[0]
        ctx.denom = denom
        return (result.sum(dim=0) / denom).to(dtype)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BLOCK': 1024}, num_stages=1,
    num_warps=1), triton.Config({'BLOCK': 2048}, num_stages=1, num_warps=8),
    triton.Config({'BLOCK': 4096}, num_stages=1, num_warps=8), triton.
    Config({'BLOCK': 8192}, num_stages=1, num_warps=16), triton.Config({
    'BLOCK': 16384}, num_stages=1, num_warps=16)], key=['N',
    'CLASS_INDICES', 'log_size_logits', 'BUFFER_DTYPE'])
@triton.jit
def _backward(PROBS, IDX, DPROBS, dprob_stride, DIN, weight, N,
    WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS: tl.constexpr,
    CLASS_INDICES: tl.constexpr, LABEL_SMOOTHING: tl.constexpr,
    IGNORE_INDEX: tl.constexpr, BUFFER_DTYPE: tl.constexpr, BLOCK: tl.constexpr
    ):
    buffer_dtype = BUFFER_DTYPE
    row = tl.program_id(0)
    start_n = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    PROBS = PROBS + row * N
    probs_start = PROBS + cols + BLOCK * start_n
    probs = -tl.load(probs_start, mask=cols < N - start_n * BLOCK, other=
        float('inf')).to(buffer_dtype)
    DIN = DIN + row * N + cols + BLOCK * start_n
    dout = tl.load(DPROBS + row * dprob_stride).to(buffer_dtype)
    if CLASS_INDICES:
        idx = tl.load(IDX + row)
        delta = start_n * BLOCK + cols == idx
        if IGNORE_INDEX >= 0:
            use_class = idx == IGNORE_INDEX
            dout = dout * (1 - use_class)
        if LABEL_SMOOTHING:
            if WEIGHTS:
                weight_ptr = weight + cols + BLOCK * start_n
                full_weights_val = tl.load(weight_ptr, mask=cols < N - 
                    start_n * BLOCK, other=0.0).to(buffer_dtype)
                weights_val = tl.load(weight + idx)
                probs = probs / full_weights_val
            probs = tl.exp(probs)
            if WEIGHTS:
                weights_total = tl.load(WEIGHT_BUFFER + row)
                numerator_contrib = weights_val * (1.0 - smoothing_factor) * (
                    probs - delta)
                mean_contrib = (weights_total * probs - full_weights_val
                    ) * smoothing_factor / N
            else:
                numerator_contrib = (1.0 - smoothing_factor) * (probs - delta)
                mean_contrib = smoothing_factor * probs - smoothing_factor / N
            din = (numerator_contrib + mean_contrib) * dout
        else:
            probs = tl.exp(probs)
            din = (probs - delta) * dout
            if WEIGHTS:
                weight_ptr = weight + idx
                weights_val = tl.load(weight_ptr)
                din = weights_val * din
    else:
        idx = tl.load(IDX + row * N + cols + BLOCK * start_n, mask=cols < N -
            start_n * BLOCK, other=0.0).to(buffer_dtype)
        full_weights_val = (1.0 - smoothing_factor
            ) * idx + smoothing_factor / N
        weights_total = tl.load(WEIGHT_BUFFER + row)
        if WEIGHTS:
            weight_ptr = weight + cols + BLOCK * start_n
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n *
                BLOCK, other=0.0).to(buffer_dtype)
            full_weights_val = weights_val * full_weights_val
        probs = probs / full_weights_val
        probs = tl.exp(probs.to(buffer_dtype))
        weighted_probs = probs * weights_total
        weighted_probs_per_class = weighted_probs - full_weights_val
        din = weighted_probs_per_class * dout
    tl.store(DIN, din.to(DIN.dtype.element_ty), mask=cols + BLOCK * start_n < N
        )


# Backward method (kernel launch code)
def _CrossEntropy_backward(ctx, dneg_logprobs):
    """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
    reduction = ctx.reduction
    if reduction == 'mean' or reduction == 'sum':
        dneg_logprobs = dneg_logprobs.expand(1)
    neg_logprobs, indices, weights_buffer = ctx.saved_tensors
    din = torch.empty_like(neg_logprobs)
    weight = ctx.WEIGHT
    buffer_dtype = ctx.buffer_dtype
    n_cols = neg_logprobs.shape[-1]
    grid = lambda opt: (neg_logprobs.numel() // n_cols, triton.cdiv(n_cols,
        opt['BLOCK']))
    log_size_logits = int(math.log(math.prod(neg_logprobs.shape) / n_cols))
    _backward[grid](neg_logprobs, indices, dneg_logprobs, dneg_logprobs.
        stride(0), din, weight, n_cols, weights_buffer, ctx.label_smoothing,
        log_size_logits, WEIGHTS=weight is not None, CLASS_INDICES=indices.
        dtype == torch.int64, LABEL_SMOOTHING=ctx.label_smoothing > 0.0,
        IGNORE_INDEX=ctx.ignore_index, BUFFER_DTYPE=_DTYPE2TRITON[buffer_dtype]
        )
    if ctx.reduction == 'mean':
        din /= ctx.denom
    return din, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, indices, weight, ignore_index, reduction,
        label_smoothing):
        buffer_dtype = None
        assert weight is None or len(weight.shape) == 1 and weight.shape[0
            ] == logits.shape[-1]
        if buffer_dtype is None:
            if logits.dtype in [torch.bfloat16, torch.float16]:
                buffer_dtype = torch.float32
            else:
                buffer_dtype = logits.dtype
        buffer_dtype_enum = _TORCH2DTYPE[buffer_dtype]
        device, dtype = logits.device, logits.dtype
        n_cols = logits.shape[-1]
        result = torch.empty((logits.shape[0],), dtype=dtype, device=device)
        neg_logprobs = torch.empty_like(logits, dtype=buffer_dtype, device=
            device)
        weights_buffer = torch.empty_like(result, dtype=buffer_dtype)
        grid = lambda opt: (logits.numel() // n_cols,)
        log_size_logits = int(math.log(math.prod(logits.shape) / n_cols))
        _forward[grid](logits, neg_logprobs, indices, result, weight,
            n_cols, weights_buffer, label_smoothing, log_size_logits,
            WEIGHTS=weight is not None, CLASS_INDICES=indices.dtype ==
            torch.int64, LABEL_SMOOTHING=label_smoothing > 0.0,
            IGNORE_INDEX=ignore_index, BUFFER_DTYPE=_DTYPE2TRITON[
            buffer_dtype_enum])
        ctx.save_for_backward(neg_logprobs, indices, weights_buffer)
        ctx.WEIGHT = weight
        ctx.label_smoothing = label_smoothing
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.buffer_dtype = buffer_dtype_enum
        if reduction == 'none':
            return result
        elif reduction == 'sum':
            return result.sum(dim=0)
        elif reduction == 'mean':
            if indices.dtype == torch.int64:
                denom = (indices != ignore_index).float()
                if weight is not None:
                    class_weights = weight[indices]
                    denom *= class_weights
                denom = denom.sum()
            else:
                denom = indices.shape[0]
            ctx.denom = denom
            return (result.sum(dim=0) / denom).to(dtype)

    @staticmethod
    def backward(ctx, dneg_logprobs):
        """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
        reduction = ctx.reduction
        if reduction == 'mean' or reduction == 'sum':
            dneg_logprobs = dneg_logprobs.expand(1)
        neg_logprobs, indices, weights_buffer = ctx.saved_tensors
        din = torch.empty_like(neg_logprobs)
        weight = ctx.WEIGHT
        buffer_dtype = ctx.buffer_dtype
        n_cols = neg_logprobs.shape[-1]
        grid = lambda opt: (neg_logprobs.numel() // n_cols, triton.cdiv(
            n_cols, opt['BLOCK']))
        log_size_logits = int(math.log(math.prod(neg_logprobs.shape) / n_cols))
        _backward[grid](neg_logprobs, indices, dneg_logprobs, dneg_logprobs
            .stride(0), din, weight, n_cols, weights_buffer, ctx.
            label_smoothing, log_size_logits, WEIGHTS=weight is not None,
            CLASS_INDICES=indices.dtype == torch.int64, LABEL_SMOOTHING=ctx
            .label_smoothing > 0.0, IGNORE_INDEX=ctx.ignore_index,
            BUFFER_DTYPE=_DTYPE2TRITON[buffer_dtype])
        if ctx.reduction == 'mean':
            din /= ctx.denom
        return din, None, None, None, None, None, None
