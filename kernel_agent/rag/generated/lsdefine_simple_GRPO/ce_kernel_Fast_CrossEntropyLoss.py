# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/lsdefine/simple_GRPO
# Source-Files: kernel/ce_kernel.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_fmpyt5sc/simple_GRPO-main/kernel/ce_kernel.py
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
from math import exp
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _chunked_cross_entropy_forward(logits_ptr, logits_row_stride, loss_ptr,
    logsumexp_ptr, labels_ptr, VOCAB_SIZE: tl.constexpr, N_CHUNKS: tl.
    constexpr, BLOCK_SIZE: tl.constexpr):
    """
        256K vocab divided in 4 chunks

        |-65536-| |-65536-| |-65536-| |-65536-|
        |-------| |-------| |-------| |-------|
        |-------| |-------| |-------| |-------|

        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        Notice we can do logsumexp for each chunk and then
        logsumexp[chunk_sum(logsumexp)] == logsumexp

        chunk_sum = log[chunk_sum(logsumexp)]
                  = log[exp(logsumexp(a)) + ... + exp(logsumexp(z))]
                  = log[exp(log[sum(exp(a))]) + ... + exp(log[sum(exp(z))])]
                  = log[sum(exp(a)) + ... + sum(exp(z))]
                  = logsumexp(x)

        This means we can perform a logsumexp for each chunk, then do a
        final logsumexp reduction!

        Ie do: logsumexp(chunked_logsumexp) - x
    """
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx
    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf')
        ).to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            loss = -1.0 * x
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    pass
    tl.store(logsumexp_ptr, logsumexp)


@triton.jit
def _cross_entropy_forward(logits_ptr, logits_row_stride, loss_ptr,
    logsumexp_ptr, labels_ptr, VOCAB_SIZE: tl.constexpr, BLOCK_SIZE: tl.
    constexpr):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        logsumexp is also stable
        Take    y =         log[sum(exp(x))]
           exp(y) =             sum(exp(x))
           exp(y) =             sum(exp(x - c)*exp(c)) Since e^(x-c)*e^c = e^x
           exp(y) =      exp(c)*sum(exp(x - c))
               y  = log(exp(c)*sum(exp(x - c)))
               y  = c + log[sum(exp(x - c))]
        This means we can set c = max(x) to make sure
        exp(x - c) always is exp(x - max(x)).
        This ensures exp(x - max(x))'s maximum is 1 as exp(0) = 1.
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf')
        ).to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx).to(tl.float32)
        loss = logsumexp - x
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f'Cannot launch Triton kernel since n = {n} exceeds the maximum CUDA blocksize = {MAX_FUSED_SIZE}.'
            )
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# Forward method (kernel launch code)
def _Fast_CrossEntropyLoss_forward(ctx, logits, labels):
    n_rows, vocab_size = logits.shape
    div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
    n_chunks = div + (mod != 0)
    losses = torch.empty(n_rows, dtype=torch.float32, device='cuda')
    if n_chunks == 1:
        BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
        logsumexp = torch.empty(n_rows, dtype=torch.float32, device='cuda')
        _cross_entropy_forward[n_rows,](logits, logits.stride(0), losses,
            logsumexp, labels, VOCAB_SIZE=vocab_size, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps)
    else:
        logsumexp = torch.empty((n_rows, n_chunks), dtype=torch.float32,
            device='cuda')
        _chunked_cross_entropy_forward[n_rows, n_chunks](logits, logits.
            stride(0), losses, logsumexp, labels, VOCAB_SIZE=vocab_size,
            N_CHUNKS=n_chunks, BLOCK_SIZE=MAX_FUSED_SIZE, num_warps=32)
        logsumexp = torch.logsumexp(logsumexp, dim=1)
        losses += logsumexp
        losses.masked_fill_(labels == -100, 0)
    pass
    ctx.save_for_backward(logits, logsumexp, labels)
    return losses


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _cross_entropy_backward(logits_ptr, logits_row_stride, dloss_ptr,
    dloss_row_stride, logsumexp_ptr, labels_ptr, VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
    """
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)
    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0
    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl
        .float32)
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x - logsumexp)
    y = tl.where(col_offsets == label_idx, y - 1.0, y)
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)


# Backward method (kernel launch code)
def _Fast_CrossEntropyLoss_backward(ctx, dlosses):
    logits, logsumexp, labels = ctx.saved_tensors
    n_rows, vocab_size = logits.shape
    BLOCK_SIZE = 4096
    div, mod = divmod(vocab_size, BLOCK_SIZE)
    n_blocks = div + (mod != 0)
    _cross_entropy_backward[n_rows, n_blocks](logits, logits.stride(0),
        dlosses, dlosses.stride(0), logsumexp, labels, VOCAB_SIZE=
        vocab_size, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return logits, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Fast_CrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels):
        n_rows, vocab_size = logits.shape
        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device='cuda')
        if n_chunks == 1:
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty(n_rows, dtype=torch.float32, device='cuda')
            _cross_entropy_forward[n_rows,](logits, logits.stride(0),
                losses, logsumexp, labels, VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        else:
            logsumexp = torch.empty((n_rows, n_chunks), dtype=torch.float32,
                device='cuda')
            _chunked_cross_entropy_forward[n_rows, n_chunks](logits, logits
                .stride(0), losses, logsumexp, labels, VOCAB_SIZE=
                vocab_size, N_CHUNKS=n_chunks, BLOCK_SIZE=MAX_FUSED_SIZE,
                num_warps=32)
            logsumexp = torch.logsumexp(logsumexp, dim=1)
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)
        pass
        ctx.save_for_backward(logits, logsumexp, labels)
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape
        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)
        _cross_entropy_backward[n_rows, n_blocks](logits, logits.stride(0),
            dlosses, dlosses.stride(0), logsumexp, labels, VOCAB_SIZE=
            vocab_size, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return logits, None, None
