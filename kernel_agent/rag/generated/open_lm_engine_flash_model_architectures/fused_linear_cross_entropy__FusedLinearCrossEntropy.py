# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/open-lm-engine/flash-model-architectures
# Source-Files: fma/functional/fused_linear_cross_entropy.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_r3l3p8y1/flash-model-architectures-main/fma/functional/fused_linear_cross_entropy.py
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
from math import ceil
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=_get_autotune_configs(), key=['BLOCK_SIZE_V'],
    reset_to_zero=['l_ptr'])
@triton.jit
def cross_entropy_forward_backward_triton_kernel(x_ptr, x_stride, y_ptr,
    y_stride, l_ptr, dx_ptr, dx_stride, logits_multiplier, B, V,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_V: tl.constexpr, reduction: tl.
    constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B
    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float('inf'), dtype=tl.float32)
    NUM_BLOCKS_V = tl.cdiv(V, BLOCK_SIZE_V)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_V[None, :
        ] * x_stride[1]
    for _ in range(NUM_BLOCKS_V):
        MASK_V = BLOCK_V < V
        MASK_BV = MASK_B[:, None] & MASK_V[None, :]
        x = tl.load(x_ptrs, mask=MASK_BV, other=-float('inf')).to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier
        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)
        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)
        BLOCK_V += BLOCK_SIZE_V
        x_ptrs += BLOCK_SIZE_V * x_stride[1]
    labels = tl.load(y_ptr + BLOCK_B * y_stride[0], mask=MASK_B)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_V[None, :
        ] * x_stride[1]
    dx_ptrs = dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_V[None, :
        ] * dx_stride[1]
    for _ in range(NUM_BLOCKS_V):
        MASK_V = BLOCK_V < V
        MASK_BV = MASK_B[:, None] & MASK_V[None, :]
        x = tl.load(x_ptrs, mask=MASK_BV).to(tl.float32)
        if logits_multiplier is not None:
            x *= logits_multiplier
        x -= M
        x = tl.exp(x)
        x /= Z
        x -= tl.where(BLOCK_V[None, :] == labels[:, None], 1, 0)
        if logits_multiplier is not None:
            x *= logits_multiplier
        if reduction == 'mean':
            x /= B
        tl.store(dx_ptrs, x, mask=MASK_BV)
        BLOCK_V += BLOCK_SIZE_V
        x_ptrs += BLOCK_SIZE_V * x_stride[1]
        dx_ptrs += BLOCK_SIZE_V * dx_stride[1]
    x = tl.load(x_ptr + BLOCK_B * x_stride[0] + labels * x_stride[1], mask=
        MASK_B).to(tl.float32)
    if logits_multiplier is not None:
        x *= logits_multiplier
    l = M + tl.log(Z) - x[:, None]
    l = tl.where(MASK_B[:, None], l, 0)
    l = tl.sum(l, axis=0)
    if reduction == 'mean':
        l /= B
    tl.atomic_add(l_ptr + tl.arange(0, 1), l, sem='relaxed')


def ceil_divide(x: int, y: int) ->int:
    return (x + y - 1) // y


def get_next_power_of_2(x: int) ->int:
    for p in _POWERS_OF_2:
        if p >= x:
            return p
    raise ValueError(
        f'x ({x}) is bigger than the max allowable power of 2 ({p})')


@custom_op(f'{LIBRARY_NAME}::cross_entropy_forward_backward_triton',
    mutates_args={'loss', 'x_grad'})
def cross_entropy_forward_backward_triton(x: torch.Tensor, labels: torch.
    Tensor, loss: torch.Tensor, x_grad: torch.Tensor, logits_multiplier: (
    float | None), reduction: str) ->None:
    B, V = x.size()
    BLOCK_SIZE_V = min(get_next_power_of_2(V), 4096 if x.dtype == torch.
        float32 else 8192)
    GRID = lambda meta: (ceil_divide(B, meta['BLOCK_SIZE_B']),)
    with torch.device(x.device):
        cross_entropy_forward_backward_triton_kernel[GRID](x_ptr=x,
            x_stride=x.stride(), y_ptr=labels, y_stride=labels.stride(),
            l_ptr=loss, dx_ptr=x_grad, dx_stride=x_grad.stride(),
            logits_multiplier=logits_multiplier, B=B, V=V, reduction=
            reduction, BLOCK_SIZE_V=BLOCK_SIZE_V)


def empty_like_contiguous(x: torch.Tensor, dtype: (torch.dtype | None)=None
    ) ->torch.Tensor:
    return torch.empty_like(x, dtype=dtype, memory_format=torch.
        contiguous_format)


def zeros_like_contiguous(x: torch.Tensor, dtype: (torch.dtype | None)=None
    ) ->torch.Tensor:
    return torch.zeros_like(x, dtype=dtype, memory_format=torch.
        contiguous_format)


# Forward method (kernel launch code)
def __FusedLinearCrossEntropy_forward(ctx, x: torch.Tensor, weight: torch.
    Tensor, labels: torch.Tensor, reduction: str, logits_multiplier: (float |
    None)) ->torch.Tensor:
    B, H = x.size()
    V = weight.size(0)
    memory_increase_factor = ceil_divide(V, H)
    chunk_size = get_next_power_of_2(ceil_divide(B, memory_increase_factor))
    num_chunks = ceil_divide(B, chunk_size)
    loss = torch.zeros((), device=x.device, dtype=torch.float32)
    x_grad = empty_like_contiguous(x)
    weight_grad = zeros_like_contiguous(weight)
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        end = min(end, B)
        _x = x[start:end]
        _logits = _x @ weight.T
        _logits_grad = empty_like_contiguous(_logits)
        _labels = labels[start:end]
        cross_entropy_forward_backward_triton(x=_logits, labels=_labels,
            loss=loss, x_grad=_logits_grad, logits_multiplier=
            logits_multiplier, reduction='sum')
        x_grad[start:end] = _logits_grad @ weight
        torch.addmm(weight_grad, _logits_grad.T, _x, alpha=1, beta=1, out=
            weight_grad)
    if reduction == 'mean':
        loss /= B
        x_grad /= B
        weight_grad /= B
    ctx.save_for_backward(x_grad, weight_grad)
    return loss


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def __FusedLinearCrossEntropy_backward(ctx, output_grad: torch.Tensor) ->tuple[
    torch.Tensor | None]:
    x_grad, weight_grad = ctx.saved_tensors
    x_grad *= output_grad
    weight_grad *= output_grad
    return x_grad, weight_grad, *([None] * 3)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _FusedLinearCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, labels: torch.
        Tensor, reduction: str, logits_multiplier: (float | None)
        ) ->torch.Tensor:
        B, H = x.size()
        V = weight.size(0)
        memory_increase_factor = ceil_divide(V, H)
        chunk_size = get_next_power_of_2(ceil_divide(B, memory_increase_factor)
            )
        num_chunks = ceil_divide(B, chunk_size)
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        x_grad = empty_like_contiguous(x)
        weight_grad = zeros_like_contiguous(weight)
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            end = min(end, B)
            _x = x[start:end]
            _logits = _x @ weight.T
            _logits_grad = empty_like_contiguous(_logits)
            _labels = labels[start:end]
            cross_entropy_forward_backward_triton(x=_logits, labels=_labels,
                loss=loss, x_grad=_logits_grad, logits_multiplier=
                logits_multiplier, reduction='sum')
            x_grad[start:end] = _logits_grad @ weight
            torch.addmm(weight_grad, _logits_grad.T, _x, alpha=1, beta=1,
                out=weight_grad)
        if reduction == 'mean':
            loss /= B
            x_grad /= B
            weight_grad /= B
        ctx.save_for_backward(x_grad, weight_grad)
        return loss

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor | None]:
        x_grad, weight_grad = ctx.saved_tensors
        x_grad *= output_grad
        weight_grad *= output_grad
        return x_grad, weight_grad, *([None] * 3)
