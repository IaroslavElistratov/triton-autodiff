# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/open-lm-engine/flash-model-architectures
# Source-Files: fma/layers/moe/cuda_implementation/__init__.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_r3l3p8y1/flash-model-architectures-main/fma/layers/moe/cuda_implementation/__init__.py
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

def ceil_divide(x: int, y: int) ->int:
    return (x + y - 1) // y


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=_get_autotune_configs(), key=['H'], reset_to_zero=
    ['y_ptr'])
@triton.jit
def ungroup_with_padding_triton_kernel(x_ptr, expert_padding_offset_ptr,
    sorted_idxs_ptr, scattered_idxs_ptr, y_ptr, T, H, K, ATOMIC_ADD: tl.
    constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    B = T * K
    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B
    scattered_idxs = tl.load(scattered_idxs_ptr + indices_b, mask=mask_b)
    x_ptrs = x_ptr + indices_b[:, None] * H
    if ATOMIC_ADD:
        y_ptrs = y_ptr + (scattered_idxs // K)[:, None] * H
    else:
        y_ptrs = y_ptr + scattered_idxs[:, None] * H
    if expert_padding_offset_ptr is not None:
        sorted_idxs = tl.load(sorted_idxs_ptr + indices_b, mask=mask_b)
        expert_padding_offset = tl.load(expert_padding_offset_ptr + sorted_idxs
            )
        x_ptrs += expert_padding_offset[:, None] * H
    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    for h in range(NUM_BLOCKS_H):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        if h < NUM_BLOCKS_H - 1:
            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_b[:, None])
            if ATOMIC_ADD:
                tl.atomic_add(y_ptrs + indices_h[None, :], x, mask=mask_b[:,
                    None], sem='relaxed')
            else:
                tl.store(y_ptrs + indices_h[None, :], x, mask=mask_b[:, None])
        else:
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]
            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_bh)
            if ATOMIC_ADD:
                tl.atomic_add(y_ptrs + indices_h[None, :], x, mask=mask_bh,
                    sem='relaxed')
            else:
                tl.store(y_ptrs + indices_h[None, :], x, mask=mask_bh)


@custom_op(f'{LIBRARY_NAME}::ungroup_with_padding_triton', mutates_args={
    'output'})
def ungroup_with_padding_triton(x: torch.Tensor, expert_padding_offset:
    torch.Tensor, sorted_idxs: torch.Tensor, scattered_idxs: torch.Tensor,
    output: torch.Tensor, T: int, H: int, K: int, ATOMIC_ADD: bool) ->None:
    GRID = lambda meta: (ceil_divide(T * K, meta['BLOCK_SIZE_B']),)
    with torch.device(x.device):
        ungroup_with_padding_triton_kernel[GRID](x_ptr=x,
            expert_padding_offset_ptr=expert_padding_offset,
            sorted_idxs_ptr=sorted_idxs, scattered_idxs_ptr=scattered_idxs,
            y_ptr=output, T=T, H=H, K=K, ATOMIC_ADD=ATOMIC_ADD)


# Forward method (kernel launch code)
@ensure_contiguous
def __UngroupWithPadding_forward(ctx, x: torch.Tensor,
    expert_padding_offset: torch.Tensor, sorted_idxs: torch.Tensor,
    scattered_idxs: torch.Tensor, top_k: int, num_tokens: int,
    pad_to_multiple_of: int) ->torch.Tensor:
    assert x.dim() == 2
    T = num_tokens
    H = x.size(-1)
    K = top_k
    assert H % 8 == 0
    output = torch.empty(T, K, H, device=x.device, dtype=x.dtype)
    ungroup_with_padding_triton(x=x, expert_padding_offset=
        expert_padding_offset, sorted_idxs=sorted_idxs, scattered_idxs=
        scattered_idxs, output=output, T=T, H=H, K=K, ATOMIC_ADD=False)
    ctx.save_for_backward(expert_padding_offset, sorted_idxs, scattered_idxs)
    ctx.x_shape = x.size()
    ctx.pad_to_multiple_of = pad_to_multiple_of
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=_get_autotune_configs(), key=['H'])
@triton.jit
def group_with_padding_triton_kernel(x_ptr, expert_padding_offset_ptr,
    sorted_idxs_ptr, scattered_idxs_ptr, y_ptr, T, H, K, NEEDS_DUPLICATION:
    tl.constexpr, BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID = tl.program_id(axis=0)
    B = T * K
    indices_b = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    mask_b = indices_b < B
    scattered_idxs = tl.load(scattered_idxs_ptr + indices_b, mask=mask_b)
    if NEEDS_DUPLICATION:
        x_ptrs = x_ptr + (scattered_idxs // K)[:, None] * H
    else:
        x_ptrs = x_ptr + scattered_idxs[:, None] * H
    y_ptrs = y_ptr + indices_b[:, None] * H
    if expert_padding_offset_ptr is not None:
        sorted_idxs = tl.load(sorted_idxs_ptr + indices_b, mask=mask_b)
        expert_padding_offset = tl.load(expert_padding_offset_ptr + sorted_idxs
            )
        y_ptrs += expert_padding_offset[:, None] * H
    NUM_BLOCKS_H = tl.cdiv(H, BLOCK_SIZE_H)
    for h in range(NUM_BLOCKS_H):
        indices_h = h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        if h < NUM_BLOCKS_H - 1:
            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_b[:, None])
            tl.store(y_ptrs + indices_h[None, :], x, mask=mask_b[:, None])
        else:
            mask_h = indices_h < H
            mask_bh = mask_b[:, None] & mask_h[None, :]
            x = tl.load(x_ptrs + indices_h[None, :], mask=mask_bh)
            tl.store(y_ptrs + indices_h[None, :], x, mask=mask_bh)


@custom_op(f'{LIBRARY_NAME}::group_with_padding_triton', mutates_args={
    'output'})
def group_with_padding_triton(x: torch.Tensor, expert_padding_offset: torch
    .Tensor, sorted_idxs: torch.Tensor, scattered_idxs: torch.Tensor,
    output: torch.Tensor, T: int, H: int, K: int, NEEDS_DUPLICATION: bool
    ) ->None:
    GRID = lambda meta: (ceil_divide(T * K, meta['BLOCK_SIZE_B']),)
    with torch.device(x.device):
        group_with_padding_triton_kernel[GRID](x_ptr=x,
            expert_padding_offset_ptr=expert_padding_offset,
            sorted_idxs_ptr=sorted_idxs, scattered_idxs_ptr=scattered_idxs,
            y_ptr=output, T=T, H=H, K=K, NEEDS_DUPLICATION=NEEDS_DUPLICATION)


# Backward method (kernel launch code)
@ensure_contiguous
def __UngroupWithPadding_backward(ctx, output_grad: torch.Tensor
    ) ->torch.Tensor:
    expert_padding_offset, sorted_idxs, scattered_idxs = ctx.saved_tensors
    pad_to_multiple_of = ctx.pad_to_multiple_of
    T, K, H = output_grad.size()
    x_grad = (torch.empty if pad_to_multiple_of == 1 else torch.zeros)(*ctx
        .x_shape, device=output_grad.device, dtype=output_grad.dtype)
    group_with_padding_triton(x=output_grad, expert_padding_offset=
        expert_padding_offset, sorted_idxs=sorted_idxs, scattered_idxs=
        scattered_idxs, output=x_grad, T=T, H=H, K=K, NEEDS_DUPLICATION=False)
    return x_grad, *([None] * 6)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _UngroupWithPadding(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, expert_padding_offset: torch.Tensor,
        sorted_idxs: torch.Tensor, scattered_idxs: torch.Tensor, top_k: int,
        num_tokens: int, pad_to_multiple_of: int) ->torch.Tensor:
        assert x.dim() == 2
        T = num_tokens
        H = x.size(-1)
        K = top_k
        assert H % 8 == 0
        output = torch.empty(T, K, H, device=x.device, dtype=x.dtype)
        ungroup_with_padding_triton(x=x, expert_padding_offset=
            expert_padding_offset, sorted_idxs=sorted_idxs, scattered_idxs=
            scattered_idxs, output=output, T=T, H=H, K=K, ATOMIC_ADD=False)
        ctx.save_for_backward(expert_padding_offset, sorted_idxs,
            scattered_idxs)
        ctx.x_shape = x.size()
        ctx.pad_to_multiple_of = pad_to_multiple_of
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) ->torch.Tensor:
        expert_padding_offset, sorted_idxs, scattered_idxs = ctx.saved_tensors
        pad_to_multiple_of = ctx.pad_to_multiple_of
        T, K, H = output_grad.size()
        x_grad = (torch.empty if pad_to_multiple_of == 1 else torch.zeros)(*
            ctx.x_shape, device=output_grad.device, dtype=output_grad.dtype)
        group_with_padding_triton(x=output_grad, expert_padding_offset=
            expert_padding_offset, sorted_idxs=sorted_idxs, scattered_idxs=
            scattered_idxs, output=x_grad, T=T, H=H, K=K, NEEDS_DUPLICATION
            =False)
        return x_grad, *([None] * 6)
