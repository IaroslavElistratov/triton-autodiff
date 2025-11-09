# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/rope.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/rope.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

@triton.jit
def _triton_rope(q_ptr, q_row_stride, k_ptr, k_row_stride, cos,
    cos_row_stride, sin, sin_row_stride, sl, bs: tl.constexpr, cos_bs: tl.
    constexpr, n_qh: tl.constexpr, n_kh: tl.constexpr, hd: tl.constexpr,
    pad_n_qh: tl.constexpr, pad_n_kh: tl.constexpr, pad_hd: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BACKWARD_PASS: tl.constexpr=False):
    pid = tl.program_id(0).to(tl.int64)
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride
    batch_idx = pid // sl
    cos_row_idx = pid % sl
    cos = cos + tl.where(cos_bs == 1, cos_row_idx * cos_row_stride, 
        batch_idx * (sl * cos_row_stride) + cos_row_idx * cos_row_stride)
    sin = sin + tl.where(cos_bs == 1, cos_row_idx * sin_row_stride, 
        batch_idx * (sl * sin_row_stride) + cos_row_idx * sin_row_stride)
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(
        0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(
        0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0,
        pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0,
        pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0
        ).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0
        ).to(sin_row.dtype)
    second_half_q_offsets = first_half_q_offsets + hd // 2
    second_half_k_offsets = first_half_k_offsets + hd // 2
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask,
        other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask,
        other=0).to(sin_row.dtype)
    if not BACKWARD_PASS:
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=
            second_q_mask)
        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=
            second_k_mask)
    else:
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=
            second_q_mask)
        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=
            second_k_mask)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

def rope_forward(q, k, cos, sin):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    batch_size, seq_len, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)
    n_row = batch_size * seq_len
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    cos_batch_size = cos.shape[0]
    _triton_rope[n_row,](q, q.stride(1), k, k.stride(1), cos, cos.stride(-2
        ), sin, sin.stride(-2), seq_len, batch_size, cos_batch_size,
        n_q_head, n_kv_head, head_dim, pad_n_q_head, pad_n_kv_head, pad_hd,
        BLOCK_SIZE=BLOCK_SIZE, BACKWARD_PASS=False)
    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


# Forward method (kernel launch code)
def _LigerRopeFunction_forward(ctx, q, k, cos, sin, position_ids=None,
    unsqueeze_dim=1):
    """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
    q, k, cos, sin = rope_forward(q, k, cos, sin)
    ctx.save_for_backward(cos, sin)
    return q, k


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

def rope_backward(dq, dk, cos, sin):
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)
    batch_size, seq_len, n_q_head, head_dim = dq.shape
    cos_batch_size = cos.shape[0]
    n_kv_head = dk.shape[2]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)
    n_row = batch_size * seq_len
    dq = dq.contiguous()
    dk = dk.contiguous()
    _triton_rope[n_row,](dq, dq.stride(1), dk, dk.stride(1), cos, cos.
        stride(-2), sin, sin.stride(-2), seq_len, batch_size,
        cos_batch_size, n_q_head, n_kv_head, head_dim, pad_n_q_head,
        pad_n_kv_head, pad_hd, BLOCK_SIZE=BLOCK_SIZE, BACKWARD_PASS=True)
    return dq.transpose(1, 2), dk.transpose(1, 2)


# Backward method (kernel launch code)
def _LigerRopeFunction_backward(ctx, dq, dk):
    """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
    cos, sin = ctx.saved_tensors
    dq, dk = rope_backward(dq, dk, cos, sin)
    return dq, dk, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LigerRopeFunction(torch.autograd.Function):
    """
    Triton implementation of the Rotary Positional Embedding (RoPE) operation. Please note that
    this implements the HuggingFace Llama & Mistral version, whose rotation matrix is slightly different
    than the original RoPE paper.

    Please find the corresponding HuggingFace implementation here:
    https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/llama/modeling_llama.py#L184

    For more details about the rotation matrix used here, please refer to:
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/2
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """
        q size: (bsz, n_q_head, seq_len, head_dim)
        k size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    def backward(ctx, dq, dk):
        """
        dq size: (bsz, n_q_head, seq_len, head_dim)
        dk size: (bsz, n_kv_head, seq_len, head_dim)
        cos size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        sin size: (1, seq_len, head_dim) or (bsz, seq_len, head_dim)
        """
        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None, None, None
