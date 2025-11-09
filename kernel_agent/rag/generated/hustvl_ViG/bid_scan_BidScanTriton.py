# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/hustvl/ViG
# Source-Files: flash-linear-attention/fla/ops/gla/bid_scan.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_2wwopzcg/ViG-main/flash-linear-attention/fla/ops/gla/bid_scan.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def triton_bid_scan(x, y, BC: tl.constexpr, BT: tl.constexpr, d_head: tl.
    constexpr, n_heads: tl.constexpr, batch_size: tl.constexpr, seq_len: tl
    .constexpr, NT: tl.constexpr):
    i_c, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch_idx = i_bh // n_heads
    head_idx = i_bh % n_heads
    block_start_seq = i_t * BT
    block_start_depth = i_c * BC
    seq_range = tl.arange(0, BT)
    depth_range = tl.arange(0, BC)
    seq_idx = block_start_seq + seq_range
    depth_idx = block_start_depth + depth_range
    mask = (seq_idx < seq_len)[:, None] & (depth_idx < d_head)
    offset_normal = (batch_idx * n_heads * seq_len * d_head + head_idx *
        seq_len * d_head + seq_idx[:, None] * d_head + depth_idx)
    offset_mirrored = (batch_idx * n_heads * seq_len * d_head + head_idx *
        seq_len * d_head + (seq_len - seq_idx - 1)[:, None] * d_head +
        depth_idx + batch_size * n_heads * seq_len * d_head)
    x_values = tl.load(x + offset_normal, mask=mask)
    tl.store(y + offset_normal, x_values, mask=mask)
    tl.store(y + offset_mirrored, x_values, mask=mask)


# Forward method (kernel launch code)
def _BidScanTriton_forward(ctx, x: torch.Tensor):
    """
        x: [batch_size, n_heads, seq_len, d_head]
        """
    batch_size, n_heads, seq_len, d_head = x.shape
    batch_size, n_heads, seq_len, d_head = int(batch_size), int(n_heads), int(
        seq_len), int(d_head)
    BC, BT = min(triton.next_power_of_2(d_head), 1), min(triton.
        next_power_of_2(seq_len), 64)
    NT, NC = triton.cdiv(seq_len, BT), triton.cdiv(d_head, BC)
    ctx.shape = batch_size, n_heads, seq_len, d_head
    ctx.triton_shape = BC, BT, NC, NT
    x = x.contiguous()
    y = x.new_empty((2 * batch_size, n_heads, seq_len, d_head))
    triton_bid_scan[NC, NT, batch_size * n_heads](x, y, BC, BT, d_head,
        n_heads, batch_size, seq_len, NT)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def triton_bid_merge(y, x, BC: tl.constexpr, BT: tl.constexpr, d_head: tl.
    constexpr, n_heads: tl.constexpr, batch_size: tl.constexpr, seq_len: tl
    .constexpr, NT: tl.constexpr):
    i_c = tl.program_id(0)
    i_t = tl.program_id(1)
    i_bh = tl.program_id(2)
    batch_idx = i_bh // n_heads
    head_idx = i_bh % n_heads
    block_start_seq = i_t * BT
    block_start_depth = i_c * BC
    seq_range = tl.arange(0, BT)
    depth_range = tl.arange(0, BC)
    seq_idx = block_start_seq + seq_range
    depth_idx = block_start_depth + depth_range
    mask = (seq_idx < seq_len)[:, None] & (depth_idx < d_head)
    offset_normal = (batch_idx * n_heads * seq_len * d_head + head_idx *
        seq_len * d_head + seq_idx[:, None] * d_head + depth_idx)
    offset_mirrored = (batch_idx * n_heads * seq_len * d_head + head_idx *
        seq_len * d_head + (seq_len - seq_idx - 1)[:, None] * d_head +
        depth_idx + batch_size * n_heads * seq_len * d_head)
    normal_vals = tl.load(y + offset_normal, mask=mask)
    mirrored_vals = tl.load(y + offset_mirrored, mask=mask)
    combined_vals = normal_vals + mirrored_vals
    tl.store(x + offset_normal, combined_vals, mask=mask)


# Backward method (kernel launch code)
def _BidScanTriton_backward(ctx, y: torch.Tensor):
    batch_size, n_heads, seq_len, d_head = ctx.shape
    BC, BT, NC, NT = ctx.triton_shape
    y = y.contiguous().view(2 * batch_size, n_heads, seq_len, d_head)
    x = y.new_empty((batch_size, n_heads, seq_len, d_head))
    triton_bid_merge[NC, NT, batch_size * n_heads](y, x, BC, BT, d_head,
        n_heads, batch_size, seq_len, NT)
    return x


# ============================================================
# autograd.Function Class Definition
# ============================================================

class BidScanTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """
        x: [batch_size, n_heads, seq_len, d_head]
        """
        batch_size, n_heads, seq_len, d_head = x.shape
        batch_size, n_heads, seq_len, d_head = int(batch_size), int(n_heads
            ), int(seq_len), int(d_head)
        BC, BT = min(triton.next_power_of_2(d_head), 1), min(triton.
            next_power_of_2(seq_len), 64)
        NT, NC = triton.cdiv(seq_len, BT), triton.cdiv(d_head, BC)
        ctx.shape = batch_size, n_heads, seq_len, d_head
        ctx.triton_shape = BC, BT, NC, NT
        x = x.contiguous()
        y = x.new_empty((2 * batch_size, n_heads, seq_len, d_head))
        triton_bid_scan[NC, NT, batch_size * n_heads](x, y, BC, BT, d_head,
            n_heads, batch_size, seq_len, NT)
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        batch_size, n_heads, seq_len, d_head = ctx.shape
        BC, BT, NC, NT = ctx.triton_shape
        y = y.contiguous().view(2 * batch_size, n_heads, seq_len, d_head)
        x = y.new_empty((batch_size, n_heads, seq_len, d_head))
        triton_bid_merge[NC, NT, batch_size * n_heads](y, x, BC, BT, d_head,
            n_heads, batch_size, seq_len, NT)
        return x
