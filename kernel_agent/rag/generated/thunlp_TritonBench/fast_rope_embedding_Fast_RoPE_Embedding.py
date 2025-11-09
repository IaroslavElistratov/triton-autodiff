# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/thunlp/TritonBench
# Source-Files: data/TritonBench_G_v1/fast_rope_embedding.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_chw4t45n/TritonBench-main/data/TritonBench_G_v1/fast_rope_embedding.py
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
def _rope_embedding(Q, Q_row_stride, cos, cos_row_stride, sin,
    sin_row_stride, seqlen, head_dim: tl.constexpr, n_heads: tl.constexpr,
    BACKWARD_PASS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    ROPE_GROUP_SIZE = 4
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim
    sin1 = tl.load(sin + row_position % seqlen * sin_row_stride + 
        half_head_dim * 0 + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + row_position % seqlen * cos_row_stride + 
        half_head_dim * 0 + col_offsets, mask=mask, other=0)
    if BACKWARD_PASS:
        sin1 = -sin1
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min(head_start + ROPE_GROUP_SIZE, n_heads)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = (row_position * Q_row_stride + k * head_dim + col_offsets +
            half_head_dim)
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)
        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

def calculate_settings(n: int) ->(int, int):
    BLOCK_SIZE: int = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f'Cannot launch Triton kernel since n = {n} exceeds the maximum CUDA blocksize = {MAX_FUSED_SIZE}.'
            )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# Forward method (kernel launch code)
def _Fast_RoPE_Embedding_forward(ctx, Q, cos, sin):
    cos, sin = cos.squeeze(), sin.squeeze()
    batch, seq_len, n_heads, head_dim = Q.shape
    Q = Q.reshape(batch * seq_len, n_heads * head_dim)
    n_rows, n_cols = Q.shape
    assert seq_len <= cos.shape[0]
    BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
    div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
    n_groups = div + (mod != 0)
    _rope_embedding[n_rows, n_groups](Q, Q.stride(0), cos, cos.stride(0),
        sin, sin.stride(0), seq_len, head_dim, n_heads, BACKWARD_PASS=False,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    ctx.BLOCK_SIZE = BLOCK_SIZE
    ctx.num_warps = num_warps
    ctx.n_groups = n_groups
    ctx.cos = cos
    ctx.sin = sin
    return Q.view(batch, seq_len, n_heads, head_dim)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _Fast_RoPE_Embedding_backward(ctx, dY):
    batch, seq_len, n_heads, head_dim = dY.shape
    dY = dY.reshape(batch * seq_len, n_heads * head_dim)
    n_rows, n_cols = dY.shape
    cos = ctx.cos
    sin = ctx.sin
    _rope_embedding[n_rows, ctx.n_groups](dY, dY.stride(0), cos, cos.stride
        (0), sin, sin.stride(0), seq_len, head_dim, n_heads, BACKWARD_PASS=
        True, BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)
    dY = dY.view(batch, seq_len, n_heads, head_dim)
    return dY, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Fast_RoPE_Embedding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.reshape(batch * seq_len, n_heads * head_dim)
        n_rows, n_cols = Q.shape
        assert seq_len <= cos.shape[0]
        BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2)
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)
        _rope_embedding[n_rows, n_groups](Q, Q.stride(0), cos, cos.stride(0
            ), sin, sin.stride(0), seq_len, head_dim, n_heads,
            BACKWARD_PASS=False, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.reshape(batch * seq_len, n_heads * head_dim)
        n_rows, n_cols = dY.shape
        cos = ctx.cos
        sin = ctx.sin
        _rope_embedding[n_rows, ctx.n_groups](dY, dY.stride(0), cos, cos.
            stride(0), sin, sin.stride(0), seq_len, head_dim, n_heads,
            BACKWARD_PASS=True, BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.
            num_warps)
        dY = dY.view(batch, seq_len, n_heads, head_dim)
        return dY, None, None
