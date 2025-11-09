# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/codefuse-ai/rodimus
# Source-Files: ops/rotary.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ykm1l2x1/rodimus-main/ops/rotary.py
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
from einops import rearrange

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[torch.
    Tensor]=None, max_seqlen: Optional[int]=None, interleaved=False,
    inplace=False, conjugate=False) ->torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, 'If cu_seqlens is passed in, then max_seqlen must be passed'
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, 'rotary_dim must be <= headdim'
    assert headdim <= 256, 'Only support headdim <= 256'
    assert seqlen_ro >= seqlen, 'seqlen_ro must be >= seqlen'
    assert cos.dtype == sin.dtype, f'cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}'
    assert x.dtype == cos.dtype, f'Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}'
    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro
    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])
    BLOCK_K = (32 if rotary_dim <= 32 else 64 if rotary_dim <= 64 else 128 if
        rotary_dim <= 128 else 256)
    grid = lambda META: (triton.cdiv(seqlen, META['BLOCK_M']), batch, nheads)
    BLOCK_M = 4 if interleaved else 8 if rotary_dim <= 64 else 4
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](output, x, cos, sin, cu_seqlens, seqlen_offsets,
            seqlen, rotary_dim, seqlen_ro, output.stride(0) if not
            is_varlen else 0, output.stride(-3), output.stride(-2), output.
            stride(-1), x.stride(0) if not is_varlen else 0, x.stride(-3),
            x.stride(-2), x.stride(-1), BLOCK_K, isinstance(seqlen_offsets,
            torch.Tensor), is_varlen, interleaved, conjugate, BLOCK_M)
    return output


@triton.jit
def rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS, seqlen,
    rotary_dim, seqlen_ro, stride_out_batch, stride_out_seqlen,
    stride_out_nheads, stride_out_headdim, stride_x_batch, stride_x_seqlen,
    stride_x_nheads, stride_x_headdim, BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr, IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2
    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = (OUT + start_idx * stride_out_seqlen + pid_head *
            stride_out_nheads)
    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)
    if not INTERLEAVED:
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] *
            stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[
            None, :] < rotary_dim_half), other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[
            None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x0 = tl.load(X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] <
            rotary_dim_half), other=0.0).to(tl.float32)
        x1 = tl.load(X + rotary_dim_half * stride_x_headdim, mask=(rm[:,
            None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0
            ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] *
            stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] <
            rotary_dim_half))
        tl.store(OUT + rotary_dim_half * stride_out_headdim, o1, mask=(rm[:,
            None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    else:
        rk_swap = rk + (rk + 1) % 2 * 2 - 1
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] *
            stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] *
            stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[
            None, :] < rotary_dim_half), other=1.0).to(tl.float32)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[
            None, :] < rotary_dim_half), other=0.0).to(tl.float32)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] <
            rotary_dim), other=0.0).to(tl.float32)
        x1 = tl.load(X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] <
            rotary_dim), other=0.0).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] *
            stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] <
            rotary_dim))


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def _ApplyRotaryEmbQKV__forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None,
    interleaved=False, seqlen_offsets: Union[int, torch.Tensor]=0):
    batch, seqlen, three, nheads, headdim = qkv.shape
    assert three == 3
    if cos_k is None and sin_k is None and qkv.is_contiguous():
        qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
        apply_rotary(qk, cos, sin, seqlen_offsets=seqlen_offsets,
            interleaved=interleaved, inplace=True)
    else:
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        q, k = qkv[:, :, 0], qkv[:, :, 1]
        apply_rotary(q, cos, sin, seqlen_offsets, interleaved=interleaved,
            inplace=True)
        apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=
            interleaved, inplace=True)
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
    if isinstance(seqlen_offsets, int):
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        ctx.seqlen_offsets = seqlen_offsets
    else:
        ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
        ctx.seqlen_offsets = None
    ctx.interleaved = interleaved
    return qkv


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _ApplyRotaryEmbQKV__backward(ctx, dqkv):
    seqlen_offsets = ctx.seqlen_offsets
    if seqlen_offsets is None:
        cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
    else:
        cos, sin, cos_k, sin_k = ctx.saved_tensors
    if cos_k is None and sin_k is None and dqkv.is_contiguous():
        dqk = rearrange(dqkv[:, :, :2], 'b s t h d -> b s (t h) d')
        apply_rotary(dqk, cos, sin, seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved, inplace=True, conjugate=True)
    else:
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
        apply_rotary(dq, cos, sin, seqlen_offsets, interleaved=ctx.
            interleaved, inplace=True, conjugate=True)
        apply_rotary(dk, cos_k, sin_k, seqlen_offsets, interleaved=ctx.
            interleaved, inplace=True, conjugate=True)
    return dqkv, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ApplyRotaryEmbQKV_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None, interleaved=
        False, seqlen_offsets: Union[int, torch.Tensor]=0):
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            apply_rotary(qk, cos, sin, seqlen_offsets=seqlen_offsets,
                interleaved=interleaved, inplace=True)
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            apply_rotary(q, cos, sin, seqlen_offsets, interleaved=
                interleaved, inplace=True)
            apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=
                interleaved, inplace=True)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            dqk = rearrange(dqkv[:, :, :2], 'b s t h d -> b s (t h) d')
            apply_rotary(dqk, cos, sin, seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved, inplace=True, conjugate=True)
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
            apply_rotary(dq, cos, sin, seqlen_offsets, interleaved=ctx.
                interleaved, inplace=True, conjugate=True)
            apply_rotary(dk, cos_k, sin_k, seqlen_offsets, interleaved=ctx.
                interleaved, inplace=True, conjugate=True)
        return dqkv, None, None, None, None, None, None
