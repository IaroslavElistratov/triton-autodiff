# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/corl-team/rebased
# Source-Files: flash_linear_attention/fla/ops/triton/gla/chunk_fuse.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_m2n9_3ro/rebased-main/flash_linear_attention/fla/ops/triton/gla/chunk_fuse.py
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
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd
from einops import rearrange

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def fused_chunk_gla_fwd_kernel(q, k, v, g, o, initial_state, final_state,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: tl.
    constexpr, BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl
    .constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.
    constexpr):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        0, i_k * BK), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        0, i_k * BK), (BT, BK), (1, 0))
    p_db = g + i_bh * s_qk_h + (BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (
        i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (
        0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (
        s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (
            DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_g *= inv_ln2
        d_b = tl.load(p_db) * inv_ln2
        b_q = b_q * scale * tl.math.exp2(b_g)
        b_k = b_k * tl.trans(tl.math.exp2(-b_g + d_b[None, :]))
        b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
        b_h *= tl.math.exp2(d_b)[:, None]
        b_h += tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_g = tl.advance(p_g, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_db += BT * DK
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV),
            (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=
            (0, 1))


# Forward method (kernel launch code)
@contiguous
@custom_fwd
def _FusedChunkGLAFunction_forward(ctx, q, k, v, g, scale, initial_state,
    output_final_state):
    ctx.g_dtype = g.dtype
    batch_size, n_heads, seq_len, d_head_qk = q.shape
    d_head_v = v.shape[-1]
    ctx.scale = scale
    BT = 16
    BK, BV = min(d_head_qk, 64), min(d_head_v, 64)
    num_stages = 1
    num_warps = 2
    NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
    g = rearrange(g, 'b h (n c) d -> b h n c d', c=BT)
    g = g.float().cumsum(-2)
    g = rearrange(g, 'b h n c d -> b h (n c) d')
    if output_final_state:
        final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v,
            dtype=torch.float32, requires_grad=False)
    else:
        final_state = None
    grid = NV, NK, batch_size * n_heads
    fused_chunk_gla_fwd_kernel[grid](q, k, v, g, o, initial_state,
        final_state, q.stride(1), q.stride(2), q.stride(3), v.stride(1), v.
        stride(2), v.stride(3), batch_size, n_heads, seq_len, scale, BT=BT,
        DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV, USE_INITIAL_STATE=
        initial_state is not None, STORE_FINAL_STATE=output_final_state,
        num_warps=num_warps, num_stages=num_stages)
    o = o.sum(0)
    chunk_size = 16
    num_chunk = seq_len // chunk_size
    q2 = rearrange(q, 'b h (n c) d -> b h n c d', n=num_chunk)
    k2 = rearrange(k, 'b h (n c) d -> b h n c d', n=num_chunk)
    v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
    g2 = rearrange(g, 'b h (n c) d -> b h n c d', n=num_chunk)
    A = semiring_cal_A.forward(q2, k2, g2) * scale
    o2 = A @ v2
    o2 = rearrange(o2, 'b h n c d -> b h (n c) d')
    o.add_(o2)
    ctx.save_for_backward(q, k, v, g, A, initial_state)
    return o.to(v), final_state


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def fused_chunk_gla_bwd_kernel(q, k, v, g, do, dq, dk, dv, initial_state,
    s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: tl.
    constexpr, BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl
    .constexpr, USE_INITIAL_STATE: tl.constexpr):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (
            1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d
            ), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d
            ), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + ((i + 1) * BT - 1
            ) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t
            ), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t,
            s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, DK
            ), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)) * inv_ln2
        d_b = tl.load(p_db) * inv_ln2
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
        b_k *= tl.math.exp2(d_b[None, :] - b_g)
        b_h *= tl.math.exp2(d_b)[None, :]
        b_h += tl.dot(b_v, b_k.to(b_v.dtype), allow_tf32=False)
        b_dq *= scale * tl.math.exp2(b_g)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t
            ), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d
            ), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d
            ), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + (T - (i - 1) * BT - 1
            ) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d
            ), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t,
            s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK
            ), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV
            ), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)) * inv_ln2
        b_db = tl.load(p_db) * inv_ln2
        g_k = tl.math.exp2(b_db[None, :] - b_g)
        b_k *= g_k
        b_q *= tl.math.exp2(tl.trans(b_g))
        b_dk = tl.trans(tl.dot(b_dh.to(b_v.dtype), tl.trans(b_v),
            allow_tf32=False)) * scale * g_k
        b_dv = tl.dot(b_k.to(b_v.dtype), b_dh.to(b_v.dtype), allow_tf32=False
            ) * scale
        b_dh *= tl.math.exp2(b_db)[:, None]
        b_dh += tl.dot(b_q.to(b_do.dtype), b_do, allow_tf32=False)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# Backward method (kernel launch code)
@contiguous
@custom_bwd
def _FusedChunkGLAFunction_backward(ctx, do, d_final_state=None):
    q, k, v, g, A, initial_state = ctx.saved_tensors
    batch_size, n_heads, seq_len, d_head_qk = q.shape
    d_head_v = v.shape[-1]
    scale = ctx.scale
    BT = 16
    BK, BV = min(d_head_qk, 64), min(d_head_v, 64)
    NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    num_stages = 1
    num_warps = 2
    dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
    dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
    dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
    grid = NV, NK, batch_size * n_heads
    fused_chunk_gla_bwd_kernel[grid](q, k, v, g, do, dq, dk, dv,
        initial_state, q.stride(1), q.stride(2), q.stride(3), v.stride(1),
        v.stride(2), v.stride(3), batch_size, n_heads, seq_len, scale, BT=
        BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV, USE_INITIAL_STATE=
        initial_state is not None, num_warps=num_warps, num_stages=num_stages)
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dg = dq * q
    dg.add_(-dk * k)
    num_chunk = seq_len // BT
    q2 = rearrange(q, 'b h (n c) d -> b h n c d', n=num_chunk)
    k2 = rearrange(k, 'b h (n c) d -> b h n c d', n=num_chunk)
    v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
    g2 = rearrange(g, 'b h (n c) d -> b h n c d', n=num_chunk)
    do2 = rearrange(do, 'b h (n c) d -> b h n c d', n=num_chunk)
    dA2 = do2 @ v2.transpose(-2, -1) * scale
    dv2 = A.transpose(-1, -2) @ do2
    dq2, dk2, dg2 = semiring_cal_A.backward(q2, k2, g2, dA2)
    dq2 = rearrange(dq2, '... h n c d -> ... h (n c) d')
    dk2 = rearrange(dk2, '... h n c d -> ... h (n c) d')
    dv2 = rearrange(dv2, '... h n c d -> ... h (n c) d')
    dg2 = rearrange(dg2, '... h n c d -> ... h (n c) d')
    dq.add_(dq2.to(dq))
    dk.add_(dk2.to(dk))
    dv.add_(dv2.to(dv))
    dg = dg.float()
    dg.add_(dg2)
    dg_cumsum = dg.cumsum(-2)
    dg = dg - dg_cumsum + dg_cumsum[:, :, -1, None]
    return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.g_dtype), None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        ctx.g_dtype = g.dtype
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        ctx.scale = scale
        BT = 16
        BK, BV = min(d_head_qk, 64), min(d_head_v, 64)
        num_stages = 1
        num_warps = 2
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        g = rearrange(g, 'b h (n c) d -> b h n c d', c=BT)
        g = g.float().cumsum(-2)
        g = rearrange(g, 'b h n c d -> b h (n c) d')
        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk,
                d_head_v, dtype=torch.float32, requires_grad=False)
        else:
            final_state = None
        grid = NV, NK, batch_size * n_heads
        fused_chunk_gla_fwd_kernel[grid](q, k, v, g, o, initial_state,
            final_state, q.stride(1), q.stride(2), q.stride(3), v.stride(1),
            v.stride(2), v.stride(3), batch_size, n_heads, seq_len, scale,
            BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None, STORE_FINAL_STATE=
            output_final_state, num_warps=num_warps, num_stages=num_stages)
        o = o.sum(0)
        chunk_size = 16
        num_chunk = seq_len // chunk_size
        q2 = rearrange(q, 'b h (n c) d -> b h n c d', n=num_chunk)
        k2 = rearrange(k, 'b h (n c) d -> b h n c d', n=num_chunk)
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        g2 = rearrange(g, 'b h (n c) d -> b h n c d', n=num_chunk)
        A = semiring_cal_A.forward(q2, k2, g2) * scale
        o2 = A @ v2
        o2 = rearrange(o2, 'b h n c d -> b h (n c) d')
        o.add_(o2)
        ctx.save_for_backward(q, k, v, g, A, initial_state)
        return o.to(v), final_state

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, d_final_state=None):
        q, k, v, g, A, initial_state = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = ctx.scale
        BT = 16
        BK, BV = min(d_head_qk, 64), min(d_head_v, 64)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 2
        dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        grid = NV, NK, batch_size * n_heads
        fused_chunk_gla_bwd_kernel[grid](q, k, v, g, do, dq, dk, dv,
            initial_state, q.stride(1), q.stride(2), q.stride(3), v.stride(
            1), v.stride(2), v.stride(3), batch_size, n_heads, seq_len,
            scale, BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None, num_warps=
            num_warps, num_stages=num_stages)
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        dg = dq * q
        dg.add_(-dk * k)
        num_chunk = seq_len // BT
        q2 = rearrange(q, 'b h (n c) d -> b h n c d', n=num_chunk)
        k2 = rearrange(k, 'b h (n c) d -> b h n c d', n=num_chunk)
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        g2 = rearrange(g, 'b h (n c) d -> b h n c d', n=num_chunk)
        do2 = rearrange(do, 'b h (n c) d -> b h n c d', n=num_chunk)
        dA2 = do2 @ v2.transpose(-2, -1) * scale
        dv2 = A.transpose(-1, -2) @ do2
        dq2, dk2, dg2 = semiring_cal_A.backward(q2, k2, g2, dA2)
        dq2 = rearrange(dq2, '... h n c d -> ... h (n c) d')
        dk2 = rearrange(dk2, '... h n c d -> ... h (n c) d')
        dv2 = rearrange(dv2, '... h n c d -> ... h (n c) d')
        dg2 = rearrange(dg2, '... h n c d -> ... h (n c) d')
        dq.add_(dq2.to(dq))
        dk.add_(dk2.to(dk))
        dv.add_(dv2.to(dv))
        dg = dg.float()
        dg.add_(dg2)
        dg_cumsum = dg.cumsum(-2)
        dg = dg - dg_cumsum + dg_cumsum[:, :, -1, None]
        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.g_dtype
            ), None, None, None
