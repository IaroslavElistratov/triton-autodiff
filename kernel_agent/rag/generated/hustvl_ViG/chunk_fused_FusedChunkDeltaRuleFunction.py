# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/hustvl/ViG
# Source-Files: flash-linear-attention/fla/ops/delta_rule/chunk_fused.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_2wwopzcg/ViG-main/flash-linear-attention/fla/ops/delta_rule/chunk_fused.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def fused_chunk_delta_rule_fwd_kernel(q, k, v, v_new, d, o, initial_state,
    final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T,
    scale, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, DK: tl.
    constexpr, DV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, CHECK: tl.constexpr):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (
        i_k * BK, 0), (BK, BT), (0, 1))
    p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (
        0, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (
        0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (
        s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, DV), (s_vo_t,
        s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (
            DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_k.dtype)
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_v_prime = tl.dot(b_d, b_h.to(b_q.dtype), allow_tf32=False)
        b_v = b_v - b_v_prime
        tl.store(p_v_new, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))
        b_o = tl.dot(b_s.to(b_q.dtype), b_v.to(b_q.dtype), allow_tf32=False)
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v.to(b_k.dtype), allow_tf32=False)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_v_new = tl.advance(p_v_new, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_d = tl.advance(p_d, (BT, 0))
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV),
            (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=
            (0, 1))


# Forward method (kernel launch code)
@contiguous
@custom_fwd
def _FusedChunkDeltaRuleFunction_forward(ctx, q, k, v, d, BT, initial_state,
    output_final_state):
    batch_size, n_heads, seq_len, d_head_qk = q.shape
    d_head_v = v.shape[-1]
    scale = 1
    ctx.scale = 1
    BT = BT
    ctx.BT = BT
    BK, BV = triton.next_power_of_2(d_head_qk), min(triton.next_power_of_2(
        d_head_v), 32)
    NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    assert NK == 1, 'NK should be 1'
    num_stages = 1
    num_warps = 4
    o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
    if output_final_state:
        final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v,
            dtype=torch.float32, requires_grad=False)
    else:
        final_state = None
    CHECK = False
    if version.parse(triton.__version__) < version.parse('2.2.0'):
        import warnings
        warnings.warn(
            "Triton<2.2.0 detected for running this kernel, which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) that lead to significant precision loss. We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
            )
        CHECK = True
    grid = NV, NK, batch_size * n_heads
    v_new = torch.empty_like(v)
    fused_chunk_delta_rule_fwd_kernel[grid](q, k, v, v_new, d, o,
        initial_state, final_state, q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3), batch_size, n_heads, seq_len,
        scale, BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
        USE_INITIAL_STATE=initial_state is not None, STORE_FINAL_STATE=
        output_final_state, CHECK=CHECK, num_warps=num_warps, num_stages=
        num_stages)
    o = o.sum(0)
    ctx.save_for_backward(q, k, v_new, d, initial_state)
    ctx.CHECK = CHECK
    return o.to(q.dtype), final_state


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def fused_chunk_delta_rule_bwd_kernel(q, k, v, d, do, dq, dk, dv, dd,
    initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T,
    scale, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, DK: tl.
    constexpr, DV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, CHECK: tl
    .constexpr):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    m_s = o_i[:, None] <= o_i[None, :]
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t
            ), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t
            ), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d
            ), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
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
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0).to(b_q.dtype)
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s = tl.where(m_s, b_s, 0).to(b_q.dtype)
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        b_d = tl.load(p_d, boundary_check=(0, 1))
        if CHECK and i == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
            b_dh -= tl.dot(b_d, b_dv.to(b_d.dtype), allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
            b_dh -= tl.dot(b_d, b_dv.to(b_d.dtype), allow_tf32=False)
        tl.store(p_dk, (b_dk * scale).to(p_dk.dtype.element_ty),
            boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    m_s = o_i[:, None] >= o_i[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (
            1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)
    NT = tl.cdiv(T, BT)
    for i in range(0, NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d
            ), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t
            ), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t,
            s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, DK
            ), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        b_dq = tl.dot(b_ds.to(b_k.dtype), b_k, allow_tf32=False)
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        if i < NT - 1:
            p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, DV), (s_vo_t,
                s_vo_d), ((i + 1) * BT, i_v * BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dd = tl.dot(b_dv.to(b_k.dtype), b_h.to(b_k.dtype), allow_tf32
                =False)
            p_dd = tl.make_block_ptr(dd + (i_bh + i_v * B * H) * s_qk_h, (T,
                DK), (s_qk_t, s_qk_d), ((i + 1) * BT, i_k * BK), (BT, BK),
                (1, 0))
            tl.store(p_dd, -b_dd.to(p_dd.dtype.element_ty), boundary_check=
                (0, 1))


# Backward method (kernel launch code)
@custom_bwd
@contiguous
def _FusedChunkDeltaRuleFunction_backward(ctx, do, d_final_state=None):
    q, k, v, d, initial_state = ctx.saved_tensors
    batch_size, n_heads, seq_len, d_head_qk = q.shape
    d_head_v = v.shape[-1]
    scale = ctx.scale
    BT = ctx.BT
    BK, BV = triton.next_power_of_2(d_head_qk), min(triton.next_power_of_2(
        d_head_v), 32)
    NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    assert NK == 1
    num_stages = 1
    num_warps = 2
    dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
    dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
    dd = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
    dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
    grid = NV, NK, batch_size * n_heads
    fused_chunk_delta_rule_bwd_kernel[grid](q, k, v, d, do, dq, dk, dv, dd,
        initial_state, q.stride(1), q.stride(2), q.stride(3), v.stride(1),
        v.stride(2), v.stride(3), batch_size, n_heads, seq_len, scale, BT=
        BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV, USE_INITIAL_STATE=
        initial_state is not None, CHECK=ctx.CHECK, num_warps=num_warps,
        num_stages=num_stages)
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    dd = dd.sum(0)
    dd[:, :, 0:BT] = 0
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dd.to(d.dtype
        ), None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedChunkDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, d, BT, initial_state, output_final_state):
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = 1
        ctx.scale = 1
        BT = BT
        ctx.BT = BT
        BK, BV = triton.next_power_of_2(d_head_qk), min(triton.
            next_power_of_2(d_head_v), 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        assert NK == 1, 'NK should be 1'
        num_stages = 1
        num_warps = 4
        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk,
                d_head_v, dtype=torch.float32, requires_grad=False)
        else:
            final_state = None
        CHECK = False
        if version.parse(triton.__version__) < version.parse('2.2.0'):
            import warnings
            warnings.warn(
                "Triton<2.2.0 detected for running this kernel, which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) that lead to significant precision loss. We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
                )
            CHECK = True
        grid = NV, NK, batch_size * n_heads
        v_new = torch.empty_like(v)
        fused_chunk_delta_rule_fwd_kernel[grid](q, k, v, v_new, d, o,
            initial_state, final_state, q.stride(1), q.stride(2), q.stride(
            3), v.stride(1), v.stride(2), v.stride(3), batch_size, n_heads,
            seq_len, scale, BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None, STORE_FINAL_STATE=
            output_final_state, CHECK=CHECK, num_warps=num_warps,
            num_stages=num_stages)
        o = o.sum(0)
        ctx.save_for_backward(q, k, v_new, d, initial_state)
        ctx.CHECK = CHECK
        return o.to(q.dtype), final_state

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, do, d_final_state=None):
        q, k, v, d, initial_state = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = ctx.scale
        BT = ctx.BT
        BK, BV = triton.next_power_of_2(d_head_qk), min(triton.
            next_power_of_2(d_head_v), 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        assert NK == 1
        num_stages = 1
        num_warps = 2
        dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dd = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        grid = NV, NK, batch_size * n_heads
        fused_chunk_delta_rule_bwd_kernel[grid](q, k, v, d, do, dq, dk, dv,
            dd, initial_state, q.stride(1), q.stride(2), q.stride(3), v.
            stride(1), v.stride(2), v.stride(3), batch_size, n_heads,
            seq_len, scale, BT=BT, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None, CHECK=ctx.CHECK,
            num_warps=num_warps, num_stages=num_stages)
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        dd = dd.sum(0)
        dd[:, :, 0:BT] = 0
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dd.to(d.dtype
            ), None, None, None
