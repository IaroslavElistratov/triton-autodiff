# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/hustvl/ViG
# Source-Files: flash-linear-attention/fla/ops/gla/bak_recurrent_fuse.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_2wwopzcg/ViG-main/flash-linear-attention/fla/ops/gla/bak_recurrent_fuse.py
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
from math import exp

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def bid_fused_recurrent_gla_fwd_kernel(q, k, v, gk, gv, o, initial_state,
    final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T,
    scale, BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.
    constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.
    constexpr, REVERSE: tl.constexpr, USE_GK: tl.constexpr, USE_GV: tl.
    constexpr):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + 0
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + 0
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + 0
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + 0
    inv_p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    inv_p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    inv_p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    inv_p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV
        ) + (T - 1) * DV
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + 0
        inv_p_gk = gk + B * H * s_qk_h + i_bh * s_qk_h + i_k * BK + tl.arange(
            0, BK) + (T - 1) * DK
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) *
            DV if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    h = tl.zeros([BV, BK], dtype=tl.float32)
    inv_h = tl.zeros([BV, BK], dtype=tl.float32)
    mask_kv = mask_bk[None, :] & mask_bv[:, None]
    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0,
            BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _inv_k = tl.load(inv_p_k, mask=mask_bk, other=0).to(tl.float32)
        _inv_v = tl.load(inv_p_v, mask=mask_bv, other=0).to(tl.float32)
        _inv_q = tl.load(inv_p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            h = h * _gk[None, :]
            _inv_gk = tl.load(inv_p_gk, mask=mask_bk, other=0).to(tl.float32)
            inv_h = inv_h * _inv_gk[None, :]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            h = h * _gv[:, None]
        h += _k[None, :] * _v[:, None]
        inv_h += _inv_k[None, :] * _inv_v[:, None]
        _o = h * _q[None, :]
        _inv_o = inv_h * _inv_q[None, :]
        _o = tl.sum(_o, axis=1)
        _inv_o = tl.sum(_inv_o, axis=1)
        fw_o = tl.load(p_o, mask=mask_bv, other=0)
        _o = _o + fw_o
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)
        bw_o = tl.load(inv_p_o, mask=mask_bv, other=0)
        _inv_o = _inv_o + bw_o
        tl.store(inv_p_o, _inv_o.to(inv_p_o.dtype.element_ty), mask=mask_bv)
        p_q += -DK if REVERSE else DK
        p_k += -DK if REVERSE else DK
        p_o += -DV if REVERSE else DV
        p_v += -DV if REVERSE else DV
        inv_p_q += -DK
        inv_p_k += -DK
        inv_p_o += -DV
        inv_p_v += -DV
        if USE_GK:
            p_gk += -DK if REVERSE else DK
            inv_p_gk += -DK
        if USE_GV:
            p_gv += -DV if REVERSE else DV
    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + (i_k * BK + tl.arange(0,
            BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)


# Forward method (kernel launch code)
@contiguous
@custom_fwd
def _BidFusedRecurrentGLAFunction_forward(ctx, q, k, v, gk, gv, scale=None,
    initial_state=None, output_final_state=False, reverse=False):
    batch_size, n_heads, seq_len, d_head_qk = q.shape
    d_head_v = v.shape[-1]
    if scale is None:
        scale = d_head_qk ** -0.5
    if gk is not None:
        gk = gk.float().exp()
    if gv is not None:
        gv = gv.float().exp()
    BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
    NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    num_stages = 1
    num_warps = 1
    o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v, dtype=torch
        .float32)
    if output_final_state:
        final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v)
    else:
        final_state = None
    grid = NV, NK, batch_size * n_heads
    bid_fused_recurrent_gla_fwd_kernel[grid](q, k, v, gk, gv, o,
        initial_state, final_state, q.stride(1), q.stride(2), q.stride(3),
        v.stride(1), v.stride(2), v.stride(3), batch_size, n_heads, seq_len,
        scale, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV, USE_INITIAL_STATE=
        initial_state is not None, STORE_FINAL_STATE=final_state is not
        None, USE_GK=gk is not None, USE_GV=gv is not None, REVERSE=reverse,
        num_warps=num_warps, num_stages=num_stages)
    o = o.sum(0)
    ctx.save_for_backward(q, k, v, gk, gv, initial_state, o)
    ctx.scale = scale
    ctx.reverse = reverse
    if final_state is not None:
        final_state = final_state.detach()
    return o.to(q.dtype), final_state


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def fused_recurrent_gla_bwd_kernel(q, k, v, gk, gv, do, dq, dk, dv,
    initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T,
    scale, BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.
    constexpr, USE_INITIAL_STATE: tl.constexpr, REVERSE: tl.constexpr,
    USE_GK: tl.constexpr, USE_GV: tl.constexpr):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if
        REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if
        REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if
        REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if
        REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * DK if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) *
            DK if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) *
            DV if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0,
            BK)[:, None]) * DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            h = h * _gk[:, None]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            h = h * _gv[None, :]
        h += _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)
        p_k += -DK if REVERSE else DK
        p_v += -DV if REVERSE else DV
        p_q += -DK if REVERSE else DK
        p_do += -DV if REVERSE else DV
        p_dq += -DK if REVERSE else DK
        if USE_GK:
            p_gk += -DK if REVERSE else DK
        if USE_GV:
            p_gv += -DV if REVERSE else DV
    tl.debug_barrier()
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if
        not REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if
        not REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if
        not REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if
        not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (
        (T - 1) * DK if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (
        (T - 1) * DV if not REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) *
            DK if not REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) *
            DV if not REVERSE else 0)
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0).to(tl.float32)
            d_h *= _gk[:, None]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0).to(tl.float32)
            d_h *= _gv[None, :]
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)
        p_do += DV if REVERSE else -DV
        p_q += DK if REVERSE else -DK
        p_k += DK if REVERSE else -DK
        p_v += DV if REVERSE else -DV
        p_dk += DK if REVERSE else -DK
        p_dv += DV if REVERSE else -DV
        if USE_GK:
            p_gk += DK if REVERSE else -DK
        if USE_GV:
            p_gv += DV if REVERSE else -DV


# Backward method (kernel launch code)
@contiguous
@custom_bwd
def _BidFusedRecurrentGLAFunction_backward(ctx, do, d_final_state=None):
    q, k, v, gk, gv, initial_state, o = ctx.saved_tensors
    batch_size, n_heads, seq_len, d_head_qk = q.shape
    d_head_v = v.shape[-1]
    scale = ctx.scale
    BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
    NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
    num_stages = 1
    num_warps = 1
    dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk, dtype=
        torch.float32)
    dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk, dtype=
        torch.float32)
    dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v, dtype=
        torch.float32)
    grid = NV, NK, batch_size * n_heads
    fused_recurrent_gla_bwd_kernel[grid](q, k, v, gk, gv, do, dq, dk, dv,
        initial_state, q.stride(1), q.stride(2), q.stride(3), v.stride(1),
        v.stride(2), v.stride(3), batch_size, n_heads, seq_len, scale, DK=
        d_head_qk, DV=d_head_v, BK=BK, BV=BV, num_warps=num_warps,
        num_stages=num_stages, USE_INITIAL_STATE=initial_state is not None,
        REVERSE=ctx.reverse, USE_GK=gk is not None, USE_GV=gv is not None)
    dq = dq.sum(0)
    dk = dk.sum(0)
    dv = dv.sum(0)
    if gk is not None:
        _dgk = dq * q.float() - dk * k.float()
        if ctx.reverse:
            dgk = _dgk.cumsum(-2)
        else:
            _dgk_cumsum = _dgk.cumsum(-2)
            dgk = _dgk + _dgk_cumsum[:, :, -1, None] - _dgk_cumsum
    else:
        dgk = None
    if gv is not None:
        _dgv = do.float() * o.float() - dv * v.float()
        if ctx.reverse:
            dgv = _dgv.cumsum(-2)
        else:
            _dgv_cumsum = _dgv.cumsum(-2)
            dgv = _dgv + _dgv_cumsum[:, :, -1, None] - _dgv_cumsum
    else:
        dgv = None
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype
        ), dgk, dgv, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class BidFusedRecurrentGLAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, q, k, v, gk, gv, scale=None, initial_state=None,
        output_final_state=False, reverse=False):
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        if scale is None:
            scale = d_head_qk ** -0.5
        if gk is not None:
            gk = gk.float().exp()
        if gv is not None:
            gv = gv.float().exp()
        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1
        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v, dtype=
            torch.float32)
        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v)
        else:
            final_state = None
        grid = NV, NK, batch_size * n_heads
        bid_fused_recurrent_gla_fwd_kernel[grid](q, k, v, gk, gv, o,
            initial_state, final_state, q.stride(1), q.stride(2), q.stride(
            3), v.stride(1), v.stride(2), v.stride(3), batch_size, n_heads,
            seq_len, scale, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None, STORE_FINAL_STATE=
            final_state is not None, USE_GK=gk is not None, USE_GV=gv is not
            None, REVERSE=reverse, num_warps=num_warps, num_stages=num_stages)
        o = o.sum(0)
        ctx.save_for_backward(q, k, v, gk, gv, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        if final_state is not None:
            final_state = final_state.detach()
        return o.to(q.dtype), final_state

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, do, d_final_state=None):
        q, k, v, gk, gv, initial_state, o = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = ctx.scale
        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1
        dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk, dtype
            =torch.float32)
        dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk, dtype
            =torch.float32)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v, dtype=
            torch.float32)
        grid = NV, NK, batch_size * n_heads
        fused_recurrent_gla_bwd_kernel[grid](q, k, v, gk, gv, do, dq, dk,
            dv, initial_state, q.stride(1), q.stride(2), q.stride(3), v.
            stride(1), v.stride(2), v.stride(3), batch_size, n_heads,
            seq_len, scale, DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            num_warps=num_warps, num_stages=num_stages, USE_INITIAL_STATE=
            initial_state is not None, REVERSE=ctx.reverse, USE_GK=gk is not
            None, USE_GV=gv is not None)
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        if gk is not None:
            _dgk = dq * q.float() - dk * k.float()
            if ctx.reverse:
                dgk = _dgk.cumsum(-2)
            else:
                _dgk_cumsum = _dgk.cumsum(-2)
                dgk = _dgk + _dgk_cumsum[:, :, -1, None] - _dgk_cumsum
        else:
            dgk = None
        if gv is not None:
            _dgv = do.float() * o.float() - dv * v.float()
            if ctx.reverse:
                dgv = _dgv.cumsum(-2)
            else:
                _dgv_cumsum = _dgv.cumsum(-2)
                dgv = _dgv + _dgv_cumsum[:, :, -1, None] - _dgv_cumsum
        else:
            dgv = None
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype
            ), dgk, dgv, None, None, None, None
