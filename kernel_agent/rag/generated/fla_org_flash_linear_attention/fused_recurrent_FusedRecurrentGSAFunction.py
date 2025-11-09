# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/gsa/fused_recurrent.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/gsa/fused_recurrent.py
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
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [4, 8]], key=['BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_GK',
    'USE_GV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['B', 'T'])
def fused_recurrent_fwd_kernel(q, k, v, g, g_gamma, gk, gv, o, h0, ht,
    cu_seqlens, scale, B, T, H: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BK: tl.constexpr, BV: tl.constexpr, REVERSE: tl.constexpr,
    USE_G: tl.constexpr, USE_G_GAMMA: tl.constexpr, USE_GK: tl.constexpr,
    USE_GV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.
        int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + (T - 1 if REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + (T - 1 if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + (T - 1 if REVERSE else 0)) * H * V + i_h * V + o_v
    p_o = o + (i_k * all + bos + (T - 1 if REVERSE else 0)
        ) * H * V + i_h * V + o_v
    if USE_G:
        p_g = g + (bos + (T - 1 if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + (T - 1 if REVERSE else 0)) * H * K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + (bos + (T - 1 if REVERSE else 0)) * H * V + i_h * V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_h = b_h * exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            b_h = b_h * exp(b_gv[None, :])
        b_h += b_k[:, None] * b_v[None, :]
        b_o = b_h * b_q[:, None]
        b_o = tl.sum(b_o, axis=0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)
        p_q += (-1 if REVERSE else 1) * H * K
        p_k += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_o += (-1 if REVERSE else 1) * H * V
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H * K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H * V
    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=m_h)


@triton.jit
def fused_recurrent_gsa_inference_kernel(q, k, v, s, g, o, hk0, hv0, hkt,
    hvt, scale, K: tl.constexpr, V: tl.constexpr, M: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, NG: tl.constexpr):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG
    b_s = tl.load(s + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = tl.load(g + i_bg * M + tl.arange(0, M)).to(tl.float32)
    b_g = exp(b_g)
    b_ok = tl.zeros([M], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_hk0 = hk0 + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[:, None
            ]
        mask_k = o_k < K
        mask_hk = (tl.arange(0, M) < M)[:, None] & mask_k[None, :]
        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.0).to(tl.float32)
        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.0).to(tl.float32
            ) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.0).to(tl.float32
            )
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)
        if i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[
                :, None]
            tl.store(p_hkt, b_hk.to(p_hkt.dtype.element_ty), mask=mask_hk)
    b_qv = tl.softmax(b_ok)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_hv0 = hv0 + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None
            ]
        mask_v = o_v < V
        mask_hv = mask_v[:, None] & (tl.arange(0, M) < M)[None, :]
        b_hv = tl.load(p_hv0, mask=mask_hv, other=0).to(tl.float32)
        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)
        tl.store(o + i_bh * V + o_v, b_ov.to(o.dtype.element_ty), mask=mask_v)
        if i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[
                :, None]
            tl.store(p_hvt, b_hv.to(p_hvt.dtype.element_ty), mask=mask_hv)


def fused_recurrent_gsa_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, s: torch.Tensor, g: torch.Tensor, initial_state: Optional[Tuple
    [torch.Tensor, torch.Tensor]]=None, output_final_state: bool=False,
    scale: float=1.0, reverse: bool=False, cu_seqlens: Optional[torch.
    LongTensor]=None) ->Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    HQ = q.shape[2]
    if HQ != H:
        raise ValueError('GQA not supported yet.')
    BK, BV, BM = min(triton.next_power_of_2(K), 64), min(triton.
        next_power_of_2(V), 64), min(triton.next_power_of_2(M), 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)
    hk0, hv0 = None, None
    if initial_state != (None, None) and initial_state is not None:
        hk0, hv0 = initial_state
    hkt, hvt = None, None
    if output_final_state:
        hkt, hvt = q.new_empty(N, H, K, M, dtype=torch.float), q.new_empty(N,
            H, M, V, dtype=torch.float)
    ok = q.new_empty(NK, *s.shape, dtype=torch.float)
    gk, gv = None, g
    grid = NM, NK, N * H
    fused_recurrent_fwd_kernel[grid](q=q, k=k, v=s, g=None, g_gamma=None,
        gk=gk, gv=gv, o=ok, h0=hk0, ht=hkt, cu_seqlens=cu_seqlens, scale=
        scale, B=B, T=T, H=H, K=K, V=M, BK=BK, BV=BM, USE_G=False,
        USE_G_GAMMA=False, USE_GK=False, USE_GV=True, REVERSE=reverse)
    ok = ok.sum(0)
    qv = ok.softmax(-1, dtype=torch.float)
    ov = q.new_empty(NM, *v.shape, dtype=torch.float)
    gk, gv = g, None
    grid = NV, NM, N * H
    fused_recurrent_fwd_kernel[grid](q=qv, k=s, v=v, g=None, g_gamma=None,
        gk=gk, gv=gv, o=ov, h0=hv0, ht=hvt, cu_seqlens=cu_seqlens, scale=
        1.0, B=B, T=T, H=H, K=M, V=V, BK=BM, BV=BV, USE_G=False,
        USE_G_GAMMA=False, USE_GK=True, USE_GV=False, REVERSE=reverse)
    ov = ov.sum(0)
    return ok, hkt, qv, ov, hvt


def fused_recurrent_gsa_inference(q: torch.Tensor, k: torch.Tensor, v:
    torch.Tensor, s: torch.Tensor, g: torch.Tensor, initial_state: Optional
    [Tuple[torch.Tensor, torch.Tensor]]=None, output_final_state: bool=
    False, scale: float=1.0) ->torch.Tensor:
    B, T, H, K, V, M = *k.shape, v.shape[-1], s.shape[-1]
    HQ = q.shape[2]
    BK, BV = min(triton.next_power_of_2(K), 64), min(triton.next_power_of_2
        (V), 64)
    NG = HQ // H
    if initial_state != (None, None) and initial_state is not None:
        hk0, hv0 = initial_state
    else:
        hk0, hv0 = q.new_zeros(B, H, K, M, dtype=torch.float), q.new_zeros(B,
            H, M, V, dtype=torch.float)
    hkt, hvt = None, None
    if output_final_state:
        if NG == 1:
            hkt, hvt = hk0, hv0
        else:
            hkt, hvt = q.new_empty(B, H, K, M, dtype=torch.float), q.new_empty(
                B, H, M, V, dtype=torch.float)
    o = v.new_empty(B, T, HQ, V)
    grid = B * HQ,
    fused_recurrent_gsa_inference_kernel[grid](q, k, v, s, g, o, hk0, hv0,
        hkt, hvt, scale=scale, K=K, V=V, M=M, BK=BK, BV=BV, NG=NG)
    return o, (hkt, hvt)


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _FusedRecurrentGSAFunction_forward(ctx, q: torch.Tensor, k: torch.
    Tensor, v: torch.Tensor, s: torch.Tensor, g: torch.Tensor, scale:
    Optional[float]=None, hk0: Optional[torch.Tensor]=None, hv0: Optional[
    torch.Tensor]=None, output_final_state: bool=False, reverse: bool=False,
    cu_seqlens: Optional[torch.LongTensor]=None) ->Tuple[torch.Tensor,
    Tuple[torch.Tensor]]:
    T = q.shape[1]
    if T == 1 and not q.requires_grad:
        o, (hkt, hvt) = fused_recurrent_gsa_inference(q=q, k=k, v=v, s=s, g
            =g, initial_state=(hk0, hv0), output_final_state=
            output_final_state, scale=scale)
        return o, hkt, hvt
    ok, hkt, qv, ov, hvt = fused_recurrent_gsa_fwd(q=q, k=k, v=v, s=s, g=g,
        initial_state=(hk0, hv0), output_final_state=output_final_state,
        scale=scale, reverse=reverse, cu_seqlens=cu_seqlens)
    ctx.save_for_backward(q, k, v, s, g, qv, hk0, hv0, ok)
    ctx.scale = scale
    ctx.reverse = reverse
    ctx.cu_seqlens = cu_seqlens
    return ov.to(q.dtype), hkt, hvt


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not
    None, 'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [4]], key=['BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_GK',
    'USE_GV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['B', 'T'])
def fused_recurrent_bwd_kernel(q, k, v, g, g_gamma, gk, gv, o, h0, do, dq,
    dk, dv, dg, dgk, dgv, dht, dh0, cu_seqlens, scale, B, T, H: tl.
    constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.
    constexpr, REVERSE: tl.constexpr, USE_G: tl.constexpr, USE_G_GAMMA: tl.
    constexpr, USE_GK: tl.constexpr, USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_INITIAL_STATE_GRADIENT: tl.
    constexpr, USE_FINAL_STATE_GRADIENT: tl.constexpr, IS_VARLEN: tl.constexpr
    ):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.
        int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    all = B * T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    NV = tl.cdiv(V, BV)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_h = m_k[:, None] & m_v[None, :]
    p_k = k + (bos + (T - 1 if REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + (T - 1 if REVERSE else 0)) * H * V + i_h * V + o_v
    p_do = do + (bos + (T - 1 if REVERSE else 0)) * H * V + i_h * V + o_v
    p_dq = dq + (i_v * all + bos + (T - 1 if REVERSE else 0)
        ) * H * K + i_h * K + o_k
    if USE_G:
        p_g = g + (bos + (T - 1 if REVERSE else 0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + (T - 1 if REVERSE else 0)) * H * K + i_h * K + o_k
    if USE_GV:
        p_gv = gv + (bos + (T - 1 if REVERSE else 0)) * H * V + i_h * V + o_v
    if USE_G_GAMMA:
        b_g_gamma = tl.load(g_gamma + i_h)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0).to(tl.float32)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h = b_h * exp(b_g)
        if USE_G_GAMMA:
            b_h = b_h * exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_h = b_h * exp(b_gk[:, None])
        if USE_GV:
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            b_h = b_h * exp(b_gv[None, :])
        b_h += b_k[:, None] * b_v[None, :]
        b_dq = b_h * b_do[None, :]
        b_dq = tl.sum(b_dq, axis=1) * scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_k)
        p_k += (-1 if REVERSE else 1) * H * K
        p_v += (-1 if REVERSE else 1) * H * V
        p_do += (-1 if REVERSE else 1) * H * V
        p_dq += (-1 if REVERSE else 1) * H * K
        if USE_G:
            p_g += (-1 if REVERSE else 1) * H
        if USE_GK:
            p_gk += (-1 if REVERSE else 1) * H * K
        if USE_GV:
            p_gv += (-1 if REVERSE else 1) * H * V
    tl.debug_barrier()
    p_q = q + (bos + (T - 1 if not REVERSE else 0)) * H * K + i_h * K + o_k
    p_k = k + (bos + (T - 1 if not REVERSE else 0)) * H * K + i_h * K + o_k
    p_v = v + (bos + (T - 1 if not REVERSE else 0)) * H * V + i_h * V + o_v
    p_do = do + (bos + (T - 1 if not REVERSE else 0)) * H * V + i_h * V + o_v
    p_dq = dq + (i_v * all + bos + (T - 1 if not REVERSE else 0)
        ) * H * K + i_h * K + o_k
    p_dk = dk + (i_v * all + bos + (T - 1 if not REVERSE else 0)
        ) * H * K + i_h * K + o_k
    p_dv = dv + (i_k * all + bos + (T - 1 if not REVERSE else 0)
        ) * H * V + i_h * V + o_v
    if USE_G:
        p_g = g + (bos + (T - 1 if not REVERSE else 0)) * H + i_h
        p_dg = dg + ((i_k * NV + i_v) * all + bos + (T - 1 if not REVERSE else
            0)) * H + i_h
    if USE_GK:
        p_gk = gk + (bos + (T - 1 if not REVERSE else 0)
            ) * H * K + i_h * K + o_k
        p_dgk = dgk + (i_v * all + bos + (T - 1 if not REVERSE else 0)
            ) * H * K + i_h * K + o_k
    if USE_GV:
        p_o = o + (bos + (T - 1 if not REVERSE else 0)) * H * V + i_h * V + o_v
        p_gv = gv + (bos + (T - 1 if not REVERSE else 0)
            ) * H * V + i_h * V + o_v
        p_dgv = dgv + (i_k * all + bos + (T - 1 if not REVERSE else 0)
            ) * H * V + i_h * V + o_v
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_dh += tl.load(p_dht, mask=m_h, other=0).to(tl.float32)
    if USE_G:
        b_dg = tl.sum(b_h * b_dh)
    if USE_GK:
        b_dgk = tl.sum(b_h * b_dh, 1)
    if USE_GV:
        b_dgv = tl.sum(b_h * b_dh, 0)
    for _ in range(T):
        b_q = tl.load(p_q, mask=m_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=m_v, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_v, other=0).to(tl.float32)
        b_dh += (b_q * scale)[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_dq = tl.load(p_dq, mask=m_k, other=0).to(tl.float32)
            b_dg += tl.sum(b_q * b_dq - b_k * b_dk)
            b_dh *= exp(b_g)
            tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty))
        if USE_G_GAMMA:
            b_dh *= exp(b_g_gamma)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
            b_dq = tl.load(p_dq, mask=m_k, other=0).to(tl.float32)
            b_dgk += b_q * b_dq - b_k * b_dk
            b_dh *= exp(b_gk)[:, None]
            tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), mask=m_k)
        if USE_GV:
            b_o = tl.load(p_o, mask=m_v, other=0).to(tl.float32)
            b_gv = tl.load(p_gv, mask=m_v, other=0).to(tl.float32)
            if i_k == 0:
                b_dgv += b_o * b_do
            b_dgv -= b_v * b_dv
            b_dh *= exp(b_gv)[None, :]
            tl.store(p_dgv, b_dgv.to(p_dgv.dtype.element_ty), mask=m_v)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)
        p_q += (1 if REVERSE else -1) * H * K
        p_k += (1 if REVERSE else -1) * H * K
        p_v += (1 if REVERSE else -1) * H * V
        p_do += (1 if REVERSE else -1) * H * V
        p_dq += (1 if REVERSE else -1) * H * K
        p_dk += (1 if REVERSE else -1) * H * K
        p_dv += (1 if REVERSE else -1) * H * V
        if USE_G:
            p_g += (1 if REVERSE else -1) * H
            p_dg += (1 if REVERSE else -1) * H
        if USE_GK:
            p_gk += (1 if REVERSE else -1) * H * K
            p_dgk += (1 if REVERSE else -1) * H * K
        if USE_GV:
            p_o += (1 if REVERSE else -1) * H * V
            p_gv += (1 if REVERSE else -1) * H * V
            p_dgv += (1 if REVERSE else -1) * H * V
    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = dh0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=m_h)


def fused_recurrent_gsa_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, s: torch.Tensor, g: torch.Tensor, qv: torch.Tensor, hk0:
    Optional[torch.Tensor]=None, hv0: Optional[torch.Tensor]=None, ok:
    Optional[torch.Tensor]=None, do: Optional[torch.Tensor]=None, dhkt:
    Optional[torch.Tensor]=None, dhvt: Optional[torch.Tensor]=None, scale:
    float=1.0, reverse: bool=False, cu_seqlens: Optional[torch.LongTensor]=None
    ) ->Tuple[torch.Tensor]:
    B, T, H, K, V, M = *q.shape, v.shape[-1], s.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV, BM = min(triton.next_power_of_2(K), 64), min(triton.
        next_power_of_2(V), 64), min(triton.next_power_of_2(M), 64)
    NK, NV, NM = triton.cdiv(K, BK), triton.cdiv(V, BV), triton.cdiv(M, BM)
    dqv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
    dsv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
    dv = q.new_empty(NM, B, T, H, V, dtype=torch.float)
    dgv = q.new_empty(NV, B, T, H, M, dtype=torch.float)
    dhv0 = torch.empty_like(hv0) if hv0 is not None else None
    grid = NV, NM, N * H
    fused_recurrent_bwd_kernel[grid](q=qv, k=s, v=v, g=None, g_gamma=None,
        gk=g, gv=None, o=None, h0=hv0, do=do, dq=dqv, dk=dsv, dv=dv, dg=
        None, dgk=dgv, dgv=None, dht=dhvt, dh0=dhv0, cu_seqlens=cu_seqlens,
        scale=1.0, B=B, T=T, H=H, K=M, V=V, BK=BM, BV=BV, USE_G=False,
        USE_G_GAMMA=False, USE_GK=True, USE_GV=False, REVERSE=reverse)
    dqv = dqv.sum(0)
    dsv = dsv.sum(0)
    dv = dv.sum(0)
    dgv = dgv.sum(0)
    dok = qv * (dqv - (qv * dqv).sum(-1, True))
    dq = q.new_empty(NM, B, T, H, K, dtype=torch.float)
    dk = q.new_empty(NM, B, T, H, K, dtype=torch.float)
    dsk = q.new_empty(NK, B, T, H, M, dtype=torch.float)
    dgk = q.new_empty(NK, B, T, H, M, dtype=torch.float)
    dhk0 = torch.empty_like(hk0) if hk0 is not None else None
    grid = NM, NK, N * H
    fused_recurrent_bwd_kernel[grid](q=q, k=k, v=s, g=None, g_gamma=None,
        gk=None, gv=g, o=ok, h0=hk0, do=dok, dq=dq, dk=dk, dv=dsk, dg=None,
        dgk=None, dgv=dgk, dht=dhkt, dh0=dhk0, cu_seqlens=cu_seqlens, scale
        =scale, B=B, T=T, H=H, K=K, V=M, BK=BK, BV=BM, USE_G=False,
        USE_G_GAMMA=False, USE_GK=False, USE_GV=True, REVERSE=reverse)
    dq = dq.sum(0)
    dk = dk.sum(0)
    dsk = dsk.sum(0)
    dgk = dgk.sum(0)
    ds = dsk.add_(dsv)
    dg = dgk.add_(dgv)
    return dq, dk, dv, ds, dg, dhk0, dhv0


# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _FusedRecurrentGSAFunction_backward(ctx, do, dhkt=None, dhvt=None):
    q, k, v, s, g, qv, hk0, hv0, ok = ctx.saved_tensors
    scale = ctx.scale
    reverse = ctx.reverse
    cu_seqlens = ctx.cu_seqlens
    dq, dk, dv, ds, dg, dhk0, dhv0 = fused_recurrent_gsa_bwd(q=q, k=k, v=v,
        s=s, g=g, qv=qv, hk0=hk0, hv0=hv0, ok=ok, do=do, dhkt=dhkt, dhvt=
        dhvt, scale=scale, reverse=reverse, cu_seqlens=cu_seqlens)
    return dq.to(q), dk.to(k), dv.to(v), ds.to(s), dg.to(g
        ), None, dhk0, dhv0, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedRecurrentGSAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, s:
        torch.Tensor, g: torch.Tensor, scale: Optional[float]=None, hk0:
        Optional[torch.Tensor]=None, hv0: Optional[torch.Tensor]=None,
        output_final_state: bool=False, reverse: bool=False, cu_seqlens:
        Optional[torch.LongTensor]=None) ->Tuple[torch.Tensor, Tuple[torch.
        Tensor]]:
        T = q.shape[1]
        if T == 1 and not q.requires_grad:
            o, (hkt, hvt) = fused_recurrent_gsa_inference(q=q, k=k, v=v, s=
                s, g=g, initial_state=(hk0, hv0), output_final_state=
                output_final_state, scale=scale)
            return o, hkt, hvt
        ok, hkt, qv, ov, hvt = fused_recurrent_gsa_fwd(q=q, k=k, v=v, s=s,
            g=g, initial_state=(hk0, hv0), output_final_state=
            output_final_state, scale=scale, reverse=reverse, cu_seqlens=
            cu_seqlens)
        ctx.save_for_backward(q, k, v, s, g, qv, hk0, hv0, ok)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        return ov.to(q.dtype), hkt, hvt

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dhkt=None, dhvt=None):
        q, k, v, s, g, qv, hk0, hv0, ok = ctx.saved_tensors
        scale = ctx.scale
        reverse = ctx.reverse
        cu_seqlens = ctx.cu_seqlens
        dq, dk, dv, ds, dg, dhk0, dhv0 = fused_recurrent_gsa_bwd(q=q, k=k,
            v=v, s=s, g=g, qv=qv, hk0=hk0, hv0=hv0, ok=ok, do=do, dhkt=dhkt,
            dhvt=dhvt, scale=scale, reverse=reverse, cu_seqlens=cu_seqlens)
        return dq.to(q), dk.to(k), dv.to(v), ds.to(s), dg.to(g
            ), None, dhk0, dhv0, None, None, None
