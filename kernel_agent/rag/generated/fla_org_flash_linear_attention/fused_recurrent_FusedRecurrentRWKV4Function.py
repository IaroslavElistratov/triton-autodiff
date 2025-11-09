# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/rwkv4/fused_recurrent.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/rwkv4/fused_recurrent.py
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

def get_block_size_c(chans: int) ->int:
    if chans < 32:
        return 32
    if chans < 64:
        return 64
    return 128


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def fused_recurrent_rwkv4_forward_kernel(w_ptr, w_s_c, u_ptr, u_s_c, k_ptr,
    k_s_b, k_s_t, k_s_c, v_ptr, v_s_b, v_s_t, v_s_c, state_ptr, state_s_b,
    state_s_abe, state_s_c, wkv_ptr, wkv_s_b, wkv_s_t, wkv_s_c,
    state_out_ptr, state_out_s_b, state_out_s_abe, state_out_s_t,
    state_out_s_c, chans, tsz, BLOCK_SIZE_C: tl.constexpr):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    cs = c_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe
    wkv_ptr = wkv_ptr + b_idx * wkv_s_b
    alpha_out_ptr = state_out_ptr + b_idx * state_out_s_b
    beta_out_ptr = state_out_ptr + b_idx * state_out_s_b + state_out_s_abe
    eps_out_ptr = state_out_ptr + b_idx * state_out_s_b + 2 * state_out_s_abe
    alpha = tl.load(alpha_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    beta = tl.load(beta_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    eps = tl.load(eps_ptr + cs * state_s_c, mask=cmask).to(tl.float32)
    w = tl.load(w_ptr + cs * w_s_c, mask=cmask).to(tl.float32)
    u = tl.load(u_ptr + cs * u_s_c, mask=cmask).to(tl.float32)
    for t in range(tsz):
        kt = tl.load(k_ptr + t * k_s_t + cs * k_s_c, mask=cmask).to(tl.float32)
        vt = tl.load(v_ptr + t * v_s_t + cs * v_s_c, mask=cmask).to(tl.float32)
        ukt = u + kt
        tau = tl.maximum(ukt, eps)
        e1a = exp(eps - tau)
        e2a = exp(ukt - tau)
        wkv = (e1a * alpha + e2a * vt) / (e1a * beta + e2a)
        tl.store(wkv_ptr + t * wkv_s_t + cs * wkv_s_c, wkv, mask=cmask)
        w_eps = w + eps
        eps = tl.maximum(w_eps, kt)
        e1b = exp(w_eps - eps)
        e2b = exp(kt - eps)
        alpha = e1b * alpha + e2b * vt
        beta = e1b * beta + e2b
        tl.store(alpha_out_ptr + t * state_out_s_t + cs * state_out_s_c,
            alpha, mask=cmask)
        tl.store(beta_out_ptr + t * state_out_s_t + cs * state_out_s_c,
            beta, mask=cmask)
        tl.store(eps_out_ptr + t * state_out_s_t + cs * state_out_s_c, eps,
            mask=cmask)


def fused_recurrent_rwkv4_forward(w: Tensor, u: Tensor, k: Tensor, v:
    Tensor, state: Tensor) ->tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape
    wkvs = k.new_empty(bsz, tsz, chans)
    state_out = k.new_empty(bsz, 3, tsz, chans)
    block_size_c = get_block_size_c(chans)

    def grid(meta: dict[str, Any]) ->tuple[int, ...]:
        return bsz, triton.cdiv(chans, meta['BLOCK_SIZE_C'])
    fused_recurrent_rwkv4_forward_kernel[grid](w, w.stride(0), u, u.stride(
        0), k, k.stride(0), k.stride(1), k.stride(2), v, v.stride(0), v.
        stride(1), v.stride(2), state, state.stride(0), state.stride(1),
        state.stride(3), wkvs, wkvs.stride(0), wkvs.stride(1), wkvs.stride(
        2), state_out, state_out.stride(0), state_out.stride(1), state_out.
        stride(2), state_out.stride(3), chans, tsz, BLOCK_SIZE_C=block_size_c)
    state_out = torch.cat((state, state_out), dim=2)
    return wkvs, state_out


# Forward method (kernel launch code)
@input_guard
def _FusedRecurrentRWKV4Function_forward(ctx: FunctionCtx, w: Tensor, u:
    Tensor, k: Tensor, v: Tensor, state: Tensor) ->tuple[Tensor, Tensor]:
    ctx.w_dtype = w.dtype
    w = -torch.exp(w.float())
    wkv, state_out = fused_recurrent_rwkv4_forward(w, u, k, v, state)
    ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
    return wkv, state_out[:, :, -1:]


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def fused_recurrent_rwkv4_backward_kernel(w_ptr, w_s_c, u_ptr, u_s_c, k_ptr,
    k_s_b, k_s_t, k_s_c, v_ptr, v_s_b, v_s_t, v_s_c, state_ptr, state_s_b,
    state_s_abe, state_s_t, state_s_c, gwkv_ptr, gwkv_s_b, gwkv_s_t,
    gwkv_s_c, gstate_out_ptr, gstate_out_s_b, gstate_out_s_abe,
    gstate_out_s_c, gw_ptr, gw_s_b, gw_s_c, gu_ptr, gu_s_b, gu_s_c, gk_ptr,
    gk_s_b, gk_s_t, gk_s_c, gv_ptr, gv_s_b, gv_s_t, gv_s_c, gstate_ptr,
    gstate_s_b, gstate_s_abe, gstate_s_c, tsz, chans, BLOCK_SIZE_C: tl.
    constexpr):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    cs = c_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe
    gk_ptr = gk_ptr + b_idx * gk_s_b
    gv_ptr = gv_ptr + b_idx * gv_s_b
    gwkv_ptr = gwkv_ptr + b_idx * gwkv_s_b
    galpha_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b
    gbeta_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + gstate_out_s_abe
    geps_out_ptr = (gstate_out_ptr + b_idx * gstate_out_s_b + 2 *
        gstate_out_s_abe)
    galpha = tl.load(galpha_out_ptr + gstate_out_s_c * cs, mask=cmask).to(tl
        .float32)
    gbeta = tl.load(gbeta_out_ptr + gstate_out_s_c * cs, mask=cmask).to(tl.
        float32)
    geps = tl.load(geps_out_ptr + gstate_out_s_c * cs, mask=cmask).to(tl.
        float32)
    w = tl.load(w_ptr + w_s_c * cs, mask=cmask).to(tl.float32)
    u = tl.load(u_ptr + u_s_c * cs, mask=cmask).to(tl.float32)
    gw = tl.zeros_like(w)
    gu = tl.zeros_like(u)
    alpha_prev = tl.load(alpha_ptr + tsz * state_s_t + state_s_c * cs, mask
        =cmask).to(tl.float32)
    beta_prev = tl.load(beta_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask
        ).to(tl.float32)
    eps_prev = tl.load(eps_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask
        ).to(tl.float32)
    for t in range(tsz):
        tc = tsz - t - 1
        kt = tl.load(k_ptr + tc * k_s_t + k_s_c * cs, mask=cmask).to(tl.float32
            )
        vt = tl.load(v_ptr + tc * v_s_t + v_s_c * cs, mask=cmask).to(tl.float32
            )
        alpha_curr = alpha_prev
        beta_curr = beta_prev
        eps_curr = eps_prev
        alpha_prev = tl.load(alpha_ptr + tc * state_s_t + state_s_c * cs,
            mask=cmask).to(tl.float32)
        beta_prev = tl.load(beta_ptr + tc * state_s_t + state_s_c * cs,
            mask=cmask).to(tl.float32)
        eps_prev = tl.load(eps_ptr + tc * state_s_t + state_s_c * cs, mask=
            cmask).to(tl.float32)
        ukt = u + kt
        tau = tl.maximum(ukt, eps_prev)
        e1 = exp(eps_prev - tau)
        e2 = exp(ukt - tau)
        euke = exp(ukt + eps_prev - 2 * tau)
        denom = e1 * beta_prev + e2
        denom_sq = denom * denom
        gwkvt = tl.load(gwkv_ptr + tc * gwkv_s_t + gwkv_s_c * cs, mask=cmask
            ).to(tl.float32)
        guk = gwkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
        gu += guk
        gk = guk
        gv = gwkvt * e2 / denom
        galpha_wkv = gwkvt * e1 / denom
        gbeta_wkv = -gwkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
        geps_wkv_denom = e1 * beta_prev + e2
        geps_wkv = gwkvt * euke * (alpha_prev - vt * beta_prev) / (
            geps_wkv_denom * geps_wkv_denom)
        e1 = exp(w + eps_prev - eps_curr)
        e2 = exp(kt - eps_curr)
        galpha_we = galpha * e1 * alpha_prev
        gw += galpha_we
        gk += galpha * e2 * vt
        gv += galpha * e2
        geps += galpha * -alpha_curr
        gbeta_we = gbeta * e1 * beta_prev
        gw += gbeta_we
        gk += gbeta * e2
        geps += gbeta * -beta_curr
        geps_mask = w + eps_prev > kt
        geps_we = tl.where(geps_mask, geps, tl.zeros_like(geps))
        gw += geps_we
        gk += tl.where(geps_mask, tl.zeros_like(geps), geps)
        tl.store(gk_ptr + tc * gk_s_t + gk_s_c * cs, gk, mask=cmask)
        tl.store(gv_ptr + tc * gv_s_t + gv_s_c * cs, gv, mask=cmask)
        galpha = galpha * e1 + galpha_wkv
        gbeta = gbeta * e1 + gbeta_wkv
        geps = galpha_we + gbeta_we + geps_we + geps_wkv
    galpha_ptr = gstate_ptr + b_idx * gstate_s_b
    gbeta_ptr = gstate_ptr + b_idx * gstate_s_b + gstate_s_abe
    geps_ptr = gstate_ptr + b_idx * gstate_s_b + 2 * gstate_s_abe
    tl.store(galpha_ptr + gstate_s_c * cs, galpha, mask=cmask)
    tl.store(gbeta_ptr + gstate_s_c * cs, gbeta, mask=cmask)
    tl.store(geps_ptr + gstate_s_c * cs, geps, mask=cmask)
    tl.store(gw_ptr + gw_s_b * b_idx + gw_s_c * cs, gw * w, mask=cmask)
    tl.store(gu_ptr + gu_s_b * b_idx + gu_s_c * cs, gu, mask=cmask)


def fused_recurrent_rwkv4_backward(w: Tensor, u: Tensor, k: Tensor, v:
    Tensor, state: Tensor, grad_wkv: Tensor, grad_state: Tensor) ->tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape
    gw = w.new_empty(bsz, chans, dtype=torch.float)
    gu = u.new_empty(bsz, chans, dtype=torch.float)
    gk = torch.empty_like(k)
    gv = torch.empty_like(v)
    gstate = k.new_empty(bsz, 3, 1, chans)
    block_size_c = get_block_size_c(chans)

    def grid(meta: dict[str, Any]) ->tuple[int, ...]:
        return bsz, triton.cdiv(chans, meta['BLOCK_SIZE_C'])
    fused_recurrent_rwkv4_backward_kernel[grid](w, w.stride(0), u, u.stride
        (0), k, k.stride(0), k.stride(1), k.stride(2), v, v.stride(0), v.
        stride(1), v.stride(2), state, state.stride(0), state.stride(1),
        state.stride(2), state.stride(3), grad_wkv, grad_wkv.stride(0),
        grad_wkv.stride(1), grad_wkv.stride(2), grad_state, grad_state.
        stride(0), grad_state.stride(1), grad_state.stride(3), gw, gw.
        stride(0), gw.stride(1), gu, gu.stride(0), gu.stride(1), gk, gk.
        stride(0), gk.stride(1), gk.stride(2), gv, gv.stride(0), gv.stride(
        1), gv.stride(2), gstate, gstate.stride(0), gstate.stride(1),
        gstate.stride(3), tsz, chans, BLOCK_SIZE_C=block_size_c)
    return gw.sum(0), gu.sum(0), gk, gv, gstate


# Backward method (kernel launch code)
@once_differentiable
@input_guard
def _FusedRecurrentRWKV4Function_backward(ctx: FunctionCtx, gwkv: Tensor,
    gstate: Tensor) ->tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
    gw, gu, gk, gv, gstate = fused_recurrent_rwkv4_backward(w, u, k, v,
        state, gwkv, gstate)
    return gw.to(ctx.w_dtype), gu.to(u), gk.to(k), gv.to(v), gstate.to(state)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedRecurrentRWKV4Function(Function):

    @staticmethod
    @input_guard
    def forward(ctx: FunctionCtx, w: Tensor, u: Tensor, k: Tensor, v:
        Tensor, state: Tensor) ->tuple[Tensor, Tensor]:
        ctx.w_dtype = w.dtype
        w = -torch.exp(w.float())
        wkv, state_out = fused_recurrent_rwkv4_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    @input_guard
    def backward(ctx: FunctionCtx, gwkv: Tensor, gstate: Tensor) ->tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        gw, gu, gk, gv, gstate = fused_recurrent_rwkv4_backward(w, u, k, v,
            state, gwkv, gstate)
        return gw.to(ctx.w_dtype), gu.to(u), gk.to(k), gv.to(v), gstate.to(
            state)
