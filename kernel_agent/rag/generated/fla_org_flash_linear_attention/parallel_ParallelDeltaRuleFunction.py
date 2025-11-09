# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/delta_rule/parallel.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/delta_rule/parallel.py
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

@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [1, 2, 4]], key=['BT', 'K', 'V'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_transform_qk_fwd_kernel(q, k, v, beta, o, A, q_new, k_new,
    A_local, scale, T, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, BT: tl.constexpr, OUTPUT_ATTENTIONS: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0),
        (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0),
        (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0),
        (BT, BV), (1, 0))
    b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(p_q.dtype.element_ty
        )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    p_T = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT,
        0), (BT, BT), (1, 0))
    b_T = tl.load(p_T, boundary_check=(0, 1))
    o_i = tl.arange(0, BT)
    m_t = o_i[:, None] >= o_i[None, :]
    b_qk = tl.where(m_t, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0).to(
        b_q.dtype)
    m_t = o_i[:, None] > o_i[None, :]
    b_kk = tl.where(m_t, tl.dot(b_k, tl.trans(b_k), allow_tf32=False), 0).to(
        b_k.dtype)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (
        BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
    b_qkT = tl.dot(b_qk, b_T, allow_tf32=False).to(b_k.dtype)
    if OUTPUT_ATTENTIONS:
        p_a = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1),
            (i_t * BT, 0), (BT, BT), (1, 0))
        tl.store(p_a, b_qkT.to(p_a.dtype.element_ty), boundary_check=(0, 1))
    b_kkT = tl.dot(b_kk, b_T, allow_tf32=False).to(b_k.dtype)
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0),
        (BT, BV), (1, 0))
    tl.store(p_o, tl.dot(b_qkT, b_v).to(p_o.dtype.element_ty),
        boundary_check=(0, 1))
    p_q_new = tl.make_block_ptr(q_new + i_bh * T * K, (T, K), (K, 1), (i_t *
        BT, 0), (BT, BK), (1, 0))
    tl.store(p_q_new, (b_q - tl.dot(b_qkT, b_k_beta, allow_tf32=False)).to(
        p_q_new.dtype.element_ty), boundary_check=(0, 1))
    p_k_new = tl.make_block_ptr(k_new + i_bh * T * K, (T, K), (K, 1), (i_t *
        BT, 0), (BT, BK), (1, 0))
    b_k_new = b_k - tl.dot(tl.trans(b_kkT), b_k_beta, allow_tf32=False)
    tl.store(p_k_new, b_k_new.to(p_k_new.dtype.element_ty), boundary_check=
        (0, 1))


@triton.heuristics({'OUTPUT_ATTENTIONS': lambda args: args['attn'] is not None}
    )
@triton.jit(do_not_specialize=['T'])
def parallel_delta_rule_fwd_kernel(q, k, k2, v, beta, o, o_new, attn, T, K:
    tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK:
    tl.constexpr, BV: tl.constexpr, OUTPUT_ATTENTIONS: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_q = tl.make_block_ptr(q + i_bh * T * K, (T, K), (K, 1), (i_t * BT, 0),
        (BT, BK), (1, 0))
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = tl.make_block_ptr(o + i_bh * T * V, (T, V), (V, 1), (i_t * BT, 0),
        (BT, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))
    for offset in range((i_t + 1) * BT - 2 * BS, i_t * BT - BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (0,
            offset), (BK, BS), (0, 1))
        p_k2 = tl.make_block_ptr(k2 + i_bh * T * K, (T, K), (K, 1), (offset,
            0), (BS, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (offset, 
            0), (BS, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (offset,),
            (BS,), (0,))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))
        m_s = tl.arange(0, BT) >= offset - i_t * BT + BS
        b_s = tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False)
        b_s = tl.where(m_s[:, None], b_s, 0)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        b_k2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None]).to(b_v
            .dtype)
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (
                i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
    for offset in range(i_t * BT - BS, -BS, -BS):
        p_k = tl.make_block_ptr(k + i_bh * T * K, (K, T), (1, K), (0,
            offset), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * T * V, (T, V), (V, 1), (offset, 
            0), (BS, BV), (1, 0))
        p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (offset,),
            (BS,), (0,))
        p_k2 = tl.make_block_ptr(k2 + i_bh * T * K, (T, K), (K, 1), (offset,
            0), (BS, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_s = tl.dot(b_q.to(b_k.dtype), b_k, allow_tf32=False)
        b_o += tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        b_k2 = (tl.load(p_k2, boundary_check=(0, 1)) * b_beta[:, None]).to(b_v
            .dtype)
        b_q -= tl.dot(b_s.to(b_v.dtype), b_k2, allow_tf32=False).to(b_q.dtype)
        if OUTPUT_ATTENTIONS:
            p_a = tl.make_block_ptr(attn + i_bh * T * T, (T, T), (T, 1), (
                i_t * BT, offset), (BT, BS), (1, 0))
            tl.store(p_a, b_s.to(p_a.dtype.element_ty), boundary_check=(0, 1))
    p_o_new = tl.make_block_ptr(o_new + i_bh * T * V, (T, V), (V, 1), (i_t *
        BT, 0), (BT, BV), (1, 0))
    tl.store(p_o_new, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2)], key=['BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def save_intra_chunk_attn(A, A_local, T, BT: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_A = tl.make_block_ptr(A + i_bh * T * T, (T, T), (T, 1), (i_t * BT, 
        i_t * BT), (BT, BT), (1, 0))
    p_A_local = tl.make_block_ptr(A_local + i_bh * T * BT, (T, BT), (BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_local = tl.load(p_A_local, boundary_check=(0, 1))
    tl.store(p_A, b_A_local.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_transform_qk_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, beta: torch.Tensor, A: torch.Tensor, scale: float, chunk_size:
    int, output_attentions: bool):
    B, H, T, K = k.shape
    BT = chunk_size
    q_new = torch.empty_like(q)
    k_new = torch.empty_like(k)
    o = torch.empty_like(v)
    grid = triton.cdiv(T, BT), B * H
    V = v.shape[-1]
    A_local = torch.empty_like(A) if output_attentions else None
    chunk_transform_qk_fwd_kernel[grid](q, k, v, beta, o, A, q_new, k_new,
        A_local, scale=scale, T=T, K=K, V=V, BT=BT, BK=triton.
        next_power_of_2(K), BV=triton.next_power_of_2(V), OUTPUT_ATTENTIONS
        =output_attentions)
    return q_new, k_new, o, A_local


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _ParallelDeltaRuleFunction_forward(ctx, q, k, v, beta, scale,
    output_attentions):
    B, H, T, K, V = *k.shape, v.shape[-1]
    assert q.shape[-1] <= 128, 'The maximum supported sequence length is 128.'
    BT, BS = 128, 32
    BK = triton.next_power_of_2(k.shape[-1])
    BV = triton.next_power_of_2(v.shape[-1])
    assert BT % BS == 0
    A = fwd_prepare_T(k, beta, BS)
    attn = q.new_zeros(B, H, T, T) if output_attentions else None
    q_new, k_new, o, A_local = chunk_transform_qk_fwd(q, k, v, beta, A,
        scale, BS, output_attentions)
    num_stages = 3 if K <= 64 else 2
    num_warps = 4
    grid = triton.cdiv(T, BT), B * H
    o_new = torch.empty_like(o)
    parallel_delta_rule_fwd_kernel[grid](q=q_new, k=k_new, k2=k, v=v, beta=
        beta, o=o, o_new=o_new, attn=attn, T=T, K=K, V=V, BT=BT, BS=BS, BK=
        BK, BV=BV, num_stages=num_stages, num_warps=num_warps)
    if output_attentions:
        grid = triton.cdiv(T, BS), B * H
        save_intra_chunk_attn[grid](A=attn, A_local=A_local, T=T, BT=BS)
    return o_new.to(q.dtype), attn


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _ParallelDeltaRuleFunction_backward(ctx, do, d_attn=None):
    raise NotImplementedError('Backward pass is not implemented. Stay tuned!')


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ParallelDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, beta, scale, output_attentions):
        B, H, T, K, V = *k.shape, v.shape[-1]
        assert q.shape[-1
            ] <= 128, 'The maximum supported sequence length is 128.'
        BT, BS = 128, 32
        BK = triton.next_power_of_2(k.shape[-1])
        BV = triton.next_power_of_2(v.shape[-1])
        assert BT % BS == 0
        A = fwd_prepare_T(k, beta, BS)
        attn = q.new_zeros(B, H, T, T) if output_attentions else None
        q_new, k_new, o, A_local = chunk_transform_qk_fwd(q, k, v, beta, A,
            scale, BS, output_attentions)
        num_stages = 3 if K <= 64 else 2
        num_warps = 4
        grid = triton.cdiv(T, BT), B * H
        o_new = torch.empty_like(o)
        parallel_delta_rule_fwd_kernel[grid](q=q_new, k=k_new, k2=k, v=v,
            beta=beta, o=o, o_new=o_new, attn=attn, T=T, K=K, V=V, BT=BT,
            BS=BS, BK=BK, BV=BV, num_stages=num_stages, num_warps=num_warps)
        if output_attentions:
            grid = triton.cdiv(T, BS), B * H
            save_intra_chunk_attn[grid](A=attn, A_local=A_local, T=T, BT=BS)
        return o_new.to(q.dtype), attn

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, d_attn=None):
        raise NotImplementedError(
            'Backward pass is not implemented. Stay tuned!')
