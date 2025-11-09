# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/lucidrains/native-sparse-attention-pytorch
# Source-Files: native_sparse_attention_pytorch/triton_native_sparse_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_hik9xh7o/native-sparse-attention-pytorch-main/native_sparse_attention_pytorch/triton_native_sparse_attention.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def divisible_by(num, den):
    return num % den == 0


def exists(v):
    return v is not None


def is_contiguous(x: Tensor):
    return x.stride(-1) == 1


def round_up_multiple(n, mult):
    return ceil(n / mult) * mult


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics(dict(EVEN_M=lambda args: divisible_by(args['seqlen_q'],
    args['BLOCK']), EVEN_N=lambda args: divisible_by(args['seqlen_k'], args
    ['BLOCK']), EVEN_HEADDIM=lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM'], QUERY_EXPAND_DIM=lambda args: 16 // args[
    'QUERY_HEAD_GROUPS']))
@triton.jit
def forward_kernel(Q, K, V, kv_block_indices, kv_block_mask, Out,
    SlidingOut, Lse, SlidingLse, softmax_scale, stride_qb, stride_qh,
    stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh,
    stride_vn, stride_ob, stride_oh, stride_om, stride_kvbl_b,
    stride_kvbl_h, stride_kvbl_m, stride_lse_b, kv_heads, seqlen_q,
    seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K, BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr, QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr, NUM_SEL_KV_BLOCKS: tl.constexpr,
    NUM_BLOCKS_PER_SEL: tl.constexpr, INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    RETURN_SLIDING_OUT: tl.constexpr):
    if RETURN_SLIDING_OUT:
        sliding = tl.program_id(2) == 0
        out_ptr = SlidingOut if sliding else Out
        lse_ptr = SlidingLse if sliding else Lse
        num_sel_kv_blocks = 0 if sliding else NUM_SEL_KV_BLOCKS
    else:
        sliding = False
        out_ptr = Out
        lse_ptr = Lse
        num_sel_kv_blocks = NUM_SEL_KV_BLOCKS
    forward_kernel_causal_and_sparse(Q, K, V, kv_block_indices,
        kv_block_mask, out_ptr, lse_ptr, softmax_scale, stride_qb,
        stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb,
        stride_vh, stride_vn, stride_ob, stride_oh, stride_om,
        stride_kvbl_b, stride_kvbl_h, stride_kvbl_m, stride_lse_b, kv_heads,
        seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K, BLOCK_HEADDIM, EVEN_M, EVEN_N, EVEN_HEADDIM,
        BLOCK, SEL_BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM,
        num_sel_kv_blocks, NUM_BLOCKS_PER_SEL, INCLUDE_BLOCK_CAUSAL, sliding)


@triton.jit
def forward_kernel_causal_and_sparse(Q, K, V, kv_block_indices,
    kv_block_mask, Out, Lse, softmax_scale, stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om, stride_kvbl_b, stride_kvbl_h,
    stride_kvbl_m, stride_lse_b, kv_heads, seqlen_q, seqlen_k,
    seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr, BLOCK: tl.constexpr, SEL_BLOCK: tl.
    constexpr, QUERY_HEAD_GROUPS: tl.constexpr, QUERY_EXPAND_DIM: tl.
    constexpr, NUM_SEL_KV_BLOCKS: tl.constexpr, NUM_BLOCKS_PER_SEL: tl.
    constexpr, INCLUDE_BLOCK_CAUSAL: tl.constexpr, SLIDING: tl.constexpr):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // kv_heads
    off_h = off_hb % kv_heads
    offs_qh = off_h * QUERY_HEAD_GROUPS + tl.arange(0, QUERY_HEAD_GROUPS)
    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + off_b * stride_qb + offs_qh[None, :, None
        ] * stride_qh + offs_m[:, None, None] * stride_qm + offs_d[None,
        None, :]
    m_i = tl.zeros([BLOCK, QUERY_HEAD_GROUPS], dtype=tl.float32) - float('inf')
    lse_ptrs = Lse + off_b * stride_lse_b + offs_qh[None, :
        ] * seqlen_q_rounded + offs_m[:, None]
    lse_i = tl.zeros([BLOCK, QUERY_HEAD_GROUPS], dtype=tl.float32) - float(
        'inf')
    out_ptrs = Out + off_b * stride_ob + offs_qh[None, :, None
        ] * stride_oh + offs_m[:, None, None] * stride_om + offs_d[None,
        None, :]
    acc_o = tl.zeros([BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM], dtype=tl.
        float32)
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, None, :] < headdim, other=0.0
                )
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None, None] < seqlen_q) & (
            offs_d[None, None, :] < headdim), other=0.0)
    q = q.reshape(BLOCK * QUERY_HEAD_GROUPS, BLOCK_HEADDIM)
    if INCLUDE_BLOCK_CAUSAL:
        start_offs_n = start_m * BLOCK + tl.arange(0, SEL_BLOCK) - (SEL_BLOCK -
            BLOCK)
        if SLIDING:
            num_kv_blocks = 2
        else:
            num_kv_blocks = 1
        for kv_block_offset_ind in range(num_kv_blocks):
            offset = kv_block_offset_ind * -SEL_BLOCK
            offs_n = start_offs_n + offset
            k_ptrs = K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None
                ] * stride_kn + offs_d[None, :]
            v_ptrs = V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None
                ] * stride_vn + offs_d[None, :]
            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    k = tl.load(k_ptrs, mask=offs_n[:, None] >= 0, other=0.0)
                else:
                    k = tl.load(k_ptrs, mask=(offs_n[:, None] >= 0) & (
                        offs_d[None, :] < headdim), other=0.0)
            elif EVEN_HEADDIM:
                k = tl.load(k_ptrs, mask=(offs_n[:, None] >= 0) & (offs_n[:,
                    None] < seqlen_k), other=0.0)
            else:
                k = tl.load(k_ptrs, mask=(offs_n[:, None] >= 0) & (offs_n[:,
                    None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            qk = tl.zeros([BLOCK * QUERY_HEAD_GROUPS, SEL_BLOCK], dtype=tl.
                float32)
            qk += tl.dot(q, tl.trans(k))
            qk = qk.reshape(BLOCK, QUERY_HEAD_GROUPS, SEL_BLOCK)
            if BLOCK != SEL_BLOCK and not SLIDING:
                block_diagonal_mask = (offs_n[None, None, :] >= 0.0) & (
                    offs_n[None, None, :] // SEL_BLOCK == offs_m[:, None,
                    None] // SEL_BLOCK)
                qk += tl.where(block_diagonal_mask, 0, float('-inf'))
            if not EVEN_N:
                qk += tl.where(offs_n[None, :] < seqlen_k, 0, float('-inf'))
            qk = qk.reshape(BLOCK, QUERY_HEAD_GROUPS, SEL_BLOCK)
            causal_mask = offs_m[:, None, None] >= offs_n[None, None, :]
            if SLIDING:
                causal_mask &= (offs_n[None, None, :] >= 0.0) & (offs_m[:,
                    None, None] - offs_n[None, None, :] <= SEL_BLOCK)
            qk += tl.where(causal_mask, 0, float('-inf'))
            m_ij = tl.maximum(tl.max(qk, 2) * softmax_scale, m_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, :, None])
            l_ij = tl.sum(p, 2)
            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o *= acc_o_scale[:, :, None]
            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    v = tl.load(v_ptrs, mask=offs_n[:, None] >= 0, other=0.0)
                else:
                    v = tl.load(v_ptrs, mask=(offs_n[:, None] >= 0) & (
                        offs_d[None, :] < headdim), other=0.0)
            elif EVEN_HEADDIM:
                v = tl.load(v_ptrs, mask=(offs_n[:, None] >= 0) & (offs_n[:,
                    None] < seqlen_k), other=0.0)
            else:
                v = tl.load(v_ptrs, mask=(offs_n[:, None] >= 0) & (offs_n[:,
                    None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
            p = p.reshape(BLOCK * QUERY_HEAD_GROUPS, SEL_BLOCK).to(v.dtype)
            causal_o = tl.dot(p, v)
            acc_o += causal_o.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)
    kv_block_indices_ptrs = (kv_block_indices + off_b * stride_kvbl_b + 
        off_h * stride_kvbl_h + offs_m * stride_kvbl_m)
    kv_block_mask_ptrs = (kv_block_mask + off_b * stride_kvbl_b + off_h *
        stride_kvbl_h + offs_m * stride_kvbl_m)
    q = q.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)
    q = tl.expand_dims(q, 2)
    q = tl.broadcast_to(q, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM,
        BLOCK_HEADDIM))
    q = q.reshape(BLOCK, 16, BLOCK_HEADDIM)
    for off_sel_kv_block in range(NUM_SEL_KV_BLOCKS):
        block_indices = tl.load(kv_block_indices_ptrs + off_sel_kv_block,
            mask=offs_m < seqlen_q, other=0)
        block_masks = tl.load(kv_block_mask_ptrs + off_sel_kv_block, mask=
            offs_m < seqlen_q, other=False)
        for off_blocks_per_sel in range(NUM_BLOCKS_PER_SEL):
            blocks_offs_n = block_indices[:, None] * (BLOCK *
                NUM_BLOCKS_PER_SEL) + tl.arange(0, BLOCK)[None, :
                ] + off_blocks_per_sel * BLOCK
            block_k_ptrs = (K + off_b * stride_kb + off_h * stride_kh + 
                blocks_offs_n[:, :, None] * stride_kn + offs_d[None, None, :])
            block_v_ptrs = (V + off_b * stride_vb + off_h * stride_vh + 
                blocks_offs_n[:, :, None] * stride_vn + offs_d[None, None, :])
            k_block = tl.load(block_k_ptrs, mask=blocks_offs_n[:, :, None] <
                seqlen_k, other=0.0)
            block_qk = tl.zeros([BLOCK, 16, BLOCK], dtype=tl.float32)
            sel_qk = tl.zeros([BLOCK, QUERY_HEAD_GROUPS, BLOCK], dtype=tl.
                float32)
            k_block = k_block.reshape(BLOCK, BLOCK, BLOCK_HEADDIM)
            k_block = k_block.permute(0, 2, 1)
            block_qk += tl.dot(q, k_block)
            block_qk = block_qk.reshape(BLOCK, QUERY_HEAD_GROUPS,
                QUERY_EXPAND_DIM, BLOCK)
            block_qk = tl.reduce(block_qk, 2, reduce_avg)
            sel_qk += block_qk
            sel_qk += tl.where(block_masks[:, None, None], 0, float('-inf'))
            m_ij = tl.maximum(tl.max(sel_qk, 2) * softmax_scale, m_i)
            block_p = tl.exp(sel_qk * softmax_scale - m_ij[:, :, None])
            l_ij = tl.sum(block_p, 2)
            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, :, None]
            v_block = tl.load(block_v_ptrs, mask=blocks_offs_n[:, :, None] <
                seqlen_k, other=0.0)
            v_block = tl.reshape(v_block, (BLOCK, BLOCK, BLOCK_HEADDIM))
            block_p = block_p.to(v_block.dtype)
            p_expanded = block_p.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK)
            p_expanded = tl.expand_dims(p_expanded, 2)
            p_expanded = tl.broadcast_to(p_expanded, (BLOCK,
                QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK))
            p_expanded = p_expanded.reshape(BLOCK, 16, BLOCK)
            block_acc_o = tl.dot(p_expanded, v_block)
            block_acc_o = block_acc_o.reshape(BLOCK, QUERY_HEAD_GROUPS,
                QUERY_EXPAND_DIM, BLOCK_HEADDIM)
            block_acc_o = tl.reduce(block_acc_o, 2, reduce_avg)
            acc_o += block_acc_o
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)
    acc_o_scale = tl.exp(m_i - lse_i)
    acc_o *= acc_o_scale[:, :, None]
    lse_i = lse_i.reshape(BLOCK, QUERY_HEAD_GROUPS)
    tl.store(lse_ptrs, lse_i)
    acc_o = acc_o.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None, None] < seqlen_q)
    else:
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None, None] < seqlen_q) &
            (offs_d[None, None, :] < headdim))


def native_sparse_attn_forward(q, k, v, kv_block_indices, kv_block_mask,
    block_size=128, include_block_causal=True, return_sliding_window_out=False
    ):
    q, k, v, kv_block_indices = [(x if is_contiguous(x) else x.contiguous()
        ) for x in (q, k, v, kv_block_indices)]
    batch, nheads, seqlen_q, dim, device = *q.shape, q.device
    _, kv_heads, seqlen_k, _ = k.shape
    assert divisible_by(nheads, kv_heads)
    head_groups = nheads // kv_heads
    assert divisible_by(block_size, 16)
    num_blocks_per_sel = block_size // 16
    num_selected_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape
    assert k.shape == (batch, kv_heads, seqlen_k, dim)
    assert v.shape == (batch, kv_heads, seqlen_k, dim)
    assert dim <= 128, 'only support head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
    assert q.dtype in [torch.float16, torch.bfloat16
        ], 'Only support fp16 and bf16'
    assert all([t.is_cuda for t in (q, k, v)])
    softmax_scale = dim ** -0.5
    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=device,
        dtype=torch.float32)
    slide_lse = torch.empty((batch, nheads, seqlen_q_rounded), device=
        device, dtype=torch.float32)
    o = torch.empty_like(q)
    slide_o = torch.empty_like(q)
    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)
    num_warps = 4 if dim <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK']), batch *
        kv_heads, 2 if return_sliding_window_out else 1)
    forward_kernel[grid](q, k, v, kv_block_indices, kv_block_mask, o,
        slide_o, lse, slide_lse, softmax_scale, q.stride(0), q.stride(1), q
        .stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.
        stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
        kv_block_indices.stride(0), kv_block_indices.stride(1),
        kv_block_indices.stride(2), lse.stride(0), kv_heads, seqlen_q,
        seqlen_k, seqlen_q_rounded, dim, seqlen_q // 32, seqlen_k // 32,
        BLOCK_HEADDIM, BLOCK=16, SEL_BLOCK=block_size, QUERY_HEAD_GROUPS=
        head_groups, NUM_SEL_KV_BLOCKS=num_selected_fine_blocks,
        NUM_BLOCKS_PER_SEL=num_blocks_per_sel, INCLUDE_BLOCK_CAUSAL=
        include_block_causal, RETURN_SLIDING_OUT=return_sliding_window_out,
        num_warps=num_warps, num_stages=1)
    return o, slide_o, lse, slide_lse


# Forward method (kernel launch code)
def _NSA_forward(self, ctx, fq, fk, fv, block_size, selected_block_indices,
    fmask, sel_scale, include_block_causal, block_dk_dv_use_dot,
    return_sliding_window_out):
    dtype = fq.dtype
    q_heads, kv_heads = fq.shape[1], fk.shape[1]
    assert divisible_by(q_heads, kv_heads)
    head_groups = q_heads // kv_heads
    fq, fk, fv = tuple(t.half() for t in (fq, fk, fv))
    out, slide_out, lse, slide_lse = native_sparse_attn_forward(fq, fk, fv,
        selected_block_indices, fmask, block_size=block_size,
        include_block_causal=include_block_causal,
        return_sliding_window_out=return_sliding_window_out)
    ctx.save_for_backward(fq, fk, fv, selected_block_indices, fmask, out,
        slide_out, lse, slide_lse)
    return_sel_grads = exists(sel_scale)
    if return_sel_grads:
        assert (sel_scale == 1.0).all(
            ), 'for now, must be straight through as multiplier of 1.'
    ctx._saved_variables = (block_size, head_groups, return_sel_grads,
        include_block_causal, block_dk_dv_use_dot, return_sliding_window_out)
    return out.type(dtype), slide_out.type(dtype), lse, slide_lse


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics(dict(QUERY_EXPAND_DIM=lambda args: 16 // args[
    'QUERY_HEAD_GROUPS']))
@triton.jit
def backward_kernel(Q, K, V, kv_block_indices, kv_block_mask,
    kv_block_grads, DO, DQ, DK, DV, LSE, D, SLIDE_DO, SLIDE_LSE, SLIDE_D,
    softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh,
    stride_kn, stride_vb, stride_vh, stride_vn, stride_dob, stride_doh,
    stride_dom, stride_dqb, stride_dqh, stride_dqm, stride_dkb, stride_dkh,
    stride_dkn, stride_dvb, stride_dvh, stride_dvn, stride_kvbl_b,
    stride_kvbl_h, stride_kvbl_m, stride_lse_b, stride_D_b, kv_heads,
    seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K, BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK: tl.constexpr,
    SEL_BLOCK: tl.constexpr, QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr, RETURN_SEL_GRADS: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr, SLIDING: tl.constexpr,
    NUM_BLOCKS_PER_SEL: tl.constexpr, BLOCK_DV_USE_DOT: tl.constexpr,
    BLOCK_DK_USE_DOT: tl.constexpr):
    off_hb = tl.program_id(1)
    off_b = off_hb // kv_heads
    off_h = off_hb % kv_heads
    off_qh = off_h * QUERY_HEAD_GROUPS
    block_id = tl.program_id(0)
    IS_CAUSAL = False
    IS_SLIDING = False
    do = DO
    lse = LSE
    delta = D
    if INCLUDE_BLOCK_CAUSAL:
        IS_CAUSAL = block_id == 0
        block_id -= 1
    if SLIDING:
        IS_SLIDING = block_id == 0
        block_id -= 1
    if IS_SLIDING:
        do = SLIDE_DO
        lse = SLIDE_LSE
        delta = SLIDE_D
    OFF_SEL_KV_BLOCKS = block_id // NUM_BLOCKS_PER_SEL
    OFF_BLOCK_PER_SEL = block_id % NUM_BLOCKS_PER_SEL
    Q += off_b * stride_qb + off_qh * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DQ += off_b * stride_dqb + off_qh * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    do += off_b * stride_dob + off_qh * stride_doh
    kv_block_indices += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    kv_block_mask += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    kv_block_grads += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    delta += off_b * stride_D_b + off_qh * seqlen_q_rounded
    lse += off_b * stride_lse_b + off_qh * seqlen_q_rounded
    start_n = tl.program_id(2)
    if IS_CAUSAL or IS_SLIDING:
        backward_kernel_one_col_block_causal(start_n, Q, K, V, do, DQ, DK,
            DV, lse, delta, softmax_scale, stride_qm, stride_kn, stride_vn,
            stride_dom, stride_dqm, stride_dkn, stride_dvn, stride_kvbl_m,
            stride_qh, stride_doh, stride_dqh, seqlen_q, seqlen_k,
            seqlen_q_rounded, headdim, BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=
            EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, BLOCK=BLOCK,
            SEL_BLOCK=SEL_BLOCK, QUERY_HEAD_GROUPS=QUERY_HEAD_GROUPS,
            QUERY_EXPAND_DIM=QUERY_EXPAND_DIM, SLIDING=IS_SLIDING)
    else:
        backward_kernel_one_col_block_sparse(start_n, Q, K, V,
            kv_block_indices, kv_block_mask, kv_block_grads, do, DQ, DK, DV,
            lse, delta, softmax_scale, stride_qm, stride_kn, stride_vn,
            stride_dom, stride_dqm, stride_dkn, stride_dvn, stride_kvbl_m,
            stride_qh, stride_doh, stride_dqh, seqlen_q, seqlen_k,
            seqlen_q_rounded, headdim, BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=
            EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, BLOCK=BLOCK,
            QUERY_HEAD_GROUPS=QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM=
            QUERY_EXPAND_DIM, RETURN_SEL_GRADS=RETURN_SEL_GRADS,
            OFF_SEL_KV_BLOCKS=OFF_SEL_KV_BLOCKS, NUM_BLOCKS_PER_SEL=
            NUM_BLOCKS_PER_SEL, OFF_BLOCK_PER_SEL=OFF_BLOCK_PER_SEL,
            BLOCK_DV_USE_DOT=BLOCK_DV_USE_DOT, BLOCK_DK_USE_DOT=
            BLOCK_DK_USE_DOT)


@triton.jit
def backward_kernel_one_col_block_causal(start_n, Q, K, V, DO, DQ, DK, DV,
    LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_dom,
    stride_dqm, stride_dkn, stride_dvn, stride_kvbl_m, stride_qh,
    stride_doh, stride_dqh, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    BLOCK_HEADDIM: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr, BLOCK: tl.constexpr, SEL_BLOCK: tl.
    constexpr, QUERY_HEAD_GROUPS: tl.constexpr, QUERY_EXPAND_DIM: tl.
    constexpr, SLIDING: tl.constexpr):
    begin_m = start_n * BLOCK // BLOCK * BLOCK
    offs_qm = begin_m + tl.arange(0, BLOCK)
    offs_n = start_n * BLOCK + tl.arange(0, SEL_BLOCK) - (SEL_BLOCK - BLOCK)
    offs_m = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_g = tl.arange(0, QUERY_HEAD_GROUPS)
    offs_d_or_lse = seqlen_q_rounded * offs_g[:, None] + offs_m
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    q_ptrs = Q + offs_g[:, None, None] * stride_qh + offs_qm[None, :, None
        ] * stride_qm + offs_d[None, None, :]
    do_ptrs = DO + offs_g[:, None, None] * stride_doh + offs_qm[None, :, None
        ] * stride_dom + offs_d[None, None, :]
    dq_ptrs = DQ + offs_g[:, None, None] * stride_dqh + offs_qm[None, :, None
        ] * stride_dqm + offs_d[None, None, :]
    dv = tl.zeros([SEL_BLOCK, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([SEL_BLOCK, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] >= 0, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] >= 0, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=(offs_n[:, None] >= 0) & (offs_d[None,
                :] < headdim), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] >= 0) & (offs_d[None,
                :] < headdim), other=0.0)
    elif EVEN_HEADDIM:
        k = tl.load(k_ptrs, mask=(offs_n[:, None] >= 0) & (offs_n[:, None] <
            seqlen_k), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] >= 0) & (offs_n[:, None] <
            seqlen_k), other=0.0)
    else:
        k = tl.load(k_ptrs, mask=(offs_n[:, None] >= 0) & (offs_n[:, None] <
            seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] >= 0) & (offs_n[:, None] <
            seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[None, :, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[None, :, None] < seqlen_q) & (
            offs_d[None, None, :] < headdim), other=0.0)
    q = q.reshape([QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM])
    qk = tl.dot(q, tl.trans(k))
    qk = qk.reshape(QUERY_HEAD_GROUPS, BLOCK, SEL_BLOCK)
    mask = offs_m[:, None] >= offs_n[None, :]
    if BLOCK != SEL_BLOCK and not SLIDING:
        block_diagonal_mask = (offs_n[None, :] >= 0) & (offs_n[None, :] //
            SEL_BLOCK == offs_m[:, None] // SEL_BLOCK)
        mask &= block_diagonal_mask
    if not EVEN_N:
        mask &= offs_n[None, :] < seqlen_k
    if SLIDING:
        mask &= (offs_n[None, :] >= 0.0) & offs_m[:, None] - offs_n[None, :
            ] <= SEL_BLOCK
    qk = tl.where(mask, qk, float('-inf'))
    qk = qk.reshape(QUERY_HEAD_GROUPS * BLOCK, SEL_BLOCK)
    if not EVEN_M & EVEN_HEADDIM:
        tl.debug_barrier()
    lse_i = tl.load(LSE + offs_d_or_lse)
    lse_i = lse_i.reshape(QUERY_HEAD_GROUPS * BLOCK)
    p = tl.exp(qk * softmax_scale - lse_i[:, None])
    if EVEN_M & EVEN_HEADDIM:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=(offs_m[None, :, None] < seqlen_q) & (
            offs_d[None, None, :] < headdim), other=0.0)
    do = do.reshape(QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM)
    dv += tl.dot(tl.trans(p.to(do.dtype)), do)
    if not EVEN_M & EVEN_HEADDIM:
        tl.debug_barrier()
    dp = tl.dot(do, tl.trans(v))
    if not EVEN_HEADDIM:
        tl.debug_barrier()
    Di = tl.load(D + offs_d_or_lse)
    Di = Di.reshape(QUERY_HEAD_GROUPS * BLOCK)
    ds = p * (dp - Di[:, None]) * softmax_scale
    ds = ds.to(q.dtype)
    dk += tl.dot(tl.trans(ds), q)
    if not EVEN_M & EVEN_HEADDIM:
        tl.debug_barrier()
    dq = tl.zeros([QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM], dtype=tl.float32)
    dq += tl.dot(ds, k)
    dq = dq.reshape(QUERY_HEAD_GROUPS, BLOCK, BLOCK_HEADDIM)
    if EVEN_M & EVEN_HEADDIM:
        tl.atomic_add(dq_ptrs, dq, sem='relaxed')
    elif EVEN_HEADDIM:
        tl.atomic_add(dq_ptrs, dq, mask=offs_m[None, :, None] < seqlen_q,
            sem='relaxed')
    else:
        tl.atomic_add(dq_ptrs, dq, mask=(offs_m[None, :, None] < seqlen_q) &
            (offs_d[None, None, :] < headdim), sem='relaxed')
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    backward_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k,
        headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)


@triton.jit
def backward_kernel_one_col_block_sparse(start_n, Q, K, V, kv_block_indices,
    kv_block_mask, kv_block_grads, DO, DQ, DK, DV, LSE, D, softmax_scale,
    stride_qm, stride_kn, stride_vn, stride_dom, stride_dqm, stride_dkn,
    stride_dvn, stride_kvbl_m, stride_qh, stride_doh, stride_dqh, seqlen_q,
    seqlen_k, seqlen_q_rounded, headdim, BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr, QUERY_HEAD_GROUPS: tl.constexpr, QUERY_EXPAND_DIM:
    tl.constexpr, RETURN_SEL_GRADS: tl.constexpr, OFF_SEL_KV_BLOCKS: tl.
    constexpr, NUM_BLOCKS_PER_SEL: tl.constexpr, OFF_BLOCK_PER_SEL: tl.
    constexpr, BLOCK_DV_USE_DOT: tl.constexpr, BLOCK_DK_USE_DOT: tl.constexpr):
    begin_m = start_n * BLOCK // BLOCK * BLOCK
    offs_qm = begin_m + tl.arange(0, BLOCK)
    offs_n = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_m = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_g = tl.arange(0, QUERY_HEAD_GROUPS)
    offs_d_or_lse = seqlen_q_rounded * offs_g[:, None] + offs_m
    q_ptrs = Q + offs_g[None, :, None] * stride_qh + offs_qm[:, None, None
        ] * stride_qm + offs_d[None, None, :]
    do_ptrs = DO + offs_g[None, :, None] * stride_doh + offs_qm[:, None, None
        ] * stride_dom + offs_d[None, None, :]
    dq_ptrs = DQ + offs_g[None, :, None] * stride_dqh + offs_qm[:, None, None
        ] * stride_dqm + offs_d[None, None, :]
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None, None] < seqlen_q) & (
            offs_d[None, None, :] < headdim), other=0.0)
    if not EVEN_M & EVEN_HEADDIM:
        tl.debug_barrier()
    lse_i = tl.load(LSE + offs_d_or_lse)
    lse_i = tl.trans(lse_i)
    if EVEN_M & EVEN_HEADDIM:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=(offs_m[:, None, None] < seqlen_q) & (
            offs_d[None, None, :] < headdim), other=0.0)
    if not EVEN_M & EVEN_HEADDIM:
        tl.debug_barrier()
    if not EVEN_HEADDIM:
        tl.debug_barrier()
    Di = tl.load(D + offs_d_or_lse)
    Di = tl.trans(Di)
    dq = tl.zeros([BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM], dtype=tl.float32)
    kv_block_indices_ptrs = kv_block_indices + offs_m * stride_kvbl_m
    kv_block_mask_ptrs = kv_block_mask + offs_m * stride_kvbl_m
    block_indices = tl.load(kv_block_indices_ptrs + OFF_SEL_KV_BLOCKS, mask
        =offs_m < seqlen_q, other=0.0)
    block_masks = tl.load(kv_block_mask_ptrs + OFF_SEL_KV_BLOCKS, mask=
        offs_m < seqlen_q, other=0.0)
    blocks_offs_n = block_indices[:, None] * (BLOCK * NUM_BLOCKS_PER_SEL
        ) + tl.arange(0, BLOCK)[None, :] + OFF_BLOCK_PER_SEL * BLOCK
    block_k_ptrs = K + blocks_offs_n[:, :, None] * stride_kn + offs_d[None,
        None, :]
    block_v_ptrs = V + blocks_offs_n[:, :, None] * stride_vn + offs_d[None,
        None, :]
    block_dv_ptrs = DV + blocks_offs_n[:, :, None] * stride_dvn + offs_d[
        None, None, :]
    block_dk_ptrs = DK + blocks_offs_n[:, :, None] * stride_dkn + offs_d[
        None, None, :]
    block_k = tl.load(block_k_ptrs, mask=blocks_offs_n[:, :, None] <
        seqlen_k, other=0.0)
    block_v = tl.load(block_v_ptrs, mask=blocks_offs_n[:, :, None] <
        seqlen_k, other=0.0)
    q_expanded = tl.expand_dims(q, 2)
    q_expanded = tl.broadcast_to(q_expanded, (BLOCK, QUERY_HEAD_GROUPS,
        QUERY_EXPAND_DIM, BLOCK_HEADDIM))
    q_expanded = q_expanded.reshape(BLOCK, 16, BLOCK_HEADDIM)
    block_k_permuted = tl.permute(block_k, (0, 2, 1))
    block_qk = tl.dot(q_expanded, block_k_permuted)
    block_qk = block_qk.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM,
        BLOCK)
    qk = tl.reduce(block_qk, 2, reduce_avg)
    masked_qk = qk + tl.where(block_masks[:, None, None], 0, float('-inf'))
    p = tl.exp(masked_qk * softmax_scale - lse_i[:, :, None])
    do_expanded = tl.expand_dims(do, 2)
    do_expanded = tl.broadcast_to(do_expanded, (BLOCK, QUERY_HEAD_GROUPS,
        QUERY_EXPAND_DIM, BLOCK_HEADDIM))
    do_expanded = do_expanded.reshape(BLOCK, 16, BLOCK_HEADDIM)
    if BLOCK_DV_USE_DOT:
        p_expanded = p.to(do.dtype)
        p_expanded = tl.expand_dims(p_expanded, 2)
        p_expanded = tl.broadcast_to(p_expanded, (BLOCK, QUERY_HEAD_GROUPS,
            QUERY_EXPAND_DIM, BLOCK))
        p_expanded = p_expanded.reshape(BLOCK, QUERY_HEAD_GROUPS *
            QUERY_EXPAND_DIM, BLOCK)
        p_expanded = tl.permute(p_expanded, (0, 2, 1))
        block_dv = tl.dot(p_expanded, do_expanded) / QUERY_EXPAND_DIM
    else:
        block_dv = p.to(do.dtype)[:, :, :, None] * do[:, :, None, :]
        block_dv = tl.sum(block_dv, 1)
    tl.atomic_add(block_dv_ptrs, block_dv, mask=block_masks[:, None, None] &
        blocks_offs_n[:, :, None] < seqlen_k, sem='relaxed')
    block_v = tl.permute(block_v, (0, 2, 1))
    dp = tl.dot(do_expanded, block_v)
    dp = dp.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK)
    dp = tl.reduce(dp, 2, reduce_avg)
    ds = p * (dp - Di[:, :, None]) * softmax_scale
    if RETURN_SEL_GRADS:
        kv_block_grads_ptrs = kv_block_grads + offs_m * stride_kvbl_m
        sel_grads = ds * qk
        sel_grads = tl.where(block_masks[:, None, None], sel_grads, 0.0)
        sel_grads = tl.sum(sel_grads, 2)
        sel_grads = tl.sum(sel_grads, 1)
        tl.atomic_add(kv_block_grads_ptrs + OFF_SEL_KV_BLOCKS, sel_grads,
            mask=offs_m < seqlen_q, sem='relaxed')
    ds_expanded = tl.expand_dims(ds, 2)
    ds_expanded = tl.broadcast_to(ds_expanded, (BLOCK, QUERY_HEAD_GROUPS,
        QUERY_EXPAND_DIM, BLOCK))
    ds_expanded = ds_expanded.reshape(BLOCK, 16, BLOCK)
    if BLOCK_DK_USE_DOT:
        ds_permuted = tl.permute(ds_expanded, (0, 2, 1))
        block_dk = tl.dot(ds_permuted.to(q_expanded.dtype), q_expanded
            ) / QUERY_EXPAND_DIM
    else:
        block_dk = ds[:, :, :, None] * q[:, :, None, :].to(ds.dtype)
        block_dk = tl.sum(block_dk, 1)
    tl.atomic_add(block_dk_ptrs, block_dk, mask=block_masks[:, None, None] &
        blocks_offs_n[:, :, None] < seqlen_k, sem='relaxed')
    block_dq = tl.dot(ds_expanded.to(block_k.dtype), block_k)
    block_dq = block_dq.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM,
        BLOCK_HEADDIM)
    block_dq = tl.reduce(block_dq, 2, reduce_avg)
    dq += block_dq
    dq = dq.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)
    if EVEN_M & EVEN_HEADDIM:
        tl.atomic_add(dq_ptrs, dq, sem='relaxed')
    elif EVEN_HEADDIM:
        tl.atomic_add(dq_ptrs, dq, mask=offs_m[:, None, None] < seqlen_q,
            sem='relaxed')
    else:
        tl.atomic_add(dq_ptrs, dq, mask=(offs_m[:, None, None] < seqlen_q) &
            (offs_d[None, None, :] < headdim), sem='relaxed')


@triton.jit
def backward_preprocess_do_o_dot(Out, DO, Delta, stride_ob, stride_oh,
    stride_om, stride_dob, stride_doh, stride_dom, qheads, seqlen_q,
    seqlen_q_rounded, headdim, BLOCK: tl.constexpr, BLOCK_HEADDIM: tl.constexpr
    ):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // qheads
    off_h = off_hb % qheads
    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:,
        None] * stride_om + offs_d[None, :], mask=(offs_m[:, None] <
        seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + offs_m[:,
        None] * stride_dom + offs_d[None, :], mask=(offs_m[:, None] <
        seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def backward_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k,
    headdim, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.
    constexpr):
    mask_n = offs_n[:, None] >= 0
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.atomic_add(dv_ptrs, dv, mask=mask_n, sem='relaxed')
            tl.atomic_add(dk_ptrs, dk, mask=mask_n, sem='relaxed')
        else:
            tl.atomic_add(dv_ptrs, dv, mask=mask_n & (offs_d[None, :] <
                headdim), sem='relaxed')
            tl.atomic_add(dk_ptrs, dk, mask=mask_n & (offs_d[None, :] <
                headdim), sem='relaxed')
    elif EVEN_HEADDIM:
        tl.atomic_add(dv_ptrs, dv, mask=mask_n & (offs_n[:, None] <
            seqlen_k), sem='relaxed')
        tl.atomic_add(dk_ptrs, dk, mask=mask_n & (offs_n[:, None] <
            seqlen_k), sem='relaxed')
    else:
        tl.atomic_add(dv_ptrs, dv, mask=mask_n & (offs_n[:, None] <
            seqlen_k) & (offs_d[None, :] < headdim), sem='relaxed')
        tl.atomic_add(dk_ptrs, dk, mask=mask_n & (offs_n[:, None] <
            seqlen_k) & (offs_d[None, :] < headdim), sem='relaxed')


def default(val, d):
    return val if exists(val) else d


def native_sparse_attn_backward(do, q, k, v, kv_block_indices,
    kv_block_mask, kv_block_grads, o, lse, dq, dk, dv, do_slide=None,
    slide_out=None, slide_lse=None, block_size=128, include_block_causal=
    True, return_sel_grads=False, sliding=False, block_dk_dv_use_dot=None):
    device = do.device
    if not is_contiguous(do):
        do = do.contiguous()
    if not is_contiguous(do_slide):
        do_slide = do_slide.contiguous()
    batch, q_heads, seqlen_q, dim = q.shape
    _, kv_heads, seqlen_k, _ = k.shape
    assert divisible_by(q_heads, kv_heads)
    head_groups = q_heads // kv_heads
    assert divisible_by(16, head_groups)
    assert divisible_by(block_size, 16)
    num_blocks_per_sel = block_size // 16
    num_sel_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape
    assert dim <= 128
    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)
    assert lse.shape == (batch, q_heads, seqlen_q_rounded)
    assert all([is_contiguous(t) for t in (q, k, v, o, dq, dk, dv)])
    softmax_scale = dim ** -0.5
    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)
    delta = torch.empty_like(lse)
    slide_delta = torch.empty_like(slide_lse)
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK']), batch * q_heads)
    backward_preprocess_do_o_dot[grid](o, do, delta, o.stride(0), o.stride(
        1), o.stride(2), do.stride(0), do.stride(1), do.stride(2), q_heads,
        seqlen_q, seqlen_q_rounded, dim, BLOCK=block_size, BLOCK_HEADDIM=
        BLOCK_HEADDIM)
    if sliding:
        backward_preprocess_do_o_dot[grid](slide_out, do_slide, slide_delta,
            slide_out.stride(0), slide_out.stride(1), slide_out.stride(2),
            do_slide.stride(0), do_slide.stride(1), do_slide.stride(2),
            q_heads, seqlen_q, seqlen_q_rounded, dim, BLOCK=block_size,
            BLOCK_HEADDIM=BLOCK_HEADDIM)
    grid = lambda META: (int(include_block_causal) + int(sliding) + 
        num_sel_fine_blocks * num_blocks_per_sel, batch * kv_heads, triton.
        cdiv(seqlen_k, META['BLOCK']))
    backward_kernel[grid](q, k, v, kv_block_indices, kv_block_mask,
        kv_block_grads, do, dq, dk, dv, lse, delta, do_slide, slide_lse,
        slide_delta, softmax_scale, q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.
        stride(2), do.stride(0), do.stride(1), do.stride(2), dq.stride(0),
        dq.stride(1), dq.stride(2), dk.stride(0), dk.stride(1), dk.stride(2
        ), dv.stride(0), dv.stride(1), dv.stride(2), kv_block_indices.
        stride(0), kv_block_indices.stride(1), kv_block_indices.stride(2),
        lse.stride(0), delta.stride(0), kv_heads, seqlen_q, seqlen_k,
        seqlen_q_rounded, dim, seqlen_q // 32, seqlen_k // 32,
        BLOCK_HEADDIM, BLOCK=16, SEL_BLOCK=block_size, QUERY_HEAD_GROUPS=
        head_groups, EVEN_M=divisible_by(seqlen_q, block_size), EVEN_N=
        divisible_by(seqlen_k, block_size), EVEN_HEADDIM=BLOCK_HEADDIM ==
        dim, RETURN_SEL_GRADS=return_sel_grads, INCLUDE_BLOCK_CAUSAL=
        include_block_causal, SLIDING=sliding, NUM_BLOCKS_PER_SEL=
        num_blocks_per_sel, BLOCK_DV_USE_DOT=default(block_dk_dv_use_dot, 
        head_groups > 1), BLOCK_DK_USE_DOT=default(block_dk_dv_use_dot, 
        head_groups > 1))
    return delta, slide_delta


# Backward method (kernel launch code)
def _NSA_backward(self, ctx, do, do_slide, _, __):
    device = do.device
    q, k, v, sel_block_indices, mask, out, slide_out, lse, slide_lse = (ctx
        .saved_tensors)
    (block_size, head_groups, return_sel_grads, include_block_causal,
        block_dk_dv_use_dot, return_sliding_window_out) = ctx._saved_variables
    do = do.half()
    do_slide = do_slide.half()
    dq = torch.zeros(q.shape, dtype=torch.float32, device=device)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=device)
    sel_grads = torch.zeros_like(sel_block_indices).float()
    native_sparse_attn_backward(do, q, k, v, sel_block_indices, mask,
        sel_grads, out, lse, dq, dk, dv, do_slide=do_slide, slide_out=
        slide_out, slide_lse=slide_lse, block_size=block_size,
        include_block_causal=include_block_causal, return_sel_grads=
        return_sel_grads, block_dk_dv_use_dot=block_dk_dv_use_dot, sliding=
        return_sliding_window_out)
    ret_sel_grads = None
    if return_sel_grads:
        ret_sel_grads = sel_grads
    return dq, dk, dv, None, None, None, ret_sel_grads, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class NSA(Function):

    @classmethod
    def forward(self, ctx, fq, fk, fv, block_size, selected_block_indices,
        fmask, sel_scale, include_block_causal, block_dk_dv_use_dot,
        return_sliding_window_out):
        dtype = fq.dtype
        q_heads, kv_heads = fq.shape[1], fk.shape[1]
        assert divisible_by(q_heads, kv_heads)
        head_groups = q_heads // kv_heads
        fq, fk, fv = tuple(t.half() for t in (fq, fk, fv))
        out, slide_out, lse, slide_lse = native_sparse_attn_forward(fq, fk,
            fv, selected_block_indices, fmask, block_size=block_size,
            include_block_causal=include_block_causal,
            return_sliding_window_out=return_sliding_window_out)
        ctx.save_for_backward(fq, fk, fv, selected_block_indices, fmask,
            out, slide_out, lse, slide_lse)
        return_sel_grads = exists(sel_scale)
        if return_sel_grads:
            assert (sel_scale == 1.0).all(
                ), 'for now, must be straight through as multiplier of 1.'
        ctx._saved_variables = (block_size, head_groups, return_sel_grads,
            include_block_causal, block_dk_dv_use_dot,
            return_sliding_window_out)
        return out.type(dtype), slide_out.type(dtype), lse, slide_lse

    @classmethod
    def backward(self, ctx, do, do_slide, _, __):
        device = do.device
        (q, k, v, sel_block_indices, mask, out, slide_out, lse, slide_lse
            ) = ctx.saved_tensors
        (block_size, head_groups, return_sel_grads, include_block_causal,
            block_dk_dv_use_dot, return_sliding_window_out
            ) = ctx._saved_variables
        do = do.half()
        do_slide = do_slide.half()
        dq = torch.zeros(q.shape, dtype=torch.float32, device=device)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=device)
        sel_grads = torch.zeros_like(sel_block_indices).float()
        native_sparse_attn_backward(do, q, k, v, sel_block_indices, mask,
            sel_grads, out, lse, dq, dk, dv, do_slide=do_slide, slide_out=
            slide_out, slide_lse=slide_lse, block_size=block_size,
            include_block_causal=include_block_causal, return_sel_grads=
            return_sel_grads, block_dk_dv_use_dot=block_dk_dv_use_dot,
            sliding=return_sliding_window_out)
        ret_sel_grads = None
        if return_sel_grads:
            ret_sel_grads = sel_grads
        return dq, dk, dv, None, None, None, ret_sel_grads, None, None, None
