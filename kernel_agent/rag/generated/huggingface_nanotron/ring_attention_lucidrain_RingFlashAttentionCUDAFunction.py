# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/huggingface/nanotron
# Source-Files: src/nanotron/nn/ring_attention_lucidrain.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ephq_whd/nanotron-main/src/nanotron/nn/ring_attention_lucidrain.py
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
from math import ceil
from math import exp
from einops import rearrange
from einops import reduce
from einops import repeat

def default(val, d):
    return val if exists(val) else d


def exists(v):
    return v is not None


@cache()
def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


@cache()
def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

def divisible_by(num, den):
    return num % den == 0


def flash_attn_forward(q, k, v, bias=None, causal=False, o=None, m=None,
    lse=None, softmax_scale=None, causal_mask_diagonal=False,
    return_normalized_output=False, load_accumulated=True, softclamp_qk_sim
    =False, softclamp_value=50.0, head_first_dim=False, remove_padding=False):
    q, k, v = [(x if is_contiguous(x) else x.contiguous()) for x in (q, k, v)]
    if head_first_dim:
        q, k, v = tuple(rearrange(t, 'b h n d -> b n h d') for t in (q, k, v))
        if exists(o):
            o = rearrange(o, 'b h n d -> b n h d')
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, 'FlashAttention only support head dimensions up to 128'
    assert q.dtype == k.dtype == v.dtype, 'All tensors must have the same type'
    assert q.dtype in [torch.float16, torch.bfloat16
        ], 'Only support fp16 and bf16'
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = default(softmax_scale, d ** -0.5)
    has_bias = exists(bias)
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        if bias.ndim == 2:
            bias = repeat(bias, 'b j -> b h i j', h=nheads, i=seqlen_q)
        if not is_contiguous(bias):
            bias = bias.contiguous()
        assert bias.shape[-2:] == (seqlen_q, seqlen_k)
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    seqlen_q_rounded = ceil(seqlen_q / 128) * 128
    if not exists(lse):
        max_neg_value = -torch.finfo(torch.float32).max
        init_fn = partial(torch.full, fill_value=max_neg_value
            ) if load_accumulated else torch.empty
        lse = init_fn((batch, nheads, seqlen_q_rounded), device=q.device,
            dtype=torch.float32)
    if not exists(m):
        max_neg_value = -torch.finfo(torch.float32).max
        init_fn = partial(torch.full, fill_value=max_neg_value
            ) if load_accumulated else torch.empty
        m = init_fn((batch, nheads, seqlen_q_rounded), device=q.device,
            dtype=torch.float32)
    if not exists(o):
        init_fn = torch.zeros_like if load_accumulated else torch.empty_like
        o = init_fn(q)
    max(triton.next_power_of_2(d), 16)

    def grid(META):
        return triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads
    if head_first_dim:
        o = rearrange(o, 'b n h d -> b h n d')
    if remove_padding:
        m = m[..., :seqlen_q]
        lse = lse[..., :seqlen_q]
    return o, m, lse


def is_contiguous(x: Tensor):
    return x.stride(-1) == 1


# Forward method (kernel launch code)
@torch.no_grad()
def _RingFlashAttentionCUDAFunction_forward(ctx, q: Tensor, k: Tensor, v:
    Tensor, mask: (Tensor | None), causal: bool, bucket_size: int,
    ring_reduce_col: bool, striped_ring_attn: bool, max_lookback_seq_len: (
    int | None), softclamp_qk_sim: bool, softclamp_value: float, ring_pg: (
    dist.ProcessGroup | None)):
    assert k.shape[-2:] == v.shape[-2:]
    q_heads, kv_heads = q.shape[-2], k.shape[-2]
    assert divisible_by(q_heads, kv_heads)
    q_head_groups = q_heads // kv_heads
    assert all(t.is_cuda for t in (q, k, v)), 'inputs must be all on cuda'
    dtype = q.dtype
    softmax_scale = q.shape[-1] ** -0.5
    if q.dtype == torch.float32:
        q = q.half()
    if k.dtype == torch.float32:
        k = k.half()
    if v.dtype == torch.float32:
        v = v.half()
    ring_size = ring_pg.size() if ring_pg else get_world_size()
    cross_attn = q.shape[-3] != k.shape[-3]
    ring_reduce_col &= not cross_attn
    striped_ring_attn &= not cross_attn
    assert k.shape[-1] == v.shape[-1
        ], 'for simplicity when doing ring passing, assume dim_values is equal to dim_queries_keys, majority of transformer do this, not a big issue'
    per_machine_seq_size = k.shape[-3]
    max_ring_passes = None
    num_lookback_buckets = float('inf')
    if exists(max_lookback_seq_len):
        assert causal
        assert not (ring_reduce_col and not divisible_by(
            per_machine_seq_size, bucket_size))
        max_ring_passes = ceil(max_lookback_seq_len / per_machine_seq_size)
        num_lookback_buckets = max_lookback_seq_len // bucket_size
    if causal:
        mask = None
    bucket_size = min(per_machine_seq_size, bucket_size)
    per_machine_seq_size // bucket_size
    orig_k, orig_v, orig_mask, q_seq_len, device = k, v, mask, q.shape[1
        ], q.device
    ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass
    kv = torch.stack((k, v))
    o = None
    m = None
    lse = None
    receive_kv = None
    receive_mask = None
    can_fuse_final_output_normalization = (not causal or causal and
        striped_ring_attn)
    for (ring_rank, (is_first, is_last)), ((kv, mask), (receive_kv,
        receive_mask)) in ring_pass_fn(kv, mask, receive_buffers=(
        receive_kv, receive_mask), max_iters=max_ring_passes, ring_size=
        ring_size, ring_pg=ring_pg):
        k, v = kv
        k, v = (repeat(t, '... h d -> ... (g h) d', g=q_head_groups) for t in
            (k, v))
        bias = None
        if exists(mask):
            bias = torch.where(mask, 0.0, float('-inf'))
        block_causal = False
        causal_mask_diagonal = False
        if causal:
            if striped_ring_attn:
                block_causal = True
                causal_mask_diagonal = get_rank(
                    ) if ring_pg is None else dist.get_rank(ring_pg
                    ) < ring_rank
            else:
                block_causal = get_rank(
                    ) if ring_pg is None else dist.get_rank(ring_pg
                    ) == ring_rank
                if (get_rank() if ring_pg is None else dist.get_rank(ring_pg)
                    ) < ring_rank:
                    continue
        o, m, lse = flash_attn_forward(q, k, v, causal=block_causal, o=o, m
            =m, lse=lse, bias=bias, softmax_scale=softmax_scale,
            causal_mask_diagonal=causal_mask_diagonal,
            return_normalized_output=can_fuse_final_output_normalization and
            is_last, load_accumulated=not is_first, softclamp_qk_sim=
            softclamp_qk_sim, softclamp_value=softclamp_value)
    if not can_fuse_final_output_normalization:
        m = m[..., :q_seq_len]
        o_scale = torch.exp(m - lse[..., :q_seq_len])
        o.mul_(rearrange(o_scale, 'b h n -> b n h 1'))
    ctx.args = (causal, softmax_scale, orig_mask, bucket_size,
        ring_reduce_col, max_ring_passes, num_lookback_buckets,
        striped_ring_attn, q_head_groups, softclamp_qk_sim, softclamp_value,
        dtype, ring_pg)
    ctx.save_for_backward(q, orig_k, orig_v, o, lse)
    o = o.type(dtype)
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,
    'SEQUENCE_PARALLEL': False}, num_warps=8, num_stages=1, pre_hook=
    init_to_zero('DQ')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,
    'SEQUENCE_PARALLEL': True}, num_warps=8, num_stages=1, pre_hook=
    init_to_zero('DQ'))], key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K',
    'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM'])
@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args[
    'BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args[
    'BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args[
    'BLOCK_HEADDIM']})
@triton.jit
def _bwd_kernel(Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm,
    stride_dob, stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm,
    stride_dkb, stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn,
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr, CAUSAL_MASK_DIAGONAL: tl.constexpr,
    SOFTCLAMP_QK_SIM: tl.constexpr, SOFTCLAMP_VALUE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr, SEQUENCE_PARALLEL: tl.constexpr, EVEN_M:
    tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr, BLOCK_M:
    tl.constexpr, BLOCK_N: tl.constexpr):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != 'none':
        Bias += off_b * stride_bb + off_h * stride_bh
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK,
                DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn,
                stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn,
                seqlen_q, seqlen_k, headdim, ATOMIC_ADD=False, BIAS_TYPE=
                BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, CAUSAL_MASK_DIAGONAL=
                CAUSAL_MASK_DIAGONAL, SOFTCLAMP_QK_SIM=SOFTCLAMP_QK_SIM,
                SOFTCLAMP_VALUE=SOFTCLAMP_VALUE, BLOCK_HEADDIM=
                BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=
                EVEN_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV,
            LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn,
            stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn,
            seqlen_q, seqlen_k, headdim, ATOMIC_ADD=True, BIAS_TYPE=
            BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, CAUSAL_MASK_DIAGONAL=
            CAUSAL_MASK_DIAGONAL, SOFTCLAMP_QK_SIM=SOFTCLAMP_QK_SIM,
            SOFTCLAMP_VALUE=SOFTCLAMP_VALUE, BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


@triton.jit
def _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE,
    D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm,
    stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k,
    headdim, ATOMIC_ADD: tl.constexpr, BIAS_TYPE: tl.constexpr, IS_CAUSAL:
    tl.constexpr, CAUSAL_MASK_DIAGONAL: tl.constexpr, SOFTCLAMP_QK_SIM: tl.
    constexpr, SOFTCLAMP_VALUE: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    begin_m = 0 if not IS_CAUSAL else start_n * BLOCK_N // BLOCK_M * BLOCK_M
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == 'vector':
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == 'matrix':
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k,
            headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)
        return
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    elif EVEN_HEADDIM:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    else:
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[
            None, :] < headdim), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[
            None, :] < headdim), other=0.0)
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        elif EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0
                )
        else:
            q = tl.load(q_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (
                offs_d[None, :] < headdim), other=0.0)
        qk = tl.dot(q, tl.trans(k))
        if SOFTCLAMP_QK_SIM:
            effective_softclamp_value = SOFTCLAMP_VALUE / softmax_scale
            qk /= effective_softclamp_value
            qk = libdevice.tanh(qk)
            dtanh = 1.0 - qk * qk
            qk *= effective_softclamp_value
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float('-inf'))
        if IS_CAUSAL:
            if CAUSAL_MASK_DIAGONAL:
                qk = tl.where(offs_m_curr[:, None] > offs_n[None, :], qk,
                    float('-inf'))
            else:
                qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk,
                    float('-inf'))
        if BIAS_TYPE != 'none':
            tl.debug_barrier()
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0
                        ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == 'matrix':
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=(offs_m_curr[:, None] <
                        seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0
                        ).to(tl.float32)
            qk = qk * softmax_scale + bias
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == 'none':
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) &
                (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        Di = tl.load(D + offs_m_curr)
        ds = p * (dp - Di[:, None]) * softmax_scale
        if SOFTCLAMP_QK_SIM:
            ds *= dtanh
        ds = ds.to(q.dtype)
        dk += tl.dot(tl.trans(ds), q)
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy='evict_last')
            elif EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, mask=offs_m_curr[:, None] < seqlen_q,
                    other=0.0, eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q,
                    eviction_policy='evict_last')
            else:
                dq = tl.load(dq_ptrs, mask=(offs_m_curr[:, None] < seqlen_q
                    ) & (offs_d[None, :] < headdim), other=0.0,
                    eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q
                    ) & (offs_d[None, :] < headdim), eviction_policy=
                    'evict_last')
        else:
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq, sem='relaxed')
            elif EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] <
                    seqlen_q, sem='relaxed')
            else:
                tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] <
                    seqlen_q) & (offs_d[None, :] < headdim), sem='relaxed')
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == 'matrix':
            b_ptrs += BLOCK_M * stride_bm
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k,
        headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)


@triton.jit
def _bwd_preprocess_do_o_dot(Out, DO, Delta, stride_ob, stride_oh,
    stride_om, stride_dob, stride_doh, stride_dom, nheads, seqlen_q,
    seqlen_q_rounded, headdim, BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.
    constexpr):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
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
def _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k,
    headdim, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.
    constexpr):
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
    else:
        tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[
            None, :] < headdim))
        tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[
            None, :] < headdim))


def circular_index_left(pos, ring_size, num=1):
    return (pos - num + ring_size) % ring_size


def circular_index_right(pos, ring_size, num=1):
    return (pos + num) % ring_size


def circular_rank_left(rank=None, ring_size=None, num=1, pg=None):
    rank = default(rank, get_rank() if pg is None else dist.get_rank(pg))
    ring_size = default(ring_size, get_world_size() if pg is None else dist
        .get_world_size(pg))
    ring_set_num = rank // ring_size
    offset = ring_set_num * ring_size
    return circular_index_left(rank, ring_size, num) + offset


def circular_rank_right(rank=None, ring_size=None, num=1, pg=None):
    rank = default(rank, get_rank() if pg is None else dist.get_rank(pg))
    ring_size = default(ring_size, get_world_size() if pg is None else dist
        .get_world_size(pg))
    ring_set_num = rank // ring_size
    offset = ring_set_num * ring_size
    return circular_index_right(rank, ring_size, num) + offset


def flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, delta=None, bias=
    None, causal=False, causal_mask_diagonal=False, softmax_scale=None,
    softclamp_qk_sim=False, softclamp_value=50.0):
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert d <= 128
    seqlen_q_rounded = ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    if not exists(delta):
        delta = torch.empty_like(lse)

        def grid(META):
            return triton.cdiv(seqlen_q, META['BLOCK_M']), batch * nheads
        _bwd_preprocess_do_o_dot[grid](o, do, delta, o.stride(0), o.stride(
            2), o.stride(1), do.stride(0), do.stride(2), do.stride(1),
            nheads, seqlen_q, seqlen_q_rounded, d, BLOCK_M=128,
            BLOCK_HEADDIM=BLOCK_HEADDIM)
    has_bias = bias is not None
    bias_type = 'none'
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = 'vector'
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = 'matrix'
        else:
            raise RuntimeError(
                'Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)'
                )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)
        ) if has_bias else (0, 0, 0)

    def grid(META):
        return triton.cdiv(seqlen_k, META['BLOCK_N']) if META[
            'SEQUENCE_PARALLEL'] else 1, batch * nheads
    _bwd_kernel[grid](q, k, v, bias, do, dq_accum, dk, dv, lse, delta,
        softmax_scale, q.stride(0), q.stride(2), q.stride(1), k.stride(0),
        k.stride(2), k.stride(1), v.stride(0), v.stride(2), v.stride(1), *
        bias_strides, do.stride(0), do.stride(2), do.stride(1), dq_accum.
        stride(0), dq_accum.stride(2), dq_accum.stride(1), dk.stride(0), dk
        .stride(2), dk.stride(1), dv.stride(0), dv.stride(2), dv.stride(1),
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d, seqlen_q // 32, 
        seqlen_k // 32, bias_type, causal, causal_mask_diagonal,
        softclamp_qk_sim, softclamp_value, BLOCK_HEADDIM)
    dq.copy_(dq_accum)
    return delta


def ring_pass(num_ring_passes: int, x: Tensor, receive_buffer: (Tensor |
    None)=None, ring_size: (int | None)=None, ring_pg: (dist.ProcessGroup |
    None)=None):
    ring_size = default(ring_size, get_world_size())
    x = x.contiguous()
    if not exists(receive_buffer):
        receive_buffer = torch.zeros_like(x)
    else:
        receive_buffer = receive_buffer.contiguous()
    send_and_receive_(x, receive_buffer, circular_rank_right(ring_size=
        ring_size, pg=ring_pg), circular_rank_left(ring_size=ring_size, pg=
        ring_pg), ring_pg=ring_pg)
    return receive_buffer, x


def send_and_receive_(x, receive_buffer, send_to_rank, receive_from_rank,
    ring_pg=None):
    send_op = dist.P2POp(dist.isend, x, send_to_rank, ring_pg)
    recv_op = dist.P2POp(dist.irecv, receive_buffer, receive_from_rank, ring_pg
        )
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    if ring_pg is not None:
        dist.barrier(ring_pg)
    else:
        dist.barrier()


# Backward method (kernel launch code)
@torch.no_grad()
def _RingFlashAttentionCUDAFunction_backward(ctx, do):
    (causal, softmax_scale, mask, bucket_size, ring_reduce_col,
        max_ring_passes, num_lookback_buckets, striped_ring_attn,
        q_head_groups, softclamp_qk_sim, softclamp_value, dtype, ring_pg
        ) = ctx.args
    q, k, v, o, lse = ctx.saved_tensors
    ring_size = ring_pg.size() if ring_pg else get_world_size()
    do = do.type(o.dtype)
    device = q.device
    if causal:
        mask = None
    q.shape[-3]
    per_machine_seq_size = k.shape[-3]
    per_machine_seq_size // bucket_size
    ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass
    device = q.device
    dq = torch.zeros(q.shape, device=device, dtype=torch.float32)
    dk = torch.zeros_like(k, device=device)
    dv = torch.zeros_like(v, device=device)
    assert k.dtype == v.dtype
    kv_and_dkv = torch.stack((k, v, dk, dv))
    receive_kv_and_dkv = None
    receive_mask = None
    delta = None
    for (ring_rank, _), ((kv_and_dkv, mask), (receive_kv_and_dkv, receive_mask)
        ) in ring_pass_fn(kv_and_dkv, mask, receive_buffers=(
        receive_kv_and_dkv, receive_mask), max_iters=max_ring_passes,
        ring_size=ring_size, ring_pg=ring_pg):
        k, v, dk, dv = kv_and_dkv
        k, v = (repeat(t, '... h d -> ... (g h) d', g=q_head_groups) for t in
            (k, v))
        bias = None
        if exists(mask):
            bias = torch.where(mask, 0.0, float('-inf'))
        if causal and striped_ring_attn:
            need_accum = True
            block_causal = True
            causal_mask_diagonal = (get_rank() if ring_pg is None else dist
                .get_rank(ring_pg)) < ring_rank
        elif causal:
            need_accum = (get_rank() if ring_pg is None else dist.get_rank(
                ring_pg)) >= ring_rank
            block_causal = (get_rank() if ring_pg is None else dist.
                get_rank(ring_pg)) == ring_rank
            causal_mask_diagonal = False
        else:
            need_accum = True
            block_causal = False
            causal_mask_diagonal = False
        if need_accum:
            ring_dq = torch.empty(q.shape, device=device, dtype=torch.float32)
            ring_dk = torch.empty_like(k)
            ring_dv = torch.empty_like(v)
            with torch.inference_mode():
                delta = flash_attn_backward(do, q, k, v, o, lse, ring_dq,
                    ring_dk, ring_dv, delta=delta, bias=bias, causal=
                    block_causal, causal_mask_diagonal=causal_mask_diagonal,
                    softmax_scale=softmax_scale, softclamp_qk_sim=
                    softclamp_qk_sim, softclamp_value=softclamp_value)
            ring_dk = reduce(ring_dk, '... (g h) d -> ... h d', g=
                q_head_groups, reduction='sum')
            ring_dv = reduce(ring_dv, '... (g h) d -> ... h d', g=
                q_head_groups, reduction='sum')
            dq.add_(ring_dq)
            dk.add_(ring_dk)
            dv.add_(ring_dv)
        if not ring_reduce_col:
            continue
        dkv = kv_and_dkv[2:]
        max_ring_passes = default(max_ring_passes, ring_size)
        dkv = ring_pass(ring_size - max_ring_passes + 1, dkv, ring_pg=ring_pg)
        dk, dv = dkv
    dq, dk, dv = (t.to(dtype) for t in (dq, dk, dv))
    return dq, dk, dv, None, None, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RingFlashAttentionCUDAFunction(Function):

    @staticmethod
    @torch.no_grad()
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, mask: (Tensor | None),
        causal: bool, bucket_size: int, ring_reduce_col: bool,
        striped_ring_attn: bool, max_lookback_seq_len: (int | None),
        softclamp_qk_sim: bool, softclamp_value: float, ring_pg: (dist.
        ProcessGroup | None)):
        assert k.shape[-2:] == v.shape[-2:]
        q_heads, kv_heads = q.shape[-2], k.shape[-2]
        assert divisible_by(q_heads, kv_heads)
        q_head_groups = q_heads // kv_heads
        assert all(t.is_cuda for t in (q, k, v)), 'inputs must be all on cuda'
        dtype = q.dtype
        softmax_scale = q.shape[-1] ** -0.5
        if q.dtype == torch.float32:
            q = q.half()
        if k.dtype == torch.float32:
            k = k.half()
        if v.dtype == torch.float32:
            v = v.half()
        ring_size = ring_pg.size() if ring_pg else get_world_size()
        cross_attn = q.shape[-3] != k.shape[-3]
        ring_reduce_col &= not cross_attn
        striped_ring_attn &= not cross_attn
        assert k.shape[-1] == v.shape[-1
            ], 'for simplicity when doing ring passing, assume dim_values is equal to dim_queries_keys, majority of transformer do this, not a big issue'
        per_machine_seq_size = k.shape[-3]
        max_ring_passes = None
        num_lookback_buckets = float('inf')
        if exists(max_lookback_seq_len):
            assert causal
            assert not (ring_reduce_col and not divisible_by(
                per_machine_seq_size, bucket_size))
            max_ring_passes = ceil(max_lookback_seq_len / per_machine_seq_size)
            num_lookback_buckets = max_lookback_seq_len // bucket_size
        if causal:
            mask = None
        bucket_size = min(per_machine_seq_size, bucket_size)
        per_machine_seq_size // bucket_size
        orig_k, orig_v, orig_mask, q_seq_len, device = k, v, mask, q.shape[1
            ], q.device
        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass
        kv = torch.stack((k, v))
        o = None
        m = None
        lse = None
        receive_kv = None
        receive_mask = None
        can_fuse_final_output_normalization = (not causal or causal and
            striped_ring_attn)
        for (ring_rank, (is_first, is_last)), ((kv, mask), (receive_kv,
            receive_mask)) in ring_pass_fn(kv, mask, receive_buffers=(
            receive_kv, receive_mask), max_iters=max_ring_passes, ring_size
            =ring_size, ring_pg=ring_pg):
            k, v = kv
            k, v = (repeat(t, '... h d -> ... (g h) d', g=q_head_groups) for
                t in (k, v))
            bias = None
            if exists(mask):
                bias = torch.where(mask, 0.0, float('-inf'))
            block_causal = False
            causal_mask_diagonal = False
            if causal:
                if striped_ring_attn:
                    block_causal = True
                    causal_mask_diagonal = get_rank(
                        ) if ring_pg is None else dist.get_rank(ring_pg
                        ) < ring_rank
                else:
                    block_causal = get_rank(
                        ) if ring_pg is None else dist.get_rank(ring_pg
                        ) == ring_rank
                    if (get_rank() if ring_pg is None else dist.get_rank(
                        ring_pg)) < ring_rank:
                        continue
            o, m, lse = flash_attn_forward(q, k, v, causal=block_causal, o=
                o, m=m, lse=lse, bias=bias, softmax_scale=softmax_scale,
                causal_mask_diagonal=causal_mask_diagonal,
                return_normalized_output=
                can_fuse_final_output_normalization and is_last,
                load_accumulated=not is_first, softclamp_qk_sim=
                softclamp_qk_sim, softclamp_value=softclamp_value)
        if not can_fuse_final_output_normalization:
            m = m[..., :q_seq_len]
            o_scale = torch.exp(m - lse[..., :q_seq_len])
            o.mul_(rearrange(o_scale, 'b h n -> b n h 1'))
        ctx.args = (causal, softmax_scale, orig_mask, bucket_size,
            ring_reduce_col, max_ring_passes, num_lookback_buckets,
            striped_ring_attn, q_head_groups, softclamp_qk_sim,
            softclamp_value, dtype, ring_pg)
        ctx.save_for_backward(q, orig_k, orig_v, o, lse)
        o = o.type(dtype)
        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        (causal, softmax_scale, mask, bucket_size, ring_reduce_col,
            max_ring_passes, num_lookback_buckets, striped_ring_attn,
            q_head_groups, softclamp_qk_sim, softclamp_value, dtype, ring_pg
            ) = ctx.args
        q, k, v, o, lse = ctx.saved_tensors
        ring_size = ring_pg.size() if ring_pg else get_world_size()
        do = do.type(o.dtype)
        device = q.device
        if causal:
            mask = None
        q.shape[-3]
        per_machine_seq_size = k.shape[-3]
        per_machine_seq_size // bucket_size
        ring_pass_fn = all_ring_pass if ring_reduce_col else null_ring_pass
        device = q.device
        dq = torch.zeros(q.shape, device=device, dtype=torch.float32)
        dk = torch.zeros_like(k, device=device)
        dv = torch.zeros_like(v, device=device)
        assert k.dtype == v.dtype
        kv_and_dkv = torch.stack((k, v, dk, dv))
        receive_kv_and_dkv = None
        receive_mask = None
        delta = None
        for (ring_rank, _), ((kv_and_dkv, mask), (receive_kv_and_dkv,
            receive_mask)) in ring_pass_fn(kv_and_dkv, mask,
            receive_buffers=(receive_kv_and_dkv, receive_mask), max_iters=
            max_ring_passes, ring_size=ring_size, ring_pg=ring_pg):
            k, v, dk, dv = kv_and_dkv
            k, v = (repeat(t, '... h d -> ... (g h) d', g=q_head_groups) for
                t in (k, v))
            bias = None
            if exists(mask):
                bias = torch.where(mask, 0.0, float('-inf'))
            if causal and striped_ring_attn:
                need_accum = True
                block_causal = True
                causal_mask_diagonal = (get_rank() if ring_pg is None else
                    dist.get_rank(ring_pg)) < ring_rank
            elif causal:
                need_accum = (get_rank() if ring_pg is None else dist.
                    get_rank(ring_pg)) >= ring_rank
                block_causal = (get_rank() if ring_pg is None else dist.
                    get_rank(ring_pg)) == ring_rank
                causal_mask_diagonal = False
            else:
                need_accum = True
                block_causal = False
                causal_mask_diagonal = False
            if need_accum:
                ring_dq = torch.empty(q.shape, device=device, dtype=torch.
                    float32)
                ring_dk = torch.empty_like(k)
                ring_dv = torch.empty_like(v)
                with torch.inference_mode():
                    delta = flash_attn_backward(do, q, k, v, o, lse,
                        ring_dq, ring_dk, ring_dv, delta=delta, bias=bias,
                        causal=block_causal, causal_mask_diagonal=
                        causal_mask_diagonal, softmax_scale=softmax_scale,
                        softclamp_qk_sim=softclamp_qk_sim, softclamp_value=
                        softclamp_value)
                ring_dk = reduce(ring_dk, '... (g h) d -> ... h d', g=
                    q_head_groups, reduction='sum')
                ring_dv = reduce(ring_dv, '... (g h) d -> ... h d', g=
                    q_head_groups, reduction='sum')
                dq.add_(ring_dq)
                dk.add_(ring_dk)
                dv.add_(ring_dv)
            if not ring_reduce_col:
                continue
            dkv = kv_and_dkv[2:]
            max_ring_passes = default(max_ring_passes, ring_size)
            dkv = ring_pass(ring_size - max_ring_passes + 1, dkv, ring_pg=
                ring_pg)
            dk, dv = dkv
        dq, dk, dv = (t.to(dtype) for t in (dq, dk, dv))
        return dq, dk, dv, None, None, None, None, None, None, None, None, None
