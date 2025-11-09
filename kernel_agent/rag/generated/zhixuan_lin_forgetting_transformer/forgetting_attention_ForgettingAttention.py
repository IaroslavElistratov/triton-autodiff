# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/zhixuan-lin/forgetting-transformer
# Source-Files: src/forgetting_transformer/ops/forgetting_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_o52u21n_/forgetting-transformer-main/src/forgetting_transformer/ops/forgetting_attention.py
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
from math import log
from math import sqrt
import time

@triton.jit
def _find_start_index_kernel(LOG_LAMBDA, START_INDEX, THRESHOLD,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    stride_threshold_z, stride_threshold_h, Z, H, M, N, P_SEQ, BLOCK_M: tl.
    constexpr, BLOCK_N: tl.constexpr, DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr):
    off_h = tl.program_id(0)
    off_z = tl.program_id(1)
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h
    THRESHOLD += off_z * stride_threshold_z + off_h * stride_threshold_h
    start_index = 0
    log_lambda_out_ptr = LOG_LAMBDA + P_SEQ * stride_log_lambda_n
    start_index_ptr = START_INDEX
    threshold = tl.load(THRESHOLD)
    for start_m in range(0, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        log_lambda_out = tl.load(log_lambda_out_ptr)
        offset_n = start_index + BLOCK_N - 1
        if not DIVISIBLE_N:
            offset_n = tl.minimum(N - 1, offset_n)
        log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
        decay = log_lambda_out - log_lambda_in
        while decay < threshold:
            start_index += BLOCK_N
            offset_n = start_index + BLOCK_N - 1
            if not DIVISIBLE_N:
                offset_n = tl.minimum(N - 1, offset_n)
            log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n
                )
            decay = log_lambda_out - log_lambda_in
        tl.store(start_index_ptr, start_index.to(START_INDEX.dtype.element_ty))
        start_index_ptr += stride_start_index_mb
        log_lambda_out_ptr += stride_log_lambda_n * BLOCK_M


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _fwd_kernel(Q, K, V, LOG_LAMBDA, SEQ_START, START_INDEX, sm_scale, L, O,
    stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
    stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    stride_oz, stride_oh, stride_om, stride_ok, Z, H, M, N, P_SEQ,
    num_groups, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N:
    tl.constexpr, IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr,
    HAS_SEQ_START: tl.constexpr, IS_ADAPTIVE: tl.constexpr, DIVISIBLE_M: tl
    .constexpr, DIVISIBLE_N: tl.constexpr):
    input_dtype = Q.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    loge2: tl.constexpr = 0.6931471805599453
    qk_scale = sm_scale * log2e
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier='.cg')
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier='.cg')
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier='.cg')
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m,
            cache_modifier='.cg')
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N
    offs_n_init = offs_n_base
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
    else:
        lo = 0
        seq_start = 0
    if IS_ADAPTIVE:
        START_INDEX += (off_z * stride_start_index_z + off_h *
            stride_start_index_h + start_m * stride_start_index_mb)
        start_index = tl.load(START_INDEX)
        lo = tl.maximum(start_index, lo)
    lo = lo // BLOCK_N * BLOCK_N
    offs_n_init += lo
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] *
        stride_kn)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] *
        stride_vk)
    log_lambda_in_ptrs = LOG_LAMBDA + offs_n_init * stride_log_lambda_n
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier='.cg')
            v = tl.load(v_ptrs, cache_modifier='.cg')
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier='.cg')
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier='.cg')
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier='.cg')
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n,
                cache_modifier='.cg')
        if BLOCK_M > 1:
            s = tl.dot(q, k, input_precision='ieee') * qk_scale
        else:
            s = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True
                ) * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s += decay_bias * log2e
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float('-inf'))
        if IS_CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
        if HAS_SEQ_START:
            s = tl.where(offs_n[None, :] >= seq_start, s, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(s - m_i_new[:, None])
        p_sum = tl.sum(p, 1)
        acc *= alpha[:, None]
        if BLOCK_M > 1:
            acc += tl.dot(p.to(input_dtype), v, input_precision='ieee')
        else:
            acc += tl.sum(p.T * v, axis=0, keep_dims=True)
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n
    if IS_CAUSAL and (LARGER_M or HAS_SEQ_START):
        is_empty_line = offs_m + P_SEQ < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float('-inf'), m_i * loge2 + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * loge2 + tl.log(l_i)
    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier='.cg')
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier='.cg')
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier='.cg')
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None],
            cache_modifier='.cg')


def get_fwd_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
    elif torch.cuda.get_device_capability() == (9, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 2, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        elif D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        raise ValueError(
            f'Unsupported device capability {torch.cuda.get_device_capability()}. Please open an issue.'
            )
    return BLOCK_M, BLOCK_N, num_stages, num_warps


def maybe_contiguous(x):
    return x.contiguous() if x.stride(-1) != 1 else x


# Forward method (kernel launch code)
def _ForgettingAttention_forward(ctx, q, k, v, log_fgate, seq_start, causal,
    sm_scale, adaptive_threshold, return_log_normalizer, return_start_index,
    record_time_key, record_attention_time, record_find_index_time):
    if record_attention_time:
        ForgettingAttention.events[record_time_key]['fwd_start_event'].record()
    assert causal, 'Only causal attention is supported'
    Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Dq == Dk == Dv, 'feature size of q, k, v should be equal'
    assert Dk in {16, 32, 64, 128
        }, 'We only support head dims in {16, 32, 64, 128}'
    B, H, M, D = q.shape
    N = k.shape[2]
    assert log_fgate.shape == (B, H, N)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    if adaptive_threshold is not None:
        if isinstance(adaptive_threshold, str):
            assert adaptive_threshold == 'auto', f'adaptive_threshold must be either the string "auto", a float, or a Tensor, but got {adaptive_threshold}.'
            max_q_norm = torch.linalg.vector_norm(q, dim=-1).max(dim=-1).values
            max_k_norm = torch.linalg.vector_norm(k, dim=-1).max(dim=-1).values
            assert max_q_norm.size() == max_k_norm.size() == (B, H)
            logit_upper_bound = max_q_norm * max_k_norm * sm_scale
            tolerance = -10
            adaptive_threshold = -(2 * logit_upper_bound + math.log(N)
                ) + tolerance
        adaptive_threshold = torch.as_tensor(adaptive_threshold, dtype=
            torch.float, device=q.device)
        try:
            adaptive_threshold = torch.broadcast_to(adaptive_threshold, (B, H))
        except RuntimeError:
            raise RuntimeError(
                f'adaptive_threshold must be either the string "auto" or broadcastable to (batch_size, num_heads) = ({B}, {H}), but got {adaptive_threshold.size()}.'
                )
        assert adaptive_threshold.size() == (B, H)
    if seq_start is not None:
        has_seq_start = True
        assert seq_start.shape == (B,)
    else:
        has_seq_start = False
        seq_start = torch.zeros((B,), device=q.device, dtype=torch.long)
    log_fgate = log_fgate.float()
    if has_seq_start:
        log_fgate = log_fgate.clone()
        mask_index = torch.arange(N, device=q.device)[None, None, :
            ] < seq_start[:, None, None]
        mask_index = torch.broadcast_to(mask_index, log_fgate.size())
        log_fgate[mask_index] = 0.0
    log_lambda = torch.cumsum(log_fgate, dim=-1, dtype=log_fgate.dtype).float()
    Hk, Hv = k.shape[1], v.shape[1]
    assert Hk == Hv, 'num of heads in k and v should be equal'
    assert H == Hk, 'groupped query attention has not been tested. You can uncomment this if you know what you are doing.'
    assert H % Hk == 0, 'number of heads in q must be a multiple of that in k & v'
    num_groups = H // Hk
    P_SEQ = N - M
    larger_m = M > N
    assert not larger_m, 'The key/value tensors must be longer than the query tensor'
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
    device = torch.cuda.device_of(q)
    with torch.cuda.device(device):
        if M > 1:
            config = get_fwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config
        else:
            BLOCK_N, num_stages, num_warps = min(128, max(16, triton.
                next_power_of_2(N))), 1, 4
            BLOCK_M = 1
        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        start_index = torch.empty((B, H, triton.cdiv(M, BLOCK_M)), dtype=
            torch.long, device=q.device)
        if adaptive_threshold is not None:
            grid = H, B
            if record_find_index_time:
                ForgettingAttention.events[record_time_key][
                    'fwd_find_index_start_event'].record()
            _find_start_index_kernel[grid](log_lambda, start_index,
                adaptive_threshold, log_lambda.stride(0), log_lambda.stride
                (1), log_lambda.stride(2), start_index.stride(0),
                start_index.stride(1), start_index.stride(2),
                adaptive_threshold.stride(0), adaptive_threshold.stride(1),
                B, H, M, N, P_SEQ, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n, num_warps=1)
            if record_find_index_time:
                ForgettingAttention.events[record_time_key][
                    'fwd_find_index_end_event'].record()
                torch.cuda.synchronize()
                elapsed = ForgettingAttention.events[record_time_key][
                    'fwd_find_index_start_event'].elapsed_time(
                    ForgettingAttention.events[record_time_key][
                    'fwd_find_index_end_event'])
                ForgettingAttention.info[record_time_key]['fwd_find_index_time'
                    ] += elapsed
                ForgettingAttention.info[record_time_key][
                    'fwd_find_index_count'] += 1
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), H, B)
        o = torch.empty_like(q)
        L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        _fwd_kernel[grid](q, k, v, log_lambda, seq_start, start_index,
            sm_scale, L, o, q.stride(0), q.stride(1), q.stride(2), q.stride
            (3), k.stride(0), k.stride(1), k.stride(2), k.stride(3), v.
            stride(0), v.stride(1), v.stride(2), v.stride(3), log_lambda.
            stride(0), log_lambda.stride(1), log_lambda.stride(2),
            start_index.stride(0), start_index.stride(1), start_index.
            stride(2), o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            B, H, M, N, P_SEQ, num_groups, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=D, IS_CAUSAL=causal, LARGER_M=larger_m,
            HAS_SEQ_START=has_seq_start, IS_ADAPTIVE=adaptive_threshold is not
            None, DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
            num_warps=num_warps, num_stages=num_stages)
    ctx.save_for_backward(q, k, v, o, L, log_lambda, seq_start,
        adaptive_threshold)
    ctx.sm_scale = sm_scale
    ctx.causal = causal
    ctx.has_seq_start = has_seq_start
    ctx.record_time_key = record_time_key
    ctx.record_attention_time = record_attention_time
    ctx.record_find_index_time = record_find_index_time
    has_extra_return = return_log_normalizer or return_start_index
    if record_attention_time:
        ForgettingAttention.events[record_time_key]['fwd_end_event'].record()
        torch.cuda.synchronize()
        elapsed = ForgettingAttention.events[record_time_key]['fwd_start_event'
            ].elapsed_time(ForgettingAttention.events[record_time_key][
            'fwd_end_event'])
        ForgettingAttention.info[record_time_key]['fwd_time'] += elapsed
        ForgettingAttention.info[record_time_key]['fwd_count'] += 1
    if has_extra_return:
        outs = (o, L if return_log_normalizer else None, start_index if
            return_start_index else None)
        return outs
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _bwd_kv_kernel(Q, K, V, LOG_LAMBDA, SEQ_START, END_INDEX, sm_scale, DO,
    DK, DV, DLOG_LAMBDA, L, D, stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh,
    stride_vn, stride_vk, stride_log_lambda_z, stride_log_lambda_h,
    stride_log_lambda_n, stride_start_index_z, stride_start_index_h,
    stride_start_index_nb, stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk, stride_dvz, stride_dvh,
    stride_dvn, stride_dvk, stride_dlog_lambda_z, stride_dlog_lambda_h,
    stride_dlog_lambda_n, Z, H, M, N, P_SEQ, num_groups, BLOCK_M: tl.
    constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL:
    tl.constexpr, DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
    HAS_SEQ_START: tl.constexpr, IS_ADAPTIVE: tl.constexpr):
    input_dtype = Q.dtype.element_ty
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    DO += off_z * stride_doz + off_h * stride_doh
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh
    DLOG_LAMBDA += off_z * stride_dlog_lambda_z + off_h * stride_dlog_lambda_h
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = lo // BLOCK_M * BLOCK_M
    else:
        lo = 0
    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] *
        stride_qk)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m_init
        ) * stride_log_lambda_n
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    log_lambda_in_ptrs = LOG_LAMBDA + offs_n * stride_log_lambda_n
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] *
        stride_dok)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk
        )
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
        )
    dlog_lambda_in_ptrs = DLOG_LAMBDA + offs_n * stride_dlog_lambda_n
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
        log_lambda_in = tl.load(log_lambda_in_ptrs)
    else:
        mask_n = offs_n < N
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])
        log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n)
    hi = M
    if IS_ADAPTIVE:
        END_INDEX += (off_z * stride_start_index_z + off_h *
            stride_start_index_h + start_n * stride_start_index_nb)
        hi = tl.minimum(tl.load(END_INDEX), M)
    else:
        hi = M
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        hi = tl.where(start_n * BLOCK_N + BLOCK_N >= seq_start - 1, hi, lo)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dlog_lambda_in = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = P_SEQ + offs_m[None, :] >= offs_n[:, None]
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
            log_lambda_out = tl.load(log_lambda_out_ptrs)
        else:
            mask_m = offs_m < M
            valid_mask = mask_m[None, :]
            q = tl.load(q_ptrs, mask=mask_m[:, None])
            log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m)
        sT = tl.dot(k, tl.trans(q), input_precision='ieee') * qk_scale
        decay_bias = log_lambda_out[None, :] - log_lambda_in[:, None]
        sT += decay_bias * log2e
        if DIVISIBLE_M:
            l = tl.load(L + offs_m)
        else:
            l = tl.load(L + offs_m, mask=mask_m)
        pT = tl.math.exp2(sT - l[None, :] * log2e)
        if not DIVISIBLE_M:
            pT = tl.where(valid_mask, pT, 0.0)
        if CAUSAL:
            pT = tl.where(causal_mask, pT, 0.0)
        if DIVISIBLE_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=mask_m[:, None])
        dv += tl.dot(pT.to(input_dtype), do, input_precision='ieee')
        if DIVISIBLE_M:
            delta = tl.load(D + offs_m)
        else:
            delta = tl.load(D + offs_m, mask=mask_m)
        dpT = tl.dot(v, tl.trans(do), input_precision='ieee')
        dsT = pT * (dpT - delta[None, :])
        if not DIVISIBLE_M:
            dsT = tl.where(valid_mask, dsT, 0.0)
        if CAUSAL:
            dsT = tl.where(causal_mask, dsT, 0.0)
        dk += tl.dot(dsT.to(input_dtype), q, input_precision='ieee')
        dlog_lambda_in += -tl.sum(dsT, axis=1)
        q_ptrs += BLOCK_M * stride_qm
        log_lambda_out_ptrs += BLOCK_M * stride_log_lambda_n
        do_ptrs += BLOCK_M * stride_dom
    dk *= sm_scale
    if HAS_SEQ_START:
        seq_mask = offs_n >= seq_start
        dk = tl.where(seq_mask[:, None], dk, 0.0)
        dv = tl.where(seq_mask[:, None], dv, 0.0)
        dlog_lambda_in = tl.where(seq_mask, dlog_lambda_in, 0.0)
    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk.to(input_dtype))
        tl.store(dv_ptrs, dv.to(input_dtype))
        tl.store(dlog_lambda_in_ptrs, dlog_lambda_in.to(tl.float32))
    else:
        tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])
        tl.store(dlog_lambda_in_ptrs, dlog_lambda_in.to(tl.float32), mask=
            mask_n)


@triton.jit
def _bwd_preprocess(Out, DO, Delta, stride_oz, stride_oh, stride_om,
    stride_ok, stride_doz, stride_doh, stride_dom, stride_dok, stride_dz,
    stride_dh, stride_dm, M, BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    DIVISIBLE_M: tl.constexpr):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok
    if DIVISIBLE_M:
        o = tl.load(o_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
    else:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_q_kernel(Q, K, V, LOG_LAMBDA, SEQ_START, START_INDEX, sm_scale, DO,
    DQ, DLOG_LAMBDA, L, D, stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh,
    stride_vn, stride_vk, stride_log_lambda_z, stride_log_lambda_h,
    stride_log_lambda_n, stride_start_index_z, stride_start_index_h,
    stride_start_index_mb, stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk, stride_dlog_lambda_z,
    stride_dlog_lambda_h, stride_dlog_lambda_n, Z, H, M, N, P_SEQ,
    num_groups, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N:
    tl.constexpr, CAUSAL: tl.constexpr, LARGER_M: tl.constexpr,
    HAS_SEQ_START: tl.constexpr, IS_ADAPTIVE: tl.constexpr, DIVISIBLE_M: tl
    .constexpr, DIVISIBLE_N: tl.constexpr):
    input_dtype = Q.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M
    DQ += off_z * stride_dqz + off_h * stride_dqh
    DLOG_LAMBDA += off_z * stride_dlog_lambda_z + off_h * stride_dlog_lambda_h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
        )
    dlog_lambda_out_ptrs = DLOG_LAMBDA + (P_SEQ + offs_m
        ) * stride_dlog_lambda_n
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
        )
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        l = tl.load(l_ptrs)
        log_lambda_out = tl.load(log_lambda_out_ptrs)
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        l = tl.load(l_ptrs, mask=mask_m)
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dlog_lambda_out = tl.zeros([BLOCK_M], dtype=tl.float32)
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
    else:
        lo = 0
        seq_start = 0
    if IS_ADAPTIVE:
        START_INDEX += (off_z * stride_start_index_z + off_h *
            stride_start_index_h + start_m * stride_start_index_mb)
        start_index = tl.load(START_INDEX)
        lo = tl.maximum(start_index, lo)
    lo = lo // BLOCK_N * BLOCK_N
    offs_n_init += lo
    k_ptrs = K + (offs_n_init[:, None] * stride_kn + offs_k[None, :] *
        stride_kk)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] *
        stride_vk)
    log_lambda_in_ptrs = LOG_LAMBDA + offs_n_init * stride_log_lambda_n
    for start_n in range(lo, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k = tl.load(k_ptrs)
            log_lambda_in = tl.load(log_lambda_in_ptrs)
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k = tl.load(k_ptrs, mask=mask_n[:, None])
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n)
        if not DIVISIBLE_N:
            valid_mask = mask_n[None, :]
        if CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        s = tl.dot(q, tl.trans(k), input_precision='ieee') * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s += decay_bias * log2e
        p = tl.math.exp2(s - l[:, None] * log2e)
        dp = tl.dot(do.to(input_dtype), tl.trans(v), input_precision='ieee')
        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_N:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        if HAS_SEQ_START:
            ds = tl.where(offs_n[None, :] >= seq_start, ds, 0.0)
        dq += tl.dot(ds.to(input_dtype), k, input_precision='ieee')
        dlog_lambda_out += tl.sum(ds, axis=1)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n
    dq *= sm_scale
    if DIVISIBLE_M:
        tmp = tl.load(dlog_lambda_out_ptrs)
    else:
        tmp = tl.load(dlog_lambda_out_ptrs, mask=mask_m)
    dlog_lambda_out += tmp
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq.to(input_dtype))
        tl.store(dlog_lambda_out_ptrs, dlog_lambda_out)
    else:
        tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])
        tl.store(dlog_lambda_out_ptrs, dlog_lambda_out, mask=mask_m)


@triton.jit
def _find_end_index_kernel(LOG_LAMBDA, END_INDEX, THRESHOLD,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_end_index_z, stride_end_index_h, stride_end_index_nb,
    stride_threshold_z, stride_threshold_h, Z, H, M, N, P_SEQ, BLOCK_M: tl.
    constexpr, BLOCK_N: tl.constexpr, DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr):
    off_h = tl.program_id(0)
    off_z = tl.program_id(1)
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    END_INDEX += off_z * stride_end_index_z + off_h * stride_end_index_h
    THRESHOLD += off_z * stride_threshold_z + off_h * stride_threshold_h
    end_index = 0
    log_lambda_in_ptr = LOG_LAMBDA
    end_index_ptr = END_INDEX
    threshold = tl.load(THRESHOLD)
    for start_n in range(0, N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_M)
        offset_n = start_n + BLOCK_N - 1
        if not DIVISIBLE_N:
            offset_n = tl.minimum(N - 1, offset_n)
        log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
        log_lambda_out = tl.load(LOG_LAMBDA + tl.minimum(end_index, M - 1) *
            stride_log_lambda_n)
        decay = log_lambda_out - log_lambda_in
        while decay >= threshold and end_index < M:
            end_index = tl.minimum(end_index + BLOCK_M, M)
            log_lambda_out = tl.load(LOG_LAMBDA + tl.minimum(end_index, M -
                1) * stride_log_lambda_n)
            decay = log_lambda_out - log_lambda_in
        tl.store(end_index_ptr, end_index.to(END_INDEX.dtype.element_ty))
        end_index_ptr += stride_end_index_nb
        log_lambda_in_ptr += stride_log_lambda_n * BLOCK_N


def get_bwd_kv_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 4, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 128, 4, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 128, 4, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 128, 2, 8
    elif torch.cuda.get_device_capability() == (9, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    else:
        raise ValueError(
            f'Unsupported device capability {torch.cuda.get_device_capability()}. Please open an issue.'
            )
    return BLOCK_M, BLOCK_N, num_stages, num_warps


def get_bwd_q_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 4, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 3, 4
    elif torch.cuda.get_device_capability() == (9, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 4, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 2, 8
    else:
        raise ValueError(
            f'Unsupported device capability {torch.cuda.get_device_capability()}. Please open an issue.'
            )
    return BLOCK_M, BLOCK_N, num_stages, num_warps


# Backward method (kernel launch code)
def _ForgettingAttention_backward(ctx, do, *ignored):
    if ctx.record_attention_time:
        ForgettingAttention.events[ctx.record_time_key]['bwd_start_event'
            ].record()
    q, k, v, o, L, log_lambda, seq_start, adaptive_threshold = (ctx.
        saved_tensors)
    sm_scale = ctx.sm_scale
    causal = ctx.causal
    has_seq_start = ctx.has_seq_start
    B, H, M, D = q.shape
    N = k.shape[2]
    Hk = k.shape[1]
    num_groups = H // Hk
    P_SEQ = N - M
    larger_m = M > N
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    device = torch.cuda.device_of(q)
    with torch.cuda.device(device):
        BLOCK_M = 64
        divisible_m = M % BLOCK_M == 0
        delta = torch.empty_like(L)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), H, B)
        _bwd_preprocess[grid](o, do, delta, o.stride(0), o.stride(1), o.
            stride(2), o.stride(3), do.stride(0), do.stride(1), do.stride(2
            ), do.stride(3), delta.stride(0), delta.stride(1), delta.stride
            (2), M, BLOCK_M=BLOCK_M, D_HEAD=D, DIVISIBLE_M=divisible_m)
        BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_kv_config(B, H, M,
            N, D, causal)
        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        end_index = torch.empty((B, H, triton.cdiv(N, BLOCK_N)), dtype=
            torch.long, device=q.device)
        if adaptive_threshold is not None:
            grid = H, B
            if ctx.record_find_index_time:
                ForgettingAttention.events[ctx.record_time_key][
                    'bwd_find_index_kv_start_event'].record()
            _find_end_index_kernel[grid](log_lambda, end_index,
                adaptive_threshold, log_lambda.stride(0), log_lambda.stride
                (1), log_lambda.stride(2), end_index.stride(0), end_index.
                stride(1), end_index.stride(2), adaptive_threshold.stride(0
                ), adaptive_threshold.stride(1), B, H, M, N, P_SEQ, BLOCK_M
                =BLOCK_M, BLOCK_N=BLOCK_N, DIVISIBLE_M=divisible_m,
                DIVISIBLE_N=divisible_n, num_warps=1)
            if ctx.record_find_index_time:
                ForgettingAttention.events[ctx.record_time_key][
                    'bwd_find_index_kv_end_event'].record()
                torch.cuda.synchronize()
                elapsed = ForgettingAttention.events[ctx.record_time_key][
                    'bwd_find_index_kv_start_event'].elapsed_time(
                    ForgettingAttention.events[ctx.record_time_key][
                    'bwd_find_index_kv_end_event'])
                ForgettingAttention.info[ctx.record_time_key][
                    'bwd_find_index_kv_time'] += elapsed
                ForgettingAttention.info[ctx.record_time_key][
                    'bwd_find_index_kv_count'] += 1
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dlog_lambda = torch.empty((B, H, N), dtype=log_lambda.dtype, device
            =q.device)
        grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), H, B)
        _bwd_kv_kernel[grid](q, k, v, log_lambda, seq_start, end_index,
            sm_scale, do, dk, dv, dlog_lambda, L, delta, q.stride(0), q.
            stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(1),
            k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2),
            v.stride(3), log_lambda.stride(0), log_lambda.stride(1),
            log_lambda.stride(2), end_index.stride(0), end_index.stride(1),
            end_index.stride(2), do.stride(0), do.stride(1), do.stride(2),
            do.stride(3), dk.stride(0), dk.stride(1), dk.stride(2), dk.
            stride(3), dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(
            3), dlog_lambda.stride(0), dlog_lambda.stride(1), dlog_lambda.
            stride(2), B, H, M, N, P_SEQ, num_groups, BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=D, BLOCK_N=BLOCK_N, CAUSAL=causal, DIVISIBLE_M=
            divisible_m, DIVISIBLE_N=divisible_n, HAS_SEQ_START=
            has_seq_start, IS_ADAPTIVE=adaptive_threshold is not None,
            num_stages=num_stages, num_warps=num_warps)
        BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_q_config(B, H, M,
            N, D, causal)
        divisible_m = M % BLOCK_M == 0
        divisible_n = N % BLOCK_N == 0
        dq = torch.empty_like(q)
        start_index = torch.empty((B, H, triton.cdiv(M, BLOCK_M)), dtype=
            torch.long, device=q.device)
        if adaptive_threshold is not None:
            grid = H, B
            if ctx.record_find_index_time:
                ForgettingAttention.events[ctx.record_time_key][
                    'bwd_find_index_q_start_event'].record()
            _find_start_index_kernel[grid](log_lambda, start_index,
                adaptive_threshold, log_lambda.stride(0), log_lambda.stride
                (1), log_lambda.stride(2), start_index.stride(0),
                start_index.stride(1), start_index.stride(2),
                adaptive_threshold.stride(0), adaptive_threshold.stride(1),
                B, H, M, N, P_SEQ, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n, num_warps=1)
            if ctx.record_find_index_time:
                ForgettingAttention.events[ctx.record_time_key][
                    'bwd_find_index_q_end_event'].record()
                torch.cuda.synchronize()
                elapsed = ForgettingAttention.events[ctx.record_time_key][
                    'bwd_find_index_q_start_event'].elapsed_time(
                    ForgettingAttention.events[ctx.record_time_key][
                    'bwd_find_index_q_end_event'])
                ForgettingAttention.info[ctx.record_time_key][
                    'bwd_find_index_q_time'] += elapsed
                ForgettingAttention.info[ctx.record_time_key][
                    'bwd_find_index_q_count'] += 1
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), H, B)
        _bwd_q_kernel[grid](q, k, v, log_lambda, seq_start, start_index,
            sm_scale, do, dq, dlog_lambda, L, delta, q.stride(0), q.stride(
            1), q.stride(2), q.stride(3), k.stride(0), k.stride(1), k.
            stride(2), k.stride(3), v.stride(0), v.stride(1), v.stride(2),
            v.stride(3), log_lambda.stride(0), log_lambda.stride(1),
            log_lambda.stride(2), start_index.stride(0), start_index.stride
            (1), start_index.stride(2), do.stride(0), do.stride(1), do.
            stride(2), do.stride(3), dq.stride(0), dq.stride(1), dq.stride(
            2), dq.stride(3), dlog_lambda.stride(0), dlog_lambda.stride(1),
            dlog_lambda.stride(2), B, H, M, N, P_SEQ, num_groups, BLOCK_M=
            BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N, CAUSAL=causal,
            LARGER_M=larger_m, HAS_SEQ_START=has_seq_start, IS_ADAPTIVE=
            adaptive_threshold is not None, DIVISIBLE_M=divisible_m,
            DIVISIBLE_N=divisible_n, num_stages=num_stages, num_warps=num_warps
            )
        if num_groups > 1:
            dk = dk.reshape((B, Hk, num_groups, N, D)).sum(2)
            dv = dv.reshape((B, Hk, num_groups, N, D)).sum(2)
    dcumsum = torch.cumsum(dlog_lambda, dim=-1, dtype=log_lambda.dtype)
    dlog_fgate = dlog_lambda + dcumsum[..., -1:] - dcumsum
    dlog_fgate = dlog_fgate.float()
    if ctx.record_attention_time:
        ForgettingAttention.events[ctx.record_time_key]['bwd_end_event'
            ].record()
        torch.cuda.synchronize()
        elapsed = ForgettingAttention.events[ctx.record_time_key][
            'bwd_start_event'].elapsed_time(ForgettingAttention.events[ctx.
            record_time_key]['bwd_end_event'])
        ForgettingAttention.info[ctx.record_time_key]['bwd_time'] += elapsed
        ForgettingAttention.info[ctx.record_time_key]['bwd_count'] += 1
    return (dq, dk, dv, dlog_fgate, None, None, None, None, None, None,
        None, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ForgettingAttention(torch.autograd.Function):
    events = defaultdict(lambda : {'fwd_start_event': torch.cuda.Event(
        enable_timing=True), 'fwd_end_event': torch.cuda.Event(
        enable_timing=True), 'bwd_start_event': torch.cuda.Event(
        enable_timing=True), 'bwd_end_event': torch.cuda.Event(
        enable_timing=True), 'fwd_find_index_start_event': torch.cuda.Event
        (enable_timing=True), 'fwd_find_index_end_event': torch.cuda.Event(
        enable_timing=True), 'bwd_find_index_kv_start_event': torch.cuda.
        Event(enable_timing=True), 'bwd_find_index_kv_end_event': torch.
        cuda.Event(enable_timing=True), 'bwd_find_index_q_start_event':
        torch.cuda.Event(enable_timing=True), 'bwd_find_index_q_end_event':
        torch.cuda.Event(enable_timing=True)})
    info = defaultdict(lambda : {'fwd_time': 0.0, 'fwd_count': 0,
        'bwd_time': 0.0, 'bwd_count': 0, 'fwd_find_index_time': 0.0,
        'fwd_find_index_count': 0, 'bwd_find_index_kv_time': 0.0,
        'bwd_find_index_kv_count': 0, 'bwd_find_index_q_time': 0.0,
        'bwd_find_index_q_count': 0})

    @staticmethod
    def forward(ctx, q, k, v, log_fgate, seq_start, causal, sm_scale,
        adaptive_threshold, return_log_normalizer, return_start_index,
        record_time_key, record_attention_time, record_find_index_time):
        if record_attention_time:
            ForgettingAttention.events[record_time_key]['fwd_start_event'
                ].record()
        assert causal, 'Only causal attention is supported'
        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, 'feature size of q, k, v should be equal'
        assert Dk in {16, 32, 64, 128
            }, 'We only support head dims in {16, 32, 64, 128}'
        B, H, M, D = q.shape
        N = k.shape[2]
        assert log_fgate.shape == (B, H, N)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)
        if adaptive_threshold is not None:
            if isinstance(adaptive_threshold, str):
                assert adaptive_threshold == 'auto', f'adaptive_threshold must be either the string "auto", a float, or a Tensor, but got {adaptive_threshold}.'
                max_q_norm = torch.linalg.vector_norm(q, dim=-1).max(dim=-1
                    ).values
                max_k_norm = torch.linalg.vector_norm(k, dim=-1).max(dim=-1
                    ).values
                assert max_q_norm.size() == max_k_norm.size() == (B, H)
                logit_upper_bound = max_q_norm * max_k_norm * sm_scale
                tolerance = -10
                adaptive_threshold = -(2 * logit_upper_bound + math.log(N)
                    ) + tolerance
            adaptive_threshold = torch.as_tensor(adaptive_threshold, dtype=
                torch.float, device=q.device)
            try:
                adaptive_threshold = torch.broadcast_to(adaptive_threshold,
                    (B, H))
            except RuntimeError:
                raise RuntimeError(
                    f'adaptive_threshold must be either the string "auto" or broadcastable to (batch_size, num_heads) = ({B}, {H}), but got {adaptive_threshold.size()}.'
                    )
            assert adaptive_threshold.size() == (B, H)
        if seq_start is not None:
            has_seq_start = True
            assert seq_start.shape == (B,)
        else:
            has_seq_start = False
            seq_start = torch.zeros((B,), device=q.device, dtype=torch.long)
        log_fgate = log_fgate.float()
        if has_seq_start:
            log_fgate = log_fgate.clone()
            mask_index = torch.arange(N, device=q.device)[None, None, :
                ] < seq_start[:, None, None]
            mask_index = torch.broadcast_to(mask_index, log_fgate.size())
            log_fgate[mask_index] = 0.0
        log_lambda = torch.cumsum(log_fgate, dim=-1, dtype=log_fgate.dtype
            ).float()
        Hk, Hv = k.shape[1], v.shape[1]
        assert Hk == Hv, 'num of heads in k and v should be equal'
        assert H == Hk, 'groupped query attention has not been tested. You can uncomment this if you know what you are doing.'
        assert H % Hk == 0, 'number of heads in q must be a multiple of that in k & v'
        num_groups = H // Hk
        P_SEQ = N - M
        larger_m = M > N
        assert not larger_m, 'The key/value tensors must be longer than the query tensor'
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            if M > 1:
                config = get_fwd_config(B, H, M, N, D, causal)
                BLOCK_M, BLOCK_N, num_stages, num_warps = config
            else:
                BLOCK_N, num_stages, num_warps = min(128, max(16, triton.
                    next_power_of_2(N))), 1, 4
                BLOCK_M = 1
            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0
            start_index = torch.empty((B, H, triton.cdiv(M, BLOCK_M)),
                dtype=torch.long, device=q.device)
            if adaptive_threshold is not None:
                grid = H, B
                if record_find_index_time:
                    ForgettingAttention.events[record_time_key][
                        'fwd_find_index_start_event'].record()
                _find_start_index_kernel[grid](log_lambda, start_index,
                    adaptive_threshold, log_lambda.stride(0), log_lambda.
                    stride(1), log_lambda.stride(2), start_index.stride(0),
                    start_index.stride(1), start_index.stride(2),
                    adaptive_threshold.stride(0), adaptive_threshold.stride
                    (1), B, H, M, N, P_SEQ, BLOCK_M=BLOCK_M, BLOCK_N=
                    BLOCK_N, DIVISIBLE_M=divisible_m, DIVISIBLE_N=
                    divisible_n, num_warps=1)
                if record_find_index_time:
                    ForgettingAttention.events[record_time_key][
                        'fwd_find_index_end_event'].record()
                    torch.cuda.synchronize()
                    elapsed = ForgettingAttention.events[record_time_key][
                        'fwd_find_index_start_event'].elapsed_time(
                        ForgettingAttention.events[record_time_key][
                        'fwd_find_index_end_event'])
                    ForgettingAttention.info[record_time_key][
                        'fwd_find_index_time'] += elapsed
                    ForgettingAttention.info[record_time_key][
                        'fwd_find_index_count'] += 1
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), H, B)
            o = torch.empty_like(q)
            L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
            _fwd_kernel[grid](q, k, v, log_lambda, seq_start, start_index,
                sm_scale, L, o, q.stride(0), q.stride(1), q.stride(2), q.
                stride(3), k.stride(0), k.stride(1), k.stride(2), k.stride(
                3), v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.
                stride(2), start_index.stride(0), start_index.stride(1),
                start_index.stride(2), o.stride(0), o.stride(1), o.stride(2
                ), o.stride(3), B, H, M, N, P_SEQ, num_groups, BLOCK_M=
                BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D, IS_CAUSAL=causal,
                LARGER_M=larger_m, HAS_SEQ_START=has_seq_start, IS_ADAPTIVE
                =adaptive_threshold is not None, DIVISIBLE_M=divisible_m,
                DIVISIBLE_N=divisible_n, num_warps=num_warps, num_stages=
                num_stages)
        ctx.save_for_backward(q, k, v, o, L, log_lambda, seq_start,
            adaptive_threshold)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.has_seq_start = has_seq_start
        ctx.record_time_key = record_time_key
        ctx.record_attention_time = record_attention_time
        ctx.record_find_index_time = record_find_index_time
        has_extra_return = return_log_normalizer or return_start_index
        if record_attention_time:
            ForgettingAttention.events[record_time_key]['fwd_end_event'
                ].record()
            torch.cuda.synchronize()
            elapsed = ForgettingAttention.events[record_time_key][
                'fwd_start_event'].elapsed_time(ForgettingAttention.events[
                record_time_key]['fwd_end_event'])
            ForgettingAttention.info[record_time_key]['fwd_time'] += elapsed
            ForgettingAttention.info[record_time_key]['fwd_count'] += 1
        if has_extra_return:
            outs = (o, L if return_log_normalizer else None, start_index if
                return_start_index else None)
            return outs
        return o

    @staticmethod
    def backward(ctx, do, *ignored):
        if ctx.record_attention_time:
            ForgettingAttention.events[ctx.record_time_key]['bwd_start_event'
                ].record()
        q, k, v, o, L, log_lambda, seq_start, adaptive_threshold = (ctx.
            saved_tensors)
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        has_seq_start = ctx.has_seq_start
        B, H, M, D = q.shape
        N = k.shape[2]
        Hk = k.shape[1]
        num_groups = H // Hk
        P_SEQ = N - M
        larger_m = M > N
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            BLOCK_M = 64
            divisible_m = M % BLOCK_M == 0
            delta = torch.empty_like(L)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), H, B)
            _bwd_preprocess[grid](o, do, delta, o.stride(0), o.stride(1), o
                .stride(2), o.stride(3), do.stride(0), do.stride(1), do.
                stride(2), do.stride(3), delta.stride(0), delta.stride(1),
                delta.stride(2), M, BLOCK_M=BLOCK_M, D_HEAD=D, DIVISIBLE_M=
                divisible_m)
            BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_kv_config(B,
                H, M, N, D, causal)
            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0
            end_index = torch.empty((B, H, triton.cdiv(N, BLOCK_N)), dtype=
                torch.long, device=q.device)
            if adaptive_threshold is not None:
                grid = H, B
                if ctx.record_find_index_time:
                    ForgettingAttention.events[ctx.record_time_key][
                        'bwd_find_index_kv_start_event'].record()
                _find_end_index_kernel[grid](log_lambda, end_index,
                    adaptive_threshold, log_lambda.stride(0), log_lambda.
                    stride(1), log_lambda.stride(2), end_index.stride(0),
                    end_index.stride(1), end_index.stride(2),
                    adaptive_threshold.stride(0), adaptive_threshold.stride
                    (1), B, H, M, N, P_SEQ, BLOCK_M=BLOCK_M, BLOCK_N=
                    BLOCK_N, DIVISIBLE_M=divisible_m, DIVISIBLE_N=
                    divisible_n, num_warps=1)
                if ctx.record_find_index_time:
                    ForgettingAttention.events[ctx.record_time_key][
                        'bwd_find_index_kv_end_event'].record()
                    torch.cuda.synchronize()
                    elapsed = ForgettingAttention.events[ctx.record_time_key][
                        'bwd_find_index_kv_start_event'].elapsed_time(
                        ForgettingAttention.events[ctx.record_time_key][
                        'bwd_find_index_kv_end_event'])
                    ForgettingAttention.info[ctx.record_time_key][
                        'bwd_find_index_kv_time'] += elapsed
                    ForgettingAttention.info[ctx.record_time_key][
                        'bwd_find_index_kv_count'] += 1
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dlog_lambda = torch.empty((B, H, N), dtype=log_lambda.dtype,
                device=q.device)
            grid = lambda META: (triton.cdiv(N, META['BLOCK_N']), H, B)
            _bwd_kv_kernel[grid](q, k, v, log_lambda, seq_start, end_index,
                sm_scale, do, dk, dv, dlog_lambda, L, delta, q.stride(0), q
                .stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride
                (1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.
                stride(2), v.stride(3), log_lambda.stride(0), log_lambda.
                stride(1), log_lambda.stride(2), end_index.stride(0),
                end_index.stride(1), end_index.stride(2), do.stride(0), do.
                stride(1), do.stride(2), do.stride(3), dk.stride(0), dk.
                stride(1), dk.stride(2), dk.stride(3), dv.stride(0), dv.
                stride(1), dv.stride(2), dv.stride(3), dlog_lambda.stride(0
                ), dlog_lambda.stride(1), dlog_lambda.stride(2), B, H, M, N,
                P_SEQ, num_groups, BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N
                =BLOCK_N, CAUSAL=causal, DIVISIBLE_M=divisible_m,
                DIVISIBLE_N=divisible_n, HAS_SEQ_START=has_seq_start,
                IS_ADAPTIVE=adaptive_threshold is not None, num_stages=
                num_stages, num_warps=num_warps)
            BLOCK_M, BLOCK_N, num_stages, num_warps = get_bwd_q_config(B, H,
                M, N, D, causal)
            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0
            dq = torch.empty_like(q)
            start_index = torch.empty((B, H, triton.cdiv(M, BLOCK_M)),
                dtype=torch.long, device=q.device)
            if adaptive_threshold is not None:
                grid = H, B
                if ctx.record_find_index_time:
                    ForgettingAttention.events[ctx.record_time_key][
                        'bwd_find_index_q_start_event'].record()
                _find_start_index_kernel[grid](log_lambda, start_index,
                    adaptive_threshold, log_lambda.stride(0), log_lambda.
                    stride(1), log_lambda.stride(2), start_index.stride(0),
                    start_index.stride(1), start_index.stride(2),
                    adaptive_threshold.stride(0), adaptive_threshold.stride
                    (1), B, H, M, N, P_SEQ, BLOCK_M=BLOCK_M, BLOCK_N=
                    BLOCK_N, DIVISIBLE_M=divisible_m, DIVISIBLE_N=
                    divisible_n, num_warps=1)
                if ctx.record_find_index_time:
                    ForgettingAttention.events[ctx.record_time_key][
                        'bwd_find_index_q_end_event'].record()
                    torch.cuda.synchronize()
                    elapsed = ForgettingAttention.events[ctx.record_time_key][
                        'bwd_find_index_q_start_event'].elapsed_time(
                        ForgettingAttention.events[ctx.record_time_key][
                        'bwd_find_index_q_end_event'])
                    ForgettingAttention.info[ctx.record_time_key][
                        'bwd_find_index_q_time'] += elapsed
                    ForgettingAttention.info[ctx.record_time_key][
                        'bwd_find_index_q_count'] += 1
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), H, B)
            _bwd_q_kernel[grid](q, k, v, log_lambda, seq_start, start_index,
                sm_scale, do, dq, dlog_lambda, L, delta, q.stride(0), q.
                stride(1), q.stride(2), q.stride(3), k.stride(0), k.stride(
                1), k.stride(2), k.stride(3), v.stride(0), v.stride(1), v.
                stride(2), v.stride(3), log_lambda.stride(0), log_lambda.
                stride(1), log_lambda.stride(2), start_index.stride(0),
                start_index.stride(1), start_index.stride(2), do.stride(0),
                do.stride(1), do.stride(2), do.stride(3), dq.stride(0), dq.
                stride(1), dq.stride(2), dq.stride(3), dlog_lambda.stride(0
                ), dlog_lambda.stride(1), dlog_lambda.stride(2), B, H, M, N,
                P_SEQ, num_groups, BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N
                =BLOCK_N, CAUSAL=causal, LARGER_M=larger_m, HAS_SEQ_START=
                has_seq_start, IS_ADAPTIVE=adaptive_threshold is not None,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_stages=num_stages, num_warps=num_warps)
            if num_groups > 1:
                dk = dk.reshape((B, Hk, num_groups, N, D)).sum(2)
                dv = dv.reshape((B, Hk, num_groups, N, D)).sum(2)
        dcumsum = torch.cumsum(dlog_lambda, dim=-1, dtype=log_lambda.dtype)
        dlog_fgate = dlog_lambda + dcumsum[..., -1:] - dcumsum
        dlog_fgate = dlog_fgate.float()
        if ctx.record_attention_time:
            ForgettingAttention.events[ctx.record_time_key]['bwd_end_event'
                ].record()
            torch.cuda.synchronize()
            elapsed = ForgettingAttention.events[ctx.record_time_key][
                'bwd_start_event'].elapsed_time(ForgettingAttention.events[
                ctx.record_time_key]['bwd_end_event'])
            ForgettingAttention.info[ctx.record_time_key]['bwd_time'
                ] += elapsed
            ForgettingAttention.info[ctx.record_time_key]['bwd_count'] += 1
        return (dq, dk, dv, dlog_fgate, None, None, None, None, None, None,
            None, None, None)
