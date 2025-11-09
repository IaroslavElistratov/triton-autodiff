# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Dao-AILab/flash-attention
# Source-Files: flash_attn/flash_attn_triton_amd/fp8.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_aqv8vcou/flash-attention-main/flash_attn/flash_attn_triton_amd/fp8.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


@triton.jit
def _cast_varlen_to_fp8_kernel_2d(X, X_fp8, Descale, cu_seqlens, H,
    MAX_SEQLEN, stride_batch, stride_seq, stride_head, stride_dim,
    stride_out_batch, stride_out_seq, stride_out_head, stride_out_dim,
    stride_desc_batch, stride_desc_head, FP8_CLAMP_VAL, FP8_MAX, BLOCK_SIZE:
    tl.constexpr, HEAD_DIM: tl.constexpr, ACTUAL_HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    if IS_VARLEN:
        seq_start = tl.load(cu_seqlens + b_id)
        seq_end = tl.load(cu_seqlens + b_id + 1)
        seqlen = seq_end - seq_start
    else:
        seq_start = 0
        seqlen = MAX_SEQLEN
    x_max_val = 0.0
    num_of_blocks = tl.cdiv(seqlen, BLOCK_SIZE)
    for blk_idx in range(0, num_of_blocks):
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)
        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim
        adj_x = (b_id * stride_batch + h_id * stride_head + seq_start *
            stride_seq + offs_seq[:, None] * stride_seq + offs_dim[None, :] *
            stride_dim)
        x_block = tl.load(X + adj_x, mask=mask_seq, other=0.0)
        block_max = tl.max(tl.abs(x_block))
        x_max_val = tl.maximum(x_max_val, block_max)
    x_max_val = tl.maximum(x_max_val, FP8_CLAMP_VAL)
    scale = FP8_MAX / x_max_val
    descale = x_max_val / FP8_MAX
    desc_ptr = Descale + b_id * stride_desc_batch + h_id
    tl.store(desc_ptr, descale)
    for blk_idx in range(0, num_of_blocks):
        offs_seq = blk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_dim = tl.arange(0, HEAD_DIM)
        mask_seq = offs_seq[:, None] < seqlen
        if ACTUAL_HEAD_DIM != HEAD_DIM:
            mask_dim = offs_dim[None, :] < ACTUAL_HEAD_DIM
            mask_seq = mask_seq & mask_dim
        addr = (b_id * stride_batch + h_id * stride_head + seq_start *
            stride_seq + offs_seq[:, None] * stride_seq + offs_dim[None, :] *
            stride_dim)
        x_block = tl.load(X + addr, mask=mask_seq, other=0.0)
        x_fp8_block = (x_block * scale).to(X_fp8.type.element_ty)
        addr_out = (b_id * stride_out_batch + h_id * stride_out_head + 
            seq_start * stride_out_seq + offs_seq[:, None] * stride_out_seq +
            offs_dim[None, :] * stride_out_dim)
        tl.store(X_fp8 + addr_out, x_fp8_block, mask=mask_seq)


@functools.cache
def arch_supports_fp8():
    return is_hip() and get_arch() in 'gfx942'


def cast_to_fp8(x: torch.Tensor, fp8_dtype: torch.dtype, layout: Literal[
    'bshd', 'thd'], clamp_val: float=1e-09, cu_seqlens: Optional[torch.
    Tensor]=None, max_seqlen: Optional[int]=None) ->tuple[torch.Tensor,
    torch.Tensor]:
    if False:
        print()
        print('cast_to_fp8')
        print('x:', x, x.shape)
        print('fp8_dtype:', fp8_dtype)
        print('cu_seqlens:', cu_seqlens)
        print('max_seqlen:', max_seqlen)
        print('clamp_val:', clamp_val)
    assert x.dtype in {torch.float16, torch.float32, torch.float64, torch.
        bfloat16} and is_dtype_fp8(fp8_dtype
        ), f'Cannot cast {x.dtype} to {fp8_dtype}'
    batch, max_seqlen_final, num_heads, head_dim = get_shape_from_layout(x,
        layout, cu_seqlens, max_seqlen)
    is_varlen = layout == 'thd'
    fp8_max = torch.finfo(fp8_dtype).max
    if False:
        print('batch:', batch)
        print('max_seqlen_final:', max_seqlen_final)
        print('num_heads:', num_heads)
        print('head_dim:', head_dim)
    padded_head_dim = 1 << (head_dim - 1).bit_length()
    padded_head_dim = max(padded_head_dim, 32)
    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    descale_factors = torch.zeros((batch, num_heads), device=x.device,
        dtype=torch.float32)
    BLOCK_SIZE = 128
    stride_batch, stride_head, stride_seq, stride_dim = get_stride_from_layout(
        x, layout)
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim = (
        get_stride_from_layout(x_fp8, layout))
    stride_desc_batch, stride_desc_head = descale_factors.stride()
    if False:
        print('stride_batch', stride_batch)
        print('stride_head', stride_head)
        print('stride_seq', stride_seq)
        print('stride_dim', stride_dim)
        print('stride_out_batch', stride_out_batch)
        print('stride_out_head', stride_out_head)
        print('stride_out_seq', stride_out_seq)
        print('stride_out_dim', stride_out_dim)
        print('stride_desc_batch', stride_desc_batch)
        print('stride_desc_head', stride_desc_head)
    grid = batch, num_heads
    _cast_varlen_to_fp8_kernel_2d[grid](x, x_fp8, descale_factors,
        cu_seqlens, num_heads, max_seqlen_final, stride_batch, stride_seq,
        stride_head, stride_dim, stride_out_batch, stride_out_seq,
        stride_out_head, stride_out_dim, stride_desc_batch,
        stride_desc_head, clamp_val, fp8_max, BLOCK_SIZE=BLOCK_SIZE,
        HEAD_DIM=padded_head_dim, ACTUAL_HEAD_DIM=head_dim, IS_VARLEN=is_varlen
        )
    if False:
        print('x_fp8:', x_fp8, x_fp8.shape)
        print('descale_factors:', descale_factors, descale_factors.shape)
    return x_fp8, descale_factors


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device='cuda').unsqueeze(
        -1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device='cuda').unsqueeze(
        0)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos


@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    x_amax = tl.max(tl.abs(x))
    x_amax = tl.where(x_amax <= 1e-09, 1e-09, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


def create_dropout_mask(dropout_p, shape, seed):
    device = 'cuda'
    rand_vals = torch.rand(shape, generator=torch.Generator(device=device).
        manual_seed(seed), device=device, dtype=torch.float32)
    return rand_vals > dropout_p


@functools.cache
def get_arch():
    return triton.runtime.driver.active.get_current_target().arch


def get_shape_from_layout(x: torch.Tensor, layout: Literal['bshd', 'bhsd',
    'thd'], cu_seqlens: Optional[torch.Tensor]=None, max_seqlen: Optional[
    int]=None) ->tuple[int, int, int, int]:
    if layout == 'bhsd':
        batch, num_heads, max_seqlen_final, head_dim = x.shape
    elif layout == 'bshd':
        batch, max_seqlen_final, num_heads, head_dim = x.shape
    elif layout == 'thd':
        total_seqlen, num_heads, head_dim = x.shape
        if cu_seqlens is None:
            raise ValueError(
                'cu_seqlens must be provided for varlen (thd) layout')
        if max_seqlen is None:
            raise ValueError(
                'max_seqlen must be provided for varlen (thd) layout')
        batch, max_seqlen_final, num_heads, head_dim = len(cu_seqlens
            ) - 1, max_seqlen, num_heads, head_dim
    else:
        assert False, 'Got unsupported layout.'
    return batch, max_seqlen_final, num_heads, head_dim


def get_shapes_from_layout(q, k, layout, cu_seqlens_q=None, cu_seqlens_k=
    None, max_seqlen_q=None, max_seqlen_k=None):
    batch_q, seqlen_q, nheads_q, head_size_q = get_shape_from_layout(q,
        layout, cu_seqlens_q, max_seqlen_q)
    batch_k, seqlen_k, nheads_k, head_size_k = get_shape_from_layout(k,
        layout, cu_seqlens_k, max_seqlen_k)
    assert batch_q == batch_k
    assert head_size_q == head_size_k
    return batch_q, nheads_q, nheads_k, head_size_q, seqlen_q, seqlen_k


def get_stride_from_layout(x: torch.Tensor, layout: Literal['bshd', 'bhsd',
    'thd']):
    if layout == 'thd':
        strides = 0, x.stride(1), x.stride(0), x.stride(2)
    elif layout == 'bhsd':
        strides = x.stride(0), x.stride(1), x.stride(2), x.stride(3)
    elif layout == 'bshd':
        strides = x.stride(0), x.stride(2), x.stride(1), x.stride(3)
    else:
        assert False, 'Got unsupported layout.'
    return strides


def get_strides_from_layout(q, k, v, o, layout):
    q_strides = get_stride_from_layout(q, layout)
    k_strides = get_stride_from_layout(k, layout)
    v_strides = get_stride_from_layout(v, layout)
    o_strides = get_stride_from_layout(o, layout)
    return q_strides, k_strides, v_strides, o_strides


def is_dtype_fp8(dtype):
    if dtype in {torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.
        float8_e5m2, torch.float8_e5m2fnuz}:
        if arch_supports_fp8():
            return True
        else:
            raise RuntimeError('This device doesnot support fp8')
    else:
        return False


def is_fp8(x):
    return is_dtype_fp8(x.dtype)


@functools.cache
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == 'hip'


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn,
    stride_vk, stride_bn, stride_sn, start_m, actual_seqlen_k,
    actual_seqlen_q, dropout_p, philox_seed, philox_ptrs, sd_mask_ptrs,
    dropout_mask_ptrs, block_min, block_max, offs_n_causal, masked_blocks,
    n_extra_tokens, alibi_slope, descale_q, descale_k, descale_v, IS_FP8:
    tl.constexpr, FP8_MAX: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M:
    tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, OFFS_M:
    tl.constexpr, OFFS_N: tl.constexpr, PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr, ENABLE_DROPOUT: tl.constexpr, PADDED_HEAD: tl
    .constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, SM_SCALE: tl.constexpr,
    USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, RETURN_SCORES: tl.
    constexpr, ACCUMULATOR_TYPE):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
    for start_n in range(block_min, block_max, BLOCK_N):
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL,
            actual_seqlen_k)
        if PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k,
                ACTUAL_BLOCK_DMODEL)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        if MASK_STEPS:
            if start_n + BLOCK_N == block_max and n_extra_tokens != 0:
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32
                    )
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float('-inf'))
        q_mask = OFFS_M[:, None] < actual_seqlen_q
        k_mask = (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k
        p_mask = q_mask & k_mask
        if IS_FP8:
            qk += tl.dot(q, k) * descale_q * descale_k
        else:
            qk += tl.dot(q, k)
        qk_scaled = qk * SM_SCALE
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float('-inf'))
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N
                ) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q,
                actual_seqlen_k)
            qk_scaled += bias
        if USE_ALIBI:
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q,
                actual_seqlen_k, global_m_positions, global_n_positions)
            qk_scaled += alibi_block
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))
        q_shifted = qk_scaled - m_ij[:, None]
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            if tl_DROPOUT_USE_PYTORCH:
                dropout_mask = tl.load(dropout_mask_ptrs, mask=p_mask)
            else:
                rng_output = tl.rand(philox_seed, philox_ptrs)
                dropout_mask = rng_output > dropout_p
                if tl_DROPOUT_DUMP:
                    tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            tl.store(sd_mask_ptrs, p, mask=p_mask)
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k,
                ACTUAL_BLOCK_DMODEL)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        if IS_FP8:
            scale_p, descale_p = compute_fp8_scaling_factors(p, FP8_MAX)
            acc += tl.dot((p * scale_p).to(v.type.element_ty), v
                ) * descale_p * descale_v
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn
    return acc, l_i, m_i


@triton.autotune(configs=autotune_configs, key=autotune_keys,
    use_cuda_graph=True)
@triton.jit
def attn_fwd(Q, K, V, bias, Cache_seqlens, Cache_batch_idx, Descale_Q,
    Descale_K, Descale_V, Descale_O, stride_descale_q_z, stride_descale_k_z,
    stride_descale_v_z, stride_descale_o_z, SM_SCALE: tl.constexpr, LSE,
    Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
    stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on, stride_bz, stride_bh,
    stride_bm, stride_bn, stride_az, stride_ah, stride_sz, stride_sh,
    stride_sm, stride_sn, stride_lse_z, stride_lse_h, stride_lse_m,
    cu_seqlens_q, cu_seqlens_k, dropout_p, philox_seed, philox_offset_base,
    sd_mask, dropout_mask, alibi_slopes, HQ: tl.constexpr, HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr, IS_VARLEN: tl.constexpr, IS_INFERENCE: tl.
    constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL:
    tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, USE_BIAS:
    tl.constexpr, ENABLE_DROPOUT: tl.constexpr, RETURN_SCORES: tl.constexpr,
    USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr):
    ACCUMULATOR_TYPE = tl.float32
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    if IS_VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    elif IS_INFERENCE:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = tl.load(Cache_seqlens + off_z)
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K
    n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_seqlen = tl.cdiv((start_m + 1) * BLOCK_M + seqlen_k -
            seqlen_q, BLOCK_N)
        n_blocks = min(n_blocks, n_blocks_seqlen)
        if n_blocks <= 0:
            o_offset = (Out + off_z * stride_oz + off_h_q * stride_oh + 
                cu_seqlens_q_start * stride_om)
            o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :
                ] * stride_on
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)
            l_offset = (LSE + off_z * stride_lse_z + off_h_q * stride_lse_h +
                cu_seqlens_q_start * stride_lse_m)
            l_ptrs = l_offset + offs_m * stride_lse_m
            l = tl.full([BLOCK_M], value=0.0, dtype=ACCUMULATOR_TYPE)
            l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            return
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL
    q_offset = (Q + off_z * stride_qz + off_h_q * stride_qh + 
        cu_seqlens_q_start * stride_qm)
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :
        ] * stride_qk
    k_offset = (K + off_z * stride_kz + off_h_k * stride_kh + 
        cu_seqlens_k_start * stride_kn)
    k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :
        ] * stride_kn
    v_offset = (V + off_z * stride_vz + off_h_k * stride_vh + 
        cu_seqlens_k_start * stride_vk)
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :
        ] * stride_vn
    if USE_BIAS:
        bias_offset = off_h_q * stride_bh
        bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[
            None, :] * stride_bn
    else:
        bias_ptrs = None
    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None
    if RETURN_SCORES:
        sd_mask_offset = sd_mask + off_z * stride_sz + off_h_q * stride_sh
        sd_mask_ptrs = sd_mask_offset + offs_m[:, None] * stride_sm + offs_n[
            None, :] * stride_sn
    else:
        sd_mask_ptrs = None
    if ENABLE_DROPOUT:
        dropout_mask_offset = (dropout_mask + off_z * stride_sz + off_h_q *
            stride_sh)
        dropout_mask_ptrs = dropout_mask_offset + offs_m[:, None
            ] * stride_sm + offs_n[None, :] * stride_sn
        batch_philox_offset = (philox_offset_base + off_z * stride_sz + 
            off_h_q * stride_sh)
        philox_ptrs = batch_philox_offset + offs_m[:, None
            ] * stride_sm + offs_n[None, :] * stride_sn
    else:
        dropout_mask_ptrs = None
        philox_ptrs = 0
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=ACCUMULATOR_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=ACCUMULATOR_TYPE)
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    if IS_FP8:
        descale_q = tl.load(Descale_Q + off_z * stride_descale_q_z + off_h_q)
        descale_k = tl.load(Descale_K + off_z * stride_descale_k_z + off_h_k)
        descale_v = tl.load(Descale_V + off_z * stride_descale_v_z + off_h_k)
    else:
        descale_q, descale_k, descale_v = 1.0, 1.0, 1.0
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and seqlen_q % BLOCK_M == 0
    if IS_CAUSAL:
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs,
            bias_ptrs, stride_kn, stride_vk, stride_bn, stride_sn, start_m,
            seqlen_k, seqlen_q, dropout_p, philox_seed, philox_ptrs,
            sd_mask_ptrs, dropout_mask_ptrs, block_min, block_max, 0, 0, 0,
            alibi_slope, descale_q, descale_k, descale_v, IS_FP8, FP8_MAX, 
            False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
            PRE_LOAD_V, False, ENABLE_DROPOUT, PADDED_HEAD,
            ACTUAL_BLOCK_DMODEL, SM_SCALE, USE_ALIBI=USE_ALIBI, USE_EXP2=
            USE_EXP2, RETURN_SCORES=RETURN_SCORES, ACCUMULATOR_TYPE=
            ACCUMULATOR_TYPE)
        block_min = block_max
        block_max = n_blocks * BLOCK_N
    tl.debug_barrier()
    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if USE_BIAS:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        if RETURN_SCORES:
            sd_mask_ptrs += n_full_blocks * BLOCK_N * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += n_full_blocks * BLOCK_N * stride_sn
            philox_ptrs += n_full_blocks * BLOCK_N * stride_sn
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs,
            bias_ptrs, stride_kn, stride_vk, stride_bn, stride_sn, start_m,
            seqlen_k, seqlen_q, dropout_p, philox_seed, philox_ptrs,
            sd_mask_ptrs, dropout_mask_ptrs, block_min, block_max,
            offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope,
            descale_q, descale_k, descale_v, IS_FP8, FP8_MAX, IS_CAUSAL,
            BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n, PRE_LOAD_V, 
            True, ENABLE_DROPOUT, PADDED_HEAD, ACTUAL_BLOCK_DMODEL,
            SM_SCALE, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, RETURN_SCORES
            =RETURN_SCORES, ACCUMULATOR_TYPE=ACCUMULATOR_TYPE)
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL,), causal_start_idx,
                dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[
                None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
    l_offset = (LSE + off_z * stride_lse_z + off_h_q * stride_lse_h + 
        cu_seqlens_q_start * stride_lse_m)
    l_ptrs = l_offset + offs_m * stride_lse_m
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        softmax_lse *= LN2
    else:
        softmax_lse = m_i + tl.math.log(l_i)
    if IS_CAUSAL:
        lse_mask = start_m_idx + tl.arange(0, BLOCK_M) < causal_start_idx
        softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, softmax_lse)
    o_offset = (Out + off_z * stride_oz + off_h_q * stride_oh + 
        cu_seqlens_q_start * stride_om)
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :
        ] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    if FP8_OUTPUT:
        scale_acc, descale_acc = compute_fp8_scaling_factors(acc, FP8_MAX)
        tl.store(Descale_O + off_z * stride_descale_o_z + off_h_q, descale_acc)
        tl.store(o_ptrs, (acc * scale_acc).to(Out.type.element_ty), mask=
            o_ptrs_mask)
    else:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second
    ):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (offset_second[
            None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n,
    transpose=False):
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :
        ]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


def attention_prefill_forward_triton_impl(q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, o: torch.Tensor, sm_scale: float, alibi_slopes:
    Optional[torch.Tensor], causal: bool, bias: Optional[torch.Tensor],
    layout: Literal['bshd', 'bhsd', 'thd'], cu_seqlens_q: Optional[torch.
    Tensor], cu_seqlens_k: Optional[torch.Tensor], max_seqlens_q: int,
    max_seqlens_k: int, cache_seqlens: Optional[Union[int, torch.Tensor]],
    cache_batch_idx: Optional[torch.Tensor], dropout_p: float, philox_seed:
    Optional[int], philox_offset: Optional[int], return_softmax: bool,
    use_exp2: bool, descale_q: Optional[torch.Tensor], descale_k: Optional[
    torch.Tensor], descale_v: Optional[torch.Tensor], descale_o: Optional[
    torch.Tensor]):
    IS_FP8 = is_fp8(q)
    if IS_FP8:
        FP8_MAX: tl.constexpr = torch.finfo(q.dtype).max
        assert is_fp8(q) and is_fp8(k) and is_fp8(v
            ), f'Non fp8 type found: q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}. All tensors must be fp8.'
        if is_fp8(o):
            FP8_OUTPUT = True
            assert descale_o is not None, f'descale_o is None. In fp8, you need to pass a tensor for descale_o along with a tensor for the output.'
        else:
            FP8_OUTPUT = False
        stride_descale_q_z = descale_q.stride(0
            ) if descale_q is not None else None
        stride_descale_k_z = descale_k.stride(0
            ) if descale_k is not None else None
        stride_descale_v_z = descale_v.stride(0
            ) if descale_v is not None else None
        stride_descale_o_z = descale_o.stride(0
            ) if descale_o is not None else None
    else:
        FP8_MAX = None
        FP8_OUTPUT = False
        descale_q = descale_k = descale_v = descale_o = None
        (stride_descale_q_z) = (stride_descale_k_z) = (stride_descale_v_z) = (
            stride_descale_o_z) = None
    is_varlen = layout == 'thd'
    use_alibi, (stride_az, stride_ah) = (True, alibi_slopes.stride()
        ) if alibi_slopes is not None else (False, (0, 0))
    is_inference = False if cache_seqlens is None else True
    if is_inference:
        assert layout == 'bshd', f'{layout} layout is not supported with inference. Use bshd layout'
    if DEBUG:
        print(f'is_inference:', is_inference)
    if bias is not None:
        assert bias.numel() < 2 ** 31
    batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k = (
        get_shapes_from_layout(q, k, layout, cu_seqlens_q, cu_seqlens_k,
        max_seqlens_q, max_seqlens_k))
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q,
        k, v, o, layout)
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    grid = lambda META: (triton.cdiv(max_seqlens_q, META['BLOCK_M']),
        nheads_q, batch)
    use_dropout = dropout_p > 0.0
    if use_dropout or return_softmax:
        sd_mask = torch.zeros((batch, nheads_q, max_seqlens_q,
            max_seqlens_k), device=q.device, dtype=torch.float32)
        if DROPOUT_USE_PYTORCH:
            dropout_mask = create_dropout_mask(dropout_p, (batch, nheads_q,
                max_seqlens_q, max_seqlens_k), seed=philox_seed)
        else:
            dropout_mask = torch.zeros((batch, nheads_q, max_seqlens_q,
                max_seqlens_k), device=q.device, dtype=torch.float32)
        scores_strides = sd_mask.stride(0), sd_mask.stride(1), sd_mask.stride(2
            ), sd_mask.stride(3)
    else:
        sd_mask = None
        dropout_mask = None
        scores_strides = 0, 0, 0, 0
    if is_varlen:
        total_seqlen_q, _, _ = q.shape
        softmax_lse = torch.zeros((nheads_q, total_seqlen_q), device=q.
            device, dtype=torch.float32)
        stride_lse_h, stride_lse_m = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.zeros((batch, nheads_q, max_seqlens_q), device=
            q.device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()
    if bias is not None:
        bias_strides = bias.stride(0), bias.stride(1), bias.stride(2
            ), bias.stride(3)
    else:
        bias_strides = 0, 0, 0, 0
    attn_fwd[grid](q, k, v, bias, cache_seqlens, cache_batch_idx, descale_q,
        descale_k, descale_v, descale_o, stride_descale_q_z,
        stride_descale_k_z, stride_descale_v_z, stride_descale_o_z,
        sm_scale, softmax_lse, o, *q_strides, *k_strides, *v_strides, *
        o_strides, *bias_strides, stride_az, stride_ah, *scores_strides,
        stride_lse_z, stride_lse_h, stride_lse_m, cu_seqlens_q,
        cu_seqlens_k, dropout_p=dropout_p, philox_seed=philox_seed,
        philox_offset_base=philox_offset, sd_mask=sd_mask, dropout_mask=
        dropout_mask, alibi_slopes=alibi_slopes, HQ=nheads_q, HK=nheads_k,
        ACTUAL_BLOCK_DMODEL=head_size, MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k, IS_CAUSAL=causal, IS_VARLEN=is_varlen,
        IS_INFERENCE=is_inference, BLOCK_DMODEL=padded_d_model, USE_BIAS=
        False if bias is None else True, USE_ALIBI=use_alibi,
        ENABLE_DROPOUT=dropout_p > 0.0, USE_EXP2=use_exp2, RETURN_SCORES=
        return_softmax, IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=FP8_OUTPUT)
    return softmax_lse, sd_mask if return_softmax else None


def attention_forward_core_ref_impl(q, k, v, sm_scale, causal, dropout_p,
    philox_seed, philox_offset, alibi_slopes, use_exp2):
    if DEBUG_CORE:
        print()
        print('attention_forward_core_ref_impl')
        print('q:', q, q.shape)
        print('k:', k, k.shape)
        print('v:', v, v.shape)
        print('sm_scale:', sm_scale)
        print('causal:', causal)
        print('dropout_p:', dropout_p)
        print('philox_seed:', philox_seed)
        print('philox_offset:', philox_offset)
        print('use_exp2:', use_exp2)
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    if DEBUG_CORE:
        print('attention_scores:', attention_scores, attention_scores.shape)
    attention_scaled_scores = sm_scale * attention_scores
    if DEBUG_CORE:
        print('attention_scaled_scores:', attention_scaled_scores,
            attention_scaled_scores.shape)
    if alibi_slopes is not None:
        L_q, L_k = q.shape[1], k.shape[1]
        if DEBUG_CORE:
            print('alibi_slopes:', alibi_slopes, alibi_slopes.shape)
        alibi_bias = compute_alibi_tensor_ref(alibi_slopes, L_q, L_k)
        if DEBUG_CORE:
            print('alibi_bias:', alibi_bias, alibi_bias.shape)
        alibi_bias = alibi_bias.reshape(-1, L_q, L_k)
        if DEBUG_CORE:
            print('alibi_bias_flat:', alibi_bias, alibi_bias.shape)
        attention_scaled_scores = attention_scaled_scores + alibi_bias
        if DEBUG_CORE:
            print('attention_scaled_scores after alibi:',
                attention_scaled_scores, attention_scaled_scores.shape)
    if causal:
        L_q, L_k = q.shape[1], k.shape[1]
        row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
        col_offset = L_q - L_k
        causal_mask = row_idx >= col_offset + col_idx
        if DEBUG_CORE:
            print('causal_mask:', causal_mask)
        attention_scaled_scores = attention_scaled_scores.masked_fill(torch
            .logical_not(causal_mask.unsqueeze(0)), float('-inf'))
        if DEBUG_CORE:
            print('attention_scaled_scores after causal:',
                attention_scaled_scores, attention_scaled_scores.shape)
    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]
    if DEBUG_CORE:
        print('max_scores:', max_scores, max_scores.shape)
    if causal:
        max_scores = torch.where(torch.isinf(max_scores), torch.zeros_like(
            max_scores), max_scores)
        if DEBUG:
            print('max_scores if causal:', max_scores, max_scores.shape)
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores
    if DEBUG_CORE:
        print('attention_shifted_scaled_scores:',
            attention_shifted_scaled_scores,
            attention_shifted_scaled_scores.shape)
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        exp_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
    else:
        exp_scores = torch.exp(attention_shifted_scaled_scores)
    if DEBUG_CORE:
        print('exp_scores:', exp_scores, exp_scores.shape)
    sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)
    if DEBUG_CORE:
        print('sum_exp_scores:', sum_exp_scores, sum_exp_scores.shape)
    if causal:
        sum_exp_scores = torch.where(sum_exp_scores == 0, torch.ones_like(
            sum_exp_scores), sum_exp_scores)
    if DEBUG_CORE:
        print('sum_exp_scores:', sum_exp_scores, sum_exp_scores.shape)
    p = exp_scores / sum_exp_scores
    if DEBUG_CORE:
        print('softmax:', p, p.shape)
    if dropout_p > 0.0:
        rand_vals = torch.rand(p.shape, generator=torch.Generator(device=p.
            device).manual_seed(philox_seed), device=p.device, dtype=p.dtype)
        dropout_mask, dropout_scale = rand_vals > dropout_p, 1.0 / (1 -
            dropout_p)
        if DEBUG_CORE:
            print('dropout_scale:', dropout_scale)
            print('dropout_mask:', dropout_mask)
        sd_mask = torch.where(dropout_mask, exp_scores, -exp_scores)
        p = torch.where(dropout_mask, p, torch.zeros_like(p)) * dropout_scale
        if DEBUG_CORE:
            print('softmax after dropout:', p)
            print('sd_mask:', sd_mask)
    else:
        sd_mask = exp_scores
    if use_exp2:
        LN2 = math.log(2)
        RCP_LN = 1 / math.log(2)
        max_scores_base2 = max_scores * RCP_LN
        softmax_lse_base2 = max_scores_base2 + torch.log2(sum_exp_scores)
        softmax_lse = softmax_lse_base2 * LN2
        softmax_lse.squeeze_(-1)
    else:
        softmax_lse = max_scores + torch.log(sum_exp_scores)
        softmax_lse = softmax_lse.squeeze(-1)
    if DEBUG_CORE:
        print('softmax_lse:', softmax_lse, softmax_lse.shape)
    o = torch.matmul(p, v)
    if DEBUG_CORE:
        print('o:', o, o.shape)
    o = o.to(torch.float16)
    sd_mask = sd_mask.to(torch.float16)
    return o, softmax_lse, sd_mask


def attention_forward_pytorch_ref_impl(q: torch.Tensor, k: torch.Tensor, v:
    torch.Tensor, out: torch.Tensor, sm_scale: float, alibi_slopes:
    Optional[torch.Tensor], causal: bool, layout: Literal['bshd', 'bhsd',
    'thd'], cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int, max_seqlen_k: int, dropout_p: float, philox_seed:
    Optional[int], philox_offset: Optional[int], use_exp2: bool):
    if layout == 'thd':
        o_ref, softmax_lse_ref, sd_mask_ref = (
            attention_varlen_forward_pytorch_ref_impl(q.clone(), k.clone(),
            v.clone(), sm_scale, causal, layout, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, dropout_p, philox_seed,
            philox_offset, alibi_slopes, use_exp2))
    else:
        o_ref, softmax_lse_ref, sd_mask_ref = (
            attention_vanilla_forward_pytorch_ref_impl(q.clone(), k.clone(),
            v.clone(), sm_scale, causal, layout, dropout_p, philox_seed,
            philox_offset, alibi_slopes, use_exp2))
    out.copy_(o_ref.to(out.dtype))
    return softmax_lse_ref, sd_mask_ref


def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal,
    layout, dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2):
    """Compute reference output and softmax_lse using PyTorch's built-in function"""
    if layout == 'bshd':
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    elif layout != 'bhsd':
        raise ValueError(f'Unknown layout {layout}')
    batch_size, nheads_q, seq_len_q, head_dim = q.shape
    batch_size, nheads_k, seq_len_k, head_dim = k.shape
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError('nheads_q must be divisible by nheads_k')
    if group_size != 1:
        q = q.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        q = q.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
    else:
        q = q.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k, seq_len_k, head_dim)
    o, softmax_lse, sd_mask = attention_forward_core_ref_impl(q, k, v,
        sm_scale, causal, dropout_p, philox_seed, philox_offset,
        alibi_slopes, use_exp2)
    if group_size != 1:
        o = o.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        o = o.reshape(batch_size, nheads_q, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_k, group_size,
            seq_len_q)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_q, seq_len_q)
        sd_mask = sd_mask.reshape(batch_size, nheads_k, group_size,
            seq_len_q, seq_len_k)
        sd_mask = sd_mask.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
    else:
        o = o.reshape(batch_size, nheads_q, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_q, seq_len_q)
        sd_mask = sd_mask.reshape(batch_size, nheads_q, seq_len_q, seq_len_k)
    if layout == 'bshd':
        o = o.transpose(1, 2)
    return o, softmax_lse, sd_mask


def attention_varlen_forward_pytorch_ref_impl(q, k, v, sm_scale, causal,
    layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2):
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")
    batch_size = cu_seqlens_q.shape[0] - 1
    nheads_q, nheads_k = q.shape[1], k.shape[1]
    head_dim = q.shape[2]
    total_L_q = q.shape[0]
    total_L_k = k.shape[0]
    o = torch.zeros((total_L_q, nheads_q, head_dim), dtype=q.dtype, device=
        q.device)
    softmax_lse = torch.zeros((total_L_q, nheads_q), dtype=torch.float32,
        device=q.device)
    sd_mask = torch.zeros((batch_size, nheads_q, max_seqlen_q, max_seqlen_k
        ), dtype=torch.float32, device=q.device)
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError('nheads_q must be divisible by nheads_k')
    for i in range(batch_size):
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()
        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        if DEBUG:
            print(
                f'Batch {i} with seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, Hq= {nheads_q}, Hk = {nheads_k}'
                )
        q_i = q[start_q:end_q, :, :]
        k_i = k[start_k:end_k, :, :]
        v_i = v[start_k:end_k, :, :]
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)
        if group_size != 1:
            q_i = q_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            k_i = k_i.unsqueeze(1).expand(-1, group_size, -1, -1)
            v_i = v_i.unsqueeze(1).expand(-1, group_size, -1, -1)
            q_i = q_i.reshape(nheads_k * group_size, seqlen_q, head_dim)
            k_i = k_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
            v_i = v_i.reshape(nheads_k * group_size, seqlen_k, head_dim)
        else:
            q_i = q_i.reshape(nheads_q, seqlen_q, head_dim)
            k_i = k_i.reshape(nheads_k, seqlen_k, head_dim)
            v_i = v_i.reshape(nheads_k, seqlen_k, head_dim)
        if alibi_slopes is not None:
            alibi_slopes_i = alibi_slopes[i]
        else:
            alibi_slopes_i = None
        o_i, softmax_lse_i, sd_mask_i = attention_forward_core_ref_impl(q_i,
            k_i, v_i, sm_scale, causal, dropout_p, philox_seed,
            philox_offset, alibi_slopes_i, use_exp2)
        if group_size != 1:
            o_i = o_i.reshape(nheads_k, group_size, seqlen_q, head_dim)
            o_i = o_i.reshape(nheads_q, seqlen_q, head_dim)
            softmax_lse_i = softmax_lse_i.reshape(nheads_k, group_size,
                seqlen_q)
            softmax_lse_i = softmax_lse_i.reshape(nheads_q, seqlen_q)
        else:
            pass
        o_i = o_i.permute(1, 0, 2)
        softmax_lse_i = softmax_lse_i.permute(1, 0)
        sd_mask_i = sd_mask_i
        o[start_q:end_q, :, :] = o_i
        softmax_lse[start_q:end_q, :] = softmax_lse_i
        sd_mask[i, :, :seqlen_q, :seqlen_k] = sd_mask_i
    return o, softmax_lse, sd_mask


def varlen_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out:
    Optional[torch.Tensor], cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch
    .Tensor, seqused_k: Optional[torch.Tensor], leftpad_k: Optional[torch.
    Tensor], block_table_: Optional[torch.Tensor], alibi_slopes: Optional[
    torch.Tensor], max_seqlen_q: int, max_seqlen_k: int, dropout_p: float,
    softmax_scale: float, zero_tensors: bool, causal: bool,
    window_size_left: int, window_size_right: int, softcap: float,
    return_softmax: bool, gen_: Optional[torch.Tensor]=None, descale_q:
    Optional[torch.Tensor]=None, descale_k: Optional[torch.Tensor]=None,
    descale_v: Optional[torch.Tensor]=None, descale_o: Optional[torch.
    Tensor]=None):
    if DEBUG:
        print()
        print('flash_attn_triton_amd.py::varlen_fwd')
        print('q:', q, q.shape)
        print('k:', k, k.shape)
        print('v:', v, v.shape)
        print('cu_seqlens_q:', cu_seqlens_q, cu_seqlens_q.shape)
        print('cu_seqlens_k:', cu_seqlens_k, cu_seqlens_k.shape)
        print('alibi_slopes:', alibi_slopes)
        print('max_seqlen_q:', max_seqlen_q)
        print('max_seqlen_k:', max_seqlen_k)
        print('dropout_p:', dropout_p)
        print('softmax_scale:', softmax_scale)
        print('causal:', causal)
        print('window_size_left:', window_size_left)
        print('window_size_right:', window_size_right)
        print('gen_:', gen_)
        print('descale_q:', descale_q, descale_q.shape if descale_q is not
            None else None)
        print('descale_k:', descale_k, descale_k.shape if descale_k is not
            None else None)
        print('descale_v:', descale_v, descale_v.shape if descale_v is not
            None else None)
    if is_fp8(q):
        assert out is not None, 'fp8 output tensor should be passed in.'
        assert descale_q is not None and descale_k is not None and descale_v is not None, f'For fp8, you need to pass descale factors for q, k and v'
    else:
        out = torch.zeros_like(q) if out is None else out.zero_()
    metadata = MetaData(sm_scale=softmax_scale)
    if return_softmax:
        metadata.return_scores = True
    metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        max_seqlen_k)
    assert metadata.layout is not None
    batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k = (
        get_shapes_from_layout(q, k, metadata.layout, cu_seqlens_q,
        cu_seqlens_k, max_seqlen_q, max_seqlen_k))
    if causal:
        metadata.need_causal(True)
    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)
    metadata.need_dropout(dropout_p, return_softmax)
    rng_state = torch.as_tensor([metadata.philox_seed, metadata.philox_offset])
    metadata.check_args(q, k, v, out)
    if USE_REF:
        if DEBUG:
            print('Using reference implementation')
        softmax_lse_ref, sd_mask_ref = attention_forward_pytorch_ref_impl(q,
            k, v, out, metadata.sm_scale, metadata.alibi_slopes, metadata.
            causal, metadata.layout, metadata.cu_seqlens_q, metadata.
            cu_seqlens_k, metadata.max_seqlens_q, metadata.max_seqlens_k,
            metadata.dropout_p, metadata.philox_seed, metadata.
            philox_offset, metadata.use_exp2)
        softmax_lse = softmax_lse_ref
        sd_mask = sd_mask_ref
    else:
        if DEBUG:
            print('Using Triton implementation')
        softmax_lse_triton, sd_mask_triton = (
            attention_prefill_forward_triton_impl(q, k, v, out, metadata.
            sm_scale, metadata.alibi_slopes, metadata.causal, None,
            metadata.layout, metadata.cu_seqlens_q, metadata.cu_seqlens_k,
            metadata.max_seqlens_q, metadata.max_seqlens_k, metadata.
            cache_seqlens, metadata.cache_batch_idx, metadata.dropout_p,
            metadata.philox_seed, metadata.philox_offset, metadata.
            return_scores, metadata.use_exp2, descale_q, descale_k,
            descale_v, descale_o))
        softmax_lse = softmax_lse_triton
        sd_mask = sd_mask_triton
    if DEBUG:
        print('varlen_fwd outputs')
        print('out:', out, out.shape)
        print('softmax_lse:', softmax_lse, softmax_lse.shape)
        print('sd_mask:', sd_mask, sd_mask.shape if sd_mask is not None else
            None)
    return out, softmax_lse, sd_mask, rng_state


def check_args(self, q, k, v, o):
    assert q.dim() == k.dim() and q.dim() == v.dim()
    batch, nheads_q, nheads_k, head_size, _, _ = get_shapes_from_layout(q,
        k, self.layout, self.cu_seqlens_q, self.cu_seqlens_k, self.
        max_seqlens_q, self.max_seqlens_k)
    if self.varlen:
        assert q.dim() == 3
        assert self.cu_seqlens_q is not None
        assert self.cu_seqlens_k is not None
        assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
        assert self.bias is None
    else:
        assert q.dim() == 4
        assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
        assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
    assert k.shape == v.shape
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
    assert q.dtype == k.dtype and q.dtype == v.dtype
    assert o.shape == q.shape
    assert nheads_q % nheads_k == 0
    assert self.layout is not None
    assert self.layout == 'thd' or not self.varlen


def need_alibi(self, alibi_slopes, batch, nheads):
    assert alibi_slopes.is_cuda
    assert alibi_slopes.dim() == 2
    assert alibi_slopes.shape[0] == batch
    assert alibi_slopes.shape[1] == nheads
    self.alibi_slopes = alibi_slopes


def need_causal(self, causal):
    self.causal = causal


def need_dropout(self, dropout_p, return_softmax=True):
    self.dropout_p = dropout_p
    self.return_softmax = return_softmax
    self.philox_seed, self.philox_offset = 114520, 1919817


def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k):
    self.varlen = True
    self.layout = 'thd'
    self.cu_seqlens_q = cu_seqlens_q
    self.cu_seqlens_k = cu_seqlens_k
    self.max_seqlens_q = max_seqlen_q
    self.max_seqlens_k = max_seqlen_k
    assert len(cu_seqlens_q) >= 2
    assert len(cu_seqlens_q) == len(cu_seqlens_k)


# Forward method (kernel launch code)
def _FlashAttnVarlenQKVPackedFP8Func_forward(ctx, qkv, cu_seqlens,
    max_seqlen, dropout_p, softmax_scale, causal, window_size, softcap,
    alibi_slopes, deterministic, return_softmax, is_grad_enabled, descale_q:
    Optional[torch.Tensor]=None, descale_k: Optional[torch.Tensor]=None,
    descale_v: Optional[torch.Tensor]=None, descale_do: Optional[torch.
    Tensor]=None):
    is_grad = is_grad_enabled and qkv.requires_grad
    if softmax_scale is None:
        softmax_scale = qkv.shape[-1] ** -0.5
    q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    if is_fp8(q) or is_fp8(k) or is_fp8(v):
        raise ValueError(
            'fp8 input and out not supported yet for this function.')
        assert descale_q is not None and descale_k is not None and descale_v is not None, f'You need to pass descale factors for q, k and v'
        q_fp8 = q
        k_fp8 = k
        v_fp8 = v
        out_fp8, descale_o = torch.zeros_like(q_fp8), torch.zeros_like(
            descale_q)
    else:
        assert descale_q is None and descale_k is None and descale_v is None, f'Found {q.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired.'
        q_fp8, descale_q = cast_to_fp8(q, torch.float8_e4m3fnuz, 'thd',
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        k_fp8, descale_k = cast_to_fp8(k, torch.float8_e4m3fnuz, 'thd',
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        v_fp8, descale_v = cast_to_fp8(v, torch.float8_e4m3fnuz, 'thd',
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        out_fp8, descale_o = torch.zeros_like(q_fp8, dtype=torch.float32), None
    q_fp8, k_fp8, v_fp8 = [maybe_contiguous(x) for x in (q_fp8, k_fp8, v_fp8)]
    _, softmax_lse, S_dmask, rng_state = flash_attn_gpu.varlen_fwd(q_fp8,
        k_fp8, v_fp8, out_fp8, cu_seqlens, cu_seqlens, None, None, None,
        alibi_slopes, max_seqlen, max_seqlen, dropout_p, softmax_scale, 
        False, causal, window_size[0], window_size[1], softcap,
        return_softmax, None, descale_q=descale_q, descale_k=descale_k,
        descale_v=descale_v, descale_o=descale_o)
    if is_grad:
        ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse,
            cu_seqlens, rng_state, descale_q, descale_k, descale_v,
            descale_o, descale_do)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
    out = out_fp8[..., :head_size_og]
    return out if not return_softmax else (out, softmax_lse, S_dmask)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _bwd_dkdv_inner(dk, dv, Q, k, v, DO, M, D, sm_scale, stride_qm,
    stride_qk, stride_dom, stride_dok, stride_dropoutm, stride_dropoutn,
    stride_deltam, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM:
    tl.constexpr, ACTUAL_HEAD_DIM: tl.constexpr, dropout_p, philox_seed,
    batch_philox_offset, dropout_offset, alibi_slope, seqlen_q, seqlen_k,
    start_n, start_m, num_steps, descale_q, descale_k, descale_v,
    descale_do, MASK: tl.constexpr, ENABLE_DROPOUT: tl.constexpr, USE_ALIBI:
    tl.constexpr, USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr, FP8_MAX: tl
    .constexpr, DEBUG_TRITON: tl.constexpr, DEBUG_TRITON_DETAIL: tl.constexpr):
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < seqlen_k
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
    do_ptrs = DO + offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634
    for blk_idx in range(num_steps):
        if DEBUG_TRITON:
            print(f'iter {blk_idx}: curr_m = {curr_m}')
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < ACTUAL_HEAD_DIM
            mask_do &= offs_k[None, :] < ACTUAL_HEAD_DIM
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        if ENABLE_DROPOUT:
            philox_offs = curr_philox_offset + offs_m[None, :
                ] * stride_dropoutm + offs_n[:, None] * stride_dropoutn
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = offs_m[None, :] * stride_dropoutm + offs_n[:,
                    None] * stride_dropoutn
                dropout_mask = tl.load(curr_dropout_offset + dropout_offs,
                    mask=mask_nm)
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)
        if IS_FP8:
            qkT = tl.dot(k, qT) * descale_q * descale_k
        else:
            qkT = tl.dot(k, qT)
        qkT_scaled = qkT * sm_scale
        if USE_ALIBI:
            relative_pos_block = offs_n[:, None
                ] + seqlen_q - seqlen_k - offs_m[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qkT_scaled += alibi_block
        if DEBUG_TRITON_DETAIL:
            if start_n == 256:
                print(f'qT: {qT.shape}\n', qT)
                print(f'k: {k.shape}\n', k)
                print(f'qkT scaled: {qkT.shape}\n', qkT_scaled)
        if USE_EXP2:
            pT = tl.math.exp2(qkT_scaled * RCP_LN2 - m[None, :] * RCP_LN2)
        else:
            pT = tl.math.exp(qkT_scaled - m[None, :])
        if MASK:
            causal_mask = offs_m[None, :] - delta_qk >= offs_n[:, None]
            mask = causal_mask & mask_nm
            if DEBUG_TRITON_DETAIL:
                if start_n == 256:
                    print(f'causal_mask: {causal_mask.shape}\n', causal_mask)
                    print(f'qkT after causal: {qkT.shape}\n', tl.where(
                        causal_mask, qkT * sm_scale, 0.0))
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        if ENABLE_DROPOUT:
            pT_dropout = tl.where(dropout_mask, pT, 0.0) * dropout_scale
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = (
                    compute_fp8_scaling_factors(pT_dropout, FP8_MAX))
                dv += tl.dot((pT_dropout * scale_p_dropout).to(do.type.
                    element_ty), do) * descale_p_dropout * descale_do
            else:
                dv += tl.dot(pT_dropout.to(do.type.element_ty), do)
        elif IS_FP8:
            scale_pT, descale_pT = compute_fp8_scaling_factors(pT, FP8_MAX)
            dv += tl.dot((pT * scale_pT).to(do.type.element_ty), do
                ) * descale_pT * descale_do
        else:
            dv += tl.dot(pT.to(do.type.element_ty), do)
        if DEBUG_TRITON_DETAIL:
            if start_n == 256:
                print(f'pT: {pT.shape}\n', pT)
        Di = tl.load(D + offs_m * stride_deltam, mask=mask_m)
        if IS_FP8:
            dpT = tl.dot(v, tl.trans(do)) * descale_v * descale_do
        else:
            dpT = tl.dot(v, tl.trans(do))
        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale
        delta_i = Di[None, :]
        dsT = pT * (dpT - delta_i)
        if IS_FP8:
            scale_dsT, descale_dsT = compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT)
                ) * descale_dsT * descale_q
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT))
        curr_m += step_m
        qT_ptrs += step_m * stride_qm
        do_ptrs += step_m * stride_dom
    return dk, dv


@triton.jit
def _bwd_dq_inner(dq, q, K, V, do, m, Delta, sm_scale, stride_qm, stride_qk,
    stride_kn, stride_kk, stride_vn, stride_vk, stride_dropoutm,
    stride_dropoutn, stride_deltam, seqlen_q, seqlen_k, BLOCK_M2: tl.
    constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr, dropout_p, philox_seed,
    batch_philox_offset, dropout_offset, alibi_slope, start_m, start_n,
    end_n, num_steps, descale_q, descale_k, descale_v, descale_do, MASK: tl
    .constexpr, ENABLE_DROPOUT: tl.constexpr, USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr, DEBUG_TRITON_DETAIL: tl.constexpr):
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < seqlen_q
    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    vT_ptrs = V + offs_n[None, :] * stride_vn + offs_k[:, None] * stride_vk
    Di = tl.load(Delta + offs_m * stride_deltam, mask=mask_m, other=0.0)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634
    for blk_idx in range(num_steps):
        if DEBUG_TRITON:
            print(f'iter {blk_idx}: curr_n = {curr_n}')
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        mask_n = offs_n < end_n
        if DEBUG_TRITON_DETAIL:
            print(
                f'start_n = {start_n}, end_n = {end_n}, offs_n: {offs_n.shape}\n{offs_n}'
                )
        if DEBUG_TRITON_DETAIL:
            print(f'mask_n: {mask_n.shape}\n{mask_n}')
        mask_kT = mask_n[None, :]
        mask_mn = mask_m[:, None] & (offs_n[None, :] < end_n)
        if PADDED_HEAD:
            mask_kT &= offs_k[:, None] < ACTUAL_HEAD_DIM
        kT = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_kT, other=0.0)
        if ENABLE_DROPOUT:
            philox_offs = curr_philox_offset + offs_m[:, None
                ] * stride_dropoutm + offs_n[None, :] * stride_dropoutn
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = offs_m[:, None] * stride_dropoutm + offs_n[
                    None, :] * stride_dropoutn
                dropout_mask = tl.load(curr_dropout_offset + dropout_offs,
                    mask=mask_mn)
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1 / (1 - dropout_p)
        if IS_FP8:
            qk = tl.dot(q, kT) * descale_q * descale_k
        else:
            qk = tl.dot(q, kT)
        qk_scaled = qk * sm_scale
        if USE_ALIBI:
            relative_pos_block = offs_m[:, None
                ] + seqlen_k - seqlen_q - offs_n[None, :]
            alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
            qk_scaled += alibi_block
        if DEBUG_TRITON_DETAIL:
            print(f'qk scaled: {qk.shape}\n', qk_scaled)
        if USE_EXP2:
            p = tl.math.exp2(qk_scaled * RCP_LN2 - m * RCP_LN2)
        else:
            p = tl.math.exp(qk_scaled - m)
        if MASK:
            causal_mask = offs_m[:, None] - delta_qk >= offs_n[None, :]
            mask = causal_mask & mask_mn
            p = tl.where(mask, p, 0.0)
        if IS_FP8:
            dp = tl.dot(do, vT) * descale_do * descale_v
        else:
            dp = tl.dot(do, vT)
        if ENABLE_DROPOUT:
            dp = tl.where(dropout_mask, dp, 0.0) * dropout_scale
        delta_i = Di[:, None]
        ds = p * (dp - delta_i)
        if IS_FP8:
            scale_ds, descale_ds = compute_fp8_scaling_factors(ds, FP8_MAX)
            dq += tl.dot((ds * scale_ds).to(kT.type.element_ty), tl.trans(kT)
                ) * descale_ds * descale_k
        else:
            dq += tl.dot(ds.to(kT.type.element_ty), tl.trans(kT))
        curr_n += step_n
        kT_ptrs += step_n * stride_kn
        vT_ptrs += step_n * stride_vn
    return dq


@triton.jit
def _bwd_kernel_dkdv_causal(Q, K, V, sm_scale, DO, DK, DV, M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk, stride_kb, stride_kh,
    stride_kn, stride_kk, stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk, stride_deltab,
    stride_deltah, stride_deltam, stride_dob, stride_doh, stride_dom,
    stride_dok, stride_dropoutb, stride_dropouth, stride_dropoutm,
    stride_dropoutn, stride_descale_q_z, stride_descale_k_z,
    stride_descale_v_z, stride_descale_do_z, stride_az, stride_ah, HQ, HK,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, Dropout_mask,
    dropout_p, philox_seed, philox_offset_base, Alibi_slopes, Descale_q,
    Descale_k, Descale_v, Descale_do, BLOCK_M: tl.constexpr, BLOCK_N: tl.
    constexpr, BLK_SLICE_FACTOR: tl.constexpr, HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr, ENABLE_DROPOUT: tl.constexpr, IS_VARLEN:
    tl.constexpr, USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, IS_FP8:
    tl.constexpr, FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr, DEBUG_TRITON_DETAIL: tl.constexpr):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    delta_qk = seqlen_q - seqlen_k
    if DEBUG_TRITON:
        print(f'\npid: {pid}, bid: {bid}, hkid: {hkid}')
    if DEBUG_TRITON:
        print(f'delta_qk = {delta_qk}')
    start_delta_q_gt_k = delta_qk
    num_blocks_skip = -delta_qk // BLOCK_N
    delta_aligned = (num_blocks_skip + 1) * BLOCK_N + delta_qk
    start_delta_q_lt_k = delta_aligned // BLOCK_M * BLOCK_M
    if delta_qk >= 0:
        start_delta = delta_qk
        if DEBUG_TRITON:
            print(
                f'q >= k: start_delta = delta_qk aligned to BLOCK_M = {start_delta_q_gt_k}'
                )
    else:
        start_delta = start_delta_q_lt_k
        if DEBUG_TRITON:
            print(
                f'q < k: start_delta = residue btw multiple BLOCK_N and delta_qk = {delta_aligned} = aligned to BLOCK_M = {start_delta_q_lt_k}'
                )
    start_n = pid * BLOCK_N
    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_kv &= mask_k[None, :]
    GROUP_SIZE = HQ // HK
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn + offs_n[
        :, None] * stride_kn + offs_k[None, :] * stride_kk
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn + offs_n[
        :, None] * stride_vn + offs_k[None, :] * stride_vk
    k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        if delta_qk >= 0:
            start_m = start_n + start_delta
            len_m = BLOCK_N
        else:
            start_m = max(start_n + delta_qk, 0)
            start_m = start_m // BLOCK_M * BLOCK_M
            residue_m = max(start_n + delta_qk - start_m, 0)
            len_m = BLOCK_N + residue_m
            if DEBUG_TRITON:
                print(f'residue_m = {residue_m}')
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        Q_ptr = Q + adj_q
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        DO_ptr = DO + adj_do
        adj_delta = (bid * stride_deltab + hqid * stride_deltah + q_start *
            stride_deltam)
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta
        if USE_ALIBI:
            alibi_offset = bid * stride_az + hqid * stride_ah
            alibi_slope = tl.load(Alibi_slopes + alibi_offset)
        else:
            alibi_slope = None
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + bid *
                stride_dropoutb + hqid * stride_dropouth)
            dropout_offset = (Dropout_mask + bid * stride_dropoutb + hqid *
                stride_dropouth)
        MASK_BLOCK_M: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
        len_m = min(len_m, seqlen_q)
        num_steps = tl.cdiv(len_m, MASK_BLOCK_M)
        if pid < num_blocks_skip:
            num_steps = 0
        if IS_FP8:
            descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0
        if DEBUG_TRITON:
            print(
                f'Masked: start_n: {start_n}; start_m: {start_m}, num_steps: {num_steps}'
                )
        dk, dv = _bwd_dkdv_inner(dk, dv, Q_ptr, k, v, DO_ptr, M_ptr,
            Delta_ptr, sm_scale, stride_qm, stride_qk, stride_dom,
            stride_dok, stride_dropoutm, stride_dropoutn, stride_deltam,
            MASK_BLOCK_M, BLOCK_N, HEAD_DIM, ACTUAL_HEAD_DIM, dropout_p,
            philox_seed, batch_philox_offset, dropout_offset, alibi_slope,
            seqlen_q, seqlen_k, start_n, start_m, num_steps, descale_q,
            descale_k, descale_v, descale_do, MASK=True, ENABLE_DROPOUT=
            ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, IS_FP8=
            IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
        start_m += num_steps * MASK_BLOCK_M
        num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M)
        end_m = start_m + num_steps * BLOCK_M
        if DEBUG_TRITON:
            print(
                f'start_m after Masked step: {start_m}; num_steps: {num_steps}'
                )
        if DEBUG_TRITON:
            print(
                f'unMasked: start_n: {start_n}, start_m: {start_m}, end_m: {end_m}, num_steps: {num_steps}'
                )
        if DEBUG_TRITON:
            print('unMasked')
        dk, dv = _bwd_dkdv_inner(dk, dv, Q_ptr, k, v, DO_ptr, M_ptr,
            Delta_ptr, sm_scale, stride_qm, stride_qk, stride_dom,
            stride_dok, stride_dropoutm, stride_dropoutn, stride_deltam,
            BLOCK_M, BLOCK_N, HEAD_DIM, ACTUAL_HEAD_DIM, dropout_p,
            philox_seed, batch_philox_offset, dropout_offset, alibi_slope,
            seqlen_q, seqlen_k, start_n, start_m, num_steps, descale_q,
            descale_k, descale_v, descale_do, MASK=False, ENABLE_DROPOUT=
            ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, IS_FP8=
            IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
    adj_dkdv = bid * stride_dkb + hkid * stride_kh + k_start * stride_dkn
    offs_dkdv = offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    tl.store(DV + adj_dkdv + offs_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv + offs_dkdv, dk, mask=mask_kv)


@triton.jit
def _bwd_kernel_dkdv_noncausal(Q, K, V, sm_scale, DO, DK, DV, M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk, stride_kb, stride_kh,
    stride_kn, stride_kk, stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk, stride_deltab,
    stride_deltah, stride_deltam, stride_dob, stride_doh, stride_dom,
    stride_dok, stride_dropoutb, stride_dropouth, stride_dropoutm,
    stride_dropoutn, stride_descale_q_z, stride_descale_k_z,
    stride_descale_v_z, stride_descale_do_z, stride_az, stride_ah, HQ, HK,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, Dropout_mask,
    dropout_p, philox_seed, philox_offset_base, Alibi_slopes, Descale_q,
    Descale_k, Descale_v, Descale_do, BLOCK_M: tl.constexpr, BLOCK_N: tl.
    constexpr, BLK_SLICE_FACTOR: tl.constexpr, HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr, ENABLE_DROPOUT: tl.constexpr, IS_VARLEN:
    tl.constexpr, USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, IS_FP8:
    tl.constexpr, FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr,
    DEBUG_TRITON: tl.constexpr, DEBUG_TRITON_DETAIL: tl.constexpr):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    start_n = pid * BLOCK_N
    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_kv &= mask_k[None, :]
    GROUP_SIZE = HQ // HK
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn + offs_n[
        :, None] * stride_kn + offs_k[None, :] * stride_kk
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn + offs_n[
        :, None] * stride_vn + offs_k[None, :] * stride_vk
    k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        Q_ptr = Q + adj_q
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        DO_ptr = DO + adj_do
        adj_delta = (bid * stride_deltab + hqid * stride_deltah + q_start *
            stride_deltam)
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta
        if USE_ALIBI:
            alibi_offset = bid * stride_az + hqid * stride_ah
            alibi_slope = tl.load(Alibi_slopes + alibi_offset)
        else:
            alibi_slope = None
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + bid *
                stride_dropoutb + hqid * stride_dropouth)
            dropout_offset = (Dropout_mask + bid * stride_dropoutb + hqid *
                stride_dropouth)
        if IS_FP8:
            descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0
        start_m = 0
        num_steps = tl.cdiv(seqlen_q, BLOCK_M)
        dk, dv = _bwd_dkdv_inner(dk, dv, Q_ptr, k, v, DO_ptr, M_ptr,
            Delta_ptr, sm_scale, stride_qm, stride_qk, stride_dom,
            stride_dok, stride_dropoutm, stride_dropoutn, stride_deltam,
            BLOCK_M, BLOCK_N, HEAD_DIM, ACTUAL_HEAD_DIM, dropout_p,
            philox_seed, batch_philox_offset, dropout_offset, alibi_slope,
            seqlen_q, seqlen_k, start_n, start_m, num_steps, descale_q,
            descale_k, descale_v, descale_do, MASK=False, ENABLE_DROPOUT=
            ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, IS_FP8=
            IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
    adj_dkdv = bid * stride_dkb + hkid * stride_kh + k_start * stride_dkn
    offs_dkdv = offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk
    tl.store(DV + adj_dkdv + offs_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv + offs_dkdv, dk, mask=mask_kv)


@triton.jit
def _bwd_kernel_dq_causal(Q, K, V, sm_scale, DO, DQ, M, Delta, stride_qb,
    stride_qh, stride_qm, stride_qk, stride_kb, stride_kh, stride_kn,
    stride_kk, stride_vb, stride_vh, stride_vn, stride_vk, stride_dqb,
    stride_dqh, stride_dqm, stride_dqk, stride_deltab, stride_deltah,
    stride_deltam, stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z,
    stride_descale_do_z, stride_az, stride_ah, HQ, HK, cu_seqlens_q,
    cu_seqlens_k, max_seqlen_q, max_seqlen_k, Dropout_mask, dropout_p,
    philox_seed, philox_offset_base, Alibi_slopes, Descale_q, Descale_k,
    Descale_v, Descale_do, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr, HEAD_DIM: tl.constexpr, ACTUAL_HEAD_DIM:
    tl.constexpr, ENABLE_DROPOUT: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr, DEBUG_TRITON: tl.
    constexpr, DEBUG_TRITON_DETAIL: tl.constexpr):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
    start_m = pid * BLOCK_M
    delta_qk = seqlen_q - seqlen_k
    if DEBUG_TRITON:
        print(
            f'end_n = start_m + BLOCK_M = {start_m} + {BLOCK_M} = {start_m + BLOCK_M}'
            )
    if start_m + BLOCK_M < delta_qk:
        if DEBUG_TRITON:
            print(
                f'start_m + BLOCK_M = {start_m} + {BLOCK_M} = {start_m + BLOCK_M} < delta_qk of {delta_qk}'
                )
        return
    offs_k = tl.arange(0, HEAD_DIM)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_q = offs_m[:, None] < seqlen_q
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_q &= mask_k[None, :]
    offs_q = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    offs_do = offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn
    K += adj_k
    V += adj_v
    GROUP_SIZE = HQ // HK
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        end_n = start_m + BLOCK_M - delta_qk
        end_n = max(min(end_n, seqlen_k), 0)
        if DEBUG_TRITON:
            print(f'delta_qk: {delta_qk}; end_n: {end_n}')
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        adj_delta = (bid * stride_deltab + hqid * stride_deltah + q_start *
            stride_deltam)
        Delta_ptr = Delta + adj_delta
        if USE_ALIBI:
            alibi_offset = bid * stride_az + hqid * stride_ah
            alibi_slope = tl.load(Alibi_slopes + alibi_offset)
        else:
            alibi_slope = None
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + bid *
                stride_dropoutb + hqid * stride_dropouth)
            dropout_offset = (Dropout_mask + bid * stride_dropoutb + hqid *
                stride_dropouth)
        q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
        do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
        m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=offs_m <
            seqlen_q)
        m = m[:, None]
        MASK_BLOCK_N: tl.constexpr = BLOCK_N // BLK_SLICE_FACTOR
        start_n = max(end_n - BLOCK_M, 0)
        num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N)
        if IS_FP8:
            descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0
        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        if DEBUG_TRITON:
            print(f'pid: {pid}; end_n: {end_n}, start_m: {start_m}')
        if DEBUG_TRITON:
            print(
                f'Masked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}'
                )
        dq = _bwd_dq_inner(dq, q, K, V, do, m, Delta_ptr, sm_scale,
            stride_qm, stride_qk, stride_kn, stride_kk, stride_vn,
            stride_vk, stride_dropoutm, stride_dropoutn, stride_deltam,
            seqlen_q, seqlen_k, BLOCK_M, MASK_BLOCK_N, HEAD_DIM,
            ACTUAL_HEAD_DIM, dropout_p, philox_seed, batch_philox_offset,
            dropout_offset, alibi_slope, start_m, start_n, end_n, num_steps,
            descale_q, descale_k, descale_v, descale_do, MASK=True,
            ENABLE_DROPOUT=ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=
            USE_EXP2, IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=
            DEBUG_TRITON, DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
        end_n -= num_steps * MASK_BLOCK_N
        num_steps = tl.cdiv(end_n, BLOCK_N)
        start_n = max(end_n - num_steps * BLOCK_N, 0)
        if DEBUG_TRITON:
            print(
                f'unMasked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}'
                )
        dq = _bwd_dq_inner(dq, q, K, V, do, m, Delta_ptr, sm_scale,
            stride_qm, stride_qk, stride_kn, stride_kk, stride_vn,
            stride_vk, stride_dropoutm, stride_dropoutn, stride_deltam,
            seqlen_q, seqlen_k, BLOCK_M, BLOCK_N, HEAD_DIM, ACTUAL_HEAD_DIM,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            alibi_slope, start_m, start_n, end_n, num_steps, descale_q,
            descale_k, descale_v, descale_do, MASK=False, ENABLE_DROPOUT=
            ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, IS_FP8=
            IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
        adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
        offs_dq = offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
        dq *= sm_scale
        tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)


@triton.jit
def _bwd_kernel_dq_noncausal(Q, K, V, sm_scale, DO, DQ, M, Delta, stride_qb,
    stride_qh, stride_qm, stride_qk, stride_kb, stride_kh, stride_kn,
    stride_kk, stride_vb, stride_vh, stride_vn, stride_vk, stride_dqb,
    stride_dqh, stride_dqm, stride_dqk, stride_deltab, stride_deltah,
    stride_deltam, stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    stride_descale_q_z, stride_descale_k_z, stride_descale_v_z,
    stride_descale_do_z, stride_az, stride_ah, HQ, HK, cu_seqlens_q,
    cu_seqlens_k, max_seqlen_q, max_seqlen_k, Dropout_mask, dropout_p,
    philox_seed, philox_offset_base, Alibi_slopes, Descale_q, Descale_k,
    Descale_v, Descale_do, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr, HEAD_DIM: tl.constexpr, ACTUAL_HEAD_DIM:
    tl.constexpr, ENABLE_DROPOUT: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr, DEBUG_TRITON: tl.
    constexpr, DEBUG_TRITON_DETAIL: tl.constexpr):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
    start_m = pid * BLOCK_M
    offs_k = tl.arange(0, HEAD_DIM)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_q = offs_m[:, None] < seqlen_q
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_q &= mask_k[None, :]
    offs_q = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    offs_do = offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn
    K += adj_k
    V += adj_v
    GROUP_SIZE = HQ // HK
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        adj_delta = (bid * stride_deltab + hqid * stride_deltah + q_start *
            stride_deltam)
        Delta_ptr = Delta + adj_delta
        if USE_ALIBI:
            alibi_offset = bid * stride_az + hqid * stride_ah
            alibi_slope = tl.load(Alibi_slopes + alibi_offset)
        else:
            alibi_slope = None
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset_base + bid *
                stride_dropoutb + hqid * stride_dropouth)
            dropout_offset = (Dropout_mask + bid * stride_dropoutb + hqid *
                stride_dropouth)
        q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
        do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
        m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=offs_m <
            seqlen_q)
        m = m[:, None]
        if IS_FP8:
            descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid)
            descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid)
            descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid)
            descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0
        start_n = 0
        end_n = seqlen_k
        num_steps = tl.cdiv(seqlen_k, BLOCK_N)
        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        dq = _bwd_dq_inner(dq, q, K, V, do, m, Delta_ptr, sm_scale,
            stride_qm, stride_qk, stride_kn, stride_kk, stride_vn,
            stride_vk, stride_dropoutm, stride_dropoutn, stride_deltam,
            seqlen_q, seqlen_k, BLOCK_M, BLOCK_N, HEAD_DIM, ACTUAL_HEAD_DIM,
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,
            alibi_slope, start_m, start_n, end_n, num_steps, descale_q,
            descale_k, descale_v, descale_do, MASK=False, ENABLE_DROPOUT=
            ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2, IS_FP8=
            IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=DEBUG_TRITON,
            DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
        adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
        offs_dq = offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk
        dq *= sm_scale
        tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)


@triton.jit
def _bwd_preprocess(O, DO, Delta, stride_ob, stride_oh, stride_om,
    stride_ok, stride_deltab, stride_deltah, stride_deltam,
    stride_descale_do_z, cu_seqlens_q, max_seqlen_q, Descale_do, BLOCK_M:
    tl.constexpr, HEAD_DIM: tl.constexpr, ACTUAL_HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr, IS_FP8: tl.constexpr):
    pid_m = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    q_start = 0
    seqlen_q = max_seqlen_q
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        seqlen_q = q_end - q_start
    else:
        q_start = 0
        seqlen_q = max_seqlen_q
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    O += bid * stride_ob + hid * stride_oh + q_start * stride_om
    DO += bid * stride_ob + hid * stride_oh + q_start * stride_om
    mask_m = offs_m < seqlen_q
    mask_md = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    if PADDED_HEAD:
        mask_md &= offs_k[None, :] < ACTUAL_HEAD_DIM
    offs_do = offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_ptrs = O + offs_do
    do_ptrs = DO + offs_do
    o = tl.load(out_ptrs, mask=mask_md, other=0.0)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0)
    if IS_FP8:
        descale_do = tl.load(Descale_do + bid * stride_descale_do_z + hid)
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do),
            axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    delta_offset = (Delta + bid * stride_deltab + hid * stride_deltah + 
        q_start * stride_deltam)
    tl.store(delta_offset + offs_m * stride_deltam, delta, mask=mask_m)


def attention_prefill_backward_triton_split_impl(do: torch.Tensor, q: torch
    .Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, softmax_lse:
    torch.Tensor, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
    sm_scale: float, alibi_slopes: Optional[torch.Tensor], causal: bool,
    layout: Literal['bshd', 'bhsd', 'thd'], cu_seqlens_q: Optional[torch.
    Tensor], cu_seqlens_k: Optional[torch.Tensor], max_seqlen_q: Optional[
    int], max_seqlen_k: Optional[int], dropout_p: float, philox_seed:
    Optional[int], philox_offset: Optional[int], use_exp2: bool, descale_q:
    Optional[torch.Tensor], descale_k: Optional[torch.Tensor], descale_v:
    Optional[torch.Tensor], descale_o: Optional[torch.Tensor], descale_do:
    Optional[torch.Tensor], descale_dq: Optional[torch.Tensor], descale_dk:
    Optional[torch.Tensor], descale_dv: Optional[torch.Tensor]):
    DEBUG_TRITON: bool = False
    DEBUG_TRITON_DETAIL: bool = False
    IS_FP8 = is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        assert is_fp8(do) and is_fp8(q) and is_fp8(k) and is_fp8(v
            ), f'Non fp8 type found: do.dtype={do.dtype}, q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}. All tensors must be fp8.'
        if is_fp8(o):
            FP8_OUTPUT = True
            assert descale_o is not None, f'descale_o is None. In fp8, you need to pass a tensor for descale_o along with a tensor o.'
            assert descale_dq is not None, f'descale_dq is None. In fp8, you need to pass a tensor for descale_dq along with a tensor dq.'
            assert descale_dk is not None, f'descale_dk is None. In fp8, you need to pass a tensor for descale_dk along with a tensor dk.'
            assert descale_dv is not None, f'descale_dv is None. In fp8, you need to pass a tensor for descale_dv along with a tensor dv.'
        else:
            FP8_OUTPUT = False
        stride_descale_q_z = descale_q.stride(0
            ) if descale_q is not None else None
        stride_descale_k_z = descale_k.stride(0
            ) if descale_k is not None else None
        stride_descale_v_z = descale_v.stride(0
            ) if descale_v is not None else None
        stride_descale_o_z = descale_o.stride(0
            ) if descale_o is not None else None
        stride_descale_do_z = descale_do.stride(0
            ) if descale_do is not None else None
    else:
        FP8_MAX = None
        FP8_OUTPUT = False
        (stride_descale_q_z) = (stride_descale_k_z) = (stride_descale_v_z) = (
            stride_descale_o_z) = (stride_descale_do_z) = None
    (batch, nheads_q, nheads_k, head_size, max_seqlen_q_final,
        max_seqlen_k_final) = (get_shapes_from_layout(q, k, layout,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k))
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q,
        k, v, o, layout)
    stride_qb, stride_qh, stride_qm, stride_qk = q_strides
    stride_kb, stride_kh, stride_kn, stride_kk = k_strides
    stride_vb, stride_vh, stride_vn, stride_vk = v_strides
    stride_ob, stride_oh, stride_om, stride_ok = o_strides
    dq_strides, dk_strides, dv_strides, do_strides = get_strides_from_layout(dq
        , dk, dv, do, layout)
    stride_dqb, stride_dqh, stride_dqm, stride_dqk = dq_strides
    stride_dkb, stride_dkh, stride_dkn, stride_dkk = dk_strides
    stride_dvb, stride_dvh, stride_dvn, stride_dvk = dv_strides
    stride_dob, stride_doh, stride_dom, stride_dok = do_strides
    IS_VARLEN = layout == 'thd'
    use_dropout = dropout_p > 0.0
    use_alibi, (stride_az, stride_ah) = (True, alibi_slopes.stride()
        ) if alibi_slopes is not None else (False, (0, 0))
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 32)
    HEAD_DIM = padded_d_model
    ACTUAL_HEAD_DIM = head_size
    NUM_WARPS, NUM_STAGES = 4, 1
    WAVES_PER_EU = 1
    PRE_BLOCK = 128
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        stride_deltab = 0
        stride_deltah, stride_deltam = delta.stride()
    else:
        stride_deltab, stride_deltah, stride_deltam = delta.stride()
    pre_grid = triton.cdiv(max_seqlen_q_final, PRE_BLOCK), batch, nheads_q
    _bwd_preprocess[pre_grid](o, do, delta, stride_ob, stride_oh, stride_om,
        stride_ok, stride_deltab, stride_deltah, stride_deltam,
        stride_descale_do_z, cu_seqlens_q, max_seqlen_q_final, descale_do,
        BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM, ACTUAL_HEAD_DIM=
        ACTUAL_HEAD_DIM, IS_VARLEN=IS_VARLEN, IS_FP8=IS_FP8)
    if DEBUG:
        print('delta:', delta, delta.shape)
    dropout_mask = None
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = (0,
        0, 0, 0)
    if use_dropout:
        dropout_mask = torch.zeros((batch, nheads_q, max_seqlen_q_final,
            max_seqlen_k_final), device=q.device, dtype=torch.float32)
        if DROPOUT_USE_PYTORCH:
            if not IS_VARLEN:
                dropout_mask = create_dropout_mask(dropout_p, (batch,
                    nheads_q, max_seqlen_q_final, max_seqlen_k_final), seed
                    =philox_seed)
            else:
                dropout_mask = create_dropout_mask_varlen(dropout_p, batch,
                    nheads_q, cu_seqlens_q, cu_seqlens_k, philox_seed)
        (stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn
            ) = dropout_mask.stride()
    grid_dkdv = (max_seqlen_k_final + BLOCK_N1 - 1
        ) // BLOCK_N1, batch, nheads_k
    grid_dq = (max_seqlen_q_final + BLOCK_M2 - 1) // BLOCK_M2, batch, nheads_k
    if causal:
        if DEBUG_TRITON:
            print(
                f'_bwd_kernel_dkdv: grid = {grid_dkdv}, block_size = ({BLOCK_M1, BLOCK_N1})'
                )
        _bwd_kernel_dkdv_causal[grid_dkdv](q, k, v, sm_scale, do, dk, dv,
            softmax_lse, delta, stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk, stride_vb,
            stride_vh, stride_vn, stride_vk, stride_dkb, stride_dkh,
            stride_dkn, stride_dkk, stride_deltab, stride_deltah,
            stride_deltam, stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm,
            stride_dropoutn, stride_descale_q_z, stride_descale_k_z,
            stride_descale_v_z, stride_descale_do_z, stride_az, stride_ah,
            nheads_q, nheads_k, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q_final, max_seqlen_k_final, dropout_mask, dropout_p,
            philox_seed, philox_offset, alibi_slopes, descale_q, descale_k,
            descale_v, descale_do, BLOCK_M1, BLOCK_N1, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM, ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN, USE_ALIBI=use_alibi, USE_EXP2=use_exp2,
            IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=FP8_OUTPUT,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES, waves_per_eu=
            WAVES_PER_EU, DEBUG_TRITON=DEBUG_TRITON, DEBUG_TRITON_DETAIL=
            DEBUG_TRITON_DETAIL)
        if DEBUG_TRITON:
            print(
                f'\n_bwd_kernel_dq: grid = {grid_dq}, block_size = ({BLOCK_M2, BLOCK_N2})'
                )
        _bwd_kernel_dq_causal[grid_dq](q, k, v, sm_scale, do, dq,
            softmax_lse, delta, stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk, stride_vb,
            stride_vh, stride_vn, stride_vk, stride_dqb, stride_dqh,
            stride_dqm, stride_dqk, stride_deltab, stride_deltah,
            stride_deltam, stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm,
            stride_dropoutn, stride_descale_q_z, stride_descale_k_z,
            stride_descale_v_z, stride_descale_do_z, stride_az, stride_ah,
            nheads_q, nheads_k, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q_final, max_seqlen_k_final, dropout_mask, dropout_p,
            philox_seed, philox_offset, alibi_slopes, descale_q, descale_k,
            descale_v, descale_do, BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM, ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN, USE_ALIBI=use_alibi, USE_EXP2=use_exp2,
            IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=FP8_OUTPUT,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES, waves_per_eu=
            WAVES_PER_EU, DEBUG_TRITON=DEBUG_TRITON, DEBUG_TRITON_DETAIL=
            DEBUG_TRITON_DETAIL)
    else:
        _bwd_kernel_dkdv_noncausal[grid_dkdv](q, k, v, sm_scale, do, dk, dv,
            softmax_lse, delta, stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk, stride_vb,
            stride_vh, stride_vn, stride_vk, stride_dkb, stride_dkh,
            stride_dkn, stride_dkk, stride_deltab, stride_deltah,
            stride_deltam, stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm,
            stride_dropoutn, stride_descale_q_z, stride_descale_k_z,
            stride_descale_v_z, stride_descale_do_z, stride_az, stride_ah,
            nheads_q, nheads_k, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q_final, max_seqlen_k_final, dropout_mask, dropout_p,
            philox_seed, philox_offset, alibi_slopes, descale_q, descale_k,
            descale_v, descale_do, BLOCK_M1, BLOCK_N1, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM, ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN, USE_ALIBI=use_alibi, USE_EXP2=use_exp2,
            IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=FP8_OUTPUT,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES, waves_per_eu=
            WAVES_PER_EU, DEBUG_TRITON=DEBUG_TRITON, DEBUG_TRITON_DETAIL=
            DEBUG_TRITON_DETAIL)
        _bwd_kernel_dq_noncausal[grid_dq](q, k, v, sm_scale, do, dq,
            softmax_lse, delta, stride_qb, stride_qh, stride_qm, stride_qk,
            stride_kb, stride_kh, stride_kn, stride_kk, stride_vb,
            stride_vh, stride_vn, stride_vk, stride_dqb, stride_dqh,
            stride_dqm, stride_dqk, stride_deltab, stride_deltah,
            stride_deltam, stride_dob, stride_doh, stride_dom, stride_dok,
            stride_dropoutb, stride_dropouth, stride_dropoutm,
            stride_dropoutn, stride_descale_q_z, stride_descale_k_z,
            stride_descale_v_z, stride_descale_do_z, stride_az, stride_ah,
            nheads_q, nheads_k, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q_final, max_seqlen_k_final, dropout_mask, dropout_p,
            philox_seed, philox_offset, alibi_slopes, descale_q, descale_k,
            descale_v, descale_do, BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR,
            HEAD_DIM, ACTUAL_HEAD_DIM, ENABLE_DROPOUT=use_dropout,
            IS_VARLEN=IS_VARLEN, USE_ALIBI=use_alibi, USE_EXP2=use_exp2,
            IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=FP8_OUTPUT,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES, waves_per_eu=
            WAVES_PER_EU, DEBUG_TRITON=DEBUG_TRITON, DEBUG_TRITON_DETAIL=
            DEBUG_TRITON_DETAIL)
    return delta


def attention_backward_core_ref_impl(do, q, k, v, o, softmax_lse, sm_scale,
    causal, dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2):
    if DEBUG_CORE:
        print()
        print('attention_backward_core_ref_impl')
        print('do:', do, do.shape)
        print('q:', q, q.shape)
        print('k:', k, k.shape)
        print('v:', v, v.shape)
        print('o:', o, o.shape)
        print('softmax_lse:', softmax_lse, softmax_lse.shape)
        print('sm_scale:', sm_scale)
        print('causal:', causal)
        print('dropout_p:', dropout_p)
        print('philox_seed:', philox_seed)
        print('philox_offset:', philox_offset)
        print('use_exp2:', use_exp2)
    do = do.to(torch.float32)
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    o = o.to(torch.float32)
    softmax_lse = softmax_lse.to(torch.float32)
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    if DEBUG_CORE:
        print('attention_scores:', attention_scores, attention_scores.shape)
    attention_scaled_scores = sm_scale * attention_scores
    if DEBUG_CORE:
        print('attention_scaled_scores:', attention_scaled_scores,
            attention_scaled_scores.shape)
    if alibi_slopes is not None:
        L_q, L_k = q.shape[1], k.shape[1]
        if DEBUG_CORE:
            print('alibi_slopes:', alibi_slopes, alibi_slopes.shape)
        alibi_bias = compute_alibi_tensor_ref(alibi_slopes, L_q, L_k)
        alibi_bias = alibi_bias.reshape(-1, L_q, L_k)
        if True:
            print('alibi_bias:', alibi_bias, alibi_bias.shape)
        attention_scaled_scores = attention_scaled_scores + alibi_bias
        if DEBUG_CORE:
            print('attention_scaled_scores after alibi:',
                attention_scaled_scores, attention_scaled_scores.shape)
    if causal:
        L_q, L_k = q.shape[1], k.shape[1]
        row_idx = torch.arange(L_q, device=q.device).unsqueeze(1)
        col_idx = torch.arange(L_k, device=q.device).unsqueeze(0)
        col_offset = L_q - L_k
        causal_mask = row_idx >= col_offset + col_idx
        if DEBUG_CORE:
            print('causal_mask:', causal_mask)
        attention_scaled_scores = attention_scaled_scores.masked_fill(torch
            .logical_not(causal_mask.unsqueeze(0)), float('-inf'))
        if DEBUG_CORE:
            print('attention_scaled_scores after causal:',
                attention_scaled_scores, attention_scaled_scores.shape)
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        attention_scaled_scores_base2 = attention_scaled_scores * RCP_LN
        softmax_lse_base2 = softmax_lse * RCP_LN
        softmax_lse_3d = softmax_lse_base2.unsqueeze(-1)
        p = torch.exp2(attention_scaled_scores_base2 - softmax_lse_3d)
    else:
        softmax_lse_3d = softmax_lse.unsqueeze(-1)
        p = torch.exp(attention_scaled_scores - softmax_lse_3d)
    if DEBUG_CORE:
        print('softmax_lse_3d:', softmax_lse_3d, softmax_lse_3d.shape)
        print('p:', p, p.shape)
    if dropout_p > 0.0:
        rand_vals = torch.rand(p.shape, generator=torch.Generator(device=p.
            device).manual_seed(philox_seed), device=p.device, dtype=p.dtype)
        dropout_mask, dropout_scale = rand_vals > dropout_p, 1.0 / (1 -
            dropout_p)
        if DEBUG:
            print('dropout_scale:', dropout_scale)
            print('dropout_mask:', dropout_mask)
        p_drop = torch.where(dropout_mask, p, torch.zeros_like(p))
        p_drop_scaled = p_drop * dropout_scale
        if DEBUG_CORE:
            print('dropout_scale:', dropout_scale)
            print('p_drop:', p_drop, p_drop.shape)
            print('p_drop_scaled:', p_drop_scaled, p_drop_scaled.shape)
        dv = torch.matmul(p_drop_scaled.transpose(-2, -1), do)
        if DEBUG_CORE:
            print('dv:', dv, dv.shape)
        dp_dropout = torch.matmul(do, v.transpose(-2, -1))
        dp = torch.where(dropout_mask, dp_dropout, torch.zeros_like(dp_dropout)
            ) * dropout_scale
        if DEBUG_CORE:
            print('dp_dropout:', dp_dropout, dp_dropout.shape)
            print('dp:', dp, dp.shape)
    else:
        dv = torch.matmul(p.transpose(-2, -1), do)
        if DEBUG_CORE:
            print('dv:', dv, dv.shape)
        dp = torch.matmul(do, v.transpose(-2, -1))
        if DEBUG_CORE:
            print('dp:', dp, dp.shape)
    if False:
        delta = torch.sum(o * do, axis=-1).unsqueeze(-1)
    else:
        delta = torch.sum(p * dp, axis=-1).unsqueeze(-1)
    if DEBUG:
        print('delta:', delta, delta.shape)
    dscores_scaled = p * (dp - delta)
    ds = dscores_scaled * sm_scale
    if DEBUG_CORE:
        print('dscores_scaled:', dscores_scaled, dscores_scaled.shape)
        print('ds:', ds, ds.shape)
    dk = torch.matmul(ds.transpose(-2, -1), q)
    dq = torch.matmul(ds, k)
    if DEBUG_CORE:
        print('dk:', dk, dk.shape)
        print('dq:', dq, dq.shape)
    dq = dq.to(torch.float16)
    dk = dk.to(torch.float16)
    dv = dv.to(torch.float16)
    delta = delta.squeeze(-1)
    if DEBUG_CORE:
        print('attention_backward_core_ref_impl output')
        print('delta:', delta, delta.shape)
        print('dv:', dv, dv.shape)
        print('dk:', dk, dk.shape)
        print('dq:', dq, dq.shape)
    return dq, dk, dv, delta


def attention_backward_pytorch_ref_impl(do: torch.Tensor, q: torch.Tensor,
    k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, softmax_lse: torch.
    Tensor, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, sm_scale:
    float, alibi_slopes: Optional[torch.Tensor], causal: bool, layout:
    Literal['bshd', 'bhsd', 'thd'], cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor], max_seqlen_q: Optional[int],
    max_seqlen_k: Optional[int], dropout_p: float, philox_seed: Optional[
    int], philox_offset: Optional[int], use_exp2: bool):
    if layout == 'thd':
        dq_ref, dk_ref, dv_ref, delta = (
            attention_varlen_backward_pytorch_ref_impl(do, q, k, v, o,
            softmax_lse, sm_scale, causal, layout, cu_seqlens_q,
            cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
            philox_seed, philox_offset, alibi_slopes, use_exp2))
    else:
        dq_ref, dk_ref, dv_ref, delta = (
            attention_vanilla_backward_pytorch_ref_impl(do, q, k, v, o,
            softmax_lse, sm_scale, causal, layout, dropout_p, philox_seed,
            philox_offset, alibi_slopes, use_exp2))
    dv.copy_(dv_ref.to(dv.dtype))
    dk.copy_(dk_ref.to(dk.dtype))
    dq.copy_(dq_ref.to(dq.dtype))
    return delta


def attention_vanilla_backward_pytorch_ref_impl(do, q, k, v, o, softmax_lse,
    sm_scale, causal, layout, dropout_p, philox_seed, philox_offset,
    alibi_slopes, use_exp2):
    if layout == 'bshd':
        if DEBUG:
            print()
            print('Changing layout to bhsd!')
        do = do.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
    elif layout == 'bhsd':
        pass
    else:
        raise ValueError(f'Unknown layout {layout}')
    batch_size, nheads_q, seq_len_q, head_dim = q.shape
    batch_size, nheads_k, seq_len_k, head_dim = k.shape
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError('nheads_q must be divisible by nheads_k')
    if group_size != 1:
        do = do.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        q = q.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        o = o.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size, nheads_k, group_size,
            seq_len_q)
        k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        do = do.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim
            )
        q = q.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k * group_size, seq_len_k, head_dim)
        o = o.reshape(batch_size * nheads_k * group_size, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size * nheads_k *
            group_size, seq_len_q)
    else:
        do = do.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        q = q.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        k = k.reshape(batch_size * nheads_k, seq_len_k, head_dim)
        v = v.reshape(batch_size * nheads_k, seq_len_k, head_dim)
        o = o.reshape(batch_size * nheads_q, seq_len_q, head_dim)
        softmax_lse = softmax_lse.reshape(batch_size * nheads_q, seq_len_q)
    dq, dk, dv, delta = attention_backward_core_ref_impl(do, q, k, v, o,
        softmax_lse, sm_scale, causal, dropout_p, philox_seed,
        philox_offset, alibi_slopes, use_exp2)
    if group_size != 1:
        dq = dq.reshape(batch_size, nheads_k, group_size, seq_len_q, head_dim)
        delta = delta.reshape(batch_size, nheads_k, group_size, seq_len_q)
        dk = dk.reshape(batch_size, nheads_k, group_size, seq_len_k, head_dim)
        dk = dk.sum(dim=2)
        dv = dv.reshape(batch_size, nheads_k, group_size, seq_len_k, head_dim)
        dv = dv.sum(dim=2)
        dq = dq.reshape(batch_size, nheads_k * group_size, seq_len_q, head_dim)
        delta = delta.reshape(batch_size, nheads_k * group_size, seq_len_q)
    else:
        dq = dq.reshape(batch_size, nheads_q, seq_len_q, head_dim)
        dk = dk.reshape(batch_size, nheads_k, seq_len_k, head_dim)
        dv = dv.reshape(batch_size, nheads_k, seq_len_k, head_dim)
        delta = delta.reshape(batch_size, nheads_q, seq_len_q)
    if layout == 'bshd':
        if DEBUG:
            print()
            print('Changing back to bshd!')
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
    elif layout == 'bhsd':
        pass
    else:
        raise ValueError(f'Unknown layout {layout}')
    return dq, dk, dv, delta


def attention_varlen_backward_pytorch_ref_impl(do, q, k, v, o, softmax_lse,
    sm_scale, causal, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k, dropout_p, philox_seed, philox_offset, alibi_slopes, use_exp2
    ):
    if layout != 'thd':
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")
    batch_size = cu_seqlens_q.shape[0] - 1
    nheads_q, head_dim = q.shape[1], q.shape[2]
    nheads_k = k.shape[1]
    group_size = nheads_q // nheads_k
    if nheads_q % nheads_k != 0:
        raise ValueError('nheads_q must be divisible by nheads_k')
    total_L_q = q.shape[0]
    total_L_k = k.shape[0]
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = torch.zeros((total_L_q, nheads_q), dtype=torch.float32, device=
        o.device)
    for i in range(batch_size):
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()
        q_i = q[start_q:end_q, :, :]
        k_i = k[start_k:end_k, :, :]
        v_i = v[start_k:end_k, :, :]
        do_i = do[start_q:end_q, :, :]
        o_i = o[start_q:end_q, :, :]
        softmax_lse_i = softmax_lse[start_q:end_q, :]
        if group_size != 1:
            q_i = q_i.view(q_i.shape[0], nheads_k, group_size, head_dim)
            do_i = do_i.view(do_i.shape[0], nheads_k, group_size, head_dim)
            o_i = o_i.view(o_i.shape[0], nheads_k, group_size, head_dim)
            softmax_lse_i = softmax_lse_i.view(softmax_lse_i.shape[0],
                nheads_k, group_size)
            k_i = k_i.unsqueeze(2).expand(-1, -1, group_size, -1)
            v_i = v_i.unsqueeze(2).expand(-1, -1, group_size, -1)
            q_i = q_i.reshape(q_i.shape[0], nheads_k * group_size, head_dim)
            do_i = do_i.reshape(do_i.shape[0], nheads_k * group_size, head_dim)
            o_i = o_i.reshape(o_i.shape[0], nheads_k * group_size, head_dim)
            softmax_lse_i = softmax_lse_i.reshape(softmax_lse_i.shape[0], 
                nheads_k * group_size)
            k_i = k_i.reshape(k_i.shape[0], nheads_k * group_size, head_dim)
            v_i = v_i.reshape(v_i.shape[0], nheads_k * group_size, head_dim)
        q_i = q_i.permute(1, 0, 2)
        k_i = k_i.permute(1, 0, 2)
        v_i = v_i.permute(1, 0, 2)
        do_i = do_i.permute(1, 0, 2)
        o_i = o_i.permute(1, 0, 2)
        softmax_lse_i = softmax_lse_i.transpose(0, 1)
        if alibi_slopes is not None:
            alibi_slopes_i = alibi_slopes[i]
        else:
            alibi_slopes_i = None
        dq_i, dk_i, dv_i, delta_i = attention_backward_core_ref_impl(do_i,
            q_i, k_i, v_i, o_i, softmax_lse_i, sm_scale, causal, dropout_p,
            philox_seed, philox_offset, alibi_slopes_i, use_exp2)
        dq_i = dq_i.permute(1, 0, 2)
        dk_i = dk_i.permute(1, 0, 2)
        dv_i = dv_i.permute(1, 0, 2)
        delta_i = delta_i.transpose(1, 0)
        if group_size != 1:
            dq_i = dq_i.view(dq_i.shape[0], nheads_k, group_size, head_dim)
            delta_i = delta_i.view(delta_i.shape[0], nheads_k, group_size)
            dk_i = dk_i.view(dk_i.shape[0], nheads_k, group_size, head_dim)
            dv_i = dv_i.view(dv_i.shape[0], nheads_k, group_size, head_dim)
            dk_i = dk_i.sum(dim=2)
            dv_i = dv_i.sum(dim=2)
            dq_i = dq_i.reshape(dq_i.shape[0], nheads_q, head_dim)
            delta_i = delta_i.reshape(delta_i.shape[0], nheads_q)
        else:
            pass
        dq[start_q:end_q, :, :] = dq_i
        dk[start_k:end_k, :, :] += dk_i
        dv[start_k:end_k, :, :] += dv_i
        delta[start_q:end_q, :] = delta_i
    return dq, dk, dv, delta


def varlen_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v:
    torch.Tensor, out: torch.Tensor, softmax_lse: torch.Tensor, dq:
    Optional[torch.Tensor], dk: Optional[torch.Tensor], dv: Optional[torch.
    Tensor], cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor], max_seqlen_q: int, max_seqlen_k:
    int, dropout_p: float, softmax_scale: float, zero_tensors: bool, causal:
    bool, window_size_left: int, window_size_right: int, softcap: float,
    deterministic: bool, gen_: Optional[torch.Tensor]=None, rng_state:
    Optional[torch.Tensor]=None, descale_q: Optional[torch.Tensor]=None,
    descale_k: Optional[torch.Tensor]=None, descale_v: Optional[torch.
    Tensor]=None, descale_o: Optional[torch.Tensor]=None, descale_do:
    Optional[torch.Tensor]=None, descale_dq: Optional[torch.Tensor]=None,
    descale_dk: Optional[torch.Tensor]=None, descale_dv: Optional[torch.
    Tensor]=None):
    if DEBUG:
        print()
        print('varlen_bwd')
        print('dout:', dout, dout.shape)
        print('q:', q, q.shape)
        print('k:', k, k.shape)
        print('v:', v, v.shape)
        print('out:', out)
        print('softmax_lse:', softmax_lse, softmax_lse.shape)
        print('dq:', dq, dq.shape if dq is not None else None)
        print('dk:', dk, dk.shape if dk is not None else None)
        print('dv:', dv, dv.shape if dv is not None else None)
        print('cu_seqlens_q:', cu_seqlens_q, cu_seqlens_q.shape)
        print('cu_seqlens_k:', cu_seqlens_k, cu_seqlens_k.shape)
        print('alibi_slopes:', alibi_slopes)
        print('max_seqlen_q:', max_seqlen_q)
        print('max_seqlen_k:', max_seqlen_k)
        print('dropout_p:', dropout_p)
        print('softmax_scale:', softmax_scale)
        print('causal:', causal)
        print('window_size_left:', window_size_left)
        print('window_size_right:', window_size_right)
        print('deterministic:', deterministic)
        print('gen_:', gen_)
        print('rng_state:', rng_state)
        print('descale_q:', descale_q, descale_q.shape if descale_q is not
            None else None)
        print('descale_k:', descale_k, descale_k.shape if descale_k is not
            None else None)
        print('descale_v:', descale_v, descale_v.shape if descale_v is not
            None else None)
        print('descale_do:', descale_do, descale_do.shape if descale_do else
            None)
    dq = torch.zeros_like(q) if dq is None else dq.zero_()
    dk = torch.zeros_like(k) if dk is None else dk.zero_()
    dv = torch.zeros_like(v) if dv is None else dv.zero_()
    if rng_state is not None:
        philox_seed, philox_offset = rng_state[0].item(), rng_state[1].item()
    else:
        philox_seed, philox_offset = None, None
    if USE_REF:
        if DEBUG:
            print('Using reference implementation')
        delta_ref = attention_backward_pytorch_ref_impl(dout, q, k, v, out,
            softmax_lse, dq, dk, dv, softmax_scale, alibi_slopes, causal,
            'thd', cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, philox_seed, philox_offset, False)
        delta = delta_ref
    else:
        if DEBUG:
            print('Using Triton implementation')
        delta_triton = attention_prefill_backward_triton_split_impl(dout, q,
            k, v, out, softmax_lse, dq, dk, dv, softmax_scale, alibi_slopes,
            causal, 'thd', cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
            max_seqlen_k, dropout_p, philox_seed, philox_offset, False,
            descale_q, descale_k, descale_v, descale_o, descale_do,
            descale_dq, descale_dk, descale_dv)
        delta = delta_triton
    if DEBUG:
        print('varlen_bwd outputs')
        print('delta:', delta, delta.shape)
        print('dv:', dv, dv.shape)
        print('dk:', dk, dk.shape)
        print('dq:', dq, dq.shape)
    return dq, dk, dv, delta


def create_dropout_mask_varlen(dropout_p, batch, nheads_q, cu_seqlens_q,
    cu_seqlens_k, philox_seed):
    device = 'cuda'
    qlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    klens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    max_qlen = qlens.max()
    max_klen = klens.max()
    dropout_mask = torch.zeros((batch, nheads_q, max_qlen, max_klen),
        device=device)
    for b in range(batch):
        qlen = qlens[b]
        klen = klens[b]
        rand_vals = torch.rand((nheads_q, qlen, klen), generator=torch.
            Generator(device=device).manual_seed(philox_seed), device=
            device, dtype=torch.float32)
        submask = rand_vals > dropout_p
        dropout_mask[b, :, :qlen, :klen] = submask
    return dropout_mask


# Backward method (kernel launch code)
def _FlashAttnVarlenQKVPackedFP8Func_backward(ctx, dout, *args):
    (q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, cu_seqlens, rng_state,
        descale_q, descale_k, descale_v, descale_o, descale_do
        ) = ctx.saved_tensors
    qkv_shape = q_fp8.shape[:-2] + (3, *q_fp8.shape[-2:])
    head_size_og = dout.size(2)
    dout_padded = dout
    if head_size_og % 8 != 0:
        dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
    if is_fp8(dout_padded):
        raise ValueError(
            'fp8 input and out not supported yet for this function.')
        assert descale_do is not None, f'You need to pass descale factors for do'
        dout_padded_fp8 = dout_padded
        dqkv, descale_dqkv = torch.zeros(qkv_shape, device=q_fp8.device
            ), torch.zeros_like(descale_q)
    else:
        assert descale_do is None, f'Found {dout.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired.'
        dout_padded_fp8, descale_do = cast_to_fp8(dout_padded, torch.
            float8_e4m3fnuz, 'thd', cu_seqlens=cu_seqlens, max_seqlen=ctx.
            max_seqlen)
        dqkv, descale_dqkv = torch.zeros(qkv_shape, dtype=torch.float32,
            device=q_fp8.device), None
    dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8 = [maybe_contiguous(x) for
        x in (dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8)]
    flash_attn_gpu.varlen_bwd(dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8,
        softmax_lse, dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens,
        cu_seqlens, ctx.alibi_slopes, ctx.max_seqlen, ctx.max_seqlen, ctx.
        dropout_p, ctx.softmax_scale, False, ctx.causal, ctx.window_size[0],
        ctx.window_size[1], ctx.softcap, ctx.deterministic, None, rng_state,
        descale_q, descale_k, descale_v, descale_o, descale_do, None, None,
        None)
    dqkv = dqkv[..., :dout.shape[-1]]
    return (dqkv, None, None, None, None, None, None, None, None, None,
        None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FlashAttnVarlenQKVPackedFP8Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale,
        causal, window_size, softcap, alibi_slopes, deterministic,
        return_softmax, is_grad_enabled, descale_q: Optional[torch.Tensor]=
        None, descale_k: Optional[torch.Tensor]=None, descale_v: Optional[
        torch.Tensor]=None, descale_do: Optional[torch.Tensor]=None):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** -0.5
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        if is_fp8(q) or is_fp8(k) or is_fp8(v):
            raise ValueError(
                'fp8 input and out not supported yet for this function.')
            assert descale_q is not None and descale_k is not None and descale_v is not None, f'You need to pass descale factors for q, k and v'
            q_fp8 = q
            k_fp8 = k
            v_fp8 = v
            out_fp8, descale_o = torch.zeros_like(q_fp8), torch.zeros_like(
                descale_q)
        else:
            assert descale_q is None and descale_k is None and descale_v is None, f'Found {q.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired.'
            q_fp8, descale_q = cast_to_fp8(q, torch.float8_e4m3fnuz, 'thd',
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            k_fp8, descale_k = cast_to_fp8(k, torch.float8_e4m3fnuz, 'thd',
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            v_fp8, descale_v = cast_to_fp8(v, torch.float8_e4m3fnuz, 'thd',
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            out_fp8, descale_o = torch.zeros_like(q_fp8, dtype=torch.float32
                ), None
        q_fp8, k_fp8, v_fp8 = [maybe_contiguous(x) for x in (q_fp8, k_fp8,
            v_fp8)]
        _, softmax_lse, S_dmask, rng_state = flash_attn_gpu.varlen_fwd(q_fp8,
            k_fp8, v_fp8, out_fp8, cu_seqlens, cu_seqlens, None, None, None,
            alibi_slopes, max_seqlen, max_seqlen, dropout_p, softmax_scale,
            False, causal, window_size[0], window_size[1], softcap,
            return_softmax, None, descale_q=descale_q, descale_k=descale_k,
            descale_v=descale_v, descale_o=descale_o)
        if is_grad:
            ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse,
                cu_seqlens, rng_state, descale_q, descale_k, descale_v,
                descale_o, descale_do)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen = max_seqlen
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_fp8[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        (q_fp8, k_fp8, v_fp8, out_fp8, softmax_lse, cu_seqlens, rng_state,
            descale_q, descale_k, descale_v, descale_o, descale_do
            ) = ctx.saved_tensors
        qkv_shape = q_fp8.shape[:-2] + (3, *q_fp8.shape[-2:])
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - 
                head_size_og % 8])
        if is_fp8(dout_padded):
            raise ValueError(
                'fp8 input and out not supported yet for this function.')
            assert descale_do is not None, f'You need to pass descale factors for do'
            dout_padded_fp8 = dout_padded
            dqkv, descale_dqkv = torch.zeros(qkv_shape, device=q_fp8.device
                ), torch.zeros_like(descale_q)
        else:
            assert descale_do is None, f'Found {dout.dtype} input tensor with descale factors. In this case, we cast to fp8 and compute the descale factors. You can pass an fp8 tensor with its descale factors if desired.'
            dout_padded_fp8, descale_do = cast_to_fp8(dout_padded, torch.
                float8_e4m3fnuz, 'thd', cu_seqlens=cu_seqlens, max_seqlen=
                ctx.max_seqlen)
            dqkv, descale_dqkv = torch.zeros(qkv_shape, dtype=torch.float32,
                device=q_fp8.device), None
        dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8 = [maybe_contiguous(x
            ) for x in (dout_padded_fp8, q_fp8, k_fp8, v_fp8, out_fp8)]
        flash_attn_gpu.varlen_bwd(dout_padded_fp8, q_fp8, k_fp8, v_fp8,
            out_fp8, softmax_lse, dqkv[:, 0], dqkv[:, 1], dqkv[:, 2],
            cu_seqlens, cu_seqlens, ctx.alibi_slopes, ctx.max_seqlen, ctx.
            max_seqlen, ctx.dropout_p, ctx.softmax_scale, False, ctx.causal,
            ctx.window_size[0], ctx.window_size[1], ctx.softcap, ctx.
            deterministic, None, rng_state, descale_q, descale_k, descale_v,
            descale_o, descale_do, None, None, None)
        dqkv = dqkv[..., :dout.shape[-1]]
        return (dqkv, None, None, None, None, None, None, None, None, None,
            None, None)
