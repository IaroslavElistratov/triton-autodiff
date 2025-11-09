# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/ROCm/aiter
# Source-Files: aiter/ops/triton/mha.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ciw0cknm/aiter-main/aiter/ops/triton/mha.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def _cast_to_fp8(x: torch.Tensor, fp8_dtype, layout, clamp_val=1e-09) ->Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Convert a tensor to FP8 format, returning an FP8 tensor and a descale factor.
    Args:
        - x (torch.Tensor): shape [batch, seq_len, heads, dim]
    Returns:
        - x_fp8 (torch.Tensor): FP8 tensor with the same shape as x
        - descale_factor (torch.Tensor): tensor of shape [batch, 1, heads, 1]
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}"
            )
    reduce_dims = 1, 3
    x_abs_max = x.abs().amax(dim=reduce_dims)
    x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))
    unsqueeze_dims = sorted(reduce_dims)
    for d in unsqueeze_dims:
        x_abs_max = x_abs_max.unsqueeze(d)
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / x_abs_max
    descale_factor = x_abs_max / fp8_max
    x_fp8 = (x * scale).to(fp8_dtype)
    return x_fp8, descale_factor


def get_num_xcds():
    return 8


def _is_fp8(x):
    if x.dtype in {torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.
        float8_e5m2, torch.float8_e5m2fnuz}:
        if arch_info.is_fp8_avail():
            return True
        else:
            raise RuntimeError('This device does not support fp8')
    else:
        return False


def get_fp8_e4m3_dtype():
    if arch_info.get_arch() in 'gfx950':
        e4m3_dtype = torch.float8_e4m3fn
    else:
        e4m3_dtype = torch.float8_e4m3fnuz
    return e4m3_dtype


def get_arch():
    try:
        arch = triton.runtime.driver.active.get_current_target().arch
    except RuntimeError:
        from jax._src.lib import gpu_triton as triton_kernel_call_lib
        arch = triton_kernel_call_lib.get_arch_details('0')
        arch = arch.split(':')[0]
    return arch


def get_device():
    return _ARCH_TO_DEVICE[get_arch()]


def is_fp8_avail():
    return get_arch() in ('gfx942', 'gfx950')


@triton.jit
def _compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    x_amax = tl.max(tl.abs(x))
    x_amax = tl.where(x_amax <= 1e-09, 1e-09, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr=8):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1
            ) + local_pid
    return pid


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _attn_fwd(q_ptr: torch.Tensor, k_ptr: torch.Tensor, v_ptr: torch.Tensor,
    descale_q_ptr: torch.Tensor, descale_k_ptr: torch.Tensor, descale_v_ptr:
    torch.Tensor, out_ptr: torch.Tensor, alibi_slopes_ptr: torch.Tensor,
    s_dmask_ptr: torch.Tensor, dropout_mask_ptr: torch.Tensor,
    softmax_lse_ptr: torch.Tensor, stride_qz_in, stride_qh_in, stride_qm_in,
    stride_qk_in, stride_kz_in, stride_kh_in, stride_kn_in, stride_kk_in,
    stride_vz_in, stride_vh_in, stride_vn_in, stride_vk_in,
    stride_descale_q_z_in, stride_descale_k_z_in, stride_descale_v_z_in,
    stride_oz_in, stride_oh_in, stride_om_in, stride_on_in,
    stride_alibi_z_in, stride_alibi_h_in, stride_sd_z_in, stride_sd_h_in,
    stride_sd_m_in, stride_sd_n_in, stride_lse_z_in, stride_lse_h_in,
    stride_lse_m_in, sm_scale, cu_seqlens_q, cu_seqlens_k, dropout_p,
    philox_seed, philox_offset_base_in, SEQLEN_Q, SEQLEN_K, IS_CAUSAL: tl.
    constexpr, NUM_Q_HEADS: tl.constexpr, NUM_K_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.
    constexpr, BLOCK_DMODEL_POW2: tl.constexpr, BLOCK_DMODEL_PE: tl.
    constexpr, RETURN_SCORES: tl.constexpr, ENABLE_DROPOUT: tl.constexpr,
    IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr, VARLEN: tl.constexpr,
    BATCH, NUM_XCD: tl.constexpr, USE_INT64_STRIDES: tl.constexpr):
    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    wid = tl.program_id(0)
    off_q_head = wid % NUM_Q_HEADS
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = wid // NUM_Q_HEADS % NUM_BLOCKS
    off_z = wid // (NUM_BLOCKS * NUM_Q_HEADS) % BATCH
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)
    HAS_PE: tl.constexpr = BLOCK_DMODEL_PE > 0
    if HAS_PE:
        offs_pe = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL_PE)
    if USE_INT64_STRIDES:
        stride_qz = tl.cast(stride_qz_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qk = tl.cast(stride_qk_in, tl.int64)
        stride_kz = tl.cast(stride_kz_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kk = tl.cast(stride_kk_in, tl.int64)
        stride_vz = tl.cast(stride_vz_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vk = tl.cast(stride_vk_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
        stride_oz = tl.cast(stride_oz_in, tl.int64)
        stride_oh = tl.cast(stride_oh_in, tl.int64)
        stride_om = tl.cast(stride_om_in, tl.int64)
        stride_on = tl.cast(stride_on_in, tl.int64)
        stride_alibi_z = tl.cast(stride_alibi_z_in, tl.int64)
        stride_alibi_h = tl.cast(stride_alibi_h_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_sd_z = tl.cast(stride_sd_z_in, tl.int64)
        stride_sd_h = tl.cast(stride_sd_h_in, tl.int64)
        stride_sd_m = tl.cast(stride_sd_m_in, tl.int64)
        stride_sd_n = tl.cast(stride_sd_n_in, tl.int64)
        stride_lse_z = tl.cast(stride_lse_z_in, tl.int64)
        stride_lse_h = tl.cast(stride_lse_h_in, tl.int64)
        stride_lse_m = tl.cast(stride_lse_m_in, tl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_qh = stride_qh_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in
        stride_alibi_z = stride_alibi_z_in
        stride_alibi_h = stride_alibi_h_in
        philox_offset_base = philox_offset_base_in
        stride_sd_z = stride_sd_z_in
        stride_sd_h = stride_sd_h_in
        stride_sd_m = stride_sd_m_in
        stride_sd_n = stride_sd_n_in
        stride_lse_z = stride_lse_z_in
        stride_lse_h = stride_lse_h_in
        stride_lse_m = stride_lse_m_in
    tl.assume(stride_qz_in >= 0)
    tl.assume(stride_qh_in >= 0)
    tl.assume(stride_qm_in >= 0)
    tl.assume(stride_qk_in >= 0)
    tl.assume(stride_kz_in >= 0)
    tl.assume(stride_kh_in >= 0)
    tl.assume(stride_kn_in >= 0)
    tl.assume(stride_kk_in >= 0)
    tl.assume(stride_vz_in >= 0)
    tl.assume(stride_vh_in >= 0)
    tl.assume(stride_vn_in >= 0)
    tl.assume(stride_vk_in >= 0)
    if IS_FP8:
        tl.assume(stride_descale_q_z_in >= 0)
        tl.assume(stride_descale_k_z_in >= 0)
        tl.assume(stride_descale_v_z_in >= 0)
        tl.assume(stride_oz_in >= 0)
        tl.assume(stride_oh_in >= 0)
        tl.assume(stride_om_in >= 0)
        tl.assume(stride_on_in >= 0)
        tl.assume(stride_alibi_z_in >= 0)
        tl.assume(stride_alibi_h_in >= 0)
    tl.assume(philox_offset_base_in >= 0)
    tl.assume(stride_sd_z_in >= 0)
    tl.assume(stride_sd_h_in >= 0)
    tl.assume(stride_sd_m_in >= 0)
    tl.assume(stride_sd_n_in >= 0)
    tl.assume(stride_lse_z_in >= 0)
    tl.assume(stride_lse_h_in >= 0)
    tl.assume(stride_lse_m_in >= 0)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K
    n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_seqlen = _cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k -
            seqlen_q, BLOCK_N)
        n_blocks = min(n_blocks, n_blocks_seqlen)
        if n_blocks <= 0:
            offs_out = (off_z * stride_oz + off_q_head * stride_oh + 
                cu_seqlens_q_start * stride_om + offs_m[:, None] *
                stride_om + offs_d[None, :] * stride_on)
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=out_ptr.type
                .element_ty)
            out_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] <
                BLOCK_DMODEL)
            tl.store(out_ptr + offs_out, acc, mask=out_mask)
            if softmax_lse_ptr is not None:
                offs_lse = (off_z * stride_lse_z + off_q_head *
                    stride_lse_h + cu_seqlens_q_start * stride_lse_m + 
                    offs_m * stride_lse_m)
                lse_mask = offs_m < SEQLEN_Q
                lse = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
                tl.store(softmax_lse_ptr + offs_lse, lse, mask=lse_mask)
            return
    grp_sz: tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    if grp_sz != 1:
        off_k_head = off_q_head // grp_sz
    else:
        off_k_head = off_q_head
    q_offs = (off_z * stride_qz + off_q_head * stride_qh + 
        cu_seqlens_q_start * stride_qm + offs_m[:, None] * stride_qm + 
        offs_d[None, :] * stride_qk)
    q_ptrs = q_ptr + q_offs
    if HAS_PE:
        q_pe_offs = (off_z * stride_qz + off_q_head * stride_qh + 
            cu_seqlens_q_start * stride_qm + offs_m[:, None] * stride_qm + 
            offs_pe[None, :] * stride_qk)
        q_pe_ptrs = q_ptr + q_pe_offs
    else:
        q_pe_ptrs = None
    k_offs = (off_z * stride_kz + off_k_head * stride_kh + 
        cu_seqlens_k_start * stride_kn + offs_d[:, None] * stride_kk + 
        offs_n[None, :] * stride_kn)
    k_ptrs = k_ptr + k_offs
    if HAS_PE:
        k_pe_offs = (off_z * stride_kz + off_k_head * stride_kh + 
            cu_seqlens_k_start * stride_kn + offs_pe[:, None] * stride_kk +
            offs_n[None, :] * stride_kn)
        k_pe_ptrs = k_ptr + k_pe_offs
    else:
        k_pe_ptrs = None
    v_offs = (off_z * stride_vz + off_k_head * stride_vh + 
        cu_seqlens_k_start * stride_vn + offs_n[:, None] * stride_vn + 
        offs_d[None, :] * stride_vk)
    v_ptrs = v_ptr + v_offs
    if alibi_slopes_ptr is not None:
        alibi_offs = off_z * stride_alibi_z + off_q_head * stride_alibi_h
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offs)
    else:
        alibi_slope = None
    if s_dmask_ptr is not None:
        s_dmask_offs = off_z * stride_sd_z + off_q_head * stride_sd_h + offs_m[
            :, None] * stride_sd_m + offs_n[None, :] * stride_sd_n
        s_dmask_ptrs = s_dmask_ptr + s_dmask_offs
    else:
        s_dmask_ptrs = None
    if dropout_mask_ptr is not None:
        dropout_mask_offs = (off_z * stride_sd_z + off_q_head * stride_sd_h +
            offs_m[:, None] * stride_sd_m + offs_n[None, :] * stride_sd_n)
        dropout_mask_ptrs = dropout_mask_ptr + dropout_mask_offs
        philox_ptrs = (philox_offset_base + off_z * stride_sd_z + 
            off_q_head * stride_sd_h + offs_m[:, None] * stride_sd_m + 
            offs_n[None, :] * stride_sd_n)
    else:
        dropout_mask_ptrs = None
        philox_ptrs = None
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)
    if BLOCK_DMODEL == BLOCK_DMODEL_POW2:
        q_mask = offs_m[:, None] < seqlen_q
    else:
        q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL
            )
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    if HAS_PE:
        q_pe = tl.load(q_pe_ptrs, mask=q_mask, other=0.0)
    else:
        q_pe = None
    if IS_FP8:
        descale_q = tl.load(descale_q_ptr + off_z * stride_descale_q_z +
            off_q_head)
        descale_k = tl.load(descale_k_ptr + off_z * stride_descale_k_z +
            off_k_head)
        descale_v = tl.load(descale_v_ptr + off_z * stride_descale_v_z +
            off_k_head)
    else:
        descale_q, descale_k, descale_v = 1.0, 1.0, 1.0
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
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
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_pe, k_ptrs,
            k_pe_ptrs, v_ptrs, stride_kn, stride_vn, stride_sd_n, start_m,
            seqlen_k, seqlen_q, dropout_p, s_dmask_ptrs, dropout_mask_ptrs,
            philox_seed, philox_ptrs, block_min, block_max, 0, 0, 0,
            alibi_slope, descale_q, descale_k, descale_v, offs_m, offs_n,
            BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE, sm_scale, False, MASK_STEPS=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT, RETURN_SCORES=RETURN_SCORES,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2, IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX, ENABLE_PIPELINING=True)
        block_min = block_max
        block_max = n_blocks * BLOCK_N
    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        if HAS_PE:
            k_pe_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vn
        if RETURN_SCORES:
            s_dmask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_pe, k_ptrs,
            k_pe_ptrs, v_ptrs, stride_kn, stride_vn, stride_sd_n, start_m,
            seqlen_k, seqlen_q, dropout_p, s_dmask_ptrs, dropout_mask_ptrs,
            philox_seed, philox_ptrs, block_min, block_max, offs_n_causal,
            masked_blocks, n_extra_tokens, alibi_slope, descale_q,
            descale_k, descale_v, offs_m, offs_n, BLOCK_M, BLOCK_N,
            BLOCK_DMODEL, BLOCK_DMODEL_POW2, BLOCK_DMODEL_PE, sm_scale,
            IS_CAUSAL, MASK_STEPS=True, ENABLE_DROPOUT=ENABLE_DROPOUT,
            RETURN_SCORES=RETURN_SCORES, PADDED_HEAD=BLOCK_DMODEL !=
            BLOCK_DMODEL_POW2, IS_FP8=IS_FP8, FP8_MAX=FP8_MAX,
            ENABLE_PIPELINING=False)
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
            out_mask_boundary = tl.full((BLOCK_DMODEL_POW2,),
                causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[
                None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
    overflow_size = end_m_idx - seqlen_q
    if softmax_lse_ptr is not None:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        mi_base2 = m_i * RCP_LN2 * sm_scale
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        softmax_lse *= LN2
        if IS_CAUSAL:
            lse_causal_mask = start_m_idx + tl.arange(0, BLOCK_M
                ) < causal_start_idx
            softmax_lse = tl.where(lse_causal_mask, 0.0, softmax_lse)
        offs_lse = (off_z * stride_lse_z + off_q_head * stride_lse_h + 
            cu_seqlens_q_start * stride_lse_m + offs_m * stride_lse_m)
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=
                tl.int32)
            lse_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(softmax_lse_ptr + offs_lse, softmax_lse, mask=lse_mask)
        else:
            tl.store(softmax_lse_ptr + offs_lse, softmax_lse)
    offs_out = (off_z * stride_oz + off_q_head * stride_oh + 
        cu_seqlens_q_start * stride_om + offs_m[:, None] * stride_om + 
        offs_d[None, :] * stride_on)
    out_mask = tl.full([BLOCK_M, BLOCK_DMODEL_POW2], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < seqlen_q)
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        out_mask = out_mask & (offs_d[None, :] < BLOCK_DMODEL)
    op = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_out, op, mask=out_mask)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_pe, k_ptrs, k_pe_ptrs, v_ptrs,
    stride_kn, stride_vk, stride_sn, start_m, seqlen_k, seqlen_q, dropout_p,
    sd_mask_ptrs, dropout_mask_ptrs, philox_seed, philox_ptrs, block_min,
    block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope,
    descale_q, descale_k, descale_v, OFFS_M: tl.constexpr, OFFS_N: tl.
    constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL:
    tl.constexpr, BLOCK_DMODEL_POW2: tl.constexpr, BLOCK_DMODEL_PE: tl.
    constexpr, SM_SCALE: tl.constexpr, IS_CAUSAL: tl.constexpr, MASK_STEPS:
    tl.constexpr, ENABLE_DROPOUT: tl.constexpr, RETURN_SCORES: tl.constexpr,
    PADDED_HEAD: tl.constexpr, IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr,
    ENABLE_PIPELINING: tl.constexpr):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    HAS_PE: tl.constexpr = BLOCK_DMODEL_PE > 0
    num_stages: tl.constexpr = None if ENABLE_PIPELINING else 1
    for start_n in tl.range(block_min, block_max, BLOCK_N, num_stages=
        num_stages):
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        k = _load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, seqlen_k)
        if HAS_PE:
            k_pe = _load_fn(k_pe_ptrs, None, k_offs_n, BLOCK_DMODEL +
                BLOCK_DMODEL_PE, seqlen_k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
        if MASK_STEPS:
            bound_cond = start_n + BLOCK_N == block_max and n_extra_tokens != 0
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
            size_n = start_n + OFFS_N[None, :]
            mask_partial = size_n < boundary_m[:, None]
            mask = tl.where(bound_cond, mask_partial, mask)
        q_mask = OFFS_M[:, None] < seqlen_q
        k_mask = (start_n + tl.arange(0, BLOCK_N))[None, :] < seqlen_k
        p_mask = q_mask & k_mask
        if IS_FP8:
            qk += tl.dot(q, k) * descale_q * descale_k
        else:
            qk += tl.dot(q, k)
            if HAS_PE:
                qk += tl.dot(q_pe, k_pe)
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            mask = mask & causal_mask
        qk = tl.where(mask, qk, float('-inf'))
        if alibi_slope is not None:
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = _compute_alibi_block(alibi_slope, seqlen_q,
                seqlen_k, global_m_positions, global_n_positions)
            qk += alibi_block / SM_SCALE
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2
        q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]
        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            rng_output = tl.rand(philox_seed, philox_ptrs)
            dropout_mask = rng_output > dropout_p
            tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            tl.store(sd_mask_ptrs, p, mask=p_mask)
        m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled
        alpha = tl.math.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]
        v = _load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        if IS_FP8:
            scale_p, descale_p = _compute_fp8_scaling_factors(p, FP8_MAX)
            acc += tl.dot((p * scale_p).to(v.type.element_ty), v
                ) * descale_p * descale_v
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)
        k_ptrs += BLOCK_N * stride_kn
        if HAS_PE:
            k_pe_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn
    return acc, l_i, m_i


@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n,
    transpose=False):
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :
        ]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


@triton.jit
def _load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second
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


def _flash_attn_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    dropout_p: float, softmax_scale: float, causal: bool, window_size_left:
    int, window_size_right: int, bias: Optional[torch.Tensor], alibi_slopes:
    Optional[torch.Tensor], return_lse: bool, return_softmax: bool,
    max_seqlen_q: int, max_seqlen_k: int, cu_seqlens_q: Optional[torch.
    Tensor]=None, cu_seqlens_k: Optional[torch.Tensor]=None, descale_q:
    Optional[torch.Tensor]=None, descale_k: Optional[torch.Tensor]=None,
    descale_v: Optional[torch.Tensor]=None, config: Optional[dict[str, any]
    ]=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bias is not None:
        raise ValueError('Bias is not supported yet in the Triton Backend')
    if window_size_left != -1 or window_size_right != -1:
        raise ValueError(
            'Sliding Window is not supported yet in the Triton Backend')
    IS_FP8 = types._is_fp8(q)
    FP8_MAX: tl.constexpr = torch.finfo(q.dtype).max
    is_varlen = True if cu_seqlens_q is not None else False
    if IS_FP8:
        o = torch.zeros(q.shape[:-1] + v.shape[-1:], dtype=torch.float32,
            device=q.device)
    else:
        o = torch.zeros(q.shape[:-1] + v.shape[-1:], dtype=q.dtype, device=
            q.device)
    if is_varlen:
        batch, seqlen_q, num_q_heads = len(cu_seqlens_q
            ) - 1, max_seqlen_q, q.shape[1]
        num_k_heads = k.shape[1]
        q_strides = 0, q.stride(1), q.stride(0), q.stride(2)
        k_strides = 0, k.stride(1), k.stride(0), k.stride(2)
        v_strides = 0, v.stride(1), v.stride(0), v.stride(2)
        o_strides = 0, o.stride(1), o.stride(0), o.stride(2)
    else:
        batch, seqlen_q, num_q_heads = q.shape[:-1]
        num_k_heads = k.shape[2]
        q_strides = q.stride(0), q.stride(2), q.stride(1), q.stride(3)
        k_strides = k.stride(0), k.stride(2), k.stride(1), k.stride(3)
        v_strides = v.stride(0), v.stride(2), v.stride(1), v.stride(3)
        o_strides = o.stride(0), o.stride(2), o.stride(1), o.stride(3)
    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    BLOCK_DMODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_DMODEL_PE_POW2 = 0 if pe_head_dim == 0 else max(triton.
        next_power_of_2(pe_head_dim), 16)
    assert pe_head_dim == 0 and BLOCK_DMODEL_PE_POW2 == 0 or v_head_dim == BLOCK_DMODEL_POW2 and pe_head_dim == BLOCK_DMODEL_PE_POW2, 'Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2.'
    assert not IS_FP8 or IS_FP8 and pe_head_dim == 0, "Positional encoding doesn't support FP8."
    if is_varlen:
        softmax_lse = torch.zeros((q.shape[0], num_q_heads), device=q.
            device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = 0, softmax_lse.stride(1
            ), softmax_lse.stride(0)
    else:
        softmax_lse = torch.zeros((batch, num_q_heads, max_seqlen_q),
            device=q.device, dtype=torch.float32)
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()
    enable_dropout = dropout_p > 0.0
    if enable_dropout:
        philox_seed = torch.randint(0, 16777215, (1,))[0].item()
        philox_offset = torch.randint(0, 16777215, (1,))[0].item()
    else:
        philox_seed = 0
        philox_offset = 0
    if return_softmax or enable_dropout:
        s_dmask = torch.zeros((batch, num_q_heads, max_seqlen_q,
            max_seqlen_k), device=q.device, dtype=torch.float32)
        dropout_mask = torch.zeros((batch, num_q_heads, max_seqlen_q,
            max_seqlen_k), device=q.device, dtype=torch.float32)
    else:
        s_dmask = None
        dropout_mask = None
    if config is None:
        config = _get_config(enable_dropout, q.dtype, has_pe=pe_head_dim > 0)
    """
    # Tuned for MI300x
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "waves_per_eu": 2,
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 1,
    }
    # Dropout significantly increases VGPR usage so use small tiles
    if enable_dropout or q.dtype == torch.float32:
        config = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "waves_per_eu": 1,
            "num_warps": 2,
            "num_ctas": 1,
            "num_stages": 1,
        }
    """
    grid = lambda META: (batch * num_q_heads * triton.cdiv(seqlen_q, META[
        'BLOCK_M']),)
    _attn_fwd[grid](q, k, v, descale_q, descale_k, descale_v, o,
        alibi_slopes, s_dmask, dropout_mask, softmax_lse, *q_strides, *
        k_strides, *v_strides, descale_q.stride(0) if descale_q is not None
         else 0, descale_k.stride(0) if descale_k is not None else 0, 
        descale_v.stride(0) if descale_v is not None else 0, *o_strides, 
        alibi_slopes.stride(0) if alibi_slopes is not None else 0, 
        alibi_slopes.stride(1) if alibi_slopes is not None else 0, s_dmask.
        stride(0) if s_dmask is not None else 0, s_dmask.stride(1) if 
        s_dmask is not None else 0, s_dmask.stride(2) if s_dmask is not
        None else 0, s_dmask.stride(3) if s_dmask is not None else 0, 
        stride_lse_z if softmax_lse is not None else 0, stride_lse_h if 
        softmax_lse is not None else 0, stride_lse_m if softmax_lse is not
        None else 0, softmax_scale, cu_seqlens_q, cu_seqlens_k, dropout_p,
        philox_seed, philox_offset, SEQLEN_Q=max_seqlen_q, SEQLEN_K=
        max_seqlen_k, IS_CAUSAL=causal, NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads, BLOCK_DMODEL=v_head_dim, BLOCK_DMODEL_POW2
        =BLOCK_DMODEL_POW2, BLOCK_DMODEL_PE=pe_head_dim, RETURN_SCORES=
        return_softmax, ENABLE_DROPOUT=enable_dropout, IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX, VARLEN=is_varlen, BATCH=batch, NUM_XCD=
        get_num_xcds(), USE_INT64_STRIDES=_USE_INT64_STRIDES, **config)
    return o, softmax_lse, s_dmask, philox_seed, philox_offset


@functools.lru_cache(maxsize=1024)
def _get_config(enable_dropout: bool, dtype: torch.dtype, has_pe: bool=False):
    if not hasattr(_get_config, '_config_dict'):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f'{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json'
        with open(fpath, 'r') as file:
            config = json.load(file)
        _get_config._config_dict['default'] = config
    if has_pe and 'pe' in _get_config._config_dict['default']['fwd']:
        return _get_config._config_dict['default']['fwd']['pe']
    elif enable_dropout or dtype == torch.float32:
        return _get_config._config_dict['default']['fwd']['dropout_or_fp32']
    else:
        return _get_config._config_dict['default']['fwd']['default']


# Forward method (kernel launch code)
def __FlashAttnFP8Func_forward(ctx, q, k, v, dropout_p, softmax_scale,
    causal, window_size, alibi_slopes, deterministic, return_lse,
    return_softmax, is_grad_enabled, config=None):
    is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    head_size_og = q.size(3)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    fp8_dtype = types.get_fp8_e4m3_dtype()
    q_fp8, descale_q = _cast_to_fp8(q, fp8_dtype, 'bshd')
    k_fp8, descale_k = _cast_to_fp8(k, fp8_dtype, 'bshd')
    v_fp8, descale_v = _cast_to_fp8(v, fp8_dtype, 'bshd')
    out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
        _flash_attn_forward(q_fp8, k_fp8, v_fp8, dropout_p, softmax_scale,
        causal=causal, window_size_left=int(window_size[0]),
        window_size_right=int(window_size[1]), bias=None, alibi_slopes=
        alibi_slopes, return_lse=return_lse, return_softmax=return_softmax and
        dropout_p > 0, max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1],
        cu_seqlens_q=None, cu_seqlens_k=None, descale_q=descale_q,
        descale_k=descale_k, descale_v=descale_v, config=config))
    if is_grad:
        ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_padded, softmax_lse,
            descale_q, descale_k, descale_v)
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
    out = out_padded[..., :head_size_og]
    result = [out]
    if return_lse:
        result.append(softmax_lse)
    if return_softmax:
        result.append(S_dmask)
    return result[0] if len(result) == 1 else tuple(result)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _bwd_dkdvdq_inner(dk, dv, Q, k, v, DO, DQ, M, D, sm_scale, stride_q_m,
    stride_q_k, stride_dq_m, stride_dq_k, stride_do_m, stride_do_k,
    stride_dropout_m, stride_dropout_n, stride_deltam, dropout_p,
    philox_seed, batch_philox_offset, dropout_offset, seqlen_q, seqlen_k,
    start_n, start_m, num_steps, descale_q, descale_k, descale_v,
    descale_do, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D_MODEL:
    tl.constexpr, BLOCK_D_MODEL_POW2: tl.constexpr, MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr, IS_FP8: tl.constexpr, FP8_MAX: tl.
    constexpr, workgroup_id):
    tl.assume(stride_q_m >= 0)
    tl.assume(stride_q_k >= 0)
    tl.assume(stride_dq_m >= 0)
    tl.assume(stride_dq_k >= 0)
    tl.assume(stride_do_m >= 0)
    tl.assume(stride_do_k >= 0)
    tl.assume(stride_deltam >= 0)
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    mask_n = offs_n < seqlen_k
    qT_ptrs_start = Q + offs_m[None, :] * stride_q_m + offs_k[:, None
        ] * stride_q_k
    dq_ptrs_start = DQ + offs_m[:, None] * stride_dq_m + offs_k[None, :
        ] * stride_dq_k
    do_ptrs_start = DO + offs_m[:, None] * stride_do_m + offs_k[None, :
        ] * stride_do_k
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    for iter in range(num_steps):
        blk_idx = (iter + workgroup_id) % num_steps
        curr_m = start_m + blk_idx * step_m
        qT_ptrs = qT_ptrs_start + blk_idx * step_m * stride_q_m
        dq_ptrs = dq_ptrs_start + blk_idx * step_m * stride_dq_m
        do_ptrs = do_ptrs_start + blk_idx * step_m * stride_do_m
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seqlen_q
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < BLOCK_D_MODEL
            mask_do &= offs_k[None, :] < BLOCK_D_MODEL
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        if ENABLE_DROPOUT:
            philox_offs = curr_philox_offset + offs_m[None, :
                ] * stride_dropout_m + offs_n[:, None] * stride_dropout_n
            rand_vals = tl.rand(philox_seed, philox_offs)
            dropout_mask = rand_vals > dropout_p
            dropout_scale = 1.0 / (1 - dropout_p)
        m = tl.load(M + offs_m * stride_deltam, mask=mask_m, other=0.0)
        if IS_FP8:
            qkT = tl.dot(k, qT) * descale_q * descale_k
        else:
            qkT = tl.dot(k, qT)
        pT = tl.math.exp(qkT * sm_scale - m[None, :])
        if MASK:
            causal_mask = offs_m[None, :] - delta_qk >= offs_n[:, None]
            mask = causal_mask & mask_nm
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        if ENABLE_DROPOUT:
            pT_dropout = tl.where(dropout_mask, pT, 0.0) * dropout_scale
            if IS_FP8:
                scale_p_dropout, descale_p_dropout = (
                    _compute_fp8_scaling_factors(pT_dropout, FP8_MAX))
                dv += tl.dot((pT_dropout * scale_p_dropout).to(do.type.
                    element_ty), do) * descale_p_dropout * descale_do
            else:
                dv += tl.dot(pT_dropout.to(do.type.element_ty), do)
        elif IS_FP8:
            scale_pT, descale_pT = _compute_fp8_scaling_factors(pT, FP8_MAX)
            dv += tl.dot((pT * scale_pT).to(do.type.element_ty), do
                ) * descale_pT * descale_do
        else:
            dv += tl.dot(pT.to(do.type.element_ty), do)
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
            scale_dsT, descale_dsT = _compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT)
                ) * descale_dsT * descale_q
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT))
        if IS_FP8:
            dq_partial = tl.dot((dsT * scale_dsT).to(k.dtype).T, k
                ) * descale_dsT * descale_k
        else:
            dq_partial = tl.dot(dsT.to(k.dtype).T, k)
        tl.atomic_add(dq_ptrs, dq_partial * sm_scale, mask=mask_m[:, None] &
            (offs_k[None, :] < BLOCK_D_MODEL), sem='relaxed')
    return dk, dv


@triton.jit
def _bwd_kernel_dkdvdq_causal(q_ptr, k_ptr, v_ptr, sm_scale, do_ptr, dk_ptr,
    dv_ptr, dq_ptr, m_ptr, delta_ptr, stride_q_b_in, stride_q_h_in,
    stride_q_m_in, stride_q_k_in, stride_k_b_in, stride_k_h_in,
    stride_k_n_in, stride_k_k_in, stride_v_b_in, stride_v_h_in,
    stride_v_n_in, stride_v_k_in, stride_dk_b_in, stride_dk_h_in,
    stride_dk_n_in, stride_dk_k_in, stride_dq_b_in, stride_dq_h_in,
    stride_dq_m_in, stride_dq_k_in, stride_delta_b_in, stride_delta_h_in,
    stride_delta_m_in, stride_do_b_in, stride_do_h_in, stride_do_m_in,
    stride_do_k_in, stride_dropout_b_in, stride_dropout_h_in,
    stride_dropout_m_in, stride_dropout_n_in, stride_descale_q_z_in,
    stride_descale_k_z_in, stride_descale_v_z_in, stride_descale_do_z_in,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_mask,
    dropout_p, philox_seed, philox_offset_base_in, descale_q_ptr,
    descale_k_ptr, descale_v_ptr, descale_do_ptr, NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr, BATCH, NUM_K_PIDS, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLK_SLICE_FACTOR: tl.constexpr, BLOCK_D_MODEL:
    tl.constexpr, BLOCK_D_MODEL_POW2: tl.constexpr, ENABLE_DROPOUT: tl.
    constexpr, IS_VARLEN: tl.constexpr, IS_FP8: tl.constexpr, FP8_MAX: tl.
    constexpr, NUM_SMS: tl.constexpr, USE_INT64_STRIDES: tl.constexpr,
    NUM_XCD: tl.constexpr):
    if USE_INT64_STRIDES:
        stride_q_b = tl.cast(stride_q_b_in, tl.int64)
        stride_q_h = tl.cast(stride_q_h_in, tl.int64)
        stride_q_m = tl.cast(stride_q_m_in, tl.int64)
        stride_q_k = tl.cast(stride_q_k_in, tl.int64)
        stride_k_b = tl.cast(stride_k_b_in, tl.int64)
        stride_k_h = tl.cast(stride_k_h_in, tl.int64)
        stride_k_n = tl.cast(stride_k_n_in, tl.int64)
        stride_k_k = tl.cast(stride_k_k_in, tl.int64)
        stride_v_b = tl.cast(stride_v_b_in, tl.int64)
        stride_v_h = tl.cast(stride_v_h_in, tl.int64)
        stride_v_n = tl.cast(stride_v_n_in, tl.int64)
        stride_v_k = tl.cast(stride_v_k_in, tl.int64)
        stride_dk_b = tl.cast(stride_dk_b_in, tl.int64)
        stride_dk_h = tl.cast(stride_dk_h_in, tl.int64)
        stride_dk_n = tl.cast(stride_dk_n_in, tl.int64)
        stride_dk_k = tl.cast(stride_dk_k_in, tl.int64)
        stride_dq_b = tl.cast(stride_dq_b_in, tl.int64)
        stride_dq_h = tl.cast(stride_dq_h_in, tl.int64)
        stride_dq_m = tl.cast(stride_dq_m_in, tl.int64)
        stride_dq_k = tl.cast(stride_dq_k_in, tl.int64)
        stride_delta_b = tl.cast(stride_delta_b_in, tl.int64)
        stride_delta_h = tl.cast(stride_delta_h_in, tl.int64)
        stride_delta_m = tl.cast(stride_delta_m_in, tl.int64)
        stride_do_b = tl.cast(stride_do_b_in, tl.int64)
        stride_do_h = tl.cast(stride_do_h_in, tl.int64)
        stride_do_m = tl.cast(stride_do_m_in, tl.int64)
        stride_do_k = tl.cast(stride_do_k_in, tl.int64)
        stride_dropout_b = tl.cast(stride_dropout_b_in, tl.int64)
        stride_dropout_h = tl.cast(stride_dropout_h_in, tl.int64)
        stride_dropout_m = tl.cast(stride_dropout_m_in, tl.int64)
        stride_dropout_n = tl.cast(stride_dropout_n_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
    else:
        stride_q_b = stride_q_b_in
        stride_q_h = stride_q_h_in
        stride_q_m = stride_q_m_in
        stride_q_k = stride_q_k_in
        stride_k_b = stride_k_b_in
        stride_k_h = stride_k_h_in
        stride_k_n = stride_k_n_in
        stride_k_k = stride_k_k_in
        stride_v_b = stride_v_b_in
        stride_v_h = stride_v_h_in
        stride_v_n = stride_v_n_in
        stride_v_k = stride_v_k_in
        stride_dk_b = stride_dk_b_in
        stride_dk_h = stride_dk_h_in
        stride_dk_n = stride_dk_n_in
        stride_dk_k = stride_dk_k_in
        stride_dq_b = stride_dq_b_in
        stride_dq_h = stride_dq_h_in
        stride_dq_m = stride_dq_m_in
        stride_dq_k = stride_dq_k_in
        stride_delta_b = stride_delta_b_in
        stride_delta_h = stride_delta_h_in
        stride_delta_m = stride_delta_m_in
        stride_do_b = stride_do_b_in
        stride_do_h = stride_do_h_in
        stride_do_m = stride_do_m_in
        stride_do_k = stride_do_k_in
        stride_dropout_b = stride_dropout_b_in
        stride_dropout_h = stride_dropout_h_in
        stride_dropout_m = stride_dropout_m_in
        stride_dropout_n = stride_dropout_n_in
        philox_offset_base = philox_offset_base_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in
    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    wid = tl.program_id(0)
    head_q_idx = wid % NUM_Q_HEADS
    head_q_idx = remap_xcd(head_q_idx, NUM_Q_HEADS, NUM_XCD)
    seq_k_blk_idx = wid // NUM_Q_HEADS % NUM_K_PIDS
    batch_idx = wid // (NUM_K_PIDS * NUM_Q_HEADS) % BATCH
    head_q_idx = head_q_idx * 29 % NUM_Q_HEADS
    head_k_idx = head_q_idx // GROUP_SIZE
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + batch_idx)
        q_end = tl.load(cu_seqlens_q + batch_idx + 1)
        k_start = tl.load(cu_seqlens_k + batch_idx)
        k_end = tl.load(cu_seqlens_k + batch_idx + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start
    dk = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    delta_qk = seqlen_q - seqlen_k
    num_blocks_skip = -delta_qk // BLOCK_N
    delta_aligned = (num_blocks_skip + 1) * BLOCK_N + delta_qk
    start_delta_q_lt_k = delta_aligned // BLOCK_M * BLOCK_M
    if delta_qk >= 0:
        start_delta = delta_qk
    else:
        start_delta = start_delta_q_lt_k
    start_n = seq_k_blk_idx * BLOCK_N
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    if PADDED_HEAD:
        mask_k = offs_k < BLOCK_D_MODEL
        mask_kv &= mask_k[None, :]
    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    adj_k = (batch_idx * stride_k_b + head_k_idx * stride_k_h + k_start *
        stride_k_n + offs_n[:, None] * stride_k_n + offs_k[None, :] *
        stride_k_k)
    adj_v = (batch_idx * stride_v_b + head_k_idx * stride_v_h + k_start *
        stride_v_n + offs_n[:, None] * stride_v_n + offs_k[None, :] *
        stride_v_k)
    k = tl.load(k_ptr + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(v_ptr + adj_v, mask=mask_kv, other=0.0)
    if delta_qk >= 0:
        start_m = start_n + start_delta
        len_m = BLOCK_N
    else:
        start_m = max(start_n + delta_qk, 0)
        start_m = start_m // BLOCK_M * BLOCK_M
        residue_m = max(start_n + delta_qk - start_m, 0)
        len_m = BLOCK_N + residue_m
    adj_q = (batch_idx * stride_q_b + head_q_idx * stride_q_h + q_start *
        stride_q_m)
    adj_dq = (batch_idx * stride_dq_b + head_q_idx * stride_dq_h + q_start *
        stride_dq_m)
    q_ptr_adj = q_ptr + adj_q
    dq_ptr_adj = dq_ptr + adj_dq
    adj_do = (batch_idx * stride_do_b + head_q_idx * stride_do_h + q_start *
        stride_do_m)
    do_ptr_adj = do_ptr + adj_do
    adj_delta = (batch_idx * stride_delta_b + head_q_idx * stride_delta_h +
        q_start * stride_delta_m)
    m_ptr_adj = m_ptr + adj_delta
    delta_ptr_adj = delta_ptr + adj_delta
    batch_philox_offset = 0
    dropout_offset = 0
    if ENABLE_DROPOUT:
        batch_philox_offset = (philox_offset_base + batch_idx *
            stride_dropout_b + head_q_idx * stride_dropout_h)
        dropout_offset = (dropout_mask + batch_idx * stride_dropout_b + 
            head_q_idx * stride_dropout_h)
    MASK_BLOCK_M: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
    len_m = min(len_m, seqlen_q)
    num_steps = tl.cdiv(len_m, MASK_BLOCK_M)
    if seq_k_blk_idx < num_blocks_skip:
        num_steps = 0
    if IS_FP8:
        descale_q = tl.load(descale_q_ptr + batch_idx * stride_descale_q_z +
            head_q_idx)
        descale_k = tl.load(descale_k_ptr + batch_idx * stride_descale_k_z +
            head_k_idx)
        descale_v = tl.load(descale_v_ptr + batch_idx * stride_descale_v_z +
            head_k_idx)
        descale_do = tl.load(descale_do_ptr + batch_idx *
            stride_descale_do_z + head_q_idx)
    else:
        descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0
    dk, dv = _bwd_dkdvdq_inner(dk, dv, q_ptr_adj, k, v, do_ptr_adj,
        dq_ptr_adj, m_ptr_adj, delta_ptr_adj, sm_scale, stride_q_m,
        stride_q_k, stride_dq_m, stride_dq_k, stride_do_m, stride_do_k,
        stride_dropout_m, stride_dropout_n, stride_delta_m, dropout_p,
        philox_seed, batch_philox_offset, dropout_offset, seqlen_q,
        seqlen_k, start_n, start_m, num_steps, descale_q, descale_k,
        descale_v, descale_do, MASK_BLOCK_M, BLOCK_N, BLOCK_D_MODEL,
        BLOCK_D_MODEL_POW2, MASK=True, ENABLE_DROPOUT=ENABLE_DROPOUT,
        IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, workgroup_id=seq_k_blk_idx)
    start_m += num_steps * MASK_BLOCK_M
    num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M)
    dk, dv = _bwd_dkdvdq_inner(dk, dv, q_ptr_adj, k, v, do_ptr_adj,
        dq_ptr_adj, m_ptr_adj, delta_ptr_adj, sm_scale, stride_q_m,
        stride_q_k, stride_dq_m, stride_dq_k, stride_do_m, stride_do_k,
        stride_dropout_m, stride_dropout_n, stride_delta_m, dropout_p,
        philox_seed, batch_philox_offset, dropout_offset, seqlen_q,
        seqlen_k, start_n, start_m, num_steps, descale_q, descale_k,
        descale_v, descale_do, BLOCK_M, BLOCK_N, BLOCK_D_MODEL,
        BLOCK_D_MODEL_POW2, MASK=False, ENABLE_DROPOUT=ENABLE_DROPOUT,
        IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, workgroup_id=seq_k_blk_idx)
    offs_dkdv = (batch_idx * stride_dk_b + head_k_idx * stride_dk_h + 
        k_start * stride_dk_n + offs_n[:, None] * stride_dk_n + offs_k[None,
        :] * stride_dk_k)
    tl.atomic_add(dv_ptr + offs_dkdv, dv, mask=mask_kv, sem='relaxed')
    dk *= sm_scale
    tl.atomic_add(dk_ptr + offs_dkdv, dk, mask=mask_kv, sem='relaxed')


@triton.jit
def _bwd_kernel_dkdvdq_noncausal(Q, K, V, sm_scale, DO, DK, DV, DQ, M,
    Delta, stride_qb_in, stride_qh_in, stride_qm_in, stride_qk_in,
    stride_kb_in, stride_kh_in, stride_kn_in, stride_kk_in, stride_vb_in,
    stride_vh_in, stride_vn_in, stride_vk_in, stride_dkb_in, stride_dkh_in,
    stride_dkn_in, stride_dkk_in, stride_dqb_in, stride_dqh_in,
    stride_dqm_in, stride_dqk_in, stride_deltab_in, stride_deltah_in,
    stride_deltam_in, stride_dob_in, stride_doh_in, stride_dom_in,
    stride_dok_in, stride_dropoutb_in, stride_dropouth_in,
    stride_dropoutm_in, stride_dropoutn_in, stride_descale_q_z_in,
    stride_descale_k_z_in, stride_descale_v_z_in, stride_descale_do_z_in,
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_mask,
    dropout_p, philox_seed, philox_offset, descale_q_ptr, descale_k_ptr,
    descale_v_ptr, descale_do_ptr, NUM_Q_HEADS: tl.constexpr, NUM_K_HEADS:
    tl.constexpr, BATCH, NUM_K_PIDS, BLOCK_M: tl.constexpr, BLOCK_N: tl.
    constexpr, BLK_SLICE_FACTOR: tl.constexpr, BLOCK_D_MODEL: tl.constexpr,
    BLOCK_D_MODEL_POW2: tl.constexpr, ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr, IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr,
    NUM_SMS: tl.constexpr, USE_INT64_STRIDES: tl.constexpr):
    if USE_INT64_STRIDES:
        stride_qb = tl.cast(stride_qb_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qk = tl.cast(stride_qk_in, tl.int64)
        stride_kb = tl.cast(stride_kb_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kk = tl.cast(stride_kk_in, tl.int64)
        stride_vb = tl.cast(stride_vb_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vk = tl.cast(stride_vk_in, tl.int64)
        stride_dkb = tl.cast(stride_dkb_in, tl.int64)
        stride_dkh = tl.cast(stride_dkh_in, tl.int64)
        stride_dkn = tl.cast(stride_dkn_in, tl.int64)
        stride_dkk = tl.cast(stride_dkk_in, tl.int64)
        stride_dqb = tl.cast(stride_dqb_in, tl.int64)
        stride_dqh = tl.cast(stride_dqh_in, tl.int64)
        stride_dqm = tl.cast(stride_dqm_in, tl.int64)
        stride_dqk = tl.cast(stride_dqk_in, tl.int64)
        stride_deltab = tl.cast(stride_deltab_in, tl.int64)
        stride_deltah = tl.cast(stride_deltah_in, tl.int64)
        stride_deltam = tl.cast(stride_deltam_in, tl.int64)
        stride_dob = tl.cast(stride_dob_in, tl.int64)
        stride_doh = tl.cast(stride_doh_in, tl.int64)
        stride_dom = tl.cast(stride_dom_in, tl.int64)
        stride_dok = tl.cast(stride_dok_in, tl.int64)
        stride_dropoutb = tl.cast(stride_dropoutb_in, tl.int64)
        stride_dropouth = tl.cast(stride_dropouth_in, tl.int64)
        stride_dropoutm = tl.cast(stride_dropoutm_in, tl.int64)
        stride_dropoutn = tl.cast(stride_dropoutn_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
    else:
        stride_qb = stride_qb_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_kb = stride_kb_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vb = stride_vb_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_dkb = stride_dkb_in
        stride_dkh = stride_dkh_in
        stride_dkn = stride_dkn_in
        stride_dkk = stride_dkk_in
        stride_dqb = stride_dqb_in
        stride_dqh = stride_dqh_in
        stride_dqm = stride_dqm_in
        stride_dqk = stride_dqk_in
        stride_deltab = stride_deltab_in
        stride_deltah = stride_deltah_in
        stride_deltam = stride_deltam_in
        stride_dob = stride_dob_in
        stride_doh = stride_doh_in
        stride_dom = stride_dom_in
        stride_dok = stride_dok_in
        stride_dropoutb = stride_dropoutb_in
        stride_dropouth = stride_dropouth_in
        stride_dropoutm = stride_dropoutm_in
        stride_dropoutn = stride_dropoutn_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in
    wid = tl.program_id(0)
    bid = wid % BATCH
    hkid = wid // BATCH % NUM_K_HEADS
    pid = wid // (BATCH * NUM_K_HEADS) % NUM_K_PIDS
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
    dk = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D_MODEL_POW2], dtype=tl.float32)
    start_n = pid * BLOCK_N
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    if PADDED_HEAD:
        mask_kv &= offs_k < BLOCK_D_MODEL
    GROUP_SIZE = NUM_Q_HEADS // NUM_K_HEADS
    adj_k = bid * stride_kb + hkid * stride_kh + k_start * stride_kn + offs_n[
        :, None] * stride_kn + offs_k[None, :] * stride_kk
    adj_v = bid * stride_vb + hkid * stride_vh + k_start * stride_vn + offs_n[
        :, None] * stride_vn + offs_k[None, :] * stride_vk
    k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_dq = bid * stride_dqb + hqid * stride_dqh + q_start * stride_dqm
        Q_ptr = Q + adj_q
        DQ_ptr = DQ + adj_dq
        adj_do = bid * stride_dob + hqid * stride_doh + q_start * stride_dom
        DO_ptr = DO + adj_do
        adj_delta = (bid * stride_deltab + hqid * stride_deltah + q_start *
            stride_deltam)
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = (philox_offset + bid * stride_dropoutb + 
                hqid * stride_dropouth)
            dropout_offset = (dropout_mask + bid * stride_dropoutb + hqid *
                stride_dropouth)
        if IS_FP8:
            descale_q = tl.load(descale_q_ptr + bid * stride_descale_q_z + hqid
                )
            descale_k = tl.load(descale_k_ptr + bid * stride_descale_k_z + hkid
                )
            descale_v = tl.load(descale_v_ptr + bid * stride_descale_v_z + hkid
                )
            descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z +
                hqid)
        else:
            descale_q, descale_k, descale_v, descale_do = 1.0, 1.0, 1.0, 1.0
        start_m = 0
        num_steps = tl.cdiv(seqlen_q, BLOCK_M)
        dk, dv = _bwd_dkdvdq_inner(dk, dv, Q_ptr, k, v, DO_ptr, DQ_ptr,
            M_ptr, Delta_ptr, sm_scale, stride_qm, stride_qk, stride_dqm,
            stride_dqk, stride_dom, stride_dok, stride_dropoutm,
            stride_dropoutn, stride_deltam, dropout_p, philox_seed,
            batch_philox_offset, dropout_offset, seqlen_q, seqlen_k,
            start_n, start_m, num_steps, descale_q, descale_k, descale_v,
            descale_do, BLOCK_M, BLOCK_N, BLOCK_D_MODEL, BLOCK_D_MODEL_POW2,
            MASK=False, ENABLE_DROPOUT=ENABLE_DROPOUT, IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX, workgroup_id=wid)
    adj_dkdv = (bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn +
        offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)
    tl.store(DV + adj_dkdv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_dkdv, dk, mask=mask_kv)


@triton.jit
def _bwd_preprocess(o_ptr, do_ptr, delta_ptr, stride_o_b, stride_o_h,
    stride_o_m, stride_o_k, stride_delta_b, stride_delta_h, stride_delta_m,
    stride_descale_do_z, cu_seqlens_q, max_seqlen_q, descale_do_ptr,
    BLOCK_M: tl.constexpr, BLOCK_D_MODEL: tl.constexpr, BLOCK_D_MODEL_POW2:
    tl.constexpr, IS_VARLEN: tl.constexpr, IS_FP8: tl.constexpr):
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
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs = bid * stride_o_b + hid * stride_o_h + q_start * stride_o_m + offs_m[
        :, None] * stride_o_m + offs_k[None, :] * stride_o_k
    mask_m = offs_m < seqlen_q
    mask = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    if PADDED_HEAD:
        mask &= offs_k[None, :] < BLOCK_D_MODEL
    o = tl.load(o_ptr + offs, mask=mask, other=0.0)
    do = tl.load(do_ptr + offs, mask=mask, other=0.0)
    if IS_FP8:
        descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hid)
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do),
            axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    offs_delta = (bid * stride_delta_b + hid * stride_delta_h + q_start *
        stride_delta_m + offs_m * stride_delta_m)
    tl.store(delta_ptr + offs_delta, delta, mask=mask_m)


@triton.jit
def _bwd_dkdv_inner(dk, dk_pe, dv, Q, k, k_pe, v, DO, M, D, sm_scale,
    stride_qm, stride_qk, stride_dom, stride_dok, stride_dropoutm,
    stride_dropoutn, stride_deltam, BLOCK_M: tl.constexpr, BLOCK_N: tl.
    constexpr, HEAD_DIM: tl.constexpr, ACTUAL_HEAD_DIM: tl.constexpr,
    PE_HEAD_DIM: tl.constexpr, dropout_p, philox_seed, batch_philox_offset,
    dropout_offset, alibi_slope, seqlen_q, seqlen_k, start_n, start_m,
    num_steps, descale_q, descale_k, descale_v, descale_do, MASK: tl.
    constexpr, ENABLE_DROPOUT: tl.constexpr, USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr, FP8_MAX: tl.constexpr,
    DEBUG_TRITON: tl.constexpr, DEBUG_TRITON_DETAIL: tl.constexpr):
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    HAS_PE: tl.constexpr = PE_HEAD_DIM > 0
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    if HAS_PE:
        offs_k_pe = HEAD_DIM + tl.arange(0, PE_HEAD_DIM)
    mask_n = offs_n < seqlen_k
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
    if HAS_PE:
        qT_pe_ptrs = Q + offs_m[None, :] * stride_qm + offs_k_pe[:, None
            ] * stride_qk
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
        if HAS_PE:
            qT_pe = tl.load(qT_pe_ptrs, mask=mask_qT, other=0.0)
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
            if HAS_PE:
                qkT += tl.dot(k_pe, qT_pe)
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
                    _compute_fp8_scaling_factors(pT_dropout, FP8_MAX))
                dv += tl.dot((pT_dropout * scale_p_dropout).to(do.type.
                    element_ty), do) * descale_p_dropout * descale_do
            else:
                dv += tl.dot(pT_dropout.to(do.type.element_ty), do)
        elif IS_FP8:
            scale_pT, descale_pT = _compute_fp8_scaling_factors(pT, FP8_MAX)
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
            scale_dsT, descale_dsT = _compute_fp8_scaling_factors(dsT, FP8_MAX)
            dk += tl.dot((dsT * scale_dsT).to(qT.type.element_ty), tl.trans(qT)
                ) * descale_dsT * descale_q
        else:
            dk += tl.dot(dsT.to(qT.type.element_ty), tl.trans(qT))
            if HAS_PE:
                dk_pe += tl.dot(dsT.to(qT_pe.type.element_ty), tl.trans(qT_pe))
        curr_m += step_m
        qT_ptrs += step_m * stride_qm
        if HAS_PE:
            qT_pe_ptrs += step_m * stride_qm
        do_ptrs += step_m * stride_dom
    return dk, dk_pe, dv


@triton.jit
def _bwd_dq_inner(dq, dq_pe, q, q_pe, K, V, do, m, Delta, sm_scale,
    stride_qm, stride_qk, stride_kn, stride_kk, stride_vn, stride_vk,
    stride_dropoutm, stride_dropoutn, stride_deltam, seqlen_q, seqlen_k,
    BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr, PE_HEAD_DIM: tl.constexpr, dropout_p,
    philox_seed, batch_philox_offset, dropout_offset, alibi_slope, start_m,
    start_n, end_n, num_steps, descale_q, descale_k, descale_v, descale_do,
    MASK: tl.constexpr, ENABLE_DROPOUT: tl.constexpr, USE_ALIBI: tl.
    constexpr, USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr, FP8_MAX: tl.
    constexpr, DEBUG_TRITON: tl.constexpr, DEBUG_TRITON_DETAIL: tl.constexpr):
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    HAS_PE: tl.constexpr = PE_HEAD_DIM > 0
    delta_qk = seqlen_q - seqlen_k
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    if HAS_PE:
        offs_k_pe = HEAD_DIM + tl.arange(0, PE_HEAD_DIM)
    mask_m = offs_m < seqlen_q
    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    if HAS_PE:
        kT_pe_ptrs = K + offs_n[None, :] * stride_kn + offs_k_pe[:, None
            ] * stride_kk
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
        if HAS_PE:
            kT_pe = tl.load(kT_pe_ptrs, mask=mask_kT, other=0.0)
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
            if HAS_PE:
                qk += tl.dot(q_pe, kT_pe)
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
            scale_ds, descale_ds = _compute_fp8_scaling_factors(ds, FP8_MAX)
            dq += tl.dot((ds * scale_ds).to(kT.type.element_ty), tl.trans(kT)
                ) * descale_ds * descale_k
        else:
            dq += tl.dot(ds.to(kT.type.element_ty), tl.trans(kT))
            if HAS_PE:
                dq_pe += tl.dot(ds.to(kT_pe.type.element_ty), tl.trans(kT_pe))
        curr_n += step_n
        kT_ptrs += step_n * stride_kn
        if HAS_PE:
            kT_pe_ptrs += step_n * stride_kn
        vT_ptrs += step_n * stride_vn
    return dq, dq_pe


@triton.jit
def _bwd_preprocess(o_ptr, do_ptr, delta_ptr, stride_o_b, stride_o_h,
    stride_o_m, stride_o_k, stride_delta_b, stride_delta_h, stride_delta_m,
    stride_descale_do_z, cu_seqlens_q, max_seqlen_q, descale_do_ptr,
    BLOCK_M: tl.constexpr, BLOCK_D_MODEL: tl.constexpr, BLOCK_D_MODEL_POW2:
    tl.constexpr, IS_VARLEN: tl.constexpr, IS_FP8: tl.constexpr):
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
    offs_k = tl.arange(0, BLOCK_D_MODEL_POW2)
    offs = bid * stride_o_b + hid * stride_o_h + q_start * stride_o_m + offs_m[
        :, None] * stride_o_m + offs_k[None, :] * stride_o_k
    mask_m = offs_m < seqlen_q
    mask = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = BLOCK_D_MODEL != BLOCK_D_MODEL_POW2
    if PADDED_HEAD:
        mask &= offs_k[None, :] < BLOCK_D_MODEL
    o = tl.load(o_ptr + offs, mask=mask, other=0.0)
    do = tl.load(do_ptr + offs, mask=mask, other=0.0)
    if IS_FP8:
        descale_do = tl.load(descale_do_ptr + bid * stride_descale_do_z + hid)
        delta = tl.sum(o.to(tl.float32) * (do.to(tl.float32) * descale_do),
            axis=1)
    else:
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    offs_delta = (bid * stride_delta_b + hid * stride_delta_h + q_start *
        stride_delta_m + offs_m * stride_delta_m)
    tl.store(delta_ptr + offs_delta, delta, mask=mask_m)


@triton.jit
def bwd_kernel_causal(Q, K, V, sm_scale, DO, DQ, DK, DV, M, Delta,
    stride_qb_in, stride_qh_in, stride_qm_in, stride_qd_in, stride_kb_in,
    stride_kh_in, stride_kn_in, stride_kd_in, stride_vb_in, stride_vh_in,
    stride_vn_in, stride_vd_in, stride_dqb_in, stride_dqh_in, stride_dqm_in,
    stride_dqd_in, stride_dkb_in, stride_dkh_in, stride_dkn_in,
    stride_dkd_in, stride_dvb_in, stride_dvh_in, stride_dvn_in,
    stride_dvd_in, stride_deltab_in, stride_deltah_in, stride_deltam_in,
    stride_dob_in, stride_doh_in, stride_dom_in, stride_dod_in,
    stride_dropoutb_in, stride_dropouth_in, stride_dropoutm_in,
    stride_dropoutn_in, stride_descale_q_z_in, stride_descale_k_z_in,
    stride_descale_v_z_in, stride_descale_do_z_in, stride_az_in,
    stride_ah_in, HQ, HK, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k, Dropout_mask, dropout_p, philox_seed,
    philox_offset_base_in, Alibi_slopes, Descale_q, Descale_k, Descale_v,
    Descale_do, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, BLOCK_M2:
    tl.constexpr, BLOCK_N2: tl.constexpr, BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr, ACTUAL_HEAD_DIM: tl.constexpr, PE_HEAD_DIM: tl.
    constexpr, ENABLE_DROPOUT: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr, DEBUG_TRITON: tl.
    constexpr, DEBUG_TRITON_DETAIL: tl.constexpr, USE_INT64_STRIDES: tl.
    constexpr):
    if USE_INT64_STRIDES:
        stride_qb = tl.cast(stride_qb_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qd = tl.cast(stride_qd_in, tl.int64)
        stride_kb = tl.cast(stride_kb_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kd = tl.cast(stride_kd_in, tl.int64)
        stride_vb = tl.cast(stride_vb_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vd = tl.cast(stride_vd_in, tl.int64)
        stride_dqb = tl.cast(stride_dqb_in, tl.int64)
        stride_dqh = tl.cast(stride_dqh_in, tl.int64)
        stride_dqm = tl.cast(stride_dqm_in, tl.int64)
        stride_dqd = tl.cast(stride_dqd_in, tl.int64)
        stride_dkb = tl.cast(stride_dkb_in, tl.int64)
        stride_dkh = tl.cast(stride_dkh_in, tl.int64)
        stride_dkn = tl.cast(stride_dkn_in, tl.int64)
        stride_dkd = tl.cast(stride_dkd_in, tl.int64)
        stride_dvb = tl.cast(stride_dvb_in, tl.int64)
        stride_dvh = tl.cast(stride_dvh_in, tl.int64)
        stride_dvn = tl.cast(stride_dvn_in, tl.int64)
        stride_dvd = tl.cast(stride_dvd_in, tl.int64)
        stride_deltab = tl.cast(stride_deltab_in, tl.int64)
        stride_deltah = tl.cast(stride_deltah_in, tl.int64)
        stride_deltam = tl.cast(stride_deltam_in, tl.int64)
        stride_dob = tl.cast(stride_dob_in, tl.int64)
        stride_doh = tl.cast(stride_doh_in, tl.int64)
        stride_dom = tl.cast(stride_dom_in, tl.int64)
        stride_dod = tl.cast(stride_dod_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_dropoutb = tl.cast(stride_dropoutb_in, tl.int64)
        stride_dropouth = tl.cast(stride_dropouth_in, tl.int64)
        stride_dropoutm = tl.cast(stride_dropoutm_in, tl.int64)
        stride_dropoutn = tl.cast(stride_dropoutn_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
        stride_az = tl.cast(stride_az_in, tl.int64)
        stride_ah = tl.cast(stride_ah_in, tl.int64)
    else:
        stride_qb = stride_qb_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qd = stride_qd_in
        stride_kb = stride_kb_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kd = stride_kd_in
        stride_vb = stride_vb_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vd = stride_vd_in
        stride_dqb = stride_dqb_in
        stride_dqh = stride_dqh_in
        stride_dqm = stride_dqm_in
        stride_dqd = stride_dqd_in
        stride_dkb = stride_dkb_in
        stride_dkh = stride_dkh_in
        stride_dkn = stride_dkn_in
        stride_dkd = stride_dkd_in
        stride_dvb = stride_dvb_in
        stride_dvh = stride_dvh_in
        stride_dvn = stride_dvn_in
        stride_dvd = stride_dvd_in
        stride_deltab = stride_deltab_in
        stride_deltah = stride_deltah_in
        stride_deltam = stride_deltam_in
        stride_dob = stride_dob_in
        stride_doh = stride_doh_in
        stride_dom = stride_dom_in
        stride_dod = stride_dod_in
        philox_offset_base = philox_offset_base_in
        stride_dropoutb = stride_dropoutb_in
        stride_dropouth = stride_dropouth_in
        stride_dropoutm = stride_dropoutm_in
        stride_dropoutn = stride_dropoutn_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in
        stride_az = stride_az_in
        stride_ah = stride_ah_in
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)
    if DEBUG_TRITON:
        print(f'\npid: {pid}, bid: {bid}, hkid: {hkid}')
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
    delta_qk = seqlen_q - seqlen_k
    if DEBUG_TRITON:
        print(f'delta_qk = {delta_qk}')
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    HAS_PE: tl.constexpr = PE_HEAD_DIM > 0
    offs_d = tl.arange(0, HEAD_DIM)
    if HAS_PE:
        offs_d_pe = HEAD_DIM + tl.arange(0, PE_HEAD_DIM)
    GROUP_SIZE: tl.constexpr = HQ // HK
    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        if HAS_PE:
            dk_pe = tl.zeros([BLOCK_N1, PE_HEAD_DIM], dtype=tl.float32)
        else:
            dk_pe = dk
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        start_delta_q_gt_k = delta_qk
        num_blocks_skip = -delta_qk // BLOCK_N1
        delta_aligned = (num_blocks_skip + 1) * BLOCK_N1 + delta_qk
        start_delta_q_lt_k = delta_aligned // BLOCK_M1 * BLOCK_M1
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
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        mask_kv = offs_n[:, None] < seqlen_k
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_kv &= mask_d[None, :]
        adj_k = (bid * stride_kb + hkid * stride_kh + k_start * stride_kn +
            offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        if HAS_PE:
            adj_k_pe = (bid * stride_kb + hkid * stride_kh + k_start *
                stride_kn + offs_n[:, None] * stride_kn + offs_d_pe[None, :
                ] * stride_kd)
        adj_v = (bid * stride_vb + hkid * stride_vh + k_start * stride_vn +
            offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
        if HAS_PE:
            k_pe = tl.load(K + adj_k_pe, mask=mask_kv, other=0.0)
        else:
            k_pe = None
        v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            if delta_qk >= 0:
                start_m = start_n + start_delta
                len_m = BLOCK_N1
            else:
                start_m = max(start_n + delta_qk, 0)
                start_m = start_m // BLOCK_M1 * BLOCK_M1
                residue_m = max(start_n + delta_qk - start_m, 0)
                len_m = BLOCK_N1 + residue_m
                if DEBUG_TRITON:
                    print(f'residue_m = {residue_m}')
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = (bid * stride_dob + hqid * stride_doh + q_start *
                stride_dom)
            DO_ptr = DO + adj_do
            adj_delta = (bid * stride_deltab + hqid * stride_deltah + 
                q_start * stride_deltam)
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
                dropout_offset = (Dropout_mask + bid * stride_dropoutb + 
                    hqid * stride_dropouth)
            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid
                    )
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid
                    )
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid
                    )
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z +
                    hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = (1.0, 1.0, 
                    1.0, 1.0)
            MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
            len_m = min(len_m, seqlen_q)
            num_steps = tl.cdiv(len_m, MASK_BLOCK_M1)
            if pid < num_blocks_skip:
                num_steps = 0
            if DEBUG_TRITON:
                print(
                    f'Masked: start_n: {start_n}; start_m: {start_m}, num_steps: {num_steps}'
                    )
            dk, dk_pe, dv = _bwd_dkdv_inner(dk, dk_pe, dv, Q_ptr, k, k_pe,
                v, DO_ptr, M_ptr, Delta_ptr, sm_scale, stride_qm, stride_qd,
                stride_dom, stride_dod, stride_dropoutm, stride_dropoutn,
                stride_deltam, MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
                ACTUAL_HEAD_DIM, PE_HEAD_DIM, dropout_p, philox_seed,
                batch_philox_offset, dropout_offset, alibi_slope, seqlen_q,
                seqlen_k, start_n, start_m, num_steps, descale_q, descale_k,
                descale_v, descale_do, MASK=True, ENABLE_DROPOUT=
                ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
            start_m += num_steps * MASK_BLOCK_M1
            num_steps = tl.cdiv(seqlen_q - start_m, BLOCK_M1)
            end_m = start_m + num_steps * BLOCK_M1
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
            dk, dk_pe, dv = _bwd_dkdv_inner(dk, dk_pe, dv, Q_ptr, k, k_pe,
                v, DO_ptr, M_ptr, Delta_ptr, sm_scale, stride_qm, stride_qd,
                stride_dom, stride_dod, stride_dropoutm, stride_dropoutn,
                stride_deltam, BLOCK_M1, BLOCK_N1, HEAD_DIM,
                ACTUAL_HEAD_DIM, PE_HEAD_DIM, dropout_p, philox_seed,
                batch_philox_offset, dropout_offset, alibi_slope, seqlen_q,
                seqlen_k, start_n, start_m, num_steps, descale_q, descale_k,
                descale_v, descale_do, MASK=False, ENABLE_DROPOUT=
                ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_kv)
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_kv)
        if HAS_PE:
            offs_dk_pe = offs_n[:, None] * stride_dkn + offs_d_pe[None, :
                ] * stride_dkd
            dk_pe *= sm_scale
            tl.store(DK + adj_dk + offs_dk_pe, dk_pe, mask=mask_kv)
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        if DEBUG_TRITON:
            print(
                f'end_n = start_m + BLOCK_M = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2}'
                )
        if start_m + BLOCK_M2 < delta_qk:
            if DEBUG_TRITON:
                print(
                    f'start_m + BLOCK_M2 = {start_m} + {BLOCK_M2} = {start_m + BLOCK_M2} < delta_qk of {delta_qk}'
                    )
            return
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        mask_q = offs_m[:, None] < seqlen_q
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_q &= mask_d[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        if HAS_PE:
            offs_q_pe = offs_m[:, None] * stride_qm + offs_d_pe[None, :
                ] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        K += bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V += bid * stride_vb + hkid * stride_vh + k_start * stride_vn
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            end_n = start_m + BLOCK_M2 - delta_qk
            end_n = max(min(end_n, seqlen_k), 0)
            if DEBUG_TRITON:
                print(f'delta_qk: {delta_qk}; end_n: {end_n}')
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = (bid * stride_dob + hqid * stride_doh + q_start *
                stride_dom)
            adj_delta = (bid * stride_deltab + hqid * stride_deltah + 
                q_start * stride_deltam)
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
                dropout_offset = (Dropout_mask + bid * stride_dropoutb + 
                    hqid * stride_dropouth)
            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            if HAS_PE:
                q_pe = tl.load(Q + adj_q + offs_q_pe, mask=mask_q, other=0.0)
            else:
                q_pe = None
            do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
            m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=offs_m <
                seqlen_q)
            m = m[:, None]
            MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
            start_n = max(end_n - BLOCK_M2, 0)
            num_steps = tl.cdiv(end_n - start_n, MASK_BLOCK_N2)
            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid
                    )
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid
                    )
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid
                    )
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z +
                    hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = (1.0, 1.0, 
                    1.0, 1.0)
            dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
            if HAS_PE:
                dq_pe = tl.zeros([BLOCK_M2, PE_HEAD_DIM], dtype=tl.float32)
            else:
                dq_pe = dq
            dq, dq_pe = _bwd_dq_inner(dq, dq_pe, q, q_pe, K, V, do, m,
                Delta_ptr, sm_scale, stride_qm, stride_qd, stride_kn,
                stride_kd, stride_vn, stride_vd, stride_dropoutm,
                stride_dropoutn, stride_deltam, seqlen_q, seqlen_k,
                BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM, ACTUAL_HEAD_DIM,
                PE_HEAD_DIM, dropout_p, philox_seed, batch_philox_offset,
                dropout_offset, alibi_slope, start_m, start_n, end_n,
                num_steps, descale_q, descale_k, descale_v, descale_do,
                MASK=True, ENABLE_DROPOUT=ENABLE_DROPOUT, USE_ALIBI=
                USE_ALIBI, USE_EXP2=USE_EXP2, IS_FP8=IS_FP8, FP8_MAX=
                FP8_MAX, DEBUG_TRITON=DEBUG_TRITON, DEBUG_TRITON_DETAIL=
                DEBUG_TRITON_DETAIL)
            end_n -= num_steps * MASK_BLOCK_N2
            num_steps = tl.cdiv(end_n, BLOCK_N2)
            start_n = max(end_n - num_steps * BLOCK_N2, 0)
            if DEBUG_TRITON:
                print(
                    f'unMasked: start_m: {start_m}, start_n: {start_n}, end_n: {end_n}, num_steps: {num_steps}'
                    )
            dq, dq_pe = _bwd_dq_inner(dq, dq_pe, q, q_pe, K, V, do, m,
                Delta_ptr, sm_scale, stride_qm, stride_qd, stride_kn,
                stride_kd, stride_vn, stride_vd, stride_dropoutm,
                stride_dropoutn, stride_deltam, seqlen_q, seqlen_k,
                BLOCK_M2, BLOCK_N2, HEAD_DIM, ACTUAL_HEAD_DIM, PE_HEAD_DIM,
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,
                alibi_slope, start_m, start_n, end_n, num_steps, descale_q,
                descale_k, descale_v, descale_do, MASK=False,
                ENABLE_DROPOUT=ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2, IS_FP8=IS_FP8, FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON, DEBUG_TRITON_DETAIL=
                DEBUG_TRITON_DETAIL)
            adj_dq = (bid * stride_dqb + hqid * stride_dqh + q_start *
                stride_dqm)
            offs_dq = offs_m[:, None] * stride_dqm + offs_d[None, :
                ] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)
            if HAS_PE:
                offs_dq_pe = offs_m[:, None] * stride_dqm + offs_d_pe[None, :
                    ] * stride_dqd
                dq_pe *= sm_scale
                tl.store(DQ + adj_dq + offs_dq_pe, dq_pe, mask=mask_q)


@triton.jit
def bwd_kernel_noncausal(Q, K, V, sm_scale, DO, DQ, DK, DV, M, Delta,
    stride_qb_in, stride_qh_in, stride_qm_in, stride_qd_in, stride_kb_in,
    stride_kh_in, stride_kn_in, stride_kd_in, stride_vb_in, stride_vh_in,
    stride_vn_in, stride_vd_in, stride_dqb_in, stride_dqh_in, stride_dqm_in,
    stride_dqd_in, stride_dkb_in, stride_dkh_in, stride_dkn_in,
    stride_dkd_in, stride_dvb_in, stride_dvh_in, stride_dvn_in,
    stride_dvd_in, stride_deltab_in, stride_deltah_in, stride_deltam_in,
    stride_dob_in, stride_doh_in, stride_dom_in, stride_dod_in,
    stride_dropoutb_in, stride_dropouth_in, stride_dropoutm_in,
    stride_dropoutn_in, stride_descale_q_z_in, stride_descale_k_z_in,
    stride_descale_v_z_in, stride_descale_do_z_in, stride_az_in,
    stride_ah_in, HQ, HK, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
    max_seqlen_k, Dropout_mask, dropout_p, philox_seed,
    philox_offset_base_in, Alibi_slopes, Descale_q, Descale_k, Descale_v,
    Descale_do, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, BLOCK_M2:
    tl.constexpr, BLOCK_N2: tl.constexpr, BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr, ACTUAL_HEAD_DIM: tl.constexpr, PE_HEAD_DIM: tl.
    constexpr, ENABLE_DROPOUT: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_ALIBI: tl.constexpr, USE_EXP2: tl.constexpr, IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr, FP8_OUTPUT: tl.constexpr, DEBUG_TRITON: tl.
    constexpr, DEBUG_TRITON_DETAIL: tl.constexpr, USE_INT64_STRIDES: tl.
    constexpr):
    if USE_INT64_STRIDES:
        stride_qb = tl.cast(stride_qb_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qd = tl.cast(stride_qd_in, tl.int64)
        stride_kb = tl.cast(stride_kb_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kd = tl.cast(stride_kd_in, tl.int64)
        stride_vb = tl.cast(stride_vb_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vd = tl.cast(stride_vd_in, tl.int64)
        stride_dqb = tl.cast(stride_dqb_in, tl.int64)
        stride_dqh = tl.cast(stride_dqh_in, tl.int64)
        stride_dqm = tl.cast(stride_dqm_in, tl.int64)
        stride_dqd = tl.cast(stride_dqd_in, tl.int64)
        stride_dkb = tl.cast(stride_dkb_in, tl.int64)
        stride_dkh = tl.cast(stride_dkh_in, tl.int64)
        stride_dkn = tl.cast(stride_dkn_in, tl.int64)
        stride_dkd = tl.cast(stride_dkd_in, tl.int64)
        stride_dvb = tl.cast(stride_dvb_in, tl.int64)
        stride_dvh = tl.cast(stride_dvh_in, tl.int64)
        stride_dvn = tl.cast(stride_dvn_in, tl.int64)
        stride_dvd = tl.cast(stride_dvd_in, tl.int64)
        stride_deltab = tl.cast(stride_deltab_in, tl.int64)
        stride_deltah = tl.cast(stride_deltah_in, tl.int64)
        stride_deltam = tl.cast(stride_deltam_in, tl.int64)
        stride_dob = tl.cast(stride_dob_in, tl.int64)
        stride_doh = tl.cast(stride_doh_in, tl.int64)
        stride_dom = tl.cast(stride_dom_in, tl.int64)
        stride_dod = tl.cast(stride_dod_in, tl.int64)
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_dropoutb = tl.cast(stride_dropoutb_in, tl.int64)
        stride_dropouth = tl.cast(stride_dropouth_in, tl.int64)
        stride_dropoutm = tl.cast(stride_dropoutm_in, tl.int64)
        stride_dropoutn = tl.cast(stride_dropoutn_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
            stride_descale_do_z = tl.cast(stride_descale_do_z_in, tl.int64)
        stride_az = tl.cast(stride_az_in, tl.int64)
        stride_ah = tl.cast(stride_ah_in, tl.int64)
    else:
        stride_qb = stride_qb_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qd = stride_qd_in
        stride_kb = stride_kb_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kd = stride_kd_in
        stride_vb = stride_vb_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vd = stride_vd_in
        stride_dqb = stride_dqb_in
        stride_dqh = stride_dqh_in
        stride_dqm = stride_dqm_in
        stride_dqd = stride_dqd_in
        stride_dkb = stride_dkb_in
        stride_dkh = stride_dkh_in
        stride_dkn = stride_dkn_in
        stride_dkd = stride_dkd_in
        stride_dvb = stride_dvb_in
        stride_dvh = stride_dvh_in
        stride_dvn = stride_dvn_in
        stride_dvd = stride_dvd_in
        stride_deltab = stride_deltab_in
        stride_deltah = stride_deltah_in
        stride_deltam = stride_deltam_in
        stride_dob = stride_dob_in
        stride_doh = stride_doh_in
        stride_dom = stride_dom_in
        stride_dod = stride_dod_in
        philox_offset_base = philox_offset_base_in
        stride_dropoutb = stride_dropoutb_in
        stride_dropouth = stride_dropouth_in
        stride_dropoutm = stride_dropoutm_in
        stride_dropoutn = stride_dropoutn_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_descale_do_z = stride_descale_do_z_in
        stride_az = stride_az_in
        stride_ah = stride_ah_in
    hkid = tl.program_id(0)
    pid = tl.program_id(1)
    bid = tl.program_id(2)
    if DEBUG_TRITON:
        print(f'\npid: {pid}, bid: {bid}, hkid: {hkid}')
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
    PADDED_HEAD: tl.constexpr = ACTUAL_HEAD_DIM != HEAD_DIM
    HAS_PE: tl.constexpr = PE_HEAD_DIM > 0
    offs_d = tl.arange(0, HEAD_DIM)
    if HAS_PE:
        offs_d_pe = HEAD_DIM + tl.arange(0, PE_HEAD_DIM)
    GROUP_SIZE: tl.constexpr = HQ // HK
    start_n = pid * BLOCK_N1
    if start_n < seqlen_k:
        dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        if HAS_PE:
            dk_pe = tl.zeros([BLOCK_N1, PE_HEAD_DIM], dtype=tl.float32)
        else:
            dk_pe = dk
        dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
        offs_n = start_n + tl.arange(0, BLOCK_N1)
        mask_kv = offs_n[:, None] < seqlen_k
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_kv &= mask_d[None, :]
        adj_k = (bid * stride_kb + hkid * stride_kh + k_start * stride_kn +
            offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        if HAS_PE:
            adj_k_pe = (bid * stride_kb + hkid * stride_kh + k_start *
                stride_kn + offs_n[:, None] * stride_kn + offs_d_pe[None, :
                ] * stride_kd)
        adj_v = (bid * stride_vb + hkid * stride_vh + k_start * stride_vn +
            offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        k = tl.load(K + adj_k, mask=mask_kv, other=0.0)
        if HAS_PE:
            k_pe = tl.load(K + adj_k_pe, mask=mask_kv, other=0.0)
        else:
            k_pe = None
        v = tl.load(V + adj_v, mask=mask_kv, other=0.0)
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            Q_ptr = Q + adj_q
            adj_do = (bid * stride_dob + hqid * stride_doh + q_start *
                stride_dom)
            DO_ptr = DO + adj_do
            adj_delta = (bid * stride_deltab + hqid * stride_deltah + 
                q_start * stride_deltam)
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
                dropout_offset = (Dropout_mask + bid * stride_dropoutb + 
                    hqid * stride_dropouth)
            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid
                    )
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid
                    )
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid
                    )
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z +
                    hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = (1.0, 1.0, 
                    1.0, 1.0)
            start_m = 0
            num_steps = tl.cdiv(seqlen_q, BLOCK_M1)
            dk, dk_pe, dv = _bwd_dkdv_inner(dk, dk_pe, dv, Q_ptr, k, k_pe,
                v, DO_ptr, M_ptr, Delta_ptr, sm_scale, stride_qm, stride_qd,
                stride_dom, stride_dod, stride_dropoutm, stride_dropoutn,
                stride_deltam, BLOCK_M1, BLOCK_N1, HEAD_DIM,
                ACTUAL_HEAD_DIM, PE_HEAD_DIM, dropout_p, philox_seed,
                batch_philox_offset, dropout_offset, alibi_slope, seqlen_q,
                seqlen_k, start_n, start_m, num_steps, descale_q, descale_k,
                descale_v, descale_do, MASK=False, ENABLE_DROPOUT=
                ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI, USE_EXP2=USE_EXP2,
                IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, DEBUG_TRITON=DEBUG_TRITON,
                DEBUG_TRITON_DETAIL=DEBUG_TRITON_DETAIL)
        adj_dv = bid * stride_dvb + hkid * stride_dvh + k_start * stride_dvn
        offs_dv = offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
        tl.store(DV + adj_dv + offs_dv, dv, mask=mask_kv)
        adj_dk = bid * stride_dkb + hkid * stride_dkh + k_start * stride_dkn
        offs_dk = offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
        dk *= sm_scale
        tl.store(DK + adj_dk + offs_dk, dk, mask=mask_kv)
        if HAS_PE:
            offs_dk_pe = offs_n[:, None] * stride_dkn + offs_d_pe[None, :
                ] * stride_dkd
            dk_pe *= sm_scale
            tl.store(DK + adj_dk + offs_dk_pe, dk_pe, mask=mask_kv)
    start_m = pid * BLOCK_M2
    if start_m < seqlen_q:
        offs_m = start_m + tl.arange(0, BLOCK_M2)
        mask_q = offs_m[:, None] < seqlen_q
        if PADDED_HEAD:
            mask_d = offs_d < ACTUAL_HEAD_DIM
            mask_q &= mask_d[None, :]
        offs_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        if HAS_PE:
            offs_q_pe = offs_m[:, None] * stride_qm + offs_d_pe[None, :
                ] * stride_qd
        offs_do = offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        K += bid * stride_kb + hkid * stride_kh + k_start * stride_kn
        V += bid * stride_vb + hkid * stride_vh + k_start * stride_vn
        for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
            adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
            adj_do = (bid * stride_dob + hqid * stride_doh + q_start *
                stride_dom)
            adj_delta = (bid * stride_deltab + hqid * stride_deltah + 
                q_start * stride_deltam)
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
                dropout_offset = (Dropout_mask + bid * stride_dropoutb + 
                    hqid * stride_dropouth)
            q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
            if HAS_PE:
                q_pe = tl.load(Q + adj_q + offs_q_pe, mask=mask_q, other=0.0)
            else:
                q_pe = None
            do = tl.load(DO + adj_do + offs_do, mask=mask_q, other=0.0)
            m = tl.load(M + adj_delta + offs_m * stride_deltam, mask=offs_m <
                seqlen_q)
            m = m[:, None]
            if IS_FP8:
                descale_q = tl.load(Descale_q + bid * stride_descale_q_z + hqid
                    )
                descale_k = tl.load(Descale_k + bid * stride_descale_k_z + hkid
                    )
                descale_v = tl.load(Descale_v + bid * stride_descale_v_z + hkid
                    )
                descale_do = tl.load(Descale_do + bid * stride_descale_do_z +
                    hqid)
            else:
                descale_q, descale_k, descale_v, descale_do = (1.0, 1.0, 
                    1.0, 1.0)
            start_n = 0
            end_n = seqlen_k
            num_steps = tl.cdiv(seqlen_k, BLOCK_N2)
            dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
            if HAS_PE:
                dq_pe = tl.zeros([BLOCK_M2, PE_HEAD_DIM], dtype=tl.float32)
            else:
                dq_pe = dq
            dq, dq_pe = _bwd_dq_inner(dq, dq_pe, q, q_pe, K, V, do, m,
                Delta_ptr, sm_scale, stride_qm, stride_qd, stride_kn,
                stride_kd, stride_vn, stride_vd, stride_dropoutm,
                stride_dropoutn, stride_deltam, seqlen_q, seqlen_k,
                BLOCK_M2, BLOCK_N2, HEAD_DIM, ACTUAL_HEAD_DIM, PE_HEAD_DIM,
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,
                alibi_slope, start_m, start_n, end_n, num_steps, descale_q,
                descale_k, descale_v, descale_do, MASK=False,
                ENABLE_DROPOUT=ENABLE_DROPOUT, USE_ALIBI=USE_ALIBI,
                USE_EXP2=USE_EXP2, IS_FP8=IS_FP8, FP8_MAX=FP8_MAX,
                DEBUG_TRITON=DEBUG_TRITON, DEBUG_TRITON_DETAIL=
                DEBUG_TRITON_DETAIL)
            adj_dq = (bid * stride_dqb + hqid * stride_dqh + q_start *
                stride_dqm)
            offs_dq = offs_m[:, None] * stride_dqm + offs_d[None, :
                ] * stride_dqd
            dq *= sm_scale
            tl.store(DQ + adj_dq + offs_dq, dq, mask=mask_q)
            if HAS_PE:
                offs_dq_pe = offs_m[:, None] * stride_dqm + offs_d_pe[None, :
                    ] * stride_dqd
                dq_pe *= sm_scale
                tl.store(DQ + adj_dq + offs_dq_pe, dq_pe, mask=mask_q)


def flash_attn_fused_backward(do: torch.Tensor, q: torch.Tensor, k: torch.
    Tensor, v: torch.Tensor, o: torch.Tensor, softmax_lse: torch.Tensor, dq:
    torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, dbias: torch.Tensor,
    sm_scale: float, alibi_slopes: Optional[torch.Tensor], causal: bool,
    cu_seqlens_q: Optional[torch.Tensor], cu_seqlens_k: Optional[torch.
    Tensor], max_seqlen_q: int, max_seqlen_k: int, dropout_p: float,
    philox_seed: Optional[int]=0, philox_offset: Optional[int]=0, descale_q:
    Optional[torch.Tensor]=None, descale_k: Optional[torch.Tensor]=None,
    descale_v: Optional[torch.Tensor]=None, descale_do: Optional[torch.
    Tensor]=None, USE_INT64_STRIDES: Optional[bool]=False, config: Optional
    [Dict[str, any]]=None):
    _LOGGER.info(
        f'FLASH_ATTN_FUSED_BKWD: do={tuple(do.shape)} q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} '
         + f'dq={tuple(dq.shape)}  dk={tuple(dk.shape)}  dv={tuple(dv.shape)}')
    if dbias is not None:
        raise ValueError('Bias is not supported yet in the Triton Backend')
    if q.shape[-1] == k.shape[-1] and k.shape[-1] > v.shape[-1]:
        raise ValueError(
            "'Fused' backward doesn't support Positional Encoding (PE). Please use 'one kernel' backward implementation for PE."
            )
    IS_FP8 = _is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        descale_strides = descale_q.stride(0), descale_k.stride(0
            ), descale_v.stride(0), descale_do.stride(0)
    else:
        FP8_MAX = None
        (stride_descale_q_z) = (stride_descale_k_z) = (stride_descale_v_z) = (
            stride_descale_do_z) = None
        descale_strides = (stride_descale_q_z, stride_descale_k_z,
            stride_descale_v_z, stride_descale_do_z)
    IS_VARLEN = True if cu_seqlens_q is not None else False
    if IS_VARLEN:
        batch, seqlen_q, num_q_heads, head_sz = len(cu_seqlens_q
            ) - 1, max_seqlen_q, q.shape[1], q.shape[2]
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = 0, q.stride(1), q.stride(0), q.stride(2)
        q_strides = 0, q.stride(1), q.stride(0), q.stride(2)
        k_strides = 0, k.stride(1), k.stride(0), k.stride(2)
        v_strides = 0, v.stride(1), v.stride(0), v.stride(2)
        o_strides = 0, o.stride(1), o.stride(0), o.stride(2)
        dq_strides = 0, dq.stride(1), dq.stride(0), dq.stride(2)
        dk_strides = 0, dk.stride(1), dk.stride(0), dk.stride(2)
        do_strides = 0, do.stride(1), do.stride(0), do.stride(2)
    else:
        batch, seqlen_q, num_q_heads, head_sz = q.shape
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = q.stride(0), q.stride(2), q.stride(1), q.stride(3)
        k_strides = k.stride(0), k.stride(2), k.stride(1), k.stride(3)
        v_strides = v.stride(0), v.stride(2), v.stride(1), v.stride(3)
        o_strides = o.stride(0), o.stride(2), o.stride(1), o.stride(3)
        dq_strides = dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3)
        dk_strides = dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3)
        do_strides = do.stride(0), do.stride(2), do.stride(1), do.stride(3)
    BLOCK_D_MODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_D_MODEL_POW2 = max(BLOCK_D_MODEL_POW2, 16)
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        delta_strides = 0, delta.stride(1), delta.stride(0)
    else:
        delta_strides = delta.stride()
    if config is None:
        config = _get_config()
    pre_grid = triton.cdiv(max_seqlen_q, config['preprocess_kernel'][
        'PRE_BLOCK']), batch, num_q_heads
    _bwd_preprocess[pre_grid](o, do, delta, *o_strides, *delta_strides,
        descale_strides[3], cu_seqlens_q, max_seqlen_q, descale_do, BLOCK_M
        =config['preprocess_kernel']['PRE_BLOCK'], BLOCK_D_MODEL=head_sz,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2, IS_VARLEN=IS_VARLEN, IS_FP8=
        IS_FP8)
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = torch.zeros((batch, num_q_heads, max_seqlen_q,
            max_seqlen_k), device=q.device, dtype=torch.float32)
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = 0, 0, 0, 0
    if BLOCK_D_MODEL_POW2 > 160 or q.dtype == torch.float32:
        config_dkdvdq = config['dkdvdq_kernel_N64']
    else:
        config_dkdvdq = config['dkdvdq_kernel_N128']
    num_k_pids = (max_seqlen_k + config_dkdvdq['BLOCK_N'] - 1
        ) // config_dkdvdq['BLOCK_N']
    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
    if causal:
        grid_dkdvdq = batch * num_q_heads * num_k_pids,
        _bwd_kernel_dkdvdq_causal[grid_dkdvdq](q, k, v, sm_scale, do, dk,
            dv, dq, softmax_lse, delta, *q_strides, *k_strides, *v_strides,
            *dk_strides, *dq_strides, *delta_strides, *do_strides, *
            dropout_strides, *descale_strides, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, dropout_mask, dropout_p,
            philox_seed, philox_offset, descale_q, descale_k, descale_v,
            descale_do, NUM_Q_HEADS=num_q_heads, NUM_K_HEADS=num_k_heads,
            BATCH=batch, NUM_K_PIDS=num_k_pids, BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2, ENABLE_DROPOUT=
            use_dropout, IS_VARLEN=IS_VARLEN, IS_FP8=IS_FP8, FP8_MAX=
            FP8_MAX, NUM_SMS=NUM_SMS, USE_INT64_STRIDES=USE_INT64_STRIDES,
            NUM_XCD=get_num_xcds(), **config_dkdvdq)
    else:
        grid_dkdvdq = batch * num_k_heads * num_k_pids,
        _bwd_kernel_dkdvdq_noncausal[grid_dkdvdq](q, k, v, sm_scale, do, dk,
            dv, dq, softmax_lse, delta, *q_strides, *k_strides, *v_strides,
            *dk_strides, *dq_strides, *delta_strides, *do_strides, *
            dropout_strides, *descale_strides, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, dropout_mask, dropout_p,
            philox_seed, philox_offset, descale_q, descale_k, descale_v,
            descale_do, NUM_Q_HEADS=num_q_heads, NUM_K_HEADS=num_k_heads,
            BATCH=batch, NUM_K_PIDS=num_k_pids, BLOCK_D_MODEL=head_sz,
            BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2, ENABLE_DROPOUT=
            use_dropout, IS_VARLEN=IS_VARLEN, IS_FP8=IS_FP8, FP8_MAX=
            FP8_MAX, NUM_SMS=NUM_SMS, USE_INT64_STRIDES=USE_INT64_STRIDES,
            **config_dkdvdq)
    return delta


def flash_attn_onekernel_backward(do: torch.Tensor, q: torch.Tensor, k:
    torch.Tensor, v: torch.Tensor, o: torch.Tensor, softmax_lse: torch.
    Tensor, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor, dbias:
    torch.Tensor, sm_scale: float, alibi_slopes: Optional[torch.Tensor],
    causal: bool, cu_seqlens_q: Optional[torch.Tensor], cu_seqlens_k:
    Optional[torch.Tensor], max_seqlen_q: int, max_seqlen_k: int, dropout_p:
    float, philox_seed: Optional[int]=0, philox_offset: Optional[int]=0,
    descale_q: Optional[torch.Tensor]=None, descale_k: Optional[torch.
    Tensor]=None, descale_v: Optional[torch.Tensor]=None, descale_do:
    Optional[torch.Tensor]=None, USE_INT64_STRIDES: Optional[bool]=False,
    config: Optional[Dict[str, any]]=None):
    _LOGGER.info(
        f'FLASH_ATTN_ONEKERNEL_BKWD: do={tuple(do.shape)} q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} '
         + f'dq={tuple(dq.shape)}  dk={tuple(dk.shape)}  dv={tuple(dv.shape)}')
    if dbias is not None:
        raise ValueError('Bias is not supported yet in the Triton Backend')
    use_alibi, (stride_az, stride_ah) = (True, alibi_slopes.stride()
        ) if alibi_slopes is not None else (False, (0, 0))
    IS_FP8 = _is_fp8(q)
    if IS_FP8:
        FP8_MAX = torch.finfo(q.dtype).max
        descale_strides = descale_q.stride(0), descale_k.stride(0
            ), descale_v.stride(0), descale_do.stride(0)
    else:
        FP8_MAX = None
        (stride_descale_q_z) = (stride_descale_k_z) = (stride_descale_v_z) = (
            stride_descale_do_z) = None
        descale_strides = (stride_descale_q_z, stride_descale_k_z,
            stride_descale_v_z, stride_descale_do_z)
    IS_VARLEN = True if cu_seqlens_q is not None else False
    if IS_VARLEN:
        batch, seqlen_q, num_q_heads = len(cu_seqlens_q
            ) - 1, max_seqlen_q, q.shape[1]
        _, num_k_heads = max_seqlen_k, k.shape[1]
        q_strides = 0, q.stride(1), q.stride(0), q.stride(2)
        q_strides = 0, q.stride(1), q.stride(0), q.stride(2)
        k_strides = 0, k.stride(1), k.stride(0), k.stride(2)
        v_strides = 0, v.stride(1), v.stride(0), v.stride(2)
        o_strides = 0, o.stride(1), o.stride(0), o.stride(2)
        dq_strides = 0, dq.stride(1), dq.stride(0), dq.stride(2)
        dk_strides = 0, dk.stride(1), dk.stride(0), dk.stride(2)
        dv_strides = 0, dv.stride(1), dv.stride(0), dv.stride(2)
        do_strides = 0, do.stride(1), do.stride(0), do.stride(2)
    else:
        batch, seqlen_q, num_q_heads = q.shape[:-1]
        _, num_k_heads = k.shape[1], k.shape[2]
        q_strides = q.stride(0), q.stride(2), q.stride(1), q.stride(3)
        k_strides = k.stride(0), k.stride(2), k.stride(1), k.stride(3)
        v_strides = v.stride(0), v.stride(2), v.stride(1), v.stride(3)
        o_strides = o.stride(0), o.stride(2), o.stride(1), o.stride(3)
        dq_strides = dq.stride(0), dq.stride(2), dq.stride(1), dq.stride(3)
        dk_strides = dk.stride(0), dk.stride(2), dk.stride(1), dk.stride(3)
        dv_strides = dv.stride(0), dv.stride(2), dv.stride(1), dv.stride(3)
        do_strides = do.stride(0), do.stride(2), do.stride(1), do.stride(3)
    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    BLOCK_D_MODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_D_MODEL_PE_POW2 = 0 if pe_head_dim == 0 else max(triton.
        next_power_of_2(pe_head_dim), 16)
    assert pe_head_dim == 0 and BLOCK_D_MODEL_PE_POW2 == 0 or v_head_dim == BLOCK_D_MODEL_POW2 and pe_head_dim == BLOCK_D_MODEL_PE_POW2, 'Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2.'
    assert not IS_FP8 or IS_FP8 and pe_head_dim == 0, "Positional encoding doesn't support FP8."
    if config is None:
        config = _get_config()
    delta = torch.zeros_like(softmax_lse)
    if IS_VARLEN:
        delta_strides = 0, delta.stride(1), delta.stride(0)
    else:
        delta_strides = delta.stride()
    pre_grid = triton.cdiv(max_seqlen_q, config['preprocess_kernel'][
        'PRE_BLOCK']), batch, num_q_heads
    _bwd_preprocess[pre_grid](o, do, delta, *o_strides, *delta_strides,
        descale_strides[3], cu_seqlens_q, max_seqlen_q, descale_do, BLOCK_M
        =config['preprocess_kernel']['PRE_BLOCK'], BLOCK_D_MODEL=v_head_dim,
        BLOCK_D_MODEL_POW2=BLOCK_D_MODEL_POW2, IS_VARLEN=IS_VARLEN, IS_FP8=
        IS_FP8)
    use_dropout = dropout_p > 0.0
    if use_dropout:
        dropout_mask = torch.zeros((batch, num_q_heads, max_seqlen_q,
            max_seqlen_k), device=q.device, dtype=torch.float32)
        dropout_strides = dropout_mask.stride()
    else:
        dropout_mask = None
        dropout_strides = 0, 0, 0, 0
    seqlen = max(max_seqlen_q, max_seqlen_k)
    config_onekernel = config['onekernel_pe'
        ] if pe_head_dim > 0 and causal and 'onekernel_pe' in config else config[
        'onekernel']
    grid = num_k_heads, triton.cdiv(seqlen, config_onekernel['BLOCK_N1']
        ), batch
    if causal:
        bwd_kernel_causal[grid](q, k, v, sm_scale, do, dq, dk, dv,
            softmax_lse, delta, *q_strides, *k_strides, *v_strides, *
            dq_strides, *dk_strides, *dv_strides, *delta_strides, *
            do_strides, *dropout_strides, *descale_strides, stride_az,
            stride_ah, num_q_heads, num_k_heads, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, dropout_mask, dropout_p,
            philox_seed, philox_offset, alibi_slopes, descale_q, descale_k,
            descale_v, descale_do, HEAD_DIM=v_head_dim, ACTUAL_HEAD_DIM=
            BLOCK_D_MODEL_POW2, PE_HEAD_DIM=pe_head_dim, ENABLE_DROPOUT=
            use_dropout, IS_VARLEN=IS_VARLEN, USE_ALIBI=use_alibi, USE_EXP2
            =True, IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=False,
            DEBUG_TRITON=False, DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES, **config_onekernel)
    else:
        bwd_kernel_noncausal[grid](q, k, v, sm_scale, do, dq, dk, dv,
            softmax_lse, delta, *q_strides, *k_strides, *v_strides, *
            dq_strides, *dk_strides, *dv_strides, *delta_strides, *
            do_strides, *dropout_strides, *descale_strides, stride_az,
            stride_ah, num_q_heads, num_k_heads, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, dropout_mask, dropout_p,
            philox_seed, philox_offset, alibi_slopes, descale_q, descale_k,
            descale_v, descale_do, HEAD_DIM=v_head_dim, ACTUAL_HEAD_DIM=
            BLOCK_D_MODEL_POW2, PE_HEAD_DIM=pe_head_dim, ENABLE_DROPOUT=
            use_dropout, IS_VARLEN=IS_VARLEN, USE_ALIBI=use_alibi, USE_EXP2
            =True, IS_FP8=IS_FP8, FP8_MAX=FP8_MAX, FP8_OUTPUT=False,
            DEBUG_TRITON=False, DEBUG_TRITON_DETAIL=False,
            USE_INT64_STRIDES=USE_INT64_STRIDES, **config_onekernel)
    return delta


@functools.lru_cache(maxsize=1024)
def _get_config():
    if not hasattr(_get_config, '_config_dict'):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f'{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json'
        with open(fpath, 'r') as file:
            config = json.load(file)
        _get_config._config_dict = config
    return _get_config._config_dict['bkwd_fused']


@functools.lru_cache(maxsize=1024)
def _get_config():
    if not hasattr(_get_config, '_config_dict'):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f'{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json'
        with open(fpath, 'r') as file:
            config = json.load(file)
        _get_config._config_dict = config
    return _get_config._config_dict['bkwd_onekernel']


def info(self, msg):
    self._logger.info(msg)


# Backward method (kernel launch code)
def __FlashAttnFP8Func_backward(ctx, do, *args):
    (q_fp8, k_fp8, v_fp8, out, softmax_lse, descale_q, descale_k, descale_v
        ) = ctx.saved_tensors
    dq, dk, dv = torch.zeros_like(q_fp8, dtype=torch.float32
        ), torch.zeros_like(k_fp8, dtype=torch.float32), torch.zeros_like(v_fp8
        , dtype=torch.float32)
    head_size_v_og = do.size(3)
    do_padded = do
    if head_size_v_og % 8 != 0:
        do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])
    fp8_dtype = types.get_fp8_e4m3_dtype()
    do_padded_fp8, descale_do = _cast_to_fp8(do_padded, fp8_dtype, 'bshd')
    if _USE_FUSED_BWD_KERNEL:
        flash_attn_fused_backward(do_padded_fp8, q_fp8, k_fp8, v_fp8, out,
            softmax_lse, dq, dk, dv, None, ctx.softmax_scale, ctx.
            alibi_slopes, ctx.causal, None, None, max_seqlen_q=q_fp8.shape[
            1], max_seqlen_k=k_fp8.shape[1], dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed, philox_offset=ctx.philox_offset,
            descale_q=descale_q, descale_k=descale_k, descale_v=descale_v,
            descale_do=descale_do, USE_INT64_STRIDES=_USE_INT64_STRIDES)
    else:
        flash_attn_onekernel_backward(do_padded_fp8, q_fp8, k_fp8, v_fp8,
            out, softmax_lse, dq, dk, dv, None, ctx.softmax_scale, ctx.
            alibi_slopes, ctx.causal, None, None, max_seqlen_q=q_fp8.shape[
            1], max_seqlen_k=k_fp8.shape[1], dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed, philox_offset=ctx.philox_offset,
            descale_q=descale_q, descale_k=descale_k, descale_v=descale_v,
            descale_do=descale_do, USE_INT64_STRIDES=_USE_INT64_STRIDES)
    return (dq, dk, dv, None, None, None, None, None, None, None, None,
        None, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _FlashAttnFP8Func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, window_size,
        alibi_slopes, deterministic, return_lse, return_softmax,
        is_grad_enabled, config=None):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        fp8_dtype = types.get_fp8_e4m3_dtype()
        q_fp8, descale_q = _cast_to_fp8(q, fp8_dtype, 'bshd')
        k_fp8, descale_k = _cast_to_fp8(k, fp8_dtype, 'bshd')
        v_fp8, descale_v = _cast_to_fp8(v, fp8_dtype, 'bshd')
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(q_fp8, k_fp8, v_fp8, dropout_p,
            softmax_scale, causal=causal, window_size_left=int(window_size[
            0]), window_size_right=int(window_size[1]), bias=None,
            alibi_slopes=alibi_slopes, return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0, max_seqlen_q=q
            .shape[1], max_seqlen_k=k.shape[1], cu_seqlens_q=None,
            cu_seqlens_k=None, descale_q=descale_q, descale_k=descale_k,
            descale_v=descale_v, config=config))
        if is_grad:
            ctx.save_for_backward(q_fp8, k_fp8, v_fp8, out_padded,
                softmax_lse, descale_q, descale_k, descale_v)
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        (q_fp8, k_fp8, v_fp8, out, softmax_lse, descale_q, descale_k, descale_v
            ) = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q_fp8, dtype=torch.float32
            ), torch.zeros_like(k_fp8, dtype=torch.float32), torch.zeros_like(
            v_fp8, dtype=torch.float32)
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8]
                )
        fp8_dtype = types.get_fp8_e4m3_dtype()
        do_padded_fp8, descale_do = _cast_to_fp8(do_padded, fp8_dtype, 'bshd')
        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(do_padded_fp8, q_fp8, k_fp8, v_fp8,
                out, softmax_lse, dq, dk, dv, None, ctx.softmax_scale, ctx.
                alibi_slopes, ctx.causal, None, None, max_seqlen_q=q_fp8.
                shape[1], max_seqlen_k=k_fp8.shape[1], dropout_p=ctx.
                dropout_p, philox_seed=ctx.philox_seed, philox_offset=ctx.
                philox_offset, descale_q=descale_q, descale_k=descale_k,
                descale_v=descale_v, descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES)
        else:
            flash_attn_onekernel_backward(do_padded_fp8, q_fp8, k_fp8,
                v_fp8, out, softmax_lse, dq, dk, dv, None, ctx.
                softmax_scale, ctx.alibi_slopes, ctx.causal, None, None,
                max_seqlen_q=q_fp8.shape[1], max_seqlen_k=k_fp8.shape[1],
                dropout_p=ctx.dropout_p, philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset, descale_q=descale_q,
                descale_k=descale_k, descale_v=descale_v, descale_do=
                descale_do, USE_INT64_STRIDES=_USE_INT64_STRIDES)
        return (dq, dk, dv, None, None, None, None, None, None, None, None,
            None, None, None)
