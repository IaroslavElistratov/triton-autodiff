# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/triton/triton_hstu_preprocess_and_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/triton/triton_hstu_preprocess_and_attention.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def autotune_max_seq_len(runtime_max_seq_len: int) ->int:
    global USE_RUNTIME_MAX_SEQ_LEN
    if USE_RUNTIME_MAX_SEQ_LEN:
        return prev_power_of_2(runtime_max_seq_len)
    else:
        if STATIC_MAX_SEQ_LENS == []:
            return 1
        for max_len in STATIC_MAX_SEQ_LENS:
            if max_len >= runtime_max_seq_len:
                return max_len
        return STATIC_MAX_SEQ_LENS[-1]


@torch.fx.wrap
def prev_power_of_2(x: int) ->int:
    if torch.compiler.is_compiling():
        x_tensor = torch.scalar_tensor(x, dtype=torch.int64)
        x_tensor_orig = x_tensor.clone()
        out = triton.next_power_of_2(x_tensor)
        return int(torch.where(torch.lt(x_tensor_orig, out), out // 2, out)
            .item())
    else:
        out = triton.next_power_of_2(x)
        return out // 2 if out > x else out


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


@triton_cc(annotations={'M': 'i32', 'N': ('i32', 16), 'K': ('i32', 16),
    'stride_xm': ('i32', 16), 'stride_xk': ('i32', 1), 'stride_wk': ('i32',
    16), 'stride_wn': ('i32', 1), 'stride_ym': ('i32', 16), 'stride_yn': (
    'i32', 1), 'stride_zm': ('i32', 16), 'stride_zn': ('i32', 1)})
@triton_autotune(configs=get_mm_configs(), key=['N', 'K'])
@triton.jit
def _addmm_fwd(x_ptr, w_ptr, y_ptr, z_ptr, M, N, K, stride_xm, stride_xk,
    stride_wk, stride_wn, stride_ym, stride_yn, stride_zm, stride_zn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, ALLOW_TF32: tl.constexpr, BROADCAST_Y: tl.constexpr
    ):
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        )
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[
            None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = (accumulator + y.to(tl.float32)).to(z_ptr.dtype.element_ty)
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


@torch.fx.wrap
def maybe_triton_addmm_fwd(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
    ) ->torch.Tensor:
    if is_sm100() or torch.version.hip is not None:
        return torch.addmm(y, x, w)
    else:
        return triton_addmm_fwd(x=x, w=w, y=y)


@torch.fx.wrap
def triton_addmm_fwd(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
    ) ->torch.Tensor:
    M, K = x.shape
    KB, N = w.shape
    assert K == KB, f'incompatible dimensions {K}, {KB}'
    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f'incompatible dimensions {N}, {NY}'
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N,
        meta['BLOCK_N']))
    _addmm_fwd[grid](x, w, y, z, M, N, K, x.stride(0), x.stride(1), w.
        stride(0), w.stride(1), y.stride(0) if not is_y_1d else 0, y.stride
        (1) if not is_y_1d else y.stride(0), z.stride(0), z.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, BROADCAST_Y=is_y_1d)
    return z


@triton_autotune(configs=_get_layer_norm_fwd_configs(), key=['BLOCK_D'])
@triton.jit
def _layer_norm_fwd(X, Y, Mean, Rstd, N, D, eps, stride_x, stride_y,
    TRAINING: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N
    X_block_ptr = tl.make_block_ptr(base=X, shape=(N, D), strides=(stride_x,
        1), offsets=(start_row, 0), block_shape=(BLOCK_N, BLOCK_D), order=(
        1, 0))
    Y_block_ptr = tl.make_block_ptr(base=Y, shape=(N, D), strides=(stride_y,
        1), offsets=(start_row, 0), block_shape=(BLOCK_N, BLOCK_D), order=(
        1, 0))
    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option='zero'
        ).to(tl.float32)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)
    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)
    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton_autotune(configs=_get_layer_norm_fwd_configs(), key=['BLOCK_D'])
@triton.jit
def _weighted_layer_norm_fwd(X, Y, W, B, Mean, Rstd, N, D, eps, stride_x,
    stride_y, IS_SWISH: tl.constexpr, TRAINING: tl.constexpr, BLOCK_D: tl.
    constexpr, BLOCK_N: tl.constexpr, COMPUTE_MEAN_AND_RSTD: tl.constexpr):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
    X_block_ptr = tl.make_block_ptr(base=X, shape=(N, D), strides=(stride_x,
        1), offsets=(start_row, 0), block_shape=(BLOCK_N, BLOCK_D), order=(
        1, 0))
    Y_block_ptr = tl.make_block_ptr(base=Y, shape=(N, D), strides=(stride_y,
        1), offsets=(start_row, 0), block_shape=(BLOCK_N, BLOCK_D), order=(
        1, 0))
    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option='zero'
        ).to(tl.float32)
    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x_block, axis=1) / D
        if TRAINING:
            tl.store(Mean + rows, mean, row_mask)
        mean = tl.expand_dims(mean, 1)
    else:
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)
    if COMPUTE_MEAN_AND_RSTD:
        _var = x_mean * x_mean
        var = tl.sum(_var, axis=1) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Rstd + rows, rstd, row_mask)
    else:
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)
    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    y = y * w[None, :] + b[None, :]
    if IS_SWISH:
        y = tl.sigmoid(y) * x_block
    tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


def triton_weighted_layer_norm_fwd(x: torch.Tensor, weight: Optional[torch.
    Tensor], bias: Optional[torch.Tensor], eps: float, mean: Optional[torch
    .Tensor]=None, rstd: Optional[torch.Tensor]=None) ->Tuple[torch.Tensor,
    torch.Tensor, torch.Tensor, int]:
    assert x.dim() == 2, f'x.dim() == {x.dim()}, expected 2'
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    learnable = weight is not None
    if learnable:
        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert weight.numel() == D
        assert bias.numel() == D
    y = torch.empty_like(x)
    compute_mean_and_rstd = mean is None or rstd is None
    if mean is None:
        mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    if rstd is None:
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    if N == 0:
        return y, mean, rstd, BLOCK_D
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)
    if learnable:
        _weighted_layer_norm_fwd[grid](x, y, weight, bias, mean, rstd, N, D,
            eps, x.stride(0), y.stride(0), IS_SWISH=False, TRAINING=True,
            BLOCK_D=BLOCK_D, COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd)
    else:
        _layer_norm_fwd[grid](x, y, mean, rstd, N, D, eps, x.stride(0), y.
            stride(0), TRAINING=True, BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd)
    return y, mean, rstd, BLOCK_D


def is_sm100() ->bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 0


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton_autotune(configs=_get_fw_configs(), key=['AUTOTUNE_Z', 'H',
    'AUTOTUNE_MAX_SEQ_LEN', 'DimQ', 'DimV', 'DeltaSize', 'IS_DELTA_Q'])
@triton.jit
def _hstu_attn_fwd(Q, K, V, workspace_ptr, sort_by_length_indices,
    seq_offsets, num_targets, Out, stride_qm, stride_qh, stride_kn,
    stride_kh, stride_vn, stride_vh, stride_om, stride_oh, alpha, Z,
    AUTOTUNE_Z, H, MAX_SEQ_LEN, AUTOTUNE_MAX_SEQ_LEN, DimQ, DimV, DeltaSize,
    contextual_seq_len, max_attn_len, HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr, ALLOW_TF32: tl.constexpr, BLOCK_D_Q: tl.
    constexpr, BLOCK_D_V: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.
    constexpr, USE_TLX: tl.constexpr, NUM_BUFFERS: tl.constexpr,
    NUM_MMA_WARPS_PER_GROUP: tl.constexpr, NUM_MMA_GROUPS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr, HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr, ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    if USE_TLX:
        _hstu_attn_fwd_compute_tlx(Q=Q, K=K, V=V, H=H, DimQ=DimQ, DimV=DimV,
            seq_offsets=seq_offsets, num_targets=num_targets, Out=Out,
            stride_qh=stride_qh, stride_kh=stride_kh, stride_vh=stride_vh,
            stride_om=stride_om, stride_oh=stride_oh, alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN, DeltaSize=DeltaSize,
            contextual_seq_len=contextual_seq_len, max_attn_len=
            max_attn_len, off_z=off_z, off_h=off_h, pid=pid,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS, IS_DELTA_Q=
            IS_DELTA_Q, ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V, HAS_CONTEXTUAL_SEQ_LEN=
            HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_BUFFERS=NUM_BUFFERS,
            NUM_MMA_WARPS_PER_GROUP=NUM_MMA_WARPS_PER_GROUP, NUM_MMA_GROUPS
            =NUM_MMA_GROUPS)
    else:
        _hstu_attn_fwd_compute(Q=Q, K=K, V=V, H=H, DimQ=DimQ, DimV=DimV,
            workspace_ptr=workspace_ptr, seq_offsets=seq_offsets,
            num_targets=num_targets, Out=Out, stride_qm=stride_qm,
            stride_qh=stride_qh, stride_kn=stride_kn, stride_kh=stride_kh,
            stride_vn=stride_vn, stride_vh=stride_vh, stride_om=stride_om,
            stride_oh=stride_oh, alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN,
            DeltaSize=DeltaSize, contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len, off_z=off_z, off_h=off_h, pid=pid,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS, IS_DELTA_Q=
            IS_DELTA_Q, ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V, HAS_CONTEXTUAL_SEQ_LEN=
            HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, ENABLE_TMA=ENABLE_TMA,
            TMA_DESC_SIZE=TMA_DESC_SIZE)


@triton.jit
def _hstu_attn_fwd_caculate_range(seq_len, start_m, n_targets,
    contextual_seq_len, max_attn_len, HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr, HAS_MAX_ATTN_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    if HAS_MULTIPLE_TARGETS:
        uih_end = seq_len - n_targets
    else:
        uih_end = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
        low = 0
        high = seq_len
    else:
        low = 0
        high = start_m + BLOCK_M
        if HAS_MAX_ATTN_LEN:
            if start_m > uih_end:
                low = uih_end - max_attn_len
            else:
                low = start_m - max_attn_len
            if HAS_CONTEXTUAL_SEQ_LEN:
                low = low if low > contextual_seq_len else 0
            else:
                low = low if low > 0 else 0
        if HAS_MULTIPLE_TARGETS:
            uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
            if uih_end < start_m:
                high = seq_len - n_targets
    return low, high, uih_end


@triton.jit
def _hstu_attn_fwd_compute(Q, K, V, H, DimQ, DimV, workspace_ptr,
    seq_offsets, num_targets, Out, stride_qm, stride_qh, stride_kn,
    stride_kh, stride_vn, stride_vh, stride_om, stride_oh, alpha,
    MAX_SEQ_LEN, DeltaSize, contextual_seq_len, max_attn_len, off_z, off_h,
    pid, HAS_MULTIPLE_TARGETS: tl.constexpr, IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr, BLOCK_D_Q: tl.constexpr, BLOCK_D_V: tl.
    constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr, HAS_MAX_ATTN_LEN: tl.constexpr,
    ENABLE_TMA: tl.constexpr, TMA_DESC_SIZE: tl.constexpr):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        start_m = (start_m_delta + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        Q_block_ptr = None
        K_block_ptr = None
        V_block_ptr = None
        if not ENABLE_TMA:
            if IS_DELTA_Q:
                Q_block_ptr = tl.make_block_ptr(base=Q + off_h * stride_qh +
                    off_z * DeltaSize * stride_qm, shape=(DeltaSize,
                    BLOCK_D_Q), strides=(stride_qm, 1), offsets=(
                    start_m_delta, 0), block_shape=(BLOCK_M, BLOCK_D_Q),
                    order=(1, 0))
            else:
                Q_block_ptr = tl.make_block_ptr(base=Q + off_h * stride_qh +
                    seq_start * stride_qm, shape=(seq_len, BLOCK_D_Q),
                    strides=(stride_qm, 1), offsets=(start_m, 0),
                    block_shape=(BLOCK_M, BLOCK_D_Q), order=(1, 0))
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero'
                )
            K_block_ptr = tl.make_block_ptr(base=K + off_h * stride_kh + 
                seq_start * stride_kn, shape=(BLOCK_D_Q, seq_len), strides=
                (1, stride_kn), offsets=(0, 0), block_shape=(BLOCK_D_Q,
                BLOCK_N), order=(0, 1))
            V_block_ptr = tl.make_block_ptr(base=V + off_h * stride_vh + 
                seq_start * stride_vn, shape=(seq_len, BLOCK_D_V), strides=
                (stride_vn, 1), offsets=(0, 0), block_shape=(BLOCK_N,
                BLOCK_D_V), order=(1, 0))
        elif IS_DELTA_Q:
            q = Q.load([(off_z * DeltaSize + start_m_delta).to(tl.int32), (
                off_h * stride_qh).to(tl.int32)])
        else:
            q = Q.load([(seq_start + start_m).to(tl.int32), (off_h *
                stride_qh).to(tl.int32)])
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if HAS_MULTIPLE_TARGETS:
            uih_end = seq_len - n_targets
        else:
            uih_end = seq_len
        if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
            low = 0
            high = seq_len
        else:
            low = 0
            high = start_m + BLOCK_M
            if HAS_MAX_ATTN_LEN:
                if start_m > uih_end:
                    low = uih_end - max_attn_len
                else:
                    low = start_m - max_attn_len
                if HAS_CONTEXTUAL_SEQ_LEN:
                    low = low if low > contextual_seq_len else 0
                else:
                    low = low if low > 0 else 0
            if HAS_MULTIPLE_TARGETS:
                uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                if uih_end < start_m:
                    high = seq_len - n_targets
        if low > 0:
            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, low))
                V_block_ptr = tl.advance(V_block_ptr, (low, 0))
        end_n = low
        for start_n in range(low, high, BLOCK_N):
            acc += _hstu_attn_fwd_one_block(start_n=start_n, seq_len=
                seq_len, offs_m=offs_m, offs_n=offs_n + start_n, q=q, K=K,
                V=V, K_block_ptr=K_block_ptr, V_block_ptr=V_block_ptr,
                offset_kh=off_h * stride_kh, offset_vh=off_h * stride_vh,
                seq_start=seq_start, n_targets=n_targets if
                HAS_MULTIPLE_TARGETS else None, alpha=alpha, MAX_SEQ_LEN=
                MAX_SEQ_LEN, contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len, HAS_MULTIPLE_TARGETS=
                HAS_MULTIPLE_TARGETS, HAS_CONTEXTUAL_SEQ_LEN=
                HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q, BLOCK_D_V=
                BLOCK_D_V, BLOCK_N=BLOCK_N, ENABLE_TMA=ENABLE_TMA)
            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            end_n += BLOCK_N
        if HAS_MULTIPLE_TARGETS:
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = (low_delta - end_n).to(tl.int32)
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                    V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(low_delta, high_delta, BLOCK_N,
                    num_stages=0):
                    acc += _hstu_attn_fwd_one_block(start_n=start_delta,
                        seq_len=seq_len, offs_m=offs_m, offs_n=offs_n +
                        start_delta, q=q, K=K, V=V, K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr, offset_kh=off_h *
                        stride_kh, offset_vh=off_h * stride_vh, seq_start=
                        seq_start, n_targets=n_targets if
                        HAS_MULTIPLE_TARGETS else None, alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN, contextual_seq_len=
                        contextual_seq_len, max_attn_len=max_attn_len,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN, ALLOW_TF32=
                        ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q, BLOCK_D_V=
                        BLOCK_D_V, BLOCK_N=BLOCK_N, ENABLE_TMA=ENABLE_TMA)
                    if not ENABLE_TMA:
                        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[
                None, :]
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton.jit
def _hstu_attn_fwd_compute_main_loop_tlx(low, high, seq_len, offs_m, offs_n,
    acc, q_tiles, k_tiles, v_tiles, q_fulls, k_fulls, v_fulls, k_empties,
    v_empties, v_dtype, n_targets, alpha, end_n, loop_trip_cnt,
    max_attn_len, HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr, HAS_MAX_ATTN_LEN: tl.constexpr,
    cid: tl.constexpr, BLOCK_N: tl.constexpr, NUM_BUFFERS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr, WAIT_FOR_Q: tl.constexpr):
    if WAIT_FOR_Q:
        q_full = tlx.local_view(q_fulls, cid)
        tlx.barrier_wait(q_full, 0)
    q_tile = tlx.local_view(q_tiles, cid)
    for start in tl.range(low + BLOCK_N, high, BLOCK_N, num_stages=0):
        buf_id = loop_trip_cnt % NUM_BUFFERS
        kv_phase = loop_trip_cnt // NUM_BUFFERS % 2
        start_n = tl.multiple_of(start, BLOCK_N)
        offs_n_start = offs_n
        offs_n = offs_n_start + start_n
        k_full = tlx.local_view(k_fulls, buf_id)
        tlx.barrier_wait(k_full, kv_phase)
        k_tile = tlx.local_view(k_tiles, buf_id)
        k_tile = tlx.local_trans(k_tile)
        qk = tlx.async_dot(q_tile, k_tile)
        qk = tlx.async_dot_wait(0, qk)
        k_empty = tlx.local_view(k_empties, buf_id)
        tlx.barrier_arrive(k_empty, 1)
        qk = qk * alpha
        invalid_mask = offs_m[:, None] == offs_n[None, :]
        max_ids = seq_len
        if HAS_MULTIPLE_TARGETS:
            max_ids = max_ids - n_targets
            offs_m = tl.where(offs_m < max_ids, offs_m, max_ids)
            offs_n = tl.where(offs_n < max_ids, offs_n, max_ids)
        offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
        invalid_mask = invalid_mask or offs_m_minus_n > 0
        if HAS_MAX_ATTN_LEN:
            invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
        if HAS_CONTEXTUAL_SEQ_LEN:
            invalid_mask = invalid_mask or offs_m[:, None] == 0 and offs_n[
                None, :] < max_ids
        scale = tl.where(invalid_mask, 1.0 / MAX_SEQ_LEN, 0.0)
        silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * scale
        silu = silu.to(v_dtype)
        v_full = tlx.local_view(v_fulls, buf_id)
        tlx.barrier_wait(v_full, kv_phase)
        v_tile = tlx.local_view(v_tiles, buf_id)
        acc = tlx.async_dot(silu, v_tile, acc)
        acc = tlx.async_dot_wait(0, acc)
        v_empty = tlx.local_view(v_empties, buf_id)
        tlx.barrier_arrive(v_empty, 1)
        end_n += BLOCK_N
        loop_trip_cnt += 1
    return acc, end_n, loop_trip_cnt


@triton.jit
def _hstu_attn_fwd_compute_main_loop_tlx_pipelined(low, high, seq_len,
    offs_m, offs_n, acc, q_tiles, k_tiles, v_tiles, q_fulls, k_fulls,
    v_fulls, k_empties, v_empties, v_dtype, n_targets, alpha, end_n,
    loop_trip_cnt, max_attn_len, HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr, HAS_MAX_ATTN_LEN: tl.constexpr,
    cid: tl.constexpr, BLOCK_N: tl.constexpr, NUM_BUFFERS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr, WAIT_FOR_Q: tl.constexpr):
    if WAIT_FOR_Q:
        q_full = tlx.local_view(q_fulls, cid)
        tlx.barrier_wait(q_full, 0)
    q_tile = tlx.local_view(q_tiles, cid)
    k_buf_id = loop_trip_cnt % NUM_BUFFERS
    k_phase = loop_trip_cnt // NUM_BUFFERS % 2
    k_full = tlx.local_view(k_fulls, k_buf_id)
    tlx.barrier_wait(k_full, k_phase)
    k_tile = tlx.local_view(k_tiles, k_buf_id)
    k_tile = tlx.local_trans(k_tile)
    if cid == 0:
        tlx.named_barrier_wait(9, 256)
    else:
        tlx.named_barrier_arrive(9, 256)
        tlx.named_barrier_wait(10, 256)
    qk = tlx.async_dot(q_tile, k_tile)
    if cid == 0:
        tlx.named_barrier_arrive(10, 256)
    qk = tlx.async_dot_wait(0, qk)
    k_empty = tlx.local_view(k_empties, k_buf_id)
    tlx.barrier_arrive(k_empty, 1)
    qk = qk * alpha
    start_n = tl.multiple_of(low, BLOCK_N)
    offs_n_start = offs_n
    offs_n = offs_n_start + start_n
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(offs_m < max_ids, offs_m, max_ids)
        offs_n = tl.where(offs_n < max_ids, offs_n, max_ids)
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    invalid_mask = invalid_mask or offs_m_minus_n > 0
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask or offs_m[:, None] == 0 and offs_n[None, :
            ] < max_ids
    scale = tl.where(invalid_mask, 1.0 / MAX_SEQ_LEN, 0.0)
    silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * scale
    silu = silu.to(v_dtype)
    loop_trip_cnt += 1
    for start in tl.range(low + BLOCK_N, high, BLOCK_N, num_stages=0):
        start_n = tl.multiple_of(start, BLOCK_N)
        offs_n = offs_n_start + start_n
        k_buf_id = loop_trip_cnt % NUM_BUFFERS
        k_phase = k_phase ^ (k_buf_id == 0)
        k_full = tlx.local_view(k_fulls, k_buf_id)
        tlx.barrier_wait(k_full, k_phase)
        k_tile = tlx.local_view(k_tiles, k_buf_id)
        k_tile = tlx.local_trans(k_tile)
        qk = tlx.async_dot(q_tile, k_tile)
        prev_silu = silu
        v_buf_id = (loop_trip_cnt - 1) % NUM_BUFFERS
        v_phase = (loop_trip_cnt - 1) // NUM_BUFFERS % 2
        v_full = tlx.local_view(v_fulls, v_buf_id)
        tlx.barrier_wait(v_full, v_phase)
        v_tile = tlx.local_view(v_tiles, v_buf_id)
        acc = tlx.async_dot(prev_silu, v_tile, acc)
        qk = tlx.async_dot_wait(1, qk)
        k_empty = tlx.local_view(k_empties, k_buf_id)
        tlx.barrier_arrive(k_empty, 1)
        qk = qk * alpha
        invalid_mask = offs_m[:, None] == offs_n[None, :]
        max_ids = seq_len
        if HAS_MULTIPLE_TARGETS:
            max_ids = max_ids - n_targets
            offs_m = tl.where(offs_m < max_ids, offs_m, max_ids)
            offs_n = tl.where(offs_n < max_ids, offs_n, max_ids)
        offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
        invalid_mask = invalid_mask or offs_m_minus_n > 0
        if HAS_MAX_ATTN_LEN:
            invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
        if HAS_CONTEXTUAL_SEQ_LEN:
            invalid_mask = invalid_mask or offs_m[:, None] == 0 and offs_n[
                None, :] < max_ids
        scale = tl.where(invalid_mask, 1.0 / MAX_SEQ_LEN, 0.0)
        silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * scale
        silu = silu.to(v_dtype)
        acc = tlx.async_dot_wait(0, acc)
        v_empty = tlx.local_view(v_empties, v_buf_id)
        tlx.barrier_arrive(v_empty, 1)
        end_n += BLOCK_N
        loop_trip_cnt += 1
    v_buf_id = (loop_trip_cnt - 1) % NUM_BUFFERS
    v_phase = (loop_trip_cnt - 1) // NUM_BUFFERS % 2
    v_full = tlx.local_view(v_fulls, v_buf_id)
    v_tile = tlx.local_view(v_tiles, v_buf_id)
    tlx.barrier_wait(v_full, v_phase)
    acc = tlx.async_dot(silu, v_tile, acc)
    acc = tlx.async_dot_wait(0, acc)
    v_empty = tlx.local_view(v_empties, v_buf_id)
    tlx.barrier_arrive(v_empty, 1)
    return acc, end_n, loop_trip_cnt


@triton.jit
def _hstu_attn_fwd_compute_tlx(Q, K, V, H, DimQ, DimV, seq_offsets,
    num_targets, Out, stride_qh, stride_kh, stride_vh, stride_om, stride_oh,
    alpha, MAX_SEQ_LEN, DeltaSize, contextual_seq_len, max_attn_len, off_z,
    off_h, pid, HAS_MULTIPLE_TARGETS: tl.constexpr, IS_DELTA_Q: tl.
    constexpr, ALLOW_TF32: tl.constexpr, BLOCK_D_Q: tl.constexpr, BLOCK_D_V:
    tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_BUFFERS:
    tl.constexpr, NUM_MMA_WARPS_PER_GROUP: tl.constexpr, NUM_MMA_GROUPS: tl
    .constexpr, HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr, HAS_MAX_ATTN_LEN: tl.
    constexpr):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if IS_DELTA_Q:
        start_m = pid * BLOCK_M
        start_m = (start_m + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m = pid * BLOCK_M
    if start_m >= seq_len:
        return
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_D_Q), tlx.dtype_of(Q),
        NUM_MMA_GROUPS)
    k_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D_Q), tlx.dtype_of(K),
        NUM_BUFFERS)
    v_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D_V), tlx.dtype_of(V),
        NUM_BUFFERS)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=
        NUM_MMA_GROUPS)
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=
        NUM_MMA_GROUPS)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    with tlx.async_tasks():
        with tlx.async_task('default'):
            _hstu_attn_fwd_load_Q_K_V(Q=Q, K=K, V=V, q_tiles=q_tiles,
                k_tiles=k_tiles, v_tiles=v_tiles, q_fulls=q_fulls, k_fulls=
                k_fulls, v_fulls=v_fulls, k_empties=k_empties, v_empties=
                v_empties, stride_qh=stride_qh, stride_kh=stride_kh,
                stride_vh=stride_vh, contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len, DeltaSize=DeltaSize, off_z=off_z,
                off_h=off_h, start_m=start_m, seq_start=seq_start, seq_len=
                seq_len, n_targets=n_targets, HAS_MULTIPLE_TARGETS=
                HAS_MULTIPLE_TARGETS, IS_DELTA_Q=IS_DELTA_Q, BLOCK_D_Q=
                BLOCK_D_Q, BLOCK_D_V=BLOCK_D_V, BLOCK_M=BLOCK_M, BLOCK_N=
                BLOCK_N, NUM_BUFFERS=NUM_BUFFERS, NUM_MMA_GROUPS=
                NUM_MMA_GROUPS, HAS_CONTEXTUAL_SEQ_LEN=
                HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN)
        with tlx.async_task(num_warps=NUM_MMA_WARPS_PER_GROUP, registers=
            232, replicate=NUM_MMA_GROUPS):
            cid = tlx.async_task_replica_id()
            acc = tl.zeros([BLOCK_M_SPLIT, BLOCK_D_V], dtype=tl.float32)
            offs_m = start_m + tl.arange(0, BLOCK_M_SPLIT
                ) + cid * BLOCK_M_SPLIT
            offs_n = tl.arange(0, BLOCK_N)
            low, high, uih_end = _hstu_attn_fwd_caculate_range(seq_len,
                start_m, n_targets, contextual_seq_len, max_attn_len,
                HAS_MULTIPLE_TARGETS, HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN, BLOCK_M, BLOCK_N)
            end_n = low
            loop_trip_cnt = 0
            acc, end_n, loop_trip_cnt = (
                _hstu_attn_fwd_compute_main_loop_tlx_pipelined(low=low,
                high=high, seq_len=seq_len, offs_m=offs_m, offs_n=offs_n,
                acc=acc, q_tiles=q_tiles, k_tiles=k_tiles, v_tiles=v_tiles,
                q_fulls=q_fulls, k_fulls=k_fulls, v_fulls=v_fulls,
                k_empties=k_empties, v_empties=v_empties, v_dtype=tlx.
                dtype_of(V), n_targets=n_targets, alpha=alpha, end_n=end_n,
                loop_trip_cnt=loop_trip_cnt, max_attn_len=max_attn_len,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN, cid=cid, BLOCK_N=BLOCK_N,
                NUM_BUFFERS=NUM_BUFFERS, MAX_SEQ_LEN=MAX_SEQ_LEN, WAIT_FOR_Q=1)
                )
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                acc, end_n, loop_trip_cnt = (
                    _hstu_attn_fwd_compute_main_loop_tlx(low=low_delta,
                    high=high_delta, seq_len=seq_len, offs_m=offs_m, offs_n
                    =offs_n, acc=acc, q_tiles=q_tiles, k_tiles=k_tiles,
                    v_tiles=v_tiles, q_fulls=q_fulls, k_fulls=k_fulls,
                    v_fulls=v_fulls, k_empties=k_empties, v_empties=
                    v_empties, v_dtype=tlx.dtype_of(V), n_targets=n_targets,
                    alpha=alpha, end_n=end_n, loop_trip_cnt=loop_trip_cnt,
                    max_attn_len=max_attn_len, HAS_MULTIPLE_TARGETS=
                    HAS_MULTIPLE_TARGETS, HAS_CONTEXTUAL_SEQ_LEN=
                    HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN=
                    HAS_MAX_ATTN_LEN, cid=cid, BLOCK_N=BLOCK_N, NUM_BUFFERS
                    =NUM_BUFFERS, MAX_SEQ_LEN=MAX_SEQ_LEN, WAIT_FOR_Q=0))
            if IS_DELTA_Q:
                start_m_delta = pid * BLOCK_M + cid * BLOCK_M_SPLIT
                offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M_SPLIT)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
                out_ptrs = off_o + offs_m_delta[:, None
                    ] * stride_om + offs_v_d[None, :]
                tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:,
                    None])
            else:
                start_m = pid * BLOCK_M + cid * BLOCK_M_SPLIT
                offs_m = start_m + tl.arange(0, BLOCK_M_SPLIT)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                off_o = Out + seq_start * stride_om + off_h * stride_oh
                out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[
                    None, :]
                tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton.jit
def _hstu_attn_fwd_load_K_or_V(K, k_tiles, k_empties, k_fulls, buf_id,
    k_phase, start_n, seq_start, offset_kh, BLOCK_D_Q: tl.constexpr,
    BLOCK_N: tl.constexpr):
    k_empty = tlx.local_view(k_empties, buf_id)
    tlx.barrier_wait(k_empty, k_phase)
    k_full = tlx.local_view(k_fulls, buf_id)
    k_tile = tlx.local_view(k_tiles, buf_id)
    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * BLOCK_D_Q)
    tlx.async_descriptor_load(K, k_tile, [(seq_start + start_n).to(tl.int32
        ), offset_kh.to(tl.int32)], k_full)


@triton.jit
def _hstu_attn_fwd_load_Q(Q, q_tiles, q_fulls, cid, off_z, off_h, stride_qh,
    start_m, seq_start, DeltaSize, IS_DELTA_Q: tl.constexpr, BLOCK_D_Q: tl.
    constexpr, BLOCK_M: tl.constexpr):
    q_full = tlx.local_view(q_fulls, cid)
    tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M * BLOCK_D_Q)
    q_tile = tlx.local_view(q_tiles, cid)
    seq_offset = start_m + cid * BLOCK_M
    if IS_DELTA_Q:
        tlx.async_descriptor_load(Q, q_tile, [(off_z * DeltaSize + start_m)
            .to(tl.int32), (off_h * stride_qh).to(tl.int32)], q_full)
    else:
        tlx.async_descriptor_load(Q, q_tile, [(seq_start + seq_offset).to(
            tl.int32), (off_h * stride_qh).to(tl.int32)], q_full)


@triton.jit
def _hstu_attn_fwd_load_Q_K_V(Q, K, V, q_tiles, k_tiles, v_tiles, q_fulls,
    k_fulls, v_fulls, k_empties, v_empties, stride_qh, stride_kh, stride_vh,
    contextual_seq_len, max_attn_len, DeltaSize, off_z, off_h, start_m,
    seq_start, seq_len, n_targets, HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr, BLOCK_D_Q: tl.constexpr, BLOCK_D_V: tl.
    constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, NUM_BUFFERS:
    tl.constexpr, NUM_MMA_GROUPS: tl.constexpr, HAS_CONTEXTUAL_SEQ_LEN: tl.
    constexpr, HAS_MAX_ATTN_LEN: tl.constexpr):
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    _hstu_attn_fwd_load_Q(Q=Q, q_tiles=q_tiles, q_fulls=q_fulls, cid=0,
        off_z=off_z, off_h=off_h, stride_qh=stride_qh, start_m=start_m,
        seq_start=seq_start, DeltaSize=DeltaSize, IS_DELTA_Q=IS_DELTA_Q,
        BLOCK_D_Q=BLOCK_D_Q, BLOCK_M=BLOCK_M_SPLIT)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    offset_kh = off_h * stride_kh
    offset_vh = off_h * stride_vh
    low, high, uih_end = _hstu_attn_fwd_caculate_range(seq_len, start_m,
        n_targets, contextual_seq_len, max_attn_len, HAS_MULTIPLE_TARGETS,
        HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN, BLOCK_M, BLOCK_N)
    kv_phase = 0
    loop_trip_cnt = 0
    buf_id = loop_trip_cnt % NUM_BUFFERS
    kv_phase = kv_phase ^ (buf_id == 0)
    start_n = tl.multiple_of(low, BLOCK_N)
    _hstu_attn_fwd_load_K_or_V(K, k_tiles, k_empties, k_fulls, buf_id,
        kv_phase, start_n, seq_start, offset_kh, BLOCK_D_Q, BLOCK_N)
    for cid in tl.range(1, NUM_MMA_GROUPS, loop_unroll_factor=
        NUM_MMA_GROUPS - 1):
        _hstu_attn_fwd_load_Q(Q, q_tiles, q_fulls, cid, off_z, off_h,
            stride_qh, start_m, seq_start, DeltaSize, IS_DELTA_Q, BLOCK_D_Q,
            BLOCK_M_SPLIT)
    _hstu_attn_fwd_load_K_or_V(V, v_tiles, v_empties, v_fulls, buf_id,
        kv_phase, start_n, seq_start, offset_vh, BLOCK_D_V, BLOCK_N)
    loop_trip_cnt += 1
    for start in range(low + BLOCK_N, high, BLOCK_N):
        buf_id = loop_trip_cnt % NUM_BUFFERS
        kv_phase = kv_phase ^ (buf_id == 0)
        start_n = tl.multiple_of(start, BLOCK_N)
        _hstu_attn_fwd_load_K_or_V(K, k_tiles, k_empties, k_fulls, buf_id,
            kv_phase, start_n, seq_start, offset_kh, BLOCK_D_Q, BLOCK_N)
        _hstu_attn_fwd_load_K_or_V(V, v_tiles, v_empties, v_fulls, buf_id,
            kv_phase, start_n, seq_start, offset_vh, BLOCK_D_V, BLOCK_N)
        loop_trip_cnt += 1
    if uih_end < start_m:
        low_delta = start_m
        high_delta = start_m + BLOCK_M
        for start_delta in tl.range(low_delta, high_delta, BLOCK_N,
            num_stages=0):
            buf_id = loop_trip_cnt % NUM_BUFFERS
            kv_phase = kv_phase ^ (buf_id == 0)
            start_n = tl.multiple_of(start_delta, BLOCK_N)
            _hstu_attn_fwd_load_K_or_V(K, k_tiles, k_empties, k_fulls,
                buf_id, kv_phase, start_n, seq_start, offset_kh, BLOCK_D_Q,
                BLOCK_N)
            _hstu_attn_fwd_load_K_or_V(V, v_tiles, v_empties, v_fulls,
                buf_id, kv_phase, start_n, seq_start, offset_vh, BLOCK_D_V,
                BLOCK_N)
            loop_trip_cnt += 1


@triton.jit
def _hstu_attn_fwd_one_block(start_n, seq_len, offs_m, offs_n, q, K, V,
    K_block_ptr, V_block_ptr, offset_kh, offset_vh, seq_start, n_targets,
    alpha, MAX_SEQ_LEN, contextual_seq_len, max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr, HAS_CONTEXTUAL_SEQ_LEN: tl.
    constexpr, HAS_MAX_ATTN_LEN: tl.constexpr, ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr, BLOCK_D_V: tl.constexpr, BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    k = None
    qk = None
    if ENABLE_TMA:
        k = K.load([(seq_start + start_n).to(tl.int32), offset_kh.to(tl.int32)]
            )
        qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * alpha
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option='zero')
        qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(offs_m > 0, offs_m, 0)
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(offs_n > 0, offs_n, 0)
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(offs_m < max_ids, offs_m, max_ids)
        offs_n = tl.where(offs_n < max_ids, offs_n, max_ids)
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    invalid_mask = invalid_mask or offs_m_minus_n > 0
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask or offs_m[:, None] == 0 and offs_n[None, :
            ] < max_ids
    scale = tl.where(invalid_mask, 1.0 / MAX_SEQ_LEN, 0.0)
    silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * scale
    v = None
    if ENABLE_TMA:
        v = V.load([(seq_start + start_n).to(tl.int32), offset_vh.to(tl.int32)]
            )
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


def triton_hstu_attention_fwd(N: int, alpha: float, q: torch.Tensor, k:
    torch.Tensor, v: torch.Tensor, seq_offsets: torch.Tensor, num_targets:
    Optional[torch.Tensor], max_attn_len: int, contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor], enable_tma: bool
    ) ->torch.Tensor:
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out
    TMA_DESC_SIZE = 128
    workspace = None
    desc_q = q
    desc_k = k
    desc_v = v
    if enable_tma and tensor_descriptor_tma:
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[L, H * DimQ], strides=[H * DimQ,
            1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[L, H * DimV], strides=[H * DimV,
            1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[L, H * DimQ], strides=[H * DimQ,
            1], block_shape=dummy_block)

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == TMA_DESC_SIZE
        return torch.empty(size, dtype=torch.int8, device='cuda')
    triton.set_allocator(alloc_fn)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_M']), Z * H)
    _hstu_attn_fwd[grid](Q=desc_q, K=desc_k, V=desc_v, workspace_ptr=
        workspace, sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets, num_targets=num_targets, Out=out,
        stride_qm=q.stride(0), stride_qh=q.stride(1), stride_kn=k.stride(0),
        stride_kh=k.stride(1), stride_vn=v.stride(0), stride_vh=v.stride(1),
        stride_om=out.stride(0), stride_oh=out.stride(1), alpha=alpha, Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z, H=H, MAX_SEQ_LEN=N, AUTOTUNE_MAX_SEQ_LEN=
        autotune_max_seq_len(N), DimQ=DimQ, DimV=DimV, DeltaSize=0,
        contextual_seq_len=contextual_seq_len, max_attn_len=max_attn_len,
        HAS_MULTIPLE_TARGETS=has_multiple_targets, IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV, HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len, HAS_SORT_BY_LENGTH_INDICES=
        has_sort_by_length_indices, ENABLE_TMA=enable_tma, TMA_DESC_SIZE=
        TMA_DESC_SIZE)
    return out


# Forward method (kernel launch code)
def __HSTUPreprocessAndAttentionFunction_forward(ctx, x: torch.Tensor,
    norm_weight: torch.Tensor, norm_bias: torch.Tensor, norm_eps: float,
    num_heads: int, attn_dim: int, hidden_dim: int, uvqk_weight: torch.
    Tensor, uvqk_bias: torch.Tensor, max_seq_len: int, seq_offsets: torch.
    Tensor, attn_alpha: float, num_targets: Optional[torch.Tensor],
    max_attn_len: int, contextual_seq_len: int, recompute_uvqk_in_backward:
    bool, recompute_normed_x_in_backward: bool, sort_by_length: bool,
    enable_tma: bool) ->Tuple[torch.Tensor, torch.Tensor]:
    normed_x, x_mean, x_rstd, BLOCK_D = triton_weighted_layer_norm_fwd(x=x,
        weight=norm_weight, bias=norm_bias, eps=norm_eps)
    uvqk = maybe_triton_addmm_fwd(x=normed_x, w=uvqk_weight, y=uvqk_bias
        ).contiguous()
    u, v, q, k = uvqk.split([hidden_dim * num_heads, hidden_dim * num_heads,
        attn_dim * num_heads, attn_dim * num_heads], dim=1)
    q = q.view(-1, num_heads, attn_dim)
    k = k.view(-1, num_heads, attn_dim)
    v = v.view(-1, num_heads, hidden_dim)
    silu_u = F.silu(u)
    sort_by_length_indices = None
    if sort_by_length:
        seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
        _, sort_by_length_indices = torch.sort(seq_lengths, descending=True,
            stable=False)
    out = triton_hstu_attention_fwd(N=max_seq_len, alpha=attn_alpha, q=q, k
        =k, v=v, seq_offsets=seq_offsets, num_targets=num_targets,
        max_attn_len=max_attn_len, contextual_seq_len=contextual_seq_len,
        sort_by_length_indices=sort_by_length_indices, enable_tma=enable_tma)
    saved_tensors = [x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight,
        seq_offsets]
    if num_targets is not None:
        saved_tensors.append(num_targets)
    if not recompute_normed_x_in_backward:
        saved_tensors.append(normed_x)
    if recompute_uvqk_in_backward:
        saved_tensors.append(uvqk_bias)
    else:
        saved_tensors.append(uvqk)
    if sort_by_length:
        saved_tensors.append(sort_by_length_indices)
    ctx.save_for_backward(*saved_tensors)
    ctx.attn_alpha = attn_alpha
    ctx.has_multiple_targets = num_targets is not None
    ctx.max_seq_len = max_seq_len
    ctx.max_attn_len = max_attn_len
    ctx.recompute_normed_x_in_backward = recompute_normed_x_in_backward
    ctx.recompute_uvqk_in_backward = recompute_uvqk_in_backward
    ctx.hidden_dim = hidden_dim
    ctx.attn_dim = attn_dim
    ctx.num_heads = num_heads
    ctx.uvqk_bias_1d = uvqk_bias.dim() == 1
    ctx.norm_eps = norm_eps
    ctx.norm_BLOCK_D = BLOCK_D
    ctx.contextual_seq_len = contextual_seq_len
    ctx.sort_by_length = sort_by_length
    ctx.enable_tma = enable_tma
    return silu_u, out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def acc_dq(dq_ptrs_trans, start_m, stride_dqm, k, dqk_trans, alpha, mask_m,
    MAX_SEQ_LEN, LOCK, BLOCK_M: tl.constexpr, ATOMIC_ADD: tl.constexpr,
    ALLOW_TF32: tl.constexpr):
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
    dq_trans = tl.load(dq_ptrs_trans + start_m * stride_dqm, mask=mask_m[
        None, :], other=0.0, eviction_policy='evict_last')
    dq_trans += tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
    dq_trans = dq_trans.to(k.dtype)
    tl.store(dq_ptrs_trans + start_m * stride_dqm, dq_trans, mask=mask_m[
        None, :], eviction_policy='evict_last')
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)


@triton_autotune(configs=_get_bw_configs(), key=['AUTOTUNE_Z', 'H',
    'AUTOTUNE_MAX_SEQ_LEN', 'DimQ', 'DimV'])
@triton.jit
def _hstu_attn_bwd(Q, K, V, tma_workspace_ptr, sort_by_length_indices,
    seq_offsets, num_targets, DOut, DQ, DK, DV, LOCK, stride_qm, stride_qh,
    stride_kn, stride_kh, stride_vn, stride_vh, stride_dom, stride_doh,
    stride_dqm, stride_dqh, stride_dkn, stride_dkh, stride_dvn, stride_dvh,
    alpha, contextual_seq_len, max_attn_len, Z, AUTOTUNE_Z, H, MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN, DimQ, DimV, HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr, HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr, BLOCK_D_Q: tl.constexpr, BLOCK_D_V: tl.
    constexpr, SEQUENCE_PARALLEL: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, UNROLL: tl.constexpr, HAS_SORT_BY_LENGTH_INDICES:
    tl.constexpr, ENABLE_TMA: tl.constexpr, TMA_DESC_SIZE: tl.constexpr,
    ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    if ENABLE_BUFFER_OPS_ASSUMES:
        tl.assume(off_hz >= 0)
        tl.assume(off_z >= 0)
        tl.assume(off_h >= 0)
        tl.assume(seq_start >= 0)
        tl.assume(stride_qm >= 0)
        tl.assume(stride_qh >= 0)
        tl.assume(stride_kn >= 0)
        tl.assume(stride_kh >= 0)
        tl.assume(stride_vn >= 0)
        tl.assume(stride_vh >= 0)
        tl.assume(stride_dom >= 0)
        tl.assume(stride_doh >= 0)
        tl.assume(stride_dqm >= 0)
        tl.assume(stride_dqh >= 0)
        tl.assume(stride_dkn >= 0)
        tl.assume(stride_dkh >= 0)
        tl.assume(stride_dvn >= 0)
        tl.assume(stride_dvh >= 0)
    Q = Q + seq_start * stride_qm
    K = K + seq_start * stride_kn
    V = V + seq_start * stride_vn
    DOut = DOut + seq_start * stride_dom
    DQ = DQ + seq_start * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start * stride_dkn
    DV = DV + seq_start * stride_dvn
    device_desc_q = None
    device_desc_k = None
    device_desc_v = None
    device_desc_do = None
    device_desc_dk = None
    device_desc_dv = None
    if ENABLE_TMA:
        device_desc_q = tl.make_tensor_descriptor(Q, shape=[seq_len, H *
            DimQ], strides=[H * DimQ, 1], block_shape=[BLOCK_M, BLOCK_D_Q])
        device_desc_do = tl.make_tensor_descriptor(DOut, shape=[seq_len, H *
            DimV], strides=[H * DimV, 1], block_shape=[BLOCK_M, BLOCK_D_V])
        device_desc_k = tl.make_tensor_descriptor(K, shape=[seq_len, H *
            DimQ], strides=[H * DimQ, 1], block_shape=[BLOCK_N, BLOCK_D_Q])
        device_desc_dk = tl.make_tensor_descriptor(DK, shape=[seq_len, H *
            DimQ], strides=[H * DimQ, 1], block_shape=[BLOCK_N, BLOCK_D_Q])
        device_desc_v = tl.make_tensor_descriptor(V, shape=[seq_len, H *
            DimV], strides=[H * DimV, 1], block_shape=[BLOCK_N, BLOCK_D_V])
        device_desc_dv = tl.make_tensor_descriptor(DV, shape=[seq_len, H *
            DimV], strides=[H * DimV, 1], block_shape=[BLOCK_N, BLOCK_D_V])
    else:
        Q += off_h * stride_qh
        K += off_h * stride_kh
        V += off_h * stride_vh
        DOut += off_h * stride_doh
        DK += off_h * stride_dkh
        DV += off_h * stride_dvh
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len:
            return
        _hstu_attn_bwd_one_col_block(start_n=start_n, seq_len=seq_len,
            n_targets=n_targets, contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len, Q=Q, K=K, V=V, DOut=DOut, DQ=DQ, DK=
            DK, DV=DV, device_desc_q=device_desc_q, device_desc_k=
            device_desc_k, device_desc_v=device_desc_v, device_desc_do=
            device_desc_do, device_desc_dk=device_desc_dk, device_desc_dv=
            device_desc_dv, LOCK=LOCK, off_h=off_h, stride_qh=stride_qh,
            stride_kh=stride_kh, stride_vh=stride_vh, stride_doh=stride_doh,
            stride_dkh=stride_dkh, stride_dvh=stride_dvh, stride_qm=
            stride_qm, stride_kn=stride_kn, stride_vn=stride_vn, stride_dom
            =stride_dom, stride_dqm=stride_dqm, stride_dkn=stride_dkn,
            stride_dvn=stride_dvn, alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN
            =HAS_MAX_ATTN_LEN, ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, UNROLL=
            UNROLL, ATOMIC_ADD=True, ENABLE_TMA=ENABLE_TMA)
    else:
        for start_n in range(0, seq_len, BLOCK_N):
            _hstu_attn_bwd_one_col_block(start_n=start_n, seq_len=seq_len,
                n_targets=n_targets, contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len, Q=Q, K=K, V=V, DOut=DOut, DQ=DQ,
                DK=DK, DV=DV, device_desc_q=device_desc_q, device_desc_k=
                device_desc_k, device_desc_v=device_desc_v, device_desc_do=
                device_desc_do, device_desc_dk=device_desc_dk,
                device_desc_dv=device_desc_dv, LOCK=LOCK, off_h=off_h,
                stride_qh=stride_qh, stride_kh=stride_kh, stride_vh=
                stride_vh, stride_doh=stride_doh, stride_dkh=stride_dkh,
                stride_dvh=stride_dvh, stride_qm=stride_qm, stride_kn=
                stride_kn, stride_vn=stride_vn, stride_dom=stride_dom,
                stride_dqm=stride_dqm, stride_dkn=stride_dkn, stride_dvn=
                stride_dvn, alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN, ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q, BLOCK_D_V=BLOCK_D_V, BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N, UNROLL=UNROLL, ATOMIC_ADD=False,
                ENABLE_TMA=ENABLE_TMA)


@triton.jit
def _hstu_attn_bwd_one_block(start_m, offs_n, offs_m, q_ptrs_trans,
    dq_ptrs_trans, do_ptrs, device_desc_q, device_desc_do, dk, dv, k, v,
    pos_offs_n, seq_len, max_ids, contextual_seq_len, max_attn_len, LOCK,
    off_h, stride_qh, stride_doh, stride_qm, stride_dom, stride_dqm, alpha,
    MAX_SEQ_LEN, HAS_MULTIPLE_TARGETS: tl.constexpr, HAS_CONTEXTUAL_SEQ_LEN:
    tl.constexpr, HAS_MAX_ATTN_LEN: tl.constexpr, ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr, ATOMIC_ADD: tl.constexpr, ENABLE_TMA: tl.
    constexpr, BLOCK_D_Q: tl.constexpr, BLOCK_D_V: tl.constexpr):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_m = pos_offs_m - contextual_seq_len + 1
        pos_offs_m = tl.where(pos_offs_m > 0, pos_offs_m, 0)
    if HAS_MULTIPLE_TARGETS:
        pos_offs_m = tl.where(pos_offs_m < max_ids, pos_offs_m, max_ids)
    if ENABLE_TMA:
        q = device_desc_q.load([start_m, (off_h * stride_qh).to(tl.int32)])
        q_trans = tl.trans(q)
    else:
        q_trans = tl.load(q_ptrs_trans + start_m * stride_qm, mask=mask_m[
            None, :], other=0.0)
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]
    invalid_mask_trans = invalid_mask_trans or pos_offs_m_minus_n > 0
    if HAS_MAX_ATTN_LEN:
        invalid_mask_trans = (invalid_mask_trans and pos_offs_m_minus_n <=
            max_attn_len)
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask_trans = invalid_mask_trans or pos_offs_m[None, :
            ] == 0 and pos_offs_n[:, None] < max_ids
    silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
    silu_trans = silu_trans.to(k.dtype)
    if ENABLE_TMA:
        do = device_desc_do.load([start_m, (off_h * stride_doh).to(tl.int32)])
    else:
        do = tl.load(do_ptrs + start_m * stride_dom, mask=mask_m[:, None],
            other=0.0)
    dv += tl.dot(silu_trans, do, allow_tf32=ALLOW_TF32)
    dqk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (
        1.0 / MAX_SEQ_LEN)
    dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
    dqk_trans = dqk_trans.to(k.dtype)
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    acc_dq(dq_ptrs_trans=dq_ptrs_trans, start_m=start_m, stride_dqm=
        stride_dqm, k=k, dqk_trans=dqk_trans, alpha=alpha, mask_m=mask_m,
        MAX_SEQ_LEN=MAX_SEQ_LEN, LOCK=LOCK, BLOCK_M=BLOCK_M, ATOMIC_ADD=
        ATOMIC_ADD, ALLOW_TF32=ALLOW_TF32)
    return dk, dv


@triton.jit
def _hstu_attn_bwd_one_col_block(start_n, seq_len, n_targets,
    contextual_seq_len, max_attn_len, Q, K, V, DOut, DQ, DK, DV,
    device_desc_q, device_desc_k, device_desc_v, device_desc_do,
    device_desc_dk, device_desc_dv, LOCK, off_h, stride_qh, stride_kh,
    stride_vh, stride_doh, stride_dkh, stride_dvh, stride_qm, stride_kn,
    stride_vn, stride_dom, stride_dqm, stride_dkn, stride_dvn, alpha,
    MAX_SEQ_LEN, HAS_MULTIPLE_TARGETS: tl.constexpr, HAS_CONTEXTUAL_SEQ_LEN:
    tl.constexpr, HAS_MAX_ATTN_LEN: tl.constexpr, ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr, BLOCK_D_V: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, UNROLL: tl.constexpr, ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr):
    if HAS_MULTIPLE_TARGETS:
        low = start_n
        if HAS_MAX_ATTN_LEN:
            high = start_n + max_attn_len + BLOCK_N
            high = high if high + n_targets < seq_len else seq_len
        else:
            high = seq_len
    else:
        low = start_n
        if HAS_MAX_ATTN_LEN:
            high = start_n + max_attn_len + BLOCK_N
            high = high if high < seq_len else seq_len
        else:
            high = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
        if low < contextual_block_end:
            low = contextual_block_end
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    if ENABLE_TMA:
        q_ptrs_trans = None
        do_ptrs = None
        k = device_desc_k.load([start_n, (off_h * stride_kh).to(tl.int32)])
        v = device_desc_v.load([start_n, (off_h * stride_vh).to(tl.int32)])
    else:
        mask_n = offs_n < seq_len
        q_ptrs_trans = Q + (offs_m[None, :] * stride_qm + offs_qk_d[:, None])
        do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_n = offs_n - contextual_seq_len + 1
        pos_offs_n = tl.where(pos_offs_n > 0, pos_offs_n, 0)
        max_ids = max_ids - contextual_seq_len + 1
    else:
        pos_offs_n = offs_n
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        pos_offs_n = tl.where(pos_offs_n < max_ids, pos_offs_n, max_ids)
    if HAS_CONTEXTUAL_SEQ_LEN:
        for start_m in range(0, contextual_seq_len, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block(start_m=start_m, offs_n=
                offs_n, offs_m=offs_m, q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans, do_ptrs=do_ptrs, device_desc_q
                =device_desc_q, device_desc_do=device_desc_do, dk=dk, dv=dv,
                k=k, v=v, pos_offs_n=pos_offs_n, seq_len=seq_len, max_ids=
                max_ids, contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len, LOCK=LOCK, off_h=off_h,
                stride_qh=stride_qh, stride_doh=stride_doh, stride_qm=
                stride_qm, stride_dom=stride_dom, stride_dqm=stride_dqm,
                alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN, HAS_MULTIPLE_TARGETS=
                HAS_MULTIPLE_TARGETS, HAS_CONTEXTUAL_SEQ_LEN=
                HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32, BLOCK_M=BLOCK_M, ATOMIC_ADD=
                ATOMIC_ADD, ENABLE_TMA=ENABLE_TMA, BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V)
    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block(start_m=start_m, offs_n=offs_n,
            offs_m=offs_m, q_ptrs_trans=q_ptrs_trans, dq_ptrs_trans=
            dq_ptrs_trans, do_ptrs=do_ptrs, device_desc_q=device_desc_q,
            device_desc_do=device_desc_do, dk=dk, dv=dv, k=k, v=v,
            pos_offs_n=pos_offs_n, seq_len=seq_len, max_ids=max_ids,
            contextual_seq_len=contextual_seq_len, max_attn_len=
            max_attn_len, LOCK=LOCK, off_h=off_h, stride_qh=stride_qh,
            stride_doh=stride_doh, stride_qm=stride_qm, stride_dom=
            stride_dom, stride_dqm=stride_dqm, alpha=alpha, MAX_SEQ_LEN=
            MAX_SEQ_LEN, HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN, HAS_MAX_ATTN_LEN
            =HAS_MAX_ATTN_LEN, ALLOW_TF32=ALLOW_TF32, BLOCK_M=BLOCK_M,
            ATOMIC_ADD=ATOMIC_ADD, ENABLE_TMA=ENABLE_TMA, BLOCK_D_Q=
            BLOCK_D_Q, BLOCK_D_V=BLOCK_D_V)
    dk = dk * alpha
    if ENABLE_TMA:
        device_desc_dv.store([start_n, (off_h * stride_dvh).to(tl.int32)],
            dv.to(k.dtype))
        device_desc_dk.store([start_n, (off_h * stride_dkh).to(tl.int32)],
            dk.to(k.dtype))
    else:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
        tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
        tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])


@triton_autotune(configs=_get_bwd_dwdb_configs(), key=['D'])
@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, N, D, BLOCK_N: tl.
    constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=
        cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=
        cols < D)


@triton.jit
def _layer_norm_bwd_dx(DX, DY, X, Mean, Rstd, stride_dx, stride_dy,
    stride_x, D, eps, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)
    dy = tl.where(mask, dy, 0.0)
    c1 = tl.sum(xhat * dy, axis=0) / D
    c2 = tl.sum(dy, axis=0) / D
    dx = (dy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)


@triton_autotune(configs=_get_layer_norm_fwd_configs(), key=['BLOCK_D'])
@triton.jit
def _weighted_layer_norm_bwd_dx(DX, DY, DW, DB, X, W, B, Mean, Rstd,
    stride_dx, stride_dy, stride_x, D, eps, IS_SWISH: tl.constexpr, N,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_N)
    blocks_per_tile = num_blocks // tile_num
    if pid < num_blocks % tile_num:
        blocks_per_tile += 1
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    acc_dw = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_db = tl.zeros([BLOCK_D], dtype=tl.float32)
    start_block = pid
    for idx in range(blocks_per_tile):
        current_block = start_block + idx * tile_num
        start_row = current_block * BLOCK_N
        X_block_ptr = tl.make_block_ptr(base=X, shape=(N, D), strides=(
            stride_x, 1), offsets=(start_row, 0), block_shape=(BLOCK_N,
            BLOCK_D), order=(1, 0))
        DX_block_ptr = tl.make_block_ptr(base=DX, shape=(N, D), strides=(
            stride_dx, 1), offsets=(start_row, 0), block_shape=(BLOCK_N,
            BLOCK_D), order=(1, 0))
        DY_block_ptr = tl.make_block_ptr(base=DY, shape=(N, D), strides=(
            stride_dy, 1), offsets=(start_row, 0), block_shape=(BLOCK_N,
            BLOCK_D), order=(1, 0))
        x_block = tl.load(X_block_ptr, boundary_check=(0, 1),
            padding_option='zero').to(tl.float32)
        dy_block = tl.load(DY_block_ptr, boundary_check=(0, 1),
            padding_option='zero').to(tl.float32)
        rows = start_row + tl.arange(0, BLOCK_N)
        row_mask = rows < N
        mean = tl.load(Mean + rows, row_mask, other=0.0)
        rstd = tl.load(Rstd + rows, row_mask, other=0.0)
        mean = tl.expand_dims(mean, 1)
        rstd = tl.expand_dims(rstd, 1)
        xhat = (x_block - mean) * rstd
        xhat = tl.where(row_mask[:, None] & col_mask[None, :], xhat, 0.0)
        wdy = w[None, :] * dy_block
        wdy = tl.where(row_mask[:, None] & col_mask[None, :], wdy, 0.0)
        if IS_SWISH:
            b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
            sigmoid_layer_norm = tl.sigmoid(xhat * w[None, :] + b[None, :])
            sigmoid_layer_norm = tl.where(row_mask[:, None] & col_mask[None,
                :], sigmoid_layer_norm, 0.0)
            sigmoid_deriv = sigmoid_layer_norm * (1 - sigmoid_layer_norm)
            x_ = wdy * x_block * sigmoid_deriv
            x_ = tl.where(row_mask[:, None] & col_mask[None, :], x_, 0.0)
            c1 = tl.sum(xhat * x_, axis=1) / D
            c2 = tl.sum(x_, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (x_ - (xhat * c1 + c2)) * rstd
            dx = dy_block * sigmoid_layer_norm + dx
            tl.store(DX_block_ptr, dx.to(DX.dtype.element_ty),
                boundary_check=(0, 1))
            partial_dw = tl.sum(dy_block * x_block * xhat * sigmoid_deriv,
                axis=0)
            partial_db = tl.sum(dy_block * x_block * sigmoid_deriv, axis=0)
        else:
            c1 = tl.sum(xhat * wdy, axis=1) / D
            c2 = tl.sum(wdy, axis=1) / D
            c1 = tl.expand_dims(c1, 1)
            c2 = tl.expand_dims(c2, 1)
            dx = (wdy - (xhat * c1 + c2)) * rstd
            tl.store(DX_block_ptr, dx.to(DX.dtype.element_ty),
                boundary_check=(0, 1))
            partial_dw = tl.sum(dy_block * xhat, axis=0)
            partial_db = tl.sum(dy_block, axis=0)
        acc_dw += partial_dw
        acc_db += partial_db
    dw_ptrs = DW + pid.to(tl.int64) * D + cols
    db_ptrs = DB + pid.to(tl.int64) * D + cols
    tl.store(dw_ptrs, acc_dw, mask=col_mask)
    tl.store(db_ptrs, acc_db, mask=col_mask)


def triton_addmm_bwd(x: torch.Tensor, w: torch.Tensor, dz: torch.Tensor,
    is_y_1d: bool) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_y_1d:
        dy = torch.sum(dz, dim=0)
    else:
        dy = dz
    dw = torch.mm(x.t(), dz)
    dx = torch.mm(dz, w.t())
    return dx, dw, dy


def triton_hstu_attention_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch
    .Tensor, v: torch.Tensor, dq: torch.Tensor, dk: torch.Tensor, dv: torch
    .Tensor, seq_offsets: torch.Tensor, num_targets: Optional[torch.Tensor],
    N: int, alpha: float, max_attn_len: int, contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor], enable_tma: bool) ->Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
    grid = lambda meta: (Z * H, triton.cdiv(N, meta['BLOCK_N']) if meta[
        'SEQUENCE_PARALLEL'] else 1)
    MIN_BLOCK_M = 16
    lock = torch.empty((Z * H, triton.cdiv(N, MIN_BLOCK_M)), dtype=torch.
        int32, device=q.device)
    AUTOTUNE_Z = prev_power_of_2(Z)
    TMA_DESC_SIZE = 128
    tma_workspace = None

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == TMA_DESC_SIZE
        return torch.empty(size, dtype=torch.int8, device='cuda')
    triton.set_allocator(alloc_fn)
    ENABLE_BUFFER_OPS_ASSUMES = torch.version.hip is not None
    _hstu_attn_bwd[grid](Q=q, K=k, V=v, tma_workspace_ptr=tma_workspace,
        sort_by_length_indices=sort_by_length_indices, seq_offsets=
        seq_offsets, num_targets=num_targets, DOut=dout, DQ=dq, DK=dk, DV=
        dv, LOCK=lock, stride_qm=q.stride(0), stride_qh=q.stride(1),
        stride_kn=k.stride(0), stride_kh=k.stride(1), stride_vn=v.stride(0),
        stride_vh=v.stride(1), stride_dom=dout.stride(0), stride_doh=dout.
        stride(1), stride_dqm=dq.stride(0), stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0), stride_dkh=dk.stride(1), stride_dvn=dv.
        stride(0), stride_dvh=dv.stride(1), alpha=alpha, contextual_seq_len
        =contextual_seq_len, max_attn_len=max_attn_len, Z=Z, AUTOTUNE_Z=
        AUTOTUNE_Z, H=H, MAX_SEQ_LEN=N, AUTOTUNE_MAX_SEQ_LEN=
        autotune_max_seq_len(N), DimQ=DimQ, DimV=DimV, HAS_MULTIPLE_TARGETS
        =num_targets is not None, HAS_CONTEXTUAL_SEQ_LEN=contextual_seq_len >
        0, HAS_MAX_ATTN_LEN=max_attn_len > 0, ALLOW_TF32=torch.backends.
        cuda.matmul.allow_tf32, BLOCK_D_Q=DimQ, BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        ENABLE_TMA=enable_tma, TMA_DESC_SIZE=TMA_DESC_SIZE,
        ENABLE_BUFFER_OPS_ASSUMES=ENABLE_BUFFER_OPS_ASSUMES)
    return dq, dk, dv


def triton_weighted_layer_norm_bwd(dy: torch.Tensor, x: torch.Tensor,
    weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], mean:
    torch.Tensor, rstd: torch.Tensor, learnable: bool, eps: float, BLOCK_D: int
    ) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    if learnable:
        assert weight is not None and bias is not None
        N, D = x.shape
        dx = torch.empty_like(x)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        tile_num = max(1, min(sms * 8, N // 4))
        _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x
            .device)
        _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.
            device)
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            dbias.zero_()
            return dx, dweight, dbias
        _weighted_layer_norm_bwd_dx[tile_num,](dx, dy, _dweight, _dbias, x,
            weight, bias, mean, rstd, dx.stride(0), dy.stride(0), x.stride(
            0), D, eps, IS_SWISH=False, N=N, BLOCK_D=BLOCK_D)

        def grid(META):
            return triton.cdiv(D, META['BLOCK_D']),
        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _layer_norm_bwd_dwdb[grid](_dweight, _dbias, dweight, dbias,
            tile_num, D, BLOCK_D=BLOCK_D)
        return dx, dweight, dbias
    else:
        N, D = x.shape
        dx = torch.empty_like(x)
        if N == 0:
            return dx, None, None
        _layer_norm_bwd_dx[N,](dx, dy, x, mean, rstd, dx.stride(0), dy.
            stride(0), x.stride(0), D, eps, BLOCK_D=BLOCK_D, num_warps=
            num_warps)
        return dx, None, None


# Backward method (kernel launch code)
def __HSTUPreprocessAndAttentionFunction_backward(ctx, dsilu_u: torch.
    Tensor, dout: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.
    Tensor, None, None, None, None, torch.Tensor, torch.Tensor, None, None,
    None, None, None, None, None, None, None, None]:
    x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight, seq_offsets = (ctx
        .saved_tensors[:7])
    idx = 7
    if ctx.has_multiple_targets:
        num_targets = ctx.saved_tensors[idx]
        idx += 1
    else:
        num_targets = None
    if ctx.recompute_normed_x_in_backward:
        normed_x, _, _, _ = triton_weighted_layer_norm_fwd(x=x, weight=
            norm_weight, bias=norm_bias, eps=ctx.norm_eps, mean=x_mean,
            rstd=x_rstd)
    else:
        normed_x = ctx.saved_tensors[idx]
        idx += 1
    if ctx.recompute_uvqk_in_backward:
        uvqk_bias = ctx.saved_tensors[idx]
        uvqk = maybe_triton_addmm_fwd(x=normed_x, w=uvqk_weight, y=uvqk_bias)
        idx += 1
    else:
        uvqk = ctx.saved_tensors[idx]
        idx += 1
    if ctx.sort_by_length:
        sort_by_length_indices = ctx.saved_tensors[idx]
    else:
        sort_by_length_indices = None
    duvqk = torch.empty_like(uvqk)
    du, dv, dq, dk = duvqk.split([ctx.hidden_dim * ctx.num_heads, ctx.
        hidden_dim * ctx.num_heads, ctx.attn_dim * ctx.num_heads, ctx.
        attn_dim * ctx.num_heads], dim=1)
    u, v, q, k = uvqk.split([ctx.hidden_dim * ctx.num_heads, ctx.hidden_dim *
        ctx.num_heads, ctx.attn_dim * ctx.num_heads, ctx.attn_dim * ctx.
        num_heads], dim=1)
    q = q.view(-1, ctx.num_heads, ctx.attn_dim)
    k = k.view(-1, ctx.num_heads, ctx.attn_dim)
    v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
    dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
    dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
    dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
    _dq, _dk, _dv = triton_hstu_attention_bwd(dout=dout, q=q, k=k, v=v, dq=
        dq, dk=dk, dv=dv, seq_offsets=seq_offsets, num_targets=num_targets,
        N=ctx.max_seq_len, max_attn_len=ctx.max_attn_len, alpha=ctx.
        attn_alpha, contextual_seq_len=ctx.contextual_seq_len,
        sort_by_length_indices=sort_by_length_indices, enable_tma=ctx.
        enable_tma)
    if dq.data_ptr() != _dq.data_ptr():
        dq.copy_(_dq)
    if dk.data_ptr() != _dk.data_ptr():
        dk.copy_(_dk)
    if dv.data_ptr() != _dv.data_ptr():
        dv.copy_(_dv)
    torch.ops.aten.silu_backward(dsilu_u, u, grad_input=du)
    d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(x=normed_x, w
        =uvqk_weight, dz=duvqk, is_y_1d=ctx.uvqk_bias_1d)
    d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(dy=
        d_normed_x, x=x, weight=norm_weight, bias=norm_bias, mean=x_mean,
        rstd=x_rstd, learnable=True, eps=ctx.norm_eps, BLOCK_D=ctx.norm_BLOCK_D
        )
    return (d_x, d_norm_weight, d_norm_bias, None, None, None, None,
        d_uvqk_weight, d_uvqk_bias, None, None, None, None, None, None,
        None, None, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _HSTUPreprocessAndAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, norm_weight: torch.Tensor, norm_bias:
        torch.Tensor, norm_eps: float, num_heads: int, attn_dim: int,
        hidden_dim: int, uvqk_weight: torch.Tensor, uvqk_bias: torch.Tensor,
        max_seq_len: int, seq_offsets: torch.Tensor, attn_alpha: float,
        num_targets: Optional[torch.Tensor], max_attn_len: int,
        contextual_seq_len: int, recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool, sort_by_length: bool,
        enable_tma: bool) ->Tuple[torch.Tensor, torch.Tensor]:
        normed_x, x_mean, x_rstd, BLOCK_D = triton_weighted_layer_norm_fwd(x
            =x, weight=norm_weight, bias=norm_bias, eps=norm_eps)
        uvqk = maybe_triton_addmm_fwd(x=normed_x, w=uvqk_weight, y=uvqk_bias
            ).contiguous()
        u, v, q, k = uvqk.split([hidden_dim * num_heads, hidden_dim *
            num_heads, attn_dim * num_heads, attn_dim * num_heads], dim=1)
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        silu_u = F.silu(u)
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(seq_lengths, descending=
                True, stable=False)
        out = triton_hstu_attention_fwd(N=max_seq_len, alpha=attn_alpha, q=
            q, k=k, v=v, seq_offsets=seq_offsets, num_targets=num_targets,
            max_attn_len=max_attn_len, contextual_seq_len=
            contextual_seq_len, sort_by_length_indices=
            sort_by_length_indices, enable_tma=enable_tma)
        saved_tensors = [x, norm_weight, norm_bias, x_mean, x_rstd,
            uvqk_weight, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if not recompute_normed_x_in_backward:
            saved_tensors.append(normed_x)
        if recompute_uvqk_in_backward:
            saved_tensors.append(uvqk_bias)
        else:
            saved_tensors.append(uvqk)
        if sort_by_length:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.attn_alpha = attn_alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_seq_len = max_seq_len
        ctx.max_attn_len = max_attn_len
        ctx.recompute_normed_x_in_backward = recompute_normed_x_in_backward
        ctx.recompute_uvqk_in_backward = recompute_uvqk_in_backward
        ctx.hidden_dim = hidden_dim
        ctx.attn_dim = attn_dim
        ctx.num_heads = num_heads
        ctx.uvqk_bias_1d = uvqk_bias.dim() == 1
        ctx.norm_eps = norm_eps
        ctx.norm_BLOCK_D = BLOCK_D
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        return silu_u, out

    @staticmethod
    def backward(ctx, dsilu_u: torch.Tensor, dout: torch.Tensor) ->Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None,
        torch.Tensor, torch.Tensor, None, None, None, None, None, None,
        None, None, None, None]:
        (x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight, seq_offsets
            ) = ctx.saved_tensors[:7]
        idx = 7
        if ctx.has_multiple_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        if ctx.recompute_normed_x_in_backward:
            normed_x, _, _, _ = triton_weighted_layer_norm_fwd(x=x, weight=
                norm_weight, bias=norm_bias, eps=ctx.norm_eps, mean=x_mean,
                rstd=x_rstd)
        else:
            normed_x = ctx.saved_tensors[idx]
            idx += 1
        if ctx.recompute_uvqk_in_backward:
            uvqk_bias = ctx.saved_tensors[idx]
            uvqk = maybe_triton_addmm_fwd(x=normed_x, w=uvqk_weight, y=
                uvqk_bias)
            idx += 1
        else:
            uvqk = ctx.saved_tensors[idx]
            idx += 1
        if ctx.sort_by_length:
            sort_by_length_indices = ctx.saved_tensors[idx]
        else:
            sort_by_length_indices = None
        duvqk = torch.empty_like(uvqk)
        du, dv, dq, dk = duvqk.split([ctx.hidden_dim * ctx.num_heads, ctx.
            hidden_dim * ctx.num_heads, ctx.attn_dim * ctx.num_heads, ctx.
            attn_dim * ctx.num_heads], dim=1)
        u, v, q, k = uvqk.split([ctx.hidden_dim * ctx.num_heads, ctx.
            hidden_dim * ctx.num_heads, ctx.attn_dim * ctx.num_heads, ctx.
            attn_dim * ctx.num_heads], dim=1)
        q = q.view(-1, ctx.num_heads, ctx.attn_dim)
        k = k.view(-1, ctx.num_heads, ctx.attn_dim)
        v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
        dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
        dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
        dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
        _dq, _dk, _dv = triton_hstu_attention_bwd(dout=dout, q=q, k=k, v=v,
            dq=dq, dk=dk, dv=dv, seq_offsets=seq_offsets, num_targets=
            num_targets, N=ctx.max_seq_len, max_attn_len=ctx.max_attn_len,
            alpha=ctx.attn_alpha, contextual_seq_len=ctx.contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices, enable_tma=ctx.
            enable_tma)
        if dq.data_ptr() != _dq.data_ptr():
            dq.copy_(_dq)
        if dk.data_ptr() != _dk.data_ptr():
            dk.copy_(_dk)
        if dv.data_ptr() != _dv.data_ptr():
            dv.copy_(_dv)
        torch.ops.aten.silu_backward(dsilu_u, u, grad_input=du)
        d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(x=
            normed_x, w=uvqk_weight, dz=duvqk, is_y_1d=ctx.uvqk_bias_1d)
        d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(dy
            =d_normed_x, x=x, weight=norm_weight, bias=norm_bias, mean=
            x_mean, rstd=x_rstd, learnable=True, eps=ctx.norm_eps, BLOCK_D=
            ctx.norm_BLOCK_D)
        return (d_x, d_norm_weight, d_norm_bias, None, None, None, None,
            d_uvqk_weight, d_uvqk_bias, None, None, None, None, None, None,
            None, None, None, None)
