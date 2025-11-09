# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/cpp/cuda_hstu_preprocess_and_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/cpp/cuda_hstu_preprocess_and_attention.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

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


def is_sm100() ->bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 0


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def __HSTUPreprocessAndAttentionFunction_forward(ctx, x: torch.Tensor,
    norm_weight: torch.Tensor, norm_bias: torch.Tensor, norm_eps: float,
    num_heads: int, attn_dim: int, hidden_dim: int, uvqk_weight: torch.
    Tensor, uvqk_bias: torch.Tensor, max_seq_len: int, seq_offsets: torch.
    Tensor, alpha: float, invalid_attn_mask_type: str, num_targets:
    Optional[torch.Tensor], rotary_weights: Optional[Tuple[torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor]]=None, attn_scale: Optional[
    torch.Tensor]=None, recompute_uvqk_in_backward: bool=False,
    recompute_normed_x_in_backward: bool=False, contextual_seq_len: int=0,
    sort_by_length: bool=False, max_attn_len: Optional[int]=None,
    full_attn_size: Optional[int]=None, silu_u: bool=True, fp8_in_addmm_fwd:
    bool=False) ->Tuple[torch.Tensor, torch.Tensor]:
    max_attn_len = max_attn_len or 0
    full_attn_size = full_attn_size or 0
    normed_x, x_mean, x_rstd, BLOCK_D, x_scale, normed_x_fp8 = (
        triton_weighted_layer_norm_quantization_fwd(x=x, weight=norm_weight,
        bias=norm_bias, eps=norm_eps, quantize_output=fp8_in_addmm_fwd))
    if fp8_in_addmm_fwd:
        assert x_scale is not None and normed_x_fp8 is not None
        uvqk = fp8_rowwise_quantize_addmm(x=normed_x, x_fp8=normed_x_fp8, w
            =uvqk_weight, y=uvqk_bias, x_scale=x_scale, custom_kernel=False,
            is_inference=False).contiguous()
    else:
        uvqk = maybe_triton_addmm_fwd(normed_x, uvqk_weight, uvqk_bias
            ).contiguous()
    u, v, q, k = uvqk.split([hidden_dim * num_heads, hidden_dim * num_heads,
        attn_dim * num_heads, attn_dim * num_heads], dim=1)
    if rotary_weights is not None:
        q_cos_weights = rotary_weights[0]
        q_sin_weights = rotary_weights[1]
        k_cos_weights = rotary_weights[2]
        k_sin_weights = rotary_weights[3]
        _q = triton_apply_rope_fwd(x=q.view(-1, num_heads, attn_dim), N=
            max_seq_len, seq_offsets=seq_offsets, cos_rope=q_cos_weights,
            sin_rope=q_sin_weights).view(-1, num_heads * attn_dim)
        _k = triton_apply_rope_fwd(x=k.view(-1, num_heads, attn_dim), N=
            max_seq_len, seq_offsets=seq_offsets, cos_rope=k_cos_weights,
            sin_rope=k_sin_weights).view(-1, num_heads * attn_dim)
        if q.data_ptr() != _q.data_ptr():
            q.copy_(_q)
        if k.data_ptr() != _k.data_ptr():
            k.copy_(_k)
    q = q.view(-1, num_heads, attn_dim)
    k = k.view(-1, num_heads, attn_dim)
    v = v.view(-1, num_heads, hidden_dim)
    if silu_u:
        u = F.silu(u)
    elif recompute_uvqk_in_backward:
        u = u.clone()
    if is_sm100():
        out = torch.ops.bw_hstu.bw_hstu_mha_fwd(max_seq_len, alpha, q, k, v,
            seq_offsets, True, num_targets, attn_scale, max_attn_len,
            full_attn_size, contextual_seq_len, None, None, None, 0, None,
            None, None, None, 1)
    else:
        out, _ = torch.ops.hstu.hstu_mha_fwd(max_seq_len, alpha, q, k, v,
            seq_offsets, True, num_targets, attn_scale, max_attn_len,
            full_attn_size, contextual_seq_len, None, None, None, 0)
    saved_tensors = [x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight,
        seq_offsets, out]
    if num_targets is not None:
        saved_tensors.append(num_targets)
    if attn_scale is not None:
        saved_tensors.append(attn_scale)
    if not recompute_normed_x_in_backward:
        saved_tensors.append(normed_x)
    if recompute_uvqk_in_backward:
        saved_tensors.append(uvqk_bias)
        if fp8_in_addmm_fwd:
            saved_tensors.append(x_scale)
            saved_tensors.append(normed_x_fp8)
    else:
        saved_tensors.append(uvqk)
    if rotary_weights is not None:
        saved_tensors.append(rotary_weights[0])
        saved_tensors.append(rotary_weights[1])
        saved_tensors.append(rotary_weights[2])
        saved_tensors.append(rotary_weights[3])
    ctx.save_for_backward(*saved_tensors)
    ctx.alpha = alpha
    ctx.invalid_attn_mask_type = invalid_attn_mask_type
    ctx.has_multiple_targets = num_targets is not None
    ctx.has_rotary_weights = rotary_weights is not None
    ctx.has_attn_scale = attn_scale is not None
    ctx.max_seq_len = max_seq_len
    ctx.max_attn_len = max_attn_len
    ctx.full_attn_size = full_attn_size
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
    ctx.silu_u = silu_u
    ctx.fp8_in_addmm_fwd = fp8_in_addmm_fwd
    return u, out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

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
def __HSTUPreprocessAndAttentionFunction_backward(ctx, _du: torch.Tensor,
    dout: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    None, None, None, None, torch.Tensor, torch.Tensor, None, None, None,
    None, None, None, None, None, None, None, None, None, None, None, None]:
    (x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight, seq_offsets, out
        ) = ctx.saved_tensors[:8]
    idx = 8
    if ctx.has_multiple_targets:
        num_targets = ctx.saved_tensors[idx]
        idx += 1
    else:
        num_targets = None
    if ctx.has_attn_scale:
        attn_scale = ctx.saved_tensors[idx]
        idx += 1
    else:
        attn_scale = None
    if ctx.recompute_normed_x_in_backward:
        normed_x, _, _, _, _, _ = triton_weighted_layer_norm_quantization_fwd(x
            =x, weight=norm_weight, bias=norm_bias, eps=ctx.norm_eps, mean=
            x_mean, rstd=x_rstd, quantize_output=ctx.fp8_in_addmm_fwd)
    else:
        normed_x = ctx.saved_tensors[idx]
        idx += 1
    if ctx.recompute_uvqk_in_backward:
        uvqk_bias = ctx.saved_tensors[idx]
        idx += 1
        if ctx.fp8_in_addmm_fwd:
            x_scale, normed_x_fp8 = ctx.saved_tensors[idx:idx + 2]
            uvqk = fp8_rowwise_quantize_addmm(x=normed_x, x_fp8=
                normed_x_fp8, w=uvqk_weight, y=uvqk_bias, x_scale=x_scale,
                custom_kernel=False, is_inference=False)
            idx += 2
        else:
            uvqk = maybe_triton_addmm_fwd(normed_x, uvqk_weight, uvqk_bias
                ).contiguous()
    else:
        uvqk = ctx.saved_tensors[idx]
        idx += 1
    if ctx.has_rotary_weights:
        q_cos_weights, q_sin_weights, k_cos_weights, k_sin_weights = (ctx.
            saved_tensors[idx:idx + 4])
        idx += 4
    else:
        q_cos_weights, q_sin_weights, k_cos_weights, k_sin_weights = (None,
            None, None, None)
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
    if ctx.recompute_uvqk_in_backward and ctx.has_rotary_weights:
        q = triton_apply_rope_fwd(x=q, N=ctx.max_seq_len, seq_offsets=
            seq_offsets, cos_rope=q_cos_weights, sin_rope=q_sin_weights)
        k = triton_apply_rope_fwd(x=k, N=ctx.max_seq_len, seq_offsets=
            seq_offsets, cos_rope=k_cos_weights, sin_rope=k_sin_weights)
    dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
    dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
    dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
    if is_sm100():
        _dq, _dk, _dv = torch.ops.bw_hstu.bw_hstu_mha_bwd(ctx.max_seq_len,
            ctx.alpha, dout, q, k, v, dq, dk, dv, seq_offsets, True,
            num_targets, attn_scale, ctx.max_attn_len, ctx.full_attn_size,
            ctx.contextual_seq_len, ctx.sort_by_length, False, 0, None,
            None, None, None, 1)
    else:
        _dq, _dk, _dv = torch.ops.hstu.hstu_mha_bwd(ctx.max_seq_len, ctx.
            alpha, dout, q, k, v, dq, dk, dv, out, seq_offsets, True,
            num_targets, attn_scale, ctx.max_attn_len, ctx.full_attn_size,
            ctx.contextual_seq_len, ctx.sort_by_length, False, 0)
    if ctx.has_rotary_weights:
        _dq = triton_apply_rope_bwd(grad=_dq, N=ctx.max_seq_len,
            seq_offsets=seq_offsets, cos_rope=q_cos_weights, sin_rope=
            q_sin_weights)
        _dk = triton_apply_rope_bwd(grad=_dk, N=ctx.max_seq_len,
            seq_offsets=seq_offsets, cos_rope=k_cos_weights, sin_rope=
            k_sin_weights)
    if dq.data_ptr() != _dq.data_ptr():
        dq.copy_(_dq)
    if dk.data_ptr() != _dk.data_ptr():
        dk.copy_(_dk)
    if dv.data_ptr() != _dv.data_ptr():
        dv.copy_(_dv)
    if ctx.silu_u:
        torch.ops.aten.silu_backward(_du, u, grad_input=du)
    elif du.data_ptr() != _du.data_ptr():
        du.copy_(_du)
    d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(x=normed_x, w
        =uvqk_weight, dz=duvqk, is_y_1d=ctx.uvqk_bias_1d)
    d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(dy=
        d_normed_x, x=x, weight=norm_weight, bias=norm_bias, mean=x_mean,
        rstd=x_rstd, learnable=True, eps=ctx.norm_eps, BLOCK_D=ctx.norm_BLOCK_D
        )
    return (d_x, d_norm_weight, d_norm_bias, None, None, None, None,
        d_uvqk_weight, d_uvqk_bias, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _HSTUPreprocessAndAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, norm_weight: torch.Tensor, norm_bias:
        torch.Tensor, norm_eps: float, num_heads: int, attn_dim: int,
        hidden_dim: int, uvqk_weight: torch.Tensor, uvqk_bias: torch.Tensor,
        max_seq_len: int, seq_offsets: torch.Tensor, alpha: float,
        invalid_attn_mask_type: str, num_targets: Optional[torch.Tensor],
        rotary_weights: Optional[Tuple[torch.Tensor, torch.Tensor, torch.
        Tensor, torch.Tensor]]=None, attn_scale: Optional[torch.Tensor]=
        None, recompute_uvqk_in_backward: bool=False,
        recompute_normed_x_in_backward: bool=False, contextual_seq_len: int
        =0, sort_by_length: bool=False, max_attn_len: Optional[int]=None,
        full_attn_size: Optional[int]=None, silu_u: bool=True,
        fp8_in_addmm_fwd: bool=False) ->Tuple[torch.Tensor, torch.Tensor]:
        max_attn_len = max_attn_len or 0
        full_attn_size = full_attn_size or 0
        normed_x, x_mean, x_rstd, BLOCK_D, x_scale, normed_x_fp8 = (
            triton_weighted_layer_norm_quantization_fwd(x=x, weight=
            norm_weight, bias=norm_bias, eps=norm_eps, quantize_output=
            fp8_in_addmm_fwd))
        if fp8_in_addmm_fwd:
            assert x_scale is not None and normed_x_fp8 is not None
            uvqk = fp8_rowwise_quantize_addmm(x=normed_x, x_fp8=
                normed_x_fp8, w=uvqk_weight, y=uvqk_bias, x_scale=x_scale,
                custom_kernel=False, is_inference=False).contiguous()
        else:
            uvqk = maybe_triton_addmm_fwd(normed_x, uvqk_weight, uvqk_bias
                ).contiguous()
        u, v, q, k = uvqk.split([hidden_dim * num_heads, hidden_dim *
            num_heads, attn_dim * num_heads, attn_dim * num_heads], dim=1)
        if rotary_weights is not None:
            q_cos_weights = rotary_weights[0]
            q_sin_weights = rotary_weights[1]
            k_cos_weights = rotary_weights[2]
            k_sin_weights = rotary_weights[3]
            _q = triton_apply_rope_fwd(x=q.view(-1, num_heads, attn_dim), N
                =max_seq_len, seq_offsets=seq_offsets, cos_rope=
                q_cos_weights, sin_rope=q_sin_weights).view(-1, num_heads *
                attn_dim)
            _k = triton_apply_rope_fwd(x=k.view(-1, num_heads, attn_dim), N
                =max_seq_len, seq_offsets=seq_offsets, cos_rope=
                k_cos_weights, sin_rope=k_sin_weights).view(-1, num_heads *
                attn_dim)
            if q.data_ptr() != _q.data_ptr():
                q.copy_(_q)
            if k.data_ptr() != _k.data_ptr():
                k.copy_(_k)
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        if silu_u:
            u = F.silu(u)
        elif recompute_uvqk_in_backward:
            u = u.clone()
        if is_sm100():
            out = torch.ops.bw_hstu.bw_hstu_mha_fwd(max_seq_len, alpha, q,
                k, v, seq_offsets, True, num_targets, attn_scale,
                max_attn_len, full_attn_size, contextual_seq_len, None,
                None, None, 0, None, None, None, None, 1)
        else:
            out, _ = torch.ops.hstu.hstu_mha_fwd(max_seq_len, alpha, q, k,
                v, seq_offsets, True, num_targets, attn_scale, max_attn_len,
                full_attn_size, contextual_seq_len, None, None, None, 0)
        saved_tensors = [x, norm_weight, norm_bias, x_mean, x_rstd,
            uvqk_weight, seq_offsets, out]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if attn_scale is not None:
            saved_tensors.append(attn_scale)
        if not recompute_normed_x_in_backward:
            saved_tensors.append(normed_x)
        if recompute_uvqk_in_backward:
            saved_tensors.append(uvqk_bias)
            if fp8_in_addmm_fwd:
                saved_tensors.append(x_scale)
                saved_tensors.append(normed_x_fp8)
        else:
            saved_tensors.append(uvqk)
        if rotary_weights is not None:
            saved_tensors.append(rotary_weights[0])
            saved_tensors.append(rotary_weights[1])
            saved_tensors.append(rotary_weights[2])
            saved_tensors.append(rotary_weights[3])
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.invalid_attn_mask_type = invalid_attn_mask_type
        ctx.has_multiple_targets = num_targets is not None
        ctx.has_rotary_weights = rotary_weights is not None
        ctx.has_attn_scale = attn_scale is not None
        ctx.max_seq_len = max_seq_len
        ctx.max_attn_len = max_attn_len
        ctx.full_attn_size = full_attn_size
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
        ctx.silu_u = silu_u
        ctx.fp8_in_addmm_fwd = fp8_in_addmm_fwd
        return u, out

    @staticmethod
    def backward(ctx, _du: torch.Tensor, dout: torch.Tensor) ->Tuple[torch.
        Tensor, torch.Tensor, torch.Tensor, None, None, None, None, torch.
        Tensor, torch.Tensor, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None]:
        (x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight,
            seq_offsets, out) = ctx.saved_tensors[:8]
        idx = 8
        if ctx.has_multiple_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        if ctx.has_attn_scale:
            attn_scale = ctx.saved_tensors[idx]
            idx += 1
        else:
            attn_scale = None
        if ctx.recompute_normed_x_in_backward:
            normed_x, _, _, _, _, _ = (
                triton_weighted_layer_norm_quantization_fwd(x=x, weight=
                norm_weight, bias=norm_bias, eps=ctx.norm_eps, mean=x_mean,
                rstd=x_rstd, quantize_output=ctx.fp8_in_addmm_fwd))
        else:
            normed_x = ctx.saved_tensors[idx]
            idx += 1
        if ctx.recompute_uvqk_in_backward:
            uvqk_bias = ctx.saved_tensors[idx]
            idx += 1
            if ctx.fp8_in_addmm_fwd:
                x_scale, normed_x_fp8 = ctx.saved_tensors[idx:idx + 2]
                uvqk = fp8_rowwise_quantize_addmm(x=normed_x, x_fp8=
                    normed_x_fp8, w=uvqk_weight, y=uvqk_bias, x_scale=
                    x_scale, custom_kernel=False, is_inference=False)
                idx += 2
            else:
                uvqk = maybe_triton_addmm_fwd(normed_x, uvqk_weight, uvqk_bias
                    ).contiguous()
        else:
            uvqk = ctx.saved_tensors[idx]
            idx += 1
        if ctx.has_rotary_weights:
            q_cos_weights, q_sin_weights, k_cos_weights, k_sin_weights = (ctx
                .saved_tensors[idx:idx + 4])
            idx += 4
        else:
            q_cos_weights, q_sin_weights, k_cos_weights, k_sin_weights = (
                None, None, None, None)
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
        if ctx.recompute_uvqk_in_backward and ctx.has_rotary_weights:
            q = triton_apply_rope_fwd(x=q, N=ctx.max_seq_len, seq_offsets=
                seq_offsets, cos_rope=q_cos_weights, sin_rope=q_sin_weights)
            k = triton_apply_rope_fwd(x=k, N=ctx.max_seq_len, seq_offsets=
                seq_offsets, cos_rope=k_cos_weights, sin_rope=k_sin_weights)
        dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
        dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
        dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
        if is_sm100():
            _dq, _dk, _dv = torch.ops.bw_hstu.bw_hstu_mha_bwd(ctx.
                max_seq_len, ctx.alpha, dout, q, k, v, dq, dk, dv,
                seq_offsets, True, num_targets, attn_scale, ctx.
                max_attn_len, ctx.full_attn_size, ctx.contextual_seq_len,
                ctx.sort_by_length, False, 0, None, None, None, None, 1)
        else:
            _dq, _dk, _dv = torch.ops.hstu.hstu_mha_bwd(ctx.max_seq_len,
                ctx.alpha, dout, q, k, v, dq, dk, dv, out, seq_offsets, 
                True, num_targets, attn_scale, ctx.max_attn_len, ctx.
                full_attn_size, ctx.contextual_seq_len, ctx.sort_by_length,
                False, 0)
        if ctx.has_rotary_weights:
            _dq = triton_apply_rope_bwd(grad=_dq, N=ctx.max_seq_len,
                seq_offsets=seq_offsets, cos_rope=q_cos_weights, sin_rope=
                q_sin_weights)
            _dk = triton_apply_rope_bwd(grad=_dk, N=ctx.max_seq_len,
                seq_offsets=seq_offsets, cos_rope=k_cos_weights, sin_rope=
                k_sin_weights)
        if dq.data_ptr() != _dq.data_ptr():
            dq.copy_(_dq)
        if dk.data_ptr() != _dk.data_ptr():
            dk.copy_(_dk)
        if dv.data_ptr() != _dv.data_ptr():
            dv.copy_(_dv)
        if ctx.silu_u:
            torch.ops.aten.silu_backward(_du, u, grad_input=du)
        elif du.data_ptr() != _du.data_ptr():
            du.copy_(_du)
        d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(x=
            normed_x, w=uvqk_weight, dz=duvqk, is_y_1d=ctx.uvqk_bias_1d)
        d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(dy
            =d_normed_x, x=x, weight=norm_weight, bias=norm_bias, mean=
            x_mean, rstd=x_rstd, learnable=True, eps=ctx.norm_eps, BLOCK_D=
            ctx.norm_BLOCK_D)
        return (d_x, d_norm_weight, d_norm_bias, None, None, None, None,
            d_uvqk_weight, d_uvqk_bias, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None)
