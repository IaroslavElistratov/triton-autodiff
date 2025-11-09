# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/triton/triton_hstu_linear.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/triton/triton_hstu_linear.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

@triton.jit
def _generate_random_mask(MASK_BUFFER, N_MASK, dropout_ratio, seed, D: tl.
    constexpr, STRIDE: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    random_offsets = pid * BLOCK_D + cols
    rand1, rand2, rand3, rand4 = tl.rand4x(seed, random_offsets)
    start_row = pid * 4
    MASK_BUFFER += start_row * STRIDE
    row_mask = start_row < N_MASK
    mask1 = rand1 > dropout_ratio
    tl.store(MASK_BUFFER + cols, mask1, mask=row_mask & col_mask)
    row_mask = start_row + 1 < N_MASK
    mask2 = rand2 > dropout_ratio
    tl.store(MASK_BUFFER + STRIDE + cols, mask2, mask=row_mask & col_mask)
    row_mask = start_row + 2 < N_MASK
    mask3 = rand3 > dropout_ratio
    tl.store(MASK_BUFFER + 2 * STRIDE + cols, mask3, mask=row_mask & col_mask)
    row_mask = start_row + 3 < N_MASK
    mask4 = rand4 > dropout_ratio
    tl.store(MASK_BUFFER + 3 * STRIDE + cols, mask4, mask=row_mask & col_mask)


@triton.jit
def rand3x(seed, offsets, n_rounds: tl.constexpr=10):
    i1, i2, i3, _ = tl.randint4x(seed, offsets, n_rounds)
    u1 = tl.uint_to_uniform_float(i1)
    u2 = tl.uint_to_uniform_float(i2)
    u3 = tl.uint_to_uniform_float(i3)
    return u1, u2, u3


def is_sm100() ->bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 0


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

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


@triton.jit
def _group_norm_mul_dropout_fwd(X, U, Y, W, B, Mean, Rstd, D, Heads, eps,
    seed, dropout_ratio, stride_x, stride_u, stride_y, SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr, TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_H)
    offsets = heads[:, None] * D + cols[None, :]
    mask_h = heads < Heads
    mask_c = cols < D
    mask = mask_c[None, :] & mask_h[:, None]
    mean = 0.0
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    mean = tl.ravel(mean)
    _var = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(mask, x - mean[:, None], 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=1) / D
    var = tl.ravel(var)
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row * Heads + heads, mean, mask=mask_h)
    tl.store(Rstd + row * Heads + heads, rstd, mask=mask_h)
    y = x_mean * rstd[:, None]
    w = tl.load(W + heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + heads, mask=mask_h).to(tl.float32)
    y = y * w[:, None] + b[:, None]
    u = tl.load(U + offsets, mask=mask, other=0.0).to(tl.float32)
    if SILU_U:
        u = fast_dividef(u, 1.0 + tl.exp(-u))
    y = y * u
    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            random_x = tl.rand(seed, random_offsets + Heads * D)
            x_keep = random_x > dropout_ratio
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            random_y = tl.rand(seed, random_offsets + 2 * Heads * D)
            y_keep = random_y > dropout_ratio
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            y_keep = random > dropout_ratio
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
    if CONCAT_UX:
        tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty),
            mask=mask)
    else:
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _ln_mul_dropout_fwd(X, U, Y, W, B, Mean, Rstd, D, eps, seed,
    dropout_ratio, stride_x, stride_u, stride_y, SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr, TRAINING: tl.constexpr, CONCAT_UX: tl.constexpr,
    FAST_DROPOUT: tl.constexpr):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    mean = 0.0
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = y * w + b
    u = tl.load(U + cols, mask=cols < D, other=0.0).to(tl.float32)
    if SILU_U:
        u = fast_dividef(u, 1.0 + tl.exp(-u))
    y = y * u
    if TRAINING:
        random_offsets = 3 * row * BLOCK_D + cols
        if CONCAT_UX:
            if FAST_DROPOUT:
                random_u, random_x, random_y = rand3x(seed, random_offsets)
            else:
                random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            if not FAST_DROPOUT:
                random_x = tl.rand(seed, random_offsets + D)
            x_keep = random_x > dropout_ratio
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            if not FAST_DROPOUT:
                random_y = tl.rand(seed, random_offsets + 2 * D)
            y_keep = random_y > dropout_ratio
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random = tl.rand(seed, random_offsets)
            y_keep = random > dropout_ratio
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
    if CONCAT_UX:
        tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton_autotune(configs=_get_layer_norm_mul_dropout_fwd_multirow_configs(),
    key=['BLOCK_D'])
@triton.jit
def _ln_mul_dropout_fwd_rng(X, U, Y, W, B, Mean, Rstd, RANDOM_MASK, N, D,
    eps, dropout_ratio, stride_x, stride_u, stride_y, stride_mask, SILU_U:
    tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, TRAINING:
    tl.constexpr, CONCAT_UX: tl.constexpr):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N
    X_block_ptr = tl.make_block_ptr(base=X, shape=(N, D), strides=(stride_x,
        1), offsets=(start_row, 0), block_shape=(BLOCK_N, BLOCK_D), order=(
        1, 0))
    U_block_ptr = tl.make_block_ptr(base=U, shape=(N, D), strides=(stride_u,
        1), offsets=(start_row, 0), block_shape=(BLOCK_N, BLOCK_D), order=(
        1, 0))
    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option='zero'
        ).to(tl.float32)
    u_block = tl.load(U_block_ptr, boundary_check=(0, 1), padding_option='zero'
        ).to(tl.float32)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N
    mean = tl.sum(x_block, axis=1) / D
    tl.store(Mean + rows, mean, mask=row_mask)
    mean = tl.expand_dims(mean, 1)
    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    rstd = tl.expand_dims(rstd, 1)
    y = x_mean * rstd
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    y = y * w[None, :] + b[None, :]
    if SILU_U:
        u_block = fast_dividef(u_block, 1.0 + tl.exp(-u_block))
    y = y * u_block
    if TRAINING:
        if CONCAT_UX:
            row_offsets = start_row + tl.arange(0, BLOCK_N)
            col_offsets = tl.arange(0, BLOCK_D)
            u_offsets = row_offsets[:, None] * stride_mask + col_offsets[
                None, :]
            x_offsets = (row_offsets[:, None] + N) * stride_mask + col_offsets[
                None, :]
            y_offsets = (row_offsets[:, None] + 2 * N
                ) * stride_mask + col_offsets[None, :]
            mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < D)
            u_keep = tl.load(RANDOM_MASK + u_offsets, mask=mask, other=True)
            x_keep = tl.load(RANDOM_MASK + x_offsets, mask=mask, other=True)
            y_keep = tl.load(RANDOM_MASK + y_offsets, mask=mask, other=True)
            u_block = tl.where(u_keep, u_block / (1.0 - dropout_ratio), 0.0)
            x_block = tl.where(x_keep, x_block / (1.0 - dropout_ratio), 0.0)
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            row_offsets = start_row + tl.arange(0, BLOCK_N)
            col_offsets = tl.arange(0, BLOCK_D)
            y_offsets = row_offsets[:, None] * stride_mask + col_offsets[
                None, :]
            mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < D)
            y_keep = tl.load(RANDOM_MASK + y_offsets, mask=mask, other=True)
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
    if CONCAT_UX:
        Y_block_ptr_u = tl.make_block_ptr(base=Y, shape=(N, 3 * D), strides
            =(stride_y, 1), offsets=(start_row, 0), block_shape=(BLOCK_N,
            BLOCK_D), order=(1, 0))
        Y_block_ptr_x = tl.make_block_ptr(base=Y, shape=(N, 3 * D), strides
            =(stride_y, 1), offsets=(start_row, D), block_shape=(BLOCK_N,
            BLOCK_D), order=(1, 0))
        Y_block_ptr_y = tl.make_block_ptr(base=Y, shape=(N, 3 * D), strides
            =(stride_y, 1), offsets=(start_row, 2 * D), block_shape=(
            BLOCK_N, BLOCK_D), order=(1, 0))
        tl.store(Y_block_ptr_u, u_block.to(Y.dtype.element_ty),
            boundary_check=(0, 1))
        tl.store(Y_block_ptr_x, x_block.to(Y.dtype.element_ty),
            boundary_check=(0, 1))
        tl.store(Y_block_ptr_y, y.to(Y.dtype.element_ty), boundary_check=(0, 1)
            )
    else:
        Y_block_ptr = tl.make_block_ptr(base=Y, shape=(N, D), strides=(
            stride_y, 1), offsets=(start_row, 0), block_shape=(BLOCK_N,
            BLOCK_D), order=(1, 0))
        tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


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


def triton_group_norm_mul_dropout_fwd(x: torch.Tensor, u: torch.Tensor,
    weight: torch.Tensor, bias: torch.Tensor, eps: float, dropout_ratio:
    float, training: bool, silu_u: bool=False, concat_ux: bool=False,
    num_heads: int=1, linear_dim: int=-1, seed: Optional[int]=None) ->Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    assert x.dim() == 2
    assert x.shape == u.shape
    assert x.shape[1] == num_heads * linear_dim
    x = switch_to_contiguous_if_needed(x)
    u = switch_to_contiguous_if_needed(u)
    N, _ = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == num_heads
    assert bias.numel() == num_heads
    if concat_ux:
        y = torch.empty((N, 3 * num_heads * linear_dim), dtype=x.dtype,
            device=x.device)
    else:
        y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype, device=
            x.device)
    mean = torch.empty((N * num_heads,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N * num_heads,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, 0, 0, 0, 0
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = triton.next_power_of_2(linear_dim)
    BLOCK_H: int = triton.next_power_of_2(num_heads)
    if BLOCK_D * BLOCK_H > MAX_FUSED_SIZE:
        raise RuntimeError(
            "This group norm doesn't support num_heads * linear_dim >= 64KB.")
    if seed is None:
        seed = torch.randint(low=0, high=2 ** 62, size=(1,), dtype=torch.int64
            ).item()
    num_warps: int = min(max(BLOCK_D * BLOCK_H // 256, 1), 8)
    _group_norm_mul_dropout_fwd[N,](x, u, y, weight, bias, mean, rstd,
        linear_dim, num_heads, eps, seed, dropout_ratio, x.stride(0), u.
        stride(0), y.stride(0), SILU_U=silu_u, BLOCK_D=BLOCK_D, BLOCK_H=
        BLOCK_H, TRAINING=training, CONCAT_UX=concat_ux, num_warps=num_warps)
    return y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed


def triton_layer_norm_mul_dropout_fwd(x: torch.Tensor, u: torch.Tensor,
    weight: torch.Tensor, bias: torch.Tensor, eps: float, dropout_ratio:
    float, training: bool, silu_u: bool=False, concat_ux: bool=False, seed:
    Optional[int]=None) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    int, int, int]:
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == D
    assert bias.numel() == D
    if concat_ux:
        y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, 0, 0, 0
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    if seed is None:
        seed = torch.randint(low=0, high=2 ** 62, size=(1,), dtype=torch.int64
            ).item()
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    sms = torch.cuda.get_device_properties('cuda').multi_processor_count
    if not FUSE_OUTPUT_LN_RNG_BLACKWELL and is_sm100(
        ) and not concat_ux and training:
        random_mask = torch.empty([N, D], dtype=torch.bool, device=x.device)
        _generate_random_mask[triton.cdiv(N, 4),](random_mask, N,
            dropout_ratio, seed, D, random_mask.stride(0), BLOCK_D)

        def grid(META):
            return triton.cdiv(N, META['BLOCK_N']),
        _ln_mul_dropout_fwd_rng[grid](x, u, y, weight, bias, mean, rstd,
            random_mask, N, D, eps, dropout_ratio, x.stride(0), u.stride(0),
            y.stride(0), random_mask.stride(0), SILU_U=silu_u, BLOCK_D=
            BLOCK_D, TRAINING=training, CONCAT_UX=concat_ux)
    else:
        _ln_mul_dropout_fwd[N,](x, u, y, weight, bias, mean, rstd, D, eps,
            seed, dropout_ratio, x.stride(0), u.stride(0), y.stride(0),
            SILU_U=silu_u, BLOCK_D=BLOCK_D, TRAINING=training, CONCAT_UX=
            concat_ux, FAST_DROPOUT=COMPUTE_OUTPUT_LN_FAST_DROPOUT,
            num_warps=num_warps)
    return y, mean, rstd, BLOCK_D, num_warps, seed


# Forward method (kernel launch code)
def _HSTUComputeOutputFunction_forward(ctx, attn: torch.Tensor, u: torch.
    Tensor, x: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.
    Tensor, output_weight: torch.Tensor, eps: float, dropout_ratio: float,
    training: bool, silu_u: bool=False, concat_ux: bool=False, group_norm:
    bool=False, num_heads: int=1, linear_dim: int=-1, seed: Optional[int]=
    None, recompute_y_in_backward: bool=False) ->torch.Tensor:
    if group_norm:
        y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
            triton_group_norm_mul_dropout_fwd(x=attn, u=u, weight=
            norm_weight, bias=norm_bias, eps=eps, dropout_ratio=
            dropout_ratio, training=training, silu_u=silu_u, concat_ux=
            concat_ux, num_heads=num_heads, linear_dim=linear_dim, seed=seed))
        ctx.BLOCK_H = BLOCK_H
    else:
        y, mean, rstd, BLOCK_D, num_warps, seed = (
            triton_layer_norm_mul_dropout_fwd(x=attn, u=u, weight=
            norm_weight, bias=norm_bias, eps=eps, dropout_ratio=
            dropout_ratio, training=training, silu_u=silu_u, concat_ux=
            concat_ux, seed=seed))
    out = maybe_triton_addmm_fwd(x=y, w=output_weight, y=x)
    saved_tensors = [attn, u, norm_weight, norm_bias, mean, rstd, output_weight
        ]
    if not recompute_y_in_backward:
        saved_tensors.append(y)
    ctx.save_for_backward(*saved_tensors)
    ctx.BLOCK_D = BLOCK_D
    ctx.num_warps = num_warps
    ctx.eps = eps
    ctx.seed = seed
    ctx.training = training
    ctx.concat_ux = concat_ux
    ctx.dropout_ratio = dropout_ratio
    ctx.num_heads = num_heads
    ctx.linear_dim = linear_dim
    ctx.group_norm = group_norm
    ctx.recompute_y_in_backward = recompute_y_in_backward
    ctx.silu_u = silu_u
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton_autotune(configs=_get_bwd_dwdb_configs(), key=[])
@triton.jit
def _group_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, N, BLOCK_N: tl.constexpr):
    col = tl.program_id(0)
    num_heads = tl.num_programs(0)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = rows < N
        offs = rows * num_heads + col
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + col, sum_dw.to(FINAL_DW.dtype.element_ty))
    tl.store(FINAL_DB + col, sum_db.to(FINAL_DB.dtype.element_ty))


@triton.jit
def _group_norm_mul_dropout_bwd_dx_du(DX, DU, DY, DW, DB, X, U, Y, W, B,
    Mean, Rstd, stride_dx, stride_du, stride_dy, stride_x, stride_u,
    stride_y, D, Heads, eps, seed, dropout_ratio, SILU_U: tl.constexpr,
    GROUP_N: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_H: tl.constexpr,
    TRAINING: tl.constexpr, CONCAT_UX: tl.constexpr, COMPUTE_Y: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    off_heads = tl.arange(0, BLOCK_H)
    mask_c = cols < D
    mask_h = off_heads < Heads
    mask = mask_c[None, :] & mask_h[:, None]
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    offsets = off_heads[:, None] * D + cols[None, :]
    x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
    if CONCAT_UX:
        du = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
        dx = tl.load(DY + Heads * D + offsets, mask=mask, other=0).to(tl.
            float32)
        dy = tl.load(DY + 2 * Heads * D + offsets, mask=mask, other=0).to(tl
            .float32)
    else:
        du = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dx = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dy = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            random_du = tl.rand(seed, random_offsets)
            du_keep = random_du > dropout_ratio
            du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
            random_dx = tl.rand(seed, random_offsets + Heads * D)
            dx_keep = random_dx > dropout_ratio
            dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
            random_dy = tl.rand(seed, random_offsets + 2 * Heads * D)
            dy_keep = random_dy > dropout_ratio
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            dy_keep = random > dropout_ratio
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
    mean = tl.load(Mean + row * Heads + off_heads)
    rstd = tl.load(Rstd + row * Heads + off_heads)
    xhat = (x - mean[:, None]) * rstd[:, None]
    w = tl.load(W + off_heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + off_heads, mask=mask_h).to(tl.float32)
    u = tl.load(U + offsets, mask=mask, other=0).to(tl.float32)
    ln = xhat * w[:, None] + b[:, None]
    du += dy * ln
    if SILU_U:
        sig_u = fast_dividef(1.0, 1.0 + tl.exp(-u))
        du = du * (sig_u + u * sig_u * (1.0 - sig_u))
        u = u * sig_u
    tl.store(DU + offsets, du.to(DU.dtype.element_ty), mask=mask)
    dy = dy * u
    wdy = w[:, None] * dy
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
        y = ln * u
        if TRAINING:
            if CONCAT_UX:
                u = tl.where(du_keep, u / (1.0 - dropout_ratio), 0.0)
                x = tl.where(dx_keep, x / (1.0 - dropout_ratio), 0.0)
                y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
            else:
                y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
        if CONCAT_UX:
            tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty),
                mask=mask)
            tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty),
                mask=mask)
        else:
            tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=1) / D
    c2 = tl.sum(wdy, axis=1) / D
    dx += (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    tl.store(DX + offsets, dx, mask=mask)
    lock_id = row % GROUP_N
    DW = DW + lock_id * Heads + off_heads
    DB = DB + lock_id * Heads + off_heads
    partial_dw = tl.sum(dy * xhat, axis=1)
    partial_dw = tl.ravel(partial_dw)
    partial_db = tl.sum(dy, axis=1)
    partial_db = tl.ravel(partial_db)
    tl.atomic_add(DW, partial_dw, mask=mask_h, sem='relaxed')
    tl.atomic_add(DB, partial_db, mask=mask_h, sem='relaxed')


@triton_autotune(configs=_get_bwd_dwdb_configs(), key=['D'])
@triton.jit
def _ln_mul_dropout_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, N, D, BLOCK_N: tl.
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
def _ln_mul_dropout_bwd_dx_du(DX, DU, DY, DW, DB, X, U, Y, W, B, Mean, Rstd,
    stride_dx, stride_du, stride_dy, stride_x, stride_u, stride_y, D, eps,
    seed, dropout_ratio, N, SILU_U: tl.constexpr, BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr, CONCAT_UX: tl.constexpr, COMPUTE_Y: tl.
    constexpr, FAST_DROPOUT: tl.constexpr):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1
    if rows_per_tile == 0:
        return
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols
    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    for _idx in range(0, rows_per_tile):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_UX:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if TRAINING:
            random_offsets = 3 * row * BLOCK_D + cols
            if CONCAT_UX:
                if FAST_DROPOUT:
                    random_du, random_dx, random_dy = rand3x(seed,
                        random_offsets)
                else:
                    random_du = tl.rand(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                if not FAST_DROPOUT:
                    random_dx = tl.rand(seed, random_offsets + D)
                dx_keep = random_dx > dropout_ratio
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                if not FAST_DROPOUT:
                    random_dy = tl.rand(seed, random_offsets + 2 * D)
                dy_keep = random_dy > dropout_ratio
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            else:
                random = tl.rand(seed, random_offsets)
                dy_keep = random > dropout_ratio
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        ln = xhat * w + b
        du += dy * ln
        if SILU_U:
            sig_u = fast_dividef(1.0, 1.0 + tl.exp(-u))
            du = du * (sig_u + u * sig_u * (1.0 - sig_u))
            u = u * sig_u
        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)
        dy = dy * u
        wdy = w * dy
        if COMPUTE_Y:
            y = ln * u
            if TRAINING:
                if CONCAT_UX:
                    u = tl.where(du_keep, u / (1.0 - dropout_ratio), 0.0)
                    x = tl.where(dx_keep, x / (1.0 - dropout_ratio), 0.0)
                    y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
                else:
                    y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
            if CONCAT_UX:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / D
        c2 = tl.sum(wdy, axis=0) / D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + cols, dx, mask=mask)
        partial_dw += dy * xhat
        partial_db += dy
        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du
        row += tile_num
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)


@triton.jit
def _ln_mul_dropout_bwd_dx_du_rng(DX, DU, DY, DW, DB, X, U, Y, W, B, Mean,
    Rstd, RANDOM_MASK, stride_dx, stride_du, stride_dy, stride_x, stride_u,
    stride_y, stride_mask, D, eps, dropout_ratio, N, SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr, TRAINING: tl.constexpr, CONCAT_UX: tl.constexpr,
    COMPUTE_Y: tl.constexpr):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1
    if rows_per_tile == 0:
        return
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols
    num_random = 1
    if CONCAT_UX:
        num_random = 3
    RANDOM_MASK += row.to(tl.int64) * stride_mask * num_random
    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    for _ in range(0, rows_per_tile):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_UX:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if TRAINING:
            if CONCAT_UX:
                du_keep = tl.load(RANDOM_MASK + cols, mask=mask, other=True)
                dx_keep = tl.load(RANDOM_MASK + stride_mask + cols, mask=
                    mask, other=True)
                dy_keep = tl.load(RANDOM_MASK + 2 * stride_mask + cols,
                    mask=mask, other=True)
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            else:
                dy_keep = tl.load(RANDOM_MASK + cols, mask=mask, other=True)
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        ln = xhat * w + b
        du += dy * ln
        if SILU_U:
            sig_u = fast_dividef(1.0, 1.0 + tl.exp(-u))
            du = du * (sig_u + u * sig_u * (1.0 - sig_u))
            u = u * sig_u
        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)
        dy = dy * u
        wdy = w * dy
        if COMPUTE_Y:
            y = ln * u
            if TRAINING:
                if CONCAT_UX:
                    u = tl.where(du_keep, u / (1.0 - dropout_ratio), 0.0)
                    x = tl.where(dx_keep, x / (1.0 - dropout_ratio), 0.0)
                    y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
                else:
                    y = tl.where(dy_keep, y / (1.0 - dropout_ratio), 0.0)
            if CONCAT_UX:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / D
        c2 = tl.sum(wdy, axis=0) / D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + cols, dx, mask=mask)
        partial_dw += dy * xhat
        partial_db += dy
        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du
        RANDOM_MASK += tile_num.to(tl.int64) * stride_mask * num_random
        row += tile_num
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)


def triton_group_norm_mul_dropout_bwd(dy: torch.Tensor, x: torch.Tensor, u:
    torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mean: torch.
    Tensor, rstd: torch.Tensor, BLOCK_D: int, BLOCK_H: int, num_warps: int,
    eps: float, training: bool, dropout_ratio: float, seed: Optional[int]=
    None, silu_u: bool=False, concat_ux: bool=False, num_heads: int=1,
    linear_dim: int=-1, compute_y: bool=False) ->Tuple[torch.Tensor, torch.
    Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    y = None
    N, dim = x.shape
    if compute_y:
        if concat_ux:
            y = torch.empty((N, 3 * num_heads * linear_dim), dtype=x.dtype,
                device=x.device)
        else:
            y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype,
                device=x.device)
    if N == 0:
        return torch.zeros_like(x), torch.zeros_like(u), torch.zeros_like(
            weight), torch.zeros_like(bias), y
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    if dim <= 1024:
        GROUP_N = 256 * 8
    elif dim <= 4096:
        GROUP_N = 128 * 8
    elif dim <= 8192:
        GROUP_N = 96 * 8
    else:
        GROUP_N = 64 * 8
    GROUP_N = N if GROUP_N > N else GROUP_N
    _dweight = torch.zeros((GROUP_N, num_heads), dtype=torch.float32,
        device=x.device)
    _dbias = torch.zeros((GROUP_N, num_heads), dtype=torch.float32, device=
        x.device)
    dweight = torch.empty((num_heads,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((num_heads,), dtype=weight.dtype, device=x.device)
    _group_norm_mul_dropout_bwd_dx_du[N,](dx, du, dy, _dweight, _dbias, x,
        u, y, weight, bias, mean, rstd, dx.stride(0), du.stride(0), dy.
        stride(0), x.stride(0), u.stride(0), y.stride(0) if compute_y else 
        0, linear_dim, num_heads, eps, seed, dropout_ratio, SILU_U=silu_u,
        GROUP_N=GROUP_N, BLOCK_D=BLOCK_D, BLOCK_H=BLOCK_H, TRAINING=
        training, CONCAT_UX=concat_ux, COMPUTE_Y=compute_y, num_warps=num_warps
        )
    _group_norm_bwd_dwdb[num_heads,](_dweight, _dbias, dweight, dbias, GROUP_N)
    return dx, du, dweight, dbias, y


def triton_layer_norm_mul_dropout_bwd(dy: torch.Tensor, x: torch.Tensor, u:
    torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, mean: torch.
    Tensor, rstd: torch.Tensor, BLOCK_D: int, num_warps: int, eps: float,
    training: bool, dropout_ratio: float, seed: Optional[int]=None, silu_u:
    bool=False, concat_ux: bool=False, compute_y: bool=False) ->Tuple[torch
    .Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    y = None
    N, D = x.shape
    if compute_y:
        if concat_ux:
            y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
        else:
            y = torch.empty_like(x)
    if N == 0:
        return torch.zeros_like(x), torch.zeros_like(u), torch.zeros((D,),
            dtype=weight.dtype, device=x.device), torch.zeros((D,), dtype=
            weight.dtype, device=x.device), y
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num = max(1, min(sms * 64, N // 4))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
    if not FUSE_OUTPUT_LN_RNG_BLACKWELL and is_sm100(
        ) and not concat_ux and training:
        random_mask = torch.empty([N, D], dtype=torch.bool, device=x.device)
        _generate_random_mask[triton.cdiv(N, 4),](random_mask, N,
            dropout_ratio, seed, D, random_mask.stride(0), BLOCK_D)
        _ln_mul_dropout_bwd_dx_du_rng[tile_num,](dx, du, dy, _dweight,
            _dbias, x, u, y, weight, bias, mean, rstd, random_mask, dx.
            stride(0), du.stride(0), dy.stride(0), x.stride(0), u.stride(0),
            y.stride(0) if compute_y else 0, random_mask.stride(0), D, eps,
            dropout_ratio, N=N, SILU_U=silu_u, BLOCK_D=BLOCK_D, TRAINING=
            training, CONCAT_UX=concat_ux, COMPUTE_Y=compute_y, num_warps=
            num_warps)
    else:
        _ln_mul_dropout_bwd_dx_du[tile_num,](dx, du, dy, _dweight, _dbias,
            x, u, y, weight, bias, mean, rstd, dx.stride(0), du.stride(0),
            dy.stride(0), x.stride(0), u.stride(0), y.stride(0) if
            compute_y else 0, D, eps, seed, dropout_ratio, N=N, SILU_U=
            silu_u, BLOCK_D=BLOCK_D, TRAINING=training, CONCAT_UX=concat_ux,
            COMPUTE_Y=compute_y, FAST_DROPOUT=
            COMPUTE_OUTPUT_LN_FAST_DROPOUT, num_warps=num_warps)

    def grid(META):
        return triton.cdiv(D, META['BLOCK_D']),
    blocks = triton.next_power_of_2(sms * 4)
    BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
    BLOCK_D = min(max(BLOCK_D, 4), 128)
    _ln_mul_dropout_bwd_dwdb[grid](_dweight, _dbias, dweight, dbias,
        tile_num, D, BLOCK_D=BLOCK_D)
    return dx, du, dweight, dbias, y


# Backward method (kernel launch code)
def _HSTUComputeOutputFunction_backward(ctx, dout: torch.Tensor) ->Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, None, None, None, None, None, None, None, None, None, None]:
    attn, u, norm_weight, norm_bias, mean, rstd, output_weight = (ctx.
        saved_tensors[:7])
    dy = torch.mm(dout, output_weight.t())
    if ctx.group_norm:
        dattn, du, d_norm_weight, d_norm_bias, y = (
            triton_group_norm_mul_dropout_bwd(dy=dy, x=attn, u=u, weight=
            norm_weight, bias=norm_bias, mean=mean, rstd=rstd, BLOCK_D=ctx.
            BLOCK_D, BLOCK_H=ctx.BLOCK_H, num_warps=ctx.num_warps, eps=ctx.
            eps, training=ctx.training, dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed, silu_u=ctx.silu_u, concat_ux=ctx.concat_ux,
            num_heads=ctx.num_heads, linear_dim=ctx.linear_dim, compute_y=
            ctx.recompute_y_in_backward))
    else:
        dattn, du, d_norm_weight, d_norm_bias, y = (
            triton_layer_norm_mul_dropout_bwd(dy=dy, x=attn, u=u, weight=
            norm_weight, bias=norm_bias, mean=mean, rstd=rstd, BLOCK_D=ctx.
            BLOCK_D, num_warps=ctx.num_warps, eps=ctx.eps, training=ctx.
            training, dropout_ratio=ctx.dropout_ratio, seed=ctx.seed,
            silu_u=ctx.silu_u, concat_ux=ctx.concat_ux, compute_y=ctx.
            recompute_y_in_backward))
    if not ctx.recompute_y_in_backward:
        y = ctx.saved_tensors[7]
    d_output_weight = torch.mm(y.t(), dout)
    return (dattn, du, dout, d_norm_weight, d_norm_bias, d_output_weight,
        None, None, None, None, None, None, None, None, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class HSTUComputeOutputFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, attn: torch.Tensor, u: torch.Tensor, x: torch.Tensor,
        norm_weight: torch.Tensor, norm_bias: torch.Tensor, output_weight:
        torch.Tensor, eps: float, dropout_ratio: float, training: bool,
        silu_u: bool=False, concat_ux: bool=False, group_norm: bool=False,
        num_heads: int=1, linear_dim: int=-1, seed: Optional[int]=None,
        recompute_y_in_backward: bool=False) ->torch.Tensor:
        if group_norm:
            y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
                triton_group_norm_mul_dropout_fwd(x=attn, u=u, weight=
                norm_weight, bias=norm_bias, eps=eps, dropout_ratio=
                dropout_ratio, training=training, silu_u=silu_u, concat_ux=
                concat_ux, num_heads=num_heads, linear_dim=linear_dim, seed
                =seed))
            ctx.BLOCK_H = BLOCK_H
        else:
            y, mean, rstd, BLOCK_D, num_warps, seed = (
                triton_layer_norm_mul_dropout_fwd(x=attn, u=u, weight=
                norm_weight, bias=norm_bias, eps=eps, dropout_ratio=
                dropout_ratio, training=training, silu_u=silu_u, concat_ux=
                concat_ux, seed=seed))
        out = maybe_triton_addmm_fwd(x=y, w=output_weight, y=x)
        saved_tensors = [attn, u, norm_weight, norm_bias, mean, rstd,
            output_weight]
        if not recompute_y_in_backward:
            saved_tensors.append(y)
        ctx.save_for_backward(*saved_tensors)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim = linear_dim
        ctx.group_norm = group_norm
        ctx.recompute_y_in_backward = recompute_y_in_backward
        ctx.silu_u = silu_u
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor) ->Tuple[torch.Tensor, torch.
        Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        None, None, None, None, None, None, None, None, None, None]:
        attn, u, norm_weight, norm_bias, mean, rstd, output_weight = (ctx.
            saved_tensors[:7])
        dy = torch.mm(dout, output_weight.t())
        if ctx.group_norm:
            dattn, du, d_norm_weight, d_norm_bias, y = (
                triton_group_norm_mul_dropout_bwd(dy=dy, x=attn, u=u,
                weight=norm_weight, bias=norm_bias, mean=mean, rstd=rstd,
                BLOCK_D=ctx.BLOCK_D, BLOCK_H=ctx.BLOCK_H, num_warps=ctx.
                num_warps, eps=ctx.eps, training=ctx.training,
                dropout_ratio=ctx.dropout_ratio, seed=ctx.seed, silu_u=ctx.
                silu_u, concat_ux=ctx.concat_ux, num_heads=ctx.num_heads,
                linear_dim=ctx.linear_dim, compute_y=ctx.
                recompute_y_in_backward))
        else:
            dattn, du, d_norm_weight, d_norm_bias, y = (
                triton_layer_norm_mul_dropout_bwd(dy=dy, x=attn, u=u,
                weight=norm_weight, bias=norm_bias, mean=mean, rstd=rstd,
                BLOCK_D=ctx.BLOCK_D, num_warps=ctx.num_warps, eps=ctx.eps,
                training=ctx.training, dropout_ratio=ctx.dropout_ratio,
                seed=ctx.seed, silu_u=ctx.silu_u, concat_ux=ctx.concat_ux,
                compute_y=ctx.recompute_y_in_backward))
        if not ctx.recompute_y_in_backward:
            y = ctx.saved_tensors[7]
        d_output_weight = torch.mm(y.t(), dout)
        return (dattn, du, dout, d_norm_weight, d_norm_bias,
            d_output_weight, None, None, None, None, None, None, None, None,
            None, None)
