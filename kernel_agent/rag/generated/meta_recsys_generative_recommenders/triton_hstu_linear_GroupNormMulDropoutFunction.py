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
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

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


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


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


# Forward method (kernel launch code)
def _GroupNormMulDropoutFunction_forward(ctx, x: torch.Tensor, u: torch.
    Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float,
    dropout_ratio: float, training: bool, concat_ux: bool=False, num_heads:
    int=1, linear_dim: int=-1, seed: Optional[int]=None) ->torch.Tensor:
    y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
        triton_group_norm_mul_dropout_fwd(x=x, u=u, weight=weight, bias=
        bias, eps=eps, dropout_ratio=dropout_ratio, training=training,
        concat_ux=concat_ux, num_heads=num_heads, linear_dim=linear_dim,
        seed=seed))
    ctx.save_for_backward(x, u, weight, bias, mean, rstd)
    ctx.BLOCK_D = BLOCK_D
    ctx.BLOCK_H = BLOCK_H
    ctx.num_warps = num_warps
    ctx.eps = eps
    ctx.seed = seed
    ctx.training = training
    ctx.concat_ux = concat_ux
    ctx.dropout_ratio = dropout_ratio
    ctx.num_heads = num_heads
    ctx.linear_dim = linear_dim
    return y


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


# Backward method (kernel launch code)
def _GroupNormMulDropoutFunction_backward(ctx, dy: torch.Tensor) ->Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None,
    None, None, None, None, None]:
    x, u, weight, bias, mean, rstd = ctx.saved_tensors
    dx, du, dweight, dbias, _ = triton_group_norm_mul_dropout_bwd(dy=dy, x=
        x, u=u, weight=weight, bias=bias, mean=mean, rstd=rstd, BLOCK_D=ctx
        .BLOCK_D, BLOCK_H=ctx.BLOCK_H, num_warps=ctx.num_warps, eps=ctx.eps,
        training=ctx.training, dropout_ratio=ctx.dropout_ratio, seed=ctx.
        seed, concat_ux=ctx.concat_ux, num_heads=ctx.num_heads, linear_dim=
        ctx.linear_dim, compute_y=False)
    return dx, du, dweight, dbias, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class GroupNormMulDropoutFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, u: torch.Tensor, weight: torch.Tensor,
        bias: torch.Tensor, eps: float, dropout_ratio: float, training:
        bool, concat_ux: bool=False, num_heads: int=1, linear_dim: int=-1,
        seed: Optional[int]=None) ->torch.Tensor:
        y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
            triton_group_norm_mul_dropout_fwd(x=x, u=u, weight=weight, bias
            =bias, eps=eps, dropout_ratio=dropout_ratio, training=training,
            concat_ux=concat_ux, num_heads=num_heads, linear_dim=linear_dim,
            seed=seed))
        ctx.save_for_backward(x, u, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.BLOCK_H = BLOCK_H
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim = linear_dim
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, None, None, None, None, None, None, None]:
        x, u, weight, bias, mean, rstd = ctx.saved_tensors
        dx, du, dweight, dbias, _ = triton_group_norm_mul_dropout_bwd(dy=dy,
            x=x, u=u, weight=weight, bias=bias, mean=mean, rstd=rstd,
            BLOCK_D=ctx.BLOCK_D, BLOCK_H=ctx.BLOCK_H, num_warps=ctx.
            num_warps, eps=ctx.eps, training=ctx.training, dropout_ratio=
            ctx.dropout_ratio, seed=ctx.seed, concat_ux=ctx.concat_ux,
            num_heads=ctx.num_heads, linear_dim=ctx.linear_dim, compute_y=False
            )
        return dx, du, dweight, dbias, None, None, None, None, None, None, None
