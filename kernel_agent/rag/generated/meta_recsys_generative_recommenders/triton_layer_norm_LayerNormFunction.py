# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/triton/triton_layer_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/triton/triton_layer_norm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

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


def switch_to_contiguous_if_needed(x: torch.Tensor) ->torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10 ** 9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


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


# Forward method (kernel launch code)
def _LayerNormFunction_forward(ctx, x: torch.Tensor, weight: Optional[torch
    .Tensor], bias: Optional[torch.Tensor], eps: float) ->torch.Tensor:
    y, mean, rstd, BLOCK_D = triton_weighted_layer_norm_fwd(x=x, weight=
        weight, bias=bias, eps=eps)
    learnable = weight is not None
    if learnable:
        ctx.save_for_backward(x, weight, bias, mean, rstd)
    else:
        ctx.save_for_backward(x, mean, rstd)
    ctx.BLOCK_D = BLOCK_D
    ctx.eps = eps
    ctx.learnable = learnable
    return y


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
def _LayerNormFunction_backward(ctx, dy: torch.Tensor) ->Tuple[torch.Tensor,
    Optional[torch.Tensor], Optional[torch.Tensor], None]:
    if ctx.learnable:
        x, weight, bias, mean, rstd = ctx.saved_tensors
    else:
        x, mean, rstd = ctx.saved_tensors
        weight, bias = None, None
    dx, dweight, dbias = triton_weighted_layer_norm_bwd(dy=dy, x=x, weight=
        weight, bias=bias, mean=mean, rstd=rstd, learnable=ctx.learnable,
        eps=ctx.eps, BLOCK_D=ctx.BLOCK_D)
    return dx, dweight, dbias, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: Optional[torch.Tensor], bias:
        Optional[torch.Tensor], eps: float) ->torch.Tensor:
        y, mean, rstd, BLOCK_D = triton_weighted_layer_norm_fwd(x=x, weight
            =weight, bias=bias, eps=eps)
        learnable = weight is not None
        if learnable:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
        else:
            ctx.save_for_backward(x, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.eps = eps
        ctx.learnable = learnable
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor) ->Tuple[torch.Tensor, Optional[
        torch.Tensor], Optional[torch.Tensor], None]:
        if ctx.learnable:
            x, weight, bias, mean, rstd = ctx.saved_tensors
        else:
            x, mean, rstd = ctx.saved_tensors
            weight, bias = None, None
        dx, dweight, dbias = triton_weighted_layer_norm_bwd(dy=dy, x=x,
            weight=weight, bias=bias, mean=mean, rstd=rstd, learnable=ctx.
            learnable, eps=ctx.eps, BLOCK_D=ctx.BLOCK_D)
        return dx, dweight, dbias, None
