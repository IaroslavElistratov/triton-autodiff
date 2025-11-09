# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/OpenSparseLLMs/Linear-MoE
# Source-Files: linear_moe/model/common_modules/layernorm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_kyohjmxq/Linear-MoE-main/linear_moe/model/common_modules/layernorm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT',
    'IS_RMS_NORM', 'HAS_BIAS'])
@triton.jit
def _layer_norm_fwd_1pass_kernel(X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean,
    Rstd, stride_x_row, stride_y_row, stride_res_row, stride_res_out_row, N,
    G, eps, IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.constexpr, HAS_RESIDUAL:
    tl.constexpr, STORE_RESIDUAL_OUT: tl.constexpr, HAS_WEIGHT: tl.
    constexpr, HAS_BIAS: tl.constexpr):
    row = tl.program_id(0)
    group = row % G
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl
            .float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + group * stride_x_row + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + group * stride_x_row + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(x, weight, bias, eps, residual=None, out_dtype=None,
    residual_dtype=None, is_rms_norm=False, num_groups=1):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N, G = *x.shape, num_groups
    if residual is not None:
        assert residual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (G * N,)
    if bias is not None:
        assert bias.shape == (G * N,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if (residual is not None or residual_dtype is not None and 
        residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device='cuda'
        ) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[M,](x, y, weight, bias, residual,
            residual_out, mean, rstd, x.stride(0), y.stride(0), residual.
            stride(0) if residual is not None else 0, residual_out.stride(0
            ) if residual_out is not None else 0, N, G, eps, is_rms_norm,
            BLOCK_N, residual is not None, residual_out is not None, weight
             is not None, bias is not None)
    return y, mean, rstd, residual_out if residual_out is not None else x


# Forward method (kernel launch code)
@contiguous
def _LayerNormFn_forward(ctx, x, weight, bias, residual=None, eps=1e-06,
    prenorm=False, residual_in_fp32=False, is_rms_norm=False, num_groups=1):
    x_shape_og = x.shape
    if x.shape[-1] % num_groups != 0:
        raise ValueError('num_channels must be divisible by num_groups')
    x = x.reshape(-1, x.shape[-1] // num_groups)
    if residual is not None:
        assert residual.shape == x_shape_og
        residual = residual.reshape_as(x)
    residual_dtype = (residual.dtype if residual is not None else torch.
        float32 if residual_in_fp32 else None)
    y, mean, rstd, residual_out = _layer_norm_fwd(x, weight, bias, eps,
        residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm,
        num_groups=num_groups)
    ctx.save_for_backward(residual_out, weight, bias, mean, rstd)
    ctx.x_shape_og = x_shape_og
    ctx.eps = eps
    ctx.is_rms_norm = is_rms_norm
    ctx.num_groups = num_groups
    ctx.has_residual = residual is not None
    ctx.prenorm = prenorm
    ctx.x_dtype = x.dtype
    y = y.reshape(x_shape_og)
    return y if not prenorm else (y, residual_out.reshape(x_shape_og))


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8), triton.Config({}, num_warps=16), triton.Config({},
    num_warps=32)], key=['N', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL',
    'IS_RMS_NORM', 'HAS_BIAS'])
@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['Y'] is not None})
@triton.jit
def _layer_norm_bwd_kernel(X, W, B, Y, DY, DX, DW, DB, DRESIDUAL,
    DRESIDUAL_IN, Mean, Rstd, stride_x_row, stride_y_row, stride_dy_row,
    stride_dx_row, stride_dres_row, stride_dres_in_row, M, N, G,
    rows_per_program, programs_per_group, IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr, HAS_DRESIDUAL: tl.constexpr, STORE_DRESIDUAL: tl
    .constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr):
    row_block_id = tl.program_id(0)
    group_id, program_id_in_group = (row_block_id // programs_per_group, 
        row_block_id % programs_per_group)
    row_start = group_id + program_id_in_group * G * rows_per_program
    row_end = min(row_start + G * rows_per_program, M)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + group_id * stride_x_row + cols, mask=mask).to(tl.
            float32)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + group_id * stride_x_row + cols, mask=mask, other=0.0
            ).to(tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row in range(row_start, row_end, G):
        x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0).to(tl
            .float32)
        dy = tl.load(DY + row * stride_dy_row + cols, mask=mask, other=0).to(tl
            .float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w if HAS_WEIGHT else xhat
            if HAS_BIAS:
                y = y + b
            tl.store(Y + row * stride_y_row + cols, y, mask=mask)
        wdy = dy
        if HAS_WEIGHT:
            wdy = dy * w
            dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + row * stride_dres_row + cols, mask=
                mask, other=0).to(tl.float32)
            dx += dres
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + row * stride_dres_in_row + cols, dx,
                mask=mask)
        tl.store(DX + row * stride_dx_row + cols, dx, mask=mask)
    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, dresidual=None,
    has_residual=False, is_rms_norm=False, x_dtype=None, recompute_output=
    False, num_groups=1):
    M, N, G = *x.shape, num_groups
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (G * N,)
    if bias is not None:
        assert bias.shape == (G * N,)
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(M, N,
        dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x
        ) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device
        ) if recompute_output else None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    S = triton.cdiv(torch.cuda.get_device_properties(x.device).
        multi_processor_count, G) * G
    dw = torch.empty((S, N), dtype=torch.float32, device=weight.device
        ) if weight is not None else None
    db = torch.empty((S, N), dtype=torch.float32, device=bias.device
        ) if bias is not None else None
    rows_per_program = triton.cdiv(M, S)
    programs_per_group = S // G
    grid = S,
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](x, weight, bias, y, dy, dx, dw, db,
            dresidual, dresidual_in, mean, rstd, x.stride(0), 0 if not
            recompute_output else y.stride(0), dy.stride(0), dx.stride(0), 
            dresidual.stride(0) if dresidual is not None else 0, 
            dresidual_in.stride(0) if dresidual_in is not None else 0, M, N,
            G, rows_per_program, programs_per_group, is_rms_norm, BLOCK_N, 
            dresidual is not None, dresidual_in is not None, weight is not
            None, bias is not None)
    dw = dw.view(G, -1, N).sum(1).to(weight).view_as(weight
        ) if weight is not None else None
    db = db.view(G, -1, N).sum(1).to(bias).view_as(bias
        ) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw,
        db, dresidual_in, y)


# Backward method (kernel launch code)
@contiguous
def _LayerNormFn_backward(ctx, dy, *args):
    x, weight, bias, mean, rstd = ctx.saved_tensors
    dy = dy.reshape(-1, dy.shape[-1] // ctx.num_groups)
    assert dy.shape == x.shape
    if ctx.prenorm:
        dresidual = args[0]
        dresidual = dresidual.reshape(-1, x.shape[-1])
        assert dresidual.shape == x.shape
    else:
        dresidual = None
    dx, dw, db, dresidual_in = _layer_norm_bwd(dy, x, weight, bias, ctx.eps,
        mean, rstd, dresidual, ctx.has_residual, ctx.is_rms_norm, x_dtype=
        ctx.x_dtype, num_groups=ctx.num_groups)
    return dx.reshape(ctx.x_shape_og), dw, db, dresidual_in.reshape(ctx.
        x_shape_og) if ctx.has_residual else None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNormFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, weight, bias, residual=None, eps=1e-06, prenorm=
        False, residual_in_fp32=False, is_rms_norm=False, num_groups=1):
        x_shape_og = x.shape
        if x.shape[-1] % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        x = x.reshape(-1, x.shape[-1] // num_groups)
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape_as(x)
        residual_dtype = (residual.dtype if residual is not None else torch
            .float32 if residual_in_fp32 else None)
        y, mean, rstd, residual_out = _layer_norm_fwd(x, weight, bias, eps,
            residual, residual_dtype=residual_dtype, is_rms_norm=
            is_rms_norm, num_groups=num_groups)
        ctx.save_for_backward(residual_out, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.num_groups = num_groups
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dy, *args):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1] // ctx.num_groups)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, x.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in = _layer_norm_bwd(dy, x, weight, bias, ctx
            .eps, mean, rstd, dresidual, ctx.has_residual, ctx.is_rms_norm,
            x_dtype=ctx.x_dtype, num_groups=ctx.num_groups)
        return dx.reshape(ctx.x_shape_og), dw, db, dresidual_in.reshape(ctx
            .x_shape_og
            ) if ctx.has_residual else None, None, None, None, None, None
