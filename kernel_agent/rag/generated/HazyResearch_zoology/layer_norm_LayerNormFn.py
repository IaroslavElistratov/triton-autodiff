# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/HazyResearch/zoology
# Source-Files: zoology/mixers/mamba_ssm/ops/triton/layer_norm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_jtfasj09/zoology-main/zoology/mixers/mamba_ssm/ops/triton/layer_norm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=pruned_configs_autotune, key=['N', 'HAS_RESIDUAL',
    'STORE_RESIDUAL_OUT', 'IS_RMS_NORM', 'HAS_BIAS'])
@triton.heuristics({'HAS_X1': lambda args: args['X1'] is not None})
@triton.heuristics({'HAS_W1': lambda args: args['W1'] is not None})
@triton.heuristics({'HAS_B1': lambda args: args['B1'] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(X, Y, W, B, RESIDUAL, X1, W1, B1, Y1,
    RESIDUAL_OUT, ROWSCALE, SEEDS, DROPOUT_MASK, Mean, Rstd, stride_x_row,
    stride_y_row, stride_res_row, stride_res_out_row, stride_x1_row,
    stride_y1_row, M, N, eps, dropout_p, IS_RMS_NORM: tl.constexpr, BLOCK_N:
    tl.constexpr, HAS_RESIDUAL: tl.constexpr, STORE_RESIDUAL_OUT: tl.
    constexpr, HAS_BIAS: tl.constexpr, HAS_DROPOUT: tl.constexpr,
    STORE_DROPOUT_MASK: tl.constexpr, HAS_ROWSCALE: tl.constexpr, HAS_X1:
    tl.constexpr, HAS_W1: tl.constexpr, HAS_B1: tl.constexpr):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols,
            n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + M + row).to(tl.uint32),
                cols, n_rounds=7) > dropout_p
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask,
                    mask=cols < N)
        x += x1
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
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)


def _layer_norm_fwd(x, weight, bias, eps, residual=None, x1=None, weight1=
    None, bias1=None, dropout_p=0.0, rowscale=None, out_dtype=None,
    residual_dtype=None, is_rms_norm=False, return_dropout_mask=False):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if weight1 is not None:
        y1 = torch.empty_like(y)
        assert y1.stride(-1) == 1
    else:
        y1 = None
    if (residual is not None or residual_dtype is not None and 
        residual_dtype != x.dtype or dropout_p > 0.0 or rowscale is not
        None or x1 is not None):
        residual_out = torch.empty(M, N, device=x.device, dtype=
            residual_dtype if residual_dtype is not None else x.dtype)
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device
        ) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    if dropout_p > 0.0:
        seeds = torch.randint(2 ** 32, (M if x1 is None else 2 * M,),
            device=x.device, dtype=torch.int64)
    else:
        seeds = None
    if return_dropout_mask and dropout_p > 0.0:
        dropout_mask = torch.empty(M if x1 is None else 2 * M, N, device=x.
            device, dtype=torch.bool)
    else:
        dropout_mask = None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[M,](x, y, weight, bias, residual, x1,
            weight1, bias1, y1, residual_out, rowscale, seeds, dropout_mask,
            mean, rstd, x.stride(0), y.stride(0), residual.stride(0) if 
            residual is not None else 0, residual_out.stride(0) if 
            residual_out is not None else 0, x1.stride(0) if x1 is not None
             else 0, y1.stride(0) if y1 is not None else 0, M, N, eps,
            dropout_p, is_rms_norm, BLOCK_N, residual is not None, 
            residual_out is not None, bias is not None, dropout_p > 0.0, 
            dropout_mask is not None, rowscale is not None)
    if dropout_mask is not None and x1 is not None:
        dropout_mask, dropout_mask1 = dropout_mask.tensor_split(2, dim=0)
    else:
        dropout_mask1 = None
    return (y, y1, mean, rstd, residual_out if residual_out is not None else
        x, seeds, dropout_mask, dropout_mask1)


# Forward method (kernel launch code)
def _LayerNormFn_forward(ctx, x, weight, bias, residual=None, x1=None,
    weight1=None, bias1=None, eps=1e-06, dropout_p=0.0, rowscale=None,
    prenorm=False, residual_in_fp32=False, is_rms_norm=False,
    return_dropout_mask=False):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if residual is not None:
        assert residual.shape == x_shape_og
        residual = residual.reshape(-1, residual.shape[-1])
        if residual.stride(-1) != 1:
            residual = residual.contiguous()
    if x1 is not None:
        assert x1.shape == x_shape_og
        assert rowscale is None, 'rowscale is not supported with parallel LayerNorm'
        x1 = x1.reshape(-1, x1.shape[-1])
        if x1.stride(-1) != 1:
            x1 = x1.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if weight1 is not None:
        weight1 = weight1.contiguous()
    if bias1 is not None:
        bias1 = bias1.contiguous()
    if rowscale is not None:
        rowscale = rowscale.reshape(-1).contiguous()
    residual_dtype = (residual.dtype if residual is not None else torch.
        float32 if residual_in_fp32 else None)
    y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1 = (
        _layer_norm_fwd(x, weight, bias, eps, residual, x1, weight1, bias1,
        dropout_p=dropout_p, rowscale=rowscale, residual_dtype=
        residual_dtype, is_rms_norm=is_rms_norm, return_dropout_mask=
        return_dropout_mask))
    ctx.save_for_backward(residual_out, weight, bias, weight1, bias1,
        rowscale, seeds, mean, rstd)
    ctx.x_shape_og = x_shape_og
    ctx.eps = eps
    ctx.dropout_p = dropout_p
    ctx.is_rms_norm = is_rms_norm
    ctx.has_residual = residual is not None
    ctx.has_x1 = x1 is not None
    ctx.prenorm = prenorm
    ctx.x_dtype = x.dtype
    y = y.reshape(x_shape_og)
    y1 = y1.reshape(x_shape_og) if y1 is not None else None
    residual_out = residual_out.reshape(x_shape_og
        ) if residual_out is not None else None
    dropout_mask = dropout_mask.reshape(x_shape_og
        ) if dropout_mask is not None else None
    dropout_mask1 = dropout_mask1.reshape(x_shape_og
        ) if dropout_mask1 is not None else None
    if not return_dropout_mask:
        if weight1 is None:
            return y if not prenorm else (y, residual_out)
        else:
            return (y, y1) if not prenorm else (y, y1, residual_out)
    elif weight1 is None:
        return (y, dropout_mask, dropout_mask1) if not prenorm else (y,
            residual_out, dropout_mask, dropout_mask1)
    else:
        return (y, y1, dropout_mask, dropout_mask1) if not prenorm else (y,
            y1, residual_out, dropout_mask, dropout_mask1)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=pruned_configs_autotune, key=['N', 'HAS_DRESIDUAL',
    'STORE_DRESIDUAL', 'IS_RMS_NORM', 'HAS_BIAS', 'HAS_DROPOUT'])
@triton.heuristics({'HAS_ROWSCALE': lambda args: args['ROWSCALE'] is not None})
@triton.heuristics({'HAS_DY1': lambda args: args['DY1'] is not None})
@triton.heuristics({'HAS_DX1': lambda args: args['DX1'] is not None})
@triton.heuristics({'HAS_B1': lambda args: args['DB1'] is not None})
@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['Y'] is not None})
@triton.jit
def _layer_norm_bwd_kernel(X, W, B, Y, DY, DX, DW, DB, DRESIDUAL, W1, DY1,
    DX1, DW1, DB1, DRESIDUAL_IN, ROWSCALE, SEEDS, Mean, Rstd, stride_x_row,
    stride_y_row, stride_dy_row, stride_dx_row, stride_dres_row,
    stride_dy1_row, stride_dx1_row, stride_dres_in_row, M, N, eps,
    dropout_p, rows_per_program, IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.
    constexpr, HAS_DRESIDUAL: tl.constexpr, STORE_DRESIDUAL: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_DROPOUT: tl.constexpr, HAS_ROWSCALE: tl.
    constexpr, HAS_DY1: tl.constexpr, HAS_DX1: tl.constexpr, HAS_B1: tl.
    constexpr, RECOMPUTE_OUTPUT: tl.constexpr):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_DY1:
        DY1 += row_start * stride_dy1_row
    if HAS_DX1:
        DX1 += row_start * stride_dx1_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DY1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_DY1:
        dw1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if HAS_B1:
            db1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if HAS_DY1:
            dy1 = tl.load(DY1 + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_DY1:
            wdy += w1 * dy1
            dw1 += dy1 * xhat
            if HAS_B1:
                db1 += dy1
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        if HAS_DX1:
            if HAS_DROPOUT:
                keep_mask = tl.rand(tl.load(SEEDS + M + row).to(tl.uint32),
                    cols, n_rounds=7) > dropout_p
                dx1 = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
            else:
                dx1 = dx
            tl.store(DX1 + cols, dx1, mask=mask)
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols,
                n_rounds=7) > dropout_p
            dx = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + row).to(tl.float32)
            dx *= rowscale
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
        if HAS_DY1:
            DY1 += stride_dy1_row
        if HAS_DX1:
            DX1 += stride_dx1_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)
    if HAS_DY1:
        tl.store(DW1 + row_block_id * N + cols, dw1, mask=mask)
        if HAS_B1:
            tl.store(DB1 + row_block_id * N + cols, db1, mask=mask)


def _layer_norm_bwd(dy, x, weight, bias, eps, mean, rstd, dresidual=None,
    dy1=None, weight1=None, bias1=None, seeds=None, dropout_p=0.0, rowscale
    =None, has_residual=False, has_x1=False, is_rms_norm=False, x_dtype=
    None, recompute_output=False):
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if dy1 is not None:
        assert weight1 is not None
        assert dy1.shape == dy.shape
        assert dy1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if seeds is not None:
        assert seeds.is_contiguous()
        assert seeds.shape == (M if not has_x1 else M * 2,)
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(M, N,
        dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and (dx.dtype != x.
        dtype or dropout_p > 0.0 or rowscale is not None or has_x1) else None
    dx1 = torch.empty_like(dx) if has_x1 and dropout_p > 0.0 else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device
        ) if recompute_output else None
    if recompute_output:
        assert weight1 is None, 'recompute_output is not supported with parallel LayerNorm'
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    _db = torch.empty((sm_count, N), dtype=torch.float32, device=bias.device
        ) if bias is not None else None
    _dw1 = torch.empty_like(_dw) if weight1 is not None else None
    _db1 = torch.empty_like(_db) if bias1 is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = sm_count,
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](x, weight, bias, y, dy, dx, _dw, _db,
            dresidual, weight1, dy1, dx1, _dw1, _db1, dresidual_in,
            rowscale, seeds, mean, rstd, x.stride(0), 0 if not
            recompute_output else y.stride(0), dy.stride(0), dx.stride(0), 
            dresidual.stride(0) if dresidual is not None else 0, dy1.stride
            (0) if dy1 is not None else 0, dx1.stride(0) if dx1 is not None
             else 0, dresidual_in.stride(0) if dresidual_in is not None else
            0, M, N, eps, dropout_p, rows_per_program, is_rms_norm, BLOCK_N,
            dresidual is not None, dresidual_in is not None, bias is not
            None, dropout_p > 0.0)
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    dw1 = _dw1.sum(0).to(weight1.dtype) if weight1 is not None else None
    db1 = _db1.sum(0).to(bias1.dtype) if bias1 is not None else None
    if (has_residual and dx.dtype == x.dtype and dropout_p == 0.0 and 
        rowscale is None):
        dresidual_in = dx
    if has_x1 and dropout_p == 0.0:
        dx1 = dx
    return (dx, dw, db, dresidual_in, dx1, dw1, db1
        ) if not recompute_output else (dx, dw, db, dresidual_in, dx1, dw1,
        db1, y)


# Backward method (kernel launch code)
def _LayerNormFn_backward(ctx, dy, *args):
    x, weight, bias, weight1, bias1, rowscale, seeds, mean, rstd = (ctx.
        saved_tensors)
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    assert dy.shape == x.shape
    if weight1 is not None:
        dy1, args = args[0], args[1:]
        dy1 = dy1.reshape(-1, dy1.shape[-1])
        if dy1.stride(-1) != 1:
            dy1 = dy1.contiguous()
        assert dy1.shape == x.shape
    else:
        dy1 = None
    if ctx.prenorm:
        dresidual = args[0]
        dresidual = dresidual.reshape(-1, dresidual.shape[-1])
        if dresidual.stride(-1) != 1:
            dresidual = dresidual.contiguous()
        assert dresidual.shape == x.shape
    else:
        dresidual = None
    dx, dw, db, dresidual_in, dx1, dw1, db1 = _layer_norm_bwd(dy, x, weight,
        bias, ctx.eps, mean, rstd, dresidual, dy1, weight1, bias1, seeds,
        ctx.dropout_p, rowscale, ctx.has_residual, ctx.has_x1, ctx.
        is_rms_norm, x_dtype=ctx.x_dtype)
    return (dx.reshape(ctx.x_shape_og), dw, db, dresidual_in.reshape(ctx.
        x_shape_og) if ctx.has_residual else None, dx1.reshape(ctx.
        x_shape_og) if dx1 is not None else None, dw1, db1, None, None,
        None, None, None, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNormFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, residual=None, x1=None, weight1=None,
        bias1=None, eps=1e-06, dropout_p=0.0, rowscale=None, prenorm=False,
        residual_in_fp32=False, is_rms_norm=False, return_dropout_mask=False):
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
            if residual.stride(-1) != 1:
                residual = residual.contiguous()
        if x1 is not None:
            assert x1.shape == x_shape_og
            assert rowscale is None, 'rowscale is not supported with parallel LayerNorm'
            x1 = x1.reshape(-1, x1.shape[-1])
            if x1.stride(-1) != 1:
                x1 = x1.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        if weight1 is not None:
            weight1 = weight1.contiguous()
        if bias1 is not None:
            bias1 = bias1.contiguous()
        if rowscale is not None:
            rowscale = rowscale.reshape(-1).contiguous()
        residual_dtype = (residual.dtype if residual is not None else torch
            .float32 if residual_in_fp32 else None)
        (y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1
            ) = (_layer_norm_fwd(x, weight, bias, eps, residual, x1,
            weight1, bias1, dropout_p=dropout_p, rowscale=rowscale,
            residual_dtype=residual_dtype, is_rms_norm=is_rms_norm,
            return_dropout_mask=return_dropout_mask))
        ctx.save_for_backward(residual_out, weight, bias, weight1, bias1,
            rowscale, seeds, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.dropout_p = dropout_p
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.has_x1 = x1 is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        y1 = y1.reshape(x_shape_og) if y1 is not None else None
        residual_out = residual_out.reshape(x_shape_og
            ) if residual_out is not None else None
        dropout_mask = dropout_mask.reshape(x_shape_og
            ) if dropout_mask is not None else None
        dropout_mask1 = dropout_mask1.reshape(x_shape_og
            ) if dropout_mask1 is not None else None
        if not return_dropout_mask:
            if weight1 is None:
                return y if not prenorm else (y, residual_out)
            else:
                return (y, y1) if not prenorm else (y, y1, residual_out)
        elif weight1 is None:
            return (y, dropout_mask, dropout_mask1) if not prenorm else (y,
                residual_out, dropout_mask, dropout_mask1)
        else:
            return (y, y1, dropout_mask, dropout_mask1) if not prenorm else (y,
                y1, residual_out, dropout_mask, dropout_mask1)

    @staticmethod
    def backward(ctx, dy, *args):
        x, weight, bias, weight1, bias1, rowscale, seeds, mean, rstd = (ctx
            .saved_tensors)
        dy = dy.reshape(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()
        assert dy.shape == x.shape
        if weight1 is not None:
            dy1, args = args[0], args[1:]
            dy1 = dy1.reshape(-1, dy1.shape[-1])
            if dy1.stride(-1) != 1:
                dy1 = dy1.contiguous()
            assert dy1.shape == x.shape
        else:
            dy1 = None
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            if dresidual.stride(-1) != 1:
                dresidual = dresidual.contiguous()
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in, dx1, dw1, db1 = _layer_norm_bwd(dy, x,
            weight, bias, ctx.eps, mean, rstd, dresidual, dy1, weight1,
            bias1, seeds, ctx.dropout_p, rowscale, ctx.has_residual, ctx.
            has_x1, ctx.is_rms_norm, x_dtype=ctx.x_dtype)
        return (dx.reshape(ctx.x_shape_og), dw, db, dresidual_in.reshape(
            ctx.x_shape_og) if ctx.has_residual else None, dx1.reshape(ctx.
            x_shape_og) if dx1 is not None else None, dw1, db1, None, None,
            None, None, None, None, None)
