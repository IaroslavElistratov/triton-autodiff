# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/zhixuan-lin/forgetting-transformer
# Source-Files: src/forgetting_transformer/model/forgetting_transformer/fuse_norm_gate.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_o52u21n_/forgetting-transformer-main/src/forgetting_transformer/model/forgetting_transformer/fuse_norm_gate.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'STORE_RESIDUAL_OUT': lambda args: args['residual_out']
     is not None, 'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_WEIGHT': lambda args: args['w'] is not None, 'HAS_BIAS': lambda
    args: args['b'] is not None})
@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps) for
    BT in [8, 16, 32, 64] for num_warps in [2, 4, 8]], key=['D', 'NB',
    'IS_RMS_NORM', 'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'])
@triton.jit
def layer_norm_gated_fwd_kernel(x, g, y, w, b, residual, residual_out, mean,
    rstd, eps, T, G: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD:
    tl.constexpr, NB: tl.constexpr, ACTIVATION: tl.constexpr, IS_RMS_NORM:
    tl.constexpr, STORE_RESIDUAL_OUT: tl.constexpr, HAS_RESIDUAL: tl.
    constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr):
    i_t = tl.program_id(0)
    o_t = i_t * BT + tl.arange(0, BT)
    o_g = o_t % G
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(residual, (T, D), (D, 1), (i_t * BT, 0),
            (BT, BD), (1, 0))
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(residual_out, (T, D), (D, 1), (i_t *
            BT, 0), (BT, BD), (1, 0))
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty),
            boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check
            =(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))
    if HAS_WEIGHT:
        b_w = tl.load(w + o_g[:, None] * D + o_d[None, :], mask=m_d[None, :]
            ).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_g[:, None] * D + o_d[None, :], mask=m_d[None, :]
            ).to(tl.float32)
    b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None
        ] if not IS_RMS_NORM else b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b
    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == 'swish':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'STORE_RESIDUAL_OUT': lambda args: args['residual_out']
     is not None, 'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_WEIGHT': lambda args: args['w'] is not None, 'HAS_BIAS': lambda
    args: args['b'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [2, 4, 8, 16]], key=['D', 'IS_RMS_NORM',
    'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'])
@triton.jit
def layer_norm_gated_fwd_kernel1(x, g, y, w, b, residual, residual_out,
    mean, rstd, eps, G: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
    ACTIVATION: tl.constexpr, IS_RMS_NORM: tl.constexpr, STORE_RESIDUAL_OUT:
    tl.constexpr, HAS_RESIDUAL: tl.constexpr, HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr):
    i_t = tl.program_id(0)
    i_g = i_t % G
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)
    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == 'swish':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)
    tl.store(y + o_d, b_y, mask=m_d)


def layer_norm_gated_fwd(x: torch.Tensor, g: torch.Tensor, weight: torch.
    Tensor, bias: torch.Tensor, activation: str='swish', eps: float=1e-05,
    residual: torch.Tensor=None, out_dtype: torch.dtype=None,
    residual_dtype: torch.dtype=None, is_rms_norm: bool=False, num_groups:
    int=1):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D, G = *x.shape, num_groups
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (G * D,)
    if bias is not None:
        assert bias.shape == (G * D,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if (residual is not None or residual_dtype is not None and 
        residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device
        ) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return triton.cdiv(T, meta['BT']),
        layer_norm_gated_fwd_kernel[grid](x=x, g=g, y=y, w=weight, b=bias,
            residual=residual, residual_out=residual_out, mean=mean, rstd=
            rstd, eps=eps, T=T, G=G, D=D, BD=BD, NB=NB, ACTIVATION=
            activation, IS_RMS_NORM=is_rms_norm)
    else:
        layer_norm_gated_fwd_kernel1[T,](x=x, g=g, y=y, w=weight, b=bias,
            residual=residual, residual_out=residual_out, mean=mean, rstd=
            rstd, eps=eps, G=G, D=D, BD=BD, ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm)
    return y, mean, rstd, residual_out if residual_out is not None else x


# Forward method (kernel launch code)
@input_guard
def _LayerNormGatedLinearFunction_forward(ctx, x: torch.Tensor, g: torch.
    Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor,
    linear_weight: torch.Tensor, linear_bias: torch.Tensor, residual:
    Optional[torch.Tensor]=None, eps: float=1e-06, prenorm: bool=False,
    residual_in_fp32: bool=False, is_rms_norm: bool=False):
    x_shape_og = x.shape
    g_shape_og = g.shape
    x = x.reshape(-1, x.shape[-1])
    g = g.reshape(-1, g.shape[-1])
    if residual is not None:
        assert residual.shape == x_shape_og
        residual = residual.reshape(-1, residual.shape[-1])
    residual_dtype = (residual.dtype if residual is not None else torch.
        float if residual_in_fp32 else None)
    y, mean, rstd, residual_out = layer_norm_gated_fwd(x=x, g=g, weight=
        norm_weight, bias=norm_bias, eps=eps, residual=residual,
        residual_dtype=residual_dtype, is_rms_norm=is_rms_norm)
    y = y.reshape(x_shape_og)
    dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled(
        ) else y.dtype
    linear_weight = linear_weight.to(dtype)
    linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
    out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
    ctx.save_for_backward(residual_out, g, norm_weight, norm_bias,
        linear_weight, mean, rstd)
    ctx.x_shape_og = x_shape_og
    ctx.g_shape_og = g_shape_og
    ctx.eps = eps
    ctx.is_rms_norm = is_rms_norm
    ctx.has_residual = residual is not None
    ctx.prenorm = prenorm
    ctx.x_dtype = x.dtype
    ctx.linear_bias_is_none = linear_bias is None
    return out if not prenorm else (out, residual_out.reshape(x_shape_og))


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'HAS_DRESIDUAL': lambda args: args['dresidual'] is not
    None, 'HAS_WEIGHT': lambda args: args['w'] is not None, 'HAS_BIAS': lambda
    args: args['b'] is not None, 'RECOMPUTE_OUTPUT': lambda args: args['y']
     is not None})
@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps) for
    BT in [8, 16, 32, 64] for num_warps in [2, 4, 8]], key=['D', 'NB',
    'IS_RMS_NORM', 'HAS_DRESIDUAL', 'HAS_WEIGHT'])
@triton.jit
def layer_norm_gated_bwd_kernel(x, g, w, b, y, dy, dx, dg, dw, db,
    dresidual, dresidual_in, mean, rstd, T, G: tl.constexpr, D: tl.
    constexpr, BS: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr, NB: tl
    .constexpr, GS: tl.constexpr, ACTIVATION: tl.constexpr, IS_RMS_NORM: tl
    .constexpr, STORE_DRESIDUAL: tl.constexpr, HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, RECOMPUTE_OUTPUT: tl.
    constexpr):
    i_s = tl.program_id(0)
    i_g, i_sg = i_s // GS, i_s % GS
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=m_d).to(tl.float32)
        b_dw = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_db = tl.zeros((BT, BD), dtype=tl.float32)
    T = min(i_sg * BS + BS, T // G)
    for i_t in range(i_sg * BS, T, BT):
        p_x = tl.make_block_ptr(x + i_g * D, (T, D), (G * D, 1), (i_t, 0),
            (BT, BD), (1, 0))
        p_g = tl.make_block_ptr(g + i_g * D, (T, D), (G * D, 1), (i_t, 0),
            (BT, BD), (1, 0))
        p_dy = tl.make_block_ptr(dy + i_g * D, (T, D), (G * D, 1), (i_t, 0),
            (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx + i_g * D, (T, D), (G * D, 1), (i_t, 0),
            (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_g * D, (T, D), (G * D, 1), (i_t, 0),
            (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
        if not IS_RMS_NORM:
            p_mean = tl.make_block_ptr(mean + i_g, (T,), (G,), (i_t,), (BT,
                ), (0,))
            b_mean = tl.load(p_mean, boundary_check=(0,))
        p_rstd = tl.make_block_ptr(rstd + i_g, (T,), (G,), (i_t,), (BT,), (0,))
        b_rstd = tl.load(p_rstd, boundary_check=(0,))
        b_xhat = (b_x - b_mean[:, None]) * b_rstd[:, None
            ] if not IS_RMS_NORM else b_x * b_rstd[:, None]
        b_xhat = tl.where(m_d[None, :], b_xhat, 0.0)
        b_y = b_xhat * b_w[None, :] if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]
        if RECOMPUTE_OUTPUT:
            p_y = tl.make_block_ptr(y + i_g * D, (T, D), (G * D, 1), (i_t, 
                0), (BT, BD), (1, 0))
            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 'swish':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 -
                b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'silu':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 -
                b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'sigmoid':
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
        b_wdy = b_dy
        if HAS_WEIGHT or HAS_BIAS:
            m_t = i_t + tl.arange(0, BT) < T
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += tl.where(m_t[:, None], b_dy * b_xhat, 0.0)
        if HAS_BIAS:
            b_db += tl.where(m_t[:, None], b_dy, 0.0)
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_c2 = tl.sum(b_wdy, axis=1) / D
            b_dx = (b_wdy - (b_xhat * b_c1[:, None] + b_c2[:, None])) * b_rstd[
                :, None]
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_dx = (b_wdy - b_xhat * b_c1[:, None]) * b_rstd[:, None]
        if HAS_DRESIDUAL:
            p_dres = tl.make_block_ptr(dresidual + i_g * D, (T, D), (G * D,
                1), (i_t, 0), (BT, BD), (1, 0))
            b_dres = tl.load(p_dres, boundary_check=(0, 1)).to(tl.float32)
            b_dx += b_dres
        if STORE_DRESIDUAL:
            p_dres_in = tl.make_block_ptr(dresidual_in + i_g * D, (T, D), (
                G * D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_dres_in, b_dx.to(p_dres_in.dtype.element_ty),
                boundary_check=(0, 1))
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, tl.sum(b_dw, axis=0), mask=m_d)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, tl.sum(b_db, axis=0), mask=m_d)


@triton.heuristics({'HAS_DRESIDUAL': lambda args: args['dresidual'] is not
    None, 'HAS_WEIGHT': lambda args: args['w'] is not None, 'HAS_BIAS': lambda
    args: args['b'] is not None, 'RECOMPUTE_OUTPUT': lambda args: args['y']
     is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [2, 4, 8]], key=['D', 'IS_RMS_NORM', 'STORE_DRESIDUAL',
    'HAS_DRESIDUAL', 'HAS_WEIGHT'])
@triton.jit
def layer_norm_gated_bwd_kernel1(x, g, w, b, y, dy, dx, dg, dw, db,
    dresidual, dresidual_in, mean, rstd, T, G: tl.constexpr, D: tl.
    constexpr, BS: tl.constexpr, BD: tl.constexpr, GS: tl.constexpr,
    ACTIVATION: tl.constexpr, IS_RMS_NORM: tl.constexpr, STORE_DRESIDUAL:
    tl.constexpr, HAS_DRESIDUAL: tl.constexpr, HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr):
    i_s = tl.program_id(0)
    i_g, i_sg = i_s // GS, i_s % GS
    o_d = tl.arange(0, BD)
    mask = o_d < D
    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=mask, other=0.0).to(tl.float32)
        b_db = tl.zeros((BD,), dtype=tl.float32)
    for i_t in range(i_sg * BS * G + i_g, min((i_sg * BS + BS) * G + i_g, T), G
        ):
        b_x = tl.load(x + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(g + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
        b_dy = tl.load(dy + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        b_rstd = tl.load(rstd + i_t)
        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(mask, b_xhat, 0.0)
        b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b
        if RECOMPUTE_OUTPUT:
            tl.store(y + i_t * D + o_d, b_y, mask=mask)
        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 'swish':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 -
                b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'silu':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 -
                b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'sigmoid':
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
        b_wdy = b_dy
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += b_dy * b_xhat
        if HAS_BIAS:
            b_db += b_dy
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_c2 = tl.sum(b_wdy, axis=0) / D
            b_dx = (b_wdy - (b_xhat * b_c1 + b_c2)) * b_rstd
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_dx = (b_wdy - b_xhat * b_c1) * b_rstd
        if HAS_DRESIDUAL:
            b_dres = tl.load(dresidual + i_t * D + o_d, mask=mask, other=0).to(
                tl.float32)
            b_dx += b_dres
        b_dx = tl.cast(b_dx, dtype=dx.dtype.element_ty,
            fp_downcast_rounding='rtne')
        b_dg = tl.cast(b_dg, dtype=dg.dtype.element_ty,
            fp_downcast_rounding='rtne')
        if STORE_DRESIDUAL:
            tl.store(dresidual_in + i_t * D + o_d, b_dx, mask=mask)
        tl.store(dx + i_t * D + o_d, b_dx, mask=mask)
        tl.store(dg + i_t * D + o_d, b_dg, mask=mask)
    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=mask)


def _cpu_device_warning():
    import warnings
    warnings.warn(
        'Triton is not supported on current platform, roll back to CPU.',
        stacklevel=1)


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int=0) ->int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(
            tensor_idx)['multiprocessor_count']
    except BaseException:
        _cpu_device_warning()
        return -1


def layer_norm_gated_bwd(dy: torch.Tensor, x: torch.Tensor, g: torch.Tensor,
    weight: torch.Tensor, bias: torch.Tensor, activation: str='swish', eps:
    float=1e-05, mean: torch.Tensor=None, rstd: torch.Tensor=None,
    dresidual: torch.Tensor=None, has_residual: bool=False, is_rms_norm:
    bool=False, x_dtype: torch.dtype=None, recompute_output: bool=False,
    num_groups: int=1):
    T, D, G = *x.shape, num_groups
    assert dy.shape == (T, D)
    if dresidual is not None:
        assert dresidual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (G * D,)
    if bias is not None:
        assert bias.shape == (G * D,)
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D,
        dtype=x_dtype, device=x.device)
    dg = torch.empty_like(g) if x_dtype is None else torch.empty(T, D,
        dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x
        ) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device
        ) if recompute_output else None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    NS = triton.cdiv(get_multiprocessor_count(x.device.index), G) * G
    BS = triton.cdiv(T, NS)
    GS = NS // G
    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device
        ) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device
        ) if bias is not None else None
    grid = NS,
    if D <= 512:
        NB = triton.cdiv(T, 2048)
        layer_norm_gated_bwd_kernel[grid](x=x, g=g, w=weight, b=bias, y=y,
            dy=dy, dx=dx, dg=dg, dw=dw, db=db, dresidual=dresidual,
            dresidual_in=dresidual_in, mean=mean, rstd=rstd, T=T, G=G, D=D,
            BS=BS, BD=BD, NB=NB, GS=GS, ACTIVATION=activation, IS_RMS_NORM=
            is_rms_norm, STORE_DRESIDUAL=dresidual_in is not None)
    else:
        layer_norm_gated_bwd_kernel1[grid](x=x, g=g, w=weight, b=bias, y=y,
            dy=dy, dx=dx, dg=dg, dw=dw, db=db, dresidual=dresidual,
            dresidual_in=dresidual_in, mean=mean, rstd=rstd, T=T, G=G, D=D,
            BS=BS, BD=BD, GS=GS, ACTIVATION=activation, IS_RMS_NORM=
            is_rms_norm, STORE_DRESIDUAL=dresidual_in is not None)
    dw = dw.view(G, -1, D).sum(1).to(weight).view_as(weight
        ) if weight is not None else None
    db = db.view(G, -1, D).sum(1).to(bias).view_as(bias
        ) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dg, dw, db, dresidual_in) if not recompute_output else (dx,
        dg, dw, db, dresidual_in, y)


# Backward method (kernel launch code)
@input_guard
def _LayerNormGatedLinearFunction_backward(ctx, dout, *args):
    x, g, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
    dout = dout.reshape(-1, dout.shape[-1])
    dy = F.linear(dout, linear_weight.t())
    dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
    assert dy.shape == x.shape
    if ctx.prenorm:
        dresidual = args[0]
        dresidual = dresidual.reshape(-1, dresidual.shape[-1])
        assert dresidual.shape == x.shape
    else:
        dresidual = None
    dx, dg, dnorm_weight, dnorm_bias, dres_in, y = layer_norm_gated_bwd(dy=
        dy, x=x, g=g, weight=norm_weight, bias=norm_bias, eps=ctx.eps, mean
        =mean, rstd=rstd, dresidual=dresidual, has_residual=ctx.
        has_residual, is_rms_norm=ctx.is_rms_norm, x_dtype=ctx.x_dtype,
        recompute_output=True)
    dlinear_weight = torch.einsum('bo,bi->oi', dout, y)
    return dx.reshape(ctx.x_shape_og), dg.reshape(ctx.g_shape_og
        ), dnorm_weight, dnorm_bias, dlinear_weight, dlinear_bias, dres_in.reshape(
        ctx.x_shape_og) if ctx.has_residual else None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LayerNormGatedLinearFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, norm_weight: torch.
        Tensor, norm_bias: torch.Tensor, linear_weight: torch.Tensor,
        linear_bias: torch.Tensor, residual: Optional[torch.Tensor]=None,
        eps: float=1e-06, prenorm: bool=False, residual_in_fp32: bool=False,
        is_rms_norm: bool=False):
        x_shape_og = x.shape
        g_shape_og = g.shape
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (residual.dtype if residual is not None else torch
            .float if residual_in_fp32 else None)
        y, mean, rstd, residual_out = layer_norm_gated_fwd(x=x, g=g, weight
            =norm_weight, bias=norm_bias, eps=eps, residual=residual,
            residual_dtype=residual_dtype, is_rms_norm=is_rms_norm)
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled(
            ) else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype
            ) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        ctx.save_for_backward(residual_out, g, norm_weight, norm_bias,
            linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dout, *args):
        x, g, norm_weight, norm_bias, linear_weight, mean, rstd = (ctx.
            saved_tensors)
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dg, dnorm_weight, dnorm_bias, dres_in, y = layer_norm_gated_bwd(dy
            =dy, x=x, g=g, weight=norm_weight, bias=norm_bias, eps=ctx.eps,
            mean=mean, rstd=rstd, dresidual=dresidual, has_residual=ctx.
            has_residual, is_rms_norm=ctx.is_rms_norm, x_dtype=ctx.x_dtype,
            recompute_output=True)
        dlinear_weight = torch.einsum('bo,bi->oi', dout, y)
        return (dx.reshape(ctx.x_shape_og), dg.reshape(ctx.g_shape_og),
            dnorm_weight, dnorm_bias, dlinear_weight, dlinear_bias, dres_in
            .reshape(ctx.x_shape_og) if ctx.has_residual else None, None,
            None, None, None)
