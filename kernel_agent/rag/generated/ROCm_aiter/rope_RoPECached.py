# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/ROCm/aiter
# Source-Files: aiter/ops/triton/rope.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ciw0cknm/aiter-main/aiter/ops/triton/rope.py
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
def _get_gptj_rotated_x(x, x_rotated_mask, BLOCK_T: tl.constexpr, BLOCK_D:
    tl.constexpr, BLOCK_D_HALF: tl.constexpr, IS_BWD: tl.constexpr=False):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)
    x_rotated = tl.reshape(x_rotated, (BLOCK_T, BLOCK_D_HALF, 2))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(x_rotated, (BLOCK_T, BLOCK_D))
    return x_rotated


@triton.jit
def _get_neox_rotated_x(x, x_rotated_mask, BLOCK_T: tl.constexpr, BLOCK_D:
    tl.constexpr, BLOCK_D_HALF: tl.constexpr, IS_BWD: tl.constexpr=False):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)
    x_rotated = tl.reshape(x_rotated, (BLOCK_T, 2, BLOCK_D_HALF))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(x_rotated, (BLOCK_T, BLOCK_D))
    x_rotated = tl.flip(x_rotated, 1)
    return x_rotated


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _rope_kernel_sbhd_cached_fwd(x_ptr, cos_ptr, sin_ptr, pos_ptr, off_ptr,
    out_ptr, stride_x_s, stride_x_b, stride_x_h, stride_x_d, stride_cos_s,
    stride_cos_b, stride_cos_h, stride_cos_d, stride_pos_s, stride_pos_b,
    stride_out_s, stride_out_b, stride_out_h, stride_out_d, S, HAVE_NOPE:
    tl.constexpr, NOPE_FIRST: tl.constexpr, INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr, IS_NEOX: tl.constexpr, HAVE_POS:
    tl.constexpr, HAVE_OFFS: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D:
    tl.constexpr, BLOCK_D_HALF: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_s = tl.program_id(2)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S
    if HAVE_POS:
        pos_offs = s_offs * stride_pos_s + b * stride_pos_b
        pos = tl.load(pos_ptr + pos_offs, mask=s_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=s_mask)
            s_cos_offs = pos + offset
        else:
            s_cos_offs = pos
    else:
        s_cos_offs = s_offs
    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where((d_cos_offs >= BLOCK_D_HALF) & (
                d_cos_offs < BLOCK_D), d_cos_offs - BLOCK_D_HALF, d_cos_offs
                ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D
        else:
            d_cos_offs = d_offs // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D
    cos_mask = s_mask[:, None] & d_cos_mask[None, :]
    cos_offs = s_cos_offs[:, None] * stride_cos_s + d_cos_offs[None, :
        ] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)
    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D
    x_mask = s_mask[:, None] & (d_offs < BLOCK_D)[None, :]
    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]
    d_offs += nope_offs
    x_offs = b * stride_x_b + s_offs[:, None
        ] * stride_x_s + h * stride_x_h + d_offs[None, :] * stride_x_d
    x = tl.load(x_ptr + x_offs, mask=x_mask)
    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(x, x_rotated_mask, BLOCK_S, BLOCK_D,
            BLOCK_D_HALF)
    else:
        x_rotated = _get_gptj_rotated_x(x, x_rotated_mask, BLOCK_S, BLOCK_D,
            BLOCK_D_HALF)
    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = b * stride_out_b + s_offs[:, None
        ] * stride_out_s + h * stride_out_h + d_offs[None, :] * stride_out_d
    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)
    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask
                =x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask
                =x_mask)


def _rope_cached_fwd(x: torch.Tensor, out: torch.Tensor, cos: torch.Tensor,
    sin: torch.Tensor, positions: torch.Tensor, offsets: torch.Tensor,
    rotate_style: int, reuse_freqs_front_part: bool, nope_first: bool,
    inplace: bool, transpose_output: bool=False) ->torch.Tensor:
    s, b, h, d = x.shape
    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False
    if have_nope:
        BLOCK_D = d // 2
        BLOCK_D_HALF = d // 4
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = b, h, triton.cdiv(s, BLOCK_S)
    pos_stride = positions.stride() if positions is not None else (1, 1)
    _rope_kernel_sbhd_cached_fwd[grid](x, cos, sin, positions, offsets, out,
        *x.stride(), *cos.stride(), *pos_stride, *out.stride(), s,
        HAVE_NOPE=have_nope, NOPE_FIRST=nope_first, INPLACE=inplace,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part, IS_NEOX=rotate_style ==
        RotateStyle.NEOX, HAVE_POS=positions is not None, HAVE_OFFS=offsets
         is not None, BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, BLOCK_D_HALF=
        BLOCK_D_HALF, num_warps=num_warps, waves_per_eu=waves_per_eu)
    return out


def rope_cached_fwd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    rotate_style: int, reuse_freqs_front_part: bool, nope_first: bool,
    transpose_output: bool=False):
    s, b, h, d = x.shape
    out = torch.empty((s, b, h, d), dtype=x.dtype, device=x.device,
        requires_grad=False)
    _rope_cached_fwd(x, out, cos, sin, None, None, rotate_style,
        reuse_freqs_front_part, nope_first, False, transpose_output)
    return out


# Forward method (kernel launch code)
def _RoPECached_forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch
    .Tensor, rotate_style: int, reuse_freqs_front_part: bool, nope_first:
    bool, transpose_output: bool=False) ->torch.Tensor:
    ctx.rotate_style = rotate_style
    ctx.reuse_freqs_front_part = reuse_freqs_front_part
    ctx.nope_first = nope_first
    ctx.transpose_output = transpose_output
    ctx.save_for_backward(cos, sin)
    return rope_cached_fwd(x, cos, sin, rotate_style,
        reuse_freqs_front_part, nope_first, transpose_output)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _rope_kernel_sbhd_cached_bwd(x_ptr, cos_ptr, sin_ptr, pos_ptr, off_ptr,
    out_ptr, stride_x_s, stride_x_b, stride_x_h, stride_x_d, stride_cos_s,
    stride_cos_b, stride_cos_h, stride_cos_d, stride_pos_s, stride_pos_b,
    stride_out_s, stride_out_b, stride_out_h, stride_out_d, S, HAVE_NOPE:
    tl.constexpr, NOPE_FIRST: tl.constexpr, INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr, IS_NEOX: tl.constexpr, HAVE_POS:
    tl.constexpr, HAVE_OFFS: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D:
    tl.constexpr, BLOCK_D_HALF: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_s = tl.program_id(2)
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S
    if HAVE_POS:
        pos_offs = s_offs * stride_pos_s + b * stride_pos_b
        pos = tl.load(pos_ptr + pos_offs, mask=s_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=s_mask)
            s_cos_offs = pos + offset
        else:
            s_cos_offs = pos
    else:
        s_cos_offs = s_offs
    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where((d_cos_offs >= BLOCK_D_HALF) & (
                d_cos_offs < BLOCK_D), d_cos_offs - BLOCK_D_HALF, d_cos_offs
                ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D
        else:
            d_cos_offs = d_offs // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D
    cos_mask = s_mask[:, None] & d_cos_mask[None, :]
    cos_offs = s_cos_offs[:, None] * stride_cos_s + d_cos_offs[None, :
        ] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)
    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D
    x_mask = s_mask[:, None] & (d_offs < BLOCK_D)[None, :]
    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]
    d_offs += nope_offs
    x_offs = b * stride_x_b + s_offs[:, None
        ] * stride_x_s + h * stride_x_h + d_offs[None, :] * stride_x_d
    x = tl.load(x_ptr + x_offs, mask=x_mask)
    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(x * sin, x_rotated_mask, BLOCK_S,
            BLOCK_D, BLOCK_D_HALF, True)
    else:
        x_rotated = _get_gptj_rotated_x(x * sin, x_rotated_mask, BLOCK_S,
            BLOCK_D, BLOCK_D_HALF, True)
    out_x = x * cos + x_rotated
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = b * stride_out_b + s_offs[:, None
        ] * stride_out_s + h * stride_out_h + d_offs[None, :] * stride_out_d
    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)
    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask
                =x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask
                =x_mask)


def _rope_cached_bwd(x: torch.Tensor, out: torch.Tensor, cos: torch.Tensor,
    sin: torch.Tensor, positions: torch.Tensor, offsets: torch.Tensor,
    rotate_style: int, reuse_freqs_front_part: bool, nope_first: bool,
    inplace: bool, transpose_output: bool=False) ->torch.Tensor:
    s, b, h, d = x.shape
    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False
    if have_nope:
        BLOCK_D = d // 2
        BLOCK_D_HALF = d // 4
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = b, h, triton.cdiv(s, BLOCK_S)
    pos_stride = positions.stride() if positions is not None else (1, 1)
    _rope_kernel_sbhd_cached_bwd[grid](x, cos, sin, positions, offsets, out,
        *x.stride(), *cos.stride(), *pos_stride, *out.stride(), s,
        HAVE_NOPE=have_nope, NOPE_FIRST=nope_first, INPLACE=inplace,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part, IS_NEOX=rotate_style ==
        RotateStyle.NEOX, HAVE_POS=positions is not None, HAVE_OFFS=offsets
         is not None, BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, BLOCK_D_HALF=
        BLOCK_D_HALF, num_warps=num_warps, waves_per_eu=waves_per_eu)
    return out


def rope_cached_bwd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    rotate_style: int, reuse_freqs_front_part: bool, nope_first: bool,
    transpose_output: bool=False):
    s, b, h, d = x.shape
    out = torch.empty((s, b, h, d), dtype=x.dtype, device=x.device,
        requires_grad=False)
    _rope_cached_bwd(x, out, cos, sin, None, None, rotate_style,
        reuse_freqs_front_part, nope_first, False, transpose_output)
    return out


# Backward method (kernel launch code)
def _RoPECached_backward(ctx, output_grads) ->Tuple[Union[torch.Tensor,
    None], ...]:
    cos, sin = ctx.saved_tensors
    return rope_cached_bwd(output_grads, cos, sin, ctx.rotate_style, ctx.
        reuse_freqs_front_part, ctx.nope_first, ctx.transpose_output
        ), None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RoPECached(autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        rotate_style: int, reuse_freqs_front_part: bool, nope_first: bool,
        transpose_output: bool=False) ->torch.Tensor:
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(cos, sin)
        return rope_cached_fwd(x, cos, sin, rotate_style,
            reuse_freqs_front_part, nope_first, transpose_output)

    @staticmethod
    def backward(ctx, output_grads) ->Tuple[Union[torch.Tensor, None], ...]:
        cos, sin = ctx.saved_tensors
        return rope_cached_bwd(output_grads, cos, sin, ctx.rotate_style,
            ctx.reuse_freqs_front_part, ctx.nope_first, ctx.transpose_output
            ), None, None
