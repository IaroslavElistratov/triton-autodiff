# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/hunto/LocalMamba
# Source-Files: classification/lib/models/mamba/local_scan.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_wrq7wx5y/LocalMamba-main/classification/lib/models/mamba/local_scan.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

# Common helper imports
from triton import cdiv
import time

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def triton_local_scan(x, y, K: tl.constexpr, flip: tl.constexpr, BC: tl.
    constexpr, BH: tl.constexpr, BW: tl.constexpr, DC: tl.constexpr, DH: tl
    .constexpr, DW: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = i_hw // NW, i_hw % NW
    _mask_h = i_h * BH + tl.arange(0, BH) < DH
    _mask_w = i_w * BW + tl.arange(0, BW) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)
    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = _tmp0 + i_h * BH * DW + tl.arange(0, BH)[:, None
        ] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    p_x = x + i_b * _tmp1 + _tmp2
    _i = (tl.arange(0, BH) + BH * i_h)[:, None]
    _j = (tl.arange(0, BW) + BW * i_w)[None, :]
    _c_offset = (DW // K * (_i // K) + _j // K) * K * K + _i % K * K + _j % K
    if flip:
        _c_offset = DH * DW - _c_offset - 1
    p_y = y + i_b * _tmp1 + _tmp0 + _c_offset
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


def pad_tensor(x, w, H, W):
    if H % w == 0 and W % w == 0:
        return x, (H, W)
    B, C = x.shape[:2]
    if len(x.shape) == 3:
        x = x.view(B, C, H, W)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    newH, newW = Hg * w, Wg * w
    x = F.pad(x, (0, newW - W, 0, newH - H))
    return x, (newH, newW)


# Forward method (kernel launch code)
def _LocalScanTriton_forward(ctx, x: torch.Tensor, K: int, flip: bool, H:
    int=None, W: int=None):
    ori_x = x
    B, C = x.shape[:2]
    if H is None or W is None:
        if len(x.shape) == 4:
            H, W = x.shape[-2:]
        elif len(x.shape) == 3:
            raise RuntimeError('x must be BCHW format to infer the H W')
    B, C, H, W = int(B), int(C), int(H), int(W)
    ctx.ori_shape = B, C, H, W
    x, (H, W) = pad_tensor(x, K, H, W)
    ctx.shape = B, C, H, W
    BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.
        next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
    NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
    ctx.triton_shape = BC, BH, BW, NC, NH, NW
    ctx.K = K
    ctx.flip = flip
    if x.stride(-1) != 1:
        x = x.contiguous()
    if len(ori_x.shape) == 4:
        y = x.new_empty((B, C, H, W))
    elif len(ori_x.shape) == 3:
        y = x.new_empty((B, C, H * W))
    triton_local_scan[NH * NW, NC, B](x, y, K, flip, BC, BH, BW, C, H, W,
        NH, NW)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def triton_local_reverse(x, y, K: tl.constexpr, flip: tl.constexpr, BC: tl.
    constexpr, BH: tl.constexpr, BW: tl.constexpr, DC: tl.constexpr, DH: tl
    .constexpr, DW: tl.constexpr, NH: tl.constexpr, NW: tl.constexpr):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = i_hw // NW, i_hw % NW
    _mask_h = i_h * BH + tl.arange(0, BH) < DH
    _mask_w = i_w * BW + tl.arange(0, BW) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)
    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = _tmp0 + i_h * BH * DW + tl.arange(0, BH)[:, None
        ] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    p_x = x + i_b * _tmp1 + _tmp2
    _i = (tl.arange(0, BH) + BH * i_h)[:, None]
    _j = (tl.arange(0, BW) + BW * i_w)[None, :]
    _o = _i * DW + _j
    _i = _o // (K * K) // (DW // K) * K + _o % (K * K) // K
    _j = _o // (K * K) % (DW // K) * K + _o % (K * K) % K
    _c_offset = _i * DW + _j
    if flip:
        _c_offset = DH * DW - _c_offset - 1
    p_y = y + i_b * _tmp1 + _tmp0 + _c_offset
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


# Backward method (kernel launch code)
def _LocalScanTriton_backward(ctx, y: torch.Tensor):
    B, C, H, W = ctx.shape
    BC, BH, BW, NC, NH, NW = ctx.triton_shape
    if y.stride(-1) != 1:
        y = y.contiguous()
    if len(y.shape) == 4 or ctx.shape != ctx.ori_shape:
        x = y.new_empty((B, C, H, W))
    else:
        x = y.new_empty((B, C, H * W))
    triton_local_reverse[NH * NW, NC, B](y, x, ctx.K, ctx.flip, BC, BH, BW,
        C, H, W, NH, NW)
    if ctx.shape != ctx.ori_shape:
        _, _, ori_H, ori_W = ctx.ori_shape
        x = x[:, :, :ori_H, :ori_W]
        if len(y.shape) == 3:
            x = x.flatten(2)
    return x, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LocalScanTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, K: int, flip: bool, H: int=None, W:
        int=None):
        ori_x = x
        B, C = x.shape[:2]
        if H is None or W is None:
            if len(x.shape) == 4:
                H, W = x.shape[-2:]
            elif len(x.shape) == 3:
                raise RuntimeError('x must be BCHW format to infer the H W')
        B, C, H, W = int(B), int(C), int(H), int(W)
        ctx.ori_shape = B, C, H, W
        x, (H, W) = pad_tensor(x, K, H, W)
        ctx.shape = B, C, H, W
        BC, BH, BW = min(triton.next_power_of_2(C), 1), min(triton.
            next_power_of_2(H), 64), min(triton.next_power_of_2(W), 64)
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.triton_shape = BC, BH, BW, NC, NH, NW
        ctx.K = K
        ctx.flip = flip
        if x.stride(-1) != 1:
            x = x.contiguous()
        if len(ori_x.shape) == 4:
            y = x.new_empty((B, C, H, W))
        elif len(ori_x.shape) == 3:
            y = x.new_empty((B, C, H * W))
        triton_local_scan[NH * NW, NC, B](x, y, K, flip, BC, BH, BW, C, H,
            W, NH, NW)
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if y.stride(-1) != 1:
            y = y.contiguous()
        if len(y.shape) == 4 or ctx.shape != ctx.ori_shape:
            x = y.new_empty((B, C, H, W))
        else:
            x = y.new_empty((B, C, H * W))
        triton_local_reverse[NH * NW, NC, B](y, x, ctx.K, ctx.flip, BC, BH,
            BW, C, H, W, NH, NW)
        if ctx.shape != ctx.ori_shape:
            _, _, ori_H, ori_W = ctx.ori_shape
            x = x[:, :, :ori_H, :ori_W]
            if len(y.shape) == 3:
                x = x.flatten(2)
        return x, None, None, None, None
