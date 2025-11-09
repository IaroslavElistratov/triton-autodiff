# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/lewandofskee/MobileMamba
# Source-Files: model/lib_mamba/csm_triton.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_0h1rwwyj/MobileMamba-main/model/lib_mamba/csm_triton.py
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

@triton.jit
def triton_cross_scan_flex(x, y, x_layout: tl.constexpr, y_layout: tl.
    constexpr, operation: tl.constexpr, onebyone: tl.constexpr, scans: tl.
    constexpr, BC: tl.constexpr, BH: tl.constexpr, BW: tl.constexpr, DC: tl
    .constexpr, DH: tl.constexpr, DW: tl.constexpr, NH: tl.constexpr, NW:
    tl.constexpr):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = i_hw // NW, i_hw % NW
    _mask_h = i_h * BH + tl.arange(0, BH) < DH
    _mask_w = i_w * BW + tl.arange(0, BW) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)
    HWRoute0 = i_h * BH * DW + tl.arange(0, BH)[:, None
        ] * DW + i_w * BW + tl.arange(0, BW)[None, :]
    HWRoute1 = i_w * BW * DH + tl.arange(0, BW)[None, :
        ] * DH + i_h * BH + tl.arange(0, BH)[:, None]
    HWRoute2 = (NH - i_h - 1) * BH * DW + (BH - 1 - tl.arange(0, BH)[:, None]
        ) * DW + (NW - i_w - 1) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (
        DH - NH * BH) * DW + (DW - NW * BW)
    HWRoute3 = (NW - i_w - 1) * BW * DH + (BW - 1 - tl.arange(0, BW)[None, :]
        ) * DH + (NH - i_h - 1) * BH + (BH - 1 - tl.arange(0, BH)[:, None]) + (
        DH - NH * BH) + (DW - NW * BW) * DH
    if scans == 1:
        HWRoute1 = HWRoute0
        HWRoute2 = HWRoute0
        HWRoute3 = HWRoute0
    elif scans == 2:
        HWRoute1 = HWRoute0
        HWRoute3 = HWRoute2
    _tmp1 = DC * DH * DW
    y_ptr_base = y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0
         else i_c * BC)
    if y_layout == 0:
        p_y1 = y_ptr_base + HWRoute0
        p_y2 = y_ptr_base + _tmp1 + HWRoute1
        p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
        p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
    else:
        p_y1 = y_ptr_base + HWRoute0 * 4 * DC
        p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
        p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
        p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC
    if onebyone == 0:
        x_ptr_base = x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0
             else i_c * BC)
        if x_layout == 0:
            p_x = x_ptr_base + HWRoute0
        else:
            p_x = x_ptr_base + HWRoute0 * DC
        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _x = tl.load(p_x + _idx_x, mask=_mask_hw)
                tl.store(p_y1 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y2 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y3 + _idx_y, _x, mask=_mask_hw)
                tl.store(p_y4 + _idx_y, _x, mask=_mask_hw)
        elif operation == 1:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                _y1 = tl.load(p_y1 + _idx_y, mask=_mask_hw)
                _y2 = tl.load(p_y2 + _idx_y, mask=_mask_hw)
                _y3 = tl.load(p_y3 + _idx_y, mask=_mask_hw)
                _y4 = tl.load(p_y4 + _idx_y, mask=_mask_hw)
                tl.store(p_x + _idx_x, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)
    else:
        x_ptr_base = x + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if x_layout ==
            0 else i_c * BC)
        if x_layout == 0:
            p_x1 = x_ptr_base + HWRoute0
            p_x2 = p_x1 + _tmp1
            p_x3 = p_x2 + _tmp1
            p_x4 = p_x3 + _tmp1
        else:
            p_x1 = x_ptr_base + HWRoute0 * 4 * DC
            p_x2 = p_x1 + DC
            p_x3 = p_x2 + DC
            p_x4 = p_x3 + DC
        if operation == 0:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_y1 + _idx_y, tl.load(p_x1 + _idx_x, mask=
                    _mask_hw), mask=_mask_hw)
                tl.store(p_y2 + _idx_y, tl.load(p_x2 + _idx_x, mask=
                    _mask_hw), mask=_mask_hw)
                tl.store(p_y3 + _idx_y, tl.load(p_x3 + _idx_x, mask=
                    _mask_hw), mask=_mask_hw)
                tl.store(p_y4 + _idx_y, tl.load(p_x4 + _idx_x, mask=
                    _mask_hw), mask=_mask_hw)
        else:
            for idxc in range(_for_C):
                _idx_x = idxc * DH * DW if x_layout == 0 else idxc
                _idx_y = idxc * DH * DW if y_layout == 0 else idxc
                tl.store(p_x1 + _idx_x, tl.load(p_y1 + _idx_y), mask=_mask_hw)
                tl.store(p_x2 + _idx_x, tl.load(p_y2 + _idx_y), mask=_mask_hw)
                tl.store(p_x3 + _idx_x, tl.load(p_y3 + _idx_y), mask=_mask_hw)
                tl.store(p_x4 + _idx_x, tl.load(p_y4 + _idx_y), mask=_mask_hw)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def _CrossScanTritonF_forward(ctx, x: torch.Tensor, in_channel_first=True,
    out_channel_first=True, one_by_one=False, scans=0):
    if one_by_one:
        if in_channel_first:
            B, _, C, H, W = x.shape
        else:
            B, H, W, _, C = x.shape
    elif in_channel_first:
        B, C, H, W = x.shape
    else:
        B, H, W, C = x.shape
    B, C, H, W = int(B), int(C), int(H), int(W)
    BC, BH, BW = 1, 32, 32
    NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
    ctx.in_channel_first = in_channel_first
    ctx.out_channel_first = out_channel_first
    ctx.one_by_one = one_by_one
    ctx.scans = scans
    ctx.shape = B, C, H, W
    ctx.triton_shape = BC, BH, BW, NC, NH, NW
    y = x.new_empty((B, 4, C, H * W)) if out_channel_first else x.new_empty((
        B, H * W, 4, C))
    triton_cross_scan_flex[NH * NW, NC, B](x.contiguous(), y, 0 if
        in_channel_first else 1, 0 if out_channel_first else 1, 0, 0 if not
        one_by_one else 1, scans, BC, BH, BW, C, H, W, NH, NW)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _CrossScanTritonF_backward(ctx, y: torch.Tensor):
    in_channel_first = ctx.in_channel_first
    out_channel_first = ctx.out_channel_first
    one_by_one = ctx.one_by_one
    scans = ctx.scans
    B, C, H, W = ctx.shape
    BC, BH, BW, NC, NH, NW = ctx.triton_shape
    if one_by_one:
        x = y.new_empty((B, 4, C, H, W)) if in_channel_first else y.new_empty((
            B, H, W, 4, C))
    else:
        x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty((
            B, H, W, C))
    triton_cross_scan_flex[NH * NW, NC, B](x, y.contiguous(), 0 if
        in_channel_first else 1, 0 if out_channel_first else 1, 1, 0 if not
        one_by_one else 1, scans, BC, BH, BW, C, H, W, NH, NW)
    return x, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CrossScanTritonF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True,
        out_channel_first=True, one_by_one=False, scans=0):
        if one_by_one:
            if in_channel_first:
                B, _, C, H, W = x.shape
            else:
                B, H, W, _, C = x.shape
        elif in_channel_first:
            B, C, H, W = x.shape
        else:
            B, H, W, C = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = 1, 32, 32
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        ctx.shape = B, C, H, W
        ctx.triton_shape = BC, BH, BW, NC, NH, NW
        y = x.new_empty((B, 4, C, H * W)
            ) if out_channel_first else x.new_empty((B, H * W, 4, C))
        triton_cross_scan_flex[NH * NW, NC, B](x.contiguous(), y, 0 if
            in_channel_first else 1, 0 if out_channel_first else 1, 0, 0 if
            not one_by_one else 1, scans, BC, BH, BW, C, H, W, NH, NW)
        return y

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        if one_by_one:
            x = y.new_empty((B, 4, C, H, W)
                ) if in_channel_first else y.new_empty((B, H, W, 4, C))
        else:
            x = y.new_empty((B, C, H, W)) if in_channel_first else y.new_empty(
                (B, H, W, C))
        triton_cross_scan_flex[NH * NW, NC, B](x, y.contiguous(), 0 if
            in_channel_first else 1, 0 if out_channel_first else 1, 1, 0 if
            not one_by_one else 1, scans, BC, BH, BW, C, H, W, NH, NW)
        return x, None, None, None, None
