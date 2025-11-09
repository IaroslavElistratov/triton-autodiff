# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/IntelLabs/EquiTriton
# Source-Files: src/equitriton/sph_harm/direct/y_2.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_qa3qiele/EquiTriton-main/src/equitriton/sph_harm/direct/y_2.py
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
def second_order_fwd(coord_ptr: tl.tensor, output_ptr: tl.tensor,
    block_size: tl.constexpr, coord_numel: tl.constexpr, output_numel: tl.
    constexpr, col_offset: tl.constexpr, output_stride: tl.constexpr):
    coord_stride = 3
    block_id = tl.program_id(0)
    coord_striding = tl.arange(0, block_size) * coord_stride
    coord_row_offset = coord_striding + block_size * coord_stride * block_id
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    y = tl.load(coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 <
        coord_numel)
    z = tl.load(coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 <
        coord_numel)
    CONST_00 = 3.87298334620742
    CONST_01 = 2.23606797749979
    CONST_02 = -1.11803398874989
    CONST_03 = 1.93649167310371
    Y20 = CONST_00 * x * z
    Y21 = CONST_00 * x * y
    Y23 = CONST_00 * y * z
    Y22 = CONST_02 * x * x + CONST_01 * y * y + CONST_02 * z * z
    Y24 = -CONST_03 * x * x + CONST_03 * z * z
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (output_striding + block_size * output_stride *
        block_id + col_offset)
    tl.store(output_ptr + output_row_offset, Y20, mask=output_row_offset <
        output_numel)
    tl.store(output_ptr + output_row_offset + 1, Y21, mask=
        output_row_offset + 1 < output_numel)
    tl.store(output_ptr + output_row_offset + 2, Y22, mask=
        output_row_offset + 2 < output_numel)
    tl.store(output_ptr + output_row_offset + 3, Y23, mask=
        output_row_offset + 3 < output_numel)
    tl.store(output_ptr + output_row_offset + 4, Y24, mask=
        output_row_offset + 4 < output_numel)


# Forward method (kernel launch code)
def _SecondOrderSphericalHarmonic_forward(ctx, coords: torch.Tensor,
    output_tensor: (torch.Tensor | None)=None, mask: (torch.Tensor | None)=
    None, block_size: int=64, col_offset: int=0):
    num_projections = 5
    if not isinstance(output_tensor, torch.Tensor):
        output_tensor = torch.empty((*coords.shape[:-1], num_projections),
            dtype=coords.dtype, device=coords.device)
    coord_numel = coords.numel()
    output_numel = output_tensor.numel()
    num_blocks = calculate_lastdim_num_blocks(coords, block_size)
    second_order_fwd[num_blocks,](coords, output_tensor, block_size,
        coord_numel, output_numel, col_offset, output_tensor.stride(-2))
    ctx.save_for_backward(coords)
    return output_tensor


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def second_order_bwd(coord_ptr: tl.tensor, coord_grad_ptr: tl.tensor,
    sph_grad_ptr: tl.tensor, block_size: tl.constexpr, coord_numel: tl.
    constexpr, output_numel: tl.constexpr, col_offset: tl.constexpr,
    output_stride: tl.constexpr):
    block_id = tl.program_id(0)
    coord_stride = 3
    coord_striding = tl.arange(0, block_size) * coord_stride
    coord_row_offset = coord_striding + block_size * coord_stride * block_id
    x = tl.load(coord_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    y = tl.load(coord_ptr + coord_row_offset + 1, mask=coord_row_offset + 1 <
        coord_numel)
    z = tl.load(coord_ptr + coord_row_offset + 2, mask=coord_row_offset + 2 <
        coord_numel)
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (output_striding + block_size * output_stride *
        block_id + col_offset)
    CONST_00 = 3.87298334620742
    CONST_01 = 2.23606797749979
    CONST_02 = 4.47213595499958
    g_Y20 = tl.load(sph_grad_ptr + output_row_offset, mask=
        output_row_offset < output_numel)
    g_Y21 = tl.load(sph_grad_ptr + output_row_offset + 1, mask=
        output_row_offset + 1 < output_numel)
    g_Y22 = tl.load(sph_grad_ptr + output_row_offset + 2, mask=
        output_row_offset + 2 < output_numel)
    g_Y23 = tl.load(sph_grad_ptr + output_row_offset + 3, mask=
        output_row_offset + 3 < output_numel)
    g_Y24 = tl.load(sph_grad_ptr + output_row_offset + 4, mask=
        output_row_offset + 4 < output_numel)
    g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    g_y = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=
        coord_row_offset + 1 < coord_numel)
    g_z = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=
        coord_row_offset + 2 < coord_numel)
    g_x += (CONST_00 * g_Y20 * z + CONST_00 * g_Y21 * y - CONST_01 * g_Y22 *
        x - CONST_00 * g_Y24 * x)
    g_y += CONST_00 * g_Y21 * x + CONST_02 * g_Y22 * y + CONST_00 * g_Y23 * z
    g_z += (CONST_00 * g_Y20 * x - CONST_01 * g_Y22 * z + CONST_00 * g_Y23 *
        y + CONST_00 * g_Y24 * z)
    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset <
        coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=
        coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=
        coord_row_offset + 2 < coord_numel)


# Backward method (kernel launch code)
def _SecondOrderSphericalHarmonic_backward(ctx, sph_grad_tensor: torch.
    Tensor, block_size: int=64, col_offset: int=0) ->torch.Tensor:
    coords, = ctx.saved_tensors
    coord_grad_output = torch.zeros_like(coords)
    num_blocks = calculate_lastdim_num_blocks(coords, block_size)
    second_order_bwd[num_blocks,](coords, coord_grad_output,
        sph_grad_tensor, block_size, coords.numel(), sph_grad_tensor.numel(
        ), col_offset, sph_grad_tensor.stride(-2))
    return coord_grad_output


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SecondOrderSphericalHarmonic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords: torch.Tensor, output_tensor: (torch.Tensor |
        None)=None, mask: (torch.Tensor | None)=None, block_size: int=64,
        col_offset: int=0):
        num_projections = 5
        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.empty((*coords.shape[:-1],
                num_projections), dtype=coords.dtype, device=coords.device)
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        second_order_fwd[num_blocks,](coords, output_tensor, block_size,
            coord_numel, output_numel, col_offset, output_tensor.stride(-2))
        ctx.save_for_backward(coords)
        return output_tensor

    @staticmethod
    def backward(ctx, sph_grad_tensor: torch.Tensor, block_size: int=64,
        col_offset: int=0) ->torch.Tensor:
        coords, = ctx.saved_tensors
        coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        second_order_bwd[num_blocks,](coords, coord_grad_output,
            sph_grad_tensor, block_size, coords.numel(), sph_grad_tensor.
            numel(), col_offset, sph_grad_tensor.stride(-2))
        return coord_grad_output
