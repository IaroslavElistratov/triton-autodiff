# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/IntelLabs/EquiTriton
# Source-Files: src/equitriton/sph_harm/direct/y_1.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_qa3qiele/EquiTriton-main/src/equitriton/sph_harm/direct/y_1.py
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
def first_order_fwd(coord_ptr: tl.tensor, output_ptr: tl.tensor, block_size:
    tl.constexpr, coord_numel: tl.constexpr, output_numel: tl.constexpr,
    col_offset: tl.constexpr, output_stride: tl.constexpr):
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
    CONST_00 = tl.sqrt(3.0)
    Y10 = CONST_00 * x
    Y11 = CONST_00 * y
    Y12 = CONST_00 * z
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (output_striding + block_size * output_stride *
        block_id + col_offset)
    tl.store(output_ptr + output_row_offset, Y10, mask=output_row_offset <
        output_numel)
    tl.store(output_ptr + output_row_offset + 1, Y11, mask=
        output_row_offset + 1 < output_numel)
    tl.store(output_ptr + output_row_offset + 2, Y12, mask=
        output_row_offset + 2 < output_numel)


# Forward method (kernel launch code)
def _FirstOrderSphericalHarmonic_forward(ctx, coords: torch.Tensor, mask: (
    torch.Tensor | None)=None, block_size: int=64, col_offset: int=0):
    output_tensor = torch.empty((*coords.shape[:-1], 3), dtype=coords.dtype,
        device=coords.device)
    coord_numel = coords.numel()
    output_numel = output_tensor.numel()
    num_blocks = calculate_lastdim_num_blocks(coords, block_size)
    first_order_fwd[num_blocks,](coords, output_tensor, block_size,
        coord_numel, output_numel, col_offset)
    ctx.save_for_backward(coords)
    return output_tensor


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def first_order_bwd(coord_ptr: tl.tensor, coord_grad_ptr: tl.tensor,
    sph_grad_ptr: tl.tensor, block_size: tl.constexpr, coord_numel: tl.
    constexpr, output_numel: tl.constexpr, col_offset: tl.constexpr,
    output_stride: tl.constexpr):
    block_id = tl.program_id(0)
    coord_stride = 3
    coord_striding = tl.arange(0, block_size) * coord_stride
    coord_row_offset = coord_striding + block_size * coord_stride * block_id
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (output_striding + block_size * output_stride *
        block_id + col_offset)
    g_Y10 = tl.load(sph_grad_ptr + output_row_offset, mask=
        output_row_offset < output_numel)
    g_Y11 = tl.load(sph_grad_ptr + output_row_offset + 1, mask=
        output_row_offset + 1 < output_numel)
    g_Y12 = tl.load(sph_grad_ptr + output_row_offset + 2, mask=
        output_row_offset + 2 < output_numel)
    g_x = tl.load(coord_grad_ptr + coord_row_offset, mask=coord_row_offset <
        coord_numel)
    g_y = tl.load(coord_grad_ptr + coord_row_offset + 1, mask=
        coord_row_offset + 1 < coord_numel)
    g_z = tl.load(coord_grad_ptr + coord_row_offset + 2, mask=
        coord_row_offset + 2 < coord_numel)
    CONST_00 = tl.sqrt(3.0)
    g_x += CONST_00 * g_Y10
    g_y += CONST_00 * g_Y11
    g_z += CONST_00 * g_Y12
    tl.store(coord_grad_ptr + coord_row_offset, g_x, mask=coord_row_offset <
        coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 1, g_y, mask=
        coord_row_offset + 1 < coord_numel)
    tl.store(coord_grad_ptr + coord_row_offset + 2, g_z, mask=
        coord_row_offset + 2 < coord_numel)


# Backward method (kernel launch code)
def _FirstOrderSphericalHarmonic_backward(ctx, sph_grad_tensor: torch.
    Tensor, block_size: int=64, col_offset: int=0) ->torch.Tensor:
    coords, = ctx.saved_tensors
    coord_grad_output = torch.zeros_like(coords)
    num_blocks = calculate_lastdim_num_blocks(coords, block_size)
    first_order_bwd[num_blocks,](coord_grad_output, sph_grad_tensor,
        block_size, coords.numel(), sph_grad_tensor.numel(), col_offset)
    return coord_grad_output


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FirstOrderSphericalHarmonic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords: torch.Tensor, mask: (torch.Tensor | None)=None,
        block_size: int=64, col_offset: int=0):
        output_tensor = torch.empty((*coords.shape[:-1], 3), dtype=coords.
            dtype, device=coords.device)
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        first_order_fwd[num_blocks,](coords, output_tensor, block_size,
            coord_numel, output_numel, col_offset)
        ctx.save_for_backward(coords)
        return output_tensor

    @staticmethod
    def backward(ctx, sph_grad_tensor: torch.Tensor, block_size: int=64,
        col_offset: int=0) ->torch.Tensor:
        coords, = ctx.saved_tensors
        coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        first_order_bwd[num_blocks,](coord_grad_output, sph_grad_tensor,
            block_size, coords.numel(), sph_grad_tensor.numel(), col_offset)
        return coord_grad_output
