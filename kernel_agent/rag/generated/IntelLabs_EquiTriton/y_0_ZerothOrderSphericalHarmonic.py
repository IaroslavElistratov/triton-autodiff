# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/IntelLabs/EquiTriton
# Source-Files: src/equitriton/sph_harm/direct/y_0.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_qa3qiele/EquiTriton-main/src/equitriton/sph_harm/direct/y_0.py
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
def zeroth_order_fwd(coord_ptr: tl.tensor, output_ptr: tl.tensor,
    block_size: tl.constexpr, coord_numel: tl.constexpr, output_numel: tl.
    constexpr, col_offset: tl.constexpr, output_stride: tl.constexpr):
    block_id = tl.program_id(0)
    output_striding = tl.arange(0, block_size) * output_stride
    output_row_offset = (output_striding + block_size * output_stride *
        block_id + col_offset)
    tl.store(output_ptr + output_row_offset, 1.0, mask=output_row_offset <
        output_numel)


# Forward method (kernel launch code)
def _ZerothOrderSphericalHarmonic_forward(ctx, coords: torch.Tensor,
    output_tensor: (torch.Tensor | None)=None, mask: (torch.Tensor | None)=
    None, block_size: int=64, col_offset: int=0):
    if not isinstance(output_tensor, torch.Tensor):
        output_tensor = torch.ones((*coords.shape[:-1], 1), dtype=coords.
            dtype, device=coords.device)
    ctx.save_for_backward(coords)
    coord_numel = coords.numel()
    output_numel = output_tensor.numel()
    num_blocks = calculate_lastdim_num_blocks(coords, block_size)
    zeroth_order_fwd[num_blocks,](coords, output_tensor, block_size,
        coord_numel, output_numel, col_offset, output_tensor.stride(-2))
    return output_tensor


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def zeroth_order_bwd(coord_ptr: tl.tensor, coord_grad_ptr: tl.tensor,
    sph_grad_ptr: tl.tensor, block_size: tl.constexpr, coord_numel: tl.
    constexpr, output_numel: tl.constexpr, col_offset: tl.constexpr,
    output_stride: tl.constexpr):
    block_id = tl.program_id(0)


# Backward method (kernel launch code)
def _ZerothOrderSphericalHarmonic_backward(ctx, sph_grad_tensor: torch.
    Tensor, block_size: int=64, col_offset: int=0) ->torch.Tensor:
    coords, = ctx.saved_tensors
    coord_grad_output = torch.zeros_like(coords)
    num_blocks = calculate_lastdim_num_blocks(coords, block_size)
    zeroth_order_bwd[num_blocks,](coord_grad_output, sph_grad_tensor,
        block_size, coords.numel(), sph_grad_tensor.numel(), col_offset,
        sph_grad_tensor.stride(-2))
    return coord_grad_output


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ZerothOrderSphericalHarmonic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords: torch.Tensor, output_tensor: (torch.Tensor |
        None)=None, mask: (torch.Tensor | None)=None, block_size: int=64,
        col_offset: int=0):
        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.ones((*coords.shape[:-1], 1), dtype=
                coords.dtype, device=coords.device)
        ctx.save_for_backward(coords)
        coord_numel = coords.numel()
        output_numel = output_tensor.numel()
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        zeroth_order_fwd[num_blocks,](coords, output_tensor, block_size,
            coord_numel, output_numel, col_offset, output_tensor.stride(-2))
        return output_tensor

    @staticmethod
    def backward(ctx, sph_grad_tensor: torch.Tensor, block_size: int=64,
        col_offset: int=0) ->torch.Tensor:
        coords, = ctx.saved_tensors
        coord_grad_output = torch.zeros_like(coords)
        num_blocks = calculate_lastdim_num_blocks(coords, block_size)
        zeroth_order_bwd[num_blocks,](coord_grad_output, sph_grad_tensor,
            block_size, coords.numel(), sph_grad_tensor.numel(), col_offset,
            sph_grad_tensor.stride(-2))
        return coord_grad_output
