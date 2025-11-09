# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/HazyResearch/fly
# Source-Files: src/models/attention/blocksparse_sum.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_lyot2jni/fly-master/src/models/attention/blocksparse_sum.py
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
from math import exp
import time

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'num_warps': lambda *args, **meta: num_warps(args[3] *
    meta['BLOCK'])})
@triton.heuristics({'TN': lambda *args, **meta: next_power_of_2(args[3] *
    meta['BLOCK'])})
@triton.jit
def _forward(X, OUT, LUT, sizemax, stride_zx, stride_zout, stride_hout, **meta
    ):
    TN = meta['TN']
    BLOCK = meta['BLOCK']
    pidhm = tl.program_id(0)
    pidz = tl.program_id(1)
    rxm = pidhm % BLOCK
    rbm = pidhm // BLOCK
    rxn = tl.arange(0, TN) % BLOCK
    rbn = tl.arange(0, TN) // BLOCK
    header = LUT + rbm * 2
    size = tl.load(header + 0)
    offset = tl.load(header + 1)
    check = rbn < size
    rbmn = tl.where(check, rbn, size - 1)
    blockid = tl.load(LUT + offset + rbmn * 4 + 0)
    rowid = tl.load(LUT + offset + rbmn * 4 + 2)
    headid = tl.load(LUT + offset + rbmn * 4 + 3)
    px = X + pidz * stride_zx + blockid * BLOCK * BLOCK + rxm * BLOCK + rxn
    x = tl.load(px, mask=check, other=0)
    x = x.to(tl.float32)
    out = tl.sum(x, axis=0)
    pout = (OUT + pidz * stride_zout + headid * stride_hout + rowid * BLOCK +
        rxm)
    tl.store(pout, out)


# Forward method (kernel launch code)
def __sum_forward(ctx, x, spdims, block, lut, maxlut, n_head, n_row, layout,
    bench, time):
    out = torch.zeros((x.shape[0], n_head, n_row), dtype=x.dtype, device=x.
        device)
    M = x.shape[0]
    meta = {'BLOCK': block}
    grid = lambda opt: [spdims[0] * spdims[1] * block, M]
    _forward[grid](x, out, lut, maxlut, x.stride(0), out.stride(0), out.
        stride(1), force_nc_cache=True, **meta)
    ctx.save_for_backward(x, lut, layout)
    ctx.spdims = spdims
    ctx.block = block
    ctx.maxlut = maxlut
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

def sparsify_broadcast_tensor(x, mask):
    """
    Arguments:
        x: (batch_size, n_head, T)
        mask: (n_head, T // block_size, S // block_size), with dtype torch.long
        block_size: int in {16, 32, 64, 128}
    Return:
        x_sparse: (batch_size, nnz(mask), block_size, 1)
    """
    block_size = x.shape[-1] // mask.shape[-2]
    x_reshaped = rearrange(x, 'b h (p blk_sz) -> b h p blk_sz 1', blk_sz=
        block_size)
    h_idx, row_idx, _ = torch.nonzero(mask, as_tuple=True)
    return x_reshaped[:, h_idx, row_idx]


# Backward method (kernel launch code)
def __sum_backward(ctx, dout):
    x, lut, layout = ctx.saved_tensors
    block = x.shape[-1]
    dx = sparsify_broadcast_tensor(dout, layout, block).expand(-1, -1, -1,
        block)
    return dx, None, None, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _sum(torch.autograd.Function):

    @staticmethod
    def make_lut(layout, block, device):
        _empty = torch.tensor([], dtype=torch.int64, device=layout.device)
        sizes = _empty.clone()
        for h in range(layout.shape[0]):
            sizes = torch.cat((sizes, layout[h, :, :].sum(-1)))
        offsets = torch.zeros_like(sizes)
        offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
        idx = torch.arange(layout.sum())
        head = layout.nonzero(as_tuple=False)[:, 0]
        rows = layout.nonzero(as_tuple=False)[:, 1]
        columns = layout.nonzero(as_tuple=False)[:, 2]
        core = torch.stack((idx, columns, rows, head), dim=1).view(-1)
        offsets = offsets * 4 + 2 * sizes.numel()
        header = torch.stack((sizes, offsets), dim=1).view(-1)
        lut = torch.cat((header, core)).type(torch.int32).to(device)
        n_head = layout.shape[0]
        n_row = layout.shape[1] * block
        return lut, int(sizes.max()), n_head, n_row

    @staticmethod
    def forward(ctx, x, spdims, block, lut, maxlut, n_head, n_row, layout,
        bench, time):
        out = torch.zeros((x.shape[0], n_head, n_row), dtype=x.dtype,
            device=x.device)
        M = x.shape[0]
        meta = {'BLOCK': block}
        grid = lambda opt: [spdims[0] * spdims[1] * block, M]
        _forward[grid](x, out, lut, maxlut, x.stride(0), out.stride(0), out
            .stride(1), force_nc_cache=True, **meta)
        ctx.save_for_backward(x, lut, layout)
        ctx.spdims = spdims
        ctx.block = block
        ctx.maxlut = maxlut
        return out

    @staticmethod
    def backward(ctx, dout):
        x, lut, layout = ctx.saved_tensors
        block = x.shape[-1]
        dx = sparsify_broadcast_tensor(dout, layout, block).expand(-1, -1, 
            -1, block)
        return dx, None, None, None, None, None, None, None, None, None
