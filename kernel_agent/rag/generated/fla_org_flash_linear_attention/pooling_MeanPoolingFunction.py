# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/utils/pooling.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/utils/pooling.py
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
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd

@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int
    ) ->torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(
        cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens
        )


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) ->torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BD': BD}, num_warps=num_warps) for
    BD in [16, 32, 64, 128] for num_warps in [1, 2, 4, 8]], key=['BT'], **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def mean_pooling_fwd_kernel(x, o, cu_seqlens, chunk_indices, T, H: tl.
    constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    p_x = tl.make_block_ptr(x + (bos * H + i_h) * D, (T, D), (H * D, 1), (
        i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_o = tl.make_block_ptr(o + (i_tg * H + i_h) * D, (D,), (1,), (i_d * BD
        ,), (BD,), (0,))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.sum(b_x, axis=0) / min(BT, T - i_t * BT)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def mean_pooling_fwd(x: torch.Tensor, chunk_size: int, cu_seqlens: Optional
    [torch.LongTensor]=None) ->torch.Tensor:
    B, T, H, D = x.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    o = x.new_empty(B, NT, H, D)

    def grid(meta):
        return triton.cdiv(D, meta['BD']), NT, B * H
    mean_pooling_fwd_kernel[grid](x, o, cu_seqlens, chunk_indices, T=T, H=H,
        D=D, BT=BT)
    return o


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _MeanPoolingFunction_forward(ctx, x: torch.Tensor, chunk_size: int,
    cu_seqlens: Optional[torch.LongTensor]=None) ->torch.Tensor:
    o = mean_pooling_fwd(x, chunk_size, cu_seqlens)
    ctx.batch_size = x.shape[0]
    ctx.seq_len = x.shape[1]
    ctx.chunk_size = chunk_size
    ctx.cu_seqlens = cu_seqlens
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BD': BD}, num_warps=num_warps) for
    BD in [16, 32, 64, 128] for num_warps in [1, 2, 4, 8]], key=['BT'], **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def mean_pooling_bwd_kernel(do, dx, cu_seqlens, chunk_indices, T, H: tl.
    constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    p_dx = tl.make_block_ptr(dx + (bos * H + i_h) * D, (T, D), (H * D, 1),
        (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    p_do = tl.make_block_ptr(do + (i_tg * H + i_h) * D, (D,), (1,), (i_d *
        BD,), (BD,), (0,))
    b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)
    b_dx = b_do / tl.full((BT,), min(BT, T - i_t * BT), dtype=tl.float32)[:,
        None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def mean_pooling_bwd(do: torch.Tensor, batch_size: int, seq_len: int,
    chunk_size: int, cu_seqlens: Optional[torch.LongTensor]=None
    ) ->torch.Tensor:
    B, T, H, D = batch_size, seq_len, *do.shape[-2:]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dx = do.new_empty(B, T, H, D)

    def grid(meta):
        return triton.cdiv(D, meta['BD']), NT, B * H
    mean_pooling_bwd_kernel[grid](do, dx, cu_seqlens, chunk_indices, T=T, H
        =H, D=D, BT=BT)
    return dx


# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _MeanPoolingFunction_backward(ctx, do) ->Tuple[torch.Tensor, None, None]:
    batch_size = ctx.batch_size
    seq_len = ctx.seq_len
    chunk_size = ctx.chunk_size
    cu_seqlens = ctx.cu_seqlens
    dx = mean_pooling_bwd(do, batch_size, seq_len, chunk_size, cu_seqlens)
    return dx, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class MeanPoolingFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, x: torch.Tensor, chunk_size: int, cu_seqlens: Optional
        [torch.LongTensor]=None) ->torch.Tensor:
        o = mean_pooling_fwd(x, chunk_size, cu_seqlens)
        ctx.batch_size = x.shape[0]
        ctx.seq_len = x.shape[1]
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        return o

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do) ->Tuple[torch.Tensor, None, None]:
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        chunk_size = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        dx = mean_pooling_bwd(do, batch_size, seq_len, chunk_size, cu_seqlens)
        return dx, None, None
