# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/modules/rotary.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/modules/rotary.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int=0) ->int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(
            tensor_idx)['multiprocessor_count']
    except BaseException:
        if triton.runtime.driver.active.get_current_target().backend == 'npu':
            return triton.runtime.driver.active.utils.get_device_properties(
                tensor_idx)['num_vectorcore']
        else:
            return 1


def rotary_embedding_fwdbwd(x: torch.Tensor, cos: torch.Tensor, sin: torch.
    Tensor, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens:
    Optional[torch.Tensor]=None, interleaved: bool=False, inplace: bool=
    False, conjugate: bool=False) ->torch.Tensor:
    """
    Args:
        x: [B, T, H, D].
        cos: [TR, R / 2]
        sin: [TR, R / 2]
        seqlen_offsets: integer or integer tensor of size [N]
        cu_seqlens: [N + 1,] or None

    Returns:
        y: [B, T, H, D]
    """
    is_varlen = cu_seqlens is not None
    B, T, H, D = x.shape
    N = B if not is_varlen else cu_seqlens.shape[0] - 1
    TR, R = cos.shape
    R2 = R * 2
    assert D <= 256, 'Only support D <= 256'
    assert TR >= T, f'TR must be >= T, got {TR} and {T}'
    assert cos.dtype == sin.dtype, f'cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}'
    assert x.dtype == cos.dtype, f'Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}'
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (N,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
    else:
        assert seqlen_offsets + T <= TR
    y = torch.empty_like(x) if not inplace else x
    if R2 < D and not inplace:
        y[..., R2:].copy_(x[..., R2:])
    BD = triton.next_power_of_2(R2)
    BT = min(128, triton.next_power_of_2(triton.cdiv(T,
        get_multiprocessor_count(x.device.index))))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if is_varlen else None
    NT = len(chunk_indices) if is_varlen else triton.cdiv(T, BT)
    grid = NT, B, H
    rotary_embedding_kernel[grid](x, cos, sin, y, cu_seqlens, chunk_indices,
        seqlen_offsets, B=B, T=T, H=H, D=D, R=R, TR=TR, BT=BT, BD=BD,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offsets, torch.Tensor),
        IS_VARLEN=is_varlen, INTERLEAVED=interleaved, CONJUGATE=conjugate)
    return y


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 
    3, 4]], key=['B', 'H', 'D', 'INTERLEAVED'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def rotary_embedding_kernel(x, cos, sin, y, cu_seqlens, chunk_indices,
    seq_offsets, T, B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, R:
    tl.constexpr, TR: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr, IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        T = eos - bos
        x = x + bos * H * D + i_h * D
        y = y + bos * H * D + i_h * D
    else:
        i_n = i_b
        x = x + i_n * T * H * D + i_h * D
        y = y + i_n * T * H * D + i_h * D
    if i_t * BT >= T:
        return
    o_t = i_t * BT + tl.arange(0, BT)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        o_cs = o_t + seq_offsets
    else:
        o_cs = o_t + tl.load(seq_offsets + i_n)
    m_t = (o_t >= 0) & (o_t < T) & (o_cs >= 0) & (o_cs < TR)
    if not INTERLEAVED:
        o_r = tl.arange(0, BD // 2)
        p_x = x + o_t[:, None] * H * D + o_r[None, :]
        p_cos = cos + (o_cs[:, None] * R + o_r[None, :])
        p_sin = sin + (o_cs[:, None] * R + o_r[None, :])
        mask = m_t[:, None] & (o_r < R)[None, :]
        b_cos = tl.load(p_cos, mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(p_sin, mask=mask, other=0.0).to(tl.float32)
        b_x0 = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x + R, mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_o0 = b_x0 * b_cos - b_x1 * b_sin
        b_o1 = b_x0 * b_sin + b_x1 * b_cos
        p_y = y + (o_t[:, None] * H * D + o_r[None, :])
        tl.store(p_y, b_o0, mask=mask)
        tl.store(p_y + R, b_o1, mask=mask)
    else:
        o_d = tl.arange(0, BD)
        o_d_swap = o_d + (o_d + 1) % 2 * 2 - 1
        o_d_repeat = tl.arange(0, BD) // 2
        p_x0 = x + o_t[:, None] * H * D + o_d[None, :]
        p_x1 = x + o_t[:, None] * H * D + o_d_swap[None, :]
        p_cos = cos + (o_cs[:, None] * R + o_d_repeat[None, :])
        p_sin = sin + (o_cs[:, None] * R + o_d_repeat[None, :])
        mask = m_t[:, None] & (o_d_repeat < R)[None, :]
        b_cos = tl.load(p_cos, mask=mask, other=1.0).to(tl.float32)
        b_sin = tl.load(p_sin, mask=mask, other=0.0).to(tl.float32)
        b_x0 = tl.load(p_x0, mask=mask, other=0.0).to(tl.float32)
        b_x1 = tl.load(p_x1, mask=mask, other=0.0).to(tl.float32)
        if CONJUGATE:
            b_sin = -b_sin
        b_o0 = b_x0 * b_cos
        b_o1 = b_x1 * b_sin
        b_y = tl.where(o_d[None, :] % 2 == 0, b_o0 - b_o1, b_o0 + b_o1)
        p_y = y + (o_t[:, None] * H * D + o_d[None, :])
        tl.store(p_y, b_y, mask=mask)


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

# Forward method (kernel launch code)
@input_guard
def _RotaryEmbeddingFunction_forward(ctx, x, cos, sin, interleaved=False,
    inplace=False, seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens:
    Optional[torch.Tensor]=None):
    y = rotary_embedding_fwdbwd(x, cos, sin, seqlen_offsets=seqlen_offsets,
        cu_seqlens=cu_seqlens, interleaved=interleaved, inplace=inplace)
    if isinstance(seqlen_offsets, int):
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.seqlen_offsets = seqlen_offsets
    else:
        ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
        ctx.seqlen_offsets = None
    ctx.interleaved = interleaved
    ctx.inplace = inplace
    return y if not inplace else x


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
@input_guard
def _RotaryEmbeddingFunction_backward(ctx, do):
    seqlen_offsets = ctx.seqlen_offsets
    if seqlen_offsets is None:
        cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
    else:
        cos, sin, cu_seqlens = ctx.saved_tensors
    if not ctx.interleaved and not ctx.inplace:
        do = do.clone()
    dx = rotary_embedding_fwdbwd(do, cos, sin, seqlen_offsets=
        seqlen_offsets, cu_seqlens=cu_seqlens, interleaved=ctx.interleaved,
        inplace=ctx.inplace, conjugate=True)
    return dx, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RotaryEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False,
        seqlen_offsets: Union[int, torch.Tensor]=0, cu_seqlens: Optional[
        torch.Tensor]=None):
        y = rotary_embedding_fwdbwd(x, cos, sin, seqlen_offsets=
            seqlen_offsets, cu_seqlens=cu_seqlens, interleaved=interleaved,
            inplace=inplace)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return y if not inplace else x

    @staticmethod
    @input_guard
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = rotary_embedding_fwdbwd(do, cos, sin, seqlen_offsets=
            seqlen_offsets, cu_seqlens=cu_seqlens, interleaved=ctx.
            interleaved, inplace=ctx.inplace, conjugate=True)
        return dx, None, None, None, None, None, None, None
