# /root/private-scrape/UNZIPED_selected/fla_org_flash_linear_attention/flash-linear-attention-main/fla/ops/gla/fused_chunk.py


# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from packaging import version

import contextlib
import functools
import inspect
import logging
import os
import sys
import warnings
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple


#from fla.ops.utils import chunk_local_cumsum

device = triton.runtime.driver.active.get_current_target().backend

device_torch_lib = getattr(torch, device)

def custom_device_ctx(index: int):
    return device_torch_lib.device(index)

def input_guard(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper



# ============================================#=========================================================================================================================================================

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

# from fla.ops.utils.index import prepare_chunk_indices

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple


def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


# from fla.utils import autotune_cache_kwargs, check_shared_mem

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'

supports_autotune_cache = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if supports_autotune_cache else {}


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False




BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BS': BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['B', 'H', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        if REVERSE:
            b_o = -b_o + b_ss + b_s
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [16, 32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['B', 'H', 'S', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    o_i = tl.arange(0, BT)
    if REVERSE:
        m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)
    else:
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        if HAS_SCALE:
            b_c *= scale
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        b_z += tl.sum(b_s, 0)


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return g


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    # keep cummulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_local_cumsum_vector_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return g


@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (N * H,)
    chunk_global_cumsum_scalar_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return z


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BS = min(32, triton.next_power_of_2(S))

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (triton.cdiv(S, BS), N * H)
    chunk_global_cumsum_vector_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        S=S,
        BS=BS,
        HEAD_FIRST=head_first,
        REVERSE=reverse
    )
    return z


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype
        )
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype
        )
    else:
        raise ValueError(
            f"Unsupported input shape {s.shape}, "
            f"which should be [B, T, H]/[B, T, H, D] if `head_first=False` "
            f"or [B, H, T]/[B, H, T, D] otherwise"
        )


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    output_dtype: Optional[torch.dtype] = torch.float,
    **kwargs
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )


#=========================================================================================================================================================


# from fla.ops.utils.op import exp, safe_exp

exp = tl.exp

@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float('-inf')))


# from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
device = 'cuda' if device == 'cpu' else device
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)

# ================================================================================================================================================




@triton.jit(do_not_specialize=['T'])
def prepare_qg_kg(
    q,
    k,
    g,
    qg,
    kg,
    scale,
    T,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * T*K + i_c * BT * K + i_k * BK + tl.arange(0, BK)

    mask = (i_k * BK + tl.arange(0, BK)) < K

    last_decay = tl.load(g + i_bh * T*K + (i_c * BT + BT - 1) * K + i_k * BK + tl.arange(0, BK))

    for _ in range(BT):
        b_q = tl.load(p_q, mask=mask, other=0)
        b_k = tl.load(p_k, mask=mask, other=0)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_q *= exp(b_g) * scale
        b_k *= exp(last_decay - b_g)
        tl.store(p_kg, b_k.to(p_kg.dtype.element_ty), mask=mask)
        tl.store(p_qg, b_q.to(p_qg.dtype.element_ty), mask=mask)
        p_q += K
        p_g += K
        p_k += K
        p_kg += K
        p_qg += K


@triton.jit(do_not_specialize=['T'])
def bwd_decay_global_cumsum(
    dq_inner,
    dq_inter,
    dk_inner,
    dk_inter,
    q,
    k,
    g,
    dg,
    T,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_k = k + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_g = g + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dg = dg + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dq_inner = dq_inner + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dk_inner = dk_inner + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dq_inter = dq_inter + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dk_inter = dk_inter + i_bh * T*K + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = (i_k * BK + tl.arange(0, BK)) < K
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(BT-1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        if j == (BT-1):
            last_g = b_g
        b_dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        b_dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        b_dq2 *= exp(b_g)
        b_dq = b_dq1 + b_dq2
        tl.store(p_dq_inter, b_dq, mask=mask)
        b_dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        b_dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        b_dk2 *= safe_exp(last_g - b_g)
        b_dk = b_dk1 + b_dk2
        tl.store(p_dk_inter, b_dk, mask=mask)
        b_q = tl.load(p_q, mask=mask, other=0)
        b_k = tl.load(p_k, mask=mask, other=0)
        b_dg = b_dq * b_q - b_dk * b_k
        cum_grad_dg += b_dg
        tl.store(p_dg, cum_grad_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= K
        p_k -= K
        p_q -= K
        p_dq_inner -= K
        p_dk_inner -= K
        p_dq_inter -= K
        p_dk_inter -= K
        p_dg -= K


@triton.jit(do_not_specialize=['T'])
def fused_chunk_gla_fwd_kernel(
    q,
    k,
    v,
    g,
    o,
    h0,
    ht,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # make block pointers
    p_q = tl.make_block_ptr(q + i_bh * T*K, (T, K), (K, 1), (0, i_k * BK), (BT, BK), (1, 0))
    p_gn = g + i_bh * T*K + (BT - 1) * K + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (K, T), (1, K), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * T*V, (T, V), (V, 1), (0, i_v * BV), (BT, BV), (1, 0))

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < K

    for i in range(0, tl.cdiv(T, BT)):
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gn = tl.load(p_gn, mask=mask, other=0).to(tl.float32)
        if CHECK and i == 0:
            b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
            b_h = b_h * exp(b_gn)[:, None] + tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)
        else:
            b_o = tl.dot(b_q.to(b_v.dtype), b_h.to(b_v.dtype), allow_tf32=False)
            b_h = b_h * exp(b_gn)[:, None] + tl.dot(b_k.to(b_v.dtype), b_v, allow_tf32=False)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_gn += BT * K

    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h.to(p_final.dtype.element_ty), boundary_check=(0, 1))


# Similar to Algorithm1 of https://arxiv.org/abs/2006.16236
@triton.jit(do_not_specialize=['T'])
def fused_chunk_gla_bwd_kernel(
    q, k, v, g,
    do,
    dq,
    dk,
    dv,
    h0,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    # clamp_min, # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    CHECK: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # [BV, BK]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < K
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = g + i_bh * T*K + ((i+1) * BT - 1) * K + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * T*V, (V, T), (1, V), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh+i_v*B*H)*T*K, (T, K), (K, 1), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        # [BT, K]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gn = tl.load(p_gn, mask=mask, other=0).to(tl.float32)

        # [V, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [V, K]
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h * exp(b_gn)[None, :] + tl.dot(b_v, b_k.to(b_v.dtype), allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype), allow_tf32=False)
            b_h = b_h * exp(b_gn)[None, :] + tl.dot(b_v, b_k.to(b_v.dtype), allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # sync threads
    b_h = None
    tl.debug_barrier()
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)

    # cum = tl.zeros([BK], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * T*K, (K, T), (1, K), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = g + i_bh * T*K + (T - (i-1) * BT - 1) * K + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * T*V, (T, V), (V, 1), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * T*V, (T, V), (V, 1), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * T*K, (T, K),
                                 (K, 1), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * T*V, (T, V),
                                 (V, 1), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        # [K, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, K]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, V]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_db = tl.load(p_gn, mask=mask, other=0).to(tl.float32)

        # inter-chunk
        # [K, V]
        if CHECK and i == 1:
            b_dk = tl.trans(tl.dot(b_dh.to(b_v.dtype), tl.trans(b_v), allow_tf32=False))
            b_dv = tl.dot((b_k).to(b_v.dtype), b_dh.to(b_v.dtype), allow_tf32=False)
            b_dh = b_dh * exp(b_db)[:, None] + tl.dot(b_q.to(b_do.dtype), b_do, allow_tf32=False)
        else:
            b_dk = tl.trans(tl.dot(b_dh.to(b_v.dtype), tl.trans(b_v), allow_tf32=False))
            b_dv = tl.dot((b_k).to(b_v.dtype), b_dh.to(b_v.dtype), allow_tf32=False)
            b_dh = b_dh * exp(b_db)[:, None] + tl.dot(b_q.to(b_do.dtype), b_do, allow_tf32=False)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def fwd_inner_chunk(
    q, k, g, A,
    scale,  # K ** -0.5
    B: tl.constexpr,  # B
    H: tl.constexpr,  # H
    T,  # T
    K: tl.constexpr,  # K
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr  # BLOCK SIZE along the K dimension
):

    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < K
    o_i = tl.arange(0, BT)

    p_q = q + i_bh * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_gq = g + i_bh * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_A = A + (i_bh + (i_k * B * H)) * (tl.cdiv(T, BT) * BT * BT) + i_t * BT * BT + tl.arange(0, BT)

    for i in range(BT):
        b_q = tl.load(p_q, mask=mask, other=0) * scale
        b_gq = tl.load(p_gq, mask=mask, other=0).to(tl.float32)
        s = b_q[None, :] * b_k * safe_exp(b_gq[None, :] - b_g)
        score = tl.sum(s, axis=1)
        score = tl.where(o_i <= i, score, 0)
        tl.store(p_A, score.to(p_A.dtype.element_ty))
        p_q += K
        p_gq += K
        p_A += BT


@triton.jit
def bwd_inner_chunk(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    T,  # T
    K: tl.constexpr,  # K
    # clamp_min, # minimum log value of the gate for numerical stability. default: -5
    BT: tl.constexpr,  # BLOCK SIZE along the sequence dimension, a.k.a. chunk size
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_k = tl.make_block_ptr(k + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_g = tl.make_block_ptr(g + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)

    mask = (i_k * BK + tl.arange(0, BK)) < K
    o_i = tl.arange(0, BT)

    p_q = q + i_bh * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_dq = dq + (i_bh) * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_gq = g + i_bh * T*K + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_dA = dA + i_bh * (tl.cdiv(T, BT) * BT * BT) + i_t * BT * BT + tl.arange(0, BT)

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)

    for i in range(BT):
        b_q = tl.load(p_q, mask=mask, other=0)
        b_gq = tl.load(p_gq, mask=mask, other=0).to(tl.float32)
        score = safe_exp(b_gq[None, :] - b_g)
        score = tl.where(o_i[:, None] <= i, score, 0)
        b_dA = tl.load(p_dA)
        b_dA = tl.where(o_i <= i, b_dA, 0)
        b_dk += (b_dA[:, None] * score * b_q[None, :])
        b_dq = tl.sum(b_dA[:, None] * score * b_k, axis=0)
        tl.store(p_dq, b_dq, mask=mask)
        p_q += K
        p_dq += K
        p_gq += K
        p_dA += BT

    p_dk = tl.make_block_ptr(dk + i_bh * T*K, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk.to(dk.dtype.element_ty), boundary_check=(0, 1))


class FusedChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, scale, initial_state, output_final_state):
        ctx.g_dtype = g.dtype
        ctx.scale = scale
        B, H, T, K, V = *k.shape, v.shape[-1]
        BT = 16  # chunk_size
        BK, BV = min(K, 64), min(V, 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 2

        g_org = g
        # cumulative decay should be in float32, otherwise the err will be accumulated and amplified.
        g = chunk_local_cumsum(g_org, chunk_size=BT)
        o = q.new_empty(NK, B, H, T, V)
        q_g = torch.empty_like(q)
        k_g = torch.empty_like(k)

        grid = (NK, triton.cdiv(T, BT), B * H)
        prepare_qg_kg[grid](
            q,
            k,
            g,
            q_g,
            k_g,
            scale,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_warps=1
        )

        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float, requires_grad=False)
        else:
            final_state = None
        # the bug still exists even for Triton 2.2 on H100 GPUs
        # so we always enable initial checks
        CHECK = True
        if version.parse(triton.__version__) < version.parse('2.2.0'):
            import warnings
            warnings.warn(
                "Triton<2.2.0 detected for running this kernel, "
                "which is known to have some weird compiler issues (refer to https://github.com/openai/triton/issues/2852) "
                "that lead to significant precision loss. "
                "We've add some initial condition checks to resolve this, sadly at the sacrifice of the speed. "
                "For optimal performance, it is recommended to install Triton>=2.2.0 (if possible)."
            )
            CHECK = True

        grid = (NV, NK, B * H)
        fused_chunk_gla_fwd_kernel[grid](
            q_g, k_g, v, g, o, initial_state, final_state,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state,
            CHECK=CHECK,
            num_warps=num_warps,
            num_stages=num_stages
        )

        o = o.sum(0)

        # intra-chunk
        chunk_size = 16
        num_chunk = T // chunk_size
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=num_chunk)
        BK = min(K, 64)
        NK = triton.cdiv(K, BK)
        A = q.new_empty(NK, B, H, triton.cdiv(T, BT), BT, BT)
        grid = (NK, triton.cdiv(T, BT), B * H)
        fwd_inner_chunk[grid](
            q, k, g, A,
            scale,
            B=B,
            H=H,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_stages=3,
            num_warps=4
        )
        A = A.sum(0)
        o2 = A @ v2
        o2 = rearrange(o2, 'b h n c d -> b h (n c) d')
        # combine inner and inter
        o.add_(o2)
        ctx.save_for_backward(q, k, v, g_org, A, initial_state)
        ctx.CHECK = CHECK
        return o.to(v), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht=None):
        q, k, v, g_org, A, initial_state = ctx.saved_tensors
        B, H, T, K, V = *k.shape, v.shape[-1]
        scale = ctx.scale

        # recomputation
        # inter-chunk
        BT = 16  # chunk_size
        g = chunk_local_cumsum(g_org, chunk_size=BT)
        BK, BV = min(K, 64), min(V, 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        q_g = torch.empty_like(q)
        k_g = torch.empty_like(k)
        grid = (NK, triton.cdiv(T, BT), B * H)
        prepare_qg_kg[grid](
            q,
            k,
            g,
            q_g,
            k_g,
            scale,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_warps=1
        )

        BK, BV = min(max(triton.next_power_of_2(K), 16), 64), min(max(triton.next_power_of_2(V), 16), 64)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 2
        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)

        grid = (NV, NK, B * H)

        fused_chunk_gla_bwd_kernel[grid](
            q_g,
            k_g,
            v,
            g,
            do,
            dq,
            dk,
            dv,
            initial_state,
            scale,
            T=T,
            B=B,
            H=H,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            CHECK=ctx.CHECK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)

        # intra chunk
        NT = T // BT
        v2 = rearrange(v, 'b h (n c) d -> b h n c d', n=NT)
        do2 = rearrange(do, 'b h (n c) d -> b h n c d', n=NT)
        dA2 = (do2 @ v2.transpose(-2, -1)) * scale
        dv2 = A.transpose(-1, -2) @ do2
        dv2 = rearrange(dv2, 'b h n c d -> b h (n c) d', n=NT)

        BK = 16
        NK = triton.cdiv(K, BK)
        dk2 = torch.empty_like(k)
        dq2 = torch.empty_like(q)

        grid = (NK, NT, B * H)
        bwd_inner_chunk[grid](
            q, k, g,
            dA2,
            dq2,
            dk2,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_warps=1,
            num_stages=3
        )

        BK = min(max(triton.next_power_of_2(K), 16), 32)
        NK = triton.cdiv(K, BK)
        dg = torch.empty_like(g, dtype=torch.float32)
        grid = (NK, triton.cdiv(T, BT), B * H)
        bwd_decay_global_cumsum[grid](
            dq2,
            dq,
            dk2,
            dk,
            q,
            k,
            g,
            dg,
            T=T,
            K=K,
            BT=BT,
            BK=BK,
            num_warps=1,
            num_stages=1
        )
        dg = rearrange(dg, 'b h (n c) d -> b h n c d', c=BT)

        def rev_cumsum_exclusive(x):
            cumsum_x = x.cumsum(-2)
            rev_cumsum_x = cumsum_x[..., -1, None, :] - cumsum_x
            return rev_cumsum_x

        rev_cumsum_dg = rev_cumsum_exclusive(dg[..., 0, :])
        dg.add_(rev_cumsum_dg.unsqueeze(-2))
        dv.add_(dv2)
        dg = rearrange(dg, 'b h n c d -> b h (n c) d')

        return dq.to(q), dk.to(k), dv.to(v), dg.to(ctx.g_dtype), None, None, None


def ceildiv(a, b):
    return -(a // -b)


def pad(x, chunk_size=16):
    T = x.shape[-2]
    padded_seq_len = ceildiv(T, chunk_size) * chunk_size
    if x.shape[-2] % chunk_size != 0:
        x = F.pad(x, (0, 0, 0, padded_seq_len - T))
    return x


def fused_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: int = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, g = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, g))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if scale == -1:
        scale = q.shape[-1] ** -0.5
    seq_len = q.shape[-2]
    q, k, v, g = map(lambda x: pad(x), [q, k, v, g])
    o, final_state = FusedChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state)
    o = o[..., :seq_len, :].contiguous()
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, final_state









# ================================

# -*- coding: utf-8 -*-

import torch
import triton
from torch.nn import functional as F

# from fla.ops.gla import fused_chunk_gla
# from fla.ops.retention import chunk_retention, parallel_retention
# from fla.ops.retention.naive import naive_retention


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['fused_chunk_gla', 'fused_chunk_gla_bwd'],
        # label name for the lines
        line_names=['fused_chunk_gla', 'fused_chunk_gla_bwd', ],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    # from fla.utils import device
    dtype = torch.bfloat16
#    dtype = torch.float32
    requires_grad = True
    B, H, D = 16, 8, 128


    # if provider in ("fused_chunk_gla", "fused_chunk_gla_bwd"):

    q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    g = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)


    do = torch.ones_like(q, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0

    if provider == 'fused_chunk_gla':
        results = triton.testing.do_bench(lambda: fused_chunk_gla(q, k, v, g), quantiles=quantiles)
    elif provider == 'fused_chunk_gla_bwd':
        results = triton.testing.do_bench(lambda: fused_chunk_gla(q, k, v, g)[0].backward(do), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)
