# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/modules/token_shift.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/modules/token_shift.py
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

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not
    None, 'USE_INITIAL_STATE': lambda args: args['cache'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [1, 
    2, 3]], key=['BD', 'NB'], **autotune_cache_kwargs)
@triton.jit
def token_shift_fwd_kernel_long(x, y, cu_seqlens, chunk_indices, cache,
    cache_out, T, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr, NB:
    tl.constexpr, IS_VARLEN: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        t_start = i_t * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        i_n = i_b
        bos, eos = i_b * T, (i_b + 1) * T
        t_start = i_t * BT
        t_end = tl.minimum(t_start + BT, T)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    for t in range(t_start, t_end):
        global_t = bos + t
        offset = global_t * D + o_d
        b_x = tl.load(x + offset, mask=m_d)
        is_first = global_t == bos
        if is_first:
            if USE_INITIAL_STATE:
                cache_off = i_n * D + o_d if IS_VARLEN else i_b * D + o_d
                b_cache = tl.load(cache + cache_off, mask=m_d)
                delta = b_cache - b_x
            else:
                delta = -b_x
        else:
            prev_off = offset - D
            b_prev = tl.load(x + prev_off, mask=m_d)
            delta = b_prev - b_x
        tl.store(y + offset, delta, mask=m_d)
        if STORE_FINAL_STATE:
            if global_t == eos - 1:
                cache_out_off = i_n * D + o_d if IS_VARLEN else i_b * D + o_d
                tl.store(cache_out + cache_out_off, b_x, mask=m_d)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not
    None, 'USE_INITIAL_STATE': lambda args: args['cache'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [1, 
    2, 3]], key=['BD'], **autotune_cache_kwargs)
@triton.jit
def token_shift_fwd_kernel_short(x, y, cu_seqlens, cache, cache_out, T, D:
    tl.constexpr, BD: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
    IS_DECODE: tl.constexpr):
    i_b, i_t = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        g_t = i_t + bos
        if g_t >= eos:
            return
        is_first_pos = i_t == 0
        is_last_pos = g_t == eos - 1
    else:
        g_t = i_t
        is_first_pos = g_t == 0
        is_last_pos = g_t == T - 1
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    if IS_VARLEN:
        base_offset = g_t * D + o_d
    else:
        base_offset = i_b * T * D + g_t * D + o_d
    b_x = tl.load(x + base_offset, mask=m_d)
    if IS_VARLEN:
        cache_offset = i_n * D + o_d
    else:
        cache_offset = i_b * D + o_d
    if IS_DECODE and USE_INITIAL_STATE:
        b_cache = tl.load(cache + cache_offset, mask=m_d)
        delta = b_cache - b_x
        tl.store(y + base_offset, delta, mask=m_d)
        if STORE_FINAL_STATE:
            tl.store(cache_out + cache_offset, b_x, mask=m_d)
        return
    if is_first_pos:
        if USE_INITIAL_STATE:
            b_cache = tl.load(cache + cache_offset, mask=m_d)
            delta = b_cache - b_x
            tl.store(y + base_offset, delta, mask=m_d)
        else:
            tl.store(y + base_offset, -b_x, mask=m_d)
        return
    if IS_VARLEN:
        prev_offset = (g_t - 1) * D + o_d
    else:
        prev_offset = i_b * T * D + (g_t - 1) * D + o_d
    prev_values = tl.load(x + prev_offset, mask=m_d)
    delta = prev_values - b_x
    tl.store(y + base_offset, delta, mask=m_d)
    if STORE_FINAL_STATE:
        if is_last_pos:
            tl.store(cache_out + cache_offset, b_x, mask=m_d)


@tensor_cache
def prepare_maxlens(cu_seqlens: torch.LongTensor) ->int:
    return torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item()


def token_shift_fwd(x: torch.Tensor, cu_seqlens: Optional[torch.Tensor]=
    None, cache: Optional[torch.Tensor]=None, output_cache: bool=False
    ) ->torch.Tensor:
    B, T, D = x.shape
    y = torch.empty_like(x)
    use_short_kernel = T <= 4096
    if cu_seqlens is not None:
        T = prepare_maxlens(cu_seqlens)
        N = len(cu_seqlens) - 1
    else:
        N = B
    if output_cache:
        cache_out = torch.empty((N, D), device=x.device, dtype=x.dtype)
    else:
        cache_out = None
    if use_short_kernel:
        if cu_seqlens is not None:
            N = len(cu_seqlens) - 1
        else:
            N = B
        BD = triton.next_power_of_2(D)
        grid = N, T
        IS_DECODE = T == 1 or B == 1 and T == N
        token_shift_fwd_kernel_short[grid](x=x, y=y, cu_seqlens=cu_seqlens,
            cache=cache, cache_out=cache_out, T=T, D=D, BD=BD,
            STORE_FINAL_STATE=output_cache, IS_DECODE=IS_DECODE)
    else:
        BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B * T),
            get_multiprocessor_count(x.device.index))))
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT
            ) if cu_seqlens is not None else None
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,
            BT)
        BD = triton.next_power_of_2(D)
        NB = triton.cdiv(B * T, 1024)

        def grid(meta):
            return triton.cdiv(D, meta['BD']), NT, N
        token_shift_fwd_kernel_long[grid](x, y, cu_seqlens, chunk_indices,
            cache, cache_out, T, D=D, BD=BD, BT=BT, NB=NB,
            STORE_FINAL_STATE=output_cache)
    return y, N, T, use_short_kernel, cache_out


# Forward method (kernel launch code)
@input_guard
def _TokenShift_forward(ctx, x: torch.Tensor, cu_seqlens: Optional[torch.
    Tensor]=None, cache: Optional[torch.Tensor]=None, output_cache: bool=False
    ):
    output, N, T, use_short_kernel, cache_out = token_shift_fwd(x,
        cu_seqlens, cache, output_cache)
    ctx.cu_seqlens = cu_seqlens
    ctx.N = N
    ctx.T = T
    ctx.use_short_kernel = use_short_kernel
    ctx.has_cache = cache is not None
    return output, cache_out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not
    None, 'USE_INITIAL_STATE': lambda args: args['grad_cache_out'] is not
    None, 'HAS_DCACHE': lambda args: args['grad_cache_in'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [1, 
    2, 3]], key=['BD', 'NB'], **autotune_cache_kwargs)
@triton.jit
def token_shift_bwd_kernel_long(dx, dy, cu_seqlens, chunk_indices,
    grad_cache_in, grad_cache_out, T, D: tl.constexpr, BD: tl.constexpr, BT:
    tl.constexpr, NB: tl.constexpr, IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, HAS_DCACHE: tl.constexpr):
    i_d, i_t_blk, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t_blk = tl.load(chunk_indices + i_t_blk * 2).to(tl.int32
            ), tl.load(chunk_indices + i_t_blk * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, eos - bos)
    else:
        bos, eos = i_b * T, (i_b + 1) * T
        t_start = i_t_blk * BT
        t_end = tl.minimum(t_start + BT, T)
    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    cache_off = i_n * D + o_d if IS_VARLEN else i_b * D + o_d
    for t in range(t_start, t_end):
        global_t = bos + t
        offset = global_t * D + o_d
        b_dy = tl.load(dy + offset, mask=m_d)
        if global_t == eos - 1:
            if HAS_DCACHE:
                b_dy_cache = tl.load(grad_cache_in + cache_off, mask=m_d)
                b_dx = -b_dy + b_dy_cache
            else:
                b_dx = -b_dy
        else:
            next_off = offset + D
            b_dx = -b_dy + tl.load(dy + next_off, mask=m_d)
        tl.store(dx + offset, b_dx, mask=m_d)
        if USE_INITIAL_STATE:
            if global_t == bos:
                tl.store(grad_cache_out + cache_off, b_dy, mask=m_d)


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not
    None, 'USE_INITIAL_STATE': lambda args: args['grad_cache_out'] is not
    None, 'HAS_DCACHE': lambda args: args['grad_cache_in'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [1, 
    2, 3]], key=['BD'], **autotune_cache_kwargs)
@triton.jit
def token_shift_bwd_kernel_short(dx, dy, cu_seqlens, grad_cache_in,
    grad_cache_out, T, D: tl.constexpr, BD: tl.constexpr, IS_VARLEN: tl.
    constexpr, USE_INITIAL_STATE: tl.constexpr, HAS_DCACHE: tl.constexpr):
    i_b, i_t = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        i_n = i_b
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        g_t = i_t + bos
        if g_t >= eos:
            return
        is_first_pos = g_t == bos
        is_last_pos = g_t == eos - 1
    else:
        g_t = i_t
        is_first_pos = g_t == 0
        is_last_pos = g_t == T - 1
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    if IS_VARLEN:
        base_offset = g_t * D + o_d
        cache_off = i_n * D + o_d
    else:
        base_offset = i_b * T * D + g_t * D + o_d
        cache_off = i_b * D + o_d
    b_dy = tl.load(dy + base_offset, mask=m_d)
    if is_last_pos:
        if HAS_DCACHE:
            b_dy_cache = tl.load(grad_cache_in + cache_off, mask=m_d)
            b_dx = -b_dy + b_dy_cache
        else:
            b_dx = -b_dy
    else:
        if IS_VARLEN:
            next_offset = (g_t + 1) * D + o_d
        else:
            next_offset = i_b * T * D + (g_t + 1) * D + o_d
        b_dx = -b_dy + tl.load(dy + next_offset, mask=m_d)
    tl.store(dx + base_offset, b_dx, mask=m_d)
    if USE_INITIAL_STATE:
        if is_first_pos:
            tl.store(grad_cache_out + cache_off, b_dy, mask=m_d)


def token_shift_bwd(dy: torch.Tensor, N: int, T: int, dcache: Optional[
    torch.Tensor]=None, cu_seqlens: Optional[torch.Tensor]=None,
    use_short_kernel: bool=True, has_init_cache: bool=False) ->torch.Tensor:
    D = dy.shape[2]
    BD = triton.next_power_of_2(D)
    dx = torch.empty_like(dy)
    if has_init_cache:
        grad_cache_out = torch.empty((N, D), device=dy.device, dtype=dy.dtype)
    else:
        grad_cache_out = None
    if use_short_kernel:
        grid = N, T
        token_shift_bwd_kernel_short[grid](dy=dy, dx=dx, cu_seqlens=
            cu_seqlens, grad_cache_in=dcache, grad_cache_out=grad_cache_out,
            T=T, D=D, BD=BD)
    else:
        BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, dy.numel() //
            D), get_multiprocessor_count(dy.device.index))))
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT
            ) if cu_seqlens is not None else None
        NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,
            BT)
        NB = triton.cdiv(N * dy.shape[1], 1024)
        BD = triton.next_power_of_2(D)

        def grid(meta):
            return triton.cdiv(D, meta['BD']), NT, N
        token_shift_bwd_kernel_long[grid](dx, dy, cu_seqlens, chunk_indices,
            dcache, grad_cache_out, T, D=D, BD=BD, BT=BT, NB=NB)
    return dx, grad_cache_out


# Backward method (kernel launch code)
@input_guard
def _TokenShift_backward(ctx, dy: torch.Tensor, dcache: Optional[torch.
    Tensor]=None):
    dx, grad_cache = token_shift_bwd(dy, ctx.N, ctx.T, dcache, ctx.
        cu_seqlens, ctx.use_short_kernel, ctx.has_cache)
    return dx, None, grad_cache, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class TokenShift(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x: torch.Tensor, cu_seqlens: Optional[torch.Tensor]=
        None, cache: Optional[torch.Tensor]=None, output_cache: bool=False):
        output, N, T, use_short_kernel, cache_out = token_shift_fwd(x,
            cu_seqlens, cache, output_cache)
        ctx.cu_seqlens = cu_seqlens
        ctx.N = N
        ctx.T = T
        ctx.use_short_kernel = use_short_kernel
        ctx.has_cache = cache is not None
        return output, cache_out

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor, dcache: Optional[torch.Tensor]=None):
        dx, grad_cache = token_shift_bwd(dy, ctx.N, ctx.T, dcache, ctx.
            cu_seqlens, ctx.use_short_kernel, ctx.has_cache)
        return dx, None, grad_cache, None
