# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/delta_rule/chunk.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/delta_rule/chunk.py
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

@classmethod
def get_shared_memory(cls, arch: str) ->int:
    try:
        return cls[arch.upper()].value
    except KeyError:
        return cls.DEFAULT.value


def _cpu_device_warning():
    warnings.warn(
        'Triton is not supported on current platform, roll back to CPU.',
        stacklevel=1)


@lru_cache(maxsize=None)
def check_shared_mem(arch: str='none', tensor_idx: int=0) ->bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


def get_all_max_shared_mem():
    try:
        return [triton.runtime.driver.active.utils.get_device_properties(i)
            ['max_shared_mem'] for i in range(device_torch_lib.device_count())]
    except BaseException:
        _cpu_device_warning()
        return [-1]


def chunk_gated_delta_rule_fwd_h(k: torch.Tensor, w: torch.Tensor, u: torch
    .Tensor, g: Optional[torch.Tensor]=None, gk: Optional[torch.Tensor]=
    None, initial_state: Optional[torch.Tensor]=None, output_final_state:
    bool=False, chunk_size: int=64, save_new_value: bool=True, cu_seqlens:
    Optional[torch.LongTensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices
            ), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, 'current kernel does not support head dimension larger than 256.'
    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32
        ) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return triton.cdiv(V, meta['BV']), N * H
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](k=k, v=u, w=w,
        v_new=v_new, g=g, gk=gk, h=h, h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets, T=T, H=H, K=K,
        V=V, BT=BT)
    return h, v_new, final_state


@triton.heuristics({'USE_G': lambda args: args['g'] is not None, 'USE_GK': 
    lambda args: args['gk'] is not None, 'USE_INITIAL_STATE': lambda args: 
    args['h0'] is not None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not
    None, 'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BV': BV}, num_warps=num_warps,
    num_stages=num_stages) for num_warps in [2, 4] for num_stages in [2, 3,
    4] for BV in [32, 64]], key=['H', 'K', 'V', 'BT'], use_cuda_graph=
    use_cuda_graph, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(k, v, w, v_new, g, gk, h,
    h0, ht, cu_seqlens, chunk_offsets, T, H: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, USE_G: tl.
    constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)
    h += (boh * H + i_h) * K * V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64,
            BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV),
                (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV),
                (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV),
                (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (0, 
            i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (
                64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(
                0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (
                128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(
                0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * stride_h, (K, V), (V, 1), (
                192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(
                0, 1))
        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (
            BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64
                ), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 
                128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 
                192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (stride_v, 1), (i_t * BT,
                i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = i_t * BT + tl.arange(0, BT) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t *
                BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K +
                o_k1, mask=o_k1 < K, other=0.0)
            b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h *
                    K + o_k2, mask=o_k2 < K, other=0.0)
                b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H * K + i_h *
                    K + o_k3, mask=o_k3 < K, other=0.0)
                b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H * K + i_h *
                    K + o_k4, mask=o_k4 < K, other=0.0)
                b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (
            64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            p_g = tl.make_block_ptr(gk + (bos * H + i_h) * K, (K, T), (1, H *
                K), (0, i_t * BT), (64, BT), (0, 1))
            b_k = (b_k * exp(b_gk_last1[:, None] - tl.load(p_g,
                boundary_check=(0, 1)))).to(b_k.dtype)
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT
                ), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_g = tl.make_block_ptr(gk + (bos * H + i_h) * K, (K, T), (
                    1, H * K), (64, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * exp(b_gk_last2[:, None] - tl.load(p_g,
                    boundary_check=(0, 1)))).to(b_k.dtype)
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t *
                BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_g = tl.make_block_ptr(gk + (bos * H + i_h) * K, (K, T), (
                    1, H * K), (128, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * exp(b_gk_last3[:, None] - tl.load(p_g,
                    boundary_check=(0, 1)))).to(b_k.dtype)
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t *
                BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_g = tl.make_block_ptr(gk + (bos * H + i_h) * K, (K, T), (
                    1, H * K), (192, i_t * BT), (64, BT), (0, 1))
                b_k = (b_k * exp(b_gk_last4[:, None] - tl.load(p_g,
                    boundary_check=(0, 1)))).to(b_k.dtype)
            b_h4 += tl.dot(b_k, b_v)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV
            ), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (
                64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(
                0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (
                64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(
                0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (
                64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(
                0, 1))


def recompute_w_u_fwd(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor,
    A: torch.Tensor, cu_seqlens: Optional[torch.LongTensor]) ->Tuple[torch.
    Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    u = torch.empty_like(v)
    w = torch.empty_like(k)
    recompute_w_u_fwd_kernel[NT, B * H](k, v, beta, w, u, A, cu_seqlens=
        cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, K=K, V=V, BT=BT,
        BK=BK, BV=BV)
    return w, u


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in [2, 4, 8] for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'], **autotune_cache_kwargs
    )
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(k, v, beta, w, u, A, cu_seqlens, chunk_indices,
    T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,
        ), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A.to(b_vb.dtype), b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A.to(b_kb.dtype), b_kb, allow_tf32=False)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int
    ) ->torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(
        cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens
        )


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int
    ) ->torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(
        cu_seqlens), chunk_size)]).cumsum(-1)


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) ->torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps) for
    num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST], key=['D', 'NB'], **
    autotune_cache_kwargs)
@triton.jit
def l2norm_fwd_kernel(x, y, rstd, eps, T: tl.constexpr, D: tl.constexpr, BD:
    tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in NUM_WARPS_AUTOTUNE], key=['D'], **autotune_cache_kwargs)
@triton.jit
def l2norm_fwd_kernel1(x, y, rstd, eps, D, BD: tl.constexpr):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None, 'IS_VARLEN': 
    lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': 128, 'BV': 128}, num_warps=8,
    num_stages=3), triton.Config({'BK': 64, 'BV': 64}, num_warps=4,
    num_stages=3), triton.Config({'BK': 32, 'BV': 32}, num_warps=2,
    num_stages=3)], key=['H', 'K', 'V', 'BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(q, k, v, h, g, g_gamma, o, cu_seqlens, chunk_indices,
    scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.
    constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
            (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT),
            (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (
            BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (
        BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (
        BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK}, num_warps=num_warps,
    num_stages=num_stages) for BK in [32, 64, 128] for num_warps in [2, 4, 
    8] for num_stages in [2, 3, 4]], key=['H', 'K', 'BT', 'IS_VARLEN'], **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel(k, g, beta, A, cu_seqlens,
    chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr, IS_VARLEN: tl.constexpr, USE_G: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,
        ), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))
    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,),
            (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp(b_g_diff)
    b_A *= b_beta[:, None]
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1),
        (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK}, num_warps=num_warps,
    num_stages=num_stages) for BK in [32, 64] for num_warps in [1, 2, 4, 8] for
    num_stages in [2, 3, 4]], key=['BC'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter(k, g, beta, A,
    cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.
    constexpr, BC: tl.constexpr, BK: tl.constexpr, NC: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT +
        i_i * BC,), (BC,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT + i_i * BC,
            i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H * K, 1), (i_t * BT + i_i * BC,
            i_k * BK), (BC, BK), (1, 0))
        b_kt = tl.make_block_ptr(k, (K, T), (1, H * K), (i_k * BK, i_t * BT +
            i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g, (K, T), (1, H * K), (i_k * BK, i_t * BT +
            i_j * BC), (BK, BC), (0, 1))
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        b_gn = tl.load(g + (i_t * BT + i_i * BC) * H * K + o_k, mask=m_k,
            other=0)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1)) * exp(b_g - b_gn[None, :])
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kt = tl.load(b_kt, boundary_check=(0, 1)) * exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_k, b_kt)
    b_A *= b_beta[:, None]
    p_A = tl.make_block_ptr(A, (T, BT), (H * BT, 1), (i_t * BT + i_i * BC, 
        i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({},
    num_warps=2), triton.Config({}, num_warps=4), triton.Config({},
    num_warps=8)], key=['BK', 'BT'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra(k, g, beta, A,
    cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.
    constexpr, BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if i_t * BT + i_i * BC >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = i_t * BT + i_i * BC + o_i < T
    o_A = (bos + i_t * BT + i_i * BC + o_i) * H * BT + i_h * BT + i_i * BC
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_beta = beta + (bos + i_t * BT + i_i * BC + o_i) * H + i_h
    b_k = tl.load(p_k, boundary_check=(0, 1)) * tl.load(p_beta, mask=m_A,
        other=0)[:, None]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    p_kt = k + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    p_gk = g + (bos + i_t * BT + i_i * BC) * H * K + i_h * K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_kt = tl.load(p_kt, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_k * b_kt[None, :] * exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_kt += H * K
        p_gk += H * K


def l2norm_fwd(x: torch.Tensor, eps: float=1e-06, output_dtype: Optional[
    torch.dtype]=None):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return triton.cdiv(T, meta['BT']),
        l2norm_fwd_kernel[grid](x=x, y=y, rstd=rstd, eps=eps, T=T, D=D, BD=
            BD, NB=NB)
    else:
        l2norm_fwd_kernel1[T,](x=x, y=y, rstd=rstd, eps=eps, D=D, BD=BD)
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def chunk_fwd_o(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch
    .Tensor, g: Optional[torch.Tensor]=None, g_gamma: Optional[torch.Tensor
    ]=None, scale: Optional[float]=None, cu_seqlens: Optional[torch.
    LongTensor]=None, chunk_size: int=64) ->torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o = torch.empty_like(v)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_fwd_kernel_o[grid](q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma, o=o,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale, T=
        T, H=H, K=K, V=V, BT=BT)
    return o


def chunk_scaled_dot_kkt_fwd(k: torch.Tensor, g: Optional[torch.Tensor]=
    None, gk: Optional[torch.Tensor]=None, beta: Optional[torch.Tensor]=
    None, cu_seqlens: Optional[torch.LongTensor]=None, chunk_size: int=64,
    output_dtype: torch.dtype=torch.float32) ->torch.Tensor:
    """
    Compute beta * K * K^T.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        gk (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths of the input tensor.
            Default: None
        chunk_size (int):
            The chunk size. Default: 64.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float32`

    Returns:
        beta * K * K^T of shape `[B, T, H, BT]` where `BT` is the chunk size.
    """
    B, T, H, K = k.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if gk is None:
        A = torch.empty(B, T, H, BT, device=k.device, dtype=output_dtype)
        chunk_scaled_dot_kkt_fwd_kernel[NT, B * H](k=k, g=g, beta=beta, A=A,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, K
            =K, BT=BT)
        return A
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)
    BK = max(triton.next_power_of_2(K), 16)
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    grid = NT, NC * NC, B * H
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](k=k, g=gk, beta=
        beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
        H=H, K=K, BT=BT, BC=BC, NC=NC)
    grid = NT, NC, B * H
    chunk_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](k=k, g=gk, beta=
        beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T,
        H=H, K=K, BT=BT, BC=BC, BK=BK)
    return A


def chunk_delta_rule_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    beta: torch.Tensor, scale: float, initial_state: torch.Tensor,
    output_final_state: bool, cu_seqlens: Optional[torch.LongTensor]=None):
    w, u, A = prepare_wy_repr_fwd(k=k, v=v, beta=beta, cu_seqlens=cu_seqlens)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=
        None, initial_state=initial_state, output_final_state=
        output_final_state, cu_seqlens=cu_seqlens)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens
        =cu_seqlens)
    return o, A, final_state


def prepare_wy_repr_fwd(k: torch.Tensor, v: torch.Tensor, beta: torch.
    Tensor, cu_seqlens: Optional[torch.LongTensor]) ->Tuple[torch.Tensor,
    torch.Tensor, torch.Tensor]:
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, cu_seqlens=cu_seqlens,
        chunk_size=64, output_dtype=torch.float32)
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=cu_seqlens)
    return w, u, A


@input_guard
def solve_tril(A: torch.Tensor, cu_seqlens: Optional[torch.Tensor]=None,
    output_dtype: torch.dtype=torch.float) ->torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    Ai = torch.zeros_like(A, dtype=output_dtype)
    if BT == 16:
        merge_fn = solve_tril_16x16_kernel
    elif BT == 32:
        merge_fn = merge_16x16_to_32x32_inverse_kernel
    elif BT == 64:
        merge_fn = merge_16x16_to_64x64_inverse_kernel
    merge_fn[NT, B * H](A=A, Ai=Ai, cu_seqlens=cu_seqlens, chunk_indices=
        chunk_indices, T=T, H=H, BT=BT, USE_TMA=is_tma_supported,
        DOT_PRECISION=FLA_TRIL_PRECISION)
    return Ai


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _ChunkDeltaRuleFunction_forward(ctx, q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, beta: torch.Tensor, scale: float, initial_state: torch
    .Tensor, output_final_state: bool, use_qk_l2norm_in_kernel: bool=False,
    cu_seqlens: Optional[torch.LongTensor]=None):
    if use_qk_l2norm_in_kernel:
        q, q_rstd = l2norm_fwd(q)
        k, k_rstd = l2norm_fwd(k)
    else:
        q_rstd, k_rstd = None, None
    o, A, final_state = chunk_delta_rule_fwd(q=q, k=k, v=v, beta=beta,
        scale=scale, initial_state=initial_state, output_final_state=
        output_final_state, cu_seqlens=cu_seqlens)
    ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state)
    ctx.scale = scale
    ctx.cu_seqlens = cu_seqlens
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    return o.to(q.dtype), final_state


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BT': BT}, num_warps=num_warps) for
    num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST], key=['D', 'NB'], **
    autotune_cache_kwargs)
@triton.jit
def l2norm_bwd_kernel(y, rstd, dy, dx, eps, T: tl.constexpr, D: tl.
    constexpr, BD: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr):
    i_t = tl.program_id(0)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (
        1, 0))
    p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (
        1, 0))
    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None
        ] * b_y * b_rstd[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in NUM_WARPS_AUTOTUNE], key=['D'], **autotune_cache_kwargs)
@triton.jit
def l2norm_bwd_kernel1(y, rstd, dy, dx, eps, D, BD: tl.constexpr):
    i_t = tl.program_id(0)
    y += i_t * D
    dx += i_t * D
    dy += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_y = tl.load(y + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
    tl.store(dx + cols, b_dx, mask=mask)


@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BV': BV}, num_warps=num_warps,
    num_stages=num_stages) for num_warps in [2, 4] for num_stages in [4, 3,
    2] for BV in [64, 32]], key=['H', 'K', 'V', 'BT', 'BV', 'USE_G'],
    use_cuda_graph=use_cuda_graph, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(q, k, w, g, dht, dh0,
    do, dh, dv, dv2, cu_seqlens, chunk_offsets, scale, T, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)
    dh += (boh * H + i_h) * K * V
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        dh0 += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K * V
    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64,
            BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV),
                (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV),
                (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV),
                (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))
    for i_t in range(NT - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1), (0, 
            i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1)
            )
        if K > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1),
                (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty),
                boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1),
                (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty),
                boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t * stride_h, (K, V), (V, 1),
                (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty),
                boundary_check=(0, 1))
        if USE_G:
            last_idx = min((i_t + 1) * BT, T) - 1
            bg_last = tl.load(g + (bos + last_idx) * H + i_h)
            bg_last_exp = exp(bg_last)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t *
                BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_g_exp = exp(b_g)
        else:
            bg_last = None
            last_idx = None
            b_g = None
            b_g_exp = None
        p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (stride_v, 1), (i_t * BT, 
            i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (
            BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh1.to(b_k.dtype))
        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 64
                ), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))
        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 
                128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))
        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 
                192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))
        if USE_G:
            m_t = i_t * BT + tl.arange(0, BT) < T
            b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))
        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (0, i_t * BT), (
            64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (0, i_t * BT), (
            64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        b_q = (b_q * scale).to(b_q.dtype)
        b_dh1 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.
            dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (64, i_t * BT
                ), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (64, i_t * BT
                ), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh2 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(
                b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (128, i_t *
                BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (128, i_t *
                BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh3 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(
                b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_k), (192, i_t *
                BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, stride_k), (192, i_t *
                BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            b_q = (b_q * scale).to(b_q.dtype)
            b_dh4 += tl.dot(b_q, b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(
                b_w.dtype))
    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64,
            BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1)
            )
        if K > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV),
                (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty),
                boundary_check=(0, 1))
        if K > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV),
                (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty),
                boundary_check=(0, 1))
        if K > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV),
                (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty),
                boundary_check=(0, 1))


@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None, 'USE_DW': lambda
    args: args['dw'] is not None, 'IS_VARLEN': lambda args: args[
    'cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G', 'USE_G_GAMMA', 'USE_DW'],
    **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dqkwg(q, k, v, h, g, g_gamma, do, dh, dq, dk, dg, w,
    dv, dw, cu_seqlens, chunk_indices, scale, B: tl.constexpr, T, H: tl.
    constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, USE_G: tl.constexpr, USE_G_GAMMA: tl.
    constexpr, USE_DW: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
        all = B * T
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V
    dh += (i_tg * H + i_h).to(tl.int64) * K * V
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    if USE_DW:
        w += (bos * H + i_h) * K
        dw += (bos * H + i_h) * K
        dv += (bos * H + i_h) * V
    if USE_G:
        dg += i_k * all * H
        b_dg_last = tl.zeros([1], dtype=tl.float32) if USE_G else None
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_g_last = b_gamma * min(BT, T - i_t * BT)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32) if USE_DW else None
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_t * BT, i_v * BV),
            (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (
            BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK),
            (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        if USE_G:
            b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        if USE_DW:
            p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v *
                BV), (BT, BV), (1, 0))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))
    if USE_DW:
        p_dw = tl.make_block_ptr(dw, (T, K), (H * K, 1), (i_t * BT, i_k *
            BK), (BT, BK), (1, 0))
        tl.store(p_dw, -b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
    tl.debug_barrier()
    p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (
        BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (
        BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_dq = tl.make_block_ptr(dq, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
        (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
        (BT, BK), (1, 0))
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        g += bos * H + i_h
        dg += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_last = tl.load(g + (min(i_t * BT + BT, T) - 1) * H)
        b_dg_last *= exp(b_g_last)
        b_dq = b_dq * exp(b_g)[:, None] * scale
        b_dg += tl.sum(b_dq * b_q, axis=1)
        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_dg -= tl.sum(b_k * b_dk, axis=1)
        b_dg_last += tl.sum(b_dk * b_k)
        b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0
            ) * scale
        b_ds2 = b_ds * tl.dot(b_q, tl.trans(b_k))
        b_dg += tl.sum(b_ds2, axis=1)
        b_dg -= tl.sum(b_ds2, axis=0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        p_dg = tl.make_block_ptr(dg, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_dg = tl.where(o_t < min(i_t * BT + BT, T) - 1, b_dg, b_dg + b_dg_last
            )
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    elif USE_G_GAMMA:
        b_dq = b_dq * exp(b_g)[:, None] * scale
        b_dk = b_dk * tl.where(m_t, exp(-b_g + b_g_last), 0)[:, None]
        b_ds = tl.where(m_A, b_ds * exp(b_g[:, None] - b_g[None, :]), 0
            ) * scale
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q)
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    else:
        b_ds = tl.where(m_A, b_ds, 0)
        b_ds = b_ds.to(b_k.dtype)
        b_dq += tl.dot(b_ds, b_k)
        b_dk += tl.dot(tl.trans(b_ds), b_q) * scale
        b_dq *= scale
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None, 'IS_VARLEN': 
    lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'USE_G'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_bwd_kernel_dv_local(q, k, g, g_gamma, do, dv, cu_seqlens,
    chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_G:
    tl.constexpr, USE_G_GAMMA: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k, (T, K), (H * K, 1), (i_t * BT, i_k * BK),
            (BT, BK), (1, 0))
        p_q = tl.make_block_ptr(q, (K, T), (1, H * K), (i_k * BK, i_t * BT),
            (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q)
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    if USE_G:
        b_A = tl.where(m_A, b_A * exp(b_g[None, :] - b_g[:, None]) * scale, 0
            ).to(do.dtype.element_ty)
    else:
        b_A = tl.where(m_A, b_A * scale, 0).to(do.dtype.element_ty)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do, (T, V), (H * V, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H * V, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS for num_stages in [2, 3, 4]],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'], **autotune_cache_kwargs
    )
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(k, v, beta, A, dw, du, dk, dv, dbeta,
    cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,
        ), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H * BT),
        (0, i_t * BT), (BT, BT), (0, 1))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 
            1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos * H + i_h) * V, (T, V), (H * V, 
            1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)
        b_dv_beta = tl.dot(b_A, b_du, allow_tf32=False)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 
            1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos * H + i_h) * K, (T, K), (H * K, 
            1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_k_beta), allow_tf32=False)
        b_dk_beta = tl.dot(b_A, b_dw, allow_tf32=False)
        b_dk = b_dk_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :],
        b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], 
        -b_dA, 0).to(k.dtype.element_ty)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos * H + i_h) * K, (T, K), (H * K, 
            1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_k_beta = (b_k * b_beta[:, None]).to(b_k.dtype)
        b_dk_beta = tl.dot(b_dA, b_k, allow_tf32=False)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += tl.dot(tl.trans(b_dA), b_k_beta, allow_tf32=False)
        b_dk += b_dk_beta * b_beta[:, None]
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    p_dbeta = tl.make_block_ptr(dbeta + bos * H + i_h, (T,), (H,), (i_t *
        BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta.to(p_dbeta.dtype.element_ty), boundary_check=(0,)
        )


def l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps:
    float=1e-06):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return triton.cdiv(T, meta['BT']),
        l2norm_bwd_kernel[grid](y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, T=T,
            D=D, BD=BD, NB=NB)
    else:
        l2norm_bwd_kernel1[T,](y=y, rstd=rstd, dy=dy, dx=dx, eps=eps, D=D,
            BD=BD)
    return dx.view(y_shape_og)


def chunk_gated_delta_rule_bwd_dhu(q: torch.Tensor, k: torch.Tensor, w:
    torch.Tensor, g: torch.Tensor, h0: torch.Tensor, dht: Optional[torch.
    Tensor], do: torch.Tensor, dv: torch.Tensor, scale: float, cu_seqlens:
    Optional[torch.LongTensor]=None, chunk_size: int=64) ->Tuple[torch.
    Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    BT = 64
    assert K <= 256, 'current kernel does not support head dimension being larger than 256.'
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices
            ), prepare_chunk_offsets(cu_seqlens, BT)
    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), N * H
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](q=q, k=k, w=w, g
        =g, dht=dht, dh0=dh0, do=do, dh=dh, dv=dv, dv2=dv2, cu_seqlens=
        cu_seqlens, chunk_offsets=chunk_offsets, scale=scale, T=T, H=H, K=K,
        V=V, BT=BT)
    return dh, dh0, dv2


def chunk_bwd_dqkwg(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do:
    torch.Tensor, h: torch.Tensor, dh: torch.Tensor, g: Optional[torch.
    Tensor]=None, g_gamma: Optional[torch.Tensor]=None, dv: Optional[torch.
    Tensor]=None, w: Optional[torch.Tensor]=None, cu_seqlens: Optional[
    torch.LongTensor]=None, chunk_size: int=64, scale: float=1.0) ->Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device=g.device
        ) if g is not None else None
    dw = torch.empty_like(w) if w is not None else None
    grid = NK, NT, B * H
    chunk_bwd_kernel_dqkwg[grid](q=q, k=k, v=v, h=h, g=g, g_gamma=g_gamma,
        do=do, dh=dh, dv=dv, w=w, dw=dw, dq=dq, dk=dk, dg=dg, cu_seqlens=
        cu_seqlens, chunk_indices=chunk_indices, scale=scale, B=B, T=T, H=H,
        K=K, V=V, BT=BT, BK=BK, BV=BV)
    if dg is not None:
        dg = dg.sum(0)
    return dq, dk, dw, dg


def chunk_bwd_dv_local(q: torch.Tensor, k: torch.Tensor, do: torch.Tensor,
    g: Optional[torch.Tensor]=None, g_gamma: Optional[torch.Tensor]=None,
    scale: float=None, cu_seqlens: Optional[torch.LongTensor]=None,
    chunk_size: int=64) ->torch.Tensor:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dv = torch.empty_like(do)
    grid = NT, B * H
    chunk_bwd_kernel_dv_local[grid](q=q, k=k, g=g, g_gamma=g_gamma, do=do,
        dv=dv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=
        scale, T=T, H=H, K=K, V=V, BT=BT, BK=BK, BV=BV)
    return dv


def chunk_delta_rule_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    beta: torch.Tensor, A: torch.Tensor, scale: float, initial_state: torch
    .Tensor, do: torch.Tensor, dht: torch.Tensor, cu_seqlens: Optional[
    torch.LongTensor]=None):
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=cu_seqlens)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=None,
        initial_state=initial_state, output_final_state=False, cu_seqlens=
        cu_seqlens)
    dv = chunk_bwd_dv_local(q=q, k=k, do=do, g=None, scale=scale,
        cu_seqlens=cu_seqlens)
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(q=q, k=k, w=w, g=None, h0=
        initial_state, dht=dht, do=do, dv=dv, scale=scale, cu_seqlens=
        cu_seqlens)
    dq, dk, dw, _ = chunk_bwd_dqkwg(q=q, k=k, v=v_new, h=h, w=w, dv=dv, do=
        do, dh=dh, g=None, scale=scale, cu_seqlens=cu_seqlens)
    dk2, dv, db = prepare_wy_repr_bwd(k=k, v=v, beta=beta, A=A, dw=dw, du=
        dv, cu_seqlens=cu_seqlens)
    dk.add_(dk2)
    return dq, dk, dv, db, dh0


def prepare_wy_repr_bwd(k: torch.Tensor, v: torch.Tensor, beta: torch.
    Tensor, A: torch.Tensor, dw: torch.Tensor, du: torch.Tensor, cu_seqlens:
    Optional[torch.LongTensor]) ->Tuple[torch.Tensor, torch.Tensor, torch.
    Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)
    prepare_wy_repr_bwd_kernel[NT, B * H](k, v, beta, A, dw, du, dk, dv,
        dbeta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H,
        K=K, V=V, BT=BT, BK=BK, BV=BV)
    return dk, dv, dbeta


# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _ChunkDeltaRuleFunction_backward(ctx, do: torch.Tensor, dht: torch.Tensor):
    q, q_rstd, k, k_rstd, v, beta, A, initial_state = ctx.saved_tensors
    dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(q=q, k=k, v=v, beta=beta, A=
        A, scale=ctx.scale, initial_state=initial_state, do=do, dht=dht,
        cu_seqlens=ctx.cu_seqlens)
    if ctx.use_qk_l2norm_in_kernel:
        dq = l2norm_bwd(q, q_rstd, dq)
        dk = l2norm_bwd(k, k_rstd, dk)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype
        ), None, dh0, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ChunkDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        beta: torch.Tensor, scale: float, initial_state: torch.Tensor,
        output_final_state: bool, use_qk_l2norm_in_kernel: bool=False,
        cu_seqlens: Optional[torch.LongTensor]=None):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None
        o, A, final_state = chunk_delta_rule_fwd(q=q, k=k, v=v, beta=beta,
            scale=scale, initial_state=initial_state, output_final_state=
            output_final_state, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state)
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, q_rstd, k, k_rstd, v, beta, A, initial_state = ctx.saved_tensors
        dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(q=q, k=k, v=v, beta=beta,
            A=A, scale=ctx.scale, initial_state=initial_state, do=do, dht=
            dht, cu_seqlens=ctx.cu_seqlens)
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype
            ), None, dh0, None, None, None, None, None
