# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/generalized_delta_rule/dplr/chunk.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/generalized_delta_rule/dplr/chunk.py
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


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 
    3, 4]], key=['BK', 'BT'], use_cuda_graph=use_cuda_graph, **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_A_kernel_intra_sub_intra(q, k, a, b, gi, ge, qg, kg, ag,
    bg, Aqk, Aqb, Aab, Aak, cu_seqlens, chunk_indices, scale: tl.constexpr,
    T, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    BK: tl.constexpr, IS_VARLEN: tl.constexpr, GATHER_SUPPORTED: tl.constexpr):
    i_t, i_b, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    if i_t * BT >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = i_t * BT + tl.arange(0, BC) < T
    last_idx = min((i_t + 1) * BT, T) - 1
    o_A = (bos + i_t * BT + tl.arange(0, BC)) * H * BT + i_h * BT
    p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT, 0), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT, 0), (BC, BK), (1, 0))
    p_a = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT, 0), (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b + (bos * H + i_h) * K, (T, K), (H * K, 1), (
        i_t * BT, 0), (BC, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, 0), (BC, BK), (1, 0))
    p_ge = tl.make_block_ptr(ge + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, 0), (BC, BK), (1, 0))
    p_g_last = gi + (bos * H + i_h) * K + last_idx * H * K + tl.arange(0, BK)
    b_g_last = tl.load(p_g_last, mask=m_k, other=0)
    p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, 0), (BC, BK), (1, 0))
    p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, 0), (BC, BK), (1, 0))
    p_ag = tl.make_block_ptr(ag + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, 0), (BC, BK), (1, 0))
    p_bg = tl.make_block_ptr(bg + (bos * H + i_h) * K, (T, K), (H * K, 1),
        (i_t * BT, 0), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1)).to(tl.float32)
    b_ge = tl.load(p_ge, boundary_check=(0, 1)).to(tl.float32)
    g_exp = exp(b_gi)
    g_exp_inv = exp(-b_gi + b_g_last[None, :])
    b_qg = b_q * g_exp
    b_kg = b_k * g_exp_inv
    b_bg = b_b * g_exp_inv
    b_ag = b_a * exp(b_ge)
    tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty, fp_downcast_rounding=
        'rtne'), boundary_check=(0, 1))
    tl.store(p_bg, b_bg.to(p_bg.dtype.element_ty, fp_downcast_rounding=
        'rtne'), boundary_check=(0, 1))
    tl.store(p_ag, b_ag.to(p_ag.dtype.element_ty, fp_downcast_rounding=
        'rtne'), boundary_check=(0, 1))
    tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty, fp_downcast_rounding=
        'rtne'), boundary_check=(0, 1))
    b_q = b_q.to(b_k.dtype)
    for j in range(0, min(BC, T - i_t * BT)):
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BK], j, dtype=tl.int16)
            b_k_j = gather(b_k, row_idx, axis=0)
            b_gk_j = gather(b_gi, row_idx, axis=0)
            b_b_j = gather(b_b, row_idx, axis=0)
        else:
            mask = tl.arange(0, BC) == j
            b_k_j = tl.sum(tl.where(mask[:, None], b_k, 0), 0)[None, :]
            b_gk_j = tl.sum(tl.where(mask[:, None], b_gi, 0), 0)[None, :]
            b_b_j = tl.sum(tl.where(mask[:, None], b_b, 0), 0)[None, :]
        tmp = exp(b_gi - b_gk_j)
        b_A_qk = tl.sum(b_q * b_k_j * tmp, 1)
        m_i = (o_i >= j).to(tl.float32)
        b_A_qk = b_A_qk * m_i
        b_A_qb = tl.sum(b_q * b_b_j * tmp, 1)
        b_A_qb = b_A_qb * m_i
        tmp2 = exp(b_ge - b_gk_j)
        b_A_ak = tl.sum(b_a * b_k_j * tmp2, 1)
        m_i2 = (o_i > j).to(tl.float32)
        b_A_ak = b_A_ak * m_i2
        b_A_ab = tl.sum(b_a * b_b_j * tmp2, 1)
        b_A_ab = b_A_ab * m_i2
        tl.store(Aqk + o_A + j, b_A_qk.to(dtype=Aqk.dtype.element_ty,
            fp_downcast_rounding='rtne'), mask=m_A)
        tl.store(Aqb + o_A + j, b_A_qb.to(dtype=Aqb.dtype.element_ty,
            fp_downcast_rounding='rtne'), mask=m_A)
        tl.store(Aab + o_A + j, b_A_ab.to(dtype=Aqb.dtype.element_ty,
            fp_downcast_rounding='rtne'), mask=m_A)
        tl.store(Aak + o_A + j, b_A_ak.to(dtype=Aqk.dtype.element_ty,
            fp_downcast_rounding='rtne'), mask=m_A)


def chunk_dplr_fwd_intra(q: torch.Tensor, k: torch.Tensor, a: torch.Tensor,
    b: torch.Tensor, gi: torch.Tensor, ge: torch.Tensor, scale: float,
    chunk_size: int, cu_seqlens: Optional[torch.LongTensor]=None):
    B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    Aqk = q.new_empty(B, T, H, BT, dtype=q.dtype)
    Aqb = q.new_empty(B, T, H, BT, dtype=q.dtype)
    Aab = q.new_empty(B, T, H, BT, dtype=torch.float)
    Aak = q.new_empty(B, T, H, BT, dtype=torch.float)
    grid = NT, B, H
    BK = max(triton.next_power_of_2(K), 16)
    qg = torch.empty_like(q)
    kg = torch.empty_like(k, dtype=q.dtype)
    ag = torch.empty_like(a, dtype=q.dtype)
    bg = torch.empty_like(b, dtype=q.dtype)
    chunk_dplr_fwd_A_kernel_intra_sub_intra[grid](q=q, k=k, a=a, b=b, gi=gi,
        ge=ge, Aqk=Aqk, Aqb=Aqb, Aab=Aab, Aak=Aak, qg=qg, kg=kg, ag=ag, bg=
        bg, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=scale,
        T=T, H=H, K=K, BT=BT, BC=BT, BK=BK, GATHER_SUPPORTED=
        is_gather_supported)
    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg


def chunk_dplr_fwd_h(kg: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u:
    torch.Tensor, bg: torch.Tensor, gk: torch.Tensor, initial_state:
    Optional[torch.Tensor]=None, output_final_state: bool=False, cu_seqlens:
    Optional[torch.LongTensor]=None, chunk_size: int=64) ->Tuple[torch.
    Tensor, torch.Tensor]:
    B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices
            ), prepare_chunk_offsets(cu_seqlens, BT)
    BK = max(triton.next_power_of_2(K), 16)
    assert BK <= 256, 'current kernel does not support head dimension larger than 256.'
    if check_shared_mem('hopper', kg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem('ampere', kg.device.index):
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    h = kg.new_empty(B, NT, H, K, V)
    final_state = kg.new_empty(N, H, K, V, dtype=torch.float32
        ) if output_final_state else None
    v_new = torch.empty_like(u)
    grid = NK, NV, N * H
    chunk_dplr_fwd_kernel_h[grid](kg=kg, v=v, w=w, bg=bg, u=u, v_new=v_new,
        h=h, gk=gk, h0=initial_state, ht=final_state, cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets, T=T, H=H, K=K, V=V, BT=BT, BC=BC, BK=
        BK, BV=BV)
    return h, v_new, final_state


@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 
    3, 4]], key=['BT', 'BK', 'BV'], use_cuda_graph=use_cuda_graph, **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_kernel_h(kg, v, w, bg, u, v_new, gk, h, h0, ht,
    cu_seqlens, chunk_offsets, T, H: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, BV: tl
    .constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.
    constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    o_k = i_k * BK + tl.arange(0, BK)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)
    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + ((boh + i_t) * H + i_h) * K * V, (K, V),
            (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_hc = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(min(BT, T - i_t * BT), BC)):
            p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (K, T), (1, 
                H * K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_bg = tl.make_block_ptr(bg + (bos * H + i_h) * K, (K, T), (1, 
                H * K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K,
                1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V,
                1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V,
                1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_v_new = tl.make_block_ptr(v_new + (bos * H + i_h) * V, (T, V),
                (H * V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_kg = tl.load(p_kg, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_bg = tl.load(p_bg, boundary_check=(0, 1))
            b_v2 = tl.dot(b_w, b_h.to(b_w.dtype)) + tl.load(p_u,
                boundary_check=(0, 1))
            b_hc += tl.dot(b_kg, b_v)
            b_hc += tl.dot(b_bg.to(b_hc.dtype), b_v2)
            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty),
                boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k,
            mask=o_k < K).to(tl.float32)
        b_h *= exp(b_g_last[:, None])
        b_h += b_hc
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty, fp_downcast_rounding=
            'rtne'), boundary_check=(0, 1))


def prepare_wy_repr_fwd(ag: torch.Tensor, v: torch.Tensor, A_ak: torch.
    Tensor, A_ab: torch.Tensor, cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, _ = ag.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)
    fwd_fn = (prepare_wy_repr_fwd_kernel_chunk64 if BT == 64 else
        prepare_wy_repr_fwd_kernel_chunk32)
    A_ab_inv = torch.empty_like(A_ab)
    fwd_fn[NT, B * H](A_ab=A_ab, A_ab_inv=A_ab_inv, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, T=T, H=H, BT=BT, BC=BC)
    w, u = wu_fwd(ag=ag, v=v, A_ak=A_ak, A_ab_inv=A_ab_inv, cu_seqlens=
        cu_seqlens, chunk_size=BT)
    return w, u, A_ab_inv


def wu_fwd(ag: torch.Tensor, v: torch.Tensor, A_ak: torch.Tensor, A_ab_inv:
    torch.Tensor, cu_seqlens: Optional[torch.LongTensor], chunk_size: int
    ) ->Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64)
    w = torch.empty_like(ag)
    u = torch.empty_like(v)
    wu_fwd_kernel[NT, B * H](ag=ag, v=v, A_ak=A_ak, A_ab_inv=A_ab_inv, w=w,
        u=u, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H,
        K=K, V=V, BT=BT, BK=BK, BV=BV)
    return w, u


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in [2, 4, 8, 16] for num_stages in [2, 3, 4]
    ], key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'], use_cuda_graph=
    use_cuda_graph, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def wu_fwd_kernel(w, u, ag, v, A_ab_inv, A_ak, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK:
    tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
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
    o_s = tl.arange(0, BT)
    p_A_ab_inv = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (T, BT),
        (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_A_ak = tl.make_block_ptr(A_ak + (bos * H + i_h) * BT, (T, BT), (H *
        BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_Aab_inv = tl.load(p_A_ab_inv, boundary_check=(0, 1))
    b_Aak = tl.load(p_A_ak, boundary_check=(0, 1))
    b_Aab_inv = tl.where(o_s[:, None] >= o_s[None, :], b_Aab_inv, 0)
    b_Aak = tl.where(o_s[:, None] > o_s[None, :], b_Aak, 0)
    b_Aak = tl.dot(b_Aab_inv, b_Aak)
    b_Aak = b_Aak.to(v.dtype.element_ty, fp_downcast_rounding='rtne')
    b_Aab_inv = b_Aab_inv.to(ag.dtype.element_ty, fp_downcast_rounding='rtne')
    for i_k in range(tl.cdiv(K, BK)):
        p_ag = tl.make_block_ptr(ag + (bos * H + i_h) * K, (T, K), (H * K, 
            1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_ag = tl.load(p_ag, boundary_check=(0, 1))
        b_w = tl.dot(b_Aab_inv, b_ag)
        tl.store(p_w, b_w.to(p_w.dtype.element_ty, fp_downcast_rounding=
            'rtne'), boundary_check=(0, 1))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_Aak, b_v)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty, fp_downcast_rounding=
            'rtne'), boundary_check=(0, 1))


def chunk_rwkv6_fwd_cumsum(g: torch.Tensor, chunk_size: int, cu_seqlens:
    Optional[torch.Tensor]=None) ->torch.Tensor:
    B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    gi, ge = torch.empty_like(g, dtype=torch.float), torch.empty_like(g,
        dtype=torch.float)

    def grid(meta):
        return triton.cdiv(meta['S'], meta['BS']), NT, B * H
    chunk_rwkv6_fwd_cumsum_kernel[grid](g, gi, ge, cu_seqlens,
        chunk_indices, T=T, H=H, S=S, BT=BT)
    return gi, ge


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BS': BS}, num_warps=num_warps,
    num_stages=num_stages) for BS in [16, 32, 64] for num_warps in [4, 8, 
    16] for num_stages in [2, 3, 4]], key=['S', 'BT'], use_cuda_graph=
    use_cuda_graph, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_rwkv6_fwd_cumsum_kernel(s, oi, oe, cu_seqlens, chunk_indices, T,
    H: tl.constexpr, S: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
    IS_VARLEN: tl.constexpr):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
    m_e = tl.where(o_i[:, None] > o_i[None, :], 1.0, 0.0).to(tl.float32)
    p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H * S, 1), (
        i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_oi = tl.make_block_ptr(oi + (bos * H + i_h) * S, (T, S), (H * S, 1),
        (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_oe = tl.make_block_ptr(oe + (bos * H + i_h) * S, (T, S), (H * S, 1),
        (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_oi = tl.dot(m_i, b_s)
    b_oe = tl.dot(m_e, b_s)
    tl.store(p_oi, b_oi.to(p_oi.dtype.element_ty, fp_downcast_rounding=
        'rtne'), boundary_check=(0, 1))
    tl.store(p_oe, b_oe.to(p_oe.dtype.element_ty, fp_downcast_rounding=
        'rtne'), boundary_check=(0, 1))


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


@triton.jit
def gather(src, index, axis, _builder=None):
    """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
    return None


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps, num_stages=num_stages) for BK in BK_LIST for BV in BK_LIST for
    num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 3, 4]], key=['BT'
    ], use_cuda_graph=use_cuda_graph, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_fwd_kernel_o(qg, v, v_new, A_qk, A_qb, h, o, cu_seqlens,
    chunk_indices, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT:
    tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
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
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (T, K), (H * K, 
            1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K * V, (K, V), (V, 1
            ), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_qg = tl.load(p_qg, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_qg, b_h)
    p_Aqk = tl.make_block_ptr(A_qk + (bos * H + i_h) * BT, (T, BT), (H * BT,
        1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_Aqb = tl.make_block_ptr(A_qb + (bos * H + i_h) * BT, (T, BT), (H * BT,
        1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (
        i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_v_new = tl.make_block_ptr(v_new + (bos * H + i_h) * V, (T, V), (H * V,
        1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H * V, 1), (
        i_t * BT, i_v * BV), (BT, BV), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_Aqk = tl.load(p_Aqk, boundary_check=(0, 1))
    b_Aqb = tl.load(p_Aqb, boundary_check=(0, 1))
    b_Aqk = tl.where(m_s, b_Aqk, 0)
    b_Aqb = tl.where(m_s, b_Aqb, 0)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
    b_o = b_o + tl.dot(b_Aqk.to(b_v.dtype), b_v) + tl.dot(b_Aqb.to(b_v_new.
        dtype), b_v_new)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_dplr_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a:
    torch.Tensor, b: torch.Tensor, gk: torch.Tensor, scale: float,
    initial_state: torch.Tensor, output_final_state: bool, cu_seqlens:
    Optional[torch.LongTensor]=None, chunk_size: int=64):
    T = q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT, cu_seqlens=cu_seqlens)
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(q=q, k=k,
        a=a, b=b, gi=gi, ge=ge, scale=scale, cu_seqlens=cu_seqlens,
        chunk_size=BT)
    del ge
    w, u, _ = prepare_wy_repr_fwd(ag=ag, A_ab=A_ab, A_ak=A_ak, v=v,
        cu_seqlens=cu_seqlens, chunk_size=BT)
    del A_ab, A_ak
    h, v_new, final_state = chunk_dplr_fwd_h(kg=kg, bg=bg, v=v, w=w, u=u,
        gk=gi, initial_state=initial_state, output_final_state=
        output_final_state, cu_seqlens=cu_seqlens, chunk_size=BT)
    del u, kg, bg, gi
    o = chunk_dplr_fwd_o(qg=qg, v=v, v_new=v_new, A_qk=A_qk, A_qb=A_qb, h=h,
        cu_seqlens=cu_seqlens, chunk_size=BT)
    del v_new, h, A_qk, A_qb
    return o, final_state


def chunk_dplr_fwd_o(qg: torch.Tensor, v: torch.Tensor, v_new: torch.Tensor,
    A_qk: torch.Tensor, A_qb: torch.Tensor, h: torch.Tensor, cu_seqlens:
    Optional[torch.LongTensor]=None, chunk_size: int=64) ->torch.Tensor:
    B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    o = torch.empty_like(v)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_dplr_fwd_kernel_o[grid](qg=qg, v=v, v_new=v_new, A_qk=A_qk, A_qb=
        A_qb, h=h, o=o, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        T=T, H=H, K=K, V=V, BT=BT)
    return o


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _ChunkDPLRDeltaRuleFunction_forward(ctx, q: torch.Tensor, k: torch.
    Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, gk: torch.
    Tensor, scale: float, initial_state: torch.Tensor, output_final_state:
    bool, cu_seqlens: Optional[torch.LongTensor]=None):
    chunk_size = 16
    o, final_state = chunk_dplr_fwd(q=q, k=k, v=v, a=a, b=b, gk=gk, scale=
        scale, initial_state=initial_state, output_final_state=
        output_final_state, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    ctx.save_for_backward(q, k, v, a, b, gk, initial_state)
    ctx.cu_seqlens = cu_seqlens
    ctx.scale = scale
    ctx.chunk_size = chunk_size
    return o.to(q.dtype), final_state


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK}, num_warps=num_warps,
    num_stages=num_stages) for num_warps in NUM_WARPS_AUTOTUNE for
    num_stages in [2, 3, 4] for BK in [32, 64]], key=['BK', 'BT', 'K'],
    use_cuda_graph=use_cuda_graph, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_dgk_kernel(dgk, dgk_offset, dgk_last, dgk_output,
    cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.constexpr, BT: tl.
    constexpr, BK: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
        i_tg = (i_b * NT + i_t).to(tl.int32)
        bos, eos = (i_b * T).to(tl.int32), (i_b * T + T).to(tl.int32)
    stride_qk = H * K
    dgk += (bos * H + i_h) * K
    dgk_offset += (bos * H + i_h) * K
    dgk_last += (i_tg * H + i_h) * K
    dgk_output += (bos * H + i_h) * K
    p_dgk_last = dgk_last + tl.arange(0, BK) + i_k * BK
    m_k = tl.arange(0, BK) + i_k * BK < K
    b_dgk_last = tl.load(p_dgk_last, mask=m_k, other=0)
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BT, BK), (1, 0))
    b_dgk = tl.load(p_dgk, boundary_check=(0, 1))
    b_dgk_offset = tl.load(p_dgk_offset, boundary_check=(0, 1))
    b_dgk_cumsum = tl.cumsum(b_dgk, 0, reverse=True)
    b_dgk_cumsum += b_dgk_last[None, :]
    b_dgk_cumsum -= b_dgk_offset
    p_dgk_output = tl.make_block_ptr(dgk_output, (T, K), (stride_qk, 1), (
        i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dgk_output, b_dgk_cumsum.to(p_dgk_output.dtype.element_ty),
        boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 
    3, 4]], key=['BK', 'BT', 'K'], use_cuda_graph=use_cuda_graph, **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_intra(q, k, a, b, gi, ge, dAqk, dAqb, dAak, dAab,
    dq, dk, da, db, dqg, dkg, dag, dbg, dgk, dgk_offset, cu_seqlens,
    chunk_indices, scale: tl.constexpr, T, H: tl.constexpr, K: tl.constexpr,
    BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, IS_VARLEN: tl.
    constexpr, GATHER_SUPPORTED: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = (i_b * T).to(tl.int32), (i_b * T + T).to(tl.int32)
    if i_t * BT >= T:
        return
    ge += (bos * H + i_h) * K
    gi += (bos * H + i_h) * K
    q += (bos * H + i_h) * K
    a += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    da += (bos * H + i_h) * K
    db += (bos * H + i_h) * K
    dqg += (bos * H + i_h) * K
    dag += (bos * H + i_h) * K
    dkg += (bos * H + i_h) * K
    dbg += (bos * H + i_h) * K
    dgk += (bos * H + i_h) * K
    dgk_offset += (bos * H + i_h) * K
    dAqk += (bos * H + i_h) * BT
    dAqb += (bos * H + i_h) * BT
    dAak += (bos * H + i_h) * BT
    dAab += (bos * H + i_h) * BT
    stride_qk = H * K
    stride_A = H * BT
    p_ge = tl.make_block_ptr(ge, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_gi = tl.make_block_ptr(gi, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    b_ge = tl.load(p_ge, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    b_da = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    b_db = tl.zeros([BC, BK], dtype=tl.float32)
    p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (stride_A, 1), (i_t * BT, 0),
        (BC, BC), (1, 0))
    p_dAab = tl.make_block_ptr(dAab, (T, BT), (stride_A, 1), (i_t * BT, 0),
        (BC, BC), (1, 0))
    p_dAqb = tl.make_block_ptr(dAqb, (T, BT), (stride_A, 1), (i_t * BT, 0),
        (BC, BC), (1, 0))
    p_dAak = tl.make_block_ptr(dAak, (T, BT), (stride_A, 1), (i_t * BT, 0),
        (BC, BC), (1, 0))
    o_i = tl.arange(0, BC)
    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK),
        (BC, BK), (1, 0))
    p_b = tl.make_block_ptr(b, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK),
        (BC, BK), (1, 0))
    p_a = tl.make_block_ptr(a, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK),
        (BC, BK), (1, 0))
    p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK),
        (BC, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
    b_dAab = tl.load(p_dAab, boundary_check=(0, 1))
    b_dAqb = tl.load(p_dAqb, boundary_check=(0, 1))
    b_dAak = tl.load(p_dAak, boundary_check=(0, 1))
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    for j in range(0, min(BC, T - i_t * BT)):
        if GATHER_SUPPORTED:
            row_idx = tl.full([1, BK], j, dtype=tl.int16)
            col_idx = tl.full([BC, 1], j, dtype=tl.int16)
            row_idx_bc = tl.full([1, BC], j, dtype=tl.int16)
            b_kj = gather(b_k, row_idx, axis=0)
            b_bj = gather(b_b, row_idx, axis=0)
            b_gij = gather(b_gi, row_idx, axis=0)
            b_gej = gather(b_ge, row_idx, axis=0)
            b_qj = gather(b_q, row_idx, axis=0)
            b_aj = gather(b_a, row_idx, axis=0)
            b_dAqk_j = gather(b_dAqk, col_idx, axis=1)
            b_dAab_j = gather(b_dAab, col_idx, axis=1)
            b_dAqb_j = gather(b_dAqb, col_idx, axis=1)
            b_dAak_j = gather(b_dAak, col_idx, axis=1)
            b_dA_qk_j = tl.sum(gather(b_dAqk, row_idx_bc, axis=0), 0)[:, None]
            b_dA_qk_j = tl.sum(gather(b_dAqk, row_idx_bc, axis=0), 0)[:, None]
            b_dA_ab_j = tl.sum(gather(b_dAab, row_idx_bc, axis=0), 0)[:, None]
            b_dA_qb_j = tl.sum(gather(b_dAqb, row_idx_bc, axis=0), 0)[:, None]
            b_dA_ak_j = tl.sum(gather(b_dAak, row_idx_bc, axis=0), 0)[:, None]
        else:
            mask_idx = tl.arange(0, BC) == j
            b_kj = tl.sum(tl.where(mask_idx[:, None], b_k, 0), 0)[None, :]
            b_bj = tl.sum(tl.where(mask_idx[:, None], b_b, 0), 0)[None, :]
            b_gij = tl.sum(tl.where(mask_idx[:, None], b_gi, 0), 0)[None, :]
            b_gej = tl.sum(tl.where(mask_idx[:, None], b_ge, 0), 0)[None, :]
            b_dAqk_j = tl.sum(tl.where(mask_idx[None, :], b_dAqk, 0), 1)[:,
                None]
            b_dAab_j = tl.sum(tl.where(mask_idx[None, :], b_dAab, 0), 1)[:,
                None]
            b_dAqb_j = tl.sum(tl.where(mask_idx[None, :], b_dAqb, 0), 1)[:,
                None]
            b_dAak_j = tl.sum(tl.where(mask_idx[None, :], b_dAak, 0), 1)[:,
                None]
            b_dA_qk_j = tl.sum(tl.where(mask_idx[:, None], b_dAqk, 0), 0)[:,
                None]
            b_dA_ab_j = tl.sum(tl.where(mask_idx[:, None], b_dAab, 0), 0)[:,
                None]
            b_dA_qb_j = tl.sum(tl.where(mask_idx[:, None], b_dAqb, 0), 0)[:,
                None]
            b_dA_ak_j = tl.sum(tl.where(mask_idx[:, None], b_dAak, 0), 0)[:,
                None]
            b_qj = tl.sum(tl.where(mask_idx[:, None], b_q, 0), 0)[None, :]
            b_aj = tl.sum(tl.where(mask_idx[:, None], b_a, 0), 0)[None, :]
        m_e = o_i[:, None] > j
        m_i = o_i[:, None] >= j
        tmp1 = exp(b_gi - b_gij)
        tmp2 = exp(b_ge - b_gij)
        b_dq += tl.where(m_i, b_dAqk_j * b_kj * tmp1, 0.0)
        b_dq += tl.where(m_i, b_dAqb_j * b_bj * tmp1, 0.0)
        b_da += tl.where(m_e, b_dAab_j * b_bj * tmp2, 0.0)
        b_da += tl.where(m_e, b_dAak_j * b_kj * tmp2, 0.0)
        m_i = o_i[:, None] <= j
        m_e = o_i[:, None] < j
        tmp1 = exp(b_gij - b_gi)
        tmp2 = exp(b_gej - b_gi)
        b_dk += tl.where(m_i, b_dA_qk_j * b_qj * tmp1, 0.0)
        b_dk += tl.where(m_e, b_dA_ak_j * b_aj * tmp2, 0.0)
        b_db += tl.where(m_i, b_dA_qb_j * b_qj * tmp1, 0.0)
        b_db += tl.where(m_e, b_dA_ab_j * b_aj * tmp2, 0.0)
    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_da = tl.make_block_ptr(da, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_dgk = tl.make_block_ptr(dgk, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_dgk_offset = tl.make_block_ptr(dgk_offset, (T, K), (stride_qk, 1), (
        i_t * BT, i_k * BK), (BC, BK), (1, 0))
    p_dqg = tl.make_block_ptr(dqg, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_dkg = tl.make_block_ptr(dkg, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_dag = tl.make_block_ptr(dag, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_dbg = tl.make_block_ptr(dbg, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BC, BK), (1, 0))
    p_gn = gi + (min(i_t * BT + BT, T) - 1) * stride_qk + o_k
    p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_da += tl.load(p_dag, boundary_check=(0, 1)) * exp(b_ge)
    b_dq += tl.load(p_dqg, boundary_check=(0, 1)) * exp(b_gi) * scale
    tmp = exp(b_gn[None, :] - b_gi)
    b_dk += tl.load(p_dkg, boundary_check=(0, 1)).to(tl.float32) * tmp
    b_db += tl.load(p_dbg, boundary_check=(0, 1)).to(tl.float32) * tmp
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_da, b_da.to(p_da.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0, 1))
    b_dgk = (b_dq * b_q + b_da * b_a - b_dk * b_k - b_db * b_b).to(tl.float32)
    b_dgk_offset = b_da * b_a
    tl.store(p_dgk, b_dgk.to(p_dgk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dgk_offset, b_dgk_offset.to(p_dgk_offset.dtype.element_ty),
        boundary_check=(0, 1))


@triton.heuristics({'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not
    None, 'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 
    3, 4]], key=['BT', 'BK', 'BV', 'V'], use_cuda_graph=use_cuda_graph, **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_dhu(qg, bg, w, gk, dht, dh0, do, dh, dv, dv2,
    cu_seqlens, chunk_offsets, T, H: tl.constexpr, K: tl.constexpr, V: tl.
    constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr, BV: tl
    .constexpr, USE_FINAL_STATE_GRADIENT: tl.constexpr, USE_INITIAL_STATE:
    tl.constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))
    mask_k = tl.arange(0, BK) < K
    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + ((boh + i_t) * H + i_h) * K * V, (K,
            V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            p_qg = tl.make_block_ptr(qg + (bos * H + i_h) * K, (K, T), (1, 
                H * K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_bg = tl.make_block_ptr(bg + (bos * H + i_h) * K, (T, K), (H *
                K, 1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (K, T), (1, H *
                K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H *
                V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H *
                V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_dv2 = tl.make_block_ptr(dv2 + (bos * H + i_h) * V, (T, V), (H *
                V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_qg = tl.load(p_qg, boundary_check=(0, 1))
            b_bg = tl.load(p_bg, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dv2 = b_dv + tl.dot(b_bg, b_dh.to(b_bg.dtype))
            tl.store(p_dv2, b_dv2.to(p_dv.dtype.element_ty), boundary_check
                =(0, 1))
            b_dh_tmp += tl.dot(b_qg, b_do.to(b_qg.dtype))
            b_dh_tmp += tl.dot(b_w, b_dv2.to(b_qg.dtype))
        last_idx = min((i_t + 1) * BT, T) - 1
        bg_last = tl.load(gk + ((bos + last_idx) * H + i_h) * K + tl.arange
            (0, BK), mask=mask_k)
        b_dh *= exp(bg_last)[:, None]
        b_dh += b_dh_tmp
    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 
    3, 4]], key=['BV', 'BT'], use_cuda_graph=use_cuda_graph, **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_dplr_bwd_kernel_dAu(v, do, v_new, A_qb, dA_qk, dA_qb, dv_new,
    cu_seqlens, chunk_indices, scale: tl.constexpr, T, H: tl.constexpr, V:
    tl.constexpr, BT: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    b_dA_qk = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA_qb = tl.zeros([BT, BT], dtype=tl.float32)
    p_A_qb = tl.make_block_ptr(A_qb + (bos * H + i_h) * BT, (T, BT), (H *
        BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_qb = tl.load(p_A_qb, boundary_check=(0, 1))
    b_A_qb = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :
        ], b_A_qb, 0.0).to(b_A_qb.dtype)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H * V, 
            1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (V, T), (1, H * V),
            (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_v_new = tl.make_block_ptr(v_new + (bos * H + i_h) * V, (V, T), (1,
            H * V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_dv_new = tl.make_block_ptr(dv_new + (bos * H + i_h) * V, (T, V),
            (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
        b_dA_qk += tl.dot(b_do, b_v)
        b_dA_qb += tl.dot(b_do, b_v_new)
        b_dv_new = tl.dot(tl.trans(b_A_qb), b_do)
        tl.store(p_dv_new, b_dv_new.to(p_dv_new.dtype.element_ty),
            boundary_check=(0, 1))
    p_dA_qk = tl.make_block_ptr(dA_qk + (bos * H + i_h) * BT, (T, BT), (H *
        BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_dA_qb = tl.make_block_ptr(dA_qb + (bos * H + i_h) * BT, (T, BT), (H *
        BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA_qk = tl.where(m_s, b_dA_qk * scale, 0.0)
    tl.store(p_dA_qk, b_dA_qk.to(p_dA_qk.dtype.element_ty), boundary_check=
        (0, 1))
    b_dA_qb = tl.where(m_s, b_dA_qb * scale, 0.0)
    tl.store(p_dA_qb, b_dA_qb.to(p_dA_qb.dtype.element_ty), boundary_check=
        (0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps, num_stages=num_stages) for num_warps in NUM_WARPS_AUTOTUNE for
    num_stages in [2, 3, 4] for BK in BK_LIST for BV in BK_LIST], key=['BT'
    ], use_cuda_graph=use_cuda_graph, **autotune_cache_kwargs)
@triton.jit
def chunk_dplr_bwd_kernel_dv(A_qk, kg, do, dv, dh, cu_seqlens,
    chunk_indices, T, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT:
    tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
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
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    A_qk += (bos * H + i_h) * BT
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    kg += (bos * H + i_h) * K
    dh += (i_tg * H + i_h) * K * V
    stride_qk = H * K
    stride_vo = H * V
    stride_A = H * BT
    for i_k in range(tl.cdiv(K, BK)):
        p_dh = tl.make_block_ptr(dh, (K, V), (V, 1), (i_k * BK, i_v * BV),
            (BK, BV), (1, 0))
        p_kg = tl.make_block_ptr(kg, (T, K), (stride_qk, 1), (i_t * BT, i_k *
            BK), (BT, BK), (1, 0))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_kg = tl.load(p_kg, boundary_check=(0, 1))
        b_dv += tl.dot(b_kg, b_dh.to(b_kg.dtype))
    p_Aqk = tl.make_block_ptr(A_qk, (BT, T), (1, stride_A), (0, i_t * BT),
        (BT, BT), (0, 1))
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :],
        tl.load(p_Aqk, boundary_check=(0, 1)), 0)
    p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v *
        BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv, (T, V), (stride_vo, 1), (i_t * BT, i_v *
        BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do)
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps, num_stages
    =num_stages) for num_warps in NUM_WARPS_AUTOTUNE for num_stages in [2, 
    3, 4]], key=['BT', 'BK', 'BV'], use_cuda_graph=use_cuda_graph, **
    autotune_cache_kwargs)
@triton.jit
def chunk_dplr_bwd_o_kernel(v, v_new, h, do, dh, dk, db, w, dq, dv, dw, gk,
    dgk_last, k, b, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.
    constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.
    constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    dh += (i_tg * H + i_h) * K * V
    dk += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    db += (bos * H + i_h) * K
    b += (bos * H + i_h) * K
    dw += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dq += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    dgk_last += (i_tg * H + i_h) * K
    gk += (bos * H + i_h) * K
    stride_qk = H * K
    stride_vo = H * V
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_db = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk_last = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        p_v_new = tl.make_block_ptr(v_new, (T, V), (stride_vo, 1), (i_t *
            BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (stride_vo, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (
            BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK),
            (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dgk_last += tl.sum((b_h * b_dh).to(tl.float32), axis=0)
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))
        b_db += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
        p_dv = tl.make_block_ptr(dv, (T, V), (stride_vo, 1), (i_t * BT, i_v *
            BV), (BT, BV), (1, 0))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_dw += tl.dot(b_dv.to(b_v.dtype), b_h.to(b_v.dtype))
    m_k = i_k * BK + tl.arange(0, BK) < K
    last_idx = min(i_t * BT + BT, T) - 1
    b_gk_last = tl.load(gk + last_idx * stride_qk + i_k * BK + tl.arange(0,
        BK), mask=m_k, other=float('-inf'))
    b_dgk_last *= exp(b_gk_last)
    p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK),
        (BT, BK), (1, 0))
    p_b = tl.make_block_ptr(b, (T, K), (stride_qk, 1), (i_t * BT, i_k * BK),
        (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_dgk_last += tl.sum(b_k * b_dk, axis=0)
    b_dgk_last += tl.sum(b_b * b_db, axis=0)
    tl.store(dgk_last + tl.arange(0, BK) + i_k * BK, b_dgk_last, mask=m_k)
    p_dw = tl.make_block_ptr(dw, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BT, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (T, K), (stride_qk, 1), (i_t * BT, i_k *
        BK), (BT, BK), (1, 0))
    tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config(triton_config, num_warps=num_warps,
    num_stages=num_stages) for num_warps in [2, 4, 8, 16] for num_stages in
    [2, 3, 4]], key=['BT', 'BK', 'BV'], use_cuda_graph=use_cuda_graph, **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(A_ab_inv, A_ak, ag, v, dw, du, dv, dv0, dag,
    dAak, dAab, cu_seqlens, chunk_indices, T, H: tl.constexpr, K: tl.
    constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.
    constexpr, IS_VARLEN: tl.constexpr):
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
    p_Aak_t = tl.make_block_ptr(A_ak + (bos * H + i_h) * BT, (BT, T), (1, H *
        BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_Aab_inv_t = tl.make_block_ptr(A_ab_inv + (bos * H + i_h) * BT, (BT, T
        ), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_dAak = tl.make_block_ptr(dAak + (bos * H + i_h) * BT, (T, BT), (H *
        BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_dAab = tl.make_block_ptr(dAab + (bos * H + i_h) * BT, (T, BT), (H *
        BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A_ab_inv_t = tl.load(p_Aab_inv_t, boundary_check=(0, 1))
    b_A_ak_t = tl.load(p_Aak_t, boundary_check=(0, 1))
    b_A_ak_t = tl.where(tl.arange(0, BT)[:, None] < tl.arange(0, BT)[None,
        :], b_A_ak_t, 0)
    b_A_ab_inv_t = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[
        None, :], b_A_ab_inv_t, 0)
    b_A_tmp_t = tl.dot(b_A_ak_t, b_A_ab_inv_t).to(v.dtype.element_ty)
    b_dA_tmp = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 
            1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv0 = tl.make_block_ptr(dv0 + (bos * H + i_h) * V, (T, V), (H * V,
            1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos * H + i_h) * V, (T, V), (H * V, 
            1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA_tmp += tl.dot(b_du.to(b_v.dtype), tl.trans(b_v))
        b_dv0 = tl.load(p_dv0, boundary_check=(0, 1))
        b_dv = b_dv0 + tl.dot(b_A_tmp_t, b_du)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    m_i = tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :]
    b_dA_tmp = tl.where(m_i, b_dA_tmp, 0)
    b_dA_ak = tl.dot(b_A_ab_inv_t, b_dA_tmp)
    b_dA_ak = tl.where(m_i, b_dA_ak, 0)
    tl.store(p_dAak, b_dA_ak, boundary_check=(0, 1))
    b_dA_ab_inv = tl.dot(b_dA_tmp, b_A_ak_t)
    for i_k in range(tl.cdiv(K, BK)):
        p_ag = tl.make_block_ptr(ag + (bos * H + i_h) * K, (T, K), (H * K, 
            1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dag = tl.make_block_ptr(dag + (bos * H + i_h) * K, (T, K), (H * K,
            1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos * H + i_h) * K, (T, K), (H * K, 
            1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_ag = tl.load(p_ag, boundary_check=(0, 1))
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA_ab_inv += tl.dot(b_dw, tl.trans(b_ag))
        b_dag = tl.dot(b_A_ab_inv_t.to(b_dw.dtype), b_dw)
        tl.store(p_dag, b_dag.to(p_dag.dtype.element_ty), boundary_check=(0, 1)
            )
    b_dA_ab_inv = tl.where(tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[
        None, :], b_dA_ab_inv, 0)
    b_dA_ab_inv = tl.dot(b_A_ab_inv_t, b_dA_ab_inv)
    b_dA_ab_inv = tl.dot(b_dA_ab_inv, b_A_ab_inv_t)
    b_dA_ab_inv = tl.where(m_i, b_dA_ab_inv, 0)
    tl.store(p_dAab, b_dA_ab_inv, boundary_check=(0, 1))


def chunk_dplr_bwd_dqk_intra(q: torch.Tensor, k: torch.Tensor, a: torch.
    Tensor, b: torch.Tensor, gi: torch.Tensor, ge: torch.Tensor, dAqk:
    torch.Tensor, dAqb: torch.Tensor, dAak: torch.Tensor, dAab: torch.
    Tensor, dqg: torch.Tensor, dkg: torch.Tensor, dag: torch.Tensor, dbg:
    torch.Tensor, dgk_last: torch.Tensor, scale: float=1.0, cu_seqlens:
    Optional[torch.LongTensor]=None, chunk_size: int=64):
    B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = min(64, triton.next_power_of_2(K)) if check_shared_mem() else min(
        32, triton.next_power_of_2(K))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dgk = torch.empty_like(gi, dtype=torch.float)
    dgk_offset = torch.empty_like(gi, dtype=torch.float)
    grid = NK, NT, B * H
    chunk_dplr_bwd_kernel_intra[grid](q=q, k=k, a=a, b=b, gi=gi, ge=ge,
        dAqk=dAqk, dAqb=dAqb, dAak=dAak, dAab=dAab, dq=dq, dk=dk, dgk=dgk,
        dgk_offset=dgk_offset, dqg=dqg, dkg=dkg, dag=dag, dbg=dbg, da=da,
        db=db, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, scale=
        scale, T=T, H=H, K=K, BT=BT, BC=BT, BK=BK, GATHER_SUPPORTED=
        is_gather_supported)
    dgk_output = torch.empty_like(dgk)

    def grid(meta):
        return NT, triton.cdiv(K, meta['BK']), B * H
    chunk_dplr_bwd_dgk_kernel[grid](dgk=dgk, dgk_offset=dgk_offset,
        dgk_last=dgk_last, dgk_output=dgk_output, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, T=T, H=H, K=K, BT=BT)
    return dq, dk, da, db, dgk_output


def chunk_dplr_bwd_dhu(qg: torch.Tensor, bg: torch.Tensor, w: torch.Tensor,
    gk: torch.Tensor, h0: torch.Tensor, dht: Optional[torch.Tensor], do:
    torch.Tensor, dv: torch.Tensor, cu_seqlens: Optional[torch.LongTensor]=
    None, chunk_size: int=64) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor
    ]:
    B, T, H, K, V = *qg.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = max(triton.next_power_of_2(K), 16)
    assert BK <= 256, 'current kernel does not support head dimension being larger than 256.'
    if check_shared_mem('hopper', qg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem('ampere', qg.device.index):
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices
            ), prepare_chunk_offsets(cu_seqlens, BT)
    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    dh = qg.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.zeros_like(dv)
    grid = NK, NV, N * H
    chunk_dplr_bwd_kernel_dhu[grid](qg=qg, bg=bg, w=w, gk=gk, dht=dht, dh0=
        dh0, do=do, dh=dh, dv=dv, dv2=dv2, cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets, T=T, H=H, K=K, V=V, BT=BT, BC=BC, BK=
        BK, BV=BV)
    return dh, dh0, dv2


def chunk_dplr_bwd_dAu(v: torch.Tensor, v_new: torch.Tensor, do: torch.
    Tensor, A_qb: torch.Tensor, scale: float, cu_seqlens: Optional[torch.
    LongTensor]=None, chunk_size: int=64) ->torch.Tensor:
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if check_shared_mem('ampere'):
        BV = min(triton.next_power_of_2(V), 128)
    elif check_shared_mem('ada'):
        BV = min(max(triton.next_power_of_2(V), 16), 64)
    else:
        BV = min(triton.next_power_of_2(V), 32)
    grid = NT, B * H
    dA_qk = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dA_qb = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dv_new = torch.empty_like(v_new)
    chunk_dplr_bwd_kernel_dAu[grid](v=v, do=do, v_new=v_new, A_qb=A_qb,
        dA_qk=dA_qk, dA_qb=dA_qb, dv_new=dv_new, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, scale=scale, T=T, H=H, V=V, BT=BT, BV=BV)
    return dv_new, dA_qk, dA_qb


def chunk_dplr_bwd_dv(A_qk: torch.Tensor, kg: torch.Tensor, do: torch.
    Tensor, dh: torch.Tensor, cu_seqlens: Optional[torch.LongTensor]=None,
    chunk_size: int=64) ->torch.Tensor:
    B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    dv = torch.empty_like(do)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_dplr_bwd_kernel_dv[grid](A_qk=A_qk, kg=kg, do=do, dv=dv, dh=dh,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, K=K,
        V=V, BT=BT)
    return dv


def chunk_dplr_bwd_o(k: torch.Tensor, b: torch.Tensor, v: torch.Tensor,
    v_new: torch.Tensor, gk: torch.Tensor, do: torch.Tensor, h: torch.
    Tensor, dh: torch.Tensor, dv: torch.Tensor, w: torch.Tensor, cu_seqlens:
    Optional[torch.LongTensor]=None, chunk_size: int=64, scale: float=1.0
    ) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *w.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64) if check_shared_mem(
        ) else min(triton.next_power_of_2(K), 32)
    BV = min(max(triton.next_power_of_2(V), 16), 64) if check_shared_mem(
        ) else min(triton.next_power_of_2(K), 32)
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(k)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)
    grid = NK, NT, B * H
    dgk_last = torch.empty(B, NT, H, K, dtype=torch.float, device=w.device)
    chunk_dplr_bwd_o_kernel[grid](k=k, b=b, v=v, v_new=v_new, h=h, do=do,
        dh=dh, dq=dq, dk=dk, db=db, dgk_last=dgk_last, w=w, dv=dv, dw=dw,
        gk=gk, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H,
        K=K, V=V, BT=BT, BK=BK, BV=BV)
    return dq, dk, dw, db, dgk_last


def chunk_dplr_bwd_wy(A_ab_inv: torch.Tensor, A_ak: torch.Tensor, v: torch.
    Tensor, ag: torch.Tensor, dw: torch.Tensor, du: torch.Tensor, dv0:
    torch.Tensor, cu_seqlens: Optional[torch.LongTensor], chunk_size: int
    ) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A_ab_inv, A_ak, v, ag, dw, du = map(lambda x: x.contiguous(), [A_ab_inv,
        A_ak, v, ag, dw, du])
    B, T, H, K, V = *dw.shape, du.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    BV = min(max(triton.next_power_of_2(V), 16), 64) if check_shared_mem(
        ) else min(max(triton.next_power_of_2(V), 16), 32)
    dA_ab = torch.empty_like(A_ab_inv, dtype=torch.float)
    dA_ak = torch.empty_like(A_ak, dtype=torch.float)
    dv = torch.empty_like(v)
    dag = torch.empty_like(ag)
    prepare_wy_repr_bwd_kernel[NT, B * H](A_ab_inv=A_ab_inv, A_ak=A_ak, ag=
        ag, v=v, dw=dw, du=du, dv=dv, dv0=dv0, dag=dag, dAak=dA_ak, dAab=
        dA_ab, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, T=T, H=H,
        K=K, V=V, BT=BT, BK=BK, BV=BV)
    return dA_ab, dA_ak, dv, dag


# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _ChunkDPLRDeltaRuleFunction_backward(ctx, do: torch.Tensor, dht: torch.
    Tensor):
    q, k, v, a, b, gk, initial_state = ctx.saved_tensors
    BT = ctx.chunk_size
    cu_seqlens = ctx.cu_seqlens
    scale = ctx.scale
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT, cu_seqlens=cu_seqlens)
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(q=q, k=k,
        a=a, b=b, gi=gi, ge=ge, scale=scale, cu_seqlens=cu_seqlens,
        chunk_size=BT)
    w, u, A_ab_inv = prepare_wy_repr_fwd(ag=ag, A_ab=A_ab, A_ak=A_ak, v=v,
        cu_seqlens=cu_seqlens, chunk_size=BT)
    del A_ab
    h, v_new, _ = chunk_dplr_fwd_h(kg=kg, bg=bg, v=v, w=w, u=u, gk=gi,
        initial_state=initial_state, cu_seqlens=cu_seqlens, chunk_size=BT)
    del u
    dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(v=v, v_new=v_new, do=do,
        A_qb=A_qb, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT)
    dh, dh0, dv_new = chunk_dplr_bwd_dhu(qg=qg, bg=bg, w=w, gk=gi, h0=
        initial_state, dht=dht, do=do, dv=dv_new_intra, cu_seqlens=
        cu_seqlens, chunk_size=BT)
    dv = chunk_dplr_bwd_dv(A_qk=A_qk, kg=kg, do=do, dh=dh, cu_seqlens=
        cu_seqlens, chunk_size=BT)
    del A_qk
    dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(k=kg, b=bg, v=v, v_new=
        v_new, do=do, h=h, dh=dh, dv=dv_new, w=w, gk=gi, cu_seqlens=
        cu_seqlens, chunk_size=BT, scale=scale)
    del v_new
    dA_ab, dA_ak, dv, dag = chunk_dplr_bwd_wy(A_ab_inv=A_ab_inv, A_ak=A_ak,
        v=v, ag=ag, dw=dw, du=dv_new, dv0=dv, cu_seqlens=cu_seqlens,
        chunk_size=BT)
    del A_ak
    dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(q=q, k=k, a=a, b=b, gi=
        gi, ge=ge, dAqk=dA_qk, dAqb=dA_qb, dAak=dA_ak, dAab=dA_ab, dgk_last
        =dgk_last, dqg=dqg, dkg=dkg, dag=dag, dbg=dbg, chunk_size=BT, scale
        =scale, cu_seqlens=cu_seqlens)
    return dq.to(q), dk.to(k), dv.to(v), da.to(a), db.to(b), dgk.to(gk
        ), None, dh0, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ChunkDPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a:
        torch.Tensor, b: torch.Tensor, gk: torch.Tensor, scale: float,
        initial_state: torch.Tensor, output_final_state: bool, cu_seqlens:
        Optional[torch.LongTensor]=None):
        chunk_size = 16
        o, final_state = chunk_dplr_fwd(q=q, k=k, v=v, a=a, b=b, gk=gk,
            scale=scale, initial_state=initial_state, output_final_state=
            output_final_state, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        ctx.save_for_backward(q, k, v, a, b, gk, initial_state)
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, k, v, a, b, gk, initial_state = ctx.saved_tensors
        BT = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        scale = ctx.scale
        gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT, cu_seqlens=cu_seqlens)
        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(q=q,
            k=k, a=a, b=b, gi=gi, ge=ge, scale=scale, cu_seqlens=cu_seqlens,
            chunk_size=BT)
        w, u, A_ab_inv = prepare_wy_repr_fwd(ag=ag, A_ab=A_ab, A_ak=A_ak, v
            =v, cu_seqlens=cu_seqlens, chunk_size=BT)
        del A_ab
        h, v_new, _ = chunk_dplr_fwd_h(kg=kg, bg=bg, v=v, w=w, u=u, gk=gi,
            initial_state=initial_state, cu_seqlens=cu_seqlens, chunk_size=BT)
        del u
        dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(v=v, v_new=v_new,
            do=do, A_qb=A_qb, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT
            )
        dh, dh0, dv_new = chunk_dplr_bwd_dhu(qg=qg, bg=bg, w=w, gk=gi, h0=
            initial_state, dht=dht, do=do, dv=dv_new_intra, cu_seqlens=
            cu_seqlens, chunk_size=BT)
        dv = chunk_dplr_bwd_dv(A_qk=A_qk, kg=kg, do=do, dh=dh, cu_seqlens=
            cu_seqlens, chunk_size=BT)
        del A_qk
        dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(k=kg, b=bg, v=v,
            v_new=v_new, do=do, h=h, dh=dh, dv=dv_new, w=w, gk=gi,
            cu_seqlens=cu_seqlens, chunk_size=BT, scale=scale)
        del v_new
        dA_ab, dA_ak, dv, dag = chunk_dplr_bwd_wy(A_ab_inv=A_ab_inv, A_ak=
            A_ak, v=v, ag=ag, dw=dw, du=dv_new, dv0=dv, cu_seqlens=
            cu_seqlens, chunk_size=BT)
        del A_ak
        dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(q=q, k=k, a=a, b=b,
            gi=gi, ge=ge, dAqk=dA_qk, dAqb=dA_qb, dAak=dA_ak, dAab=dA_ab,
            dgk_last=dgk_last, dqg=dqg, dkg=dkg, dag=dag, dbg=dbg,
            chunk_size=BT, scale=scale, cu_seqlens=cu_seqlens)
        return dq.to(q), dk.to(k), dv.to(v), da.to(a), db.to(b), dgk.to(gk
            ), None, dh0, None, None
