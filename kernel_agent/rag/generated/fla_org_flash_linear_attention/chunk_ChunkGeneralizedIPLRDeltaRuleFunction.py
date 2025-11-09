# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/generalized_delta_rule/iplr/chunk.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/generalized_delta_rule/iplr/chunk.py
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

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in [2, 4] + ([] if check_shared_mem('hopper') else [8])], key
    =['BT', 'BK', 'BV'], use_cuda_graph=use_cuda_graph, **autotune_cache_kwargs
    )
@triton.jit(do_not_specialize=['T'])
def chunk_generalized_iplr_delta_rule_fwd_kernel_h(k, v, d, b, u, v_new, h,
    h0, ht, cu_seqlens, chunk_offsets, T, H: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
    BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE:
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
            p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H *
                K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_b = tl.make_block_ptr(b + (bos * H + i_h) * K, (K, T), (1, H *
                K), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_d = tl.make_block_ptr(d + (bos * H + i_h) * K, (T, K), (H * K,
                1), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V,
                1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V,
                1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_v_new = tl.make_block_ptr(v_new + (bos * H + i_h) * V, (T, V),
                (H * V, 1), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_b = tl.load(p_b, boundary_check=(0, 1))
            b_v2 = tl.dot(b_d, b_h.to(b_d.dtype)) + tl.load(p_u,
                boundary_check=(0, 1))
            b_hc += tl.dot(b_k, b_v)
            b_hc += tl.dot(b_b, b_v2.to(b_k.dtype))
            tl.store(p_v_new, b_v2.to(p_v_new.dtype.element_ty),
                boundary_check=(0, 1))
        b_h += b_hc
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (i_k *
            BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BK': BK, 'BV': BV}, num_warps=
    num_warps) for BK in BKV_LIST for BV in BKV_LIST for num_warps in [2, 4,
    8]], key=['BT'], use_cuda_graph=use_cuda_graph, **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def chunk_generalized_iplr_delta_rule_fwd_kernel_o(q, k, v, u, b, h, o,
    cu_seqlens, chunk_indices, scale, T, H: tl.constexpr, K: tl.constexpr,
    V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    IS_VARLEN: tl.constexpr):
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
    b += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    u += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h) * K * V
    stride_qk = H * K
    stride_vo = H * V
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_Aqk = tl.zeros([BT, BT], dtype=tl.float32)
    b_Aqb = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (stride_qk, 1), (i_t * BT, i_k *
            BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_qk), (i_k * BK, i_t *
            BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (
            BK, BV), (1, 0))
        p_b = tl.make_block_ptr(b, (K, T), (1, stride_qk), (i_k * BK, i_t *
            BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_Aqk += tl.dot(b_q, b_k)
        b_Aqb += tl.dot(b_q, b_b)
    o_i = tl.arange(0, BT)
    m_A = o_i[:, None] >= o_i[None, :]
    b_Aqk = tl.where(m_A, b_Aqk, 0)
    b_Aqb = tl.where(m_A, b_Aqb, 0)
    p_v = tl.make_block_ptr(v, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV),
        (BT, BV), (1, 0))
    p_u = tl.make_block_ptr(u, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV),
        (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (stride_vo, 1), (i_t * BT, i_v * BV),
        (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_u = tl.load(p_u, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_Aqk.to(b_v.dtype), b_v) + tl.dot(b_Aqb.to(b_u.
        dtype), b_u)) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({}, num_warps=num_warps) for
    num_warps in NUM_WARPS], key=['BT', 'BK', 'BV'], **autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def wu_fwd_kernel(w, u, a, k, v, A, cu_seqlens, chunk_indices, T, H: tl.
    constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.
    constexpr, BV: tl.constexpr, IS_VARLEN: tl.constexpr):
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
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H * BT, 1),
        (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_Aak = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_a = tl.make_block_ptr(a + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_w = tl.make_block_ptr(w + (bos * H + i_h) * K, (T, K), (H * K, 1),
            (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_a = tl.load(p_a, boundary_check=(0, 1))
        b_w = tl.dot(b_A, b_a)
        b_Aak += tl.dot(b_a, tl.trans(b_k))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))
    b_Aak = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :],
        b_Aak, 0)
    b_Aak = b_Aak.to(k.dtype.element_ty)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos * H + i_h) * V, (T, V), (H * V, 1),
            (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = tl.dot(b_Aak, b_v).to(v.dtype.element_ty)
        b_u = tl.dot(b_A, b_v)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))


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


def chunk_generalized_iplr_delta_rule_fwd(q: torch.Tensor, k: torch.Tensor,
    v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale: float,
    initial_state: torch.Tensor, output_final_state: bool, cu_seqlens:
    Optional[torch.LongTensor]=None, chunk_size: int=64):
    T = q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    w, u, _ = prepare_wy_repr_fwd(a=a, b=b, k=k, v=v, cu_seqlens=cu_seqlens,
        chunk_size=BT)
    h, v_new, final_state = chunk_generalized_iplr_delta_rule_fwd_h(k=k, v=
        v, b=b, w=w, u=u, initial_state=initial_state, output_final_state=
        output_final_state, cu_seqlens=cu_seqlens, chunk_size=BT)
    o = chunk_generalized_iplr_delta_rule_fwd_o(q=q, k=k, v=v, v_new=v_new,
        b=b, h=h, scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT)
    return o, final_state


def chunk_generalized_iplr_delta_rule_fwd_h(k: torch.Tensor, v: torch.
    Tensor, w: torch.Tensor, u: torch.Tensor, b: torch.Tensor,
    initial_state: Optional[torch.Tensor]=None, output_final_state: bool=
    False, cu_seqlens: Optional[torch.LongTensor]=None, chunk_size: int=64
    ) ->Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, u.shape[-1]
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
    if check_shared_mem('hopper', k.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem('ampere', k.device.index):
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'
    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32
        ) if output_final_state else None
    v_new = torch.empty_like(u)
    grid = NK, NV, N * H
    chunk_generalized_iplr_delta_rule_fwd_kernel_h[grid](k=k, v=v, d=w, b=b,
        u=u, v_new=v_new, h=h, h0=initial_state, ht=final_state, cu_seqlens
        =cu_seqlens, chunk_offsets=chunk_offsets, T=T, H=H, K=K, V=V, BT=BT,
        BC=BC, BK=BK, BV=BV)
    return h, v_new, final_state


def chunk_generalized_iplr_delta_rule_fwd_o(q: torch.Tensor, k: torch.
    Tensor, v: torch.Tensor, v_new: torch.Tensor, b: torch.Tensor, h: torch
    .Tensor, scale: Optional[float]=None, cu_seqlens: Optional[torch.
    LongTensor]=None, chunk_size: int=64) ->torch.Tensor:
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = k.shape[-1] ** -0.5
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    o = torch.empty_like(v)

    def grid(meta):
        return triton.cdiv(V, meta['BV']), NT, B * H
    chunk_generalized_iplr_delta_rule_fwd_kernel_o[grid](q=q, k=k, v=v, u=
        v_new, b=b, h=h, o=o, cu_seqlens=cu_seqlens, chunk_indices=
        chunk_indices, scale=scale, T=T, H=H, K=K, V=V, BT=BT)
    return o


def prepare_wy_repr_fwd(a: torch.Tensor, b: torch.Tensor, v: torch.Tensor,
    k: torch.Tensor, cu_seqlens: Optional[torch.LongTensor], chunk_size: int=64
    ) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K = a.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(BT, 32)
    BK = min(max(triton.next_power_of_2(K), 16), 64)
    A = torch.empty(B, T, H, BT, device=a.device, dtype=a.dtype)
    fwd_fn = (prepare_wy_repr_fwd_kernel_chunk64 if BT == 64 else
        prepare_wy_repr_fwd_kernel_chunk32)
    fwd_fn[NT, B * H](a=a, b=b, A=A, cu_seqlens=cu_seqlens, chunk_indices=
        chunk_indices, T=T, H=H, K=K, BT=BT, BK=BK, BC=BC)
    w, u = wu_fwd(a=a, v=v, k=k, A=A, cu_seqlens=cu_seqlens, chunk_size=
        chunk_size)
    return w, u, A


def wu_fwd(a: torch.Tensor, v: torch.Tensor, k: torch.Tensor, A: torch.
    Tensor, cu_seqlens: Optional[torch.LongTensor], chunk_size: int) ->Tuple[
    torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *a.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT
        ) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    u = torch.empty_like(v)
    w = torch.empty_like(a)
    wu_fwd_kernel[NT, B * H](a=a, v=v, w=w, u=u, A=A, k=k, cu_seqlens=
        cu_seqlens, chunk_indices=chunk_indices, T=T, H=H, K=K, V=V, BT=BT,
        BK=BK, BV=BV)
    return w, u


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


# Forward method (kernel launch code)
@input_guard
@autocast_custom_fwd
def _ChunkGeneralizedIPLRDeltaRuleFunction_forward(ctx, q: torch.Tensor, k:
    torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, scale:
    float, initial_state: torch.Tensor, output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor]=None):
    chunk_size = 64
    o, final_state = chunk_generalized_iplr_delta_rule_fwd(q=q, k=k, v=v, a
        =a, b=b, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens,
        chunk_size=chunk_size)
    return o.to(q.dtype), final_state


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
@input_guard
@autocast_custom_bwd
def _ChunkGeneralizedIPLRDeltaRuleFunction_backward(ctx, do: torch.Tensor,
    dht: torch.Tensor):
    raise NotImplementedError(
        'Backward pass for ChunkGeneralizedIPLRDeltaRuleFunction is not implemented yet. Stay tuned!'
        )


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ChunkGeneralizedIPLRDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a:
        torch.Tensor, b: torch.Tensor, scale: float, initial_state: torch.
        Tensor, output_final_state: bool, cu_seqlens: Optional[torch.
        LongTensor]=None):
        chunk_size = 64
        o, final_state = chunk_generalized_iplr_delta_rule_fwd(q=q, k=k, v=
            v, a=a, b=b, scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            chunk_size=chunk_size)
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        raise NotImplementedError(
            'Backward pass for ChunkGeneralizedIPLRDeltaRuleFunction is not implemented yet. Stay tuned!'
            )
