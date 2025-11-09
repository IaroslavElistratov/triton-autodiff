import os
import contextlib
import functools
import inspect
import math
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
from torch.nn import functional as F
import triton
import triton.language as tl


from kernel_agent.autodiff import autodiff


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'


supports_autotune_cache = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if supports_autotune_cache else {}


exp = tl.exp




# todo-now: support .autotune and .heuristics
# @triton.heuristics({
#     'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
#     'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
#     'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
# })
# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8, 16]
#     ],
#     key=['BK', 'BV'],
#     **autotune_cache_kwargs
# )
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv6_fwd_kernel(
    q,  # query [B, H, T, K]/[B, T, H, K]
    k,  # key [B, H, T, K]/[B, T, H, K]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H] or None
    u,  # bonus [B, H, K]
    o,  # output [NK, B, H, T, V]/[NK, B, T, H, V]
    h0,  # initial hidden state [B, H, K, V]
    ht,  # final hidden state [B, H, K, V]
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,  # whether to reverse the recurrence
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_w = w + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_kv = b_k[:, None] * b_v[None, :]
        b_o = tl.sum((b_h + b_kv * b_u[:, None]) * b_q[:, None], 0)
        b_h = b_h * exp(b_w)[:, None] + b_kv
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * H*K
        p_k += (-1 if REVERSE else 1) * H*K
        p_v += (-1 if REVERSE else 1) * H*V
        p_w += (-1 if REVERSE else 1) * H*K
        p_o += (-1 if REVERSE else 1) * H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


# todo-now: add grad dh0?
@autodiff(inputs_require_grad=(0, 1, 2, 3, 4))
def fused_recurrent_rwkv6_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float)

    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_fwd_kernel[grid](
        q,
        k,
        v,
        w,
        u,
        o,
        h0,
        ht,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
    )
    o = o.sum(0)
    return o, ht


SWEEP = [
    {"B": 4, "T": 256, "H": 32, "D": 32, "required": True},
    # todo-now: fix advancing to next phase if all required shapes passed, even if TimeoutErr
    # {"B": 8, "T": 2048, "H": 64, "D": 64, "required": False},
    # {"B": 8, "T": 4096, "H": 64, "D": 64, "required": False},
    # {"B": 8, "T": 8192, "H": 64, "D": 64, "required": False},
]

def make_args(dims, device=DEVICE, dtype=torch.bfloat16):
    dims = dict(dims)
    B, T, H, D = dims["B"], dims["T"], dims["H"], dims["D"]
    r = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    w = F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=dtype)).requires_grad_(True)
    u = torch.randn(H, D, device=device, dtype=dtype).requires_grad_(True)
    scale = 1.0 / math.sqrt(D)
    return (r, k, v, w, u), {"scale": scale}

# temporarily disable no bwd-file flop reference to validate it
# def flops(dims, mode):
#     if mode == "bwd":
#         # forward recurrence ~2*B*H*T*D^2, backward approx 3.5x inflation
#         B, T, H, D = dims["B"], dims["T"], dims["H"], dims["D"]
#         forward = 2.0 * B * H * T * D * D
#         return forward * 3.5
#     raise ValueError(f"flops only implemented for backward mode; got {mode}")


def setup():
    (r, k, v, w, u), kwargs = make_args(SWEEP[0])
    fused_recurrent_rwkv6_fwd(r, k, v, w, u, **kwargs)


if __name__ == "__main__":
    setup()
