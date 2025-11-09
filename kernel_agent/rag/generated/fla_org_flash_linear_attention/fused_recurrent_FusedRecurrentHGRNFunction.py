# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/hgrn/fused_recurrent.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/hgrn/fused_recurrent.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BD': BD}, num_warps=num_warps) for
    BD in [32, 64, 128] for num_warps in [1, 2, 4, 8]], key=['D'], **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_hgrn_fwd_kernel(x, g, o, h0, ht, cu_seqlens, T, D: tl.
    constexpr, BD: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_x = x + bos * D + o_d
    p_g = g + bos * D + o_d
    p_o = o + bos * D + o_d
    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_n * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = exp(b_g) * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)
        p_x += D
        p_g += D
        p_o += D
    if STORE_FINAL_STATE:
        p_ht = ht + i_n * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)


def fused_recurrent_hgrn_fwd(x: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor=None, output_final_state: bool=False,
    cu_seqlens: Optional[torch.LongTensor]=None) ->Tuple[torch.Tensor,
    torch.Tensor]:
    B, T, D = x.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    o = torch.empty_like(x)
    final_state = x.new_empty(N, D) if output_final_state else None

    def grid(meta):
        return triton.cdiv(D, meta['BD']), N
    fused_recurrent_hgrn_fwd_kernel[grid](x=x, g=g, o=o, h0=initial_state,
        ht=final_state, cu_seqlens=cu_seqlens, T=T, D=D)
    return o, final_state


# Forward method (kernel launch code)
@input_guard
def _FusedRecurrentHGRNFunction_forward(ctx, x: torch.Tensor, g: torch.
    Tensor, initial_state: torch.Tensor=None, output_final_state: bool=
    False, cu_seqlens: Optional[torch.LongTensor]=None):
    o, ht = fused_recurrent_hgrn_fwd(x=x, g=g, initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens)
    ctx.save_for_backward(g, o, initial_state)
    ctx.cu_seqlens = cu_seqlens
    return o, ht


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.autotune(configs=[triton.Config({'BD': BD}, num_warps=num_warps) for
    BD in [32, 64, 128] for num_warps in [1, 2, 4, 8]], key=['D'], **
    autotune_cache_kwargs)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_hgrn_bwd_kernel(g, o, h0, dx, dg, do, dht, dh0,
    cu_seqlens, T, D: tl.constexpr, BD: tl.constexpr, USE_INITIAL_STATE: tl
    .constexpr, USE_FINAL_STATE_GRADIENT: tl.constexpr, IS_VARLEN: tl.constexpr
    ):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_g = g + (bos + T - 1) * D + o_d
    p_o = o + (bos + T - 2) * D + o_d
    p_dx = dx + (bos + T - 1) * D + o_d
    p_dg = dg + (bos + T - 1) * D + o_d
    p_do = do + (bos + T - 1) * D + o_d
    b_dh = tl.zeros([BD], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = dht + i_n * D + o_d
        b_dh += tl.load(p_dht, mask=mask, other=0).to(tl.float32)
    for i in range(T - 1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        if i > 0:
            b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_n * D + o_d, mask=mask, other=0).to(tl.float32
                )
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * exp(b_g)
        b_dg = b_dh * b_o
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)
        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D
    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_n * D + o_d
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask)


def fused_recurrent_hgrn_bwd(g: torch.Tensor, o: torch.Tensor, do: torch.
    Tensor, dht: torch.Tensor=None, initial_state: torch.Tensor=None,
    cu_seqlens: Optional[torch.LongTensor]=None) ->Tuple[torch.Tensor,
    torch.Tensor]:
    B, T, D = do.shape
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    dx = torch.empty_like(o, dtype=torch.float)
    dg = torch.empty_like(g, dtype=torch.float)
    dh0 = torch.empty_like(initial_state, dtype=torch.float
        ) if initial_state is not None else None

    def grid(meta):
        return triton.cdiv(D, meta['BD']), N
    fused_recurrent_hgrn_bwd_kernel[grid](g=g, o=o, h0=initial_state, dx=dx,
        dg=dg, do=do, dht=dht, dh0=dh0, cu_seqlens=cu_seqlens, T=T, D=D)
    return dx, dg, dh0


# Backward method (kernel launch code)
@input_guard
def _FusedRecurrentHGRNFunction_backward(ctx, do, dht=None):
    g, o, initial_state = ctx.saved_tensors
    cu_seqlens = ctx.cu_seqlens
    dx, dg, dh0 = fused_recurrent_hgrn_bwd(g=g, o=o, do=do, dht=dht,
        initial_state=initial_state, cu_seqlens=cu_seqlens)
    return dx, dg, dh0, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedRecurrentHGRNFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x: torch.Tensor, g: torch.Tensor, initial_state: torch
        .Tensor=None, output_final_state: bool=False, cu_seqlens: Optional[
        torch.LongTensor]=None):
        o, ht = fused_recurrent_hgrn_fwd(x=x, g=g, initial_state=
            initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens)
        ctx.save_for_backward(g, o, initial_state)
        ctx.cu_seqlens = cu_seqlens
        return o, ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        cu_seqlens = ctx.cu_seqlens
        dx, dg, dh0 = fused_recurrent_hgrn_bwd(g=g, o=o, do=do, dht=dht,
            initial_state=initial_state, cu_seqlens=cu_seqlens)
        return dx, dg, dh0, None, None
