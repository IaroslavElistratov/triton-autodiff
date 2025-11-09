# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/ops/comba/fused_recurrent.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/ops/comba/fused_recurrent.py
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
import time

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not
    None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_comba_fwd_kernel(q, k, p, v, g, beta, o, h0, ht,
    cu_seqlens, scale, T, B: tl.constexpr, H: tl.constexpr, HV: tl.
    constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.
    constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.
    constexpr, IS_BETA_HEADWISE: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.
    constexpr, IS_VARLEN: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_p = p + (bos * H + i_h) * K + o_k
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_p = tl.load(p_p, mask=mask_k, other=0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
            b_p = b_p / tl.sqrt(tl.sum(b_p * b_p) + 1e-06)
        b_q = b_q * scale
        b_v -= tl.sum(b_h * b_p[:, None], 0)
        b_h *= exp(b_g)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_p += H * K
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_comba_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.
    Tensor, p: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, scale:
    float, initial_state: torch.Tensor, output_final_state: bool,
    use_qk_l2norm_in_kernel: bool=False, cu_seqlens: Optional[torch.
    LongTensor]=None) ->Tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported yet'
    num_stages = 3
    num_warps = 1
    o = q.new_empty(NK, *v.shape)
    if output_final_state:
        final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
    else:
        final_state = None
    grid = NK, NV, N * HV
    fused_recurrent_comba_fwd_kernel[grid](q=q, k=k, p=p, v=v, g=g, beta=
        beta, o=o, h0=initial_state, ht=final_state, cu_seqlens=cu_seqlens,
        scale=scale, T=T, B=B, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        IS_BETA_HEADWISE=beta.ndim == v.ndim, USE_QK_L2NORM_IN_KERNEL=
        use_qk_l2norm_in_kernel, num_warps=num_warps, num_stages=num_stages)
    o = o.squeeze(0)
    return o, final_state


# Forward method (kernel launch code)
@input_guard
def _FusedRecurrentCombaFunction_forward(ctx, q: torch.Tensor, k: torch.
    Tensor, p: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.
    Tensor, scale: float, initial_state: torch.Tensor, output_final_state:
    bool, use_qk_l2norm_in_kernel: bool=False, cu_seqlens: Optional[torch.
    LongTensor]=None):
    o, final_state = fused_recurrent_comba_fwd(q=q, k=k, p=p, v=v, g=g,
        beta=beta, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, use_qk_l2norm_in_kernel=
        use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)
    return o, final_state


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
@input_guard
def _FusedRecurrentCombaFunction_backward(ctx, do, dht):
    raise NotImplementedError(
        "Backward pass is not implemented yet and we do not have plans to implement it because we haven't figured out how to compute dg without materializing the full hidden states for all time steps."
        )


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedRecurrentCombaFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, p: torch.Tensor, v:
        torch.Tensor, g: torch.Tensor, beta: torch.Tensor, scale: float,
        initial_state: torch.Tensor, output_final_state: bool,
        use_qk_l2norm_in_kernel: bool=False, cu_seqlens: Optional[torch.
        LongTensor]=None):
        o, final_state = fused_recurrent_comba_fwd(q=q, k=k, p=p, v=v, g=g,
            beta=beta, scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, use_qk_l2norm_in_kernel=
            use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens)
        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it because we haven't figured out how to compute dg without materializing the full hidden states for all time steps."
            )
