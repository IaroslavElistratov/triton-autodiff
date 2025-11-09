# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/fla-org/flash-linear-attention
# Source-Files: fla/modules/grpo.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rh2l8zei/flash-linear-attention-main/fla/modules/grpo.py
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
from math import log

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE': BLOCK_SIZE},
    num_warps=NUM_WARPS, num_stages=NUM_STAGES) for BLOCK_SIZE in [1024, 
    2048, 4096, 8192] for NUM_WARPS in NUM_WARPS_AUTOTUNE for NUM_STAGES in
    [1, 2, 4]], key=['B', 'N'], **autotune_cache_kwargs)
@triton.jit
def grpo_fwd_kernel(logits_ptr, ref_logp_ptr, input_ids_ptr, advantages_ptr,
    completion_mask_ptr, loss_ptr, lse_ptr, beta, save_kl: tl.constexpr, B,
    M, N, L, start_idx, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    off_b = row_idx // L
    N = tl.cast(N, tl.int64)
    loss_ptr += row_idx
    completion_mask_ptr += row_idx
    not_skip = tl.load(completion_mask_ptr).to(tl.int1)
    if not_skip == 1:
        ref_logp_ptr += row_idx
        lse_ptr += row_idx
        advantages_ptr += off_b
        logits_ptr += N * (row_idx + off_b)
        input_ids_ptr += row_idx + (off_b + 1) * start_idx
        base_cols = tl.arange(0, BLOCK_SIZE)
        m_i = -float('inf')
        l_i = 0.0
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(logits_ptr + cols, mask=mask, other=-float('inf')
                ).to(tl.float32)
            m_ij = tl.max(logits)
            new_m_i = tl.maximum(m_i, m_ij)
            l_i = l_i * exp(m_i - new_m_i) + tl.sum(exp(logits - new_m_i))
            m_i = new_m_i
        lse = log(l_i) + m_i
        idx = tl.load(input_ids_ptr)
        x = tl.load(logits_ptr + idx).to(tl.float32)
        advantage = tl.load(advantages_ptr).to(tl.float32)
        ref_logp = tl.load(ref_logp_ptr)
        logp = x - lse
        diff = ref_logp - logp
        kl = exp(diff) - diff - 1
        loss = kl * beta - advantage
        tl.store(loss_ptr, loss.to(loss_ptr.dtype.element_ty))
        tl.store(lse_ptr, lse.to(lse_ptr.dtype.element_ty))
        if save_kl:
            tl.store(loss_ptr + M, kl.to(loss_ptr.dtype.element_ty))
    else:
        tl.store(loss_ptr, 0.0)
        if save_kl:
            tl.store(loss_ptr + M, 0.0)


# Forward method (kernel launch code)
@input_guard
def _GrpoLoss_forward(ctx, logits, ref_logp, input_ids, advantages, beta,
    completion_mask, save_kl, inplace=True):
    ctx.input_shape = logits.shape
    B, L_ADD_1, N = ctx.input_shape
    L = L_ADD_1 - 1
    M = B * L
    input_ids_start_index = input_ids.size(1) - L
    if not save_kl:
        loss = torch.empty(B, L, device=logits.device, dtype=torch.float32)
    else:
        loss = torch.empty(B * 2, L, device=logits.device, dtype=torch.float32)
    lse = torch.empty(B, L, device=logits.device, dtype=torch.float32)
    if completion_mask is None:
        completion_mask = torch.ones(B, L, device=logits.device, dtype=
            torch.int32)
    else:
        loss[:B].masked_fill_(completion_mask.logical_not(), 0.0)
    grpo_fwd_kernel[M,](logits_ptr=logits, ref_logp_ptr=ref_logp,
        input_ids_ptr=input_ids, advantages_ptr=advantages,
        completion_mask_ptr=completion_mask, loss_ptr=loss, lse_ptr=lse,
        beta=beta, save_kl=save_kl, B=B, M=M, N=N, L=L, start_idx=
        input_ids_start_index)
    ctx.beta = beta
    ctx.save_for_backward(lse, logits, input_ids, advantages, completion_mask)
    ctx.ref_logp = ref_logp
    ctx.inplace = inplace
    return loss


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({}, num_warps=NUM_WARPS, num_stages
    =NUM_STAGES) for NUM_WARPS in [32] for NUM_STAGES in [4]], key=['B',
    'N'], **autotune_cache_kwargs)
@triton.jit
def grpo_bwd_kernel(dloss_ptr, dlogits_ptr, logits_ptr, ref_logp_ptr,
    input_ids_ptr, advantages_ptr, completion_mask_ptr, lse_ptr, beta, B, N,
    L, start_idx, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    off_b = row_idx // L
    N = tl.cast(N, tl.int64)
    dlogits_ptr += N * (row_idx + off_b)
    base_cols = tl.arange(0, BLOCK_SIZE)
    completion_mask_ptr += row_idx
    not_skip = tl.load(completion_mask_ptr).to(tl.int1)
    if not_skip == 1:
        lse_ptr += row_idx
        dloss_ptr += row_idx
        advantages_ptr += off_b
        ref_logp_ptr += row_idx
        logits_ptr += N * (row_idx + off_b)
        input_ids_ptr += row_idx + (off_b + 1) * start_idx
        dloss = tl.load(dloss_ptr).to(tl.float32)
        lse = tl.load(lse_ptr).to(tl.float32)
        idx = tl.load(input_ids_ptr)
        x = tl.load(logits_ptr + idx).to(tl.float32)
        advantage = tl.load(advantages_ptr).to(tl.float32)
        ref_logp = tl.load(ref_logp_ptr)
        tl.debug_barrier()
        logp = x - lse
        dlogp = (beta * (-1.0 * exp(ref_logp - logp) + 1) - advantage) * dloss
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(logits_ptr + cols, mask=mask, other=-float('inf')
                ).to(tl.float32)
            probs = exp(logits - lse)
            dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
            tl.store(dlogits_ptr + cols, dlogits.to(dlogits_ptr.dtype.
                element_ty), mask=mask)
    else:
        dlogits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            tl.store(dlogits_ptr + cols, dlogits.to(dlogits_ptr.dtype.
                element_ty), mask=mask)


# Backward method (kernel launch code)
@input_guard
def _GrpoLoss_backward(ctx, dloss):
    lse, logits, input_ids, advantages, completion_mask = ctx.saved_tensors
    inplace = ctx.inplace
    B, L_ADD_1, N = ctx.input_shape
    L = L_ADD_1 - 1
    M = B * L
    input_ids_start_index = input_ids.size(1) - L
    dlogits = logits if inplace else torch.empty_like(logits)
    BN = min(65536, triton.next_power_of_2(N))
    grpo_bwd_kernel[M,](dloss_ptr=dloss, dlogits_ptr=dlogits, logits_ptr=
        logits, ref_logp_ptr=ctx.ref_logp, input_ids_ptr=input_ids,
        advantages_ptr=advantages, completion_mask_ptr=completion_mask,
        lse_ptr=lse, beta=ctx.beta, B=B, N=N, L=L, BLOCK_SIZE=BN, start_idx
        =input_ids_start_index)
    dlogits[:, -1, :].fill_(0.0)
    return dlogits.view(*ctx.input_shape
        ), None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class GrpoLoss(torch.autograd.Function):

    @input_guard
    @staticmethod
    def forward(ctx, logits, ref_logp, input_ids, advantages, beta,
        completion_mask, save_kl, inplace=True):
        ctx.input_shape = logits.shape
        B, L_ADD_1, N = ctx.input_shape
        L = L_ADD_1 - 1
        M = B * L
        input_ids_start_index = input_ids.size(1) - L
        if not save_kl:
            loss = torch.empty(B, L, device=logits.device, dtype=torch.float32)
        else:
            loss = torch.empty(B * 2, L, device=logits.device, dtype=torch.
                float32)
        lse = torch.empty(B, L, device=logits.device, dtype=torch.float32)
        if completion_mask is None:
            completion_mask = torch.ones(B, L, device=logits.device, dtype=
                torch.int32)
        else:
            loss[:B].masked_fill_(completion_mask.logical_not(), 0.0)
        grpo_fwd_kernel[M,](logits_ptr=logits, ref_logp_ptr=ref_logp,
            input_ids_ptr=input_ids, advantages_ptr=advantages,
            completion_mask_ptr=completion_mask, loss_ptr=loss, lse_ptr=lse,
            beta=beta, save_kl=save_kl, B=B, M=M, N=N, L=L, start_idx=
            input_ids_start_index)
        ctx.beta = beta
        ctx.save_for_backward(lse, logits, input_ids, advantages,
            completion_mask)
        ctx.ref_logp = ref_logp
        ctx.inplace = inplace
        return loss

    @input_guard
    @staticmethod
    def backward(ctx, dloss):
        lse, logits, input_ids, advantages, completion_mask = ctx.saved_tensors
        inplace = ctx.inplace
        B, L_ADD_1, N = ctx.input_shape
        L = L_ADD_1 - 1
        M = B * L
        input_ids_start_index = input_ids.size(1) - L
        dlogits = logits if inplace else torch.empty_like(logits)
        BN = min(65536, triton.next_power_of_2(N))
        grpo_bwd_kernel[M,](dloss_ptr=dloss, dlogits_ptr=dlogits,
            logits_ptr=logits, ref_logp_ptr=ctx.ref_logp, input_ids_ptr=
            input_ids, advantages_ptr=advantages, completion_mask_ptr=
            completion_mask, lse_ptr=lse, beta=ctx.beta, B=B, N=N, L=L,
            BLOCK_SIZE=BN, start_idx=input_ids_start_index)
        dlogits[:, -1, :].fill_(0.0)
        return dlogits.view(*ctx.input_shape
            ), None, None, None, None, None, None, None
