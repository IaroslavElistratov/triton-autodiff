# SPDX-License-Identifier: BSD-2-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/linkedin/Liger-Kernel
# Source-Files: src/liger_kernel/ops/grpo_loss.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_d1vk6zhm/Liger-Kernel-main/src/liger_kernel/ops/grpo_loss.py
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

@triton.jit
def _grpo_loss_fwd_kernel(LOGITS, OLD_LOGP, REF_LOGP, INPUT_IDS,
    COMPLETION_MASK, ADVANTAGES, LOSS, LSE, KL, IS_CLIPPED, TEMPERATURE,
    BETA: tl.constexpr, EPS_LOW, EPS_HIGH, L: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr=4096):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)
    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            return
    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LOSS += off_b * L + off_l
    LSE += off_b * L + off_l
    IS_CLIPPED += off_b * L + off_l
    m_i = float('-inf')
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float('-inf')).to(
            tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)
    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse
    if OLD_LOGP is None:
        old_logp = logp
    else:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    advantage = tl.load(ADVANTAGES).to(tl.float32)
    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)
    is_clipped = per_token_loss1 < per_token_loss2
    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
        per_token_loss += BETA * kl
        tl.store(KL, kl)
    tl.store(LOSS, per_token_loss)
    tl.store(LSE, lse)
    tl.store(IS_CLIPPED, is_clipped)


# Forward method (kernel launch code)
def _GrpoLossFunction_forward(ctx, logits, old_logp, ref_logp,
    completion_ids, advantages, completion_mask, temperature, beta, eps_low,
    eps_high, inplace):
    assert logits.is_contiguous() and completion_ids.is_contiguous()
    assert old_logp is None or old_logp.is_contiguous()
    assert ref_logp is not None and ref_logp.is_contiguous(
        ) if beta != 0.0 else True
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    if completion_mask is not None:
        assert completion_mask.is_contiguous()
    loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    lse = torch.zeros_like(loss)
    is_clipped = torch.zeros_like(loss)
    kl = torch.zeros_like(loss) if beta != 0.0 else None
    kwargs = {'BLOCK_N': 2048, 'num_stages': 2, 'num_warps': 1}
    _grpo_loss_fwd_kernel[B, L](logits, old_logp, ref_logp, completion_ids,
        completion_mask, advantages, loss, lse, kl, is_clipped, temperature,
        beta, eps_low, eps_high, L, N, **kwargs)
    ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids,
        advantages, completion_mask, lse)
    ctx.infos = temperature, beta, eps_low, eps_high, inplace
    return loss, kl, is_clipped


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _grpo_loss_bwd_kernel(DLOSS, DLOGITS, LOGITS, OLD_LOGP, REF_LOGP,
    INPUT_IDS, ADVANTAGES, COMPLETION_MASK, LSE, TEMPERATURE, BETA: tl.
    constexpr, EPS_LOW, EPS_HIGH, loss_stride0, loss_stride1, L: tl.
    constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr=4096):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)
    DLOGITS += off_b * (L + 1) * N + off_l * N
    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS + cols, 0.0, mask=cols < N)
            return
    LOGITS += off_b * (L + 1) * N + off_l * N
    DLOSS += off_b * loss_stride0 + off_l * loss_stride1
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LSE += off_b * L + off_l
    dloss = tl.load(DLOSS).to(tl.float32)
    lse = tl.load(LSE).to(tl.float32)
    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse
    if OLD_LOGP is None:
        old_logp = logp
    else:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    advantage = tl.load(ADVANTAGES).to(tl.float32)
    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    mask = per_token_loss2 >= per_token_loss1
    dlogp = -per_token_loss1 * mask
    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        dlogp += BETA * (1 - tl.exp(ref_logp - logp))
    dlogp = dlogp * dloss / TEMPERATURE
    tl.debug_barrier()
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=-float('inf')).to(
            tl.float32) / TEMPERATURE
        probs = tl.exp(logits - lse)
        dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
        tl.store(DLOGITS + cols, dlogits, mask=cols < N)


# Backward method (kernel launch code)
def _GrpoLossFunction_backward(ctx, *args):
    dloss = args[0]
    (logits, old_logp, ref_logp, completion_ids, advantages,
        completion_mask, lse) = ctx.saved_tensors
    temperature, beta, eps_low, eps_high, inplace = ctx.infos
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    dlogits = logits.data if inplace else torch.empty_like(logits)
    kwargs = {'BLOCK_N': 4096, 'num_stages': 1, 'num_warps': 16}
    _grpo_loss_bwd_kernel[B, L](dloss, dlogits, logits, old_logp, ref_logp,
        completion_ids, advantages, completion_mask, lse, temperature, beta,
        eps_low, eps_high, *dloss.stride(), L, N, **kwargs)
    dlogits[:, -1, :] = 0
    return dlogits, None, None, None, None, None, None, None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class GrpoLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, old_logp, ref_logp, completion_ids, advantages,
        completion_mask, temperature, beta, eps_low, eps_high, inplace):
        assert logits.is_contiguous() and completion_ids.is_contiguous()
        assert old_logp is None or old_logp.is_contiguous()
        assert ref_logp is not None and ref_logp.is_contiguous(
            ) if beta != 0.0 else True
        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1
        if completion_mask is not None:
            assert completion_mask.is_contiguous()
        loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
        lse = torch.zeros_like(loss)
        is_clipped = torch.zeros_like(loss)
        kl = torch.zeros_like(loss) if beta != 0.0 else None
        kwargs = {'BLOCK_N': 2048, 'num_stages': 2, 'num_warps': 1}
        _grpo_loss_fwd_kernel[B, L](logits, old_logp, ref_logp,
            completion_ids, completion_mask, advantages, loss, lse, kl,
            is_clipped, temperature, beta, eps_low, eps_high, L, N, **kwargs)
        ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids,
            advantages, completion_mask, lse)
        ctx.infos = temperature, beta, eps_low, eps_high, inplace
        return loss, kl, is_clipped

    @staticmethod
    def backward(ctx, *args):
        dloss = args[0]
        (logits, old_logp, ref_logp, completion_ids, advantages,
            completion_mask, lse) = ctx.saved_tensors
        temperature, beta, eps_low, eps_high, inplace = ctx.infos
        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1
        dlogits = logits.data if inplace else torch.empty_like(logits)
        kwargs = {'BLOCK_N': 4096, 'num_stages': 1, 'num_warps': 16}
        _grpo_loss_bwd_kernel[B, L](dloss, dlogits, logits, old_logp,
            ref_logp, completion_ids, advantages, completion_mask, lse,
            temperature, beta, eps_low, eps_high, *dloss.stride(), L, N, **
            kwargs)
        dlogits[:, -1, :] = 0
        return (dlogits, None, None, None, None, None, None, None, None,
            None, None)
