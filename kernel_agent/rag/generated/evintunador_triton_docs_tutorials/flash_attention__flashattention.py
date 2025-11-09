# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/evintunador/triton_docs_tutorials
# Source-Files: 09_flash_attention/flash_attention.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_4ctjy3wy/triton_docs_tutorials-main/09_flash_attention/flash_attention.py
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
from triton import cdiv

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _attn_fwd_inner(Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
    block_index_QO, softmax_scale, stride_K_N, stride_V_N, BLOCK_SIZE_QO:
    tl.constexpr, BLOCK_SIZE_KV: tl.constexpr, DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr, N: tl.constexpr,
    Dh: tl.constexpr):
    """
    arrows indicate direction of this pid's for loop; each arrow is a different PID
                N of K & V
                ------------>
                ------------>
    N of Q      ------------>
                ------------>
                ------------>
    but if we actually take into account the causal mask then really it's more like
                N of K & V
                >
                --->
    N of Q      ------>
                --------->
                ------------>
    and to get even more accurate, we do the diagonal in our second call of this inner kernel
                N of K & V
                x
                   x
    N of Q            x
                         x
                            x
    and then the first call gets all the parts below the diagonal
                N of K & V
                
                -->
    N of Q      ----->
                -------->
                ----------->
    """
    if DIAGONAL:
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
    else:
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO
    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo
    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        mask_KV_N = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.0)
        S = tl.dot(Q, K_T) * softmax_scale
        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= offsets_KV_N[None, :]
            S += tl.where(causal_mask, 0, -1000000.0)
        M_new = tl.maximum(M, tl.max(S, axis=1))
        S -= M_new[:, None]
        P = tl.exp2(S)
        L_new = tl.sum(P, axis=1)
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.0)
        O = O * alpha[:, None]
        O = tl.dot(P, V, acc=O)
        M = M_new
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV
    return O, L, M


@triton.autotune([triton.Config({'BLOCK_SIZE_QO': BLOCK_SIZE_QO,
    'BLOCK_SIZE_KV': BLOCK_SIZE_KV}, num_stages=num_stages, num_warps=
    num_warps) for BLOCK_SIZE_QO in [16] for BLOCK_SIZE_KV in [16] for
    num_stages in [3] for num_warps in [4]], key=['Dh'])
@triton.jit
def attn_fwd(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, softmax_scale, stride_Q_B,
    stride_Q_H, stride_Q_N, stride_Q_Dh, stride_K_B, stride_K_H, stride_K_N,
    stride_K_Dh, stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh, stride_LSE_B,
    stride_LSE_H, stride_LSE_N, B, H: tl.constexpr, N: tl.constexpr, Dh: tl
    .constexpr, BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr):
    rln2: tl.constexpr = 1.4426950408889634
    softmax_scale *= rln2
    """
    let's show that e^x = 2^(x * rln2)
    e^x = (2^(log_2(e)))^x since a = 2^log_2(a)
    then using the power rule
    (2^(log_2(e)))^x = 2^(x * log_2(e))
    fundamental property of logarithm is log_2(e) = 1/log_e(2)
    therefore e^x = 2^(x * 1/log_e(2)) 
    AKA e^x = 2^(x * rln2)
    then later in the backward pass we'll have to remember to account for this in the gradient
    """
    tl.static_assert(BLOCK_SIZE_KV <= Dh)
    block_index_QO = tl.program_id(0)
    index_BH = tl.program_id(1)
    index_B = index_BH // H
    index_H = index_BH % H
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)
    Q_offsets = offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :
        ] * stride_Q_Dh
    K_T_offsets = offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :
        ] * stride_K_N
    V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :
        ] * stride_V_Dh
    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.0)
    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1000000.0, dtype=tl.float32)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)
    O, L, M = _attn_fwd_inner(Q, O, L, M, K_ptr, V_ptr, K_T_offsets,
        V_offsets, block_index_QO, softmax_scale, stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV, False, offsets_QO_N, offsets_KV_N, N, Dh)
    O, L, M = _attn_fwd_inner(Q, O, L, M, K_ptr, V_ptr, K_T_offsets,
        V_offsets, block_index_QO, softmax_scale, stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV, True, offsets_QO_N, offsets_KV_N, N, Dh)
    O = O / L[:, None]
    LSE = M + tl.math.log2(L)
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)
    O_offsets = offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :
        ] * stride_O_Dh
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None])


# Forward method (kernel launch code)
def __flashattention_forward(ctx, q, k, v, scale):
    assert q.shape == k.shape == v.shape
    assert q.shape[-1
        ] <= 128, f'flash attention only supports head dimension of 128 less but got {q.shape[-1]}'
    B, H, N, Dh = q.shape
    assert q.device == k.device and q.device == v.device
    assert q.dtype == k.dtype == v.dtype == torch.float32
    O = torch.empty_like(q)
    LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE_QO']), B * H)
    """
        imagine for a launch grid of (3, 2) wiwth 3 SMs that can each hold 2 PIDs
        we'd have PIDs:
        [0, 0] \\ SM0
        [1, 0] /
        [2, 0] \\ SM1
        [0, 1] /
        [1, 1] \\ SM2
        [2, 1] /
        """
    attn_fwd[grid](q, k, v, O, LSE, scale, q.stride(0), q.stride(1), q.
        stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2), k.
        stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), O.
        stride(0), O.stride(1), O.stride(2), O.stride(3), LSE.stride(0),
        LSE.stride(1), LSE.stride(2), B, H, N, Dh)
    ctx.save_for_backward(q, k, v, O, LSE)
    ctx.grid = grid
    ctx.B, ctx.H, ctx.N, ctx.Dh = B, H, N, Dh
    ctx.scale = scale
    return O


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _attn_backward_KV(K, V, dLdK, dLdV, Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
    stride_N, stride_Dh, H, N, Dh: tl.constexpr, BLOCK_SIZE_ROW: tl.
    constexpr, BLOCK_SIZE_COL: tl.constexpr, start_ROW, start_COL,
    num_steps, scale, ln2: tl.constexpr, rln2: tl.constexpr, MASK: tl.constexpr
    ):
    """
    this sub-kernel will be looking at a specific chunk of K & V , where we call
    the sequence length of K & V the columns of our NxN attention matrix,
    and iterating through rows of Q's sequence length to calculate that 
    chunk of dLdK and dLdV
                    N of K & V
               |    |   |   |   |
               |    |   |   |   |
    N of Q     |    |   |   |   |
               |    |   |   |   |
              \\|/  \\|/ \\|/ \\|/ \\|/
    arrows indicate direction of this pid's for loop; each arrow is a different PID
    """
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)
    Q_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :
        ] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :
        ] * stride_Dh
    for block_idx in range(num_steps):
        mask_N = offsets_ROW < N
        Q_T = tl.load(Q_ptr + Q_T_offsets, mask=mask_N[None, :], other=0.0)
        LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_N, other=0.0)
        dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask_N[:, None], other=0.0
            )
        Delta = tl.load(Delta_ptr + offsets_ROW, mask=mask_N, other=0.0)
        S_T = tl.dot(K, Q_T)
        P_T = tl.exp2(S_T - LSE[None, :])
        if MASK:
            mask = offsets_COL[:, None] <= offsets_ROW[None, :]
            P_T = tl.where(mask, P_T, 0.0)
        dLdV = tl.dot(P_T, dLdO, acc=dLdV)
        dLdP_T = tl.dot(V, tl.trans(dLdO))
        dLdS_T = P_T * (dLdP_T - Delta[None, :]) * ln2
        dLdK = tl.dot(dLdS_T, tl.trans(Q_T), acc=dLdK)
        offsets_ROW += BLOCK_SIZE_ROW
        Q_ptr += BLOCK_SIZE_ROW * stride_N
        dLdO_ptr += BLOCK_SIZE_ROW * stride_N
    return dLdK, dLdV


@triton.jit
def _attn_backward_Q(dLdQ, Q, dLdO, LSE, K_ptr, V_ptr, Delta_ptr, stride_N,
    stride_Dh, H, N, Dh: tl.constexpr, BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr, start_ROW, start_COL, num_steps, scale,
    ln2: tl.constexpr, rln2: tl.constexpr, MASK: tl.constexpr):
    """
    this sub-kernel will be looking at a specific chunk of Q and iterating through
    rows of K & V to calculate that chunk of dLdQ
    I say "rows" of K and V but really we refer to them as colums since we're thinking
    not in terms of the (B, H, N, D) shaped matrices but rather the (B, H, N, N) shaped
    attention logits, where the first N are split up by "BLOCK_SIZE_ROW" and the second N
    is split up by "BLOCK_SIZE_COL"
                    N of K & V
               ------------------->
               ------------------->
    N of Q     ------------------->
               ------------------->
               ------------------->
    arrows indicate direction of this pid's for loop; each arrow is a different PID
    """
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)
    K_and_V_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :
        ] * stride_N
    Delta = tl.load(Delta_ptr + offsets_ROW, mask=offsets_ROW < N, other=0.0)
    for block_idx in range(num_steps):
        K_T = tl.load(K_ptr + K_and_V_T_offsets, mask=(offsets_COL < N)[
            None, :], other=0.0)
        V_T = tl.load(V_ptr + K_and_V_T_offsets, mask=(offsets_COL < N)[
            None, :], other=0.0)
        S = tl.dot(Q, K_T)
        P = tl.exp2(S - LSE)
        if MASK:
            mask = offsets_ROW[:, None] >= offsets_COL[None, :]
            P = tl.where(mask, P, 0.0)
        dLdP = tl.dot(dLdO, V_T)
        dLdS = P * (dLdP - Delta[:, None]) * ln2
        dLdQ += tl.dot(dLdS, tl.trans(K_T))
        offsets_COL += BLOCK_SIZE_COL
        K_ptr += BLOCK_SIZE_COL * stride_N
        V_ptr += BLOCK_SIZE_COL * stride_N
    return dLdQ


@triton.autotune([triton.Config({'BLOCK_SIZE_MACRO': BLOCK_SIZE_MACRO,
    'BLOCK_SIZE_MICRO': BLOCK_SIZE_MICRO}, num_stages=num_stages, num_warps
    =num_warps) for BLOCK_SIZE_MICRO in [16] for BLOCK_SIZE_MACRO in [32] for
    num_stages in [3] for num_warps in [4] if BLOCK_SIZE_MACRO >
    BLOCK_SIZE_MICRO], key=['Dh'])
@triton.jit
def attn_backward(Q_ptr, K_ptr, V_ptr, dLdO_ptr, dLdQ_ptr, dLdK_ptr,
    dLdV_ptr, LSE_ptr, Delta_ptr, scale, stride_B, stride_H, stride_N,
    stride_Dh, H, N, Dh: tl.constexpr, BLOCK_SIZE_MICRO: tl.constexpr,
    BLOCK_SIZE_MACRO: tl.constexpr):
    ln2: tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // H
    idx_head = idx_batch_head % H
    batch_head_jump = idx_batch * stride_B + idx_head * stride_H
    Q_ptr += batch_head_jump
    K_ptr += batch_head_jump
    V_ptr += batch_head_jump
    dLdO_ptr += batch_head_jump
    dLdQ_ptr += batch_head_jump
    dLdK_ptr += batch_head_jump
    dLdV_ptr += batch_head_jump
    batch_head_jump = idx_batch_head * N
    LSE_ptr += batch_head_jump
    Delta_ptr += batch_head_jump
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)
    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO
    pid = tl.program_id(0)
    start_COL = pid * BLOCK_SIZE_COL_1
    start_ROW = start_COL
    num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1
    offsets_COL_1 = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    offsets_Dh = tl.arange(0, Dh)
    KV_offsets = offsets_COL_1[:, None] * stride_N + offsets_Dh[None, :
        ] * stride_Dh
    KV_mask = offsets_COL_1[:, None] < N
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.0)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.0)
    K *= scale * rln2
    dLdK = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    dLdK, dLdV = _attn_backward_KV(K, V, dLdK, dLdV, Q_ptr, dLdO_ptr,
        LSE_ptr, Delta_ptr, stride_N, stride_Dh, H, N, Dh, BLOCK_SIZE_ROW_1,
        BLOCK_SIZE_COL_1, start_ROW, start_COL, num_steps, scale, ln2, rln2,
        MASK=True)
    start_ROW += BLOCK_SIZE_COL_1
    N_adj = tl.cdiv(N, BLOCK_SIZE_COL_1) * BLOCK_SIZE_COL_1
    num_steps = (N_adj - start_ROW) // BLOCK_SIZE_ROW_1
    dLdK, dLdV = _attn_backward_KV(K, V, dLdK, dLdV, Q_ptr, dLdO_ptr,
        LSE_ptr, Delta_ptr, stride_N, stride_Dh, H, N, Dh, BLOCK_SIZE_ROW_1,
        BLOCK_SIZE_COL_1, start_ROW, start_COL, num_steps, scale, ln2, rln2,
        MASK=False)
    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK, mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV, mask=KV_mask)
    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO
    start_ROW = pid * BLOCK_SIZE_ROW_2
    start_COL = start_ROW
    num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    QO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :
        ] * stride_Dh
    mask_ROW = offsets_ROW < N
    Q = tl.load(Q_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.0)
    Q *= scale * rln2
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.0)
    LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_ROW, other=0.0)[:, None]
    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, Dh], dtype=tl.float32)
    dLdQ = _attn_backward_Q(dLdQ, Q, dLdO, LSE, K_ptr, V_ptr, Delta_ptr,
        stride_N, stride_Dh, H, N, Dh, BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps, scale, ln2, rln2, MASK=True)
    end_COL = start_COL
    start_COL = 0
    num_steps = end_COL // BLOCK_SIZE_COL_2
    dLdQ = _attn_backward_Q(dLdQ, Q, dLdO, LSE, K_ptr, V_ptr, Delta_ptr,
        stride_N, stride_Dh, H, N, Dh, BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps, scale, ln2, rln2, MASK=False)
    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ, mask=mask_ROW[:, None])


@triton.autotune([triton.Config({'PRE_BLOCK_SIZE_ROW': PRE_BLOCK_SIZE_ROW},
    num_stages=num_stages, num_warps=num_warps) for PRE_BLOCK_SIZE_ROW in [
    32] for num_stages in [3] for num_warps in [4]], key=['Dh'])
@triton.jit
def attn_backward_preprocess(O_ptr, dLdO_ptr, Delta_ptr, stride_O_B,
    stride_O_H, stride_O_N, stride_O_Dh, stride_dLdO_B, stride_dLdO_H,
    stride_dLdO_N, stride_dLdO_Dh, stride_Delta_B, stride_Delta_H,
    stride_Delta_N, N, Dh: tl.constexpr, PRE_BLOCK_SIZE_ROW: tl.constexpr):
    """the job of this kernel is to pre-compute Delta since Delta is used by both of the following two kernels"""
    index_BH = tl.program_id(1)
    row = tl.program_id(0)
    row_offsets = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    col_offsets = tl.arange(0, Dh)
    mask = row_offsets < N
    O_ptr += index_BH * stride_O_H
    O_offsets = row_offsets[:, None] * stride_O_N + col_offsets[None, :
        ] * stride_O_Dh
    O = tl.load(O_ptr + O_offsets, mask=mask[:, None], other=0.0)
    dLdO_ptr += index_BH * stride_dLdO_H
    dLdO_offsets = row_offsets[:, None] * stride_dLdO_N + col_offsets[None, :
        ] * stride_dLdO_Dh
    dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask[:, None], other=0.0)
    Delta = tl.sum(dLdO.to(tl.float32) * O.to(tl.float32), axis=1)
    Delta_ptr += index_BH * stride_Delta_H
    tl.store(Delta_ptr + row_offsets, Delta, mask=mask)


# Backward method (kernel launch code)
def __flashattention_backward(ctx, dLdO):
    q, k, v, O, LSE = ctx.saved_tensors
    grid = ctx.grid
    scale = ctx.scale
    B, H, N, Dh = ctx.B, ctx.H, ctx.N, ctx.Dh
    dLdq = torch.empty_like(q)
    dLdk = torch.empty_like(k)
    dLdv = torch.empty_like(v)
    dLdO = dLdO.contiguous()
    assert q.stride() == k.stride() == v.stride() == O.stride() == dLdO.stride(
        )
    Delta = torch.empty_like(LSE)
    pre_grid = lambda meta: (triton.cdiv(N, meta['PRE_BLOCK_SIZE_ROW']), B * H)
    attn_backward_preprocess[pre_grid](O, dLdO, Delta, O.stride(0), O.
        stride(1), O.stride(2), O.stride(3), dLdO.stride(0), dLdO.stride(1),
        dLdO.stride(2), dLdO.stride(3), Delta.stride(0), Delta.stride(1),
        Delta.stride(2), N, Dh)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_MACRO']), B * H)
    attn_backward[grid](q, k, v, dLdO, dLdq, dLdk, dLdv, LSE, Delta, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), H, N, Dh)
    return dLdq, dLdk, dLdv, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _flashattention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        assert q.shape == k.shape == v.shape
        assert q.shape[-1
            ] <= 128, f'flash attention only supports head dimension of 128 less but got {q.shape[-1]}'
        B, H, N, Dh = q.shape
        assert q.device == k.device and q.device == v.device
        assert q.dtype == k.dtype == v.dtype == torch.float32
        O = torch.empty_like(q)
        LSE = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        grid = lambda args: (triton.cdiv(N, args['BLOCK_SIZE_QO']), B * H)
        """
        imagine for a launch grid of (3, 2) wiwth 3 SMs that can each hold 2 PIDs
        we'd have PIDs:
        [0, 0] \\ SM0
        [1, 0] /
        [2, 0] \\ SM1
        [0, 1] /
        [1, 1] \\ SM2
        [2, 1] /
        """
        attn_fwd[grid](q, k, v, O, LSE, scale, q.stride(0), q.stride(1), q.
            stride(2), q.stride(3), k.stride(0), k.stride(1), k.stride(2),
            k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3), LSE.stride(
            0), LSE.stride(1), LSE.stride(2), B, H, N, Dh)
        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.grid = grid
        ctx.B, ctx.H, ctx.N, ctx.Dh = B, H, N, Dh
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        grid = ctx.grid
        scale = ctx.scale
        B, H, N, Dh = ctx.B, ctx.H, ctx.N, ctx.Dh
        dLdq = torch.empty_like(q)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)
        dLdO = dLdO.contiguous()
        assert q.stride() == k.stride() == v.stride() == O.stride(
            ) == dLdO.stride()
        Delta = torch.empty_like(LSE)
        pre_grid = lambda meta: (triton.cdiv(N, meta['PRE_BLOCK_SIZE_ROW']),
            B * H)
        attn_backward_preprocess[pre_grid](O, dLdO, Delta, O.stride(0), O.
            stride(1), O.stride(2), O.stride(3), dLdO.stride(0), dLdO.
            stride(1), dLdO.stride(2), dLdO.stride(3), Delta.stride(0),
            Delta.stride(1), Delta.stride(2), N, Dh)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_MACRO']), B * H)
        attn_backward[grid](q, k, v, dLdO, dLdq, dLdk, dLdv, LSE, Delta,
            scale, q.stride(0), q.stride(1), q.stride(2), q.stride(3), H, N, Dh
            )
        return dLdq, dLdk, dLdv, None
