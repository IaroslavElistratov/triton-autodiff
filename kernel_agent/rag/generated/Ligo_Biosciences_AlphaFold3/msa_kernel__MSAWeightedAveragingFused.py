# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/Ligo-Biosciences/AlphaFold3
# Source-Files: src/models/components/msa_kernel.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_od7mclz5/AlphaFold3-main/src/models/components/msa_kernel.py
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
from math import exp
from math import log

def nearest_pow2(n: int):
    power = math.ceil(math.log2(n))
    next_power_of_two = 2 ** power
    return next_power_of_two


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def MSAFwdFused(v_si_ptr, b_ij_ptr, g_si_ptr, output_ptr, vw_ptr,
    logsumexp_ptr, C_hidden, N_head, C_LEN_POW2: tl.constexpr, RES_LEN_POW2:
    tl.constexpr, SEQ_LEN: tl.constexpr, RES_LEN: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr):
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)
    z_off = pid_z.to(tl.int64)
    h_off = pid_h.to(tl.int64)
    i_off = pid_i.to(tl.int64) * BLOCK_SIZE_ROW
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    offs_c = tl.arange(0, C_LEN_POW2)
    log2_e = 1.44269504089
    prev_row_max = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    new_row_max = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    l = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    for j in range(0, RES_LEN, BLOCK_SIZE_COL):
        offs_j = j + tl.arange(0, BLOCK_SIZE_COL)
        b_offs = z_off * RES_LEN * RES_LEN * N_head + offs_i[:, None
            ] * RES_LEN * N_head + offs_j[None, :] * N_head + h_off
        ij_mask = (offs_i < RES_LEN)[:, None] & (offs_j < RES_LEN)[None, :]
        b = tl.load(b_ij_ptr + b_offs, ij_mask, -INF)
        new_row_max = tl.maximum(tl.max(b, axis=1, keep_dims=True),
            prev_row_max)
        w = tl.exp2(log2_e * (b - new_row_max))
        l *= tl.exp2(log2_e * (prev_row_max - new_row_max))
        l += tl.sum(w, axis=1, keep_dims=True)
        for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
            for ch in range(0, C_hidden, 1):
                offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
                si_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + 
                    offs_s[None, :] * RES_LEN * N_head * C_hidden + offs_i[
                    :, None] * N_head * C_hidden + h_off * C_hidden + ch)
                sj_off = (z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + 
                    offs_s[None, :] * RES_LEN * N_head * C_hidden + offs_j[
                    :, None] * N_head * C_hidden + h_off * C_hidden + ch)
                si_mask = (offs_s < SEQ_LEN)[None, :] & (offs_i < RES_LEN)[
                    :, None]
                sj_mask = (offs_s < SEQ_LEN)[None, :] & (offs_j < RES_LEN)[
                    :, None]
                v = tl.load(v_si_ptr + sj_off, sj_mask, 0)
                vw = tl.load(output_ptr + si_off, si_mask, 0)
                vw = vw * tl.exp2(log2_e * (prev_row_max - new_row_max))
                vw = tl.dot(w, v, acc=vw)
                tl.store(output_ptr + si_off, vw, si_mask)
        prev_row_max = new_row_max
    for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
        for ch in range(0, C_hidden, 1):
            offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
            si_off = z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + offs_s[
                None, :] * RES_LEN * N_head * C_hidden + offs_i[:, None
                ] * N_head * C_hidden + h_off * C_hidden + ch
            si_mask = (offs_s < SEQ_LEN)[None, :] & (offs_i < RES_LEN)[:, None]
            g = tl.load(g_si_ptr + si_off, si_mask, 0)
            g = tl.sigmoid(g)
            vw = tl.load(output_ptr + si_off, si_mask, 0)
            vw = vw / l
            out = g * vw
            tl.store(output_ptr + si_off, out, si_mask)
            tl.store(vw_ptr + si_off, vw, si_mask)
    lse_off = z_off * RES_LEN * N_head + offs_i[:, None] * N_head + h_off
    lse_mask = (offs_i < RES_LEN)[:, None]
    tl.store(logsumexp_ptr + lse_off, new_row_max + tl.log(l), lse_mask)


# Forward method (kernel launch code)
def __MSAWeightedAveragingFused_forward(ctx, v, b, g):
    """
        Fuse the softmax and linear combination step of MSA.
        """
    n_batches, n_seq, n_res, no_heads, c_hidden = v.shape
    out = torch.empty((n_batches, n_seq, n_res, no_heads * c_hidden),
        device=g.device, dtype=g.dtype)
    vw = torch.empty((n_batches, n_seq, n_res, no_heads * c_hidden), device
        =g.device, dtype=g.dtype)
    logsumexp = torch.empty((n_batches, n_res, 1, no_heads), device=g.
        device, dtype=g.dtype)
    BLOCK_SIZE_ROW = 32
    BLOCK_SIZE_COL = 16
    BLOCK_SIZE_SEQ = 16
    n_res_pow2 = nearest_pow2(n_res)
    c_hidden_pow2 = nearest_pow2(c_hidden)
    grid = n_batches, no_heads, triton.cdiv(n_res, BLOCK_SIZE_ROW)
    MSAFwdFused[grid](v, b, g, out, vw, logsumexp, c_hidden, no_heads,
        c_hidden_pow2, n_res_pow2, n_seq, n_res, BLOCK_SIZE_ROW,
        BLOCK_SIZE_SEQ, BLOCK_SIZE_COL)
    ctx.save_for_backward(vw, v, b, g, logsumexp)
    ctx.n_batches = n_batches
    ctx.no_heads = no_heads
    ctx.n_seq = n_seq
    ctx.n_res = n_res
    ctx.c_hidden = c_hidden
    return out


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def MSABwdFused(b_ij_ptr, logsumexp_ptr, N_head, RES_LEN: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr):
    pid_zh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_z = pid_zh // N_head
    pid_h = pid_zh % N_head
    log2_e = 1.44269504089
    z_off = pid_z.to(tl.int64)
    h_off = pid_h.to(tl.int64)
    i_off = pid_i.to(tl.int64) * BLOCK_SIZE_ROW
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    lse_off = z_off * RES_LEN * N_head + offs_i[:, None] * N_head + h_off
    lse_mask = (offs_i < RES_LEN)[:, None]
    logsumexp = tl.load(logsumexp_ptr + lse_off, lse_mask, 0)
    for j in range(0, RES_LEN, BLOCK_SIZE_COL):
        offs_j = j + tl.arange(0, BLOCK_SIZE_COL)
        b_offs = z_off * RES_LEN * RES_LEN * N_head + offs_i[:, None
            ] * RES_LEN * N_head + offs_j[None, :] * N_head + h_off
        ij_mask = (offs_i < RES_LEN)[:, None] & (offs_j < RES_LEN)[None, :]
        b = tl.load(b_ij_ptr + b_offs, ij_mask, -INF)
        b = tl.exp2(log2_e * (b - logsumexp))
        tl.store(b_ij_ptr + b_offs, b, ij_mask)


# Backward method (kernel launch code)
def __MSAWeightedAveragingFused_backward(ctx, do):
    """
        TODO: Currently experiencing some precision issues with the Softmax, but computation
        otherwise is correct.
        """
    split_heads = nn.Unflatten(dim=-1, unflattened_size=(ctx.no_heads, ctx.
        c_hidden))
    vw, v, b, g, logsumexp = ctx.saved_tensors
    assert do.is_contiguous()
    BLOCK_SIZE_ROW = 64
    BLOCK_SIZE_COL = 64
    BLOCK_SIZE_SEQ = 16
    n_res_pow2 = nearest_pow2(ctx.n_res)
    c_hidden_pow2 = nearest_pow2(ctx.c_hidden)
    grid = ctx.n_batches * ctx.no_heads, triton.cdiv(ctx.n_res, BLOCK_SIZE_ROW)
    MSABwdFused[grid](b, logsumexp, ctx.no_heads, ctx.n_res, BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL)
    G = F.sigmoid(g)
    A = split_heads(do) * G
    dv = torch.einsum('bsrhc,brRh->bsRhc', A, b)
    C = torch.einsum('brshc,bsRhc->brRhc', torch.transpose(A, dim0=1, dim1=
        2), v)
    A_vwT = A * torch.einsum('bsrhc,brRh->bsRhc', v, torch.transpose(b,
        dim0=1, dim1=2))
    A_vwT = torch.sum(A_vwT, 1).unsqueeze(2)
    db = b * torch.sum(C - A_vwT, -1)
    dg = G * (1 - G) * split_heads(do * vw)
    return dv, db, dg


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _MSAWeightedAveragingFused(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v, b, g):
        """
        Fuse the softmax and linear combination step of MSA.
        """
        n_batches, n_seq, n_res, no_heads, c_hidden = v.shape
        out = torch.empty((n_batches, n_seq, n_res, no_heads * c_hidden),
            device=g.device, dtype=g.dtype)
        vw = torch.empty((n_batches, n_seq, n_res, no_heads * c_hidden),
            device=g.device, dtype=g.dtype)
        logsumexp = torch.empty((n_batches, n_res, 1, no_heads), device=g.
            device, dtype=g.dtype)
        BLOCK_SIZE_ROW = 32
        BLOCK_SIZE_COL = 16
        BLOCK_SIZE_SEQ = 16
        n_res_pow2 = nearest_pow2(n_res)
        c_hidden_pow2 = nearest_pow2(c_hidden)
        grid = n_batches, no_heads, triton.cdiv(n_res, BLOCK_SIZE_ROW)
        MSAFwdFused[grid](v, b, g, out, vw, logsumexp, c_hidden, no_heads,
            c_hidden_pow2, n_res_pow2, n_seq, n_res, BLOCK_SIZE_ROW,
            BLOCK_SIZE_SEQ, BLOCK_SIZE_COL)
        ctx.save_for_backward(vw, v, b, g, logsumexp)
        ctx.n_batches = n_batches
        ctx.no_heads = no_heads
        ctx.n_seq = n_seq
        ctx.n_res = n_res
        ctx.c_hidden = c_hidden
        return out

    @staticmethod
    def backward(ctx, do):
        """
        TODO: Currently experiencing some precision issues with the Softmax, but computation
        otherwise is correct.
        """
        split_heads = nn.Unflatten(dim=-1, unflattened_size=(ctx.no_heads,
            ctx.c_hidden))
        vw, v, b, g, logsumexp = ctx.saved_tensors
        assert do.is_contiguous()
        BLOCK_SIZE_ROW = 64
        BLOCK_SIZE_COL = 64
        BLOCK_SIZE_SEQ = 16
        n_res_pow2 = nearest_pow2(ctx.n_res)
        c_hidden_pow2 = nearest_pow2(ctx.c_hidden)
        grid = ctx.n_batches * ctx.no_heads, triton.cdiv(ctx.n_res,
            BLOCK_SIZE_ROW)
        MSABwdFused[grid](b, logsumexp, ctx.no_heads, ctx.n_res,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL)
        G = F.sigmoid(g)
        A = split_heads(do) * G
        dv = torch.einsum('bsrhc,brRh->bsRhc', A, b)
        C = torch.einsum('brshc,bsRhc->brRhc', torch.transpose(A, dim0=1,
            dim1=2), v)
        A_vwT = A * torch.einsum('bsrhc,brRh->bsRhc', v, torch.transpose(b,
            dim0=1, dim1=2))
        A_vwT = torch.sum(A_vwT, 1).unsqueeze(2)
        db = b * torch.sum(C - A_vwT, -1)
        dg = G * (1 - G) * split_heads(do * vw)
        return dv, db, dg
