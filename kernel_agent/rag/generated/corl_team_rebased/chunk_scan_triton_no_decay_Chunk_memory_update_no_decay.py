# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/corl-team/rebased
# Source-Files: flash_linear_attention/fla/ops/triton/gla/block_parallel/inter_chunk_contribution/chunk_scan_triton_no_decay.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_m2n9_3ro/rebased-main/flash_linear_attention/fla/ops/triton/gla/block_parallel/inter_chunk_contribution/chunk_scan_triton_no_decay.py
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

@triton.jit
def _fwd_recurrence(S, O, NUM_BLOCK, D_MODEL_K: tl.constexpr, D_MODEL_V: tl
    .constexpr, BLOCK_MODEL: tl.constexpr):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)
    S = (S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d *
        D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] *
        D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]
        )
    O = (O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d *
        D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] *
        D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None,
        :] + D_MODEL_K * D_MODEL_V)
    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    acc += tl.load(S)
    S += D_MODEL_K * D_MODEL_V
    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V
    for i in range(NUM_BLOCK - 2):
        S_i = tl.load(S)
        acc = acc + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V


# Forward method (kernel launch code)
@custom_fwd
@contiguous
def _Chunk_memory_update_no_decay_forward(ctx, to_add):
    B, H, N, D_k, D_v = to_add.shape
    output = torch.empty_like(to_add)
    BLOCK_MODEL = 32
    assert D_k % 32 == 0
    assert D_v % 32 == 0
    grid = B * H, D_k // BLOCK_MODEL, D_v // BLOCK_MODEL
    ctx.grid = grid
    ctx.BLOCK_MODEL = BLOCK_MODEL
    _fwd_recurrence[grid](to_add, output, D_MODEL_K=D_k, D_MODEL_V=D_v,
        NUM_BLOCK=N, BLOCK_MODEL=BLOCK_MODEL)
    output[:, :, 0] = 0
    ctx.save_for_backward(output)
    return output.to(to_add.dtype)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _bwd_recurrence(S, DS, NUM_BLOCK, NUM_SPLIT_K, NUM_SPLIT_V, D_MODEL_K:
    tl.constexpr, D_MODEL_V: tl.constexpr, BLOCK_MODEL: tl.constexpr):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)
    S = (S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d *
        D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] *
        D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None,
        :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V)
    DS = (DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d *
        D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] *
        D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None,
        :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V)
    Dacc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1):
        DS_i = tl.load(DS)
        Dacc += DS_i
        tl.store(S, Dacc.to(S.dtype.element_ty))
        S -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V


# Backward method (kernel launch code)
@custom_bwd
@contiguous
def _Chunk_memory_update_no_decay_backward(ctx, DO):
    output, = ctx.saved_tensors
    B, H, N, D_k, D_v = output.shape
    num_block = N
    BLOCK_MODEL = 32
    grid = B * H, D_k // BLOCK_MODEL, D_v // BLOCK_MODEL
    _bwd_recurrence[grid](output, DO, NUM_BLOCK=num_block, NUM_SPLIT_K=D_k //
        BLOCK_MODEL, NUM_SPLIT_V=D_v // BLOCK_MODEL, D_MODEL_K=D_k,
        D_MODEL_V=D_v, BLOCK_MODEL=BLOCK_MODEL)
    output[:, :, -1] = 0
    return output


# ============================================================
# autograd.Function Class Definition
# ============================================================

class Chunk_memory_update_no_decay(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    @contiguous
    def forward(ctx, to_add):
        B, H, N, D_k, D_v = to_add.shape
        output = torch.empty_like(to_add)
        BLOCK_MODEL = 32
        assert D_k % 32 == 0
        assert D_v % 32 == 0
        grid = B * H, D_k // BLOCK_MODEL, D_v // BLOCK_MODEL
        ctx.grid = grid
        ctx.BLOCK_MODEL = BLOCK_MODEL
        _fwd_recurrence[grid](to_add, output, D_MODEL_K=D_k, D_MODEL_V=D_v,
            NUM_BLOCK=N, BLOCK_MODEL=BLOCK_MODEL)
        output[:, :, 0] = 0
        ctx.save_for_backward(output)
        return output.to(to_add.dtype)

    @staticmethod
    @custom_bwd
    @contiguous
    def backward(ctx, DO):
        output, = ctx.saved_tensors
        B, H, N, D_k, D_v = output.shape
        num_block = N
        BLOCK_MODEL = 32
        grid = B * H, D_k // BLOCK_MODEL, D_v // BLOCK_MODEL
        _bwd_recurrence[grid](output, DO, NUM_BLOCK=num_block, NUM_SPLIT_K=
            D_k // BLOCK_MODEL, NUM_SPLIT_V=D_v // BLOCK_MODEL, D_MODEL_K=
            D_k, D_MODEL_V=D_v, BLOCK_MODEL=BLOCK_MODEL)
        output[:, :, -1] = 0
        return output
