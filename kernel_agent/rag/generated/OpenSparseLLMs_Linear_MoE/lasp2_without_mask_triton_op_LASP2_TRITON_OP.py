# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/OpenSparseLLMs/Linear-MoE
# Source-Files: linear_moe/sequence_modeling/lasp2/lasp2_without_mask_triton_op.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_kyohjmxq/Linear-MoE-main/linear_moe/sequence_modeling/lasp2/lasp2_without_mask_triton_op.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _fwd_kernel(Q, Out, KV, n: tl.constexpr, d: tl.constexpr, e: tl.
    constexpr, BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr, E_FBLOCK: tl.
    constexpr, CBLOCK: tl.constexpr, NUM_CBLOCK: tl.constexpr):
    off_bh = tl.program_id(0)
    off_nc = tl.program_id(1)
    off_e = tl.program_id(2)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK
    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e + e_offset
    Q_block_ptr = Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(
        0, d)[None, :]
    O_block_ptr = Out + o_offset + tl.arange(0, CBLOCK)[:, None
        ] * e + tl.arange(0, E_FBLOCK)[None, :]
    KV_block_ptr = KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(
        0, E_FBLOCK)[None, :]
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q = tl.load(Q_block_ptr).to(tl.float32)
    qkv = tl.dot(q, kv)
    tl.store(O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_m_calculate(K, V, KV, n: tl.constexpr, d: tl.constexpr, e: tl.
    constexpr, BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr, D_FBLOCK: tl.
    constexpr, E_FBLOCK: tl.constexpr, NUM_FBLOCK: tl.constexpr, CBLOCK: tl
    .constexpr, NUM_CBLOCK: tl.constexpr):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK
    block_offset = off_block * BLOCK
    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e
    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    K_trans_block_ptr = K + k_offset + k_block_offset + d_offset + tl.arange(
        0, CBLOCK)[None, :] * d + tl.arange(0, D_FBLOCK)[:, None]
    V_block_ptr = V + v_offset + v_block_offset + e_offset + tl.arange(0,
        CBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    KV_block_ptr = (KV + kv_offset + kv_block_offset + d_offset * e +
        e_offset + tl.arange(0, D_FBLOCK)[:, None] * e + tl.arange(0,
        E_FBLOCK)[None, :])
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for _ in range(NUM_CBLOCK):
        k_trans = tl.load(K_trans_block_ptr).to(tl.float32)
        v = tl.load(V_block_ptr).to(tl.float32)
        kv += tl.dot(k_trans, v)
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_m_cumsum(KV, d: tl.constexpr, e: tl.constexpr, NUM_BLOCK: tl.
    constexpr, D_FBLOCK: tl.constexpr, E_FBLOCK: tl.constexpr):
    off_bh = tl.program_id(0)
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    KV_block_ptr = KV + kv_offset + d_offset * e + e_offset + tl.arange(0,
        D_FBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for _ in range(NUM_BLOCK):
        kv_current = tl.load(KV_block_ptr).to(tl.float32)
        kv = kv + kv_current
        KV_block_ptr += d * e
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


@triton.jit
def _fwd_m_update(KV, GKV, d: tl.constexpr, e: tl.constexpr, NUM_BLOCK: tl.
    constexpr, D_FBLOCK: tl.constexpr, E_FBLOCK: tl.constexpr):
    off_bh = tl.program_id(0)
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    gkv_offset = off_bh * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    KV_block_ptr = KV + kv_offset + d_offset * e + e_offset + tl.arange(0,
        D_FBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    GKV_block_ptr = GKV + gkv_offset + tl.arange(0, d)[:, None
        ] * e + tl.arange(0, E_FBLOCK)[None, :]
    kv = tl.load(GKV_block_ptr).to(tl.float32)
    for _ in range(NUM_BLOCK):
        kv_current = tl.load(KV_block_ptr).to(tl.float32)
        tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))
        kv = kv + kv_current
        KV_block_ptr += d * e


def lasp2_forward(q, k, v, kv, KV, BLOCK=128, CBLOCK=64):
    b, h, n, d = q.shape
    e = v.shape[-1]
    o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
    NUM_BLOCK = q.shape[2] // BLOCK
    NUM_CBLOCK = BLOCK // CBLOCK
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0
    with torch.cuda.device(q.device.index):
        grid = b * h, NUM_FBLOCK, NUM_FBLOCK
        _fwd_m_update[grid](kv, KV, d, e, NUM_BLOCK=NUM_BLOCK, D_FBLOCK=
            D_FBLOCK, E_FBLOCK=E_FBLOCK)
    with torch.cuda.device(q.device.index):
        grid = b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK
        _fwd_kernel[grid](q, o, kv, n, d, e, BLOCK=BLOCK, NUM_BLOCK=
            NUM_BLOCK, E_FBLOCK=E_FBLOCK, CBLOCK=CBLOCK, NUM_CBLOCK=NUM_CBLOCK)
    return o


def prepare_m(q, k, v, BLOCK=128, CBLOCK=64):
    b, h, n, d = q.shape
    e = v.shape[-1]
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0
    grid = b * h, NUM_FBLOCK, NUM_FBLOCK
    NUM_BLOCK = q.shape[2] // BLOCK
    NUM_CBLOCK = BLOCK // CBLOCK
    kv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32,
        device=q.device)
    with torch.cuda.device(q.device.index):
        grid = b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK
        _fwd_m_calculate[grid](k, v, kv, n, d, e, BLOCK=BLOCK, NUM_BLOCK=
            NUM_BLOCK, D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK, NUM_FBLOCK=
            NUM_FBLOCK, CBLOCK=CBLOCK, NUM_CBLOCK=NUM_CBLOCK)
        grid = b * h, NUM_FBLOCK, NUM_FBLOCK
        _fwd_m_cumsum[grid](kv, d, e, NUM_BLOCK=NUM_BLOCK, D_FBLOCK=
            D_FBLOCK, E_FBLOCK=E_FBLOCK)
    return kv


# Forward method (kernel launch code)
def _LASP2_TRITON_OP_forward(ctx, q, k, v, cp_group: torch.distributed.
    ProcessGroup):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    n = q.shape[-2]
    if n > 128:
        BLOCK = 256
        CBLOCK = 64
    else:
        BLOCK = min(n, 128)
        CBLOCK = min(n, 64)
    kv = prepare_m(q, k, v, BLOCK, CBLOCK)
    cp_rank = get_context_parallel_rank()
    cp_world_size = get_context_parallel_world_size()
    m = kv[:, :, -1].contiguous()
    all_m = torch.empty([cp_world_size, *m.shape], dtype=kv.dtype, device=
        kv.device)
    torch.distributed.all_gather_into_tensor(all_m, m, group=cp_group)
    if cp_rank > 0:
        prefixsum_m = all_m[:cp_rank]
        KV = torch.sum(prefixsum_m, dim=0)
    else:
        KV = torch.zeros_like(m)
    o = lasp2_forward(q, k, v, kv, KV, BLOCK, CBLOCK)
    ctx.save_for_backward(q, k, v, kv)
    ctx.cp_group = cp_group
    ctx.cp_world_size = cp_world_size
    ctx.BLOCK = BLOCK
    ctx.CBLOCK = CBLOCK
    return o


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def _bwd_dm_calculate(Q, DO, DKV, n: tl.constexpr, d: tl.constexpr, e: tl.
    constexpr, BLOCK: tl.constexpr, NUM_BLOCK: tl.constexpr, D_FBLOCK: tl.
    constexpr, E_FBLOCK: tl.constexpr, NUM_FBLOCK: tl.constexpr, CBLOCK: tl
    .constexpr, NUM_CBLOCK: tl.constexpr):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_de = tl.program_id(2)
    off_d = off_de // NUM_FBLOCK
    off_e = off_de % NUM_FBLOCK
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    o_block_offset = block_offset * e
    kv_block_offset = off_block * d * e
    qk_offset = off_bh * n * d
    o_offset = off_bh * n * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    DKV_block_ptr = (DKV + kv_offset + kv_block_offset + d_offset * e +
        e_offset + tl.arange(0, D_FBLOCK)[:, None] * e + tl.arange(0,
        E_FBLOCK)[None, :])
    Q_trans_block_ptr = Q + qk_offset + qk_block_offset + d_offset + tl.arange(
        0, CBLOCK)[None, :] * d + tl.arange(0, D_FBLOCK)[:, None]
    DO_block_ptr = DO + o_offset + o_block_offset + e_offset + tl.arange(0,
        CBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for _ in range(NUM_CBLOCK):
        do = tl.load(DO_block_ptr).to(tl.float32)
        q_trans = tl.load(Q_trans_block_ptr).to(tl.float32)
        dkv += tl.dot(q_trans, do)
        DO_block_ptr += CBLOCK * e
        Q_trans_block_ptr += CBLOCK * d
    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_dm_cumsum(DKV, d: tl.constexpr, e: tl.constexpr, NUM_BLOCK: tl.
    constexpr, D_FBLOCK: tl.constexpr, E_FBLOCK: tl.constexpr):
    off_bh = tl.program_id(0)
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    DKV_block_ptr = (DKV + kv_offset + d_offset * e + e_offset + NUM_BLOCK *
        d * e + tl.arange(0, D_FBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK
        )[None, :])
    dkv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)
    for _ in range(NUM_BLOCK - 1, -1, -1):
        DKV_block_ptr -= d * e
        dkv_current = tl.load(DKV_block_ptr).to(tl.float32)
        dkv = dkv + dkv_current
    DKV_block_ptr += NUM_BLOCK * d * e
    tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))


@triton.jit
def _bwd_dm_update(DKV, GDKV, d: tl.constexpr, e: tl.constexpr, NUM_BLOCK:
    tl.constexpr, D_FBLOCK: tl.constexpr, E_FBLOCK: tl.constexpr):
    off_bh = tl.program_id(0)
    off_d = tl.program_id(1)
    off_e = tl.program_id(2)
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e
    gkv_offset = off_bh * d * e
    d_offset = off_d * D_FBLOCK
    e_offset = off_e * E_FBLOCK
    DKV_block_ptr = (DKV + kv_offset + d_offset * e + e_offset + NUM_BLOCK *
        d * e + tl.arange(0, D_FBLOCK)[:, None] * e + tl.arange(0, E_FBLOCK
        )[None, :])
    GDKV_block_ptr = GDKV + gkv_offset + d_offset * e + tl.arange(0, D_FBLOCK)[
        :, None] * e + tl.arange(0, e)[None, :]
    dkv = tl.load(GDKV_block_ptr).to(tl.float32)
    for _ in range(NUM_BLOCK - 1, -1, -1):
        DKV_block_ptr -= d * e
        dkv_current = tl.load(DKV_block_ptr).to(tl.float32)
        tl.store(DKV_block_ptr, dkv.to(DKV_block_ptr.dtype.element_ty))
        dkv = dkv + dkv_current


@triton.jit
def _bwd_kernel(K, V, DO, DQ, DK, DV, KV, DKV, n: tl.constexpr, d: tl.
    constexpr, e: tl.constexpr, BLOCK: tl.constexpr, NUM_BLOCK: tl.
    constexpr, D_FBLOCK: tl.constexpr, E_FBLOCK: tl.constexpr, CBLOCK: tl.
    constexpr, NUM_CBLOCK: tl.constexpr):
    off_bh = tl.program_id(0)
    off_nc = tl.program_id(1)
    off_de = tl.program_id(2)
    off_n = off_nc // NUM_CBLOCK
    off_c = off_nc % NUM_CBLOCK
    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    d_offset = off_de * D_FBLOCK
    e_offset = off_de * E_FBLOCK
    qk_offset = off_bh * n * d + (n_offset + c_offset) * d
    v_offset = off_bh * n * e + (n_offset + c_offset) * e
    o_offset = off_bh * n * e + (n_offset + c_offset) * e
    kv_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e
    kv_trans_offset = off_bh * (NUM_BLOCK + 1) * d * e + off_n * d * e
    DO_block_ptr = DO + o_offset + tl.arange(0, CBLOCK)[:, None
        ] * e + tl.arange(0, e)[None, :]
    DQ_block_ptr = DQ + qk_offset + d_offset + tl.arange(0, CBLOCK)[:, None
        ] * d + tl.arange(0, D_FBLOCK)[None, :]
    KV_trans_block_ptr = KV + kv_trans_offset + d_offset * e + tl.arange(0,
        D_FBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    kv_trans = tl.load(KV_trans_block_ptr).to(tl.float32)
    do = tl.load(DO_block_ptr).to(tl.float32)
    dq = tl.dot(do, kv_trans)
    tl.store(DQ_block_ptr, dq.to(DQ_block_ptr.dtype.element_ty))
    DK_trans_block_ptr = DK + qk_offset + d_offset + tl.arange(0, CBLOCK)[
        None, :] * d + tl.arange(0, D_FBLOCK)[:, None]
    V_trans_block_ptr = V + v_offset + tl.arange(0, CBLOCK)[None, :
        ] * e + tl.arange(0, e)[:, None]
    DKV_block_ptr = DKV + kv_offset + d_offset * e + tl.arange(0, D_FBLOCK)[
        :, None] * e + tl.arange(0, e)[None, :]
    v_trans = tl.load(V_trans_block_ptr).to(tl.float32)
    dkv = tl.load(DKV_block_ptr).to(tl.float32)
    dk_trans = tl.dot(dkv, v_trans)
    tl.store(DK_trans_block_ptr, dk_trans.to(DK_trans_block_ptr.dtype.
        element_ty))
    DKV_block_ptr_ = DKV + kv_offset + e_offset + tl.arange(0, d)[:, None
        ] * e + tl.arange(0, E_FBLOCK)[None, :]
    K_block_ptr = K + qk_offset + tl.arange(0, CBLOCK)[:, None
        ] * d + tl.arange(0, d)[None, :]
    DV_block_ptr = DV + v_offset + e_offset + tl.arange(0, CBLOCK)[:, None
        ] * e + tl.arange(0, E_FBLOCK)[None, :]
    k = tl.load(K_block_ptr).to(tl.float32)
    dkv_ = tl.load(DKV_block_ptr_).to(tl.float32)
    dv = tl.dot(k, dkv_)
    tl.store(DV_block_ptr, dv.to(DV_block_ptr.dtype.element_ty))


def lasp2_backward(q, k, v, do, kv, dkv, DKV, BLOCK=128, CBLOCK=64):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    grid = q.shape[0] * q.shape[1], 1
    b, h, n, d = q.shape
    e = v.shape[-1]
    NUM_BLOCK = n // BLOCK
    assert BLOCK % CBLOCK == 0
    NUM_CBLOCK = BLOCK // CBLOCK
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0
    with torch.cuda.device(q.device.index):
        grid = b * h, NUM_FBLOCK, NUM_FBLOCK
        _bwd_dm_update[grid](dkv, DKV, d, e, NUM_BLOCK=NUM_BLOCK, D_FBLOCK=
            D_FBLOCK, E_FBLOCK=E_FBLOCK)
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0
    with torch.cuda.device(q.device.index):
        grid = b * h, NUM_BLOCK * NUM_CBLOCK, NUM_FBLOCK
        _bwd_kernel[grid](k, v, do, dq, dk, dv, kv, dkv, n, d, e, BLOCK=
            BLOCK, NUM_BLOCK=NUM_BLOCK, D_FBLOCK=D_FBLOCK, E_FBLOCK=
            E_FBLOCK, CBLOCK=CBLOCK, NUM_CBLOCK=NUM_CBLOCK)
    return dq, dk, dv


def prepare_dm(q, k, v, do, BLOCK=128, CBLOCK=64):
    b, h, n, d = q.shape
    e = v.shape[-1]
    NUM_BLOCK = n // BLOCK
    assert BLOCK % CBLOCK == 0
    NUM_CBLOCK = BLOCK // CBLOCK
    dkv = torch.empty((b, h, NUM_BLOCK + 1, d, e), dtype=torch.float32,
        device=q.device)
    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK
    assert d % NUM_FBLOCK == 0
    assert e % NUM_FBLOCK == 0
    with torch.cuda.device(q.device.index):
        grid = b * h, NUM_BLOCK, NUM_FBLOCK * NUM_FBLOCK
        _bwd_dm_calculate[grid](q, do, dkv, n, d, e, BLOCK=BLOCK, NUM_BLOCK
            =NUM_BLOCK, D_FBLOCK=D_FBLOCK, E_FBLOCK=E_FBLOCK, NUM_FBLOCK=
            NUM_FBLOCK, CBLOCK=CBLOCK, NUM_CBLOCK=NUM_CBLOCK)
        grid = b * h, NUM_FBLOCK, NUM_FBLOCK
        _bwd_dm_cumsum[grid](dkv, d, e, NUM_BLOCK=NUM_BLOCK, D_FBLOCK=
            D_FBLOCK, E_FBLOCK=E_FBLOCK)
    return dkv


# Backward method (kernel launch code)
def _LASP2_TRITON_OP_backward(ctx, do):
    q, k, v, kv = ctx.saved_tensors
    do = do.contiguous()
    cp_group = ctx.cp_group
    cp_world_size = ctx.cp_world_size
    BLOCK = ctx.BLOCK
    CBLOCK = ctx.CBLOCK
    dkv = prepare_dm(q, k, v, do, BLOCK, CBLOCK)
    cp_rank = get_context_parallel_rank()
    dm = dkv[:, :, -1].contiguous()
    all_dm = torch.empty([cp_world_size, *dm.shape], dtype=dkv.dtype,
        device=dkv.device)
    torch.distributed.all_gather_into_tensor(all_dm, dm, group=cp_group)
    if cp_rank < cp_world_size - 1:
        suffixsum_dm = all_dm[cp_rank + 1:]
        DKV = torch.sum(suffixsum_dm, dim=0)
    else:
        DKV = torch.zeros_like(dm)
    dq, dk, dv = lasp2_backward(q, k, v, do, kv, dkv, DKV, BLOCK, CBLOCK)
    return dq, dk, dv, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LASP2_TRITON_OP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cp_group: torch.distributed.ProcessGroup):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        n = q.shape[-2]
        if n > 128:
            BLOCK = 256
            CBLOCK = 64
        else:
            BLOCK = min(n, 128)
            CBLOCK = min(n, 64)
        kv = prepare_m(q, k, v, BLOCK, CBLOCK)
        cp_rank = get_context_parallel_rank()
        cp_world_size = get_context_parallel_world_size()
        m = kv[:, :, -1].contiguous()
        all_m = torch.empty([cp_world_size, *m.shape], dtype=kv.dtype,
            device=kv.device)
        torch.distributed.all_gather_into_tensor(all_m, m, group=cp_group)
        if cp_rank > 0:
            prefixsum_m = all_m[:cp_rank]
            KV = torch.sum(prefixsum_m, dim=0)
        else:
            KV = torch.zeros_like(m)
        o = lasp2_forward(q, k, v, kv, KV, BLOCK, CBLOCK)
        ctx.save_for_backward(q, k, v, kv)
        ctx.cp_group = cp_group
        ctx.cp_world_size = cp_world_size
        ctx.BLOCK = BLOCK
        ctx.CBLOCK = CBLOCK
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, kv = ctx.saved_tensors
        do = do.contiguous()
        cp_group = ctx.cp_group
        cp_world_size = ctx.cp_world_size
        BLOCK = ctx.BLOCK
        CBLOCK = ctx.CBLOCK
        dkv = prepare_dm(q, k, v, do, BLOCK, CBLOCK)
        cp_rank = get_context_parallel_rank()
        dm = dkv[:, :, -1].contiguous()
        all_dm = torch.empty([cp_world_size, *dm.shape], dtype=dkv.dtype,
            device=dkv.device)
        torch.distributed.all_gather_into_tensor(all_dm, dm, group=cp_group)
        if cp_rank < cp_world_size - 1:
            suffixsum_dm = all_dm[cp_rank + 1:]
            DKV = torch.sum(suffixsum_dm, dim=0)
        else:
            DKV = torch.zeros_like(dm)
        dq, dk, dv = lasp2_backward(q, k, v, do, kv, dkv, DKV, BLOCK, CBLOCK)
        return dq, dk, dv, None, None
