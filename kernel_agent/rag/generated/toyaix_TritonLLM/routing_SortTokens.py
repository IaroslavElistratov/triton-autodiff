# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/toyaix/TritonLLM
# Source-Files: tritonllm/triton_kernels/routing.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_wam_m4a5/TritonLLM-main/tritonllm/triton_kernels/routing.py
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
from functools import partial

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def _cdiv_pow2(n, log2_k):
    return n + ((1 << log2_k) - 1) >> log2_k


@triton.jit
def _expt_data_compute(Hist, MDTileStarts, tile_starts_stridem, MDTileInfo,
    tile_info_stridem, first_tile_dim_log2, SIZES: tl.constexpr, BLOCK: tl.
    constexpr):
    pid = tl.program_id(0)
    expt_id = pid // SIZES
    buff_id = pid % SIZES
    MDTileStarts += buff_id * tile_starts_stridem
    MDTileInfo += buff_id * tile_info_stridem
    n_tokens = tl.load(Hist + expt_id)
    tile_dim_log2 = first_tile_dim_log2 + buff_id
    n_blocks = _cdiv_pow2(n_tokens, tile_dim_log2)
    tile_off = tl.load(MDTileStarts + expt_id)
    MDTileInfo += tile_off
    for block_off in range(0, n_blocks, BLOCK):
        block_offs = block_off + tl.arange(0, BLOCK)
        data = (block_offs << 16) + expt_id
        tl.store(MDTileInfo + block_offs, data, mask=block_offs < n_blocks)


@triton.jit
def _expt_data_memset(Hist, n_expts_tot, MDStarts, tile_starts_stridem,
    MDTileInfo, first_tile_dim_log2, SIZES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid <= SIZES:
        MDStarts += pid * tile_starts_stridem
        x_tile = tl.zeros([BLOCK], dtype=MDStarts.dtype.element_ty)
        Tile_ptrs = MDStarts + tl.arange(0, BLOCK)
        tile_dim_log2 = tl.where(pid == 0, 0, pid + first_tile_dim_log2 - 1)
        for i in range(0, n_expts_tot + 1, BLOCK):
            offs_n = tl.arange(0, BLOCK) + i
            mask_n0 = offs_n < n_expts_tot
            hist_tok = tl.load(Hist + offs_n, mask=mask_n0, other=0)
            hist_tile = _cdiv_pow2(hist_tok, tile_dim_log2)
            tile_starts = tl.cumsum(hist_tile, 0) + x_tile
            x_tile += tl.sum(hist_tile, 0).to(MDStarts.dtype.element_ty)
            tl.store(Tile_ptrs, tile_starts - hist_tile)
            Tile_ptrs += BLOCK
    else:
        pid -= SIZES + 1
        TileInfoOut = MDTileInfo + pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(TileInfoOut, 4294967295)


@triton.jit
def _combined_routing_compute(GatherIndx, ScatterIndx, GateScal, ExptScal,
    ExptIndx, PartialOffs, stride_pm, stride_pn, TokensStart, n_tokens,
    BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr, Hist, MDTileStarts,
    tile_starts_stridem, MDTileInfo, tile_info_stridem, first_tile_dim_log2,
    SIZES: tl.constexpr, BLOCK: tl.constexpr, blocks2a):
    pid = tl.program_id(0)
    if pid < blocks2a:
        _expt_data_compute(Hist, MDTileStarts, tile_starts_stridem,
            MDTileInfo, tile_info_stridem, first_tile_dim_log2, SIZES, BLOCK)
    else:
        pid -= blocks2a
        _routing_compute_indx(pid, GatherIndx, ScatterIndx, GateScal,
            ExptScal, ExptIndx, PartialOffs, stride_pm, stride_pn,
            TokensStart, n_tokens, BLOCK_M, N_EXPTS_ACT)


@triton.jit
def _combined_routing_memset(Indx, size, sentinel, BLOCK: tl.constexpr,
    ExpertHist, FinalExpertOffs, hist_size, n_expts_tot, PartialHist,
    shape_pm, stride_pm, stride_pn, MDStarts, tile_starts_stridem, blocks1a,
    MDTileInfo, first_tile_dim_log2, SIZES: tl.constexpr, BLOCK_A: tl.
    constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """
    This kernel essentially combines 6 different pieces of functionality,
    statically branching on the value of tl.program_id(0) to decide which
    codepath to take.

        pid == 0:                                  create the token cumsum
        1 <= pid <= SIZES:                         create a tile cumsum
        SIZES < pid < blocks1a:                    initialise MDTileInfo to 0xffffffff
        blocks1a <= pid < blocks1a + n_expts_tot:  compute_indx_offs
        pid == blocks1a + n_expts_tot:             compute_expt_offs
        pid > blocks1a + n_expts_tot:              initialise Indx to sentinel

    As each of these is a relatively trivial workload, launching them from
    this single trampoline is beneficial as they can execute on different
    streaming multiprocesses in parallel.
    """
    pid = tl.program_id(0)
    if pid < blocks1a:
        _expt_data_memset(ExpertHist, n_expts_tot, MDStarts,
            tile_starts_stridem, MDTileInfo, first_tile_dim_log2, SIZES,
            BLOCK_A)
    elif pid == n_expts_tot + blocks1a:
        _routing_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size,
            BLOCK_N)
    elif pid < n_expts_tot + blocks1a:
        _routing_compute_indx_offs(PartialHist, shape_pm, stride_pm,
            stride_pn, BLOCK_M, pid - blocks1a)
    else:
        offs = (pid - n_expts_tot - blocks1a - 1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < size
        tl.store(Indx + offs, sentinel, mask=mask)


@triton.jit
def _routing_compute_expt_offs(ExpertHist, FinalExpertOffs, hist_size,
    BLOCK_N: tl.constexpr):
    loop_iterations = (hist_size + BLOCK_N - 1) // BLOCK_N
    x = tl.zeros([BLOCK_N], ExpertHist.dtype.element_ty)
    for i in range(loop_iterations):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < hist_size
        hist2 = tl.load(ExpertHist + offs_n, mask=mask_n)
        tok_starts = tl.cumsum(hist2, 0) - hist2 + x
        x += tl.sum(hist2, 0)
        tl.store(FinalExpertOffs + offs_n, tok_starts, mask=mask_n)
        offs_n += BLOCK_N


@triton.jit
def _routing_compute_indx(pid_m, GatherIndx, ScatterIndx, GateScal,
    ExptScal, ExptIndx, PartialOffs, stride_pm, stride_pn, TokensStart,
    n_tokens, BLOCK_M: tl.constexpr, N_EXPTS_ACT: tl.constexpr):
    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    n_gates = n_tokens * N_EXPTS_ACT
    tl.static_assert(N_EXPTS_ACT * BLOCK_M <= 32768)
    local_offs = tl.arange(0, N_EXPTS_ACT * BLOCK_M)
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + local_offs
    expert = tl.load(ExptIndx + offs, mask=offs < n_gates, other=-1).to(tl.
        uint32)
    kv_pairs = (expert << 16 | local_offs).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    expert = kv_pairs >> 16
    offs = pid_m * BLOCK_M * N_EXPTS_ACT + (kv_pairs & 65535)
    mask = expert != 65535
    gate_scal = tl.load(ExptScal + offs, mask=mask)
    x = kv_pairs & 4294901760 | 1
    expts_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = expts_and_inclusive_run_lengths - 1 & 65535
    gates = tl.load(PartialOffs + pid_m * stride_pm + expert * stride_pn,
        mask=mask)
    gates += tl.load(TokensStart + expert, mask=mask)
    gates += exclusive_run_lengths
    tl.store(ScatterIndx + offs, gates, mask=mask)
    tl.store(GatherIndx + gates, offs, mask=mask)
    tl.store(GateScal + gates, gate_scal, mask=mask)


@triton.jit
def _routing_compute_indx_offs(PartialHist, shape_pm, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr, expt_id):
    offs_m = tl.arange(0, BLOCK_M)
    curr_sum = 0
    for _ in range(0, shape_pm, BLOCK_M):
        offs = offs_m * stride_pm + expt_id * stride_pn
        curr = tl.load(PartialHist + offs, mask=offs_m < shape_pm)
        out = tl.cumsum(curr, 0) + curr_sum
        curr_sum += tl.sum(curr, 0)
        tl.store(PartialHist + offs, out - curr, mask=offs_m < shape_pm)
        offs_m += BLOCK_M


def _compute_expt_data_internal(expt_hist, n_expts_tot, n_gates):
    MEMSET_BLOCK = 512
    HIST2_BLOCK_M = 512
    device = expt_hist.device
    n_expts_tot = n_expts_tot
    cdiv = triton.cdiv
    block_m_log2_end = 9 if is_hip() else 8
    block_m_num = block_m_log2_end - block_m_log2_start
    if n_gates <= n_expts_tot:
        max_n_tiles = n_gates
    else:
        max_n_tiles = n_expts_tot - 1 - (n_expts_tot - n_gates - 1
            ) // 2 ** block_m_log2_start
    pad = lambda x: cdiv(x, MEMSET_BLOCK) * MEMSET_BLOCK
    dtype = torch.int32
    token_offs_combined = torch.empty((block_m_num + 1, pad(n_expts_tot + 1
        )), dtype=dtype, device=device)
    token_offs_raw = token_offs_combined[0][:n_expts_tot + 1]
    token_offs_pad = token_offs_combined[1:]
    block_pid_map = torch.empty((block_m_num, pad(max_n_tiles)), dtype=
        dtype, device=device)
    memset_grid = torch.numel(block_pid_map) // MEMSET_BLOCK
    token_offs_pad = token_offs_pad[:, :n_expts_tot + 1]
    block_pid_map = block_pid_map[:, :max_n_tiles]
    blocks1 = memset_grid + block_m_num + 1
    blocks2 = n_expts_tot * block_m_num
    return (token_offs_combined, token_offs_raw, token_offs_pad,
        block_pid_map, blocks1, blocks2, MEMSET_BLOCK, HIST2_BLOCK_M,
        block_m_log2_start, block_m_num)


def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        return None
    return active_driver.get_current_target()


@constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == 'hip'


# Forward method (kernel launch code)
def _SortTokens_forward(ctx, expt_scal, expt_indx, n_expts_tot, bitmatrix):
    HIST_BLOCK_M = 32
    INDX_OFFS_BLOCK_M = 512
    MEMSET_BLOCK = 1024
    cdiv = triton.cdiv
    device = expt_scal.device
    dtype = expt_scal.dtype
    n_tokens_raw, _ = bitmatrix.shape
    n_tokens_pad, n_expts_act = expt_scal.shape
    n_gates_pad = n_tokens_pad * n_expts_act
    hist, partial_hist = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
    hist = hist[:n_expts_tot]
    assert hist.dtype == torch.int32
    expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
    combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32, device=
        device)
    topk_indx = combined_indx[:n_gates_pad]
    gate_indx = combined_indx[n_gates_pad:]
    gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)
    (token_offs_combined, token_offs_raw, token_offs_pad, block_pid_map,
        blocks1a, blocks2a, MEMSET_BLOCK_A, HIST2_BLOCK_M,
        block_m_log2_start, block_m_num) = _compute_expt_data_internal(hist,
        n_expts_tot, n_gates_pad)
    blocks1b = cdiv(n_gates_pad * 2, MEMSET_BLOCK) + n_expts_tot + 1
    blocks2b = cdiv(n_tokens_pad, HIST_BLOCK_M)
    _combined_routing_memset[blocks1a + blocks1b,](combined_indx, 
        n_gates_pad * 2, -1, MEMSET_BLOCK, hist, expt_offs, hist.shape[0],
        n_expts_tot, partial_hist, partial_hist.shape[0], partial_hist.
        stride(0), partial_hist.stride(1), token_offs_combined,
        token_offs_combined.stride(0), blocks1a, block_pid_map,
        block_m_log2_start, SIZES=block_m_num, BLOCK_A=MEMSET_BLOCK_A,
        BLOCK_N=512, BLOCK_M=INDX_OFFS_BLOCK_M)
    indx_offs = partial_hist
    _combined_routing_compute[blocks2a + blocks2b,](topk_indx, gate_indx,
        gate_scal, expt_scal, expt_indx, indx_offs, indx_offs.stride(0),
        indx_offs.stride(1), expt_offs, n_tokens_raw, HIST_BLOCK_M,
        n_expts_act, hist, token_offs_pad, token_offs_pad.stride(0),
        block_pid_map, block_pid_map.stride(0), block_m_log2_start,
        block_m_num, HIST2_BLOCK_M, blocks2a)
    ctx.n_tokens_raw = n_tokens_raw
    ctx.n_tokens_pad = n_tokens_pad
    ctx.n_expts_act = n_expts_act
    ctx.save_for_backward(gate_indx)
    return (hist, topk_indx, gate_indx, gate_scal, token_offs_raw,
        token_offs_pad, block_pid_map)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
def _SortTokens_backward(ctx, _0, _1, _2, dgate_scal, _3, _4, _5):
    gate_indx, = ctx.saved_tensors
    dgate_scal = dgate_scal[gate_indx]
    dgate_scal = dgate_scal.reshape(ctx.n_tokens_pad, ctx.n_expts_act)
    return dgate_scal, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class SortTokens(torch.autograd.Function):

    @staticmethod
    def forward(ctx, expt_scal, expt_indx, n_expts_tot, bitmatrix):
        HIST_BLOCK_M = 32
        INDX_OFFS_BLOCK_M = 512
        MEMSET_BLOCK = 1024
        cdiv = triton.cdiv
        device = expt_scal.device
        dtype = expt_scal.dtype
        n_tokens_raw, _ = bitmatrix.shape
        n_tokens_pad, n_expts_act = expt_scal.shape
        n_gates_pad = n_tokens_pad * n_expts_act
        hist, partial_hist = bitmatrix.sum(partials_block_size=HIST_BLOCK_M)
        hist = hist[:n_expts_tot]
        assert hist.dtype == torch.int32
        expt_offs = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
        combined_indx = torch.empty(n_gates_pad * 2, dtype=torch.int32,
            device=device)
        topk_indx = combined_indx[:n_gates_pad]
        gate_indx = combined_indx[n_gates_pad:]
        gate_scal = torch.empty(n_gates_pad, dtype=dtype, device=device)
        (token_offs_combined, token_offs_raw, token_offs_pad, block_pid_map,
            blocks1a, blocks2a, MEMSET_BLOCK_A, HIST2_BLOCK_M,
            block_m_log2_start, block_m_num) = _compute_expt_data_internal(hist
            , n_expts_tot, n_gates_pad)
        blocks1b = cdiv(n_gates_pad * 2, MEMSET_BLOCK) + n_expts_tot + 1
        blocks2b = cdiv(n_tokens_pad, HIST_BLOCK_M)
        _combined_routing_memset[blocks1a + blocks1b,](combined_indx, 
            n_gates_pad * 2, -1, MEMSET_BLOCK, hist, expt_offs, hist.shape[
            0], n_expts_tot, partial_hist, partial_hist.shape[0],
            partial_hist.stride(0), partial_hist.stride(1),
            token_offs_combined, token_offs_combined.stride(0), blocks1a,
            block_pid_map, block_m_log2_start, SIZES=block_m_num, BLOCK_A=
            MEMSET_BLOCK_A, BLOCK_N=512, BLOCK_M=INDX_OFFS_BLOCK_M)
        indx_offs = partial_hist
        _combined_routing_compute[blocks2a + blocks2b,](topk_indx,
            gate_indx, gate_scal, expt_scal, expt_indx, indx_offs,
            indx_offs.stride(0), indx_offs.stride(1), expt_offs,
            n_tokens_raw, HIST_BLOCK_M, n_expts_act, hist, token_offs_pad,
            token_offs_pad.stride(0), block_pid_map, block_pid_map.stride(0
            ), block_m_log2_start, block_m_num, HIST2_BLOCK_M, blocks2a)
        ctx.n_tokens_raw = n_tokens_raw
        ctx.n_tokens_pad = n_tokens_pad
        ctx.n_expts_act = n_expts_act
        ctx.save_for_backward(gate_indx)
        return (hist, topk_indx, gate_indx, gate_scal, token_offs_raw,
            token_offs_pad, block_pid_map)

    @staticmethod
    def backward(ctx, _0, _1, _2, dgate_scal, _3, _4, _5):
        gate_indx, = ctx.saved_tensors
        dgate_scal = dgate_scal[gate_indx]
        dgate_scal = dgate_scal.reshape(ctx.n_tokens_pad, ctx.n_expts_act)
        return dgate_scal, None, None, None
