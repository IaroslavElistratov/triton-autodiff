# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/pengzhangzhi/Open-dLLM
# Source-Files: veomni/distributed/moe/fused_moe.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ww7oku23/Open-dLLM-main/veomni/distributed/moe/fused_moe.py
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
from math import exp

def group_gemm_same_nk(a: torch.Tensor, b: torch.Tensor, cumsum_M: torch.
    Tensor, max_M: int, transpose_a: bool=False, transpose_b: bool=False,
    activation: Optional[ActivationType]=None, save_activation: bool=False,
    c: Optional[torch.Tensor]=None):
    """Grouped gemm for same nk

    Keyword arguments:
    a -- lhs matrixs to be matrix multiplied
    b -- rhs matrixs to be matrix multiplied
    cumsum_M -- matrixs's size cumsum on M
    max_M -- matrixs's max size on M
    transpose_a -- transpose `a` or not
    transpose_b -- transpose `b` or not
    activation -- activation type if needed
    save_activation -- return the activation's input or not
    c -- which tensor accumulate to, c = c + ggemm(a, b)
    """
    if transpose_b:
        G, N, K = b.shape
    else:
        G, K, N = b.shape
    assert not transpose_a, 'Transpose A not tested yet.'
    assert a.dtype in [torch.bfloat16, torch.float16], a.dtype
    assert b.dtype in [torch.bfloat16, torch.float16], b.dtype
    assert a.device == b.device, f'a.device = {a.device}, b.device = {b.device}'
    assert len(cumsum_M) == b.shape[0]
    assert activation is None or activation in list(ActivationType
        ), f'Not implemented: activation is {activation}.'
    assert activation or not save_activation, "Can't save activation since activation type is None"
    assert a.is_contiguous() and b.is_contiguous(
        ), 'Not implemented: Noncontiguous input.'
    c_is_none = c is None
    if c_is_none:
        c = torch.empty((a.shape[1] if transpose_a else a.shape[0], N),
            dtype=a.dtype, device=a.device)
    if save_activation:
        act = torch.empty_like(c)
    with torch.cuda.device(a.device):
        group_gemm_same_nk_kernel[lambda x: (triton.cdiv(max_M, x['BLOCK_M'
            ]) * triton.cdiv(N, x['BLOCK_N']), x['G'])](a_ptr=a, b_ptr=b,
            c_ptr=c, act_ptr=act if save_activation else None, cumsum_M=
            cumsum_M, max_M=max_M, total_M=a.shape[0], G=G, K=K, N=N,
            TRANSPOSE_A=transpose_a, TRANSPOSE_B=transpose_b,
            ACCUMULATE_TO_C=not c_is_none, ACTIVATION=activation,
            SAVE_ACTIVATION=save_activation)
    if save_activation:
        return c, act
    return c


@pretuned(algo_key=algo_key_scaled(['total_M', 'N', 'K'], [5000, 1, 1], [
    'TRANSPOSE_A', 'TRANSPOSE_B']), fallback={'BLOCK_M': 128, 'BLOCK_N': 
    128, 'BLOCK_K': 32, 'GROUP': 8})
@triton.heuristics(values={'N_ALIGNED': lambda args: args['N'] % args[
    'BLOCK_N'] == 0, 'K_ALIGNED': lambda args: args['K'] % args['BLOCK_K'] ==
    0, 'HAS_ACTIVATION': lambda args: args['ACTIVATION'] is not None})
@triton.jit
def group_gemm_same_nk_kernel(a_ptr, b_ptr, c_ptr, act_ptr, cumsum_M, max_M,
    total_M, G: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl
    .constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, TRANSPOSE_A:
    tl.constexpr, TRANSPOSE_B: tl.constexpr, ACCUMULATE_TO_C: tl.constexpr,
    GROUP: tl.constexpr, N_ALIGNED: tl.constexpr, K_ALIGNED: tl.constexpr,
    ACTIVATION: tl.constexpr, HAS_ACTIVATION: tl.constexpr, SAVE_ACTIVATION:
    tl.constexpr):
    m, n = get_pid_mn(tl.program_id(axis=0), max_M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_M + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_M + gid)
    m_size = (gtid_end - gtid_start).to(tl.uint64)
    if m * BLOCK_M >= m_size:
        return
    a_ptr += gtid_start * K
    b_ptr += gid * K * N
    c_ptr += gtid_start * N
    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_am = offs_m % m_size.to(tl.int64)
    offs_bn = offs_n % N
    blk_k = tl.arange(0, BLOCK_K)
    stride_am, stride_ak = (K, 1) if not TRANSPOSE_A else (1, m_size)
    stride_bk, stride_bn = (N, 1) if not TRANSPOSE_B else (1, K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + blk_k[None, :] * stride_ak
        )
    b_ptrs = b_ptr + (blk_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )
    c_ptrs = c_ptr + N * offs_m[:, None] + 1 * offs_n[None, :]
    if ACCUMULATE_TO_C:
        c = load_with_pred_2d(c_ptrs, False, N_ALIGNED, offs_m[:, None] <
            m_size, offs_n[None, :] < N, other=0)
    else:
        c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = load_with_pred_1d(a_ptrs, K_ALIGNED, blk_k[None, :] < K - k *
            BLOCK_K, other=0)
        b = load_with_pred_1d(b_ptrs, K_ALIGNED, blk_k[:, None] < K - k *
            BLOCK_K, other=0)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    if HAS_ACTIVATION:
        c = make_blocked(c, c_ptr.dtype.element_ty)
        if SAVE_ACTIVATION:
            store_with_pred_2d(act_ptr + gtid_start * N + N * offs_m[:,
                None] + offs_n[None, :], c, False, N_ALIGNED, offs_m[:,
                None] < m_size, offs_n[None, :] < N)
        c = activation_fwd(c, ACTIVATION)
    store_with_pred_2d(c_ptrs, c, False, N_ALIGNED, offs_m[:, None] <
        m_size, offs_n[None, :] < N)


@triton.heuristics(values={'N_ALIGNED': lambda args: args['N'] % args[
    'BLOCK_N'] == 0})
@triton.jit
def _moe_gather_kernel(X, Y, index, num_elts_in, num_elts_out, N: tl.
    constexpr, TOPK: tl.constexpr, STRIDE_XM: tl.constexpr, STRIDE_XN: tl.
    constexpr, STRIDE_OM: tl.constexpr, STRIDE_ON: tl.constexpr, STRIDE_IM:
    tl.constexpr, STRIDE_IN: tl.constexpr, BLOCK_N: tl.constexpr, N_ALIGNED:
    tl.constexpr):
    """
    X: m * topk x n
    Y: m x n
    index: m x topk
    code:
        repeated-X: m * topk x n -> reduce(sum_over_topk) -> m x n
        Y: Y[arange(m)] = sum_over_topk(repeated-X[arange(m) * topk])
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)
    block_idx = tl.program_id(axis=1).to(tl.int64)
    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    y = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        x_index = tl.load(index + pid_m.to(tl.int64) * STRIDE_IM + i *
            STRIDE_IN)
        tl.device_assert(x_index < num_elts_in, 'Input OOB')
        x = load_with_pred_1d(X + x_index.to(tl.int64) * STRIDE_XM + n.to(
            tl.int64) * STRIDE_XN, N_ALIGNED, mask=n < N, other=0)
        y += x
    tl.device_assert(pid_m < num_elts_out, 'Output OOB')
    Y = Y + pid_m.to(tl.int64) * STRIDE_OM + n.to(tl.int64) * STRIDE_ON
    store_with_pred_1d(Y, y, N_ALIGNED, mask=n < N)


@triton.heuristics(values={'N_ALIGNED': lambda args: args['N'] % args[
    'BLOCK_N'] == 0})
@triton.jit
def _moe_scatter_kernel(X, O, index, num_elts_in, num_elts_out, N: tl.
    constexpr, TOPK: tl.constexpr, STRIDE_XM: tl.constexpr, STRIDE_XN: tl.
    constexpr, STRIDE_OM: tl.constexpr, STRIDE_ON: tl.constexpr, STRIDE_IM:
    tl.constexpr, STRIDE_IN: tl.constexpr, BLOCK_N: tl.constexpr, N_ALIGNED:
    tl.constexpr):
    """
    X: m x n
    O: m * topk x n
    index: m x topk

    code:
        X: m x n -> repeat -> m x topk x n -> m * topk x n
            X[arange(m) * topk] = X[arange(m)]

        O[index] = X
            O[index[arange(m) * topk]] = X[arange(m) * topk]
    """
    pid_m = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)
    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.device_assert(pid_m < num_elts_in, 'Input OOB.')
    X = X + pid_m * STRIDE_XM + n * STRIDE_XN
    x = load_with_pred_1d(X, N_ALIGNED, mask=n < N, other=0)
    for i in tl.static_range(TOPK):
        o_index = tl.load(index + pid_m * STRIDE_IM + i * STRIDE_IN)
        tl.device_assert(o_index < num_elts_out, 'Output OOB.')
        tmp_index = o_index.to(tl.int64) * STRIDE_OM
        out = O + tmp_index + n * STRIDE_ON
        store_with_pred_1d(out, x, N_ALIGNED, mask=n < N)


def moe_gather(x: torch.Tensor, index: torch.Tensor, out_dtype=None):
    assert x.is_cuda and index.is_cuda
    M, topk = index.shape
    assert x.shape[0] == M * topk
    N = x.shape[1]
    assert x.device == index.device, f'x.device = {x.device}, index.device = {index.device}'
    out_dtype = out_dtype or x.dtype
    out = torch.empty(M, N, dtype=out_dtype, device=x.device)
    grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_N']))
    with torch.cuda.device(x.device):
        _moe_gather_kernel[grid](x, out, index, num_elts_in=M * topk,
            num_elts_out=M, N=N, TOPK=topk, STRIDE_XM=x.stride(0),
            STRIDE_XN=x.stride(1), STRIDE_OM=out.stride(0), STRIDE_ON=out.
            stride(1), STRIDE_IM=index.stride(0), STRIDE_IN=index.stride(1),
            BLOCK_N=1024)
    return out


def moe_scatter(x: torch.Tensor, index: torch.Tensor, out_dtype=None):
    assert x.is_cuda and index.is_cuda
    assert x.shape[0] == index.shape[0]
    assert x.device == index.device, f'x.device = {x.device}, index.device = {index.device}'
    M, N = x.shape
    topk = index.shape[1]
    out_dtype = out_dtype or x.dtype
    out = torch.empty(M * topk, N, dtype=out_dtype, device=x.device)
    assert lambda : index.unique().numel() == M * topk, 'Holes in output?'
    grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_N']))
    with torch.cuda.device(x.device):
        _moe_scatter_kernel[grid](x, out, index, num_elts_in=M,
            num_elts_out=M * topk, N=N, TOPK=topk, STRIDE_XM=x.stride(0),
            STRIDE_XN=x.stride(1), STRIDE_OM=out.stride(0), STRIDE_ON=out.
            stride(1), STRIDE_IM=index.stride(0), STRIDE_IN=index.stride(1),
            BLOCK_N=1024)
    return out


@triton.jit
def activation_fwd(x: tl.tensor, ACTIVATION: tl.constexpr):
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    if ACTIVATION == 'gelu':
        y = gelu(x)
    elif ACTIVATION == 'gelu_new':
        y = gelu_new(x)
    elif ACTIVATION == 'silu':
        y = silu(x)
    else:
        tl.static_assert(False, f'Unsupported activation of {ACTIVATION}')
    return y.to(orig_dtype)


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    x = x.to(tl.float32)
    return x * 0.5 * (1.0 + tl.erf(x * _sqrt1_2))


@triton.jit
def gelu_new(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(_sqrt2pi * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def silu(x):
    """https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html"""
    x = x.to(tl.float32)
    return x * tl.sigmoid(x)


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.
    tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)


@triton.jit
def load_with_pred_2d(ptr, skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr, mask_0: tl.tensor, mask_1: tl.
    tensor, other=0):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_0 and mask_1, other=other)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, mask_0, other=other)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_1, other=other)
    else:
        return tl.load(ptr)


@triton.jit
def store_with_pred_1d(ptr, value, skip_boundary_check: tl.constexpr, mask:
    tl.tensor):
    if not skip_boundary_check:
        tl.store(ptr, value, mask)
    else:
        tl.store(ptr, value)


@triton.jit
def store_with_pred_2d(ptr, value, skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr, mask_0: tl.tensor, mask_1: tl.tensor):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_0 and mask_1)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, mask_0)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_1)
    else:
        tl.store(ptr, value)


@triton.jit
def get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def make_blocked(t: tl.tensor, intermediate_type: tl.dtype) ->tl.tensor:
    """Forcibly convert tensor (from "mma" layout) into "blocked" layout.

    `intermediate_type` affects performance. Usually `tl.bfloat16` or `tl.float16` should be used.
    INTERNALLY `t` IS CONVERTED TO `intermediate_type` AND BACK  SO THE PRECISION CAN DROP.

    ATM Triton does such conversion prior to storing tensor into global memory. This usually doesn't
    matter as we usually only store the accumulator once. However, if we'd like to perform some
    element-wise operation on the accumulator and save both pre-op and post-op results, Triton will
    do the conversion twice, and hence hurt performance.

    In such cases, forcibly convert tensor eagerly can help performance. This is not guaranteed, so
    be sure to benchmark before applying this "optimization".

    NOTE: Once Triton can optimize away multiple layout conversions, this hack should be removed.
    """
    return t.to(intermediate_type).expand_dims(0).reshape(t.shape)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.heuristics(values={'BLOCK_ALIGNED': lambda args: args['num_elts'] %
    args['BLOCK_SIZE'] == 0})
@triton.jit
def _expert_histogram_kernel(out_ptr, x_ptr, num_elts, num_bins,
    NUM_BINS_LAST_UNUSED: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_ALIGNED: tl.constexpr):
    pid = tl.program_id(0)
    in_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = load_with_pred_1d(x_ptr + in_off, BLOCK_ALIGNED, in_off <
        num_elts, NUM_BINS_LAST_UNUSED - 1).to(tl.int32)
    tl.device_assert(data < num_bins or data == NUM_BINS_LAST_UNUSED - 1,
        'Out-of-bound element found.')
    count = tl.histogram(data, NUM_BINS_LAST_UNUSED)
    out_off = tl.arange(0, NUM_BINS_LAST_UNUSED)
    tl.atomic_add(out_ptr + out_off, count, mask=out_off < num_bins, sem=
        'relaxed')


def expert_histogram(input: torch.Tensor, num_bins: int) ->torch.Tensor:
    """Returns histogram of `input`, with bin width 1. Note that for each individual `num_bins`,
    a separate Triton kernel is generated (mostly). So if `num_bins` varies between calls, you
    probably should go for some other histogram method.
    """
    assert input.is_cuda
    assert input.dtype == torch.int32 or input.dtype == torch.int64
    assert input.numel() < (1 << 31) - 1, 'Too many elements.'
    flattened = input.flatten().contiguous()
    NUM_BINS_LAST_UNUSED = triton.next_power_of_2(num_bins + 1)
    out = torch.zeros([num_bins], dtype=torch.int32, device=input.device)
    BLOCK_SIZE = 1024
    num_elts = flattened.numel()
    grid = triton.cdiv(num_elts, BLOCK_SIZE),
    with torch.cuda.device(input.device):
        _expert_histogram_kernel[grid](out_ptr=out, x_ptr=flattened,
            num_elts=num_elts, num_bins=num_bins, NUM_BINS_LAST_UNUSED=
            NUM_BINS_LAST_UNUSED, BLOCK_SIZE=BLOCK_SIZE)
    return out[:num_bins]


# Forward method (kernel launch code)
def _FusedMoeExpertFunction_forward(ctx, num_experts, gate_weights,
    expert_index, hidden_states, fc1_1_weight, fc1_2_weight, fc2_weight):
    splits = expert_histogram(expert_index, num_experts)
    scatter_index = expert_index.flatten().argsort(stable=True).argsort().int(
        ).view(expert_index.shape)
    scatter_output = moe_scatter(hidden_states, scatter_index)
    cumsum_t = torch.cumsum(splits, dim=0)
    fc1_1_output = group_gemm_same_nk(a=scatter_output, b=fc1_1_weight,
        cumsum_M=cumsum_t, max_M=scatter_output.shape[0], transpose_a=False,
        transpose_b=True)
    fc1_2_output = group_gemm_same_nk(a=scatter_output, b=fc1_2_weight,
        cumsum_M=cumsum_t, max_M=scatter_output.shape[0], transpose_a=False,
        transpose_b=True)
    fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
    fc1_activation = fc1_1_activation * fc1_2_output
    reshaped_gate_weight = gate_weights.reshape(-1, 1)
    scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
    scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
    fc1_weighted_output = fc1_activation * scattered_gate_weight
    fc2_output = group_gemm_same_nk(a=fc1_weighted_output, b=fc2_weight,
        cumsum_M=cumsum_t, max_M=scatter_output.shape[0], transpose_a=False,
        transpose_b=True)
    expert_output = moe_gather(fc2_output, scatter_index)
    output = expert_output.reshape(hidden_states.shape)
    ctx.num_experts = num_experts
    ctx.save_for_backward(gate_weights, fc1_1_weight, fc1_2_weight,
        fc2_weight, hidden_states, scatter_index, scatter_output, cumsum_t,
        fc1_1_output, fc1_2_output, fc1_activation, scattered_gate_weight,
        fc1_weighted_output)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@pretuned(algo_key=algo_key_scaled(['M', 'N', 'total_K'], [1, 1, 5000], [
    'TRANSPOSE_A', 'TRANSPOSE_B']), fallback={'BLOCK_M': 128, 'BLOCK_N': 
    128, 'BLOCK_K': 32, 'GROUP': 8})
@triton.heuristics(values={'M_ALIGNED': lambda args: args['M'] % args[
    'BLOCK_M'] == 0, 'N_ALIGNED': lambda args: args['N'] % args['BLOCK_N'] ==
    0})
@triton.jit
def group_gemm_same_mn_kernel(a_ptr, b_ptr, c_ptr, cumsum_K, total_K, G: tl
    .constexpr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr, ACCUMULATE_TO_C: tl.constexpr, GROUP: tl.
    constexpr, M_ALIGNED: tl.constexpr, N_ALIGNED: tl.constexpr):
    m, n = get_pid_mn(tl.program_id(axis=0), M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_K + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_K + gid)
    k = (gtid_end - gtid_start).to(tl.uint64)
    if TRANSPOSE_A:
        a_block_ptr = tl.make_block_ptr(base=a_ptr + gtid_start * M, shape=
            (M, k), strides=(1, M), offsets=(m * BLOCK_M, 0), block_shape=(
            BLOCK_M, BLOCK_K), order=(0, 1))
    else:
        a_block_ptr = tl.make_block_ptr(base=a_ptr + gtid_start * M, shape=
            (M, k), strides=(k, 1), offsets=(m * BLOCK_M, 0), block_shape=(
            BLOCK_M, BLOCK_K), order=(1, 0))
    if TRANSPOSE_B:
        b_block_ptr = tl.make_block_ptr(base=b_ptr + gtid_start * N, shape=
            (k, N), strides=(1, k), offsets=(0, n * BLOCK_N), block_shape=(
            BLOCK_K, BLOCK_N), order=(0, 1))
    else:
        b_block_ptr = tl.make_block_ptr(base=b_ptr + gtid_start * N, shape=
            (k, N), strides=(N, 1), offsets=(0, n * BLOCK_N), block_shape=(
            BLOCK_K, BLOCK_N), order=(1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr + gid * M * N, shape=(M, N),
        strides=(N, 1), offsets=(m * BLOCK_M, n * BLOCK_N), block_shape=(
        BLOCK_M, BLOCK_N), order=(1, 0))
    if k == 0:
        if not ACCUMULATE_TO_C:
            store_block_with_pred_2d(c_block_ptr, tl.zeros((BLOCK_M,
                BLOCK_N), dtype=tl.float32).to(c_block_ptr.dtype.element_ty
                ), M_ALIGNED, N_ALIGNED)
        else:
            pass
        return
    if ACCUMULATE_TO_C:
        out = tl.load(c_block_ptr).to(tl.float32)
    else:
        out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(k.to(tl.int64), BLOCK_K)):
        a = load_block_with_pred_2d(a_block_ptr, M_ALIGNED, False)
        b = load_block_with_pred_2d(b_block_ptr, False, N_ALIGNED)
        out += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    store_block_with_pred_2d(c_block_ptr, out.to(c_block_ptr.dtype.
        element_ty), M_ALIGNED, N_ALIGNED)


@triton.jit
def load_block_with_pred_2d(ptr, skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(0, 1))
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(0,))
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(1,))
    else:
        return tl.load(ptr)


@triton.jit
def store_block_with_pred_2d(ptr, value, skip_boundary_check_0: tl.
    constexpr, skip_boundary_check_1: tl.constexpr):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(0, 1))
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(0,))
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(1,))
    else:
        tl.store(ptr, value)


def group_gemm_same_mn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    cumsum_K: torch.Tensor, max_K: int, transpose_a: bool=False,
    transpose_b: bool=False):
    G, M, N = c.shape
    assert a.dtype in [torch.bfloat16, torch.float16], a.dtype
    assert b.dtype in [torch.bfloat16, torch.float16], b.dtype
    assert a.device == b.device, f'a.device = {a.device}, b.device = {b.device}'
    assert a.device == c.device, f'a.device = {a.device}, c.device = {c.device}'
    assert c is not None, c
    assert len(cumsum_K) == c.shape[0], f'{len(cumsum_K), c.shape}'
    assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous(
        ), 'Not implemented: Noncontiguous input.'
    with torch.cuda.device(a.device):
        group_gemm_same_mn_kernel[lambda x: (triton.cdiv(M, x['BLOCK_M']) *
            triton.cdiv(N, x['BLOCK_N']), x['G'])](a_ptr=a, b_ptr=b, c_ptr=
            c, cumsum_K=cumsum_K, total_K=b.shape[0], G=G, M=M, N=N,
            TRANSPOSE_A=transpose_a, TRANSPOSE_B=transpose_b,
            ACCUMULATE_TO_C=False)


# Backward method (kernel launch code)
def _FusedMoeExpertFunction_backward(ctx, grad_output):
    (gate_weights, fc1_1_weight, fc1_2_weight, fc2_weight, hidden_states,
        scatter_index, scatter_output, cumsum_t, fc1_1_output, fc1_2_output,
        fc1_activation, scattered_gate_weight, fc1_weighted_output
        ) = ctx.saved_tensors
    hidden_dim = grad_output.shape[-1]
    grad_output = grad_output.view(-1, hidden_dim)
    grad_fc2_output = moe_scatter(grad_output, scatter_index)
    grad_fc1_weighted_output = group_gemm_same_nk(a=grad_fc2_output, b=
        fc2_weight, cumsum_M=cumsum_t, max_M=grad_output.shape[0],
        transpose_b=False)
    grad_fc2_weight = None
    if fc2_weight.requires_grad:
        grad_fc2_weight = torch.empty_like(fc2_weight)
        group_gemm_same_mn(a=grad_fc2_output, b=fc1_weighted_output, c=
            grad_fc2_weight, cumsum_K=cumsum_t, max_K=grad_output.shape[0],
            transpose_a=True, transpose_b=False)
    grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight
    grad_scattered_gate_weight = torch.sum(fc1_activation *
        grad_fc1_weighted_output, dim=-1)
    grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
    grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)
    fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
    grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
    grad_fc1_2_output = fc1_1_activation * grad_fc1_activation
    grad_scatter_output_2 = group_gemm_same_nk(a=grad_fc1_2_output, b=
        fc1_2_weight, cumsum_M=cumsum_t, max_M=grad_output.shape[0],
        transpose_b=False)
    grad_fc1_2_weight = None
    if fc1_2_weight.requires_grad:
        grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
        group_gemm_same_mn(a=grad_fc1_2_output, b=scatter_output, c=
            grad_fc1_2_weight, cumsum_K=cumsum_t, max_K=grad_output.shape[0
            ], transpose_a=True, transpose_b=False)
    grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation,
        fc1_1_output)
    grad_scatter_output_1 = group_gemm_same_nk(a=grad_fc1_1_output, b=
        fc1_1_weight, cumsum_M=cumsum_t, max_M=grad_output.shape[0],
        transpose_b=False)
    grad_fc1_1_weight = None
    if fc1_1_weight.requires_grad:
        grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
        group_gemm_same_mn(a=grad_fc1_1_output, b=scatter_output, c=
            grad_fc1_1_weight, cumsum_K=cumsum_t, max_K=grad_output.shape[0
            ], transpose_a=True, transpose_b=False)
    grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
    grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
    grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)
    return (None, grad_gate_weight, None, grad_hidden_states,
        grad_fc1_1_weight, grad_fc1_2_weight, grad_fc2_weight)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FusedMoeExpertFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, num_experts, gate_weights, expert_index, hidden_states,
        fc1_1_weight, fc1_2_weight, fc2_weight):
        splits = expert_histogram(expert_index, num_experts)
        scatter_index = expert_index.flatten().argsort(stable=True).argsort(
            ).int().view(expert_index.shape)
        scatter_output = moe_scatter(hidden_states, scatter_index)
        cumsum_t = torch.cumsum(splits, dim=0)
        fc1_1_output = group_gemm_same_nk(a=scatter_output, b=fc1_1_weight,
            cumsum_M=cumsum_t, max_M=scatter_output.shape[0], transpose_a=
            False, transpose_b=True)
        fc1_2_output = group_gemm_same_nk(a=scatter_output, b=fc1_2_weight,
            cumsum_M=cumsum_t, max_M=scatter_output.shape[0], transpose_a=
            False, transpose_b=True)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_activation = fc1_1_activation * fc1_2_output
        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
        fc1_weighted_output = fc1_activation * scattered_gate_weight
        fc2_output = group_gemm_same_nk(a=fc1_weighted_output, b=fc2_weight,
            cumsum_M=cumsum_t, max_M=scatter_output.shape[0], transpose_a=
            False, transpose_b=True)
        expert_output = moe_gather(fc2_output, scatter_index)
        output = expert_output.reshape(hidden_states.shape)
        ctx.num_experts = num_experts
        ctx.save_for_backward(gate_weights, fc1_1_weight, fc1_2_weight,
            fc2_weight, hidden_states, scatter_index, scatter_output,
            cumsum_t, fc1_1_output, fc1_2_output, fc1_activation,
            scattered_gate_weight, fc1_weighted_output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (gate_weights, fc1_1_weight, fc1_2_weight, fc2_weight,
            hidden_states, scatter_index, scatter_output, cumsum_t,
            fc1_1_output, fc1_2_output, fc1_activation,
            scattered_gate_weight, fc1_weighted_output) = ctx.saved_tensors
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)
        grad_fc2_output = moe_scatter(grad_output, scatter_index)
        grad_fc1_weighted_output = group_gemm_same_nk(a=grad_fc2_output, b=
            fc2_weight, cumsum_M=cumsum_t, max_M=grad_output.shape[0],
            transpose_b=False)
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(a=grad_fc2_output, b=fc1_weighted_output, c=
                grad_fc2_weight, cumsum_K=cumsum_t, max_K=grad_output.shape
                [0], transpose_a=True, transpose_b=False)
        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight
        grad_scattered_gate_weight = torch.sum(fc1_activation *
            grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation
        grad_scatter_output_2 = group_gemm_same_nk(a=grad_fc1_2_output, b=
            fc1_2_weight, cumsum_M=cumsum_t, max_M=grad_output.shape[0],
            transpose_b=False)
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(a=grad_fc1_2_output, b=scatter_output, c=
                grad_fc1_2_weight, cumsum_K=cumsum_t, max_K=grad_output.
                shape[0], transpose_a=True, transpose_b=False)
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation,
            fc1_1_output)
        grad_scatter_output_1 = group_gemm_same_nk(a=grad_fc1_1_output, b=
            fc1_1_weight, cumsum_M=cumsum_t, max_M=grad_output.shape[0],
            transpose_b=False)
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(a=grad_fc1_1_output, b=scatter_output, c=
                grad_fc1_1_weight, cumsum_K=cumsum_t, max_K=grad_output.
                shape[0], transpose_a=True, transpose_b=False)
        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
        grad_hidden_states = moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)
        return (None, grad_gate_weight, None, grad_hidden_states,
            grad_fc1_1_weight, grad_fc1_2_weight, grad_fc2_weight)
