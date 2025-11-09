# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/pengzhangzhi/Open-dLLM
# Source-Files: veomni/distributed/moe/moe_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ww7oku23/Open-dLLM-main/veomni/distributed/moe/moe_layer.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

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

# Forward method (kernel launch code)
def _EPGroupGemm_forward(ctx, permute_tokens, cumsum, fc1_1_weight,
    fc1_2_weight, fc2_weight):
    fc1_1_output = group_gemm_same_nk(a=permute_tokens, b=fc1_1_weight,
        cumsum_M=cumsum, max_M=permute_tokens.shape[0], transpose_a=False,
        transpose_b=True)
    fc1_2_output = group_gemm_same_nk(a=permute_tokens, b=fc1_2_weight,
        cumsum_M=cumsum, max_M=permute_tokens.shape[0], transpose_a=False,
        transpose_b=True)
    fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
    fc1_output = fc1_1_activation * fc1_2_output
    fc2_output = group_gemm_same_nk(a=fc1_output, b=fc2_weight, cumsum_M=
        cumsum, max_M=permute_tokens.shape[0], transpose_a=False,
        transpose_b=True)
    ctx.save_for_backward(permute_tokens, cumsum, fc1_1_weight,
        fc1_2_weight, fc2_weight, fc1_1_output, fc1_2_output)
    return fc2_output


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
def _EPGroupGemm_backward(ctx, grad_output):
    (permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight,
        fc1_1_output, fc1_2_output) = ctx.saved_tensors
    grad_fc1_output = group_gemm_same_nk(a=grad_output, b=fc2_weight,
        cumsum_M=cumsum, max_M=grad_output.shape[0], transpose_b=False)
    fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
    fc1_output = fc1_1_activation * fc1_2_output
    grad_fc2_weight = None
    if fc2_weight.requires_grad:
        grad_fc2_weight = torch.empty_like(fc2_weight)
        group_gemm_same_mn(a=grad_output, b=fc1_output, c=grad_fc2_weight,
            cumsum_K=cumsum, max_K=grad_output.shape[0], transpose_a=True,
            transpose_b=False)
    grad_fc1_2_output = fc1_1_activation * grad_fc1_output
    grad_fc1_1_activation = grad_fc1_output * fc1_2_output
    grad_scatter_output_2 = group_gemm_same_nk(a=grad_fc1_2_output, b=
        fc1_2_weight, cumsum_M=cumsum, max_M=grad_output.shape[0],
        transpose_b=False)
    grad_fc1_2_weight = None
    if fc1_2_weight.requires_grad:
        grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
        group_gemm_same_mn(a=grad_fc1_2_output, b=permute_tokens, c=
            grad_fc1_2_weight, cumsum_K=cumsum, max_K=grad_output.shape[0],
            transpose_a=True, transpose_b=False)
    grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation,
        fc1_1_output)
    grad_scatter_output_1 = group_gemm_same_nk(a=grad_fc1_1_output, b=
        fc1_1_weight, cumsum_M=cumsum, max_M=grad_output.shape[0],
        transpose_b=False)
    grad_fc1_1_weight = None
    if fc1_1_weight.requires_grad:
        grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
        group_gemm_same_mn(a=grad_fc1_1_output, b=permute_tokens, c=
            grad_fc1_1_weight, cumsum_K=cumsum, max_K=grad_output.shape[0],
            transpose_a=True, transpose_b=False)
    grad_permute_tokens = grad_scatter_output_1 + grad_scatter_output_2
    return (grad_permute_tokens, None, grad_fc1_1_weight, grad_fc1_2_weight,
        grad_fc2_weight)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class EPGroupGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, permute_tokens, cumsum, fc1_1_weight, fc1_2_weight,
        fc2_weight):
        fc1_1_output = group_gemm_same_nk(a=permute_tokens, b=fc1_1_weight,
            cumsum_M=cumsum, max_M=permute_tokens.shape[0], transpose_a=
            False, transpose_b=True)
        fc1_2_output = group_gemm_same_nk(a=permute_tokens, b=fc1_2_weight,
            cumsum_M=cumsum, max_M=permute_tokens.shape[0], transpose_a=
            False, transpose_b=True)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output
        fc2_output = group_gemm_same_nk(a=fc1_output, b=fc2_weight,
            cumsum_M=cumsum, max_M=permute_tokens.shape[0], transpose_a=
            False, transpose_b=True)
        ctx.save_for_backward(permute_tokens, cumsum, fc1_1_weight,
            fc1_2_weight, fc2_weight, fc1_1_output, fc1_2_output)
        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (permute_tokens, cumsum, fc1_1_weight, fc1_2_weight, fc2_weight,
            fc1_1_output, fc1_2_output) = ctx.saved_tensors
        grad_fc1_output = group_gemm_same_nk(a=grad_output, b=fc2_weight,
            cumsum_M=cumsum, max_M=grad_output.shape[0], transpose_b=False)
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output
        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            group_gemm_same_mn(a=grad_output, b=fc1_output, c=
                grad_fc2_weight, cumsum_K=cumsum, max_K=grad_output.shape[0
                ], transpose_a=True, transpose_b=False)
        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output
        grad_scatter_output_2 = group_gemm_same_nk(a=grad_fc1_2_output, b=
            fc1_2_weight, cumsum_M=cumsum, max_M=grad_output.shape[0],
            transpose_b=False)
        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            group_gemm_same_mn(a=grad_fc1_2_output, b=permute_tokens, c=
                grad_fc1_2_weight, cumsum_K=cumsum, max_K=grad_output.shape
                [0], transpose_a=True, transpose_b=False)
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation,
            fc1_1_output)
        grad_scatter_output_1 = group_gemm_same_nk(a=grad_fc1_1_output, b=
            fc1_1_weight, cumsum_M=cumsum, max_M=grad_output.shape[0],
            transpose_b=False)
        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            group_gemm_same_mn(a=grad_fc1_1_output, b=permute_tokens, c=
                grad_fc1_1_weight, cumsum_K=cumsum, max_K=grad_output.shape
                [0], transpose_a=True, transpose_b=False)
        grad_permute_tokens = grad_scatter_output_1 + grad_scatter_output_2
        return (grad_permute_tokens, None, grad_fc1_1_weight,
            grad_fc1_2_weight, grad_fc2_weight)
