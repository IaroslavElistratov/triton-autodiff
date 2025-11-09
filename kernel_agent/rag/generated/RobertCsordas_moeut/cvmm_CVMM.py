# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/RobertCsordas/moeut
# Source-Files: moeut/cvmm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_jy69wtnh/moeut-master/moeut/cvmm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

def get_dtype():
    if not torch.is_autocast_enabled():
        return torch.float32
    return torch.get_autocast_gpu_dtype()


# Forward method (kernel launch code)
def _CVMM_forward(ctx, x: torch.Tensor, sel_index: torch.Tensor, sel: torch
    .Tensor, keys: torch.Tensor, out_index: Optional[torch.Tensor]=None,
    reduction_weight: Optional[torch.Tensor]=None):
    ctx.save_for_backward(x, keys, sel, sel_index, out_index, reduction_weight)
    out_type = get_dtype()
    if out_index is None:
        out_index = torch.tensor(-1).cuda()
    res = cvmm_triton_call(x, sel_index, sel, keys, out_type, out_index)
    if reduction_weight is not None:
        res = res.view(*reduction_weight.shape, res.shape[-1])
        res = (reduction_weight.unsqueeze(-2).type_as(res) @ res).squeeze(-2)
    ctx.op_type = out_type
    ctx.keys_type = keys.dtype
    ctx.dtype = out_type
    return res


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N':
    64, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=
    4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 4}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 8}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 16}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 16}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 64}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4,
    num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32,
    'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8, 'K_BLOCKS': 32}, num_stages=4,
    num_warps=4)], key=['M', 'N', 'K', 'out_dtype_id', 'allow_tf32',
    'dtype_id'], reset_to_zero=['c_ptr'])
@triton.jit
def cvmm_backward_kernel3(a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr,
    out_index_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
    stride_co, stride_cm, stride_cn, stride_index, stride_sel,
    stride_out_index, out_index_is_none: tl.constexpr, out_dtype_id: tl.
    constexpr, allow_tf32: tl.constexpr, dtype_id: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K:
    tl.constexpr, GROUP_SIZE_M: tl.constexpr, K_BLOCKS: tl.constexpr):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    k_block_id = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs_this = a_ptr + offs_am[:, None] * stride_am
    b_ptrs_this = b_ptr + offs_bn[None, :] * stride_bn
    block_start_index = k_block_id * BLOCK_SIZE_K * K_BLOCKS
    block_end_index = min(block_start_index + BLOCK_SIZE_K * K_BLOCKS, K) - 1
    first_mat = tl.load(sel_ptr + stride_sel * block_start_index)
    last_mat = tl.load(sel_ptr + stride_sel * block_end_index)
    for matrix_index in range(first_mat, last_mat + 1):
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        start_i = block_start_index
        end_i = block_end_index + 1
        while start_i < end_i:
            middle = (start_i + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix < matrix_index:
                start_i = middle + 1
            else:
                end_i = middle
        start_i2 = start_i
        end_i = block_end_index + 1
        while start_i2 < end_i:
            middle = (start_i2 + end_i) // 2
            middle_matrix = tl.load(sel_ptr + middle * stride_sel)
            if middle_matrix <= matrix_index:
                start_i2 = middle + 1
            else:
                end_i = middle
        end_i = start_i2
        count = end_i - start_i
        block_mem_indices_f_base = start_i + tl.arange(0, BLOCK_SIZE_K)
        if count > 0:
            for k in range((count + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K):
                block_mem_indices_f = (block_mem_indices_f_base + k *
                    BLOCK_SIZE_K)
                block_mem_indices = block_mem_indices_f % K
                a_index = tl.load(index_ptr + stride_index * block_mem_indices)
                if out_index_is_none:
                    b_index = a_index
                else:
                    b_index = tl.load(out_index_ptr + stride_out_index *
                        block_mem_indices)
                sel_ok = block_mem_indices_f < end_i
                a_ptrs = a_ptrs_this + a_index[None, :] * stride_ak
                b_ptrs = b_ptrs_this + b_index[:, None] * stride_bk
                a = tl.load(a_ptrs, mask=sel_ok[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=sel_ok[:, None], other=0.0)
                if dtype_id == 1:
                    a = a.to(tl.float16)
                    b = b.to(tl.float16)
                elif dtype_id == 2:
                    a = a.to(tl.bfloat16)
                    b = b.to(tl.bfloat16)
                accumulator += tl.dot(a, b, allow_tf32=allow_tf32)
            if out_dtype_id == 1:
                c = accumulator.to(tl.float16)
            elif out_dtype_id == 2:
                c = accumulator.to(tl.bfloat16)
            else:
                c = accumulator
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_co * matrix_index + stride_cm * offs_cm[
                :, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.atomic_add(c_ptrs, c, mask=c_mask)


def cvmm_triton_backward(x: torch.Tensor, sel_index: torch.Tensor, sel:
    torch.Tensor, grads: torch.Tensor, n_experts: int, key_dtype: torch.
    dtype, op_dtype: torch.dtype, out_index: torch.Tensor):
    x = x.flatten(end_dim=-2)
    x = x.transpose(0, 1)
    grads = grads.flatten(end_dim=-2)
    sel = sel.flatten()
    M, _ = x.shape
    K, N = grads.shape
    out = torch.zeros((n_experts, M, N), device=x.device, dtype=key_dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv
        (N, META['BLOCK_SIZE_N']), triton.cdiv(K, META['BLOCK_SIZE_K'] *
        META['K_BLOCKS']))
    out_index_is_none = False
    if out_index.numel() == 1 and out_index == -1:
        out_index_is_none = True
    cvmm_backward_kernel3[grid](x, grads, out, sel_index, sel, out_index, M,
        N, K, x.stride(0), x.stride(1), grads.stride(0), grads.stride(1),
        out.stride(0), out.stride(1), out.stride(2), sel_index.stride(0),
        sel.stride(0), 0 if out_index_is_none else out_index.stride(0),
        out_index_is_none=out_index_is_none, out_dtype_id=dtype_to_type_id(
        out.dtype), dtype_id=dtype_to_type_id(op_dtype), allow_tf32=False)
    return out


def dtype_to_type_id(dtype: torch.dtype):
    if dtype == torch.float32:
        return 0
    elif dtype == torch.float16:
        return 1
    elif dtype == torch.bfloat16:
        return 2
    raise ValueError('Unknown dtype')


# Backward method (kernel launch code)
def _CVMM_backward(ctx, grad_output):
    x, keys, sel, sel_index, out_index, reduction_weight = ctx.saved_tensors
    keys_dt = keys
    if reduction_weight is not None:
        grad_output_w = reduction_weight.unsqueeze(-1).type_as(grad_output
            ) @ grad_output.unsqueeze(-2)
    else:
        grad_output_w = grad_output
    out_index_is_none = False
    if out_index is None:
        out_index_is_none = True
        out_index = torch.tensor(-1).cuda()
    grad_w = cvmm_triton_backward(x, sel_index, sel, grad_output_w, keys_dt
        .shape[0], ctx.keys_type, ctx.dtype, out_index=out_index)
    grad_w_off = None
    bw_index = sel_index if out_index_is_none else out_index
    bw_index_out = torch.tensor(-1).cuda()
    if reduction_weight is not None:
        bw_index_out = bw_index
        bw_index = bw_index // reduction_weight.shape[-1]
    grad_x_full = cvmm_triton_call(grad_output, bw_index, sel, keys_dt.
        transpose(1, 2), ctx.op_type, bw_index_out)
    grad_x_full = grad_x_full.view(*x.shape[:-1], -1, x.shape[-1])
    if reduction_weight is not None:
        grad_x = (reduction_weight.view(*grad_x_full.shape[:-1]).unsqueeze(
            -2).type_as(grad_x_full) @ grad_x_full).squeeze(-2)
        grad_w_off = (grad_x_full.type_as(reduction_weight) @ x.unsqueeze(-
            1).type_as(reduction_weight)).squeeze(-1).view_as(reduction_weight)
    elif grad_x_full.shape[-2] != 1:
        grad_x = grad_x_full.sum(-2)
    else:
        grad_x = grad_x_full
    grad_x = grad_x.view_as(x)
    return grad_x, None, None, grad_w, None, grad_w_off


# ============================================================
# autograd.Function Class Definition
# ============================================================

class CVMM(torch.autograd.Function):
    warned = False

    @staticmethod
    def forward(ctx, x: torch.Tensor, sel_index: torch.Tensor, sel: torch.
        Tensor, keys: torch.Tensor, out_index: Optional[torch.Tensor]=None,
        reduction_weight: Optional[torch.Tensor]=None):
        ctx.save_for_backward(x, keys, sel, sel_index, out_index,
            reduction_weight)
        out_type = get_dtype()
        if out_index is None:
            out_index = torch.tensor(-1).cuda()
        res = cvmm_triton_call(x, sel_index, sel, keys, out_type, out_index)
        if reduction_weight is not None:
            res = res.view(*reduction_weight.shape, res.shape[-1])
            res = (reduction_weight.unsqueeze(-2).type_as(res) @ res).squeeze(
                -2)
        ctx.op_type = out_type
        ctx.keys_type = keys.dtype
        ctx.dtype = out_type
        return res

    @staticmethod
    def backward(ctx, grad_output):
        x, keys, sel, sel_index, out_index, reduction_weight = (ctx.
            saved_tensors)
        keys_dt = keys
        if reduction_weight is not None:
            grad_output_w = reduction_weight.unsqueeze(-1).type_as(grad_output
                ) @ grad_output.unsqueeze(-2)
        else:
            grad_output_w = grad_output
        out_index_is_none = False
        if out_index is None:
            out_index_is_none = True
            out_index = torch.tensor(-1).cuda()
        grad_w = cvmm_triton_backward(x, sel_index, sel, grad_output_w,
            keys_dt.shape[0], ctx.keys_type, ctx.dtype, out_index=out_index)
        grad_w_off = None
        bw_index = sel_index if out_index_is_none else out_index
        bw_index_out = torch.tensor(-1).cuda()
        if reduction_weight is not None:
            bw_index_out = bw_index
            bw_index = bw_index // reduction_weight.shape[-1]
        grad_x_full = cvmm_triton_call(grad_output, bw_index, sel, keys_dt.
            transpose(1, 2), ctx.op_type, bw_index_out)
        grad_x_full = grad_x_full.view(*x.shape[:-1], -1, x.shape[-1])
        if reduction_weight is not None:
            grad_x = (reduction_weight.view(*grad_x_full.shape[:-1]).
                unsqueeze(-2).type_as(grad_x_full) @ grad_x_full).squeeze(-2)
            grad_w_off = (grad_x_full.type_as(reduction_weight) @ x.
                unsqueeze(-1).type_as(reduction_weight)).squeeze(-1).view_as(
                reduction_weight)
        elif grad_x_full.shape[-2] != 1:
            grad_x = grad_x_full.sum(-2)
        else:
            grad_x = grad_x_full
        grad_x = grad_x.view_as(x)
        return grad_x, None, None, grad_w, None, grad_w_off
