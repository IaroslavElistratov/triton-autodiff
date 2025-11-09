# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/shawntan/scattermoe
# Source-Files: scattermoe/parallel_experts.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_q3__3sbf/scattermoe-main/scattermoe/parallel_experts.py
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

@triton.jit
def _compute_expert_block(E_idx, E_mask, M_in_idx, N_block, N_mask, X_ptr,
    stride_xm, stride_xk, W_ptr, stride_we, stride_wk, stride_wn, K, acc,
    no_k_mask, BLOCK_K, allow_tf32=True):
    K_block = tl.arange(0, BLOCK_K)
    X_blk_ptrs = X_ptr + M_in_idx[:, None] * stride_xm + K_block[None, :
        ] * stride_xk
    W_blk_ptrs = W_ptr + K_block[:, None] * stride_wk + N_block[None, :
        ] * stride_wn + E_idx * stride_we
    iters = tl.cdiv(K, BLOCK_K)
    for K_block_id in range(iters):
        if no_k_mask:
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None])
            w = tl.load(W_blk_ptrs, mask=N_mask[None, :])
        else:
            K_mask = K_block_id * BLOCK_K + K_block < K
            x = tl.load(X_blk_ptrs, mask=E_mask[:, None] & K_mask[None, :])
            w = tl.load(W_blk_ptrs, mask=K_mask[:, None] & N_mask[None, :])
        X_blk_ptrs += BLOCK_K * stride_xk
        W_blk_ptrs += BLOCK_K * stride_wk
        acc = tl.dot(x, w, acc, allow_tf32=allow_tf32)
    return acc


@triton.autotune(configs=_scatter2scatter_configs(), key=['M', 'N', 'K'])
@triton.heuristics({'NO_K_MASK': lambda args: args['K'] % args['BLOCK_K'] ==
    0, 'NO_N_MASK': lambda args: args['N'] % args['BLOCK_N'] == 0})
@triton.jit
def _scatter2scatter(X_ptr, stride_xm: tl.constexpr, stride_xk: tl.
    constexpr, W_ptr, stride_we, stride_wk: tl.constexpr, stride_wn: tl.
    constexpr, Y_ptr, stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    B_ptr, stride_be: tl.constexpr, stride_bn: tl.constexpr,
    grouped_idx_ptr, expert_idxs_ptr, FAN_OUT: tl.constexpr, M, K: tl.
    constexpr, N: tl.constexpr, E: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
    allow_tf32: tl.constexpr, x_grouped: tl.constexpr, y_grouped: tl.
    constexpr, NO_K_MASK: tl.constexpr, NO_N_MASK: tl.constexpr):
    pid = tl.program_id(axis=0)
    N_BLOCK_COUNT = tl.cdiv(N, BLOCK_N)
    M_block_id = pid // N_BLOCK_COUNT
    N_block_id = pid % N_BLOCK_COUNT
    M_block = M_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_block < N
    M_boundary_mask = M_block < FAN_OUT * M
    E_idxs = tl.load(expert_idxs_ptr + M_block, mask=M_boundary_mask, other=E)
    no_k_mask = K % BLOCK_K == 0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    E_first_idx = tl.min(E_idxs)
    E_last_idx = tl.minimum(tl.max(E_idxs), E - 1)
    M_idx = tl.load(grouped_idx_ptr + M_block, mask=M_boundary_mask).to(tl.
        int32)
    for E_idx in range(E_first_idx, E_last_idx + 1):
        E_mask = E_idxs == E_idx
        E_M_idx = M_idx
        if x_grouped:
            M_in_idx = M_block
        else:
            M_in_idx = E_M_idx // FAN_OUT
        acc = _compute_expert_block(E_idx, E_mask, M_in_idx, N_block,
            N_mask, X_ptr, stride_xm, stride_xk, W_ptr, stride_we,
            stride_wk, stride_wn, K, acc, no_k_mask, BLOCK_K, allow_tf32=
            allow_tf32)
    if B_ptr is not None:
        B_blk_ptrs = B_ptr + E_idxs[:, None] * stride_be + N_block[None, :
            ] * stride_bn
        acc += tl.load(B_blk_ptrs, mask=M_boundary_mask[:, None] & N_mask[
            None, :])
    if y_grouped:
        M_out_idx = M_block
    else:
        M_out_idx = M_idx
    Y_blk_ptrs = Y_ptr + (M_out_idx[:, None] * stride_ym + N_block[None, :] *
        stride_yn)
    tl.store(Y_blk_ptrs, acc, mask=M_boundary_mask[:, None] & N_mask[None, :])


def scatter2scatter(X, W, sorted_expert_idxs, sorted_scattered_idxs, k, b=
    None, x_grouped=False, y_grouped=False, out=None):
    assert sorted_scattered_idxs.size(0) == sorted_expert_idxs.size(0)
    assert sorted_scattered_idxs.size(0) == X.size(0) * k
    y_dim = W.size(-1)
    L_scattered = sorted_expert_idxs.size(0)
    if out is None:
        output = torch.empty((L_scattered, y_dim), device=X.device, dtype=X
            .dtype)
    else:
        assert out.size(0) == L_scattered and out.size(1) == y_dim
        output = out
    scatter2scatter_compileable(output, W, X, k, sorted_expert_idxs,
        sorted_scattered_idxs, b, x_grouped, y_grouped)
    return output


@torch.library.custom_op('scattermoe::scatter2scatter', mutates_args={'output'}
    )
def scatter2scatter_compileable(output: torch.Tensor, W: torch.Tensor, X:
    torch.Tensor, k: int, sorted_expert_idxs: torch.Tensor,
    sorted_scattered_idxs: torch.Tensor, b: Optional[torch.Tensor],
    x_grouped: bool, y_grouped: bool) ->None:

    def grid(META):
        grid_num = triton.cdiv(sorted_expert_idxs.size(0), META['BLOCK_M']
            ) * triton.cdiv(META['N'], META['BLOCK_N']),
        return grid_num
    if b is None:
        b = None
        stride_be = stride_bk = 0
    else:
        stride_be, stride_bk = b.stride()
    _scatter2scatter[grid](X, X.stride(0), X.stride(1), W, W.stride(0), W.
        stride(1), W.stride(2), output, output.stride(0), output.stride(1),
        b, stride_be, stride_bk, grouped_idx_ptr=sorted_scattered_idxs,
        expert_idxs_ptr=sorted_expert_idxs, FAN_OUT=k, M=X.size(0), K=X.
        size(1), N=output.size(1), E=W.size(0), BLOCK_M=BLOCK_M, ACC_TYPE=
        tl.float32, allow_tf32=ALLOW_TF32, x_grouped=x_grouped, y_grouped=
        y_grouped)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def _ParallelLinear_forward(ctx, x: torch.Tensor, expert_weights: torch.
    Tensor, k: int, sorted_expert_idxs: torch.Tensor, sorted_scattered_idxs:
    torch.Tensor, expert_offsets: torch.Tensor, expert_biases: Optional[
    torch.Tensor]=None, gates: Optional[torch.Tensor]=None, grouped_in:
    bool=False, grouped_out: bool=False):
    with torch.device(x.device):
        output = kernels.ops.scatter2scatter(X=x, W=expert_weights, b=
            expert_biases, k=k, sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs, x_grouped=
            grouped_in, y_grouped=grouped_out)
        if gates is not None:
            output_expanded = output.view(gates.size(0), gates.size(1),
                output.size(-1))
            output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
        else:
            output_expanded = None
        ctx.save_for_backward(x, expert_weights, expert_biases,
            sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
            gates, output_expanded)
        ctx.grouped_in = grouped_in
        ctx.grouped_out = grouped_out
        ctx.k = k
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=_config_grouping(), key=['K'])
@triton.heuristics({'NO_K_MASK': lambda args: args['K'] % args['BLOCK_K'] == 0}
    )
@triton.jit
def _group(src_ptr, stride_sn, stride_sk, has_coeff: tl.constexpr,
    coeff_ptr, FAN_OUT: tl.constexpr, tgt_ptr, stride_tn, stride_ti,
    grouped_idx_ptr, N, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl
    .constexpr, NO_K_MASK: tl.constexpr):
    pid = tl.program_id(axis=0)
    N_block_id = pid
    N_blk = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    N_mask = N_blk < N
    N_blk = tl.max_contiguous(tl.multiple_of(N_blk % N, BLOCK_N), BLOCK_N)
    N_idx = tl.load(grouped_idx_ptr + N_blk, mask=N_mask, other=0)
    K_blk = tl.arange(0, BLOCK_K)
    src_blk_ptrs = src_ptr + (N_idx // FAN_OUT)[:, None] * stride_sn + K_blk[
        None, :] * stride_sk
    tgt_blk_ptrs = tgt_ptr + N_blk[:, None] * stride_tn + K_blk[None, :
        ] * stride_ti
    if has_coeff:
        c = tl.load(coeff_ptr + N_idx, mask=N_mask)[:, None]
    iters = tl.cdiv(K, BLOCK_K)
    for i in range(0, iters):
        if NO_K_MASK or i < iters - 1:
            block = tl.load(src_blk_ptrs, mask=N_mask[:, None])
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=N_mask[:, None])
        else:
            K_mask = i * BLOCK_K + K_blk < K
            mask = N_mask[:, None] & K_mask[None, :]
            block = tl.load(src_blk_ptrs, mask=mask)
            if has_coeff:
                block *= c
            tl.store(tgt_blk_ptrs, block, mask=mask)
        src_blk_ptrs += BLOCK_K * stride_sk
        tgt_blk_ptrs += BLOCK_K * stride_ti


@triton.autotune(configs=_config_XtY(), key=['M', 'N', 'K'])
@triton.heuristics({'NO_K_MASK': lambda args: args['K'] % args['BLOCK_K'] ==
    0, 'NO_N_MASK': lambda args: args['N'] % args['BLOCK_N'] == 0})
@triton.jit
def _groupXtY(DY_ptr, stride_dym, stride_dyk, X_ptr, stride_xm, stride_xn,
    DW_ptr, stride_dwe, stride_dwk, stride_dwn, Db_ptr, stride_dbe,
    stride_dbn, expert_offsets_ptr, M, K: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACC_TYPE: tl.constexpr, allow_tf32: tl.constexpr, NO_K_MASK: tl.
    constexpr, NO_N_MASK: tl.constexpr):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    num0 = tl.num_programs(0)
    num1 = tl.num_programs(1)
    pid0, pid1 = tl.swizzle2d(pid0, pid1, num0, num1, 4)
    K_BLOCK_COUNT = tl.cdiv(K, BLOCK_K)
    E_idx = pid0 // K_BLOCK_COUNT
    K_block_id = pid0 % K_BLOCK_COUNT
    N_block_id = pid1
    if E_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(expert_offsets_ptr + E_idx - 1).to(tl.int32)
    end_idx = tl.load(expert_offsets_ptr + E_idx).to(tl.int32)
    if end_idx > start_idx:
        M_block = tl.max_contiguous(start_idx + tl.arange(0, BLOCK_M), BLOCK_M)
        K_block = K_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        K_mask = K_block < K
        K_block = tl.max_contiguous(tl.multiple_of(K_block % K, BLOCK_K),
            BLOCK_K)
        N_block = N_block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        N_mask = N_block < N
        N_block = tl.max_contiguous(tl.multiple_of(N_block % N, BLOCK_N),
            BLOCK_N)
        M_idxs = M_block
        xt_blk_ptrs = X_ptr + K_block[:, None] * stride_xn + M_idxs[None, :
            ] * stride_xm
        dy_blk_ptrs = DY_ptr + M_idxs[:, None] * stride_dym + N_block[None, :
            ] * stride_dyk
        if Db_ptr is not None and K_block_id == 0:
            _xty_and_bias(E_idx, start_idx, end_idx, M_block, K_block,
                K_mask, N_block, N_mask, dy_blk_ptrs, stride_dym,
                xt_blk_ptrs, stride_xm, DW_ptr, stride_dwe, stride_dwk,
                stride_dwn, Db_ptr, stride_dbe, stride_dbn, BLOCK_M,
                BLOCK_N, BLOCK_K, ACC_TYPE, allow_tf32, NO_K_MASK,
                NO_N_MASK, compute_bias=True)
        else:
            _xty_and_bias(E_idx, start_idx, end_idx, M_block, K_block,
                K_mask, N_block, N_mask, dy_blk_ptrs, stride_dym,
                xt_blk_ptrs, stride_xm, DW_ptr, stride_dwe, stride_dwk,
                stride_dwn, Db_ptr, stride_dbe, stride_dbn, BLOCK_M,
                BLOCK_N, BLOCK_K, ACC_TYPE, allow_tf32, NO_K_MASK,
                NO_N_MASK, compute_bias=False)


@triton.jit
def _xty_and_bias(E_idx, start_idx, end_idx, M_block, K_block, K_mask,
    N_block, N_mask, dy_blk_ptrs, stride_dym, xt_blk_ptrs, stride_xm,
    DW_ptr, stride_dwe, stride_dwk, stride_dwn, Db_ptr, stride_dbe,
    stride_dbn, BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE, allow_tf32, NO_K_MASK,
    NO_N_MASK, compute_bias: tl.constexpr):
    if compute_bias:
        db_acc = tl.zeros((BLOCK_N,), dtype=ACC_TYPE)
    else:
        db_acc = None
    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_TYPE)
    iters = tl.cdiv(end_idx - start_idx, BLOCK_M)
    for i in range(0, iters):
        M_mask = i * BLOCK_M + M_block < end_idx
        if NO_K_MASK:
            xt = tl.load(xt_blk_ptrs, mask=M_mask[None, :])
        else:
            xt = tl.load(xt_blk_ptrs, mask=K_mask[:, None] & M_mask[None, :])
        if NO_N_MASK:
            dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None])
        else:
            dy = tl.load(dy_blk_ptrs, mask=M_mask[:, None] & N_mask[None, :])
        acc += tl.dot(xt, dy, out_dtype=ACC_TYPE, allow_tf32=allow_tf32)
        xt_blk_ptrs += BLOCK_M * stride_xm
        dy_blk_ptrs += BLOCK_M * stride_dym
        if compute_bias:
            db_acc += tl.sum(dy, axis=0)
    DW_blk_ptrs = DW_ptr + E_idx * stride_dwe + K_block[:, None
        ] * stride_dwk + N_block[None, :] * stride_dwn
    acc = acc.to(DW_blk_ptrs.dtype.element_ty)
    tl.store(DW_blk_ptrs, acc, mask=K_mask[:, None] & N_mask[None, :])
    if compute_bias:
        Db_blk_ptrs = Db_ptr + E_idx * stride_dbe + N_block * stride_dbn
        tl.store(Db_blk_ptrs, db_acc, mask=N_mask)


def group(A, sorted_expert_idxs, coeff=None, fan_out=1, out=None):
    N = sorted_expert_idxs.size(0)
    K = A.size(1)
    assert A.size(0) * fan_out == N
    if out is not None:
        Y = out
    else:
        Y = torch.empty((N, K), dtype=A.dtype, device=A.device)
    group_compileable(A, K, N, Y, coeff, coeff is not None, fan_out,
        sorted_expert_idxs)
    return Y


@torch.library.custom_op('scattermoe::groupXtY', mutates_args={'DW'})
def groupXtY_compileable(E: int, DW: torch.Tensor, Db: Optional[torch.
    Tensor], DY: torch.Tensor, X: torch.Tensor, expert_offsets: torch.Tensor
    ) ->None:

    def grid(META):
        grid = E * triton.cdiv(META['K'], META['BLOCK_K']), triton.cdiv(META
            ['N'], META['BLOCK_N'])
        return grid
    if Db is None:
        stride_dbe = 0
        stride_dbn = 0
    else:
        stride_dbe, stride_dbn = Db.stride()
    _groupXtY[grid](DY, DY.stride(0), DY.stride(1), X, X.stride(0), X.
        stride(1), DW, DW.stride(0), DW.stride(1), DW.stride(2), Db,
        stride_dbe, stride_dbn, expert_offsets, M=DY.size(0), N=DY.size(-1),
        K=X.size(-1), ACC_TYPE=tl.float32, allow_tf32=ALLOW_TF32)


def group_bwd_W(DY, X, expert_offsets, E, has_bias=False):
    DWt = torch.zeros((E, DY.size(-1), X.size(-1)), device=DY.device, dtype
        =DY.dtype)
    DW = DWt.permute(0, 2, 1)
    if has_bias:
        Db = torch.zeros((E, DY.size(-1)), device=DY.device, dtype=DY.dtype)
    else:
        Db = None
    groupXtY_compileable(E, DW, Db, DY, X, expert_offsets)
    return DW, Db


@torch.library.custom_op('scattermoe::group', mutates_args={'Y'})
def group_compileable(A: torch.Tensor, K: int, N: int, Y: torch.Tensor,
    coeff: torch.Tensor, has_coeff: bool, fan_out: int, sorted_expert_idxs:
    torch.Tensor) ->None:

    def grid(META):
        grid_num = triton.cdiv(META['N'], META['BLOCK_N']),
        return grid_num
    _group[grid](A, A.stride(0), A.stride(1), has_coeff, coeff, fan_out, Y,
        Y.stride(0), Y.stride(1), sorted_expert_idxs, N, K)


# Backward method (kernel launch code)
def _ParallelLinear_backward(ctx, grad_out: torch.Tensor):
    with torch.device(grad_out.device):
        (x, expert_weights, expert_biases, sorted_expert_idxs,
            sorted_scattered_idxs, expert_offsets, gates, output_expanded
            ) = ctx.saved_tensors
        k = ctx.k
        grouped_in = ctx.grouped_in
        grouped_out = ctx.grouped_out
        if gates is not None:
            d_gates = (output_expanded @ grad_out.unsqueeze(-1)).squeeze(-1)
            gates_flat = gates.flatten()
            gate_fan = gates.size(1)
            grouped_grad_out = output_expanded.flatten(0, 1)
        else:
            d_gates = None
            gates_flat = None
            gate_fan = 1
            grouped_grad_out = None
        if grouped_out:
            grouped_grad_out = grad_out
        else:
            grouped_grad_out = kernels.ops.group(grad_out,
                sorted_scattered_idxs, fan_out=gate_fan, coeff=gates_flat,
                out=grouped_grad_out)
        if grouped_in:
            grouped_x = x
            d_expanded_input = None
        else:
            grouped_x = kernels.ops.group(x, sorted_scattered_idxs, fan_out=k)
            d_expanded_input = grouped_x
        d_weights, d_biases = kernels.ops.group_bwd_W(DY=grouped_grad_out,
            X=grouped_x, expert_offsets=expert_offsets, E=expert_weights.
            size(0), has_bias=expert_biases is not None)
        d_expanded_input = kernels.ops.scatter2scatter(X=grouped_grad_out,
            x_grouped=True, W=expert_weights.permute(0, 2, 1),
            sorted_expert_idxs=sorted_expert_idxs, sorted_scattered_idxs=
            sorted_scattered_idxs, k=1, y_grouped=grouped_in, out=
            d_expanded_input)
        if k == 1:
            d_input = d_expanded_input
        else:
            d_input = d_expanded_input.view(x.size(0), k, d_expanded_input.
                size(-1)).sum(-2)
    return (d_input, d_weights, None, None, None, None, d_biases, d_gates,
        None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class ParallelLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, expert_weights: torch.Tensor, k: int,
        sorted_expert_idxs: torch.Tensor, sorted_scattered_idxs: torch.
        Tensor, expert_offsets: torch.Tensor, expert_biases: Optional[torch
        .Tensor]=None, gates: Optional[torch.Tensor]=None, grouped_in: bool
        =False, grouped_out: bool=False):
        with torch.device(x.device):
            output = kernels.ops.scatter2scatter(X=x, W=expert_weights, b=
                expert_biases, k=k, sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs, x_grouped=
                grouped_in, y_grouped=grouped_out)
            if gates is not None:
                output_expanded = output.view(gates.size(0), gates.size(1),
                    output.size(-1))
                output = (gates.unsqueeze(1) @ output_expanded).squeeze(1)
            else:
                output_expanded = None
            ctx.save_for_backward(x, expert_weights, expert_biases,
                sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
                gates, output_expanded)
            ctx.grouped_in = grouped_in
            ctx.grouped_out = grouped_out
            ctx.k = k
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        with torch.device(grad_out.device):
            (x, expert_weights, expert_biases, sorted_expert_idxs,
                sorted_scattered_idxs, expert_offsets, gates, output_expanded
                ) = ctx.saved_tensors
            k = ctx.k
            grouped_in = ctx.grouped_in
            grouped_out = ctx.grouped_out
            if gates is not None:
                d_gates = (output_expanded @ grad_out.unsqueeze(-1)).squeeze(-1
                    )
                gates_flat = gates.flatten()
                gate_fan = gates.size(1)
                grouped_grad_out = output_expanded.flatten(0, 1)
            else:
                d_gates = None
                gates_flat = None
                gate_fan = 1
                grouped_grad_out = None
            if grouped_out:
                grouped_grad_out = grad_out
            else:
                grouped_grad_out = kernels.ops.group(grad_out,
                    sorted_scattered_idxs, fan_out=gate_fan, coeff=
                    gates_flat, out=grouped_grad_out)
            if grouped_in:
                grouped_x = x
                d_expanded_input = None
            else:
                grouped_x = kernels.ops.group(x, sorted_scattered_idxs,
                    fan_out=k)
                d_expanded_input = grouped_x
            d_weights, d_biases = kernels.ops.group_bwd_W(DY=
                grouped_grad_out, X=grouped_x, expert_offsets=
                expert_offsets, E=expert_weights.size(0), has_bias=
                expert_biases is not None)
            d_expanded_input = kernels.ops.scatter2scatter(X=
                grouped_grad_out, x_grouped=True, W=expert_weights.permute(
                0, 2, 1), sorted_expert_idxs=sorted_expert_idxs,
                sorted_scattered_idxs=sorted_scattered_idxs, k=1, y_grouped
                =grouped_in, out=d_expanded_input)
            if k == 1:
                d_input = d_expanded_input
            else:
                d_input = d_expanded_input.view(x.size(0), k,
                    d_expanded_input.size(-1)).sum(-2)
        return (d_input, d_weights, None, None, None, None, d_biases,
            d_gates, None, None)
