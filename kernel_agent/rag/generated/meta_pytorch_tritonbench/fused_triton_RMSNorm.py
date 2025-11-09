# SPDX-License-Identifier: BSD-3-Clause
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-pytorch/tritonbench
# Source-Files: tritonbench/operators/rms_norm/fused_triton.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ct7v_342/tritonbench-main/tritonbench/operators/rms_norm/fused_triton.py
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
from math import ceil
from math import sqrt

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
def _RMSNorm_forward(ctx, x, normalized_shape, weight, eps):
    y = torch.empty_like(x)

    def rmsnorm_ref(inp, w, eps=1e-06):
        rms = 1.0 / torch.sqrt(torch.mean(inp.square(), dim=-1, keepdim=
            True) + eps)
        return (inp * rms * w).to(inp.dtype), rms
    y, rms = rmsnorm_ref(x, weight, eps)
    ctx.save_for_backward(x, weight, rms)
    ctx.eps = eps
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=[triton.Config({'M_INCREMENT': M_INCREMENT},
    num_warps=w) for M_INCREMENT in [1, 2, 4, 8, 16] for w in [2, 4, 8]],
    key=['M', 'N'])
@triton.jit
def _rms_norm_bwd_fused(DX, DY, DW, X, W, RMS, stride, N, M, BLOCK_SIZE_N:
    tl.constexpr, BLOCK_SIZE_M: tl.constexpr, M_INCREMENT: tl.constexpr,
    N_POW_2: tl.constexpr):
    pid = tl.program_id(0)
    start_row = pid * BLOCK_SIZE_M
    grad_w = tl.full([BLOCK_SIZE_N], 0, tl.float32)
    cols = tl.arange(0, BLOCK_SIZE_N)
    if N_POW_2:
        col_mask = None
    else:
        col_mask = cols < N
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)[None, :]
    for cur_row in tl.range(0, BLOCK_SIZE_M, M_INCREMENT):
        rows = start_row + cur_row + tl.arange(0, M_INCREMENT)
        row_indices = rows * stride
        row_mask = rows < M
        rms = tl.load(RMS + rows, mask=row_mask).to(tl.float32)[:, None]
        if N_POW_2:
            index_mask = row_mask[:, None]
        else:
            index_mask = row_mask[:, None] & col_mask[None, :]
        indices = row_indices[:, None] + cols[None, :]
        x = tl.load(X + indices, mask=index_mask, other=0).to(tl.float32)
        dy = tl.load(DY + indices, mask=index_mask, other=0).to(tl.float32)
        m = dy * w
        row_dot = tl.sum(m * x, axis=1)[:, None]
        scale = -(1.0 / N) * rms * rms * rms
        dx = rms * m
        dx += scale * row_dot * x
        tl.store(DX + indices, dx, mask=index_mask)
        grad_w += tl.sum(dy * x * rms, axis=0)
    tl.store(DW + pid * N + cols, grad_w, mask=col_mask)


# Backward method (kernel launch code)
def _RMSNorm_backward(ctx, dy):
    x, w, rms = ctx.saved_tensors
    x_arg = x.reshape(-1, x.shape[-1])
    N = w.shape[0]
    dw = torch.empty((N,), dtype=w.dtype, device=w.device)
    dx = torch.empty_like(dy)
    M, N = x_arg.shape
    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
    BLOCK_SIZE_M = min(2048, triton.next_power_of_2(M // (8 * NUM_SMS)))
    PARTIAL_SIZE = math.ceil(M / BLOCK_SIZE_M)
    _dw = torch.empty((PARTIAL_SIZE, N), dtype=w.dtype, device=w.device)
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = triton.next_power_of_2(N)
    assert BLOCK_SIZE <= MAX_FUSED_SIZE, "This layer norm doesn't support feature dim >= 64KB."
    _rms_norm_bwd_fused[PARTIAL_SIZE,](dx, dy, _dw, x_arg, w, rms, x_arg.
        stride(0), N, M, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE,
        N_POW_2=N % BLOCK_SIZE == 0)
    dw = torch.sum(_dw, dim=0)
    return dx, None, dw, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps):
        y = torch.empty_like(x)

        def rmsnorm_ref(inp, w, eps=1e-06):
            rms = 1.0 / torch.sqrt(torch.mean(inp.square(), dim=-1, keepdim
                =True) + eps)
            return (inp * rms * w).to(inp.dtype), rms
        y, rms = rmsnorm_ref(x, weight, eps)
        ctx.save_for_backward(x, weight, rms)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, rms = ctx.saved_tensors
        x_arg = x.reshape(-1, x.shape[-1])
        N = w.shape[0]
        dw = torch.empty((N,), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        M, N = x_arg.shape
        NUM_SMS = torch.cuda.get_device_properties('cuda'
            ).multi_processor_count
        BLOCK_SIZE_M = min(2048, triton.next_power_of_2(M // (8 * NUM_SMS)))
        PARTIAL_SIZE = math.ceil(M / BLOCK_SIZE_M)
        _dw = torch.empty((PARTIAL_SIZE, N), dtype=w.dtype, device=w.device)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = triton.next_power_of_2(N)
        assert BLOCK_SIZE <= MAX_FUSED_SIZE, "This layer norm doesn't support feature dim >= 64KB."
        _rms_norm_bwd_fused[PARTIAL_SIZE,](dx, dy, _dw, x_arg, w, rms,
            x_arg.stride(0), N, M, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=
            BLOCK_SIZE, N_POW_2=N % BLOCK_SIZE == 0)
        dw = torch.sum(_dw, dim=0)
        return dx, None, dw, None, None
