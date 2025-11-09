# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/XunhaoLai/native-sparse-attention-triton
# Source-Files: native_sparse_attention/ops/triton/linear_compress.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_ottww3ye/native-sparse-attention-triton-main/native_sparse_attention/ops/triton/linear_compress.py
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
from math import floor
from einops import rearrange

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def linear_compress_fwd_kernel(X, Y, W, cu_seqlens_x, cu_seqlens_y,
    stride_xn, stride_xh, stride_xd, stride_wh, stride_wk, stride_wd,
    stride_wD, stride_yn, stride_yh, stride_yd, NUM_HEADS: tl.constexpr,
    KERNEL_SIZE: tl.constexpr, KERNEL_STRIDE: tl.constexpr, HEADd_DIM: tl.
    constexpr, HEADD_DIM: tl.constexpr, BLOCK_OUTPUT_SEQ_SIZE: tl.constexpr,
    BLOCK_KERNEL_SIZE: tl.constexpr, BLOCK_HEADd_DIM: tl.constexpr,
    BLOCK_HEADD_DIM: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    pid_k = tl.program_id(1)
    pid_D = tl.program_id(2)
    x_start = tl.load(cu_seqlens_x + pid_b)
    x_end = tl.load(cu_seqlens_x + pid_b + 1)
    x_len = x_end - x_start
    y_start = tl.load(cu_seqlens_y + pid_b)
    y_end = tl.load(cu_seqlens_y + pid_b + 1)
    y_len = y_end - y_start
    if pid_k * BLOCK_OUTPUT_SEQ_SIZE >= y_len:
        return
    off_kernel_size = tl.arange(0, BLOCK_KERNEL_SIZE)
    off_d = tl.arange(0, BLOCK_HEADd_DIM)
    off_output_seq_size = tl.arange(0, BLOCK_OUTPUT_SEQ_SIZE)
    x_base_ptrs = X + pid_h * stride_xh + x_start * stride_xn + ((pid_k *
        BLOCK_OUTPUT_SEQ_SIZE * KERNEL_STRIDE + off_output_seq_size *
        KERNEL_STRIDE)[:, None] + off_kernel_size[None, :])[:, :, None
        ] * stride_xn + off_d[None, None, :] * stride_xd
    x_base_mask = ((pid_k * BLOCK_OUTPUT_SEQ_SIZE * KERNEL_STRIDE + 
        off_output_seq_size * KERNEL_STRIDE)[:, None] + off_kernel_size[
        None, :] < x_len)[:, :, None]
    w_ptrs = tl.make_block_ptr(base=W + pid_h * stride_wh, shape=(
        KERNEL_SIZE, HEADd_DIM, HEADD_DIM), strides=(stride_wk, stride_wd,
        stride_wD), offsets=(0, 0, pid_D * BLOCK_HEADD_DIM), block_shape=(
        BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM, BLOCK_HEADD_DIM), order=(2, 1, 0))
    y_ptrs = tl.make_block_ptr(base=Y + y_start * stride_yn + pid_h *
        stride_yh, shape=(y_len, HEADD_DIM), strides=(stride_yn, stride_yd),
        offsets=(pid_k * BLOCK_OUTPUT_SEQ_SIZE, pid_D * BLOCK_HEADD_DIM),
        block_shape=(BLOCK_OUTPUT_SEQ_SIZE, BLOCK_HEADD_DIM), order=(1, 0))
    y_d = tl.full((BLOCK_OUTPUT_SEQ_SIZE, BLOCK_HEADD_DIM), 0, dtype=tl.float32
        )
    for i in range(0, HEADd_DIM, BLOCK_HEADd_DIM):
        x_ptrs = x_base_ptrs + i * stride_xd
        x_mask = x_base_mask & (i + off_d < HEADd_DIM)[None, None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0)
        x = tl.reshape(x, (BLOCK_OUTPUT_SEQ_SIZE, BLOCK_KERNEL_SIZE *
            BLOCK_HEADd_DIM))
        w = tl.load(w_ptrs, boundary_check=(0, 1, 2), padding_option='zero')
        w = tl.reshape(w, (BLOCK_KERNEL_SIZE * BLOCK_HEADd_DIM,
            BLOCK_HEADD_DIM))
        y_d += tl.dot(x, w)
        w_ptrs = tl.advance(w_ptrs, (0, BLOCK_HEADd_DIM, 0))
    tl.store(y_ptrs, y_d.to(y_ptrs.dtype.element_ty), boundary_check=(0, 1))


# Forward method (kernel launch code)
def _LinearCompress_forward(ctx, x: torch.Tensor, w: torch.Tensor,
    cu_seqlens: torch.Tensor, kernel_size: int, kernel_stride: int) ->Tuple[
    torch.Tensor, torch.Tensor]:
    """
        Compress key and value tensor with kernel_size and kernel_stride. Similar to conv_compress.

        Args:
            x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
            w (torch.Tensor): weight for each head, shape (num_heads, kernel_size * head_dim, head_dim)
            cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen
            kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
            kernel_stride (int): stride for each compress kernel

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
        """
    assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
    assert x.dtype == w.dtype
    assert cu_seqlens.dtype == torch.int32
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    assert w.shape[0] == num_heads
    assert w.shape[1] == kernel_size * head_dim
    assert w.shape[2] == head_dim
    assert kernel_size % kernel_stride == 0
    assert kernel_size in {16, 32, 64, 128}
    assert head_dim % 8 == 0
    torch.cuda.set_device(x.device)
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch
        .int32) + 1
    y_seqlens[seqlens < kernel_size] = 0
    y_cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=x.
        device), torch.cumsum(y_seqlens.to(x.device), dim=0)], dim=0).to(torch
        .int32)
    y = torch.zeros(y_cu_seqlens[-1], num_heads, head_dim, dtype=x.dtype,
        device=x.device)
    block_kernel_size = max(16, triton.next_power_of_2(kernel_size))
    block_head_dim = 8 if IS_HOPPER_GPU else 4
    block_headD_dim = 32
    block_output_seq_size = 64
    w = w.reshape(num_heads, kernel_size, head_dim, head_dim).contiguous()
    grid = lambda META: (batch_size * num_heads, triton.cdiv(y_seqlens.max(
        0)[0].item(), META['BLOCK_OUTPUT_SEQ_SIZE']), triton.cdiv(head_dim,
        META['BLOCK_HEADD_DIM']))
    linear_compress_fwd_kernel[grid](x, y, w, cu_seqlens, y_cu_seqlens, x.
        stride(0), x.stride(1), x.stride(2), w.stride(0), w.stride(1), w.
        stride(2), w.stride(3), y.stride(0), y.stride(1), y.stride(2),
        num_heads, kernel_size, kernel_stride, head_dim, head_dim,
        block_output_seq_size, block_kernel_size, block_head_dim,
        block_headD_dim)
    ctx.save_for_backward(x, w, cu_seqlens, y_seqlens, y_cu_seqlens)
    ctx.kernel_size = kernel_size
    ctx.kernel_stride = kernel_stride
    ctx.block_kernel_size = block_kernel_size
    ctx.block_headd_dim = block_head_dim
    ctx.block_headD_dim = block_headD_dim
    ctx.block_output_seq_size = block_output_seq_size
    return y, y_cu_seqlens


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def linear_compress_bwd_kernel(DX, DY, DW, X, W, cu_seqlens_x, cu_seqlens_y,
    stride_xn, stride_xh, stride_xd, stride_wh, stride_wk, stride_wd,
    stride_wD, stride_dxn, stride_dxh, stride_dxd, stride_dwh, stride_dwk,
    stride_dwd, stride_dwD, stride_dyn, stride_dyh, stride_dyd, NUM_HEADS:
    tl.constexpr, NUM_PARALLEL_HEADd_NUM: tl.constexpr, KERNEL_SIZE: tl.
    constexpr, KERNEL_STRIDE: tl.constexpr, HEADd_DIM: tl.constexpr,
    HEADD_DIM: tl.constexpr, BLOCK_KERNEL_SIZE: tl.constexpr,
    BLOCK_HEADd_DIM: tl.constexpr, BLOCK_HEADD_DIM: tl.constexpr,
    BLOCK_OUTPUT_SEQ_SIZE: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // NUM_HEADS
    pid_h = pid_bh % NUM_HEADS
    pid_k = tl.program_id(1)
    pid_Dd = tl.program_id(2)
    pid_D = pid_Dd // NUM_PARALLEL_HEADd_NUM
    pid_d = pid_Dd % NUM_PARALLEL_HEADd_NUM
    x_start = tl.load(cu_seqlens_x + pid_b)
    x_end = tl.load(cu_seqlens_x + pid_b + 1)
    x_len = x_end - x_start
    y_start = tl.load(cu_seqlens_y + pid_b)
    y_end = tl.load(cu_seqlens_y + pid_b + 1)
    y_len = y_end - y_start
    if pid_k * BLOCK_OUTPUT_SEQ_SIZE >= y_len:
        return
    off_kernel_size = tl.arange(0, BLOCK_KERNEL_SIZE)
    off_d = tl.arange(0, BLOCK_HEADd_DIM)
    off_D = tl.arange(0, BLOCK_HEADD_DIM)
    off_output_seq_size = tl.arange(0, BLOCK_OUTPUT_SEQ_SIZE)
    x_ptrs = X + pid_h * stride_xh + x_start * stride_xn + ((pid_k *
        BLOCK_OUTPUT_SEQ_SIZE * KERNEL_STRIDE + off_output_seq_size *
        KERNEL_STRIDE)[:, None] + off_kernel_size[None, :])[:, :, None
        ] * stride_xn + (pid_d * BLOCK_HEADd_DIM + off_d)[None, None, :
        ] * stride_xd
    x_mask = ((pid_k * BLOCK_OUTPUT_SEQ_SIZE * KERNEL_STRIDE + 
        off_output_seq_size * KERNEL_STRIDE)[:, None] + off_kernel_size[
        None, :] < x_len)[:, :, None] & (pid_d * BLOCK_HEADd_DIM + off_d <
        HEADd_DIM)[None, None, :]
    dx_ptrs = DX + pid_h * stride_dxh + x_start * stride_dxn + ((pid_k *
        BLOCK_OUTPUT_SEQ_SIZE * KERNEL_STRIDE + off_output_seq_size *
        KERNEL_STRIDE)[:, None] + off_kernel_size[None, :])[:, :, None
        ] * stride_dxn + (pid_d * BLOCK_HEADd_DIM + off_d)[None, None, :
        ] * stride_dxd
    w_ptrs = tl.make_block_ptr(base=W + pid_h * stride_wh, shape=(
        KERNEL_SIZE, HEADd_DIM, HEADD_DIM), strides=(stride_wk, stride_wd,
        stride_wD), offsets=(0, pid_d * BLOCK_HEADd_DIM, pid_D *
        BLOCK_HEADD_DIM), block_shape=(BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM,
        BLOCK_HEADD_DIM), order=(2, 1, 0))
    dw_ptrs = DW + pid_h * stride_dwh + off_kernel_size[:, None, None
        ] * stride_dwk + (pid_d * BLOCK_HEADd_DIM + off_d)[None, :, None
        ] * stride_dwd + (pid_D * BLOCK_HEADD_DIM + off_D)[None, None, :
        ] * stride_dwD
    dw_mask = (off_kernel_size < KERNEL_SIZE)[:, None, None] & (pid_d *
        BLOCK_HEADd_DIM + off_d < HEADd_DIM)[None, :, None] & (pid_D *
        BLOCK_HEADD_DIM + off_D < HEADD_DIM)[None, None, :]
    dy_ptrs = tl.make_block_ptr(base=DY + y_start * stride_dyn + pid_h *
        stride_dyh, shape=(y_len, HEADD_DIM), strides=(stride_dyn,
        stride_dyd), offsets=(pid_k * BLOCK_OUTPUT_SEQ_SIZE, pid_D *
        BLOCK_HEADD_DIM), block_shape=(BLOCK_OUTPUT_SEQ_SIZE,
        BLOCK_HEADD_DIM), order=(1, 0))
    dy = tl.load(dy_ptrs, boundary_check=(0, 1), padding_option='zero')
    w = tl.load(w_ptrs, boundary_check=(0, 1, 2), padding_option='zero')
    w = tl.reshape(w, (BLOCK_KERNEL_SIZE * BLOCK_HEADd_DIM, BLOCK_HEADD_DIM))
    dx = tl.dot(dy, tl.trans(w))
    dx = tl.reshape(dx, (BLOCK_OUTPUT_SEQ_SIZE, BLOCK_KERNEL_SIZE,
        BLOCK_HEADd_DIM))
    tl.atomic_add(dx_ptrs, dx.to(dx_ptrs.dtype.element_ty), mask=x_mask)
    x = tl.load(x_ptrs, mask=x_mask, other=0)
    x = tl.reshape(x, (BLOCK_OUTPUT_SEQ_SIZE, BLOCK_KERNEL_SIZE *
        BLOCK_HEADd_DIM))
    dw = tl.dot(tl.trans(x), dy)
    dw = tl.reshape(dw, (BLOCK_KERNEL_SIZE, BLOCK_HEADd_DIM, BLOCK_HEADD_DIM))
    tl.atomic_add(dw_ptrs, dw.to(dw_ptrs.dtype.element_ty), mask=dw_mask)


# Backward method (kernel launch code)
def _LinearCompress_backward(ctx, dy: torch.Tensor, *args) ->Any:
    x, w, cu_seqlens, y_seqlens, y_cu_seqlens = ctx.saved_tensors
    kernel_size = ctx.kernel_size
    kernel_stride = ctx.kernel_stride
    block_kernel_size = ctx.block_kernel_size
    block_head_dim = ctx.block_headd_dim
    block_headD_dim = ctx.block_headD_dim
    block_output_seq_size = ctx.block_output_seq_size
    total_len, num_heads, head_dim = x.shape
    batch_size = cu_seqlens.shape[0] - 1
    dx = torch.zeros(cu_seqlens[-1], num_heads, head_dim, dtype=torch.
        float32, device=x.device)
    dw = torch.zeros(num_heads, kernel_size, head_dim, head_dim, dtype=
        torch.float32, device=x.device)
    grid = lambda META: (batch_size * num_heads, triton.cdiv(y_seqlens.max(
        0)[0].item(), META['BLOCK_OUTPUT_SEQ_SIZE']), triton.cdiv(head_dim,
        META['BLOCK_HEADD_DIM']) * triton.cdiv(head_dim, META[
        'BLOCK_HEADd_DIM']))
    linear_compress_bwd_kernel[grid](dx, dy, dw, x, w, cu_seqlens,
        y_cu_seqlens, x.stride(0), x.stride(1), x.stride(2), w.stride(0), w
        .stride(1), w.stride(2), w.stride(3), dx.stride(0), dx.stride(1),
        dx.stride(2), dw.stride(0), dw.stride(1), dw.stride(2), dw.stride(3
        ), dy.stride(0), dy.stride(1), dy.stride(2), num_heads, head_dim //
        block_head_dim, kernel_size, kernel_stride, head_dim, head_dim,
        block_kernel_size, block_head_dim, block_headD_dim,
        block_output_seq_size)
    return dx.to(x.dtype), rearrange(dw.to(x.dtype), 'n k d D -> n (k d) D'
        ), None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LinearCompress(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, cu_seqlens: torch.
        Tensor, kernel_size: int, kernel_stride: int) ->Tuple[torch.Tensor,
        torch.Tensor]:
        """
        Compress key and value tensor with kernel_size and kernel_stride. Similar to conv_compress.

        Args:
            x (torch.Tensor): key_states or value_states, shape (total_len, num_heads, head_dim)
            w (torch.Tensor): weight for each head, shape (num_heads, kernel_size * head_dim, head_dim)
            cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen
            kernel_size (int): kernel_size, each (kernel_size, head_dim) blocks will be compressed to (1, head_dim)
            kernel_stride (int): stride for each compress kernel

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: compressed states and corresponding cu_seqlens.
        """
        assert x.dtype == torch.float16 or x.dtype == torch.bfloat16
        assert x.dtype == w.dtype
        assert cu_seqlens.dtype == torch.int32
        total_len, num_heads, head_dim = x.shape
        batch_size = cu_seqlens.shape[0] - 1
        assert w.shape[0] == num_heads
        assert w.shape[1] == kernel_size * head_dim
        assert w.shape[2] == head_dim
        assert kernel_size % kernel_stride == 0
        assert kernel_size in {16, 32, 64, 128}
        assert head_dim % 8 == 0
        torch.cuda.set_device(x.device)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(
            torch.int32) + 1
        y_seqlens[seqlens < kernel_size] = 0
        y_cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device=
            x.device), torch.cumsum(y_seqlens.to(x.device), dim=0)], dim=0).to(
            torch.int32)
        y = torch.zeros(y_cu_seqlens[-1], num_heads, head_dim, dtype=x.
            dtype, device=x.device)
        block_kernel_size = max(16, triton.next_power_of_2(kernel_size))
        block_head_dim = 8 if IS_HOPPER_GPU else 4
        block_headD_dim = 32
        block_output_seq_size = 64
        w = w.reshape(num_heads, kernel_size, head_dim, head_dim).contiguous()
        grid = lambda META: (batch_size * num_heads, triton.cdiv(y_seqlens.
            max(0)[0].item(), META['BLOCK_OUTPUT_SEQ_SIZE']), triton.cdiv(
            head_dim, META['BLOCK_HEADD_DIM']))
        linear_compress_fwd_kernel[grid](x, y, w, cu_seqlens, y_cu_seqlens,
            x.stride(0), x.stride(1), x.stride(2), w.stride(0), w.stride(1),
            w.stride(2), w.stride(3), y.stride(0), y.stride(1), y.stride(2),
            num_heads, kernel_size, kernel_stride, head_dim, head_dim,
            block_output_seq_size, block_kernel_size, block_head_dim,
            block_headD_dim)
        ctx.save_for_backward(x, w, cu_seqlens, y_seqlens, y_cu_seqlens)
        ctx.kernel_size = kernel_size
        ctx.kernel_stride = kernel_stride
        ctx.block_kernel_size = block_kernel_size
        ctx.block_headd_dim = block_head_dim
        ctx.block_headD_dim = block_headD_dim
        ctx.block_output_seq_size = block_output_seq_size
        return y, y_cu_seqlens

    @staticmethod
    def backward(ctx, dy: torch.Tensor, *args) ->Any:
        x, w, cu_seqlens, y_seqlens, y_cu_seqlens = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        kernel_stride = ctx.kernel_stride
        block_kernel_size = ctx.block_kernel_size
        block_head_dim = ctx.block_headd_dim
        block_headD_dim = ctx.block_headD_dim
        block_output_seq_size = ctx.block_output_seq_size
        total_len, num_heads, head_dim = x.shape
        batch_size = cu_seqlens.shape[0] - 1
        dx = torch.zeros(cu_seqlens[-1], num_heads, head_dim, dtype=torch.
            float32, device=x.device)
        dw = torch.zeros(num_heads, kernel_size, head_dim, head_dim, dtype=
            torch.float32, device=x.device)
        grid = lambda META: (batch_size * num_heads, triton.cdiv(y_seqlens.
            max(0)[0].item(), META['BLOCK_OUTPUT_SEQ_SIZE']), triton.cdiv(
            head_dim, META['BLOCK_HEADD_DIM']) * triton.cdiv(head_dim, META
            ['BLOCK_HEADd_DIM']))
        linear_compress_bwd_kernel[grid](dx, dy, dw, x, w, cu_seqlens,
            y_cu_seqlens, x.stride(0), x.stride(1), x.stride(2), w.stride(0
            ), w.stride(1), w.stride(2), w.stride(3), dx.stride(0), dx.
            stride(1), dx.stride(2), dw.stride(0), dw.stride(1), dw.stride(
            2), dw.stride(3), dy.stride(0), dy.stride(1), dy.stride(2),
            num_heads, head_dim // block_head_dim, kernel_size,
            kernel_stride, head_dim, head_dim, block_kernel_size,
            block_head_dim, block_headD_dim, block_output_seq_size)
        return dx.to(x.dtype), rearrange(dw.to(x.dtype), 'n k d D -> n (k d) D'
            ), None, None, None
