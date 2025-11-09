# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/woct0rdho/transformers-qwen3-moe-fused
# Source-Files: qwen3_moe_fused/kernels/fast_lora.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_n8hgmd0z/transformers-qwen3-moe-fused-master/qwen3_moe_fused/kernels/fast_lora.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

def get_num_sms() ->int:
    return torch.cuda.get_device_properties('cuda').multi_processor_count


def is_int_tensor(x: torch.Tensor) ->bool:
    return x.dtype in {torch.uint8, torch.int8, torch.int16, torch.int32,
        torch.int64}


def _maybe_dequant(weight, quant_state):
    if quant_state is None:
        return weight
    else:
        out = dequantize_4bit(weight, quant_state)
        return out


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=get_autotune_configs(), prune_configs_by={
    'early_config_prune': prune_configs}, key=get_autotune_keys())
@triton.jit
def _grouped_gemm_forward_kernel(x_ptr, w_ptr, m_sizes_ptr, y_ptr, M: int,
    N: tl.constexpr, K: tl.constexpr, NUM_EXPERTS: tl.constexpr, NUM_SMS:
    tl.constexpr, stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_we: tl.constexpr, stride_wn: tl.constexpr, stride_wk: tl.
    constexpr, stride_ym: tl.constexpr, stride_yn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr=64, BLOCK_SIZE_N: tl.constexpr=64,
    BLOCK_SIZE_K: tl.constexpr=64) ->None:
    tidx = tl.program_id(0)
    m_end = 0
    processed_tiles = 0
    for expert_idx in range(NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        if m_size > 0:
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles
            while (tidx >= processed_tiles and tidx < processed_tiles +
                num_tiles_per_expert):
                tile_idx = tidx - processed_tiles
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                offs_m = m_start + tile_m_idx * BLOCK_SIZE_M + tl.arange(0,
                    BLOCK_SIZE_M)
                x_ptrs = x_ptr + stride_xm * offs_m[:, None
                    ] + stride_xk * offs_k[None, :]
                mask_m = offs_m < m_end
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[
                    :, None] + stride_wk * offs_k[None, :]
                mask_n = offs_n < N
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=
                    tl.float32)
                for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offs_k < K
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                    w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :])
                    accumulator += tl.dot(x, w.T)
                    offs_k += BLOCK_SIZE_K
                    x_ptrs += stride_xk * BLOCK_SIZE_K
                    w_ptrs += stride_wk * BLOCK_SIZE_K
                y = accumulator.to(y_ptr.dtype.element_ty)
                y_ptrs = y_ptr + stride_ym * offs_m[:, None
                    ] + stride_yn * offs_n[None, :]
                tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])
                tidx += NUM_SMS
            processed_tiles += num_tiles_per_expert


@triton.autotune(configs=_autotune_configs, key=[])
@triton.jit
def _silu_mul_forward_kernel(e_ptr, g_ptr, h_ptr, n_elements: int,
    BLOCK_SIZE: tl.constexpr=128) ->None:
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    e = tl.load(e_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask)
    f = e * tl.sigmoid(e)
    f = f.to(e_ptr.dtype.element_ty)
    h = f * g
    tl.store(h_ptr + offsets, h, mask=mask)


def grouped_gemm_forward(x: torch.Tensor, w: torch.Tensor, m_sizes: torch.
    Tensor, dtype: Optional[torch.dtype]=None) ->torch.Tensor:
    assert x.is_cuda
    assert w.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_sizes.ndim == 1
    M, K = x.shape
    E, N, _ = w.shape
    assert w.shape[2] == K
    assert m_sizes.numel() == E
    if dtype is None:
        dtype = x.dtype
    y = torch.empty((M, N), device=x.device, dtype=dtype)
    NUM_SMS = get_num_sms()
    grid = lambda META: (NUM_SMS,)
    _grouped_gemm_forward_kernel[grid](x, w, m_sizes, y, M, N, K, E,
        NUM_SMS, x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.
        stride(2), y.stride(0), y.stride(1))
    return y


def silu_mul_forward(e: torch.Tensor, g: torch.Tensor) ->torch.Tensor:
    assert e.is_cuda
    assert g.device == e.device
    assert e.is_contiguous()
    assert g.is_contiguous()
    assert g.numel() == e.numel()
    n_elements = e.numel()
    h = torch.empty_like(e)
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    _silu_mul_forward_kernel[grid](e, g, h, n_elements)
    return h


# Forward method (kernel launch code)
def _FastLora_forward(ctx, x, Gq, Gqs, Ag, Bg, Sg, Uq, Uqs, Au, Bu, Su, Wq,
    Wqs, Aw, Bw, Sw, m_sizes):
    if Gqs is None:
        Gq = Gq.to(x.dtype)
    else:
        Gqs.dtype = x.dtype
    Ag = Ag.to(x.dtype)
    Bg = Bg.to(x.dtype)
    if Uqs is None:
        Uq = Uq.to(x.dtype)
    else:
        Uqs.dtype = x.dtype
    Au = Au.to(x.dtype)
    Bu = Bu.to(x.dtype)
    if Wqs is None:
        Wq = Wq.to(x.dtype)
    else:
        Wqs.dtype = x.dtype
    Aw = Aw.to(x.dtype)
    Bw = Bw.to(x.dtype)

    def mv(_w, _x):
        return grouped_gemm_forward(_x, _w, m_sizes, x.dtype)

    def mv_lora(_x, _wq, _wqs, _a, _b, _s):
        _w = _maybe_dequant(_wq, _wqs)
        _y = mv(_w, _x)
        _y += mv(_b, mv(_a, _x)) * _s
        return _y
    e = mv_lora(x, Gq, Gqs, Ag, Bg, Sg)
    g = mv_lora(x, Uq, Uqs, Au, Bu, Su)
    h = silu_mul_forward(e, g)
    y = mv_lora(h, Wq, Wqs, Aw, Bw, Sw)
    ctx.custom_saved_tensors = Gq, Gqs, Sg, Uq, Uqs, Su, Wq, Wqs, Sw, m_sizes
    ctx.save_for_backward(x, Ag, Bg, Au, Bu, Aw, Bw, e, g)
    return y


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.autotune(configs=get_autotune_configs(), prune_configs_by={
    'early_config_prune': prune_configs}, key=get_autotune_keys())
@triton.jit
def _grouped_gemm_backward_dw_kernel(x_ptr, y_ptr, m_sizes_ptr, w_ptr, M:
    int, N: tl.constexpr, K: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    NUM_SMS: tl.constexpr, stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_ym: tl.constexpr, stride_yn: tl.constexpr, stride_we: tl.
    constexpr, stride_wn: tl.constexpr, stride_wk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr=64, BLOCK_SIZE_N: tl.constexpr=64,
    BLOCK_SIZE_K: tl.constexpr=64) ->None:
    tidx = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles_per_expert = num_n_tiles * num_k_tiles
    for tile_idx in range(tidx, num_tiles_per_expert, NUM_SMS):
        tile_n_idx = tile_idx % num_n_tiles
        tile_k_idx = tile_idx // num_n_tiles
        offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tile_k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_n = offs_n < N
        mask_k = offs_k < K
        m_end = 0
        for expert_idx in range(NUM_EXPERTS):
            m_start = m_end
            m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
            m_end = m_start + m_size
            if m_size > 0:
                offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
                x_ptrs = x_ptr + stride_xm * offs_m[:, None
                    ] + stride_xk * offs_k[None, :]
                y_ptrs = y_ptr + stride_ym * offs_m[:, None
                    ] + stride_yn * offs_n[None, :]
                accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=
                    tl.float32)
                for _ in range(tl.cdiv(m_size, BLOCK_SIZE_M)):
                    mask_m = offs_m < m_end
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                    y = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :])
                    accumulator += tl.dot(y.T, x)
                    offs_m += BLOCK_SIZE_M
                    x_ptrs += stride_xm * BLOCK_SIZE_M
                    y_ptrs += stride_ym * BLOCK_SIZE_M
                w = accumulator.to(w_ptr.dtype.element_ty)
                w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[
                    :, None] + stride_wk * offs_k[None, :]
                tl.store(w_ptrs, w, mask=mask_n[:, None] & mask_k[None, :])


@triton.autotune(configs=get_autotune_configs(), prune_configs_by={
    'early_config_prune': prune_configs}, key=get_autotune_keys())
@triton.jit
def _grouped_gemm_forward_transposed_kernel(x_ptr, w_ptr, m_sizes_ptr,
    y_ptr, M: int, N: tl.constexpr, K: tl.constexpr, NUM_EXPERTS: tl.
    constexpr, NUM_SMS: tl.constexpr, stride_xm: tl.constexpr, stride_xk:
    tl.constexpr, stride_we: tl.constexpr, stride_wk: tl.constexpr,
    stride_wn: tl.constexpr, stride_ym: tl.constexpr, stride_yn: tl.
    constexpr, BLOCK_SIZE_M: tl.constexpr=64, BLOCK_SIZE_N: tl.constexpr=64,
    BLOCK_SIZE_K: tl.constexpr=64) ->None:
    tidx = tl.program_id(0)
    m_end = 0
    processed_tiles = 0
    for expert_idx in range(NUM_EXPERTS):
        m_start = m_end
        m_size = tl.load(m_sizes_ptr + expert_idx).to(tl.int32)
        m_end = m_start + m_size
        if m_size > 0:
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
            num_tiles_per_expert = num_m_tiles * num_n_tiles
            while (tidx >= processed_tiles and tidx < processed_tiles +
                num_tiles_per_expert):
                tile_idx = tidx - processed_tiles
                tile_m_idx = tile_idx % num_m_tiles
                tile_n_idx = tile_idx // num_m_tiles
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                offs_m = m_start + tile_m_idx * BLOCK_SIZE_M + tl.arange(0,
                    BLOCK_SIZE_M)
                x_ptrs = x_ptr + stride_xm * offs_m[:, None
                    ] + stride_xk * offs_k[None, :]
                mask_m = offs_m < m_start + m_size
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_ptrs = w_ptr + stride_we * expert_idx + stride_wn * offs_n[
                    :, None] + stride_wk * offs_k[None, :]
                mask_n = offs_n < N
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=
                    tl.float32)
                for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offs_k < K
                    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :])
                    w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :])
                    accumulator += tl.dot(x, w.T)
                    offs_k += BLOCK_SIZE_K
                    x_ptrs += stride_xk * BLOCK_SIZE_K
                    w_ptrs += stride_wk * BLOCK_SIZE_K
                y = accumulator.to(y_ptr.dtype.element_ty)
                y_ptrs = y_ptr + stride_ym * offs_m[:, None
                    ] + stride_yn * offs_n[None, :]
                tl.store(y_ptrs, y, mask=mask_m[:, None] & mask_n[None, :])
                tidx += NUM_SMS
            processed_tiles += num_tiles_per_expert


@triton.autotune(configs=_autotune_configs, key=[])
@triton.jit
def _silu_mul_backward_inplace_kernel(dh_ptr, e_ptr, g_ptr, n_elements: int,
    BLOCK_SIZE: tl.constexpr=128) ->None:
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dh = tl.load(dh_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offsets, mask=mask)
    se = tl.sigmoid(e)
    ese = e * se
    f = ese.to(e_ptr.dtype.element_ty)
    h = f * g
    df = dh * g
    dg = f * dh
    de = df.to(tl.float32) * se * (1.0 + e - ese)
    de = de.to(e_ptr.dtype.element_ty)
    tl.store(dh_ptr + offsets, h, mask=mask)
    tl.store(e_ptr + offsets, de, mask=mask)
    tl.store(g_ptr + offsets, dg, mask=mask)


def grouped_gemm_backward_dw(x: torch.Tensor, y: torch.Tensor, m_sizes:
    torch.Tensor, dtype: torch.dtype) ->torch.Tensor:
    assert x.is_cuda
    assert y.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert y.ndim == 2
    assert m_sizes.ndim == 1
    M, K = x.shape
    _, N = y.shape
    assert y.shape[0] == M
    E = m_sizes.numel()
    w = torch.zeros((E, N, K), device=x.device, dtype=dtype)
    NUM_SMS = get_num_sms()
    grid = lambda META: (NUM_SMS,)
    _grouped_gemm_backward_dw_kernel[grid](x, y, m_sizes, w, M, N, K, E,
        NUM_SMS, x.stride(0), x.stride(1), y.stride(0), y.stride(1), w.
        stride(0), w.stride(1), w.stride(2))
    return w


def grouped_gemm_forward_transposed(x: torch.Tensor, w: torch.Tensor,
    m_sizes: torch.Tensor, dtype: Optional[torch.dtype]=None) ->torch.Tensor:
    assert x.is_cuda
    assert w.device == x.device
    assert m_sizes.device == x.device
    assert is_int_tensor(m_sizes)
    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()
    assert x.ndim == 2
    assert w.ndim == 3
    assert m_sizes.ndim == 1
    M, K = x.shape
    E, _, N = w.shape
    assert w.shape[1] == K
    assert m_sizes.numel() == E
    if dtype is None:
        dtype = x.dtype
    y = torch.empty((M, N), device=x.device, dtype=dtype)
    NUM_SMS = get_num_sms()
    grid = lambda META: (NUM_SMS,)
    _grouped_gemm_forward_transposed_kernel[grid](x, w, m_sizes, y, M, N, K,
        E, NUM_SMS, x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.
        stride(2), y.stride(0), y.stride(1))
    return y


def silu_mul_backward_inplace(dh: torch.Tensor, e: torch.Tensor, g: torch.
    Tensor) ->None:
    assert e.is_cuda
    assert g.device == e.device
    assert dh.device == e.device
    assert e.is_contiguous()
    assert g.is_contiguous()
    assert dh.is_contiguous()
    assert g.numel() == e.numel()
    assert dh.numel() == e.numel()
    n_elements = e.numel()
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    _silu_mul_backward_inplace_kernel[grid](dh, e, g, n_elements)


# Backward method (kernel launch code)
def _FastLora_backward(ctx, dy):
    Gq, Gqs, Sg, Uq, Uqs, Su, Wq, Wqs, Sw, m_sizes = ctx.custom_saved_tensors
    x, Ag, Bg, Au, Bu, Aw, Bw, e, g = ctx.saved_tensors

    def vm(_x, _w):
        return grouped_gemm_forward_transposed(_x, _w, m_sizes, x.dtype)

    def vv(_y, _x):
        return grouped_gemm_backward_dw(_x, _y, m_sizes, x.dtype)

    def emv(_w, _x):
        return torch.einsum('eri,eji->erj', _w, _x).to(x.dtype)

    def evm(_x, _w):
        return torch.einsum('eij,eir->ejr', _x, _w).to(x.dtype)

    def vm_lora(_x, _wq, _wqs, _a, _b, _s):
        _w = _maybe_dequant(_wq, _wqs)
        _y = vm(_x, _w)
        _y += vm(vm(_x, _b), _a) * _s
        return _y
    dh = vm_lora(dy, Wq, Wqs, Aw, Bw, Sw)
    silu_mul_backward_inplace(dh, e, g)
    h, de, dg = dh, e, g
    del dh, e, g
    dW = vv(dy, h)
    dAw = evm(Bw, dW) * Sw
    dBw = emv(dW, Aw) * Sw
    dU = vv(dg, x)
    dAu = evm(Bu, dU) * Su
    dBu = emv(dU, Au) * Su
    dG = vv(de, x)
    dAg = evm(Bg, dG) * Sg
    dBg = emv(dG, Ag) * Sg
    dx = vm_lora(de, Gq, Gqs, Ag, Bg, Sg)
    dx += vm_lora(dg, Uq, Uqs, Au, Bu, Su)
    return (dx, None, None, dAg, dBg, None, None, None, dAu, dBu, None,
        None, None, dAw, dBw, None, None)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class FastLora(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, Gq, Gqs, Ag, Bg, Sg, Uq, Uqs, Au, Bu, Su, Wq, Wqs,
        Aw, Bw, Sw, m_sizes):
        if Gqs is None:
            Gq = Gq.to(x.dtype)
        else:
            Gqs.dtype = x.dtype
        Ag = Ag.to(x.dtype)
        Bg = Bg.to(x.dtype)
        if Uqs is None:
            Uq = Uq.to(x.dtype)
        else:
            Uqs.dtype = x.dtype
        Au = Au.to(x.dtype)
        Bu = Bu.to(x.dtype)
        if Wqs is None:
            Wq = Wq.to(x.dtype)
        else:
            Wqs.dtype = x.dtype
        Aw = Aw.to(x.dtype)
        Bw = Bw.to(x.dtype)

        def mv(_w, _x):
            return grouped_gemm_forward(_x, _w, m_sizes, x.dtype)

        def mv_lora(_x, _wq, _wqs, _a, _b, _s):
            _w = _maybe_dequant(_wq, _wqs)
            _y = mv(_w, _x)
            _y += mv(_b, mv(_a, _x)) * _s
            return _y
        e = mv_lora(x, Gq, Gqs, Ag, Bg, Sg)
        g = mv_lora(x, Uq, Uqs, Au, Bu, Su)
        h = silu_mul_forward(e, g)
        y = mv_lora(h, Wq, Wqs, Aw, Bw, Sw)
        ctx.custom_saved_tensors = (Gq, Gqs, Sg, Uq, Uqs, Su, Wq, Wqs, Sw,
            m_sizes)
        ctx.save_for_backward(x, Ag, Bg, Au, Bu, Aw, Bw, e, g)
        return y

    @staticmethod
    def backward(ctx, dy):
        Gq, Gqs, Sg, Uq, Uqs, Su, Wq, Wqs, Sw, m_sizes = (ctx.
            custom_saved_tensors)
        x, Ag, Bg, Au, Bu, Aw, Bw, e, g = ctx.saved_tensors

        def vm(_x, _w):
            return grouped_gemm_forward_transposed(_x, _w, m_sizes, x.dtype)

        def vv(_y, _x):
            return grouped_gemm_backward_dw(_x, _y, m_sizes, x.dtype)

        def emv(_w, _x):
            return torch.einsum('eri,eji->erj', _w, _x).to(x.dtype)

        def evm(_x, _w):
            return torch.einsum('eij,eir->ejr', _x, _w).to(x.dtype)

        def vm_lora(_x, _wq, _wqs, _a, _b, _s):
            _w = _maybe_dequant(_wq, _wqs)
            _y = vm(_x, _w)
            _y += vm(vm(_x, _b), _a) * _s
            return _y
        dh = vm_lora(dy, Wq, Wqs, Aw, Bw, Sw)
        silu_mul_backward_inplace(dh, e, g)
        h, de, dg = dh, e, g
        del dh, e, g
        dW = vv(dy, h)
        dAw = evm(Bw, dW) * Sw
        dBw = emv(dW, Aw) * Sw
        dU = vv(dg, x)
        dAu = evm(Bu, dU) * Su
        dBu = emv(dU, Au) * Su
        dG = vv(de, x)
        dAg = evm(Bg, dG) * Sg
        dBg = emv(dG, Ag) * Sg
        dx = vm_lora(de, Gq, Gqs, Ag, Bg, Sg)
        dx += vm_lora(dg, Uq, Uqs, Au, Bu, Su)
        return (dx, None, None, dAg, dBg, None, None, None, dAu, dBu, None,
            None, None, dAw, dBw, None, None)
