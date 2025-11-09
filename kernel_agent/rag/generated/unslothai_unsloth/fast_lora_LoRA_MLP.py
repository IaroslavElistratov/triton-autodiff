# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/unslothai/unsloth
# Source-Files: unsloth/kernels/fast_lora.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_cfgjhluw/unsloth-main/unsloth/kernels/fast_lora.py
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

@torch_compile
def fbgemm_fp8_linear(X, weight, weight_scale, bias=None):
    return FbgemmFp8Linear.apply(X, weight, weight_scale, bias)


@torch_compile
def fp8_block_quant_forward(X, weight, weight_scale):
    return FP8BlockQuantLinear.apply(X, weight, weight_scale)


@torch_compile
def fp8_linear(X, weight, weight_scale, bias=None):
    if weight_scale.ndim == 2 and weight_scale.shape[1] > 1:
        out = fp8_block_quant_forward(X, weight, weight_scale)
    else:
        out = fbgemm_fp8_linear(X, weight, weight_scale, bias)
    return out


def weight_dequant(x: torch.Tensor, s: torch.Tensor, dtype=torch.bfloat16):
    if s.shape[1] == 1:
        if x.shape[0] == s.shape[0]:
            y = x.to(dtype) * s.to(dtype)
        elif x.shape[1] == s.shape[0]:
            y = x.t().to(dtype) * s.to(dtype)
            y = y.t()
        else:
            raise ValueError(
                f'Incompatible shapes x.shape={x.shape!r}, s.shape={s.shape!r}'
                )
        return y
    else:
        return weight_dequant_block(x, s, dtype=dtype)


def weight_dequant_block(x: torch.Tensor, s: torch.Tensor, block_size: int=
    128, dtype=torch.bfloat16) ->torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N,
        meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def cdequantize_blockwise_fp32(*args, **kwargs):
    raise RuntimeError(
        'XPU BNB support is not implemented yet. cdequantize_blockwise_fp32 should not be called now.'
        )


@torch.inference_mode
def fast_dequantize(W, quant_state=None, out=None, use_global_buffer=False):
    if quant_state is None:
        return W
    if W.dtype == torch.float8_e4m3fn:
        return weight_dequant(W, quant_state)
    if type(quant_state) is not list:
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2
    pass
    n_elements_absmax = absmax.numel()
    device = W.device
    if out is None:
        out = torch_empty(shape, dtype=dtype, device=device, requires_grad=
            False)
    else:
        assert out.shape == shape
        assert out.dtype == dtype
    out_absmax = torch_empty(n_elements_absmax, dtype=torch_float32, device
        =device, requires_grad=False)
    ptr_out_absmax = get_ptr(out_absmax)
    cdequantize_blockwise_fp32(get_ptr(code2), get_ptr(absmax), get_ptr(
        absmax2), ptr_out_absmax, ctypes_c_int(blocksize2), ctypes_c_int(
        n_elements_absmax))
    out_absmax += offset
    fx = (cdequantize_blockwise_fp16_nf4 if dtype == torch_float16 else
        cdequantize_blockwise_bf16_nf4)
    fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
        ctypes_c_int(blocksize), ctypes_c_int(out.numel()))
    is_transposed = True if W.shape[0] == 1 else False
    return out.t() if is_transposed else out


def get_ptr(x: Optional[torch.Tensor]):
    raise RuntimeError(
        'XPU BNB support is not implemented yet. This function should not be called.'
        )


def matmul_lora(X, W, W_quant, A, B, s, out=None):
    dtype = X.dtype
    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False
    pass
    if W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant)
    else:
        W = fast_dequantize(W.t(), W_quant, use_global_buffer=True)
        out = torch_matmul(X, W, out=out)
    if W_quant is not None:
        del W
    if A is not None:
        A, B = A.t(), B.t()
        XA = torch_matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha=s)
    pass
    return out.view(batch, seq_len, -1) if reshape else out


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

# Forward method (kernel launch code)
@torch_amp_custom_fwd
def _LoRA_MLP_forward(ctx, X: torch.Tensor, gateW, gateW_quant, gateA,
    gateB, gateS, upW, upW_quant, upA, upB, upS, downW, downW_quant, downA,
    downB, downS, _forward_function, _backward_function, inplace=True):
    dtype = X.dtype
    e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
    g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
    h = _forward_function(e, g)
    i = matmul_lora(h, downW, downW_quant, downA, downB, downS)
    ctx.custom_saved_tensors = (gateW, gateW_quant, gateS, upW, upW_quant,
        upS, downW, downW_quant, downS, _backward_function)
    ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
    ctx.inplace = inplace
    return i


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
@torch_amp_custom_bwd
def _LoRA_MLP_backward(ctx, dY: torch.Tensor):
    (gateW, gateW_quant, gateS, upW, upW_quant, upS, downW, downW_quant,
        downS, _backward_function) = ctx.custom_saved_tensors
    gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors
    batch, seq_len, hd = X.shape
    dY = dY.view(-1, dY.shape[-1])
    X = X.view(-1, X.shape[-1])
    e = e.view(-1, e.shape[-1])
    g = g.view(-1, g.shape[-1])
    dtype = X.dtype
    gateA, gateB, upA, upB, downA, downB = gateA.to(dtype), gateB.to(dtype
        ), upA.to(dtype), upB.to(dtype), downA.to(dtype), downB.to(dtype)
    gateA, gateB, upA, upB, downA, downB = gateA.t(), gateB.t(), upA.t(
        ), upB.t(), downA.t(), downB.t()
    DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
    DW, e, g = _backward_function(DW, e, g)
    h, df, de = DW, e, g
    d_downA = torch.empty_like(downA)
    d_downB = torch.empty_like(downB)
    d_gateA = torch.empty_like(gateA)
    d_gateB = torch.empty_like(gateB)
    d_upA = torch.empty_like(upA)
    d_upB = torch.empty_like(upB)
    d_downA.addmm_(h.t(), dY @ downB.t(), alpha=downS, beta=0)
    d_downB.addmm_(downA.t() @ h.t(), dY, alpha=downS, beta=0)
    d_upA.addmm_(X.t(), df @ upB.t(), alpha=upS, beta=0)
    d_upB.addmm_(upA.t() @ X.t(), df, alpha=upS, beta=0)
    d_gateA.addmm_(X.t(), de @ gateB.t(), alpha=gateS, beta=0)
    d_gateB.addmm_(gateA.t() @ X.t(), de, alpha=gateS, beta=0)
    upW = fast_dequantize(upW.t(), upW_quant)
    dX = torch.matmul(df, upW.t(), out=X if ctx.inplace else None)
    del upW
    dX.addmm_(df @ upB.t(), upA.t(), alpha=upS)
    gateW = fast_dequantize(gateW.t(), gateW_quant)
    dX.addmm_(de, gateW.t())
    del gateW
    dX.addmm_(de @ gateB.t(), gateA.t(), alpha=gateS)
    return dX.view(batch, seq_len, hd), None, None, d_gateA.t(), d_gateB.t(
        ), None, None, None, d_upA.t(), d_upB.t(), None, None, None, d_downA.t(
        ), d_downB.t(), None, None, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LoRA_MLP(torch.autograd.Function):
    """
    ### LoRA weights
    G = G + Ag @ Bg
    U = U + Au @ Bu
    W = W + Aw @ Bw

    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)

    ### Down projection LoRA weights
    dC/dAw = dC/dW @ B.T
    dC/dBw = A.T @ dC/dW
    dC/dAw =       h.T @ dY @ B.T
    dC/dBw = A.T @ h.T @ dY

    ### Up projection LoRA weights
    dC/dAu =       X.T @ (D @ W.T * f) @ B.T
    dC/dBu = A.T @ X.T @ (D @ W.T * f)

    ### Gate projection LoRA weights
    dC/dAg =       X.T @ (D @ W.T * df * g) @ B.T
    dC/dBg = A.T @ X.T @ (D @ W.T * df * g)

    Don't forget to see our blog post for more details!
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X: torch.Tensor, gateW, gateW_quant, gateA, gateB,
        gateS, upW, upW_quant, upA, upB, upS, downW, downW_quant, downA,
        downB, downS, _forward_function, _backward_function, inplace=True):
        dtype = X.dtype
        e = matmul_lora(X, gateW, gateW_quant, gateA, gateB, gateS)
        g = matmul_lora(X, upW, upW_quant, upA, upB, upS)
        h = _forward_function(e, g)
        i = matmul_lora(h, downW, downW_quant, downA, downB, downS)
        ctx.custom_saved_tensors = (gateW, gateW_quant, gateS, upW,
            upW_quant, upS, downW, downW_quant, downS, _backward_function)
        ctx.save_for_backward(gateA, gateB, upA, upB, downA, downB, X, e, g)
        ctx.inplace = inplace
        return i
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY: torch.Tensor):
        (gateW, gateW_quant, gateS, upW, upW_quant, upS, downW, downW_quant,
            downS, _backward_function) = ctx.custom_saved_tensors
        gateA, gateB, upA, upB, downA, downB, X, e, g = ctx.saved_tensors
        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X = X.view(-1, X.shape[-1])
        e = e.view(-1, e.shape[-1])
        g = g.view(-1, g.shape[-1])
        dtype = X.dtype
        gateA, gateB, upA, upB, downA, downB = gateA.to(dtype), gateB.to(dtype
            ), upA.to(dtype), upB.to(dtype), downA.to(dtype), downB.to(dtype)
        gateA, gateB, upA, upB, downA, downB = gateA.t(), gateB.t(), upA.t(
            ), upB.t(), downA.t(), downB.t()
        DW = matmul_lora(dY, downW.t(), downW_quant, downB, downA, downS)
        DW, e, g = _backward_function(DW, e, g)
        h, df, de = DW, e, g
        d_downA = torch.empty_like(downA)
        d_downB = torch.empty_like(downB)
        d_gateA = torch.empty_like(gateA)
        d_gateB = torch.empty_like(gateB)
        d_upA = torch.empty_like(upA)
        d_upB = torch.empty_like(upB)
        d_downA.addmm_(h.t(), dY @ downB.t(), alpha=downS, beta=0)
        d_downB.addmm_(downA.t() @ h.t(), dY, alpha=downS, beta=0)
        d_upA.addmm_(X.t(), df @ upB.t(), alpha=upS, beta=0)
        d_upB.addmm_(upA.t() @ X.t(), df, alpha=upS, beta=0)
        d_gateA.addmm_(X.t(), de @ gateB.t(), alpha=gateS, beta=0)
        d_gateB.addmm_(gateA.t() @ X.t(), de, alpha=gateS, beta=0)
        upW = fast_dequantize(upW.t(), upW_quant)
        dX = torch.matmul(df, upW.t(), out=X if ctx.inplace else None)
        del upW
        dX.addmm_(df @ upB.t(), upA.t(), alpha=upS)
        gateW = fast_dequantize(gateW.t(), gateW_quant)
        dX.addmm_(de, gateW.t())
        del gateW
        dX.addmm_(de @ gateB.t(), gateA.t(), alpha=gateS)
        return dX.view(batch, seq_len, hd), None, None, d_gateA.t(), d_gateB.t(
            ), None, None, None, d_upA.t(), d_upB.t(
            ), None, None, None, d_downA.t(), d_downB.t(
            ), None, None, None, None
    pass
