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


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

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


# Forward method (kernel launch code)
@torch_amp_custom_fwd
def _LoRA_QKV_forward(ctx, X: torch.Tensor, QW, QW_quant, QA, QB, QS, KW,
    KW_quant, KA, KB, KS, VW, VW_quant, VA, VB, VS, inplace=True):
    dtype = X.dtype
    Q = matmul_lora(X, QW, QW_quant, QA, QB, QS)
    K = matmul_lora(X, KW, KW_quant, KA, KB, KS)
    V = matmul_lora(X, VW, VW_quant, VA, VB, VS)
    ctx.custom_saved_tensors = (QW, QW_quant, QS, KW, KW_quant, KS, VW,
        VW_quant, VS)
    ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB)
    ctx.inplace = inplace
    return Q, K, V


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

# Backward method (kernel launch code)
@torch_amp_custom_bwd
def _LoRA_QKV_backward(ctx, dQ, dK, dV):
    QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = (ctx.
        custom_saved_tensors)
    X, QA, QB, KA, KB, VA, VB = ctx.saved_tensors
    batch, seq_len, hd = X.shape
    dQ = dQ.view(-1, dQ.shape[-1])
    dK = dK.reshape(-1, dK.shape[-1])
    dV = dV.view(-1, dV.shape[-1])
    X = X.view(-1, X.shape[-1])
    dtype = X.dtype
    QA, QB, KA, KB, VA, VB = QA.to(dtype), QB.to(dtype), KA.to(dtype), KB.to(
        dtype), VA.to(dtype), VB.to(dtype)
    QA, QB, KA, KB, VA, VB = QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()
    d_QA = torch.empty_like(QA)
    d_QB = torch.empty_like(QB)
    d_KA = torch.empty_like(KA)
    d_KB = torch.empty_like(KB)
    d_VA = torch.empty_like(VA)
    d_VB = torch.empty_like(VB)
    d_QA.addmm_(X.t(), dQ @ QB.t(), alpha=QS, beta=0)
    d_QB.addmm_(QA.t() @ X.t(), dQ, alpha=QS, beta=0)
    d_KA.addmm_(X.t(), dK @ KB.t(), alpha=KS, beta=0)
    d_KB.addmm_(KA.t() @ X.t(), dK, alpha=KS, beta=0)
    d_VA.addmm_(X.t(), dV @ VB.t(), alpha=VS, beta=0)
    d_VB.addmm_(VA.t() @ X.t(), dV, alpha=VS, beta=0)
    QW = fast_dequantize(QW.t(), QW_quant)
    dX = torch.matmul(dQ, QW.t(), out=X if ctx.inplace else None)
    del QW
    dX.addmm_(dQ @ QB.t(), QA.t(), alpha=QS)
    KW = fast_dequantize(KW.t(), KW_quant)
    dX.addmm_(dK, KW.t())
    del KW
    dX.addmm_(dK @ KB.t(), KA.t(), alpha=KS)
    VW = fast_dequantize(VW.t(), VW_quant)
    dX.addmm_(dV, VW.t())
    del VW
    dX.addmm_(dV @ VB.t(), VA.t(), alpha=VS)
    return dX.view(batch, seq_len, hd), None, None, d_QA.t(), d_QB.t(
        ), None, None, None, d_KA.t(), d_KB.t(), None, None, None, d_VA.t(
        ), d_VB.t(), None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class LoRA_QKV(torch.autograd.Function):
    """
    ### LoRA weights
    Wq = Wq + Aq @ Bq
    Wk = Wk + Ak @ Bk
    Wv = Wv + Av @ Bv
    Q = X @ Wq = X @ Wq + X @ Aq @ Bq
    K = X @ Wk = X @ Wk + X @ Ak @ Bk
    V = X @ Wv = X @ Wv + X @ Av @ Bv

    ### Backpropagation chain rule
    See our blogpost for more details.

    dC/dWq = X.T @ D(Wq)
    dC/dWk = X.T @ D(Wk)
    dC/dWv = X.T @ D(Wv)
    We then sum them all find dC/dX

    ### Q projection LoRA weights
    dC/dAq =       X.T @ D(Wq) @ B.T
    dC/dBq = A.T @ X.T @ D(Wq)

    ### K projection LoRA weights
    dC/dAk =       X.T @ D(Wk) @ B.T
    dC/dBk = A.T @ X.T @ D(Wk)

    ### V projection LoRA weights
    dC/dAv =       X.T @ D(Wv) @ B.T
    dC/dBv = A.T @ X.T @ D(Wv)
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X: torch.Tensor, QW, QW_quant, QA, QB, QS, KW,
        KW_quant, KA, KB, KS, VW, VW_quant, VA, VB, VS, inplace=True):
        dtype = X.dtype
        Q = matmul_lora(X, QW, QW_quant, QA, QB, QS)
        K = matmul_lora(X, KW, KW_quant, KA, KB, KS)
        V = matmul_lora(X, VW, VW_quant, VA, VB, VS)
        ctx.custom_saved_tensors = (QW, QW_quant, QS, KW, KW_quant, KS, VW,
            VW_quant, VS)
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB)
        ctx.inplace = inplace
        return Q, K, V
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = (ctx.
            custom_saved_tensors)
        X, QA, QB, KA, KB, VA, VB = ctx.saved_tensors
        batch, seq_len, hd = X.shape
        dQ = dQ.view(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])
        dV = dV.view(-1, dV.shape[-1])
        X = X.view(-1, X.shape[-1])
        dtype = X.dtype
        QA, QB, KA, KB, VA, VB = QA.to(dtype), QB.to(dtype), KA.to(dtype
            ), KB.to(dtype), VA.to(dtype), VB.to(dtype)
        QA, QB, KA, KB, VA, VB = QA.t(), QB.t(), KA.t(), KB.t(), VA.t(), VB.t()
        d_QA = torch.empty_like(QA)
        d_QB = torch.empty_like(QB)
        d_KA = torch.empty_like(KA)
        d_KB = torch.empty_like(KB)
        d_VA = torch.empty_like(VA)
        d_VB = torch.empty_like(VB)
        d_QA.addmm_(X.t(), dQ @ QB.t(), alpha=QS, beta=0)
        d_QB.addmm_(QA.t() @ X.t(), dQ, alpha=QS, beta=0)
        d_KA.addmm_(X.t(), dK @ KB.t(), alpha=KS, beta=0)
        d_KB.addmm_(KA.t() @ X.t(), dK, alpha=KS, beta=0)
        d_VA.addmm_(X.t(), dV @ VB.t(), alpha=VS, beta=0)
        d_VB.addmm_(VA.t() @ X.t(), dV, alpha=VS, beta=0)
        QW = fast_dequantize(QW.t(), QW_quant)
        dX = torch.matmul(dQ, QW.t(), out=X if ctx.inplace else None)
        del QW
        dX.addmm_(dQ @ QB.t(), QA.t(), alpha=QS)
        KW = fast_dequantize(KW.t(), KW_quant)
        dX.addmm_(dK, KW.t())
        del KW
        dX.addmm_(dK @ KB.t(), KA.t(), alpha=KS)
        VW = fast_dequantize(VW.t(), VW_quant)
        dX.addmm_(dV, VW.t())
        del VW
        dX.addmm_(dV @ VB.t(), VA.t(), alpha=VS)
        return dX.view(batch, seq_len, hd), None, None, d_QA.t(), d_QB.t(
            ), None, None, None, d_KA.t(), d_KB.t(), None, None, None, d_VA.t(
            ), d_VB.t(), None, None
    pass
