
# %%
## Simplified

# - this is non causal only (so no need for the 2nd call to _attn_fwd_inner)

# // one block + sm_scale (float arg) is constexpr

# // !!! non causal



# CHANGE LOG
#   - made kernels args that are used as loop bounds -- to be tl.constexpr
#     - start_m
#     - almost all args to _attn_fwd
#
#   - reduced the range of inputs for benchmark (for speed of iter)

#   - ==> yeah problem is that variable needed for loop boundaries "start_m = tl.program_id(0)" -- 



# Made sm_scale a constepr (it was the only non ptr arg) -- so need special handing to compute grad wrt it

# %%
STAGE = 1
if (STAGE & 1):
  print("1: ", True)
if (STAGE & 2):
  print("2: ", True)

# %%
import pytest
import torch

import triton
import triton.language as tl

print(triton.__path__)

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = torch.device("cuda:0")


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True

# %% [markdown]
# # Official tutorial (before blackwell support)
# https://github.com/triton-lang/triton/blob/105cb56487cd8a433b8fbfe9cc63c1f1c04a4b2a/python/tutorials/06-fused-attention.py


# %% [markdown]
# ## Simplified
# 
# - this is non causal only (so no need for the 2nd call to _attn_fwd_inner)

# %%


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr):
    # range of values handled by this stage
    # causal = False
    # answer-now: hardcoded N_CTX=16, here bc even though I'm passing compiled time constant -- it still shomehow makes the bounds dynamic
    #   _attn_fwd_inner is inlined into the outer kernel, but its N_CTX is forwarded as an SSA argument so the entry point keeps a uniform parameter that the host sets at launch (just like grid size).
    #   Until the very last “constant-prop + canonicalize” sweep runs, the loop boundary therefore looks dynamic
    # lo, hi = 0, N_CTX
    lo, hi = 0, 16

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)

        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# answer-now: added do not specilize, other wise trtion adds them as constants as oppose to arguments -- and this is different from my causal=True case -- thus to avoid chaning what params I pass to the bwd kenrel in my bwd.py every time I filp "causal" flag -- thus here I just specialized them
@triton.jit(do_not_specialize=["stride_qz", "stride_qh", "stride_qm", "stride_qk",  "stride_kn", "stride_kk",  "stride_vk", "stride_vn",  "stride_om", "stride_on", "Z", "H", "N_CTX"])
def _attn_fwd(Q, K, V, sm_scale: tl.constexpr, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  # 
              stride_kn, stride_kk,  # 
              stride_vk, stride_vn,  # 
              stride_om, stride_on,  #
              Z, H, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):


    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = 0 # tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)

    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  #
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                    4 - STAGE, offs_m, offs_n, N_CTX
                                    )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))




_COMPILED_KERNEL = None


def stub(q, k, v, causal, sm_scale):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    print("stage: ", stage)

    # answer-now: lauch only one block in PID-1 -- to make loops unrollable
    # assert q.shape[2] / args["BLOCK_M"] == 1
    grid = lambda args: (1, q.shape[0] * q.shape[1], 1)

    # answer-now: this works but unrolls the loop 128 times! 5500 lines in the IR
    # BLOCK_M = 128
    # BLOCK_N = 64

    BLOCK_M = 16
    BLOCK_N = 16

    # assert q.shape[2] / BLOCK_M == 1
    assert q.shape[2] == BLOCK_M
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    global _COMPILED_KERNEL
    _COMPILED_KERNEL = _attn_fwd[grid](
        q, k, v, sm_scale, M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        # k.stride(0), k.stride(1), 
        k.stride(2), k.stride(3),  #
        # v.stride(0), v.stride(1), 
        v.stride(2), v.stride(3),  #
        # o.stride(0), o.stride(1),
        o.stride(2), o.stride(3),  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        STAGE=stage,  #
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    # comment: these are the args passed to the fwd kerenl -- unlike in all my preivous kernel I actually want to try to pass all the ints valeus to bwd kenrel as well (instead of hardcoding them)
    # (Q, K, V, sm_scale: tl.constexpr, M, Out,  #
    # stride_qz, stride_qh, stride_qm, stride_qk,  #
    # stride_kz, stride_kh, stride_kn, stride_kk,  #
    # stride_vz, stride_vh, stride_vk, stride_vn,  #
    # stride_oz, stride_oh, stride_om, stride_on,  #
    # Z, H, N_CTX,  #
    # HEAD_DIM: tl.constexpr,  #
    # BLOCK_M: tl.constexpr,  #
    # BLOCK_N: tl.constexpr,  #
    # STAGE: tl.constexpr  #
    # )

    # print("q")
    # print("k")
    # print("v")
    # print("sm_scale", sm_scale)
    # print("M")
    # print("o")
    # print("q.stride(0)", q.stride(0))
    # print("q.stride(1)", q.stride(1))
    # print("q.stride(2)", q.stride(2))
    # print("q.stride(3)", q.stride(3))
    # print("k.stride(0)", k.stride(0))
    # print("k.stride(1)", k.stride(1))
    # print("k.stride(2)", k.stride(2))
    # print("k.stride(3)", k.stride(3))
    # print("v.stride(0)", v.stride(0))
    # print("v.stride(1)", v.stride(1))
    # print("v.stride(2)", v.stride(2))
    # print("v.stride(3)", v.stride(3))
    # print("o.stride(0)", o.stride(0))
    # print("o.stride(1)", o.stride(1))
    # print("o.stride(2)", o.stride(2))
    # print("o.stride(3)", o.stride(3))
    # print("q.shape[0]", q.shape[0])
    # print("q.shape[1]", q.shape[1])

    return o # , M


kernel = _attn_fwd
attention = stub

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 128, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)

    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    # comment: commented this out bc I'm using smaller shapes now but the bwd has this assert (requries a larger shape)
    #    PRE_BLOCK = 128
    #    assert N_CTX % PRE_BLOCK == 0

    # tri_out.backward(dout)
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)
    return tri_out, ref_out

# NOTE: this works but unrolls for 128 times -- so try smaller shapes
# tri_out, ref_out = test_op(1, 1, 128, 64, True)




# NOTE: causal False
tri_out, ref_out = test_op(1, 1, 16, 16, False)


# # %%
# tri_out[0, 0, :4, :4]

# # %%
# ref_out[0, 0, :4, :4]

# %%
# print("IR", _COMPILED_KERNEL.asm['ttir']) # triton IR