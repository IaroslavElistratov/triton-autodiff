import torch
import triton
import triton.language as tl

@triton.jit
def _mk_block_ptr(base, m_idx, n_idx, stride_m, stride_n, BM: tl.constexpr, BN: tl.constexpr):
    # Device helper: rebuild a pointer grid without broadcasting the pointer itself.
    # BM/BN are constexpr tile sizes. Casts strides to int64 to satisfy addptr rules.
    stride_m = tl.cast(stride_m, tl.int64)
    stride_n = tl.cast(stride_n, tl.int64)
    grid = base + tl.zeros((BM, BN), dtype=tl.int64)
    return grid + m_idx[:, None] * stride_m + n_idx[None, :] * stride_n

# Legend:
#    local grads for <y>                         (fine-grained: backward ops emitted when differentiating a single forward value y)
#    ~~~~~~~~~~ grad branch for <X> ~~~~~~~~~~   (coarse: ops contributing to grad of kernel input X)

@triton.jit
def backward__attn_fwd(Q, K, V, M, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kn, stride_kk, stride_vk, stride_vn, stride_om, stride_on, Z, H, grad_Q, grad_K, grad_V, grad_M, grad_Out):
    off_hz = tl.program_id(axis=1)
    off_z = off_hz // H  # assumes non-negative
    qvk_offset = tl.cast(off_z, tl.int64)
    qvk_offset_1 = tl.cast(stride_qz, tl.int64)
    qvk_offset_2 = qvk_offset * qvk_offset_1
    off_h = off_hz % H  # assumes non-negative
    qvk_offset_3 = tl.cast(off_h, tl.int64)
    qvk_offset_4 = tl.cast(stride_qh, tl.int64)
    qvk_offset_5 = qvk_offset_3 * qvk_offset_4
    qvk_offset_6 = qvk_offset_2 + qvk_offset_5
    O_block_ptr = Out + qvk_offset_6
    start_m = tl.program_id(axis=0)
    unnamed = 16
    Q_block_ptr = start_m * unnamed
    Q_block_ptr_1 = tl.cast(Q_block_ptr, tl.int64)
    offs_m = tl.arange(0, 16)
    q = tl.cast(offs_m, tl.int64)
    q_1 = Q_block_ptr_1 + q
    O_block_ptr_7 = _mk_block_ptr(O_block_ptr, q_1, q, stride_om, stride_on, 16, 16)
    Q_block_ptr_2 = Q + qvk_offset_6
    Q_block_ptr_5 = _mk_block_ptr(Q_block_ptr_2, q_1, q, stride_qm, stride_qk, 16, 16)
    q_8 = tl.load(Q_block_ptr_5)
    K_block_ptr = K + qvk_offset_6
    K_block_ptr_3 = _mk_block_ptr(K_block_ptr, q, q, stride_kk, stride_kn, 16, 16)
    k_5 = tl.load(K_block_ptr_3)
    qk = tl.dot(q_8, k_5)
    unnamed_2 = 0.7213475108146667
    qk_1 = qk * unnamed_2
    _elementwise_max = tl.max(qk, axis=1)
    unnamed_3 = 0.7213475108146667
    m_ij = _elementwise_max * unnamed_3
    unnamed_4 = float('-inf')
    m_ij_1 = tl.maximum(m_ij, unnamed_4)
    qk_2 = m_ij_1[:, None]
    qk_3 = qk_1 - qk_2
    p = tl.exp2(qk_3)
    p_1 = tl.cast(p, tl.float16)
    V_block_ptr = V + qvk_offset_6
    V_block_ptr_3 = _mk_block_ptr(V_block_ptr, q, q, stride_vk, stride_vn, 16, 16)
    v_4 = tl.load(V_block_ptr_3)
    alpha = unnamed_4 - m_ij_1
    alpha_1 = tl.exp2(alpha)
    acc = alpha_1[:, None]
    unnamed_5 = 0.0
    acc_1 = acc * unnamed_5
    acc_2 = tl.dot(p_1, v_4) + acc_1
    _sum_combine = tl.sum(p, axis=1)
    l_i = alpha_1 + _sum_combine
    acc_3 = l_i[:, None]
    acc_4 = acc_2 / acc_3
    element_ty = tl.cast(acc_4, tl.float16)
    tl.store(O_block_ptr_7, element_ty)
    m_ptrs = off_hz * unnamed
    m_ptrs_1 = M + tl.cast(m_ptrs, tl.int64)
    m_ptrs_2 = m_ptrs_1 + tl.zeros((16,), dtype=tl.int64)
    offs_m_1 = Q_block_ptr + offs_m
    m_ptrs_3 = m_ptrs_2 + tl.cast(offs_m_1, tl.int64)
    m_i = tl.log2(l_i)
    m_i_1 = m_ij_1 + m_i
    tl.store(m_ptrs_3, m_i_1)

    # ~~~~~~~~~~ grad branch for {grad_Q,grad_K} ~~~~~~~~~~
    m_ptrs_4 = grad_M + tl.cast(m_ptrs, tl.int64)
    m_ptrs_5 = m_ptrs_4 + tl.zeros((16,), dtype=tl.int64)
    m_ptrs_6 = m_ptrs_5 + tl.cast(offs_m_1, tl.int64)
    bwd_m_i = tl.load(m_ptrs_6)

    # ~~~~~~~~~~ grad branch for {grad_Q,grad_K,grad_V} ~~~~~~~~~~
    O_block_ptr_8 = grad_Out + qvk_offset_6
    O_block_ptr_10 = _mk_block_ptr(O_block_ptr_8, q_1, q, stride_om, stride_on, 16, 16)
    bwd_O_block_ptr = tl.load(O_block_ptr_10)
    # grads wrt element_ty
    bwd_element_ty = tl.cast(bwd_O_block_ptr, tl.float32)
    # grads wrt acc
    bwd_acc_3 = (1.0 / acc_3) * bwd_element_ty
    bwd_acc_4 = tl.cast(bwd_acc_3, tl.float16)

    # ~~~~~~~~~~ grad branch for {grad_Q,grad_K} ~~~~~~~~~~
    bwd_acc_8 = tl.dot(bwd_acc_4, tl.trans(v_4))
    # grads wrt p
    bwd_p = tl.cast(bwd_acc_8, tl.float32)

    # ~~~~~~~~~~ grad branch for grad_V ~~~~~~~~~~
    bwd_acc_12 = tl.dot(tl.trans(p_1), bwd_acc_4)

    # ~~~~~~~~~~ grad branch for {grad_Q,grad_K} ~~~~~~~~~~
    # grads wrt acc
    bwd_acc_14 = tl.sum(bwd_acc_3, axis=1)[:, None]
    bwd_acc_16 = unnamed_5 * bwd_acc_14
    bwd_acc_23 = (-1.0 * (acc_2 / (acc_3 * acc_3))) * bwd_element_ty
    bwd_acc_25 = tl.sum(bwd_acc_23, axis=1)[:, None]
    bwd_acc_26 = tl.reshape(bwd_acc_25, (16,))
    bwd_acc_27 = bwd_acc_26 + ((1.0 / (l_i * 0.6931470036506653)) * bwd_m_i)
    bwd_acc_28 = bwd_acc_27 + tl.reshape(bwd_acc_16, (16,))
    # grads wrt alpha
    bwd_alpha_3 = (0.6931470036506653 * alpha_1) * bwd_acc_28
    bwd_m_i_8 = bwd_m_i + (bwd_alpha_3 * -1.0)
    # grads wrt _sum_combine
    bwd_p_1 = bwd_p + bwd_acc_27[:, None]
    # grads wrt p
    bwd_p_5 = (0.6931470036506653 * p) * bwd_p_1
    # grads wrt qk
    bwd_qk_2 = bwd_p_5 * -1.0
    bwd_qk_4 = tl.sum(bwd_qk_2, axis=1)[:, None]
    bwd_m_i_9 = bwd_m_i_8 + tl.reshape(bwd_qk_4, (16,))
    # grads wrt m_ij
    bwd_m_ij_3 = 1.0
    bwd_m_ij_5 = 0.0
    bwd_m_ij_8 = tl.where((m_ij >= unnamed_4), bwd_m_ij_3, bwd_m_ij_5) * bwd_m_i_9
    bwd_m_ij_10 = unnamed_3 * bwd_m_ij_8
    # grads wrt qk
    bwd_qk_7 = unnamed_2 * bwd_p_5
    bwd_qk_8 = bwd_qk_7 + tl.where(qk == _elementwise_max[:, None], 1.0, 0.0) * bwd_m_ij_10[:, None]
    bwd_qk_9 = tl.cast(bwd_qk_8, tl.float16)

    # ~~~~~~~~~~ grad branch for grad_Q ~~~~~~~~~~
    bwd_qk_13 = tl.dot(bwd_qk_9, tl.trans(k_5))

    # ~~~~~~~~~~ grad branch for grad_K ~~~~~~~~~~
    bwd_qk_17 = tl.dot(tl.trans(q_8), bwd_qk_9)

    # ~~~~~~~~~~ grad branch for grad_Q ~~~~~~~~~~
    Q_block_ptr_6 = grad_Q + qvk_offset_6
    Q_block_ptr_7 = _mk_block_ptr(Q_block_ptr_6, q_1, q, stride_qm, stride_qk, 16, 16)
    # grads wrt q
    tl.atomic_add(Q_block_ptr_7, tl.cast(bwd_qk_13, tl.float16))

    # ~~~~~~~~~~ grad branch for grad_K ~~~~~~~~~~
    K_block_ptr_4 = grad_K + qvk_offset_6
    K_block_ptr_5 = _mk_block_ptr(K_block_ptr_4, q, q, stride_kk, stride_kn, 16, 16)
    # grads wrt k
    tl.atomic_add(K_block_ptr_5, tl.cast(bwd_qk_17, tl.float16))

    # ~~~~~~~~~~ grad branch for grad_V ~~~~~~~~~~
    V_block_ptr_4 = grad_V + qvk_offset_6
    V_block_ptr_5 = _mk_block_ptr(V_block_ptr_4, q, q, stride_vk, stride_vn, 16, 16)
    # grads wrt v
    tl.atomic_add(V_block_ptr_5, tl.cast(bwd_acc_12, tl.float16))


# commenting manually
# N_CTX = 128
# N_CTX = (stride_qh // stride_qm)
# fwd_m_ptrs = (fwd_off_hz * N_CTX)


BLOCK_M = 16




def stub(q, k, v, causal, sm_scale, upstream):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    print("stage: ", stage)

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    print("grid: ", grid)

    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    grad_o = upstream


    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    grad_m = torch.zeros_like(M)
    backward__attn_fwd[grid](
        q, k, v,
        M, o,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(2), k.stride(3),  #
        v.stride(2), v.stride(3),  #
        o.stride(2), o.stride(3),  #

        q.shape[0],  # Z
        q.shape[1],  # H
        # N_CTX=q.shape[2],  #
        # HEAD_DIM=HEAD_DIM_K,  #
        # STAGE=stage,  #
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,

        grad_q,  # arg17
        grad_k, # arg18
        grad_v, # arg19
        # question-now: not sure: m at this postion?
        grad_m,  # arg20
        grad_o, # arg21
    )

    return grad_q, grad_k, grad_v


#   my actual forward / backward test is in run.py
# Z = 1
# H = 2
# N_CTX = 16
# HEAD_DIM = 64

# answer-now:MUST be same as fwd shapes! bc the kernel is speciazed
Z = 256
H = 64
N_CTX = 16
HEAD_DIM = 16

causal=False
dtype=torch.float16
DEVICE = torch.device("cuda:0")

torch.manual_seed(20)
q = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
k = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
v = torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
sm_scale = 0.5
upstream = torch.randn_like(q)

torch_q = q.clone().detach().requires_grad_(True)
torch_k = k.clone().detach().requires_grad_(True)
torch_v = v.clone().detach().requires_grad_(True)
torch_upstream = upstream.clone().detach()


# reference implementation
def torch_fn(q, k, v):
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    return ref_out



grad_q, grad_k, grad_v = stub(q, k, v, causal, sm_scale, upstream)

print("grad_q", grad_q[0, 0, :5, :5])
print("grad_k", grad_k[0, 0, :5, :5])
print("grad_v", grad_v[0, 0, :5, :5])


_torch_output = torch_fn(torch_q, torch_k, torch_v)
_torch_output.backward(torch_upstream)

print("torch_q.grad", torch_q.grad[0, 0, :5, :5])
print("torch_k.grad", torch_k.grad[0, 0, :5, :5])
print("torch_v.grad", torch_v.grad[0, 0, :5, :5])



if torch.allclose(grad_q, torch_q.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_k, torch_k.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_v, torch_v.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")