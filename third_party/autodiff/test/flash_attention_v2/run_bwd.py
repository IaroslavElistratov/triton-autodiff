import numpy as np
import torch
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget


# Create a GPU target
target = GPUTarget("cuda", arch=89, warp_size=32)

# Compile the IR
bwd_kernel = compile("out.ttir", target=target)




torch.manual_seed(20)

def test_op(Z=1, H=2, N_CTX=16, HEAD_DIM=16, causal=True, dtype=torch.float16, sm_scale=0.5):

    ######## COPY FROM FWD ########

    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # answer-now: note they use rand like upstream (not ones)
    dout = torch.randn_like(q)


    ######## COPY FROM FWD ########

    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    o = torch.empty_like(q)
    stage = 3 if causal else 1
    print("stage: ", stage)
    extra_kern_args = {}


    grid = (1, q.shape[0] * q.shape[1], 1)

    BLOCK_M = 16
    BLOCK_N = 16

    # assert q.shape[2] / BLOCK_M == 1
    assert q.shape[2] == BLOCK_M
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    ################################



    grad_q = torch.zeros_like(q)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    grad_m = torch.zeros_like(M)
    # note: ones!
    grad_o = torch.ones_like(o)

    # enqueue kernel
    compiled_kernel = bwd_kernel[grid](  #

        # # fwd values:
        # q,      # FLOAT PTR!
        # k,      # FLOAT PTR!
        # v,      # FLOAT PTR!
        # 0.5,    #   sm_scale
        # M,      # FLOAT PTR!
        # o,      # FLOAT PTR!
        # 256,    #   q.stride(0)
        # 256,    #   q.stride(1)
        # 16,     #   q.stride(2)
        # 1,      #   q.stride(3)
        # 256,    #   k.stride(0)
        # 256,    #   k.stride(1)
        # 16,     #   k.stride(2)
        # 1,      #   k.stride(3)
        # 256,    #   v.stride(0)
        # 256,    #   v.stride(1)
        # 16,     #   v.stride(2)
        # 1,      #   v.stride(3)
        # 256,    #   o.stride(0)
        # 256,    #   o.stride(1)
        # 16,     #   o.stride(2)
        # 1,      #   o.stride(3)
        # 1,      #   q.shape[0]
        # 1,      #   q.shape[1]


        # fwd values:
        q,      # FLOAT PTR!
        k,      # FLOAT PTR!
        v,      # FLOAT PTR!
        # sm_scale,     # constexpr
        M,      # FLOAT PTR!
        o,      # FLOAT PTR!
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],

        # HEAD_DIM=HEAD_DIM_K,  # constexpr
        # STAGE=stage,          # constexpr
        # BLOCK_M=BLOCK_M,      # constexpr
        # BLOCK_N=BLOCK_N       # constexpr

        # grads:
        grad_q,
        grad_k,
        grad_v,
        grad_m,
        grad_o,

    )
    return compiled_kernel, grad_x_arg, grad_weight, grad_bias, grad_mean, grad_rstd

# !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<f32>, !tt.ptr<f16>,
# i32, i32, i32, i32,
# i32, i32, i32, i32,
# i32, i32, i32, i32,
# i32,
# !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<f16>, !tt.ptr<f32>, !tt.ptr<f16>





# NOTE: this works but unrolls for 128 times -- so try smaller shapes
# tri_out, ref_out = test_op(1, 1, 128, 64, True)




test_op()

# # todo-now:

# # reference implementation
# M = torch.tril(torch.ones((N_CTX, N_CTX), device=DEVICE))
# p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
# if causal:
#     p[:, :, M == 0] = float("-inf")
# p = torch.softmax(p.float(), dim=-1).half()
# # p = torch.exp(p)
# ref_out = torch.matmul(p, v)
# ref_out.backward(dout)
# ref_dv, v.grad = v.grad.clone(), None
# ref_dk, k.grad = k.grad.clone(), None
# ref_dq, q.grad = q.grad.clone(), None



# # triton implementation
# tri_out = attention(q, k, v, causal, sm_scale).half()

# assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
# rtol = 0.0
# # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
# # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
# # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)
# return tri_out, ref_out



















"""


M = 1151
N = 8192
dtype = torch.float16
device = DEVICE

# create data
x_shape = (M, N)
w_shape = (x_shape[-1], )
weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
dy = .1 * torch.randn_like(x)
# x.requires_grad_(True)
# quantiles = [0.5, 0.2, 0.8]


# # Run our implementation
with torch.no_grad():
    compiled_kernel, grad_x_arg, grad_weight, grad_bias, grad_mean, grad_rstd = bwd(x, weight, bias)
    print("grad_x_arg: ", grad_x_arg)
    print("grad_weight: ", grad_weight)
    print("grad_bias: ", grad_bias)

print("\n" * 3)
# # Forward pass with PyTorch's LayerNorm
x.requires_grad = True
weight.requires_grad = True
bias.requires_grad = True
torch_ln = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps=1e-5)
torch_ln.backward(torch.ones_like(torch_ln))
print("x.grad: ", x.grad)
print("weight.grad: ", weight.grad)
print("bias.grad: ", bias.grad)
"""