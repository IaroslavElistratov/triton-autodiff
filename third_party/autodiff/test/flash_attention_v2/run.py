import numpy as np
import torch
import triton

from utils import stub


torch.manual_seed(20)
# torch.set_printoptions(sci_mode=False, linewidth=1000)
DEVICE = torch.device("cuda:0")

B=1
NUM_HEADS=1
SEQ_LEN=32
HEAD_DIM=16
dtype=torch.float16

causal=False
sm_scale=0.5

q = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
k = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
v = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)


def torch_fn(q, k, v, causal, sm_scale):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    if causal:
        M = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device=DEVICE))
        p[:, :, M == 0] = float("-inf")

    # p = torch.softmax(p.float(), dim=-1).half()

    p = p.float()
    p = p - p.max(axis=3, keepdim=True)[0]
    p = torch.exp(p)
    p = (p / p.sum(dim=3, keepdim=True))
    p = p.half()

    ref_out = torch.matmul(p, v)
    return ref_out


#### test forward ####
print("\n" * 4, "forward:")

output_torch = torch_fn(q, k, v, causal, sm_scale)
output_triton = stub(q, k, v, causal, sm_scale)
print("output_torch:", output_torch[0, 0, :4, :4])
print("output_triton:", output_triton[0, 0, :4, :4])

max_difference = torch.max(torch.abs(output_torch - output_triton))
print(f'The maximum difference between torch and triton is '
      f'{max_difference}')
# assert max_difference == 0.0

if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


#### test backward ####
print("\n" * 4, "backward:")

upstream = torch.randn_like(q)

q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

my_out = stub(q, k, v, causal, sm_scale)
my_out.backward(upstream)
print("grad q[0, 0, :4, :4]: ", q.grad[0, 0, :4, :4])
print("grad k[0, 0, :4, :4]: ", k.grad[0, 0, :4, :4])
print("grad v[0, 0, :4, :4]: ", v.grad[0, 0, :4, :4])
print()

# compare with pytorch

torch_q = q.clone().detach().requires_grad_(True)
torch_k = k.clone().detach().requires_grad_(True)
torch_v = v.clone().detach().requires_grad_(True)

torch_out = torch_fn(torch_q, torch_k, torch_v, causal, sm_scale)
torch_out.backward(upstream)
print("grad torch_q", torch_q.grad[0, 0, :4, :4])
print("grad torch_k", torch_k.grad[0, 0, :4, :4])
print("grad torch_v", torch_v.grad[0, 0, :4, :4])
print()

# todo: rtol is high because there's a bug how I handle truncation (see issue #10)
rtol = 0.1
if torch.allclose(my_out, torch_out, atol=1e-2, rtol=rtol):
    print("✅ [out] Triton and Torch match")
else:
    print("❌ out Triton and Torch differ")

if torch.allclose(q.grad, torch_q.grad, atol=1e-2, rtol=rtol):
    print("✅ [q grad] Triton and Torch match")
else:
    print("❌ [q grad] Triton and Torch differ")

if torch.allclose(k.grad, torch_k.grad, atol=1e-2, rtol=rtol):
    print("✅ [k grad] Triton and Torch match")
else:
    print("❌ [k grad] Triton and Torch differ")

if torch.allclose(v.grad, torch_v.grad, atol=1e-2, rtol=rtol):
    print("✅ [v grad] Triton and Torch match")
else:
    print("❌ [v grad] Triton and Torch differ")

