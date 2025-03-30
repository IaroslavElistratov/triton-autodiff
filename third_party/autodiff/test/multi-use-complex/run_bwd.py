import numpy as np
import torch
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget


target = GPUTarget("cuda", arch=89, warp_size=32)
bwd_kernel = compile("out.ttir", target=target)


def bwd(a, upstream): # , BLOCK_SIZE=4
    return bwd_kernel[(1, 1, 1)](a, upstream)

np_a = np.array([0.3990, 0.5167, 0.0249, 0.9401])

a = torch.from_numpy(np_a).to(dtype=torch.float32, device='cuda:0')
upstream = torch.ones_like(a)
_compiled_kernel = bwd(a, upstream)
print("grad a: ", a)
print()

# compare with pytorch

torch_a = torch.from_numpy(np_a).to(device='cuda:0')
torch_a.requires_grad = True
torch_output = ((torch_a + 0.5) * torch_a) * torch_a
torch_output.backward(torch.ones_like(torch_output))

print("torch grad a: ", torch_a.grad)
print()


if torch.allclose(a, torch_a.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
