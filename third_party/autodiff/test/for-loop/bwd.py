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

def bwd(a, upstream, BLOCK_SIZE=4):
    out = torch.empty_like(a)
    grad_a = torch.zeros_like(a)
    grad_out = upstream
    _compiled_kernel = bwd_kernel[(1, 1, 1)](a, out, grad_a, grad_out)
    return _compiled_kernel, grad_a


np_a = np.array([0.3990, 0.5167, 0.0249, 0.9401])

a = torch.from_numpy(np_a).to(dtype=torch.float32, device='cuda:0')

upstream = torch.ones_like(a)
_compiled_kernel, grad_a = bwd(a, upstream)
print("grad a: ", grad_a)
print()

# compare with pytorch
torch_a = torch.from_numpy(np_a).to(device='cuda:0')
torch_a.requires_grad = True

def torch_fn(torch_a):

    accum = torch.zeros_like(torch_a)

    for i in range(3):
      y = torch_a * i
      accum = accum + y
    return accum

torch_output = torch_fn(torch_a)
torch_output.backward(torch.ones_like(torch_output))

print("torch grad a: ", torch_a.grad)
print()


if torch.allclose(grad_a, torch_a.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
