import numpy as np

import torch
DEVICE = torch.device("cuda:0")

import triton
from triton.compiler import compile
from triton.backends.compiler import GPUTarget


target = GPUTarget("cuda", arch=89, warp_size=32)
bwd_kernel = compile("out.ttir", target=target)

def bwd(a, b, c, d, upstream):
    assert a.device == DEVICE and b.device == DEVICE and upstream.device == DEVICE
    grid = (1, 1, 1)
    out = torch.empty_like(upstream)
    grad_a = torch.zeros_like(a)
    grad_b = torch.zeros_like(b)
    grad_c = torch.zeros_like(c)
    grad_d = torch.zeros_like(d)
    out_grad = upstream
    _compiled_kernel = bwd_kernel[grid](a, b, c, d, out, grad_a, grad_b, grad_c, grad_d, out_grad)
    return _compiled_kernel, grad_a, grad_b, grad_c, grad_d

np_a = np.random.rand(16, 16)
np_b = np.random.rand(16, 16)
np_c = np.random.rand(16, 16)
np_d = np.random.rand(16, 16)

a = torch.from_numpy(np_a).to(dtype=torch.float32, device='cuda:0')
b = torch.from_numpy(np_b).to(dtype=torch.float32, device='cuda:0')
c = torch.from_numpy(np_c).to(dtype=torch.float32, device='cuda:0')
d = torch.from_numpy(np_d).to(dtype=torch.float32, device='cuda:0')
upstream = torch.ones(a.shape[0], b.shape[1]).to(dtype=torch.float32, device='cuda:0')
_compiled_kernel, grad_a, grad_b, grad_c, grad_d = bwd(a, b, c, d, upstream)

print("grad a[:3, :3]: ", grad_a[:3, :3])
print("grad b[:3, :3]: ", grad_b[:3, :3])
print()

# compare with pytorch

torch_a = torch.from_numpy(np_a).to(device='cuda:0')
torch_a.requires_grad = True
torch_b = torch.from_numpy(np_b).to(device='cuda:0')
torch_b.requires_grad = True
torch_c = torch.from_numpy(np_c).to(device='cuda:0')
torch_c.requires_grad = True
torch_d = torch.from_numpy(np_d).to(device='cuda:0')
torch_d.requires_grad = True

def torch_fn(a, b, c, d):
    mm_1 = torch.matmul(a, b)
    l = mm_1 * 2
    mm_2 = torch.matmul(c, d)
    out = mm_2 + l
    return out

torch_output = torch_fn(torch_a, torch_b, torch_c, torch_d)
torch_output.backward(torch.ones_like(torch_output))

print("torch grad a[:3, :3]: ", torch_a.grad[:3, :3])
print()

print("torch grad b[:3, :3]: ", torch_b.grad[:3, :3])
print()


if torch.allclose(grad_a, torch_a.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(grad_b, torch_b.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


if torch.allclose(grad_c, torch_c.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


if torch.allclose(grad_d, torch_d.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
