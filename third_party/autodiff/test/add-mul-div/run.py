import os
os.environ['TRITON_ALWAYS_COMPILE']='1'


import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        output_ptr,
    ):
    offsets = tl.arange(0, 4)

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    c = tl.load(c_ptr + offsets)

    x = a + 0.5
    y = x * b
    z = y / c

    tl.store(output_ptr + offsets, z)

def stub(a, b, c):
    output = torch.empty_like(a)
    grid = (1, 1, 1)
    kernel[grid](a, b, c, output)
    return output

torch.manual_seed(0)
size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)
c = torch.rand(size, device=DEVICE)

def torch_fn(a, b, c):
    x = a + 0.5
    y = x * b
    z = y / c
    return z

output_torch = torch_fn(a, b, c)
output_triton = stub(a, b, c)
max_difference = torch.max(torch.abs(output_torch - output_triton))

# print(output_torch)
# print(output_triton)
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')

# assert max_difference == 0.0
if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")




#### test backward ####

upstream = torch.randn_like(a)
a.requires_grad = True
b.requires_grad = True
c.requires_grad = True

from triton.backends.api import autodiff

my_op, bwd_kernel = autodiff(kernel, stub, idx_upstream=3)

# todo: rm warmup
bwd_kernel[1, 1, 1](a, b, c, torch.ones_like(a))
my_out = my_op(a, b, c)
my_out.backward(upstream)


# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)
torch_c = c.clone().detach().requires_grad_(True)

torch_output = torch_fn(torch_a, torch_b, torch_c)
torch_output.backward(upstream)
# print("torch grad a: ", torch_a.grad)
# print("torch grad b: ", torch_b.grad)
# print("torch grad c: ", torch_c.grad)
# print()


if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(c.grad, torch_c.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")