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
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
    offsets = tl.arange(0, BLOCK_SIZE)

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)

    x = a + 0.5
    y = x * b

    tl.store(output_ptr + offsets, y, mask=offsets<4)

def stub(kernel, a, b):
    output = torch.empty_like(a)
    assert a.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    kernel[grid](a, b, output, BLOCK_SIZE=4)
    return output

torch.manual_seed(0)
size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)

def torch_fn(a, b):
    return (a + 0.5) * b

output_torch = torch_fn(a, b)
output_triton = stub(kernel, a, b)
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

from triton.backends.autodiff import autodiff

my_op, bwd_kernel = autodiff(kernel, stub, grid=(1,), idx_upstream=2)

# todo: rm warmup
stub(bwd_kernel, a, b)


my_out = my_op(a, b)
my_out.backward(upstream)
print("grad a: ", a.grad)
print("grad b: ", b.grad)
print()


# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)

torch_output = torch_fn(torch_a, torch_b)
torch_output.backward(upstream)

print("torch grad a: ", torch_a.grad)
print("torch grad b: ", torch_b.grad)
print()


if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
