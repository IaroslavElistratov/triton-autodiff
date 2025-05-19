import os
os.environ['TRITON_ALWAYS_COMPILE']='1'


import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")



@triton.jit
def kernel(
        a_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
    offsets = tl.arange(0, BLOCK_SIZE)

    a = tl.load(a_ptr + offsets)

    x = a + 0.5
    y = a * x
    # z = a * y

    tl.store(output_ptr + offsets, y)

def stub(kernel, a):
    output = torch.empty_like(a)
    kernel[(1, 1, 1)](a, output, BLOCK_SIZE=4)
    return output

torch.manual_seed(0)

size = 4
a = torch.rand(size, device=DEVICE)
# print("a: ", a)

def torch_fn(a):
    return a * (a + 0.5)

output_torch = torch_fn(a)
output_triton = stub(kernel, a)
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


# backward

upstream = torch.randn_like(a)
a.requires_grad = True


from triton.backends.autodiff import autodiff

my_op, bwd_kernel = autodiff(kernel, stub, grid=(1,), idx_upstream=1)

# todo: rm warmup
stub(bwd_kernel, a)
my_out = my_op(a)
my_out.backward(upstream)

# print("grad a: ", grad_a)
# print()

# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)

torch_output = torch_fn(torch_a)
torch_output.backward(upstream)

# print("torch grad a: ", torch_a.grad)
# print()


if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
