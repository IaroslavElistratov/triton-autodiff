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
        output_1_ptr,
        output_2_ptr,
    ):
    offsets = tl.arange(0, 4)

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)

    _1 = a * b

    # branch 1
    _2 = _1 + 2
    tl.store(output_1_ptr + offsets, _2) # _3

    # branch 2
    _4 = _1 - 4
    # _5 = _4 / 2.0
    tl.store(output_2_ptr + offsets, _4) # _6

def stub(kernel a, b):
    output_1 = torch.empty_like(a)
    output_2 = torch.empty_like(a)
    grid = (1, 1, 1)
    kernel[grid](a, b, output_1, output_2)
    return output_1, output_2

torch.manual_seed(0)

size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)

def torch_fn(torch_a, torch_b):
    # return ((torch_a + 0.5) * torch_a) * torch_a
    _1 = torch_a * torch_b
    _2 = _1 + 2
    _4 = _1 - 4
    return _2, _4

output_triton_1, output_triton_2 = stub(kernel, a, b)
output_torch_1, output_torch_2 = stub(torch_a, torch_b)
# max_difference = torch.max(torch.abs(output_torch - output_triton))
# assert max_difference == 0.0

if torch.allclose(output_triton_1, output_torch_1, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(output_triton_2, output_torch_2, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


#### backward ####

upstream = torch.randn_like(a)
a.requires_grad = True

from triton.backends.api import autodiff

# todo-now: support multiple upstream grads?
my_op, bwd_kernel = autodiff(kernel, stub, idx_upstream=[2, 3])

# todo: rm warmup
stub(bwd_kernel, a, b)
my_out_1, my_out_2 = my_op(a, b)
# todo-now: but how .backward then?
my_out.backward(upstream)

# print("grad a: ", grad_a)
# print()

# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)

torch_output_1, torch_output_2 = torch_fn(torch_a, torch_b)
torch_output.backward(upstream)

# print("torch grad a: ", torch_a.grad)
# print()


if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
