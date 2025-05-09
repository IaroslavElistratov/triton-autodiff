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
    ):
    offsets = tl.arange(0, 16)

    offsets_2d = (16 * offsets[:, None]) + offsets[None, :]

    a = tl.load(a_ptr + offsets_2d)
    b = tl.load(b_ptr + offsets_2d)

    l = tl.dot(a, b)
    tl.store(output_ptr + offsets_2d, l)

def stub(kernel, a, b):
    output = torch.empty(a.shape[0], b.shape[1]).to(dtype=torch.float32, device='cuda:0')
    # grid = lambda meta: (triton.cdiv(output.numel(), meta['BLOCK_SIZE']), )
    grid = (1, 1, 1)
    kernel[grid](a, b, output) # BLOCK_SIZE=4
    return output

torch.manual_seed(0)
# fits in a single block -- less complex kernel (and thus the IR) bc not computing idx using block_size and pid in this case;
# Also, bc it's exactly the size of the block -- no need to add masks when loading -- further simplifies loading
size = 16
a = torch.rand((size, size), device=DEVICE).to(dtype=torch.float32, device='cuda:0')
b = torch.rand((size, size), device=DEVICE).to(dtype=torch.float32, device='cuda:0')

def torch_fn(a, b):
    return torch.matmul(a, b)


#### test forward ####

output_torch = torch_fn(a, b)
output_triton = stub(kernel, a, b)
max_difference = torch.max(torch.abs(output_torch - output_triton))

# print("output_torch:", output_torch[:3, :3])
# print("output_triton:", output_triton[:3, :3])
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')

# assert max_difference == 0.0
if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


#### test backward ####

upstream = torch.randn(a.shape[0], b.shape[1]).to(dtype=torch.float32, device='cuda:0')
a.requires_grad = True
b.requires_grad = True

from triton.backends.api import autodiff

my_op, bwd_kernel = autodiff(kernel, stub, idx_upstream=2)

# todo: rm warmup
stub(bwd_kernel, a, b)
my_out = my_op(a, b)
my_out.backward(upstream)
print("grad a[:3, :3]: ", a.grad[:3, :3])
print("grad b[:3, :3]: ", b.grad[:3, :3])
print()


# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)

torch_output = torch_fn(torch_a, torch_b)
torch_output.backward(upstream)

print("torch grad a[:3, :3]: ", torch_a.grad[:3, :3])
print("torch grad b[:3, :3]: ", torch_b.grad[:3, :3])
print()


if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
