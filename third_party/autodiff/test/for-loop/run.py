import os
os.environ['TRITON_ALWAYS_COMPILE']='1'

import torch
torch.manual_seed(0)

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

    accum = tl.load(output_ptr + offsets)

    # changed to (1, 3) -- bc range(2): i=0, i=1 -- and thus derivative wrt a is just grad-out
    for i in range(3):
      # answer-now:
      # use the loop variable here, bc if use a constant, in the IR, the loop gets flattened
      y = a * i
      accum += y

    tl.store(output_ptr + offsets, accum)


def stub(kernel, a):
    output = torch.zeros_like(a)
    n_elements = output.numel()
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    grid = (1, )
    print("grid: ", grid)
    kernel[grid](a, output, BLOCK_SIZE=4)
    return output

def torch_fn(torch_a):
    accum = torch.zeros_like(torch_a)
    for i in range(3):
      y = torch_a * i
      accum = accum + y
    return accum


size = 4
a = torch.randn(size, device=DEVICE)

output_torch = torch_fn(a)
output_triton = stub(kernel, a)
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

upstream = torch.randn_like(a)
a.requires_grad = True


# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)

torch_output = torch_fn(torch_a)
torch_output.backward(upstream)

print("torch grad a: ", torch_a.grad)
print()


# triton autograd

from triton.backends.api import autodiff

# todo-high: grid with meta is not supported when calling bwd kernel (presumably BLOCK_SIZE got already inlined)
my_op, bwd_kernel = autodiff(kernel, stub, grid=(1, ), idx_upstream=1)

# todo: rm warmup
stub(bwd_kernel, a)
my_out = my_op(a)
# todo-now: calling this doubles torch grad!
my_out.backward(upstream)
print("grad a: ", a.grad)
print()


if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
