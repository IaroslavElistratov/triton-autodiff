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
    # b = tl.load(b_ptr + offsets)

    accum = tl.load(output_ptr + offsets)
    for i in range(3):
      # x = a + b
      # y = x * i

      # answer-now:
      # use the loop variable here, bc if use a constant, in the IR, the loop gets flattened
      y = a * i
      accum += y

    tl.store(output_ptr + offsets, accum)

def stub(a, b):
    output = torch.empty_like(a)
    assert a.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _compiled_kernel = kernel[grid](a, output, BLOCK_SIZE=4)
    return output, _compiled_kernel

torch.manual_seed(0)
size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)
# print("a: ", a)
# print("b: ", b)

output_torch = (a + 0.5) * b
output_triton, _compiled_kernel = stub(a, b)
max_difference = torch.max(torch.abs(output_torch - output_triton))
assert max_difference == 0.0

# print(output_torch)
# print(output_triton)
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')

with open("inp.ttir", "w") as f:
    f.write(_compiled_kernel.asm['ttir'])
