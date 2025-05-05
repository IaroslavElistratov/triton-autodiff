# 5) masked and multi-block: Add + Mull
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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 10 here is numel in the input;
    # because I'm launching ceil(10, 4) -- 3 instances of this fn
    # because each fn processes 4 elements, 3 instances will process 3*4=12
    # elements. But the input only has 10 elements.
    # So for the 3rd instance need to mask of last two elements
    a = tl.load(a_ptr + offsets, mask=offsets<10)
    b = tl.load(b_ptr + offsets, mask=offsets<10)

    x = a + 0.5
    y = x * b

    tl.store(output_ptr + offsets, y, mask=offsets<10)

def stub(a, b):
    output = torch.empty_like(a)
    assert a.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _compiled_kernel = kernel[grid](a, b, output, BLOCK_SIZE=4)
    return output, _compiled_kernel


torch.manual_seed(0)

size = 10
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
