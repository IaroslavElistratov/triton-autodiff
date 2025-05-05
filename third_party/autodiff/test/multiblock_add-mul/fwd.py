# 4) multi-block: Add + Mull
import os
os.environ['TRITON_ALWAYS_COMPILE']='1'


import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def add_kernel(a_ptr,
               b_ptr,
               output_ptr,  # *Pointer* to output vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    a = tl.load(a_ptr + offsets) # , mask=offsets<4
    b = tl.load(b_ptr + offsets) # , mask=offsets<4

    x = a + 0.5
    y = x * b

    # tl.debug_barrier()

    tl.store(output_ptr + offsets, y) # , mask=offsets<4
    # tl.atomic_add(output_ptr + offsets, y) # , mask=offsets<4
def g1(a, b):
    # We need to preallocate the output.
    output = torch.empty_like(a)
    assert a.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _compiled_kernel = add_kernel[grid](a, b, output, BLOCK_SIZE=4)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output, _compiled_kernel
torch.manual_seed(0)

size = 8
a = torch.rand(size, device=DEVICE)
print("a: ", a)
b = torch.rand(size, device=DEVICE)
print("b: ", b)
output_torch = (a + 0.5) * b
output_triton, _compiled_kernel = g1(a, b)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

with open("inp.ttir", "w") as f:
    f.write(_compiled_kernel.asm['ttir'])

# a.dtype
# print("IR", _compiled_kernel.asm['ttir']) # triton IR
# print("TTGIR", _compiled_kernel.asm['ttgir']) # triton gpu IR
# print("LLIR", _compiled_kernel.asm["llir"]) # llvm IR
# print("PTX", _compiled_kernel.asm['ptx'])