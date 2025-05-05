import os
os.environ['TRITON_ALWAYS_COMPILE']='1'

import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def kernel(x_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    offsets = tl.arange(0, BLOCK_SIZE)
    # Load x and y from DRAM, masking out any extra elements in case the input is not a multiple of the block size.
    x = tl.load(x_ptr + offsets)
    output = x + 42
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output)

def stub(x: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    _compiled_kernel = kernel[grid](x, output, BLOCK_SIZE=4)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output, _compiled_kernel

torch.manual_seed(0)
# fits in a single block -- less complex kernel (and thus the IR) bc not computing idx using block_size and pid in this case;
# Also, bc it's exactly the size of the block -- no need to add masks when loading -- further simplifies loading
size = 4
x = torch.rand(size, device=DEVICE)
output_torch = x + 42
output_triton, _compiled_kernel = stub(x)
# print(output_torch)
# print(output_triton)
max_difference = torch.max(torch.abs(output_torch - output_triton))
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')
assert max_difference == 0.0

with open("inp.ttir", "w") as f:
    f.write(_compiled_kernel.asm['ttir'])
