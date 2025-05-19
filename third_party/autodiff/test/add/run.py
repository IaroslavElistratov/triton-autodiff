import os
os.environ['TRITON_ALWAYS_COMPILE']='1'

import torch
torch.manual_seed(0)

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def kernel(
        x_ptr,  # *Pointer* to first input vector.
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

def stub(kernel, x):
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
    kernel[grid](x, output, BLOCK_SIZE=4)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# fits in a single block -- less complex kernel (and thus the IR) bc not computing idx using block_size and pid in this case;
# Also, bc it's exactly the size of the block -- no need to add masks when loading -- further simplifies loading
size = 4
a = torch.rand(size, device=DEVICE)

def torch_fn(a):
    return a + 42

output_torch = torch_fn(a)
output_triton = stub(kernel, a)
# print(output_torch)
# print(output_triton)

max_difference = torch.max(torch.abs(output_torch - output_triton))
# print(f'The maximum difference between torch and triton is '
#       f'{max_difference}')
# assert max_difference == 0.0
if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


#### test backward ####


upstream = torch.randn(4, device=DEVICE)
a.requires_grad = True

from triton.backends.autodiff import autodiff

my_op, bwd_kernel = autodiff(kernel, stub, grid=(1,), idx_upstream=1)

# todo: rm warmup
print("\n" * 3, "bwd_kernel warmup")
stub(bwd_kernel, a)

my_out = my_op(a)
my_out.backward(upstream)
print("grad a: ", a.grad)
print()

# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)
torch_output = torch_fn(torch_a)
torch_output.backward(upstream)
print("torch grad a: ", torch_a.grad)
print()


if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
