from functools import partial

import torch
torch.manual_seed(0)
DEVICE = torch.device("cuda:0")

import triton
import triton.language as tl


from api import autodiff




def stub(a, b):
    output = torch.empty_like(a)
    assert a.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    grid = lambda meta: (triton.cdiv(n_elements, 4), )
    print("[fwd stub] a, b, out", a, b, output)
    kernel[grid](a, b, output) # , BLOCK_SIZE=4)
    return output


# @grad(stub, idx_upstream=2)
@triton.jit
def kernel(
        a_ptr,
        b_ptr,
        output_ptr,
        # BLOCK_SIZE: tl.constexpr,
    ):
    offsets = tl.arange(0, 4) # BLOCK_SIZE)

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)

    x = a + 0.5
    y = x * b

    tl.store(output_ptr + offsets, y, mask=offsets<4)



size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)
a.requires_grad = True
b.requires_grad = True
upstream = torch.ones_like(a)



my_op, bwd_kernel = autodiff(kernel, stub)
my_out = my_op(a, b)
print("my_out: ", my_out)



# let bwd kernel generate autodiffed version -- once
# running once to let my hook trigger and create bwd CompiledKernel, otherwise errs bc the bwd_stub tries to pass more args to the bwd_kernel at which time the compile hook hasn't triggered yet, thus hasn't modifed signature yet (to expect 6 args) -- thus passing more args failed
# you need to call the backward function once with original inputs just to let it create compileKernel and trigger post hook to replace taht compiledKernel with the differned compiledKernel

# bc my hook runs after the first comaplation, first call to the stub return original result (does not call my DifferenciatedCompiledKernel)
bwd_kernel[1, 1, 1](a, b, torch.zeros_like(a)) # , BLOCK_SIZE=4


my_out.backward(upstream)
print(a.grad)
print(b.grad)


# device = driver.active.get_current_device()
# kernel_cache, target, backend, _binder = bwd_kernel.device_caches[device]
# print("kernel_cache", kernel_cache)





# compare with pytorch

torch_a = torch.clone(a).detach().requires_grad_().to(device='cuda:0')
torch_b = torch.clone(b).detach().requires_grad_().to(device='cuda:0')

torch_out = (torch_a + 0.5) * torch_b
torch_out.backward(torch.ones_like(torch_out))

if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
