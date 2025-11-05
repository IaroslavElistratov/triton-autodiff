import os
os.environ['TRITON_ALWAYS_COMPILE']='1'

import torch
import triton
import triton.language as tl

from triton.backends.autodiff import autodiff


torch.manual_seed(0)
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

    p = a * b

    # branch 1
    z = p + 2
    tl.store(output_1_ptr + offsets, z)

    # branch 2
    y = p - 4
    tl.store(output_2_ptr + offsets, y)


@autodiff(kernel, idxs_buffers=(2, 3))
def stub(a, b):
    output_1 = torch.empty_like(a)
    output_2 = torch.empty_like(a)
    kernel[(1, )](a, b, output_1, output_2)
    return output_1, output_2


size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)

def torch_fn(torch_a, torch_b):
    p = torch_a * torch_b
    z = p + 2
    y = p - 4
    return z, y

triton_output_1, triton_output_2 = stub(a, b)
torch_output_1, torch_output_2 = torch_fn(a, b)

# max_difference = torch.max(torch.abs(output_torch - output_triton))
# assert max_difference == 0.0

if torch.allclose(triton_output_1, torch_output_1, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(triton_output_2, torch_output_2, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


#### backward ####

upstream = torch.randn_like(a)
a.requires_grad = True
b.requires_grad = True

# toy loss to combine the two outs -- so that when
# ran .backward grads wrt both of the variables are defined
def loss_fn(out_1, out_2):
    return out_1 + out_2


my_out_1, my_out_2 = stub(a, b)
my_loss = loss_fn(my_out_1, my_out_2)
my_loss.backward(upstream)

assert my_out_1.grad_fn is not None
assert my_out_2.grad_fn is not None
print()
print("my_out_1.grad_fn: ", my_out_1.grad_fn)
print("my_out_2.grad_fn: ", my_out_2.grad_fn)


# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)

torch_output_1, torch_output_2 = torch_fn(torch_a, torch_b)
torch_loss = loss_fn(torch_output_1, torch_output_2)
torch_loss.backward(upstream)

print()
print("a.grad", a.grad)
print("b.grad", b.grad)

print()
print("torch_a.grad", torch_a.grad)
print("torch_b.grad", torch_b.grad)

if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
