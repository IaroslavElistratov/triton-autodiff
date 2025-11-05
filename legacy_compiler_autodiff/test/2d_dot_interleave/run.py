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
        c_ptr,
        d_ptr,
        output_ptr,
    ):
    offsets = tl.arange(0, 16)

    offsets_2d = (16 * offsets[:, None]) + offsets[None, :]

    # mm 1
    a = tl.load(a_ptr + offsets_2d)
    b = tl.load(b_ptr + offsets_2d)
    l = tl.dot(a, b)
    # interleave mul
    i = l * 0.751411
    # mm 2
    c = tl.load(c_ptr + offsets_2d)
    d = tl.load(d_ptr + offsets_2d)
    # note: accumulate into previous value
    out = tl.dot(c, d, acc=i)

    tl.store(output_ptr + offsets_2d, out)


@autodiff(kernel, idxs_buffers=4)
def stub(a, b, c, d):
    output = torch.empty(a.shape[0], b.shape[1]).to(device=DEVICE)
    # grid = lambda meta: (triton.cdiv(output.numel(), meta['BLOCK_SIZE']), )
    kernel[(1, )](a, b, c, d, output)
    return output


shape = (16, 16)
a = torch.randn(shape, device=DEVICE)
b = torch.randn(shape, device=DEVICE)
c = torch.randn(shape, device=DEVICE)
d = torch.randn(shape, device=DEVICE)


def torch_fn(torch_a, torch_b, torch_c, torch_d):
    mm_1 = torch.matmul(torch_a, torch_b)
    l = mm_1 * 0.751411
    mm_2 = torch.matmul(torch_c, torch_d)
    return mm_2 + l

torch_out = torch_fn(a, b, c, d)
triton_out = stub(a, b, c, d)
# print("torch_out:", torch_out[:4, :4])
# print("triton_out:", triton_out[:4, :4])

max_difference = torch.max(torch.abs(torch_out - triton_out))
print(f'The maximum difference between torch and triton is '
      f'{max_difference}')
# assert max_difference == 0.0

if torch.allclose(torch_out, triton_out, atol=1e-2, rtol=0.001):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


#### backward ####

upstream = torch.rand_like(torch_out)

a.requires_grad = True
b.requires_grad = True
c.requires_grad = True
d.requires_grad = True

triton_out = stub(a, b, c, d)
triton_out.backward(upstream)

# compare with pytorch

torch_a = a.clone().detach().requires_grad_(True)
torch_b = b.clone().detach().requires_grad_(True)
torch_c = c.clone().detach().requires_grad_(True)
torch_d = d.clone().detach().requires_grad_(True)

torch_out = torch_fn(torch_a, torch_b, torch_c, torch_d)
torch_out.backward(upstream)

if torch.allclose(a.grad, torch_a.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(b.grad, torch_b.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(c.grad, torch_c.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(d.grad, torch_d.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
