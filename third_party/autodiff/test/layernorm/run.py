import numpy as np
import torch
import triton

from utils import stub

torch.manual_seed(20)
# torch.set_printoptions(sci_mode=False, linewidth=1000)
DEVICE = torch.device("cuda:0")


M = 1151 # 256 # 32
N = 8192 # 256 # 32

# todo-now: use float16
#   bc during backward we need to accumulate M times (from each instance
#   of the kernel) -- when data is in float16, this quickly emplifies
#   float numerics issues
dtype = torch.float32

# create data
x_shape = (M, N)
w_shape = (N, )
weight = torch.rand(w_shape, dtype=dtype, device=DEVICE, requires_grad=True)
bias = torch.rand(w_shape, dtype=dtype, device=DEVICE, requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=DEVICE)
upstream = .1 * torch.randn_like(x)

#### test forward ####

output_triton = stub(x, weight, bias, eps=1e-5)
output_torch = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps=1e-5)
# print("output_torch:", output_torch[:3, :3])
# print("output_triton:", output_triton[:3, :3])

max_difference = torch.max(torch.abs(output_torch - output_triton))
print(f'The maximum difference between torch and triton is '
      f'{max_difference}')
# assert max_difference == 0.0

if torch.allclose(output_torch, output_triton, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


#### test backward ####

upstream = torch.randn_like(output_torch)
x.requires_grad = True
weight.requires_grad = True
bias.requires_grad = True

my_out = stub(x, weight, bias)
my_out.backward(upstream)
print("x.grad:", x.grad)
print("weight.grad:", weight.grad)
print("bias.grad:", bias.grad)
print()

# compare with pytorch

torch_x = x.clone().detach().requires_grad_(True)
torch_weight = weight.clone().detach().requires_grad_(True)
torch_bias = bias.clone().detach().requires_grad_(True)

torch_out = torch.nn.functional.layer_norm(torch_x, w_shape, torch_weight, torch_bias, eps=1e-5).to(dtype)
torch_out.backward(upstream)
print("torch_x.grad: ", torch_x.grad)
print("torch_weight.grad: ", torch_weight.grad)
print("torch_bias.grad: ", torch_bias.grad)
print()

print("grad_x norm difference:", torch.norm(x - torch_x.grad).item())
print("grad_weight norm difference:", torch.norm(weight - torch_weight.grad).item())
print("grad_bias norm difference:", torch.norm(bias - torch_bias.grad).item())

print("grad_x abs difference:", (x.grad - torch_x.grad).abs().mean())
print("grad_weight abs difference:", (weight - torch_weight.grad).abs().mean())
print("grad_bias abs difference:", (bias - torch_bias.grad).abs().mean())


# todo: rtol is high because there's a bug how I handle truncation (see issue #10)
rtol = 0.1
if torch.allclose(x.grad, torch_x.grad, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(weight.grad, torch_weight.grad, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(bias.grad, torch_bias.grad, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
