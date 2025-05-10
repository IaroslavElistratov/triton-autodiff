import numpy as np
import torch
import triton

from utils import kernel, stub

torch.manual_seed(20)
DEVICE = torch.device("cuda:0")

# answer-now: some input types to the fn are float16 and some are float32

M = 1151 # 256
N = 8192 # 256
dtype = torch.float16

# create data
x_shape = (M, N)
w_shape = (x_shape[-1], )
weight = torch.rand(w_shape, dtype=dtype, device=DEVICE, requires_grad=True)
bias = torch.rand(w_shape, dtype=dtype, device=DEVICE, requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=DEVICE)
upstream = .1 * torch.randn_like(x)
x.requires_grad_(True)
quantiles = [0.5, 0.2, 0.8]


#### test forward ####

output_triton = stub(kernel, x, weight, bias)
output_torch = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps=1e-5)

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

upstream = torch.randn_like(output_torch)
x.requires_grad = True
weight.requires_grad = True
bias.requires_grad = True


from triton.backends.api import autodiff

my_op, bwd_kernel = autodiff(kernel, stub, grid=(M,), non_stub_args_idxs=[4,5], idx_upstream=1)

# todo: rm warmup
stub(bwd_kernel, x, weight, bias)
my_out = my_op(x, weight, bias)
my_out.backward(upstream)

print("x grad: ", x.grad)
print("weight grad: ", weight.grad)
print("bias grad: ", bias.grad)
print()

# compare with pytorch


torch_x = x.clone().detach().requires_grad_(True)
torch_weight = weight.clone().detach().requires_grad_(True)
torch_bias = bias.clone().detach().requires_grad_(True)


torch_out = torch.nn.functional.layer_norm(torch_x, w_shape, torch_weight, torch_bias, eps=1e-5)
torch_out.backward(upstream)
print("torch_x.grad: ", torch_x.grad)
print("torch_weight.grad: ", torch_weight.grad)
print("torch_bias.grad: ", torch_bias.grad)

# print("\nPyTorch reference gradients:")
# print("dx norm difference:", torch.norm(dx_small - x_torch.grad).item())
# print("dweight norm difference:", torch.norm(dweight_small - weight_torch.grad).item())
# print("dbias norm difference:", torch.norm(dbias_small - bias_torch.grad).item())



if torch.allclose(x.grad, torch_x.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(weight.grad, torch_weight.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(bias.grad, torch_bias.grad, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")




def dynamic_func(X, Y, W, B, Mean, Rstd, extra_0, extra_1, extra_2, extra_3, extra_4, extra_5, **options):
    params = {
        'X': X,
        'Y': Y,
        'W': W,
        'B': B,
        'Mean': Mean,
        'Rstd': Rstd,
        'extra_0': extra_0,
        'extra_1': extra_1,
        'extra_2': extra_2,
        'extra_3': extra_3,
        'extra_4': extra_4,
        'extra_5': extra_5
    }
    specialization = [
        specialize_impl(X, specialize_extra, False, True, True),
        specialize_impl(Y, specialize_extra, False, True, True),
        specialize_impl(W, specialize_extra, False, True, True),
        specialize_impl(B, specialize_extra, False, True, True),
        specialize_impl(Mean, specialize_extra, False, True, True),
        specialize_impl(Rstd, specialize_extra, False, True, True),
        specialize_impl(extra_0, specialize_extra, False, True, True),
        specialize_impl(extra_1, specialize_extra, False, True, True),
        specialize_impl(extra_2, specialize_extra, False, True, True),
        specialize_impl(extra_3, specialize_extra, False, True, True),
        specialize_impl(extra_4, specialize_extra, False, True, True),
        specialize_impl(extra_5, specialize_extra, False, True, True)
    ]
    return params, specialization, options