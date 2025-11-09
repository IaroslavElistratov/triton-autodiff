# SPDX-License-Identifier: MIT
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/BobMcDear/attorch
# Source-Files: attorch/glu_layer.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_leort3lx/attorch-main/attorch/glu_layer.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

# Common helper imports
from triton import cdiv
from torch.cuda.amp import custom_bwd
from torch.cuda.amp import custom_fwd
from math import log

@triton.jit
def apply_act_func(input, drop_p, seed, offset, param, act_func: tl.
    constexpr, dropout: tl.constexpr):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', 'hardshrink',
            and 'softshrink'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Input transformed by the desired activation function,
        potentially with fused dropout.
    """
    if act_func == 'sigmoid':
        input = input.to(tl.float32)
        output = sigmoid(input)
    if act_func == 'logsigmoid':
        input = input.to(tl.float32)
        output = logsigmoid(input)
    elif act_func == 'tanh':
        input = input.to(tl.float32)
        output = tanh(input)
    elif act_func == 'relu':
        output = relu(input)
    elif act_func == 'gelu':
        input = input.to(tl.float32)
        output = gelu(input)
    elif act_func == 'geluapprox':
        input = input.to(tl.float32)
        output = geluapprox(input)
    elif act_func == 'silu':
        input = input.to(tl.float32)
        output = silu(input)
    elif act_func == 'relu6':
        output = relu6(input)
    elif act_func == 'hardsigmoid':
        output = hardsigmoid(input)
    elif act_func == 'hardtanh':
        output = hardtanh(input)
    elif act_func == 'hardswish':
        output = hardswish(input)
    elif act_func == 'selu':
        input = input.to(tl.float32)
        output = selu(input)
    elif act_func == 'mish':
        input = input.to(tl.float32)
        output = mish(input)
    elif act_func == 'softplus':
        input = input.to(tl.float32)
        output = softplus(input)
    elif act_func == 'softsign':
        output = softsign(input)
    elif act_func == 'tanhshrink':
        input = input.to(tl.float32)
        output = tanhshrink(input)
    elif act_func == 'leaky_relu':
        output = leaky_relu(input, param)
    elif act_func == 'elu':
        input = input.to(tl.float32)
        output = elu(input, param)
    elif act_func == 'celu':
        input = input.to(tl.float32)
        output = celu(input, param)
    elif act_func == 'hardshrink':
        output = hardshrink(input, param)
    elif act_func == 'softshrink':
        output = softshrink(input, param)
    if dropout:
        output = apply_dropout(output, drop_p, seed, offset)
    return output


@triton.jit
def celu(input, alpha):
    """
    Applies CELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        alpha: Alpha value.

    Returns:
        Input transformed by CELU.
    """
    return relu(input) + tl.minimum(0, alpha * (tl.exp(input / alpha) - 1))


@triton.jit
def elu(input, alpha):
    """
    Applies ELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        alpha: Alpha value.

    Returns:
        Input transformed by ELU.
    """
    return tl.where(input <= 0, alpha * (tl.exp(input) - 1), input)


@triton.jit
def gelu(input):
    """
    Applies GELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input


@triton.jit
def geluapprox(input):
    """
    Applies the tanh approximation of GELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by the tanh approximation of GELU.
    """
    cdf = 0.5 * (1 + tanh(0.7978845608 * input * (1 + 0.044715 * input *
        input)))
    return cdf * input


@triton.jit
def hardshrink(input, lambd):
    """
    Applies hard shrink to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: Lambda value.

    Returns:
        Input transformed by hard shrink.
    """
    return tl.where(tl.abs(input) < lambd, 0, input)


@triton.jit
def hardsigmoid(input):
    """
    Applies hard sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard sigmoid.
    """
    return tl.maximum(0, tl.minimum(1, input / 6 + 0.5))


@triton.jit
def hardswish(input):
    """
    Applies hard Swish to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard Swish.
    """
    return input * relu6(input + 3) / 6


@triton.jit
def hardtanh(input):
    """
    Applies hard tanh to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard tanh.
    """
    return tl.maximum(-1, tl.minimum(1, input))


@triton.jit
def leaky_relu(input, negative_slope):
    """
    Applies leaky ReLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        negative_slope: Slope of the negative component.

    Returns:
        Input transformed by leaky ReLU.
    """
    return relu(input) + negative_slope * tl.minimum(0, input)


@triton.jit
def logsigmoid(input):
    """
    Applies the log of sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by the log of sigmoid.
    """
    return tl.log(sigmoid(input))


@triton.jit
def mish(input):
    """
    Applies Mish to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by Mish.
    """
    return input * tanh(tl.log(1 + tl.exp(input)))


@triton.jit
def relu(input):
    """
    Applies ReLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by ReLU.
    """
    return tl.maximum(0, input)


@triton.jit
def relu6(input):
    """
    Applies ReLU6 to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by ReLU6.
    """
    return tl.minimum(relu(input), 6)


@triton.jit
def selu(input):
    """
    Applies SELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by SELU.
    """
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    return scale * (tl.maximum(0, input) + tl.minimum(0, alpha * (tl.exp(
        input) - 1)))


@triton.jit
def sigmoid(input):
    """
    Applies sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by sigmoid.
    """
    return 1 / (1 + tl.exp(-input))


@triton.jit
def silu(input):
    """
    Applies SiLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by SiLU.
    """
    return input * sigmoid(input)


@triton.jit
def softplus(input):
    """
    Applies softplus to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by softplus.
    """
    return tl.log(1 + tl.exp(input))


@triton.jit
def softshrink(input, lambd):
    """
    Applies softshrink to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: Lambda value.

    Returns:
        Input transformed by softshrink.
    """
    return tl.where(input > lambd, input - lambd, tl.where(input < -lambd, 
        input + lambd, 0))


@triton.jit
def softsign(input):
    """
    Applies softsign to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by softsign.
    """
    return input / (1 + tl.abs(input))


@triton.jit
def tanh(input):
    """
    Applies tanh to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by tanh.
    """
    return 2 * sigmoid(2 * input) - 1


@triton.jit
def tanhshrink(input):
    """
    Applies tanh shrink to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by tanh shrink.
    """
    return input - tanh(input)


@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    """
    Randomly zeroes elements in the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        offset: Offset to generate the mask for.

    Returns:
        Input with elements randomly zeroed out.
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def glu_forward_kernel(input1_pointer, input2_pointer, output_pointer, size,
    param, act_func: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Applies the gated linear unit with an arbitrary activation function
    to the input.

    Args:
        input1_pointer: Pointer to the first half of the input to gate.
            The first half must be contiguous and contain size elements.
        input2_pointer: Pointer to the second half of the input to gate.
            The second half must be contiguous and contain size elements.
        output_pointer: Pointer to a container the result is written to.
            The container must be contiguous and contain size elements.
        size: Number of elements in each half of the input.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', and 'hardshrink'.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input1 = tl.load(input1_pointer + offset, mask=mask)
    input2 = tl.load(input2_pointer + offset, mask=mask)
    output = input1 * apply_act_func(input2, None, None, None, param,
        act_func, False)
    tl.store(output_pointer + offset, output, mask=mask)


# Forward method (kernel launch code)
@custom_fwd(device_type='cuda')
def _GLUAutoGrad_forward(ctx: Context, input: Tensor, dim: int, act_func: str
    ) ->Tensor:
    """
        Applies the gated linear unit with an arbitrary activation function
        to the input.

        Args:
            ctx: Context for variable storage.
            input: Input to gate.
                Can have arbitrary shape but dimension dim must be even.
            dim: Dimension over which to gate.
            act_func: Name of activation function to apply.
                Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
                'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
                'softplus', 'softsign', 'tanhshrink', 'leaky_relu_PARAM',
                'elu_PARAM', 'celu_PARAM', 'hardshrink_PARAM', and 'softshrink_PARAM'
                where PARAM stands for the parameter in the case of parameterized
                activation functions (e.g., 'leaky_relu_0.01' for leaky ReLU with a
                negative slope of 0.01).

        Returns:
            Input transformed by the gated linear unit
            with an arbitrary activation function.
        """
    param = None
    if '_' in act_func:
        comps = act_func.split('_')
        act_func = '_'.join(comps[:-1])
        param = float(comps[-1])
    input1, input2 = input.chunk(2, dim=dim)
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    requires_grad = input.requires_grad
    size = input1.numel()
    output = torch.empty_like(input1)
    ctx.param = param
    ctx.act_func = act_func
    ctx.dim = dim
    ctx.size = size
    if requires_grad:
        ctx.save_for_backward(input1, input2)
    grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
    glu_forward_kernel[grid](input1, input2, output, size, param, act_func)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def apply_act_func_grad(output_grad, input, drop_p, seed, offset, param,
    act_func: tl.constexpr, dropout: tl.constexpr):
    """
    Calculates the gradient of an activation function.

    Args:
        output_grad: Output gradients. The output gradients must be
            loaded and cannot be a pointer.
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', 'hardshrink',
            and 'softshrink'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Gradient of the desired activation function.
    """
    if act_func == 'sigmoid':
        input = input.to(tl.float32)
        output = sigmoid_grad(input)
    if act_func == 'logsigmoid':
        input = input.to(tl.float32)
        output = logsigmoid_grad(input)
    elif act_func == 'tanh':
        input = input.to(tl.float32)
        output = tanh_grad(input)
    elif act_func == 'relu':
        output = relu_grad(input)
    elif act_func == 'gelu':
        input = input.to(tl.float32)
        output = gelu_grad(input)
    elif act_func == 'geluapprox':
        input = input.to(tl.float32)
        output = geluapprox_grad(input)
    elif act_func == 'silu':
        input = input.to(tl.float32)
        output = silu_grad(input)
    elif act_func == 'relu6':
        output = relu6_grad(input)
    elif act_func == 'hardsigmoid':
        output = hardsigmoid_grad(input)
    elif act_func == 'hardtanh':
        output = hardtanh_grad(input)
    elif act_func == 'hardswish':
        output = hardswish_grad(input)
    elif act_func == 'selu':
        input = input.to(tl.float32)
        output = selu_grad(input)
    elif act_func == 'mish':
        input = input.to(tl.float32)
        output = mish_grad(input)
    elif act_func == 'softplus':
        input = input.to(tl.float32)
        output = softplus_grad(input)
    elif act_func == 'softsign':
        output = softsign_grad(input)
    elif act_func == 'tanhshrink':
        input = input.to(tl.float32)
        output = tanhshrink_grad(input)
    elif act_func == 'leaky_relu':
        output = leaky_relu_grad(input, param)
    elif act_func == 'elu':
        input = input.to(tl.float32)
        output = elu_grad(input, param)
    elif act_func == 'celu':
        input = input.to(tl.float32)
        output = celu_grad(input, param)
    elif act_func == 'hardshrink':
        output = hardshrink_grad(input, param)
    elif act_func == 'softshrink':
        output = softshrink_grad(input, param)
    if dropout:
        output_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)
    return output_grad * output


@triton.jit
def celu_grad(input, alpha):
    """
    Calculates the gradient of CELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        alpha: Alpha value.

    Returns:
        Gradient of CELU.
    """
    return tl.where(input <= 0, tl.exp(input / alpha), 1)


@triton.jit
def elu_grad(input, alpha):
    """
    Calculates the gradient of ELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        alpha: Alpha value.

    Returns:
        Gradient of ELU.
    """
    return tl.where(input <= 0, alpha * tl.exp(input), 1)


@triton.jit
def gelu_grad(input):
    """
    Calculates the gradient of GELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    cdf_grad = 0.39894228 * tl.exp(-0.5 * input * input)
    return cdf_grad * input + cdf


@triton.jit
def geluapprox_grad(input):
    """
    Calculates the gradient of the tanh approximation of GELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of the tanh approximation of GELU.
    """
    tanh_res = tanh(0.7978845608 * input * (1 + 0.044715 * input * input))
    sech_sq = 1 - tanh_res * tanh_res
    return 0.5 * (1 + tanh_res + input * sech_sq * 0.7978845608 * (1 + 3 * 
        0.044715 * input * input))


@triton.jit
def hardshrink_grad(input, lambd):
    """
    Calculates the gradient of hard shrink.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: Lambda value.

    Returns:
        Gradient of hard shrink.
    """
    return tl.where(tl.abs(input) < lambd, 0, 1)


@triton.jit
def hardsigmoid_grad(input):
    """
    Calculates the gradient of hard sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard sigmoid.
    """
    return tl.where((-3 < input) & (input < 3), 1 / 6, 0)


@triton.jit
def hardswish_grad(input):
    """
    Calculates the gradient of hard Swish.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard Swish.
    """
    return (relu6(input + 3) + input * relu6_grad(input + 3)) / 6


@triton.jit
def hardtanh_grad(input):
    """
    Calculates the gradient of hard tanh.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard tanh.
    """
    return tl.where((-1 < input) & (input < 1), 1, 0)


@triton.jit
def leaky_relu_grad(input, negative_slope):
    """
    Calculates the gradient of leaky ReLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        negative_slope: Slope of the negative component.

    Returns:
        Gradient of leaky ReLU.
    """
    return tl.where(input <= 0, negative_slope, 1)


@triton.jit
def logsigmoid_grad(input):
    """
    Calculates the gradient of the log of sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of the log of sigmoid.
    """
    return 1 / (1 + tl.exp(input))


@triton.jit
def mish_grad(input):
    """
    Calculates the gradient of Mish.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of Mish.
    """
    exp = tl.exp(input)
    delta = exp * (exp + 2) + 2
    return exp * (exp * (4 * input + 6 + exp * (exp + 4)) + 4 * (input + 1)
        ) / (delta * delta)


@triton.jit
def relu6_grad(input):
    """
    Calculates the gradient of ReLU6.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of ReLU6.
    """
    return tl.where((0 < input) & (input < 6), 1, 0)


@triton.jit
def relu_grad(input):
    """
    Calculates the gradient of ReLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of ReLU.
    """
    return tl.where(input <= 0, 0, 1)


@triton.jit
def selu_grad(input):
    """
    Calculates the gradient of SELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of SELU.
    """
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    return scale * tl.where(input <= 0, alpha * tl.exp(input), 1)


@triton.jit
def sigmoid_grad(input):
    """
    Calculates the gradient of sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of sigmoid.
    """
    output_sigmoid = sigmoid(input)
    return output_sigmoid * (1 - output_sigmoid)


@triton.jit
def silu_grad(input):
    """
    Calculates the gradient of SiLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of SiLU.
    """
    output_sigmoid = sigmoid(input)
    return output_sigmoid * (input * (1 - output_sigmoid) + 1)


@triton.jit
def softplus_grad(input):
    """
    Calculates the gradient of softplus.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of softplus.
    """
    return sigmoid(input)


@triton.jit
def softshrink_grad(input, lambd):
    """
    Calculates the gradient of softshrink.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        lambd: Lambda value.

    Returns:
        Gradient of softshrink.
    """
    return tl.where(tl.abs(input) < lambd, 0, 1)


@triton.jit
def softsign_grad(input):
    """
    Calculates the gradient of softsign.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of softsign.
    """
    denom = 1 + tl.abs(input)
    return 1 / (denom * denom)


@triton.jit
def tanh_grad(input):
    """
    Calculates the gradient of tanh.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of tanh.
    """
    output_tanh = tanh(input)
    return 1 - output_tanh * output_tanh


@triton.jit
def tanhshrink_grad(input):
    """
    Calculates the gradient of tanh shrink.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of tanh shrink.
    """
    return 1 - tanh_grad(input)


@triton.jit
def apply_dropout_grad(output_grad, drop_p, seed, offset):
    """
    Calculates the input gradient of dropout.

    Args:
        output_grad: Output gradients. The output gradients must be
            loaded and cannot be a pointer.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        offset: Offset to generate the mask for.

    Returns:
        Gradient of dropout.
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, output_grad / (1 - drop_p))


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def glu_backward_kernel(output_grad_pointer, input1_pointer, input2_pointer,
    input1_grad_pointer, input2_grad_pointer, size, param, act_func: tl.
    constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Calculates the input gradient of the gated linear unit.

    Args:
        output_grad_pointer: Pointer to the unit's output gradients.
            The output gradients must be contiguous and contain size elements.
        input1_pointer: Pointer to the first half of the input that was gated.
            The first half must be contiguous and contain size elements.
        input2_pointer: Pointer to the second half of the input that was gated.
            The second half must be contiguous and contain size elements.
        input1_grad_pointer: Pointer to a container the first half's gradients are written to.
            The container must be contiguous and contain size elements.
        input2_grad_pointer: Pointer to a container the second half's gradients are written to.
            The container must be contiguous and contain size elements.
        size: Number of elements in each half of the input.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
            'softplus', 'softsign', 'tanhshrink', 'leaky_relu', 'elu', 'celu', 'hardshrink',
            and 'softshrink'.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input1 = tl.load(input1_pointer + offset, mask=mask)
    input2 = tl.load(input2_pointer + offset, mask=mask)
    input1_grad = output_grad * apply_act_func(input2, None, None, None,
        param, act_func, False)
    input2_grad = output_grad * input1 * apply_act_func_grad(1, input2,
        None, None, None, param, act_func, False)
    tl.store(input1_grad_pointer + offset, input1_grad, mask=mask)
    tl.store(input2_grad_pointer + offset, input2_grad, mask=mask)


# Backward method (kernel launch code)
@custom_bwd(device_type='cuda')
def _GLUAutoGrad_backward(ctx: Context, output_grad: Tensor) ->Tuple[
    Optional[Tensor], ...]:
    """
        Calculates the input gradient of the gated linear unit.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the gated linear unit.
        """
    input1, input2 = ctx.saved_tensors
    input1_grad = torch.empty_like(input1)
    input2_grad = torch.empty_like(input2)
    grid = lambda META: (cdiv(ctx.size, META['BLOCK_SIZE']),)
    glu_backward_kernel[grid](output_grad, input1, input2, input1_grad,
        input2_grad, ctx.size, ctx.param, ctx.act_func)
    return torch.concat([input1_grad, input2_grad], dim=ctx.dim), None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class GLUAutoGrad(torch.autograd.Function):
    """
    Autodiff for gated linear unit.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx: Context, input: Tensor, dim: int, act_func: str) ->Tensor:
        """
        Applies the gated linear unit with an arbitrary activation function
        to the input.

        Args:
            ctx: Context for variable storage.
            input: Input to gate.
                Can have arbitrary shape but dimension dim must be even.
            dim: Dimension over which to gate.
            act_func: Name of activation function to apply.
                Options are 'sigmoid', 'logsigmoid', 'tanh', 'relu', 'gelu', 'geluapprox', 'silu',
                'relu6', 'hardsigmoid', 'hardtanh', 'hardswish', 'selu', 'mish',
                'softplus', 'softsign', 'tanhshrink', 'leaky_relu_PARAM',
                'elu_PARAM', 'celu_PARAM', 'hardshrink_PARAM', and 'softshrink_PARAM'
                where PARAM stands for the parameter in the case of parameterized
                activation functions (e.g., 'leaky_relu_0.01' for leaky ReLU with a
                negative slope of 0.01).

        Returns:
            Input transformed by the gated linear unit
            with an arbitrary activation function.
        """
        param = None
        if '_' in act_func:
            comps = act_func.split('_')
            act_func = '_'.join(comps[:-1])
            param = float(comps[-1])
        input1, input2 = input.chunk(2, dim=dim)
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        requires_grad = input.requires_grad
        size = input1.numel()
        output = torch.empty_like(input1)
        ctx.param = param
        ctx.act_func = act_func
        ctx.dim = dim
        ctx.size = size
        if requires_grad:
            ctx.save_for_backward(input1, input2)
        grid = lambda META: (cdiv(size, META['BLOCK_SIZE']),)
        glu_forward_kernel[grid](input1, input2, output, size, param, act_func)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx: Context, output_grad: Tensor) ->Tuple[Optional[Tensor
        ], ...]:
        """
        Calculates the input gradient of the gated linear unit.

        Args:
            ctx: Context containing stored variables.
            output_grad: Output gradients.
                Must be the same shape as the output.

        Returns:
            Input gradient of the gated linear unit.
        """
        input1, input2 = ctx.saved_tensors
        input1_grad = torch.empty_like(input1)
        input2_grad = torch.empty_like(input2)
        grid = lambda META: (cdiv(ctx.size, META['BLOCK_SIZE']),)
        glu_backward_kernel[grid](output_grad, input1, input2, input1_grad,
            input2_grad, ctx.size, ctx.param, ctx.act_func)
        return torch.concat([input1_grad, input2_grad], dim=ctx.dim
            ), None, None
