# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/open-lm-engine/flash-model-architectures
# Source-Files: fma/functional/swiglu/__init__.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_r3l3p8y1/flash-model-architectures-main/fma/functional/swiglu/__init__.py
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
from math import exp

@torch.compiler.disable
def _get_cpp_function(function_name: str, module_name: str, source_files:
    list[str], build_directory: str) ->Callable:
    module_name = f'{_CPP_MODULE_PREFIX}_{module_name}'
    extra_cflags = ['-O3', '-Wall', '-shared', '-fPIC', '-fdiagnostics-color']
    extra_cuda_cflags = ['-O3', '-lineinfo']
    extra_include_paths = [os.path.dirname(__file__), os.path.dirname(os.
        path.dirname(__file__)) + '/cutlass/include', os.path.dirname(os.
        path.dirname(__file__)) + '/cutlass/tools/util/include']
    module = _ALL_COMPILED_MODULES.get(module_name, None)
    if module is None:
        if torch.distributed.is_initialized():
            os.makedirs(build_directory, exist_ok=True)
            if _GLOBAL_RANK == 0:
                module = load_cpp_extension(module_name, sources=
                    source_files, with_cuda=True, extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_directory, verbose=True)
            torch.distributed.barrier()
            if _GLOBAL_RANK != 0:
                module = load_cpp_extension(module_name, sources=
                    source_files, with_cuda=True, extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_directory, verbose=False)
        else:
            if _WORLD_SIZE > 1:
                build_directory = os.path.join(build_directory, str(uuid4()))
            os.makedirs(build_directory, exist_ok=True)
            module = load_cpp_extension(module_name, sources=source_files,
                with_cuda=True, extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags, extra_include_paths=
                extra_include_paths, build_directory=build_directory,
                verbose=True)
            if _WORLD_SIZE > 1:
                rmtree(build_directory, ignore_errors=True)
        _ALL_COMPILED_MODULES[module_name] = module
    return getattr(module, function_name)


def _run(*args, **kwargs):
    nonlocal cpp_function
    if cpp_function is None:
        cpp_function = _get_cpp_function(function_name=_run.__name__,
            module_name=module_name, source_files=source_files,
            build_directory=build_directory)
    full_args = []
    full_args.extend(args)
    for variable_name in args_spec.args[len(args):]:
        full_args.append(kwargs[variable_name])
    return cpp_function(*full_args)


def cpp_jit(function_name: (str | None)=None, extra_source_files: list[str]
    =[], build_directory: (str | None)=None, depth: int=1) ->Callable:
    """wrapper to compile C++/CUDA source code at runtime.

    Args:
        function_name (str | None, optional): name of the function to expose from the C++ file, the python function
            name should match the funcion name in the C++ file if this is not specified. Defaults to None.
        extra_source_files (list[str], optional): any extra files to use for compilation, by default it scans the
            directory of the python stub file. Defaults to [].
        build_directory (str | None, optional): directory in which to place the build artifacts. Defaults to None.
        depth (int, optional): number of times dirname is called to get the build path. Defaults to 2.

    Returns:
        Callable: returns the wrapped function that can be used to call the C++ functions from python
    """
    cpp_function = None
    args_spec = None
    source_files = []
    source_files.extend(extra_source_files)
    calling_filename = inspect.stack()[1].filename
    calling_directory = os.path.dirname(calling_filename)
    for dirname, _, filenames in os.walk(calling_directory):
        filenames = [os.path.join(dirname, f) for f in filenames]
        filenames = filter(lambda f: os.path.splitext(f)[1] in ['.cu',
            '.cpp'], filenames)
        source_files.extend(filenames)
    if build_directory is None:
        module_name = calling_directory
        for _ in range(depth):
            module_name = os.path.dirname(module_name)
        module_name = os.path.basename(module_name)
        build_directory = os.path.join(os.path.dirname(os.path.dirname(
            __file__)), 'build', module_name)

    def _run(*args, **kwargs):
        nonlocal cpp_function
        if cpp_function is None:
            cpp_function = _get_cpp_function(function_name=_run.__name__,
                module_name=module_name, source_files=source_files,
                build_directory=build_directory)
        full_args = []
        full_args.extend(args)
        for variable_name in args_spec.args[len(args):]:
            full_args.append(kwargs[variable_name])
        return cpp_function(*full_args)

    def _wrapper(function: Callable) ->Callable:
        nonlocal args_spec
        args_spec = inspect.getfullargspec(function)
        _run.__doc__ = function.__doc__
        _run.__name__ = (function.__name__ if function_name is None else
            function_name)
        _run.__signature__ = inspect.signature(function)
        return _run
    return _wrapper


def ceil_divide(x: int, y: int) ->int:
    return (x + y - 1) // y


@triton.jit
def sigmoid(x, output_dtype: tl.constexpr=None):
    if output_dtype is None:
        output_dtype = x.dtype
    x = x.to(tl.float32)
    x = tanh(0.5 * x, output_dtype=tl.float32)
    x = 0.5 * x + 0.5
    x = x.to(output_dtype)
    return x


@triton.jit
def tanh(x, output_dtype: tl.constexpr=None):
    if output_dtype is None:
        output_dtype = x.dtype
    x = x.to(tl.float32)
    x = tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', '=f,f', [x],
        dtype=tl.float32, is_pure=True, pack=1)
    x = x.to(output_dtype)
    return x


def empty_like_contiguous(x: torch.Tensor, dtype: (torch.dtype | None)=None
    ) ->torch.Tensor:
    return torch.empty_like(x, dtype=dtype, memory_format=torch.
        contiguous_format)


def get_num_elements_and_hidden_size(x: torch.Tensor) ->tuple[int]:
    hidden_size = x.size(-1)
    num_elements = x.numel() // hidden_size
    return num_elements, hidden_size


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton.jit
def swiglu_forward_triton_kernel(gate_ptr, gate_stride_b, up_ptr,
    output_ptr, output_stride_b, B, H, BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)
    indices_b = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_b = indices_b < B
    mask_h = indices_h < H
    mask = mask_b[:, None] & mask_h[None, :]
    indices = indices_b[:, None] * gate_stride_b + indices_h[None, :]
    gate = tl.load(gate_ptr + indices, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices, mask=mask)
    output = up * gate * sigmoid(gate)
    indices = indices_b[:, None] * output_stride_b + indices_h[None, :]
    tl.store(output_ptr + indices, output, mask=mask)


@custom_op(f'{LIBRARY_NAME}::{_FORWARD_KERNEL_NAME}', mutates_args={'output'})
@cpp_jit()
def swiglu_forward_cuda(gate: torch.Tensor, up: torch.Tensor, output: torch
    .Tensor, BLOCK_SIZE: int) ->None:
    ...


@custom_op(f'{LIBRARY_NAME}::swiglu_forward_triton', mutates_args={'output'})
def swiglu_forward_triton(gate: torch.Tensor, up: torch.Tensor, output:
    torch.Tensor) ->None:
    B, H = get_num_elements_and_hidden_size(gate)
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_H = 64
    with torch.device(gate.device):
        swiglu_forward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),
            ceil_divide(H, BLOCK_SIZE_H)](gate_ptr=gate, gate_stride_b=gate
            .stride(-2), up_ptr=up, output_ptr=output, output_stride_b=
            output.stride(-2), B=B, H=H, BLOCK_SIZE_B=BLOCK_SIZE_B,
            BLOCK_SIZE_H=BLOCK_SIZE_H)


# Forward method (kernel launch code)
@ensure_contiguous
def __Swiglu_forward(ctx, gate: torch.Tensor, up: torch.Tensor,
    kernel_backend_forward: KernelBackend, kernel_backend_backward:
    KernelBackend) ->torch.Tensor:
    output = empty_like_contiguous(gate)
    if kernel_backend_forward == KernelBackend.cuda:
        swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE=1024)
    elif kernel_backend_forward == KernelBackend.triton:
        swiglu_forward_triton(gate=gate, up=up, output=output)
    else:
        raise ValueError(
            f'unexpected kernel_backend ({kernel_backend_forward})')
    ctx.save_for_backward(gate, up)
    ctx.kernel_backend_backward = kernel_backend_backward
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

@triton.jit
def swiglu_backward_triton_kernel(gate_ptr, gate_stride_b, up_ptr,
    output_grad_ptr, output_grad_stride_b, gate_grad_ptr, up_grad_ptr, B, H,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_H: tl.constexpr):
    BLOCK_ID_B = tl.program_id(axis=0)
    BLOCK_ID_H = tl.program_id(axis=1)
    indices_b = BLOCK_ID_B * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    indices_h = BLOCK_ID_H * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_b = indices_b < B
    mask_h = indices_h < H
    mask = mask_b[:, None] & mask_h[None, :]
    indices_gate = indices_b[:, None] * gate_stride_b + indices_h[None, :]
    indices_output = indices_b[:, None] * output_grad_stride_b + indices_h[
        None, :]
    gate = tl.load(gate_ptr + indices_gate, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + indices_gate, mask=mask)
    output_grad = tl.load(output_grad_ptr + indices_output, mask=mask)
    gate_sigmoid = sigmoid(gate)
    gate_silu = gate * gate_sigmoid
    gate_grad = output_grad * up * (gate_sigmoid + gate_silu * (1 -
        gate_sigmoid))
    up_grad = output_grad * gate_silu
    tl.store(gate_grad_ptr + indices_gate, gate_grad, mask=mask)
    tl.store(up_grad_ptr + indices_gate, up_grad, mask=mask)


@custom_op(f'{LIBRARY_NAME}::{_BACKWARD_KERNEL_NAME}', mutates_args={
    'gate_grad', 'up_grad'})
@cpp_jit()
def swiglu_backward_cuda(gate: torch.Tensor, up: torch.Tensor, output_grad:
    torch.Tensor, gate_grad: torch.Tensor, up_grad: torch.Tensor,
    BLOCK_SIZE: int) ->None:
    ...


@custom_op(f'{LIBRARY_NAME}::swiglu_backward_triton', mutates_args={
    'gate_grad', 'up_grad'})
def swiglu_backward_triton(gate: torch.Tensor, up: torch.Tensor,
    output_grad: torch.Tensor, gate_grad: torch.Tensor, up_grad: torch.Tensor
    ) ->None:
    B, H = get_num_elements_and_hidden_size(gate)
    BLOCK_SIZE_B = 64
    BLOCK_SIZE_H = 64
    with torch.device(gate.device):
        swiglu_backward_triton_kernel[ceil_divide(B, BLOCK_SIZE_B),
            ceil_divide(H, BLOCK_SIZE_H)](gate_ptr=gate, gate_stride_b=gate
            .stride(-2), up_ptr=up, output_grad_ptr=output_grad,
            output_grad_stride_b=output_grad.stride(-2), gate_grad_ptr=
            gate_grad, up_grad_ptr=up_grad, B=B, H=H, BLOCK_SIZE_B=
            BLOCK_SIZE_B, BLOCK_SIZE_H=BLOCK_SIZE_H)


# Backward method (kernel launch code)
@ensure_contiguous
def __Swiglu_backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor |
    None]:
    gate, up = ctx.saved_tensors
    gate_grad = empty_like_contiguous(gate)
    up_grad = empty_like_contiguous(up)
    kernel_backend_backward = ctx.kernel_backend_backward
    if kernel_backend_backward == KernelBackend.cuda:
        swiglu_backward_cuda(gate=gate, up=up, output_grad=output_grad,
            gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=1024)
    elif kernel_backend_backward == KernelBackend.triton:
        swiglu_backward_triton(gate=gate, up=up, output_grad=output_grad,
            gate_grad=gate_grad, up_grad=up_grad)
    else:
        raise ValueError('unexpected kernel_backend')
    return gate_grad, up_grad, None, None


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _Swiglu(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor,
        kernel_backend_forward: KernelBackend, kernel_backend_backward:
        KernelBackend) ->torch.Tensor:
        output = empty_like_contiguous(gate)
        if kernel_backend_forward == KernelBackend.cuda:
            swiglu_forward_cuda(gate=gate, up=up, output=output, BLOCK_SIZE
                =1024)
        elif kernel_backend_forward == KernelBackend.triton:
            swiglu_forward_triton(gate=gate, up=up, output=output)
        else:
            raise ValueError(
                f'unexpected kernel_backend ({kernel_backend_forward})')
        ctx.save_for_backward(gate, up)
        ctx.kernel_backend_backward = kernel_backend_backward
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor | None]:
        gate, up = ctx.saved_tensors
        gate_grad = empty_like_contiguous(gate)
        up_grad = empty_like_contiguous(up)
        kernel_backend_backward = ctx.kernel_backend_backward
        if kernel_backend_backward == KernelBackend.cuda:
            swiglu_backward_cuda(gate=gate, up=up, output_grad=output_grad,
                gate_grad=gate_grad, up_grad=up_grad, BLOCK_SIZE=1024)
        elif kernel_backend_backward == KernelBackend.triton:
            swiglu_backward_triton(gate=gate, up=up, output_grad=
                output_grad, gate_grad=gate_grad, up_grad=up_grad)
        else:
            raise ValueError('unexpected kernel_backend')
        return gate_grad, up_grad, None, None
