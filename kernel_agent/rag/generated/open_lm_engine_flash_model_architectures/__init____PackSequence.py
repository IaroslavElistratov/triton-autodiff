# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/open-lm-engine/flash-model-architectures
# Source-Files: fma/functional/sequence_packing/__init__.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_r3l3p8y1/flash-model-architectures-main/fma/functional/sequence_packing/__init__.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# SHARED HELPERS (Used by both forward and backward)
# ============================================================

# These helpers are called by both forward() and backward() methods

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


@custom_op(f'{LIBRARY_NAME}::pack_unpack_sequence_cuda', mutates_args={
    'output'})
@cpp_jit()
def pack_unpack_sequence_cuda(x: torch.Tensor, output: torch.Tensor,
    cu_seqlens: torch.Tensor, padding_side: str, pack: bool, BLOCK_SIZE: int
    ) ->None:
    ...


@triton.jit
def _copy_array(source_ptr, destination_ptr, b, s, t, S, N, pack, BLOCK_SIZE):
    unpacked_offset = (b * S + s) * N
    packed_offset = t * N
    for i in range(tl.cdiv(N, BLOCK_SIZE)):
        indices = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = indices < N
        if pack:
            source = tl.load(source_ptr + unpacked_offset + indices, mask=mask)
            tl.store(destination_ptr + packed_offset + indices, source,
                mask=mask)
        else:
            source = tl.load(source_ptr + packed_offset + indices, mask=mask)
            tl.store(destination_ptr + unpacked_offset + indices, source,
                mask=mask)


@custom_op(f'{LIBRARY_NAME}::pack_unpack_sequence_triton', mutates_args={
    'output'})
def pack_unpack_sequence_triton(x: torch.Tensor, output: torch.Tensor,
    cu_seqlens: torch.Tensor, padding_side: str, pack: bool) ->None:
    if pack:
        B, S = x.size()[:2]
        N = x.numel() // (B * S)
    else:
        B, S = output.size()[:2]
        N = output.numel() // (B * S)
    BLOCK_SIZE = 4096
    NUM_WARPS = 32
    with torch.device(x.device):
        pack_unpack_sequence_triton_kernel[S, B](x_ptr=x, output_ptr=output,
            cu_seqlens_ptr=cu_seqlens, S=S, N=N, PADDING_SIDE=padding_side,
            PACK=pack, BLOCK_SIZE=BLOCK_SIZE, num_warps=NUM_WARPS)


@triton.jit
def pack_unpack_sequence_triton_kernel(x_ptr, output_ptr, cu_seqlens_ptr, S,
    N, PADDING_SIDE: tl.constexpr, PACK: tl.constexpr, BLOCK_SIZE: tl.constexpr
    ):
    s = tl.program_id(axis=0)
    b = tl.program_id(axis=1)
    cu_seqlens_ptrs = cu_seqlens_ptr + b
    start = tl.load(cu_seqlens_ptrs)
    end = tl.load(cu_seqlens_ptrs + 1)
    seqlens = end - start
    if PADDING_SIDE == 'left':
        pad_tokens = S - seqlens
        if s >= pad_tokens:
            _copy_array(x_ptr, output_ptr, b, s, start + s - pad_tokens, S,
                N, PACK, BLOCK_SIZE)
    elif s < seqlens:
        _copy_array(x_ptr, output_ptr, b, s, start + s, S, N, PACK, BLOCK_SIZE)


# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

def _pack_sequence(x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape:
    tuple[int], padding_side: str, kernel_backend: KernelBackend
    ) ->torch.Tensor:
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    if kernel_backend == KernelBackend.cuda:
        pack_unpack_sequence_cuda(x=x, output=output, cu_seqlens=cu_seqlens,
            padding_side=padding_side, pack=True, BLOCK_SIZE=1024)
    elif kernel_backend == KernelBackend.triton:
        pack_unpack_sequence_triton(x=x, output=output, cu_seqlens=
            cu_seqlens, padding_side=padding_side, pack=True)
    else:
        raise ValueError(f'unexpected kernel_backend ({kernel_backend})')
    return output


# Forward method (kernel launch code)
@ensure_contiguous
def __PackSequence_forward(ctx, x: torch.Tensor, cu_seqlens: torch.Tensor,
    output_shape: tuple[int], padding_side: str, kernel_backend_forward:
    KernelBackend, kernel_backend_backward: KernelBackend) ->torch.Tensor:
    ctx.save_for_backward(cu_seqlens)
    ctx.padding_side = padding_side
    ctx.x_shape = x.size()
    ctx.kernel_backend_backward = kernel_backend_backward
    output = _pack_sequence(x=x, cu_seqlens=cu_seqlens, output_shape=
        output_shape, padding_side=padding_side, kernel_backend=
        kernel_backend_forward)
    return output


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

def _unpack_sequence(x: torch.Tensor, cu_seqlens: torch.Tensor,
    output_shape: tuple[int], padding_side: str, kernel_backend: KernelBackend
    ) ->torch.Tensor:
    output = torch.zeros(*output_shape, device=x.device, dtype=x.dtype)
    if kernel_backend == KernelBackend.cuda:
        pack_unpack_sequence_cuda(x=x, output=output, cu_seqlens=cu_seqlens,
            padding_side=padding_side, pack=False, BLOCK_SIZE=1024)
    elif kernel_backend == KernelBackend.triton:
        pack_unpack_sequence_triton(x=x, output=output, cu_seqlens=
            cu_seqlens, padding_side=padding_side, pack=False)
    else:
        raise ValueError(f'unexpected kernel_backend ({kernel_backend})')
    return output


# Backward method (kernel launch code)
@ensure_contiguous
def __PackSequence_backward(ctx, output_grad: torch.Tensor) ->tuple[torch.
    Tensor | None]:
    x_grad = _unpack_sequence(x=output_grad, cu_seqlens=ctx.saved_tensors[0
        ], output_shape=ctx.x_shape, padding_side=ctx.padding_side,
        kernel_backend=ctx.kernel_backend_backward)
    return x_grad, *([None] * 5)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _PackSequence(torch.autograd.Function):

    @staticmethod
    @ensure_contiguous
    def forward(ctx, x: torch.Tensor, cu_seqlens: torch.Tensor,
        output_shape: tuple[int], padding_side: str, kernel_backend_forward:
        KernelBackend, kernel_backend_backward: KernelBackend) ->torch.Tensor:
        ctx.save_for_backward(cu_seqlens)
        ctx.padding_side = padding_side
        ctx.x_shape = x.size()
        ctx.kernel_backend_backward = kernel_backend_backward
        output = _pack_sequence(x=x, cu_seqlens=cu_seqlens, output_shape=
            output_shape, padding_side=padding_side, kernel_backend=
            kernel_backend_forward)
        return output

    @staticmethod
    @ensure_contiguous
    def backward(ctx, output_grad: torch.Tensor) ->tuple[torch.Tensor | None]:
        x_grad = _unpack_sequence(x=output_grad, cu_seqlens=ctx.
            saved_tensors[0], output_shape=ctx.x_shape, padding_side=ctx.
            padding_side, kernel_backend=ctx.kernel_backend_backward)
        return x_grad, *([None] * 5)
