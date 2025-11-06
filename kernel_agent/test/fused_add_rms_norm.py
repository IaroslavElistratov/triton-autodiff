# credit: linkedin_Liger_Kernel/Liger-Kernel-main/benchmark/scripts/benchmark_fused_add_rms_norm.py

import functools
import importlib
import operator

from typing import Callable

import math
import operator

import torch
import triton
import triton.language as tl

# from liger_kernel.ops.utils import calculate_settings

def is_hip() -> bool:
    return torch.version.hip is not None


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# from liger_kernel.ops.utils import compare_version


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


# from liger_kernel.ops.utils import ensure_contiguous

def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


# from liger_kernel.ops.utils import torch_to_triton_dtype

torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

# if compare_version("triton", operator.ge, "3.0.0"):
#     try:
#         # typical import path with dispatch available
# from triton.language.extra.libdevice import rsqrt
# except ModuleNotFoundError:
#     # for working with NGC containers
#     from triton.language.extra.cuda.libdevice import rsqrt
# else:
from triton.language.math import rsqrt


_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)


@triton.jit
def _fused_add_rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    S_ptr,  # output residual
    S_row_stride,
    X_ptr,
    X_row_stride,
    R_ptr,  # input residual
    R_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,  # constexpr so the `if` blocks can be optimized out
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes the following:
    1. hidden_states = residual + hidden_states
    2. residual = hidden_states
    3. hidden_states = rmsnorm(hidden_states)

    This is a commonly used pattern in the decoder layers of LLMs.
    Some examples:
    1. https://github.com/huggingface/transformers/blob/0dc2df5ddafe3cb5824ad24e85beba13e0aa6726/src/transformers/models/qwen3/modeling_qwen3.py#L271
    2. https://github.com/huggingface/transformers/blob/0dc2df5ddafe3cb5824ad24e85beba13e0aa6726/src/transformers/models/llama4/modeling_llama4.py#L393

    This kernel is inspired by the rms_norm forward kernel, and is adapted to support the residual addition in the forward pass.
    The backward pass is also adapted to support the residual addition in the backward pass.
    """

    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    S_ptr += row_idx * S_row_stride
    X_ptr += row_idx * X_row_stride
    R_ptr += row_idx * R_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    R_row = tl.load(R_ptr + col_offsets, mask=mask, other=0)
    S_row = X_row + R_row
    tl.store(S_ptr + col_offsets, S_row, mask=mask)
    S_row_dtype = S_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    # On Llama, only rstd is computed on fp32
    if casting_mode == _CASTING_MODE_LLAMA:
        S_row = S_row.to(tl.float32)

    # Gemma computes everything on fp32, and then casts back the output to the original dtype
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row.to(tl.float32)
        S_row = S_row.to(tl.float32)

    if casting_mode == _CASTING_MODE_NONE:
        eps = eps.to(S_row_dtype)
        offset = offset.to(S_row_dtype)

    mean_square = tl.sum(S_row * S_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)

    # We can save time by caching rms with minimal memory overhead
    # because rms is much smaller compared to X_row, as rms is for each row.
    # However, on the computation side, it can save 4 operations (*, sum, /, sqrt).
    tl.store(RSTD_ptr, rstd)

    S_row = S_row * rstd

    # On Llama, the multiplication with the weight is done on the original dtype
    if casting_mode == _CASTING_MODE_LLAMA:
        S_row = S_row.to(S_row_dtype)

    Y_row = S_row * (offset + W_row)

    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row.to(S_row_dtype)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _fused_add_rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dS_out_ptr,
    dS_out_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program: tl.constexpr,
    casting_mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    has_dS_out: tl.constexpr,
):
    """
    This kernel is adapted from the rms_norm backward kernel, and is adapted to support the residual
    addition in the backward pass. For the following code pattern:
    1. hidden_states = residual + hidden_states
    2. residual = hidden_states
    3. hidden_states = rmsnorm(hidden_states)

    The gradient of hidden_states and residual comes out be exactly same. The value of this gradient is
    the sum of the gradient of the hidden_states in step 3 and the gradient of the residual in step 2.

    The backward pass computation logic is same as the rms_norm backward kernel, except that the gradient
    of the hidden_states in step 3 and the gradient of the residual in step 2 are summed up.
    """

    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    dY_ptr += row_start * dY_row_stride
    dX_ptr += row_start * dX_row_stride
    if has_dS_out:
        dS_out_ptr += row_start * dS_out_row_stride

    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row + offset

    for _ in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)

        # Get cached rms
        rstd_row = tl.load(RSTD_ptr)

        X_row = X_row.to(tl.float32)

        # Different bacward graphs for different casting modes
        if casting_mode == _CASTING_MODE_LLAMA:
            m = (dY_row * W_row).to(tl.float32)

        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row.to(tl.float32)
            m = dY_row * W_row
        else:
            m = dY_row * W_row

        dX_row = rstd_row * m

        if has_dS_out:
            dS_out_row = tl.load(dS_out_ptr + col_offsets, mask=mask, other=0.0)
            dX_row += (rstd_row) * (
                -(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
            ) + dS_out_row
            dS_out_ptr += dS_out_row_stride
        else:
            dX_row += (rstd_row) * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)

        # calculate the gradient of W
        if casting_mode == _CASTING_MODE_LLAMA:
            dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
        else:
            # here X_row is already in fp32 (see previous if block)
            dW_row += dY_row * (X_row * rstd_row)

        tl.store(dX_ptr + col_offsets, dX_row.to(X_dtype), mask=mask)

        dY_ptr += dY_row_stride
        dX_ptr += dX_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride

    tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


_str_to_casting_mode = {
    "llama": _CASTING_MODE_LLAMA.value,
    "gemma": _CASTING_MODE_GEMMA.value,
    "none": _CASTING_MODE_NONE.value,
}


def fused_add_rms_norm_forward(X, R, W, eps, offset, casting_mode):
    if not isinstance(casting_mode, int):
        assert casting_mode in _str_to_casting_mode, f"Invalid casting mode: {casting_mode}"
        casting_mode = _str_to_casting_mode[casting_mode]
    else:
        assert casting_mode in _str_to_casting_mode.values(), f"Invalid casting mode: {casting_mode}"

    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    R = R.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    S = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    # RSTD is to cache rstd for each row
    # RSTD is always computed/stored in fp32 if we are using Llama or Gemma casting mode
    rstd_dtype = torch.float32 if casting_mode in (_CASTING_MODE_LLAMA.value, _CASTING_MODE_GEMMA.value) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    # Check constraints.
    assert X.shape[1] == W.shape[0], "Incompatible hidden size dimension between tensor1.shape[1] and tensor2.shape[0]"

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        kernel_args["grf_mode"] = "large"

    # TODO: add _block_fused_add_rms_norm_forward_kernel
    _fused_add_rms_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        S,
        S.stride(0),
        X,
        X.stride(0),
        R,
        R.stride(0),
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        offset,
        casting_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,  # XPU-specific optimization
    )

    return Y.view(*shape), S.view(*shape), RSTD, BLOCK_SIZE, num_warps, casting_mode


def fused_add_rms_norm_backward(dY, dS_out, S, W, RSTD, offset, casting_mode, BLOCK_SIZE, num_warps, in_place):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    dS_out = dS_out.view(-1, dim)
    S = S.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = 1
    if S.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(S.device).multi_processor_count
    elif S.device.type == "xpu":
        sm_count = torch.xpu.get_device_properties(S.device).gpu_eu_count

    # fp32 for numerical stability especially.
    _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    if in_place is True:
        dX = dY
    else:
        dX = torch.empty_like(dY)

    # XPU-specific optimization
    kernel_args = {}
    if S.device.type == "xpu":
        kernel_args["grf_mode"] = "large"

    # TODO: add _block_fused_add_rms_norm_backward_kernel
    _fused_add_rms_norm_backward_kernel[grid](
        dY,
        dY.stride(0),
        dS_out,
        dS_out.stride(0),
        dX,
        dX.stride(0),
        S,
        S.stride(0),
        torch_to_triton_dtype[S.dtype],
        W,
        W.stride(0),
        RSTD,
        RSTD.stride(0),
        _dW,
        _dW.stride(0),
        n_rows,
        n_cols,
        offset,
        rows_per_program,
        casting_mode,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        has_dS_out=dS_out is not None,
        **kernel_args,  # XPU-specific optimization
    )

    dX = dX.view(*shape)
    dW = _dW.sum(dim=0).to(W.dtype)

    return dX, dX, dW  # dR is equal to dX


class LigerFusedAddRMSNormFunction(torch.autograd.Function):
    """
    Performs a fused operation that first adds a residual tensor to the hidden_states tensor (`X`), then applies RMSNorm (Root Mean Square Normalization) to the result using the weight tensor `W`, with optional offset and casting mode.

    This class implements the following sequence, commonly used in transformer decoder layers:
        1. hidden_states = residual + hidden_states
        2. residual = hidden_states (after addition)
        3. hidden_states = rmsnorm(hidden_states)

    Both the normalized hidden_states and the updated residual are returned as outputs.

    Some models use an 'offset' to shift the weight tensor `W` by a constant value. For example, Gemma
    uses an offset of 1.0, so the computation becomes `(X / RMS(X)) * (W + 1.0)` instead of the usual
    `(X / RMS(X)) * W`. You can pass the offset value as an argument to the forward function.

    In addition, different models cast their inputs at different places during RMSNorm computation. For
    example, Gemma casts everything to fp32 before starting the computation, while Llama casts only the
    inverse RMS to fp32. You can specify the casting mode using the `casting_mode` argument. We currently
    support the following casting modes (they match HuggingFace Transformers' implementations):
    - 'llama': matches the Llama implementation, where only the inverse RMS is computed on fp32.
    - 'gemma': matches the Gemma implementation, where everything is cast to fp32, then computed, then cast back to the original dtype.
    - 'none': no casting is done. The computation is done in the original dtype. This saves memory and is slightly faster, but has more error w.r.t. the original implementation.

    The `in_place` option determines whether to modify dY in-place to store dX. This defaults to `True` to save memory.
    """

    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, R, W, eps, offset=0.0, casting_mode="llama", in_place=False):
        """
        X: (B, T, H) or (BxT, H)
        W: (H,)
        """
        # TODO: add row_mode
        Y, S, RSTD, BLOCK_SIZE, num_warps, casting_mode = fused_add_rms_norm_forward(X, R, W, eps, offset, casting_mode)
        ctx.offset = offset
        ctx.casting_mode = casting_mode
        ctx.in_place = in_place
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(S, W, RSTD)
        return Y, S

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY, dS_out):
        """
        Y: (B, T, H) or (BxT, H)
        """
        S, W, RSTD = ctx.saved_tensors
        dX, dR, dW = fused_add_rms_norm_backward(
            dY,
            dS_out,
            S,
            W,
            RSTD,
            ctx.offset,
            ctx.casting_mode,
            ctx.BLOCK_SIZE,
            ctx.num_warps,
            ctx.in_place,
        )

        return dX, dR, dW, None, None, None, None, None










import torch
import torch.nn as nn
import triton

# from utils import QUANTILES
# from utils import SingleBenchmarkRunInput
# from utils import SingleBenchmarkRunOutput
# from utils import _test_memory
# from utils import parse_benchmark_script_args
# from utils import run_benchmarks

# =======================================================================================================================================================================================================================
import argparse
import csv
import json
import os
import time

from collections import OrderedDict
from dataclasses import asdict
from dataclasses import dataclass
from importlib.metadata import version
from itertools import zip_longest
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch

# from liger_kernel.utils import infer_device


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

device = infer_device()

LIGER_KERNEL_VERSION = None # version("liger-kernel")

QUANTILES = [0.5, 0.2, 0.8]


@dataclass
class SingleBenchmarkRunInput:
    x: Union[int, float]
    kernel_provider: str
    kernel_operation_mode: Optional[str] = ""
    extra_benchmark_config: Optional[Dict[str, Any]] = None


@dataclass
class SingleBenchmarkRunOutput:
    # 20th percentile
    y_20: float
    # 50th percentile (median)
    y_50: float
    # 80th percentile
    y_80: float


@dataclass
class BenchmarkData:
    """
    BenchmarkData is a dataclass to store the benchmark data for a a completed benchmark
    run on all x-values for a given kernel/kernel operation mode/metric/extra_benchmark_config
    """

    kernel_name: str
    kernel_provider: str
    metric_name: str
    metric_unit: str
    gpu_name: str
    x_name: str
    x_label: str
    x_values: List[float]
    y_values_50: List[float]
    y_values_20: List[float]
    y_values_80: List[float]
    timestamp: str
    kernel_operation_mode: Optional[str] = None
    extra_benchmark_config_str: Optional[str] = None
    liger_version: str = LIGER_KERNEL_VERSION


@dataclass
class BenchmarkDataCSVRow:
    # The ordering of field names here will be the order of columns in the CSV
    kernel_name: str
    kernel_provider: str
    kernel_operation_mode: Union[str, None]
    metric_name: str
    metric_unit: str
    x_name: str
    x_label: str
    x_value: float
    y_value_50: float
    y_value_20: float
    y_value_80: float
    extra_benchmark_config_str: Union[str, None]
    gpu_name: str
    timestamp: str
    liger_version: str


def _test_memory(
    func: Callable,
    _iter: int = 10,
    quantiles: Optional[List[float]] = None,
    return_mode="mean",
) -> float:
    assert return_mode in ["min", "max", "mean", "median"]
    total_mem = []

    for _ in range(_iter):
        getattr(torch, device).memory.reset_peak_memory_stats()
        func()
        # Convert to MB
        mem = getattr(torch, device).max_memory_allocated() / 2**20
        total_mem.append(mem)

    total_mem = torch.tensor(total_mem, dtype=torch.float)
    if quantiles is not None:
        quantiles_data = torch.quantile(total_mem, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(quantiles_data) == 1:
            quantiles_data = quantiles_data[0]
        return quantiles_data
    return getattr(torch, return_mode)(total_mem).item()


def get_current_file_directory() -> str:
    """
    Returns the directory path of the current Python file.
    """
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory path of the current file
    return os.path.dirname(current_file_path)


def sleep(seconds):
    def decorator(function):
        def wrapper(*args, **kwargs):
            time.sleep(seconds)
            return function(*args, **kwargs)

        return wrapper

    return decorator


def _print_benchmarking_banner(metric_name: str, kernel_name: str):
    print("**************************************")
    print(f"     BENCHMARKING {metric_name.upper()} for {kernel_name.upper()}")
    print("**************************************")


def get_formatted_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_gpu_name():
    """
    Returns the current GPU name, formatted to serve as a directory name
    """
    torch_device = getattr(torch, device)
    if torch_device.is_available():
        gpu_name = torch_device.get_device_name(torch_device.current_device())
        return gpu_name
    else:
        raise Exception("Benchmarks can only be run on GPU.")


def update_benchmark_data_csv(
    benchmark_data_list: List[BenchmarkData],
    filename: str = "all_benchmark_data.csv",
    overwrite: bool = True,
):
    """
    Update the CSV file with the new benchmark data. If the file does not exist, create it.
    If an entry already exists for the benchmark, then overwrite it if `overwrite` is True.
    """

    def create_unique_key(row):
        # This unique key is used to determine if a benchmark run already exists in the CSV
        # If the key is the same, then the benchmark run already exists and will optionally
        # be overwritten. Otherwise, it is considered a new benchmark run and appended.
        return (
            row["kernel_name"],
            row["kernel_provider"],
            row["kernel_operation_mode"] if row["kernel_operation_mode"] else "",
            row["metric_name"],
            row["x_name"],
            str(row["x_value"]),
            (row["extra_benchmark_config_str"] if row["extra_benchmark_config_str"] else ""),
            row["gpu_name"],
        )

    fieldnames = BenchmarkDataCSVRow.__annotations__.keys()

    # Make filename path relative to current file
    filename_abs_path = os.path.join(get_current_file_directory(), "../data", filename)
    file_exists = os.path.isfile(filename_abs_path)

    # Read existing data into a list of dicts
    existing_data = []
    if file_exists:
        with open(filename_abs_path, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                existing_data.append(row)

    existing_data_dict = OrderedDict((create_unique_key(row), row) for row in existing_data)

    for benchmark_data in benchmark_data_list:
        benchmark_data_dict = asdict(benchmark_data)
        x_values = benchmark_data_dict.pop("x_values")
        y_values_50 = benchmark_data_dict.pop("y_values_50")
        y_values_20 = benchmark_data_dict.pop("y_values_20")
        y_values_80 = benchmark_data_dict.pop("y_values_80")

        # Need to convert benchmark_data into multiple rows based on x_values and y_values
        for x_value, y_value_50, y_value_20, y_value_80 in zip_longest(x_values, y_values_50, y_values_20, y_values_80):
            if y_value_50 is None:
                y_value_50 = float("nan")
            if y_value_20 is None:
                y_value_20 = float("nan")
            if y_value_80 is None:
                y_value_80 = float("nan")

            row = BenchmarkDataCSVRow(
                x_value=x_value,
                y_value_50=y_value_50,
                y_value_20=y_value_20,
                y_value_80=y_value_80,
                **benchmark_data_dict,
            )
            row_dict = asdict(row)

            row_key = create_unique_key(row_dict)

            if row_key in existing_data_dict:
                if overwrite:
                    # If overwriting, update the row
                    existing_data_dict[row_key] = row_dict
                else:
                    # If not overwriting, skip this row
                    pass
            else:
                existing_data_dict[row_key] = row_dict
    os.makedirs(os.path.dirname(filename_abs_path), exist_ok=True)
    with open(filename_abs_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_data_dict.values():
            writer.writerow(row)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        return super().default(self, obj)


def print_benchmark_data(benchmark_data_list: List[BenchmarkData]) -> str:
    print("********** Benchmark Data **********")
    formatted_list = [obj.__dict__ for obj in benchmark_data_list]
    print(json.dumps(formatted_list, indent=2))


def run_benchmarks(
    bench_test_fn: Callable,
    kernel_name: str,
    metric_name: str,
    metric_unit: str,
    x_name: str,
    x_label: str,
    x_values: List[Union[float, int]],
    kernel_providers: List[str],
    kernel_operation_modes: Optional[List[str]] = [None],
    extra_benchmark_configs: Optional[List[Dict[str, Any]]] = None,
    overwrite: bool = False,
):
    """
    Run benchmarks given a bench_test_fn that takes in a SingleBenchmarkRunInput as input and
    saves data to the CSV file.

    Args:
        - bench_test_fn: The benchmark test function to run. This function should take in a
            SingleBenchmarkRunInput as input and return a SingleBenchmarkRunOutput.
        - kernel_name: The name of the kernel being benchmarked (e.g. "swiglu")
        - metric_name: The name of the metric being benchmarked (e.g. "speed" or "memory")
        - metric_unit: The unit of the metric being benchmarked (e.g. "ms" or "MB")
        - x_name: The name of the x-axis (e.g. "T" for sequence length)
        - x_label: The label of the x-axis (e.g. "sequence length")
        - x_values: The list of x-values to run the benchmark on (e.g. [2**i for i in range(10, 14)])
        - kernel_providers: The list of kernel providers to run the benchmark on (e.g. ["liger", "huggingface"])
        - kernel_operation_modes: The list of kernel operation modes to run the benchmark on (e.g. ["full", "backward"])
        - extra_benchmark_configs: The list of extra benchmark configurations to run the benchmark on.
        - overwrite: Whether to overwrite the existing benchmark data entry if it already exists.
    """

    assert len(kernel_operation_modes) >= 1
    assert len(kernel_providers) >= 1

    _print_benchmarking_banner(metric_name=metric_name, kernel_name=kernel_name)

    gpu_name = get_gpu_name()
    benchmark_data_list = []
    for extra_benchmark_config in extra_benchmark_configs:
        for kernel_operation_mode in kernel_operation_modes:
            for kernel_provider in kernel_providers:
                y_values_50 = []
                y_values_20 = []
                y_values_80 = []

                for x in x_values:
                    single_benchmark_run_input = SingleBenchmarkRunInput(
                        x=x,
                        kernel_provider=kernel_provider,
                        kernel_operation_mode=kernel_operation_mode,
                        extra_benchmark_config=extra_benchmark_config,
                    )
                    benchmark_result: SingleBenchmarkRunOutput = bench_test_fn(single_benchmark_run_input)
                    y_values_50.append(benchmark_result.y_50)
                    y_values_20.append(benchmark_result.y_20)
                    y_values_80.append(benchmark_result.y_80)

                benchmark_run_data = BenchmarkData(
                    kernel_name=kernel_name,
                    kernel_operation_mode=kernel_operation_mode,
                    kernel_provider=kernel_provider,
                    metric_name=metric_name,
                    metric_unit=metric_unit,
                    gpu_name=gpu_name,
                    x_name=x_name,
                    x_label=x_label,
                    x_values=x_values,
                    y_values_50=y_values_50,
                    y_values_20=y_values_20,
                    y_values_80=y_values_80,
                    extra_benchmark_config_str=json.dumps(extra_benchmark_config, cls=CustomEncoder),
                    timestamp=get_formatted_time(),
                    liger_version=LIGER_KERNEL_VERSION,
                )

                benchmark_data_list.append(benchmark_run_data)

    print_benchmark_data(benchmark_data_list)

    update_benchmark_data_csv(benchmark_data_list=benchmark_data_list, overwrite=overwrite)


def parse_benchmark_script_args():
    parser = argparse.ArgumentParser(description="Benchmarking script for Liger-Kernel")

    # Add an optional --overwrite flag
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Flag to overwrite existing benchmark data with current run.",
    )

    args = parser.parse_args()
    return args


# ==============================================================================================================================================================================

# from liger_kernel.transformers.fused_add_rms_norm import LigerFusedAddRMSNorm
# from liger_kernel.transformers.rms_norm import LigerRMSNorm
# from liger_kernel.utils import infer_device

import torch
import torch.nn as nn

# from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction


class LigerFusedAddRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        init_fn="ones",
        in_place=False,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size))
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place = (eps, offset, casting_mode, in_place)

    def forward(self, hidden_states, residual):
        return LigerFusedAddRMSNormFunction.apply(
            hidden_states,
            residual,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
        )

    def extra_repr(self):
        return (
            f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}"
        )




device = infer_device()


class NaiveAddRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Naive implementation of the add residual rms norm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        residual = residual.to(torch.float32)
        hidden_states = hidden_states + residual
        residual = hidden_states
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype), residual.to(input_dtype)


class AddLigerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        AddLigerRMSNorm is equivalent to NaiveAddRMSNorm class above, but uses the LigerRMSNorm kernel.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.rms_norm = LigerRMSNorm(hidden_size, eps, in_place=False)

    def forward(self, hidden_states, residual):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        residual = residual.to(torch.float32)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states)
        return self.weight * hidden_states.to(input_dtype), residual.to(input_dtype)


def bench_speed_fused_residual_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)

    # Fused Add RMS Norm
    fused_add_rms_norm = LigerFusedAddRMSNorm(hidden_size=N, eps=eps).to(device)
    # Naive implementation
    naive_rms_norm = NaiveAddRMSNorm(hidden_size=N, eps=eps).to(device)
    # # LigerRMSNorm without fused residual addition
    # liger_rms_norm = AddLigerRMSNorm(hidden_size=N, eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    r = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    ds = torch.randn_like(r)
    x.requires_grad_(True)
    r.requires_grad_(True)
    # utility functions

    def y_fwd():
        if provider == "liger_fused_add_rms_norm":
            return fused_add_rms_norm(x, r)

        if provider == "huggingface":
            return naive_rms_norm(x, r)

        if provider == "liger_rms_norm":
            return liger_rms_norm(x, r)

    if mode == "forward":
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            y_fwd,
            grad_to_none=[x, r],
            # comment: changed
            # rep=500,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "backward":
        y, s = y_fwd()
        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: (torch.autograd.backward((y, s), (dy, ds), retain_graph=True)),
            grad_to_none=[x, r],
            # comment: changed
            # rep=500,
            rep=100,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y, s = y_fwd()
            torch.autograd.backward((y, s), (dy, ds))

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            grad_to_none=[x, r],
            # comment: changed
            # rep=500,
            rep=100,
            quantiles=QUANTILES,
        )

    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


def bench_memory_fused_residual_rms_norm(input: SingleBenchmarkRunInput) -> SingleBenchmarkRunOutput:
    N = input.x
    provider = input.kernel_provider

    extra_benchmark_config = input.extra_benchmark_config
    M = extra_benchmark_config["M"]
    eps = extra_benchmark_config["eps"]
    dtype = extra_benchmark_config["dtype"]

    x_shape = (M, N)

    fused_add_rms_norm = LigerFusedAddRMSNorm(hidden_size=N, eps=eps).to(device)
    naive_rms_norm = NaiveAddRMSNorm(hidden_size=N, eps=eps).to(device)
    # liger_rms_norm = AddLigerRMSNorm(hidden_size=N, eps=eps).to(device)

    x = torch.randn(x_shape, dtype=dtype, device=device)
    r = torch.randn(x_shape, dtype=dtype, device=device)
    dy = torch.randn_like(x)
    ds = torch.randn_like(r)
    x.requires_grad_(True)
    r.requires_grad_(True)

    # utility functions
    def y_fwd():
        if provider == "liger_fused_add_rms_norm":
            return fused_add_rms_norm(x, r)
        if provider == "huggingface":
            return naive_rms_norm(x, r)
        # if provider == "liger_rms_norm":
        #     return liger_rms_norm(x, r)

    def full():
        y, s = y_fwd()
        torch.autograd.backward((y, s), (dy, ds))

    mem_50, mem_20, mem_80 = _test_memory(full, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "fused_add_rms_norm",
        "x_name": "H",
        "x_label": "hidden size",
        "x_values": [2**i for i in range(10, 16)],
        "kernel_providers": ["liger_fused_add_rms_norm", "huggingface"], # , "liger_rms_norm"
        "extra_benchmark_configs": [{"M": 2048, "dtype": torch.float32, "eps": 1e-6}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_residual_rms_norm,
        kernel_operation_modes=["forward", "full", "backward"],
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_residual_rms_norm,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
