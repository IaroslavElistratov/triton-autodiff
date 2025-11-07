"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.

The following line
https://github.com/linkedin/Liger-Kernel/blob/7382a8761f9af679482b968f9348013d933947c7/src/liger_kernel/ops/utils.py#L23
is based on code from Unsloth, located at:
https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

Modifications made by Yanning Chen, 2024.
"""

# todo: attribute to ligner
# linkedin_Liger_Kernel/Liger-Kernel-main/src/liger_kernel/ops/fused_linear_cross_entropy.py

# python -m kernel_agent.test.bwd.fused_linear_cross_entropy


from ..fused_linear_cross_entropy import fused_linear_cross_entropy_forward, is_hip, MAX_FUSED_SIZE


import torch
import triton

#################### utils ####################

import argparse
import csv
import json
import os
import time

from collections import OrderedDict
from dataclasses import asdict
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


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


#################################


from typing import Optional
import functools
import importlib
import operator
from typing import Callable
from packaging.version import Version


import triton.language as tl




def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


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


def compare_version(package: str, operator: Callable, target: str):
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        return False
    pkg_version = Version(pkg.__version__)
    return operator(pkg_version, Version(target))


def get_amp_custom_fwd_bwd() -> Callable:
    device = infer_device()
    if compare_version("torch", operator.ge, "2.4.0"):
        return (
            functools.partial(torch.amp.custom_fwd, device_type=device),
            functools.partial(torch.amp.custom_bwd, device_type=device),
        )
    return torch.cuda.amp.custom_fwd, torch.cuda.amp.custom_bwd


amp_custom_fwd, amp_custom_bwd = get_amp_custom_fwd_bwd()


torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """

    # Get the program ID and convert it to int64 to avoid overflow
    program_id = tl.program_id(0).to(tl.int64)

    # Locate the start index
    X_ptr += program_id * X_stride

    # Load the gradient output value
    grad_output = tl.load(grad_output_ptr)

    # Perform the element-wise multiplication
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


def fused_linear_cross_entropy_backward(grad_output, grad_input, grad_weight, grad_bias):
    # If cross entropy is the last layer, grad_output is 1.0. Skip the mul to save time
    if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        # We use a Triton kernel instead of a PyTorch operation because modifying inputs in-place
        # for gradient storage and backward multiple times causes anomalies with PyTorch but not with Triton.
        BT, H = grad_input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))

        element_mul_kernel[(n_rows,)](
            grad_input,
            grad_input.stride(-2),
            grad_output,
            H,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not is_hip() else 16,
        )

        # handle grad_weight
        if grad_weight is not None:
            V, H = grad_weight.shape
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_weight,
                grad_weight.stride(-2),
                grad_output,
                H,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )

        if grad_bias is not None:
            V = grad_bias.shape[0]
            n_rows = V

            element_mul_kernel[(n_rows,)](
                grad_bias,
                grad_bias.stride(-1),
                grad_output,
                1,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=32 if not is_hip() else 16,
            )
    return grad_input, grad_weight, grad_bias


class LigerFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        _input,
        weight,
        target,
        bias=None,
        ce_weight=None,
        ignore_index=-100,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction="mean",
        softcap=None,
        return_z_loss: bool = False,
        accum_dtype=None,
        use_token_scaling: bool = False,
    ):
        """
        Fusing the last linear layer with cross-entropy loss
            Reference: https://github.com/mgmalek/efficient_cross_entropy

        Handle the forward and backward pass of the final linear layer via cross-entropy loss by avoiding
        the materialization of the large logits tensor. Since Cross Entropy Loss is the last layer, we can
        compute the gradient at the forward pass. By doing so, we don't have to store the _input and target
        for the backward pass.

        _input: (B*T, H) where B is batch size, T is sequence length, H is hidden dimension.
        target: (B*T) where each value is in [0, V-1]
        weight: (V, H) where V is the number of classes
        bias: (V) where V is the number of classes
        ce_weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size V and floating point dtype
        ignore_index: the index to ignore in the target
        label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction: reduction to apply
        accum_dtype (torch.dtype): the dtype of intermediate result buffers for weight and bias gradient accumulations.
            Recommended to set `accum_dtype` to higher precision, e.g. `torch.float32`, if the training is unstable with original dtype. Default: `None`, performing accumulations in original dtype
        use_token_scaling (bool): whether to scale each token's loss by its predicted probability (detached).
            When True, each token's loss is multiplied by the model's predicted probability for that token's true class.
            Default: False.
        """

        loss, z_loss, grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_forward(
            _input=_input,
            weight=weight,
            target=target,
            bias=bias,
            ce_weight=ce_weight,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=softcap,
            return_z_loss=return_z_loss,
            accum_dtype=accum_dtype,
            use_token_scaling=use_token_scaling,
        )
        # downcast to dtype and store for backward
        ctx.save_for_backward(
            grad_input.detach(),
            grad_weight.detach() if grad_weight is not None else None,
            grad_bias.detach() if bias is not None else None,
        )
        ctx.return_z_loss = return_z_loss
        return loss, z_loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_output2):
        if ctx.return_z_loss:
            del grad_output2  # z_loss is only for logging
        (grad_input, grad_weight, grad_bias) = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = fused_linear_cross_entropy_backward(
            grad_output, grad_input, grad_weight, grad_bias
        )
        return (
            grad_input,
            grad_weight,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # use_token_scaling
        )




class LigerFusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be 'mean' or 'sum' or 'none'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.accum_dtype = accum_dtype
        self.use_token_scaling = use_token_scaling

    def forward(self, lin_weight, _input, target, bias=None):
        loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ce_weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
            self.accum_dtype,
            self.use_token_scaling,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss


device = infer_device()


class TorchLMHeadCE(torch.nn.Module):
    """Ground truth implementation of the linear fused with torch based cross entropy loss.

    :param H: hidden size
    :param V: vocab size
    :param ignore_index: index to ignore
    :param reduction: reduction method
    """

    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, x, y):
        logits = self.lin(x)
        return self.ce_loss(logits, y)


class LigerLMHeadCE(torch.nn.Module):
    def __init__(self, H: int, V: int, dtype: torch.dtype, ignore_index: int = -100, accum_dtype=None):
        super().__init__()
        self.lin = torch.nn.Linear(in_features=H, out_features=V, bias=False, dtype=dtype)
        self.ce_loss = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean", accum_dtype=accum_dtype
        )

    def forward(self, x, y):
        return self.ce_loss(self.lin.weight, x, y)


#############################################################################
# Test the memory consumption of the linear fused cross entropy loss
#############################################################################


def bench_memory_fused_linear_cross_entropy(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    BT = input.x
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider

    lm_head_ce = None
    if provider == "liger":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    elif provider == "liger-fp32-accum":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype, accum_dtype=torch.float32).to(device)
    else:
        lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    def fwd():
        return lm_head_ce(_input, target)

    def full():
        y = fwd()
        y.backward()

    mem_50, mem_20, mem_80 = _test_memory(full, _iter=10, quantiles=QUANTILES)

    return SingleBenchmarkRunOutput(
        y_20=mem_20,
        y_50=mem_50,
        y_80=mem_80,
    )


# #############################################################################
# # Test the speed of the fused linear cross entropy loss
# #############################################################################


def bench_speed_fused_linear_cross_entropy(
    input: SingleBenchmarkRunInput,
) -> SingleBenchmarkRunOutput:
    BT = input.x
    H = input.extra_benchmark_config["H"]
    V = input.extra_benchmark_config["V"]
    dtype = input.extra_benchmark_config["dtype"]
    provider = input.kernel_provider
    mode = input.kernel_operation_mode

    lm_head_ce = None
    if provider == "liger":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype).to(device)
    elif provider == "liger-fp32-accum":
        lm_head_ce = LigerLMHeadCE(H=H, V=V, dtype=dtype, accum_dtype=torch.float32).to(device)
    else:
        lm_head_ce = TorchLMHeadCE(H=H, V=V, dtype=dtype).to(device)

    _input = torch.randn(BT, H, requires_grad=True, dtype=dtype, device=device)
    target = torch.randint(V, (BT, 1), dtype=torch.long, device=device).squeeze(1)

    def fwd():
        return lm_head_ce(_input, target)

    # if mode == "forward":
    #     ms_50, ms_20, ms_80 = triton.testing.do_bench(
    #         fwd,
    #         rep=100,
    #         quantiles=QUANTILES,
    #     )
    # elif mode == "no-grad-forward":
    #     with torch.no_grad():
    #         ms_50, ms_20, ms_80 = triton.testing.do_bench(
    #             fwd,
    #             rep=100,
    #             quantiles=QUANTILES,
    #         )
    if mode == "backward":
        y = fwd()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            lambda: y.backward(retain_graph=True),
            grad_to_none=[_input],
            # comment: changed
            # rep=100,
            rep=50,
            quantiles=QUANTILES,
        )
    elif mode == "full":

        def full():
            y = fwd()
            y.backward()

        ms_50, ms_20, ms_80 = triton.testing.do_bench(
            full,
            # comment: changed
            # rep=100,
            rep=50,
            quantiles=QUANTILES,
        )
    return SingleBenchmarkRunOutput(
        y_20=ms_20,
        y_50=ms_50,
        y_80=ms_80,
    )


if __name__ == "__main__":
    args = parse_benchmark_script_args()

    common_configs = {
        "kernel_name": "fused_linear_cross_entropy",
        "x_name": "BT",
        "x_label": "B x T",
        # comment:
        # i chagned from ""x_values": [2**i for i in range(12, 16)],"
        "x_values": [2**i for i in range(12, 14)],
        "kernel_providers": ["liger", "liger-fp32-accum", "huggingface"],
        "extra_benchmark_configs": [{"H": 4096, "V": 128256, "mode": "forward", "dtype": torch.bfloat16}],
        "overwrite": args.overwrite,
    }

    run_benchmarks(
        bench_test_fn=bench_speed_fused_linear_cross_entropy,
        kernel_operation_modes=["backward", "full"], # "forward", "no-grad-forward"
        metric_name="speed",
        metric_unit="ms",
        **common_configs,
    )
    run_benchmarks(
        bench_test_fn=bench_memory_fused_linear_cross_entropy,
        kernel_operation_modes=["full"],
        metric_name="memory",
        metric_unit="MB",
        **common_configs,
    )
