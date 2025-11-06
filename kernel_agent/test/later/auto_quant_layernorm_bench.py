# Pulled the exact upstream helpers (fla/utils.py, fla/modules/layernorm.py, fla/modules/fused_bitlinear.py) verbatim into a single launcher (kernel_agent/
# standalone_quant_layernorm_bench.py) and wired the Triton benchmark to instantiate the 1.58-bit FusedBitLinear kernel instead of the original fused LayerNorm

# Source files embedded verbatim from:
#   fla_org_flash_linear_attention/flash-linear-attention-main/fla/utils.py
#   fla_org_flash_linear_attention/flash-linear-attention-main/fla/modules/layernorm.py
#   fla_org_flash_linear_attention/flash-linear-attention-main/fla/modules/fused_bitlinear.py
#   fla_org_flash_linear_attention/flash-linear-attention-main/benchmarks/modules/benchmark_layernorm.py
# Only the fused LayerNorm benchmark branch swaps in FusedBitLinear, per instructions.

# Performance:
#          T  naive_ln   fused_ln  naive_gn  fused_gn  naive_ln_bwd  fused_ln_bwd  naive_gn_bwd  fused_gn_bwd
# 0    128.0  0.035936   0.159200  0.069232  0.033664      0.086208      0.545248      0.149536      0.715744
# 1    256.0  0.066800   0.230720  0.107552  0.062160      0.148624      0.605136      0.260128      0.158784
# 2    512.0  0.120384   0.379248  0.181920  0.116096      0.396928      1.180448      0.516160      0.331040
# 3   1024.0  0.233424   0.707328  0.388768  0.228032      0.894016      2.418400      1.339424      0.735520
# 4   2048.0  0.454688   1.349248  0.787152  0.446128      1.750240      4.902080      2.816544      1.508640
# 5   4096.0  0.898608   2.690528  1.559584  0.882176      3.514688      9.747424      5.703616      3.051488
# 6   8192.0  1.780288   5.295152  3.103520  1.755136      7.153120     19.445552     11.511632      6.144128
# 7  16384.0  3.551360  10.519216  6.196256  3.510976     14.547280     38.184929     23.112225     12.304192


# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import triton
import triton.testing

FLA_UTILS_SRC = r'''
# -*- coding: utf-8 -*-

import contextlib
import functools
import inspect
import logging
import os
import sys
import warnings
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch
import triton
from packaging import version

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fla import __version__

FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'


supports_autotune_cache = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": FLA_CACHE_RESULTS} if supports_autotune_cache else {}


@lru_cache(maxsize=1)
def check_environments():
    """
    Checks the current operating system, Triton version, and Python version,
    issuing warnings if they don't meet recommendations.
    This function's body only runs once due to lru_cache.
    """
    # Check Operating System
    if sys.platform == 'win32':
        logger.warning(
            "Detected Windows operating system. Triton does not have an official Windows release, "
            "thus FLA will not be adapted for Windows, and any potential errors will not be fixed. "
            "Please consider using a Linux environment for compatibility."
        )

    triton_version = version.parse(triton.__version__)
    required_triton_version = version.parse("3.2.0")

    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.2.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton."
        )

    # Check Python version
    py_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = version.parse("3.11")

    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience."
        )

    return None


check_environments()


def get_abs_err(x, y):
    return (x.detach()-y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach()-y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (FLA_CI_ENV and (error_rate < 0.01 or abs_atol <= 0.3)):
        if error_rate > ratio:
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def input_guard(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = custom_device_ctx(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version
            require_version(version, hint)
            return fn(ctx,
                      *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                      **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
        return wrapper
    return decorator


class Action(Enum):
    NONE = "none"
    NOTIFY = "notify"
    NOTIFY_ALWAYS = "notify_always"
    RAISE = "raise"


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: Optional[str] = None,
    warn_if_greater_or_equal_version: bool = False,
    raise_if_greater_or_equal_version: bool = False,
    raise_if_both_names: bool = False,
    additional_message: Optional[str] = None,
):
    """
    Decorator to notify users about deprecated keyword arguments, replacing them with a new name if specified.

    This decorator allows you to:
    - Notify users when a keyword argument is deprecated.
    - Automatically replace deprecated keyword arguments with new ones.
    - Raise an error if deprecated arguments are used, depending on the specified conditions.

    By default, the decorator notifies the user about the deprecated argument while the `fla.__version__` < specified `version`
    in the decorator. To keep notifications with any version `warn_if_greater_or_equal_version=True` can be set.

    Args:
        old_name (`str`):
            Name of the deprecated keyword argument.
        version (`str`):
            The version in which the keyword argument was (or will be) deprecated.
        new_name (`Optional[str]`, *optional*):
            The new name for the deprecated keyword argument.
            If specified, the deprecated keyword argument will be replaced with this new name.
        warn_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to show warning if current `fla` version is greater or equal to the deprecated version.
        raise_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if current `fla` version is greater or equal to the deprecated version.
        raise_if_both_names (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if both deprecated and new keyword arguments are set.
        additional_message (`Optional[str]`, *optional*):
            An additional message to append to the default deprecation message.

    Raises:
        ValueError:
            If `raise_if_greater_or_equal_version` is `True` and the current version >= the deprecated one,
            or if `raise_if_both_names` is `True` and both old and new keyword arguments are provided.

    Returns:
        Callable:
            A wrapped function that handles the deprecated keyword arguments according to the specified parameters.

    Example usage with renaming argument:

        ```python
        @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="6.0.0")
        def my_function(do_reduce_labels):
            print(do_reduce_labels)

        my_function(reduce_labels=True)  # Will show a deprecation warning and use do_reduce_labels=True
        ```

    Example usage without renaming argument:

        ```python
        @deprecate_kwarg("max_size", version="6.0.0")
        def my_function(max_size):
            print(max_size)

        my_function(max_size=1333)  # Will show a deprecation warning
        ```

    """
    deprecated_version = version.parse(version)
    current_version = version.parse(__version__)
    is_greater_or_equal_version = current_version >= deprecated_version

    if is_greater_or_equal_version:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        # Required for better warning message
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        is_instance_method = "self" in function_named_args
        is_class_method = "cls" in function_named_args

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            # Get class + function name (just for better warning message)
            func_name = func.__name__
            if is_instance_method:
                func_name = f"{args[0].__class__.__name__}.{func_name}"
            elif is_class_method:
                func_name = f"{args[0].__name__}.{func_name}"

            minimum_action = Action.NONE
            message = None

            # deprecated kwarg and its new version are set for function call -> replace it with new name
            if old_name in kwargs and new_name in kwargs:
                minimum_action = Action.RAISE if raise_if_both_names else Action.NOTIFY_ALWAYS
                message = (
                    f"Both `{old_name}` and `{new_name}` are set for `{func_name}`. "
                    f"Using `{new_name}={kwargs[new_name]}` and ignoring deprecated `{old_name}={kwargs[old_name]}`."
                )
                kwargs.pop(old_name)

            # only deprecated kwarg is set for function call -> replace it with new name
            elif old_name in kwargs and new_name is not None and new_name not in kwargs:
                minimum_action = Action.NOTIFY
                message = (
                    f"`{old_name}` is deprecated {version_message} for `{func_name}`. "
                    f"Use `{new_name}` instead."
                )
                kwargs[new_name] = kwargs.pop(old_name)

            # deprecated kwarg is not set for function call and new name is not specified -> just notify
            elif old_name in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`."

            if message is not None and additional_message is not None:
                message = f"{message} {additional_message}"

            # update minimum_action if argument is ALREADY deprecated (current version >= deprecated version)
            if is_greater_or_equal_version:
                # change to (NOTIFY, NOTIFY_ALWAYS) -> RAISE if specified
                # in case we want to raise error for already deprecated arguments
                if raise_if_greater_or_equal_version and minimum_action != Action.NONE:
                    minimum_action = Action.RAISE

                # change to NOTIFY -> NONE if specified (NOTIFY_ALWAYS can't be changed to NONE)
                # in case we want to ignore notifications for already deprecated arguments
                elif not warn_if_greater_or_equal_version and minimum_action == Action.NOTIFY:
                    minimum_action = Action.NONE

            # raise error or notify user
            if minimum_action == Action.RAISE:
                raise ValueError(message)
            elif minimum_action in (Action.NOTIFY, Action.NOTIFY_ALWAYS):
                # DeprecationWarning is ignored by default, so we use FutureWarning instead
                warnings.warn(message, FutureWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return wrapper


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str = '2.4') -> bool:
    return version.parse(torch.__version__) >= version.parse(version_s)


def _cpu_device_warning():
    warnings.warn(('Triton is not supported on current platform, roll back to CPU.'), stacklevel=1)


@lru_cache(maxsize=None)
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']
    except BaseException:
        # Maybe we use a NPU device.
        if triton.runtime.driver.active.get_current_target().backend == 'npu':
            return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['num_vectorcore']
        else:
            return 1


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        _cpu_device_warning()
        return 'cpu'


def map_triton_backend_to_torch_device() -> str:
    backend = get_available_device()        # 'cuda' | 'hip' | 'xpu' | 'cpu' | ...
    return {'cuda': 'cuda', 'hip': 'cuda', 'xpu': 'xpu'}.get(backend, backend)


# For AMD GPUs, the triton backend is 'hip', while for Nvidia GPUs, the triton backend is 'cuda'.
# However, the torch backend is 'cuda' for both Nvidia and AMD GPUs.
# Therefore, we need to check the triton backend to determine the actual GPU vendor.
device = get_available_device() if get_available_device() != 'hip' else 'cuda'
device_torch_lib = getattr(torch, device)
device_platform = get_available_device()
device_name = map_triton_backend_to_torch_device()

is_amd = (device_platform == 'hip')
is_intel = (device_platform == 'xpu')
is_nvidia = (device_platform == 'cuda')
is_intel_alchemist = (is_intel and 'Intel(R) Arc(TM) A' in torch.xpu.get_device_name(0))
is_nvidia_hopper = (is_nvidia and ('NVIDIA H' in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9))
use_cuda_graph = (is_nvidia and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1')

# Nvidia Ampere or newer, haven't check AMD and intel yet.
is_tf32_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 8)
is_gather_supported = hasattr(triton.language, 'gather')
is_tma_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 9) \
    and os.environ.get('FLA_USE_TMA', '0') == '1' and \
    (hasattr(triton.language, '_experimental_make_tensor_descriptor') or hasattr(triton.language, 'make_tensor_descriptor'))

if is_nvidia and not is_tf32_supported:
    # Make old card happy, since triton will use tf32 by default.
    # This is a workaround for old nvidia card.
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

if is_tma_supported:
    logger.info('TMA is supported, using TMA by default.')

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device=torch.device(device_name, device_torch_lib.current_device()), dtype=torch.int8)

    triton.set_allocator(alloc_fn)


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)['max_shared_mem']
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        _cpu_device_warning()
        return [-1]


class Backend(Enum):
    ADA = 101376       # RTX 4090
    AMPERE = 166912    # A100
    HOPPER = 232448    # H100
    DEFAULT = 102400   # Default

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@lru_cache(maxsize=None)
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False


if check_pytorch_version('2.4'):
    device = 'cuda' if device == 'cpu' else device
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=device)

    def custom_device_ctx(index: int):
        return device_torch_lib.device(index)
else:
    assert device == 'cuda', 'Only cuda device is supported for PyTorch version < 2.4.0.'
    autocast_custom_fwd = device_torch_lib.amp.custom_fwd
    autocast_custom_bwd = device_torch_lib.amp.custom_bwd

    def custom_device_ctx(index: int):
        return torch.cuda.device(index)

'''

FLA_MODULES_LAYERNORM_SRC = r'''
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Copyright (c) 2023, Tri Dao
# https://github.com/state-spaces/mamba/blob/fb7b5310fa865dbd62aa059b1e26f2b431363e2a/mamba_ssm/ops/triton/layernorm.py
# Implement residual + layer_norm / rms_norm.

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_module
from torch.distributed.tensor.parallel import ParallelStyle

from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None


def layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    upcast: bool = False
):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(
        dtype
    )
    return out if not prenorm else (out, x)


def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    upcast: bool = False
):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)


def group_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    is_rms_norm: bool = False,
    prenorm: bool = False,
    upcast: bool = False
):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    residual = x
    x, weight = [
        rearrange(data, "... (g d) -> ... g d", g=num_groups) for data in (x, weight)
    ]
    if bias is not None:
        bias = rearrange(bias, '... (g d) -> ... g d', g=num_groups)
    if not is_rms_norm:
        mean = x.mean(dim=-1, keepdim=True)
        x = x - mean
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = rearrange(out, "... g d -> ... (g d)")
    out = out.to(dtype)
    return out if not prenorm else (out, residual)


class GroupNormRef(nn.Module):

    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        is_rms_norm: bool = False
    ) -> GroupNormRef:
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.is_rms_norm = is_rms_norm

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.num_groups}, {self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        if self.is_rms_norm:
            s += f", is_rms_norm={self.is_rms_norm}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False):
        return group_norm_ref(
            x,
            self.weight,
            self.bias,
            num_groups=self.num_groups,
            residual=residual,
            eps=self.eps,
            is_rms_norm=self.is_rms_norm,
            prenorm=prenorm,
            upcast=True
        )


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [32, 64, 128]
        for num_warps in [2, 4, 8]
    ],
    key=['D', 'NB', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM'],
    **autotune_cache_kwargs
)
@triton.jit
def layer_norm_fwd_kernel(
    x,  # pointer to the input
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    res,  # pointer to the res
    res_out,  # pointer to the res
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    i_t = tl.program_id(0)

    o_t = i_t * BT + tl.arange(0, BT)
    o_g = o_t % G
    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(res, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(res_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_g[:, None] * D + o_d[None, :], mask=m_d[None, :]).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_g[:, None] * D + o_d[None, :], mask=m_d[None, :]).to(tl.float32)
    b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # Write output
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['D', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM'],
    **autotune_cache_kwargs
)
@triton.jit
def layer_norm_fwd_kernel1(
    x,  # pointer to the input
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    res,  # pointer to the res
    res_out,  # pointer to the res
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    G: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    i_t = tl.program_id(0)
    i_g = i_t % G

    x += i_t * D
    y += i_t * D
    if HAS_RESIDUAL:
        res += i_t * D
    if STORE_RESIDUAL_OUT:
        res_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(res + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(res_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # Write output
    tl.store(y + o_d, b_y, mask=m_d)


@triton.heuristics({
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=['D', 'NB', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL', 'IS_RMS_NORM'],
    **autotune_cache_kwargs
)
@triton.jit
def layer_norm_bwd_kernel(
    x,  # pointer to the input
    w,  # pointer to the weights
    b,  # pointer to the biases
    y,  # pointer to the output to be recomputed
    dy,  # pointer to the output gradient
    dx,  # pointer to the input gradient
    dw,  # pointer to the partial sum of weights gradient
    db,  # pointer to the partial sum of biases gradient
    dres,
    dres_in,
    mean,
    rstd,
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    GS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_g, i_sg = i_s // GS, i_s % GS

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=m_d).to(tl.float32)
        b_dw = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_db = tl.zeros((BT, BD), dtype=tl.float32)

    T = min(i_sg * BS + BS, T // G)
    for i_t in range(i_sg * BS, T, BT):
        p_x = tl.make_block_ptr(x + i_g * D, (T, D), (G*D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dy = tl.make_block_ptr(dy + i_g * D, (T, D), (G*D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx + i_g * D, (T, D), (G*D, 1), (i_t, 0), (BT, BD), (1, 0))
        # [BT, BD]
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
        b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)

        if not IS_RMS_NORM:
            p_mean = tl.make_block_ptr(mean + i_g, (T,), (G,), (i_t,), (BT,), (0,))
            b_mean = tl.load(p_mean, boundary_check=(0,))
        p_rstd = tl.make_block_ptr(rstd + i_g, (T,), (G,), (i_t,), (BT,), (0,))
        b_rstd = tl.load(p_rstd, boundary_check=(0,))
        # Compute dx
        b_xhat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
        b_xhat = tl.where(m_d[None, :], b_xhat, 0.0)

        b_y = b_xhat * b_w[None, :] if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]
        if RECOMPUTE_OUTPUT:
            p_y = tl.make_block_ptr(y + i_g * D, (T, D), (G*D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))

        b_wdy = b_dy

        if HAS_WEIGHT or HAS_BIAS:
            m_t = (i_t + tl.arange(0, BT)) < T
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += tl.where(m_t[:, None], b_dy * b_xhat, 0.0)
        if HAS_BIAS:
            b_db += tl.where(m_t[:, None], b_dy, 0.0)
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_c2 = tl.sum(b_wdy, axis=1) / D
            b_dx = (b_wdy - (b_xhat * b_c1[:, None] + b_c2[:, None])) * b_rstd[:, None]
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_dx = (b_wdy - b_xhat * b_c1[:, None]) * b_rstd[:, None]
        if HAS_DRESIDUAL:
            p_dres = tl.make_block_ptr(dres + i_g * D, (T, D), (G*D, 1), (i_t, 0), (BT, BD), (1, 0))
            b_dres = tl.load(p_dres, boundary_check=(0, 1)).to(tl.float32)
            b_dx += b_dres
        # Write dx
        if STORE_DRESIDUAL:
            p_dres_in = tl.make_block_ptr(dres_in + i_g * D, (T, D), (G*D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_dres_in, b_dx.to(p_dres_in.dtype.element_ty), boundary_check=(0, 1))

        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, tl.sum(b_dw, axis=0), mask=m_d)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, tl.sum(b_db, axis=0), mask=m_d)


@triton.heuristics({
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8]
    ],
    key=['D', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL', 'IS_RMS_NORM'],
    **autotune_cache_kwargs
)
@triton.jit
def layer_norm_bwd_kernel1(
    x,  # pointer to the input
    w,  # pointer to the weights
    b,  # pointer to the biases
    y,  # pointer to the output to be recomputed
    dy,  # pointer to the output gradient
    dx,  # pointer to the input gradient
    dw,  # pointer to the partial sum of weights gradient
    db,  # pointer to the partial sum of biases gradient
    dres,
    dres_in,
    mean,
    rstd,
    T,
    G: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    GS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_g, i_sg = i_s // GS, i_s % GS

    o_d = tl.arange(0, BD)
    mask = o_d < D

    if HAS_WEIGHT:
        b_w = tl.load(w + i_g * D + o_d, mask=mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b_b = tl.load(b + i_g * D + o_d, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_t in range(i_sg * BS * G + i_g, min((i_sg * BS + BS) * G + i_g, T), G):
        b_x = tl.load(x + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
        b_dy = tl.load(dy + i_t * D + o_d, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        b_rstd = tl.load(rstd + i_t)
        # Compute dx
        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(mask, b_xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
            if HAS_BIAS:
                b_y = b_y + b_b
            tl.store(y + i_t * D + o_d, b_y, mask=mask)
        b_wdy = b_dy
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += b_dy * b_xhat
        if HAS_BIAS:
            b_db += b_dy
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_c2 = tl.sum(b_wdy, axis=0) / D
            b_dx = (b_wdy - (b_xhat * b_c1 + b_c2)) * b_rstd
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_dx = (b_wdy - b_xhat * b_c1) * b_rstd
        if HAS_DRESIDUAL:
            b_dres = tl.load(dres + i_t * D + o_d, mask=mask, other=0).to(tl.float32)
            b_dx += b_dres
        # Write dx
        b_dx = tl.cast(b_dx, dtype=dx.dtype.element_ty, fp_downcast_rounding='rtne')
        if STORE_DRESIDUAL:
            tl.store(dres_in + i_t * D + o_d, b_dx, mask=mask)
        tl.store(dx + i_t * D + o_d, b_dx, mask=mask)

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=mask)


def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
    num_groups: int = 1,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D, G = *x.shape, num_groups
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (G * D,)
    if bias is not None:
        assert bias.shape == (G * D,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        res_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        res_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        def grid(meta): return (triton.cdiv(T, meta['BT']), )
        layer_norm_fwd_kernel[grid](
            x,
            y,
            weight,
            bias,
            residual,
            res_out,
            mean,
            rstd,
            eps,
            T=T,
            G=G,
            D=D,
            BD=BD,
            NB=NB,
            IS_RMS_NORM=is_rms_norm,
            HAS_RESIDUAL=residual is not None,
            STORE_RESIDUAL_OUT=res_out is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
        )
    else:
        layer_norm_fwd_kernel1[(T,)](
            x,
            y,
            weight,
            bias,
            residual,
            res_out,
            mean,
            rstd,
            eps,
            G=G,
            D=D,
            BD=BD,
            IS_RMS_NORM=is_rms_norm,
            HAS_RESIDUAL=residual is not None,
            STORE_RESIDUAL_OUT=res_out is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
        )
    # res_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, res_out if res_out is not None else x


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    dres: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
    num_groups: int = 1,
):
    T, D, G = *x.shape, num_groups
    assert dy.shape == (T, D)
    if dres is not None:
        assert dres.shape == (T, D)
    if weight is not None:
        assert weight.shape == (G * D,)
    if bias is not None:
        assert bias.shape == (G * D,)
    # allocate output
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dres_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # each program handles one group only
    NS = triton.cdiv(get_multiprocessor_count(x.device.index), G) * G
    BS = triton.cdiv(T, NS)
    GS = NS // G

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None
    grid = (NS,)

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        layer_norm_bwd_kernel[grid](
            x,
            weight,
            bias,
            y,
            dy,
            dx,
            dw,
            db,
            dres,
            dres_in,
            mean,
            rstd,
            T=T,
            G=G,
            D=D,
            BS=BS,
            BD=BD,
            NB=NB,
            GS=GS,
            IS_RMS_NORM=is_rms_norm,
            HAS_DRESIDUAL=dres is not None,
            STORE_DRESIDUAL=dres_in is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
        )
    else:
        layer_norm_bwd_kernel1[grid](
            x,
            weight,
            bias,
            y,
            dy,
            dx,
            dw,
            db,
            dres,
            dres_in,
            mean,
            rstd,
            T=T,
            G=G,
            D=D,
            BS=BS,
            BD=BD,
            GS=GS,
            IS_RMS_NORM=is_rms_norm,
            HAS_DRESIDUAL=dres is not None,
            STORE_DRESIDUAL=dres_in is not None,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
        )
    dw = dw.view(G, -1, D).sum(1).to(weight).view_as(weight) if weight is not None else None
    db = db.view(G, -1, D).sum(1).to(bias).view_as(bias) if bias is not None else None
    # Don't need to compute dres_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dres_in = dx
    return (dx, dw, db, dres_in) if not recompute_output else (dx, dw, db, dres_in, y)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual: torch.Tensor = None,
        eps: float = 1e-5,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
        num_groups: int = 1
    ):
        x_shape_og = x.shape

        if x.shape[-1] % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        # reshape input data into 2D tensor
        x = x.reshape(-1, (x.shape[-1] // num_groups))
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape_as(x)
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, mean, rstd, res_out = layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            num_groups=num_groups
        )
        ctx.save_for_backward(res_out, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.num_groups = num_groups
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, res_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dy, *args):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, (dy.shape[-1] // ctx.num_groups))
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, x.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in = layer_norm_bwd(
            dy,
            x,
            weight,
            bias,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            num_groups=ctx.num_groups
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
            None
        )


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False
):
    return LayerNormFunction.apply(
        x,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm
    )


def group_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    num_groups: int = 1
):
    return LayerNormFunction.apply(
        x,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        num_groups
    )


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False
):
    return LayerNormFunction.apply(
        x,
        weight,
        bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True
    )


def layer_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    num_groups: int = 1
):
    return LayerNormLinearFunction.apply(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
        num_groups
    )


def rms_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False
):
    return layer_norm_linear(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
        residual=residual,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True
    )


def group_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
    num_groups: int = 1
):
    return layer_norm_linear(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
        residual=residual,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=is_rms_norm,
        num_groups=num_groups
    )


class LayerNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> LayerNorm:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )


class GroupNorm(nn.Module):

    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        is_rms_norm: bool = False
    ) -> GroupNorm:
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.is_rms_norm = is_rms_norm

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.num_groups}, {self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        if self.is_rms_norm:
            s += f", is_rms_norm={self.is_rms_norm}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return group_norm(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=self.is_rms_norm,
            num_groups=self.num_groups
        )


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> RMSNorm:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class LayerNormLinearFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-5,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
        num_groups=1
    ):
        x_shape_og = x.shape

        if x.shape[-1] % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        # reshape input data into 2D tensor
        x = x.reshape(-1, (x.shape[-1] // num_groups))
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape_as(x)
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, mean, rstd, res_out = layer_norm_fwd(
            x,
            norm_weight,
            norm_bias,
            eps,
            residual,
            out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(),
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
            num_groups=num_groups
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(res_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.num_groups = num_groups
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, res_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dout, *args):
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dy = dy.reshape(-1, (dy.shape[-1] // ctx.num_groups))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, x.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(
            dy,
            x,
            norm_weight,
            norm_bias,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True,
            num_groups=ctx.num_groups
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y.view(-1, linear_weight.shape[-1]))
        return (
            dx.reshape(ctx.x_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
            None
        )


class LayerNormLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> LayerNormLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_linear(
            x=x,
            norm_weight=self.weight,
            norm_bias=self.bias,
            linear_weight=weight,
            linear_bias=bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False
        )


class GroupNormLinear(nn.Module):

    def __init__(
        self,
        num_groups: int,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        is_rms_norm: bool = False
    ) -> GroupNormLinear:
        super().__init__()

        if hidden_size % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.is_rms_norm = is_rms_norm

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.num_groups}, {self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        if self.is_rms_norm:
            s += f", is_rms_norm={self.is_rms_norm}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_linear(
            x=x,
            norm_weight=self.weight,
            norm_bias=self.bias,
            linear_weight=weight,
            linear_bias=bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=self.is_rms_norm,
            num_groups=self.num_groups
        )


class RMSNormLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5
    ) -> RMSNormLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_linear(
            x=x,
            norm_weight=self.weight,
            norm_bias=self.bias,
            linear_weight=weight,
            linear_bias=bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True
        )


class NormParallel(ParallelStyle):

    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False):
        super().__init__()
        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
            # us to simply just use from_local
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated_param)

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            # if the passed in input DTensor is not sharded on the sequence dim, we need to redistribute it
            if input_tensor.placements != sequence_sharding:
                input_tensor = input_tensor.redistribute(
                    placements=sequence_sharding, async_op=True
                )
            return input_tensor
        elif isinstance(input_tensor, torch.Tensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            return DTensor.from_local(
                input_tensor, device_mesh, sequence_sharding, run_check=False
            )
        else:
            raise ValueError(
                f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.sequence_sharding),
            partial(self._prepare_output_fn, self.use_local_output),
        )

'''

FLA_MODULES_FUSED_BITLINEAR_SRC = r'''
# /root/private-scrape/UNZIPED_selected/fla_org_flash_linear_attention/flash-linear-attention-main/fla/modules/fused_bitlinear.py


# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Implementations of BitLinear layer with fused LayerNorm and quantized Linear layer.
# [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
# [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)

# Code adapted from https://github.com/ridgerchu/matmulfreellm/

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.modules.layernorm import RMSNorm
from fla.utils import autotune_cache_kwargs, get_multiprocessor_count, input_guard, is_amd, require_version

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if is_amd else [1, 2, 4, 8, 16, 32]


def activation_quant(x):
    """
    Per-token quantization to 8 bits. No grouping is needed for quantization.

    Args:
        x: An activation tensor with shape [n, d].

    Returns:
        A quantized activation tensor with shape [n, d].
    """
    # Compute the scale factor
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    # Compute the scale factor
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
    **autotune_cache_kwargs
)
@triton.jit
def layer_norm_fwd_kernel_quant(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    RESIDUAL_OUT,  # pointer to the residual
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd

    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b

    # Aply quantization to the output
    scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
    # Quantize and then de-quantize the tensor
    y = tl.extra.cuda.libdevice.round(y * scale)
    y = tl.maximum(tl.minimum(y, 127), -128) / scale

    # Write output
    tl.store(Y + cols, y, mask=mask)


def layer_norm_fwd_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    layer_norm_fwd_kernel_quant[(M,)](
        x,
        y,
        weight,
        bias,
        residual,
        residual_out,
        mean,
        rstd,
        x.stride(0),
        y.stride(0),
        residual.stride(0) if residual is not None else 0,
        residual_out.stride(0) if residual_out is not None else 0,
        N,
        eps,
        is_rms_norm,
        BLOCK_N,
        residual is not None,
        residual_out is not None,
        weight is not None,
        bias is not None,
    )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


@triton.heuristics({
    "RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
    **autotune_cache_kwargs
)
@triton.jit
def layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DRESIDUAL,
    DRESIDUAL_IN,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dres_row,
    stride_dres_in_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w if HAS_WEIGHT else xhat
            if HAS_BIAS:
                y = y + b

            # Aply quantization to the output
            scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
            # Quantize and then de-quantize the tensor
            y = tl.extra.cuda.libdevice.round(y * scale)
            y = tl.maximum(tl.minimum(y, 127), -128) / scale

            tl.store(Y + cols, y, mask=mask)
        wdy = dy
        if HAS_WEIGHT:
            wdy = dy * w
            dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
):
    M, N = x.shape
    # allocate output
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(M, N, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    sm_count = get_multiprocessor_count(x.device.index)
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device) if weight is not None else None
    _db = torch.empty((sm_count, N), dtype=torch.float32, device=bias.device) if bias is not None else None
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    layer_norm_bwd_kernel[grid](
        x,
        weight,
        bias,
        y,
        dy,
        dx,
        _dw,
        _db,
        dresidual,
        dresidual_in,
        mean,
        rstd,
        x.stride(0),
        0 if not recompute_output else y.stride(0),
        dy.stride(0),
        dx.stride(0),
        dresidual.stride(0) if dresidual is not None else 0,
        dresidual_in.stride(0) if dresidual_in is not None else 0,
        M,
        N,
        eps,
        rows_per_program,
        is_rms_norm,
        BLOCK_N,
        dresidual is not None,
        dresidual_in is not None,
        weight is not None,
        bias is not None,
    )
    dw = _dw.sum(0).to(weight.dtype) if weight is not None else None
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)


class LayerNormLinearQuantFn(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = residual.dtype if residual is not None else (torch.float32 if residual_in_fp32 else None)
        y, mean, rstd, residual_out = layer_norm_fwd_quant(
            x,
            norm_weight,
            norm_bias,
            eps,
            residual,
            out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(),
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = weight_quant(linear_weight).to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dout, *args):
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = layer_norm_bwd(
            dy,
            x,
            norm_weight,
            norm_bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
        return (
            dx.reshape(ctx.x_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def layer_norm_linear_quant_fn(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    return LayerNormLinearQuantFn.apply(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
    )


def rms_norm_linear_quant(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False
):
    return layer_norm_linear_quant_fn(
        x=x,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
        residual=residual,
        eps=eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=True
    )


@require_version("triton>=3.0", "Triton >= 3.0 is required to do online quantization.")
def bit_linear(x, weight, bias=None, norm_weight=None, norm_bias=None, eps=1e-8):
    """
    A functional version of BitLinear that applies quantization to activations and weights.

    Args:
        x: Input tensor with shape [n, d].
        weight: Weight tensor with shape [out_features, in_features].
        bias: Bias tensor with shape [out_features] (optional).
        norm_weight: Weight tensor for RMS normalization with shape [in_features].
        norm_bias: Bias tensor for RMS normalization with shape [in_features].
        eps: A small constant for numerical stability in normalization.

    Returns:
        Output tensor with shape [n, out_features].
    """
    return layer_norm_linear_quant_fn(
        x,
        norm_weight,
        norm_bias,
        weight,
        bias,
        is_rms_norm=True
    )


class BitLinear(nn.Linear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        norm_eps: float = 1e-8
    ):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        # Initialize the superclass nn.Linear with the given parameters
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)

        self.norm = RMSNorm(in_features, eps=norm_eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().extra_repr()}, norm_eps={self.norm.eps})"

    def forward(self, x):
        """
        Overrides the forward pass to include quantization.

        Args:
            x: An input tensor with shape [n, d].

        Returns:
            An output tensor with shape [n, d].
        """
        # Weight tensor
        w = self.weight

        # Apply RMS normalization to the input
        x_norm = self.norm(x)

        # Apply quantization to both activations and weights
        # Uses Straight-Through Estimator (STE) trick with .detach() for gradient flow
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        # Perform linear operation with quantized values
        y = F.linear(x_quant, w_quant)

        return y


class FusedBitLinear(BitLinear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(self, in_features, out_features, bias=False):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        # Initialize the superclass nn.Linear with the given parameters
        super(FusedBitLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return layer_norm_linear_quant_fn(
            x,
            self.norm.weight,
            self.norm.bias,
            self.weight,
            self.bias,
            is_rms_norm=True
        )

'''

BENCHMARK_SRC = r'''
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import triton

from fla.modules import GroupNorm
from fla.modules.fused_bitlinear import FusedBitLinear


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg``
        line_vals=['naive_ln',  'fused_ln', 'naive_gn', 'fused_gn',
                   'naive_ln_bwd',  'fused_ln_bwd', 'naive_gn_bwd', 'fused_gn_bwd'],
        # label name for the lines
        line_names=['naive_ln',  'fused_ln', 'naive_gn', 'fused_gn',
                    'naive_ln_bwd',  'fused_ln_bwd', 'naive_gn_bwd', 'fused_gn_bwd'],
        # line styles
        styles=[('green', '-'), ('blue', '--'), ('red', '-.'),
                ('cyan', ':'), ('yellow', 'dotted'), ('cyan', '--'), ('cyan', '-'), ('black', ':')],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    from fla.utils import device
    dtype = torch.bfloat16
    requires_grad = True
    B, D = 16, 1024

    x = torch.randn(B * T, D, device=device, requires_grad=requires_grad, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    results = 0, 0, 0
    if provider.startswith('naive_ln'):
        norm = nn.LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('fused_ln'):
        # swap fused LN reference with quantized LayerNorm+linear block
        norm = FusedBitLinear(D, D, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('naive_gn'):
        norm = nn.GroupNorm(4, D).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('fused_gn'):
        norm = GroupNorm(4, D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x), quantiles=quantiles)
    if provider.startswith('naive_ln_bwd'):
        norm = nn.LayerNorm(D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    if provider.startswith('fused_ln_bwd'):
        # swap fused LN reference with quantized LayerNorm+linear block
        norm = FusedBitLinear(D, D, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    if provider.startswith('naive_gn_bwd'):
        norm = nn.GroupNorm(4, D).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    if provider.startswith('fused_gn_bwd'):
        norm = GroupNorm(4, D, elementwise_affine=True, bias=True).to(device=device, dtype=dtype)
        results = triton.testing.do_bench(lambda: norm(x).backward(x), quantiles=quantiles)
    return results


if __name__ == '__main__':
    benchmark.run(print_data=True)

'''


def _load_embedded_module(name: str, source: str, package: str | None = None):
    module = types.ModuleType(name)
    embed_path = Path(__file__).with_name(f"__embedded_{name.replace('.', '_')}.py")
    embed_path.write_text(source)
    module.__file__ = str(embed_path)
    module.__package__ = package or name.rpartition('.')[0]
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    sys.modules[name] = module
    return module

fla_pkg = types.ModuleType("fla")
fla_pkg.__path__ = []
sys.modules["fla"] = fla_pkg

fla_modules_pkg = types.ModuleType("fla.modules")
fla_modules_pkg.__path__ = []
sys.modules["fla.modules"] = fla_modules_pkg
setattr(fla_pkg, "modules", fla_modules_pkg)

fla_utils_mod = _load_embedded_module("fla.utils", FLA_UTILS_SRC, package="fla")
setattr(fla_pkg, "utils", fla_utils_mod)

layernorm_mod = _load_embedded_module("fla.modules.layernorm", FLA_MODULES_LAYERNORM_SRC, package="fla.modules")
setattr(fla_modules_pkg, "layernorm", layernorm_mod)
setattr(fla_modules_pkg, "GroupNorm", getattr(layernorm_mod, "GroupNorm"))
setattr(fla_modules_pkg, "LayerNorm", getattr(layernorm_mod, "LayerNorm"))

fused_bitlinear_mod = _load_embedded_module("fla.modules.fused_bitlinear", FLA_MODULES_FUSED_BITLINEAR_SRC, package="fla.modules")
setattr(fla_modules_pkg, "fused_bitlinear", fused_bitlinear_mod)

exec(compile(BENCHMARK_SRC, "<embedded benchmark_layernorm.py>", "exec"), globals())
