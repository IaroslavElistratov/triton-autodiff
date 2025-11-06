# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


"""
Performance:
        T      rwkv6   rwkv6_bwd
0   128.0   0.819456    5.646624
1   256.0   1.781760   11.546368
2   512.0   3.484320   23.400064
3  1024.0   6.964976   47.019312
4  2048.0  13.904192   94.287552
5  4096.0  27.865152  189.171875
6  8192.0  56.230560  380.042114
"""

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
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice
import triton.testing
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

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float('-inf')))


if not is_gather_supported:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather


if hasattr(triton.language, '_experimental_make_tensor_descriptor'):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, 'make_tensor_descriptor'):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """
    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None

@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BK', 'BV'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv6_fwd_kernel(
    q,  # query [B, H, T, K]/[B, T, H, K]
    k,  # key [B, H, T, K]/[B, T, H, K]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H] or None
    u,  # bonus [B, H, K]
    o,  # output [NK, B, H, T, V]/[NK, B, T, H, V]
    h0,  # initial hidden state [B, H, K, V]
    ht,  # final hidden state [B, H, K, V]
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,  # whether to reverse the recurrence
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_w = w + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_o = o + ((i_k * all + bos) + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_kv = b_k[:, None] * b_v[None, :]
        b_o = tl.sum((b_h + b_kv * b_u[:, None]) * b_q[:, None], 0)
        b_h = b_h * exp(b_w)[:, None] + b_kv
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * H*K
        p_k += (-1 if REVERSE else 1) * H*K
        p_v += (-1 if REVERSE else 1) * H*V
        p_w += (-1 if REVERSE else 1) * H*K
        p_o += (-1 if REVERSE else 1) * H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['BK', 'BV'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv6_bwd_kernel_dq(
    k,  # key [B, H, T, V]/[B, T, H, V]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H]
    u,  # bonus [B, H, K]
    do,  # gradient of output [B, H, T, V]/[B, T, H, V]
    dq,  # gradient of query [NV, B, H, T, K]/[NV, B, T, H, K]
    dq1,  # gradient of query_aux [NV, B, H, T, K]/[NV, B, T, H, K]
    h0,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_w = w + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_do = do + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_dq = dq + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_dq1 = dq1 + ((i_v * all + bos) + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_kv = b_k[:, None] * b_v[None, :]

        b_hq = b_h * b_do[None, :]
        b_dq = tl.sum(b_hq + b_kv * b_u[:, None] * b_do[None, :], 1) * scale
        b_dq1 = tl.sum(b_hq, 1)
        b_h = b_h * exp(b_w)[:, None]
        b_h += b_kv
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=mask_k)
        tl.store(p_dq1, b_dq1.to(p_dq1.dtype.element_ty), mask=mask_k)

        p_k += (-1 if REVERSE else 1) * H*K
        p_v += (-1 if REVERSE else 1) * H*V
        p_w += (-1 if REVERSE else 1) * H*K
        p_do += (-1 if REVERSE else 1) * H*V
        p_dq += (-1 if REVERSE else 1) * H*K
        p_dq1 += (-1 if REVERSE else 1) * H*K


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
    ],
    key=['BK', 'BV'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv6_bwd_kernel_dkv(
    q,  # query [B, H, T, K]/[B, T, H, K]
    k,  # key [B, H, T, V]/[B, T, H, V]
    v,  # value [B, H, T, V]/[B, T, H, V]
    w,  # log gate [B, H, T]/[B, T, H]
    u,  # bonus [B, H, K]
    do,  # gradient of output [B, H, T, V]/[B, T, H, V]
    dk,  # gradient of key [NV, B, H, T, K]/[NK, B, T, H, K]
    dk1,  # gradient of key_aux [NV, B, H, T, K]/[NK, B, T, H, K]
    dv,  # gradient of value [NK, B, H, T, V]/[NV, B, T, H, V]
    dh0,  # gradient of initial hidden state [N, H, K, V]
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_k, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    p_w = w + (bos + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_do = do + (bos + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    p_dk = dk + ((i_v * all + bos) + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_dk1 = dk1 + ((i_v * all + bos) + ((T-1) if not REVERSE else 0)) * H*K + i_h * K + o_k
    p_dv = dv + ((i_k * all + bos) + ((T-1) if not REVERSE else 0)) * H*V + i_h * V + o_v
    p_u = u + i_h * K + o_k

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_u = tl.load(p_u, mask=mask_k, other=0).to(tl.float32)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T - 1, -1, -1):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_v, other=0).to(tl.float32)
        b_dkv = b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], 1)
        tl.store(p_dk1, b_dk.to(p_dk1.dtype.element_ty), mask=mask_k)
        b_dk += tl.sum(b_dkv * b_u[:, None] * b_v[None, :], 1)
        b_dv = tl.sum((b_dh + (b_dkv * b_u[:, None])) * b_k[:, None], 0)

        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=mask_k)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=mask_v)
        b_dh *= exp(b_w)[:, None]
        b_dh += b_dkv

        p_q += (-1 if not REVERSE else 1) * H*K
        p_k += (-1 if not REVERSE else 1) * H*K
        p_v += (-1 if not REVERSE else 1) * H*V
        p_w += (-1 if not REVERSE else 1) * H*K
        p_do += (-1 if not REVERSE else 1) * H*V
        p_dk += (-1 if not REVERSE else 1) * H*K
        p_dk1 += (-1 if not REVERSE else 1) * H*K
        p_dv += (-1 if not REVERSE else 1) * H*V

    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), mask=mask_h)


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT, 'BK': BK}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
    ],
    key=['K'],
    **autotune_cache_kwargs
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv6_bwd_kernel_dw(
    q,
    k,
    dq,
    dk,
    dw,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    REVERSE: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    i_k, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos
    NT = tl.cdiv(T, BT)

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.) if not REVERSE else tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BK], dtype=tl.float32)

    i_t = 0 if not REVERSE else NT - 1
    for _ in range(NT):
        p_q = tl.make_block_ptr(q + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T-1, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + 1, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T-1, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
        b_dq = tl.load(p_dq, boundary_check=(0, 1)).to(tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dw = (b_q * b_dq * scale) - b_k * b_dk
        b_c = b_z[None, :] + tl.dot(m_i, b_dw, allow_tf32=False)
        tl.store(p_dw, b_c.to(p_dw.dtype.element_ty), boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_dw, 0)

        i_t += (1 if not REVERSE else -1)


def fused_recurrent_rwkv6_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    h0 = initial_state
    ht = q.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    o = q.new_empty(NK, *v.shape, dtype=torch.float)

    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_fwd_kernel[grid](
        q,
        k,
        v,
        w,
        u,
        o,
        h0,
        ht,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
    )
    o = o.sum(0)
    return o, ht


def fused_recurrent_rwkv6_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    do: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = min(triton.next_power_of_2(K), 16), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dq = q.new_empty(NV, *q.shape, dtype=torch.float)
    dq1 = torch.empty_like(dq)

    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_bwd_kernel_dq[grid](
        k,
        v,
        w,
        u,
        do,
        dq,
        dq1,
        initial_state,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
    )
    dq = dq.sum(0)
    dq1 = dq1.sum(0)

    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)

    dk = q.new_empty(NV, *k.shape, dtype=torch.float)
    dk1 = q.new_empty(NV, *k.shape, dtype=torch.float)
    dv = q.new_empty(NK, *v.shape, dtype=torch.float)

    dh0 = torch.empty_like(initial_state) if initial_state is not None else None
    grid = (NV, NK, N * H)
    fused_recurrent_rwkv6_bwd_kernel_dkv[grid](
        q,
        k,
        v,
        w,
        u,
        do,
        dk,
        dk1,
        dv,
        dh0,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REVERSE=reverse,
    )
    dk = dk.sum(0)
    dk1 = dk1.sum(0)
    dv = dv.sum(0)

    dw = torch.empty_like(w)
    def grid(meta): return (triton.cdiv(meta['K'], meta['BK']), N * H)
    fused_recurrent_rwkv6_bwd_kernel_dw[grid](
        q,
        k,
        dq1,
        dk1,
        dw,
        cu_seqlens,
        scale,
        T=T,
        H=H,
        K=K,
        REVERSE=not reverse,
    )
    du = (do.float() * v).sum(-1, True, dtype=torch.float) * q * k * scale
    du = du.sum((0, 1))
    return dq, dk, dv, dw, du, dh0


class FusedRecurrentRWKV6Function(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        reverse: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        o, ht = fused_recurrent_rwkv6_fwd(
            q=q,
            k=k,
            v=v,
            w=w,
            u=u,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        ctx.save_for_backward(q, k, v, w, u, initial_state)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        return o.to(v), ht

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, w, u, initial_state = ctx.saved_tensors

        dq, dk, dv, dw, du, dh0 = fused_recurrent_rwkv6_bwd(
            q=q,
            k=k,
            v=v,
            w=w,
            u=u,
            do=do,
            scale=ctx.scale,
            initial_state=initial_state,
            reverse=ctx.reverse,
            cu_seqlens=ctx.cu_seqlens,
        )
        dh0 = dh0.to(initial_state) if dh0 is not None else dh0
        return dq.to(q), dk.to(k), dv.to(v), dw.to(w), du.to(u), None, dh0, None, None, None


def fused_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: Optional[int] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        r (torch.Tensor):
            reception of shape `[B, T, H, K]`.
            Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        w (torch.Tensor):
            data-dependent decays of shape `[B, T, H, K]`. in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `[H, K]`
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (Optional[torch.Tensor]):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.rwkv6 import fused_recurrent_rwkv6
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> u = torch.randn(H, K, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = fused_recurrent_rwkv6(
            q, k, v, g, u,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_rwkv6(
            q, k, v, g, u,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
        >>> assert o.allclose(o_var.view(o.shape))
        >>> assert ht.allclose(ht_var)
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and r.shape[1] < r.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({r.shape[1]}) < num_heads ({r.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {r.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = FusedRecurrentRWKV6Function.apply(
        r,
        k,
        v,
        w,
        u,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
    return o, final_state


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        # comment: changed otherwise OOM
        # x_vals=[128 * 2 ** i for i in range(0, 8)],
        x_vals=[128 * 2 ** i for i in range(0, 7)],
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg`
        line_vals=['rwkv6', 'rwkv6_bwd'],
        # label name for the lines
        line_names=['rwkv6', 'rwkv6_bwd'],
        # # line styles
        styles=[
            ('green', '-'),      # rwkv6
            ('cyan', ':'),       # rwkv6_bwd
        ],
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Performance",
        args={},
    )
)
def benchmark(T, provider):
    dtype = torch.bfloat16
    requires_grad = True
    # Read B, H, D from environment variables, default to 16, 8, 128 if not set
    B = int(os.getenv('BENCH_B', '8'))  # Batch size
    H = int(os.getenv('BENCH_H', '64'))   # Number of heads
    D = int(os.getenv('BENCH_D', '64'))  # Dimension per head
    with torch.no_grad():
        q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        if provider.startswith('rwkv6'):
            w = F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=dtype)).requires_grad_(True)
            u = torch.randn(H, D, device=device, dtype=dtype).requires_grad_(True)

    do = torch.ones_like(v, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'rwkv6':
        results = triton.testing.do_bench(lambda: fused_recurrent_rwkv6(q, k, v, w, u), quantiles=quantiles)
    elif provider == 'rwkv6_bwd':
        results = triton.testing.do_bench(lambda: fused_recurrent_rwkv6(q, k, v, w, u)[0].backward(do), quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider {provider}")
    return results


def main():
    if not torch.cuda.is_available():
        print("skip RWKV6 standalone benchmark: CUDA device is not available")
        return
    benchmark.run(print_data=True)


if __name__ == '__main__':
    main()
