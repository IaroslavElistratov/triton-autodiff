# SPDX-License-Identifier: Apache-2.0
# Modified: extracted and consolidated transitive functions; adjusted imports/names/formatting.
# Source-Repo: https://github.com/meta-recsys/generative-recommenders
# Source-Files: generative_recommenders/ops/triton/triton_addmm.py
# See: THIRD_PARTY_LICENSES.md (license text + NOTICE)
#
# Extracted autograd.Function from /var/folders/wf/5ynhbbrn49z46nwvn4vy2pkw0000gn/T/TRITON_EXTRACT_rzwbk0zd/generative-recommenders-main/generative_recommenders/ops/triton/triton_addmm.py
import torch
import torch.nn.functional as F
from torch.autograd import Function

import triton
import triton.language as tl

# ============================================================
# FORWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from forward() method

@triton_cc(annotations={'M': 'i32', 'N': ('i32', 16), 'K': ('i32', 16),
    'stride_xm': ('i32', 16), 'stride_xk': ('i32', 1), 'stride_wk': ('i32',
    16), 'stride_wn': ('i32', 1), 'stride_ym': ('i32', 16), 'stride_yn': (
    'i32', 1), 'stride_zm': ('i32', 16), 'stride_zn': ('i32', 1)})
@triton_autotune(configs=get_mm_configs(), key=['N', 'K'])
@triton.jit
def _addmm_fwd(x_ptr, w_ptr, y_ptr, z_ptr, M, N, K, stride_xm, stride_xk,
    stride_wk, stride_wn, stride_ym, stride_yn, stride_zm, stride_zn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, ALLOW_TF32: tl.constexpr, BROADCAST_Y: tl.constexpr
    ):
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        )
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[
            None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = (accumulator + y.to(tl.float32)).to(z_ptr.dtype.element_ty)
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


@triton_autotune(configs=get_mm_configs(pre_hook=
    _addmm_tma_set_block_size_hook), key=['N', 'K', 'WARP_SPECIALIZE'])
@triton.jit
def _addmm_fwd_tma_persistent(x_desc, w_desc, y_desc, z_desc, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, ALLOW_TF32: tl.constexpr, BROADCAST_Y: tl.
    constexpr, WARP_SPECIALIZE: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True,
        warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m,
            GROUP_M, NUM_SMS)
        offs_xm = pid_m * BLOCK_M
        offs_wn = pid_n * BLOCK_N
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
            offs_k = k * BLOCK_K
            x = x_desc.load([offs_xm, offs_k])
            w = w_desc.load([offs_k, offs_wn])
            accumulator = tl.dot(x, w, accumulator, allow_tf32=ALLOW_TF32)
        if BROADCAST_Y:
            y = y_desc.load([0, offs_wn])
        else:
            y = y_desc.load([offs_xm, offs_wn])
        z = (accumulator + y.to(tl.float32)).to(z_desc.dtype)
        z_desc.store([offs_xm, offs_wn], z)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + tile_id % group_size_m
    pid_n = tile_id % num_pid_in_group // group_size_m
    return pid_m, pid_n


def _check_tma_alignment(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor,
    min_alignment: int=16) ->bool:
    """Check if tensors meet TMA alignment requirements.

    TMA (Tensor Memory Accelerator) on H100 requires:
    1. Base addresses to be 64-byte aligned
    2. Dimensions to be multiples of 64 for optimal performance
    3. Contiguous inner dimensions (stride=1)

    Args:
        x: Input tensor [M, K]
        w: Weight tensor [K, N]
        y: Bias tensor [N] or [M, N]
        min_alignment: Minimum alignment requirement (default: 64)

    Returns:
        True if all tensors meet TMA alignment requirements
    """
    _, K = x.shape
    KB, N = w.shape
    assert K == KB, f'incompatible dimensions {K}, {KB}'
    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f'incompatible dimensions {N}, {NY}'
    return K % min_alignment == 0 and N % min_alignment == 0


@torch.fx.wrap
def triton_addmm_fwd(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
    ) ->torch.Tensor:
    M, K = x.shape
    KB, N = w.shape
    assert K == KB, f'incompatible dimensions {K}, {KB}'
    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f'incompatible dimensions {N}, {NY}'
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N,
        meta['BLOCK_N']))
    _addmm_fwd[grid](x, w, y, z, M, N, K, x.stride(0), x.stride(1), w.
        stride(0), w.stride(1), y.stride(0) if not is_y_1d else 0, y.stride
        (1) if not is_y_1d else y.stride(0), z.stride(0), z.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, BROADCAST_Y=is_y_1d)
    return z


@torch.fx.wrap
def triton_addmm_fwd_tma_persistent(x: torch.Tensor, w: torch.Tensor, y:
    torch.Tensor, warp_specialize: bool=False) ->torch.Tensor:
    M, K = x.shape
    _, N = w.shape
    is_y_1d = y.dim() == 1
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z
    dummy_block = [1, 1]
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)
    NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count

    def grid(meta):
        nonlocal x_desc, w_desc, z_desc
        BLOCK_M = meta['BLOCK_M']
        BLOCK_N = meta['BLOCK_N']
        return min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),
    _addmm_fwd_tma_persistent[grid](x_desc, w_desc, y_desc, z_desc, M, N, K,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32, BROADCAST_Y=
        is_y_1d, WARP_SPECIALIZE=warp_specialize, NUM_SMS=NUM_SMS)
    return z


def is_sm100() ->bool:
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and props.minor == 0


# Forward method (kernel launch code)
def __AddMmFunction_forward(ctx, x: torch.Tensor, w: torch.Tensor, y: torch
    .Tensor) ->torch.Tensor:
    ctx.save_for_backward(x, w)
    ctx.is_y_1d = y.dim() == 1
    if is_sm100() and TMA_AVAILABLE and _check_tma_alignment(x, w, y):
        return triton_addmm_fwd_tma_persistent(x, w, y, warp_specialize=True)
    else:
        return triton_addmm_fwd(x, w, y)


# ============================================================
# BACKWARD Triton Kernels
# ============================================================

# Kernels called (directly or transitively) from backward() method

def triton_addmm_bwd(x: torch.Tensor, w: torch.Tensor, dz: torch.Tensor,
    is_y_1d: bool) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_y_1d:
        dy = torch.sum(dz, dim=0)
    else:
        dy = dz
    dw = torch.mm(x.t(), dz)
    dx = torch.mm(dz, w.t())
    return dx, dw, dy


# Backward method (kernel launch code)
def __AddMmFunction_backward(ctx, dz: torch.Tensor) ->Tuple[torch.Tensor,
    torch.Tensor, torch.Tensor]:
    x, w = ctx.saved_tensors
    return triton_addmm_bwd(x, w, dz, ctx.is_y_1d)


# ============================================================
# autograd.Function Class Definition
# ============================================================

class _AddMmFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
        ) ->torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.is_y_1d = y.dim() == 1
        if is_sm100() and TMA_AVAILABLE and _check_tma_alignment(x, w, y):
            return triton_addmm_fwd_tma_persistent(x, w, y, warp_specialize
                =True)
        else:
            return triton_addmm_fwd(x, w, y)

    @staticmethod
    def backward(ctx, dz: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor,
        torch.Tensor]:
        x, w = ctx.saved_tensors
        return triton_addmm_bwd(x, w, dz, ctx.is_y_1d)
