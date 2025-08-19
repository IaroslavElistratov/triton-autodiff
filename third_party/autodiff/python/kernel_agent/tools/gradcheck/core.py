from __future__ import annotations
import math, random
from typing import Callable, Sequence, Tuple, Union, Optional, List
import torch

Tensor = torch.Tensor
TensorOrTensors = Union[Tensor, Sequence[Tensor]]

def _as_tuple(x: TensorOrTensors) -> Tuple[Tensor, ...]:
    return (x,) if isinstance(x, Tensor) else tuple(x)

def _dot64(a: Tensor, b: Tensor) -> float:
    return float((a.reshape(-1).to(torch.float64) * b.reshape(-1).to(torch.float64)).sum().item())

def _phi(forward_fn, inputs: Tuple[Tensor, ...], upstream: Tuple[Tensor, ...]) -> float:
    with torch.no_grad():
        outs = _as_tuple(forward_fn(*inputs))
    return sum(_dot64(y, g) for y, g in zip(outs, upstream))

def _choose_eps(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16): return 2e-2
    if dtype == torch.float64: return 1e-5
    return 1e-3  # fp32 default


# @torch.no_grad()
def gradient_check(
    forward_fn: Callable[..., TensorOrTensors],
    backward_fn: Callable[..., Tuple[Optional[Tensor], ...]],  # returns per-input grads
    inputs: Sequence[Tensor],
    *,
    mode: str = "coord",            # "coord" or "directional"
    num_checks: int = 32,           # per-input samples (coord) or number of directions (directional)
    eps: Optional[float] = None,
    rtol: float = 2e-2,
    atol: float = 1e-3,
    seed: int = 0,
    only_inputs: Optional[Sequence[int]] = None,
) -> Tuple[bool, dict]:
    """
    Black-box central-difference gradient check for Triton kernels.
    Scalarizes via phi(x) = sum_j <f_j(x), g_j>.
    - mode="coord": sample coordinates per floating input and compare partials.
    - mode="directional": sample random directions and compare directional derivs
    """
    rng = random.Random(seed)
    inputs = tuple(t.detach().clone() for t in inputs)

    # Forward once to infer outputs and dtype
    outs0 = _as_tuple(forward_fn(*inputs))
    assert len(outs0) >= 1, "forward_fn returned no outputs"
    base_dtype = outs0[0].dtype
    if eps is None:
        eps = _choose_eps(base_dtype)

    # Upstream (same arity as outputs)
    torch.manual_seed(seed)
    upstream = tuple(torch.randn_like(y, dtype=y.dtype) for y in outs0)

    # Analytic grads at base point (single call)
    ana_grads = backward_fn(*inputs, upstream)  # assumes (inputs..., upstream)
    if not isinstance(ana_grads, (tuple, list)) or len(ana_grads) != len(inputs):
        raise TypeError("backward_fn must return a tuple of grads matching inputs")

    # Select floating inputs
    included = [i for i, t in enumerate(inputs) if t.is_floating_point()]
    if only_inputs is not None:
        included = [i for i in only_inputs if inputs[i].is_floating_point()]
    if not included:
        raise ValueError("No floating-point inputs selected")

    total = 0
    failed = 0
    max_abs = 0.0
    max_rel = 0.0

    if mode == "coord":
        # For each floating input tensor, sample coordinates and check partials
        for i in included:
            x = inputs[i]
            g = ana_grads[i]; assert g is not None
            grad_flat = g.reshape(-1).to(torch.float64)
            N = x.numel()
            if N == 0: continue
            idxs = list(range(N)) if num_checks >= N else rng.sample(range(N), num_checks)
            for k in idxs:
                x_plus = list(inputs); x_minus = list(inputs)
                xp = x.clone(); xm = x.clone()
                xp.view(-1)[k] += eps; xm.view(-1)[k] -= eps
                x_plus[i] = xp; x_minus[i] = xm
                num = (_phi(forward_fn, tuple(x_plus), upstream) -
                       _phi(forward_fn, tuple(x_minus), upstream)) / (2.0 * eps)
                ana = float(grad_flat[k].item())
                err = abs(num - ana)
                rel = err / max(1.0, abs(num), abs(ana))
                total += 1
                failed += int(err > (atol + rtol * max(abs(num), abs(ana))))
                max_abs = max(max_abs, err)
                max_rel = max(max_rel, rel)

    elif mode == "directional":
        # Sample random directions; global L2 normalize across included inputs
        for _ in range(num_checks):
            r: List[Optional[Tensor]] = [None] * len(inputs)
            flat64 = []
            for i in included:
                ri = torch.randn_like(inputs[i], dtype=torch.float32).to(inputs[i].dtype)
                r[i] = ri
                flat64.append(ri.reshape(-1).to(torch.float64))
            norm = float(torch.linalg.vector_norm(torch.cat(flat64)).item())
            for i in included:
                r[i] = (r[i] / norm).to(inputs[i].dtype)

            x_plus = list(inputs); x_minus = list(inputs)
            for i in included:
                x_plus[i]  = inputs[i] + eps * r[i]
                x_minus[i] = inputs[i] - eps * r[i]
            num = (_phi(forward_fn, tuple(x_plus), upstream) -
                   _phi(forward_fn, tuple(x_minus), upstream)) / (2.0 * eps)
            ana = 0.0
            for i in included:
                gi = ana_grads[i]
                if gi is not None:
                    ana += _dot64(r[i], gi)
            err = abs(num - ana)
            rel = err / max(1.0, abs(num), abs(ana))
            total += 1
            failed += int(err > (atol + rtol * max(abs(num), abs(ana))))
            max_abs = max(max_abs, err)
            max_rel = max(max_rel, rel)
    else:
        raise ValueError("mode must be 'coord' or 'directional'")

    ok = total > 0 and failed == 0
    stats = dict(ok=ok, total=total, failed=failed, max_abs_err=max_abs,
                 max_rel_err=max_rel, eps=eps, rtol=rtol, atol=atol, mode=mode)
    return ok, stats
