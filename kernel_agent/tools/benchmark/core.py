from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import triton.testing as tt

Tensor = torch.Tensor
MaybeTensors = Union[Tensor, Sequence[Tensor]]
OutputSel = Union[str, int, Sequence[int], Callable[[MaybeTensors], MaybeTensors]]

def _as_tuple(x: MaybeTensors) -> Tuple[Tensor, ...]:
    return (x,) if isinstance(x, torch.Tensor) else tuple(x)

def _select_outputs(y: MaybeTensors, sel: OutputSel) -> Tuple[Tensor, ...]:
    yt = _as_tuple(y)
    if sel == "auto":
        return yt
    if isinstance(sel, int):
        return (yt[sel],)
    if callable(sel):
        return _as_tuple(sel(y))
    return tuple(yt[i] for i in sel)

@dataclass(frozen=True)
class BenchRecord:
    dims: Dict[str, Any]
    time_ms: float
    tflops: Optional[float] = None

def bench_op(
    my_op: Callable[..., MaybeTensors],
    sidecar: Dict[str, Any],
    *,
    mode: str = "bwd",                  # "fwd" | "bwd"
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    outputs: OutputSel = "auto",        # which tensors to drive backward through
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Benchmark an autograd-backed op.

    sidecar must expose:
      - SWEEP: Iterable[dict]
      - make_args(dims, device, dtype) -> (args, kwargs)
      - optional flops(dims, mode) -> int|float for TFLOPS reporting

    - Returns a per-sweep, per-shape summary via _reduce_bench (not raw per-run timings),
      which avoids mixing heterogeneous shapes into a single median/min.
    """
    if mode not in ("fwd", "bwd"):
        raise ValueError("mode must be 'fwd' or 'bwd'")

    results: List[BenchRecord] = []

    if not isinstance(sidecar, dict):
        raise TypeError("bench_op expects sidecar as a dict namespace")

    sweep = list(sidecar.get("SWEEP", [{}]))
    make_args_fn = sidecar.get("make_args")
    if not callable(make_args_fn):
        raise RuntimeError("sidecar['make_args'] must be callable")

    def _ensure_requires_grad(x: Any) -> Any:
        # try:
        if isinstance(x, torch.Tensor):
            x.requires_grad_(True)
            return x
        # Shallow map for simple sequences of tensors
        if isinstance(x, (list, tuple)):
            seq = [(_ensure_requires_grad(t)) for t in x]
            return type(x)(seq)
        # except Exception:
        #     pass
        return x

    for dims in sweep:
        print(f"[benchmark] running with {dims}")
        args, kwargs = make_args_fn(dims, device=device, dtype=dtype)
        # Ensure autograd is enabled on inputs so outputs require grad
        # try:
        args = tuple(_ensure_requires_grad(t) for t in args)
        # except Exception:
        #     args = (_ensure_requires_grad(args),)
        if isinstance(kwargs, dict):
            for k, v in list(kwargs.items()):
                kwargs[k] = _ensure_requires_grad(v)

        # Warm compile outside the timed region
        y_warm = my_op(*args, **kwargs)

        if mode == "fwd":
            def run():
                _ = my_op(*args, **kwargs)
        else:
            ys = _select_outputs(y_warm, outputs)
            torch.manual_seed(seed)
            ups = tuple(torch.randn_like(t) for t in ys)

            def run():
                torch.autograd.backward(ys, ups, retain_graph=True)

        ms = float(tt.do_bench(run))
        flops_fn = sidecar.get("flops")
        flop = float(flops_fn(dims, mode)) if callable(flops_fn) else None
        tflops = flop * 1e-12 / (ms * 1e-3) if flop is not None else None
        results.append(BenchRecord(dims=dict(dims), time_ms=ms, tflops=tflops))

    # Collapse raw per-run samples into per-shape metrics **for this sweep only**.
    # Cross-iteration acceptance (geomean speedup vs best, patience) is handled in PerfTracker.
    return _reduce_bench(results)


def _dims_key(d: Dict[str, Any]) -> str:
    """Stable 'k=v' key for a dims dict to identify a shape within a sweep.
    Stable string key like "B=1|H=8|N_CTX=4096".
    """
    return "|".join(f"{k}={d[k]}" for k in sorted(d or {}))

def _reduce_bench(records: Sequence[BenchRecord]) -> Dict[str, Any]:
    """
    Per-shape reducer for ONE sweep. Condense multiple raw samples per
    shape (e.g., duplicated shapes in the sweep) into per-shape metrics.

    Notes
    -----
    - Groups BenchRecords by shape key (built from dims).
    - For each shape:
        - Aggregates all timings seen in this sweep for that shape.
        - Computes median_ms and best_ms.
        - Records per-shape max_tflops if available.
    - For the whole sweep: computes peak_tflops = max(per‑shape max_tflops) and returns the original sweep order.

    No cross‑shape medians here (handles within-sweep aggregation only).
    Cross‑iteration decisions live in PerfTracker.

    Returns
    -------
    dict with:
      per_shape: {shape_key: {"dims", "median_ms", "best_ms", ["max_tflops"]}}
          Per‑shape latency (median and best) computed within this sweep only.
      order: [shape_key, ...]
          Encounter order for display/selection.
      peak_tflops: float|None
          Peak TFLOPS across shapes in this sweep (informational).
    """
    import statistics as st

    if not records:
        return {"per_shape": {}, "order": [], "peak_tflops": None}

    groups: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    # Group timings by shape
    for r in records:
        k = _dims_key(r.dims)
        if k not in groups:
            groups[k] = {"dims": dict(r.dims), "times": [], "tflops": []}
            order.append(k)
        groups[k]["times"].append(float(r.time_ms))
        if r.tflops is not None:
            groups[k]["tflops"].append(float(r.tflops))

    # Compute per‑shape medians, bests, and local TFLOPS peaks
    per_shape: Dict[str, Dict[str, Any]] = {}
    peak: Optional[float] = None
    for k, g in groups.items():
        times = g["times"]
        med = float(st.median(times))
        bst = float(min(times))
        entry: Dict[str, Any] = {"dims": g["dims"], "median_ms": med, "best_ms": bst}
        if g["tflops"]:
            mx = float(max(g["tflops"]))
            entry["max_tflops"] = mx
            peak = mx if peak is None else max(peak, mx)
        per_shape[k] = entry

    return {"per_shape": per_shape, "order": order, "peak_tflops": peak}

