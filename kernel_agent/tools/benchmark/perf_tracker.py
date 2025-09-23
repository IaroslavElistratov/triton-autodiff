from __future__ import annotations
from typing import Any, Dict, List, Optional
import math

def _dims_key(d: Dict[str, Any]) -> str:
    """Return a stable key like 'B=1|H=8|N_CTX=4096' for a dims dict.
    Identifies shapes consistently across sweeps/iterations so can compare
    similar shapes when computing speedups.
    """
    return "|".join(f"{k}={d[k]}" for k in sorted(d or {}))

def _as_shape_map(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize reducer output to {shape_key: {dims, median_ms, best_ms, max_tflops}}.
    Allows direct key-based matching against the accepted best_by_shape without
    scanning lists, simpler lookups for per-shape ratios.
    """
    per = summary.get("per_shape") or {}
    if isinstance(per, dict):
        return per
    # Backward‑compat: if reducer returns a list, rekey it.
    out: Dict[str, Dict[str, Any]] = {}
    for e in per:
        out[_dims_key(e.get("dims", {}))] = e
    return out

def _geomean(values: List[float]) -> Optional[float]:
    """Geometric mean for ratios. Returns None if empty or invalid."""
    xs = [float(x) for x in values if x and x > 0.0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None

class PerfTracker:
    """
    Minimal performance policy and state.

    Tracks
    ------
    latest_metrics: dict
        Last sweep summary from the reducer (per_shape + peak TFLOPS),
        augmented with per‑shape speedups vs best and the geomean.
    best_by_shape: dict
        Current accepted baseline per shape (seeded on first full‑parity run)
    best_metrics: dict|None
        Snapshot of the accepted best: {by_shape, order, ever_max_tflops, accepted_at_iter, geomean_speedup_vs_prev_best}.
    ever_max_tflops: float|None
        Monotone high‑water TFLOPS across all iterations.

    Gate
    ----
    - First full‑parity sweep: accept and seed best_by_shape.
    - Otherwise: accept if geomean_speedup_vs_best >= (1 + min_rel_improvement) under full parity.

    Patience
    --------
    Count consecutive non‑improving full‑parity iterations. Expose stop_reason="patience" when threshold is hit.
    """

    def __init__(self, *, min_rel_improvement: float, patience_perf_stop: int) -> None:
        self.min_rel_improvement = float(min_rel_improvement)
        self.patience_perf_stop = int(patience_perf_stop)
        self.best_by_shape: Dict[str, Dict[str, Any]] = {}
        self.best_metrics: Optional[Dict[str, Any]] = None
        self.latest_metrics: Optional[Dict[str, Any]] = None
        self.ever_max_tflops: Optional[float] = None
        self._non_improve = 0
        self.stop_reason: Optional[str] = None

    @staticmethod
    def _full_parity(stats: Dict[str, Any]) -> bool:
        """Return True iff all shapes in the sweep passed gradcheck.
        Performance acceptance only after correctness. The orchestrator already
        treats parity as the first gate; this mirrors that requirement.
        """
        total = int(stats.get("num_total", 0))
        passed = int(stats.get("num_passed", 0))
        return bool(total) and (passed == total)

    def update(self, cand: Dict[str, Any], grad_stats: Dict[str, Any], *, it: int) -> Dict[str, Any]:
        """
        Consume one sweep summary (reducer output) and gradcheck stats; update
        latest/best state and return a compact decision.

        Inputs
        ------
        - cand: reducer output for the current sweep (per_shape, order, peak_tflops).
        - grad_stats: gradcheck totals (num_passed, num_total).
        - it: iteration index for bookkeeping in accepted best.

        Returns
        -------
        {"full_parity": bool, "improved": bool, "geomean": float|None}
        """
        # 1) Always record the newest sweep summary (immutable copy)
        self.latest_metrics = dict(cand)

        # 2) Update TFLOPS high‑water mark (monotone max across iterations)
        pk = cand.get("peak_tflops")
        if pk is None:
            # tolerate older field name if present
            pk = cand.get("max_tflops")
        if isinstance(pk, (int, float)):
            self.ever_max_tflops = pk if self.ever_max_tflops is None else max(self.ever_max_tflops, float(pk))

        # 3) Enforce full parity before considering performance (correctness first)
        full_parity = self._full_parity(grad_stats)

        # 4) Compute per‑shape speedups vs current best and the geomean across shapes
        per_now = _as_shape_map(cand)
        order = list(cand.get("order") or per_now.keys())
        speedups: Dict[str, float] = {}
        ratios: List[float] = []
        if self.best_by_shape:
            for k in order:
                if k in per_now and k in self.best_by_shape:
                    base = float(self.best_by_shape[k]["median_ms"])
                    cur = float(per_now[k]["median_ms"])
                    if base > 0.0 and cur > 0.0:
                        r = base / cur  # >1 means faster than best
                        speedups[k] = r
                        ratios.append(r)
        S_geo = _geomean(ratios) if ratios else None

        # Attach reporting to latest snapshot for visibility in CLI
        self.latest_metrics["speedups_vs_best"] = speedups if self.best_by_shape else {}
        self.latest_metrics["geomean_speedup_vs_best"] = S_geo if self.best_by_shape else None

        # 5) Acceptance policy
        improved = False
        if full_parity:
            # seed on first full-parity
            if not self.best_by_shape and per_now:
                improved = True  # seed best on first full‑parity run
            # otherwise require geomean >= 1+min_rel_improvement
            else:
                thr = 1.0 + self.min_rel_improvement
                improved = (S_geo is not None) and (S_geo >= thr)

        if improved:
            # Promote current per‑shape results to the new best
            self.best_by_shape = {k: dict(v) for k, v in per_now.items()}
            self.best_metrics = {
                "by_shape": self.best_by_shape,
                "order": list(order),
                "ever_max_tflops": self.ever_max_tflops,
                "accepted_at_iter": it,
                "geomean_speedup_vs_prev_best": (S_geo if S_geo is not None else 1.0),
            }
            self._non_improve = 0
        else:
            # Only consider performance when parity is full; otherwise ignore perf for acceptance
            if full_parity:
                self._non_improve += 1
                if self._non_improve >= self.patience_perf_stop:
                    self.stop_reason = "patience"

        return {"full_parity": full_parity, "improved": improved, "geomean": S_geo}

    # used on rollback
    def reset_patience(self) -> None:
        """Clear the non-improvement streak and any pending patience stop."""
        self._non_improve = 0
        self.stop_reason = None
