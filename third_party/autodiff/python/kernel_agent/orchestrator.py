from __future__ import annotations
from dataclasses import dataclass
import os, re

# --- repo patch tool (patch-text only) ---------------------------------------
try:
    from tools.apply_patch import apply_patch as repo_apply_patch  # OSS repo tool
except Exception as e:
    raise RuntimeError(
        "Could not import tools.apply_patch.apply_patch from the OSS repo."
    ) from e


# --- tiny config --------------------------------------------------------------
@dataclass
class Config:
    max_iters: int = 6
    patience: int = 2
    min_rel_improvement: float = 0.02  # require >= +2% throughput to accept
    snippet_max_lines: int = 120        # bound context shown to the LLM


# --- tiny LLM patch interface -------------------------------------------------
class PatchProvider:
    """
    Protocol-like minimal interface. Provide an implementation that returns a
    single apply_patch.md patch (no prose).
    """
    def propose_patch(self, *, phase: str, device: str, target_file: str,
                      kernel_snippet: str, grad_summary: str,
                      bench_summary: str, profile_hint: str) -> str:
        raise NotImplementedError


# --- helpers (keep tiny) ------------------------------------------------------
_PASS_RX = re.compile(r"\b(pass|ok|success)\b", re.I)

def _parse_fd(text: str) -> tuple[bool, str]:
    """Heuristic FD pass/fail parser."""
    msg = (text or "").strip()
    return bool(_PASS_RX.search(msg)), msg

def _norm_bench(x) -> dict[str, float]:
    """Normalize benchmark() result to {'throughput': float} if possible."""
    if isinstance(x, (int, float)):
        return {"throughput": float(x)}
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if isinstance(v, (int, float)):
                out[k.lower()] = float(v)
        return out
    # last resort: loose parse from string
    try:
        return {"throughput": float(str(x).strip().split()[0])}
    except Exception:
        return {}

def _better(new: dict[str, float], best: dict[str, float] | None, min_rel: float) -> bool:
    """Strict improvement gate (relative throughput)."""
    if not new: return False
    if best is None or "throughput" not in best: return "throughput" in new
    if "throughput" not in new: return False
    base = best["throughput"]
    return new["throughput"] >= base * (1.0 + min_rel)

def _summ_bench(m: dict[str, float] | None) -> str:
    if not m: return "no bench yet"
    return ", ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, (int, float)))

def _read_snippet(path: str, max_lines: int) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[:max_lines])
    except Exception as e:
        return f"(snippet unavailable: {e})"


# --- main orchestrator --------------------------------------------------------
class KernelOptimizer:
    """
    Deterministic controller:
      init: get_user_forwrad() -> naive_grad()
      loop:
        - gradient_check() -> if FAIL: LLM 'fix' patch -> apply_patch -> continue
        - benchmark()
        - accept best-so-far only if >= min_rel_improvement
        - LLM 'optimize' patch -> apply_patch
    """
    def __init__(self, cfg: Config, patcher: PatchProvider):
        self.cfg = cfg
        self.patcher = patcher

    def run(self, *,
            # black-box tools you already implemented
            get_user_forwrad,
            naive_grad,
            gradient_check,
            benchmark,
            profile,
            get_user_dvice_info,
            ) -> dict:
        # ---- bootstrap (outside the loop) ----
        device = get_user_dvice_info()
        fwd_fp = get_user_forwrad()
        if not os.path.isfile(fwd_fp):
            raise FileNotFoundError(f"forward file not found: {fwd_fp}")

        bwd_fp = naive_grad(fwd_fp)
        if not os.path.isfile(bwd_fp):
            raise FileNotFoundError(f"backward file not found: {bwd_fp}")

        best_metrics: dict[str, float] | None = None
        best_path = bwd_fp
        non_improve = 0

        # ---- fixed optimization loop ----
        for it in range(self.cfg.max_iters):
            # 1) correctness gate
            ok, fd_sum = _parse_fd(gradient_check(bwd_fp))
            if not ok:
                patch = self.patcher.propose_patch(
                    phase="fix",
                    device=device,
                    target_file=bwd_fp,
                    kernel_snippet=_read_snippet(bwd_fp, self.cfg.snippet_max_lines),
                    grad_summary=fd_sum,
                    bench_summary=_summ_bench(best_metrics),
                    profile_hint="(n/a, fix first)",
                )
                # Minimal assumption: repo apply_patch applies patch-text to files in-tree.
                repo_apply_patch(patch)  # in-place; path presumed unchanged
                # retry correctness in next iteration
                continue

            # 2) performance
            cand = _norm_bench(benchmark(bwd_fp))
            # (optional) profile to get a hint—but don't depend on it
            try:
                prof_path = profile(bwd_fp)
                prof_hint = f"profile: {os.path.basename(prof_path)}" if prof_path else "(no profile)"
            except Exception:
                prof_hint = "(no profile)"

            improved = _better(cand, best_metrics, self.cfg.min_rel_improvement)
            if improved:
                best_metrics = cand
                best_path = bwd_fp
                non_improve = 0
            else:
                non_improve += 1
                if non_improve >= self.cfg.patience:
                    break  # plateau

            # 3) ask for an optimization patch and apply
            patch = self.patcher.propose_patch(
                phase="optimize",
                device=device,
                target_file=bwd_fp,
                kernel_snippet=_read_snippet(bwd_fp, self.cfg.snippet_max_lines),
                grad_summary="OK",
                bench_summary=_summ_bench(best_metrics),
                profile_hint=prof_hint,
            )
            repo_apply_patch(patch)  # in-place; path presumed unchanged

        return {
            "best_metrics": best_metrics or {},
            "best_backward_fp": best_path,
            "device_info": device,
        }