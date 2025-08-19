from __future__ import annotations
from dataclasses import dataclass
import os, re

from gpt_oss.tools.apply_patch import apply_patch
from .utils import compile_kernel, _read_snippet


@dataclass
class Config:
    max_iters: int = 6
    patience: int = 2
    min_rel_improvement: float = 0.02  # require >= +2% throughput to accept
    snippet_max_lines: int = 120        # bound context shown to the LLM



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
    def __init__(self, cfg: Config, patcher):
        self.cfg = cfg
        self.patcher = patcher

    def run(self, *,
            fwd_fp,
            naive_grad,
            # gradient_check,
            # benchmark,
            # profile,
            get_user_dvice_info,
            ) -> dict:

        if not os.path.isfile(fwd_fp):
            raise FileNotFoundError(f"forward file not found: {fwd_fp}")

        try:
            fwd_kernel = compile_kernel(fwd_fp)
        except Exception as e:
            raise RuntimeError("kernel malformed, provide a well-formed kernel") from e

        try:
            # todo-now: currently problem is that my sysytem retuns TTIR (not trtion-lang) thus output of my system cannot be used direcrly for downstream
            # bwd_fp = naive_grad(fwd_kernel)
            naive_bwd_fp = naive_grad(fwd_kernel)
            # temporary hack, in future "naive_grad" should prodice a triton-lang and output its file path
            bwd_dir = naive_bwd_fp.split("/out.ttir")[0]
            bwd_fp = os.path.join(bwd_dir, "backward.py")
        except Exception as e:
            raise RuntimeError("naive autograd failed") from e

        # best_metrics: dict[str, float] | None = None
        device = get_user_dvice_info()
        best_path = bwd_fp
        non_improve = 0


        # optimization loop
        for it in range(self.cfg.max_iters):

            # todo-high: lift differenciated TTIR to triton-lang
            # then can just use output of my system directly as the initial version of the backward kernel to be optimized
            # (removing patcher.naive_autodiff filed, instead just using output of trtion-autodiff as patcher.kernel_snippet)
            # and avoiding this special casing
            if it > 0:
                # 1) correctness gate
                bwd_kernel = compile_kernel(bwd_fp)

                ok, stats = gradient_check(
                    forward_fn=fwd_kernel,
                    backward_fn=lambda *inp_up: bwd_kernel(*inp_up),
                    inputs=(A, B),
                    mode="coord",
                )

                if not ok:
                    patch = self.patcher.propose_patch(
                        phase="fix",
                        target_file=bwd_fp,
                        naive_kernel=_read_snippet(naive_bwd_fp, self.cfg.snippet_max_lines),
                        kernel_snippet=_read_snippet(bwd_fp, self.cfg.snippet_max_lines),
                        grad_summary=stats,
                        # bench_summary=_summ_bench(best_metrics),
                        # profile_hint="(n/a, fix first)",
                    )
                    # apply_patch applies patch-text to files in-tree
                    # in-place; path presumed unchanged
                    apply_patch(patch)
                    # retry correctness in next iteration
                    continue

            # # 2) performance
            # cand = _norm_bench(benchmark(bwd_fp))
            # # (optional) profile to get a hint—but don't depend on it
            # try:
            #     prof_path = profile(bwd_fp)
            #     prof_hint = f"profile: {os.path.basename(prof_path)}" if prof_path else "(no profile)"
            # except Exception:
            #     prof_hint = "(no profile)"

            # improved = _better(cand, best_metrics, self.cfg.min_rel_improvement)
            # if improved:
            #     best_metrics = cand
            #     best_path = bwd_fp
            #     non_improve = 0
            # else:
            #     non_improve += 1
            #     if non_improve >= self.cfg.patience:
            #         break  # plateau

            # 3) ask for an optimization patch and apply
            kernel_snippet = _read_snippet(bwd_fp, self.cfg.snippet_max_lines) if os.path.isfile(bwd_fp) else ""
            phase = "optimize" if kernel_snippet else "init"
            patch = self.patcher.propose_patch(
                phase=phase,
                target_file=bwd_fp,
                naive_kernel=_read_snippet(naive_bwd_fp, self.cfg.snippet_max_lines),
                kernel_snippet=kernel_snippet,
                grad_summary="OK",
                # bench_summary=_summ_bench(best_metrics),
                # profile_hint=prof_hint,
            )
            apply_patch(patch)  # in-place; path presumed unchanged

        return {
            # "best_metrics": best_metrics or {},
            "best_backward_fp": best_path,
            "device_info": device,
        }