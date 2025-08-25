from __future__ import annotations
from dataclasses import dataclass
import os, re

import torch

from gpt_oss.tools.apply_patch import apply_patch
from .utils import compile_kernel, _read_snippet


from ..api import raise_to_triton_lang, load_raised_jit
from trtion_autodiff.api import raise_to_triton_lang, load_raised_jit



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
            fwd_fp: str,
            naive_grad,
            gradient_check,
            # benchmark,
            # profile,
            get_user_dvice_info,
            ) -> dict:

        if not os.path.isfile(fwd_fp):
            raise FileNotFoundError(f"forward file not found: {fwd_fp}")

        try:
            # naive_grad expects a compiled Triton kernel object; users should set `compiled_kernel` in setup()
            fwd_stub, fwd_ns = compile_kernel(fwd_fp, return_ns=True)
            # todo-now: use hook
            fwd_kernel = fwd_ns["compiled_kernel"]
            # raise RuntimeError("Forward file must define `forward(*inputs)` or `stub(*inputs)` for gradcheck inputs")
            # raise RuntimeError("Unable to infer inputs for gradient_check; provide `make_args` in forward file")

        except Exception as e:
            raise RuntimeError("kernel malformed, provide a well-formed kernel") from e


        # using output of triton-autograd directly as the initial version of the backward kernel
        # to be optimized -- "seeding a problem with a draft" (removing patcher.naive_autodif instead just using output of trtion-autodiff as patcher.kernel_snippet)

        # TTIR from autodiff then raise to Python once; use as seed and target
        bwd_ttir_fp = naive_grad(fwd_kernel)
        raised_py_path = raise_to_triton_lang(bwd_ttir_fp)
        bwd_kernel = load_raised_jit(raised_py_path)    # JITFunction


        # best_metrics: dict[str, float] | None = None
        device = get_user_dvice_info()
        best_path = bwd_fp
        non_improve = 0


        # optimization loop
        for it in range(self.cfg.max_iters):

            # 1) correctness gate
            bwd_stub = compile_kernel(bwd_fp)

            # todo: hide in a helper
            make_args = fwd_ns["make_args"]
            args, kwargs = make_args(fwd_ns["SWEEP"][0])
            # upstream = tuple(torch.randn_like(out) for out in (fwd_stub(*args, **kwargs),))

            ok, stats = gradient_check(
                forward_fn=fwd_stub,
                backward_fn=bwd_stub,
                inputs=args,
                mode="coord",
            )

            if not ok:
                patch = self.patcher.propose_patch(
                    phase="fix",
                    target_file=bwd_fp,
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
                kernel_snippet=kernel_snippet, # _read_snippet(bwd_fp, self.cfg.snippet_max_lines),
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