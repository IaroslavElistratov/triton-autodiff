from __future__ import annotations
from dataclasses import dataclass
import os, re

import torch

from gpt_oss.tools.apply_patch import apply_patch
from .utils import _read_snippet, compile_kernel as create_op
from .tools.gradcheck.core import check_op_backward_parity



@dataclass
class Config:
    max_iters: int = 6
    patience: int = 2
    min_rel_improvement: float = 0.10   # require >= +10% throughput to accept
    snippet_max_lines: int = 120        # bound context shown to the LLM



class KernelOptimizer:
    """
    Deterministic controller:
      init: get_user_forwrad() -> naive_grad (via triton-autodiff/api.py)
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
            # benchmark,
            # profile,
            get_user_dvice_info,
            ) -> dict:

        if not os.path.isfile(fwd_fp):
            raise FileNotFoundError(f"forward file not found: {fwd_fp}")

        # try:
        # except Exception as e:
        #     raise RuntimeError("kernel malformed, provide a well-formed kernel") from e

        # TTIR from autodiff then raise to Python once; use as seed and target
        # using output of triton-autograd directly as the initial version of the backward kernel
        # to be optimized -- "seeding a problem with a draft" (removing patcher.naive_autodif instead just using output of trtion-autodiff as patcher.kernel_snippet)

        # todo-high: support overwritting stub (current api.py integration doens't support it)

        # directly re-use api.py as otherwise i'd need to re-impl all the below funcs which i need
        # raise_to_triton_lang, load_raised_jit, wrap_bwd_kernel, DifferentiatedCompiledKernel, helper, autodiff
        op, bwd_fp, ns = create_op(fwd_fp, overwrite_fp=None)


        # best_metrics: dict[str, float] | None = None
        device = get_user_dvice_info()
        best_path = bwd_fp
        # non_improve = 0

        # optimization loop
        for it in range(self.cfg.max_iters):

            # 1) correctness gate
            if it > 0:
                op, _, _ = create_op(fwd_fp, overwrite_fp=bwd_fp)

            # Build inputs for parity check from user's helpers
            make_args = ns.get("make_args")
            if not callable(make_args):
                raise RuntimeError("User kernel must define make_args(dims) -> (args, kwargs)")
            sweep = ns.get("SWEEP")
            dims = sweep[0] if isinstance(sweep, (list, tuple)) and sweep else {}
            args, _kwargs = make_args(dims)

            # (a, b), _ = mod.make_args(mod.SWEEP[0]) 

            # compare grads: reference torch implementation vs my fused op

            # gradient_check uses autograd, its expectations are: my_op(*inputs) -> true outputs,
            # those outputs must be on a graph back to inputs. The stub satisfies this after @autodiff
            # because it runs the autograd‑wrapped kernel and returns the real outputs. The parity core
            # clones inputs, builds random upstreams, and compares torch.autograd.grad results per input.
            #
            # Grid handling remains in the stub, so gradient_check does not need to know meta params or shapes.
            # No change required to check_op_backward_parity.
            ok, stats = check_op_backward_parity(
                ref_fwd=ns["torch_fn"],
                my_op=op,
                inputs=args,
                outputs="auto",
            )

            if not ok:
                patch = self.patcher.propose_patch(
                    phase="fix",
                    # todo: pass fwd kernel to the model as well, for more context
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


            # todo-now: add benchmark; initially, without re-trace; only then with re-trace

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
            patch = self.patcher.propose_patch(
                phase="optimize",
                target_file=bwd_fp,
                kernel_snippet=_read_snippet(bwd_fp, self.cfg.snippet_max_lines),
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