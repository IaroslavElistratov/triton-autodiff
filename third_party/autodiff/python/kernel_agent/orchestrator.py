from __future__ import annotations
from dataclasses import dataclass
import os, re, json

import torch

from gpt_oss.tools.apply_patch import apply_patch as _apply_patch_raw
from .utils import _read_snippet, compile_kernel as create_op
from .tools.gradcheck.core import check_op_backward_parity

# Verbose flag: set KERNEL_AGENT_VERBOSE=1|true to enable detailed logs
VERBOSE = str(os.environ.get("KERNEL_AGENT_VERBOSE", "")).strip().lower() in ("1", "true", "yes", "y")

# Parse target file paths from patch headers (Update/Add/Delete).
def _extract_update_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for ln in patch_text.splitlines():
        s = ln.strip()
        if s.startswith("*** Update File:") or s.startswith("*** Add File:") or s.startswith("*** Delete File:"):
            try:
                paths.append(s.split(":", 1)[1].strip())
            except Exception:
                pass
    return paths

def _patch_has_effect(patch_text: str) -> bool:
    """Detect whether patch contains any real change hunks or add/delete ops."""
    lines = [ln.strip() for ln in patch_text.splitlines()]
    if any(ln.startswith(("*** Add File:", "*** Delete File:")) for ln in lines):
        return True
    if any(ln.startswith("@@") for ln in lines):
        return True
    for ln in lines:
        if ln and ln[0] in "+-" and not ln.startswith("***"):
            return True
    return False

def _read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""

def _apply_and_report(patch_text: str, it: int, stage: str) -> tuple[bool, list[str]]:
    """Apply patch iff it has effect; report whether any bytes changed and which targets were touched."""
    # 1) Parse targets from patch headers.
    targets = _extract_update_paths(patch_text)
    # 0) Skip apply if patch has no effect (prevents false positives in change detection).
    if not _patch_has_effect(patch_text):
        if VERBOSE:
            print(f"[kernel-agent][it={it}] No-op '{stage}' patch; no hunks/add/delete — skipping apply")
        return False, targets
    # 2) Snapshot raw bytes of each target before applying the patch.
    before = {p: _read_bytes(p) for p in targets}
    # 3) Apply the patch in-tree.
    _apply_patch_raw(patch_text)
    # 4) Re-read bytes and mark changed if any target differs.
    changed = any(_read_bytes(p) != before.get(p, b"") for p in targets)
    if VERBOSE:
        if changed:
            print(f"[kernel-agent][it={it}] Applied '{stage}' patch; changed: {targets or ['(no explicit targets)']}")
        else:
            print(f"[kernel-agent][it={it}] No-op '{stage}' patch; nothing changed for: {targets or ['(no explicit targets)']}")
    return changed, targets



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
            get_user_device_info,
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
        if VERBOSE:
            print("[kernel-agent] Starting run")
            print(f"[kernel-agent] Forward file: {fwd_fp}")
            print("[kernel-agent] Compiling and tracing user kernel via create_op(...) (seed backward)")
        op, bwd_fp, ns = create_op(fwd_fp, overwrite_fp=None)
        if VERBOSE:
            print(f"[kernel-agent] Initial backward path: {bwd_fp}")


        # best_metrics: dict[str, float] | None = None
        device = get_user_device_info()
        best_path = bwd_fp
        # non_improve = 0

        # optimization loop
        for it in range(self.cfg.max_iters):
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Begin iteration")

            # 1) correctness gate
            if it > 0:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Rebuilding op with current backward: {bwd_fp}")
                op, _, _ = create_op(fwd_fp, overwrite_fp=bwd_fp)

            # Build inputs for parity check from user's helpers
            make_args = ns.get("make_args")
            if not callable(make_args):
                raise RuntimeError("User kernel must define make_args(dims) -> (args, kwargs)")
            sweep = ns.get("SWEEP")
            dims = sweep[0] if isinstance(sweep, (list, tuple)) and sweep else {}
            args, _kwargs = make_args(dims)
            if VERBOSE:
                shapes = tuple(getattr(t, "shape", None) for t in args)
                print(f"[kernel-agent][it={it}] Inputs dims={dims}, shapes={shapes}")

            # (a, b), _ = mod.make_args(mod.SWEEP[0]) 

            # breadcrumb for LLM continuity
            self.patcher.remember("iteration", f"it={it}, bwd_file={bwd_fp}, dims={dims}")

            # compare grads: reference torch implementation vs my fused op

            # gradient_check uses autograd, its expectations are: my_op(*inputs) -> true outputs,
            # those outputs must be on a graph back to inputs. The stub satisfies this after @autodiff
            # because it runs the autograd‑wrapped kernel and returns the real outputs. The parity core
            # clones inputs, builds random upstreams, and compares torch.autograd.grad results per input.
            #
            # Grid handling remains in the stub, so gradient_check does not need to know meta params or shapes.
            # No change required to check_op_backward_parity.
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Running gradient_check (parity)")
            ok, stats = check_op_backward_parity(
                ref_fwd=ns["torch_fn"],
                my_op=op,
                inputs=args,
                outputs="auto",
                # tests/mamtul: backward casts to fp16 before dot and accumulates/atomics in fp16, while Torch grads accumulate in fp32;
                # later proper fix: keep accumulators fp32 and cast only at tl.atomic_add
                # todo-high: rm; too-high deltas
                atol=0.07,
                rtol=0.02,
            )
            if VERBOSE:
                print(f"[kernel-agent][it={it}] gradient_check ok={ok}")
                print(f"[kernel-agent][it={it}] gradient_check stats={json.dumps(stats, default=str) if isinstance(stats, (dict, list)) else stats}")

            self.patcher.remember("gradcheck", f"ok={ok}\n{json.dumps(stats, default=str) if isinstance(stats, (dict, list)) else stats}")

            if not ok:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Parity failed — requesting 'fix' patch from LLM")
                patch = self.patcher.propose_patch(
                    phase="fix",
                    # todo: pass fwd kernel to the model as well, for more context
                    bwd_file=bwd_fp,
                    fwd_kernel_snippet=_read_snippet(fwd_fp, self.cfg.snippet_max_lines),
                    bwd_kernel_snippet=_read_snippet(bwd_fp, self.cfg.snippet_max_lines),
                    grad_summary=stats,
                    # bench_summary=_summ_bench(best_metrics),
                    # profile_hint="(n/a, fix first)",
                )
                # apply_patch applies patch-text to files in-tree
                # in-place; path presumed unchanged
                if VERBOSE:
                    preview = str(patch)[:800]
                    print(f"[kernel-agent][it={it}] LLM thinking (fix):\n{getattr(self.patcher, 'last_thinking', '')}")
                    print(f"[kernel-agent][it={it}] LLM patch (fix) preview:\n{preview}")
                if not isinstance(patch, str) or not patch.lstrip().startswith("*** Begin Patch"):
                    print("[kernel-agent] LLM returned non-patch content; skipping apply for 'fix' phase")
                    break
                self.patcher.remember("llm.patch.fix", patch[:1200])
                changed, targets = _apply_and_report(patch, it, "fix")
                self.patcher.remember("apply.fix", f"changed={changed}, targets={targets}")
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
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Requesting 'optimize' patch from LLM")
            patch = self.patcher.propose_patch(
                phase="optimize",
                bwd_file=bwd_fp,
                fwd_kernel_snippet=_read_snippet(fwd_fp, self.cfg.snippet_max_lines),
                bwd_kernel_snippet=_read_snippet(bwd_fp, self.cfg.snippet_max_lines),
                grad_summary="OK",
                # bench_summary=_summ_bench(best_metrics),
                # profile_hint=prof_hint,
            )
            if VERBOSE:
                preview = str(patch)[:800]
                print(f"[kernel-agent][it={it}] LLM thinking (optimize):\n{getattr(self.patcher, 'last_thinking', '')}")
                print(f"[kernel-agent][it={it}] LLM patch (optimize) preview:\n{preview}")
            if not isinstance(patch, str) or not patch.lstrip().startswith("*** Begin Patch"):
                print("[kernel-agent] LLM returned non-patch content; skipping apply for 'optimize' phase")
                break

            self.patcher.remember("llm.patch.optimize", patch[:1200])
            changed, targets = _apply_and_report(patch, it, "optimize")  # in-place
            self.patcher.remember("apply.optimize", f"changed={changed}, targets={targets}")
            if VERBOSE:
                print(f"[kernel-agent][it={it}] End iteration")

        return {
            # "best_metrics": best_metrics or {},
            "best_backward_fp": best_path,
            "device_info": device,
        }