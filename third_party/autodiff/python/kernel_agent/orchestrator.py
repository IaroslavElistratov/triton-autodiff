from __future__ import annotations
from dataclasses import dataclass
import os, re, json

import torch

from gpt_oss.tools.apply_patch import apply_patch as _apply_patch_raw
from .utils import _read_snippet, compile_kernel as create_op, CompileError
from .tools.gradcheck.core import check_op_backward_parity
from .tools.benchmark import bench_from_file, reduce_bench

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
    # todo: a better way?
    snippet_max_lines: int = 400        # bound context shown to the LLM



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

    def _llm_request_and_apply(self, it: int, stage: str, *, bwd_fp: str, fwd_fp: str, grad_summary: object) -> None:
        """Request a patch from the LLM with one retry for empty/invalid output,
        apply it, and if apply fails, request a one-shot repair patch and apply.
        Distinguishes max_tokens truncation from generic invalid patch.

        Side-effects:
          - Remembers breadcrumbs for llm.patch.<stage>, apply.<stage>, and errors
          - Prints verbose previews when VERBOSE=true
        """
        assert stage in ("fix", "optimize")
        phase = "fix" if stage == "fix" else "optimize"

        def _propose(fwd_snip: str, bwd_snip: str) -> str:
            return self.patcher.propose_patch(
                phase=phase,
                bwd_file=bwd_fp,
                fwd_kernel_snippet=fwd_snip,
                bwd_kernel_snippet=bwd_snip,
                grad_summary=grad_summary,
            )

        # Initial snippets
        fwd_snip = _read_snippet(fwd_fp, self.cfg.snippet_max_lines)
        bwd_snip = _read_snippet(bwd_fp, self.cfg.snippet_max_lines)

        # First request
        patch = _propose(fwd_snip, bwd_snip)
        if VERBOSE:
            preview = str(patch)[:800]
            print(f"[kernel-agent][it={it}] LLM thinking ({stage}):\n{getattr(self.patcher, 'last_thinking', '')}")
            print(f"[kernel-agent][it={it}] LLM patch ({stage}) preview:\n{preview}")

        # Retry once for empty/invalid patch
        if not isinstance(patch, str) or not patch.lstrip().startswith("*** Begin Patch"):
            stop_reason = getattr(self.patcher, "last_stop_reason", "")
            if stop_reason.lower() == "max_tokens":
                self.patcher.remember(f"llm.patch.{stage}.error", "generation truncated (max_tokens); retrying once")
            else:
                self.patcher.remember(f"llm.patch.{stage}.error", "invalid or empty patch; retrying once")
            patch = _propose(fwd_snip, bwd_snip)
            if not isinstance(patch, str) or not patch.lstrip().startswith("*** Begin Patch"):
                stop_reason = getattr(self.patcher, "last_stop_reason", "")
                if stop_reason.lower() == "max_tokens":
                    raise RuntimeError(f"LLM stopped due to max_tokens during {stage}; no patch produced")
                raise RuntimeError(f"LLM returned invalid/empty patch twice for '{stage}'")

        # Remember and attempt apply
        self.patcher.remember(f"llm.patch.{stage}", patch[:1200])
        try:
            changed, targets = _apply_and_report(patch, it, stage)
            self.patcher.remember(f"apply.{stage}", f"changed={changed}, targets={targets}")
        except Exception as e:
            # Surface patcher error to model and let it repair the patch once
            msg = f"apply_error: {type(e).__name__}: {e}"
            self.patcher.remember(f"apply.{stage}.error", msg)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Apply error ({stage}): {msg}")
            patch = _propose(fwd_snip, bwd_snip)
            if not isinstance(patch, str) or not patch.lstrip().startswith("*** Begin Patch"):
                raise RuntimeError(f"LLM returned invalid/empty patch on {stage}-retry after apply error")
            changed, targets = _apply_and_report(patch, it, f"{stage}-retry")
            self.patcher.remember(f"apply.{stage}-retry", f"changed={changed}, targets={targets}")

    def _create_op_with_fix(self, it: int, fwd_fp: str, overwrite_fp: str | None):
        try:
            return create_op(fwd_fp, overwrite_fp=overwrite_fp)
        except CompileError as ce:
            info = ce.info
            target = info.get("bwd_file") or info.get("fwd_file") or (overwrite_fp or fwd_fp)
            summary = f"compile_error[{info.get('phase','?')}]: {info.get('error_type')}: {info.get('error_message')}"
            if VERBOSE:
                print(f"[kernel-agent][it={it}] {summary}")
            # Pass the full info dict; _llm_request_and_apply accepts any object as grad_summary
            self._llm_request_and_apply(it, "fix", bwd_fp=str(target), fwd_fp=fwd_fp, grad_summary=info)
            return create_op(fwd_fp, overwrite_fp=overwrite_fp)

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
        # using output of triton-autodiff directly as the initial version of the backward kernel
        # to be optimized -- "seeding a problem with a draft" (removing patcher.naive_autodif instead just using output of trtion-autodiff as patcher.kernel_snippet)

        # todo-high: support overwritting stub (current api.py integration doens't support it)

        # directly re-use api.py as otherwise i'd need to re-impl all the below funcs which i need
        # raise_to_triton_lang, load_raised_jit, wrap_bwd_kernel, DifferentiatedCompiledKernel, helper, autodiff
        if VERBOSE:
            print("[kernel-agent] Starting run")
            print(f"[kernel-agent] Forward file: {fwd_fp}")
            print("[kernel-agent] Compiling and tracing user kernel via create_op(...) (seed backward)")
        op, bwd_fp, ns = self._create_op_with_fix(0, fwd_fp, overwrite_fp=None)
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
                op, _, _ = self._create_op_with_fix(it, fwd_fp, overwrite_fp=bwd_fp)

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
            torch_fn = ns.get("torch_fn")
            if not torch_fn:
                raise RuntimeError("Please define torch_fn semantically equivalent to your triton kernel + stub")
            ok, stats = check_op_backward_parity(
                ref_fwd=torch_fn,
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
                self._llm_request_and_apply(it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp, grad_summary=stats)
                # retry correctness in next iteration
                continue

            # Benchmark current backward
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Benchmarking backward")
            bench_records = bench_from_file(
                fwd_fp,
                overwrite_bwd_fp=bwd_fp,
                mode="bwd",
            )
            cand = reduce_bench(bench_records)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] bench: {cand}")

            # simple plateau logic
            if 'best_metrics' not in locals():
                best_metrics = None
            if 'non_improve' not in locals():
                non_improve = 0

            improved = (best_metrics is None) or (
                cand["median_ms"] <= (1.0 - self.cfg.min_rel_improvement) * best_metrics["median_ms"]
            )

            if improved:
                best_metrics = cand
                best_path = bwd_fp
                non_improve = 0
            else:
                non_improve += 1
                if non_improve >= self.cfg.patience:
                    break

            # 3) ask for an optimization patch and apply
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Requesting 'optimize' patch from LLM")
            self._llm_request_and_apply(it, "optimize", bwd_fp=bwd_fp, fwd_fp=fwd_fp, grad_summary="OK")
            if VERBOSE:
                print(f"[kernel-agent][it={it}] End iteration")

        return {
            "best_metrics": best_metrics or {},
            "best_backward_fp": best_path,
            "device_info": device,
        }