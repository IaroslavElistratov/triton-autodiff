from __future__ import annotations
from dataclasses import dataclass
import os, re, json, sys

import torch

from gpt_oss.tools.apply_patch import apply_patch as _apply_patch_raw
from .llm import _ensure_update_file_target  # normalize target header so model needn't guess file path
from .utils import _read_snippet, compile_kernel as create_op, CompileError
from .tools.gradcheck.core import check_op_backward_parity
from .tools.benchmark import bench_op, reduce_bench
from .strategy import make_strategy, GLOBAL_GUARDRAILS

# Verbose flag: set KERNEL_AGENT_VERBOSE=1|true to enable detailed logs
VERBOSE = str(os.environ.get("KERNEL_AGENT_VERBOSE", "")).strip().lower() in ("1", "true", "yes", "y")

def _read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""


def _file_mtime_ns(p: str) -> int | None:
    try:
        return os.stat(p).st_mtime_ns
    except Exception:
        return None



@dataclass
class Config:
    max_iters: int = 6
    patience: int = 2
    min_rel_improvement: float = 0.10   # require >= +10% throughput to accept
    # todo: a better way?
    snippet_max_lines: int = 400        # bound context shown to the LLM


FIX_HEADER = (
    "Phase = fix. ONLY restore correctness to pass gradcheck.\n"
    + GLOBAL_GUARDRAILS
)

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
        # Strategy encapsulates per-step phase text and temperature; keeps the
        # main loop clean and allows switching behavior via env
        self.strategy = make_strategy(os.environ.get("KERNEL_AGENT_STRATEGY", "regular"))
        # Toggle optimize path: "simple" (strategy + error echo, default) or "legacy" (original _llm_request_and_apply helper)
        self.optimize_mode = str(os.environ.get("KERNEL_AGENT_OPTIMIZE_MODE", "simple")).strip().lower()

    def _llm_request_and_apply(self, it: int, stage: str, *, bwd_fp: str, fwd_fp: str, header: str, state_facts: dict, temperature: float | None = None) -> bool:
        """Ask for one patch, apply with one retry on apply error, return True if apply succeeded.

        Separation of concerns:
          - llm.propose_patch: ensures a non-empty apply_patch.md block, normalizes target path,
            and distinguishes max_tokens vs generic no-patch. It may raise on generation failure.
          - _llm_request_and_apply (this): applies the patch via patcher, performs one
            apply-repair retry on failure, detects change via before/after bytes, and breadcrumbs errors.
            No preflight/anchors; no prompt error injection; no generation retry duplication.
            These are brital, instead for patch application errors just realy on the patcher to raise an error.
        """
        assert stage in ("fix", "optimize")
        if VERBOSE:
            print(f"[kernel-agent][it={it}] LLM phase='{stage}'")

        # context shown to LLM
        fwd_snip = _read_snippet(fwd_fp, self.cfg.snippet_max_lines)
        bwd_snip = _read_snippet(bwd_fp, self.cfg.snippet_max_lines)

        if temperature is not None:
            try:
                self.patcher.temperature = float(temperature)
            except Exception:
                pass

        # 1) Propose once (provider already does one strict retry on empty/no-op and may raise,
        #    including a specific max_tokens error). Avoid duplicate generation retries here
        #
        # Provider already did a strict retry and raised/returned accordingly.
        # Avoid duplicate generation retries here.
        patch = self.patcher.propose_patch(
            phase=header,
            bwd_file=bwd_fp,
            fwd_kernel_snippet=fwd_snip,
            bwd_kernel_snippet=bwd_snip,
            state_facts=state_facts or {},
        )

        if VERBOSE:
            print(f"[kernel-agent][it={it}] LLM patch preview:\n{str(patch)[:800]}")

        # Normalize the file header so the model doesn't spend tokens on it and
        # we avoid target-path drift in apply.
        patch = _ensure_update_file_target(patch, bwd_fp)

        # Detect change on raw bytes
        before = _read_bytes(bwd_fp)

        try:
            _apply_patch_raw(patch)
        except Exception as e:
            # 2. Surface exact patcher error and reprompt once with the error attached.
            # rely on the patcher to surface validation errors at apply time; no preflight checks; the patcher remains the source of truth
            err_msg = f"{type(e).__name__}: {e}"
            self.patcher.remember("apply.error", err_msg)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] apply error: {err_msg}")
            fix_header = header + (
                "\nRETRY: Your previous patch failed to apply.\n"
                f"apply_patch error:\n{err_msg}\n"
                "Produce a corrected patch and call functions.apply_patch again. No prose."
            )
            patch = self.patcher.propose_patch(
                phase=fix_header,
                bwd_file=bwd_fp,
                fwd_kernel_snippet=fwd_snip,
                bwd_kernel_snippet=bwd_snip,
                state_facts=state_facts or {},
            )
            patch2 = (patch or "").strip()
            if not patch2:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] still empty after apply error reprompt")
                return False

            # retry once on apply error; if the second attempt still fails, do not crash the process—treat as failed iteration;
            # keep this under its own try/except because the application of the 2nd patch
            # can independantly fail and it did happen in the past
            try:
                # Re-normalize header on retry and apply again.
                _apply_patch_raw(_ensure_update_file_target(patch2, bwd_fp))
            except Exception as e2:
                err2 = f"{type(e2).__name__}: {e2}"
                self.patcher.remember("apply.error.retry", err2)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] apply retry error: {err2}")
                # Optimize path: uses the boolean to control phase advancement (no advance on False), in both legacy and phased modes;
                # Fix paths: (compile/gradcheck/bench errors) intentionally ignore the return bool and proceed to next iteration
                return False

        # 3) Record + report change. Detect change on raw bytes for simplicity.
        after = _read_bytes(bwd_fp)
        changed = (after != before)
        # Keep a compact preview for breadcrumbs while avoiding token bloat.
        self.patcher.remember(f"llm.patch.{stage}", str(patch)[:1200])
        self.patcher.remember(f"apply.{stage}", ("ok" if changed else "no-change"))
        if VERBOSE:
            print(f"[kernel-agent][it={it}] {'changed' if changed else 'no change'} in '{stage}'")
        return changed

    def _create_op_with_fix(self, it: int, fwd_fp: str, overwrite_fp: str | None):
        try:
            return create_op(fwd_fp, overwrite_fp=overwrite_fp)
        except CompileError as ce:
            info = ce.info
            target = info.get("bwd_file") or info.get("fwd_file") or (overwrite_fp or fwd_fp)
            summary = f"compile_error[{info.get('phase','?')}]: {info.get('error_type')}: {info.get('error_message')}"
            if VERBOSE:
                print(f"[kernel-agent][it={it}] {summary}")
            # route via helper with explicit fix header and state_facts
            _ = self._llm_request_and_apply(
                it, "fix", bwd_fp=str(target), fwd_fp=fwd_fp,
                header=FIX_HEADER,
                state_facts={"compile_error": info},
                temperature=0.25,
            )
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
        stop_reason = "max_iters"

        # optimization loop
        for it in range(self.cfg.max_iters):
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Begin iteration")

            # 1) correctness gate
            if it > 0:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Rebuilding op with current backward: {bwd_fp}")
                op, _, ns = self._create_op_with_fix(it, fwd_fp, overwrite_fp=bwd_fp)

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
            try:
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
            # todo-now:
            # because I don't run backward in create_op. Although we have code there to catch errors, backward isn't invoked, so the error isn't caught and it hard-fails later when .backward is called (e.g., during gradcheck).
            # so maybe cleaner to re-emptively run backward in create_op -- so that there's taht concrete boundray, in which if create_op ran then can be certain that both fwd and bwd well formed
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] gradcheck error: {err}")
                self.patcher.remember("gradcheck.error", err)
                _ = self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=FIX_HEADER,
                    state_facts={"runtime_error": {"stage": "gradcheck", "error": err}, "dims": dims},
                    temperature=0.35,
                )
                # retry correctness in next iteration
                continue
            if VERBOSE:
                print(f"[kernel-agent][it={it}] gradient_check ok={ok}")
                print(f"[kernel-agent][it={it}] gradient_check stats={json.dumps(stats, default=str) if isinstance(stats, (dict, list)) else stats}")

            self.patcher.remember("gradcheck", f"ok={ok}\n{json.dumps(stats, default=str) if isinstance(stats, (dict, list)) else stats}")

            if not ok:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Parity failed — requesting 'fix' patch from LLM")
                _ = self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=FIX_HEADER,
                    state_facts={"grad_summary": stats, "dims": dims},
                    temperature=0.35,
                )
                # retry correctness in next iteration
                continue

            # Benchmark the current autograd op (backward), independent of optimize path
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Benchmarking backward")
            try:
                # Restrict benchmark shapes during Phase 1 only
                sidecar = ns
                if self.strategy.i == 0:
                    sweep = ns["SWEEP"]
                    sidecar = dict(ns)
                    sidecar["SWEEP"] = sweep[:1]
                bench_records = bench_op(
                    op,              # autograd-backed op from create_op(...)
                    sidecar,         # sidecar providing SWEEP and make_args
                    mode="bwd",
                )
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] bench error: {err}")
                self.patcher.remember("bench.error", err)
                _ = self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=FIX_HEADER,
                    state_facts={"runtime_error": {"stage": "bench", "error": err}, "dims": dims},
                    temperature=0.35,
                )
                continue
            cand = reduce_bench(bench_records)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] bench: {cand}")

            # Simple plateau logic governed by cfg.patience and min_rel_improvement.
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
                    stop_reason = "patience"
                    if VERBOSE:
                        print(f"[kernel-agent][it={it}] Early stop: patience reached (non_improve={non_improve})")
                    break

            # 3) Optimize step (toggleable to keep loop disentangled from policy)
            if self.optimize_mode == "legacy":
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Requesting 'optimize' patch from LLM (legacy path)")
                phase_text, temp = self.strategy.next_phase(parity_ok=ok, last_runtime=cand.get("median_ms"))
                changed = self._llm_request_and_apply(
                    it, "optimize", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=phase_text,
                    state_facts={"gradcheck_ok": ok, "dims": dims, "bench": cand},
                    temperature=temp,
                )
                self.strategy.advance(changed=changed, parity_ok=ok)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] End iteration")
                continue

            # Strategy-based path: generate phase text + temp, request patch, and
            # let the patcher apply/raise; echo apply errors back into next prompt.
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Requesting 'optimize' patch from LLM")

            phase_text, temp = self.strategy.next_phase(parity_ok=ok, last_runtime=cand.get("median_ms"))
            changed = self._llm_request_and_apply(
                it, "optimize", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                header=phase_text,
                state_facts={"gradcheck_ok": ok, "dims": dims, "bench": cand},
                temperature=temp,
            )
            self.strategy.advance(changed=changed, parity_ok=ok)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] End iteration")

        return {
            "best_metrics": best_metrics or {},
            "best_backward_fp": best_path,
            "device_info": device,
            "stop_reason": stop_reason,
        }