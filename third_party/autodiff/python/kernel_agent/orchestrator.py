from __future__ import annotations
from dataclasses import dataclass
import os, re, json, sys

import torch

from gpt_oss.tools.apply_patch import apply_patch as _apply_patch_raw
from .llm import _ensure_update_file_target  # normalize target header so model needn't guess file path
from .utils import _read_snippet, compile_kernel as create_op, CompileError
from .tools.gradcheck.core import check_op_backward_parity_sweep
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

        # todo: [cleanup] llm.py should be responsible for everyhting related to patch proposal and its
        # errors, do not spread that logic across both llm.py and this file. Remove these try/except around
        # self.patcher.propose_patch and make it the responsibility of llm.py
        #
        # llm.py detects and classifies max-tokens, then raises; this try/catch in the orchestrator
        # just catches that exception so the loop doesn’t crash
        try:
            patch = self.patcher.propose_patch(
                phase=header,
                bwd_file=bwd_fp,
                fwd_kernel_snippet=fwd_snip,
                bwd_kernel_snippet=bwd_snip,
                state_facts=state_facts or {},
            )
        except Exception as e:
            msg = str(e)
            # Treat generation limits (max_tokens) and generic proposal errors as non-fatal
            self.patcher.remember("llm.propose.error", msg)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] propose error: {msg}")
            return False

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
            try:
                patch = self.patcher.propose_patch(
                    phase=fix_header,
                    bwd_file=bwd_fp,
                    fwd_kernel_snippet=fwd_snip,
                    bwd_kernel_snippet=bwd_snip,
                    state_facts=state_facts or {},
                )
            except Exception as e_propose_retry:
                err_retry = f"{type(e_propose_retry).__name__}: {e_propose_retry}"
                self.patcher.remember("llm.propose.error.retry", err_retry)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] propose retry error: {err_retry}")
                return False
            patch2 = (patch or "").strip()
            if not patch2:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] still empty after apply error reprompt")
                return False

            # todo-high: debatable to cath this, as you can just advance a bunch of iterations with failed patches
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

    def _create_op_with_fix(self, it: int | None, fwd_fp: str, overwrite_fp: str | None):
        try:
            return create_op(fwd_fp, overwrite_fp=overwrite_fp)

        # catch only the CompileError, because there are other types of errors which create_op
        # raises -- want to surface them to the user, only want to catch the CompileError
        except CompileError as ce:

            # for initial build (outside the optimization loop), bubble up
            if it is None:
                raise ce

            # info = ce.info
            err = f"{type(ce).__name__}: {ce}"
            self.patcher.remember("llm.compile_error", err)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] compile error: {err}")
            # route via helper with explicit fix header and state_facts
            _ = self._llm_request_and_apply(
                it, "fix", bwd_fp=self.bwd_fp, fwd_fp=fwd_fp,
                header=FIX_HEADER,
                state_facts={"compile_error": err},
                temperature=0.25,
            )
            # catch here as well, because this create_op can independently error
            try:
                return create_op(fwd_fp, overwrite_fp=overwrite_fp)
            except CompileError as ce_retry:
                # info_retry = ce_retry.info
                err_retry = f"{type(ce_retry).__name__}: {ce_retry}"
                self.patcher.remember("llm.compile_error", err_retry)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] compile retry error: {err_retry}")
                _ = self._llm_request_and_apply(
                    it, "fix", bwd_fp=self.bwd_fp, fwd_fp=fwd_fp,
                    header=FIX_HEADER,
                    state_facts={"compile_error": err_retry},
                    temperature=0.25,
                )
                # ugly but need signature consistent with create_op signature
                # because the caller of _create_op_with_fix can assign to a tuple
                return None, None, None

    def run(self, *,
            fwd_fp: str,
            # benchmark,
            # profile,
            get_user_device_info,
            ) -> dict:

        if not os.path.isfile(fwd_fp):
            raise FileNotFoundError(f"forward file not found: {fwd_fp}")


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

        op, bwd_fp, ns = self._create_op_with_fix(None, fwd_fp, overwrite_fp=None)
        self.bwd_fp = bwd_fp
        sweep, make_args, torch_fn = ns.get("SWEEP"), ns.get("make_args"), ns.get("torch_fn")
        if not callable(make_args) or not isinstance(sweep, (list, tuple)):
            raise RuntimeError("User kernel must define make_args and SWEEP")
        if not torch_fn:
            raise RuntimeError("Please define torch_fn semantically equivalent to your triton kernel + stub")

        # compute shapes for all dims upfront for logging/breadcrumbs
        # shapes = [(inp.shape for inp in make_args(i)[0]) for i in sweep]
        shapes = []
        for _dims in sweep:
            _args, _kwargs = make_args(_dims)
            if isinstance(_args, (list, tuple)):
                shapes.append(tuple(t.shape for t in _args))
            else:
                shapes.append((_args.shape, ))


        if VERBOSE:
            print(f"[kernel-agent] Initial backward path: {bwd_fp}")


        device = get_user_device_info()
        # Initialize plateau tracking to avoid unbound locals on early returns
        best_metrics = None
        non_improve = 0
        stop_reason = "max_iters"

        # optimization loop
        for it in range(self.cfg.max_iters):
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Begin iteration")

            # only first shape on it==0, then all shapes;
            # running parity and bench across all entries after attempt 0 makes "loops re‑introduced"
            # observable and blocks phase advance until the same kernel passes on every shape;
            # the raised naive backward is unrolled and shape‑specialized; it will fail on varied
            # shapes until loops are restored
            #
            # failed patch apply on attempt 0 "continue"s to attempt 1 and triggers full-SWEEP anyway,
            # the model will see the failures with dims included in the stats and fix accordingly
            # todo-high: add phase-specific header even when gradcheck / benchmark fails (together with FIX_HEADER) ?
            sidecar = ns
            if it == 0:
                sidecar = dict(ns)
                sidecar["SWEEP"] = ns["SWEEP"][:1]

            # 1) correctness gate
            if it > 0:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Rebuilding op with current backward: {bwd_fp}")
                # Rebuild only the op; the sidecar namespace (ns) remains unchanged across iterations by design
                op, _, _ = self._create_op_with_fix(it, fwd_fp, overwrite_fp=bwd_fp)
                # failed to compile kernel
                if not op:
                    continue

            if VERBOSE:
                print(f"[kernel-agent][it={it}] Inputs shapes={shapes}")

            # breadcrumb for LLM continuity
            self.patcher.remember("iteration", f"it={it}, bwd_file={bwd_fp}, shapes={shapes}")

            # gradient_check uses autograd, its expectations are: my_op(*inputs) -> true outputs,
            # those outputs must be on a graph back to inputs. The stub satisfies this after @autodiff
            # because it runs the autograd‑wrapped kernel and returns the real outputs. The parity core
            # clones inputs, builds random upstreams, and compares torch.autograd.grad results per input.
            #
            # Grid handling remains in the stub, so gradient_check does not need to know meta params or shapes.
            # No change required to check_op_backward_parity.
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Running gradient_check (parity)")

            try:
                # phase 1 (and beyond): run parity over the full SWEEP to enforce loop re-introduction
                ok, stats = check_op_backward_parity_sweep(
                    ref_fwd=torch_fn,
                    my_op=op,
                    sidecar=sidecar,
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
                    state_facts={"runtime_error": {"stage": "gradcheck", "error": err}, "shapes": shapes},
                    temperature=0.35,
                )
                # retry correctness in next iteration
                continue
            if VERBOSE:
                print(f"[kernel-agent][it={it}] gradient_check ok={ok}, stats={stats}")

            # Breadcrumb: minimal
            self.patcher.remember("gradcheck", stats)

            if not ok:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Parity failed on sweep — requesting 'fix' patch from LLM")
                _ = self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=FIX_HEADER,
                    state_facts={"grad_summary": stats},
                    temperature=0.35,
                )
                # retry correctness in next iteration
                continue

            # Benchmark the current autograd op (backward), independent of optimize path
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Benchmarking backward")
            try:
                # benchmark across the full SWEEP to align with parity gating
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
                    state_facts={"runtime_error": {"stage": "bench", "error": err}, "shapes": shapes},
                    temperature=0.35,
                )
                continue
            cand = reduce_bench(bench_records)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] bench: {cand}")

            # Simple plateau logic governed by cfg.patience and min_rel_improvement
            improved = (best_metrics is None) or (
                cand["median_ms"] <= (1.0 - self.cfg.min_rel_improvement) * best_metrics["median_ms"]
            )

            if improved:
                best_metrics = cand
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
                    state_facts={"bench": cand, "grad_summary": stats},
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
                state_facts={"bench": cand, "grad_summary": stats},
                temperature=temp,
            )
            self.strategy.advance(changed=changed, parity_ok=ok)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] End iteration")

        return {
            "best_metrics": best_metrics,
            "backward_fp": bwd_fp,
            "device_info": device,
            "stop_reason": stop_reason,
        }