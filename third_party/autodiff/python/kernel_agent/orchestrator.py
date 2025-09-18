from __future__ import annotations
from dataclasses import dataclass
import os, re, json, sys, shutil

import torch

from gpt_oss.tools.apply_patch import apply_patch as _apply_patch_raw
from .llm import _ensure_update_file_target  # normalize target header so model needn't guess file path
from .utils import _read_snippet, compile_kernel as create_op, CompileError
from .worker import run_gradcheck_child, run_bench_child
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
    patience_perf_stop: int = 2  # stop loop after this many non-improving full-parity iterations (performance patience)
    patience_parity_restore: int = 2  # allow this many parity-regression iterations before restoring snapshot
    min_rel_improvement: float = 0.10   # require >= +10% throughput to accept
    # todo: a better way?
    snippet_max_lines: int = 400        # bound context shown to the LLM


FIX_HEADER = (
    "Phase = fix. ONLY restore correctness to pass gradcheck.\n"
    + GLOBAL_GUARDRAILS
)

class Rollback:
    """Lock-wins snapshot manager for the backward kernel file.

    - Snapshots the current kernel to a sidecar file on improved parity/perf
    - Restores from that snapshot on plateau/regress (e.g., patience stop)
    - Tracks best parity coverage across iterations
    """
    def __init__(self, backward_fp: str, patience_parity_restore: int = 0) -> None:
        self.backward_fp = backward_fp
        self.lock_fp = f"{backward_fp}.lock"
        self.best_pass_count: int = 0
        self._parity_regress_streak: int = 0
        self._patience_parity_restore: int = int(max(0, patience_parity_restore))
        self._log(
            f"init: patience_parity_restore={self._patience_parity_restore} | lock={self.lock_fp}"
        )

    def _log(self, text: str) -> None:
        if VERBOSE:
            print(f"[kernel-agent][rollback] {text}")

    def snapshot(self, note: str = "") -> None:
        """Save current kernel contents to the lock file.
        Single I/O choke point used by higher-level triggers (parity/perf).
        Keeping it here avoids duplicate try/except noise and centralizes logging.
        """
        try:
            shutil.copyfile(self.backward_fp, self.lock_fp)
            msg = f"snapshot: {self.backward_fp} -> {self.lock_fp}"
            if note:
                msg += f" ({note})"
            self._log(msg)
        except Exception as e:
            self._log(f"warning: snapshot failed: {type(e).__name__}: {e}")

    def _restore(self) -> None:
        """Restore kernel from the last snapshot, if present."""
        try:
            if os.path.isfile(self.lock_fp):
                shutil.copyfile(self.lock_fp, self.backward_fp)
                self._log(f"restore: {self.lock_fp} -> {self.backward_fp}")
        except Exception as e:
            self._log(f"warning: restore failed: {type(e).__name__}: {e}")

    def maybe_snapshot_or_restore(self, stats) -> bool:
        """Lock on parity improvement; optionally restore after sustained regression.
        Correctness-first. Snapshot immediately on increases in num_passed.
        If parity regresses, tolerate a few attempts (patience_parity_restore) to
        let the model iterate on a risky refactor before restoring the last lock.
        Returns True iff a restore occurred (caller can skip fix prompt and re-run).
        """
        # Parity regression handling:
        # - If fewer shapes pass than our best so far, revert to the locked snapshot.
        # - If more shapes pass (but not all), snapshot this incremental improvement.
        # - If equal, leave the current file as-is.
        curr_passed, total = int(stats.get("num_passed", 0)), int(stats.get("num_total", 0))
        if curr_passed < self.best_pass_count:
            self._parity_regress_streak += 1
            self._log(
                f"parity regress: {curr_passed}/{total} < best {self.best_pass_count} | streak {self._parity_regress_streak}/{self._patience_parity_restore}"
            )
            if self._parity_regress_streak >= self._patience_parity_restore:
                self._log("parity regress threshold reached -> restore")
                self._restore()
                self._parity_regress_streak = 0
                return True
        elif curr_passed > self.best_pass_count:
            self.best_pass_count = curr_passed
            prev = self.best_pass_count
            self._log(f"parity improve: best {prev} -> {self.best_pass_count} of {total} -> snapshot")
            self.snapshot(f"parity {curr_passed}/{total}")
            self._parity_regress_streak = 0
            return False
        else:
            self._log(f"parity unchanged: {curr_passed}/{total} == best {self.best_pass_count}")
            return False

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
        self.strategy_name = os.environ.get("KERNEL_AGENT_STRATEGY", "regular")
        self.strategy = make_strategy(self.strategy_name)

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
        # don't keep patch in the breadcrumbs, because model keeps the summary
        # of the changes in the kernel docstring
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
            # todo: but _llm_request_and_apply already catches errors twice -- no, it only catches patch application fails
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

        # Rollback manager: owns the lock-wins snapshot and pass-count tracking
        rollback = Rollback(bwd_fp, self.cfg.patience_parity_restore)

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
            self.patcher.remember("iteration", f"it={it}, shapes={shapes}")

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
                # Run parity in an isolated child process;
                # phase 1 (and beyond): run parity over the full SWEEP to enforce loop re-introduction;
                # tests/mamtul: backward casts to fp16 before dot and accumulates/atomics in fp16, while Torch grads accumulate in fp32;
                # later proper fix: keep accumulators fp32 and cast only at tl.atomic_add
                # todo-high: rm; too-high deltas
                ok, stats = run_gradcheck_child(fwd_fp, overwrite_fp=bwd_fp)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] gradcheck error: {err}")
                self.patcher.remember("gradcheck.error", err)
                _ = self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    # prefix every "fix" request with the current phase header. To keep model anchored on the phase goal
                    # while it fixes concrete errors. Important esp in Phase-1, since the raised backward is unrolled and SWEEP is multi‑shape
                    header=f"{self.strategy.get_header.strip()}\n\n{FIX_HEADER}",
                    state_facts={"runtime_error": {"stage": "gradcheck", "error": err}, "shapes": shapes},
                    temperature=0.35,
                )
                # retry correctness in next iteration
                continue
            if VERBOSE:
                print(f"[kernel-agent][it={it}] gradient_check ok={ok}, stats={stats}")

            # Breadcrumb: minimal
            self.patcher.remember("gradcheck", stats)

            was_restored = rollback.maybe_snapshot_or_restore(stats)

            if not ok:
                if was_restored:
                    # avoid showing stale errors to the model after a restore by skipping the fix prompt and
                    # advancing to re-run gradcheck on the restored kernel next iteration
                    #
                    # Restore happened this iteration: skip prompting the model with stale failures
                    # and immediately re-test on the restored kernel next iteration.
                    # Also reset perf patience counter so a prior non_improve streak doesn't trip early.
                    non_improve = 0  # reset perf patience after restore
                    if VERBOSE:
                        print(f"[kernel-agent][it={it}] Restore performed; skipping fix to re-test on restored kernel next iteration")
                    continue
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Parity failed on sweep — requesting 'fix' patch from LLM")
                _ = self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=f"{self.strategy.get_header.strip()}\n\n{FIX_HEADER}",
                    state_facts={"grad_summary": stats},
                    temperature=0.35,
                )
                # retry correctness in next iteration
                continue

            # Benchmark the current autograd op (backward), independent of optimize path
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Benchmarking backward")
            try:
                # Benchmark in an isolated child process
                cand = run_bench_child(fwd_fp, overwrite_fp=bwd_fp)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] bench error: {err}")
                self.patcher.remember("bench.error", err)
                _ = self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=f"{self.strategy.get_header.strip()}\n\n{FIX_HEADER}",
                    state_facts={"runtime_error": {"stage": "bench", "error": err}, "shapes": shapes},
                    temperature=0.35,
                )
                continue
            if VERBOSE:
                print(f"[kernel-agent][it={it}] bench: {cand}")

            # Simple plateau logic governed by cfg.patience and min_rel_improvement
            # Only consider performance when parity is full; otherwise ignore perf for acceptance
            full_parity = bool(stats.get("num_total", 0)) and (int(stats.get("num_passed", 0)) == int(stats.get("num_total", 0)))
            improved = full_parity and ((best_metrics is None) or (
                cand["median_ms"] <= (1.0 - self.cfg.min_rel_improvement) * best_metrics["median_ms"]
            ))

            if improved:
                best_metrics = cand
                non_improve = 0
                # lock perf improvement only under full parity
                rollback.snapshot("perf-improved (on full parity)")
            else:
                non_improve += 1
                # patience triggers only in full parity mode
                if full_parity and (non_improve >= self.cfg.patience_perf_stop):
                    stop_reason = "patience"
                    if VERBOSE:
                        print(f"[kernel-agent][it={it}] Early stop: patience reached (non_improve={non_improve})")
                    break


            # 3) Optimize step (toggleable to keep loop disentangled from policy)

            if VERBOSE:
                print(f"[kernel-agent][it={it}] Requesting 'optimize' patch from LLM")

            phase_text, temp = self.strategy.next_phase(parity_ok=ok, last_runtime=cand.get("median_ms"))
            changed = self._llm_request_and_apply(
                it, "optimize", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                header=phase_text,
                state_facts={"bench": cand, "grad_summary": stats},
                temperature=temp,
            )
            if not changed:
                continue

            if self.strategy_name == "regular":
                advance_changed = changed and ok
                self.strategy.advance(changed=advance_changed, parity_ok=ok)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] End iteration")

            elif self.strategy_name == "phased":
                advance_changed = changed and ok and self.strategy.verify_guardrails(bwd_fp)
                self.strategy.advance(changed=advance_changed, parity_ok=ok)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] End iteration")

            else:
                print(self.strategy_name)
                raise ValueError(f"Unreachable, mode should be either 'regular' or 'phased'. Got {self.strategy_name}")

        return {
            "best_metrics": best_metrics,
            "backward_fp": bwd_fp,
            "device_info": device,
            "stop_reason": stop_reason,
        }