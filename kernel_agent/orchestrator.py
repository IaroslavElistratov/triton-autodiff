from __future__ import annotations
from dataclasses import dataclass
import os, re, json, sys, shutil

import torch

from gpt_oss.tools.apply_patch import apply_patch as _apply_patch_raw
from .llm import _ensure_update_file_target  # normalize target header so model needn't guess file path
from .utils import _read_snippet, compile_kernel as create_op, UserError, _env_truthy
from .worker import run_gradcheck_child, run_bench_child
from .strategy import make_strategy, PhasedStrategy
from .worker import run_compile_child


VERBOSE = _env_truthy("KERNEL_AGENT_VERBOSE", "1")

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

        # let _llm_request_and_apply be the single point which appends these guardrails
        # Shared guardrails injected every turn (regular and phased)
        patch_header = (
            "\nPatch guardrails:\n"
            # Tool-only: patches must be delivered via functions.apply_patch. This aligns
            # model guidance with the sampler/orchestrator behavior and avoids analysis/final scraping.
            "- Exactly one patch: call functions.apply_patch({patch: ...}) once.\n"
            # Analysis is allowed for reasoning, but must not contain the tool call.
            # That's not needed strictly speaking but I think cleaner when model output tool call in the final channel
            "- Use analysis for planning only (no patcher tool call); call apply_patch once as your final action.\n"
            "- No rule echoing; diff only.\n"
            # note: attention kernel is about that size, to introduce for loop need to at least indent almost all of the lines in the kernel (around 120 lines)
            # "- ≤120 changed lines per patch.\n"
            "- Include at least one '-' anchor line per hunk.\n"
            "- Do NOT change the backward stub's signature.\n"
            "- Single backward kernel and single stub."
        )

        assert stage in ("fix", "optimize")
        if VERBOSE:
            print(f"[kernel-agent][it={it}] LLM phase='{stage}'")

        # context shown to LLM
        fwd_snip = _read_snippet(fwd_fp, self.cfg.snippet_max_lines)
        bwd_snip = _read_snippet(bwd_fp, self.cfg.snippet_max_lines)

        if temperature is not None:
            self.patcher.temperature = float(temperature)

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
                phase=header + patch_header,
                bwd_file=bwd_fp,
                fwd_kernel_snippet=fwd_snip,
                bwd_kernel_snippet=bwd_snip,
                state_facts=state_facts,
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
            fix_header = (
                header
                + "\nRETRY: Your previous patch failed to apply.\n"
                + f"apply_patch error:\n{err_msg}\n"
                + "Produce a corrected patch and call functions.apply_patch again. No prose.\n"
                + patch_header
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

    # todo-low: more descriptive name run_catch_errs_and_retry;
    # previously I wrapped create_op in _create_op_with_fix -- which allowed llm to fix it, but lost that functionality after i moved e.g. compile->bwd into child process,
    # so i expose the below method to "run and retry" which takes in a callable, and wraps it in some structure similar to _create_op_with_fix,
    # so that in the orchestrator's loop I can call self.run_with_fix(run_gradcheck_child) and then self.run_with_fix(run_bench_child) to restore compile-fix semantics
    def run_with_fix(self, it, fn, temperature, err_category):
        try:
            return True, fn()
        # raise only UserError, catch the rest of the errors
        except UserError as ue:
            # User-facing forward-file error: do not loop, surface to caller
            raise ue
        except Exception as ce:

            # for initial build (outside the optimization loop), bubble up
            if it is None:
                raise ce

            err = f"{type(ce).__name__}: {ce}"
            self.patcher.remember(err_category, err)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] {err_category}: {err}")

            self._llm_request_and_apply(
                it, "fix", bwd_fp=self.bwd_fp, fwd_fp=self.fwd_fp,
                # don't provide phase header for fixing exceptions
                header="Phase = fix. ONLY fix the exception.\n",
                state_facts={err_category: err}, # , "shapes": shapes
                temperature=temperature,
            )
            # _llm_request_and_apply also catches errors, but only for patch application fails;

            # catch here as well, because this create_op can independently error
            try:
                return True, fn()
            except UserError as ue2:
                raise ue2
            except Exception as ce_retry:
                err_retry = f"{type(ce_retry).__name__}: {ce_retry}"
                self.patcher.remember(err_category, err_retry)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] {err_category} on retry: {err_retry}")
                self._llm_request_and_apply(
                    it, "fix", bwd_fp=self.bwd_fp, fwd_fp=self.fwd_fp,
                    # don't provide phase header for fixing exceptions
                    header="Phase = fix. ONLY fix the exception.\n",
                    state_facts={err_category: err_retry}, # , "shapes": shapes
                    temperature=temperature,
                )
                return False, None

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

        # directly re-use api.py as otherwise i'd need to re-impl all the below funcs which i need
        # raise_to_triton_lang, load_raised_jit, wrap_bwd_kernel, DifferentiatedCompiledKernel, helper, autodiff
        if VERBOSE:
            print("[kernel-agent] Starting run")
            print(f"[kernel-agent] Forward file: {fwd_fp}")
            print("[kernel-agent] Compiling and tracing user kernel via create_op(...) (seed backward)")

        def run_create_op():
            # for consistency call this in a child process as well (run_compile_child) even though compiler generated bwd doesn't OOB;
            # and if it OOBs a failure here should terminate the program anyway (so the that safety around "run_compile_child" is redundant)
            return run_compile_child(fwd_fp, overwrite_fp=None)
        child_ran_ok, bwd_fp = self.run_with_fix(None, run_create_op, 0.25, "compile_error")
        self.bwd_fp = bwd_fp
        self.fwd_fp = fwd_fp

        # Rollback manager: owns the lock-wins snapshot and pass-count tracking
        rollback = Rollback(bwd_fp, self.cfg.patience_parity_restore)

        # solves the problem of not making any snapshot until a kernel finally passes all tests:
        # when llm is called, it can messup the kernel (pass rate 1/6 -> 0/6), in which case
        # rollback.maybe_snapshot_or_restore below will do nothing bc the first thing it will see is (0/6)
        rollback.snapshot("naive_backward")
        # without this the rollback logic doesn't count iterations regressed under tolerance;
        # todo: setting to 1 isn't general -- it's possible that for some kernels my naive grad will
        # fail (not guarantied that it will always pass one test (plus it also depends on the shapes
        # in the tests which users have). So alternatively, call gradcheck here followed by
        # rollback.maybe_snapshot_or_restore(stats) here
        rollback.best_pass_count = 1


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

            # todo-now: handle that logic where i run the first iteartion only on the 1st shape, i broke this logic whenre moved gradchekc and bench calls to children
            # sidecar = ns
            # if it == 0:
            #     sidecar = dict(ns)
            #     sidecar["SWEEP"] = ns["SWEEP"][:1]

            if VERBOSE:
                print(f"[kernel-agent][it={it}]") #  Inputs shapes={shapes}"

            # breadcrumb for LLM continuity
            self.patcher.remember("iteration", f"it={it}") # , shapes={shapes}

            # gradient_check uses autograd, its expectations are: my_op(*inputs) -> true outputs,
            # those outputs must be on a graph back to inputs. The stub satisfies this after @autodiff
            # because it runs the autograd‑wrapped kernel and returns the real outputs. The parity core
            # clones inputs, builds random upstreams, and compares torch.autograd.grad results per input.
            #
            # Grid handling remains in the stub, so gradient_check does not need to know meta params or shapes.
            # No change required to check_op_backward_parity.
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Running gradient_check (parity)")

            # Run parity in an isolated child process;
            # phase 1 (and beyond): run parity over the full SWEEP to enforce loop re-introduction;
            # tests/mamtul: backward casts to fp16 before dot and accumulates/atomics in fp16, while Torch grads accumulate in fp32;
            # later proper fix: keep accumulators fp32 and cast only at tl.atomic_add
            # todo-high: rm; too-high deltas
            def _run_gradcheck_child():
                return run_gradcheck_child(fwd_fp, overwrite_fp=bwd_fp)
            child_ran_ok, payload_gradcheck = self.run_with_fix(it, _run_gradcheck_child, 0.25, "gradcheck_error")
            if not child_ran_ok:
                continue
            # (grad_passed, grad_stats) can be just (None, ) don't assume it's a tuple
            parity_ok, grad_stats = payload_gradcheck
            if VERBOSE:
                print(f"[kernel-agent][it={it}] gradient_check ok={parity_ok}, grad_stats={grad_stats}")

            # Breadcrumb: minimal
            self.patcher.remember("gradcheck", grad_stats)

            was_restored = rollback.maybe_snapshot_or_restore(grad_stats)
            phase_text, temp = self.strategy.current_phase(parity_ok)

            # note "parity_ok" does not mean "if err in the child occurred", instead it means "if not full parity is achieved" (aka "if gracheck did't pass on full SWEEP")
            if not parity_ok:
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

                # when child returns ok (no err was raised) but parity_ok is False (some gradcheck test failed), call the fix prompt, then continue;
                # can't just rm this and just do "if not parity_ok: continue" bc that would just continue on parity failure (no LLM "fix" turn)
                # -- the kernel would never change
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Parity failed on sweep — requesting 'fix' patch from LLM")

                # Decide the fix prompt header:
                # - If we are in Phase 1, include the phase header bc Phase 1's explicit goal is to restore
                #   parity (by re-introducing loops). So add that header to anchor fixes to loop re-introduction.
                # - Otherwise (regular strategy or later phases), issue a correctness-only header
                #   to avoid confusing the model with optimization goals while fixing parity.
                is_phase_0 = isinstance(self.strategy, PhasedStrategy) and self.strategy.i == 0
                fix_header = (phase_text + "\n" if is_phase_0 else "") + "ONLY restore correctness to pass gradcheck.\n"

                self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=fix_header,
                    state_facts={"grad_summary": grad_stats},
                    # not using phase temp (temp) for fix prompts, fix turns should be conservative and stable
                    temperature=0.25,
                )
                # retry correctness in next iteration
                continue

            # Benchmark the current autograd op (backward), independent of optimize path
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Benchmarking backward")

            # Benchmark in an isolated child process
            def _run_bench_child():
                return run_bench_child(fwd_fp, overwrite_fp=bwd_fp)
            child_ran_ok, cand = self.run_with_fix(it, _run_bench_child, temp, "bench_error")
            if not child_ran_ok:
                continue

            if VERBOSE:
                print(f"[kernel-agent][it={it}] bench: {cand}")

            # Simple plateau logic governed by cfg.patience and min_rel_improvement
            # Only consider performance when parity is full; otherwise ignore perf for acceptance
            full_parity = bool(grad_stats.get("num_total", 0)) and (int(grad_stats.get("num_passed", 0)) == int(grad_stats.get("num_total", 0)))
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


            # comment:
            # i guess i can think of it that the only time phase header and constraints are shown in here
            # and all the previous llm calls were basically fixes in one from or another (e.g. patch fixes, grad-correctness fixes)
            changed = self._llm_request_and_apply(
                it, "optimize", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                header=phase_text,
                state_facts={"bench": cand, "grad_summary": grad_stats},
                temperature=temp,
            )
            if not changed:
                continue

            if self.strategy_name == "regular":
                # at this point both ok_gracheck and ok_bench and changed are all true
                self.strategy.advance(changed=changed, parity_ok=parity_ok)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] End iteration")

            elif self.strategy_name == "phased":
                # at this point both ok_gracheck and ok_bench and changed are all true
                # so just check for verify_guardrails
                advance_changed = self.strategy.verify_guardrails(bwd_fp)
                self.strategy.advance(changed=advance_changed, parity_ok=parity_ok)
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