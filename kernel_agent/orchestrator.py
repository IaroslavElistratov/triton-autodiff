from __future__ import annotations
from dataclasses import dataclass
import os, re, json, sys
import hashlib
from pathlib import Path

import torch

from gpt_oss.tools.apply_patch import apply_patch as _apply_patch_raw
from .utils import _read_snippet, compile_kernel as create_op, UserError, _env_truthy, save_file_bytes, restore_file_bytes, redact_torch_fn, filter_traceback_for_llm
from .worker import run_gradcheck_child, run_bench_child, run_compile_child
from .strategy import make_strategy
from .rollback import Rollback
from .tools.benchmark import PerfTracker
from .rag import _load_index, _embed_query, _cosine


VERBOSE = _env_truthy("KERNEL_AGENT_VERBOSE", "1")

def _read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""


@dataclass
class Config:
    max_iters: int = 6
    patience_perf_stop: int = 2  # stop loop after this many non-improving full-parity iterations (performance patience)
    patience_parity_restore: int = 2  # allow this many parity-regression iterations before restoring snapshot
    min_rel_improvement: float = 0.10   # require >= +10% throughput to accept
    # todo: a better way?
    snippet_max_lines: int = 400        # bound context shown to the LLM


# Runtime exceptions (gradcheck/bench/compile child fails): handled once per iteration in run_with_fix;
#   make one LLM fix attempt, do not re-run the child in the same iteration—let the main loop continue.
# Patch-apply errors: handled only inside _llm_request_and_apply, with one reprompt that includes the patcher error.
# Generation retries: live only in llm.py.

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

    # todo-now: don't ignore this fn's return status
    def _llm_request_and_apply(
        self,
        it: int,
        stage: str,
        *,
        bwd_fp: str,
        fwd_fp: str,
        header: str,
        state_facts: dict,
        temperature: float | None = None,
    ) -> bool:
        """Propose exactly one patch; if apply fails, retry once to **fix the same patch**.
        Consistent with "one LLM turn -> one attempt". Returns True iff file bytes changed.

        Separation of concerns:
          - llm.propose_patch: ensures a non-empty apply_patch.md block, normalizes target path,
            and distinguishes max_tokens vs generic no-patch. It may raise on generation failure.
          - _llm_request_and_apply (this): applies the patch via patcher, performs one
            apply-repair retry on failure, detects change via before/after bytes, and breadcrumbs errors.
            For patch application errors just rely on the patcher to raise an error.

        Proposes exactly one patch; if its apply fails, it tries to salvage **the same patch**
        once. The next outer iteration always makes a fresh proposal (no hidden carry‑over).
        This separation keeps state clean and rollback simple.
        """
        # Stage types:
        # - "init": Initial code generation after phase advance (no errors to fix yet)
        # - "fix": Fix existing code after validation failure
        # - "optimize": Performance optimization after correctness achieved
        assert stage in ("init", "fix", "optimize")
        if VERBOSE:
            print(f"[kernel-agent][it={it}] LLM phase='{stage}'")

        if temperature is not None:
            self.patcher.temperature = float(temperature)

        # llm.py already did a retry and raised/returned accordingly. Avoid duplicate generation retries here.
        # llm.py detects and classifies max-tokens, then raises; this try/catch in the orchestrator
        # just catches that exception so the loop doesn’t crash
        # todo: [cleanup] llm.py should be responsible for everyhting related to patch proposal and its
        # errors, do not spread that logic across both llm.py and this file. Remove these try/except around
        # self.patcher.propose_patch and make it responsibility of llm.py
        def _propose(header: str):
            try:
                patch = self.patcher.propose_patch(
                    phase=header,
                    strategy=self.strategy,
                    fwd_fp=fwd_fp,
                    bwd_fp=bwd_fp,
                    state_facts=state_facts,
                    it=it,
                )
                return patch, None
            except Exception as err_propose:
                # import traceback
                err_propose = f"{type(err_propose).__name__}: {err_propose}"
                self.patcher.remember("llm.propose.error", err_propose)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] propose error: {err_propose}")
                    # print(f"[kernel-agent][it={it}] Full traceback:\n{traceback.format_exc()}")
                return None, err_propose

        def _apply_once(patch) -> str | None:
            # the OSS patcher (gpt_oss.tools.apply_patch) calls apply_commit(...) -> write_file(...), which
            # opens the target with text mode "wt" and writes directly (no transaction/rollback). If an
            # exception occurs mid‑write, the file can be left truncated or partially written
            existed_before, prev_bytes = save_file_bytes(bwd_fp)
            err_apply: str | None = None
            try:
                _apply_patch_raw(patch)
            # don't shadow err_apply
            except Exception as exc:
                # restore to pre‑apply bytes to avoid leaving a partial file when apply fails midway
                restore_err = restore_file_bytes(bwd_fp, existed_before, prev_bytes)
                if restore_err and VERBOSE:
                    print(f"[kernel-agent][it={it}] restore error: {type(restore_err).__name__}: {restore_err}")

                err_apply = f"{type(exc).__name__}: {exc}"
                # rely on the patcher to surface validation errors at apply time;
                # no preflight checks; the patcher remains the source of truth
                self.patcher.remember("apply.error", err_apply)
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] apply error: {err_apply}")
            finally:
                # stash the result for inline finalize on the next LLM call
                self.patcher.finalize_last_tool_call(json.dumps({
                    "tool": "apply_patch",
                    "stage": stage,
                    "status": ("error" if err_apply else "applied"),
                    "error": err_apply,
                }))
            return err_apply

        # 1) Propose once
        patch, err_propose = _propose(header)
        # max_tokens
        if err_propose:
            return False

        # 2) Apply (with one repair attempt on apply error)
        before = _read_bytes(bwd_fp)
        err_apply = _apply_once(patch)

        # Perform one immediate apply-error retry inside _llm_request_and_apply;
        # if that also fails, return False and let the outer loop advance.
        # This salvages potentially a good patch using the exact apply error once, and keeps next
        # iteration a clean fresh proposal with no hidden carry-over.
        # comment:
        # Can semantically think that _llm_request_and_apply proposes a single patch and on re-try
        # attempts to fix **the same patch** (instead of proposing a separate independent patch)

        if err_apply:
            retry_header = (
                header
                + "\nRETRY: Your previous patch failed to apply. "
                "Fix the SAME diff below; do not start a new redesign; keep intent identical. "
                "Adjust anchors/context only if needed.\n"
                f"apply_patch error:\n{err_apply}\n"
                f"Previous patch (verbatim):\n{str(patch).strip()}\n"
                "Produce a corrected patch and call functions.apply_patch again. No prose.\n"
            )
            patch2, err2_propose = _propose(retry_header)
            if err2_propose:
                return False
            err2_apply = _apply_once(patch2)
            if err2_apply:
                return False

        # 3) Change detection
        after = _read_bytes(bwd_fp)
        changed = (after != before)
        # don't keep full patch in breadcrumbs to reduce token overhead
        self.patcher.remember(f"apply.{stage}", ("patch applied successfully" if changed else "no-change"))
        if VERBOSE:
            print(f"[kernel-agent][it={it}] {'patch applied successfully' if changed else 'no change'} in '{stage}'")
        return changed


    # expose the below method to "run and retry" which takes in a callable, so that in the orchestrator's loop I can call
    # self.run_with_fix(run_gradcheck_child) and then self.run_with_fix(run_bench_child) to restore compile-fix semantics
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

            # Extract and filter traceback before showing to LLM.
            # Worker errors (from compile/gradcheck/bench children) attach structured payload with full traceback.
            # Filter removes middleware frames (triton runtime, autodiff internals) keeping only:
            #   - User-controlled code (generated/ backward, test/ forward, tools/)
            #   - Last frame (actual error location, even if in internal code)
            # This gives LLM enough context to diagnose without 20+ lines of framework noise.
            #
            # Example filtering result (hypothetical):
            #   Before: 15 frames (worker.py, utils.py, triton runtime, autodiff, torch internals, generated code, utils.py)
            #   After:  3 frames (test/attention.py:269, generated/raised.py:5 @triton.jit, utils.py:187 raise)
            import traceback
            if hasattr(ce, 'worker_payload') and isinstance(ce.worker_payload, dict):
                payload = ce.worker_payload
                # Worker sent {"etype": "...", "emsg": "...", "traceback": "..."}
                if "traceback" in payload:
                    # Filter traceback to show only relevant frames (last frame + user-controlled paths)
                    filtered_tb = filter_traceback_for_llm(payload['traceback'])
                    err = f"{payload.get('etype', type(ce).__name__)}: {payload.get('emsg', str(ce))}\n\nTraceback:\n{filtered_tb}"
                else:
                    err = f"{payload.get('etype', type(ce).__name__)}: {payload.get('emsg', str(ce))}"
            else:
                # Regular exception (not from worker) - capture traceback here
                tb = "".join(traceback.format_exception(type(ce), ce, ce.__traceback__))
                # Filter traceback to show only relevant frames
                filtered_tb = filter_traceback_for_llm(tb)
                err = f"{type(ce).__name__}: {ce}\n\nTraceback:\n{filtered_tb}"

            self.patcher.remember(err_category, err)
            if VERBOSE:
                print(f"[kernel-agent][it={it}] {err_category}: {err}")

            # One fix attempt for this iteration. Do not re-run fn() here.
            self._llm_request_and_apply(
                it,
                "fix",
                bwd_fp=self.bwd_fp,
                fwd_fp=self.fwd_fp,
                header="Phase = fix. ONLY fix the exception.\n",
                state_facts={err_category: err},
                temperature=temperature,
            )

            # Defer re-test to next iteration.
            return False, None

    def run(self, *,
            fwd_fp: str,
            # benchmark,
            # profile,
            get_user_device_info,
            ) -> dict:

        assert self.strategy.name == "rag_adaptation"

        if not os.path.isfile(fwd_fp):
            raise FileNotFoundError(f"forward file not found: {fwd_fp}")

        # Set run start time for hook to distinguish within-run vs cross-run
        # Hook uses this to decide: regenerate skeleton (new run) vs preserve LLM edits (same run)
        import time
        os.environ["KERNEL_AGENT_START_TIME"] = str(time.time())

        # Initialize backward kernel using RAG retrieval
        if VERBOSE:
            print("[kernel-agent] Starting run")
            print(f"[kernel-agent] Forward file: {fwd_fp}")

        # Set fwd_fp early so run_with_fix can use it if it needs to call _llm_request_and_apply
        self.fwd_fp = fwd_fp
        self.bwd_fp = None  # Will be set after initialization

        # Backward kernel initialization: retrieve similar backward from RAG index
        # in api.py mlir passes are not called, because:
        #   1. Orchestrator writes retrieved backward to raised.py (here)
        #   2. First compile: compile_kernel(fwd_fp, overwrite_fp=raised_py)
        #     - bc overwrite_fp is set, MLIR generation is NOT called

        # RAG initialization: retrieve most similar backward kernel
        if VERBOSE:
            print("[kernel-agent] Using RAG to retrieve initial backward kernel")
            print("[kernel-agent] Strategy: RAGAdaptationStrategy (two-phase: reference then backward)")

        # Get the forward source for embedding
        fwd_source = redact_torch_fn(fwd_fp, None)
        if not fwd_source or not fwd_source.strip():
            raise ValueError(f"Forward kernel file is empty or could not be read: {fwd_fp}")

        # Retrieve most similar backward kernel using existing rag.py internal functions
        index_path = Path(__file__).parent / "kernel_embeddings.pkl"
        min_sim = float(os.environ.get("KERNEL_AGENT_RAG_MIN_SIM", "0.75"))

        # Load index and embed query (same logic as build_rag_block)
        try:
            embeddings, documents, backward_docs, file_list, openai_model = _load_index(str(index_path))
            query_embedding = _embed_query(fwd_source, model=openai_model)
        except Exception as e:
            raise ValueError(f"Failed to load RAG index or embed query: {e}")

        # Find best match above threshold
        best_match = None
        best_similarity = -1.0
        best_content = None
        for fp in file_list:
            sim = _cosine(query_embedding, embeddings[fp])
            if sim >= min_sim and sim > best_similarity:
                content = backward_docs.get(fp, "")
                if content:  # Only consider if backward exists
                    best_match = fp
                    best_similarity = sim
                    best_content = content

        if not best_match:
            raise ValueError(f"No similar kernels found in RAG index (similarity >= {min_sim:.2f}). "
                           "Try lowering --min-sim threshold.")

        # Store RAG FWD+BWD for use in Phase 2 (backward generation)
        # RAGAdaptationStrategy shows these only in Phase 2, not Phase 1
        retrieved_fwd = documents.get(best_match, "")
        retrieved_bwd = best_content

        if VERBOSE:
            print(f"[kernel-agent] Retrieved backward from '{best_match}' (similarity: {best_similarity:.3f})")
            print(f"[kernel-agent] Retrieved FWD: {len(retrieved_fwd)} chars, BWD: {len(retrieved_bwd)} chars")

        # Store RAG references on strategy (used in Phase 2 prompts)
        self.strategy.rag_fwd = retrieved_fwd
        self.strategy.rag_bwd = retrieved_bwd

        # Set backward file path (file will be generated by compile hook on first run)
        # Hook has access to compile_signature, so it can generate proper skeleton
        # with correct tensor parameters identified via call-site analysis
        digest = hashlib.sha256(fwd_source.encode()).hexdigest()[:10]
        gen_dir = f"generated/{digest}"
        os.makedirs(gen_dir, exist_ok=True)
        bwd_fp = f"{gen_dir}/raised.py"

        if VERBOSE:
            print(f"[kernel-agent] Backward file path: {bwd_fp}")
            print(f"[kernel-agent] File will be generated by compile hook with proper scaffolding")

        self.bwd_fp = bwd_fp

        # Rollback manager: owns the lock-wins snapshot and pass-count tracking
        rollback = Rollback(
            bwd_fp,
            self.cfg.patience_parity_restore,
            strategy=self.strategy,
            chain=self.patcher._sampler._chain,
        )
        # solves the problem of not making any snapshot until a kernel finally passes all tests:
        # when llm is called, it can messup the kernel (pass rate 1/6 -> 0/6), in which case
        # rollback.maybe_snapshot_or_restore below will do nothing bc the first thing it will see is (0/6)
        rollback.snapshot("naive_backward")

        # Set initial best_pass_count
        # RAG-retrieved kernels: don't assume anything - kernel might not even compile for this forward
        rollback.best_pass_count = 0

        # Accept baseline performance (>= 1.0) since they're already optimized
        # Retrieved kernels are already optimized, unlikely to improve significantly
        # Bar of 10% improvement (default min_rel_improvement) would reject all changes
        # Accept any change that doesn't regress performance
        min_improvement = 0.0
        tracker = PerfTracker(min_rel_improvement=min_improvement,
                              patience_perf_stop=self.cfg.patience_perf_stop)

        if VERBOSE:
            print(f"[kernel-agent] Initial backward path: {bwd_fp}")
            print(f"[kernel-agent] Using accept-eq policy: will accept >= 1.0x speedup (baseline performance)")


        device = get_user_device_info()
        stop_reason = "max_iters"

        # optimization loop
        for it in range(self.cfg.max_iters):
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Begin iteration")

            # RAGAdaptationStrategy never throttles SWEEP

            # only first shape on it==0, then all shapes;
            # running parity and bench across all entries after attempt 0 makes "loops re‑introduced"
            # observable and blocks phase advance until the same kernel passes on every shape;
            # the raised naive backward is unrolled and shape‑specialized; it will fail on varied
            # shapes until loops are restored

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
            if VERBOSE: print(f"[kernel-agent][it={it}] Running gradient_check (parity)")


            ###### parity check (fwd or bwd) ######


            # Run parity in an isolated child process;
            # phase 1 (and beyond): run parity over the full SWEEP to enforce loop re-introduction;
            # tests/mamtul: backward casts to fp16 before dot and accumulates/atomics in fp16, while Torch grads accumulate in fp32;

            if self.strategy.i == 0:
                # Phase 1: Validate PyTorch reference against Triton forward
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Phase 1: Validating PyTorch reference against Triton forward")

                # Only validate forward outputs
                os.environ["GRADCHECK_FORWARD_ONLY"] = "1"

                # Run validation (gradcheck will only check forward outputs)
                def _run_reference_validation():
                    return run_gradcheck_child(fwd_fp, overwrite_fp=bwd_fp)
                child_ran_ok, payload_gradcheck = self.run_with_fix(it, _run_reference_validation, 0.25, "reference_validation_error")
                if not child_ran_ok:
                    continue

                # Check if reference validation passed
                forward_match_ok, grad_stats = payload_gradcheck
                # In Phase 1, parity_ok means forward outputs match, not gradients
                parity_ok = forward_match_ok  # Keep variable name for compatibility

                if forward_match_ok:
                    self.strategy.pytorch_reference_validated = True
                    if VERBOSE:
                        print(f"[kernel-agent][it={it}] Phase 1: PyTorch reference forward outputs validated!")
                        print(f"[kernel-agent][it={it}] Advancing to Phase 2 for backward kernel generation...")
                else:
                    # Reset flag on validation failure (e.g., after rollback from Phase 2)
                    self.strategy.pytorch_reference_validated = False
                    if VERBOSE:
                        print(f"[kernel-agent][it={it}] Phase 1: PyTorch reference validation failed, will retry")

                grad_summary_text = ""

            else:
                # Normal gradcheck for Phase 2 or other strategies
                # Clear forward-only flag for Phase 2
                os.environ.pop("GRADCHECK_FORWARD_ONLY", None)

                def _run_gradcheck_child():
                    return run_gradcheck_child(fwd_fp, overwrite_fp=bwd_fp)
                child_ran_ok, payload_gradcheck = self.run_with_fix(it, _run_gradcheck_child, 0.25, "gradcheck_error")
                if not child_ran_ok:
                    continue
                # (grad_passed, grad_stats) can be just (None, ) don't assume it's a tuple
                parity_ok, grad_stats = payload_gradcheck
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] gradient_check ok={parity_ok}, grad_stats={grad_stats}")

                # Extract formatted summary_text for LLM prompt (gradcheck module pre-formatted it).
                # Pass only formatted text, not full dict, to reduce coupling with llm.py.
                grad_summary_text = grad_stats.get("summary_text", str(grad_stats))

            # Don't add gradcheck to history - it appears in state_facts when prompting,
            # and previous iterations' gradcheck results aren't useful for current fixes.

            was_restored = rollback.maybe_snapshot_or_restore(grad_stats)

            # Deferred phase advance gate: if last iteration applied a patch, only advance
            # now (before computing the phase header) if the current kernel passes the
            # per-phase parity threshold. This ensures we do not move to the next phase until
            # the patch (which as applied, with the current phase header, in the previous iteration)
            # is validated by gradcheck.
            # Solves advancing on stale parity and prompting with the wrong phase.
            # Do not advance immediately after _llm_request_and_apply (in the previous iteration),
            # using parity from the previous kernel, it misalignes prompts and flips SWEEP early.

            # Phase Advance Detection
            # Track phase transitions (e.g., Phase 1->2) to avoid using stale validation results
            # BUG FIX: Previously, we would terminate after Phase 2 using Phase 1's parity_ok
            # Now we detect advances and ensure fresh validation before termination decisions
            phase_before_advance = self.strategy.i  # TIMING: Phase BEFORE advance

            # this call may change self.strategy.i (e.g., 0->1)
            self.strategy.maybe_advance(bwd_fp, payload_gradcheck)

            # Check if we just advanced phases
            phase_after_advance = self.strategy.i  # TIMING: Phase AFTER advance
            phase_just_advanced = (phase_before_advance != phase_after_advance)

            # CRITICAL: Reset conversation chain when advancing from Phase 1 to Phase 2
            # Phase 1 shows the LLM how to write PyTorch reference implementation
            # Phase 2 must start with a FRESH conversation to avoid bias toward PyTorch code
            # Without this reset, the LLM remembers Phase 1 and writes pure PyTorch backward instead of Triton
            if phase_just_advanced and phase_after_advance == 1:
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Phase transition detected (0→1), resetting conversation chain to start fresh")
                self.patcher._sampler._chain.restore_anchor(None)

            phase_text, temp = self.strategy.current_phase(parity_ok)

            # RAG-only mode: stop immediately after parity is achieved IN PHASE 2
            # Retrieved kernels are already optimized - once adapted to pass gradcheck, no further optimization needed
            # Only terminate if we're in Phase 2 (backward generation), not Phase 1 (reference validation)
            # Don't terminate using stale parity_ok after phase advance!
            # TIMING: This reflects phase AFTER advance (may have just changed above)
            #
            # CORNER CASE: Phase Just Advanced
            # SEMANTIC CONFUSION without phase_just_advanced tracking:
            #   parity_ok=True means "Phase 1 passed" (BEFORE advance)
            #   in_phase2=True means "now in Phase 2" (AFTER advance)
            #   -> Would misinterpret as "Phase 2 passed" and terminate prematurely!
            # When phase advances (e.g., 0->1), parity_ok is from OLD phase (forward validation)
            # We must NOT terminate using stale results - need to generate code for NEW phase first
            #
            # if just advanced phases - parity_ok is from the previous phase!
            # Will call LLM to generate initial code for new phase, then validate
            # Skip termination check (parity_ok is stale from previous phase)
            in_phase2 = not (self.strategy.i == 0)

            if parity_ok and in_phase2 and not phase_just_advanced:
                # TIMING: Safe to terminate - parity_ok and in_phase2 are from same context (no advance)
                # Both refer to Phase 2: validation passed AND still in Phase 2
                # Only terminate if we're in Phase 2 AND we have fresh Phase 2 validation results
                stop_reason = "rag_parity_achieved"
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] RAG adaptation complete - parity achieved on all SWEEP shapes (backward gradients validated)")
                break

            # Previously skipped LLM call after phase advance, causing immediate gradcheck
            # on unimplemented backward stub. Now correctly generate initial code for new phases.
            # Call LLM after phase advance (to generate initial code) OR after parity failure (to fix)
            # Note: "parity_ok" means either "fwd parity" or "backward" (gradcheck passed on all SWEEP shapes)
            if not parity_ok or phase_just_advanced:

                if was_restored:
                    # reset perf patience counter after rollback
                    tracker.reset_patience()
                    if VERBOSE:
                        print(f"[kernel-agent][it={it}] Restore performed; skipping fix to re-test on restored kernel next iteration")

                    # avoid showing stale errors to the model after a restore by skipping the fix prompt and
                    # advancing to re-run gradcheck on the restored kernel next iteration;
                    # Restore happened this iteration: skip prompting the model with stale failures
                    # and immediately re-test on the restored kernel next iteration
                    continue

                # Two LLM call scenarios:
                # (1) Initial generation after phase advance - no gradcheck results yet (use "init" stage)
                # (2) Fix after parity failure - include gradcheck error summary (use "fix" stage)

                # when child returns ok (no err was raised) but parity_ok is False (some gradcheck test failed), call the fix prompt, then continue;
                # can't just rm this and just do "if not parity_ok: continue" bc that would just continue on parity failure (no LLM "fix" turn)
                # -- the kernel would never change
                self._llm_request_and_apply(
                    it,
                    "init" if phase_just_advanced else "fix",
                    bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=phase_text if phase_just_advanced else phase_text + "\nONLY restore correctness to pass gradcheck.\n",
                    state_facts={"grad_summary": grad_summary_text},
                    # not using phase temp (temp) for fix prompts, fix turns should be conservative and stable
                    temperature=0.25,
                )
                # retry correctness in next iteration
                continue


            ###### benchmark ######


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

            # Feed per-sweep reducer output to the tracker:
            # - latest_metrics: last sweep + per-shape speedups and geomean vs best
            # - best_metrics: accepted snapshot + ever_max_tflops
            decision = tracker.update(cand, grad_stats, it=it)
            if decision["improved"]:
                # lock perf improvement only under full parity; persist strategy phase index
                rollback.snapshot("perf-improved (on full parity)")
            elif tracker.stop_reason:
                stop_reason = tracker.stop_reason
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Early stop {stop_reason}")
                break


            ###### phase specific prompt ######


            # if VERBOSE: print(f"[kernel-agent][it={it}] Requesting 'optimize' patch from LLM")

            # # i guess i can think of it that the only time phase header and constraints are shown in here
            # # and all the previous llm calls were basically fixes in one from or another (e.g. patch fixes, grad-correctness fixes)
            # # Extract formatted summary_text for LLM prompt (gradcheck module pre-formatted it).
            # # Pass only formatted text, not full dict, to reduce coupling with llm.py.
            # grad_summary_text = grad_stats.get("summary_text", str(grad_stats))

            # changed = self._llm_request_and_apply(
            #     it, "optimize", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
            #     header=phase_text,
            #     state_facts={"bench": cand, "grad_summary": grad_summary_text},
            #     temperature=temp,
            # )
            # if not changed:
            #     continue

            # NOTE: any gradcheck stats are now stale (right after the patch was applied above)

            if VERBOSE: print(f"[kernel-agent][it={it}] End iteration")

        return {
            "best_metrics": tracker.best_metrics,
            "latest_metrics": tracker.latest_metrics,
            "backward_fp": bwd_fp,
            "device_info": device,
            "stop_reason": stop_reason,
        }
