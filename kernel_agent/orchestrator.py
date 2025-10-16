from __future__ import annotations
from dataclasses import dataclass
import os, re, json, sys
import hashlib
from pathlib import Path

import torch

from gpt_oss.tools.apply_patch import apply_patch as _apply_patch_raw
from .utils import _read_snippet, compile_kernel as create_op, UserError, _env_truthy, save_file_bytes, restore_file_bytes, redact_torch_fn, filter_traceback_for_llm
from .worker import run_gradcheck_child, run_bench_child, run_compile_child
from .strategy import make_strategy, PhasedStrategy
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
    use_compiler: bool = True           # use MLIR compiler to generate initial backward
    use_rag: bool = False           # use RAG retrieval for initial backward (when compiler is False)


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
        assert stage in ("fix", "optimize")
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

        if not os.path.isfile(fwd_fp):
            raise FileNotFoundError(f"forward file not found: {fwd_fp}")


        # Initialize backward kernel using either compiler or RAG retrieval
        # TODO: (future) Consider seed abstraction (--seed {auto,rag,compiler,user,none})
        # where "auto" tries RAG first with fallback to compiler, "user" allows custom
        # initial kernel, and "none" starts from scratch. Current dual-boolean approach
        # works but seed abstraction would be cleaner and more extensible.
        if VERBOSE:
            print("[kernel-agent] Starting run")
            print(f"[kernel-agent] Forward file: {fwd_fp}")

        # Set fwd_fp early so run_with_fix can use it if it needs to call _llm_request_and_apply
        self.fwd_fp = fwd_fp
        self.bwd_fp = None  # Will be set after initialization

        # Backward kernel initialization: select method based on flags
        # - --rag alone: Use RAG to retrieve similar backward as starting point
        # - --compiler alone: Use MLIR compiler to generate backward
        # - --rag --compiler: Use compiler for initialization; RAG provides prompt augmentation only
        #   (retrieved backward is shown to LLM as reference via KERNEL_AGENT_RAG_PROMPTS in llm.py,
        #    but compiler-generated backward is used as the actual starting kernel for optimization)
        if self.cfg.use_rag and not self.cfg.use_compiler:

            # in api.py mlir passes compiler is not called, because:
            #   1. Orchestrator writes retrieved backward to raised.py (here)
            #   2. First compile: compile_kernel(fwd_fp, overwrite_fp=raised_py)
            #     - so because overwrite_fp is set, compiler is NOT called

            # RAG initialization: retrieve most similar backward kernel
            if VERBOSE:
                print("[kernel-agent] Using RAG to retrieve initial backward kernel")
                print("[kernel-agent] Strategy: RAGAdaptationStrategy (single adaptation phase)")

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
                               "Try lowering --rag-min-sim or use --compiler instead.")

            # Store RAG FWD+BWD for use in prompt (not in file)
            # RAG backward will be shown as readonly reference, not edited directly
            retrieved_fwd = documents.get(best_match, "")
            retrieved_bwd = best_content

            if VERBOSE:
                print(f"[kernel-agent] Retrieved backward from '{best_match}' (similarity: {best_similarity:.3f})")
                print(f"[kernel-agent] Retrieved FWD: {len(retrieved_fwd)} chars, BWD: {len(retrieved_bwd)} chars")

            # Store RAG references on strategy (used for prompt reference section)
            self.strategy.rag_fwd = retrieved_fwd
            self.strategy.rag_bwd = retrieved_bwd

            # Generate initial raised.py with backward stub skeleton
            # Forward will be prepended by compile_kernel hook during first gradcheck
            # LLM will write backward for USER's forward, using RAG as reference
            initial_content = self.strategy.generate_initial_file(fwd_source)

            digest = hashlib.sha256(fwd_source.encode()).hexdigest()[:10]
            gen_dir = f"generated/{digest}"
            os.makedirs(gen_dir, exist_ok=True)
            bwd_fp = f"{gen_dir}/raised.py"

            with open(bwd_fp, "w") as f:
                f.write(initial_content)

            if VERBOSE:
                print(f"[kernel-agent] Wrote backward stub skeleton to: {bwd_fp}")
                print(f"[kernel-agent] Forward will be prepended automatically during first gradcheck")

            self.bwd_fp = bwd_fp

        elif self.cfg.use_compiler:
            # TTIR from autodiff then raise to Python once; use as seed and target
            # using output of triton-autodiff directly as the initial version of the backward kernel
            # to be optimized -- "seeding a problem with a draft" (removing patcher.naive_autodiff instead
            # just using output of triton-autodiff as patcher.kernel_snippet)
            if VERBOSE:
                print("[kernel-agent] Using MLIR compiler to generate initial backward kernel")

            def run_create_op():
                # for consistency call this in a child process as well (run_compile_child) even though compiler generated bwd doesn't OOB;
                # and if it OOBs a failure here should terminate the program anyway (so the that safety around "run_compile_child" is redundant)
                return run_compile_child(fwd_fp, overwrite_fp=None)
            child_ran_ok, bwd_fp = self.run_with_fix(None, run_create_op, 0.25, "compile_error")
            self.bwd_fp = bwd_fp

        else:
            # Should not reach here due to validation in main.py
            raise ValueError("No initialization method specified (need --compiler or --rag)")

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

        # Set initial best_pass_count based on initialization method
        # Compiler-generated kernels: assume at least 1 test passes (unrolled kernel typically works for single shape)
        # RAG-retrieved kernels: don't assume anything - kernel might not even compile for this forward
        if self.cfg.use_rag and not self.cfg.use_compiler:
            # RAG-only initialization - retrieved kernel may need adaptation before it passes tests
            rollback.best_pass_count = 0
        else:
            # Compiler initialization (with or without RAG prompt augmentation)
            rollback.best_pass_count = 1

        # For RAG-initialized kernels, accept baseline performance (>= 1.0) since they're already optimized
        # For compiler-generated kernels, require the configured improvement threshold
        rag_init = self.cfg.use_rag and not self.cfg.use_compiler
        # for rag_init: after parity, accept any change that doesn't regress performance
        #  - retrieved kernels are already optimized, unlikely to improve significantly
        #  - bar of 10% improvement (default min_rel_improvement) would reject all changes
        min_improvement = 0.0 if rag_init else self.cfg.min_rel_improvement
        tracker = PerfTracker(min_rel_improvement=min_improvement,
                              patience_perf_stop=self.cfg.patience_perf_stop)

        if VERBOSE:
            print(f"[kernel-agent] Initial backward path: {bwd_fp}")
            if rag_init:
                print(f"[kernel-agent] Using accept-eq policy: will accept >= 1.0x speedup (baseline performance)")
            else:
                print(f"[kernel-agent] Requiring {self.cfg.min_rel_improvement:.0%} improvement for acceptance")


        device = get_user_device_info()
        stop_reason = "max_iters"

        # optimization loop
        for it in range(self.cfg.max_iters):
            if VERBOSE:
                print(f"[kernel-agent][it={it}] Begin iteration")

            # todo: move this inside PhasedStrategy
            # Phase-0: throttle SWEEP to the first shape in children;
            # RAGAdaptationStrategy never throttles since it has name="rag_adaptation"
            is_readability_phase = self.strategy.name == "phased" and self.strategy.i == 0
            if is_readability_phase:
                os.environ["KERNEL_AGENT_SWEEP_LIMIT"] = "1"
            else:
                os.environ.pop("KERNEL_AGENT_SWEEP_LIMIT", None)

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

            # Store only formatted text in history to avoid duplication with state_facts.
            # The summary_text already contains all important info (errors, per-input details).
            # state_facts will show current iteration's formatted summary, so history doesn't need raw dict.
            self.patcher.remember("gradcheck", grad_stats.get("summary_text", str(grad_stats)))

            was_restored = rollback.maybe_snapshot_or_restore(grad_stats)

            # Deferred phase advance gate: if last iteration applied a patch, only advance
            # now (before computing the phase header) if the current kernel passes the
            # per-phase parity threshold. This ensures we do not move to the next phase until
            # the patch (which as applied, with the current phase header, in the previous iteration)
            # is validated by gradcheck.
            # Solves advancing on stale parity and prompting with the wrong phase.
            # Do not advance immediately after _llm_request_and_apply (in the previous iteration),
            # using parity from the previous kernel, it misalignes prompts and flips SWEEP early.
            self.strategy.maybe_advance(bwd_fp, payload_gradcheck)

            phase_text, temp = self.strategy.current_phase(parity_ok)

            # RAG-only mode: stop immediately after parity is achieved
            # Retrieved kernels are already optimized - once adapted to pass gradcheck, no further optimization needed
            if parity_ok and rag_init:
                stop_reason = "rag_parity_achieved"
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] RAG adaptation complete - parity achieved on all SWEEP shapes")
                break

            # note "parity_ok" does not mean "if err in the child occurred", instead it means "if not full parity is achieved" (aka "if gracheck did't pass on full SWEEP")
            if not parity_ok:
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

                # when child returns ok (no err was raised) but parity_ok is False (some gradcheck test failed), call the fix prompt, then continue;
                # can't just rm this and just do "if not parity_ok: continue" bc that would just continue on parity failure (no LLM "fix" turn)
                # -- the kernel would never change
                if VERBOSE:
                    print(f"[kernel-agent][it={it}] Parity failed on sweep — requesting 'fix' patch from LLM")

                # Decide the fix prompt header:
                # - Phase 1 (loops): include phase header since the explicit goal is parity via loop re-introduction
                # - RAG adaptation: include phase header since adaptation is the core goal
                # - Otherwise (regular strategy or other phases): issue correctness-only header to avoid confusing
                #   the model with optimization goals while fixing parity
                is_loop_phase = self.strategy.name == "phased" and self.strategy.i == 1
                is_rag_adaptation = self.strategy.name == "rag_adaptation"
                include_phase = is_loop_phase or is_rag_adaptation
                fix_header = (phase_text + "\n" if include_phase else "") + "ONLY restore correctness to pass gradcheck.\n"

                # Extract formatted summary_text for LLM prompt (gradcheck module pre-formatted it).
                # Pass only formatted text, not full dict, to reduce coupling with llm.py.
                grad_summary_text = grad_stats.get("summary_text", str(grad_stats))

                self._llm_request_and_apply(
                    it, "fix", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                    header=fix_header,
                    state_facts={"grad_summary": grad_summary_text},
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


            # 3) Optimize step (toggleable to keep loop disentangled from policy)

            if VERBOSE:
                print(f"[kernel-agent][it={it}] Requesting 'optimize' patch from LLM")


            # comment:
            # i guess i can think of it that the only time phase header and constraints are shown in here
            # and all the previous llm calls were basically fixes in one from or another (e.g. patch fixes, grad-correctness fixes)
            # Extract formatted summary_text for LLM prompt (gradcheck module pre-formatted it).
            # Pass only formatted text, not full dict, to reduce coupling with llm.py.
            grad_summary_text = grad_stats.get("summary_text", str(grad_stats))

            changed = self._llm_request_and_apply(
                it, "optimize", bwd_fp=bwd_fp, fwd_fp=fwd_fp,
                header=phase_text,
                state_facts={"bench": cand, "grad_summary": grad_summary_text},
                temperature=temp,
            )
            if not changed:
                continue

            # NOTE: any gradcheck stats are now stale (right after the patch was applied above)

            # defers the actual phase increment until i see fresh gradcheck/bench in the next loop
            # (on the post-optimize kernel). Record a pending request only if a patch landed.
            # The actual advance will occur at the start of the next iteration when gradcheck passes the per-phase gate
            self.strategy.pending_advance_from = self.strategy.i

            if VERBOSE:
                print(f"[kernel-agent][it={it}] End iteration")

        return {
            "best_metrics": tracker.best_metrics,
            "latest_metrics": tracker.latest_metrics,
            "backward_fp": bwd_fp,
            "device_info": device,
            "stop_reason": stop_reason,
        }
