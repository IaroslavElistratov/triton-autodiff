import os
import shutil

from .utils import _env_truthy


VERBOSE = _env_truthy("KERNEL_AGENT_VERBOSE", "1")

class Rollback:
    """Lock-wins snapshot manager for the backward kernel file.

    - Snapshots the current kernel to a sidecar file on improved parity/perf
    - Restores from that snapshot on plateau/regress (e.g., patience stop)
    - Tracks best parity coverage across iterations
    """
    def __init__(self, backward_fp: str, patience_parity_restore: int = 0, *, strategy, chain) -> None:
        self.backward_fp = backward_fp
        self.lock_fp = f"{backward_fp}.lock"
        self.best_pass_count: int = 0
        self._parity_regress_streak: int = 0
        self._patience_parity_restore: int = patience_parity_restore
        # Strategy reference (used to persist/restore phase index)
        self._strategy = strategy
        # Track the strategy phase index saved alongside the lock snapshot
        self._saved_phase_index: int | None = None
        self._log(
            f"init: patience_parity_restore={self._patience_parity_restore} | lock={self.lock_fp}"
        )

        ### handle previous_response_id
        # Track the OpenAI Responses chain state (prev_response_id) alongside the lock snapshot
        self._saved_prev_response_id: str | None = None
        # ChainState, allows to call get_prev_response_id
        self._chain = chain

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
            # Remember the strategy phase index at snapshot time for later restore
            if self._strategy.name == "phased":
                self._saved_phase_index = self._strategy.i
            elif self._strategy.name == "rag_adaptation":
                self._saved_phase_index = self._strategy.i
            # don't snapshot any pending tool call ("PendingTool") -- which is unfinalized function_call
            # that must be acknowledged on very next API request via function_call_output.
            # Because i snapshot code after the tool has been already applied, no need
            # to snapshot ephemeral pending call/output;
            # Also capture the current prev_response_id via chain -- anchors the server-side
            # convo to this finalized turn so a future restore can chain from the same
            # hidden state (avoids stale context)
            if os.environ.get("KERNEL_AGENT_BACKEND") == "openai" and self._chain is not None:
                self._saved_prev_response_id = self._chain.get_prev_anchor()
        except Exception as e:
            self._log(f"warning: snapshot failed: {type(e).__name__}: {e}")

    def _restore(self) -> None:
        """Restore kernel from the last snapshot, if present, and optionally
        restore the strategy phase to the value saved at snapshot time.
        Keeping strategy rewinding here ensures kernel bytes and strategy phase
        remain aligned when a restore occurs.
        """
        try:
            if os.path.isfile(self.lock_fp):
                shutil.copyfile(self.lock_fp, self.backward_fp)
                self._log(f"restore: {self.lock_fp} -> {self.backward_fp}")
                # If we saved a phase index, restore it now so policy state matches the restored kernel
                # if self._strategy.name == "phased":
                #     # Drop any pending phase advance request (stale kernel)
                #     self._strategy.pending_advance_from = None
                #     self._strategy.set_phase_index(self._saved_phase_index)
                #     if VERBOSE:
                #         print(f"[kernel-agent][rollback] strategy phase restored to i={self._saved_phase_index}")
                if self._strategy.name == "rag_adaptation":
                    # Restore phase index and reset validation flag if rolling back to Phase 1
                    self._strategy.set_phase_index(self._saved_phase_index)
                    if self._saved_phase_index == 0:
                        self._strategy.pytorch_reference_validated = False
                    if VERBOSE:
                        print(f"[kernel-agent][rollback] RAGAdaptationStrategy phase restored to i={self._saved_phase_index}")
                        if self._saved_phase_index == 0:
                            print(f"[kernel-agent][rollback] Reset pytorch_reference_validated flag on rollback to Phase 1")
                # clears any pending and sets prev_response_id to the saved anchor so the
                # next turn starts clean, avoiding stale function_call_output acks against the wrong turn.
                if os.environ.get("KERNEL_AGENT_BACKEND") == "openai" and self._chain is not None:
                    self._chain.restore_anchor(self._saved_prev_response_id)
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

        # lock phase 0 -> phase 1 even if same number of gradcheck passed shapes
        # (bc phase 1 is more readable kernel, so worth locking)
        is_higher_phase = (self._saved_phase_index is not None) and (self._strategy.i > self._saved_phase_index)
        same_count_but_higher_phase = (curr_passed == self.best_pass_count) and is_higher_phase

        # note: not counting _parity_regress_streak when curr_passed == self.best_pass_count -- potentially should?

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
        elif (curr_passed > self.best_pass_count) or same_count_but_higher_phase:
            prev = self.best_pass_count
            self.best_pass_count = curr_passed
            self._log(f"parity improve: best {prev} -> {self.best_pass_count} of {total} -> snapshot")
            self.snapshot(f"parity {curr_passed}/{total}")
            self._parity_regress_streak = 0
            return False
        else:
            self._log(f"parity unchanged: {curr_passed}/{total} == best {self.best_pass_count}")
            return False
