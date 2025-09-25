from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import os

from .utils import _env_truthy


# Default verbose ON unless explicitly disabled
VERBOSE = _env_truthy("KERNEL_AGENT_VERBOSE", "1")

# Strategy module: encapsulates per-step instructions and temperature, so the
# orchestrator can toggle this behavior without changing its main loop.

@dataclass(frozen=True)
class Phase:
    name: str
    goal: str
    guardrails: str
    temp: float


# Default phase sequence — light guidance per step; can be replaced/tuned.
PHASES: Sequence[Phase] = (
    Phase(
        "0 / Readability",

        # Prioritize high‑impact edits:
        # split long expressions into named steps and factor repeated subexpressions;
        # group code into ordered sections with one‑line headers: indexing -> block pointers -> loads -> forward compute -> grads -> atomics;
        # compute and reuse base block pointers once per tensor;
        # rename opaque temporaries with consistent semantic names;
        # hoist and reuse casts and repeated offset products into named base offsets;
        # remove dead or unused intermediates;
        # replace magic numbers with named tl.constexpr or locals and comment intent (BM, BN, SCALE=0.7213475108146667, NEG_INF);
        # collapse trivial reshape/broadcast churn only when semantics are identical; drop redundant tl.cast only where it cannot meaningfully change precision;
        # keep accumulation dtypes and cast boundaries exactly as in the input;
        # you can modify _mk_block_ptr for readability as well.
        # add minimal docstring/comments
        #
        # No new Python loops, no control flow changes, no API changes, no moving ops across data dependencies, no reordering of reductions.

        "Refactor the Triton kernel for readability only. Preserve exact math and memory semantics: same function/helper names and signatures, @triton.jit, program_id axis mapping, strides/indexing algebra, tile sizes, reduction axes and order, IO dtypes, memory‑access order, and every tl.atomic_* call. No new control flow, helpers, or reordering across data dependencies. High‑impact edits only: rename opaque temporaries to semantic names; hoist and reuse base offsets/casts; compute base block pointers once; replace magic numbers with named constants; split long expressions and factor repeats; group into sections (indexing -> pointers -> loads -> forward compute -> grads -> atomics); remove dead intermediates; keep dtype/cast boundaries unchanged. If uncertain, keep the original. Output the cleaned code.",
        # "Improve Readability without changing semantics. E.g. you can: rename variables; split long expressions; hoist constants; reorder independent statements; delete dead code; add comments/docstring. No math or memory-access semantics change.",
        "Stub & kernel signatures unchanged; no new Python loops; no change in tl.atomic_* usage.",
        0.15,
    ),
    # Add tail masks where appropriate.
    Phase("1 / Re-introduce loops",
          "The initial backward kernel covers the gradients for exactly one iteration of the original forward loop (loop flattened). You should re-introduce back the for-loops in the backward kernel, as it'll generalize the backward kernel to multi-tiled shapes and allow to pass the gradcheck.\n",
          "Re-introduce loops. Keep atomics. Keep pointer math.",
          0.25,
    ),

    Phase(
        "2 / Atomics->private",
        # "Do NOT blindly swap atomics for direct store, remove atomics while preserving numerics semantics: privatize accumulation per CTA and write once per output tile. Potentially, adjust grid/tiling or add an explicit reduction; do not loop over the wrong axis. No other unrelated kernel changes.",
        # "Privatize accumulators per CTA. One write per output tile. Remove atomics by privatizing accumulation within a CTA and writing each output tile once after a local reduction. ",
        "Privatize accumulators per CTA and write each output tile once after a local reduction. Remove atomics by local accumulation. You may change tiling/parallelization or add an explicit reduction. " +
        "Do not drop required reductions, reduce over the wrong axis, or swap atomics for direct stores. Preserve numerics across all test shapes (gradcheck must pass). No unrelated changes.\n",
        (
            "Checklist:\n"
            "- One write per output tile after a per-CTA reduction.\n"
            "- No direct-store swaps in place of atomics.\n"
            "- Ensure reducing over correct axis.\n"
            "- Tiling/parallelization/grid changes allowed; explicit reduction allowed.\n"
            "- Gradcheck must pass across the sweep.\n"
            "- No unrelated changes.\n"
        ),
        0.5,
    ),
    # todo: implement guardrails_check_phase3
    # todo-high: make it an open-end goal instead? since i can't verify "Coalesce loads/stores" anyway
    Phase("3 / Coalesce/layout", "Coalesce loads/stores; adopt tl.make_block_ptr; adjust tile shapes.", "Algorithm unchanged.", 0.5),
    # Phase("4 / Meta tune", "Sweep BLOCK_SIZE_{M,N,K}, num_warps, num_stages.", "Emit one patch per turn.", 0.25),
    # todo: requires ability to change the fwd kernel
    # Phase("5 / Recompute vs read", "Recompute-vs-read forward intermediates.", "No new atomics.", 0.25),
)

class BaseStrategy:
    def next_phase(self, parity_ok: bool) -> Tuple[str, float]:
        # Default: a simple optimize header with global guardrails and neutral temperature
        return "Phase = optimize.\n", 0.7

    def maybe_advance(self, bwd_fp, payload_gradcheck) -> None:
        # No-op in the base class.
        pass

class RegularStrategy(BaseStrategy):
    def __init__(self, temp: float = 0.7) -> None:
        self.name = "regular"
        self._temp = temp

    # added this method to be compatible with PhasedStrategy.get_header
    # because orchestrator unconditionally calls self.strategy.get_header
    @property
    def get_header(self):
        return "Phase = optimize. Improve performance without changing numerics."

    def current_phase(self, parity_ok: bool) -> Tuple[str, float]:
        header = (
            "Phase = optimize. Improve performance without changing numerics.\n"
            " * atomics: kernel uses atomics -- try privatizing the accumulation to the same memory location to a single CTA to avoid atomics, as it'll clearly improve the performance. "
        )
        return header, self._temp

class PhasedStrategy(BaseStrategy):
    def __init__(self, phases: Sequence[Phase] = PHASES) -> None:
        self.name = "phased"
        self.phases = list(phases)
        self.i = 0
        # Deferred phase-advance request recorded when an optimize patch is applied.
        # Only advance at the beginning of the NEXT iteration after seeing gradcheck
        # for the post-optimize kernel. This keeps prompts in the old phase until the
        # patch proves itself on the subsequent gradcheck (avoids advancing on stale
        # pre-optimize parity and misguiding the next LLM turn)
        self.pending_advance_from: int | None = None

    # @property
    # def get_header(self):
    #     return self.phases[self.i].goal

    def set_phase_index(self, i: int) -> None:
        """Force strategy index to a given phase (used on rollback restore)."""
        self.i = i
        if VERBOSE:
            print(f"[kernel-agent] Strategy phase restored to i={i}")

    def current_phase(self, parity_ok: bool) -> Tuple[str, float]:
        # emit a concise header describing the allowed scope for this step.
        p = self.phases[self.i]
        header = (
            f"Phase = {p.name}. Phase Goals = {p.goal}.\n"
            f"Success Gates: gradcheck_ok={parity_ok}.\n"
            f"Phase Guardrails: {p.guardrails}\n"
        )
        return header, p.temp

    def _verify_guardrails(self, backward_fp: str) -> bool:
        """Return True if current phase-specific guardrails are satisfied.

        comment:
        **Assumption: guardrails run after the LLM patch was applied for the current phase.**
        'pending_advance_from' ensures that this is the case.

        Phase 1 -> require reintroduced loops (guardrails_check_phase1)
        Phase 2 -> require atomics removed (guardrails_check_phase2)
        Other phases -> currently no extra checks
        """
        checks = {
            0: guardrails_check_phase0,
            1: guardrails_check_phase1,
            2: guardrails_check_phase2,
            3: guardrails_check_phase3,
        }
        # todo-high:
        # test all guardrails because a recent model patch can violate older (previously passing) guardrails,
        # if add this don't need the phase save and restore on rollback functionality
        fn = checks.get(self.i)
        ok = fn(backward_fp)
        return ok

    def maybe_advance(self, bwd_fp, payload_gradcheck) -> None:
        """Advance one phase only after post-optimize gradcheck proves the patch.
        - Uses parity from the current iteration (post-optimize kernel)
        - Require phase-specific code guardrails and parity guardrails
        """

        if self.pending_advance_from is None:
            if VERBOSE:
                print(f"[kernel-agent] Phase maybe_advance was called but pending_advance_from is None. Not advancing.")
            return

        # cannot be not None and not self.i
        assert self.pending_advance_from == self.i, "Unreachable"

        _is_full_parity, grad_stats = payload_gradcheck

        # parity guardrails
        passed = int(grad_stats.get("num_passed", 0))
        total  = int(grad_stats.get("num_total", 0))

        # Phase-specific parity thresholds for advancement
        # Phase-0: allow advance if >=1 shape passes; later phases require full sweep.
        parity_gate = (passed >= 1) if self.i == 0 else (total > 0 and passed == total)

        # code guardrails must also hold for the current phase
        code_gate = self._verify_guardrails(bwd_fp)

        if code_gate and parity_gate:
            self.i += 1
        # clear request either way
        self.pending_advance_from = None

        if VERBOSE:
            ok = code_gate and parity_gate
            # -1 bc just advanced above
            phase_name = self.phases[self.i-1].name
            status = "guardrails satisfied" if ok else "guardrails NOT satisfied"
            action = "advancing to next phase" if ok else "holding at current phase"
            print(f"[kernel-agent] Phase gate: {status} for '{phase_name}'; {action}")


def make_strategy(mode: str, default_temp: float = 0.7) -> BaseStrategy:
    # Factory: allows toggling strategy with an env flag without touching the loop.
    return PhasedStrategy() if mode == "phased" else RegularStrategy(default_temp)



# Phase-specific guardrail checks

# todo-now: the guarrails for this should be "gradcheck passes on single shape". And actaully gradrails for phase 1 is also IMPLCITILY assuuming passing all the shapes -- but currently this logic is hidden in the loop strcuture
#   ==> i think better to refactor and paass gradcheck stats here to the gurarails check as well -- so that guradrails_ checks below can decdie wearther ot  incrrmer or not based on weather that e.g. 1 shaep passeed; or all shaeps passed
# todo: assert no change in counts of tl.load, tl.store, tl.atomic_, and forbid edits to stub/kernel signatures
def guardrails_check_phase0(backward_fp: str) -> bool:
    try:
        with open(backward_fp, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s = ln.lstrip()
                if not s or s.startswith("#"):
                    continue
                # Forbid introducing new loops in Phase-0 (readability only)
                if (s.startswith("for ") or s.startswith("while ")) and s.rstrip().endswith(":") and (len(ln) - len(s) > 0):
                    return False
        return True
    except OSError:
        return False

def guardrails_check_phase1(backward_fp: str) -> bool:
    """
    Phase-1 (Refactor only) guardrail: return True if the current backward
    kernel contains at least one Python 'for' loop (heuristic: a line starting
    with 'for ' after indentation and ending with ':', ignoring comments).

    Rationale: only allow advancing to Phase 2 after loops were reintroduced.
    """
    try:
        with open(backward_fp, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                s = ln.lstrip()
                if not s or s.startswith("#"):
                    continue
                # Looks like an indented loop inside a function
                if s.startswith("for ") and s.rstrip().endswith(":") and (len(ln) - len(s) > 0):
                    return True
        return False
    except OSError:
        # Hold at Phase 1 if we can't verify
        return False

# comment:
# this check isn't particularly needed because in the orchestrator, on it>1
# i flip running gradcheck on the entire SWEEP, assuming user specified
# multiple shapes and given that my autograd unrolls and requires a single
# iteration -- so if the gradcheck passes, this means very likely the model
# introduced the loops already
def guardrails_check_phase2(backward_fp: str) -> bool:
    """
    Phase-2 (Atomics->private) guardrail: return True if the current backward
    kernel contains no Triton atomic operations. Conservative False on read error.

    Rationale: only allow advancing to Phase 3 after the model removed atomics.
    We keep the check lightweight by scanning for "tl.atomic_" in the file.
    """
    try:
        with open(backward_fp, "r", encoding="utf-8", errors="ignore") as f:
            return ("tl.atomic_" not in f.read())
    except OSError:
        # Hold at Phase 2 if we can't verify
        return False

def guardrails_check_phase3(backward_fp: str) -> bool:
    """
    Phase-3 guardrails are not implemented yet. This function intentionally raises
    to make the missing implementation explicit when invoked.
    """
    # raise NotImplementedError("Phase 3 guardrails are not implemented")
    return True
