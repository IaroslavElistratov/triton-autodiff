from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import os

from .utils import _env_truthy, generate_backward_stub_skeleton


# Default verbose ON unless explicitly disabled
VERBOSE = _env_truthy("KERNEL_AGENT_VERBOSE", "1")

# RAG reference formatting constants
_RAG_MAX_CHARS = 15000  # Max chars per kernel in reference section (to stay within token budget)
_SEP_MAJOR = "=" * 80   # Major section separator
_SEP_MINOR = "─" * 80   # Minor section separator
_SEP_HEADER = "=" * 60  # Header separator for initial file comments

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

    def workflow_section(self) -> str:
        """Return workflow context for system prompt.

        Describes the agent's role, workflow, and high-level goal.
        Override in subclasses for strategy-specific workflow descriptions.
        """
        return (
            "\n### Workflow context\n"
            "You are a Triton kernel optimizer. You are called as part of the workflow: generate initial backward pass -> [gradcheck -> optimize -> benchmark] the part in the brackets repeats in a for-loop. You are the 'optimize' step.\n"
            "You will have multiple turns to refine the backward kernel.\n" # , so do not propose large overly-eager kernel rewrites.
            "You will be provided a backward kernel which computes per-input gradients, use it as the starting point and make edits to improve its performance.\n"
            "You must adhere to user's Phase-specific goals and guardrails.\n"
            # "Do not try to derive backward mathematically from scratch this is hallucination- and error- prone, instead use the provided backward kernel and gradient annotations for your reference.\n"
        )

    def kernel_details_section(self) -> str:
        """Return initial kernel details for system prompt.

        Describes characteristics of the starting backward kernel that are specific
        to how it was generated (compiler vs RAG vs other methods).
        Override in subclasses or return empty string if not applicable.
        """
        return (
            # todo: show this only in the 1st iter?
            # comment: do not instruct to e.g. "for loops" or "remove atomics" -- this is handled in Phase[s]. Below is just general info only
            "\n### Initial backward kernel details\n"
            # " * signature: `backward_kernel(arg1, arg2, grad_arg1, grad_arg2)` for every *pointer* arg 'i' in inputs, there's a corresponding 'arg_i' containing pointer to gradient tensors wrt that input 'i').\n"
            "* variable names inside the kernel contain prefixes fwd_*, bwd_* -- the former means this is some intermediate value from the forward pass recomputed in backward, the latter means this is a value added by a derivative formula of some forward operator.\n"
            "* single-iteration unrolled: the initial backward kernel covers the gradients for exactly one iteration of the original forward loop (loop flattened).\n"
            # " * single-iteration unroll: the forward loop is flattened; this backward kernel computes gradients for exactly one loop iteration (one tile/chunk) and does not iterate over the full extent used in the benchmark sweep.\n"
            # " * single-iteration unroll: loops from the forward kernel are unrolled; the provided backward kernel corresponds to differentiated version of exactly one iteration of those loops.\n"
        )

    def allowed_edits_section(self) -> str:
        """Return allowed edits constraints for system prompt.

        Describes what the LLM is allowed to modify in the backward kernel.
        Override in subclasses for strategy-specific constraints.

        Compiler-generated kernels: Stub signature is correct, don't change it.
        RAG-retrieved kernels: Stub signature may need adaptation to match forward.
        """
        return (
            # "You must modify backward kernel; but preserve function names and pointer/mask semantics.\n"
            "The backward file contains BOTH the backward Triton kernel and a generated backward stub; you can (and likely should) edit both.\n"
            "Do NOT change the backward stub's signature, you can edit body of the stub but not its signature.\n"
            "You can edit the backward kernel signature and its body (but not stub's signature). Do not rename or move the file.\n"
            "You must only have a single backward kernel and a single backward stub, do not attempt to create multiple backward kernels or stubs.\n"
            "You can edit _mk_block_ptr when it's present.\n"
        )

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
        next_phase_name = self.phases[self.i+1].name

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
            status = "guardrails satisfied" if ok else "guardrails NOT satisfied"
            action = "advancing to next phase" if ok else "holding at current phase"
            print(f"[kernel-agent] Phase gate: {status} for '{next_phase_name}'; {action}")


class RAGAdaptationStrategy(BaseStrategy):
    """Single-phase strategy for RAG-initialized kernels.

    Unlike compiler-generated kernels that need multi-phase refinement (readability,
    loops, atomics, coalesce), RAG-retrieved kernels are already optimized and just
    need adaptation to the specific forward kernel.
    """
    def __init__(self, rag_fwd: Optional[str] = None, rag_bwd: Optional[str] = None):
        self.name = "rag_adaptation"
        # todo-low: cleanup (rm for this phase)
        self.i = 0  # for compatibility with rollback system
        # No phase advance since there's only one phase
        self.pending_advance_from = None

        # Store retrieved FWD+BWD for use in prompt as reference pattern
        # These are shown as readonly reference, not edited directly
        self.rag_fwd = rag_fwd
        self.rag_bwd = rag_bwd

    def workflow_section(self) -> str:
        """Return RAG-specific workflow context emphasizing adaptation over optimization."""
        return (
            "\n### Workflow context\n"
            "You are a Triton kernel adapter. Your task: write a backward kernel for YOUR forward (shown in working file).\n"
            "You can try adapt a retrieved backward kernel to work with a specific forward kernel.\n"
            "Workflow: RAG retrieval -> [write/adapt backward -> gradcheck] repeats until gradcheck passes on all shapes.\n"
            "\n"
            "A retrieved FWD+BWD pair is provided as a REFERENCE PATTERN (see reference section below).\n"
            "The retrieved backward was written for a DIFFERENT forward kernel and needs adaptation/rewrite to compute correct gradients for THIS forward.\n"
            "Your job: write backward for YOUR forward using the retrieved pair as a pattern guide.\n"
            "\n"
            "CRITICAL APPROACH:\n"
            "1. SEMANTIC ANALYSIS FIRST: Compare RETRIEVED FWD vs YOUR FWD (what's algorithmically different?)\n"
            "2. UNDERSTAND PATTERN: How does retrieved BWD mirror its FWD structure?\n"
            "3. APPLY PATTERN: Write BWD for YOUR FWD following the same backward-mirrors-forward principle\n"
            "\n"
            "Your goal is correctness (pass gradcheck), not optimization. The retrieved kernel is already optimized.\n"
            "You will have multiple turns to refine the adaptation. Adhere to Phase-specific adaptation goals.\n"
        )

    def kernel_details_section(self) -> str:
        """RAG kernels don't have compiler-specific characteristics, return empty."""
        return ""

    def rag_reference_section(self, rag_fwd: str, rag_bwd: str) -> str:
        """Build readonly RAG reference section for prompt.

        This shows the retrieved FWD+BWD pair as a reference pattern,
        NOT as code to edit directly. Forces semantic comparison with user's forward.

        Args:
            rag_fwd: Retrieved forward kernel source
            rag_bwd: Retrieved backward kernel source

        Returns:
            Formatted reference block with comparison instructions
        """
        # Truncate if too long (stay within token budget)
        if len(rag_fwd) > _RAG_MAX_CHARS:
            rag_fwd = rag_fwd[:_RAG_MAX_CHARS] + "\n... [truncated] ..."
        if len(rag_bwd) > _RAG_MAX_CHARS:
            rag_bwd = rag_bwd[:_RAG_MAX_CHARS] + "\n... [truncated] ..."

        return f"""
{_SEP_MAJOR}
REFERENCE ONLY (do not edit this section)
{_SEP_MAJOR}

The sections below show a RETRIEVED FWD+BWD pair written for a DIFFERENT kernel.
Use this as a REFERENCE PATTERN to understand how backward mirrors forward structure.

CRITICAL: Your task is to write backward for YOUR forward (shown in working file above).
DO NOT copy-paste this reference code. Instead:
1. Compare RETRIEVED FWD vs YOUR FWD (what's algorithmically different?)
2. Understand the pattern (how does retrieved BWD mirror its FWD?)
3. Apply that pattern to write BWD for YOUR FWD

{_SEP_MINOR}
RETRIEVED FORWARD (this is what the retrieved backward was written for):
{_SEP_MINOR}

{rag_fwd}

{_SEP_MINOR}
RETRIEVED BACKWARD (pattern reference - shows how backward mirrors forward):
{_SEP_MINOR}

{rag_bwd}

{_SEP_MAJOR}
END REFERENCE SECTION
{_SEP_MAJOR}

"""

    def generate_initial_file(self, fwd_source: str) -> str:
        """Generate initial raised.py content with backward stub skeleton.

        Uses utils.generate_backward_stub_skeleton() to parse forward and create
        minimal backward_stub, then formats it with file header.

        Args:
            fwd_source: Forward kernel source code containing stub function

        Returns:
            String containing file header + backward stub skeleton
        """
        skeleton = generate_backward_stub_skeleton(fwd_source)
        return f"""# {_SEP_HEADER}
# YOUR BACKWARD (write this using RAG reference)
# Forward kernel will be prepended by compile hook
# {_SEP_HEADER}

{skeleton}"""

    def allowed_edits_section(self) -> str:
        """Return RAG-specific allowed edits - stub signatures MUST be adapted to match forward."""
        return (
            "The backward file contains BOTH the backward Triton kernel and a backward stub; you can (and likely should) edit both.\n"
            "You can edit both the backward kernel and the backward stub (their signatures and bodies) to match YOUR forward kernel.\n"
            "The retrieved stub name/signature is from a DIFFERENT forward - you MUST adapt it to match YOUR forward's expectations.\n"
            "\n"
            "CRITICAL: Follow the auto-generated '# SIGNATURE CONTRACT' comment that shows:\n"
            "  - Exact signature gradcheck will call\n"
            "  - Exact tuple of gradients to return\n"
            "  - NO ctx parameter, NO None padding in returns\n"
            "\n"
            "The retrieved backward has torch.autograd.Function signature like '_SomeClass_backward(ctx, do)'.\n"
            "That's INCOMPATIBLE. Extract the gradient computation logic but rewrite stub to match the format above.\n"
            "\n"
            "### TRANSFORMATION TEMPLATE (follow this structure):\n"
            "\n"
            "BEFORE (torch.autograd.Function - what RAG gives you):\n"
            "```\n"
            "@staticmethod\n"
            "def backward(ctx, do):\n"
            "    q, k, v, softmax_lse = ctx.saved_tensors\n"
            "    sm_scale = ctx.sm_scale\n"
            "    dq, dk, dv = _some_bwd_kernel(...)\n"
            "    return dq, dk, dv, None, None\n"
            "```\n"
            "\n"
            "AFTER (triton autodiff raised.py - what you must create):\n"
            "```\n"
            "# SIGNATURE CONTRACT comment shows exact function signature\n"
            "def backward_stub(...)  # Use exact signature from SIGNATURE CONTRACT\n"
            "    # 1. Allocate outputs (match forward)\n"
            "    # 2. Initialize grad buffers (zeros for grads wrt inputs, upstream[s] for grads wrt outputs)\n"
            "    # 3. Launch backward kernel with adapted logic\n"
            "    # 4. Return exactly what SIGNATURE CONTRACT shows\n"
            "```\n"
            "\n"
            "### ARCHITECTURAL CONTEXT:\n"
            "\n"
            "Retrieved backward is torch.autograd.Function.backward\n"
            "Required backward is triton autodiff raised.py (kernel-level autodiff with Python wrapper).\n"
            "These are INCOMPATIBLE systems - you must TRANSFORM, not just rename:\n"
            "\n"
            "- backward(ctx, do)              ->  See SIGNATURE CONTRACT comment for exact signature\n"
            "- ctx.saved_tensors              ->  eliminated (use stub params + allocate outputs like fwd)\n"
            "- do (single upstream)           ->  upstream_0, upstream_1, ... (kwargs)\n"
            "- returns (grad1, grad2, None)   ->  See SIGNATURE CONTRACT for exact return tuple\n"
            "\n"
            "Extract ONLY: gradient computation logic, kernel calls, math operations\n"
            "Discard: ctx usage, function signature, return format, @staticmethod decorator\n"
            "\n"
            "### RETURN VALUE REQUIREMENTS:\n"
            "\n"
            "Follow the '# Must return tuple:' line in the SIGNATURE CONTRACT comment.\n"
            "It shows exactly which gradients to return (only tensor parameters, no None padding).\n"
            "\n"
            "Do not rename or move the file.\n"
            "You must only have a single backward kernel and a single backward stub, do not attempt to create multiple backward kernels or stubs.\n"
            "You can edit _mk_block_ptr when it's present.\n"
        )

    def current_phase(self, parity_ok: bool) -> Tuple[str, float]:
        """Return adaptation phase instructions and temperature."""
        # Use lower temperature for adaptation (precision critical)
        # Higher temp after baseline acceptance for optional optimization
        temp = 0.25 if not parity_ok else 0.5

        header = (
            "Phase = Adapt Retrieved Backward Kernel\n"
            "\n"
            "Task: Adapt retrieved backward to compute correct gradients for THIS forward kernel.\n"
            # "No algorithm changes.\n"
            "The retrieved backward was written for a DIFFERENT forward kernel - its signatures,\n"
            "argument order, and indexing will NOT match yours. Perform adaptation to\n"
            "establish correspondence between retrieved backward and MY forward:\n"
            "\n"
            "Adaptations required (not exhaustive):\n"
            "- Function signatures: Match kernel/stub names to MY forward's naming convention\n"
            "- Argument order: Backward must accept forward's inputs in SAME order forward defines them\n"
            "  (Critical: gradcheck expects specific signature - wrong order = immediate failure)\n"
            "- Dtypes: Match MY forward's input/output dtypes exactly\n"
            "- Indexing: Fix strides, offsets, pointer arithmetic for MY tensor shapes\n"
            "  (Access patterns may differ - backward can read/write in different order than forward)\n"
            "- Boundaries: Update masks and tail handling for MY tensor shapes\n"
            "\n"
            "Preserve from retrieved backward:\n"
            "- Algorithmic structure (retrieved backward computes correct gradients for similar -- but NOT exactly our -- fwd kernel, so need adapt operations to compute correct gradients for our fwd kernel specifically)\n"
            "- Atomics (if present - preserve location and accumulation semantics)\n"
            "- Parallelization strategy (grid/loop structure may differ from forward - that's expected)\n"
            "\n"
            "### KEY TRANSFORMATION STEPS:\n"
            "\n"
            "1. Use signature from '# SIGNATURE CONTRACT' comment (NOT from RAG code)\n"
            "2. Allocate outputs like forward stub does\n"
            "3. Initialize gradients: zeros_like(inputs), upstream_N.clone() for outputs\n"
            "4. Replace ctx.* with stub params\n"
            "5. Return exactly what '# Must return tuple:' specifies\n"
            "\n"
            f"Success Gates: gradcheck_ok={parity_ok} (must pass on all SWEEP shapes)\n"
            "Focus on structural correspondence with MY forward, not optimization.\n"
        )
        return header, temp

    # todo-low: cleanup (rm for this phase)
    def maybe_advance(self, bwd_fp, payload_gradcheck) -> None:
        """No-op for single-phase strategy."""
        pass

    def set_phase_index(self, i: int) -> None:
        """For compatibility with rollback - always stays at phase 0."""
        if i != 0 and VERBOSE:
            print(f"[kernel-agent] RAGAdaptationStrategy only has one phase (attempted to set i={i})")
        self.i = 0


def make_strategy(mode: str, default_temp: float = 0.7) -> BaseStrategy:
    # Factory: allows toggling strategy with an env flag without touching the loop.
    if mode == "phased":
        return PhasedStrategy()
    elif mode == "rag_adaptation":
        return RAGAdaptationStrategy()
    else:
        return RegularStrategy(default_temp)



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
