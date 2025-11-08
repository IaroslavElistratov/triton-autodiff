from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .utils import _env_truthy
from .rag import _load_index, _embed_query, _cosine


# Default verbose ON unless explicitly disabled
VERBOSE = _env_truthy("KERNEL_AGENT_VERBOSE", "1")


def _parse_rag_exclusions() -> list[str]:
    raw = os.environ.get("KERNEL_AGENT_RAG_EXCLUDE_SUBSTR", "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item and item.strip()]


def _filter_rag_paths(paths: Sequence[str], excluder: "_PathExcluder") -> tuple[list[str], list[tuple[str, str]]]:
    filtered: list[str] = []
    excluded: list[tuple[str, str]] = []
    for path in paths:
        reason = excluder.match(path)
        if reason:
            excluded.append((path, reason))
            if VERBOSE:
                print(f"[kernel-agent][RAG] Excluding '{path}' (matched '{reason}')")
            continue
        filtered.append(path)
    return filtered, excluded

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


class BaseStrategy:
    def next_phase(self, parity_ok: bool) -> Tuple[str, float]:
        # Default: a simple optimize header with global guardrails and neutral temperature
        return "Phase = optimize.\n", 0.7

    def maybe_advance(self, bwd_fp, payload_gradcheck) -> None:
        # No-op in the base class.
        pass

    def exception_fix_header(self) -> str:
        """Return phase-appropriate header for exception fixing.

        Strategies can use internal state (e.g., self.i for phase) to customize
        the fix header. Default implementation returns generic fix prompt.

        Returns:
            Header string for LLM prompt when fixing exceptions.
        """
        # Default: generic fix header
        return "Phase = fix. Fix the exception.\n"

    def workflow_section(self) -> str:
        """Return workflow context for system prompt.

        Describes the agent's role, workflow, and high-level goal.
        Override in subclasses for strategy-specific workflow descriptions.
        """
        return (
            "\n### Workflow context\n"
            "You are a Triton kernel optimizer. You are called as part of the workflow: generate initial backward pass -> [gradcheck -> optimize -> benchmark] the part in the brackets repeats in a for-loop. You are the 'optimize' step.\n"
            "You will have multiple turns to refine the backward kernel.\n"
            # , so do not propose large overly-eager kernel rewrites.
            "You will be provided a backward kernel which computes per-input gradients, use it as the starting point and make edits to improve its performance.\n"
            "You must adhere to user's Phase-specific goals and guardrails.\n"
            # "Do not try to derive backward mathematically from scratch this is hallucination- and error- prone, instead use the provided backward kernel and gradient annotations for your reference.\n"
        )

    # def kernel_details_section(self) -> str:
    #     """Return initial kernel details for system prompt.

    #     Describes characteristics of the starting backward kernel that are specific
    #     to how it was generated (compiler vs RAG vs other methods).
    #     Override in subclasses or return empty string if not applicable.
    #     """
    #     return (
    #         # "\n### Initial backward kernel details\n"
    #         # " * signature: `backward_kernel(arg1, arg2, grad_arg1, grad_arg2)` for every *pointer* arg 'i' in inputs, there's a corresponding 'arg_i' containing pointer to gradient tensors wrt that input 'i').\n"
    #         # "* variable names inside the kernel contain prefixes fwd_*, bwd_* -- the former means this is some intermediate value from the forward pass recomputed in backward, the latter means this is a value added by a derivative formula of some forward operator.\n"
    #         # "* single-iteration unrolled: the initial backward kernel covers the gradients for exactly one iteration of the original forward loop (loop flattened).\n"
    #         # " * single-iteration unroll: the forward loop is flattened; this backward kernel computes gradients for exactly one loop iteration (one tile/chunk) and does not iterate over the full extent used in the benchmark sweep.\n"
    #         # " * single-iteration unroll: loops from the forward kernel are unrolled; the provided backward kernel corresponds to differentiated version of exactly one iteration of those loops.\n"
    #     )

    def allowed_edits_section(self) -> str:
        """Return allowed edits constraints for system prompt.

        Describes what the LLM is allowed to modify in the backward kernel.
        Override in subclasses for strategy-specific constraints.
        """
        return (
            "The backward file contains the forward stub, backward Triton kernel, backward stub, and StubOverrideDCK class.\n"
            "You can edit: backward kernel, backward stub body, forward stub body, and StubOverrideDCK class.\n"
            "You must only have a single backward stub.\n"
            "Do not rename or move the file.\n"
            "\n"
            "You may edit stub and kernel signatures when required. Keep every call-site and signature consistent when you do so, restoring argument order and return structure to match the contract used by gradcheck.\n"
            "### CRITICAL: AVOID RECOMPUTING FORWARD INTERMEDIATES\n"
            "Ensure forward() saves required intermediates and backward_stub consumes them; do not fall back to recomputing forward kernels inside backward_stub.\n"
            "If you need new buffers, modify StubOverrideDCK to pass them while keeping existing stub signatures intact.\n"
            "\n"
        )

# for ablations
class RegularStrategy(BaseStrategy):
    def __init__(self, temp: float = 0.7) -> None:
        self.name = "regular"
        self._temp = temp
        self.phase_just_advanced = False  # For compatibility with orchestrator

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

    def maybe_advance(self, bwd_fp, payload_gradcheck) -> None:
        """No-op for RegularStrategy which doesn't have phases.

        Note: phase_just_advanced is already reset by orchestrator at start of iteration.
        We redundantly set it to False here for clarity (RegularStrategy never transitions).
        """
        self.phase_just_advanced = False  # Always False since no phases


class _PathExcluder:
    def __init__(self, raw_tokens: Sequence[str]):
        cleaned = [tok.strip() for tok in raw_tokens if tok and tok.strip()]
        self.raw_tokens = cleaned
        self.tokens_lower = [tok.lower() for tok in cleaned]

    def match(self, path: str) -> Optional[str]:
        if not self.tokens_lower:
            return None
        lowered = path.lower()
        for token in self.tokens_lower:
            if token in lowered:
                return token
        return None


class RAGAdaptationStrategy(BaseStrategy):
    """Two-phase strategy for reference-based gradient validation.

    Phase 1: Generate and validate PyTorch reference implementation
    Phase 2: Generate backward kernel and validate against reference gradients

    Unlike compiler-generated kernels that need multi-phase refinement (readability,
    loops, atomics, coalesce), RAG-retrieved kernels are already optimized and just
    need adaptation to the specific forward kernel.
    """

    def __init__(
        self,
        rag_fwd: Optional[str] = None,
        rag_bwd: Optional[str] = None,
        *,
        excluded_substrings: Optional[Sequence[str]] = None,
    ):
        self.name = "rag_adaptation"
        # Phase index (0=reference generation, 1=backward generation)
        self.i = 0
        # Track if we just transitioned phases
        self.phase_just_advanced = False

        # Phase 1 state: PyTorch reference

        # Source code of validated reference
        self.pytorch_reference_code = None
        # Whether reference passed validation
        self.pytorch_reference_validated = False

        # Phase 2 state: RAG patterns (optional)
        self.rag_fwd = rag_fwd
        self.rag_bwd = rag_bwd
        # Store retrieval metadata when orchestrator calls retrieve_references
        self.last_retrieval = {}
        self._path_excluder = _PathExcluder(excluded_substrings or [])

    def retrieve_references(self, redacted_forward: str) -> dict:
        """Retrieve reference forward/backward code snippets via embeddings."""
        index_path = Path(__file__).parent / "kernel_embeddings.pkl"
        min_sim = float(os.environ.get("KERNEL_AGENT_RAG_MIN_SIM", "0.75"))

        try:
            embeddings, documents, backward_docs, file_list, openai_model = _load_index(str(index_path))
            query_embedding = _embed_query(redacted_forward, model=openai_model)
        except Exception as e:
            raise ValueError(f"Failed to load RAG index or embed query: {e}")

        filtered_paths, excluded_info = _filter_rag_paths(file_list, self._path_excluder)
        best_match = None
        best_similarity = -1.0
        candidate_scores: list[tuple[str, float]] = []
        for fp in filtered_paths:
            sim = _cosine(query_embedding, embeddings[fp])
            candidate_scores.append((fp, sim))
            if sim >= min_sim and sim > best_similarity:
                content = backward_docs.get(fp, "")
                if content:
                    best_match = fp
                    best_similarity = sim

        if VERBOSE and candidate_scores:
            top5 = sorted(candidate_scores, key=lambda item: item[1], reverse=True)[:5]
            print("[kernel-agent][RAG] Top candidates:")
            for path, sim in top5:
                print(f"  - {path} (sim={sim:.3f})")

        if not best_match:
            extra = ""
            if excluded_info and self._path_excluder.tokens_lower:
                first_hits = ", ".join(path for path, _ in excluded_info[:3])
                tokens = ", ".join(self._path_excluder.raw_tokens)
                extra = (
                    f" Excluded {len(excluded_info)} candidates via rag-exclude filter"
                    f" (substrings: {tokens}; first match: {first_hits})."
                )
            raise ValueError(
                f"No similar kernels found in RAG index (similarity >= {min_sim:.2f}). "
                "Try lowering --min-sim threshold." + extra
            )

        forward_ref = documents.get(best_match, "")
        backward_ref = backward_docs.get(best_match, "")

        payload = {
            "forward": forward_ref,
            "backward": backward_ref,
            "match_path": best_match,
            "similarity": best_similarity,
            "excluded_matches": [path for path, _ in excluded_info],
        }
        self.last_retrieval = payload
        return payload

    def workflow_section(self) -> str:
        """Return workflow context based on current phase."""
        if self.i == 0:
            # Phase 1 Prompts - Emphasize Naive, Simple References
            # BUG FIX: LLM was generating complex 60-line loopy implementations trying to
            # replicate Triton's tiling/streaming behavior.
            # ALSO: LLM tried using built-ins like F.scaled_dot_product_attention, which
            # don't exist for arbitrary kernels in the wild.
            # SOLUTION: Updated prompts to emphasize straightforward, naive implementations
            # following Triton's own tutorial style (simple, no complex loops).
            return (
                "\n### Workflow context\n"
                "You are generating a PyTorch reference implementation for gradient validation.\n"
                "Your task: Write a SIMPLE, NAIVE PyTorch equivalent of the Triton forward kernel.\n"
                "This reference will be used to validate gradients via PyTorch's autograd.\n"
                "\n"
                "Workflow: Generate naive PyTorch reference -> Validate outputs match Triton -> Use for gradient validation\n"
                "\n"
                "CRITICAL: Keep the reference SIMPLE, NAIVE!\n"
                "- Create straightforward reference implementation for forward kernels \n"
                "- Do NOT replicate tiling/streaming with loops\n"
                "- Do NOT try to use built-in functions (they won't exist for arbitrary kernels)\n"
                "- Write the most straightforward implementation, even if it's memory-intensive\n"
                "- It's OK if the reference OOMs on large shapes - we test on smaller shapes first\n"
                # "- Tolerances (atol=1e-2) handle minor numerical differences from tiling\n"
                "\n"
                "You will have multiple turns to refine until outputs match.\n"
            )
        else:
            ref_block_line = (
                "A retrieved FWD+BWD pair is provided as a REFERENCE PATTERN (see reference section below).\n"
                if self.phase_just_advanced
                else "You already saw the retrieved FWD+BWD reference when entering Phase 2; reuse that pattern from memory (reference block hidden this turn).\n"
            )
            return (
                "\n### Workflow context\n"
                "You are a Triton kernel adapter. Your task: write a backward kernel for YOUR forward (shown in working file).\n"
                "You can try adapt a retrieved backward kernel to work with a specific forward kernel.\n"
                "Workflow: RAG retrieval -> [write/adapt backward -> gradcheck] repeats until gradcheck passes on all shapes.\n"
                "\n"
                f"{ref_block_line}"
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

    # def kernel_details_section(self) -> str:
    #     """RAG kernels don't have compiler-specific characteristics, return empty."""
    #     return ""

    def rag_reference_section(self, rag_fwd: str, rag_bwd: str) -> str:
        """Show RAG references only at Phase 2 entry (backward generation init).

        Phase 1: PyTorch reference generation - no RAG content (returns empty)
        Phase 2 entry: Show RAG FWD+BWD as pattern reference (only once)
        Phase 2 fix iterations: Hide RAG, rely on conversation chaining from Phase 2 entry turn
        """
        # Phase 1: Hide RAG (don't show Triton kernels when generating PyTorch reference)
        if self.i == 0:
            if VERBOSE: print("[kernel-agent] RAG reference hidden (Phase 1: generating PyTorch reference)")
            return ""

        # Phase 2 entry: Show RAG FWD+BWD once when transitioning
        if self.phase_just_advanced:
            if VERBOSE: print(f"[kernel-agent] RAG reference shown (Phase 2 entry: generating backward kernel)")
            return self._build_rag_reference_section(rag_fwd, rag_bwd)

        # Phase 2 fix iterations: Hide RAG, rely on conversation chaining
        if VERBOSE: print("[kernel-agent] RAG reference hidden (Phase 2 fix iteration: relying on conversation chaining)")
        return ""

    def _build_rag_reference_section(self, rag_fwd: str, rag_bwd: str) -> str:
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

    def allowed_edits_section(self) -> str:
        """Return allowed edits based on current phase."""
        if self.i == 0:
            # Phase 1: PyTorch reference generation
            return (
                "You are writing a PyTorch reference implementation.\n"
                "Add a new function called `pytorch_reference_impl` that:\n"
                "1. Takes the same arguments as the Triton forward stub\n"
                "2. Returns the same output shape/dtype as the Triton forward\n"
                "3. Implements the same mathematical operation in pure PyTorch\n"
                "\n"
                "CRITICAL: Write a SIMPLE, NAIVE implementation for CORRECTNESS!\n"
                "- Use standard PyTorch operations (matmul, softmax, etc.)\n"
                "- Write the MOST STRAIGHTFORWARD implementation, even if it's memory-intensive\n"
                "- Do NOT replicate the Triton kernel's tiling/streaming behavior\n"
                "- Do NOT write complex loops to mimic block processing\n"
                "- Do NOT try to use built-in functions like F.scaled_dot_product_attention\n"
                "  (they may not exist or match semantics for arbitrary kernels)\n"
                "\n"
                "IMPORTANT: It's OK if the reference OOMs on large shapes!\n"
                "- The reference is for gradient validation, not production use\n"
                "- Naive implementations that materialize full matrices are acceptable\n"
                "- Focus on CORRECTNESS, not efficiency\n"
                "\n"
                "Example:\n"
                "```python\n"
                "# For attention - naive implementation:\n"
                "def pytorch_reference_impl(q, k, v, causal=False, sm_scale=0.5, **kwargs):\n"
                "    # Materialize full attention matrix (may OOM on large shapes - that's OK!)\n"
                "    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale\n"
                "    if causal:\n"
                "        mask = torch.tril(torch.ones_like(scores))\n"
                "        scores = scores.masked_fill(mask == 0, float('-inf'))\n"
                "    p = torch.softmax(scores, dim=-1)\n"
                "    return torch.matmul(p, v)\n"
                "```\n"
                "\n"
                "The reference should be TRUSTED, SIMPLE, and OBVIOUSLY CORRECT.\n"
                "Tolerances (atol=1e-2) handle minor numerical differences from tiling.\n"
            )
        else:
            # Phase 2: Backward kernel generation
            base_section = super().allowed_edits_section()

            # RAG-specific adaptation instructions
            backward_specific = (
            "The backward file contains BOTH the backward Triton kernel and a backward stub.\n"
            "You can edit both the backward kernel and the backward stub (their signatures and bodies) to match YOUR forward kernel.\n"
            "The retrieved stub name/signature is from a DIFFERENT forward - you MUST adapt it to match YOUR forward's expectations.\n"
            "\n"
            "### CRITICAL: Signature Contract\n"
            "\n"
            "The auto-generated '# SIGNATURE CONTRACT' comment shows EXACT calling code gradcheck will use.\n"
            "Match it precisely - parameter order, upstream_N kwargs, return tuple (tensor grads only, no None padding).\n"
            "\n"
            "Retrieved backward has torch.autograd.Function signature (ctx, do) - INCOMPATIBLE.\n"
            "Extract gradient logic, rewrite stub to match SIGNATURE CONTRACT.\n"
            "\n"
            "### TRANSFORMATION SEQUENCE:\n"
            "\n"
            "1. SEMANTIC ANALYSIS FIRST (see detailed section below):\n"
            "   - Understand what mathematical transformation RETRIEVED forward computes\n"
            "   - Understand what mathematical transformation YOUR forward computes\n"
            "   - Identify where the algorithms diverge structurally\n"
            "   - Determine how each divergence affects the derivative\n"
            "\n"
            "2. MECHANICAL ADAPTATION (after semantic understanding):\n"
            "\n"
            "BEFORE (torch.autograd.Function - what RAG gives you):\n"
            "```\n"
            "@staticmethod\n"
            "def backward(ctx, grad_out):\n"
            "    saved_vals = ctx.saved_tensors\n"
            "    params = ctx.params\n"
            "    grad_inputs = _bwd_kernel(...)\n"
            "    return grad_inputs, None, None\n"
            "```\n"
            "\n"
            "AFTER (what you must create):\n"
            "```\n"
            "# SIGNATURE CONTRACT comment shows exact function signature\n"
            "def backward_stub(...)  # Use exact signature from SIGNATURE CONTRACT\n"
            "    # Allocate outputs (match forward)\n"
            "    # Initialize grad buffers (zeros for grads wrt inputs, upstream[s] for grads wrt outputs)\n"
            "    # Launch backward kernel with adapted logic from semantic analysis\n"
            "    # Return exactly what SIGNATURE CONTRACT shows\n"
            "```\n"
            "\n"
            # todo-now: outdated? becuase not llm has control over the StubOverwiteDck
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
            # "\n"
            # "### VALIDATION:\n"
            # "\n"
            # "The gradcheck will:\n"
            # "1. Run pytorch_reference_impl with autograd enabled\n"
            # "2. Call .backward() to get reference gradients via PyTorch autograd\n"
            # "3. Compare your Triton backward gradients against PyTorch reference gradients\n"
            # "\n"
            # "Your gradients must match PyTorch's gradients (atol=1e-2, rtol=0).\n"
            "\n"
            "Do not rename or move the file.\n"
            "You must only have a single backward stub, do not attempt to create multiple backward stubs.\n"
            )

            # Combine with base section for stashing optimization
            return backward_specific + "\n" + base_section

    def current_phase(self, parity_ok: bool) -> Tuple[str, float]:
        """Return phase-specific instructions and temperature."""
        if self.i == 0:
            # Phase 1: Generate PyTorch reference
            header = (
                "Phase = Generate PyTorch Reference\n"
                "\n"
                "Task: Write a SIMPLE, NAIVE PyTorch reference that matches the Triton forward kernel.\n"
                # "This reference will be used to validate gradients via PyTorch's autograd.\n"
                "\n"
                "Requirements:\n"
                "1. Function name: pytorch_reference_impl\n"
                "2. Same inputs as forward stub (ignore tiling parameters)\n"
                "3. Same output shape/dtype as forward\n"
                "4. SIMPLE, NAIVE PyTorch code (no loops, no tiling, no built-ins)\n"
                "\n"
                "Reference implementations must be simple and straightforward.\n"
                "Do NOT try to replicate the tiling/streaming behavior with loops.\n"
                "Do NOT try to use built-in functions like F.scaled_dot_product_attention.\n"
                "Just compute the same mathematical operation in the most straightforward PyTorch way.\n"
                "It's OK if the reference OOMs on large shapes - we validate on smaller shapes first.\n"
            )
            return header, 0.1  # Low temperature for deterministic reference
        else:
            # Phase 2: Generate backward kernel
            temp = 0.25 if not parity_ok else 0.5

            # Semantic analysis preamble - forces model to think top-down before diving into implementation
            semantic_preamble = (
                "\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "CRITICAL: START WITH HIGH-LEVEL SEMANTIC ANALYSIS (not implementation details)\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "\n"
                "Before making ANY code changes, spend reasoning tokens on semantic understanding:\n"
                "\n"
                "1. UNDERSTAND BOTH ALGORITHMS:\n"
                "\n"
                "   For YOUR forward AND RETRIEVED forward:\n"
                "   - What mathematical transformation does each compute?\n"
                "   - What are the key operations and data dependencies?\n"
                "\n"
                "2. COMPARE AND IDENTIFY DIVERGENCE:\n"
                "\n"
                "   Do they compute fundamentally different things, or the same thing differently?\n"
                "   Which parts of the computation are semantically different?\n"
                "   \n"
                "   The comparison reveals your strategy:\n"
                "   \n"
                "   - Small divergence (e.g., one computes subset of operations the other does):\n"
                "     → Adapt gradient terms for these specific differences\n"
                "   \n"
                "   - Large divergence (e.g., different algorithms but similar computational pattern):\n"
                "     → Understand the pattern, rewrite gradient math for YOUR algorithm\n"
                "   \n"
                "   - Completely different (e.g., fundamentally different operation types):\n"
                "     → RAG similarity failed; derive YOUR gradients from first principles\n"
                "\n"
                "3. DETERMINE GRADIENT IMPLICATIONS:\n"
                "\n"
                "   For each divergence identified:\n"
                "   - How does this difference change the derivative?\n"
                "   - What does the chain rule require given this algorithmic structure?\n"
                "   - Which gradient terms exist in retrieved backward but shouldn't in yours (or vice versa)?\n"
                "   - Which parts of retrieved backward can be reused vs rewritten?\n"
                "\n"
                "   The comparison naturally guides adaptation - no explicit categorization needed.\n"
                "\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "DO NOT spend reasoning tokens on (until high-level semantic analysis is done):\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "- Block size arithmetic (TILE, BLOCK_M, BLOCK_N, divisibility constraints)\n"
                "- Stride/offset micro-calculations\n"
                "- Naming conventions (variable naming, docstring style)\n"
                "\n"
                "Algorithm thinking: \"Is this kernel computing the CORRECT mathematical derivative?\"\n"
                "Implementation thinking: \"Are the memory strides correct?\" ← Do this AFTER algorithm is right\n"
                "\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "Before calling apply_patch, verify in your analysis:\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "- Understood what mathematical transformation RETRIEVED forward performs\n"
                "- Understood what mathematical transformation YOUR forward performs\n"
                "- Identified where the two algorithms diverge structurally\n"
                "- Analyzed how each divergence affects the derivative (chain rule implications)\n"
                "- Determined which gradient terms need to change based on algorithmic differences\n"
                "\n"
                "If you skip directly to implementation details (strides, offsets, block sizes),\n"
                "you will waste iterations fixing wrong things.\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "\n"
            )

            # Success criteria - distinguish partial success from failure
            success_criteria = (
                f"\n"
                f"SUCCESS CRITERIA (gradcheck_ok={parity_ok}):\n"
                f"- ALL shapes pass → SUCCESS, don't modify algorithm\n"
                f"- N shapes pass, M OOM → Algorithm CORRECT for passing shapes. OutOfMemoryError = hardware limit (NOT bug).\n"
                f"  DO NOT modify kernel algorithm. Regression (N pass → fewer) = you broke working code.\n"
                f"- Wrong gradients → Algorithm incorrect, re-analyze what YOUR forward computes vs what RETRIEVED forward computes\n"
                f"\n"
            )
            header = (
                "Phase = Adapt Retrieved Backward Kernel\n"
                + semantic_preamble
                + "\n"
                "Task: Adapt retrieved backward to compute correct gradients for THIS forward kernel.\n"
                "The retrieved backward was written for a DIFFERENT forward kernel - its signatures,\n"
                "argument order, and indexing will NOT match yours. Perform adaptation to\n"
                "establish correspondence between retrieved backward and MY forward.\n"
                "\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "❌ ANTI-PATTERN (causes iteration waste):\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "- Blindly copying retrieved backward's code without understanding the algorithm\n"
                "- Tweaking parameter names, strides, block sizes without analyzing forward differences\n"
                "- Assuming retrieved backward works as-is without checking if forwards match algorithmically\n"
                "\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "✓ CORRECT PATTERN:\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "- Understand what mathematical transformation RETRIEVED forward performs\n"
                "- Understand what mathematical transformation YOUR forward performs\n"
                "- Understand what gradients the retrieved backward computes (the math, not the code)\n"
                "- ADAPT the gradient math for YOUR forward's algorithm (different algorithms → different derivatives)\n"
                "- Implement adapted gradient computation in backward kernel\n"
                "\n"
                "Gradient formulas themselves need adaptation based on semantic differences.\n"
                "═══════════════════════════════════════════════════════════════════════════════\n"
                "\n"
                "After semantic analysis, adapt in priority order:\n"
                "1. HIGH: What gradients are computed (mathematical correctness for YOUR forward's algorithm)\n"
                "2. MEDIUM: Interface (signatures, return tuple per SIGNATURE CONTRACT)\n"
                "3. LOW: How gradients are computed (memory layout, block sizes) - only after gradients pass\n"
                "\n"
                "CRITICAL IMPLEMENTATION DETAILS:\n"
                "- Argument order: Must match SIGNATURE CONTRACT exactly (wrong order = immediate failure)\n"
                "- Dtypes: Match YOUR forward's dtypes exactly\n"
                "- Atomics: If present in retrieved, preserve location and semantics\n"
                "- Gradient init: zeros_like(inputs) for input grads, upstream_N.clone() for output grads\n"
                "- Grid: Backward parallelization may differ from forward - expected\n"
                "\n"
                + success_criteria
                + "Focus on structural correspondence with MY forward, not optimization.\n"
                "ONLY restore correctness to pass gradcheck.\n"
            )
            return header + semantic_preamble, temp

    def maybe_advance(self, bwd_fp, payload_gradcheck) -> None:
        """Advance from Phase 0 to Phase 1 after PyTorch reference is validated.

        Note: phase_just_advanced is reset to False by orchestrator at start of each iteration.
        We redundantly set it to False here for clarity.
        """
        self.phase_just_advanced = False
        if self.i == 0 and self.pytorch_reference_validated:
            # Advance from reference generation to backward generation
            self.i = 1
            self.phase_just_advanced = True  # Mark that we just advanced (will be reset next iteration)
            if VERBOSE: print("[kernel-agent] RAGAdaptationStrategy: Advancing from PyTorch reference to backward generation")
        # Phase 1 doesn't advance (stays at backward generation)

    def exception_fix_header(self) -> str:
        """Return phase-appropriate header for exception fixing.

        Uses self.i (phase index) to determine appropriate guidance.
        Phase 1 requires semantic analysis to avoid tactical fix loops,
        while Phase 0 just needs correct PyTorch implementation.
        """
        if self.i == 0:
            # Phase 0: PyTorch reference generation
            return (
                "Phase = fix. Fix the exception in PyTorch reference.\n"
                "\n"
                "The PyTorch reference must be SIMPLE and NAIVE.\n"
                "It should compute the same mathematical operation as YOUR forward kernel.\n"
                "Do NOT use built-in functions like F.scaled_dot_product_attention.\n"
                "Just implement the math directly with basic PyTorch operations.\n"
                "\n"
            )
        else:
            # Phase 1: Backward kernel (self.i == 1)
            # Semantic analysis critical to avoid tactical loops (evidenced by LOGS/3_out.txt)
            return (
                "Phase = fix. Fix the exception below.\n"
                "\n"
                "CRITICAL: Before making tactical fixes, verify semantic correctness:\n"
                "- What mathematical transformation does YOUR forward compute?\n"
                "- What mathematical transformation does RETRIEVED forward compute?\n"
                "- What gradients are required by the chain rule for YOUR forward's algorithm?\n"
                "- Is the current backward kernel computing the CORRECT gradients for YOUR forward?\n"
                "\n"
                "If you haven't compared YOUR forward vs RETRIEVED forward algorithms, DO THAT FIRST.\n"
                "Do not fix dtypes, strides, or atomics until you verify the gradient math is correct.\n"
                "\n"
            )

    def set_phase_index(self, i: int) -> None:
        """Set phase index for rollback compatibility."""
        assert i in [0, 1], "RAGAdaptationStrategy has only two phases"
        self.i = i
        self.phase_just_advanced = False  # Reset flag when manually setting phase
        if VERBOSE:
            phase_name = "PyTorch reference" if i == 0 else "Backward generation"
            print(f"[kernel-agent] RAGAdaptationStrategy phase set to {i} ({phase_name})")


def make_strategy(mode: str, default_temp: float = 0.7) -> BaseStrategy:
    # Factory: allows toggling strategy with an env flag without touching the loop.
    exclusions = _parse_rag_exclusions()
    if mode == "phased":
        raise ValueError("Phased mode is temporarily disabled. Use 'regular' or 'rag_adaptation' instead.")
    elif mode == "rag_adaptation":
        return RAGAdaptationStrategy(excluded_substrings=exclusions)  # Use new reference-based strategy
    else:
        return RegularStrategy(default_temp)



# Phase-specific guardrail checks

# todo-now: the guarrails for this should be "gradcheck passes on single shape". And actaully gradrails for phase 1 is also IMPLCITILY assuuming passing all the shapes -- but currently this logic is hidden in the loop strcuture
#   ==> i think better to refactor and paass gradcheck stats here to the gurarails check as well -- so that guradrails_ checks below can decdie wearther ot  incrrmer or not based on weather that e.g. 1 shaep passeed; or all shaeps passed
# todo: assert no change in counts of tl.load, tl.store, tl.atomic_, and forbid edits to stub/kernel signatures
def guardrails_check_phase0(generated_fp: str) -> bool:
    try:
        with open(generated_fp, "r", encoding="utf-8", errors="ignore") as f:
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

def guardrails_check_phase1(generated_fp: str) -> bool:
    """
    Phase-1 (Refactor only) guardrail: return True if the current backward
    kernel contains at least one Python 'for' loop (heuristic: a line starting
    with 'for ' after indentation and ending with ':', ignoring comments).

    Rationale: only allow advancing to Phase 2 after loops were reintroduced.
    """
    try:
        with open(generated_fp, "r", encoding="utf-8", errors="ignore") as f:
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
