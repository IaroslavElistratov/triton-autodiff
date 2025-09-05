from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# Strategy module: encapsulates per-step instructions and temperature, so the
# orchestrator can toggle this behavior without changing its main loop.

@dataclass(frozen=True)
class Phase:
    name: str
    goal: str
    guardrails: str
    temp: float

# Shared guardrails injected every turn (regular and phased).
GLOBAL_GUARDRAILS = (
    "Guardrails:\n"
    "- Exactly one apply_patch.md block.\n"
    "- No rule echoing; diff only.\n"
    "- ≤120 changed lines per patch.\n"
    "- Include at least one '-' anchor line per hunk.\n"
    "- Preserve function names and all 'backward_*' args.\n"
    "- Single backward kernel and single stub."
)

# Default phase sequence — light guidance per step; can be replaced/tuned.
PHASES: Sequence[Phase] = (
    Phase("1 / Refactor only", "Re-introduce loops and tail masks. Keep atomics.", "Only loop structure and pointer math.", 0.25),
    Phase("2 / Index & strides", "Replace modulo wrapping with tail masks. Use real strides or tl.make_block_ptr.", "No atomics changes.", 0.25),
    Phase("3 / Atomics->private", "Privatize accumulators per CTA. One write per output tile.", "No new kernels or API changes.", 0.5),
    Phase("4 / Coalesce/layout", "Coalesce loads/stores; adopt block pointers; adjust tile shapes.", "Algorithm unchanged.", 0.5),
    Phase("5 / Meta tune", "Sweep BLOCK_SIZE_{M,N,K}, num_warps, num_stages.", "Emit one patch per turn.", 0.25),
    Phase("6 / Recompute vs read", "Recompute-vs-read forward intermediates.", "No new atomics.", 0.25),
)

class BaseStrategy:
    def next_phase(self, parity_ok: bool, last_runtime: Optional[float]) -> Tuple[str, float]:
        # Default: a simple optimize header with global guardrails and neutral temperature
        return "Phase = optimize.\n" + GLOBAL_GUARDRAILS, 0.7
    def advance(self, changed: bool, parity_ok: bool) -> None:
        # No-op in the base class.
        pass

class RegularStrategy(BaseStrategy):
    def __init__(self, temp: float = 0.7) -> None:
        self._temp = temp
    def next_phase(self, parity_ok: bool, last_runtime: Optional[float]) -> Tuple[str, float]:
        header = "Phase = optimize. Improve performance without changing numerics.\n" + GLOBAL_GUARDRAILS
        return header, self._temp

class PhasedStrategy(BaseStrategy):
    def __init__(self, phases: Sequence[Phase] = PHASES) -> None:
        self.phases = list(phases)
        self.i = 0
    def next_phase(self, parity_ok: bool, last_runtime: Optional[float]) -> Tuple[str, float]:
        # Emit a concise header describing the allowed scope for this step.
        p = self.phases[self.i]
        header = (
            f"Phase = {p.name}. ONLY do: {p.goal}.\n"
            f"Success gates: gradcheck_ok={parity_ok}.\n"
            + GLOBAL_GUARDRAILS + "\n"
            f"Guardrails (phase-specific): {p.guardrails}"
        )
        return header, p.temp
    def advance(self, changed: bool, parity_ok: bool) -> None:
        # Advance to the next phase only on a successful, gradcheck-passing change.
        if changed and parity_ok and self.i < len(self.phases) - 1:
            self.i += 1

def make_strategy(mode: str, default_temp: float = 0.7) -> BaseStrategy:
    # Factory: allows toggling strategy with an env flag without touching the loop.
    return PhasedStrategy() if mode == "phased" else RegularStrategy(default_temp)
