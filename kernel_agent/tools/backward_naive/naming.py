from enum import Enum
import keyword
import re
from typing import Optional, Set

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sanitize(base: Optional[str]) -> str:
    s = (base or "v").strip()
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if not s or s[0].isdigit():
        s = f"v_{s}"
    if keyword.iskeyword(s):
        s = f"{s}_"
    return s if _IDENT.match(s) else "v"


class NameStyle(str, Enum):
    # Naming policy is chosen by the raiser. We intentionally avoid any policy
    # in codegen/utils to keep one single authority for final Python identifiers.
    UNIQUE = "unique"                  # always unique suffixing when needed
    REUSE_AFTER_LAST_USE = "reuse"     # reuse base name only after last use


# comment:
# REUSE in NameAllocator basically allows to re-use same variable name in another
# variable, if and only if the first variable is never gonna be used downstream

# todo:
# name_style: str = "reuse" doest work -- either fix or delete it and NameAllocator to simplify logic.
# Root cause: reuse mode can drop single-use temp bindings (like fwd_q_1 = fwd_Q_block_ptr_1 + fwd_q)
# while the addptr fallback still references that temp, producing a NameError.
# For addptr/make_block_ptr calls, one index argument was a temporary like fwd_q_1 = fwd_Q_block_ptr_1 + fwd_q.
# With reuse on, that temporary got elided (no assignment printed) because it’s single-use.
# But the call site still referenced fwd_q_1 instead of inlining the expression, so the generated Python
# referenced a name that was never defined -> NameError



class NameAllocator:
    """Minimal allocator for emission-time names.

    Rationale:
    - Uniqueness must be enforced where we have liveness/region context: the
      raiser. Codegen emits clean stems only; utils just sanitizes.
    - REUSE mode enables base-name reuse after last use without changing
      semantics; UNIQUE mode is deterministic and conservative.
    """
    def __init__(self, style: NameStyle = NameStyle.UNIQUE):
        self.style = style
        self._used: Set[str] = set()
        self._reusable: Set[str] = set()

    def claim(self, base_hint: Optional[str]) -> str:
        """Return a unique Python identifier derived from base_hint.

        - UNIQUE: first claim is base; subsequent claims get _1/_2…
        - REUSE: if base was released at last use, reclaim it; otherwise suffix.
        """
        base = _sanitize(base_hint)
        if self.style == NameStyle.REUSE_AFTER_LAST_USE:
            if base in self._reusable and base not in self._used:
                self._reusable.discard(base)
                self._used.add(base)
                return base
        if base not in self._used:
            self._used.add(base)
            return base
        i = 1
        while True:
            cand = f"{base}_{i}"
            if cand not in self._used:
                self._used.add(cand)
                return cand
            i += 1

    def release(self, name: str) -> None:
        """Signal last use of a name.

        - UNIQUE: no-op; we never want to rebind the same identifier.
        - REUSE: allow the base to be reclaimed by a later claim.
        """
        if self.style == NameStyle.REUSE_AFTER_LAST_USE:
            if name in self._used:
                self._used.discard(name)
            self._reusable.add(name)


