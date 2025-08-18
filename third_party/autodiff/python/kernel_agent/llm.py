from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol

from evals.chat_completions_sampler import ChatCompletionsSampler  # gpt-oss

APPLY_PATCH_MD_SPEC = """
You MUST output ONLY one patch using apply_patch.md format:

*** Begin Patch
*** Update File: <relative/path/to/file.py>
@@
- old line
+ new line
*** End Patch

Rules:
- No prose outside the patch.
- Keep changes minimal and correctness-preserving.
"""

@dataclass
class PatchContext:
    device_info: str
    kernel_snippet: str
    grad_check_summary: str
    benchmark_summary: str
    profile_hint: str
    target_file_path: str
    phase: str = "optimize"  # "fix" | "optimize"

class PatchProvider(Protocol):
    def propose_patch(self, ctx: PatchContext) -> str: ...

@dataclass
class LLMClient(PatchProvider):
    model: str = "gpt-4.1-2025-04-14"
    max_tokens: int = 1500

    def __post_init__(self) -> None:
        if ChatCompletionsSampler is None:
            raise RuntimeError("ChatCompletionsSampler not found; add gpt-oss to PYTHONPATH.")
        self._sampler = ChatCompletionsSampler(model=self.model, system_message=None, max_tokens=self.max_tokens)

    def propose_patch(self, ctx: PatchContext) -> str:
        sys = "You are an expert GPU optimization assistant. Output ONLY apply_patch.md patch; no prose."
        user = f"""{APPLY_PATCH_MD_SPEC}
