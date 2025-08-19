from __future__ import annotations
from dataclasses import dataclass

# Prefer the OSS sampler shipped in the repo.
try:
    from evals.chat_completions_sampler import ChatCompletionsSampler
except Exception as e:
    raise RuntimeError(
        "ChatCompletionsSampler not found; ensure the OSS repo is on PYTHONPATH."
    ) from e


_APPLY_PATCH_MD_SPEC = """
Return ONLY one patch in apply_patch.md format (no prose):

*** Begin Patch
*** Update File: <relative/path/to/file.py>
@@
- old line
+ new line
*** End Patch
"""


@dataclass
class MinimalLLMPatchProvider:
    model: str = "gpt-oss-20b"
    temperature: float = 0.0
    max_tokens: int = 1536

    def __post_init__(self) -> None:
        self._sampler = ChatCompletionsSampler(
            model=self.model,
            system_message=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def propose_patch(self, *, phase: str, device: str, target_file: str,
                      kernel_snippet: str, grad_summary: str,
                      bench_summary: str, profile_hint: str) -> str:
        system = "You are a CUDA/Triton kernel optimizer. Output ONLY an apply_patch.md patch. No prose."
        user = f"""{_APPLY_PATCH_MD_SPEC}

Phase: {phase}
Device: {device}
Target file: {target_file}

Kernel snippet:
```

{kernel\_snippet}

```

Gradient check summary:
{grad_summary}

Benchmark summary:
{bench_summary}

Profiler hint:
{profile_hint}
"""
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        resp = self._sampler(msgs)
        text = (getattr(resp, "response_text", "") or "").strip()
        # If the model added any extra text, keep only the patch block.
        begin, end = "*** Begin Patch", "*** End Patch"
        if begin in text and end in text:
            s, e = text.index(begin), text.index(end) + len(end)
            return text[s:e].strip()
        return text  # trust the model if already clean