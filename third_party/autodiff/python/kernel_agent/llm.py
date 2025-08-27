from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Any

from gpt_oss.evals.types import SamplerResponse
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)




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
    reasoning_effort: str
    temperature: float = 0.0
    max_tokens: int = 1536
    context: int | None = None
    last_thinking: str = ""

    def __post_init__(self) -> None:
        self._sampler = _GenerateSampler(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            context=self.context,
            reasoning_effort=self.reasoning_effort,
        )

    # todo: use pply_patch.md instead of my custom instructions belo
    def propose_patch(self, *, phase: str,
                      fwd_kernel_snippet: str,
                      bwd_file: str, bwd_kernel_snippet: str,
                      grad_summary: str) -> str:
                    #   bench_summary: str, profile_hint: str) -> str:
        system = (
            "You are a CUDA/Triton kernel optimizer. Output ONLY an apply_patch.md patch. No prose. "
            "Backward file contains a Python function `backward(*inputs, *grads)` which computes per-input gradients. "
            "Use this backward kernel provided to you as the starting point and make edits to improve its performance. "
            "Do not rewrite backward kernel from scratch; preserve function names/signatures and pointer/mask semantics. "
            # todo: attach stub, so that model sees details it
            # "Do not add a stub for that kernel, this is already handled outside of this file -- just assume the stub is present"
            "More details about the initial backward kernel: "
            "1. signature: `backward(*inputs, arg_1, arg_2)` for every *pointer* arg 'i' in inputs, there's a corresponding 'arg_i' containing pointer to gradient tensors wrt that input 'i'). "
            "2. recomputing intermediate activations from the forward pass: variable names inside the kernel contain prefixes fwd_*, bwd_* -- the former means this is some intermideate value from the forward pass recomputed in backward, the latter means this is a value added by a derivative formular of some forward operator. "
            "3. heavily unrolled: for loops from the forward kernel were unrolled -- can start by fixing that, as it would clearly improvement the performance "
            # "If gradient summary is OK, don't second guess it -- assume the gradient is correct"
        )
        user = f'''{_APPLY_PATCH_MD_SPEC}

Phase: {phase}

Forward kernel:
{fwd_kernel_snippet}

Backward file: {bwd_file}
Backward kernel:
{bwd_kernel_snippet}

Gradient check summary:
{grad_summary}

'''
# Benchmark summary:
# {bench_summary}

# Profiler hint:
# {profile_hint}

        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        resp = self._sampler(msgs)
        text = (getattr(resp, "response_text", "") or "").strip()

        # todo-now:
        # slices the model output between the sentinels *** Begin Patch and *** End Patch and discards everything else.
        # If no sentinel block is found, it returns the whole text as-is.

        # If the model added any extra text, keep only the patch block.
        # Capture thinking, if provided by backend
        try:
            self.last_thinking = (resp.response_metadata or {}).get("thinking", "")  # type: ignore[attr-defined]
        except Exception:
            self.last_thinking = ""

        begin, end = "*** Begin Patch", "*** End Patch"
        patch_text = text
        if begin in text and end in text:
            s, e = text.index(begin), text.index(end) + len(end)
            patch_text = text[s:e].strip()

        # Ensure target file line points to requested file
        patch_lines = patch_text.splitlines()
        for i, ln in enumerate(patch_lines):
            if ln.strip().startswith("*** Update File:"):
                patch_lines[i] = f"*** Update File: {target_file}"
                break
        return "\n".join(patch_lines)


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


# copy from gpt_oss/generate.py (adapted to Harmony)
class _GenerateSampler:

    def __init__(self, temperature: float, max_tokens: int, context: int, reasoning_effort: str) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backend = os.environ.get("KERNEL_AGENT_BACKEND", "stub").lower()
        self.checkpoint = os.environ.get("KERNEL_AGENT_CHECKPOINT", "")
        self.context = context
        self.reasoning_effort = reasoning_effort
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.generator: Any | None = None
        self._init_backend()

    def _init_backend(self) -> None:
        match self.backend:
            case "torch":
                from gpt_oss.torch.utils import init_distributed
                from gpt_oss.torch.model import TokenGenerator as TorchGenerator
                device = init_distributed()
                self.generator = TorchGenerator(self.checkpoint, device=device)
            case "triton":
                from gpt_oss.torch.utils import init_distributed
                from gpt_oss.triton.model import TokenGenerator as TritonGenerator
                device = init_distributed()
                self.generator = TritonGenerator(self.checkpoint, context=self.context, device=device)
            case "vllm":
                from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator
                self.generator = VLLMGenerator(self.checkpoint, tensor_parallel_size=2)
            case "stub":
                # class _NoopPatcher:
                #     def propose_patch(self, *_, **__):
                #         return "*** Begin Patch\n*** End Patch"
                # llm = _NoopPatcher()
                # Generator stub that echoes back a no-op patch
                class _StubGen:
                    def generate(self, *_ , **__):
                        # Emit a trivial no-op apply_patch block
                        text = "*** Begin Patch\n*** End Patch"
                        # Return once, in tokenized form; our stub encoding just decodes raw
                        yield (0,)
                self.generator = _StubGen()
            case _:
                raise ValueError(f"Invalid backend: {self.backend}")


    # todo-low: simplfiy
    def __call__(self, message_list: list[dict[str, str]]) -> SamplerResponse:
        """ uses Harmony encoding for single-response generation """

        # Extract the simple system and user contents
        system_text = next((m.get("content", "") for m in message_list if m.get("role") == "system"), "")
        user_text = next((m.get("content", "") for m in message_list if m.get("role") == "user"), "")

        # Build Harmony messages: structured system controls + developer instructions + user text
        system_message = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(
                REASONING_EFFORT.get(self.reasoning_effort, ReasoningEffort.LOW)
            ),
        )
        messages = [system_message]
        if system_text:
            dev = DeveloperContent.new().with_instructions(system_text)
            messages.append(Message.from_role_and_content(Role.DEVELOPER, dev))
        messages.append(Message.from_role_and_content(Role.USER, user_text))

        conversation = Conversation.from_messages(messages)
        input_tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        stop_tokens = self.encoding.stop_tokens_for_assistant_actions()

        generated: list[int] = []
        for out in self.generator.generate(
            input_tokens,
            stop_tokens=stop_tokens,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            return_logprobs=False,
        ):
            token = int(out[0]) if isinstance(out, tuple) else int(out)
            generated.append(token)

        # Parse the completion tokens into Harmony messages and extract final text
        entries = self.encoding.parse_messages_from_completion_tokens(generated, Role.ASSISTANT)
        final_text_parts: list[str] = []
        thinking_parts: list[str] = []
        for entry in entries:
            entry_dict = entry.to_dict()
            channel = entry_dict.get("channel")
            parts = [c.get("text", "") for c in entry_dict.get("content", []) if isinstance(c, dict) and c.get("text")]
            if channel == "final":
                final_text_parts.extend(parts)
            elif channel and channel != "tool":
                thinking_parts.extend(parts)
        text = "".join(final_text_parts) if final_text_parts else self.encoding.decode(generated)
        # In stub mode, if nothing meaningful generated, return empty no-op patch
        if not text.strip():
            text = "*** Begin Patch\n*** End Patch"
        return SamplerResponse(
            response_text=text,
            actual_queried_message_list=message_list,
            response_metadata={"thinking": "\n".join(thinking_parts)},
        )

