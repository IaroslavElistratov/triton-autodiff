from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Any, Callable

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

def _env_truthy(name: str, default: str = "0") -> bool:
    """Parse boolean-like env flags from environment."""
    val = os.environ.get(name, default)
    return str(val).lower() not in ("0", "", "false", "no", "off")




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
    # Max breadcrumbs kept for prompt context; small to avoid token bloat.
    history_max_items: int = 8

    # Optional streaming sink for thinking tokens; if None and
    # KERNEL_AGENT_STREAM_THINKING is truthy, a default console printer is used.
    on_thinking_chunk: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self._sampler = _GenerateSampler(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            context=self.context,
            reasoning_effort=self.reasoning_effort,
        )
        # Minimal per-run context to give the model continuity across iterations.
        # Stored as small text snippets. Not a chat transcript.
        self._history: list[str] = []

    # breadcrumb API for per-iteration context used by the orchestrator.
    def remember(self, kind: str, text: str) -> None:
        """Record a compact breadcrumb for later prompts."""
        try:
            s = str(text).strip()
        except Exception:
            s = "<unprintable>"
        if not s:
            return
        self._history.append(f"[{kind}]\n{s}")
        if len(self._history) > self.history_max_items:
            self._history = self._history[-self.history_max_items:]

    # Render breadcrumbs as a single block for injection into prompts.
    def _history_block(self) -> str:
        return ("\n\n".join(self._history)) if self._history else "(none)"

    def _trim_history(self) -> None:
        if len(self.history) > self.history_cap:
            self.history = self.history[-self.history_cap:]

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
            # todo-high: allow to re-write from scratch?
            "Do not rewrite backward kernel from scratch; preserve function names/signatures and pointer/mask semantics. "
            # todo: attach stub, so that model sees details it
            # "Do not add a stub for that kernel, this is already handled outside of this file -- just assume the stub is present"
            "More details about the initial backward kernel: "
            "1. signature: `backward(*inputs, arg_1, arg_2)` for every *pointer* arg 'i' in inputs, there's a corresponding 'arg_i' containing pointer to gradient tensors wrt that input 'i'). "
            "2. recomputing intermediate activations from the forward pass: variable names inside the kernel contain prefixes fwd_*, bwd_* -- the former means this is some intermideate value from the forward pass recomputed in backward, the latter means this is a value added by a derivative formular of some forward operator. "
            "3. heavily unrolled: for loops from the forward kernel were unrolled -- can start by fixing that, as it would clearly improvement the performance "
            "If you have NO actual change to propose, return an EMPTY no-op patch:\n*** Begin Patch\n*** End Patch\n"
            # "If gradient summary is OK, don't second guess it -- assume the gradient is correct"
        )
        user = f'''{_APPLY_PATCH_MD_SPEC}

Phase: {phase}

Context from previous iterations:
{self._history_block()}

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

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        # Streaming toggle via a single env flag; install default sink if enabled.
        thinking_sink = self.on_thinking_chunk
        if thinking_sink is None and _env_truthy("KERNEL_AGENT_STREAM", "0"):
            def _print_sink(chunk: str) -> None:
                # keep minimal/no prefix to avoid noisy logs; orchestrator can add one
                print(chunk, end="", flush=True)
            thinking_sink = _print_sink

        resp = self._sampler(msgs, on_thinking_chunk=thinking_sink)
        text = (getattr(resp, "response_text", "") or "").strip()

        # Capture thinking, if provided by backend (may contain patch when final was truncated)
        try:
            self.last_thinking = (resp.response_metadata or {}).get("thinking", "")  # type: ignore[attr-defined]
        except Exception:
            self.last_thinking = ""

        begin, end = "*** Begin Patch", "*** End Patch"

        def _extract_patch(src: str) -> str | None:
            if not src:
                return None
            if begin in src and end in src:
                s, e = src.index(begin), src.index(end) + len(end)
                return src[s:e].strip()
            return None

        # Prefer patch from final text; otherwise try thinking; otherwise synthesize no-op
        patch_text = _extract_patch(text)
        if patch_text is None:
            patch_text = _extract_patch(self.last_thinking)
        if patch_text is None:
            # No real changes: return a true no-op patch (avoid emitting an Update line without hunks).
            return f"{begin}\n{end}"

        # If an explicit Update target is present, correct it to the requested file; otherwise leave as-is.
        if "*** Update File:" in patch_text:
            patch_lines = patch_text.splitlines()
            for i, ln in enumerate(patch_lines):
                if ln.strip().startswith("*** Update File:"):
                    patch_lines[i] = f"*** Update File: {bwd_file}"
            return "\n".join(patch_lines)
        return patch_text


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
    def __call__(self, message_list: list[dict[str, str]], on_thinking_chunk: Callable[[str], None] | None = None) -> SamplerResponse:
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
        # Avoid stopping at channel boundaries; let the model emit the final patch block
        stop_tokens = []

        generated: list[int] = []
        # streaming controls (single toggle)
        parse_every = 8
        emitted_chars = 0

        for idx, out in enumerate(self.generator.generate(
            input_tokens,
            stop_tokens=stop_tokens,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            return_logprobs=False,
        )):
            token = int(out[0]) if isinstance(out, tuple) else int(out)
            generated.append(token)

            # Stream decoded deltas and early-stop on patch terminator.
            if on_thinking_chunk and (idx + 1) % parse_every == 0:
                try:
                    decoded = self.encoding.decode(generated)
                    if len(decoded) > emitted_chars:
                        on_thinking_chunk(decoded[emitted_chars:])
                        emitted_chars = len(decoded)
                    if "*** End Patch" in decoded:
                        break
                except Exception:
                    # streaming should never be fatal
                    pass

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
        # Cap thinking for metadata only (do not re-stream to avoid duplicates).
        all_thinking = "".join(thinking_parts)
        max_thinking_chars = int(os.environ.get("KERNEL_AGENT_THINKING_MAX_CHARS", "0") or "0")
        if max_thinking_chars > 0 and len(all_thinking) > max_thinking_chars:
            all_thinking = all_thinking[-max_thinking_chars:]
        # In stub mode, if nothing meaningful generated, return empty no-op patch
        if not text.strip():
            text = "*** Begin Patch\n*** End Patch"
        capture_thinking = _env_truthy("KERNEL_AGENT_CAPTURE_THINKING", "1")
        return SamplerResponse(
            response_text=text,
            actual_queried_message_list=message_list,
            response_metadata={"thinking": (all_thinking if capture_thinking else "")},
        )

