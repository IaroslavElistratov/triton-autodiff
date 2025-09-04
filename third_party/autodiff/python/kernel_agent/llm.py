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


VERBOSE = str(os.environ.get("KERNEL_AGENT_VERBOSE", "")).strip().lower() in ("1", "true", "yes", "y")

BEGIN_PATCH = "*** Begin Patch"
END_PATCH = "*** End Patch"

# helpers (strict hunk check + target normalization)

def print_model_prompt(msg):
    RED, RESET = "\x1b[31m", "\x1b[0m"
    print("=" * 60 + " MODEL SEES" + "=" * 60)
    print(f"{RED}{msg}{RESET}\n")
    print("=" * 130)

def extract_patch(text: str | None) -> str | None:
    if not text:
        return None
    i = text.find(BEGIN_PATCH)
    j = text.rfind(END_PATCH)
    return text[i:j+len(END_PATCH)] if i != -1 and j != -1 else None

def has_real_change(patch_text: str | None) -> bool:
    if not patch_text or (BEGIN_PATCH not in patch_text) or (END_PATCH not in patch_text):
        return False
    for ln in patch_text.splitlines():
        s = ln.lstrip()
        if not s or s.startswith(("***", "@@", "+++", "---")):
            continue
        if s[0] in "+-":
            return True
    return False

def _ensure_update_file_target(patch_text: str, target_file: str) -> str:
    """Ensure patch targets exactly one file by inserting/replacing the Update header.

    - If an "*** Update File:" header exists, rewrite its path to target_file
    - If no file header exists at all, insert "*** Update File: {target_file}"
      immediately after "*** Begin Patch". Intentionally do not support
      Add/Delete/Move in this workflow to keep the LLM output minimal.
    """
    lines = patch_text.splitlines()
    begin_idx = None
    updated = False
    for i, ln in enumerate(lines):
        if ln.startswith("*** Begin Patch"):
            begin_idx = i
        if ln.startswith("*** Update File:"):
            lines[i] = f"*** Update File: {target_file}"
            updated = True
            break
    if not updated and begin_idx is not None:
        insert_at = begin_idx + 1
        lines.insert(insert_at, f"*** Update File: {target_file}")
    return "\n".join(lines)


def _env_truthy(name: str, default: str = "0") -> bool:
    """Parse boolean-like env flags from environment."""
    val = os.environ.get(name, default)
    return str(val).lower() not in ("0", "", "false", "no", "off")


# Canonical apply_patch.md contract presented to the model;
# the patcher is the source of truth for validation and application.
_APPLY_PATCH_SPEC = """
Return ONE apply_patch.md block. No prose.

Use this exact envelope (do NOT include file headers; the system will add them):
*** Begin Patch
@@ [optional hunk header]
- old line from the current file
+ new line to write
*** End Patch

Rules:
- Do not include any of: "*** Update File:", "*** Add File:", "*** Delete File:", or "*** Move to:".
- At least one '-' line per hunk to anchor to real lines (no pure insert-only hunks).
- Hunk lines must be prefixed with one of:
  - blank space ( ) for unchanged context
  - '-' for removed text (from the current file)
  - '+' for inserted text
- Do not include any text outside the patch block.

Minimal example:
*** Begin Patch
@@ def some_function(...):
-    x = old_value
+    x = new_value
*** End Patch
"""


@dataclass
class MinimalLLMPatchProvider:
    reasoning_effort: str
    temperature: float = 0.0
    max_tokens: int = 1536
    context: int | None = None
    last_thinking: str = ""
    # Tracks backend stop reason (e.g., "max_tokens") to disambiguate truncation
    # from other failure modes and report errors upstream.
    last_stop_reason: str = ""
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

        # system prompt
        system = (
            "You are a CUDA/Triton kernel optimizer. You are called as part of the workflow: generate initial backward pass -> [gradcheck -> optimize -> benchmark] the part in the brackets repeats in a for-loop. You are the 'optimize' step. "
            "You will be called multiple times to try to improve your kernel, so don't try to output final solution in one shot. You will have opportunities to refine it later."
            "Output ONLY an apply_patch.md patch. No prose. If you wrote any analysis above, end with exactly one apply_patch.md block. "
            "Backward file contains a Python function `backward(*inputs, *grads)` which computes per-input gradients. "
            "Use this backward kernel provided to you as the starting point and make edits to improve its performance. "
            # "Do not try to derive backward mathematically from scratch this is hallucination- and error- prone, instead use the provided backward kernel and gradient annotations for your reference."
            "You can rewrite backward kernel from scratch; but preserve function names and pointer/mask semantics. "
            "The backward file contains BOTH the backward Triton kernel and a generated backward stub; you can (and likely should) edit both. "
            "You can change function signatures (but you must preserve function args prefixed with backward_*). Do not rename or move the file. "
            "More details about the initial backward kernel: "
            "1. signature: `backward(arg1, arg2, grad_arg1, grad_arg2)` for every *pointer* arg 'i' in inputs, there's a corresponding 'arg_i' containing pointer to gradient tensors wrt that input 'i'). "
            "2. recomputing intermediate activations from the forward pass: variable names inside the kernel contain prefixes fwd_*, bwd_* -- the former means this is some intermediate value from the forward pass recomputed in backward, the latter means this is a value added by a derivative formula of some forward operator. "
            "3. heavily unrolled: for loops from the forward kernel were unrolled -- you should re-introduce back the for-loops, as it'll clearly improvement the performance "
            "4. atomics: kernel uses atomics -- try privatizing the accumulation to the same memory location to a single CTA to avoid atomics, as it'll clearly improvement the performance "
            "If gradient summary is OK assume the kernel and stub compute gradients correctly -- do not second guess it. "
            "Reply with substantive code changes, not with comment/docstring 'touch' patches. "
            # "If you have NO actual change to propose, return an EMPTY no-op patch:\n*** Begin Patch\n*** End Patch\n"
            # "Assume contiguous inputs.; When appropriate, use tail masks to support ragged tiles."
            "You must only have a single backward kernel and a single backward stub, do not attempt to create multiple backward kernels or stubs."
            "Emit ONE apply_patch.md patch only. "
        )
        # user prompt
        spec_text = _APPLY_PATCH_SPEC
        user = (
            spec_text
            + f"\n\nPhase: {phase}\n"
            # todo-high: maybe don't manually save it but let llm an option to write a note for the next iteration and work done in the current iteration
            + f"Context from previous iterations:\n{self._history_block()}\n\n"
            + "Forward snippet:\n" + fwd_kernel_snippet + "\n\n"
            # Benchmark summary: {bench_summary}
            # Profiler hint: {profile_hint}
            # path is not shown to the model; the workflow injects the target file name
            + "Backward snippet:\n" + bwd_kernel_snippet + "\n"
            + f"Gradcheck: {grad_summary}\n"
        )

        if VERBOSE:
            # colored preview of the prompt for debugging
            print_model_prompt(user)

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

        # Record stop reason to distinguish truncation vs normal stops
        try:
            self.last_stop_reason = (resp.response_metadata or {}).get("stop_reason", "")  # type: ignore[attr-defined]
        except Exception:
            self.last_stop_reason = ""
        if self.last_stop_reason:
            # breadcrumb for visibility in next prompt and logs
            self.remember("llm.stop_reason", self.last_stop_reason)

        # Prefer patch from final text only; if absent, retry once below
        # Do not try to extract from thinking channel
        patch_text = extract_patch(text)

        # One strict retry if the model ignored the format or produced an empty/no-op patch.
        # Also capture the retry's stop_reason to distinguish true truncation.
        if (patch_text is None or not has_real_change(patch_text)):
            retry_system = "Return ONE non-empty apply_patch.md block. No prose."
            retry_user = user + "\nIMPORTANT: Your previous output had no usable patch. Emit exactly one patch block."
            retry_msgs = [{"role": "system", "content": retry_system}, {"role": "user", "content": retry_user}]
            resp2 = self._sampler(retry_msgs, on_thinking_chunk=thinking_sink)
            text2 = (getattr(resp2, "response_text", "") or "").strip()
            patch2 = extract_patch(text2)
            # Update stop reason from retry attempt as well
            try:
                stop2 = (resp2.response_metadata or {}).get("stop_reason", "")
            except Exception:
                stop2 = ""
            if stop2:
                self.last_stop_reason = stop2
                self.remember("llm.stop_reason.retry", self.last_stop_reason)
            if patch2:
                patch_text = patch2

        # If still invalid, fail fast. If hit the token limit, report that
        # explicitly so the orchestrator can avoid misclassifying it as a
        # patching/apply failure.
        if (patch_text is None or not has_real_change(patch_text)):
            if (self.last_stop_reason or "").lower() == "max_tokens":
                raise RuntimeError("LLM stopped due to max_tokens; output truncated; no patch produced")
            raise RuntimeError("LLM returned no actionable patch")

        # Ensure target file line points to requested file (fix any mismatched path)
        patch_text = _ensure_update_file_target(patch_text, bwd_file)
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
        stopped_on_end_patch = False

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
                        stopped_on_end_patch = True
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
        # Detect if we likely hit the token limit without finishing the patch
        hit_token_limit = (not stopped_on_end_patch) and (len(generated) >= int(self.max_tokens or 0))
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
            response_metadata={
                "thinking": (all_thinking if capture_thinking else ""),
                "stop_reason": ("max_tokens" if hit_token_limit else ("end_patch" if stopped_on_end_patch else "eos")),
            },
        )

