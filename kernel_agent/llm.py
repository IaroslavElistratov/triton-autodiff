from __future__ import annotations
from dataclasses import dataclass
import os
# Parse Harmony function-call arguments from the tool channel.
# Rationale: tool-only extraction gives a deterministic boundary and
# avoids brittle substring scraping of "*** Begin/End Patch" in text
import json
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
    # StreamableParser yields channelized messages and assistant actions
    # (function/tool calls) using Harmony control tokens, so do not rely
    # on decoded text markers to find start/stop boundaries
    StreamableParser,
    # ToolDescription advertises a single function tool to the model so it can
    # deliver the patch payload structurally instead of free-form text
    ToolDescription,
    load_harmony_encoding,
)


from .utils import _env_truthy


VERBOSE = _env_truthy("KERNEL_AGENT_VERBOSE", "1")
KERNEL_AGENT_STREAM = _env_truthy("KERNEL_AGENT_STREAM", "1")

BEGIN_PATCH = "*** Begin Patch"
END_PATCH = "*** End Patch"

# helpers (strict hunk check + target normalization)

def print_model_prompt(msg_system, msg_user):
    RED, GREEN, RESET = "\x1b[31m", "\x1b[92m", "\x1b[0m"
    print("=" * 60 + " MODEL SEES" + "=" * 60)
    print(f"{RED}{msg_system}{RESET}\n")
    print(f"{GREEN}{msg_user}{RESET}\n")
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

# let llm.py be the single point which appends these guardrails
# enforcing a single functions.apply_patch({patch: ...}) call gives
# protocol-level boundaries (no "*** End Patch" loops; no early <|end|> cuts),
# and lets the host take control immediately after the tool payload arrives.
PATCH_SPEC = """
Contract: Call the function tool functions.apply_patch once with arguments {"patch": "<one apply_patch.md block>"}. No prose.
You may use the analysis channel for planning, but do not include the patcher tool call in analysis.

Patch guardrails:
- Emit exactly one tool call: functions.apply_patch({"patch": "..."}).
- Use analysis for reasoning; do not place the patcher tool call in analysis; call apply_patch once as your final action.
- Do not include any of: "*** Update File:", "*** Add File:", "*** Delete File:", or "*** Move to:".
- Do NOT include file name in your patch -- the system will add it automatically
- At least one '-' line per hunk to anchor to real lines (no pure insert-only hunks).
- Hunk lines must be prefixed with one of:
  - blank space ( ) for unchanged context
  - '-' for removed text (from the current file)
  - '+' for inserted text
- Do not include any text outside the patch block.
- Do NOT change the backward stub's signature.

Use this exact envelope:
*** Begin Patch
@@ [hunk header]
- old line from the current file
+ new line to write
*** End Patch

Minimal example:
*** Begin Patch
@@ def some_function(...):
-    x = old_value
+    x = new_value
*** End Patch
"""
# attention kernel is about that size, to introduce for loop need to at least indent almost all of the lines in the kernel (around 120 lines)
# "- ≤120 changed lines per patch.\n"


MEM_SPEC = """
MEMORY REQUIREMENT:
- Rewrite the backward kernel's docstring to briefly note this iteration (<= 15 lines; no code/diffs).
- Focus on: what you changed; what previously failed and why; and what key takeaways worth remembering for the next iteration.
- Do this in the SAME patch as your code edits. Do NOT submit a docstring-only patch.
"""


@dataclass
class MinimalLLMPatchProvider:
    reasoning_effort: str
    temperature: float = 0.0
    max_tokens: int = 1536
    context: int | None = None
    # Tracks backend stop reason (e.g., "max_tokens") to disambiguate truncation
    # from other failure modes and report errors upstream.
    last_stop_reason: str = ""
    # Max breadcrumbs kept for prompt context; small to avoid token bloat.
    history_max_items: int = 1

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

    # called by the orchestrator
    def finalize_last_tool_call(self, output: str) -> None:
        """Stash tool output; next OpenAI turn will inline-ack before the new prompt.

        Provider only provides the output; the sampler manages Responses API plumbing.
        OpenAI-only behavior: other backends do not use Responses API function_call_output.
        """
        backend = str(os.environ.get("KERNEL_AGENT_BACKEND", "triton")).lower()
        if backend != "openai":
            return
        # Delegate to sampler; it handles ChainState details
        self._sampler._chain.stash_output(output)

    # Allow per-phase temperature overrides (optional)
    def set_phase_temperature(self, mapping: dict[str, float]) -> None:
        self._phase_temp = getattr(self, "_phase_temp", {})
        self._phase_temp.update(mapping)

    def _phase_temperature(self, phase_key: str) -> float:
        if hasattr(self, "_phase_temp"):
            for key, val in self._phase_temp.items():
                if key in str(phase_key):
                    try:
                        return float(val)
                    except Exception:
                        pass
        return getattr(self, "temperature", 0.3)

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

    # def _trim_history(self) -> None:
    #     if len(self.history) > self.history_cap:
    #         self.history = self.history[-self.history_cap:]

    def propose_patch(self, *, phase: str,
                      fwd_kernel_snippet: str,
                      bwd_file: str, bwd_kernel_snippet: str,
                      state_facts=None) -> str:

        system = (
            "\n### Workflow context\n"
            "You are a Triton kernel optimizer. You are called as part of the workflow: generate initial backward pass -> [gradcheck -> optimize -> benchmark] the part in the brackets repeats in a for-loop. You are the 'optimize' step.\n"
            "You will have multiple turns to refine the backward kernel.\n" # , so do not propose large overly-eager kernel rewrites.
            "You will be provided a backward kernel which computes per-input gradients, use it as the starting point and make edits to improve its performance.\n"
            "You must adhere to user's Phase-specific goals and guardrails.\n"
            # "Do not try to derive backward mathematically from scratch this is hallucination- and error- prone, instead use the provided backward kernel and gradient annotations for your reference.\n"

            "\n### Allowed edits\n"
            # "You must modify backward kernel; but preserve function names and pointer/mask semantics.\n"
            "The backward file contains BOTH the backward Triton kernel and a generated backward stub; you can (and likely should) edit both.\n"
            "Do NOT change the backward stub's signature, you can edit body of the stub but not its signature.\n"
            "You can edit the backward kernel signature and its body (but not stub's signature). Do not rename or move the file.\n"
            "You must only have a single backward kernel and a single backward stub, do not attempt to create multiple backward kernels or stubs.\n"
            "You can edit _mk_block_ptr when it's present.\n"

            # todo: show this only in the 1st iter?
            # comment: do not instruct to e.g. "for loops" or "remove atomics" -- this is handled in Phase[s]. Below is just general info only
            "\n### Initial backward kernel details\n"
            # " * signature: `backward_kernel(arg1, arg2, grad_arg1, grad_arg2)` for every *pointer* arg 'i' in inputs, there's a corresponding 'arg_i' containing pointer to gradient tensors wrt that input 'i').\n"
            "* variable names inside the kernel contain prefixes fwd_*, bwd_* -- the former means this is some intermediate value from the forward pass recomputed in backward, the latter means this is a value added by a derivative formula of some forward operator.\n"
            "* single-iteration unrolled: the initial backward kernel covers the gradients for exactly one iteration of the original forward loop (loop flattened).\n"
            # " * single-iteration unroll: the forward loop is flattened; this backward kernel computes gradients for exactly one loop iteration (one tile/chunk) and does not iterate over the full extent used in the benchmark sweep.\n"
            # " * single-iteration unroll: loops from the forward kernel are unrolled; the provided backward kernel corresponds to differentiated version of exactly one iteration of those loops.\n"

            "\n### Patch requirements\n"
            "In each turn, you must call functions.apply_patch({patch: ...}) at most once, and only as your final action for that turn.\n"
            "Before calling it, form a high-level plan of your changes in your private reasoning and rehearse the patch.\n"
            "When you call functions.apply_patch, output only a real diff—no rule echoing, no commentary, no placeholders. Include the *** Begin Patch / *** End Patch envelope inside the patch argument; do not echo the rule list.\n"
            "Use the analysis channel for planning (no functions.apply_patch in analysis). Then, as your final action, call the function tool functions.apply_patch with arguments {\"patch\": \"<one apply_patch.md block>\"}. No prose.\n"

            "\n### Docstring requirements\n"
            # not "what to try next;" -- because this will be dictated by Strategy, model should not decide that
            "Maintain a concise iteration note (<= 15 lines; no code/diffs; no future plans) in the backward Triton kernel's docstring. That note should contain a concise but informative note to your future self about this iteration.\n"
            "Whenever you modify the backward kernel code you MUST rewrite/update this docstring in the same patch as your code edits, focusing on key takeaways worth remembering for the next iteration (e.g. what changed, what failed and why, etc.).\n"
            "This docstring update is always required alongside your code edits. Do not submit a docstring-only patch.\n"

            "\n### Minor\n"
            # observed error cases:
            # "Reply with substantive code changes.\n"
            # "Assume contiguous inputs.; When appropriate, use tail masks to support ragged tiles.\n"
            "- If gradient summary is OK assume the backward kernel and stub compute gradients correctly -- do not second guess it.\n"
            "- Always keep the backward_kernel's call-site (in backward_stub) and the backward_kernel's signature in sync.\n"
            "   If you change the backward_kernel signature, don't forget to update the kernel's call-site in the backward_stub and vice versa.\n"
            "   Remember that the backward_kernel is called from the backward_stub. So when you e.g. add/replace/remove arguments in the backward_kernel function definition, always make sure that you also updated the place (in the backward_stub) where this backward_kernel is called from.\n"
            "- Under no circumstance replace triton kernel with pytorch operations.\n"
            "- Inside kernel use `tl.*` APIs, not `triton.*`\n"
        )

        # # Include initial kernel details only when not phased, or when phased and in Phase 1.
        # phase_text = str(phase).strip().lower()
        # is_phased = str(os.environ.get("KERNEL_AGENT_STRATEGY", "")).strip().lower() == "phased"
        # include_initial_details = (not is_phased) or (is_phased and phase_text.startswith("phase = 1"))
        # initial_kernel_details = initial_kernel_details if include_initial_details else ""
        # system = base + initial_kernel_details + tail

        # user prompt
        facts_lines = "\n".join(f"{k}={v}" for k, v in (state_facts or {}).items())
        user = (
            f"\n{phase}\n"
            # todo-high: maybe don't manually save it but let llm an option to write a note for the next iteration and work done in the current iteration
            + f"Context from previous iterations:\n{self._history_block()}\n\n"
            + "Forward snippet:\n" + fwd_kernel_snippet + "\n\n"
            # path is not shown to the model; the workflow injects the target file name
            + "Backward snippet:\n" + bwd_kernel_snippet + "\n"
            # optional: gradcheck, profiler hint, bench -- info is carried in state_facts below
            + (f"State:\n{facts_lines}\n" if facts_lines else "")
            + PATCH_SPEC
            + MEM_SPEC
        )

        if VERBOSE:
            # colored preview of the prompt for debugging
            print_model_prompt(system, user)

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        # Streaming toggle via a single env flag; install default sink if enabled.
        thinking_sink = self.on_thinking_chunk
        if thinking_sink is None and KERNEL_AGENT_STREAM:
            def _print_sink(chunk: str) -> None:
                # keep minimal/no prefix to avoid noisy logs; orchestrator can add one
                print(chunk, end="", flush=True)
            thinking_sink = _print_sink

        # Temperature is controlled by the orchestrator

        # First attempt: sampler is tool-only, so response_text is the tool
        # payload (or empty). Keep this single-turn, single-action
        resp = self._sampler(msgs, on_thinking_chunk=thinking_sink)
        text = (getattr(resp, "response_text", "") or "").strip()

        # Capture thinking, if provided by backend (may contain patch when final was truncated)
        # self.last_thinking = resp.response_metadata.get("thinking", "")
        self.last_stop_reason = resp.response_metadata.get("stop_reason", "")  # type: ignore[attr-defined]
        if self.last_stop_reason:
            # minimal breadcrumbs: persist the backend stop reason for visibility across turns
            self.remember("llm.stop_reason", self.last_stop_reason)

        # The sampler returns the tool payload; extract_patch double-checks
        # that a single apply_patch.md window exists and is non-empty
        patch_text = extract_patch(text)

        # One strict retry if the model ignored the format or produced an empty/no-op patch.
        # If the first attempt hit the token limit, explicitly instruct the model to
        # emit ONLY the patch block on retry.
        if (patch_text is None or not has_real_change(patch_text)):

            # this isn't really needed becuase above we save "remember" stop reason anyway
            # # If the sampler hit the token cap with no tool call, surface that explicitly.
            # reason_line = "Previous output truncated (max_tokens). " if (self.last_stop_reason == "max_tokens") else ""

            # Minimal second attempt: demand the tool call and forbid analysis patch text.
            # This is intentionally terse to reduce drift and token bloat.
            retry_msgs = [
                {"role": "system", "content": "Return a single patch now by calling functions.apply_patch({patch: ...}). No analysis patch text."},
                {"role": "user", "content": user}
            ]

            # reached_limit = (self.last_stop_reason or "").lower() == "max_tokens"
            # if reached_limit:
            #     retry_system = "Previous output truncated (max_tokens). Return ONE complete apply_patch.md block only. No prose."
            #     retry_user = user + "\nIMPORTANT: Your previous response truncated at the token limit. Emit exactly one apply_patch.md patch now. Do not include any analysis text."
            # else:
            #     retry_system = "Return ONE non-empty apply_patch.md block. No prose."
            #     retry_user = user + "\nIMPORTANT: Your previous output had no usable patch. Emit exactly one patch block."
            # retry_msgs = [{"role": "system", "content": retry_system}, {"role": "user", "content": retry_user}]

            resp2 = self._sampler(retry_msgs, on_thinking_chunk=thinking_sink)
            text2 = (getattr(resp2, "response_text", "") or "").strip()
            patch2 = extract_patch(text2)
            # Update stop reason from retry attempt as well
            stop2 = resp2.response_metadata.get("stop_reason", "")
            if stop2:
                # Persist retry stop reason breadcrumb for cross-turn visibility
                self.last_stop_reason = str(stop2)
                self.remember("llm.stop_reason.retry", self.last_stop_reason)
            if patch2:
                patch_text = patch2

        # If still invalid, fail fast. If we twice hit max_tokens with no usable patch,
        # make that explicit so the orchestrator can log/handle it distinctly.
        if (patch_text is None or not has_real_change(patch_text)):
            if self.last_stop_reason == "max_tokens":
                raise RuntimeError("LLM stopped due to max_tokens; output truncated; no patch produced")
            raise RuntimeError("LLM returned no actionable patch")

        # Ensure target file line points to requested file (fix any mismatched path)
        # Normalize the file header so the model doesn't spend
        # tokens on it and we avoid target-path drift in apply
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
        self.backend = os.environ.get("KERNEL_AGENT_BACKEND", "triton").lower()
        self.checkpoint = os.environ.get("KERNEL_AGENT_CHECKPOINT", "")
        self.context = context
        self.reasoning_effort = reasoning_effort
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.generator: Any | None = None
        self._init_backend()
        # Responses API chaining state (OpenAI backend only)
        self._chain = ChainState()

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
            case "openai":
                # OpenAI backend: no local token generator.
                # Keep a persistent API client to avoid re-initializing per call.
                self.generator = None
                # Resolve model once from a single project-specific env.
                self._oa_model = os.environ.get("KERNEL_AGENT_OPENAI_MODEL", "gpt-5-mini")
                # Lazy import keeps OpenAI optional unless backend=openai.
                from openai import OpenAI  # type: ignore
                # Use default constructor; it reads OPENAI_API_KEY from env.
                self._oa_client = OpenAI()
            case _:
                raise ValueError(f"Invalid backend: {self.backend}")


    # Tool-only Harmony streaming and parse.
    #
    # Why tool-only:
    # - Deterministic boundary: assistant actions mark tool start/stop; no text scraping.
    # - Avoids two failure classes seen in logs: (1) endless "*** End Patch" in analysis,
    #   (2) breaking on the first <|end|> that only closes analysis, not final.
    # - Clean handoff: host applies the patch and controls the loop.
    def __call__(self, message_list: list[dict[str, str]],
                 on_thinking_chunk: Callable[[str], None] | None = None) -> SamplerResponse:
        """
        Single‑turn generation via Harmony (local backends) or OpenAI Responses API (for --backend openai).

        Local/Harmony contract:
          - The model emits one functions.apply_patch tool call with the patch payload.
          - We stream tokens, stop after the first tool payload, and return it (do not apply it here).

        OpenAI/Responses contract:
          - The model must emit one function_call named "apply_patch" with {"patch": "<apply_patch.md>"}.
          - We make one non‑stream responses.create() call with a single tool and tool_choice="required".
          - We parse resp.output for that function_call and return its arguments; we do not apply the patch here.
        """

        system_text = next((m.get("content", "") for m in message_list if m.get("role") == "system"), "")
        user_text   = next((m.get("content", "") for m in message_list if m.get("role") == "user"), "")

        # OpenAI Responses API path
        if self.backend == "openai":
            client = self._oa_client
            model = self._oa_model
            tools = [{
                "type": "function",
                "name": "apply_patch",
                "description": "Apply a single apply_patch.md diff",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch": {
                            "type": "string",
                            "description": "*** Begin Patch ... *** End Patch"
                        }
                    },
                    "required": ["patch"],
                    # disallow extra args; only 'patch' allowed
                    "additionalProperties": False
                },
                # enforce JSON schema for tool args (may be ignored by some SDKs)
                "strict": True
            }]

            # these models don't support temperature arg
            is_reasoning = model.startswith("gpt-5") or model.startswith("o")
            temperature_kw = {} if is_reasoning else {"temperature": self.temperature}

            # Build request input applying inline tool-acknowledge if pending exists
            input_payload, prev_kw, used_inline_ack = self._chain.build_input(user_text)

            resp = client.responses.create(
                model=model,
                instructions=system_text,
                input=input_payload,
                tools=tools,
                tool_choice="required",
                reasoning={
                    "effort": self.reasoning_effort,
                    # todo:
                    # "summary": "auto",
                },
                max_output_tokens=self.max_tokens,
                **temperature_kw,
                **prev_kw,
            )

            # note this is needed: bc "client.responses.create" above can throw and my orchestrator catches this potential failure,
            # so preserve pending state on failure and clear it only after a successful call. Thus "used_inline_ack" variable
            #
            # After a successful inline acknowledge (ack), clear the pending-ack state.
            # Don’t clear pending state before the API call to avoid losing data on failure.
            self._chain.clear_if_used(used_inline_ack)

            patch_text = ""
            reasoning_summary = ""
            # capture tool call id to acknowledge the function_call
            last_call_id = None

            try:
                for item in (getattr(resp, "output", []) or []):
                    t = getattr(item, "type", None)

                    if t == "reasoning":
                        for part in getattr(item, "summary", []):
                            if getattr(part, "type", "") == "summary_text":
                                reasoning_summary += part.get("text", "")

                    # Responses returns custom function calls as function_call items
                    elif t == "function_call" and getattr(item, "name", "") == "apply_patch":
                        # capture the specific tool call id to finalize this turn
                        last_call_id = item.call_id
                        raw_args = getattr(item, "arguments", "") or ""
                        try:
                            obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            patch_text = (obj.get("patch", "") if isinstance(obj, dict) else str(raw_args)) or ""
                        except Exception:
                            patch_text = str(raw_args) or ""
                        break

                if not patch_text:
                    # Fallback if the model emitted plain text instead of a tool call
                    patch_text = extract_patch(getattr(resp, "output_text", "") or "") or ""
            except Exception:
                patch_text = ""

            if patch_text:
                pt = patch_text.strip()
                if BEGIN_PATCH not in pt and END_PATCH not in pt:
                    patch_text = f"{BEGIN_PATCH}\n{pt}\n{END_PATCH}"
                else:
                    patch_text = pt

            if VERBOSE and reasoning_summary:
                print("=== Reasoning summary ===\n" + reasoning_summary + "\n")

            if VERBOSE:
                try:
                    usage_dict = resp.model_dump()["usage"]
                    input_tokens = usage_dict["input_tokens"]
                    output_tokens = usage_dict["output_tokens"]
                    total_tokens = usage_dict["total_tokens"]
                    input_cached = usage_dict["input_tokens_details"]["cached_tokens"]
                    reasoning_tokens = usage_dict["output_tokens_details"]["reasoning_tokens"]
                    print(
                        f"=== Token usage === input={input_tokens} (cached={input_cached}) "
                        f"output={output_tokens} total={total_tokens} reasoning_tokens={reasoning_tokens}"
                    )
                except Exception:
                    pass

            # Update chain state so the next turn can finalize inline
            self._chain.on_response(resp.id, last_call_id)

            finish = "produced_patch" if patch_text else "no_patch"
            return SamplerResponse(
                response_text=patch_text,
                actual_queried_message_list=message_list,
                response_metadata={
                    "tool": ("functions.apply_patch" if patch_text else ""),
                    "stop_reason": finish,
                    "usage": resp.usage,
                    "reasoning_summary": reasoning_summary,
                    # expose raw response id to assist with debugging/analytics/chaining
                    "response_id": resp.id,
                    # expose call id so callers can finalize later if desired
                    "tool_call_id": last_call_id,
                },
            )


        # System + Developer (advertise the tool) + User
        sys_msg = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(
                REASONING_EFFORT[self.reasoning_effort]
            ),
        )
        # Attaches Developer/tool block only when developer instructions are provided
        dev = (
            DeveloperContent.new()
            .with_instructions(system_text)
            .with_function_tools([
                ToolDescription.new(
                    "apply_patch",
                    "Apply a single apply_patch.md diff",
                    parameters={
                        "type": "object",
                        "properties": {
                            "patch": {
                                "type": "string",
                                "description": "*** Begin Patch ... *** End Patch"
                            }
                        },
                        "required": ["patch"],
                    },
                )
            ])
        ) if system_text else DeveloperContent.new()
        msgs = [sys_msg]
        if system_text:
            msgs.append(Message.from_role_and_content(Role.DEVELOPER, dev))
        msgs.append(Message.from_role_and_content(Role.USER, user_text))

        convo = Conversation.from_messages(msgs)
        input_tokens = self.encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

        # Stream and parse assistant actions. Using Harmony-provided stop tokens for
        # assistant actions ensures parser sees complete tool envelopes.
        parser = StreamableParser(self.encoding, role=Role.ASSISTANT)
        token_count = 0
        for tok in self.generator.generate(
            input_tokens,
            stop_tokens=self.encoding.stop_tokens_for_assistant_actions(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            return_logprobs=False,
        ):
            parser.process(int(tok))
            token_count += 1
            if on_thinking_chunk and parser.last_content_delta and parser.current_channel != "final":
                on_thinking_chunk(parser.last_content_delta)

        # Prefer the first functions.apply_patch action; ignore analysis/final text completely.
        # Rationale: tool channel gives us a structured JSON/string payload and clear
        # boundaries. This eliminates sentinel scraping and channel mis-detections.
        patch_text = ""
        for m in parser.messages:
            d = m.to_dict()
            if d.get("recipient") == "functions.apply_patch":
                arg: str | dict | None = None
                for c in d.get("content") or []:
                    if isinstance(c, dict):
                        if "arguments" in c and c["arguments"]:
                            arg = c["arguments"]  # JSON string
                        elif "text" in c and c["text"]:
                            arg = c["text"]       # raw string
                # Accept {"patch": "..."} or any JSON with a single string field; else raw string.
                # This mirrors gpt-oss behavior and tolerates minor schema drift while keeping
                # the contract simple for the model.
                if isinstance(arg, str):
                    if arg.lstrip().startswith("{"):
                        try:
                            obj = json.loads(arg)
                            patch_text = obj.get("patch") or next((v for v in obj.values() if isinstance(v, str)), "")
                        except Exception:
                            patch_text = arg
                    else:
                        patch_text = arg
                elif isinstance(arg, dict):
                    patch_text = str(arg.get("patch", ""))
                break  # first tool call wins

        # Classify stop reason: assistant_action if tool found; else max_tokens if we hit cap.
        hit_limit = bool(self.max_tokens) and (token_count >= int(self.max_tokens)) and not bool(patch_text)

        # Return only the patch payload; empty string signals orchestrator to reprompt.
        return SamplerResponse(
            response_text=(patch_text or ""),   # empty -> orchestrator will reprompt
            actual_queried_message_list=message_list,
            response_metadata={
                # Expose minimal breadcrumbs for logging/metrics; no free-form thinking here.
                "tool": ("functions.apply_patch" if patch_text else ""),
                "stop_reason": ("produced_patch" if patch_text else ("max_tokens" if hit_limit else "no_patch")),
            },
        )




####### llm api backends #######



# 1) open-ai

@dataclass
class PendingTool:
    """Holds a single pending tool call that must be finalized on the next turn.

    - response_id: the OpenAI Responses turn id (id of an the whole response) where the function_call was emitted
        pass it as previous_response_id on the next call so the server reuses that turn’s hidden state/reasoning.
    - call_id:     the specific function_call.id (inside that response) to acknowledge
        must reference it once to submit the tool’s output and finalize the pending function call.
    - output:      the tool's actual output to return to the model
    """
    response_id: str | None = None
    call_id: str | None = None
    output: str | None = None


# comment:
# basically, with previous_response_id, the prior response must be finalized. If that prior turn ended with a function_call,
# the API requires to provide the tool’s output to close the loop. Otherwise the server refuses to chain and raises see "No tool output found".
# So finalize by including a {"type":"function_call_output"} item in the next request. 

@dataclass
class ChainState:
    """Encapsulates chaining state for OpenAI Responses API.

    prev_response_id stores only finalized response ids.
    pending stores a function call awaiting output; its output is supplied inline
    on the next create() call, in the same request as the new user message.
    """
    prev_response_id: str | None = None
    pending: PendingTool | None = None

    def build_input(self, user_text: str) -> tuple[object, dict, bool]:
        """Construct input payload and prev kw for responses.create.

        Returns (input_payload, prev_kw, used_inline_ack).
        If we have a pending tool with output, we prepend function_call_output and
        set previous_response_id to that pending turn id. Otherwise we pass through
        a typed user message and chain using the last finalized prev_response_id.
        """

        # If model ended the turn with a function_call on the previous iteration. Server marks
        # the turn as "requires tool output." If pass previous_response_id to this pending turn, the
        # server blocks with "No tool output found" bc it’s still waiting for the tool’s result.
        # Thus the logic for adding function_call_output below

        # If have a pending tool output from the previous turn, finalize it inline
        # before sending the new user input. This both closes the previous tool call
        # and starts the next turn in a single request.

        # When pending exists, it always has response_id and call_id, so
        # only require that output is present to finalize inline.
        if self.pending and (self.pending.output is not None):
            input_payload = [
                {
                    # Finalize the pending tool call with a result; this closes the turn
                    # so the next chained call can see hidden reasoning and tool outputs.
                    "type": "function_call_output",
                    "call_id": self.pending.call_id,
                    "output": self.pending.output,
                },
                {
                    "type": "message",
                    "role": "user",
                    # input_text item is only valid inside a message item. So using a message
                    # wrapper for the user turn. Otherwise BadRequestError: Error code: 400
                    "content": [{"type": "input_text", "text": user_text}],
                },
            ]
            prev_kw = {"previous_response_id": self.pending.response_id}
            return input_payload, prev_kw, True
        # No pending ack: regular chaining to the last finalized turn (if any).
        # For schema consistency, always send typed items (wrap user text in a message).
        prev_kw = {"previous_response_id": self.prev_response_id} if self.prev_response_id else {}
        input_payload = [{
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }]
        return input_payload, prev_kw, False

    def clear_if_used(self, used_inline_ack: bool) -> None:
        """Clear pending ack only after a successful API call that used it."""
        if used_inline_ack:
            self.pending = None

    def on_response(self, response_id: str, call_id: str | None) -> None:
        """Update chain state based on model output.

        - If a new function_call was emitted, record it as pending.
        - Otherwise, the turn is finalized; record as prev_response_id for next call.
        """
        if call_id:
            self.pending = PendingTool(response_id=response_id, call_id=call_id, output=None)
        else:
            self.pending = None
            self.prev_response_id = response_id

    def stash_output(self, text: str) -> None:
        """Stash tool output to be inlined on the next request, if pending exists."""
        if self.pending:
            self.pending.output = text
