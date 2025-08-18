"""
- `gpt_oss/tools/backward_naive/backward_tool.py`: new tool that
    - parses input, executes code,
    - imports `generate_naive_backward` from `.core`
    - and returns the backward code
- `gpt_oss/grad.py`:
    - Imports `TritonBackwardTool`
    - Adds `--triton-backward` flag
    - Registers tool with Harmony when enabled
    - Adds processing branch for `triton_backward.*` messages and prints tool status
"""


from typing import Any, AsyncIterator, Optional

import json
import hashlib
import os
import re

from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
    ToolDescription,
)

from ..tool import Tool
from .utils import run_with_timeout, pick_compiled_kernel, extract_request, try_make_slice_payload
from triton.runtime.jit import JITFunction



def _is_compiled_kernel(candidate: Any) -> bool:
    return hasattr(candidate, "asm") and isinstance(getattr(candidate, "asm"), dict) and "ttir" in candidate.asm


class TritonBackwardTool(Tool):
    def __init__(self, name: str | None = None) -> None:
        super().__init__()
        if name is not None and name != "triton_backward":
            raise ValueError("Tool name is fixed to 'triton_backward'")
        
        # Soft cap for raw text payloads to protect LLM context window
        self.MAX_RAW_BYTES = int(os.environ.get("TB_MAX_RAW_BYTES", "65536"))

    @classmethod
    def get_tool_name(cls) -> str:
        return "triton_backward"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return (
            "Call `triton_backward.run` with JSON: "
            '{"code": "<python module>", "setup": "<python snippet>", '
            '"format": "json|raw"}.\n'
            "Rules:\n"
            "- Define exactly one @triton.jit kernel. In `setup`, run your stub and assign the **LAUNCH** object (has `.asm['ttir']`) to COMPILED_KERNEL (or _compiled_kernel).\n"
            "- Default output is JSON {digest, path}. Use `slice` to page TTIR. Use `format:'raw'` only for tiny graphs.\n"
            "Any for-loops inside the kernel MUST have static bounds."
            "Minimal valid example (\\n escaped):\n"
            '{\n'
            '  "code": "import triton\\nimport triton.language as tl\\nimport torch\\n@triton.jit\\n'
            'def k(a_ptr,b_ptr,o_ptr):\\n  off=tl.arange(0,4)\\n  tl.store(o_ptr+off, tl.load(a_ptr+off)*tl.load(b_ptr+off))\\n\\n'
            'def stub(a,b):\\n  o=torch.empty_like(a)\\n  launch = k[(1,1,1)](a,b,o)\\n  return o, launch\\n",\n'
            '  "setup": "import torch\\na=torch.rand(4, device=\\"cuda\\"); b=torch.rand(4, device=\\"cuda\\")\\n_, COMPILED_KERNEL = stub(a,b)",\n'
            '  "format": "json"\n'
            '}\n'
        )

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description="Generate a naive backward TTIR from a compiled Triton kernel.",
            tools=[
                ToolDescription.new(
                    name="run",
                    description=(
                        "Compile the provided Triton kernel and return its backward TTIR. "
                        "Provide Python source for `code` and `setup`. "
                        "In `setup`, bind COMPILED_KERNEL (or _compiled_kernel) to the **LAUNCH** object."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python module defining exactly one @triton.jit kernel.",
                            },
                            "setup": {
                                "type": "string",
                                "description": "Python snippet that prepares inputs and assigns COMPILED_KERNEL (or _compiled_kernel) to the LAUNCH object.",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["raw", "json"],
                                "default": "json",
                                "description": "Return compact JSON {digest, path} (default) or the raw TTIR text.",
                            },
                        },
                        "required": ["code", "setup"],
                    },
                )
                ,
                ToolDescription.new(
                    name="slice",
                    description="Return a byte-range from the generated TTIR identified by digest.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "digest": {"type": "string", "description": "Digest (10+ hex chars) returned by run(format=json)."},
                            "offset": {"type": "integer", "default": 0, "minimum": 0},
                            "limit":  {"type": "integer", "default": 65536, "minimum": 1},
                        },
                        "required": ["digest"],
                    },
                )
            ],
        )

    def make_response(
        self,
        content: Content,
        *,
        author: Author | None = None,
        channel: str | None = None,
    ) -> Message:
        tool_name = self.get_tool_name()
        author = Author(role=Role.TOOL, name=f"{tool_name}")

        message = Message(author=author, content=[content]).with_recipient("assistant")
        if channel:
            message = message.with_channel(channel)
        return message

    def _extract_request(self, text: str) -> tuple[str, str, Optional[str]]:
        code, setup, fmt = extract_request(text)
        return code, setup, fmt

    def _load_function_from_code(self, code: str, function_name: str | None) -> tuple[dict[str, Any], Any]:
        """
        Execute the provided Python `code` and extract exactly one Triton JITFunction.

        * executes code in an isolated namespace,
        * enumerates ALL JITFunction instances,
        * requires exactly one (explicit, deterministic),
        * raises with a clear error if there are 0 or >1 kernels.
        """
        del function_name  # single-kernel enforcement; parameter is ignored

        local_ns: dict[str, Any] = {}
        try:
            # Directly execute user-provided code in an isolated namespace.
            run_with_timeout(lambda: exec(code, local_ns, local_ns), self.CODE_EXEC_TIMEOUT_S)
        except Exception as e:
            raise RuntimeError(
                "Failed to execute `code`. Ensure it is valid Python and defines a Triton kernel decorated with @triton.jit.\n"
                "Tip: Import triton and triton.language as tl, and bind the kernel to a top-level name.\n"
                f"Exec error: {e}"
            ) from e

        kernels: dict[str, Any] = {name: value for name, value in local_ns.items() if isinstance(value, JITFunction)}

        if not kernels:
            raise RuntimeError(
                "No Triton JITFunction found.\n"
                "Expected your `code` to define exactly one top-level function decorated with @triton.jit, e.g.:\n"
                "@triton.jit\n"
                "def my_kernel(...): ...\n"
            )

        if len(kernels) > 1:
            available = ", ".join(sorted(kernels.keys()))
            raise RuntimeError(
                f"Multiple Triton kernels found: [{available}]. Expose exactly one top-level @triton.jit kernel in `code`."
            )

        selected_kernel = next(iter(kernels.values()))
        return local_ns, selected_kernel

    # Tunable limits (seconds) — configurable via env
    CODE_EXEC_TIMEOUT_S = float(os.environ.get("TB_CODE_TIMEOUT_S", "15"))

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        text = message.content[0].text if message.content else ""
        try:
            # Optional fast-path: slice read without recompilation
            payload = try_make_slice_payload(text)
            if payload is not None:
                yield self.make_response(TextContent(text=payload), channel=channel)
                return

            code, setup, fmt = extract_request(text)

            # Execute code and require exactly one @triton.jit kernel to be present
            local_ns, _ = self._load_function_from_code(code, None)

            # Execute setup (may set COMPILED_KERNEL or _compiled_kernel)
            try:
                run_with_timeout(lambda: exec(setup, local_ns, local_ns), self.CODE_EXEC_TIMEOUT_S)
            except Exception as e:
                raise RuntimeError(
                    "Failed to execute `setup`. Ensure it creates CUDA tensors and launches the kernel once.\n"
                    f"Setup error: {e}"
                ) from e

            # Resolve compiled kernel
            compiled_kernel = pick_compiled_kernel(local_ns)

            # Lazy import to avoid hard dependency unless the tool is used
            from .core import generate_naive_backward  # type: ignore

            backward_code: str = generate_naive_backward(compiled_kernel)  # type: ignore

            if fmt == "json" or (fmt == "raw" and len(backward_code.encode("utf-8")) > self.MAX_RAW_BYTES):
                fwd_ttir = compiled_kernel.asm["ttir"]
                digest10 = hashlib.sha256(fwd_ttir.encode()).hexdigest()[:10]
                path = f"generated/{digest10}/out.ttir"
                payload_dict: dict[str, Any] = {"digest": digest10, "path": path}
                if fmt == "raw":
                    payload_dict["note"] = (
                        f"raw TTIR exceeded {self.MAX_RAW_BYTES} bytes; returning json pointer instead. "
                        "Use triton_backward.slice to read windows."
                    )
                payload = json.dumps(payload_dict)
                yield self.make_response(TextContent(text=payload), channel=channel)
            else:
                yield self.make_response(TextContent(text=backward_code), channel=channel)
        except Exception as e:
            err = f"Error generating backward pass: {e}"
            yield self.make_response(TextContent(text=err), channel=channel)


