"""
- `gpt_oss/tools/triton_backward/backward_tool.py`: new tool that
    - parses input, executes code,
    - imports `generate_naive_backward` from `.core`
    - and returns the backward code
- `gpt_oss/grad.py`:
    - Imports `TritonBackwardTool`
    - Adds `--triton-backward` flag
    - Registers tool with Harmony when enabled
    - Adds processing branch for `triton_backward.*` messages and prints tool status
"""


from typing import Any, AsyncIterator

import json

from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)

from ..tool import Tool


class TritonBackwardTool(Tool):
    def __init__(self, name: str = "triton_backward"):
        assert name == "triton_backward"

    @classmethod
    def get_tool_name(cls) -> str:
        return "triton_backward"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return (
            """
Send Python code that defines a Triton kernel decorated with @triton.jit (or JITFunction),
or send a JSON object of the form {"code": "...", "function_name": "..."}.
This tool will execute the code, locate the target function, call generate_naive_backward(fn),
and return the generated backward pass as a string.
"""
        ).strip()

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        return ToolNamespaceConfig(
            name=self.get_tool_name(),
            description=self.instruction,
            tools=[],
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

    def _extract_code_and_function_name(self, text: str) -> tuple[str, str | None]:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "code" in obj:
                code = str(obj["code"])  # type: ignore
                function_name = obj.get("function_name")  # type: ignore
                if function_name is not None:
                    function_name = str(function_name)
                return code, function_name
        except Exception:
            pass
        return text, None

    def _load_function_from_code(self, code: str, function_name: str | None) -> Any:
        local_ns: dict[str, Any] = {}
        try:
            exec(code, local_ns, local_ns)
        except Exception as e:
            raise RuntimeError(f"Failed to execute provided code: {e}") from e

        candidate: Any | None = None
        if function_name:
            found = local_ns.get(function_name)
            if not callable(found):
                raise RuntimeError(f"Function `{function_name}` not found after executing code")
            candidate = found
        else:
            # Pick the last defined callable as a reasonable default
            for name, value in local_ns.items():
                if callable(value):
                    candidate = value
            if candidate is None:
                raise RuntimeError("No callable function found in provided code")
        return candidate

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        text = message.content[0].text if message.content else ""
        try:
            code, fn_name = self._extract_code_and_function_name(text)
            fn_obj = self._load_function_from_code(code, fn_name)

            # Lazy import to avoid hard dependency unless the tool is used
            try:
                from .core import generate_naive_backward  # type: ignore
            except Exception as e1:
                try:
                    from gpt_oss.tools.triton_backward.core import generate_naive_backward  # type: ignore
                except Exception as e2:
                    raise RuntimeError(
                        "Could not import `generate_naive_backward`. Ensure the core implementation is available."
                    ) from e2

            backward_code: str = generate_naive_backward(fn_obj)  # type: ignore
            yield self.make_response(TextContent(text=backward_code), channel=channel)
        except Exception as e:
            err = f"Error generating backward pass: {e}"
            yield self.make_response(TextContent(text=err), channel=channel)


