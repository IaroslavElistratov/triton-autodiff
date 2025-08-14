"""
- `gpt_oss/tools/backward_naive/backward_tool.py`: Triton backward tool that
    - parses input, executes code,
    - imports `generate_naive_backward` from `.core`,
    - and returns the generated backward code (TTIR string)
- `gpt_oss/tools/kernel_loop.py`:
    - (Example CLI) Shows how to register and route to `TritonBackwardTool`
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
from triton.runtime.jit import JITFunction


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

Send Python code that defines and binds a Triton JITFunction (e.g., `@triton.jit`),
as a JSON object with fields:
  - code: required, Python module text defining and binding a JITFunction variable
    NOTE: any for-loops MUST have static bounds
  - setup: required, Python snippet executed after code (e.g., to create inputs)
  - warmup_call: required, a single Python snippet to run the kernel once

The tool executes code, then runs setup and warmup to JIT-compile the kernel, obtains the compiled kernel,
and returns the generated backward TTIR as a string.

Example JSON request:
```
{
  "code": """
import triton
import triton.language as tl

@triton.jit
def mm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # for-loops must have static bounds
    for _ in range(0, 2):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

def stub(a, b, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)
    _compiled_kernel = mm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c, _compiled_kernel
""",
  "setup": """
import torch
M = N = K = 32
a = torch.randn((M, K), device='cuda', dtype=torch.float16)
b = torch.randn((K, N), device='cuda', dtype=torch.float16)
""",
  "warmup_call": """
stub(a, b)
"""
}
```

What to provide:
- Expose exactly one top-level python function decorated with `@triton.jit` (the tool will pick it).
- A stub that chooses an appropriate grid, launches the kernel, and returns the compiled kernel (return either `_compiled_kernel` or `(output, _compiled_kernel)`).
- Concrete pytorch tensor inputs (device=cuda) and a warmup call to the stub to compile the kernel.
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

    def _extract_request(self, text: str) -> tuple[str, None | str, list[str], None | str]:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "code" in obj:
                code = str(obj["code"])  # type: ignore
                function_name = None
                if "setup" not in obj or not obj["setup"]:
                    raise ValueError("Missing required field: setup")
                setup = str(obj["setup"])  # type: ignore
                if "warmup_call" not in obj or not obj["warmup_call"]:
                    raise ValueError("Missing required field: warmup_call")
                warmups: list[str] = [str(obj["warmup_call"])]
                return code, function_name, warmups, setup
        except Exception as e:
            raise ValueError(
                "Invalid request. Expected JSON with required fields: code, setup, warmup_call.\n"
                "Tip: See the Example JSON request in the tool description."
            ) from e

    def _load_function_from_code(self, code: str, function_name: str | None) -> tuple[dict[str, Any], Any]:
        local_ns: dict[str, Any] = {}
        try:
            exec(code, local_ns, local_ns)
        except Exception as e:
            raise RuntimeError(
                "Failed to execute `code`. Ensure it is valid Python and defines a Triton kernel decorated with @triton.jit.\n"
                "Tip: Import triton and triton.language as tl, and bind the kernel to a top-level name.\n"
                f"Exec error: {e}"
            ) from e

        candidate: Any | None = None
        # Require that the code expose a JITFunction instance
        for name, value in local_ns.items():
            if isinstance(value, JITFunction):
                candidate = value
        if candidate is None:
            raise RuntimeError(
                "No Triton JITFunction found.\n"
                "Expected your `code` to define a top-level function decorated with @triton.jit, e.g.:\n"
                "@triton.jit\n"
                "def my_kernel(...): ...\n"
                "Tip: Expose exactly one top-level JITFunction variable so the tool can pick it."
            )
        return local_ns, candidate

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        text = message.content[0].text if message.content else ""
        try:
            code, fn_name, warmups, setup = self._extract_request(text)
            local_ns, fn_obj = self._load_function_from_code(code, fn_name)

            # Required setup snippet (e.g., create inputs)
            try:
                exec(setup, local_ns, local_ns)
            except Exception as e:
                raise RuntimeError(
                    "Failed to execute `setup`.\n"
                    "Ensure you import torch and create CUDA tensors with correct shapes/dtypes.\n"
                    f"Setup error: {e}"
                ) from e

            # Required warmup to trigger Triton JIT compilation (single call);
            # capture the compiled kernel if the stub returns it
            try:
                result = eval(warmups[0], local_ns, local_ns)
            except Exception as e:
                raise RuntimeError(
                    "Failed to execute `warmup_call`. It must run the kernel once and return the compiled kernel or (output, compiled_kernel).\n"
                    "Example: out, _compiled_kernel = stub(...); return out, _compiled_kernel  OR  return _compiled_kernel\n"
                    f"Warmup error: {e}"
                ) from e
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

            # Expect the warmup result to contain the compiled kernel; accept either the kernel
            # itself or a tuple (output, compiled_kernel)
            if hasattr(result, 'asm') and isinstance(getattr(result, 'asm'), dict) and 'ttir' in result.asm:
                compiled_kernel = result
            elif isinstance(result, tuple) and len(result) == 2 and hasattr(result[1], 'asm') and isinstance(getattr(result[1], 'asm'), dict) and 'ttir' in result[1].asm:
                compiled_kernel = result[1]
            else:
                raise RuntimeError(
                    "Warmup did not yield a compiled kernel.\n"
                    f"Got type: {type(result).__name__}. Expected a compiled kernel or a tuple (output, compiled_kernel).\n"
                    "Fix: Modify your stub to return the compiled kernel object. Example pattern:\n"
                    "_compiled_kernel = my_kernel[grid](...); return out, _compiled_kernel"
                )

            backward_code: str = generate_naive_backward(compiled_kernel)  # type: ignore
            yield self.make_response(TextContent(text=backward_code), channel=channel)
        except Exception as e:
            err = f"Error generating backward pass: {e}"
            yield self.make_response(TextContent(text=err), channel=channel)


