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


from typing import Any, AsyncIterator, Callable, Optional

import json
import hashlib
import os
import torch
import queue
import threading

from openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
    ToolNamespaceConfig,
)

from ..tool import Tool
from .utils import extract_request, run_with_timeout, pick_compiled_kernel, try_make_slice_payload
from triton.runtime.jit import JITFunction


class TritonBackwardTool(Tool):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_tool_name(cls) -> str:
        return "triton_backward"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return (
            '''

Send Python code that defines exactly one Triton JITFunction (python function decorated with @triton.jit),
as a JSON object with fields:
  - code  (required): Python module text that defines ONE top-level @triton.jit kernel
                        and any helper stubs it needs.
                        NOTE: any for-loops inside the kernel MUST have static bounds
  - setup (required): Python snippet executed after `code`. It must:
                        - import torch & build CUDA tensors
                        - launch the kernel ONCE to force JIT compilation
                        - assign the compiled kernel object to `_compiled_kernel`
  - json  (optional): if true, return a small JSON pointer {digest, path} instead of the full TTIR string.

The tool executes `code`, then runs `setup` to JIT-compile the kernel, obtains the compiled kernel,
and returns the generated backward TTIR as a string (or a tiny pointer when `json: true`).

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
c, _compiled_kernel = stub(a, b)
""",
  "json": true
}

Contract:
- The tool will execute `code`, then `setup`.
- It discovers `_compiled_kernel` (or the only compiled kernel object in scope),
  runs the autodiff pass, and returns the backward TTIR (or a JSON pointer when `json: true`).

Fetching large TTIR incrementally:
- After receiving `{ "digest": "<digest10>", "path": "generated/<digest10>/out.ttir" }`,
  call the tool later with:
  { "slice": { "digest": "<digest10>", "offset": 0, "limit": 65536 } }
```

What to provide:
- Expose exactly one top-level python function decorated with `@triton.jit` (the tool will pick it).
- A stub that chooses an appropriate grid, launches the kernel, and returns the compiled kernel.

'''
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

    def _extract_request(self, text: str) -> tuple[str, str, bool]:
        # Delegate to shared utils for validation and extraction
        return extract_request(text)

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

    # Tunable limits (seconds)
    CODE_EXEC_TIMEOUT_S = 15.0     # for `exec(code)` and `exec(setup)`

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        channel = message.channel
        text = message.content[0].text if message.content else ""
        try:
            # Optional fast path: slice read without recompilation
            payload = try_make_slice_payload(text)
            if payload is not None:
                yield self.make_response(TextContent(text=payload), channel=channel)
                return

            code, setup, as_json = self._extract_request(text)
            local_ns, _ = self._load_function_from_code(code, None)

            # Required setup snippet (prepare tensors AND launch once)
            try:
                run_with_timeout(lambda: exec(setup, local_ns, local_ns), self.CODE_EXEC_TIMEOUT_S)
            except Exception as e:
                raise RuntimeError(
                    "Failed to execute `setup`. Ensure it creates CUDA tensors and launches the kernel once.\n"
                    f"Setup error: {e}"
                ) from e

            # Lazy import to avoid hard dependency unless the tool is used
            from .core import generate_naive_backward  # type: ignore

            # Find the compiled kernel in the namespace
            compiled_kernel = pick_compiled_kernel(local_ns)

            backward_code: str = generate_naive_backward(compiled_kernel)  # type: ignore

            if as_json:
                fwd_ttir = compiled_kernel.asm["ttir"]
                digest10 = hashlib.sha256(fwd_ttir.encode()).hexdigest()[:10]
                path = f"generated/{digest10}/out.ttir"
                payload = json.dumps({"digest": digest10, "path": path})
                yield self.make_response(TextContent(text=payload), channel=channel)
            else:
                yield self.make_response(TextContent(text=backward_code), channel=channel)
        except Exception as e:
            err = f"Error generating backward pass: {e}"
            yield self.make_response(TextContent(text=err), channel=channel)


