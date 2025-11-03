# todo: need this ?
# Extract names returned by stub function

from __future__ import annotations

import inspect
import os
import textwrap
from dataclasses import dataclass
from typing import Callable, Optional

from .utils import redact_torch_fn


def _render_stub_override_dck(stub_name: str) -> str:
    return textwrap.dedent(
        f"""
        import torch

        class StubOverrideDCK(torch.autograd.Function):
            \"\"\"Connects forward/backward stubs for automatic differentiation.\"\"\"

            @staticmethod
            def forward(ctx, *all_stub_inputs):
                result = {stub_name}(*all_stub_inputs)

                ten = [x for x in all_stub_inputs if isinstance(x, torch.Tensor)]
                ctx.save_for_backward(*ten)

                ctx.non_ten = [x for x in all_stub_inputs if not isinstance(x, torch.Tensor)]
                ctx.is_ten = [isinstance(x, torch.Tensor) for x in all_stub_inputs]

                return result

            @staticmethod
            def backward(ctx, *upstreams):
                # Reconstruct all forward inputs (mix tensors + non-tensors)
                it_t = iter(ctx.saved_tensors)
                it_n = iter(ctx.non_ten)
                all_inps = [next(it_t) if t else next(it_n) for t in ctx.is_ten]

                # keep full dict comprehension literal so generated file stays readable
                kw_up = {{f"upstream_{{i}}": g for i, g in enumerate(upstreams)}}
                grads_for_tensors = backward_{stub_name}(*all_inps, **kw_up)

                if len(grads_for_tensors) != sum(ctx.is_ten):
                    raise RuntimeError(
                        "Backward stub must return one grad per tensor input (in stub order)."
                    )

                # Align gradients to forward inputs (Tensor -> grad, non-Tensor -> None)
                it_g = iter(grads_for_tensors)
                per_input = [next(it_g) if t else None for t in ctx.is_ten]
                return tuple(per_input)
        """
    ).strip()


# autograd.Function[s] doesn't support kwargs, but user might be
# using their stub with kwargs this helper adds the kwarg support
def wrap_with_kwargs(stub_fn: Callable[..., object], override_cls: type) -> Callable[..., object]:
    """Preserve the stub's calling convention while routing through override_cls."""
    sig = inspect.signature(stub_fn)
    names = [param.name for param in sig.parameters.values()]

    def wrapped(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        ordered = [bound.arguments[name] for name in names]
        return override_cls.apply(*ordered)

    wrapped.__name__ = stub_fn.__name__
    wrapped.__qualname__ = getattr(stub_fn, "__qualname__", stub_fn.__name__)
    wrapped.__doc__ = stub_fn.__doc__
    wrapped.__signature__ = sig  # type: ignore[attr-defined]
    wrapped.__wrapped__ = stub_fn
    return wrapped


@dataclass
class StubInfo:
    stub_name: str
    positional_params: list[str]
    kwonly_params: list[str]
    upstream_count: int
    grad_param_names: list[str]



def discover_stub_info(forward_path: str) -> StubInfo:
    """Import the user module once to read the decorator metadata off the stub.

    The @kernel_agent.autodiff decorator is the source
    of truth for stub shape: importing the module executes
    user top-level code, creates the real stub object, and tags it with:
      * __is_autodiff_stub__      – marker so workers can locate it later
      * __autodiff_idxs__         – positional indices that require gradients

    Grab the stub, inspect its signature, and build a StubInfo (argument order,
    kw-only args, upstream kwargs, tensor names). The scaffold writer then uses
    StubInfo to emit the signature contract, backward placeholder, and
    StubOverrideDCK wrapper.

    IOW: let the decorator tell which stub arguments need gradients, then bake
    that guidance straight into raised.py so the LLM sees the exact call signature
    and the corresponding “grad_x = torch.zeros_like(x)” hints for every tensor input.
    """


    # 1) try to find stub, by whatever defines __is_autodiff_stub__
    # (which is what my @autodiff decorator writes)

    import importlib.util

    spec = importlib.util.spec_from_file_location("__ka_forward__", forward_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load forward module: {forward_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates = [
        (name, value)
        for name, value in vars(module).items()
        if callable(value) and getattr(value, "__is_autodiff_stub__", False)
    ]
    if not candidates:
        raise RuntimeError("No autodiff stub found in forward module")
    stub_name, stub_fn = candidates[0]

    # 2) inspect its signature

    signature = inspect.signature(stub_fn)
    # Preserve the stub's positional parameters in order so the generated scaffold matches exactly
    positional = [p.name for p in signature.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    # Keep any keyword-only parameters defined on the stub (besides pre-existing upstream_* kwargs)
    kwonly = [p.name for p in signature.parameters.values() if p.kind == p.KEYWORD_ONLY and not p.name.startswith("upstream_")]
    # Some user stubs already expose upstream_* kwargs; reuse them instead of fabricating new names
    upstream_names = [p.name for p in signature.parameters.values() if p.name.startswith("upstream_")]

    idxs = getattr(stub_fn, "__autodiff_idxs__", ())
    grad_names = [positional[idx] for idx in idxs if 0 <= idx < len(positional)]
    return StubInfo(
        stub_name=stub_name,
        positional_params=positional,
        kwonly_params=kwonly,
        # TODO-NOW: stop fabricating upstream_0 when the stub signature has zero upstream kwargs
        upstream_count=len(upstream_names) or 1,
        grad_param_names=grad_names,
    )


def write_initial_backward(
    *,
    forward_path: str,
    dst_path: str,
    retrieved_backward: Optional[str] = None,
) -> StubInfo:
    """Render raised.py using StubInfo: forward copy, signature comment, stub scaffold, DCK.

    Called once per run before any workers spawn. The goal is to hand the LLM a
    complete template that already spells out “gradcheck will call this exactly as …”
    and “here are the gradient buffers you need to fill”, using the real stub
    signature and decorator-provided tensor indices.
    """
    info = discover_stub_info(forward_path)
    forward_src = redact_torch_fn(forward_path, None)
    if not forward_src:
        raise RuntimeError(f"Failed to read forward source: {forward_path}")
    forward_src = _prepare_forward_source(forward_src, info.stub_name)

    header_forward = textwrap.dedent(
        """\
        # ============================================================
        # Forward kernel and stub (copied from user file)
        # Backward can call these to recompute intermediates
        # ============================================================"""
    ).strip()

    header_backward = textwrap.dedent(
        """\
        # ============================================================
        # Backward kernel and stub
        # ============================================================"""
    ).strip()

    # Signature comment tells the LLM exactly how gradcheck will invoke backward_* and which grads to return.
    signature_comment = _make_signature_comment(info).strip()
    # Placeholder backward stub mirrors the user signature and raises until filled in with real logic.
    backward_stub = _make_backward_stub(info)
    dck = _render_stub_override_dck(info.stub_name)

    # spell sections out explicitly to keep zero-indent formatting in the generated file
    sections = [
        header_forward,
        forward_src.strip(),
        header_backward,
        signature_comment,
        backward_stub.strip(),
        dck,
    ]

    content = "\n\n".join(sections) + "\n"

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    return info


# comment:
# essentially: inspect the stub’s signature, read user annotations for which args require grads, then auto-generate a docstring that tells the LLM to create gradients w.r.t. those input indices (and include examples that initialize zero-grads with the right shapes)
def _make_signature_comment(info: StubInfo) -> str:
    pieces: list[str] = []
    if info.positional_params:
        pieces.append(", ".join(info.positional_params))

    kw_section: list[str] = []
    if info.kwonly_params:
        kw_section.extend(info.kwonly_params)

    upstream_names = [f"upstream_{i}" for i in range(info.upstream_count)]
    if upstream_names:
        kw_section.extend(upstream_names)

    if kw_section:
        pieces.append(", ".join(["*"] + kw_section))

    call_signature = ", ".join(pieces)

    if info.grad_param_names:
        grad_clause = "Must return tuple: (" + ", ".join(f"grad_{name}" for name in info.grad_param_names) + ")"
    else:
        grad_clause = "Must return gradients for every tensor input in original order."

    return textwrap.dedent(
        f"""
        # SIGNATURE CONTRACT (from forward stub):
        # Gradcheck will call this EXACTLY as:
        #   backward_{info.stub_name}({call_signature})
        #
        # {grad_clause}
        """
    )


def _make_backward_stub(info: StubInfo) -> str:
    params: list[str] = list(info.positional_params)

    if info.kwonly_params or info.upstream_count:
        params.append("*")
        params.extend(info.kwonly_params)
        params.extend(f"upstream_{i}" for i in range(info.upstream_count))

    param_str = ", ".join(params)
    grad_targets = info.grad_param_names or ["input_tensor"]

    lines = [
        f"def backward_{info.stub_name}({param_str}):",
        '    """Generated placeholder for backward pass."""',
        "    # 1. Allocate gradient buffers for each tensor input",
    ]
    # Show explicit grad_foo examples so the LLM knows exactly which tensors require gradients.
    lines.extend(f"    #    grad_{name} = torch.zeros_like({name})" for name in grad_targets)
    lines.extend([
        "    # 2. Launch backward kernels to populate gradients",
        "    #    backward_kernel[grid](...)",
        "    # 3. Return gradients matching the tensor input order",
        "    raise ValueError(",
        f'        "backward_{info.stub_name} not implemented!\\n"',
        '        "Must allocate gradient buffers for each tensor input, launch backward kernels, and return gradients in input order."',
        "    )",
    ])
    return "\n".join(lines)


def _prepare_forward_source(forward_src: str, stub_name: str) -> str:
    """Clean up forward source before embedding into raised.py."""
    cleaned = _strip_autodiff_decorator(forward_src, stub_name)
    cleaned = _remove_autodiff_import(cleaned)
    cleaned = _strip_main_guard(cleaned)
    return cleaned


def _strip_autodiff_decorator(forward_src: str, stub_name: str) -> str:
    """Remove the @autodiff decorator line that guards the stub definition."""
    lines = forward_src.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("@autodiff"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            next_line = lines[j].strip() if j < len(lines) else ""
            if next_line.startswith(f"def {stub_name}"):
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _remove_autodiff_import(src: str) -> str:
    """Drop autodiff helper imports that become unused after decorator removal."""
    out: list[str] = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("from kernel_agent.autodiff import autodiff"):
            continue
        if stripped == "import kernel_agent.autodiff":
            continue
        out.append(line)
    return "\n".join(out)


def _strip_main_guard(src: str) -> str:
    """Remove `if __name__ == \"__main__\":` blocks so generated file stays inert."""
    lines = src.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith('if __name__ == "__main__"') or stripped.startswith("if __name__ == '__main__'"):
            guard_indent = len(line) - len(line.lstrip())
            i += 1
            while i < len(lines):
                nxt = lines[i]
                nxt_stripped = nxt.strip()
                nxt_indent = len(nxt) - len(nxt.lstrip())
                if nxt_stripped and nxt_indent <= guard_indent:
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)
