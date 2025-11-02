from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import torch

try:  # PyTorch 2.3+ exposes default_partition via functorch.compile
    from functorch.compile import default_partition  # type: ignore
except ImportError:  # Fallback for older versions
    from torch._functorch.partitioners import default_partition  # type: ignore

Tensor = torch.Tensor


# avoid: default min cut partitioner (min_cut_rematerialization_partition) -- https://github.com/pytorch/pytorch/blob/main/torch/_functorch/partitioners.py#L993
#
# use default_partition instead of the higher-level torch.compile partitioner
# so the joint graph stays partitioned, but without recomputing fwd nodes in bwd; matches the
# manual contract where I only want a straight cut between forward and backward
# without torch deciding what to recompute in backward (in which case the backward
# graph will have additional nodes -- from forward -- so the bwd graph will no longer
# be 1:1 clean differentiated fwd, which will likely confuse the llm);
# the backward FX graph remains a straight differentiated version of the forward without min-cut rematerialization
def _stash_only_partition(
    joint_module: torch.fx.GraphModule,
    joint_inputs,
    *,
    num_fwd_outputs: int,
    **kw,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    """Default partition with recompute tags stripped."""
    for node in joint_module.graph.nodes:
        node.meta.pop("recompute", None)
    return default_partition(
        joint_module,
        joint_inputs,
        num_fwd_outputs=num_fwd_outputs,
        **kw,
    )


def capture_reference_backward(
    pytorch_ref: Callable,
    test_inputs: Sequence[Tensor],
    test_kwargs: Dict[str, Any],
    *,
    stats: Dict[str, Any],
    verbose: bool = False,
) -> Tuple[Callable, Dict[str, Any]]:
    """Wrap reference with AOTAutograd, run probe fwd/bwd, and return capture dict."""
    capture: Dict[str, Any] = {}
    ref_callable = pytorch_ref
    try:
        from torch._functorch.aot_autograd import aot_function

        def _fw_compiler(gm, _):
            capture["forward_graph"] = str(gm.graph)
            return gm

        def _bw_compiler(gm, _):
            # print generated python code instead of fx graph
            # https://github.com/pytorch/pytorch/blob/0674e0a0f14775f920296e9dfb8b61e4960bf99d/torch/fx/graph.py#L1745
            # https://github.com/pytorch/pytorch/blob/0674e0a0f14775f920296e9dfb8b61e4960bf99d/torch/fx/graph.py#L432-L858
            capture["backward_graph"] = gm.code
            if verbose:
                print("[AOT Capture] backward graph FX:")
                print(gm.code)
            return gm

        def _partition(joint_module, flat_inputs, *, num_fwd_outputs, **kw):
            capture["joint_graph"] = str(joint_module.graph)
            return _stash_only_partition(
                joint_module,
                flat_inputs,
                num_fwd_outputs=num_fwd_outputs,
                **kw,
            )

        ref_callable = aot_function(
            pytorch_ref,
            fw_compiler=_fw_compiler,
            bw_compiler=_bw_compiler,
            partition_fn=_partition,
        )
    except Exception as err:
        capture["error"] = f"{type(err).__name__}: {err}"
        if verbose:
            print(f"[Reference Check] AOTAutograd capture disabled: {capture['error']}")
        return ref_callable, capture

    # Run probe forward/backward once to populate backward graph if needed
    inputs = [_clone_for_capture(x, require_grad=True) for x in test_inputs]
    kwargs = _clone_for_capture(test_kwargs, require_grad=True)
    try:
        out = ref_callable(*inputs, **kwargs)
        if not isinstance(out, (list, tuple)):
            out = (out,)
        terms = [o for o in out if isinstance(o, torch.Tensor) and o.is_floating_point()]
        if terms:
            (sum(t.sum() for t in terms)).backward()
    except Exception as err:
        capture.setdefault("backward_error", f"{type(err).__name__}: {err}")

    stats["aot_reference"] = {
        key: _truncate_graph_dump(val) if isinstance(val, str) else val
        for key, val in capture.items()
    }
    return ref_callable, capture


def _clone_for_capture(obj: Any, *, require_grad: bool) -> Any:
    if isinstance(obj, torch.Tensor):
        out = obj.detach().clone()
        if require_grad and out.is_floating_point():
            out.requires_grad_(True)
        return out
    if isinstance(obj, dict):
        return {k: _clone_for_capture(v, require_grad=require_grad) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_for_capture(v, require_grad=require_grad) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_for_capture(v, require_grad=require_grad) for v in obj)
    if isinstance(obj, set):
        return {_clone_for_capture(v, require_grad=require_grad) for v in obj}
    return obj


def _truncate_graph_dump(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated, {len(text)} chars] ..."
