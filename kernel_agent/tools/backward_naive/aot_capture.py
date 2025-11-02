from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import operator

import torch
from torch._functorch.partitioners import (
    _extract_fwd_bwd_modules,
    _extract_fwd_bwd_outputs,
    _extract_graph_with_inputs_outputs,
    _is_fwd_seed_offset,
    _is_primal,
    is_sym_node,
)
from torch.utils._ordered_set import OrderedSet

Tensor = torch.Tensor


# avoid: default min cut partitioner (min_cut_rematerialization_partition) -- https://github.com/pytorch/pytorch/blob/main/torch/_functorch/partitioners.py#L993
#
# use default_partition instead of the higher-level torch.compile partitioner
# so the joint graph stays NOT partitioned; matches the
# manual contract where I only want a straight cut between forward and backward
# without torch deciding what to recompute in backward (in which case the backward
# graph will have additional nodes -- from forward -- so the bwd graph will no longer
# be 1:1 clean differentiated fwd, which will likely confuse the llm)
def _no_recompute_partition(
    joint_module: torch.fx.GraphModule,
    joint_inputs,
    *,
    num_fwd_outputs,
    static_lifetime_input_indices: Sequence[int] | None = None,
    static_lifetime_input_nodes: OrderedSet | None = None,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    """Partition joint graph without min-cut rematerialization, keeping a straight forward/backward split."""
    # do this to mirror the manual contract from Phase 0: I want to show the LLM the raw differentiated math
    # without the min_cut_rematerialization_partition that default_partition triggers when recomputable ops appear
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, _ = _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs, fwd_outputs, "forward"
    )
    forward_node_names = OrderedSet(
        node.name for node in forward_only_graph.nodes if node.op != "output"
    )
    saved_values = []
    saved_sym_nodes = []

    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            continue
        if is_sym_node(node):
            saved_sym_nodes.append(node)
        elif "tensor_meta" not in node.meta and node.op == "call_function":
            users = node.users
            assert all(user.target == operator.getitem for user in users)
            saved_values.extend(users)
        else:
            backward_usages = [n for n in node.users if n.name not in forward_node_names]
            if "tensor_meta" in node.meta and all(is_sym_node(n) for n in backward_usages):
                saved_sym_nodes.extend(backward_usages)
            else:
                saved_values.append(node)

    saved_values = list(dict.fromkeys(saved_values).keys())
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())

    return _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
        static_lifetime_input_nodes=static_lifetime_input_nodes,
    )


def wrap_reference_with_aot(pytorch_ref: Callable, verbose: bool) -> Tuple[Callable, Dict[str, Any]]:
    """Wrap pytorch_ref with AOTAutograd and collect FX graphs when possible.

    Mirrors the manual contract: Phase 0 wraps the torch reference via aot_function
    with custom compilers and a partition function that keeps the forward/backward
    split without invoking rematerialization heuristics.
    """
    capture: Dict[str, Any] = {}
    ref_callable = pytorch_ref
    try:
        from torch._functorch.aot_autograd import aot_function

        def _fw_compiler(fx_module, _flat_inputs):
            capture["forward_graph"] = str(fx_module.graph)
            return fx_module

        def _bw_compiler(fx_module, _flat_inputs):
            capture["backward_graph"] = str(fx_module.graph)
            return fx_module

        def _partition(joint_module, flat_inputs, num_fwd_outputs):
            capture["joint_graph"] = str(joint_module.graph)
            return _no_recompute_partition(
                joint_module,
                flat_inputs,
                num_fwd_outputs=num_fwd_outputs,
                static_lifetime_input_indices=None,
                static_lifetime_input_nodes=None,
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


def ensure_backward_aot_capture(
    ref_callable: Callable,
    capture: Dict[str, Any],
    test_inputs: Sequence[Tensor],
    test_kwargs: Dict[str, Any],
    verbose: bool = False,
) -> None:
    """Run a lightweight backward pass to populate missing backward FX graphs."""
    if "forward_graph" not in capture or capture.get("backward_graph"):
        return
    capture_inputs = [_clone_for_capture(x, require_grad=True) for x in test_inputs]
    capture_kwargs = _clone_for_capture(test_kwargs, require_grad=True)
    try:
        capture_out = ref_callable(*capture_inputs, **capture_kwargs)
        if not isinstance(capture_out, (list, tuple)):
            capture_out = (capture_out,)
        capture_terms = [o for o in capture_out if isinstance(o, torch.Tensor) and o.is_floating_point()]
        if capture_terms:
            loss = sum(t.sum() for t in capture_terms)
            loss.backward()
    except Exception as err:
        capture.setdefault("backward_error", f"{type(err).__name__}: {err}")

    if verbose and capture.get("backward_graph"):
        print("[AOT Capture] backward graph FX:")
        print(capture["backward_graph"])


def attach_aot_capture(stats: Dict[str, Any], capture: Dict[str, Any]) -> None:
    """Attach captured FX graphs to gradcheck stats with safe truncation."""
    if not capture:
        return
    stats["aot_reference"] = {
        key: _truncate_graph_dump(val) if isinstance(val, str) else val
        for key, val in capture.items()
    }


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
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} chars] ..."
