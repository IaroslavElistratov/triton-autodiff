"""
Generate a backward stub (Python source) from a user's forward stub that launches
a Triton/CUDA-style kernel via:  kernel[grid](...)

Merged features:
- AST-only, no LibCST
- Preserves the user stub body but removes its `return`
- Precomputes grad args at codegen time (no runtime loops)
- Supports removing folded positional args before the backward launch
- Forwards original keyword args (including **kwargs) to the backward kernel
- Returns `None` for requested grads that cannot be mapped
- Avoids requiring `import torch` by using Tensor.new_zeros for zero-like grads
"""
import ast
import textwrap
from typing import List, Dict, Optional


def _src_of(node: ast.AST, src: str) -> str:
    seg = ast.get_source_segment(src, node)
    return seg if seg is not None else ast.unparse(node)


def _find_func(src: str, name: str) -> ast.FunctionDef:
    try:
        mod = ast.parse(src)
    except SyntaxError as e:
        raise ValueError(f"Failed to parse stub_src: {e}") from e
    for n in mod.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise ValueError(f"function {name!r} not found in provided source")


def _find_kernel_call(func_node: ast.FunctionDef, src: str, kernel_name: str):
    """
    Find the first call like:  <kernel_name>[...](...)
    Returns (call_node, grid_code:str, posargs_code:list[str], kwargs_code:list[str or '**expr']).
    """
    class Finder(ast.NodeVisitor):
        def __init__(self):
            self.hit: Optional[ast.Call] = None
        def visit_Call(self, node: ast.Call):
            if self.hit is not None:
                return
            func = node.func
            if isinstance(func, ast.Subscript):
                val = func.value
                if isinstance(val, ast.Name) and val.id == kernel_name:
                    self.hit = node
                    return
            self.generic_visit(node)

    f = Finder()
    f.visit(func_node)
    if f.hit is None:
        raise ValueError(f"{kernel_name}[...] call not found inside {func_node.name}")

    call = f.hit
    grid_code = _src_of(call.func.slice, src)
    posargs_code = [_src_of(a, src) for a in call.args]
    kwargs_code = [
        (f"{kw.arg}={_src_of(kw.value, src)}" if kw.arg is not None else f"**{_src_of(kw.value, src)}")
        for kw in call.keywords
    ]
    return call, grid_code, posargs_code, kwargs_code


def gen_bwd_stub(
    stub_src: str,
    stub_name: str,
    kernel_name: str,
    bwd_kernel_name: str,
    upstream_param: str,
    tensor_params: List[str],           # params in stub that require grads, in order
    tensor_arg_idxs: List[int],         # positions in *positional* kernel args that are tensors
    upstream_map: Dict[int, str],       # original arg_idx -> upstream tensor name in bwd stub
    folded_const_idxs: Optional[List[int]] = None,  # original positional arg indices to drop for bwd
    bwd_stub_name: Optional[str] = None,
) -> str:
    """
    Synthesize a backward stub function that:
      1) Replays the forward stub body (excluding its `return`) to rebuild needed tensors.
      2) Precomputes grad vars for *kept* positional args.
      3) Calls the backward kernel with: kept positional args + grad vars + original kwargs.
      4) Returns grads in the order of `tensor_params` (uses None when not found).

    Folding:
      - `folded_const_idxs` are indices into the *original positional* argument list of the
        forward kernel call. These are removed before invoking the backward kernel.
      - `tensor_arg_idxs` and `upstream_map` are specified in the original index space and
        are shifted automatically after folding.
    """
    fn = _find_func(stub_src, stub_name)
    call, grid, posargs, kwargs = _find_kernel_call(fn, stub_src, kernel_name)

    # Build new function signature
    param_names = [a.arg for a in fn.args.args]
    if upstream_param in param_names:
        raise ValueError(f"upstream_param {upstream_param!r} duplicates an existing parameter")
    bwd_params = param_names + [upstream_param]
    bwd_name = bwd_stub_name or f"{stub_name}_bwd"

    # Replay original body but drop any explicit returns
    body_lines = [textwrap.indent(_src_of(stmt, stub_src), "    ")
                  for stmt in fn.body if not isinstance(stmt, ast.Return)]

    # Validate indices
    n_pos = len(posargs)
    folded = sorted(set(folded_const_idxs or []))
    def _chk_space(idxs, label):
        bad = [i for i in idxs if not (0 <= i < n_pos)]
        if bad:
            raise ValueError(f"{label} contains invalid arg indices {bad}; kernel has {n_pos} positional args")
    _chk_space(upstream_map.keys(), "upstream_map")
    _chk_space(tensor_arg_idxs, "tensor_arg_idxs")
    _chk_space(folded, "folded_const_idxs")

    # Compute kept positional args and index shift
    keep_pos = [i for i in range(n_pos) if i not in folded]
    kept_posargs = [posargs[i] for i in keep_pos]
    def _shift(i): return i - sum(f < i for f in folded)

    shifted_upstream_map = { _shift(i): name for i, name in upstream_map.items() if i not in folded }
    shifted_tensor_idxs = [ _shift(i) for i in tensor_arg_idxs if i not in folded ]

    # Precompute grad vars for kept positional args
    grad_var_names: List[str] = []
    grad_lines: List[str] = []
    for j, arg_code in enumerate(kept_posargs):
        base = arg_code if arg_code.isidentifier() else f"arg{j}"
        gname = f"grad_{base}"
        if j in shifted_upstream_map:
            src_name = shifted_upstream_map[j]
            grad_lines.append(f"    {gname} = {src_name}.clone()")
            grad_var_names.append(gname)
        elif j in shifted_tensor_idxs:
            grad_lines.append(f"    {gname} = {arg_code}.new_zeros({arg_code}.shape)")
            grad_var_names.append(gname)
        # else: non-tensor kept arg → no grad slot

    # Backward kernel call: kept positional args + grads + original kwargs
    call_args = ", ".join(kept_posargs + grad_var_names)
    call_kwargs = (", " + ", ".join(kwargs)) if kwargs else ""
    bwd_call = f"    {bwd_kernel_name}[{grid}]({call_args}{call_kwargs})"

    # Map kept positional arg source names to grad var names
    name_to_grad: Dict[str, str] = {}
    for j, arg_code in enumerate(kept_posargs):
        base = arg_code if arg_code.isidentifier() else f"arg{j}"
        name_to_grad[arg_code] = f"grad_{base}"

    # Return grads aligned to function parameters
    ret_exprs: List[str] = []
    for pname in tensor_params:
        ret_exprs.append(name_to_grad.get(pname, "None"))

    ret_tuple = ", ".join(ret_exprs)
    if len(ret_exprs) == 1:
        ret_tuple += ","

    bwd_def = [
        f"def {bwd_name}({', '.join(bwd_params)}):",
        *body_lines,
        "    # --- codegen: precomputed grad args (no runtime loops) ---",
        *grad_lines,
        bwd_call,
        f"    return ({ret_tuple})",
    ]
    return "\n".join(bwd_def)


# ---------------- minimal demo ----------------
if __name__ == "__main__":
#     user_stub = '''
# import torch
# def stub_impl(a):
#     out = torch.empty_like(a)
#     kernel[1,](a, out, BLOCK_SIZE=a.numel())
#     return out
# '''.strip()

#     code = gen_bwd_stub(
#         stub_src=user_stub,
#         stub_name="stub_impl",
#         kernel_name="kernel",
#         bwd_kernel_name="kernel_bwd",
#         upstream_param="upstream",
#         tensor_params=["a"],          # only 'a' requires a returned grad
#         tensor_arg_idxs=[0, 1],       # 0:'a', 1:'out' are tensors
#         upstream_map={1: "upstream"}, # index 1 (out) gets the upstream
#         folded_const_idxs=[],         # no positional constants in this stub
#     )
#     print(code)








    user_stub = '''
def stub_impl(
        a,
        b,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_K=16
    ):

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    # todo: passing grid with meta args isn't supported yet
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1)
    print("grid: ", grid)
    kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),

        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    c = ...
    return c
'''.strip()

    code = gen_bwd_stub(
        stub_src=user_stub,
        stub_name="stub_impl",
        kernel_name="kernel",
        bwd_kernel_name="kernel_bwd",
        upstream_param="upstream",
        tensor_params=["a", "b"],          # only 'a' requires a returned grad
        tensor_arg_idxs=[0, 1, 2],       # 0:'a', 1:'out' are tensors
        upstream_map={2: "upstream"}, # index 1 (out) gets the upstream
        folded_const_idxs=[],         # no positional constants in this stub
    )
    print(code)