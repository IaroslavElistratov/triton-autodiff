# codegen_bwd_stub.py
import ast, textwrap

def _src_of(node, src):
    seg = ast.get_source_segment(src, node)
    return seg if seg is not None else ast.unparse(node)

def _find_func(src, name):
    mod = ast.parse(src)
    for n in mod.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise ValueError(f"function {name} not found")

def _find_kernel_call(func_node, src, kernel_name):
    # find first: kernel[...](...)
    class Finder(ast.NodeVisitor):
        def __init__(self): self.hit = None
        def visit_Call(self, node):
            if isinstance(node.func, ast.Subscript):
                val = node.func.value
                if isinstance(val, ast.Name) and val.id == kernel_name:
                    self.hit = node
                    return
            self.generic_visit(node)
    f = Finder(); f.visit(func_node)
    if f.hit is None: raise ValueError("kernel[...] call not found")
    call = f.hit
    grid_code = _src_of(call.func.slice, src)
    args_code = [_src_of(a, src) for a in call.args]
    kw_code = [f"{kw.arg}={_src_of(kw.value, src)}" for kw in call.keywords]
    return call, grid_code, args_code, kw_code

def gen_bwd_stub(
    stub_src: str,
    stub_name: str,
    kernel_name: str,
    bwd_kernel_name: str,
    upstream_param: str,
    tensor_params: list[str],        # names of function params that need grads, in order
    tensor_arg_idxs: list[int],      # positions in kernel call args that are tensor inputs
    upstream_map: dict[int, str],    # arg_idx -> upstream tensor name in bwd stub
    bwd_stub_name: str | None = None
) -> str:
    fn = _find_func(stub_src, stub_name)
    call, grid, posargs, kwargs = _find_kernel_call(fn, stub_src, kernel_name)

    # original header/body (preserve user code)
    header = _src_of(ast.parse(stub_src).body[0], stub_src)
    # reconstruct header cleanly to inject upstream param
    param_names = [a.arg for a in fn.args.args]
    bwd_params = param_names + [upstream_param]
    bwd_name = bwd_stub_name or f"{stub_name}_bwd"

    body_lines = [textwrap.indent(_src_of(stmt, stub_src), "    ") for stmt in fn.body]

    # precompute grad vars
    grad_lines = []
    grad_var_names = []
    for i, arg_code in enumerate(posargs):
        # try a readable grad name
        base = arg_code if arg_code.isidentifier() else f"arg{i}"
        gname = f"grad_{base}"
        if i in upstream_map:
            grad_lines.append(f"    {gname} = {upstream_map[i]}.clone()")
            grad_var_names.append(gname)
        elif i in tensor_arg_idxs:
            grad_lines.append(f"    {gname} = torch.zeros_like({arg_code})")
            grad_var_names.append(gname)
        # else: non‑tensor → no grad slot

    # build bwd call
    call_args = ", ".join(posargs + grad_var_names)
    call_kwargs = (", " + ", ".join(kwargs)) if kwargs else ""
    bwd_call = f"    {bwd_kernel_name}[{grid}]({call_args}{call_kwargs})"

    # return only grads for tensor params in the function signature order
    ret_vars = []
    for pname in tensor_params:
        # find which kernel arg index corresponded to this pname
        idxs = [i for i, a in enumerate(posargs) if a == pname]
        if not idxs: continue
        idx = idxs[0]
        base = pname if pname.isidentifier() else f"arg{idx}"
        ret_vars.append(f"grad_{base}")
    ret_tuple = ", ".join(ret_vars)
    if len(ret_vars) == 1: ret_tuple += ","  # single‑element tuple syntax

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
    user_stub = '''
import torch
def stub_impl(a):
    out = torch.empty_like(a)
    kernel[1,](a, out, BLOCK_SIZE=a.numel())
    return out
'''.strip()

    code = gen_bwd_stub(
        stub_src=user_stub,
        stub_name="stub_impl",
        kernel_name="kernel",
        bwd_kernel_name="kernel_bwd",
        upstream_param="upstream",
        tensor_params=["a"],        # only 'a' requires a returned grad
        tensor_arg_idxs=[0, 1],     # kernel args 0:'a', 1:'out' are tensors
        upstream_map={1: "upstream"}# index 1 (out) gets the upstream
    )
    print(code)
