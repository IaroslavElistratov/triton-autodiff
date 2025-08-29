import re
from typing import List, Tuple

# --- pointer positions from compile signature like: "[('*fp32','D'), ('constexpr',4), ...]"
def _ptr_arg_idxs_from_signature(sig: str):
    body = sig[sig.find("[")+1:sig.rfind("]")]
    items = [x.replace("(", "").replace(")", "") for x in body.split(", (")]
    idxs = []
    for i, it in enumerate(items):
        s = it.replace(" ", "")
        if s.startswith("'*fp") or s.startswith("\"*fp"):
            idxs.append(i)
    return idxs

def _shift_indices(indices, folded):
    folded = sorted(folded)
    return [i - sum(f < i for f in folded) for i in indices]


import ast, textwrap, inspect, sys

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
    class Finder(ast.NodeVisitor):
        def __init__(self): self.hit = None
        def visit_Call(self, node):
            if isinstance(node.func, ast.Subscript):
                val = node.func.value
                if isinstance(val, ast.Name) and val.id == kernel_name:
                    self.hit = node; return
            self.generic_visit(node)
    f = Finder(); f.visit(func_node)
    if f.hit is None: raise ValueError("kernel[...] call not found")
    call = f.hit
    grid_code = _src_of(call.func.slice, src)
    posargs_code = [_src_of(a, src) for a in call.args]
    kwargs_code = [
        (f"{kw.arg}={_src_of(kw.value, src)}" if kw.arg is not None else f"**{_src_of(kw.value, src)}")
        for kw in call.keywords
    ]
    return call, grid_code, posargs_code, kwargs_code

def _render_bwd_paramlist(fn, up_names, src):
    a = fn.args
    parts = []

    # positional-only
    if getattr(a, "posonlyargs", []):
        parts += [p.arg for p in a.posonlyargs]
        parts.append("/")

    # positional-or-keyword (preserve defaults)
    n, m = len(a.args), len(a.defaults)
    for i, p in enumerate(a.args):
        if i >= n - m:
            d = a.defaults[i - (n - m)]
            parts.append(f"{p.arg}={_src_of(d, src)}")
        else:
            parts.append(p.arg)

    # vararg or start keyword-only section
    if a.vararg is not None:
        parts.append(f"*{a.vararg.arg}")
    elif a.kwonlyargs or up_names:
        parts.append("*")

    # existing keyword-only (preserve defaults)
    for p, d in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(p.arg if d is None else f"{p.arg}={_src_of(d, src)}")

    # upstreams as keyword-only
    parts += up_names

    # **kwargs passthrough
    if a.kwarg is not None:
        parts.append(f"**{a.kwarg.arg}")

    return ", ".join(parts)

def gen_bwd_stub_auto(
    stub_src: str,
    *,
    stub_name: str,
    fwd_kernel_name: str,
    bwd_kernel_sym: str,
    idxs_buffers,        # original fwd indices of output buffers
    idx_folded,          # positional indices specialized away in fwd
    ptr_arg_idxs,        # positional indices of pointer args in the fwd call
):
    fn = _find_func(stub_src, stub_name)
    call, grid, posargs, kwargs = _find_kernel_call(fn, stub_src, fwd_kernel_name)

    # find launch stmt index
    launch_idx = None
    for i, s in enumerate(fn.body):
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
            f = s.value.func
            if isinstance(f, ast.Subscript) and isinstance(f.value, ast.Name) and f.value.id == fwd_kernel_name:
                launch_idx = i; break
    assert launch_idx is not None

    # split body around the launch, drop 'return'
    pre_nodes  = [s for s in fn.body[:launch_idx] if not isinstance(s, ast.Return)]
    post_nodes = [s for s in fn.body[launch_idx+1:] if not isinstance(s, ast.Return)]
    pre_lines  = [textwrap.indent(_src_of(s, stub_src), "    ") for s in pre_nodes]
    post_lines = [textwrap.indent(_src_of(s, stub_src), "    ") for s in post_nodes]

    # fold positional args
    folded = sorted(idx_folded)
    kept_posargs = [a for j, a in enumerate(posargs) if j not in folded]
    # shift outputs and pointer indices into kept-args space
    shifted_upstream = _shift_indices(sorted(idxs_buffers), folded)
    shifted_ptrs     = set(_shift_indices(ptr_arg_idxs, folded))

    # name upstream params
    up_names = [f"upstream_{k}" for k in range(len(shifted_upstream))]
    up_map = {i: up_names[k] for k, i in enumerate(shifted_upstream)}

    # params of the new stub = original stub params (with defaults) + upstreams as kw-only
    param_names = [a.arg for a in fn.args.args]
    bwd_name = f"backward_{stub_name}"
    header_params = _render_bwd_paramlist(fn, up_names, stub_src)

    # precompute grad vars for each kept positional arg
    grad_lines, grad_vars = [], []
    for j, a in enumerate(kept_posargs):
        base = a if a.isidentifier() else f"arg{j}"
        g = f"grad_{base}"
        if j in up_map:
            grad_lines.append(f"    {g} = {up_map[j]}.clone()")
            grad_vars.append(g)
        elif j in shifted_ptrs:
            grad_lines.append(f"    {g} = torch.zeros_like({a})")
            grad_vars.append(g)
        # else: non‑tensor, skip

    # build backward launch (replace the fwd one)
    call_kwargs = (", " + ", ".join(kwargs)) if kwargs else ""
    bwd_call = f"    {bwd_kernel_sym}[{grid}]({', '.join(kept_posargs + grad_vars)}{call_kwargs})"

    # return grads for original tensor *parameters* in declaration order
    name_to_grad = {a: f"grad_{(a if a.isidentifier() else f'arg{j}')}" for j, a in enumerate(kept_posargs)}
    tensor_param_names = [p for p in param_names
                          if p in posargs and posargs.index(p) in set(i for i in range(len(posargs)) if i not in folded) and
                             _shift_indices([posargs.index(p)], folded)[0] in shifted_ptrs]
    ret = ", ".join(name_to_grad[p] for p in tensor_param_names) or ""
    if len(tensor_param_names) == 1: ret += ","

    # assemble function
    lines = [f"def {bwd_name}({header_params}):",
             *pre_lines,
             "    # --- codegen stub ---",
             *grad_lines,
             bwd_call,
             *post_lines,
             f"    return ({ret})"]
    return "\n".join(lines)

def _append_stub_into_raised(raised_py_path: str, bwd_stub_src: str, *, alias_to: str | None):
    with open(raised_py_path, "a") as f:
        f.write("\n# --- autodiff: generated backward stub ---\n")
        f.write("import torch\n")
        if alias_to:
            f.write(f"kernel_bwd = {alias_to}\n")
        else:
            f.write("from triton.runtime.jit import JITFunction as _JF\n")
            f.write("kernel_bwd = next(v for v in globals().values() if isinstance(v, _JF))\n")
        f.write("\n")
        f.write(bwd_stub_src)
        f.write("\n")




# ---------------- CLI entry ----------------
if __name__ == "__main__":

    if len(sys.argv) != 8:
        print(
            "Usage: python emit_stub.py <stub_src> <stub_name> <fwd_kernel_name> <bwd_kernel_sym> <idxs_buffers> <idx_folded> <signature_key>"
        )
        raise SystemExit(1)
    stub_src = sys.argv[1]
    stub_name = sys.argv[2]
    fwd_kernel_name = sys.argv[3]
    bwd_kernel_sym = sys.argv[4]
    # parse tuples/lists like "(1, 2)" or "[1, 2]"
    idxs_buffers = tuple(ast.literal_eval(sys.argv[5]))
    idx_folded = tuple(ast.literal_eval(sys.argv[6]))
    signature_key = sys.argv[7]
    ptr_arg_idxs = tuple(_ptr_arg_idxs_from_signature(signature_key))

    code = gen_bwd_stub_auto(
        stub_src,
        stub_name=stub_name,
        fwd_kernel_name=fwd_kernel_name,
        bwd_kernel_sym=bwd_kernel_sym,
        idxs_buffers=idxs_buffers,
        idx_folded=idx_folded,
        ptr_arg_idxs=ptr_arg_idxs,
    )
    print(code)



def test():

    user_stub = '''

def kernel():
  pass

def stub(a, b, BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape; K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), 1, 1)
    kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    )
    return c
'''.strip()

    sig = (
        "[('*fp16','D'), ('*fp16','D'), ('*fp16','D'), "  # a_ptr, b_ptr, c_ptr  → tensors
        "('i32',''), ('i32',''), ('i32',''), "            # M, N, K
        "('i64',''), ('i64',''), ('i64',''), "            # a.stride(0/1), b.stride(0)
        "('i64',''), ('i64',''), ('i64','')]"             # b.stride(1), c.stride(0/1)
        "{'num_warps': 4}"
    )




    # import inspect
    stub_src = user_stub # inspect.getsource(stub_spec)
    # folded indices (positional) were already computed in your hook
    idx_folded = (12, 13, 14) #  [p[0] for p in compile_dict["constants"]]  # works with your current structure
    # pointer positions from compile signature
    ptr_idxs = _ptr_arg_idxs_from_signature(sig) # str(compile_dict["signature"]))
    # which fwd-call args carry upstream
    idxs_bufs = (2, ) # getattr(jit_fn, "_idxs_buffers", ())
    # generate and append
    code = gen_bwd_stub_auto(
        stub_src,
        stub_name="stub", # stub_spec.__name__,
        fwd_kernel_name="kernel", # jit_fn.fn.__name__,
        bwd_kernel_sym="kernel_bwd",
        idxs_buffers=tuple(idxs_bufs),
        idx_folded=tuple(idx_folded),
        ptr_arg_idxs=tuple(ptr_idxs),
    )
    print(code)
    # alias = f"backward_{jit_fn.fn.__name__}"   # matches raise.py’s function name
    # _append_stub_into_raised(raised_py_path, code, alias_to=alias)


    # code = gen_bwd_stub(
    #     ### extract from user source ####
    #     stub_src=user_stub,
    #     #### fix these -- enforce user to use these specific names ###
    #     stub_name="stub_impl",
    #     kernel_name="kernel",
    #     bwd_kernel_name="kernel_bwd",

    #     #### hardcode with upstream_1, upstream_2 etc -- based on the number of elements in the idxs_buffers set  ###
    #     upstream_param="upstream",
    #     #### get this from the python inspect.signatrue ? Aleternatively hardcode ####
    #     tensor_params=["a", "b"],          # only 'a' requires a returned grad
    #     ### get this from the kernel signature -- for every ptr in the sig, this is a tensor arg ###
    #     tensor_arg_idxs=[0, 1, 2],       # 0:'a', 1:'out' are tensors
    #     #### info contained in idxs_buffers set ####
    #     upstream_map={2: "upstream"}, # index 1 (out) gets the upstream
    #     #### these are my "idx_folded = _autodiff_info[-1]" ###
    #     folded_const_idxs=[],         # no positional constants in this stub
    # )
    # print(code)