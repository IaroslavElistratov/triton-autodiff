# AST helper functions for kernel call site analysis
# Adapted from emit_stub.py to avoid code duplication

def _src_of(node, src):
    """Get source segment from AST node."""
    import ast
    seg = ast.get_source_segment(src, node)
    return seg if seg is not None else ast.unparse(node)


def _find_func(src, name):
    """Find function definition by name in source."""
    import ast
    mod = ast.parse(src)
    for n in mod.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise ValueError(f"function {name} not found")


def _find_kernel_call(func_node, src, kernel_name):
    """Find kernel[grid](...) call in function body."""
    import ast

    class Finder(ast.NodeVisitor):
        def __init__(self):
            self.hit = None

        def visit_Call(self, node):
            if isinstance(node.func, ast.Subscript):
                val = node.func.value
                if isinstance(val, ast.Name) and val.id == kernel_name:
                    self.hit = node
                    return
            self.generic_visit(node)

    f = Finder()
    f.visit(func_node)
    if f.hit is None:
        raise ValueError(f"kernel[...] call not found for {kernel_name}")

    call = f.hit
    grid_code = _src_of(call.func.slice, src)
    posargs_code = [_src_of(a, src) for a in call.args]
    kwargs_code = [
        (f"{kw.arg}={_src_of(kw.value, src)}" if kw.arg is not None
         else f"**{_src_of(kw.value, src)}")
        for kw in call.keywords
    ]
    return call, grid_code, posargs_code, kwargs_code


def _returned_names(func_node):
    """Extract names returned by stub function."""
    import ast

    returns = []
    class ReturnFinder(ast.NodeVisitor):
        def visit_Return(self, node):
            if node.value:
                if isinstance(node.value, ast.Name):
                    returns.append(node.value.id)
                elif isinstance(node.value, (ast.Tuple, ast.List)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Name):
                            returns.append(elt.id)

    ReturnFinder().visit(func_node)
    return returns


def extract_stub_info(stub_source, stub_name):
    """Extract parameter names and return count from stub source."""
    import ast
    try:
        mod = ast.parse(stub_source)
        for node in mod.body:
            if isinstance(node, ast.FunctionDef) and node.name == stub_name:
                # Get parameter names
                param_names = [arg.arg for arg in node.args.args]
                if node.args.kwonlyargs:
                    param_names.extend([arg.arg for arg in node.args.kwonlyargs])

                # Count returns
                returns = []
                class ReturnFinder(ast.NodeVisitor):
                    def visit_Return(self, n):
                        if n.value:
                            if isinstance(n.value, ast.Name):
                                returns.append(n.value.id)
                            elif isinstance(n.value, (ast.Tuple, ast.List)):
                                for elt in n.value.elts:
                                    if isinstance(elt, ast.Name):
                                        returns.append(elt.id)
                ReturnFinder().visit(node)

                return param_names, len(returns)
    except:
        pass
    return None, 0


def analyze_stub_for_signature(fwd_source, stub_name, kernel_name, compile_signature):
    """
    Analyze stub to determine tensor parameters and return count.
    Uses kernel call site analysis - NO name heuristics.

    Instead of matching stub param names to kernel param names, this
    matches stub params to their POSITIONS in the kernel call, the
    checks if those positions are pointer types in compile_signature.

    Call-site analysis (current approach)
    - Parse stub AST to find kernel[grid](...) call
    - Match stub params by POSITION in kernel call (not by name)
    - Check compile_signature to see if that position is a pointer type
    - Example: stub calls kernel[grid](x, weight, eps, ...)
    -   Position 0: x → compile_signature[0] = '*fp32' → tensor
    -   Position 1: weight → compile_signature[1] = '*fp32' → tensor
    -   Position 3: eps → compile_signature[2] = 'fp32' → scalar (no *)
    - Correctly identifies (x, weight, bias) as tensors regardless of kernel param names

    compile_signature has ground truth from Triton compilation

    Args:
        fwd_source: Source code containing stub definition
        stub_name: Name of stub function (e.g., "stub")
        kernel_name: Name of kernel function (e.g., "_layer_norm_fwd_fused")
        compile_signature: Ordered dict {param_name: type_str} from kernel compilation
                          e.g., {'X': '*fp32', 'W': '*fp32', 'eps': 'fp32', ...}

    Returns:
        (param_names, tensor_params, num_returns)
        - param_names: List of all stub parameter names
        - tensor_params: List of stub params that are tensors (identified via call-site analysis)
        - num_returns: Number of return values from stub
    """
    import ast

    # Parse stub AST to find function definition
    fn = _find_func(fwd_source, stub_name)

    # Find kernel[grid](...) call in stub body
    call, grid, posargs_code, kwargs_code = _find_kernel_call(fn, fwd_source, kernel_name)

    # Get stub parameter names from function signature
    param_names = [arg.arg for arg in fn.args.args]

    # Get pointer positions from compile_signature
    # compile_signature is ordered dict: {kernel_param_name: type_str}
    # Position i corresponds to i-th kernel parameter
    # Type starting with "*" indicates pointer/tensor
    ptr_positions = {i for i, typ in enumerate(compile_signature.values())
                     if isinstance(typ, str) and typ.startswith("*")}

    # Identify which stub params are tensors by matching to kernel call positions
    # Algorithm:
    # 1. For each stub parameter name
    # 2. Find if it appears in kernel call positional args (as simple name)
    # 3. If yes, get its position in the call
    # 4. Check if that position is a pointer in compile_signature
    tensor_params = []
    for param in param_names:
        if param in posargs_code:  # Simple name match in call args
            pos = posargs_code.index(param)  # Position in kernel call
            if pos in ptr_positions:  # Is that position a pointer?
                tensor_params.append(param)

    # Count returns from stub
    num_returns = len(_returned_names(fn))

    return param_names, tensor_params, num_returns


def build_signature_comment_from_stub_analysis(stub_name, fwd_source, kernel_name, compile_signature):
    """
    Build signature comment using kernel call site analysis.

    Uses actual kernel call structure to determine tensor parameters - no name heuristics.
    Parses stub AST to find kernel[grid](...) call, matches stub params to their positions
    in the call, then checks compile_signature to see which positions are pointer types.

    Args:
        stub_name: Name of stub function (e.g., "stub")
        fwd_source: Full source code containing stub definition and kernel call
        kernel_name: Name of kernel that stub calls (e.g., "_layer_norm_fwd_fused")
        compile_signature: Kernel compile signature dict {param_name: type_str}

    Returns:
        Formatted signature contract comment string

    Raises:
        ValueError: If stub or kernel call not found in source
        SyntaxError: If source code cannot be parsed
    """
    # Use kernel call site analysis - no heuristics
    param_names, tensor_params, num_returns = analyze_stub_for_signature(
        fwd_source, stub_name, kernel_name, compile_signature
    )
    ret_string = ", ".join(f"grad_{p}" for p in tensor_params)

    return build_signature_comment(f"backward_{stub_name}", param_names, num_returns, ret_string)


def build_signature_comment(bwd_name, param_names, num_returns, ret_string):
    """
    Build signature comment showing exact call pattern and return values.
    Shared between emit_stub.py and new.py.

    Args:
        bwd_name: Name of backward stub
        param_names: List of parameter names
        num_returns: Number of upstream gradients
        ret_string: String showing what gradients to return (e.g., "grad_q, grad_k, grad_v")
    """
    up_names = [f"upstream_{i}" for i in range(num_returns)]

    lines = [
        "",
        "# SIGNATURE CONTRACT (from forward stub):",
        "# Gradcheck will call this EXACTLY as:",
        f"#   {bwd_name}({', '.join(param_names)}"
    ]

    if up_names:
        lines[-1] += f", *, {', '.join(up_names)}"
    lines[-1] += ")"

    lines.extend([
        "#",
        f"# Must return tuple: ({ret_string})",
        ""
    ])

    return "\n".join(lines)


def generate_backward_stub_with_scaffolding(stub_name, fwd_source, kernel_name, compile_signature):
    """
    Generate backward_stub skeleton with proper scaffolding using compile_signature.

    Moved skeleton generation from orchestrator (old approach), because
    orchestrator runs BEFORE kernel compilation -> no access to compile_signature

    Skeleton generation in compile hook (current approach)
    - Hook runs AFTER kernel compilation (triggered by first forward pass)
    - Has access to compile_signature with ground truth type information
    - Uses call-site analysis to match params by position
    - Generates correct scaffolding:
    -   grad_x = torch.zeros_like(x)     # Only for tensors
    -   grad_weight = torch.zeros_like(weight)
    -   # NOT for scalars like eps
    -   return (grad_x, grad_weight)  # Correct count

    Why this timing is critical:
    - Orchestrator: runs once at start, no kernel compilation yet
    - First forward pass: triggers kernel compilation, hook fires
    - Hook has compile_signature available from jit_fn._compile_signature
    - This is the ONLY point where we have both source code AND type info

    Args:
        stub_name: Name of forward stub (e.g., "stub")
        fwd_source: Full source code containing stub and kernel call
        kernel_name: Name of kernel function
        compile_signature: Kernel compile signature dict from Triton

    Returns:
        str: Complete backward_stub skeleton with scaffolding
    """
    # Use call-site analysis to identify tensor params
    param_names, tensor_params, num_returns = analyze_stub_for_signature(
        fwd_source, stub_name, kernel_name, compile_signature
    )

    # Build signature string (same as forward stub)
    # Note: We don't have access to default values easily, so just list params
    bwd_name = f"backward_{stub_name}"
    params_str = ", ".join(param_names)

    # Build upstream kwargs
    upstream_params = ", ".join(f"upstream_{i}" for i in range(num_returns))

    # Generate grad allocation code for tensor params
    grad_allocs = []
    for tparam in tensor_params:
        grad_allocs.append(f"    grad_{tparam} = torch.zeros_like({tparam})")

    # Build return tuple
    ret_parts = [f"grad_{tp}" for tp in tensor_params]
    if len(ret_parts) == 1:
        ret_tuple = f"({ret_parts[0]},)"
    else:
        ret_tuple = f"({', '.join(ret_parts)})"

    skeleton_lines = [
        f"def {bwd_name}({params_str}, *, {upstream_params}):",
        '    """Backward pass for YOUR forward kernel.',
        '    TODO: Implement gradient computation',
        '    - Allocate gradient buffers (see scaffolding below)',
        '    - Call backward kernel(s) to compute gradients',
        '    - Return gradients for tensor inputs in correct order',
        '    """',
        '    # Allocate gradient buffers for tensor inputs',
    ]

    skeleton_lines.extend(grad_allocs)

    skeleton_lines.extend([
        '',
        '    # TODO: Call backward kernel(s) to compute gradients',
        '    # Example:',
        '    #   backward_kernel[grid](',
        f'    #       {", ".join(param_names)},',
        f'    #       {", ".join(f"grad_{tp}" for tp in tensor_params)},',
                      # todo-now: don't hardcode -- can be multiple upstream grads
        f'    #       upstream_0,  # gradient from loss',
        '    #       ...',
        '    #   )',
        '',
        f'    return {ret_tuple}',
    ])

    return "\n".join(skeleton_lines)
