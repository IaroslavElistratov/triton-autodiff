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


def build_signature_comment_from_compile_sig(stub_name, param_names, num_returns, compile_signature):
    """
    Build signature comment for backward stub. Used by both emit_stub.py and new.py.
    This handles the simple case where we have compile signature directly.

    For the complex case with folding (emit_stub.py), use build_signature_comment().
    """
    # Determine tensor params from compile signature
    tensor_params = []
    for param in param_names:
        # Handle case mismatch: kernel uses uppercase (Q, K, V), stub uses lowercase (q, k, v)
        sig_key = param.upper() if param.upper() in compile_signature else param
        if sig_key not in compile_signature:
            # Fallback: case-insensitive search
            for k in compile_signature:
                if k.lower() == param.lower():
                    sig_key = k
                    break

        if sig_key in compile_signature:
            param_type = compile_signature[sig_key]
            # Same check as emit_stub.py: isinstance(t, str) and t.startswith("*")
            if isinstance(param_type, str) and param_type.startswith("*"):
                tensor_params.append(param)

    # Build return string
    ret = ", ".join(f"grad_{p}" for p in tensor_params)

    # Build comment using common function
    return build_signature_comment(f"backward_{stub_name}", param_names, num_returns, ret)


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


# Keep old name for backward compatibility
def generate_signature_comment(stub_name, param_names, num_returns, compile_signature):
    """Deprecated: Use build_signature_comment_from_compile_sig instead."""
    return build_signature_comment_from_compile_sig(stub_name, param_names, num_returns, compile_signature)
