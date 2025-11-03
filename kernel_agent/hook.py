import runpy
import inspect, functools

from .bwd_sig_comment import build_signature_comment_from_stub_analysis, generate_backward_stub_with_scaffolding


# todo-now:
# mv to orchestrator.py populating initial raise.py; and keep the hook only basically monkey patching user's stub in the user code file to make it call the code from raised.py


# Prepend forward kernel+stub source to raised.py so both are in one file.
# Benefits:
# - Namespace isolation solved: backward can call forward kernels (both in same namespace)
# - LLM can edit both: enables coordination (e.g., forward saves intermediates, backward uses them)
# - Runtime binding: load both stubs from raised.py, edits take effect at runtime

# # todo: cleanup
# # the "key" arg is just a python string with input signatures of the kernel
# # but I want some folder name -- one way is to hash it
# hash_object = hashlib.sha256(key.encode())
# dir_name = hash_object.hexdigest()[:10]
# if VERBOSE: print("dir_name: ", dir_name)

# os.makedirs(f"generated/{dir_name}", exist_ok=True)


# # FIRST TRACE: Generate complete skeleton from scratch

# if VERBOSE:
#     print(f"[hook] First trace for {stub_name}, generating fresh skeleton")

# # Get forward source and strip @autodiff decorator
# fwd_source = get_fwd_source_from_module(mod_name)
# if not fwd_source or not fwd_source.strip():
#     raise RuntimeError(f"Failed to extract forward source for {stub_name}")

# # Strip @autodiff decorator from forward source
# # Original forward file has decorator that creates proxies - we don't need it in raised.py
# import re
# fwd_source = re.sub(r'^@autodiff.*$\n?', '', fwd_source, flags=re.MULTILINE)

# # Generate signature comment using call-site analysis
# sig_comment = build_signature_comment_from_stub_analysis(
#     stub_name,
#     fwd_source,
#     jit_fn.fn.__name__,
#     jit_fn._compile_signature
# )

# # Generate backward_stub skeleton with proper scaffolding
# stub_skeleton = generate_backward_stub_with_scaffolding(
#     stub_name,
#     fwd_source,
#     jit_fn.fn.__name__,
#     jit_fn._compile_signature
# )

# # Generate DCK template
# dck_content = gen_dck_template(stub_name)

# # todo: Compiler mode: preserve compiler-generated backward (from MLIR)
# # currently overwrites compiler generated stub with the stub_skeleton

# # Build complete file content
# raised_content = (
#     "# ============================================================\n"
#     "# Forward kernel and stub (copied from user file)\n"
#     "# Backward can call these to recompute intermediates\n"
#     "# ============================================================\n\n"
#     f"{fwd_source}\n\n"
#     "# ============================================================\n"
#     "# Backward kernel and stub\n"
#     "# ============================================================\n"
#     f"{sig_comment}\n"
#     f"{stub_skeleton}\n\n"
#     f"{dck_content}"
# )

# # Write complete skeleton to file
# with open(raised_py_path, "w") as f:
#     f.write(raised_content)

# if VERBOSE:
#     print(f"[hook] Generated skeleton: {raised_py_path}")











# def my_post_hook(key, repr, fn, compile, is_manual_warmup, already_compiled):
#     jit_fn = fn.jit_function

#     if already_compiled or not hasattr(jit_fn, "_is_fwd_kernel"):
#         return True

#     raised_py_path = _AD_OVERWRITE_FP.get()
#     if not raised_py_path:
#         raise RuntimeError(
#             "autodiff hook requires overwrite_fp – call your kernel under "
#             "triton.backends.autodiff.autodiff_overwrite_fp(path_to_raised_py)."
#         )
#     if not os.path.isfile(raised_py_path):
#         raise FileNotFoundError(f"raised.py not found at {raised_py_path}")

#     mod_name, stub_name = jit_fn._autodiff_stub_info

#     if VERBOSE:
#         print(f"[hook] loading raised.py for {stub_name} from {raised_py_path}")

#     raised_module = runpy.run_path(raised_py_path)

#     custom_dck = raised_module.get("StubOverrideDCK")
#     bwd_stub_name = f"backward_{stub_name}"
#     bwd_stub = raised_module.get(bwd_stub_name)
#     fwd_stub = raised_module.get(stub_name)

#     if custom_dck:
#         setattr(jit_fn, "_CustomDCK", custom_dck)

#     if bwd_stub and callable(bwd_stub):
#         setattr(jit_fn, "_generated_bwd_stub", bwd_stub)
#     if fwd_stub and callable(fwd_stub):
#         setattr(jit_fn, "_generated_fwd_stub", fwd_stub)

#     if not custom_dck and (bwd_stub is None or fwd_stub is None):
#         raise RuntimeError(
#             f"{raised_py_path} is missing StubOverrideDCK and forward/backward stubs "
#             f"for '{stub_name}'."
#         )

#     setattr(jit_fn, "_generated_bwd_stub_path", raised_py_path)

#     return True



def my_post_hook(key, repr, fn, compile, is_manual_warmup, already_compiled):
    jit_fn = fn.jit_function  # JITFunction

    # compile hook registers on all instances of JITFunction,
    # but i want this hook to trigger only on fwd JITFunctons
    if not already_compiled and hasattr(jit_fn, '_is_fwd_kernel'):

        if VERBOSE: print(f"[my hook] compile hook triggered on the fwd JITFunction: {fn.name}")

        raised_py_path = _AD_OVERWRITE_FP.get()
        if not raised_py_path:
            raise RuntimeError(
                "autodiff hook requires overwrite_fp – call your kernel under "
                "triton.backends.autodiff.autodiff_overwrite_fp(path_to_raised_py)."
            )
        if not os.path.isfile(raised_py_path):
            raise FileNotFoundError(f"raised.py not found at {raised_py_path}")



        # Store compile signature for later use (needed for RAG path)
        if not hasattr(jit_fn, "_compile_signature"):
            jit_fn._compile_signature = compile["signature"]


        # Load DCK and/or stubs from raised.py
        # New approach (raised.py with DCK): Load CustomDCK class
        # NOTE: LLM edits are picked up because we rebind each time
        raised_module = runpy.run_path(raised_py_path)

        # Try loading custom DCK first (new raised.py files)
        CustomDCK = raised_module.get("StubOverrideDCK")
        assert CustomDCK
        # New path: LLM-editable DCK in raised.py
        setattr(jit_fn, "_CustomDCK", CustomDCK)

    return True


# dont set it inside autograd fn, but rather set it here on module lvl (triggers at import)
# Bc I'm registering hooks on the fwd JITFucntions -- so I need to register my hook
# before the first user invocation of fwd JITFucntions (so that my wrapping can see
# as many fwd signatures as possible)
triton.knobs.runtime.jit_post_compile_hook = my_post_hook


import sys, importlib, inspect, textwrap

def get_stub_src_from_module(mod_name, stub_name: str) -> str:
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    stub_obj = getattr(mod, stub_name)  # assumes the stub is a top-level def
    stub_obj = inspect.unwrap(stub_obj) # in case user decorated the stub too
    return textwrap.dedent(inspect.getsource(stub_obj))  # -> str

def get_fwd_source_from_module(mod_name: str) -> str:
    """Extract forward kernel+stub source from user module, with same filtering as shown to LLM."""
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    fwd_file = getattr(mod, "__file__", None)
    if not fwd_file:
        return ""
    # Use redact_torch_fn for consistent filtering (removes test helpers, keeps kernel+stub)
    # TRITON_AUTODIFF_DIR points to 'triton_autodiff', kernel_agent is one level up
    repo_root = os.path.abspath(os.path.join(dir, ".."))
    kernel_agent_path = os.path.join(repo_root, "kernel_agent")
    if kernel_agent_path not in sys.path:
        sys.path.insert(0, kernel_agent_path)
    from utils import redact_torch_fn
    return redact_torch_fn(fwd_file, None)


def gen_dck_template(stub_name: str) -> str:
    """Generate LLM-editable DCK class for raised.py.

    Creates a working StubOverrideDCK that calls stub/backward_stub directly,
    with optimization guidance in docstring for saving forward intermediates.
    """
    bwd_name = f"backward_{stub_name}"

    return f'''
class StubOverrideDCK(torch.autograd.Function):
    """
    Connects forward/backward stubs for automatic differentiation.

    OPTIMIZATION: To avoid wasteful recomputation of forward intermediates,
    edit this class to save/restore them via ctx.save_for_backward().
    See "CRITICAL: AVOID RECOMPUTING FORWARD INTERMEDIATES" in prompts for pattern.
    """

    @staticmethod
    def forward(ctx, *all_stub_inputs):
        # Call forward stub (defined above in this file)
        result = {stub_name}(*all_stub_inputs)

        # Save tensor inputs for backward
        ten = [x for x in all_stub_inputs if isinstance(x, torch.Tensor)]
        ctx.save_for_backward(*ten)

        # Save non-tensor inputs (scalars, shapes, etc.)
        ctx.non_ten = [x for x in all_stub_inputs if not isinstance(x, torch.Tensor)]
        ctx.is_ten = [isinstance(x, torch.Tensor) for x in all_stub_inputs]

        return result

    @staticmethod
    def backward(ctx, *upstreams):
        # Reconstruct all forward inputs (mix tensors + non-tensors)
        it_t = iter(ctx.saved_tensors)
        it_n = iter(ctx.non_ten)
        all_inps = [next(it_t) if t else next(it_n) for t in ctx.is_ten]

        # Prepare upstream gradients as keyword args
        kw_up = {{f"upstream_{{i}}": g for i, g in enumerate(upstreams)}}

        # Call backward stub (defined above in this file)
        grads_for_tensors = {bwd_name}(*all_inps, **kw_up)

        # Validate return count (must match number of tensor inputs)
        if len(grads_for_tensors) != sum(ctx.is_ten):
            raise RuntimeError("Backward stub must return one grad per tensor input (in stub order).")

        # Align gradients to forward inputs (Tensor → grad, non-Tensor → None)
        it_g = iter(grads_for_tensors)
        per_input = [next(it_g) if t else None for t in ctx.is_ten]
        return tuple(per_input)
'''


# now i'm sucessfully generating both raised user bwd kernel and bwd stub into a file.
# Now basically i want to replace user's stub (fucntion with some name) in their module
# with the StubOverrideDCK -- where StubOverrideDCK's fwd calls users stub and StubOverrideDCK.bwd
# calls the generated stub (which in turn calls the genrated bwd kenrel);
# I can't import form the generated_file (which contains the bwd_kernel and bwd_stub) -- becuase it's
# compitely in a random place on disk -- instead seems what i can do is parse the genrated
# file and python compile that and exec it into some new namespace to turn these
# strings into a python fucntions, which i then case use to e.g.
# closure capture into the StubOverrideDCK.backawrd


# autograd.Function[s] doesn't support kwargs, but user might be
# using their stub with kwargs this helper adds the kwarg support
def _add_kwarg_support(fwd_stub, bwd_proxy, fwd_kernel):
    sig   = inspect.signature(fwd_stub)
    names = [p.name for p in sig.parameters.values()]

    @functools.wraps(fwd_stub)  # keeps __name__, __qualname__, __doc__ and __wrapped__ for inspect.unwrap
    def wrapped(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs); bound.apply_defaults()
        # fixed positional order for c++ apply
        ordered = [bound.arguments[n] for n in names]

        # Check if custom DCK available (new raised.py files)
        CustomDCK = getattr(fwd_kernel, "_CustomDCK", None)
        if CustomDCK:
            # Use LLM-editable DCK from raised.py (no stubs tuple)
            return CustomDCK.apply(*ordered)
        else:
            # Fallback to framework DCK (old raised.py or first call before hook)
            return StubOverrideDCK.apply((fwd_stub, bwd_proxy), *ordered)
    return wrapped



def autodiff(kernel, idxs_buffers, stub_name=None):
    # for user it's more natual to specify "kernel=..." (and not "fwd_kernel")
    # but fwd_kernel reflects the semantics better
    fwd_kernel = kernel
    def inner(fwd_stub):
        # assert isinstance(idxs_buffers, (tuple, int)), f"idxs_buffers must be either tuple or int, got {type(idxs_buffers)}"
        idxs = (idxs_buffers,) if isinstance(idxs_buffers, int) else tuple(idxs_buffers)
        # tag kernel for the hook
        fwd_kernel.idxs_buffers = idxs
        fwd_kernel._is_fwd_kernel = True
        fwd_kernel._autodiff_stub_info = (fwd_stub.__module__, fwd_stub.__name__)  # tell hook which stub
        # Late-resolving proxies for both forward and backward stubs.
        #
        # Why proxies are needed:
        # - This decorator decorates the stub (not the kernel as in legacy API)
        # - So can't return StubOverrideDCK directly because it's created later inside the hook
        # - When decorator runs at import time, don't have access to generated backward stub yet
        # - Raised.py doesn't exist yet (hook creates it during first kernel compilation)
        #
        # Solution:
        # 1. Hook (when bwd/fwd stubs are created) installs them on fwd_kernel attributes:
        #    - setattr(fwd_kernel, "_generated_bwd_stub", bwd_stub)
        #    - setattr(fwd_kernel, "_generated_fwd_stub", fwd_stub)
        # 2. Proxies check these attributes at runtime and use generated versions when available
        # 3. This enables LLM to edit both forward and backward in raised.py - edits take effect
        #    because we rebind stubs from raised.py on each iteration

        def _fwd_stub_proxy(*args, **kwargs):
            # Check if hook installed edited forward from raised.py
            fn = getattr(fwd_kernel, "_generated_fwd_stub", None)
            if fn is not None:
                # Use LLM-editable version from raised.py (enables forward/backward coordination)
                return fn(*args, **kwargs)
            # todo: remove
            # Fallback to original user stub on first call (before hook installs generated version)
            # Hook fires during kernel compilation inside first stub call, so proxy executes before hook
            return fwd_stub(*args, **kwargs)

        def _bwd_stub_proxy(*args, **kwargs):
            fn = getattr(fwd_kernel, "_generated_bwd_stub", None)
            if fn is None:
                raise RuntimeError("backward stub not ready; run forward once")
            return fn(*args, **kwargs)

        # Preserve original user stub's signature on the proxy function.
        # _add_kwarg_support (below) uses inspect.signature() to extract parameter names for argument binding.
        # Without this fix, it sees the proxy's (*args, **kwargs) signature instead of the real stub signature.
        # This causes bind_partial to create ordered = [(q,k,v), {}] instead of [q, k, v],
        # which then makes StubOverrideDCK.forward call _fwd_stub_proxy((q,k,v), {}) instead of (q, k, v),
        # leading to "missing required positional argument" errors when the proxy falls back to fwd_stub.
        _fwd_stub_proxy.__signature__ = inspect.signature(fwd_stub)
        _fwd_stub_proxy.__wrapped__ = fwd_stub  # For inspect.unwrap in case signature extraction uses it

        wrapped = _add_kwarg_support(_fwd_stub_proxy, _bwd_stub_proxy, fwd_kernel)   # Pass proxies + kernel
        setattr(wrapped, "__is_autodiff_stub__", True)            # tag for loader detection
        return wrapped

    return inner
