import runpy
import inspect, functools

from .common import *


# todo: replace the two subprocess helpers with import-first calls
# the two helpers below are ugly -- reorganize the package so that you can just
# import "raise" and "emit_stub" functions instead of needing to call them via subprocess.run
def raise_to_triton_lang(ttir_path: str):
    out_dir = os.path.dirname(ttir_path)
    os.makedirs(out_dir, exist_ok=True)
    # Assume TRITON_AUTODIFF_DIR points to the subdir 'triton_autodiff'.
    # The top-level 'kernel_agent' lives one level up from there.
    repo_root = os.path.abspath(os.path.join(dir, ".."))
    raise_py = os.path.join(repo_root, "kernel_agent", "tools", "backward_naive", "raise.py")
    proc = subprocess.run([sys.executable, raise_py, ttir_path],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"raise.py failed on {ttir_path}")
    dst = os.path.join(out_dir, "raised.py")
    with open(dst, "w") as f:
        f.write(proc.stdout)
    return dst

def emit_stub_to_file(
    raised_py_path: str,
    stub_src: str,
    stub_name: str,
    fwd_kernel_name: str,
    bwd_kernel_sym: str,
    idxs_buffers,
    idx_folded,
    signature_key: str,
):
    # Assume TRITON_AUTODIFF_DIR points to 'triton_autodiff' and locate tools one level up
    repo_root = os.path.abspath(os.path.join(dir, ".."))
    emit_stub_py = os.path.join(repo_root, "kernel_agent", "tools", "backward_naive", "emit_stub.py")
    proc = subprocess.run(
        [
            sys.executable,
            emit_stub_py,
            stub_src,
            stub_name,
            fwd_kernel_name,
            bwd_kernel_sym,
            repr(tuple(idxs_buffers)),
            repr(tuple(idx_folded)),
            signature_key,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"emit_stub.py failed for {raised_py_path}")
    code = proc.stdout
    # append into raised.py
    print("writting stub to ", raised_py_path)
    with open(raised_py_path, "a") as f:
        f.write("\n")
        f.write(code)
        f.write("\n")
    return code



def load_raised_jit(raised_py_path):
  """
  Import generated raised.py and return the first @triton.jit JITFunction found.
  """
  import importlib.util, sys, uuid
  from triton.runtime.jit import JITFunction

  mod_name = f"autodiff_raised_{uuid.uuid4().hex[:8]}"
  spec = importlib.util.spec_from_file_location(mod_name, raised_py_path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[mod_name] = mod
  assert spec.loader is not None, f"Failed to load spec for {raised_py_path}"
  spec.loader.exec_module(mod)

  candidates = [obj for obj in vars(mod).values() if isinstance(obj, JITFunction)]
  if not candidates:
      raise RuntimeError(f"No @triton.jit kernel found in {raised_py_path}")
  return candidates[0]

# Context for per-call override of backward kernel path
_AD_OVERWRITE_FP = contextvars.ContextVar("ad_overwrite_fp", default=None)
# capture artifacts produced by the autodiff hook
_AD_ARTIFACTS = contextvars.ContextVar("ad_artifacts", default=None)

@contextlib.contextmanager
def autodiff_overwrite_fp(path: str):
  token = _AD_OVERWRITE_FP.set(path)
  assert str(path).endswith(".py"), "backward overwrite expects a triton-lang (not ttir) kernel"
  try:
    yield
  finally:
    _AD_OVERWRITE_FP.reset(token)

@contextlib.contextmanager
def record_autodiff_artifacts():
  """
  Capture backward file pointer (raised.py) for the current trace.
  Usage:
      with record_autodiff_artifacts():
          ... launch a Triton kernel once ...
      bwd_fp = get_last_bwd_fp()
  """
  token = _AD_ARTIFACTS.set(None)
  try:
    yield
  finally:
    _AD_ARTIFACTS.reset(token)

def get_last_bwd_fp() -> Optional[str]:
  """
  Return the most recent backward file pointer (raised.py path) produced by the autodiff hook.
  None if no kernels were traced yet in this process.
  """
  return _AD_ARTIFACTS.get()







def my_post_hook(key, repr, fn, compile, is_manual_warmup, already_compiled):


    jit_fn = fn.jit_function  # JITFunction

    # compile hook registers on all instances of JITFunction,
    # but i want this hook to trigger only on fwd JITFunctons
    if not already_compiled and hasattr(jit_fn, '_is_fwd_kernel'):

        if VERBOSE: print(f"[my hook] compile hook triggered on the fwd JITFunction: {fn.name}")

        compile_dict = compile
        # bwd_jit_fn = jit_fn._bwd_kernel

        # 1) extract fwd_compiled_kernel

        device = driver.active.get_current_device()
        fwd_kernel_cache, target, backend, _binder = jit_fn.device_caches[device]
        # bwd_kernel_cache, target, backend, _binder = bwd_jit_fn.device_caches[device]

        # get the kernel using the same key
        fwd_compiled_kernel = fwd_kernel_cache[key]

        # 2) write fwd IR

        # todo: cleanup
        # the "key" arg is just a python string with input signatures of the kernel
        # but I want some folder name -- one way is to hash it
        hash_object = hashlib.sha256(key.encode())
        dir_name = hash_object.hexdigest()[:10]
        if VERBOSE: print("dir_name: ", dir_name)

        os.makedirs(f"generated/{dir_name}", exist_ok=True)
        with open(f"generated/{dir_name}/inp.ttir", "w") as f:
          f.write(fwd_compiled_kernel.asm['ttir'])

        # 3) autodiff

        # optionally override via context manager;
        # this is runtime overwrite
        raised_py_path = _AD_OVERWRITE_FP.get()

        # new API installs a Python-level backward stub on the forward JITFunction, so backward
        # is not tied to a single forward specialization.
        # To allow retracing: enforce a single shared backward kernel+stub across all
        # forward traces (in other words, require callers to use autodiff_overwrite_fp).
        # Without the assert below, each new forward signature will silently create
        # a separate backward kernel+stub for each fwd trace, so multi-shape gradcheck
        # (e.g. passing different N_CTX to fwd attention kernel) will pass
        # **but for the wrong reason** (per-trace backward pairs).
        # Instead I want to enforce the following behavior: first trace passes; second trace
        # fails until the single backward kernel+stub is generalized by the llm to handle the
        # new shape as well
        if len(fwd_kernel_cache) > 1:
            # for call sight clarity: if no explicit overwrite is provided, fall back to the
            # last artifacts captured in‑process. This keeps a single shared backward across
            # retraces without requiring the caller to use the context manager;
            # To stash/read the previous bwd_fp, use a per‑kernel persistent pointer instead of
            # the short‑lived ContextVar (_AD_ARTIFACTS), the ContextVar is intentionally reset
            # at the end of record_autodiff_artifacts() to avoid cross‑kernel contamination,
            # so it is usually None here, instead read the last installed path from the JITFunction._generated_bwd_stub_path
            if not raised_py_path:
                raised_py_path = jit_fn._generated_bwd_stub_path

            assert raised_py_path, "Forward kernel retraces, but backward overwrite is not provided. Aborting to avoid creating another backward kernel + stub pair."

        mod_name, stub_name = jit_fn._autodiff_stub_info

        # run the mlir pass to generate TTIR,
        # and then raise that ttir to triton-lang
        # (via raise_to_triton_lang below)
        if not raised_py_path:
            run_mlir_pass(f"generated/{dir_name}")
            bwd_fp = f"generated/{dir_name}/out.ttir"

            # 4) create callable python fn for bwd

            ### generate kernel ###

            raised_py_path = raise_to_triton_lang(bwd_fp)

            ### generate stub ###

            # folded indices (positional) were already computed in your hook
            idx_folded = list(p[0] for p in compile_dict["constants"])  # works with current structure
            # which fwd-call args carry upstream
            idxs_bufs = getattr(jit_fn, "idxs_buffers", ())

            # user stub lives in the same module as their kenrel
            stub_src = get_stub_src_from_module(mod_name, stub_name)

            # generate and append via CLI script
            emit_stub_to_file(
                raised_py_path,
                stub_src,
                stub_name,
                jit_fn.fn.__name__,
                f"backward_{jit_fn.fn.__name__}",
                tuple(idxs_bufs),
                tuple(idx_folded),
                builtins.repr(compile_dict["signature"]),
            )


            bwd_stub = runpy.run_path(raised_py_path)[f"backward_{stub_name}"]
            # _bwd_stub_proxy then uses it at runtime to load the backward generated stub
            setattr(jit_fn, "_generated_bwd_stub", bwd_stub)
            # remember path on the kernel for future retraces. We avoid relying on the
            # thread‑local ContextVar because it is reset after setup() and can point to
            # artifacts of a different kernel if multiple kernels are traced interleaved
            setattr(jit_fn, "_generated_bwd_stub_path", raised_py_path)

            # expose artifacts in context store
            # publish raised.py as the backward file pointer
            _AD_ARTIFACTS.set(raised_py_path)

        else:

            # compile_kernel(..., overwrite_fp=...) path already re‑executes the user module, runs setup(),
            # and attaches the edited stub from the on disk generated/[sha]/raised.py even when the JIT hook doesn’t fire
            # because the forward specialization is cached. Allows for reload without re‑running the MLIR pass

            # ensure the bwd stub is installed on the forward JITFunction when overwrite_fp is used
            # so StubOverrideDCK.backward can find it without regenerating. The hook already does this
            # on fresh generations; we add a defensive install here for overwrite path

            # NOTE: LLM edits are picked up because I rebind each time
            bwd_stub = runpy.run_path(raised_py_path)[f"backward_{stub_name}"]
            assert callable(bwd_stub)
            setattr(jit_fn, "_generated_bwd_stub", bwd_stub)
            setattr(jit_fn, "_generated_bwd_stub_path", raised_py_path)

        # if overwrite_fp is provided then raise the kernel stored in the provided file
        # bwd_jit_fn._raised = load_raised_jit(raised_py_path)    # JITFunction
        # print("bwd_jit_fn._raised", bwd_jit_fn._raised)

        # jit_fn._raised = load_raised_jit(raised_py_path)    # JITFunction
        # print("jit_fn._raised", jit_fn._raised)



        # comment:
        # for this path, don't need to attach any entry into the cache of the backward_jit_fucntion
        # (as oppose to the USE_RAISED=False path) becuase USE_RAISED=True path is integrated basically calling
        # the pytohn JITFunction object (bwd_kerne._raised) from the wrap_bwd_kernel (so no populating of bwd jut fucntion caches is needed)


        # still keep _autodiff_info on bwd_jit_fn (folded indices) for the wrapper logic
        # (for both USE_RAISED=True and USE_RAISED=False) becuase wrap_bwd_kernel reads it unconditionally

        # recover all arguments that have been folded into the CompiledKernel (need for later removal
        # of these folded args from fwd_kernel_args inside wrapped_bwd_kernel before passing them bwd_kernel);
        # includes compile‑time constants (declared `constexpr`) AND automatically specialised (ints/bools/tuples …)
        folded = list(p[0] for p in compile_dict["constants"])
        names = [jit_fn.arg_names[i] for i in folded]

        if VERBOSE:
            print("[my_hook] compile_dict['signature']", compile_dict["signature"])
            print("[my_hook] compile_dict['constants']", compile_dict["constants"])
            print("[my hook] jit_fn.params:", jit_fn.params)
            print("jit_fn.signature.parameters", jit_fn.signature.parameters)

            print("Hard‑coded parameter indices:", folded)
            print("Hard‑coded parameter names:  ", names)

    return True


# dont set it inside autograd fn, but rather set it here on module lvl (triggers at import)
# Bc I'm registering hooks on the fwd JITFucntions -- so I need to register my hook
# before the first user invocation of fwd JITFucntions (so that my wrapping can see
# as many fwd signatures as possible)
triton.knobs.runtime.jit_post_compile_hook = my_post_hook



# same as DifferentiatedCompiledKernel (DCK), but operating on the level of stubs (not on the lvel of kernels as DCK does);
# useful to provide llm with ability to overwirte stubs
class StubOverrideDCK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, stubs, *all_stub_inputs):
        fwd_stub, bwd_stub = stubs
        out = fwd_stub(*all_stub_inputs)
        ten = [x for x in all_stub_inputs if isinstance(x, torch.Tensor)]
        ctx.save_for_backward(*ten)
        ctx.non_ten = [x for x in all_stub_inputs if not isinstance(x, torch.Tensor)]
        ctx.is_ten = [isinstance(x, torch.Tensor) for x in all_stub_inputs]
        ctx.bwd_stub = bwd_stub
        return out

    @staticmethod
    def backward(ctx, *upstreams): # wrt stub outputs

        # reconstruct all fwd kernel args
        it_t = iter(ctx.saved_tensors)
        it_n = iter(ctx.non_ten)
        all_inps = [next(it_t) if t else next(it_n) for t in ctx.is_ten]

        # in emit_stub.py i made upstream args (to the generated bwd stub) to be keyword only
        # and then appended the added upstream_* args to the end bwd stub's arg list.
        # becuase I want to preserve defult args which user orig stub might have,
        # and not break python’s rule that non‑default params cannot follow defaulted ones
        kw_up = {f"upstream_{i}": g for i, g in enumerate(upstreams)}
        # call bwd stub with all fwd stub inputs + upstream grads
        grads_for_tensors = ctx.bwd_stub(*all_inps, **kw_up)

        # todo: layernorm tests fail because retrun order mismatches -- my code expects outputs
        # of the stub be in same order as inputs to the stub;
        # IOW: wrapper expects the backward stub to return one gradient per tensor
        # input parameter of the stub, in the stub’s declaration order
        if len(grads_for_tensors) != sum(ctx.is_ten):
            raise RuntimeError("Backward stub must return one grad per tensor input (in stub order).")

        # align to forward inputs (Tensor -> grad, non‑Tensor -> None)
        it_g = iter(grads_for_tensors)
        per_input = [next(it_g) if t else None for t in ctx.is_ten]
        return (None, *per_input)  # first arg (stubs tuple) has no grad

import sys, importlib, inspect, textwrap

def get_stub_src_from_module(mod_name, stub_name: str) -> str:
    mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)
    print("[get_stub_src_from_module] mod", mod)
    stub_obj = getattr(mod, stub_name)  # assumes the stub is a top-level def
    stub_obj = inspect.unwrap(stub_obj) # in case user decorated the stub too
    return textwrap.dedent(inspect.getsource(stub_obj))  # -> str



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
def _add_kwarg_support(fwd_stub, bwd_proxy):
    sig   = inspect.signature(fwd_stub)
    names = [p.name for p in sig.parameters.values()]

    @functools.wraps(fwd_stub)  # keeps __name__, __qualname__, __doc__ and __wrapped__ for inspect.unwrap
    def wrapped(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs); bound.apply_defaults()
        # fixed positional order for c++ apply
        ordered = [bound.arguments[n] for n in names]
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
        # late-resolving proxy that the hook will fill;
        # when writting approach which decorates stub (not the kernel as in the legacy api) --
        # i couldn't easily return StubOverrideDCK from my stub decorator (becuase StubOverrideDCK
        # gets created only later and from inside the hook) -- but when my autodiff decorator runs
        # don't yet have access to the bwd generated stub.
        # Solution is to basically 1) inside the hook (when the bwd stub is created) install it on some
        # kernel's attribute and 2) make StubOverrideDCK.backward to check for that attribute on the kernel
        # (to see if the bwd_stub has been installed there by the hook or not)
        def _bwd_stub_proxy(*args, **kwargs):
            fn = getattr(fwd_kernel, "_generated_bwd_stub", None)
            if fn is None:
                raise RuntimeError("backward stub not ready; run forward once")
            return fn(*args, **kwargs)
        wrapped = _add_kwarg_support(fwd_stub, _bwd_stub_proxy)   # do not return .apply directly
        setattr(wrapped, "__is_autodiff_stub__", True)            # tag for loader detection
        return wrapped

    return inner
