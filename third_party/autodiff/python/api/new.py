from .common import *

def raise_to_triton_lang(ttir_path: str):
    out_dir = os.path.dirname(ttir_path)
    os.makedirs(out_dir, exist_ok=True)
    # Use repo root (TRITON_AUTODIFF_DIR) to locate raise.py in third_party tree
    # This avoids relying on the backend file location.
    raise_py = os.path.join(dir, "third_party/autodiff/python/kernel_agent/tools/backward_naive/raise.py")
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
    # Use repo root (TRITON_AUTODIFF_DIR) to locate emit_stub.py in third_party tree
    emit_stub_py = os.path.join(
        dir, "third_party/autodiff/python/kernel_agent/tools/backward_naive/emit_stub.py"
    )
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
        bwd_jit_fn = jit_fn._bwd_kernel

        # 1) extract fwd_compiled_kernel

        device = driver.active.get_current_device()
        fwd_kernel_cache, target, backend, _binder = jit_fn.device_caches[device]
        bwd_kernel_cache, target, backend, _binder = bwd_jit_fn.device_caches[device]

        assert len(fwd_kernel_cache) == 1, "Temporary limitation: retracing is not yet supported."

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

            # get user stub source lazily (module is fully initialized now)
            mod_name = jit_fn.fn.__module__
            # stub_name = getattr(jit_fn, "stub_name", "stub")
            stub_name = "stub"
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


            ### create StubOverrideDCK and install into user's module ###
            create_stub_op(raised_py_path, mod_name, is_override_stub=True)


        # if overwrite_fp is provided then raise the kernel stored in the provided file
        bwd_jit_fn._raised = load_raised_jit(raised_py_path)    # JITFunction
        print("bwd_jit_fn._raised", bwd_jit_fn._raised)

        # expose artifacts in context store
        # publish raised.py as the backward file pointer
        _AD_ARTIFACTS.set(raised_py_path)

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
        # todo-high: ugly
        bwd_jit_fn._autodiff_info.append(folded)

        if VERBOSE:
            print("[my_hook] compile_dict['signature']", compile_dict["signature"])
            print("[my_hook] compile_dict['constants']", compile_dict["constants"])
            print("[my hook] jit_fn.params:", jit_fn.params)
            print("jit_fn.signature.parameters", jit_fn.signature.parameters)

            print("Hard‑coded parameter indices:", folded)
            print("Hard‑coded parameter names:  ", names)

    return False


# dont set it inside autograd fn, but rather set it here on module lvl (triggers at import)
# Bc I'm registering hooks on the fwd JITFucntions -- so I need to register my hook
# before the first user invocation of fwd JITFucntions (so that my wrapping can see
# as many fwd signatures as possible)
triton.knobs.runtime.jit_post_compile_hook = my_post_hook








# # same as DifferentiatedCompiledKernel (DCK), but operating on the level of stubs (not on the lvel of kernels as DCK does);
# # useful to provide llm with ability to overwirte stubs
# class StubOverrideDCK(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, stubs, *all_stub_inputs):

#         fwd_stub, bwd_stub = stubs
#         out = fwd_stub(*all_stub_inputs)

#         # ugly workaround because save_for_backward only works for tensor inputs
#         tensor_stub_inputs = [a for a in all_stub_inputs if isinstance(a, torch.Tensor)]
#         ctx.save_for_backward(*tensor_stub_inputs)
#         ctx.non_tensor_inputs = [a for a in all_stub_inputs if not isinstance(a, torch.Tensor)]
#         ctx.arg_types = [isinstance(a, torch.Tensor) for a in all_stub_inputs]

#         ctx.bwd_stub = bwd_stub

#         return out

#     @staticmethod
#     def backward(ctx, *upstream_grads): # wrt stub outputs

#         # reconstruct all fwd kernel args
#         all_stub_inputs = []
#         tensor_idx = 0
#         non_tensor_idx = 0
#         for is_tensor in ctx.arg_types:
#             if is_tensor:
#                 all_stub_inputs.append(ctx.saved_tensors[tensor_idx])
#                 tensor_idx += 1
#             else:
#                 all_stub_inputs.append(ctx.non_tensor_inputs[non_tensor_idx])
#                 non_tensor_idx += 1

#         # call bwd stub with all fwd stub inputs + upstream grads
#         downstream_grads = ctx.bwd_stub(*all_stub_inputs, *upstream_grads)

#         downstream_per_input = []
#         tensor_idx = 0
#         for i, is_tensor in enumerate(ctx.arg_types):
#             if is_tensor:
#                 downstream_per_input.append(downstream_grads[tensor_idx])
#                 tensor_idx += 1
#             else:
#                 downstream_per_input.append(None)

#     return (None, *per_input_grads)


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
    def backward(ctx, *upstreams):
        it_t = iter(ctx.saved_tensors)
        it_n = iter(ctx.non_ten)
        all_inps = [next(it_t) if t else next(it_n) for t in ctx.is_ten]
        # in emit_stub.py i made upstream args (to the generated bwd stub) to be keyword only
        # and then appended the added upstream_* args to the end bwd stub's arg list.
        # becuase I want to preserve defult args which user orig stub might have,
        # and not break python’s rule that non‑default params cannot follow defaulted ones
        kw_up = {f"upstream_{i}": g for i, g in enumerate(upstreams)}
        grads_for_tensors = ctx.bwd_stub(*all_inps, **kw_up)

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
# calls the generated stub (which in turn calls the genrated bwd kenrel)
#
# I can't import form the generated_file (which contains the bwd_kernel and bwd_stub) -- becuase it's
# compitely in a random place on disk -- instead seems what i can do is parse the genrated
# file and python compile that and exec it into some new namespace to turn these
# strings into a python fucntions, which i then case use to e.g.
# closure capture into the StubOverrideDCK.backawrd
def create_stub_op(gen_fp, usr_module_name, name="stub", *, is_override_stub=False):
    import runpy

    # gen_fp = "/abs/path/to/generated/autogen.py"
    assert os.path.exists(gen_fp)
    bwd_stub = runpy.run_path(gen_fp)[f"backward_{name}"]      # function object

    # ns = {}
    # exec(compile(generated_source_text, "<autogen>", "exec"), ns)
    # bwd_stub = ns[f"backward_{name}"]

    usr_module = sys.modules.get(usr_module_name)

    fwd_stub = getattr(usr_module, name)
    stubs = (fwd_stub, bwd_stub)

    op = partial(StubOverrideDCK.apply, stubs)

    # note: in this case, ofc replacing the fwd user stub,
    # not fwd user kernel -- becuase the StubOverrideDCK
    # operates on the lvl of stubs
    if is_override_stub:
        # install to user module -- overwrite
        # user fwd stub with the StubOverrideDCK
        setattr(usr_module, name, op)

    return op

