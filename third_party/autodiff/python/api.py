import sys
import os
# os.environ['TRITON_ALWAYS_COMPILE']='1'
import hashlib
import inspect
import subprocess
from functools import partial
import shutil
from typing import Optional
import contextvars
import contextlib

import torch
torch.manual_seed(0)
DEVICE = torch.device("cuda:0")

import triton
import triton.language as tl
from triton.runtime import driver
from triton.runtime.jit import JITFunction
from triton.compiler import compile as compile_kernel
from triton.backends.compiler import GPUTarget


# True: raises backward TTIR to trtion-lang then re-lowers it to TTIR,
# False: uses backward TTIR directly
USE_RAISED = True

VERBOSE = int(os.environ.get('VERBOSE', 0))
assert VERBOSE in [0, 1, 2]

dir = os.getenv("TRITON_AUTODIFF_DIR")
if dir is None:
    raise ValueError("Please specify TRITON_AUTODIFF_DIR, see README.")

# timeout for triton-opt invocations (seconds)
SUBPROCESS_TIMEOUT_S = float(os.environ.get("TRITON_OPT_TIMEOUT_S", "60"))


def _locate_triton_opt(base_dir: Optional[str]) -> str:
    env_path = os.getenv("TRITON_OPT_BIN")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    if base_dir:
        search_root = os.path.join(base_dir, "build")
        if os.path.isdir(search_root):
            for root, _dirs, files in os.walk(search_root):
                if "triton-opt" in files:
                    candidate = os.path.join(root, "triton-opt")
                    if os.access(candidate, os.X_OK):
                        return candidate
    which_path = shutil.which("triton-opt")
    if which_path:
        return which_path
    raise FileNotFoundError("Could not find `triton-opt`. Set TRITON_OPT_BIN or add to PATH.")

tool = _locate_triton_opt(dir)

# # Optional hint to your local triton-autodiff checkout
# _BASE_DIR = os.getenv("TRITON_AUTODIFF_DIR")

# # Use the robust locator everywhere below
# tool = _locate_triton_opt(_BASE_DIR)

# tool = f"{dir}/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt"


def run_mlir_pass(path):

    os.makedirs(path, exist_ok=True)

    inp_path = f"{path}/inp.ttir"
    out_path = f"{path}/out.ttir"

    # produce bwd ttir
    with open(out_path, "w") as f_out:
        try:
            subprocess.run(
                [tool, "--convert-triton-to-autodiff", "--mlir-print-debuginfo", inp_path],
                stdout=f_out,
                stderr=subprocess.DEVNULL,  # suppress verbose compiler diagnostics
                check=True,                 # raise on failure; we map to a friendly message
                timeout=SUBPROCESS_TIMEOUT_S,
            )
        except subprocess.CalledProcessError:
            raise RuntimeError(
                "triton-opt failed while generating the backward TTIR.\n"
                "Hint: Check that your kernel has static loop bounds, shapes line up, and dtypes are supported."
            )

    # optionally, produce readable fwd ttir
    if VERBOSE >= 1:
        # ugly, needed bc fwd.py files create out.ttir files with default SSA names (%1, %2, ...)
        # and with location info (containing variable names). Here I ran "--mlir-use-nameloc-as-prefix"
        # on it and write to the same files to avoid creating redundant files
        with open(inp_path, "r+") as f_out:
            _ = f_out.read()         # Read existing content
            f_out.seek(0)            # Move cursor to the beginning
            try:
                # Overwrite from the start
                subprocess.run(
                    [tool, "--mlir-use-nameloc-as-prefix", "--mlir-print-debuginfo", inp_path],
                    stdout=f_out,
                    stderr=subprocess.DEVNULL,  # suppress verbose diagnostics
                    check=True,
                    timeout=SUBPROCESS_TIMEOUT_S,
                )
                f_out.truncate()               # Remove remaining old content
            except subprocess.CalledProcessError:
                raise RuntimeError("triton-opt failed while pretty-printing the forward TTIR.")



    # if VERBOSE == 2:

    #     def draw_dot(path, mode):
    #     assert mode in ["fwd", "bwd"]

    #     vis_dir = path + "/vis"
    #     os.makedirs(vis_dir, exist_ok=True)

    #     # a. optionally, produce vis dot
    #     with open(f"{vis_dir}/{mode}.dot", "w") as f:
    #         ttir_path = f"{path}/inp.ttir" if mode == "fwd" else f"{path}/out.ttir"
    #         subprocess.run([tool, "-mlir-use-nameloc-as-prefix", "--view-op-graph", ttir_path], stderr=f,
    #                     # suppress stdout, otherwise prints _inp_readable again
    #                     stdout=subprocess.DEVNULL)

    #     with open(f"{vis_dir}/{mode}.svg", "w") as f:
    #         subprocess.run(["dot", "-Tsvg", f"{vis_dir}/{mode}.dot"], stdout=f)

    #     # b. optionally cluster nodes
    #     subprocess.run(["python", "cluster_dot.py", "--strict", f"{vis_dir}/{mode}.dot", f"{vis_dir}/{mode}_grouped.dot"])

    #     with open(f"{vis_dir}/{mode}_grouped.svg", "w") as f:
    #         subprocess.run(["dot", "-Tsvg", f"{vis_dir}/{mode}_grouped.dot"], stdout=f)

    #     # os.remove(f"{vis_dir}/{mode}.dot")
    #     # os.remove(f"{vis_dir}/{mode}_grouped.dot")

    #     # optionally, produce vis dot
    #     draw_dot(path, mode="fwd")

    #     # optionally, produce vis dot
    #     draw_dot(path, mode="bwd")


def raise_to_triton_lang(ttir_path):
  # Run raiser and write the resulting Triton code to raised.py in the same directory
  out_dir = os.path.dirname(ttir_path)
  os.makedirs(out_dir, exist_ok=True)
  raise_py = os.path.join(dir, "third_party/autodiff/python/kernel_agent/tools/backward_naive/raise.py")
  proc = subprocess.run([sys.executable, raise_py, ttir_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  if proc.returncode != 0:
    print(proc.stderr)
    raise RuntimeError(f"raise.py failed on {ttir_path}")
  with open(os.path.join(out_dir, "raised.py"), "w") as f:
    f.write(proc.stdout)
  return os.path.join(out_dir, "raised.py")


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

    def remove_constexpr(jit_fn, bwd_jit_fn, key):
        # remove constexpr -- bc backward signature or key should not
        # have them (bc they will NOT be provided to the bwd kernel)

        if VERBOSE: print("[remove_constexpr] key: ", key)
        # need to also modify self.signature bc it's used in create_binder -> create_function_from_signature
        #   > in JITFunction.run "binder = create_function_from_signature(self.signature, self.params, backend)"

        # key = key.replace(", ('constexpr', 4)", "")

        # replaces all occurrences of , ('constexpr', [some integer]) in the string
        sub_strs = key.split("[")[1].split("]")[0].split("), (")
        # if VERBOSE: print('sub_strs: ', sub_strs)
        # >>> ["('*fp32', 'D'", "'*fp32', 'D'", "'constexpr', 4", "'*fp32', 'D'", "'*fp32', 'D')"]

        # iterate over dict whose keys are tuples of ints, and
        # extract all ints from all keys into a single list
        idx_const_ints = [i for key in compile_dict['constants'].keys() for i in key]
        num_const_args = len(idx_const_ints)
        if VERBOSE: print("[remove_constexpr] idx_const_ints", idx_const_ints)

        sig_params = list(jit_fn.signature.parameters.values())
        # reverse to avoid shifting issues
        for i, s in reversed(list(enumerate(sig_params))):
            if i in idx_const_ints:
                sub_strs.pop(i)
                sig_params.pop(i)
                bwd_jit_fn.params.pop(i)
        new_key = "[" + "), (".join(sub_strs)
        new_key += "]" if new_key[-1] == ")" else ")]"
        new_key += key.split("]")[1]
        bwd_jit_fn.signature = bwd_jit_fn.signature.replace(parameters=sig_params)

        if VERBOSE:
            print("[remove_constexpr] new_key", new_key)
            print("[remove_constexpr] bwd_jit_fn.signature: ", bwd_jit_fn.signature)
            print("[remove_constexpr] bwd_jit_fn self.params:", bwd_jit_fn.params)

        return new_key, num_const_args

    def key_add_args(key):

        # can't run the binder to automatically create specialization and options (both needed to create key)
        # bc here I don't have acces to *arg, **kwargs from inside the compile_hook
        # there doesn't seem to be a direct way to extract the full args and kwargs
        # from the compile hook -- thus doing the appraoch below
        # # _bound_args, specialization, options = binder(*args, **kwargs)

        if VERBOSE: print("[key_add_args] key: ", key)
        # key:  [('*fp32', 'D'), ('*fp32', 'D')]{'debug': False}

        split = key.split("]")
        if VERBOSE: print("[key_add_args] split: ", split)
        # split:  ["[('*fp32', 'D'), ('*fp32', 'D')", "{'debug': False}"]

        new_key = split[0] + ", "
        for name, str_type in compile_dict["signature"].items():
            if "*" in str_type:
                # todo-low: don't hardcode D
                new_key += f"('{str_type}', 'D'), "
        # cut ", "
        new_key = new_key[:len(new_key)-2]
        # add what I split by
        new_key += "]"
        new_key += split[1]

        if VERBOSE: print("[key_add_args] new_key: ", new_key)
        # new_key:  [('*fp32', 'D'), ('*fp32', 'D'), ('*f32', 'D'), ('*f32', 'D')]{'debug': False}
        return new_key

    def rebuild_binder(jit_fn, delta, backend):

        from triton.runtime.jit import (
            KernelParam,                        # helper for arg metadata
            create_function_from_signature,     # binder factory
        )

        # 1. extend the Python signature *before* we rebuild the binder
        sig_params = list(jit_fn.signature.parameters.values())
        for i in range(delta):
            p = inspect.Parameter(
                f"grad_{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                # leave annotation empty so the binder infers the correct pointer
                # dtype from the actual runtime tensors (e.g. *fp32, *bf16, ...)
                annotation=inspect._empty
            )
            jit_fn.params.append(KernelParam(len(jit_fn.params), p, False, False))
            sig_params.append(p)

        if VERBOSE:
            print(f"[rebuild_binder] adding {delta} args")
            print("[rebuild_binder] jit_fn.signature: ", jit_fn.signature)
            print("[rebuild_binder] jit_fn.params:", jit_fn.params)

        jit_fn.signature = jit_fn.signature.replace(parameters=sig_params)

        # 2. build a fresh binder
        # this is dynamic_func that maps user launch args -> positional tuple
        new_binder = create_function_from_signature(
                        jit_fn.signature,
                        jit_fn.params,
                        backend)

        if VERBOSE: print("new_binder: ", new_binder)
        return new_binder


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

        if USE_RAISED:

            if not raised_py_path:
                raised_py_path = raise_to_triton_lang(bwd_fp)
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

        else:

            # runtime override was provided but USE_RAISED is False, fail early with a clear message
            if raised_py_path:
                raise RuntimeError("autodiff_overwrite_fp provided but USE_RAISED=False; enable USE_RAISED to load a .py override")

            # CompiledKernel
            bwd_compiled_kernel = compile_kernel(
                f"generated/{dir_name}/out.ttir",
                target=target,
                # keep original CompiledKernel.options to preserve same PTX flavour
                # options={k: compile_dict[k] for k in BACKEND_OPTS if k in compile_dict}
            )

            # todo: seems to automatically lowered to ttgir not ttir
            # if VERBOSE: print(bwd_compiled_kernel.asm.keys())

            # 5) keep original fwd CompiledKernel with autograd.Function and add
            # corresponding cache entry (with differentiated CompiledKernel) to
            # cache of bwd JITFunction

            # 5.1. remove constexpr from: key, signature, params
            new_key, num_const_args = remove_constexpr(jit_fn, bwd_jit_fn, key)

            # 5.2. add new args to: key, signature, params
            # add new args to key
            new_key = key_add_args(new_key)

            # fwd_compiled_kernel has constexprs (in its signature) while bwd_compiled_kernel does not
            num_fwd_args = len(fwd_compiled_kernel.src.signature) - num_const_args
            num_bwd_args = len(bwd_compiled_kernel.src.signature)
            num_added_args = num_bwd_args - num_fwd_args

            new_binder = rebuild_binder(bwd_jit_fn, num_added_args, backend)

            # 5.3. add to bwd CompiledKernel into the cache

            # keep forward cache entry as is, don't delete it

            bwd_kernel_cache[new_key] = bwd_compiled_kernel
            if VERBOSE: print("bwd_kernel_cache[new_key]: ", bwd_kernel_cache[new_key])
            bwd_jit_fn.device_caches[device] = (bwd_kernel_cache, target, backend, new_binder)

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


# it's not as much as a stub, but more like helper to wrap_bwd_kernel
# from kernel_inputs -- the true stub is the user thing, this thing
# just piggy backs on the true stub
def wrap_bwd_kernel(fwd_kernel, bwd_kernel, idxs_buffers, grid, kernel_inputs, all_upstream):

    # # todo: these were injected automatically (launch‑options)
    # kwargs.pop("debug", None)
    # kwargs.pop("num_warps", None)
    # kwargs.pop("num_ctas", None)

    # if VERBOSE:  print("[wrap_bwd_kernel] kernel_inputs", kernel_inputs)

    # fwd specializes away some arguments (so that they aren't
    # arguments in the fwd TTIR, and thus not arguments in bwd
    # TTIR as well) -- so don't pass them to bwd TTIR
    idx_folded = bwd_kernel._autodiff_info[-1]
    if VERBOSE:
        print("[wrap_bwd_kernel] bwd_kernel._autodiff_info: ", bwd_kernel._autodiff_info)
        print("[wrap_bwd_kernel] idx_folded: ", idx_folded)

    # user provided idx of outputs (idx_folded) in terms of all args to fwd python kernel
    # (JITFunction), when it compiled, some of the args potentially got specialized away.
    # Thus here need to shift that user specified index to account for these args (that got
    # specialized away) if they were located before the user provided idx_upstream

    # reverse to prevent shifting issues when popping
    for i in reversed(sorted(idx_folded)):
        kernel_inputs.pop(i)

    # for upstream each idx, shift it by how many args
    # before it has been folded

    # sort because usr can pass idxs in arbitrary order
    idxs_buffers = reversed(sorted(idxs_buffers))

    shifted_idxs_buffers = []
    for idx in idxs_buffers:
        num_folded_before = sum(x < idx for x in idx_folded)
        idx_shifted = idx - num_folded_before
        shifted_idxs_buffers.append(idx_shifted)
        print(f"output-buffer at idx {idx} was shifted by {num_folded_before}")

    # pass (from the AG.bwd inputs) upstream
    # grads wrt all (not just one) outputs
    bwd_args = []
    for i, arg in enumerate(kernel_inputs):
        # grad wrt an output -- fill with upstream
        if i in shifted_idxs_buffers:
            # clone bc torch's AG.Func contract is "NEVER to modify these in-place"
            # https://docs.pytorch.org/docs/stable/notes/extending.html;
            # Otherwise doubles grads when later executing torch_fn impls
            bwd_args.append(all_upstream[i].clone())
            continue
        # grad wrt an input -- fill with zeros
        if isinstance(arg, torch.Tensor):
            bwd_args.append(torch.zeros_like(arg))

    # todo:
    #  some err handling for weird cases where fwd JITFcuntio has't ran
    #  with that signature yet -- so my wrapping didnt' take place -- so
    #  calling the bellow will fail

    # if VERBOSE: print("[wrap_bwd_kernel] fwd_args:", kernel_inputs)
    if VERBOSE: print("[wrap_bwd_kernel] bwd_args:", bwd_args)


    kernel_impl = bwd_kernel._raised if USE_RAISED else bwd_kernel
    kernel_impl[grid](*kernel_inputs, *bwd_args)
    # bwd_kernel.run(grid=grid, warmup=False, *kernel_inputs, *bwd_args)

    if VERBOSE: print("[wrap_bwd_kernel] bwd_args (after calling bwd_kernel): ", bwd_args)

    return (*bwd_args,)


class DifferentiatedCompiledKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, kernels, idxs_buffers, grid, *fwd_kernel_inputs):
        if VERBOSE: print("\n"*3, "Op.forward")

        fwd_kernel, wrapped_bwd_kernel = kernels

        # todo:
        # bc calling the fwd kernel will write output inplace of output buffers -- but my bwd expects a cleanly initialized buffers
        #  - clone them and store on the ctx BEFORE they got overwritten by calling the fwd kernel?
        #  - even better: on cpp side remove the entire fwd subgraph leading to the output?
        #    Bc fwd result which I re-compute during bwd will not be used anywhere anyway

        # if VERBOSE: print("[Op.forward] fwd_kernel_inputs (before calling kernel)", *fwd_kernel_inputs)
        fwd_kernel[grid](*fwd_kernel_inputs)
        # if VERBOSE: print("[Op.forward] fwd_kernel_inputs (after calling kernel)", *fwd_kernel_inputs)

        ctx.mark_dirty(*[fwd_kernel_inputs[i] for i in idxs_buffers])

        # ugly workaround because save_for_backward only works for tensor inputs
        tensor_fwd_kernel_inputs = [a for a in fwd_kernel_inputs if isinstance(a, torch.Tensor)]
        ctx.save_for_backward(*tensor_fwd_kernel_inputs)
        ctx.non_tensor_inputs = [a for a in fwd_kernel_inputs if not isinstance(a, torch.Tensor)]
        ctx.arg_types = [isinstance(a, torch.Tensor) for a in fwd_kernel_inputs]

        ctx.wrapped_bwd_kernel = wrapped_bwd_kernel
        ctx.grid = grid

        return (*tensor_fwd_kernel_inputs, )


    @staticmethod
    def backward(ctx, *all_upstream):
        if VERBOSE: print("\n"*3, "Op.backward")

        # AG.bwd expects same num of args (upstream grads) as the outputs of AG.fwd;
        # because in AG.fwd I'm returning ALL the kernel tensor args, here
        # I need to input grads wrt ALL OF THEM -- not just grad wrt the single
        # output returned from the user stub;

        # reconstruct all fwd kernel args
        fwd_kernel_inputs = []
        tensor_idx = 0
        non_tensor_idx = 0
        for is_tensor in ctx.arg_types:
            if is_tensor:
                fwd_kernel_inputs.append(ctx.saved_tensors[tensor_idx])
                tensor_idx += 1
            else:
                fwd_kernel_inputs.append(ctx.non_tensor_inputs[non_tensor_idx])
                non_tensor_idx += 1

        # todo-high:
        #  note output buffers (in this case at idx=1) have my grad_fn
        #  attached -- but I'll be runnign kernel on them again: undesirable?
        # if VERBOSE: print("[Op.backward] reconstructed fwd_kernel_inputs", fwd_kernel_inputs)
        # > tensor([...], requires_grad=True),
        # > tensor([...], grad_fn=<DifferentiatedCompiledKernelBackward>)]

        # wrapped_bwd_kernel returns grad_inputs that it initialized, after running the
        # generated_bwd_kernel these are populated -- and contain grads wrt original tensor inputs
        grads = ctx.wrapped_bwd_kernel(ctx.grid, fwd_kernel_inputs, all_upstream)
        if VERBOSE: print("[Op.backward] grads", grads)

        all_outs = []
        tensor_idx = 0
        for i, is_tensor in enumerate(ctx.arg_types):
            if is_tensor:
                all_outs.append(grads[tensor_idx])
                tensor_idx += 1
            else:
                all_outs.append(None)

        return (None, None, None, *all_outs, )


def helper(spec, kernels, idxs):
    # returns a subclass of `DifferentiatedCompiledKernel` whose
    # .apply accepts keywords;
    # its signature staying generic (ctx, *args, **kwargs) -- is
    # fine as long as it can consume the ordered list I pass in

    target_sig = inspect.signature(spec)
    params = target_sig.parameters.values()
    if VERBOSE: print("[helper] target_sig", target_sig)
    if VERBOSE: print("[helper] params", params)

    class _Helper(DifferentiatedCompiledKernel):
        # __signature__ = target_sig          # IDE/help friendly
        __doc__       = DifferentiatedCompiledKernel.__doc__
        __name__      = DifferentiatedCompiledKernel.__name__
        __qualname__  = DifferentiatedCompiledKernel.__qualname__

        @classmethod
        def apply(cls, *args, **kwargs):
            if VERBOSE:
                # print("[_Helper.apply] args", args)
                print("[_Helper.apply] kwargs", kwargs)
                print("[_Helper.apply] grid", cls.grid)
            bound = target_sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            # fixed positional order for c++ apply
            ordered = [bound.arguments[p.name] for p in params]
            return super().apply(kernels, idxs, cls.grid, *ordered)

        @classmethod
        def __class_getitem__(cls, grid):
            # mutate the existing class
            cls.grid = grid
            return cls.apply

    return _Helper



def create_new_jitfn(jit_func):
    assert isinstance(jit_func, JITFunction)

    # create a new JITFunction with the same base py-fn and parameters;
    # new object, empty cache -- each JITFunction instance owns its own device_caches dict
    new = JITFunction(
        jit_func.fn,
        version=jit_func.version,
        do_not_specialize=jit_func.do_not_specialize,
        do_not_specialize_on_alignment=jit_func.do_not_specialize_on_alignment,
        debug=jit_func.debug,
        noinline=jit_func.noinline,
        repr=jit_func._repr,
        launch_metadata=jit_func.launch_metadata
    )

    new._autodiff_info = []

    # copy any pre-run hooks
    new.pre_run_hooks = list(jit_func.pre_run_hooks)

    return new


# todo-low: can idxs_buffers determine automatically:
#   - in AG.fwd -- run kernel once and see which inputs were changed as result of executing kernel;
#   - or, in mlir pass output idx of all inputs which are used in store nodes
def autodiff(idxs_buffers): # , overwrite_fp=None

    def inner(fwd_kernel):

        nonlocal idxs_buffers
        assert isinstance(idxs_buffers, (tuple, int)), f"idxs_buffers must be either tuple or int, got {type(idxs_buffers)}"
        # make the most common case slightly more convenient for usr
        if isinstance(idxs_buffers, int):
            idxs_buffers = (idxs_buffers, )

        bwd_kernel = create_new_jitfn(fwd_kernel)

        # optionally, allows to overwrite backward kernel with a kernel stored at the provided file pointer
        # if overwrite_fp=None, generates a new backward kernel and uses it
        # else just "load_raised_jit" from that path
        #
        # not setting it here becuase it's not user who sets it but rather I do it,
        # and execution-order wise, I do it after the user facing autodiff function has ran
        # so no use to set it here
        # fwd_kernel.overwrite_fp = overwrite_fp

        # allows to associate a bwd JITFcuntion with this specific fwdKernel
        # so that, from inside the compile hook (which will be triggered on the fwd JITFunciton)
        # I can install the bwd CompiledKernel **on the backward JITFcuntion** (NOT fwd JITFcuntion)
        fwd_kernel._bwd_kernel = bwd_kernel

        # I'm registering compile hook on all JITFunction[s] (both forward and backward);
        # this flag is needed to be able to early exit from the hook (avoids triggering
        # the autograd machinery on already differentiated kernels)
        fwd_kernel._is_fwd_kernel = True

        wrapped_bwd_kernel = partial(wrap_bwd_kernel, fwd_kernel, bwd_kernel, idxs_buffers)
        kernels = (fwd_kernel, wrapped_bwd_kernel)

        # takes the signature form fwd_kernel's python fn
        op = helper(fwd_kernel.fn, kernels, idxs_buffers)
        return op

    return inner
