import sys
import os
import builtins
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


# True: raises backward TTIR to trtion-lang; and emits stub
# False: uses backward TTIR directly
USE_LEGACY_API = False

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
def autodiff(idxs_buffers, stub_name=None): # , overwrite_fp=None

    def inner(fwd_kernel):

        nonlocal idxs_buffers, stub_name
        assert isinstance(idxs_buffers, (tuple, int)), f"idxs_buffers must be either tuple or int, got {type(idxs_buffers)}"
        # make the most common case slightly more convenient for usr
        if isinstance(idxs_buffers, int):
            idxs_buffers = (idxs_buffers, )

        # for bwd stub gen
        fwd_kernel.idxs_buffers = idxs_buffers

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

        if not USE_LEGACY_API:
            # in this case do not want to overwrite user kernel with DifferentiatedCompiledKernel
            # becuase instead i'm overwritting their stub (not kernel) with StubOverrideDCK
            # and StubOverrideDCK.forward calls the original unchanged user fwd
            return fwd_kernel
        else:
            wrapped_bwd_kernel = partial(wrap_bwd_kernel, fwd_kernel, bwd_kernel, idxs_buffers)
            kernels = (fwd_kernel, wrapped_bwd_kernel)

            # takes the signature form fwd_kernel's python fn
            op = helper(fwd_kernel.fn, kernels, idxs_buffers)
            return op

    return inner

