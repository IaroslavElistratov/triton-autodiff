import sys
import hashlib
import subprocess
import os
import shutil
os.environ['TRITON_ALWAYS_COMPILE'] = '1'

import triton
# from triton.compiler import compile
# from triton.backends.compiler import GPUTarget
# from triton.runtime.jit import JITFunction
from typing import Callable, Optional, Any


VERBOSE = int(os.environ.get('VERBOSE', 0))
assert VERBOSE in [0, 1, 2]

dir = os.getenv("TRITON_AUTODIFF_DIR")
if dir is None:
    raise ValueError("Please specify TRITON_AUTODIFF_DIR, see README.")

# todo: don't hardcode
tool = f"{dir}/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt"

# def _locate_triton_opt(base_dir: str) -> str:
#     """
#     Locate `triton-opt`:
#       1) Use TRITON_OPT_BIN if set and executable.
#       2) Search under {base_dir}/build/**/triton-opt.
#       3) Fallback to PATH.
#     """
#     env_path = os.getenv("TRITON_OPT_BIN")
#     if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
#         return env_path

#     search_root = os.path.join(base_dir, "build")
#     if os.path.isdir(search_root):
#         for root, _dirs, files in os.walk(search_root):
#             if "triton-opt" in files:
#                 return os.path.join(root, "triton-opt")

#     which_path = shutil.which("triton-opt")
#     if which_path:
#         return which_path

#     raise FileNotFoundError(
#         "Could not find `triton-opt`. Set TRITON_OPT_BIN to its full path or ensure it is on PATH."
#     )


# tool = _locate_triton_opt(dir)

def run_mlir_pass(path):

  os.makedirs(path, exist_ok=True)

  # produce bwd ttir
  with open(f"{path}/out.ttir", "w") as f:
    subprocess.run([tool, "--convert-triton-to-autodiff", "--mlir-print-debuginfo", f"{path}/inp.ttir"], stdout=f, check=True)

  if VERBOSE >= 1:
    # optionally, produce readable fwd ttir

    # ugly, needed bc fwd.py files create out.ttir files with default SSA names (%1, %2, ...)
    # and with location info (containing variable names). Here I ran "--mlir-use-nameloc-as-prefix"
    # on it and write to the same files to avoid creating redundant files
    with open(f"{path}/inp.ttir", "r+") as f:
        content = f.read()         # Read existing content
        f.seek(0)                  # Move cursor to the beginning
        # Overwrite from the start
        subprocess.run([tool, "--mlir-use-nameloc-as-prefix", "--mlir-print-debuginfo", f"{path}/inp.ttir"], stdout=f, check=True)
        f.truncate()               # Remove remaining old content

    # if VERBOSE == 2:

    #   def draw_dot(path, mode):
    #     assert mode in ["fwd", "bwd"]

    #     vis_dir = path + "/vis"
    #     os.makedirs(vis_dir, exist_ok=True)

    #     # a. optionally, produce vis dot
    #     with open(f"{vis_dir}/{mode}.dot", "w") as f:
    #       ttir_path = f"{path}/inp.ttir" if mode == "fwd" else f"{path}/out.ttir"
    #       subprocess.run([tool, "-mlir-use-nameloc-as-prefix", "--view-op-graph", ttir_path], stderr=f,
    #                     # suppress stdout, otherwise prints _inp_readable again
    #                     stdout=subprocess.DEVNULL)

    #     with open(f"{vis_dir}/{mode}.svg", "w") as f:
    #       subprocess.run(["dot", "-Tsvg", f"{vis_dir}/{mode}.dot"], stdout=f)

    #     # b. optionally cluster nodes
    #     subprocess.run(["python", "cluster_dot.py", "--strict", f"{vis_dir}/{mode}.dot", f"{vis_dir}/{mode}_grouped.dot"])

    #     with open(f"{vis_dir}/{mode}_grouped.svg", "w") as f:
    #       subprocess.run(["dot", "-Tsvg", f"{vis_dir}/{mode}_grouped.dot"], stdout=f)

    #     # os.remove(f"{vis_dir}/{mode}.dot")
    #     # os.remove(f"{vis_dir}/{mode}_grouped.dot")

    #   # optionally, produce vis dot
    #   draw_dot(path, mode="fwd")

    #   # optionally, produce vis dot
    #   draw_dot(path, mode="bwd")



# def create_new_jitfn(jit_func):
#     assert isinstance(jit_func, JITFunction)

#     # create a new JITFunction with the same base py-fn and parameters;
#     # new object, empty cache -- each JITFunction instance owns its own device_caches dict
#     new = JITFunction(
#         jit_func.fn,
#         version=jit_func.version,
#         do_not_specialize=jit_func.do_not_specialize,
#         do_not_specialize_on_alignment=jit_func.do_not_specialize_on_alignment,
#         debug=jit_func.debug,
#         noinline=jit_func.noinline,
#         repr=jit_func._repr,
#         launch_metadata=jit_func.launch_metadata
#     )

#     return new


# def generate_naive_backward(key, repr, fn, compile, is_manual_warmup, already_compiled):

#     # 1) extract fwd_compiled_kernel

#     # get the kernel using the same key
#     fwd_compiled_kernel = 

#     # 2) write fwd IR

#     # todo: cleanup
#     # the "key" arg is just a python string with input signatures of the kernel
#     # but I want some folder name -- one way is to hash it
#     hash_object = hashlib.sha256(key.encode())
#     dir_name = hash_object.hexdigest()[:10]
#     if VERBOSE: print("dir_name: ", dir_name)

#     os.makedirs(f"generated/{dir_name}", exist_ok=True)
#     with open(f"generated/{dir_name}/inp.ttir", "w") as f:
#       f.write(fwd_compiled_kernel.asm['ttir'])

#     # 3) autodiff
#     run_mlir_pass(f"generated/{dir_name}")

#     # # 4) create callable python fn for bwd
#     # # CompiledKernel(
#     #     f"generated/{dir_name}/out.ttir",
#     #     target=target,
#     #     # keep original CompiledKernel.options to preserve same PTX flavour
#     #     # options={k: compile_dict[k] for k in BACKEND_OPTS if k in compile_dict}
#     # )



def generate_naive_backward(compiled_kernel: Any) -> str:
  """
  Given a compiled Triton kernel object (duck-typed: has asm dict with 'ttir'),
  run the autodiff pass and return the generated backward TTIR as a string.

  Expected usage in user stub:
    _compiled_kernel = my_kernel[grid](...)
    return out, _compiled_kernel  # or just return _compiled_kernel
  """
  try:
    ttir_text = compiled_kernel.asm['ttir']
  except Exception as e:
    raise TypeError(
      "Invalid compiled_kernel: missing asm['ttir'].\n"
      "Fix: Return the object produced by a Triton kernel launch, e.g. `_compiled_kernel = my_kernel[grid](...)`."
    ) from e
  hash_object = hashlib.sha256(ttir_text.encode())
  dir_name = hash_object.hexdigest()[:10]

  if VERBOSE: print("dir_name: ", dir_name)

  # 2) write fwd IR
  os.makedirs(f"generated/{dir_name}", exist_ok=True)
  with open(f"generated/{dir_name}/inp.ttir", "w") as f:
    f.write(ttir_text)

  # 3) autodiff -> writes out.ttir
  run_mlir_pass(f"generated/{dir_name}")

  # Return the generated backward TTIR as a string
  out_path = f"generated/{dir_name}/out.ttir"
  with open(out_path, "r") as f:
    return f.read()






"""
# ir_code =

# IRSource (created inside triton.compile) uses extension of the file to figure
# out to call "ir.parse_mlir_module(self.path, context)""

# with open("third_party/autodiff/test/kernel.ttir", "w") as f:
#     f.write(ir_code)

# Create a GPU target
target = GPUTarget("cuda", arch=89, warp_size=32)

# Compile the IR
add_bwd_kernel = compile("third_party/autodiff/test/out.ttir", target=target)

# The IRSource class handles parsing the IR file and setting up the compilation pipeline
# The rest of the compilation process (from IR to PTX to cubin) remains the same as the normal workflow
def add_bwd(upstream, BLOCK_SIZE=4):
    # ...
    _compiled_kernel = add_bwd_kernel[grid](a_grad, upstream) # BLOCK_SIZE
    return a_grad, _compiled_kernel

a_grad, _compiled_kernel = add_bwd(upstream)
"""


"""
def stub(a, b):
    ...
    _compiled_kernel = kernel[grid](a, b, output) # BLOCK_SIZE=4
    return output, _compiled_kernel

output_triton, _compiled_kernel = stub(a, b)

with open("inp.ttir", "w") as f:
    f.write(_compiled_kernel.asm['ttir'])
"""
