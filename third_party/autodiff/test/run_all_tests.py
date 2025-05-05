#!/usr/bin/env python3

import os
import sys
import subprocess




dir = "/home/iaro/Desktop/my_triton_autodiff/working_files/triton"
tool = f"{dir}/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt"



def draw_dot(mode):
  assert mode in ["fwd", "bwd"]

  vis_dir = test_dir + "/vis"
  os.makedirs(vis_dir, exist_ok=True)

  # a. optionally, produce vis dot
  with open(f"{vis_dir}/{mode}.dot", "w") as f:
    ttir_path = f"{test_dir}/inp.ttir" if mode == "fwd" else f"{test_dir}/out.ttir"
    subprocess.run([tool, "-mlir-use-nameloc-as-prefix", "--view-op-graph", ttir_path], stderr=f,
                  # suppress stdout, otherwise prints _inp_readable again
                  stdout=subprocess.DEVNULL)

  with open(f"{vis_dir}/{mode}.svg", "w") as f:
    subprocess.run(["dot", "-Tsvg", f"{vis_dir}/{mode}.dot"], stdout=f)

  # b. optionally cluster nodes
  subprocess.run(["python", "cluster_dot.py", "--strict", f"{vis_dir}/{mode}.dot", f"{vis_dir}/{mode}_grouped.dot"])

  with open(f"{vis_dir}/{mode}_grouped.svg", "w") as f:
    subprocess.run(["dot", "-Tsvg", f"{vis_dir}/{mode}_grouped.dot"], stdout=f)

  os.remove(f"{vis_dir}/{mode}.dot")
  os.remove(f"{vis_dir}/{mode}_grouped.dot")


# TODO:
# for test_name in ["multi-use-complex"]:
#   - bug introduced by: "[WIP] don't insert grads for later nodes above grads for previous nodes when using same upstream"
#   - Cant find fwd py fn

# todo:
# "for-loop-mm_static-bounds_interleave-mul"
#   - Cant find fwd py fn


# for test_name in ["add"]:

# WORKS:
for test_name in ["add", "add-mul", "div", "add-mul-div", "math-ops",
                  "multiblock_add-mul", "mask_multiblock_add-mul",
                  "2d_dot", "multi-use",
                  "for-loop", "for-loop-mm_static-bounds",
                  "LOCAL_for-loop-mm-static-bounds_simple", "LOCAL_for-loop-mm-static-bounds_actual"]:

# GRAD WRONG:
#   "layernorm"
#   "flash_attention_v2"


  test_dir = f"{dir}/third_party/autodiff/test/{test_name}"
  print("~" * 20 + f" Running {test_name} " + "~" * 20)

  # 1. produce fwd ttir
  subprocess.run([sys.executable, f"{test_dir}/run_fwd.py"], cwd=test_dir)

  # 2. optionally, produce readable fwd ttir

  # with open(f"{test_dir}/_inp_readable.ttir", "w") as f:
  #   subprocess.run([tool, "--mlir-use-nameloc-as-prefix", "--mlir-print-debuginfo", f"{test_dir}/inp.ttir"], stdout=f)

  # this is a bit ugly but needed bc run_fwd.py files create out.ttir files with default SSA names (%1, %2, ...)
  # and with location info (containing variable names). Here I ran "--mlir-use-nameloc-as-prefix" on it and write
  # to the same files to avoid creating redundant files
  with open(f"{test_dir}/inp.ttir", "r+") as f:
      content = f.read()         # Read existing content
      f.seek(0)                  # Move cursor to the beginning
      # Overwrite from the start
      subprocess.run([tool, "--mlir-use-nameloc-as-prefix", "--mlir-print-debuginfo", f"{test_dir}/inp.ttir"], stdout=f)
      f.truncate()               # Remove remaining old content

  # 3. optionally, produce vis dot
  draw_dot(mode="fwd")

  # 4. produce bwd ttir
  with open(f"{test_dir}/out.ttir", "w") as f:
    subprocess.run([tool, "--convert-triton-to-autodiff", "--mlir-print-debuginfo", f"{test_dir}/inp.ttir"], stdout=f)

  # 5. run bwd ttir
  subprocess.run([sys.executable, f"{test_dir}/run_bwd.py"], cwd=test_dir)

  # 6. optionally, produce vis dot
  draw_dot(mode="bwd")

print("~" * 50)