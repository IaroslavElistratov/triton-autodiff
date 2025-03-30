#!/usr/bin/env python3

import sys
import subprocess


dir = "/home/iaro/Desktop/my_triton_autodiff/working_files/triton"
tool = f"{dir}/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt"


for test_name in ["add", "add-mul", "div", "add-mul-div", "math-ops",
                  "multiblock_add-mul", "mask_multiblock_add-mul",
                  "2d_dot"]:
  test_dir = f"{dir}/third_party/autodiff/test/{test_name}"
  print("~" * 20 + f" Running {test_name} " + "~" * 20)
  with open(f"{test_dir}/out.ttir", "w") as f:
    subprocess.run([tool, "--convert-triton-to-autodiff", f"{test_dir}/inp.ttir"], stdout=f)
  subprocess.run([sys.executable, f"{test_dir}/run_bwd.py"], cwd=test_dir)
print("~" * 50)