#!/usr/bin/env python3

import os
import sys
import subprocess

if __name__ == "__main__":

  tests = [
    "add", "add-mul", "div", "add-mul-div", "math-ops",
    "multiblock_add-mul", "mask_multiblock_add-mul",
    "2d_dot", "2d_dot_interleave", "multi-use", "multi-output_multi-use",
    "for-loop", "for-loop-mm_static-bounds",
    "flash_attention_v2", "layernorm"
  ]

  if int(os.environ.get('EXTRA', 0)):
    tests.extend([
      "LOCAL_for-loop-mm-static-bounds_simple",
      "LOCAL_for-loop-mm-static-bounds_actual"
    ])

  for test_name in tests:
    print("~" * 20 + f" Running {test_name} " + "~" * 20)
    # produce fwd and bwd ttir; and compare outs and grads with torch
    subprocess.run([sys.executable, "run.py"], cwd=test_name)
  print("~" * 50)
