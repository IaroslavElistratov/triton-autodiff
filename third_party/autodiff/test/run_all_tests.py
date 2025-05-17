#!/usr/bin/env python3

import sys
import subprocess

if __name__ == "__main__":

  for test_name in ["add", "add-mul", "div", "add-mul-div", "math-ops",
                    "multiblock_add-mul", "mask_multiblock_add-mul",
                    "2d_dot", "multi-use",
                    "for-loop", "for-loop-mm_static-bounds",
                    "LOCAL_for-loop-mm-static-bounds_simple", "LOCAL_for-loop-mm-static-bounds_actual",
                    "flash_attention_v2", "layernorm"]:
  # for test_name in ["flash_attention_v2"]:

    print("~" * 20 + f" Running {test_name} " + "~" * 20)

    # produce fwd and bwd ttir; and compare outs and grads with torch
    subprocess.run([sys.executable, "run.py"], cwd=test_name)

  print("~" * 50)
