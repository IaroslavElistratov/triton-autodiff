# python -m kernel_agent.tools.gradcheck.debug --file-path /root/triton-autodiff/third_party/autodiff/python/kernel_agent/test/attention.py

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Tuple

import torch

from kernel_agent.utils import compile_kernel as create_op
from api import autodiff_overwrite_fp
from kernel_agent.tools.gradcheck.core import check_op_backward_parity_sweep


def main() -> None:
    ap = argparse.ArgumentParser("kernel-agent-gradcheck")
    ap.add_argument("--file-path", metavar="FILE", type=str, required=True,
                    help="Path to the forward kernel file (e.g. test/matmul.py)")
    args = ap.parse_args()

    fwd_fp: str = args.file_path
    if not os.path.isfile(fwd_fp):
        raise FileNotFoundError(f"forward file not found: {fwd_fp}")


    # os.environ["VERBOSE"] = "1"

    print("[gradcheck] Forward file:", fwd_fp)
    print("[gradcheck] Compiling and tracing user kernel via create_op(...)")

    op, bwd_fp, ns = create_op(fwd_fp, overwrite_fp=None)

    # see detailed comment in api.py
    with autodiff_overwrite_fp(bwd_fp):

        ok, stats = check_op_backward_parity_sweep(
            ref_fwd=ns["torch_fn"],
            my_op=op,
            sidecar=ns,
            outputs="auto",
            # atol=0.1,
            # rtol=0.04,

            # atol=0.0015,
            # rtol=0.04,

            atol=0.07,
            rtol=0.02,
        )

    # Print result
    print("ok:", ok)
    print("stats:", json.dumps(stats, default=str))


if __name__ == "__main__":
    main()


