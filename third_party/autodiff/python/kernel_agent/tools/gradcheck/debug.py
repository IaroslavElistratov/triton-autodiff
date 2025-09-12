# python -m kernel_agent.tools.gradcheck.debug --file-path /root/triton-autodiff/third_party/autodiff/python/kernel_agent/test/matmul.py --verbose

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Tuple

import torch

from kernel_agent.utils import compile_kernel as create_op
from kernel_agent.tools.gradcheck.core import check_op_backward_parity_sweep


def main() -> None:
    ap = argparse.ArgumentParser("kernel-agent-gradcheck")
    ap.add_argument("--file-path", metavar="FILE", type=str, required=True,
                    help="Path to the forward kernel file (e.g. test/matmul.py)")
    ap.add_argument("--verbose", action="store_true", help="Print detailed logs")
    args = ap.parse_args()

    fwd_fp: str = args.file_path
    if not os.path.isfile(fwd_fp):
        raise FileNotFoundError(f"forward file not found: {fwd_fp}")

    if args.verbose:
        print("[gradcheck] Forward file:", fwd_fp)
        print("[gradcheck] Compiling and tracing user kernel via create_op(...)")

    op, bwd_fp, ns = create_op(fwd_fp, overwrite_fp=None)

    if args.verbose:
        print("[gradcheck] Backward path:", bwd_fp)

    if not callable(ns.get("make_args")) or not isinstance(ns.get("SWEEP"), (list, tuple)):
        raise RuntimeError("User kernel must define make_args and SWEEP")
    if args.verbose and ns.get("SWEEP"):
        dims0 = ns["SWEEP"][0]
        try:
            args0, _ = ns["make_args"](dims0)
            shapes = tuple(getattr(t, "shape", None) for t in args0)
        except Exception:
            shapes = ()
        print(f"[gradcheck] Inputs dims={dims0}, shapes={shapes}")

    ok, stats = check_op_backward_parity_sweep(
        ref_fwd=ns["torch_fn"],
        my_op=op,
        make_args=ns["make_args"],
        sweep=ns["SWEEP"],
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


