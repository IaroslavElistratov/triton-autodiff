# python -m kernel_agent.tools.gradcheck.debug --file-path /root/triton-autodiff/third_party/autodiff/python/kernel_agent/test/matmul.py --verbose

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Tuple

import torch

from kernel_agent.utils import compile_kernel as create_op
from kernel_agent.tools.gradcheck.core import check_op_backward_parity


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

    make_args = ns.get("make_args")
    if not callable(make_args):
        raise RuntimeError("User kernel must define make_args(dims) -> (args, kwargs)")
    sweep = ns.get("SWEEP")
    dims = sweep[0] if isinstance(sweep, (list, tuple)) and sweep else {}
    args_tuple, _kwargs = make_args(dims)

    if args.verbose:
        shapes = tuple(getattr(t, "shape", None) for t in args_tuple)
        print(f"[gradcheck] Inputs dims={dims}, shapes={shapes}")

    ok, stats = check_op_backward_parity(
        ref_fwd=ns["torch_fn"],
        my_op=op,
        inputs=args_tuple,
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


