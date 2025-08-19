from __future__ import annotations
import argparse

from .orchestrator import KernelOptimizer, Config
from .llm import MinimalLLMPatchProvider

# ---- Your black-box functions (already implemented elsewhere) ---------------
from user_tools import (  # type: ignore
    get_user_forwrad,      # NOTE: name preserved as provided
    naive_grad,
    gradient_check,
    benchmark,
    profile,
    get_user_dvice_info,   # NOTE: name preserved as provided
)

def main() -> None:
    ap = argparse.ArgumentParser("kernel-agent-min")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-rel-impr", type=float, default=0.02)
    ap.add_argument("--model", default="gpt-oss-20b")
    args = ap.parse_args()

    cfg = Config(max_iters=args.max_iters, patience=args.patience,
                 min_rel_improvement=args.min_rel_impr)
    llm = MinimalLLMPatchProvider(model=args.model)
    agent = KernelOptimizer(cfg, llm)

    out = agent.run(
        get_user_forwrad=get_user_forwrad,
        naive_grad=naive_grad,
        gradient_check=gradient_check,
        benchmark=benchmark,
        profile=profile,
        get_user_dvice_info=get_user_dvice_info,
    )
    print("Best metrics:", out["best_metrics"])
    print("Best backward kernel:", out["best_backward_fp"])
    print("Device:", out["device_info"])

if __name__ == "__main__":
    main()