from __future__ import annotations
import argparse
from .config import OptimizeConfig
from .toolbox import Toolbox
from .llm import LLMClient
from .orchestrator import KernelOptimizer

# TODO: provide actual tool implementation
from user_tools import (  # type: ignore
    get_user_forwrad, naive_grad, gradient_check, benchmark, profile, get_user_dvice_info,
    apply_patch, # oss tool
)

def main() -> None:
    p = argparse.ArgumentParser("kernel-agent")
    p.add_argument("--record-root", default="artifacts")
    p.add_argument("--max-iters", type=int, default=8)
    p.add_argument("--min-rel-impr", type=float, default=0.02)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--target-throughput", type=float, default=0.0)
    p.add_argument("--time-budget-s", type=int, default=0)
    p.add_argument("--model", default="gpt-4.1-2025-04-14")
    args = p.parse_args()

    cfg = OptimizeConfig(
        record_root=args.record_root,
        max_iters=args.max_iters,
        min_rel_improvement=args.min_rel_impr,
        patience=args.patience,
        target_throughput=(args.target_throughput or None),
        time_budget_s=(args.time_budget_s or None),
    )

    tb = Toolbox(
        get_user_forwrad=get_user_forwrad,
        naive_grad=naive_grad,
        gradient_check=gradient_check,
        benchmark=benchmark,
        profile=profile,
        get_user_dvice_info=get_user_dvice_info,
        apply_patch=apply_patch,
    )
    llm = LLMClient(model=args.model)
    m = KernelOptimizer(cfg, tb, llm).run()

    print("Run:", m.run_id)
    print("Best metrics:", m.best_metrics)
    print("Manifest:", m.save(cfg.record_root))

if __name__ == "__main__":
    main()
