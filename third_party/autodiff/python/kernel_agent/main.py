from __future__ import annotations
import argparse
import os

from .orchestrator import KernelOptimizer, Config
from .llm import MinimalLLMPatchProvider


from .tools import (  # type: ignore
    naive_grad,
    # gradient_check,
    # benchmark,
    # profile,
    # get_user_dvice_info,
)

def main() -> None:
    ap = argparse.ArgumentParser("kernel-agent")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min-rel-impr", type=float, default=0.02)
    ap.add_argument("--file-path", metavar="FILE", type=str, default="", help="Path to the forward kernel to be optimized")

    ap.add_argument("--backend", type=str, default="triton", choices=["triton", "torch", "vllm"], help="Inference backend for local sampler")
    ap.add_argument("--checkpoint", metavar="FILE", type=str, default="", help="Path to the SafeTensors checkpoint")

    ap.add_argument("-c", "--context", metavar="CONTEXT", type=int, default=32768, help="Max context length (tokens)")
    ap.add_argument("-r", "--reasoning-effort", metavar="REASONING_EFFORT", type=str, default="high", choices=["high", "medium", "low"], help="Reasoning effort")
    args = ap.parse_args()

    # Map selected backend options into environment for the local sampler
    if args.backend:
        os.environ["KERNEL_AGENT_BACKEND"] = args.backend
    if args.checkpoint:
        os.environ["KERNEL_AGENT_CHECKPOINT"] = args.checkpoint

    cfg = Config(max_iters=args.max_iters, patience=args.patience,
                 min_rel_improvement=args.min_rel_impr)
    llm = MinimalLLMPatchProvider(
        temperature=0.7,
        # todo-low: rm
        max_tokens=1536,
        reasoning_effort=args.reasoning_effort,
        context=args.context,
    )
    agent = KernelOptimizer(cfg, llm)

    out = agent.run(
        fwd_fp=args.file_path,
        naive_grad=naive_grad,
        # gradient_check=gradient_check,
        # benchmark=benchmark,
        # profile=profile,
        # todo:
        get_user_dvice_info=(lambda: "N/A"),
    )
    print("Best metrics:", out.get("best_metrics", {}))
    print("Best backward kernel:", out["best_backward_fp"])
    print("Device:", out["device_info"])

if __name__ == "__main__":
    main()