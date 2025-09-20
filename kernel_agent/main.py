# python -m pip install -e kernel_agent
# cd /root/triton-autodiff
# export TRITON_AUTODIFF_DIR=$(pwd)/triton_autodiff
# kernel-agent --backend triton --checkpoint /workspace/gpt-oss/gpt-oss-20b/original/ --file-path kernel_agent/test/matmul.py --reasoning-effort medium --mode phased > kernel_agent/LOGS/out.txt
# kernel-agent --backend triton --checkpoint /workspace/gpt-oss/gpt-oss-120b/original/ --file-path kernel_agent/test/attention.py --reasoning-effort medium --mode phased > kernel_agent/LOGS/out.txt


from __future__ import annotations
import argparse
import os

from .orchestrator import KernelOptimizer, Config
# from .llm import MinimalLLMPatchProvider


def main() -> None:
    ap = argparse.ArgumentParser("kernel-agent")
    ap.add_argument("--max-iters", type=int, default=32)
    ap.add_argument("--patience_perf_stop", type=int, default=4)
    ap.add_argument("--patience_parity_restore", type=int, default=4)
    ap.add_argument("--min-rel-impr", type=float, default=0.10)
    ap.add_argument("--file-path", metavar="FILE", type=str, required=True, help="Path to the forward kernel to be optimized")

    ap.add_argument("--backend", type=str, default="stub", choices=["stub", "triton", "torch", "vllm"], help="Inference backend for local sampler")
    ap.add_argument("--checkpoint", metavar="FILE", type=str, help="Path to the SafeTensors checkpoint")

    ap.add_argument("-c", "--context", metavar="CONTEXT", type=int, default=32768, help="Max context length (tokens)")
    ap.add_argument("-r", "--reasoning-effort", metavar="REASONING_EFFORT", type=str, default="high", choices=["high", "medium", "low"], help="Reasoning effort")
    ap.add_argument("--mode", type=str, default="regular", choices=["regular", "phased"], help="Optimization strategy mode")
    args = ap.parse_args()

    # Map selected backend options into environment for the local sampler
    if args.backend:
        os.environ["KERNEL_AGENT_BACKEND"] = args.backend
    if args.checkpoint:
        os.environ["KERNEL_AGENT_CHECKPOINT"] = args.checkpoint
    # Strategy toggle (regular | phased)
    os.environ["KERNEL_AGENT_STRATEGY"] = args.mode

    cfg = Config(max_iters=args.max_iters,
                 patience_perf_stop=args.patience_perf_stop,
                 patience_parity_restore=args.patience_parity_restore,
                 min_rel_improvement=args.min_rel_impr)
    # Defer heavy LLM imports unless we're actually iterating
    if args.max_iters > 0:
        from .llm import MinimalLLMPatchProvider  # lazy import
        llm = MinimalLLMPatchProvider(
            temperature=0.7,
            max_tokens=32768,
            reasoning_effort=args.reasoning_effort,
            context=args.context,
        )
    else:
        class _Noop:
            def propose_patch(self, *_, **__):
                return "*** Begin Patch\n*** End Patch"
        llm = _Noop()
    agent = KernelOptimizer(cfg, llm)

    out = agent.run(
        fwd_fp=args.file_path,
        # benchmark=benchmark,
        # profile=profile,
        # todo:
        get_user_device_info=(lambda: "N/A"),
    )
    print("Best metrics:", out.get("best_metrics", {}))
    print("Best backward file:", out["backward_fp"])
    print("Device:", out["device_info"])
    stop_reason = out.get("stop_reason", "")
    if stop_reason:
        print("Stop reason:", stop_reason)

if __name__ == "__main__":
    main()