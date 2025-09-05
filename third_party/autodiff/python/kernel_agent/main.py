# python -m pip install -e /root/triton-autodiff/third_party/autodiff/python
# cd /root/triton-autodiff/third_party/autodiff/python/kernel_agent
# export KERNEL_AGENT_VERBOSE=1 KERNEL_AGENT_CAPTURE_THINKING=1 KERNEL_AGENT_STREAM=1 TRITON_AUTODIFF_DIR=/root/triton-autodiff
# kernel-agent --backend triton --checkpoint /workspace/gpt-oss/gpt-oss-20b/original/ --file-path /root/triton-autodiff/third_party/autodiff/python/kernel_agent/test/matmul.py


from __future__ import annotations
import argparse
import os

from .orchestrator import KernelOptimizer, Config
# from .llm import MinimalLLMPatchProvider


def main() -> None:
    ap = argparse.ArgumentParser("kernel-agent")
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
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

    cfg = Config(max_iters=args.max_iters, patience=args.patience,
                 min_rel_improvement=args.min_rel_impr)
    # Defer heavy LLM imports unless we're actually iterating
    if args.max_iters > 0:
        from .llm import MinimalLLMPatchProvider  # lazy import
        llm = MinimalLLMPatchProvider(
            temperature=0.7,
            max_tokens=12288,
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
    print("Best backward kernel:", out["best_backward_fp"])
    print("Device:", out["device_info"])

if __name__ == "__main__":
    main()