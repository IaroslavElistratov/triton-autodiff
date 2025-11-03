# python -m pip install -e kernel_agent
# cd /root/triton-autodiff
# export TRITON_AUTODIFF_DIR=$(pwd)/triton_autodiff

# Example usage:
# kernel-agent --backend triton --checkpoint /workspace/gpt-oss/gpt-oss-120b/original/ --reasoning-effort high --file-path kernel_agent/test/attention.py > kernel_agent/LOGS/out.txt
# kernel-agent --backend openai --openai-model gpt-5 --reasoning-effort high --file-path kernel_agent/test/attention.py > kernel_agent/LOGS/out.txt
# kernel-agent --backend openai --openai-model gpt-5 --reasoning-effort high --file-path kernel_agent/test/layernorm.py --min-sim 0.7 > kernel_agent/LOGS/out.txt

from __future__ import annotations
import argparse
import os

from .orchestrator import KernelOptimizer, Config
from .llm import MinimalLLMPatchProvider


def main() -> None:
    ap = argparse.ArgumentParser("kernel-agent")
    ap.add_argument("--max-iters", type=int, default=15)
    ap.add_argument("--patience_perf_stop", type=int, default=4)
    ap.add_argument("--patience_parity_restore", type=int, default=4)
    ap.add_argument("--min-rel-impr", type=float, default=0.10)
    ap.add_argument("--file-path", metavar="FILE", type=str, required=True, help="Path to the forward kernel to be optimized")

    ap.add_argument("--backend", type=str, default="triton", choices=["triton", "torch", "vllm", "openai"], help="LLM backend: local samplers or OpenAI API")
    ap.add_argument("--openai-model", type=str, default="gpt-5-mini", help="OpenAI model name (only when --backend openai)")
    ap.add_argument("--checkpoint", metavar="FILE", type=str, help="Path to the SafeTensors checkpoint (ignored when --backend openai)")
    ap.add_argument("-r", "--reasoning-effort", metavar="REASONING_EFFORT", type=str, default="medium", choices=["high", "medium", "low"], help="Reasoning effort")

    ap.add_argument("-c", "--context", metavar="CONTEXT", type=int, default=262144, help="Max context length (tokens; ignored when --backend openai)")

    # Legacy flag - kept for error message only
    ap.add_argument("--compiler", action="store_true", help=argparse.SUPPRESS)  # Hidden, shows error if used

    # RAG configuration (always enabled by default)
    ap.add_argument("--min-sim", type=float, default=0.75, help="Minimum cosine similarity threshold for RAG retrieval")
    ap.add_argument("--topk", type=int, default=1, help="Number of RAG examples to retrieve (currently only 1 is used)")

    args = ap.parse_args()

    if args.compiler:
        # [LEGACY]
        #
        #   TTIR from autodiff then raise to Python once; use as seed and target
        #   using output of triton-autodiff directly as the initial version of the backward kernel
        #   to be optimized -- "seeding a problem with a draft" (removing patcher.naive_autodiff instead
        #   just using output of triton-autodiff as patcher.kernel_snippet)
        #
        #   select method based on flags:
        #     --rag alone: Use RAG to retrieve similar backward as starting point
        #     --compiler alone: Use MLIR compiler to generate backward
        #     --rag --compiler: Use compiler for initialization; RAG provides prompt augmentation only
        #       (retrieved backward is shown to LLM as reference, but compiler-generated backward is used
        #       as the actual starting kernel for optimization)
        ap.error(
            "⚠️  ERROR: --compiler flag is temporarily not supported.\n"
            "\n"
            "MLIR compiler-based backward generation is currently disabled.\n"
        )
    # Map selected backend options into environment for the local sampler
    if args.backend:
        os.environ["KERNEL_AGENT_BACKEND"] = args.backend
    if args.checkpoint:
        os.environ["KERNEL_AGENT_CHECKPOINT"] = args.checkpoint
    if args.backend == "openai" and args.openai_model:
        os.environ["KERNEL_AGENT_OPENAI_MODEL"] = args.openai_model

    # Always use RAG adaptation strategy
    os.environ["KERNEL_AGENT_STRATEGY"] = "rag_adaptation"

    # Always set RAG configuration
    os.environ["KERNEL_AGENT_RAG_TOPK"] = str(int(args.topk))
    os.environ["KERNEL_AGENT_RAG_MIN_SIM"] = str(float(args.min_sim))

    # Initialize configuration (RAG is always enabled)
    cfg = Config(max_iters=args.max_iters,
                 patience_perf_stop=args.patience_perf_stop,
                 patience_parity_restore=args.patience_parity_restore,
                 min_rel_improvement=args.min_rel_impr)
    llm = MinimalLLMPatchProvider(
        temperature=0.7,
        max_tokens=262144,
        reasoning_effort=args.reasoning_effort,
        context=args.context,
        snippet_max_lines=cfg.snippet_max_lines,
    )
    agent = KernelOptimizer(cfg, llm)

    out = agent.run(
        fwd_fp=args.file_path,
        # benchmark=benchmark,
        # profile=profile,
        # todo:
        get_user_device_info=(lambda: "N/A"),
    )
    print("Latest metrics:", out.get("latest_metrics", {}))  # last sweep with per-shape + speedups
    print("Best metrics:", out.get("best_metrics", {}))      # accepted snapshot + ever_max_tflops
    print("Best generated file:", out["generated_fp"])
    print("Device:", out["device_info"])
    if out.get("stop_reason"):
        print("Stop reason:", out["stop_reason"])

if __name__ == "__main__":
    main()
