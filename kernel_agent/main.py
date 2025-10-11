# python -m pip install -e kernel_agent
# cd /root/triton-autodiff
# export TRITON_AUTODIFF_DIR=$(pwd)/triton_autodiff
# kernel-agent --backend triton --checkpoint /workspace/gpt-oss/gpt-oss-20b/original/ --file-path kernel_agent/test/matmul.py --reasoning-effort medium --mode phased > kernel_agent/LOGS/out.txt
# kernel-agent --backend triton --checkpoint /workspace/gpt-oss/gpt-oss-120b/original/ --file-path kernel_agent/test/attention.py --reasoning-effort medium --mode phased > kernel_agent/LOGS/out.txt

# kernel-agent --backend openai --openai-model gpt-5-mini --file-path kernel_agent/test/matmul.py --mode phased --reasoning-effort medium --rag > kernel_agent/LOGS/out.txt
# kernel-agent --backend openai --openai-model gpt-5 --file-path kernel_agent/test/attention.py --mode phased --reasoning-effort medium --rag > kernel_agent/LOGS/out.txt

from __future__ import annotations
import argparse
import os

from .orchestrator import KernelOptimizer, Config
from .llm import MinimalLLMPatchProvider


def main() -> None:
    ap = argparse.ArgumentParser("kernel-agent")
    ap.add_argument("--max-iters", type=int, default=32)
    ap.add_argument("--patience_perf_stop", type=int, default=4)
    ap.add_argument("--patience_parity_restore", type=int, default=4)
    ap.add_argument("--min-rel-impr", type=float, default=0.10)
    ap.add_argument("--file-path", metavar="FILE", type=str, required=True, help="Path to the forward kernel to be optimized")

    ap.add_argument("--backend", type=str, default="triton", choices=["triton", "torch", "vllm", "openai"], help="LLM backend: local samplers or OpenAI API")
    ap.add_argument("--openai-model", type=str, default="gpt-5-mini", help="OpenAI model name (only when --backend openai)")
    ap.add_argument("--checkpoint", metavar="FILE", type=str, help="Path to the SafeTensors checkpoint (ignored when --backend openai)")
    ap.add_argument("-r", "--reasoning-effort", metavar="REASONING_EFFORT", type=str, default="medium", choices=["high", "medium", "low"], help="Reasoning effort")

    ap.add_argument("-c", "--context", metavar="CONTEXT", type=int, default=32768, help="Max context length (tokens; ignored when --backend openai)")
    ap.add_argument("--mode", type=str, default="regular", choices=["regular", "phased"], help="Optimization strategy mode")

    # RAG configuration (optional; default off). When enabled, appends a compact block with
    # retrieved backward references to the LLM prompt using a prebuilt embeddings index.
    ap.add_argument("--rag", action="store_true", help="Enable RAG: include retrieved backward references in prompts")
    ap.add_argument("--rag-topk", type=int, default=2, help="Number of RAG examples to include (default: 2)")
    ap.add_argument("--rag-min-sim", type=float, default=0.75, help="Minimum cosine similarity to include an example (default: 0.75)")
    args = ap.parse_args()

    # Map selected backend options into environment for the local sampler
    if args.backend:
        os.environ["KERNEL_AGENT_BACKEND"] = args.backend
    if args.checkpoint:
        os.environ["KERNEL_AGENT_CHECKPOINT"] = args.checkpoint
    if args.backend == "openai" and args.openai_model:
        os.environ["KERNEL_AGENT_OPENAI_MODEL"] = args.openai_model
    # Strategy toggle (regular | phased)
    os.environ["KERNEL_AGENT_STRATEGY"] = args.mode

    if args.rag:
        os.environ["KERNEL_AGENT_RAG"] = "1"
        os.environ["KERNEL_AGENT_RAG_TOPK"] = str(int(args.rag_topk))
        os.environ["KERNEL_AGENT_RAG_MIN_SIM"] = str(float(args.rag_min_sim))

    cfg = Config(max_iters=args.max_iters,
                 patience_perf_stop=args.patience_perf_stop,
                 patience_parity_restore=args.patience_parity_restore,
                 min_rel_improvement=args.min_rel_impr)
    llm = MinimalLLMPatchProvider(
        temperature=0.7,
        max_tokens=32768,
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
    print("Best backward file:", out["backward_fp"])
    print("Device:", out["device_info"])
    if out.get("stop_reason"):
        print("Stop reason:", out["stop_reason"])

if __name__ == "__main__":
    main()