# Triton-Autodiff

⚠️ **This project is under active development -- it is not yet stable, and not yet feature complete.**

<!-- Not production ready yet, but I think the core ideas are valid, so will keep improving robustness. -->


# Motivation

This repo aims to take in an arbitrary triton forward stub and generate efficient backward kernels for it.

<!-- Given a triton stub:
  - creates corresponding backward triton kernels
  - then wraps this pair of forward / backward triton stubs into a torch.autograd.Function

The project aims to support arbitrary triton kernels. -->

*(Phil)osophy :)*

**Acknowledging that llms are not currently on par with humans for writing gpu kernels, this tool doesn’t seek to fully automate kernel development, it only tries to automate writing of backward kernel **given** your fwd kernel.**
**Ultimately you stay in control of writing fwd kernels, and the tool is trying to automate the tedious parts (bwd generation).**


# Implementation details

Collected dataset of 500 fwd-bwd triton stub pairs (and all the kernels they call) from github (among repos with permissive licenses).
See `kernel_agent/rag/generated`.

Since each of 500 entries in the dataset is a pair of (fwd, bwd) kernels:
- embedding all fwd kernels into vector space;
- at inference time, given a fwd kernel (which i want to generate efficient bwd for): embedding it in the same way, then finding N closest *fwd* kernels from the dataset, and - for each selected kernel - retrieving its *bwd* and pasting into llm's context

very simple, but helps llm a lot.

Overall, the project implements the following features (so far):
- rag over dataset of 500 fwd-bwd pairs
- grad correctness checks
- benchmark
- code snapshots and rollbacks
- orchestration
- process isolation
- 3 phases
- ... (much more to come!)


# Setup


```shell
git clone --depth 1 https://github.com/IaroslavElistratov/triton-autodiff
python -m pip install -e kernel_agent
```

Embeddings are provided as part of the repo. OPTIONALLY, you can generate new ones:
```shell
python kernel_agent/rag/rag_kernel_embedder.py --output kernel_agent/rag/kernel_embeddings.pkl kernel_agent/rag/generated
```


# Examples of usage

🔥 see examples of full traces in `kernel_agent/test/LOGS`.

see usage examples in `kernel_agent/test`.

```shell
kernel-agent --backend openai --openai-model gpt-5 --reasoning-effort medium --file-path kernel_agent/test/attention.py --min-sim 0.6 --topk 4 --rag-exclude tutorial,fa2_original__attention > kernel_agent/LOGS/atten.txt
kernel-agent --backend openai --openai-model gpt-5 --reasoning-effort medium --file-path kernel_agent/test/layernorm.py --min-sim 0.7 --topk 4 --rag-exclude tutorial,layer_norm_triton,fused_triton_LayerNorm,chengzeyi_stable_fast > kernel_agent/LOGS/layernorm.txt

# WIP:
# kernel-agent --backend openai --openai-model gpt-5 --reasoning-effort low --file-path kernel_agent/test/swiglu.py --min-sim 0.7 --topk 4 --rag-exclude liger_kernel/ops/swiglu > kernel_agent/LOGS/swiglu.txt
# kernel-agent --backend openai --openai-model gpt-5 --reasoning-effort medium --file-path kernel_agent/test/qwen3_moe_fused.py --min-sim 0.5 --topk 4 --rag-exclude woct0rdho > kernel_agent/LOGS/moe.txt
# kernel-agent --backend openai --openai-model gpt-5 --reasoning-effort medium --file-path kernel_agent/test/rwkv6_fused_recurrent.py --min-sim 0.7 --topk 4 --rag-exclude fla/ops/rwkv6/fused_recurrent > kernel_agent/LOGS/rwkv.txt
```

feel free to paly with: `--min-sim` (RAG minimal similarity), `--topk` (RAG num examples to include), `--rag-exclude` (excludes paths containing given substrings), `--max-iters` (number iterations of the main loop), `--reasoning-effort`

code examples follow this pattern:
- as part of your source file, provide `SWEEP` (tells what shapes your stub expects), `make_args` (constructs example inputs), `setup`
- decorate your stub (NOT the kernel) with `@autodiff`
- specify `inputs_require_grad` -- tells which idxs (of args to your stub) will need grads
- `required` -- mark only small shapes which will work (without OOM) with naive torch fwd reference
- do not use `--rag-exclude` -- this is purely for my testing (to avoid cheating)
<!-- - If a forward stub returns extra tensors (from your fwd stub) which you don’t want to differentiate through, detach them (or convert them to non-floating types) before returning.
    return main_output, buffer.detach() -->

temporary limitations (to be fixed):
- your code, and all the helpers it calls, must be in the same file —- for now works best when the whole file is relatively short (<1k lines)
- your file is being exec’ed so avoid heavy initialization / work in module level (put them in functions so exec doesn’t reach them by default)
- does not support heavily quantized kernels (for now)
- does not support computing grads inside fwd kernel (e.g. fused cross-entropy like)


<!-- Flags compatibility
-------------------

- --backend: stub | triton | torch | vllm | openai
- --checkpoint: used by triton/torch/vllm; ignored by openai/stub
- --openai-model: only used by openai (defaults to gpt-5-mini)
- --reasoning-effort: used by Harmony local backends (triton/torch/vllm); ignored by openai
- Always used: --file-path, --max-iters, --patience_perf_stop, --patience_parity_restore, --min-rel-impr, --min-sim, --topk -->


# Licensing

All my original code in this repo is licensed under the Apache License, Version 2.0.
Files in `kernel_agent/rag/generated/` and any files that carry their own license headers are
under their respective licenses. See `THIRD_PARTY_LICENSES.md` for a list.
Many thanks to all the kernel authors!
