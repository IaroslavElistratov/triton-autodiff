LLM Optimizer
==============


```shell
read -s -p "OPENAI_API_KEY: " OPENAI_API_KEY; echo; export OPENAI_API_KEY
MAX_JOBS=20  pip install -e triton_autodiff --verbose --no-build-isolation
export TRITON_AUTODIFF_DIR=$(pwd)/triton_autodiff
ln -s $TRITON_AUTODIFF_DIR/third_party/autodiff/python/api.py $TRITON_AUTODIFF_DIR/python/triton/backends/autodiff.py

python -m pip install -e kernel_agent
triton_autodiff && kernel-agent --backend triton --checkpoint /workspace/gpt-oss/gpt-oss-120b/original/ --file-path kernel_agent/test/attention.py --reasoning-effort medium --mode phased > kernel_agent/LOGS/out.txt
```

Flags compatibility
-------------------

- --backend: stub | triton | torch | vllm | openai
- --checkpoint: used by triton/torch/vllm; ignored by openai/stub
- --openai-model: only used by openai (defaults to gpt-5-mini)
- --context: used by triton; ignored by torch/vllm/openai
- --reasoning-effort: used by Harmony local backends (triton/torch/vllm); ignored by openai
- Always used: --file-path, --mode, --max-iters, --patience_perf_stop, --patience_parity_restore, --min-rel-impr