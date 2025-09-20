LLM Optimizer
==============


```shell
MAX_JOBS=20  pip install -e triton_autodiff --verbose --no-build-isolation
export TRITON_AUTODIFF_DIR=$(pwd)/triton_autodiff
ln -s $TRITON_AUTODIFF_DIR/third_party/autodiff/python/api.py $TRITON_AUTODIFF_DIR/python/triton/backends/autodiff.py

python -m pip install -e kernel_agent
triton_autodiff && kernel-agent --backend triton --checkpoint /workspace/gpt-oss/gpt-oss-120b/original/ --file-path kernel_agent/test/attention.py --reasoning-effort medium --mode phased > kernel_agent/LOGS/out.txt
```