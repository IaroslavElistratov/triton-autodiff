from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.testing as tt

from ...utils import compile_kernel as create_op  # reuse your loader

@dataclass(frozen=True)
class BenchRecord:
    dims: Dict[str, Any]
    time_ms: float
    tflops: Optional[float] = None

def _prep_args_for_mode(args: Sequence[Any], *, mode: str) -> Tuple[Tuple[Any, ...], Tuple[int, ...]]:
    if mode != "bwd":
        return tuple(args), tuple()
    out: List[Any] = []
    grad_idx: List[int] = []
    for i, a in enumerate(args):
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            a = a.detach().clone().requires_grad_(True)
            grad_idx.append(i)
        out.append(a)
    return tuple(out), tuple(grad_idx)

def bench_triton(
    stub_fn,
    sidecar,
    *,
    mode: str = "fwd",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> List[BenchRecord]:
    """
    sidecar must expose:
      SWEEP: list[dict]
      make_args(dims, device, dtype) -> (args, kwargs)
    optional:
      flops(dims, mode) -> float
    """
    if mode not in ("fwd", "bwd"):
        raise ValueError("mode must be 'fwd' or 'bwd'")

    results: List[BenchRecord] = []
    has_flops = hasattr(sidecar, "flops")
    sweep = list(getattr(sidecar, "SWEEP", [{}]))

    for dims in sweep:
        args, kwargs = sidecar.make_args(dims, device=device, dtype=dtype)

        # warm compile outside timed region
        warm = stub_fn(*args, **kwargs)
        warm = warm[0] if isinstance(warm, (tuple, list)) else warm

        if mode == "fwd":
            def run():
                _ = stub_fn(*args, **kwargs)
        else:
            # backward only: build a graph once, then time .backward()
            bwd_args, _ = _prep_args_for_mode(args, mode="bwd")
            o = stub_fn(*bwd_args, **kwargs)
            o = o[0] if isinstance(o, (tuple, list)) else o
            upstream = torch.randn_like(o)
            def run():
                o.backward(upstream, retain_graph=True)

        ms = float(tt.do_bench(run))
        flop = float(sidecar.flops(dims, mode)) if has_flops else None
        tflops = flop * 1e-12 / (ms * 1e-3) if flop is not None else None
        results.append(BenchRecord(dims=dict(dims), time_ms=ms, tflops=tflops))

    return results

def reduce_bench(records: Sequence[BenchRecord]) -> Dict[str, float]:
    import statistics as st
    if not records:
        return {"median_ms": float("nan"), "best_ms": float("nan")}
    times = [r.time_ms for r in records]
    out: Dict[str, float] = {
        "median_ms": float(st.median(times)),
        "best_ms": float(min(times)),
    }
    mx = max((r.tflops for r in records if r.tflops is not None), default=None)
    if mx is not None:
        out["max_tflops"] = float(mx)
    return out

def bench_from_file(
    fwd_fp: str,
    *,
    overwrite_bwd_fp: Optional[str] = None,
    mode: str = "bwd",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> List[BenchRecord]:
    """
    Compile and run the user's module via utils.compile_kernel then bench.
    Pass overwrite_bwd_fp to use the current edited backward stub.
    """
    _op, _bwd, ns = create_op(fwd_fp, overwrite_fp=overwrite_bwd_fp)
    stub = ns.get("stub")
    if not callable(stub):
        raise RuntimeError("expected stub(...) in user module")
    if not callable(ns.get("make_args")):
        raise RuntimeError("expected make_args(dims)->(args, kwargs) in user module")
    return bench_triton(stub, ns, mode=mode, device=device, dtype=dtype)





# if __name__ == "__main__":

#     BATCH, N_HEADS, HEAD_DIM = 1, 2, 64
#     # vary seq length for fixed head and batch=4
#     configs = []
#     configs.append(
#         triton.testing.Benchmark(
#             x_names=["N_CTX"],
#             x_vals=[256, 512, 1024, 2048, 4096], # , 8192, 16384
#             line_arg="provider",
#             line_vals=["stub_fast", "stub_naive", "torch"],
#             line_names=["stub_fast", "stub_naive", "torch"],
#             styles=[("red", "-"), ("pink", "dotted"), ("blue", "-")],
#             # ("orange", "dotted"),
#             ylabel="TFLOPS",
#             plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-BWD-causal=True",
#             args={
#                 "H": N_HEADS,
#                 "BATCH": BATCH,
#                 "HEAD_DIM": HEAD_DIM,
#                 "mode": "bwd",
#                 # "causal": False,
#             },
#         ))


#     @triton.testing.perf_report(configs)
#     def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):

#         # torch._functorch.config.donated_buffer=False

#         assert mode == "bwd"
#         dtype = torch.float16
#         sm_scale = 1.3

#         q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
#         k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
#         v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

#         # inside the stub o is allocated as q like
#         upstream = torch.randn_like(q)

#         if provider == "stub":
#             o = stub_fast(q, k, v, causal, sm_scale)
#             bwd = lambda: o.backward(upstream, retain_graph=True)
#             ms = triton.testing.do_bench(bwd)

#         elif provider == "torch":
#             o = torch_fn(q, k, v, causal, sm_scale)
#             bwd = lambda: o.backward(upstream, retain_graph=True)
#             ms = triton.testing.do_bench(bwd)

#         # todo-now: generailize for other kernels
#         flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
#         total_flops = 2 * flops_per_matmul
#         # if causal:
#         total_flops *= 0.5

#         # todo: this is really only for open-ai's bwd
#         # due to mode "bwd"
#         # total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
#         return total_flops * 1e-12 / (ms * 1e-3)



#     # only works on post-Ampere GPUs right now
#     # Disable Triton autodiff compile hook globally for this benchmark run
#     import triton.runtime.jit as triton_jit
#     _old_compiled_hook = triton_jit.JITFunction.compiled_hook
#     triton_jit.JITFunction.compiled_hook = None
#     try:
#         bench_flash_attention.run(print_data=True)
#     finally:
#         # Restore after benchmark (optional)
#         triton_jit.JITFunction.compiled_hook = _old_compiled_hook

