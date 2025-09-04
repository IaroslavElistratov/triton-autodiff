from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import torch
import triton
import triton.testing as tt

from ...utils import compile_kernel as create_op

@dataclass(frozen=True)
class BenchRecord:
    dims: Dict[str, Any]
    time_ms: float
    tflops: Optional[float] = None


def bench_triton(
    my_op: Callable[..., Any],
    sidecar,
    *,
    mode: str = "fwd",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> List[BenchRecord]:
    """
    Benchmark the autograd-wrapped op that composes forward+backward.

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
        warm = my_op(*args, **kwargs)
        warm = warm[0] if isinstance(warm, (tuple, list)) else warm

        if mode == "fwd":
            def run():
                _ = my_op(*args, **kwargs)
        else:
            # Build graph once, time only autograd backward through op outputs
            out = my_op(*args, **kwargs)
            outs: List[torch.Tensor] = list(out) if isinstance(out, (tuple, list)) else [out]
            upstreams = [torch.randn_like(o) for o in outs]

            def run():
                if len(outs) == 1:
                    outs[0].backward(upstreams[0], retain_graph=True)
                else:
                    torch.autograd.backward(tuple(outs), tuple(upstreams), retain_graph=True)

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
#             ms = tt.do_bench(bwd)

#         elif provider == "torch":
#             o = torch_fn(q, k, v, causal, sm_scale)
#             bwd = lambda: o.backward(upstream, retain_graph=True)
#             ms = tt.do_bench(bwd)

#         # todo: generailize for other kernels
#         flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
#         # total_flops = ..
#         return total_flops * 1e-12 / (ms * 1e-3)

