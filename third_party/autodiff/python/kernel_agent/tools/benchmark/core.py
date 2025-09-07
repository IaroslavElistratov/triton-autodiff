from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import triton.testing as tt

Tensor = torch.Tensor
MaybeTensors = Union[Tensor, Sequence[Tensor]]
OutputSel = Union[str, int, Sequence[int], Callable[[MaybeTensors], MaybeTensors]]

def _as_tuple(x: MaybeTensors) -> Tuple[Tensor, ...]:
    return (x,) if isinstance(x, torch.Tensor) else tuple(x)

def _select_outputs(y: MaybeTensors, sel: OutputSel) -> Tuple[Tensor, ...]:
    yt = _as_tuple(y)
    if sel == "auto":
        return yt
    if isinstance(sel, int):
        return (yt[sel],)
    if callable(sel):
        return _as_tuple(sel(y))
    return tuple(yt[i] for i in sel)

@dataclass(frozen=True)
class BenchRecord:
    dims: Dict[str, Any]
    time_ms: float
    tflops: Optional[float] = None

def bench_op(
    my_op: Callable[..., MaybeTensors],
    sidecar: Dict[str, Any],
    *,
    mode: str = "fwd",                  # "fwd" | "bwd"
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    outputs: OutputSel = "auto",        # which tensors to drive backward through
    seed: int = 0,
) -> List[BenchRecord]:
    """
    Benchmark an autograd-backed op.

    sidecar must expose:
      - SWEEP: Iterable[dict]
      - make_args(dims, device, dtype) -> (args, kwargs)
      - optional flops(dims, mode) -> int|float for TFLOPS reporting
    """
    if mode not in ("fwd", "bwd"):
        raise ValueError("mode must be 'fwd' or 'bwd'")

    results: List[BenchRecord] = []

    if not isinstance(sidecar, dict):
        raise TypeError("bench_op expects sidecar as a dict namespace")

    sweep = list(sidecar.get("SWEEP", [{}]))
    make_args_fn = sidecar.get("make_args")
    if not callable(make_args_fn):
        raise RuntimeError("sidecar['make_args'] must be callable")

    def _ensure_requires_grad(x: Any) -> Any:
        try:
            if isinstance(x, torch.Tensor):
                x.requires_grad_(True)
                return x
            # Shallow map for simple sequences of tensors
            if isinstance(x, (list, tuple)):
                seq = [(_ensure_requires_grad(t)) for t in x]
                return type(x)(seq)
        except Exception:
            pass
        return x

    for dims in sweep:
        print(f"[benchmark] running with {dims}")
        args, kwargs = make_args_fn(dims, device=device, dtype=dtype)
        # Ensure autograd is enabled on inputs so outputs require grad
        try:
            args = tuple(_ensure_requires_grad(t) for t in args)
        except Exception:
            args = (_ensure_requires_grad(args),)
        if isinstance(kwargs, dict):
            for k, v in list(kwargs.items()):
                kwargs[k] = _ensure_requires_grad(v)

        # Warm compile outside the timed region
        y_warm = my_op(*args, **kwargs)

        if mode == "fwd":
            def run():
                _ = my_op(*args, **kwargs)
        else:
            ys = _select_outputs(y_warm, outputs)
            torch.manual_seed(seed)
            ups = tuple(torch.randn_like(t) for t in ys)

            def run():
                torch.autograd.backward(ys, ups, retain_graph=True)

        ms = float(tt.do_bench(run))
        flops_fn = sidecar.get("flops")
        flop = float(flops_fn(dims, mode)) if callable(flops_fn) else None
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

