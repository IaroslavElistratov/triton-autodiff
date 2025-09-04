import torch
import triton
import triton.testing as tt

def bench_triton(stub_fn, sidecar, *, mode="fwd", device="cuda", dtype=torch.float16):
    """
    mode: "fwd" (time forward call) or "bwd" (time backward of the forward output).
    sidecar: must expose SWEEP (list[dict]) and make_args(dims)->(args, kwargs).
    If sidecar defines flops(dims, mode)->int|float, we also return TFLOPS.
    """
    results = []
    has_flops = hasattr(sidecar, "flops")

    for dims in sidecar.SWEEP:
        args, kwargs = sidecar.make_args(dims, device=device, dtype=dtype)

        # One warm call to trigger JIT compile outside the timed region
        out = stub_fn(*args, **kwargs)

        # Some stubs return (output, compiled_kernel); normalize to `o`
        o = out[0] if isinstance(out, (tuple, list)) else out

        if mode == "fwd":
            def run():
                _ = stub_fn(*args, **kwargs)
        elif mode == "bwd":
            upstream = torch.randn_like(o)
            def run():
                o.backward(upstream, retain_graph=True)
        else:
            raise ValueError("mode must be 'fwd' or 'bwd'")

        ms = tt.do_bench(run)  # Triton’s timing helper (handles sync/warmups)
        rec = {"dims": dict(dims), "time_ms": ms}

        if has_flops:
            flop = float(sidecar.flops(dims, mode))
            rec["tflops"] = flop * 1e-12 / (ms * 1e-3)
        results.append(rec)
        if "tflops" in rec:
            print(f"{dims} -> {rec['time_ms']:.3f} ms, {rec['tflops']:.2f} TFLOPS")
        else:
            print(f"{dims} -> {rec['time_ms']:.3f} ms")
    return results


















if __name__ == "__main__":

    BATCH, N_HEADS, HEAD_DIM = 1, 2, 64
    # vary seq length for fixed head and batch=4
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[256, 512, 1024, 2048, 4096], # , 8192, 16384
            line_arg="provider",
            line_vals=["stub_fast", "stub_naive", "torch"],
            line_names=["stub_fast", "stub_naive", "torch"],
            styles=[("red", "-"), ("pink", "dotted"), ("blue", "-")],
            # ("orange", "dotted"),
            ylabel="TFLOPS",
            plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-BWD-causal=True",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "mode": "bwd",
                # "causal": False,
            },
        ))


    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, provider, device=DEVICE):

        # torch._functorch.config.donated_buffer=False

        assert mode == "bwd"
        dtype = torch.float16
        sm_scale = 1.3

        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

        # inside the stub o is allocated as q like
        upstream = torch.randn_like(q)

        if provider == "stub_fast":
            o = stub_fast(q, k, v, causal, sm_scale)
            bwd = lambda: o.backward(upstream, retain_graph=True)
            ms = triton.testing.do_bench(bwd)


        elif provider == "stub_naive":
            # o = stub(q, k, v, causal, sm_scale)
            # bwd = lambda: o.backward(upstream, retain_graph=True)
            # ms = triton.testing.do_bench(bwd)
            return

        elif provider == "torch":
            o = torch_fn(q, k, v, causal, sm_scale)
            bwd = lambda: o.backward(upstream, retain_graph=True)
            ms = triton.testing.do_bench(bwd)

        else:
            raise ValueError("Unreachable")

        flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
        total_flops = 2 * flops_per_matmul
        # if causal:
        total_flops *= 0.5

        # todo: this is really only for open-ai's bwd
        # due to mode "bwd"
        # total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        return total_flops * 1e-12 / (ms * 1e-3)



    # only works on post-Ampere GPUs right now
    # Disable Triton autodiff compile hook globally for this benchmark run
    import triton.runtime.jit as triton_jit
    _old_compiled_hook = triton_jit.JITFunction.compiled_hook
    triton_jit.JITFunction.compiled_hook = None
    try:
        bench_flash_attention.run(print_data=True)
    finally:
        # Restore after benchmark (optional)
        triton_jit.JITFunction.compiled_hook = _old_compiled_hook

