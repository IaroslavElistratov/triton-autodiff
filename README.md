# Triton-Autodiff

A fork of [Triton](https://github.com/openai/triton) with an experimental automatic differentiation support.

This work clones the main Triton repository, but intends to minimize
divergences in the core. Most of the autodiff work is in [third_party/autodiff](third_party/autodiff)
subdirectory.

**NOTE: This project is in its early stages and under heavy development -- it is not stable, not feature complete, and APIs will change.**


# Motivation

This repo aims to take in an arbitrary triton kernel and return a callable that supports automatic differentiation out of the box.

Given a triton kernel and its stub:
  - creates a corresponding backward_triton_kernel
  - then wraps this pair of forward / backward triton kernels into a torch.autograd.Function

Below I show how this repo helps to simplify your kernel definitions. The project aims to support arbitrary triton kernels.


# 🔥 Example 1: Flash Attention v2

Let's look at flash attention v2 impl from [official triton tutorial](https://github.com/triton-lang/triton/blob/105cb56487cd8a433b8fbfe9cc63c1f1c04a4b2a/python/tutorials/06-fused-attention.py).

I'll show code snippets below. **For end to end example see: [autodiff/test/flash_attention_v2/run.py](third_party/autodiff/test/flash_attention_v2/run.py)**

<details>
  <summary>❗ CLICK TO EXPAND DIFF ❗</summary>

```diff

+ # Define your forward kernel as usual:

+ # Unchanged (omitted for brevity)
@triton.jit
def _attn_fwd_inner
  ...

+ # Unchanged (omitted for brevity)
+ # temporary limitation: autotune is not supported for now
- @triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd
  ...


+ # No need to write backward kernels and backward stubs by hand anymore!

-@triton.jit
-def _attn_bwd_preprocess(O, DO,  #
-                         Delta,  #
-                         Z, H, N_CTX,  #
-                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
-                         ):
-    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
-    off_hz = tl.program_id(1)
-    off_n = tl.arange(0, HEAD_DIM)
-    # load
-    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
-    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
-    delta = tl.sum(o * do, axis=1)
-    # write-back
-    tl.store(Delta + off_hz * N_CTX + off_m, delta)
-
-
-# The main inner-loop logic for computing dK and dV.
-@triton.jit
-def _attn_bwd_dkdv(dk, dv,  #
-                   Q, k, v, sm_scale,  #
-                   DO,  #
-                   M, D,  #
-                   # shared by Q/K/V/DO.
-                   stride_tok, stride_d,  #
-                   H, N_CTX, BLOCK_M1: tl.constexpr,  #
-                   BLOCK_N1: tl.constexpr,  #
-                   HEAD_DIM: tl.constexpr,  #
-                   # Filled in by the wrapper.
-                   start_n, start_m, num_steps,  #
-                   MASK: tl.constexpr):
-    offs_m = start_m + tl.arange(0, BLOCK_M1)
-    offs_n = start_n + tl.arange(0, BLOCK_N1)
-    offs_k = tl.arange(0, HEAD_DIM)
-    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
-    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
-    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
-    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
-    curr_m = start_m
-    step_m = BLOCK_M1
-    for blk_idx in range(num_steps):
-        qT = tl.load(qT_ptrs)
-        # Load m before computing qk to reduce pipeline stall.
-        offs_m = curr_m + tl.arange(0, BLOCK_M1)
-        m = tl.load(M + offs_m)
-        qkT = tl.dot(k, qT)
-        pT = tl.math.exp2(qkT - m[None, :])
-        # Autoregressive masking.
-        if MASK:
-            mask = (offs_m[None, :] >= offs_n[:, None])
-            pT = tl.where(mask, pT, 0.0)
-        do = tl.load(do_ptrs)
-        # Compute dV.
-        ppT = pT
-        ppT = ppT.to(tl.float16)
-        dv += tl.dot(ppT, do)
-        # D (= delta) is pre-divided by ds_scale.
-        Di = tl.load(D + offs_m)
-        # Compute dP and dS.
-        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
-        dsT = pT * (dpT - Di[None, :])
-        dsT = dsT.to(tl.float16)
-        dk += tl.dot(dsT, tl.trans(qT))
-        # Increment pointers.
-        curr_m += step_m
-        qT_ptrs += step_m * stride_tok
-        do_ptrs += step_m * stride_tok
-    return dk, dv
-
-
-# the main inner-loop logic for computing dQ
-@triton.jit
-def _attn_bwd_dq(dq, q, K, V,  #
-                 do, m, D,
-                 # shared by Q/K/V/DO.
-                 stride_tok, stride_d,  #
-                 H, N_CTX,  #
-                 BLOCK_M2: tl.constexpr,  #
-                 BLOCK_N2: tl.constexpr,  #
-                 HEAD_DIM: tl.constexpr,
-                 # Filled in by the wrapper.
-                 start_m, start_n, num_steps,  #
-                 MASK: tl.constexpr):
-    offs_m = start_m + tl.arange(0, BLOCK_M2)
-    offs_n = start_n + tl.arange(0, BLOCK_N2)
-    offs_k = tl.arange(0, HEAD_DIM)
-    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
-    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
-    # D (= delta) is pre-divided by ds_scale.
-    Di = tl.load(D + offs_m)
-    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
-    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
-    curr_n = start_n
-    step_n = BLOCK_N2
-    for blk_idx in range(num_steps):
-        kT = tl.load(kT_ptrs)
-        vT = tl.load(vT_ptrs)
-        qk = tl.dot(q, kT)
-        p = tl.math.exp2(qk - m)
-        # Autoregressive masking.
-        if MASK:
-            offs_n = curr_n + tl.arange(0, BLOCK_N2)
-            mask = (offs_m[:, None] >= offs_n[None, :])
-            p = tl.where(mask, p, 0.0)
-        # Compute dP and dS.
-        dp = tl.dot(do, vT).to(tl.float32)
-        ds = p * (dp - Di[:, None])
-        ds = ds.to(tl.float16)
-        # Compute dQ.
-        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
-        dq += tl.dot(ds, tl.trans(kT))
-        # Increment pointers.
-        curr_n += step_n
-        kT_ptrs += step_n * stride_tok
-        vT_ptrs += step_n * stride_tok
-    return dq
-
-
-@triton.jit
-def _attn_bwd(Q, K, V, sm_scale,  #
-              DO,  #
-              DQ, DK, DV,  #
-              M, D,
-              # shared by Q/K/V/DO.
-              stride_z, stride_h, stride_tok, stride_d,  #
-              H, N_CTX,  #
-              BLOCK_M1: tl.constexpr,  #
-              BLOCK_N1: tl.constexpr,  #
-              BLOCK_M2: tl.constexpr,  #
-              BLOCK_N2: tl.constexpr,  #
-              BLK_SLICE_FACTOR: tl.constexpr,  #
-              HEAD_DIM: tl.constexpr):
-    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
-
-    bhid = tl.program_id(2)
-    off_chz = (bhid * N_CTX).to(tl.int64)
-    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
-    pid = tl.program_id(0)
-
-    # offset pointers for batch/head
-    Q += adj
-    K += adj
-    V += adj
-    DO += adj
-    DQ += adj
-    DK += adj
-    DV += adj
-    M += off_chz
-    D += off_chz
-
-    # load scales
-    offs_k = tl.arange(0, HEAD_DIM)
-
-    start_n = pid * BLOCK_N1
-    start_m = start_n
-
-    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
-    offs_n = start_n + tl.arange(0, BLOCK_N1)
-
-    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
-    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
-
-    # load K and V: they stay in SRAM throughout the inner loop.
-    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
-    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
-
-    num_steps = BLOCK_N1 // MASK_BLOCK_M1
-
-    dk, dv = _attn_bwd_dkdv(dk, dv,  #
-                            Q, k, v, sm_scale,  #
-                            DO,  #
-                            M, D,  #
-                            stride_tok, stride_d,  #
-                            H, N_CTX,  #
-                            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
-                            start_n, start_m, num_steps,  #
-                            MASK=True  #
-                            )
-
-    start_m += num_steps * MASK_BLOCK_M1
-    num_steps = (N_CTX - start_m) // BLOCK_M1
-
-    # Compute dK and dV for non-masked blocks.
-    dk, dv = _attn_bwd_dkdv(  #
-        dk, dv,  #
-        Q, k, v, sm_scale,  #
-        DO,  #
-        M, D,  #
-        stride_tok, stride_d,  #
-        H, N_CTX,  #
-        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
-        start_n, start_m, num_steps,  #
-        MASK=False  #
-    )
-
-    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
-    tl.store(dv_ptrs, dv)
-
-    # Write back dK.
-    dk *= sm_scale
-    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
-    tl.store(dk_ptrs, dk)
-
-    # THIS BLOCK DOES DQ:
-    start_m = pid * BLOCK_M2
-    end_n = start_m + BLOCK_M2
-
-    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
-    offs_m = start_m + tl.arange(0, BLOCK_M2)
-
-    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
-    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
-    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
-
-    m = tl.load(M + offs_m)
-    m = m[:, None]
-
-    # Compute dQ for masked (diagonal) blocks.
-    # NOTE: This code scans each row of QK^T backward (from right to left,
-    # but inside each call to _attn_bwd_dq, from left to right), but that's
-    # not due to anything important.  I just wanted to reuse the loop
-    # structure for dK & dV above as much as possible.
-    num_steps = BLOCK_M2 // MASK_BLOCK_N2
-    dq = _attn_bwd_dq(dq, q, K, V,  #
-                      do, m, D,  #
-                      stride_tok, stride_d,  #
-                      H, N_CTX,  #
-                      BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,  #
-                      start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,  #
-                      MASK=True  #
-                      )
-    end_n -= num_steps * MASK_BLOCK_N2
-    # stage 2
-    num_steps = end_n // BLOCK_N2
-    dq = _attn_bwd_dq(dq, q, K, V,  #
-                      do, m, D,  #
-                      stride_tok, stride_d,  #
-                      H, N_CTX,  #
-                      BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
-                      start_m, end_n - num_steps * BLOCK_N2, num_steps,  #
-                      MASK=False  #
-                      )
-    # Write back dQ.
-    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
-    dq *= LN2
-    tl.store(dq_ptrs, dq)


+ # Define your forward stub as usual:

+ # The only change to your stub is to explicitly pass kernel function as the first argument
+ # Everything else is unchanged (omitted for brevity below "...")
- def stub_forward(ctx, q, k, v, causal, sm_scale):
+ def stub_forward(kernel, ctx, q, k, v, causal, sm_scale):
  ...
-  _attn_fwd[grid](
+  kernel[grid](
  ...


-def stub_backward(ctx, do):
-    q, k, v, o, M = ctx.saved_tensors
-    assert do.is_contiguous()
-    assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
-    dq = torch.empty_like(q)
-    dk = torch.empty_like(k)
-    dv = torch.empty_like(v)
-    BATCH, N_HEAD, N_CTX = q.shape[:3]
-    PRE_BLOCK = 128
-    NUM_WARPS, NUM_STAGES = 4, 5
-    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
-    BLK_SLICE_FACTOR = 2
-    RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
-    arg_k = k
-    arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
-    PRE_BLOCK = 128
-    assert N_CTX % PRE_BLOCK == 0
-    pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
-    delta = torch.empty_like(M)
-    _attn_bwd_preprocess[pre_grid](
-        o, do,  #
-        delta,  #
-        BATCH, N_HEAD, N_CTX,  #
-        BLOCK_M=PRE_BLOCK, HEAD_DIM=ctx.HEAD_DIM  #
-    )
-    grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
-    _attn_bwd[grid](
-        q, arg_k, v, ctx.sm_scale, do, dq, dk, dv,  #
-        M, delta,  #
-        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
-        N_HEAD, N_CTX,  #
-        BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
-        BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
-        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
-        HEAD_DIM=ctx.HEAD_DIM,  #
-        num_warps=NUM_WARPS,  #
-        num_stages=NUM_STAGES  #
-    )
-
-    return dq, dk, dv, None, None



causal=False
sm_scale=0.5

q = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
k = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
v = torch.empty((B, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)

grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)


#### backward ####

# in your case, upstream gradient can come from next torch operation which was called on the output of your kernel ("my_out" below)
upstream = torch.randn_like(q)

q.requires_grad = True
k.requires_grad = True
v.requires_grad = True

+ from triton.backends.autodiff import autodiff, right_partial

+ # temporary limitation:
+ #   here need right_partial because binding trailing arguments (not arguments at the beginning), because kernel signature
+ #   happen to have these at the end. Now this uses the stub which only needs grad args (non grad args have been bound)
+ stub = right_partial(stub, causal, sm_scale)

+ my_op, bwd_kernel = autodiff(kernel, stub, grid, idx_upstream=5, non_stub_args_idxs=[3])

+ # temporary limitation: run stub once before calling my_op
+ stub(bwd_kernel, q, k, v)

+ my_out = my_op(q, k, v)
+ my_out.backward(upstream)

+ # now grads have been populated for: q.grad, k.grad, v.grad
```

</details>




# 🧪 Example 2: LayerNorm

Let's look at layer-norm impl from [official triton tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html#sphx-glr-getting-started-tutorials-05-layer-norm-py).

I'll show code snippets below. **For end to end example see: [autodiff/test/layernorm/run.py](third_party/autodiff/test/layernorm/run.py)**

<details>
  <summary>❗ CLICK TO EXPAND DIFF ❗</summary>

```diff

+ # Define your forward kernel as usual:

+ # Unchanged (omitted for brevity)

@triton.jit
def _layer_norm_fwd_fused


+ # No need to write backward kernels and backward stubs by hand anymore!

-@triton.jit
-def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
-                             DY,  # pointer to the output gradient
-                             DW,  # pointer to the partial sum of weights gradient
-                             DB,  # pointer to the partial sum of biases gradient
-                             X,  # pointer to the input
-                             W,  # pointer to the weights
-                             Mean,  # pointer to the mean
-                             Rstd,  # pointer to the 1/std
-                             Lock,  # pointer to the lock
-                             stride,  # how much to increase the pointer when moving by 1 row
-                             N,  # number of columns in X
-                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
-    # Map the program id to the elements of X, DX, and DY it should compute.
-    row = tl.program_id(0)
-    cols = tl.arange(0, BLOCK_SIZE_N)
-    mask = cols < N
-    X += row * stride
-    DY += row * stride
-    DX += row * stride
-    # Offset locks and weights/biases gradient pointer for parallel reduction
-    lock_id = row % GROUP_SIZE_M
-    Lock += lock_id
-    Count = Lock + GROUP_SIZE_M
-    DW = DW + lock_id * N + cols
-    DB = DB + lock_id * N + cols
-    # Load data to SRAM
-    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
-    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
-    w = tl.load(W + cols, mask=mask).to(tl.float32)
-    mean = tl.load(Mean + row)
-    rstd = tl.load(Rstd + row)
-    # Compute dx
-    xhat = (x - mean) * rstd
-    wdy = w * dy
-    xhat = tl.where(mask, xhat, 0.)
-    wdy = tl.where(mask, wdy, 0.)
-    c1 = tl.sum(xhat * wdy, axis=0) / N
-    c2 = tl.sum(wdy, axis=0) / N
-    dx = (wdy - (xhat * c1 + c2)) * rstd
-    # Write dx
-    tl.store(DX + cols, dx, mask=mask)
-    # Accumulate partial sums for dw/db
-    partial_dw = (dy * xhat).to(w.dtype)
-    partial_db = (dy).to(w.dtype)
-    while tl.atomic_cas(Lock, 0, 1) == 1:
-        pass
-    count = tl.load(Count)
-    # First store doesn't accumulate
-    if count == 0:
-        tl.atomic_xchg(Count, 1)
-    else:
-        partial_dw += tl.load(DW, mask=mask)
-        partial_db += tl.load(DB, mask=mask)
-    tl.store(DW, partial_dw, mask=mask)
-    tl.store(DB, partial_db, mask=mask)
-
-    # need a barrier to ensure all threads finished before
-    # releasing the lock
-    tl.debug_barrier()
-
-    # Release the lock
-    tl.atomic_xchg(Lock, 0)
-
-
-@triton.jit
-def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
-                         DB,  # pointer to the partial sum of biases gradient
-                         FINAL_DW,  # pointer to the weights gradient
-                         FINAL_DB,  # pointer to the biases gradient
-                         M,  # GROUP_SIZE_M
-                         N,  # number of columns
-                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
-    # Map the program id to the elements of DW and DB it should compute.
-    pid = tl.program_id(0)
-    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
-    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
-    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
-    # Iterate through the rows of DW and DB to sum the partial sums.
-    for i in range(0, M, BLOCK_SIZE_M):
-        rows = i + tl.arange(0, BLOCK_SIZE_M)
-        mask = (rows[:, None] < M) & (cols[None, :] < N)
-        offs = rows[:, None] * N + cols[None, :]
-        dw += tl.load(DW + offs, mask=mask, other=0.)
-        db += tl.load(DB + offs, mask=mask, other=0.)
-    # Write the final sum to the output.
-    sum_dw = tl.sum(dw, axis=0)
-    sum_db = tl.sum(db, axis=0)
-    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
-    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)





+ # Define your forward stub as usual:

+ # The only change to your stub is to explicitly pass kernel function as the first argument
+ # Everything else is unchanged (omitted for brevity below "...")
- def stub_forward(ctx, x, normalized_shape, weight, bias, eps):
+ def stub_forward(kernel, ctx, x, normalized_shape, weight, bias, eps):
  ...
-  _layer_norm_fwd_fused[grid](
+  kernel[grid](
  ...


-def stub_backward(ctx, dy):
-    x, w, b, m, v = ctx.saved_tensors
-    # heuristics for amount of parallel reduction stream for DW/DB
-    N = w.shape[0]
-    GROUP_SIZE_M = 64
-    if N <= 8192: GROUP_SIZE_M = 96
-    if N <= 4096: GROUP_SIZE_M = 128
-    if N <= 1024: GROUP_SIZE_M = 256
-    # allocate output
-    locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
-    _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
-    _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
-    dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
-    db = torch.empty((N, ), dtype=w.dtype, device=w.device)
-    dx = torch.empty_like(dy)
-    # enqueue kernel using forward pass heuristics
-    # also compute partial sums for DW and DB
-    x_arg = x.reshape(-1, x.shape[-1])
-    M, N = x_arg.shape
-    _layer_norm_bwd_dx_fused[(M, )](  #
-        dx, dy, _dw, _db, x, w, m, v, locks,  #
-        x_arg.stride(0), N,  #
-        BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
-        GROUP_SIZE_M=GROUP_SIZE_M,  #
-        num_warps=ctx.num_warps)
-    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
-    # accumulate partial sums in separate kernel
-    _layer_norm_bwd_dwdb[grid](
-        _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
-        BLOCK_SIZE_M=32,  #
-        BLOCK_SIZE_N=128, num_ctas=1)
-    return dx, None, dw, db, None



+ # temporary limitation:
- dtype = torch.float16
+ dtype = torch.float32


x_shape = (M, N)
w_shape = (N, )
weight = torch.rand(w_shape, dtype=dtype, device=DEVICE, requires_grad=True)
bias = torch.rand(w_shape, dtype=dtype, device=DEVICE, requires_grad=True)
x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=DEVICE)


#### test backward ####

# in your case, upstream gradient can come from next torch operation which was called on the output of your kernel ("my_out" below)
upstream = .1 * torch.randn_like(x)

x.requires_grad = True
weight.requires_grad = True
bias.requires_grad = True

+ from triton.backends.autodiff import autodiff

+ my_op, bwd_kernel = autodiff(kernel, stub, grid=(M,), non_stub_args_idxs=[4,5], idx_upstream=1)

+ # temporary limitation: run stub once before calling my_op
+ stub(bwd_kernel, x, weight, bias)

+ my_out = my_op(x, weight, bias)
+ my_out.backward(upstream)

+ # now grads have been populated for: x.grad, weight.grad, bias.grad
```

</details>



# 🔧 Other examples

<!-- **See 10 more examples in [third_party/autodiff/test](backend/autodiff/test)**. -->
For more examples see [third_party/autodiff/test](third_party/autodiff/test).


<!-- 
# How to use it?

TBD: For now see examples at [third_party/autodiff/test](third_party/autodiff/test). -->



---

# Upstream README


<div align="center">
  <img src="https://lh5.googleusercontent.com/wzQKEsTFkrgNQO9JjhGH5wFvslJr1saLtLaJ_a6Fp_gNENpvt3VG7BmztwngU9hFJaU4CPwGiw1opQtDvTkLrxWRbO_a12Q-pdESWHgtmheIHcPbOL5ZMC4TSiJVe5ty1w=w3517" alt="Triton logo">
</div>

| **`Documentation`** | **`Nightly Wheels`** |
|-------------------- | -------------------- |
| [![Documentation](https://github.com/triton-lang/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/) | [![Wheels](https://github.com/triton-lang/triton/actions/workflows/wheels.yml/badge.svg?branch=release/2.0.x)](https://github.com/triton-lang/triton/actions/workflows/wheels.yml) |

# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.  See also these third-party [Triton puzzles](https://github.com/srush/Triton-Puzzles), which can all be run using the Triton interpreter -- no GPU required.

# Quick Installation

You can install the latest stable release of Triton from pip:

```shell
pip install triton
```

Binary wheels are available for CPython 3.9-3.13.

# Enabling Blackwell Support

The main branch now features support for NVIDIA Blackwell GPUs using 5th
generation tensor cores. To enable this, you will need two additional steps:

1. Build a pre-release PyTorch from source with CUDA 12.8
2. Build triton from the latest source


First, to build pytorch you need to have CUDA 12.8 installed locally. If not,
follow the [instructions for your platform](https://developer.nvidia.com/cuda-downloads)
```bash
# Clone and checkout pytorch 2.6 release candidate
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.6.0-rc9
git submodule sync
git submodule update --init --recursive -j 8

# Install build dependencies (assumes you already have a system compiler)
pip install -r requirements.txt
pip install mkl-static mkl-include wheel

# Build PyTorch (will take a long time)
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=$CUDA_HOME
export TORCH_CUDA_ARCH_LIST=Blackwell
python setup.py develop

# Optional, package build into a wheel to install on other machines.
python setup.py bdist_wheel
ls dist  # Wheel should be output in this directory
```

Note that if you use the domain libraries (`torchvision`, `torchtext`,
`torchaudio`, etc.) these will need to be built from source as well, otherwise
their custom PyTorch extensions will not work.

Finally, follow the instructions below to install triton from source.

# Install from source

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

pip install ninja cmake wheel pybind11 # build-time dependencies
pip install -e python
```

Or with a virtualenv:

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install ninja cmake wheel pybind11 # build-time dependencies
pip install -e python
```

# Building with a custom LLVM

Triton uses LLVM to generate code for GPUs and CPUs.  Normally, the Triton build
downloads a prebuilt LLVM, but you can also build LLVM from source and use that.

LLVM does not have a stable API, so the Triton build will not work at an
arbitrary LLVM version.

1. Find the version of LLVM that Triton builds against.  Check
`cmake/llvm-hash.txt` to see the current version. For example, if it says:
       49af6502c6dcb4a7f7520178bd14df396f78240c

   This means that the version of Triton you have builds against
   [LLVM](https://github.com/llvm/llvm-project) 49af6502.

2. `git checkout` LLVM at this revision.  Optionally, make additional
   modifications to LLVM.

3. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run

       $ cd $HOME/llvm-project  # your clone of LLVM.
       $ mkdir build
       $ cd build
       $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
       $ ninja

4. Grab a snack, this will take a while.

5. Build Triton as above, but set the following environment variables.

       # Modify as appropriate to point to your LLVM build.
       $ export LLVM_BUILD_DIR=$HOME/llvm-project/build

       $ cd <triton install>
       $ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e python

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- vscode intellisense has some difficulty figuring out how to build Triton's C++
  (probably because, in our build, users don't invoke cmake directly, but
  instead use setup.py).  Teach vscode how to compile Triton as follows.

    - Do a local build. Run command `pip install -e python`
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find python/build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/python/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`
    - In vscode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe.

```shell
# One-time setup.  Note this will reinstall local Triton because torch
# overwrites it with the public version.
$ make dev-install

# To run all tests (requires a GPU)
$ make test

# Or, to run tests without a gpu
$ make test-nogpu
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Helpful environment variables**

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`
- `MLIR_DUMP_PATH` specifies where `MLIR_ENABLE_DUMP` will dump to. If unset will dump to stderr.
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_REPRODUCER_PATH=<reproducer_path>` will generate an MLIR reproducer file
  at `<reproducer_path>` before each MLIR compiler stage. If any of the stages fail,
  `<reproducer_path>` will be a local MLIR reproducer captured right before the failing pass.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.

  An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.
- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `TRITON_ENABLE_ASAN=1` invokes the LLVM address sanitizer for
  memory leak and out of bounds access detection. Currently only supported on the AMD
  backend. This must be run using the ASAN libraries documented [here](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).

  When enabling the address sanitizer it is recommended to disable various memory caching strategies
  both within the ROCm stack and PyTorch. This will give the address sanitizer the best chance at finding the
  memory fault where it originates. See this [test](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/test/test_address_sanitizer.py) for more details.

- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of allowing fp fusion (mul+add->fma).
- `MLIR_ENABLE_DIAGNOSTICS=<comma-separated>` controls diagnostic emission in MLIR.
  Options are: `warnings`, `remarks`, `stacktraces`, `operations`.
  Use comma-separated values to customize output. For example,
  `MLIR_ENABLE_DIAGNOSTICS=remarks,operations` enables remarks and IR operations,
  while `MLIR_ENABLE_DIAGNOSTICS=warnings,stacktraces` enables warnings with
  stacktraces. By default, only errors are shown. Setting `warnings` includes
  errors and warnings; `remarks` includes errors, warnings, and remarks.
- `MLIR_ENABLE_REMARK` is deprecated. Please use `MLIR_ENABLE_DIAGNOSTICS=remarks`.
- `TRITON_KERNEL_DUMP` enables the dumping of the IR from each compilation stage and the final ptx/amdgcn.
- `TRITON_DUMP_DIR` specifies the directory to save the dumped IR and ptx/amdgcn when `TRITON_KERNEL_DUMP` is set to 1.
- `TRITON_KERNEL_OVERRIDE` enables the override of the compiled kernel with a user-specified IR/ptx/amdgcn at the beginning of each compilation stage.
- `TRITON_OVERRIDE_DIR` specifies the directory from which to load the IR/ptx/amdgcn files when `TRITON_KERNEL_OVERRIDE` is set to 1.
- `TRITON_F32_DEFAULT` sets the default input precision of `tl.dot` when using 32-bit floats, which can be either `ieee`, `tf32`, or `tf32x3`.

**Kernel Override Steps**

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
# Step 1: Run the kernel once to dump kernel's IRs and ptx/amdgcn in $TRITON_DUMP_DIR
# Step 2: Copy $TRITON_DUMP_DIR/<kernel_hash> to $TRITON_OVERRIDE_DIR
# Step 3: Delete the stages that you do not want to override and modify the stage you do want to override
# Step 4: Run the kernel again to see the overridden result
```


# Changelog

Version 2.0 is out! New features include:

- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/triton-lang/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).

# Compatibility

Supported Platforms:

- Linux

Supported Hardware:

- NVIDIA GPUs (Compute Capability 8.0+)
- AMD GPUs (ROCm 6.2+)
- Under development: CPUs

# Development Container (Dev Container)

**Dev Containers** for the Triton project are available from
the [triton-dev-containers repository](https://github.com/redhat-et/triton-dev-containers)

### Key Benefits:
- **Consistency**: All developers can work with the same development
  environment, ensuring uniform behavior across different systems.
- **Isolation**: The container prevents potential conflicts with software
  installed on your local machine.
- **Portability**: Easily share the development environment with team members,
  minimizing onboarding time and setup issues.

### How to Use the Dev Container:

For detailed instructions on how to use the dev containers please see
the [dev container user guide](https://github.com/redhat-et/triton-dev-containers/blob/main/.devcontainer/devcontainer.md)
