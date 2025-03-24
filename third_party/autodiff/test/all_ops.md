Programming Model
-----------------
    program_id
    num_programs


Creation Ops
------------
    arange
    cat
    full
    zeros
    zeros_like
    cast


Shape Manipulation Ops
----------------------
    broadcast
    broadcast_to
    expand_dims
    interleave
    join
    permute
    ravel
    reshape
    split
    trans
    view


Memory/Pointer Ops
----------
    load
    store
    make_block_ptr
    advance


Indexing Ops
------------
    flip
    where
    swizzle2d


Math Ops
--------
    abs
    cdiv
    ceil
    clamp
    cos
    div_rn
    erf
    exp
    exp2
    + fdiv
    floor
    fma
    log
    log2
    maximum
    minimum
    rsqrt
    sigmoid
    sin
    softmax
    sqrt
    sqrt_rn
    umulhi


Linear Algebra Ops
------------------
    dot
    dot_scaled





- # later

  Reduction Ops
  -------------
      argmax
      argmin
      max
      min
      reduce
      sum
      xor_sum

  Scan/Sort Ops
  -------------
      associative_scan
      cumprod
      cumsum
      histogram
      sort
      gather

  Iterators
  -----------------
      range
      static_range

  Atomic Ops
  ----------
      atomic_add
      atomic_and
      atomic_cas
      atomic_max
      atomic_min
      atomic_or
      atomic_xchg
      atomic_xor

  Random Number Generation
  ------------------------
      randint4x
      randint
      rand
      randn



- # won't support

  Inline Assembly
  -----------------
      inline_asm_elementwise


  Compiler Hint Ops
  -----------------
      assume
      debug_barrier
      max_constancy
      max_contiguous
      multiple_of


  Debug Ops
  -----------------
      static_print
      static_assert
      device_print
      device_assert