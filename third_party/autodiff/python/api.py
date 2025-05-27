import sys
import os
os.environ['TRITON_ALWAYS_COMPILE']='1'
import hashlib
import inspect
import subprocess
from functools import partial

import torch
torch.manual_seed(0)
DEVICE = torch.device("cuda:0")

import triton
import triton.language as tl
from triton.runtime import driver
from triton.runtime.jit import JITFunction


VERBOSE = int(os.environ.get('VERBOSE', 0))
assert VERBOSE in [0, 1, 2]

dir = os.getenv("TRITON_AUTODIFF_DIR")
if dir is None:
    raise ValueError("Please specify TRITON_AUTODIFF_DIR, see README.")

# todo: don't hardcode
tool = f"{dir}/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt"


def run_mlir_pass(path):

  os.makedirs(path, exist_ok=True)

  # produce bwd ttir
  with open(f"{path}/out.ttir", "w") as f:
    subprocess.run([tool, "--convert-triton-to-autodiff", "--mlir-print-debuginfo", f"{path}/inp.ttir"], stdout=f)

  if VERBOSE >= 1:
    # optionally, produce readable fwd ttir

    # with open(f"{path}/_inp_readable.ttir", "w") as f:
    #   subprocess.run([tool, "--mlir-use-nameloc-as-prefix", "--mlir-print-debuginfo", f"{path}/inp.ttir"], stdout=f)

    # this is a bit ugly but needed bc fwd.py files create out.ttir files with default SSA names (%1, %2, ...)
    # and with location info (containing variable names). Here I ran "--mlir-use-nameloc-as-prefix" on it and write
    # to the same files to avoid creating redundant files
    with open(f"{path}/inp.ttir", "r+") as f:
        content = f.read()         # Read existing content
        f.seek(0)                  # Move cursor to the beginning
        # Overwrite from the start
        subprocess.run([tool, "--mlir-use-nameloc-as-prefix", "--mlir-print-debuginfo", f"{path}/inp.ttir"], stdout=f)
        f.truncate()               # Remove remaining old content

    if VERBOSE == 2:

      def draw_dot(path, mode):
        assert mode in ["fwd", "bwd"]

        vis_dir = path + "/vis"
        os.makedirs(vis_dir, exist_ok=True)

        # a. optionally, produce vis dot
        with open(f"{vis_dir}/{mode}.dot", "w") as f:
          ttir_path = f"{path}/inp.ttir" if mode == "fwd" else f"{path}/out.ttir"
          subprocess.run([tool, "-mlir-use-nameloc-as-prefix", "--view-op-graph", ttir_path], stderr=f,
                        # suppress stdout, otherwise prints _inp_readable again
                        stdout=subprocess.DEVNULL)

        with open(f"{vis_dir}/{mode}.svg", "w") as f:
          subprocess.run(["dot", "-Tsvg", f"{vis_dir}/{mode}.dot"], stdout=f)

        # b. optionally cluster nodes
        subprocess.run(["python", "cluster_dot.py", "--strict", f"{vis_dir}/{mode}.dot", f"{vis_dir}/{mode}_grouped.dot"])

        with open(f"{vis_dir}/{mode}_grouped.svg", "w") as f:
          subprocess.run(["dot", "-Tsvg", f"{vis_dir}/{mode}_grouped.dot"], stdout=f)

        # todo: fails when running be tests (from another dir)
        # os.remove(f"{vis_dir}/{mode}.dot")
        # os.remove(f"{vis_dir}/{mode}_grouped.dot")

      # optionally, produce vis dot
      draw_dot(path, mode="fwd")

      # optionally, produce vis dot
      draw_dot(path, mode="bwd")





def create_new_jitfn(jit_func):
    assert isinstance(jit_func, JITFunction)

    # Create a new JITFunction with the same base function and parameters
    new = JITFunction(
        jit_func.fn,
        version=jit_func.version,
        do_not_specialize=jit_func.do_not_specialize,
        do_not_specialize_on_alignment=jit_func.do_not_specialize_on_alignment,
        debug=jit_func.debug,
        noinline=jit_func.noinline,
        repr=jit_func._repr,
        launch_metadata=jit_func.launch_metadata
    )

    new._autodiff_info = []

    # Copy any pre-run hooks
    new.pre_run_hooks = list(jit_func.pre_run_hooks)

    return new

# it's not as much as a stub, but more like helper to wrap_bwd_kernel from kernel_inputs -- the true stub is the user thing, this thing just piggy backs on the true stub
def wrap_bwd_kernel(fwd_kernel, bwd_kernel, idxs_buffers, grid, kernel_inputs, all_upstream):


    # # todo: understand more
    # # these kwargs are injected automatically
    # # launch‑options, not ordinary kernel parameters
    # _kwargs = kwargs.copy()
    # _kwargs.pop("debug", None)
    # _kwargs.pop("num_warps", None)
    # _kwargs.pop("num_ctas", None)
    # # enable_fp_fusion
    # # launch_cooperative_grid

    # todo-low: add input checks
    # assert a.device == DEVICE and b.device == DEVICE and upstream.device == DEVICE

    # todo-high: [support re-tracing]
    # for now still relying on bwd_kernel._autodiff_info, but for re-tracing seems need a more general
    # appraoch (supports interleaving calls to different CompiledKernels in the cache), but the bwd_kernel._autodiff_info is
    # more limited as it only keeps info about the last CompiledKernel -- last compiled != last called
    #   Thus passing fwd_kernel here
    #     to extract it's cache directly;
    #     then find what CompiledKernel in the fwd pass corresponds to the current args (kernel_inputs);
    #     and then extract what idx were specialized in fwd CompiledKernel directly from that CompiledKernel
    #  passing fwd_kernel to support re-tracing: to be able to dynamically figure which FWD
    #   compileKernel does the kernel_inputs (passed to the current fn) correspond to, to extract what inputs were
    #   specized in the fwd -- so that you don't need to pass that specialized idx (for a particular fwd CompiledKernel)
    #   through some global dict

    # if VERBOSE:  print("[wrap_bwd_kernel] kernel_inputs", kernel_inputs)

    # fwd specializes away some arguments (so that they aren't arguments in the fwd TTIR,
    # and thus not arguments in bwd TTIR as well) -- so don't pass them to bwd TTIR
    if VERBOSE: print("[wrap_bwd_kernel] bwd_kernel._autodiff_info: ", bwd_kernel._autodiff_info)
    idx_folded = bwd_kernel._autodiff_info[-1]
    if VERBOSE: print("[wrap_bwd_kernel] idx_folded: ", idx_folded)
    # user provided idx of outputs (idx_folded) in terms of all args to fwd python kernel
    # (JITFunction), when it compiled, some of the args potentially got specialized away.
    # Thus here need to shift that user specified index to account for these args (that got
    # specialized away) if they were located before the user provided idx_upstream

    # reverse to prevent shifting issues when popping;
    for i in reversed(sorted(idx_folded)):
        kernel_inputs.pop(i)


    # for upstream each idx, shift it by how many args
    # before it has been folded

    # sort because usr can pass idxs in arbitrary order
    idxs_buffers = reversed(sorted(idxs_buffers))

    shifted_idxs_buffers = []
    for idx in idxs_buffers:
        num_folded_before = sum(x < idx for x in idx_folded)
        idx_shifted = idx - num_folded_before
        shifted_idxs_buffers.append(idx_shifted)
        print(f"output-buffer at idx {idx} was shifted by {num_folded_before}")

    # if VERBOSE: print("[wrap_bwd_kernel] idx_folded", idx_folded)
    # if VERBOSE: print("num_folded_before_upstream:", num_folded_before_upstream)

    print("all_upstream: ", all_upstream)

    # pass (from the AG.bwd inputs) upstream grads wrt to all (not just one) outputs
    bwd_args = []
    # idx_upstream = 0
    for i, arg in enumerate(kernel_inputs):
        # grad wrt an output -- fill with upstream
        if i in shifted_idxs_buffers:
            bwd_args.append(all_upstream[i]) # all_upstream[idx_upstream]
            # idx_upstream += 1
            continue
        # grad wrt an input -- fill with zeros
        if isinstance(arg, torch.Tensor):
            bwd_args.append(torch.zeros_like(arg))

    # todo:
    #   some err handling for weird cases where fwd JITFcuntio has't ran
    #   with that signature yet -- so my wrapping didnt' take place -- so
    #   calling the bellow will fail

    # if VERBOSE: print("[wrap_bwd_kernel] fwd_args:", kernel_inputs)
    if VERBOSE: print("[wrap_bwd_kernel] bwd_args:", bwd_args)

    bwd_kernel[grid](*kernel_inputs, *bwd_args)
    # bwd_kernel.run(grid=grid, warmup=False, *kernel_inputs, *bwd_args)

    if VERBOSE: print("[wrap_bwd_kernel] bwd_args (after calling bwd_kernel): ", bwd_args)

    # don't need to pop grad wrt upstream anymore bc now (when my AG.Function works on lvl of kernels, but not stubs)
    # I actually do need to return grads wrt each of the inputs of the kernel inputs (including the out buffers):
    #
    # # remove upstream grad
    # # Use num_folded_before_upstream, otherwise assumes all args are tensors (IOW: grad inputs are 1:1 with
    # # inputs) -- but it's not always the case, so this causes idx (in terms of fwd args) not match idx (in terms of
    # # grad args)
    # bwd_args.pop(idx_upstream - num_folded_before_upstream)

    # print("[wrap_bwd_kernel] bwd_args (after popping grad wrt out): ", bwd_args)

    # #   capture stub and wrap_bwd_kernel args by closure -- instead of passing them as inputs to forward() -- otherwise autograd requires to return same number of grads
    # #   cannot just pass "def forward(ctx, stub, wrap_bwd_kernel, *stub_inputs)" and later bind stub and wrap_bwd_kernel -- bc even if bind and thus won't need to feed them them at runtime, autograd expect I should return 4 args (as the number of args to autograd.Function.forward)
    # #   E.g. for flash aten kernel user fwd stub creates some additional tensor args and passes them to the kernel (e.g. M, OUT) and feeds them to the kernel but the user calls stub with "my_op(q, k, v)" -- so the autograd.Function.forward also only expects "q, k, v"
    # #       but kernel actually sees (q, k, v, M, OUT, [other non-tensor arguments]) -- and your wrap_bwd_kernel create grad tensors for all tensor arguments (to feed to bwd_kernel) and then returns all added grad_tensor arguments
    # #       (which would also contain grad_M, grad_OUT) but because these M or OUT weren't passed to the autograd.Function.forwad, in autograd.Function.backward, it's incorrect to return grads wrt these values
    # #       so need a way to only return grad wrt fwd stub args (and NOT wrt all bwd_kernel tensor args)
    return (*bwd_args,)


def my_post_hook(key, repr, fn, compile, is_manual_warmup, already_compiled):

    def key_add_args(key):

        # can't run the binder to automatically create specialization and options (both needed to create key)
        # bc here is that I don't have acces to *arg, **kwargs from inside the compile_hook
        # there doesn't seem to be a direct way to extract the full args and kwargs from the compile hook
        # -- thus doing the appraoch below
        # # _bound_args, specialization, options = binder(*args, **kwargs)

        if VERBOSE: print("[key_add_args] key: ", key)
        # key:  [('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D')]{'debug': False}

        split = key.split("]")
        if VERBOSE: print("[key_add_args] split: ", split)
        # split:  ["[('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D')", "{'debug': False}"]

        new_key = split[0] + ", "
        for name, str_type in compile_dict["signature"].items():
            if "*" in str_type:
                # todo-low: don't hardcode D
                # Right now Triton only seem to define two single‑letter tags 
                # D	-- BaseBackend.get_arg_specialization – given to any int or tensor pointer whose value / address is divisible by 16 when align=True is in force. Produces tt.divisibility = 16, i.e. the backend may assume 16‑byte alignment.
                # S	-- HIPBackend.get_arg_specialization (AMD GPUs) when buffer‑ops are on and the tensor’s storage fits in ±2 GB. Adds tt.pointer_range = 32, telling the compiler it can emit 32‑bit (small) addresses.
                new_key += f"('{str_type}', 'D'), "
        # cut ", "
        new_key = new_key[:len(new_key)-2]
        # add what I split by
        new_key += "]"
        new_key += split[1]

        if VERBOSE: print("[key_add_args] new_key: ", new_key)
        # new_key:  [('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D'), ('*f32', 'D'), ('*f32', 'D'), ('*f32', 'D')]{'debug': False}
        return new_key

    def rebuild_binder(jit_fn, delta, backend):
        """
        1 – update signatures	create_function_from_signature looks at fn.jit_function.signature and fn.jit_function.metadata.arg_types. If you forget to patch either, the binder will still have the old arity and you will hit TypeError: dynamic_func() takes N positional arguments… at launch 
        2 – new binder	The helper builds the little Python function (dynamic_func) that maps user launch args → positional tuple for the GPU call
        """

        if VERBOSE: print("rebuild_binder")
        import inspect

        from triton.runtime.jit import (
            KernelParam,                        # helper for arg metadata
            create_function_from_signature,     # binder factory
        )

        if VERBOSE: print(f"adding {delta} args")

        # 1. extend the Python signature *before* we rebuild the binder
        sig_params = list(jit_fn.signature.parameters.values())
        for i in range(delta):
            p = inspect.Parameter(
                f"grad_{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation="tl.float16*" # "tl.pointer"
            )
            jit_fn.params.append(KernelParam(len(jit_fn.params), p, False, False)) # dns= dns_oa= , annotation="tl.float16*"
            sig_params.append(p)
        if VERBOSE: print("jit_fn.signature: ", jit_fn.signature)
        if VERBOSE: print("jit_fn self.params:", jit_fn.params)
        jit_fn.signature = jit_fn.signature.replace(parameters=sig_params)

        # 2. build a fresh binder
        new_binder = create_function_from_signature(
                        jit_fn.signature,
                        jit_fn.params,
                        backend)

        return new_binder


    jit_fn = fn.jit_function  # JITFunction

    # compile hook registers on all instances of JITFunction, but i want this hook to trigger only on fwd JITFunctons
    if not already_compiled and hasattr(jit_fn, '_is_fwd_kernel'):

        if VERBOSE: print("[my hook] compile hook triggered on the fwd JITFunction!")

        compile_dict = compile
        bwd_jit_fn = jit_fn._bwd_kernel

        if VERBOSE: print(f"Kernel {fn.name} just finished executing!")
        if VERBOSE: print(f"Representation: {repr}")

        # The fn parameter passed to the hook contains a jit_function attribute that refers to the JITFunction instance.
        # Each JITFunction keeps its kernels in device_caches[device], which is a tuple where the first element is the kernel cache dictionary.
        # The same key that's passed to the hook is the one used to store the kernel in the cache.

        # 1) extract fwd_compiled_kernel

        # Access the kernel from the cache
        device = driver.active.get_current_device()
        fwd_kernel_cache, target, backend, _binder = jit_fn.device_caches[device]
        bwd_kernel_cache, target, backend, _binder = bwd_jit_fn.device_caches[device]

        # Get the kernel using the same key
        fwd_compiled_kernel = fwd_kernel_cache[key]

        # 2) write fwd IR

        # todo: cleanup
        # the "key" arg is just a python string with input signatures of the kernel
        # but I want some folder name -- one way is to hash it
        hash_object = hashlib.sha256(key.encode())
        dir_name = hash_object.hexdigest()[:10]
        if VERBOSE: print("dir_name: ", dir_name)

        os.makedirs(f"generated/{dir_name}", exist_ok=True)
        with open(f"generated/{dir_name}/inp.ttir", "w") as f:
          f.write(fwd_compiled_kernel.asm['ttir'])

        # 3) autodiff
        run_mlir_pass(f"generated/{dir_name}")

        # 4) create callable python fn for bwd
        from triton.compiler import compile
        from triton.backends.compiler import GPUTarget

        bwd_compiled_kernel = compile(
            f"generated/{dir_name}/out.ttir",
            target=target,
            # preserve the original CompiledKernel.options so Triton does not pick a different PTX flavour
            # options={k: compile_dict[k] for k in BACKEND_OPTS if k in compile_dict}
        )
        assert isinstance(bwd_compiled_kernel, triton.compiler.compiler.CompiledKernel)
        # question-now: seems to automatically lowered to ttgir not ttir
        # if VERBOSE: print(bwd_compiled_kernel.asm.keys())

        # good, I can confirm this is my autodiff'ed IR
        # if VERBOSE: print(bwd_compiled_kernel.asm['ttgir'])


        # 5) keep original fwd CompiledKernel with autograd.Function and add corresponding cache entry (with differentiated CompiledKernel) to cache of bwd JITFunction

        # if VERBOSE: print(dir(jit_fn.device_caches[device][0][key]))
        # '_init_handles', 'asm', 'function', 'hash', 'kernel', 'launch_enter_hook', 'launch_exit_hook', 'launch_metadata', 'metadata', 'module', 'name', 'packed_metadata', 'src'

        #   5.1. remove constexpr from: key, signature, params

        def remove_constexpr(bwd_jit_fn, key):
            if VERBOSE: print("[remove_constexpr] key: ", key)
            # need to also modify self.signature bc it's used in create_binder -> create_function_from_signature
            #   > inf JITFunction.run "binder = create_function_from_signature(self.signature, self.params, backend)"
            if VERBOSE: print("[remove_constexpr] bwd_jit_fn.signature:", bwd_jit_fn.signature)
            if VERBOSE: print("[remove_constexpr] bwd_jit_fn.params:", bwd_jit_fn.params)

            # remove constexpr -- bc backward signature or key should not have them (bc they will NOT be provided to the bwd kernel)
            # remove ", ('constexpr', 4)"
            # key = key.replace(", ('constexpr', 4)", "")

            # replaces all occurrences of , ('constexpr', [some integer]) in the string
            sub_strs = key.split("[")[1].split("]")[0].split("), (")
            # if VERBOSE: print('sub_strs: ', sub_strs)
            # >>> sub_strs:  ["('*fp32', 'D'", "'*fp32', 'D'", "'constexpr', 4", "'*fp32', 'D'", "'*fp32', 'D')"]

            # iterate over dict whose keys are tuples of ints, and extract all ints from all keys into a single list
            idx_const_ints = [i for key in compile_dict['constants'].keys() for i in key]
            num_const_args = len(idx_const_ints)
            print("[remove_constexpr] idx_const_ints", idx_const_ints)

            sig_params = list(fn.jit_function.signature.parameters.values())
            # reverse to avoid shifting issues
            for i, s in reversed(list(enumerate(sig_params))):
                if i in idx_const_ints:
                    sub_strs.pop(i)
                    sig_params.pop(i)
                    bwd_jit_fn.params.pop(i)
            new_key = "[" + "), (".join(sub_strs)
            new_key += "]" if new_key[-1] == ")" else ")]"
            new_key += key.split("]")[1]
            if VERBOSE: print("[remove_constexpr] new_key", new_key)
            bwd_jit_fn.signature = bwd_jit_fn.signature.replace(parameters=sig_params)
            if VERBOSE: print("[remove_constexpr] bwd_jit_fn.signature: ", bwd_jit_fn.signature)
            if VERBOSE: print("[remove_constexpr] bwd_jit_fn self.params:", bwd_jit_fn.params)

            return new_key, num_const_args

        new_key, num_const_args = remove_constexpr(bwd_jit_fn, key)

        #   5.2. add new args to: key, signature, params

        # add new args to key
        new_key = key_add_args(new_key)
        # key [('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D')]{'debug': False}
        # new key [('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D')]{'debug': False}

        # need to account for, otherwise may not correctly count differences in args, bc fwd_compiled_kernel has constexprs (in its signature) while bwd_compiled_kernel does not!
        #   >> fwd_compiled_kernel.src.signature:  {'x_ptr': '*fp32', 'output_ptr': '*fp32', 'BLOCK_SIZE': 'constexpr'}
        #   >> bwd_compiled_kernel.src.signature:  {0: '*f32', 1: '*f32', 2: '*f32', 3: '*f32'}

        num_fwd_args = len(fwd_compiled_kernel.src.signature) - num_const_args
        num_bwd_args = len(bwd_compiled_kernel.src.signature)
        num_added_args = num_bwd_args - num_fwd_args

        new_binder = rebuild_binder(bwd_jit_fn, num_added_args, backend)
        if VERBOSE: print("new_binder: ", new_binder)

        if VERBOSE: print("fn.jit_function.signature:", fn.jit_function.signature)
        if VERBOSE: print("fn.jit_function.params:", fn.jit_function.params)


        #   5.3. add to bwd CompiledKernel into the cache

        # keep forward cache entry as is, don't delete it

        # previously I incorrectly stored at the same key -- so the grad fn is basically keyed on singatures to the fwd kernel
        # key on a new_key (containing added args) not on the old key, otherwise:
        #   when you pass e.g. 6 args to the bwd JITFunction on the next call, it checks the cache for CompiledKernel with signature which has 6 args -- didn't find one (bc here you're storing the bwd CompiledKernel under the *original key* which only has 3 args) and thus re-compiles
        #   JITFunction looking for key: [('*fp32','D'), … 6 items …]{'debug':False}
        # IOW, it was re-wrapping bc previously (in my post hook) I saved bwd graph while key'ing on the original signature (3 args). But now when passing 6 args -- it fails to find compiledKerenl with a key which has 6 args and thus re-compiles
        bwd_kernel_cache[new_key] = bwd_compiled_kernel
        if VERBOSE: print("bwd_kernel_cache[new_key]: ", bwd_kernel_cache[new_key])
        bwd_jit_fn.device_caches[device] = (bwd_kernel_cache, target, backend, new_binder)

        # question-now: recomputing them?
        # s.non_constexpr_indices = [i for (i, p) in enumerate(s.params) if not p.is_constexpr] # RM
        # s.specialised_indices = [i for (i, p) in enumerate(s.params) if (not p.do_not_specialize) and (not p.is_constexpr)] # RM

        # todo: Iterate over bwd_jit_fn.device_caches.keys() and duplicate the backward kernel per device, or error out if torch.cuda.current_device() differs between forward and backward.
        # # if multiple gpus, patch caches of all devices?
        # for d, (cch, t, b, _) in bwd_jit_fn.device_caches.items():
        #     if key in cch:
        #         cch[key] = new_k
        #         bwd_jit_fn.device_caches[d] = (cch, t, b, new_binder)

        if VERBOSE: print("[my_hook] compile_dict['signature']", compile_dict["signature"])
        if VERBOSE: print("[my_hook] compile_dict['constants']", compile_dict["constants"])
        if VERBOSE: print("[my hook] jit_fn.params:", jit_fn.params)
        if VERBOSE: print("fn.jit_function.signature.parameters", fn.jit_function.signature.parameters)


        # recover all arguments that have been folded into the CompiledKernel (need for later removal
        # of these folded args from fwd_kernel_args inside wrapped_bwd_kernel before passing them bwd_kernel);
        # includes compile‑time constants (declared `constexpr`) AND automatically specialised (ints/bools/tuples …)
        folded = list(p[0] for p in compile_dict["constants"])
        names = [fn.jit_function.arg_names[i] for i in folded]
        if VERBOSE: print("Hard‑coded parameter indices:", sorted(folded))
        if VERBOSE: print("Hard‑coded parameter names:  ", names)
        # todo-now: ugly
        bwd_jit_fn._autodiff_info.append(folded)
        print("setting _autodiff_info", bwd_jit_fn._autodiff_info)

    return False



# dont set it inside autograd fn, but rather set it here on module lvl (triggers at import)
# Bc I'm registering hooks on the fwd JITFucntions -- so I need to register my hook
# before the first user invocation of fwd JITFucntions (so that my wrapping can see
# as many fwd signatures as possible)
triton.runtime.jit.JITFunction.compiled_hook = my_post_hook


class DifferentiatedCompiledKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, kernels, idxs_buffers, grid, *fwd_kernel_inputs):
        if VERBOSE: print("\n"*3, "Op.forward")

        fwd_kernel, wrapped_bwd_kernel = kernels

        # todo:
        # bc calling the fwd kernel will write output inplace of some original arguments (the
        # ones which are output buffers) -- but my bwd expects a cleanly initialized buffers
        #  - clone them and store on the ctx BEFORE they got overwritten by calling the fwd kernel?
        #  - actually maybe even better: on cpp side remove the entire fwd subgraph leaning the ouput?
        #    Bc fwd result which I re-compute during bwd will not be used anywhere anyway -- wasted computations

        # if VERBOSE: print("[Op.forward] fwd_kernel_inputs (before calling kernel)", *fwd_kernel_inputs)
        # calling wrap_bwd_kernel inside AG is also nice because clary separates roles:
        #   - diff'ing everything what's called inside this AG class (kernel) -- I take care of
        #   - diff'ing anything outside this class (i.e. stub logic including pre/post processing) -- torch's responsibility
        # outs = fwd_kernel[grid](*fwd_kernel_inputs)
        fwd_kernel[grid](*fwd_kernel_inputs)
        # self.run(grid=grid, warmup=False, *args, **kwargs)

        # if VERBOSE: print("[Op.forward] fwd_kernel_inputs (after calling kernel)", *fwd_kernel_inputs)

        # todo: in AG.backward you can return none wrt all of them
        for i in idxs_buffers:
            ctx.mark_dirty(fwd_kernel_inputs[i])

        # if VERBOSE: print("saving for bwd: ", fwd_kernel_inputs)
        # ugly workaround because save_for_backward only works for tensor inputs
        tensor_fwd_kernel_inputs = [a for a in fwd_kernel_inputs if isinstance(a, torch.Tensor)]
        ctx.save_for_backward(*tensor_fwd_kernel_inputs)
        ctx.non_tensor_inputs = [a for a in fwd_kernel_inputs if not isinstance(a, torch.Tensor)]
        ctx.arg_types = [isinstance(a, torch.Tensor) for a in fwd_kernel_inputs]

        ctx.wrapped_bwd_kernel = wrapped_bwd_kernel
        ctx.grid = grid

        # need to return values from AG.fwd, even though caller discards output of this fn ("kernel[...]" -- the result is not assigned by the caller)
        #
        # i think: When the engine attaches grad_fn to the returned tensor it also sets
        #   requires_grad=True automatically if any upstream input in the graph needs
        #   gradients. So the buffer now participates in back-prop exactly as a normal
        #   output would, even though Python code in the stub (caller of this Autograd.forward)
        #   discards the return value
        #
        # return only tensor inputs I guess -- bc for everything I return here, I believe torch will try
        #   to attached "grad_fn=DifferentiatedCompiledKernel" -- which wouldn't make sense for non
        #   tensor args (e.g. BLOCK_SIZE=4 kwargs to the kernel)
        return (*tensor_fwd_kernel_inputs, )


    @staticmethod
    def backward(ctx, *all_upstream):
        if VERBOSE: print("\n"*3, "Op.backward")

        # because in AG.fwd I'm returning ALL the kernel tensor args, in the AG.bwd,
        # I need to input grads wrt ALL OF THEM -- not just grad wrt the single
        # output returned form the user stub;
        # AG.bwd expects same num of args (upstream grads) as the outputs of AG.fwd

        # reconstruct all fwd kernel args
        fwd_kernel_inputs = []
        tensor_idx = 0
        non_tensor_idx = 0
        for is_tensor in ctx.arg_types:
            if is_tensor:
                fwd_kernel_inputs.append(ctx.saved_tensors[tensor_idx])
                tensor_idx += 1
            else:
                fwd_kernel_inputs.append(ctx.non_tensor_inputs[non_tensor_idx])
                non_tensor_idx += 1

        # todo-now: note output buffers (in this case at idx=1) have my grad_fn attached -- but in AG.bwd I'll be runnign kernel on them again: undesirable?
        # if VERBOSE: print("[Op.backward] reconstructed fwd_kernel_inputs", fwd_kernel_inputs)
        # > tensor([...], device='cuda:0', requires_grad=True),
        # > tensor([...], device='cuda:0', grad_fn=<DifferentiatedCompiledKernelBackward>)]

        # wrapped_bwd_kernel does return grad_inputs that it initialized, after running the
        # generated_bwd_kernel these are populated -- and contain grads wrt original tensor inputs
        grads = ctx.wrapped_bwd_kernel(ctx.grid, fwd_kernel_inputs, all_upstream)
        if VERBOSE: print("[Op.backward] grads", grads)

        # at this point your "grads" value has grads wrt args of fwd kernel (including grad wrt kernel out itself
        # -- not popping it in wrap_bwd_kernel because here I'm required to return grads wrt ALL inputs which passed to AG.fwd)
        all_outs = []
        tensor_idx = 0
        for i, is_tensor in enumerate(ctx.arg_types):
            if is_tensor:
                all_outs.append(grads[tensor_idx])
                tensor_idx += 1
            else:
                all_outs.append(None)

        return (None, None, None, *all_outs, )



# todo-low: can determine automatically:
#   - in AG.fwd -- run kernel once and see which inputs were changed as result of executing kernel;
#   - Or, in mlir pass output idx of all inputs which are used in store nodes
def autodiff(idxs_buffers):

    def helper(spec, kernels, idxs):
        # returns a subclass of `DifferentiatedCompiledKernel` whose .apply accepts keywords;
        # DifferentiatedCompiledKernel.forward signature staying (ctx, *args, **kwargs) -- fine as long as it can consume the ordered list I pass in

        target_sig = inspect.signature(spec)
        params = target_sig.parameters.values()
        if VERBOSE: print("[helper] target_sig", target_sig)
        if VERBOSE: print("[helper] params", params)

        class _Helper(DifferentiatedCompiledKernel):
            # __signature__ = target_sig          # IDE/help friendly
            __doc__       = DifferentiatedCompiledKernel.__doc__
            __name__      = DifferentiatedCompiledKernel.__name__
            __qualname__  = DifferentiatedCompiledKernel.__qualname__

            @classmethod
            def apply(cls, *args, **kwargs):
                # if VERBOSE: print("[_Helper.apply] args", args)
                if VERBOSE: print("[_Helper.apply] kwargs", kwargs)
                # if VERBOSE: print("[_Helper.apply] grid", cls.grid)
                bound = target_sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                # fixed positional order for c++ apply
                ordered = [bound.arguments[p.name] for p in params]
                return super().apply(kernels, idxs, cls.grid, *ordered)

            # operates on the class level -- not on the instances of the class,
            # because in pytorch instances of torch.autograd.Function never created
            @classmethod
            def __class_getitem__(cls, grid):
                # mutate the existing class
                cls.grid = grid
                return cls.apply

        return _Helper

    def inner(fwd_kernel):

        nonlocal idxs_buffers
        assert isinstance(idxs_buffers, (tuple, int)), f"idxs_buffers must be either tuple or int, got {type(idxs_buffers)}"
        # make the most common case slightly more convenient for usr
        if isinstance(idxs_buffers, int):
            idxs_buffers = (idxs_buffers, )

        # new object, empty cache -- each JITFunction instance owns its own device_caches dict
        bwd_kernel = create_new_jitfn(fwd_kernel)

        # device = torch.cuda.current_device()
        # print("[autodiff] bwd_kernel cache:", bwd_kernel.device_caches[device][0])

        # allows to associate a bwd JITFcuntion with this specific fwdKernel
        # so that, from inside the compile hook (which will be triggered on the fwd JITFunciton)
        # I can install the bwd CompiledKernel **on the backward JITFcuntion** (NOT fwd JITFcuntion)
        fwd_kernel._bwd_kernel = bwd_kernel

        # I'm registering compile hook on all JITFunction[s] (both forward and backward);
        # this flag is needed to be able to early exit from the hook (avoids triggering
        # the autograd machinery on already differentiated kernels)
        fwd_kernel._is_fwd_kernel = True

        wrapped_bwd_kernel = partial(wrap_bwd_kernel, fwd_kernel, bwd_kernel, idxs_buffers)
        kernels = (fwd_kernel, wrapped_bwd_kernel)

        # takes the signature form fwd_kernel's python fn
        op = helper(fwd_kernel.fn, kernels, idxs_buffers)

        # avoid user needing to pass grid parameter explicitly
        # because wrapped_bwd_kernel will called from inside the user stub inplace of the original
        # kernel -- it will be called like so "wrapped_bwd_kernel[grid](...)";
        # make_indexable is needed to enable wrapped_bwd_kernel support this calling convention
        return op

    return inner
