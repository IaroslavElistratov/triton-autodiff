import sys
import os
os.environ['TRITON_ALWAYS_COMPILE']='1'
import hashlib
import subprocess
from functools import partial
from collections import defaultdict

import torch
torch.manual_seed(0)
DEVICE = torch.device("cuda:0")

import triton
import triton.language as tl
from triton.runtime import driver
from triton.runtime.jit import JITFunction


VERBOSE = int(os.environ.get('VERBOSE', 0))
assert VERBOSE in [0, 1, 2]

dir = "/home/iaro/Desktop/my_triton_autodiff/working_files/triton"
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





bwd_kernel = None

class StashArgsCtx:
    _active_instance = None

    def __enter__(self):
        type(self)._active_instance = self
        self.args = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        type(self)._active_instance = None


# hook is called on the Python side with the same *args, **kwargs you pass when you launch the kernel
# to keep track of the shapes of the tensors (seems by default this info is not accessible form the bwd hook alone)
def shape_track_hook(*args, **kwargs):

    # fwd_shapes.clear()

    ctx = StashArgsCtx._active_instance
    if ctx is not None:
        _kwargs = kwargs.copy()
        # these kwargs are injected automatically
        # launch‑options, not ordinary kernel parameters
        _kwargs.pop("debug", None)
        # todo-high: understand more
        _kwargs.pop("num_warps", None)
        _kwargs.pop("num_ctas", None)
        # enable_fp_fusion
        # launch_cooperative_grid

        ctx.args = [*args, *_kwargs] # [weakref.ref(arg) for arg in args]
        # if VERBOSE: print("[shape_track_hook] kwargs", _kwargs)



def clone_jit_function(jit_func):
    assert isinstance(jit_func, JITFunction)

    # Create a new JITFunction with the same base function and parameters
    cloned = JITFunction(
        jit_func.fn,
        version=jit_func.version,
        do_not_specialize=jit_func.do_not_specialize,
        do_not_specialize_on_alignment=jit_func.do_not_specialize_on_alignment,
        debug=jit_func.debug,
        noinline=jit_func.noinline,
        repr=jit_func._repr,
        launch_metadata=jit_func.launch_metadata
    )

    cloned._autodiff_info = []

    # Copy any pre-run hooks
    cloned.pre_run_hooks = list(jit_func.pre_run_hooks)

    return cloned


# it's not as much as a stub, but more like helper to create_bwd_kernel_inputs from kernel_inputs -- the true stub is the user thing, this thing just piggy backs on the true stub
def create_bwd_kernel_inputs(bwd_kernel, idx_upstream, grid, non_stub_args_idxs, kernel_inputs, upstream):

    # todo: add input checks
    # assert a.device == DEVICE and b.device == DEVICE and upstream.device == DEVICE

    # fwd specializes away some arguments (so that they aren't arguments in the fwd TTIR,
    # and thus not arguments in bwd TTIR as well) -- so don't pass them bwd TTIR
    idx_folded = bwd_kernel._autodiff_info[-1]
    if VERBOSE: print("[create_bwd_kernel_inputs] idx_folded: ", idx_folded)
    # user provided idx of output (idx_upstream) in terms of all args to fwd python kernel
    # (JITFunction), when it compiled, some of the args potentially got specialized away.
    # Thus here need to shift that user specified index to account for these args (that got
    # specialized away) if they were located before the user provided idx_upstream
    num_folded_before_upstream = 0
    # reverse to prevent shifting issues when popping
    for i in reversed(idx_folded):
        kernel_inputs.pop(i)
        if i < idx_upstream:
            num_folded_before_upstream += 1

    # if VERBOSE: print("[create_bwd_kernel_inputs] idx_folded", idx_folded)
    # if VERBOSE: print("num_folded_before_upstream:", num_folded_before_upstream)

    bwd_args = []
    for i, arg in enumerate(kernel_inputs):
        if i == (idx_upstream - num_folded_before_upstream):
            bwd_args.append(upstream)
            continue
        if isinstance(arg, torch.Tensor):
            bwd_args.append(torch.zeros_like(arg))

    # if VERBOSE: print("[create_bwd_kernel_inputs] fwd_args:", kernel_inputs)
    # if VERBOSE: print("[create_bwd_kernel_inputs] bwd_args:", bwd_args)
    bwd_kernel[grid](*kernel_inputs, *bwd_args)
    # if VERBOSE: print("[create_bwd_kernel_inputs] grads", bwd_args)

    # remove upstream grad
    # Use num_folded_before_upstream, otherwise assumes all args are tensors (IOW: grad inputs are 1:1 with
    # inputs) -- but it's not always the case, so this causes idx (in terms of fwd args) not match idx (in terms of
    # grad args)
    bwd_args.pop(idx_upstream - num_folded_before_upstream)


    # todo-now: an automatic way of solving this seems would require a new implementation of python API (V3)
    #   create_bwd_kernel_inputs really computes grad wrt inputs to fwd_KERNEL not the inputs to the fwd_stub -- how should you wire this into the autograd system?
    #       seems like you need do step of: given grads wrt fwd_kernel inputs; select only the ones that are grads wrt stub_fwd inputs
    #
    #   capture stub and create_bwd_kernel_inputs by closure -- instead of passing them as inputs to forward() -- otherwise autograd requires to return same number of grads
    #   cannot just pass "def forward(ctx, stub, create_bwd_kernel_inputs, *stub_inputs)" and later bind stub and create_bwd_kernel_inputs -- bc even if bind and thus won't need to feed them them at runtime, autograd still sees 4 args and therefore will err when by bwd retunrs only 2 grads (wrt to the 2 args) -- it would expect I should return 4 args (as the number of args to autograd.Function.forward)
    #   E.g. for flash aten kernel user fwd stub creates some additional tensor args and passes them to the kernel (e.g. M, OUT) and feeds them to the kernel but the user calls stub with "my_op(q, k, v)" -- so the autograd.Function.forward also only expects "q, k, v"
    #       but kernel actually sees (q, k, v, M, OUT, [other non-tensor arguments]) -- and your create_bwd_kernel_inputs creates grad tensors for all tensor arguments (to feed to bwd_kernel) and then returns all added grad_tensor arguments
    #       (which would also contain grad_M, grad_OUT) but because these M or OUT weren't passed to the autograd.Function.forwad, in autograd.Function.backward, it's incorrect to return grads wrt these values
    #       so need a way to only return grad wrt fwd stub args (and NOT wrt all bwd_kernel tensor args)

    # use reverse to avoid shifting issues
    for i in reversed(non_stub_args_idxs):
        # for the stub args which are **after** the popped upstream_idx,
        # shift them by 1 to account for the grad_out that I popped just above
        bwd_args.pop(i) if i < idx_upstream else bwd_args.pop(i - 1)

    # unpack list
    return (*bwd_args,)


# def my_post_hook(stub):
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

    def rebuild_binder(fn, delta, backend):
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
        sig_params = list(fn.jit_function.signature.parameters.values())
        for i in range(delta):
            p = inspect.Parameter(
                f"grad_{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation="tl.float16*" # "tl.pointer"
            )
            fn.jit_function.params.append(KernelParam(len(fn.jit_function.params), p, False, False)) # dns= dns_oa= , annotation="tl.float16*"
            sig_params.append(p)
        if VERBOSE: print("fn.jit_function.signature: ", fn.jit_function.signature)
        if VERBOSE: print("fn.jit_function self.params:", fn.jit_function.params)
        fn.jit_function.signature = fn.jit_function.signature.replace(parameters=sig_params)

        # 2. build a fresh binder
        new_binder = create_function_from_signature(
                        fn.jit_function.signature,
                        fn.jit_function.params,
                        backend)

        return new_binder


    if not already_compiled:

        # todo-now: find a cleaner way -- compile hook registers on all instances of JITFunction, but i want this hook to trigger only on bwd JITFuncton
        if fn.jit_function is not bwd_kernel:
            if VERBOSE: print("fn.jit_function is not bwd_kernel", id(fn.jit_function), id(bwd_kernel))
            return
        else:
            if VERBOSE: print("post hook triggered on the bwd!")

        compile_dict = compile

        if VERBOSE: print(f"Kernel {fn.name} just finished executing!")
        if VERBOSE: print(f"Representation: {repr}")

        # The fn parameter passed to the hook contains a jit_function attribute that refers to the JITFunction instance.
        # Each JITFunction keeps its kernels in device_caches[device], which is a tuple where the first element is the kernel cache dictionary.
        # The same key that's passed to the hook is the one used to store the kernel in the cache.

        # 1) extract fwd_compiled_kernel
        # Get the device
        device = driver.active.get_current_device()
        # Access the kernel from the cache
        jit_fn = fn.jit_function  # This is the JITFunction instance
        kernel_cache, target, backend, _binder = jit_fn.device_caches[device]  # First element is the kernel cache dict
        fwd_compiled_kernel = kernel_cache[key]  # Get the kernel using the same key


        # todo: cleanup
        # the "key" arg is just a python string with input signatures of the kernel
        # but I want some folder name -- one way is to hash it
        hash_object = hashlib.sha256(key.encode())
        dir_name = hash_object.hexdigest()[:10]
        if VERBOSE: print("dir_name: ", dir_name)

        # 2) write fwd IR
        os.makedirs(f"generated/{dir_name}", exist_ok=True)
        with open(f"generated/{dir_name}/inp.ttir", "w") as f:
          f.write(fwd_compiled_kernel.asm['ttir'])

        # 3) autodiff
        run_mlir_pass(f"generated/{dir_name}")

        # 4) create executable python fn for bwd
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


        # 5) replace original fwd CompiledKernel with autograd.Function (replaces cache entry in the cache of JITFunction)

        #   bwd_kernel is a CompiledKernel and I can directly swap original compiled kernel with this CompiledKernel
        #   And because this never calls a stub (no possibility for recursion).

        # if VERBOSE: print(dir(jit_fn.device_caches[device][0][key]))
        # '_init_handles', 'asm', 'function', 'hash', 'kernel', 'launch_enter_hook', 'launch_exit_hook', 'launch_metadata', 'metadata', 'module', 'name', 'packed_metadata', 'src'

        #   5.1. remove constexpr form: key, signature, params

        def remove_constexpr(key):
            if VERBOSE: print("[remove_constexpr] key: ", key)
            # need to also modify self.signature bc it's used in create_binder -> create_function_from_signature
            #   > inf JITFunction.run "binder = create_function_from_signature(self.signature, self.params, backend)"
            if VERBOSE: print("[remove_constexpr] fn.jit_function.signature:", fn.jit_function.signature)
            if VERBOSE: print("[remove_constexpr] fn.jit_function.params:", fn.jit_function.params)

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
                    fn.jit_function.params.pop(i)
            new_key = "[" + "), (".join(sub_strs)
            new_key += "]" if new_key[-1] == ")" else ")]"
            new_key += key.split("]")[1]
            if VERBOSE: print("[remove_constexpr] new_key", new_key)
            fn.jit_function.signature = fn.jit_function.signature.replace(parameters=sig_params)
            if VERBOSE: print("[remove_constexpr] fn.jit_function.signature: ", fn.jit_function.signature)
            if VERBOSE: print("[remove_constexpr] fn.jit_function self.params:", fn.jit_function.params)

            return new_key, num_const_args

        new_key, num_const_args = remove_constexpr(key)

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

        new_binder = rebuild_binder(fn, num_added_args, backend)
        if VERBOSE: print("new_binder: ", new_binder)

        if VERBOSE: print("fn.jit_function.signature:", fn.jit_function.signature)
        if VERBOSE: print("fn.jit_function.params:", fn.jit_function.params)


        #   5.3. add to bwd CompiledKernel into the cache

        del jit_fn.device_caches[device][0][key]
        # previously I incorrectly stored at the same key -- so the grad fn is basically keyed on singatures to the fwd kernel
        # key on a new_key (containing added args) not on the old key, otherwise:
        #   when you pass 6 args to the bwd JITFunction on the next call, it checks the cache for CompiledKernel with signature which has 6 args -- didn't find one (bc here you're storing the bwd CompiledKernel under the *original key* which only has 3 args) and thus re-compiles
        #   JITFunction looking for key: [('*fp32','D'), … 6 items …]{'debug':False}
        # IOW
        #   otherwise re-wraps -- you'd think that it should bc the signature of args taht will be passed to bwd_kernel is the same as was passed to bwd_kernel (just line above) -- so it seems like shouldn't trigger a re-tracing, but in fact, i guess bc I modified function signature in the hook after last re-teacing, it tries to trace again!
        #   no actually it was re-wrapping bc previously (in my post hook) I saved bwd graph while key'ing on the original signature (3 args). But now when passing 6 args -- it fails to find compiledKerenl with a key which has 6 args and thus re-compiles
        kernel_cache[new_key] = bwd_compiled_kernel
        if VERBOSE: print("kernel_cache[new_key]: ", kernel_cache[new_key])
        fn.jit_function.device_caches[device] = (kernel_cache, target, backend, new_binder)

        # s = fn.jit_function # RM
        # s.device_caches = defaultdict(s.create_binder) # RM
        # s.device_caches = defaultdict() # RM

        # # question-now: does recomputing them help?
        # s.non_constexpr_indices = [i for (i, p) in enumerate(s.params) if not p.is_constexpr] # RM
        # s.specialised_indices = [i for (i, p) in enumerate(s.params) if (not p.do_not_specialize) and (not p.is_constexpr)] # RM

        # todo: Iterate over fn.jit_function.device_caches.keys() and duplicate the backward kernel per device, or error out if torch.cuda.current_device() differs between forward and backward.
        # # if multiple gpus, patch caches of all devices?
        # for d, (cch, t, b, _) in fn.jit_function.device_caches.items():
        #     if key in cch:
        #         cch[key] = new_k
        #         fn.jit_function.device_caches[d] = (cch, t, b, new_binder)


        # quick sanity check
        # assert len(fwd_compiled_kernel.metadata.arg_types) == bwd_compiled_kernel.metadata.arg_types

        if VERBOSE: print("[my_hook] compile_dict['signature']", compile_dict["signature"])
        if VERBOSE: print("[my_hook] compile_dict['constants']", compile_dict["constants"])
        if VERBOSE: print("[my hook] jit_fn.params:", jit_fn.params)
        if VERBOSE: print("fn.jit_function.signature.parameters", fn.jit_function.signature.parameters)


        # recover all arguments that have been folded into the CompiledKernel (need for later removal
        # of these folded args from fwd_kernel_args inside create_bwd_kernel_inputs before passing them bwd_kernel);
        # includes compile‑time constants (declared `constexpr`) AND automatically specialised (ints/bools/tuples …)
        folded = list(p[0] for p in compile_dict["constants"])
        names = [fn.jit_function.arg_names[i] for i in folded]
        if VERBOSE: print("Hard‑coded parameter indices:", sorted(folded))
        if VERBOSE: print("Hard‑coded parameter names:  ", names)
        fn.jit_function._autodiff_info.append(folded)

    return False





# double nesting is needed here to support decorator with arguments
# and I want make user pass the stub here -- bc need some generic way for usr
# to specify what their stub fn is wt me hardcoding it
#
# def grad(stub, idx_upstream):

#     def inner(kernel):
#         kernel.add_pre_run_hook(shape_track_hook)

#         # todo: add a hook to a specific instance of a JITFunction
#         # Assign the hook to JITFunction's compiled_hook

#         triton.runtime.jit.JITFunction.compiled_hook = my_post_hook
#         return kernel

#     return inner



class DifferentiatedCompiledKernel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, stub, create_bwd_kernel_inputs, post_process_fn, *stub_inputs):
        if VERBOSE: print("\n"*3, "Op.forward")

        # basically the problem is that for Op.backward you need to save in args produced inside the fwd_stub (can get access right before executing the fwd kernel)
        #   ==> can overload kernel pre-hook to get access to them
        with StashArgsCtx() as arg_ctx:
            # if VERBOSE: print("[op fwd] stub_inputs:", stub_inputs)
            outs = stub(*stub_inputs)
        # now executing the stub should populate FWD_ARGS (bc we registered pre-hook on the kernel, and that kernel was called inside the stub)

        # todo:
        # note the execution of the kernel might have over-written zeroed out output buffers (initialized in the stub) with actual kernel outputs

        # note: you assume nothing modifies these arg in the StashArgsCtx before Function.backward is called (between Function.fwd and Function.bwd)
        kernel_inputs = arg_ctx.args
        # if VERBOSE: print("saving for bwd: ", kernel_inputs)
        # if VERBOSE: print("kernel_inputs: ", len(kernel_inputs))

        # ugly workaround for the fact that save_for_backward only works for tensor inputs
        # note stashing kernel inputs, not stub inputs
        ctx.save_for_backward(*[a for a in kernel_inputs if isinstance(a, torch.Tensor)])
        # assign to cxt to extract from .backward
        ctx.non_tensor_inputs = [a for a in kernel_inputs if not isinstance(a, torch.Tensor)]
        ctx.arg_types = [isinstance(a, torch.Tensor) for a in kernel_inputs]

        ctx.create_bwd_kernel_inputs = create_bwd_kernel_inputs
        ctx.post_process_fn = post_process_fn
        return outs


    @staticmethod
    def backward(ctx, upstream):
        if VERBOSE: print("\n"*3, "Op.backward")

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

        # IOW: create_grad_inputs
        # if VERBOSE: print("[Op.backward] fwd_kernel_inputs", fwd_kernel_inputs)
        grads = ctx.create_bwd_kernel_inputs(fwd_kernel_inputs, upstream)
        # if VERBOSE: print("[Op.backward] grads", grads)

        if ctx.post_process_fn is not None:
            grads = ctx.post_process_fn(*grads)

        return (None, None, None, *grads,)



def autodiff(kernel, stub, grid, idx_upstream, non_stub_args_idxs=None, post_process_fn=None):

    if non_stub_args_idxs is None:
        non_stub_args_idxs = []

    global bwd_kernel
    bwd_kernel = clone_jit_function(kernel)

    kernel.add_pre_run_hook(shape_track_hook)
    triton.runtime.jit.JITFunction.compiled_hook = my_post_hook

    global create_bwd_kernel_inputs
    create_bwd_kernel_inputs = partial(create_bwd_kernel_inputs, bwd_kernel, idx_upstream, grid, non_stub_args_idxs)
    fwd_stub = partial(stub, kernel)
    my_op = partial(DifferentiatedCompiledKernel.apply, fwd_stub, create_bwd_kernel_inputs, post_process_fn)
    return my_op, bwd_kernel



#### utils  ####

from functools import partial

def right_partial(func, *args):
    return lambda *fargs: func(*fargs, *args)
