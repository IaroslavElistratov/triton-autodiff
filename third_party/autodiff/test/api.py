import os
os.environ['TRITON_ALWAYS_COMPILE']='1'
import hashlib
from functools import partial
from collections import defaultdict

import torch
torch.manual_seed(0)
DEVICE = torch.device("cuda:0")

import triton
import triton.language as tl
from triton.runtime import driver
from triton.runtime.jit import JITFunction

from triton.backends.run_all_tests import main


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
    # Clear previous shapes if needed
    # fwd_shapes.clear()  # Uncomment if you want to reset on each call

    ctx = StashArgsCtx._active_instance
    if ctx is not None:
        ctx.args = args # [weakref.ref(arg) for arg in args]
        # todo-now: you're just ignoring kwargs



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

    # Copy any pre-run hooks
    cloned.pre_run_hooks = list(jit_func.pre_run_hooks)

    return cloned


# it's not as much as a stub, but more like helper to create_bwd_kernel_inputs from kernel_inputs -- the true stub is the user thing, this thing just piggy backs on the true stub
def create_bwd_kernel_inputs(bwd_kernel, idx_upstream, kernel_inputs, upstream):

    # todo: add input checks
    # assert a.device == DEVICE and b.device == DEVICE and upstream.device == DEVICE

    # todo: extract this from the kernel
    grid = (1, 1, 1)

    bwd_args = []
    for i, arg in enumerate(kernel_inputs):
        if i == idx_upstream:
            bwd_args.append(upstream)
            continue
        if isinstance(arg, torch.Tensor):
            bwd_args.append(torch.zeros_like(arg))

    # # print("[create_bwd_kernel_inputs] kernel_inputs:", kernel_inputs)
    print("[create_bwd_kernel_inputs] fwd_args:", kernel_inputs)
    print("[create_bwd_kernel_inputs] bwd_args:", bwd_args)
    bwd_kernel[grid](*kernel_inputs, *bwd_args)
    print("[create_bwd_kernel_inputs] grads", bwd_args)

    # [orig_arg, orig_arg, orig_arg_OUT, grad, grad, grad_OUT,]
    # num_args = len([*kernel_inputs, *bwd_args])

    # remove upstream grad
    bwd_args.pop(idx_upstream)
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

        print("[key_add_args] key: ", key)
        # key:  [('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D')]{'debug': False}

        split = key.split("]")
        print("[key_add_args] split: ", split)
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

        print("[key_add_args] new_key: ", new_key)
        # new_key:  [('*fp32', 'D'), ('*fp32', 'D'), ('*fp32', 'D'), ('*f32', 'D'), ('*f32', 'D'), ('*f32', 'D')]{'debug': False}
        return new_key

    def rebuild_binder(fn, delta, backend):
        """
        1 – update signatures	create_function_from_signature looks at fn.jit_function.signature and fn.jit_function.metadata.arg_types. If you forget to patch either, the binder will still have the old arity and you will hit TypeError: dynamic_func() takes N positional arguments… at launch 
        2 – new binder	The helper builds the little Python function (dynamic_func) that maps user launch args → positional tuple for the GPU call
        """

        print("rebuild_binder")
        import inspect

        from triton.runtime.jit import (
            KernelParam,                        # Triton’s helper for arg metadata
            create_function_from_signature,     # binder factory
        )

        print(f"adding {delta} args")

        # 1. extend the Python signature *before* we rebuild the binder
        sig_params = list(fn.jit_function.signature.parameters.values())
        for i in range(delta):
            p = inspect.Parameter(
                f"extra_{i}",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation="tl.float16*" # "tl.pointer"
            )
            fn.jit_function.params.append(KernelParam(len(fn.jit_function.params), p, False, False)) # dns= dns_oa= , annotation="tl.float16*"
            sig_params.append(p)
        print("fn.jit_function.signature: ", fn.jit_function.signature)
        print("fn.jit_function self.params:", fn.jit_function.params)
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
            print("fn.jit_function is not bwd_kernel", id(fn.jit_function), id(bwd_kernel))
            return
        else:
            print("post hook triggered on the bwd!")

        compile_dict = compile

        print(f"Kernel {fn.name} just finished executing!")
        print(f"Representation: {repr}")

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
        print("dir_name: ", dir_name)

        # 2) write fwd IR
        os.makedirs(f"generated/{dir_name}", exist_ok=True)
        with open(f"generated/{dir_name}/inp.ttir", "w") as f:
          f.write(fwd_compiled_kernel.asm['ttir'])

        # 3) autodiff
        main(f"generated/{dir_name}", run_py=False)

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
        # question-now: seem automtically lowered to ttgir not ttir
        # print(bwd_compiled_kernel.asm.keys())

        # good, I can confirm this is my autodiff'ed IR
        # print(bwd_compiled_kernel.asm['ttgir'])


        # 5) replace original fwd CompiledKernel with autograd.Function (replaces cache entry in the cache of JITFunction)

        #   bwd_kernel is a CompiledKernel and I can directly swap original compiled kernel with this CompiledKernel
        #   And because this never calls a stub (no possibility for recursion).

        # print(dir(jit_fn.device_caches[device][0][key]))
        # '_init_handles', 'asm', 'function', 'hash', 'kernel', 'launch_enter_hook', 'launch_exit_hook', 'launch_metadata', 'metadata', 'module', 'name', 'packed_metadata', 'src'

        #   5.1. remove constexpr form: key, signature, params

        def remove_constexpr(key):
            print("[remove_constexpr] key: ", key)
            # need to also modify self.signature bc it's used in create_binder -> create_function_from_signature
            #   > inf JITFunction.run "binder = create_function_from_signature(self.signature, self.params, backend)"
            print("[remove_constexpr] fn.jit_function.signature:", fn.jit_function.signature)
            print("[remove_constexpr] fn.jit_function.params:", fn.jit_function.params)

            # remove constexpr -- bc backward signature or key should not have them (bc they will NOT be provided to the bwd kernel)
            # remove ", ('constexpr', 4)"
            # key = key.replace(", ('constexpr', 4)", "")
            # replaces all occurrences of , ('constexpr', [some integer]) in the string

            sub_strs = key.split("[")[1].split("]")[0].split("), (")
            # print('sub_strs: ', sub_strs)
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
            print("[remove_constexpr] new_key", new_key)
            fn.jit_function.signature = fn.jit_function.signature.replace(parameters=sig_params)
            print("[remove_constexpr] fn.jit_function.signature: ", fn.jit_function.signature)
            print("[remove_constexpr] fn.jit_function self.params:", fn.jit_function.params)

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
        print("new_binder: ", new_binder)

        print("fn.jit_function.signature:", fn.jit_function.signature)
        print("fn.jit_function.params:", fn.jit_function.params)


        #   5.3. add to bwd CompiledKernel into the cache

        del jit_fn.device_caches[device][0][key]
        # peevishly I incorrectly stored at the same key -- so the grad fn is basically keyed on singatures to the fwd kernel
        # key on a new_key (containing added args) not on the old key, otherwise:
        #   when you pass 6 args to the bwd JITFunction on the next call, it checks the cache for CompiledKernel with signature which has 6 args -- didn't find one (bc here you're storing the bwd CompiledKernel under the *original key* which only has 3 args) and thus re-compiles
        #   JITFunction looking for key: [('*fp32','D'), … 6 items …]{'debug':False}
        # IOW
        #   otherwise re-wraps -- you'd think that it should bc the signature of args taht will be passed to bwd_kernel is the same as was passed to bwd_kernel (just line above) -- so it seems like shouldn't trigger a re-tracing, but in fact, i guess bc I modified function signature in the hook after last re-teacing, it tries to trace again!
        #   no actually it was re-wrapping bc previously (in my post hook) I saved bwd graph while key'ing on the original signature (3 args). But now when passing 6 args -- it fails to find compiledKerenl with a key which has 6 args and thus re-compiles
        kernel_cache[new_key] = bwd_compiled_kernel
        print("kernel_cache[new_key]: ", kernel_cache[new_key])
        fn.jit_function.device_caches[device] = (kernel_cache, target, backend, new_binder)

        # s = fn.jit_function # RM
        # s.device_caches = defaultdict(s.create_binder) # RM
        # s.device_caches = defaultdict() # RM

        # # question-now: does recomputing them help?
        # s.non_constexpr_indices = [i for (i, p) in enumerate(s.params) if not p.is_constexpr] # RM
        # s.specialised_indices = [i for (i, p) in enumerate(s.params) if (not p.do_not_specialize) and (not p.is_constexpr)] # RM

        # # if multiple gpus, patch caches of all devices?
        # for d, (cch, t, b, _) in fn.jit_function.device_caches.items():
        #     if key in cch:
        #         cch[key] = new_k
        #         fn.jit_function.device_caches[d] = (cch, t, b, new_binder)


        # quick sanity check
        # assert len(fwd_compiled_kernel.metadata.arg_types) == bwd_compiled_kernel.metadata.arg_types

        print("[my_hook] compile_dict['signature']", compile_dict["signature"])
        print("[my_hook] compile_dict['constants']", compile_dict["constants"])
        print("[my hook] jit_fn.params:", jit_fn.params)
        print("fn.jit_function.signature.parameters", fn.jit_function.signature.parameters)

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

    # question-now:
    #   create_bwd_kernel_inputs really computes grad wrt inputs to fwd_KERNEL not the inputs to the fwd_stub -- how should you wire this into the autograd system?
    #   seems like you need do step of: given grads wrt fwd_kernel inputs; select only the ones that are grads wrt stub_fwd inputs

    @staticmethod
    def forward(ctx, stub, create_bwd_kernel_inputs, *stub_inputs):
        print("\n"*3, "Op.forward")

        # basically the problem is that for Op.backward you need to save in args produced inside the fwd_stub (can get access right before executing the fwd kernel)
        #   ==> can overload kernel pre-hook to get access to them
        with StashArgsCtx() as arg_ctx:
            # print("[op fwd] stub_inputs:", stub_inputs)
            outs = stub(*stub_inputs)
        # now executing the stub should populate FWD_ARGS (bc we registered pre-hook on the kernel, and that kernel was called inside the stub)

        # todo:
        # note the execution of the kernel might have over-written zeroed out output buffers (initialized in the stub) with actual kernel outputs

        # note: you assume nothing modifies these arg in the StashArgsCtx before Function.backward is called (between Function.fwd and Function.bwd)
        kernel_inputs = arg_ctx.args
        # print("saving for bwd: ", kernel_inputs)
        # print("kernel_inputs: ", len(kernel_inputs))

        # ugly workaround for the fact that save_for_backward only works for tensor inputs
        # note stashing kernel inputs, not stub inputs
        ctx.save_for_backward(*[a for a in kernel_inputs if isinstance(a, torch.Tensor)])
        # assign to cxt to extract from .backward
        ctx.non_tensor_inputs = [a for a in kernel_inputs if not isinstance(a, torch.Tensor)]
        ctx.arg_types = [isinstance(a, torch.Tensor) for a in kernel_inputs]

        ctx.create_bwd_kernel_inputs = create_bwd_kernel_inputs
        return outs


    @staticmethod
    def backward(ctx, upstream):
        print("\n"*3, "Op.backward")

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

        # todo: don't hardcode idx
        # IOW: create_grad_inputs
        # print("[Op.backward] fwd_kernel_inputs", fwd_kernel_inputs)
        grads = ctx.create_bwd_kernel_inputs(fwd_kernel_inputs, upstream)
        # print("[Op.backward] grads", grads)

        # todo-now:
        #   capture stub and create_bwd_kernel_inputs by closure -- instead of passing them as inputs to forward() -- otherwise autograd requres to return same numebr of grads
        #   cannot just pass "def forward(ctx, stub, create_bwd_kernel_inputs, *stub_inputs)" and later bind stub and create_bwd_kernel_inputs -- bc even if bind and thus won't need to feed them them at runtime, autograd still sees 4 argueets and therefore will err when by bwd retunrs only 2 grads (wrt to the 2 args) -- it would expect I should return 4 args (as the number of args to autograd.Fcutnion.forward)
        return (None, None, *grads,)



def autodiff(kernel, stub, idx_upstream):

    global bwd_kernel
    bwd_kernel = clone_jit_function(kernel)

    kernel.add_pre_run_hook(shape_track_hook)
    triton.runtime.jit.JITFunction.compiled_hook = my_post_hook

    global create_bwd_kernel_inputs
    create_bwd_kernel_inputs = partial(create_bwd_kernel_inputs, bwd_kernel, idx_upstream)
    fwd_stub = partial(stub, kernel)
    my_op = partial(DifferentiatedCompiledKernel.apply, fwd_stub, create_bwd_kernel_inputs)
    return my_op, bwd_kernel

