import os
os.environ['TRITON_ALWAYS_COMPILE']='1'


import torch
torch.manual_seed(0)

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


from triton.runtime import driver

from run_all_tests import main

"""
kwargs = {
    'signature': signature,
    'device': device,
    'constants': constants,
    'num_warps': options.num_warps,
    'num_ctas': options.num_ctas,
    'num_stages': options.num_stages,
    'enable_fp_fusion': options.enable_fp_fusion,
    'launch_cooperative_grid': options.launch_cooperative_grid,
    'extern_libs': options.extern_libs,
    'configs': configs,
    'specialization_data': specialization_data,
    'is_warmup': is_warmup,
}

return hook(
    key=key,
    repr=repr,
    fn=JitFunctionInfo(module, name, self),
    compile={"key": key, **kwargs},
    is_manual_warmup=is_warmup,
    already_compiled=False,
)
"""

import hashlib


# fwd_shapes = []
# import weakref

FWD_ARGS = None

# tensor_ptr_to_shape = weakref.WeakValueDictionary()


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

    # for arg in args:
    #     if isinstance(arg, torch.Tensor):
    #         tensor_ptr_to_shape[arg.data_ptr()] = arg

    # # Process positional arguments
    # for i, arg in enumerate(args):
    #     if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):  # Simple check for tensor-like objects
    #         fwd_shapes.append((f"arg_{i}", arg.shape, arg.dtype))
    #     elif hasattr(arg, 'data_ptr'):  # This is a raw pointer, we can't get shape directly
    #         fwd_shapes.append((f"arg_{i}", "pointer", None))

    # # Process keyword arguments
    # for name, arg in kwargs.items():
    #     if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
    #         fwd_shapes.append((name, arg.shape, arg.dtype))
    #     elif hasattr(arg, 'data_ptr'):
    #         fwd_shapes.append((name, "pointer", None))


# double nesting is needed here to support decorator with arguments
# and I want make user pass the stub here -- bc need some generic way for usr
# to specify what their stub fn is wt me hardcoding it
#
# todo: use stub and idx_upstream passed by the user
def grad(stub, idx_upstream):

    def inner(kernel):
        kernel.add_pre_run_hook(shape_track_hook)

        # todo: add a hook to a specific instance of a JITFunction
        # Assign the hook to JITFunction's compiled_hook

        triton.runtime.jit.JITFunction.compiled_hook = my_post_hook

        return kernel

    return inner

BWD_KERNEL = None

# def my_post_hook(stub):
def my_post_hook(key, repr, fn, compile, is_manual_warmup, already_compiled):
    if not already_compiled:
        compile_dict = compile
        print(f"Kernel {fn.name} just finished executing!")
        print(f"Representation: {repr}")

        # I see the post execution hook is called from here -- but how do i get "kenrel" (output of self.compile) from inside the hook?
        # JITFunction.run
        #     # compile the kernel
        #     src = self.ASTSource(self, signature, constexprs, attrs)
        #     kernel = self.compile(src, target=target, options=options.__dict__)
        #     kernel_cache[key] = kernel
        #     self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=False)


        # 1. The fn parameter passed to the hook contains a jit_function attribute that refers to the JITFunction instance.
        # 2. Each JITFunction keeps its kernels in device_caches[device], which is a tuple where the first element is the kernel cache dictionary.
        # 3. The same key that's passed to the hook is the one used to store the kernel in the cache.

        # Get the device
        device = driver.active.get_current_device()

        # Access the kernel from the cache
        jit_fn = fn.jit_function  # This is the JITFunction instance
        kernel_cache = jit_fn.device_caches[device][0]  # First element is the kernel cache dict
        _compiled_kernel = kernel_cache[key]  # Get the kernel using the same key

        # # Now you have the kernel object
        # print(f"Retrieved compiled kernel object: {_compiled_kernel}")



        # the "key" arg is just a python string with input signatures of the kernel
        # but I want some folder name -- one way is to hash it
        hash_object = hashlib.sha256(key.encode())
        dir_name = hash_object.hexdigest()[:10]

        with open(f"{dir_name}/inp.ttir", "w") as f:
          f.write(_compiled_kernel.asm['ttir'])

        main(f"./{dir_name}", run_py=False)


        # create executable python fn for bwd
        from triton.compiler import compile
        from triton.backends.compiler import GPUTarget

        target = GPUTarget("cuda", arch=89, warp_size=32)
        bwd_kernel = compile("out.ttir", target=target)

        # Extract signature from the compile dictionary
        signature = compile_dict["signature"]
        print(f"Kernel signature: {signature}")

        print("repr: ", repr)

        # for (name, shape, dtype) in fwd_shapes:
        #     print("shape:", shape)


        # grad_args = []
        # for ptr, tensor in tensor_ptr_to_shape.items():
        #     # print(ptr, tensor)
        #     grad_args.append(torch.zeros_like(tensor))

        # todo-now: need to input idx for upstream ptr from the user -- as to pass it during the bwd pass
        global BWD_KERNEL
        BWD_KERNEL = bwd_kernel

    return False



def stub_bwd(kernel_inputs, upstream, idx_upstream):

    # todo: extract this from the kernel
    grid = (1, 1, 1)

    bwd_args = []
    for i, arg in enumerate(kernel_inputs):
        if i == idx_upstream:
            bwd_args.append(upstream)
            continue
        if isinstance(arg, torch.Tensor):
            bwd_args.append(torch.zeros_like(arg))

    # # print("[stub_bwd] kernel_inputs:", kernel_inputs)
    print("[stub_bwd] bwd_args:", bwd_args)
    BWD_KERNEL[grid](*kernel_inputs, *bwd_args)
    print("[stub_bwd] grads", bwd_args)

    # [stub_bwd] kernel_inputs: [tensor([0.3990, 0.5167, 0.0249, 0.9401], device='cuda:0', requires_grad=True), tensor([0.9722, 0.7910, 0.4690, 0.3300], device='cuda:0', requires_grad=True), tensor([0., 0., 0., 0.], device='cuda:0')]
    # [stub_bwd] bwd_args: [tensor([0., 0., 0., 0.], device='cuda:0'), tensor([0., 0., 0., 0.], device='cuda:0'), tensor([1., 1., 1., 1.], device='cuda:0')]
    # [stub_bwd] grads [tensor([0.9722, 0.7910, 0.4690, 0.3300], device='cuda:0'), tensor([0.8990, 1.0167, 0.5249, 1.4401], device='cuda:0'), tensor([1., 1., 1., 1.], device='cuda:0')

    # remove upstream grad
    bwd_args.pop(idx_upstream)
    # unpack list
    return (*bwd_args,)



def stub(a, b):
    output = torch.empty_like(a)
    assert a.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    kernel[grid](a, b, output, BLOCK_SIZE=4)
    return output

# @autodiff
@grad(stub, idx_upstream=2)
@triton.jit
def kernel(
        a_ptr,
        b_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
    offsets = tl.arange(0, BLOCK_SIZE)

    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)

    x = a + 0.5
    y = x * b

    tl.store(output_ptr + offsets, y, mask=offsets<4)



size = 4
a = torch.rand(size, device=DEVICE)
b = torch.rand(size, device=DEVICE)
a.requires_grad = True
b.requires_grad = True
upstream = torch.ones_like(a)




class Op(torch.autograd.Function):


    # question-now: bwd_stub really computes grad wrt inputs to fwd_KERNEL not the inputs to the fwd_stub -- how should you wire this into the autograd system?
    #   seems like you need do step of: given grads wrt fwd_kernel inputs; select only the ones that are grads wrt stub_fwd inputs
    @staticmethod
    def forward(ctx, *stub_inputs):
        # ctx is a context object that can be used to stash information for backward computation. You can cache arbitrary
        # objects for use in the backward pass using the ctx.save_for_backward method.

        # now executing the stub should populate FWD_ARGS (bc we registered pre-hook on the kernel, and that kernel was called inside the stub)
        with StashArgsCtx() as arg_ctx:
            outs = stub(*stub_inputs)

        # answer-now:
        # basically the problem is that for Op.backward you need to save in args produced inside the fwd_stub (can get access right before executing the fwd kernel)
        #   ==> can overload kernel pre-hook to get access to them
        # todo:
        # note the execution of the kernel might have  over-written zeroed out output buffers (initialized in the stub) with actual kernel outputs

        kernel_inputs = arg_ctx.args
        print("saving for bwd: ", kernel_inputs)

        # note stashing kernel inputs, not stub inputs
        ctx.save_for_backward(*kernel_inputs)
        return outs

    @staticmethod
    def backward(ctx, upstream):
        inputs = ctx.saved_tensors
        # returns grads wrt all inputs
        # todo: don't hardcode idx
        return stub_bwd(inputs, upstream, idx_upstream=2)



my_op = Op.apply

my_out = my_op(a, b)
my_out.backward(upstream)






# compare with pytorch

torch_a = torch.clone(a).detach().requires_grad_().to(device='cuda:0')
torch_b = torch.clone(b).detach().requires_grad_().to(device='cuda:0')
# torch_a.requires_grad = True
# torch_b.requires_grad = True

torch_out = (torch_a + 0.5) * torch_b
torch_out.backward(torch.ones_like(torch_out))

if torch.allclose(a.grad, torch_a.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

if torch.allclose(b.grad, torch_b.grad.to(dtype=torch.float32), atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
