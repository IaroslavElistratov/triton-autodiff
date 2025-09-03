# True: raises backward TTIR to trtion-lang; and emits stub
# False: uses backward TTIR directly
USE_LEGACY_API = False

if USE_LEGACY_API:
    from .legacy import autodiff
else:
    from .new import autodiff

__all__ = ["autodiff"]


# todo-low: maybe polymorphic over the old/new APIs?
# def autodiff(*args, **kwargs):
#     """
#     New API (preferred):
#         @autodiff(kernel=kernel, idxs_buffers=..., stub_name="stub")
#         def stub(...): ...

#     Legacy (still accepted):
#         @autodiff(idxs_buffers=...)
#         @triton.jit
#         def kernel(...): ...
#     """
#     # Legacy pattern: autodiff(idxs_buffers=...)(JITFunction)
#     if args and hasattr(args[0], "fn") and kwargs.get("kernel") is None:
#         fwd_kernel = args[0]
#         idxs = kwargs.pop("idxs_buffers", None)
#         if idxs is None:
#             raise TypeError("autodiff legacy usage requires idxs_buffers=...")
#         if isinstance(idxs, int):
#             idxs = (idxs,)
#         # annotate kernel so the compile hook can find the stub later
#         fwd_kernel.idxs_buffers = tuple(idxs)
#         fwd_kernel._is_fwd_kernel = True
#         # assume a top-level 'stub' in the same module unless overridden later
#         fwd_kernel._autodiff_stub_info = (fwd_kernel.fn.__module__, "stub")
#         return fwd_kernel

#     # New API passthrough
#     return _autodiff_new(*args, **kwargs)
