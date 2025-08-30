# True: raises backward TTIR to trtion-lang; and emits stub
# False: uses backward TTIR directly
USE_LEGACY_API = False

if USE_LEGACY_API:
    from .legacy import autodiff
else:
    from .new import autodiff

__all__ = ["autodiff"]


