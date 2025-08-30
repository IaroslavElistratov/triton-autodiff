from .common import USE_LEGACY_API, autodiff

if USE_LEGACY_API:
    from . import legacy as _backend  # noqa: F401
else:
    from . import new as _backend  # noqa: F401

__all__ = ["autodiff"]


