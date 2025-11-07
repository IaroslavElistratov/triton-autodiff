from __future__ import annotations

from typing import Callable


__all__ = ["autodiff"]

def autodiff(*args, **kwargs):
    """Lightweight decorator to tag stubs that require gradients.

    Only records metadata (no runtime hook integration).
    """

    if 'idxs_buffers' not in kwargs:
        raise TypeError("autodiff() missing required keyword argument 'idxs_buffers'")

    idxs = kwargs.pop('idxs_buffers')
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {list(kwargs)}")

    if isinstance(idxs, int):
        idxs_tuple = (idxs,)
    else:
        idxs_tuple = tuple(idxs)

    def decorator(fwd_stub: Callable) -> Callable:
        fwd_stub.__is_autodiff_stub__ = True
        fwd_stub.__autodiff_idxs__ = idxs_tuple
        return fwd_stub

    return decorator
