from importlib import import_module

# todo: ugly, fix this when re-organizing folders; problem is that current python package tries to import from above its path

def _load(symbols: tuple[str, ...]):
    m = import_module("triton_autodiff_api")  # preferred: your canonical API
    return tuple(getattr(m, s) for s in symbols)


(record_autodiff_artifacts,
 get_last_bwd_fp,
 autodiff_overwrite_fp,
 StubOverrideDCK) = _load((
    "record_autodiff_artifacts",
    "get_last_bwd_fp",
    "autodiff_overwrite_fp",
    "StubOverrideDCK",
))

__all__ = [
    "record_autodiff_artifacts",
    "get_last_bwd_fp",
    "autodiff_overwrite_fp",
    "StubOverrideDCK",
]


