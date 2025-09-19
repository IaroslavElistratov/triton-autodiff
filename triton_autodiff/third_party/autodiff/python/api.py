# todo: ugly, fix this when re-organizing folders; problem is that current python package tries to import from above its path
try:
    from third_party.autodiff.python.api import autodiff
except ModuleNotFoundError:  # third_party not on sys.path, load by file path
    import importlib.util, os, sys
    _here = os.path.dirname(__file__)
    _repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    _api_dir = os.path.join(_repo_root, "third_party", "autodiff", "python", "api")
    _api_init = os.path.join(_api_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "triton_autodiff_api",
        _api_init,
        submodule_search_locations=[_api_dir],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["triton_autodiff_api"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    autodiff = mod.autodiff