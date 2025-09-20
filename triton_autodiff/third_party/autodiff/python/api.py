# Expose autodiff API under triton.backends.autodiff via this file.
# This file is symlinked to `python/triton/backends/autodiff.py` in editable installs.

# can't do "from triton.third_party.autodiff.python.api" because it isn’t part of the installed triton package.
# The Python package that ends up on sys.path is .../python/triton. There is no triton.third_party package.
# The third_party/autodiff/... tree lives outside python/triton, so Python cannot resolve triton.third_party...
# unless explicitly package and install that whole subtree under triton/third_party. Which I avoid to not pollute the triton.* namespace

import os, sys

base = os.environ.get("TRITON_AUTODIFF_DIR")
assert base
repo_root = os.path.abspath(os.path.join(base, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from triton_autodiff.third_party.autodiff.python.api import (
    autodiff,
    StubOverrideDCK,
    record_autodiff_artifacts,
    autodiff_overwrite_fp,
    get_last_bwd_fp,
)

__all__ = ["autodiff", "StubOverrideDCK", "record_autodiff_artifacts", "autodiff_overwrite_fp", "get_last_bwd_fp"]