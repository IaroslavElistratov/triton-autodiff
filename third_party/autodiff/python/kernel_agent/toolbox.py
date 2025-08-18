from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

# Default to the repo's apply_patch tool but allow DI for tests.
def _identity_apply_patch(fp: str, patch: str) -> str:  # replace with real tool if needed
    raise NotImplementedError("wire your repo's apply_patch here")

@dataclass
class Toolbox:
    """
    Host-side tools; all accept/return file paths or small values.
    """
    get_user_forwrad: Callable[[], str]
    naive_grad: Callable[[str], str]
    gradient_check: Callable[[str], str]
    benchmark: Callable[[str], object]
    profile: Callable[[str], str]
    get_user_dvice_info: Callable[[], str]
    apply_patch: Callable[[str, str], str] = _identity_apply_patch

    def apply_patch_text(self, file_fp: str, patch_text: str) -> str:
        return self.apply_patch(file_fp, patch_text)
