from __future__ import annotations
import shutil, os
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from .utils import ensure_dir, write_json, sha256_file, short_sig

@dataclass
class IterationRecord:
    iter_idx: int
    candidate_bwd_fp: str
    grad_check_pass: bool
    grad_summary: str
    bench_metrics: Dict[str, float] | None = None
    profile_fp: Optional[str] = None
    improved: Optional[bool] = None
    notes: Optional[str] = None

@dataclass
class Manifest:
    run_id: str
    record_dir: str
    device_info: str

    # File pointers (source of truth = disk)
    forward_fp: str
    current_backward_fp: str
    best_backward_fp: str

    # Identities & metrics
    current_code_id: str
    best_code_id: str
    best_metrics: Dict[str, float] = field(default_factory=dict)

    # History & caches
    history: List[IterationRecord] = field(default_factory=list)
    neg_cache: Dict[str, str] = field(default_factory=dict)

    # Status
    status: str = "running"

    def save(self, root: str) -> str:
        ensure_dir(self.record_dir)
        path = os.path.join(self.record_dir, "state.json")
        write_json(path, asdict(self))
        return path

def code_id_for_file(path: str) -> str:
    return sha256_file(path)

def neg_key(kind: str, code_id: str) -> str:
    return f"{kind}:{short_sig(code_id)}"

def copy_as_best(src_fp: str, dst_dir: str) -> str:
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, "best_backward.py")
    shutil.copyfile(src_fp, dst)
    return dst
