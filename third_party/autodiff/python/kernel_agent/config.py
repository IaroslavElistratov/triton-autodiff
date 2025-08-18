from dataclasses import dataclass
from typing import Optional

@dataclass
class OptimizeConfig:
    # Orchestration & gates
    max_iters: int = 8
    patience: int = 2                   # stop after N non-improving iterations
    min_rel_improvement: float = 0.02   # require >=2% relative throughput to count as better
    stop_on_grad_fail_repeats: int = 2
    target_throughput: Optional[float] = None  # early stop if reached

    # Run management
    record_root: str = "artifacts"
    run_id: Optional[str] = None
    time_budget_s: Optional[int] = None
    save_profiles: bool = True

    # LLM output contract
    patch_format: str = "apply_patch_md"
