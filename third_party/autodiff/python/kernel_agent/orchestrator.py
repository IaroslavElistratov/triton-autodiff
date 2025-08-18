from __future__ import annotations
import os, time
from .config import OptimizeConfig
from .llm import PatchContext, PatchProvider
from .state import Manifest, IterationRecord, code_id_for_file, copy_as_best, neg_key
from .toolbox import Toolbox
from .utils import (
    ensure_dir, now_run_id, read_excerpt,
    parse_grad_check_output, normalize_bench_metrics,
    strictly_better, summarize_bench, summarize_profile_file,
)

class KernelOptimizer:
    """
    Deterministic loop (compile is implicit inside your tools):
      boot: get_user_forwrad -> naive_grad
      repeat:
        gradient_check -> if fail: LLM fix (diff) -> apply -> continue
        benchmark (+optional profile)
        if not improved enough and not converged: LLM optimize (diff) -> apply -> continue
    """

    def __init__(self, cfg: OptimizeConfig, tools: Toolbox, patcher: PatchProvider) -> None:
        self.cfg = cfg
        self.t = tools
        self.pp = patcher
        self._deadline = (time.time() + cfg.time_budget_s) if cfg.time_budget_s else None

    def run(self) -> Manifest:
        # 0) prepare run
        run_id = self.cfg.run_id or now_run_id()
        record_dir = os.path.join(self.cfg.record_root, run_id)
        ensure_dir(record_dir)

        # 1) bootstrap (outside the loop)
        device_info = self.t.get_user_dvice_info()
        forward_fp = self.t.get_user_forwrad()
        backward_fp = self.t.naive_grad(forward_fp)

        m = Manifest(
            run_id=run_id,
            record_dir=record_dir,
            device_info=device_info,
            forward_fp=forward_fp,
            current_backward_fp=backward_fp,
            best_backward_fp=backward_fp,
            current_code_id=code_id_for_file(backward_fp),
            best_code_id=code_id_for_file(backward_fp),
        )
        m.save(self.cfg.record_root)

        non_improving = 0
        for i in range(self.cfg.max_iters):
            if self._timed_out(): break

            # 2.1 correctness (FD)
            fd_msg = self.t.gradient_check(m.current_backward_fp)
            fd_ok, fd_sum = parse_grad_check_output(fd_msg)
            if not fd_ok:
                # negative-cache repeated FD failure on this code hash
                key = neg_key("fd", m.current_code_id)
                if m.neg_cache.get(key) == (fd_sum or "")[:200]:
                    # same failure signature on same code; bail early for this candidate
                    m.history.append(IterationRecord(
                        iter_idx=i, candidate_bwd_fp=m.current_backward_fp,
                        grad_check_pass=False, grad_summary=fd_sum or "fd fail (repeat)"
                    ))
                    m.save(self.cfg.record_root)
                    break
                m.neg_cache[key] = (fd_sum or "")[:200]

                # ask LLM to FIX correctness
                patch = self._ask_patch(
                    phase="fix", bwd_fp=m.current_backward_fp, device_info=m.device_info,
                    grad_summary=fd_sum, bench_summary=summarize_bench(m.best_metrics),
                    profile_hint="(skip; focusing on correctness)"
                )
                m.current_backward_fp = self.t.apply_patch_text(m.current_backward_fp, patch)
                m.current_code_id = code_id_for_file(m.current_backward_fp)
                m.history.append(IterationRecord(
                    iter_idx=i, candidate_bwd_fp=m.current_backward_fp,
                    grad_check_pass=False, grad_summary=fd_sum, notes="fix applied"
                ))
                m.save(self.cfg.record_root)
                # back to start of loop
                continue

            # 2.2 performance (correctness gated)
            raw_bench = self.t.benchmark(m.current_backward_fp)
            cand_metrics = normalize_bench_metrics(raw_bench)
            prof_fp = self.t.profile(m.current_backward_fp) if self.cfg.save_profiles else None

            improved = strictly_better(cand_metrics, m.best_metrics, self.cfg.min_rel_improvement)
            m.history.append(IterationRecord(
                iter_idx=i, candidate_bwd_fp=m.current_backward_fp,
                grad_check_pass=True, grad_summary="OK",
                bench_metrics=cand_metrics, profile_fp=prof_fp, improved=improved
            ))

            if improved:
                m.best_metrics = cand_metrics
                m.best_backward_fp = m.current_backward_fp
                m.best_code_id = m.current_code_id
                copy_as_best(m.current_backward_fp, m.record_dir)
                non_improving = 0
            else:
                non_improving += 1

            m.save(self.cfg.record_root)

            # early-stop checks
            if self.cfg.target_throughput and "throughput" in m.best_metrics:
                if m.best_metrics["throughput"] >= self.cfg.target_throughput:
                    break
            if non_improving >= self.cfg.patience:
                break

            # 2.3 optimization: ask LLM for a perf patch
            patch = self._ask_patch(
                phase="optimize", bwd_fp=m.current_backward_fp, device_info=m.device_info,
                grad_summary="OK", bench_summary=summarize_bench(m.best_metrics),
                profile_hint=summarize_profile_file(prof_fp),
            )
            m.current_backward_fp = self.t.apply_patch_text(m.current_backward_fp, patch)
            m.current_code_id = code_id_for_file(m.current_backward_fp)

        m.status = "done"
        m.save(self.cfg.record_root)
        return m

    # ------------------------- helpers -------------------------

    def _ask_patch(self, *, phase: str, bwd_fp: str, device_info: str,
                   grad_summary: str, bench_summary: str, profile_hint: str) -> str:
        ctx = PatchContext(
            device_info=device_info,
            kernel_snippet=read_excerpt(bwd_fp),
            grad_check_summary=grad_summary,
            benchmark_summary=bench_summary,
            profile_hint=profile_hint,
            target_file_path=bwd_fp,
            phase=phase,
        )
        return self.pp.propose_patch(ctx)

    def _timed_out(self) -> bool:
        return self._deadline is not None and time.time() > self._deadline
