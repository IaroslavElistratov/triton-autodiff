from __future__ import annotations
import hashlib, json, os, re, time
from pathlib import Path
from typing import Dict, Optional

PASS_RX = re.compile(r"\b(pass|ok|success)\b", re.I)

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def now_run_id(prefix: str = "run") -> str:
    ts = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    return f"{prefix}_{ts}"

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def short_sig(s: str, n: int = 8) -> str:
    return s[:n]

def write_json(path: str, obj: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

# --- tiny parsers / summarizers (keep prompts small) -------------------------

def parse_grad_check_output(msg: str) -> tuple[bool, str]:
    ok = bool(PASS_RX.search(msg or ""))
    return ok, msg

def normalize_bench_metrics(raw) -> Dict[str, float]:
    """
    Accepts float|dict|object-like and normalizes to {'throughput': float} if possible.
    """
    if raw is None:
        return {}
    if isinstance(raw, (int, float)):
        return {"throughput": float(raw)}
    if isinstance(raw, dict):
        # prefer 'throughput' if present, else ms->throughput heuristic if you add it later
        return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    # last resort string parse (user tools can return text)
    try:
        num = float(str(raw).strip().split()[0])
        return {"throughput": num}
    except Exception:
        return {}

def strictly_better(cand: Dict[str, float], best: Dict[str, float] | None, min_rel: float) -> bool:
    if not cand: return False
    if not best or "throughput" not in best: return True
    if "throughput" not in cand: return False
    base = best["throughput"]
    return cand["throughput"] >= base * (1.0 + min_rel)

def read_excerpt(path: str, max_lines: int = 120) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    # simple heuristic: show head if file is short, else around first kernel def
    head = "".join(lines[:max_lines])
    return head

def summarize_bench(metrics: Dict[str, float] | None) -> str:
    if not metrics: return "no bench yet"
    bits = []
    if "throughput" in metrics: bits.append(f"throughput={metrics['throughput']:.4g}")
    return ", ".join(bits) or "bench summary unavailable"

def summarize_profile_file(path: Optional[str]) -> str:
    if not path: return "(no profile)"
    # keep it small; you can plug a real parser later
    return f"profile: see {Path(path).name}"
