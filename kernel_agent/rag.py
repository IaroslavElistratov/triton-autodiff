#!/usr/bin/env python3
"""
RAG prompt block builder for kernel_agent.

Adds a compact "Retrieved backward references — adapt, don't copy" section
to the LLM prompt when a local embeddings index is available.

Design:
- Load embeddings pickle produced by RAG/rag_kernel_embedder.py.
- Use vector similarity (cosine) to find similar forward kernels.
- Embed query using OpenAI API (one call per optimization run, cached).
- Threshold by min_sim, de-duplicate identical backward docs.
- Compose a concise, token-budgeted block with forward preview and backward snippet.

Usage:
  # As a library (called by orchestrator):
  kernel-agent --backend triton --file-path test/attention.py --rag --rag-topk 3 --rag-min-sim 0.75

  # As a debug CLI tool:
  python -m kernel_agent.rag --file-path test/attention.py --topk 3 --min-sim 0.75 --show-block
"""

from __future__ import annotations

import os
import pickle
import hashlib
from typing import Any, List, Tuple


def _truthy(val: str | None, default: str = "0") -> bool:
    """Parse boolean-like env flags; treat non-empty/true-ish as True."""
    v = (val if val is not None else default).strip().lower()
    return v not in ("", "0", "false", "no", "off")


def _truncate(text: str, max_chars: int) -> str:
    """Truncate at a character budget, prefer cutting at a newline near the end."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    return cut if nl < int(max_chars * 0.9) else cut[:nl]


def _load_index(index_path: str):
    """Load embeddings pickle produced by rag_kernel_embedder.py.

    The pickle contains KernelPair objects, but we only extract primitive dicts
    (embeddings, documents, backward_documents), so a minimal stub suffices.
    """
    class _KernelPair:
        def __init__(self, file_path: str, forward: str, backward: str):
            self.file_path = file_path
            self.forward = forward
            self.backward = backward

    class _SafeUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if name == "KernelPair":
                return _KernelPair
            return pickle.Unpickler.find_class(self, module, name)

    with open(index_path, "rb") as f:
        data: dict[str, Any] = _SafeUnpickler(f).load()
    embeddings = data["embeddings"]           # dict[str, np.ndarray] (normalized)
    documents = data["documents"]             # dict[str, str] (FORWARD sections)
    backward_docs = data.get("backward_documents", {}) or {}
    file_list = data["file_list"]             # list[str]
    openai_model = data.get("openai_model", "text-embedding-3-large")
    return embeddings, documents, backward_docs, file_list, openai_model


def _embed_query(query_text: str, model: str = "text-embedding-3-large"):
    """Embed query using OpenAI API. Returns normalized embedding vector."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package required for RAG. Run: pip install openai")

    client = OpenAI()  # Uses OPENAI_API_KEY env var
    response = client.embeddings.create(model=model, input=[query_text])
    embedding = response.data[0].embedding

    # Normalize the embedding (OpenAI embeddings are already normalized, but be explicit)
    import math
    norm = math.sqrt(sum(x * x for x in embedding))
    return [x / (norm + 1e-8) for x in embedding]


def _cosine(a, b) -> float:
    """Cosine similarity for normalized vectors; 'b' may be numpy array or list."""
    if hasattr(b, "tolist"):
        b = b.tolist()
    return float(sum(x * y for x, y in zip(a, b)))


def _dedupe_by_backward(files: list[str], backward_docs: dict[str, str]) -> list[str]:
    """Keep first occurrence for identical backward docs (hash by MD5 of text)."""
    seen: set[str] = set()
    out: list[str] = []
    for fp in files:
        bwd = backward_docs.get(fp, "")
        h = hashlib.md5(bwd.encode("utf-8")).hexdigest()[:16]
        if h in seen:
            continue
        seen.add(h)
        out.append(fp)
    return out


def build_rag_block(
    *,
    index_path: str,
    fwd_source: str,
    top_k: int = 2,
    min_sim: float = 0.75,
    token_budget_chars: int = 2000 * 4,  # rough chars/token
    debug: bool = False,
) -> str:
    """
    Build a compact RAG block with forward previews and backward references.

    Uses vector similarity (cosine) to find similar forward kernels:
    1. Embed query using OpenAI API
    2. Compute cosine similarities with all indexed embeddings
    3. Filter by min_sim threshold
    4. Return top-k results after de-duplication

    Returns empty string when disabled or unavailable; caller may append it to
    the LLM header verbatim.
    """
    from .utils import _env_truthy as _verbose_check
    VERBOSE = _verbose_check("KERNEL_AGENT_VERBOSE", "1")

    if not _truthy(os.environ.get("KERNEL_AGENT_RAG"), "0"):
        return ""

    if not index_path or not os.path.exists(index_path):
        if VERBOSE:
            print(f"[kernel-agent][RAG] Index not found: {index_path}")
        return ""

    try:
        embeddings, documents, backward_docs, file_list, openai_model = _load_index(index_path)
    except Exception as e:
        if VERBOSE:
            print(f"[kernel-agent][RAG] Failed to load index: {type(e).__name__}: {e}")
        return ""

    # Embed the query using OpenAI API
    try:
        query_embedding = _embed_query(fwd_source, model=openai_model)
    except Exception as e:
        if VERBOSE:
            print(f"[kernel-agent][RAG] Failed to embed query: {type(e).__name__}: {e}")
        return ""

    # Compute cosine similarities with all indexed embeddings
    all_similarities: List[Tuple[str, float]] = []
    for fp in file_list:
        stored_embedding = embeddings[fp]
        sim = _cosine(query_embedding, stored_embedding)
        all_similarities.append((fp, sim))

    # Sort all similarities (descending)
    all_similarities.sort(key=lambda x: x[1], reverse=True)

    # Debug output: show top 10 and statistics
    if debug:
        print(f"\n📊 Top 10 most similar kernels (threshold: {min_sim:.2f}):")
        for i, (fp, sim) in enumerate(all_similarities[:10], 1):
            marker = "✓" if sim >= min_sim else "✗"
            print(f"  [{i}] {marker} {sim:.4f} → {fp}")

        above_threshold_count = sum(1 for _, sim in all_similarities if sim >= min_sim)
        print(f"\n📈 Statistics:")
        print(f"  Kernels above threshold: {above_threshold_count}/{len(file_list)}")
        print(f"  Top similarity: {all_similarities[0][1]:.4f}")
        if len(all_similarities) > 1:
            print(f"  Median similarity: {all_similarities[len(all_similarities)//2][1]:.4f}")

    # Filter by min_sim threshold
    similarities = [(fp, sim) for fp, sim in all_similarities if sim >= min_sim]

    if VERBOSE and similarities:
        top_sim = similarities[0][1] if similarities else 0.0
        print(f"[kernel-agent][RAG] Found {len(similarities)} kernels >= {min_sim:.2f} (top: {top_sim:.3f})")

    # Take top-k and extract file paths
    ranked = [fp for fp, _ in similarities[:top_k]]

    # De-duplicate by backward content
    ranked = _dedupe_by_backward(ranked, backward_docs)[:top_k]

    if not ranked:
        if VERBOSE:
            print(f"[kernel-agent][RAG] No similar kernels found (min_sim={min_sim:.2f})")
        return ""

    if VERBOSE:
        print(f"[kernel-agent][RAG] Retrieved {len(ranked)} backward reference(s)")

    # Compose block under budget
    per_example_chars = max(200, token_budget_chars // max(1, len(ranked)))
    lines: list[str] = []
    lines.append("[Retrieved backward references — adapt, don't copy]\n")
    lines.append("- Use these as patterns to ADAPT my current backward. They are similar but NOT exactly for my forward.\n")
    lines.append("- Edit ONLY my backward file to match MY forward's signature and semantics; no unrelated refactors.\n")

    for fp in ranked:
        # fwd_full = documents.get(fp, "")
        bwd_full = backward_docs.get(fp, "")
        bwd_snip = _truncate(bwd_full, per_example_chars)

        if VERBOSE:
            print(f"[kernel-agent][RAG DEBUG] {fp}: bwd_full={len(bwd_full)} chars, bwd_snip={len(bwd_snip)} chars, per_example_chars={per_example_chars}")

        if bwd_snip:
            lines.append("Backward (reference):")
            lines.append(bwd_snip)
        else:
            if VERBOSE:
                print(f"[kernel-agent][RAG DEBUG] WARNING: bwd_snip is empty for {fp}!")

    lines.append("\nAdaptation checklist for MY backward:")
    lines.append("- Match argument order, dtypes, tl.constexpr, grid mapping (program_id axes).")
    lines.append("- Match pointer math/strides, masks/tail handling, loop bounds/steps.")
    lines.append("- Keep accumulation dtype and cast boundaries consistent with my forward.")
    # lines.append("- Keep tl.atomic_* unless current phase explicitly removes them.")

    out = "\n".join(lines)
    truncated = _truncate(out, token_budget_chars)

    if VERBOSE:
        print(f"[kernel-agent][RAG DEBUG] Block length before truncate: {len(out)}")
        print(f"[kernel-agent][RAG DEBUG] Block length after truncate: {len(truncated)}")
        print(f"[kernel-agent][RAG DEBUG] Token budget: {token_budget_chars}")

    return truncated



def main() -> int:
    """CLI debug tool for RAG retrieval. Run: python -m kernel_agent.rag --help"""
    import argparse
    from pathlib import Path
    from .utils import redact_torch_fn

    # Compute default index path (invariant to cwd)
    default_index = str(Path(__file__).parent / "kernel_embeddings.pkl")

    ap = argparse.ArgumentParser(
        "kernel_agent.rag",
        description="Debug RAG retrieval: show what similar kernels would be retrieved",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--file-path", required=True, help="Path to forward .py file")
    ap.add_argument("--index", type=str, default=default_index, help="Path to embeddings index")
    ap.add_argument("--topk", type=int, default=2, help="Top-k references to retrieve")
    ap.add_argument("--min-sim", type=float, default=0.75, help="Minimum cosine similarity threshold")
    ap.add_argument("--show-block", action="store_true", help="Print the full RAG block")
    args = ap.parse_args()

    if not os.path.exists(args.index):
        print(f"❌ Index not found: {args.index}")
        return 1

    # Read forward kernel
    fwd_source = redact_torch_fn(args.file_path, None)
    if not fwd_source:
        print("❌ Failed to read or redact forward file")
        return 1

    print("== Forward summary ==")
    print(f"file: {args.file_path}")
    print(f"chars: {len(fwd_source)} | lines: {len(fwd_source.splitlines())}")

    # Enable RAG and call with debug=True
    orig_rag = os.environ.get("KERNEL_AGENT_RAG")
    os.environ["KERNEL_AGENT_RAG"] = "1"

    try:
        block = build_rag_block(
            index_path=args.index,
            fwd_source=fwd_source,
            top_k=args.topk,
            min_sim=args.min_sim,
            token_budget_chars=7500,
            debug=True,
        )

        if args.show_block:
            print("\n== RAG block (as sent to model) ==")
            if block:
                print(block)
                print(f"\n[Block size: {len(block)} chars, ~{len(block)//4} tokens]")
            else:
                print("(empty - no similar kernels found)")

        if not block:
            print(f"\n⚠️  RAG block is empty (no kernels above threshold)")

    finally:
        if orig_rag is None:
            os.environ.pop("KERNEL_AGENT_RAG", None)
        else:
            os.environ["KERNEL_AGENT_RAG"] = orig_rag

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
