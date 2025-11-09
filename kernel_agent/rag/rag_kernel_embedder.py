#!/usr/bin/env python3
"""
RAG system for Triton kernel matching.
Embeds triton kernels and enables similarity search to find relevant backward implementations.
"""

import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class KernelPair:
    """Container for forward/backward kernel pair from a single file."""

    def __init__(self, file_path: str, forward: str, backward: str):
        self.file_path = file_path
        self.forward = forward
        self.backward = backward
        self.repo = Path(file_path).parts[0] if Path(file_path).parts else ""
        self.function_name = Path(file_path).stem


def extract_forward_backward_kernels(file_path: str) -> Dict[str, str]:
    """
    Extract combined forward and backward Triton kernel sections from generated files.

    Generated files have clean sections created by autograd_function_writer.py:
        # SHARED HELPERS (Used by both forward and backward)
        [shared kernel code and imports]

        # FORWARD Triton Kernels
        [forward kernel code with decorators and comments]

        # BACKWARD Triton Kernels
        [backward kernel code with decorators and comments]

        # autograd.Function Class Definition
        [class methods - ignore]

    Returns dict with only 2 fields:
        'forward_with_shared': SHARED + separator + FORWARD (for embedding)
        'backward': Just BACKWARD (for retrieval - no SHARED needed)

    If no SHARED section exists, returns just FORWARD without separator.

    Raises ValueError if:
        - FORWARD section is missing or empty (all files must have FORWARD)
        - BACKWARD section is missing or empty (all files must have BACKWARD)

    No AST parsing needed - the generated files already have perfectly structured
    sections with exactly what we need. Just extract text between markers.
    Preserves everything: decorators, comments, formatting, whitespace.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Section markers used in generated files
    shared_marker = "# SHARED HELPERS (Used by both forward and backward)"
    forward_marker = "# FORWARD Triton Kernels"
    backward_marker = "# BACKWARD Triton Kernels"
    class_marker = "# autograd.Function Class Definition"

    shared_text = ""
    forward_text = ""
    backward_text = ""

    # Extract shared section (if exists)
    if shared_marker in content:
        # Get everything after shared marker
        after_shared = content.split(shared_marker, 1)[1]

        # Files with SHARED must have FORWARD marker - otherwise malformed
        if forward_marker not in after_shared:
            raise ValueError(f"Malformed file: has SHARED HELPERS marker but no FORWARD marker: {file_path}")

        shared_text = after_shared.split(forward_marker, 1)[0]

    # Extract forward section
    if forward_marker in content:
        # Get everything after forward marker
        after_forward = content.split(forward_marker, 1)[1]

        # Stop at backward marker or class marker (whichever comes first)
        if backward_marker in after_forward:
            forward_text = after_forward.split(backward_marker, 1)[0]
        elif class_marker in after_forward:
            forward_text = after_forward.split(class_marker, 1)[0]
        else:
            forward_text = after_forward

    # Extract backward section
    if backward_marker in content:
        # Get everything after backward marker
        after_backward = content.split(backward_marker, 1)[1]

        # Stop at class marker
        if class_marker in after_backward:
            backward_text = after_backward.split(class_marker, 1)[0]
        else:
            backward_text = after_backward

    # Clean up: remove leading/trailing whitespace but preserve internal structure
    shared_text = shared_text.strip()
    forward_text = forward_text.strip()
    backward_text = backward_text.strip()

    # All generated files must have FORWARD section
    if not forward_text:
        raise ValueError(f"Malformed file: missing or empty FORWARD section: {file_path}")

    # All generated files must have BACKWARD section
    if not backward_text:
        raise ValueError(f"Malformed file: missing or empty BACKWARD section: {file_path}")

    # Create combined versions for embedding and retrieval
    # Add section separator for clarity when SHARED exists
    separator = "\n\n" + "="*60 + "\n\n"

    # WHY include SHARED in forward but NOT backward?
    # ------------------------------------------------
    # RAG workflow:
    #   1. User provides their code (SHARED + FORWARD) → embedded as query
    #   2. We search against indexed keys (SHARED + FORWARD)
    #   3. We retrieve matched result (BACKWARD only) → shown to LLM along with user's query
    #
    # Critical: The LLM sees BOTH the user's query (which includes their SHARED+FORWARD)
    # AND the retrieved backward. No need to duplicate SHARED helpers in the backward
    # since they're already visible to the LLM in the user's query text.

    # Combine SHARED + FORWARD for embedding
    if shared_text:
        forward_with_shared = shared_text + separator + forward_text
    else:
        forward_with_shared = forward_text

    # Return BACKWARD only (no SHARED - it's already in the user's query shown to LLM)
    return {
        'forward_with_shared': forward_with_shared,  # For embedding (KEY)
        'backward': backward_text  # For retrieval (no SHARED)
    }


class KernelEmbedder:
    """
    Manages kernel embeddings and similarity search using OpenAI embeddings.
    Uses text-embedding-3-large for high-quality code embeddings.
    """

    def __init__(self, openai_model: str = 'text-embedding-3-large'):
        """
        Initialize embedder. OpenAI client is created lazily when needed.

        Args:
            openai_model: OpenAI embedding model to use (default: text-embedding-3-large)
        """
        self.openai_model = openai_model
        self.openai_client = None  # Initialize lazily when needed for embedding

        self.embeddings = {}  # file_path -> embedding vector
        self.documents = {}  # file_path -> FORWARD kernel section text (embedded)
        self.backward_documents = {}  # file_path -> BACKWARD kernel section text (for retrieval)
        self.kernel_pairs = {}  # file_path -> KernelPair object for clean access
        self.file_list = []  # Ordered list of file paths
        self.embedding_dim = None  # Will be set after first embedding

    def _init_openai_client(self):
        """Initialize OpenAI client when needed. Called lazily on first use."""
        if self.openai_client is not None:
            return  # Already initialized

        try:
            from openai import OpenAI
            self.openai_client = OpenAI()  # Uses OPENAI_API_KEY env var
            print(f"Using OpenAI embedding model: {self.openai_model}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def get_kernel_pair(self, file_path: str) -> KernelPair:
        """Get the forward/backward pair for a file."""
        if file_path not in self.kernel_pairs:
            # Fallback for old pickles without kernel_pairs
            return KernelPair(
                file_path,
                self.documents.get(file_path, ""),
                self.backward_documents.get(file_path, "")
            )
        return self.kernel_pairs[file_path]

    def _truncate_text(self, text: str, max_tokens: int = 6000) -> str:
        """
        Truncate text to fit within token limit.

        Uses conservative character-based approximation: ~3.2 chars per token
        (Triton kernel code measured at 3.24-3.32 chars/token with cl100k_base).
        Target 6000 tokens to stay well below 8192 token limit with safety margin.
        """
        max_chars = int(max_tokens * 3.2)  # Conservative: 3.2 chars per token for Triton kernels
        if len(text) <= max_chars:
            return text

        # Truncate and add marker
        truncated = text[:max_chars]
        # Try to truncate at line boundary for cleaner cut
        last_newline = truncated.rfind('\n')
        if last_newline > max_chars * 0.9:  # If within last 10%, use it
            truncated = truncated[:last_newline]

        return truncated

    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode texts using OpenAI API with batching and rate limit handling.

        OpenAI supports up to 2048 inputs per request, uses smaller batches for safety.
        Truncates documents that exceed 8192 token limit.
        Returns normalized embeddings as numpy array.
        """
        # Initialize OpenAI client if not already done (lazy initialization)
        self._init_openai_client()

        batch_size = 100  # Conservative batch size
        all_embeddings = []

        # Truncate texts that are too long
        truncated_texts = [self._truncate_text(t) for t in texts]

        # Count how many were truncated
        num_truncated = sum(1 for orig, trunc in zip(texts, truncated_texts) if len(orig) != len(trunc))
        if num_truncated > 0 and show_progress:
            print(f"Note: Truncated {num_truncated} documents that exceeded 8192 token limit")

        total_batches = (len(truncated_texts) + batch_size - 1) // batch_size

        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            if show_progress:
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")

            # Call OpenAI API with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.embeddings.create(
                        model=self.openai_model,
                        input=batch
                    )

                    # Extract embeddings from response
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise RuntimeError(f"Failed to get embeddings after {max_retries} attempts: {e}")

            # Rate limiting: small delay between batches to avoid hitting limits
            if i + batch_size < len(texts):
                time.sleep(0.1)

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Normalize embeddings (OpenAI embeddings are already normalized, but explicit is better)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)

        return embeddings_array

    def build_index(self, generated_dir: str, save_path: str = "kernel_embeddings.pkl"):
        """
        Build embeddings for all kernel files in generated directory.

        Extracts text from structured sections (no AST parsing needed):
        - SHARED + FORWARD sections are embedded for similarity search
        - BACKWARD sections (without SHARED) are stored for retrieval

        Workflow:
        1. User provides SHARED+FORWARD as query (their complete forward context)
        2. Similarity search matches against indexed SHARED+FORWARD patterns
        3. Retrieved result is BACKWARD only (user's query already contains SHARED)

        Why not include SHARED in backward retrieval:
        - LLM sees both the user's query (with SHARED+FORWARD) and the retrieved backward
        - No need to duplicate SHARED helpers since they're already visible in query text
        - Keeps retrieval focused on the actual backward implementation

        Saves embeddings to disk for fast reloading.
        """
        generated_path = Path(generated_dir)

        # Recursively find all .py files
        py_files = list(generated_path.rglob("*.py"))
        print(f"Found {len(py_files)} Python files")

        # Extract forward and backward kernel sections
        # Embed SHARED+FORWARD, store BACKWARD (without SHARED) for retrieval
        self.backward_documents = {}

        valid_files = 0
        skipped_malformed = 0
        separator = "\n\n" + "="*60 + "\n\n"  # Match actual separator we insert between SHARED and FORWARD
        files_with_shared_kernels = 0

        for file_path in py_files:
            try:
                # Extract sections by splitting at markers - simple string operations
                # May raise ValueError for malformed files (missing FORWARD/BACKWARD, etc.)
                sections = extract_forward_backward_kernels(str(file_path))
                forward_with_shared = sections['forward_with_shared']
                backward = sections['backward']
            except ValueError as e:
                # Skip malformed files (missing FORWARD/BACKWARD section, etc.)
                skipped_malformed += 1
                continue

            # Detect if file has SHARED section with actual Triton kernels
            # Just having helpers isn't important - we care about @triton.jit/@triton.autotune in SHARED
            has_shared_kernels = False
            if separator in forward_with_shared:
                # Extract SHARED section (before separator)
                shared_section = forward_with_shared.split(separator)[0]
                # Check if SHARED has Triton kernel decorators
                has_shared_kernels = '@triton.jit' in shared_section or '@triton.autotune' in shared_section

            # Store section text
            rel_path = str(file_path.relative_to(generated_path))
            self.documents[rel_path] = forward_with_shared  # SHARED+FORWARD for embedding
            self.backward_documents[rel_path] = backward  # BACKWARD only (no SHARED)

            # Create KernelPair for cleaner access in tests
            self.kernel_pairs[rel_path] = KernelPair(rel_path, forward_with_shared, backward)
            self.file_list.append(rel_path)
            valid_files += 1

            if has_shared_kernels:
                files_with_shared_kernels += 1

            if valid_files % 50 == 0:
                print(f"Processed {valid_files} files...")

        print(f"Extracted kernels from {valid_files} files")
        print(f"  Files with SHARED Triton kernels: {files_with_shared_kernels} ({100*files_with_shared_kernels/valid_files:.1f}%)")
        print(f"  Skipped {skipped_malformed} malformed files (missing FORWARD/BACKWARD section)")
        print(f"Generating embeddings for SHARED+FORWARD sections...")

        # Generate embeddings in batch using configured backend
        docs_list = [self.documents[f] for f in self.file_list]
        embeddings_array = self.encode(docs_list, show_progress=True)

        # Store normalized embeddings
        for file_path, embedding in zip(self.file_list, embeddings_array):
            self.embeddings[file_path] = embedding

        # Store embedding dimension for later verification
        if len(embeddings_array) > 0:
            self.embedding_dim = embeddings_array[0].shape[0]
            print(f"Embedding dimension: {self.embedding_dim}")

        # Save to disk
        self.save(save_path)
        print(f"Saved embeddings to {save_path}")

        return valid_files

    def save(self, path: str):
        """Save embeddings and metadata to disk."""
        data = {
            'embeddings': self.embeddings,
            'documents': self.documents,  # Forward kernel sections (embedded)
            'backward_documents': self.backward_documents,  # Backward kernel sections (for retrieval)
            'kernel_pairs': self.kernel_pairs,  # KernelPair objects for clean access
            'file_list': self.file_list,
            'openai_model': self.openai_model,
            'embedding_dim': self.embedding_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load embeddings from disk."""
        print(f"Loading embeddings from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.embeddings = data['embeddings']
        self.documents = data['documents']
        self.file_list = data['file_list']
        self.backward_documents = data.get('backward_documents', {})
        self.kernel_pairs = data.get('kernel_pairs', {})  # Backward compatible
        self.embedding_dim = data.get('embedding_dim', None)

        # Display model info if available
        saved_model = data.get('openai_model', 'unknown')

        print(f"Loaded {len(self.embeddings)} embeddings")
        if self.backward_documents:
            print(f"Loaded {len(self.backward_documents)} backward kernel sections")
        if self.kernel_pairs:
            print(f"Loaded {len(self.kernel_pairs)} kernel pairs")
        if self.embedding_dim:
            print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Model: {saved_model}")

    def search_similar(self, query_code: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find top-k most similar kernel files to query code.
        Returns list of (file_path, similarity_score) tuples.
        """
        if not self.embeddings:
            raise ValueError("No embeddings loaded. Call build_index() or load() first.")

        # Check if query_code is already in our documents (for test self-retrieval)
        # This allows testing without API key when query is an exact match of stored document
        query_embedding = None
        for file_path in self.file_list:
            if self.documents.get(file_path) == query_code:
                # Use the stored embedding for this exact document
                query_embedding = self.embeddings[file_path]
                break

        # If not found in documents, embed the query (requires API key)
        if query_embedding is None:
            query_embedding = self.encode([query_code], show_progress=False)[0]

        # Compute cosine similarities
        # Note: embeddings are already normalized when stored (see encode() line 240)
        # query_embedding is also normalized (either from storage or encode())
        similarities = {}
        for file_path in self.file_list:
            stored_embedding = self.embeddings[file_path]

            # Direct dot product since both vectors are already normalized
            # Cosine similarity = dot product of normalized vectors (range: -1 to 1)
            similarity = np.dot(query_embedding, stored_embedding)
            similarities[file_path] = float(similarity)

        # Sort by similarity
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:top_k]

    def get_document(self, file_path: str) -> str:
        """Get the FORWARD kernel document for a file (what was embedded)."""
        return self.documents.get(file_path, "")

    def get_backward_kernels(self, file_path: str) -> str:
        """
        Get the BACKWARD kernel document for a file.
        This is what you paste into LLM context as examples for gradient computation.
        """
        return self.backward_documents.get(file_path, "")


def main():
    """Build kernel embeddings index using OpenAI embeddings."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Build kernel embeddings index using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build embeddings with default model (text-embedding-3-large)
  python rag_kernel_embedder.py ../generated/

  # Use different OpenAI model
  python rag_kernel_embedder.py ../generated/ --model text-embedding-3-small

  # Custom output path
  python rag_kernel_embedder.py ../generated/ --output my_embeddings.pkl

Requires OPENAI_API_KEY environment variable to be set.
Estimated cost: ~$0.40 for 500+ files with text-embedding-3-large
        """
    )

    parser.add_argument(
        'generated_dir',
        help='Path to generated kernels directory'
    )
    parser.add_argument(
        '--model',
        default='text-embedding-3-large',
        help='OpenAI embedding model to use (default: text-embedding-3-large)'
    )
    parser.add_argument(
        '--output', '-o',
        default='kernel_embeddings.pkl',
        help='Output file path (default: kernel_embeddings.pkl)'
    )

    args = parser.parse_args()

    # Verify OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=your-api-key")
        sys.exit(1)

    # Build index
    print(f"Building embeddings index...")
    print(f"Model: {args.model}")
    print(f"Estimated cost: ~$0.40 for 500+ files")

    embedder = KernelEmbedder(openai_model=args.model)
    num_files = embedder.build_index(args.generated_dir, save_path=args.output)

    print(f"\nIndexed {num_files} files with triton kernels")
    print(f"Embeddings saved to {args.output}")


if __name__ == "__main__":
    main()