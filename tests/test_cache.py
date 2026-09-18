"""Tests for VectorSemanticCache and ExactHashCache."""

import pytest
import numpy as np

from src.cache import ExactHashCache, VectorSemanticCache, EmbeddingModelWrapper


def test_exact_hash_cache():
    cache = ExactHashCache()
    assert len(cache) == 0

    cache.set("How do I reset my password?", "Click forgot password.", metadata={"source": "faq"})
    assert len(cache) == 1

    # Exact match
    res = cache.get("How do I reset my password?")
    assert res is not None
    assert res[0] == "Click forgot password."

    # Normalized match (case and punctuation variations)
    res_norm = cache.get("how do i reset my password???")
    assert res_norm is not None
    assert res_norm[0] == "Click forgot password."

    # Miss
    assert cache.get("How do I delete my account?") is None

    # Clear
    cache.clear()
    assert len(cache) == 0


def test_vector_semantic_cache_basic():
    # Use lightweight all-MiniLM-L6-v2 for fast testing
    cache = VectorSemanticCache(model_name="all-MiniLM-L6-v2", threshold=0.85, exact_fallback=True)
    cache.clear()

    # Store entry
    entry = cache.store(
        query="How do I return a damaged item?",
        response="Go to Orders and click return item.",
        entry_id="TP-001"
    )
    assert entry.id == "TP-001"
    assert cache.size == 1

    # Query with identical text (Exact match)
    hit_exact = cache.query("How do I return a damaged item?")
    assert hit_exact.is_hit is True
    assert hit_exact.hit_type == "exact"
    assert hit_exact.similarity_score == 1.0
    assert hit_exact.response == "Go to Orders and click return item."

    # Query with semantic paraphrase
    hit_semantic = cache.query("What is the procedure for returning broken merchandise?", threshold=0.70)
    assert hit_semantic.is_hit is True
    assert hit_semantic.similarity_score >= 0.70
    assert hit_semantic.response == "Go to Orders and click return item."

    # Query with completely unrelated text (Semantic Miss)
    hit_unrelated = cache.query("How to make chocolate chip cookies from scratch?", threshold=0.75)
    assert hit_unrelated.is_hit is False
    assert hit_unrelated.hit_type == "miss"
    assert hit_unrelated.similarity_score < 0.75

    # Batch store
    batch_items = [
        ("What is Docker?", "Docker is a container platform.", {}),
        ("What is Kubernetes?", "Kubernetes is a container orchestrator.", {}),
    ]
    entries = cache.store_batch(batch_items)
    assert len(entries) == 2
    assert cache.size == 3


def test_pairwise_similarity_computation():
    cache = VectorSemanticCache(model_name="all-MiniLM-L6-v2")
    pairs = [
        ("How to stop docker container", "How to kill docker container"),
        ("Apple stock price today", "Recipe for apple pie dessert"),
    ]
    sims = cache.compute_pairwise_similarities(pairs)
    assert len(sims) == 2
    assert sims[0] > 0.80  # Related
    assert sims[1] < 0.60  # Distant
