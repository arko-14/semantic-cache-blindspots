"""
High-Performance Vector Semantic Cache and Exact Hash Fallback Engine.

Implements:
1. ExactHashCache: SHA-256 normalized hash cache for O(1) exact query matching.
2. EmbeddingModelWrapper: Cached HuggingFace SentenceTransformer wrapper with L2 normalization.
3. VectorSemanticCache: In-memory vector similarity cache with cosine similarity search,
   configurable similarity threshold tau, latency telemetry, and exact hash fallback.
"""

from __future__ import annotations

import hashlib
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CacheEntry:
    """Represents an entry stored in the cache."""
    id: str
    query: str
    embedding: np.ndarray
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"<CacheEntry id={self.id} query={self.query[:30]!r}>"


@dataclass
class CacheQueryResult:
    """Result of querying the semantic or hash cache."""
    is_hit: bool
    hit_type: str  # "exact", "semantic", or "miss"
    similarity_score: float
    matched_entry: Optional[CacheEntry] = None
    response: Optional[str] = None
    latency_ms: float = 0.0
    threshold_used: float = 0.85
    candidate_scores: List[Tuple[str, float]] = field(default_factory=list)


class ExactHashCache:
    """O(1) exact match cache using SHA-256 hash over normalized query text."""

    def __init__(self) -> None:
        self._store: Dict[str, Tuple[str, Any]] = {}

    @staticmethod
    def normalize_text(text: str) -> str:
        """Standardize text for exact matching: lowercase, strip punctuation & whitespace."""
        text = text.lower().strip()
        # Remove surrounding punctuation
        text = text.strip(string.punctuation)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text

    def _hash(self, normalized_text: str) -> str:
        return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Optional[Tuple[str, Any]]:
        norm = self.normalize_text(query)
        h = self._hash(norm)
        return self._store.get(h)

    def set(self, query: str, response: str, metadata: Any = None) -> None:
        norm = self.normalize_text(query)
        h = self._hash(norm)
        self._store[h] = (response, metadata)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class EmbeddingModelWrapper:
    """Wrapper for SentenceTransformer embedding models with caching and normalization."""

    _models: Dict[str, Any] = {}

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = self._load_model(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    @classmethod
    def _load_model(cls, model_name: str):
        if model_name not in cls._models:
            from sentence_transformers import SentenceTransformer
            cls._models[model_name] = SentenceTransformer(model_name)
        return cls._models[model_name]

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Generate embeddings. If normalize=True, vectors have L2 norm = 1.0."""
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        if single_input:
            return embeddings[0]
        return embeddings


class VectorSemanticCache:
    """
    In-memory vector semantic cache with cosine similarity search.
    
    Features:
    - Cosine similarity matching: u · v (for unit vectors).
    - Configurable similarity threshold tau (default 0.85).
    - Stage-0 Exact hash match fallback (optional).
    - Sub-millisecond vector similarity search over thousands of entries.
    - Full telemetry: hit type, similarity score, latency.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.85,
        exact_fallback: bool = True,
        embedding_model: Optional[EmbeddingModelWrapper] = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.exact_fallback = exact_fallback
        self.encoder = embedding_model or EmbeddingModelWrapper(model_name=model_name)

        self._exact_cache = ExactHashCache() if exact_fallback else None
        self._entries: List[CacheEntry] = []
        self._embeddings_matrix: Optional[np.ndarray] = None  # shape (N, D)

    @property
    def size(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._embeddings_matrix = None
        if self._exact_cache:
            self._exact_cache.clear()

    def store(
        self,
        query: str,
        response: str,
        entry_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> CacheEntry:
        """Store a new prompt-response pair in the semantic cache."""
        t0 = time.perf_counter()
        if embedding is None:
            embedding = self.encoder.encode(query, normalize=True)

        if entry_id is None:
            entry_id = f"entry_{len(self._entries):05d}"

        entry = CacheEntry(
            id=entry_id,
            query=query,
            embedding=embedding,
            response=response,
            metadata=metadata or {},
            created_at=time.time(),
        )

        self._entries.append(entry)

        # Update embeddings matrix
        if self._embeddings_matrix is None:
            self._embeddings_matrix = embedding.reshape(1, -1)
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, embedding.reshape(1, -1)])

        # Update exact cache
        if self._exact_cache is not None:
            self._exact_cache.set(query, response, entry)

        return entry

    def store_batch(
        self,
        items: List[Tuple[str, str, Optional[Dict[str, Any]]]],
        batch_size: int = 64,
    ) -> List[CacheEntry]:
        """Batch store multiple queries and responses for high throughput."""
        if not items:
            return []

        queries = [item[0] for item in items]
        embeddings = self.encoder.encode(queries, normalize=True)

        created_entries = []
        for i, (q, resp, meta) in enumerate(items):
            entry_id = f"entry_{len(self._entries):05d}"
            entry = CacheEntry(
                id=entry_id,
                query=q,
                embedding=embeddings[i],
                response=resp,
                metadata=meta or {},
                created_at=time.time(),
            )
            self._entries.append(entry)
            created_entries.append(entry)
            if self._exact_cache is not None:
                self._exact_cache.set(q, resp, entry)

        new_matrix = np.array([e.embedding for e in created_entries])
        if self._embeddings_matrix is None:
            self._embeddings_matrix = new_matrix
        else:
            self._embeddings_matrix = np.vstack([self._embeddings_matrix, new_matrix])

        return created_entries

    def query(
        self,
        query: str,
        threshold: Optional[float] = None,
        top_k: int = 1,
    ) -> CacheQueryResult:
        """
        Query the cache with an incoming prompt.
        
        Returns:
            CacheQueryResult with is_hit, similarity_score, matched_entry, latency.
        """
        t0 = time.perf_counter()
        tau = threshold if threshold is not None else self.threshold

        # Stage 0: Exact hash match check
        if self._exact_cache is not None:
            exact_hit = self._exact_cache.get(query)
            if exact_hit is not None:
                resp, entry = exact_hit
                latency = (time.perf_counter() - t0) * 1000.0
                return CacheQueryResult(
                    is_hit=True,
                    hit_type="exact",
                    similarity_score=1.0,
                    matched_entry=entry,
                    response=resp,
                    latency_ms=latency,
                    threshold_used=tau,
                    candidate_scores=[(entry.id, 1.0)] if entry else [],
                )

        # If cache is empty
        if self._embeddings_matrix is None or len(self._entries) == 0:
            latency = (time.perf_counter() - t0) * 1000.0
            return CacheQueryResult(
                is_hit=False,
                hit_type="miss",
                similarity_score=0.0,
                latency_ms=latency,
                threshold_used=tau,
            )

        # Stage 1: Dense Vector Similarity Search
        query_vec = self.encoder.encode(query, normalize=True)  # shape (D,)

        # Dot product against all stored embeddings (cosine similarity because both are L2 normalized)
        # _embeddings_matrix is (N, D), query_vec is (D,) -> sim_scores is (N,)
        sim_scores = np.dot(self._embeddings_matrix, query_vec)

        best_idx = int(np.argmax(sim_scores))
        best_score = float(sim_scores[best_idx])

        # Get top-k candidate scores for diagnostics
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        candidate_scores = [(self._entries[idx].id, float(sim_scores[idx])) for idx in top_indices]

        latency = (time.perf_counter() - t0) * 1000.0

        if best_score >= tau:
            matched = self._entries[best_idx]
            return CacheQueryResult(
                is_hit=True,
                hit_type="semantic",
                similarity_score=best_score,
                matched_entry=matched,
                response=matched.response,
                latency_ms=latency,
                threshold_used=tau,
                candidate_scores=candidate_scores,
            )
        else:
            return CacheQueryResult(
                is_hit=False,
                hit_type="miss",
                similarity_score=best_score,
                matched_entry=None,
                response=None,
                latency_ms=latency,
                threshold_used=tau,
                candidate_scores=candidate_scores,
            )

    def compute_similarity(self, query_a: str, query_b: str) -> float:
        """Compute the cosine similarity between two query strings."""
        vecs = self.encoder.encode([query_a, query_b], normalize=True)
        return float(np.dot(vecs[0], vecs[1]))

    def compute_pairwise_similarities(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Compute pairwise cosine similarities for a batch of query pairs efficiently."""
        if not pairs:
            return []

        queries_a = [p[0] for p in pairs]
        queries_b = [p[1] for p in pairs]

        vecs_a = self.encoder.encode(queries_a, normalize=True)
        vecs_b = self.encoder.encode(queries_b, normalize=True)

        # Element-wise dot product for each row: sum(vecs_a * vecs_b, axis=1)
        sims = np.sum(vecs_a * vecs_b, axis=1)
        return [float(s) for s in sims]
