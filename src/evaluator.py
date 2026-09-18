"""
Semantic Cache Evaluation Engine.

Provides deep statistical analysis, pairwise similarity distributions,
overlap zone quantification, and threshold sweep analytics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.cache import EmbeddingModelWrapper, VectorSemanticCache


@dataclass
class DistributionStats:
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    @classmethod
    def from_scores(cls, scores: List[float]) -> DistributionStats:
        if not scores:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        arr = np.array(scores)
        return cls(
            count=len(arr),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
            p10=float(np.percentile(arr, 10)),
            p25=float(np.percentile(arr, 25)),
            p50=float(np.percentile(arr, 50)),
            p75=float(np.percentile(arr, 75)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )


@dataclass
class EvaluatedPair:
    id: str
    category: str
    subcategory: str
    domain: str
    query_a: str
    query_b: str
    response_a: str
    response_b: str
    is_cache_safe: bool
    hazard_severity: str
    similarity: float


@dataclass
class OverlapZoneAnalysis:
    safe_min: float
    safe_p10: float
    safe_median: float
    safe_mean: float
    unsafe_max: float
    unsafe_p90: float
    unsafe_median: float
    unsafe_mean: float
    overlap_range_min: float
    overlap_range_max: float
    has_overlap: bool
    overlap_pair_count: int
    critical_hazard_above_085: int
    critical_hazard_above_090: int
    dangerous_negations_above_085: int
    dangerous_negations_above_090: int


@dataclass
class ThresholdMetrics:
    threshold: float
    total_safe: int
    total_unsafe: int
    true_hits: int
    false_hits: int
    true_hit_rate: float  # TPR (Recall on safe)
    false_hit_rate: float  # FPR (Fatal rate on unsafe)
    precision: float
    accuracy: float
    negation_false_hits: int
    entity_false_hits: int
    numerical_false_hits: int
    critical_false_hits: int


class CacheEvaluationEngine:
    """Evaluation Engine for benchmarking semantic caching failure modes."""

    def __init__(
        self,
        benchmark_data: Optional[List[Dict[str, Any]]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if benchmark_data is not None:
            self.dataset = benchmark_data
        elif dataset_path is not None:
            with open(dataset_path, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
        else:
            default_path = Path(__file__).resolve().parent.parent / "dataset" / "benchmark_300.json"
            if default_path.exists():
                with open(default_path, "r", encoding="utf-8") as f:
                    self.dataset = json.load(f)
            else:
                from dataset.generate_dataset import get_all_benchmark_data
                self.dataset = get_all_benchmark_data()

    def evaluate_model(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        encoder: Optional[EmbeddingModelWrapper] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate pairwise cosine similarity across all 300 benchmark pairs.
        
        Returns a comprehensive dictionary containing:
        - Evaluated pairs with individual similarity scores
        - Category-level distribution statistics
        - Subcategory-level distribution statistics
        - Domain-level distribution statistics
        - Overlap zone analysis
        - Threshold sweeps (tau from 0.70 to 0.99)
        """
        if encoder is None:
            encoder = EmbeddingModelWrapper(model_name=model_name)

        pairs = [(item["query_a"], item["query_b"]) for item in self.dataset]
        queries_a = [p[0] for p in pairs]
        queries_b = [p[1] for p in pairs]

        vecs_a = encoder.encode(queries_a, normalize=True)
        vecs_b = encoder.encode(queries_b, normalize=True)

        similarities = np.sum(vecs_a * vecs_b, axis=1)

        evaluated_pairs: List[EvaluatedPair] = []
        category_scores: Dict[str, List[float]] = {}
        subcategory_scores: Dict[str, List[float]] = {}
        domain_scores: Dict[str, List[float]] = {}

        safe_scores: List[float] = []
        unsafe_scores: List[float] = []

        for i, item in enumerate(self.dataset):
            sim = float(similarities[i])
            cat = item["category"]
            subcat = item["subcategory"]
            domain = item["domain"]
            is_safe = item["is_cache_safe"]

            ev_pair = EvaluatedPair(
                id=item["id"],
                category=cat,
                subcategory=subcat,
                domain=domain,
                query_a=item["query_a"],
                query_b=item["query_b"],
                response_a=item["response_a"],
                response_b=item["response_b"],
                is_cache_safe=is_safe,
                hazard_severity=item["hazard_severity"],
                similarity=sim,
            )
            evaluated_pairs.append(ev_pair)

            category_scores.setdefault(cat, []).append(sim)
            subcategory_scores.setdefault(subcat, []).append(sim)
            domain_scores.setdefault(domain, []).append(sim)

            if is_safe:
                safe_scores.append(sim)
            else:
                unsafe_scores.append(sim)

        # Calculate Distributions
        cat_stats = {cat: asdict(DistributionStats.from_scores(scs)) for cat, scs in category_scores.items()}
        subcat_stats = {sc: asdict(DistributionStats.from_scores(scs)) for sc, scs in subcategory_scores.items()}
        domain_stats = {d: asdict(DistributionStats.from_scores(scs)) for d, scs in domain_scores.items()}

        safe_stats = DistributionStats.from_scores(safe_scores)
        unsafe_stats = DistributionStats.from_scores(unsafe_scores)

        # Overlap Zone Analysis
        # Overlap exists if max(unsafe) > min(safe)
        has_overlap = unsafe_stats.max > safe_stats.min
        overlap_range_min = float(min(safe_stats.min, unsafe_stats.min)) if has_overlap else 0.0
        overlap_range_max = float(unsafe_stats.max)

        # Count unsafe pairs with similarity >= safe_stats.min
        overlap_pairs = [p for p in evaluated_pairs if not p.is_cache_safe and p.similarity >= safe_stats.min]

        # Specific danger counts
        crit_085 = sum(1 for p in evaluated_pairs if not p.is_cache_safe and p.hazard_severity == "critical" and p.similarity >= 0.85)
        crit_090 = sum(1 for p in evaluated_pairs if not p.is_cache_safe and p.hazard_severity == "critical" and p.similarity >= 0.90)
        neg_085 = sum(1 for p in evaluated_pairs if p.category == "negation" and p.similarity >= 0.85)
        neg_090 = sum(1 for p in evaluated_pairs if p.category == "negation" and p.similarity >= 0.90)

        overlap_analysis = OverlapZoneAnalysis(
            safe_min=safe_stats.min,
            safe_p10=safe_stats.p10,
            safe_median=safe_stats.median,
            safe_mean=safe_stats.mean,
            unsafe_max=unsafe_stats.max,
            unsafe_p90=unsafe_stats.p90,
            unsafe_median=unsafe_stats.median,
            unsafe_mean=unsafe_stats.mean,
            overlap_range_min=safe_stats.min,
            overlap_range_max=unsafe_stats.max,
            has_overlap=has_overlap,
            overlap_pair_count=len(overlap_pairs),
            critical_hazard_above_085=crit_085,
            critical_hazard_above_090=crit_090,
            dangerous_negations_above_085=neg_085,
            dangerous_negations_above_090=neg_090,
        )

        # Threshold Sweep Evaluation
        threshold_sweeps: List[ThresholdMetrics] = []
        sweep_thresholds = [round(t, 2) for t in np.arange(0.70, 0.995, 0.02)]
        if 0.85 not in sweep_thresholds:
            sweep_thresholds.append(0.85)
        if 0.90 not in sweep_thresholds:
            sweep_thresholds.append(0.90)
        sweep_thresholds.sort()

        total_safe = len(safe_scores)
        total_unsafe = len(unsafe_scores)

        for tau in sweep_thresholds:
            true_hits = sum(1 for p in evaluated_pairs if p.is_cache_safe and p.similarity >= tau)
            false_hits = sum(1 for p in evaluated_pairs if not p.is_cache_safe and p.similarity >= tau)
            neg_fp = sum(1 for p in evaluated_pairs if p.category == "negation" and p.similarity >= tau)
            ent_fp = sum(1 for p in evaluated_pairs if p.category == "entity_swap" and p.similarity >= tau)
            num_fp = sum(1 for p in evaluated_pairs if p.category == "numerical_temporal_swap" and p.similarity >= tau)
            crit_fp = sum(1 for p in evaluated_pairs if not p.is_cache_safe and p.hazard_severity == "critical" and p.similarity >= tau)

            tpr = (true_hits / total_safe) if total_safe > 0 else 0.0
            fpr = (false_hits / total_unsafe) if total_unsafe > 0 else 0.0
            precision = (true_hits / (true_hits + false_hits)) if (true_hits + false_hits) > 0 else 1.0
            accuracy = (true_hits + (total_unsafe - false_hits)) / (total_safe + total_unsafe)

            threshold_sweeps.append(
                ThresholdMetrics(
                    threshold=tau,
                    total_safe=total_safe,
                    total_unsafe=total_unsafe,
                    true_hits=true_hits,
                    false_hits=false_hits,
                    true_hit_rate=round(tpr, 4),
                    false_hit_rate=round(fpr, 4),
                    precision=round(precision, 4),
                    accuracy=round(accuracy, 4),
                    negation_false_hits=neg_fp,
                    entity_false_hits=ent_fp,
                    numerical_false_hits=num_fp,
                    critical_false_hits=crit_fp,
                )
            )

        return {
            "model_name": model_name,
            "embedding_dimension": encoder.dimension,
            "total_benchmark_pairs": len(self.dataset),
            "category_statistics": cat_stats,
            "subcategory_statistics": subcat_stats,
            "domain_statistics": domain_stats,
            "safe_distribution": asdict(safe_stats),
            "unsafe_distribution": asdict(unsafe_stats),
            "overlap_zone_analysis": asdict(overlap_analysis),
            "threshold_sweeps": [asdict(tm) for tm in threshold_sweeps],
            "evaluated_pairs": [asdict(p) for p in evaluated_pairs],
        }

    def to_dataframe(self, evaluated_pairs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert evaluated pairs to Pandas DataFrame for downstream tabular manipulation."""
        return pd.DataFrame(evaluated_pairs)
