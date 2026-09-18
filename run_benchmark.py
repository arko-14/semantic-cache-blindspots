"""
Unified Benchmark & Publication Pipeline for Semantic Cache Blindspots.

Executes:
1. Full 300-Query Benchmark Dataset Generation & Validation
2. Single-Stage Vector Semantic Cache Evaluation (MiniLM-L6 & BGE-Small)
3. Two-Stage Hybrid Verified Semantic Cache Evaluation
4. 3 Publication Visualizations:
   - Plot 1: Cosine Similarity Overlap Histogram
   - Plot 2: Semantic Cache ROC Curve
   - Plot 3: Financial Break-Even Matrix
5. Full Statistical Telemetry & JSON Export
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dataset.generate_dataset import validate_and_export_dataset
from src.cache import EmbeddingModelWrapper, VectorSemanticCache
from src.evaluator import CacheEvaluationEngine
from src.filter import Stage2MicroFilter
from src.plotter import (
    plot_cache_roc_curve,
    plot_cosine_overlap_histogram,
    plot_financial_breakeven_matrix,
)
from src.two_stage_cache import TwoStageSemanticCache


def evaluate_two_stage_cache(
    dataset: list,
    model_name: str,
    thresholds: list,
    encoder: EmbeddingModelWrapper,
) -> list:
    """Evaluate 2-Stage Verified Cache across threshold sweeps."""
    sweeps = []
    total_safe = sum(1 for d in dataset if d["is_cache_safe"])
    total_unsafe = sum(1 for d in dataset if not d["is_cache_safe"])

    micro_filter = Stage2MicroFilter()

    # Pre-encode all queries
    queries_a = [d["query_a"] for d in dataset]
    queries_b = [d["query_b"] for d in dataset]
    vecs_a = encoder.encode(queries_a, normalize=True)
    vecs_b = encoder.encode(queries_b, normalize=True)
    sims = np.sum(vecs_a * vecs_b, axis=1)

    for tau in thresholds:
        true_hits = 0
        false_hits = 0
        neg_fp = 0
        ent_fp = 0
        num_fp = 0
        crit_fp = 0

        for i, item in enumerate(dataset):
            sim = float(sims[i])
            is_safe = item["is_cache_safe"]

            # Stage 1: Vector check
            if sim >= tau:
                # Stage 2: Micro-filter verification
                filt_res = micro_filter.verify(item["query_a"], item["query_b"])
                if filt_res.is_verified:
                    if is_safe:
                        true_hits += 1
                    else:
                        false_hits += 1
                        if item["category"] == "negation":
                            neg_fp += 1
                        elif item["category"] == "entity_swap":
                            ent_fp += 1
                        elif item["category"] == "numerical_temporal_swap":
                            num_fp += 1
                        if item["hazard_severity"] == "critical":
                            crit_fp += 1

        tpr = true_hits / total_safe if total_safe > 0 else 0.0
        fpr = false_hits / total_unsafe if total_unsafe > 0 else 0.0
        precision = true_hits / (true_hits + false_hits) if (true_hits + false_hits) > 0 else 1.0

        sweeps.append({
            "threshold": tau,
            "true_hits": true_hits,
            "false_hits": false_hits,
            "true_hit_rate": round(tpr, 4),
            "false_hit_rate": round(fpr, 4),
            "precision": round(precision, 4),
            "negation_false_hits": neg_fp,
            "entity_false_hits": ent_fp,
            "numerical_false_hits": num_fp,
            "critical_false_hits": crit_fp,
        })

    return sweeps


def main():
    import numpy as np

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]Semantic Cache Blindspots — Full Benchmark & Publication Pipeline[/bold cyan]\n"
        "[dim]Reproducible Evaluation of Anisotropy Blindspots & 2-Stage Verification Defense[/dim]",
        border_style="cyan"
    ))

    dataset_dir = ROOT_DIR / "dataset"
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Generation
    console.print("[bold yellow]1. Validating 300-Query Benchmark Dataset...[/bold yellow]")
    dataset = validate_and_export_dataset(dataset_dir)

    evaluator = CacheEvaluationEngine(benchmark_data=dataset)
    models = ["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"]

    benchmark_summary = {}

    for model_name in models:
        console.print(f"\n[bold magenta]2. Running Evaluations on: [underline]{model_name}[/underline][/bold magenta]")
        encoder = EmbeddingModelWrapper(model_name=model_name)

        # Single-Stage Evaluation
        t0 = time.time()
        single_stage_results = evaluator.evaluate_model(model_name=model_name, encoder=encoder)
        eval_time = time.time() - t0

        # Two-Stage Evaluation
        sweep_thresholds = [s["threshold"] for s in single_stage_results["threshold_sweeps"]]
        two_stage_sweeps = evaluate_two_stage_cache(dataset, model_name, sweep_thresholds, encoder)

        benchmark_summary[model_name] = {
            "single_stage": single_stage_results,
            "two_stage_sweeps": two_stage_sweeps,
        }

        # Print Comparison Table at Default Threshold 0.85
        s1_085 = next(s for s in single_stage_results["threshold_sweeps"] if s["threshold"] == 0.85)
        s2_085 = next(s for s in two_stage_sweeps if s["threshold"] == 0.85)

        comp_table = Table(title=f"1-Stage vs 2-Stage Hybrid Filter Comparison @ τ = 0.85 ({model_name})", header_style="bold green")
        comp_table.add_column("Architecture", style="bold")
        comp_table.add_column("True Hit Rate (TPR)", justify="right", style="green")
        comp_table.add_column("Fatal False Hit Rate (FPR)", justify="right", style="red")
        comp_table.add_column("Fatal False Hits", justify="right", style="bold red")
        comp_table.add_column("Critical Fatal Collisions", justify="right", style="bold red")
        comp_table.add_column("Cache Precision", justify="right")

        comp_table.add_row(
            "1-Stage Vector Cache (Naive)",
            f"{s1_085['true_hit_rate']*100:.1f}%",
            f"{s1_085['false_hit_rate']*100:.1f}%",
            f"{s1_085['false_hits']} / 200",
            str(s1_085["critical_false_hits"]),
            f"{s1_085['precision']*100:.1f}%",
        )
        comp_table.add_row(
            "2-Stage Verified Hybrid Filter",
            f"{s2_085['true_hit_rate']*100:.1f}%",
            f"[bold green]{s2_085['false_hit_rate']*100:.1f}%[/bold green]",
            f"[bold green]{s2_085['false_hits']} / 200[/bold green]",
            f"[bold green]{s2_085['critical_false_hits']}[/bold green]",
            f"[bold green]{s2_085['precision']*100:.1f}%[/bold green]",
        )
        console.print(comp_table)

    # 3. Generate Visualizations for Primary Model (BGE-Small)
    console.print("\n[bold yellow]3. Generating Publication-Grade Visualizations...[/bold yellow]")
    bge_results = benchmark_summary["BAAI/bge-small-en-v1.5"]["single_stage"]
    bge_two_stage = benchmark_summary["BAAI/bge-small-en-v1.5"]["two_stage_sweeps"]

    p1_path = results_dir / "plot1_cosine_overlap_histogram.png"
    p2_path = results_dir / "plot2_semantic_cache_roc_curve.png"
    p3_path = results_dir / "plot3_financial_breakeven_matrix.png"

    plot_cosine_overlap_histogram(bge_results, p1_path, model_name="BAAI/bge-small-en-v1.5")
    plot_cache_roc_curve(bge_results["threshold_sweeps"], bge_two_stage, p2_path, model_name="BAAI/bge-small-en-v1.5")
    plot_financial_breakeven_matrix(p3_path)

    # 4. Save Final Summary JSON
    json_path = results_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2, ensure_ascii=False)

    console.print(f"\n[bold green]✔ Full Benchmark Pipeline Completed Successfully![/bold green]")
    console.print(f"  • JSON Results: {json_path}")
    console.print(f"  • Plot 1: {p1_path}")
    console.print(f"  • Plot 2: {p2_path}")
    console.print(f"  • Plot 3: {p3_path}")


if __name__ == "__main__":
    main()
