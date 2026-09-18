"""
Day 1 Evaluation Script: Benchmark Pairwise Cosine Similarity Distributions.

Evaluates:
- all-MiniLM-L6-v2 (384-dim)
- BAAI/bge-small-en-v1.5 (384-dim)

Over the 300-query categorized benchmark dataset.
Outputs rich terminal visuals and saves results to results/day1_similarity_distribution.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is in python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dataset.generate_dataset import validate_and_export_dataset
from src.cache import EmbeddingModelWrapper, VectorSemanticCache
from src.evaluator import CacheEvaluationEngine


def run_evaluation():
    console = Console()
    console.print(Panel.fit(
        "[bold cyan]Semantic Cache Blindspots — Day 1 Evaluation Pipeline[/bold cyan]\n"
        "[dim]Testing Vector Cosine Similarity Distributions & Failure Modes across 300 Query Pairs[/dim]",
        border_style="cyan"
    ))

    # 1. Validate & Export Dataset
    dataset_dir = ROOT_DIR / "dataset"
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold yellow]Step 1: Generating & Validating 300-Query Benchmark Dataset...[/bold yellow]")
    dataset = validate_and_export_dataset(dataset_dir)
    console.print(f"[green]✔ Dataset validated with {len(dataset)} pairs across 4 categories and 7 domains.[/green]\n")

    evaluator = CacheEvaluationEngine(benchmark_data=dataset)

    models_to_test = [
        "all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
    ]

    all_results = {}

    for model_name in models_to_test:
        console.print(f"[bold magenta]Step 2: Evaluating Embedding Model: [underline]{model_name}[/underline]...[/bold magenta]")
        t0 = time.time()
        encoder = EmbeddingModelWrapper(model_name=model_name)
        load_time = time.time() - t0
        console.print(f"[dim]Model loaded in {load_time:.2f}s (Embedding Dim: {encoder.dimension})[/dim]")

        t0 = time.time()
        results = evaluator.evaluate_model(model_name=model_name, encoder=encoder)
        eval_time = time.time() - t0
        console.print(f"[dim]Computed 300 pairwise cosine similarities in {eval_time*1000:.1f}ms[/dim]\n")

        all_results[model_name] = results

        # ----------------------------------------------------
        # TABLE 1: Category Distribution Statistics
        # ----------------------------------------------------
        cat_table = Table(title=f"Cosine Similarity Distribution by Category ({model_name})", header_style="bold cyan")
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Mean ± Std", justify="center")
        cat_table.add_column("Median", justify="right")
        cat_table.add_column("Min", justify="right", style="red")
        cat_table.add_column("Max", justify="right", style="green")
        cat_table.add_column("P10", justify="right")
        cat_table.add_column("P90", justify="right")

        for cat, stats in results["category_statistics"].items():
            cat_name = cat.replace("_", " ").title()
            style = "bold green" if cat == "true_paraphrase" else "bold red"
            cat_table.add_row(
                Text(cat_name, style=style),
                str(stats["count"]),
                f"{stats['mean']:.3f} ± {stats['std']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                f"{stats['p10']:.3f}",
                f"{stats['p90']:.3f}",
            )
        console.print(cat_table)
        console.print()

        # ----------------------------------------------------
        # TABLE 2: Overlap Zone & Hypothesis Validation
        # ----------------------------------------------------
        overlap = results["overlap_zone_analysis"]
        overlap_table = Table(title=f"Dangerous Overlap Zone Analysis ({model_name})", header_style="bold yellow")
        overlap_table.add_column("Metric", style="bold")
        overlap_table.add_column("Value", justify="right")
        overlap_table.add_column("Implication / Hazard Description", style="dim")

        overlap_table.add_row(
            "True Paraphrase Min Similarity",
            f"{overlap['safe_min']:.4f}",
            "Lowest similarity score for a legitimate, safe cache hit"
        )
        overlap_table.add_row(
            "Unsafe Query Max Similarity",
            f"{overlap['unsafe_max']:.4f}",
            "Highest similarity score for a fatal false positive"
        )
        overlap_table.add_row(
            "Empirical Overlap Range",
            f"[{overlap['overlap_range_min']:.3f}, {overlap['overlap_range_max']:.3f}]",
            "CRITICAL: Unsafe queries in this zone score HIGHER than safe queries!"
        )
        overlap_table.add_row(
            "Unsafe Pairs in Overlap Range",
            f"{overlap['overlap_pair_count']} / 200 ({overlap['overlap_pair_count']/200*100:.1f}%)",
            "Percentage of dangerous queries that overlap with legitimate queries"
        )
        overlap_table.add_row(
            "Logical Negations with Cosine >= 0.85",
            f"{overlap['dangerous_negations_above_085']} / 75 ({overlap['dangerous_negations_above_085']/75*100:.1f}%)",
            "Negations that standard cache thresholds (0.85) disastrously hit"
        )
        overlap_table.add_row(
            "Critical Severity Hazards >= 0.85",
            f"{overlap['critical_hazard_above_085']} / 68 ({overlap['critical_hazard_above_085']/68*100:.1f}%)",
            "Lethal medical/financial inverted queries hitting default cache"
        )
        console.print(overlap_table)
        console.print()

        # ----------------------------------------------------
        # TABLE 3: Threshold Sweeps (TPR vs FPR)
        # ----------------------------------------------------
        sweep_table = Table(title=f"Threshold Sweep Performance Matrix ({model_name})", header_style="bold magenta")
        sweep_table.add_column("Threshold (τ)", justify="center", style="bold")
        sweep_table.add_column("True Hit Rate (TPR)", justify="right", style="green")
        sweep_table.add_column("False Hit Rate (FPR)", justify="right", style="red")
        sweep_table.add_column("Negation FPs", justify="right")
        sweep_table.add_column("Entity FPs", justify="right")
        sweep_table.add_column("Numerical FPs", justify="right")
        sweep_table.add_column("Critical Fatal Hits", justify="right", style="bold red")
        sweep_table.add_column("Cache Precision", justify="right")

        # Highlight key thresholds
        key_tau = [0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
        for tm in results["threshold_sweeps"]:
            if tm["threshold"] in key_tau:
                is_default = tm["threshold"] == 0.85
                t_str = f"{tm['threshold']:.2f}" + (" (Default)" if is_default else "")
                style = "bold yellow" if is_default else ""
                sweep_table.add_row(
                    Text(t_str, style=style),
                    f"{tm['true_hit_rate']*100:.1f}%",
                    f"{tm['false_hit_rate']*100:.1f}%",
                    str(tm["negation_false_hits"]),
                    str(tm["entity_false_hits"]),
                    str(tm["numerical_false_hits"]),
                    str(tm["critical_false_hits"]),
                    f"{tm['precision']*100:.1f}%",
                )
        console.print(sweep_table)
        console.print()

        # ----------------------------------------------------
        # TABLE 4: Top 8 Most Fatal False Positive Blindspots
        # ----------------------------------------------------
        hazards_table = Table(title=f"Top Fatal Cache Collisions with High Cosine Similarity ({model_name})", header_style="bold red")
        hazards_table.add_column("ID", style="bold")
        hazards_table.add_column("Cosine", justify="right", style="bold red")
        hazards_table.add_column("Category", style="cyan")
        hazards_table.add_column("Cached Prompt (Query A)", style="dim")
        hazards_table.add_column("Incoming Probe (Query B)", style="bold")
        hazards_table.add_column("Hazard Severity", style="red")

        # Sort unsafe pairs by highest similarity
        unsafe_pairs = [p for p in results["evaluated_pairs"] if not p["is_cache_safe"]]
        unsafe_pairs.sort(key=lambda x: x["similarity"], reverse=True)

        for p in unsafe_pairs[:8]:
            hazards_table.add_row(
                p["id"],
                f"{p['similarity']:.4f}",
                p["category"],
                p["query_a"],
                p["query_b"],
                p["hazard_severity"].upper(),
            )
        console.print(hazards_table)
        console.print("\n" + "="*80 + "\n")

    # 3. Save Summary to JSON
    json_out_path = results_dir / "day1_similarity_distribution.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        # Don't save entire raw query texts twice, save full structure
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]✔ Day 1 Evaluation Complete! Results saved to: {json_out_path}[/bold green]")
    return all_results


if __name__ == "__main__":
    run_evaluation()
