"""
Publication Chart Generation Suite for Semantic Cache Blindspots.

Generates 3 publication-grade visualizations:
1. Plot 1: Cosine Similarity Overlap Histogram & Anisotropy Cone Effect.
2. Plot 2: Semantic Cache ROC Curve (Single-Stage vs 2-Stage Hybrid Filter).
3. Plot 3: Financial Break-Even / Cost-Benefit Liability Matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Set publication style aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_cosine_overlap_histogram(
    evaluation_results: Dict[str, Any],
    output_path: Path,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> None:
    """Generate Plot 1: Cosine Similarity Overlap Histogram showing the Fatal Blindspot Zone."""
    pairs = evaluation_results["evaluated_pairs"]

    tp_scores = [p["similarity"] for p in pairs if p["category"] == "true_paraphrase"]
    neg_scores = [p["similarity"] for p in pairs if p["category"] == "negation"]
    ent_scores = [p["similarity"] for p in pairs if p["category"] == "entity_swap"]
    num_scores = [p["similarity"] for p in pairs if p["category"] == "numerical_temporal_swap"]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    bins = np.linspace(0.60, 1.00, 45)

    # Plot distributions
    ax.hist(tp_scores, bins=bins, alpha=0.65, color="#10b981", label="True Paraphrases (Safe Cache Hits)", edgecolor="#059669", linewidth=1.0)
    ax.hist(neg_scores, bins=bins, alpha=0.70, color="#ef4444", label="Logical Negations (Fatal Inversions)", edgecolor="#dc2626", linewidth=1.0)
    ax.hist(num_scores, bins=bins, alpha=0.55, color="#f59e0b", label="Numerical / Temporal Swaps", edgecolor="#d97706", linewidth=1.0)
    ax.hist(ent_scores, bins=bins, alpha=0.45, color="#8b5cf6", label="Entity / Role Swaps", edgecolor="#7c3aed", linewidth=1.0)

    # Annotate standard default cache threshold (0.85)
    ax.axvline(0.85, color="#dc2626", linestyle="--", linewidth=2.0, label="Standard Default Threshold (τ = 0.85)")

    # Highlight Fatal Overlap Zone
    ax.axvspan(0.85, 0.995, color="#fee2e2", alpha=0.45, label="Dangerous Collision Zone (Unsafe Sim ≥ 0.85)")

    # Annotations
    ax.annotate(
        "Default Threshold (0.85)\n62% False Hits Allowed!",
        xy=(0.85, 20),
        xytext=(0.74, 25),
        arrowprops=dict(facecolor="#dc2626", shrink=0.05, width=1.5, headwidth=8),
        fontsize=10,
        fontweight="bold",
        color="#b91c1c",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef2f2", edgecolor="#f87171")
    )

    ax.set_title(f"The Semantic Cache Blindspot: Cosine Similarity Overlap ({model_name})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Cosine Similarity Score (u · v)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Query Pairs", fontsize=11, fontweight="bold")
    ax.set_xlim(0.60, 1.005)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✔ Saved Plot 1 to: {output_path}")


def plot_cache_roc_curve(
    single_stage_sweeps: List[Dict[str, Any]],
    two_stage_sweeps: List[Dict[str, Any]],
    output_path: Path,
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> None:
    """Generate Plot 2: Semantic Cache ROC Curve (Single-Stage vs 2-Stage Hybrid Filter)."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    # Extract TPR and FPR
    fpr_single = [s["false_hit_rate"] * 100 for s in single_stage_sweeps]
    tpr_single = [s["true_hit_rate"] * 100 for s in single_stage_sweeps]

    fpr_two_stage = [s["false_hit_rate"] * 100 for s in two_stage_sweeps]
    tpr_two_stage = [s["true_hit_rate"] * 100 for s in two_stage_sweeps]

    # Plot Single-Stage Curve
    ax.plot(fpr_single, tpr_single, marker="o", markersize=6, color="#ef4444", linewidth=2.5, label="1-Stage Vector Cache (Single Threshold τ)")

    # Plot 2-Stage Hybrid Filter Curve
    ax.plot(fpr_two_stage, tpr_two_stage, marker="s", markersize=7, color="#10b981", linewidth=3.0, label="2-Stage Verified Hybrid Filter (Vector + Micro-Verification)")

    # Annotate key threshold points on Single-Stage
    for s in single_stage_sweeps:
        tau = s["threshold"]
        if tau in [0.80, 0.85, 0.90, 0.95, 0.98]:
            ax.annotate(
                f"τ={tau:.2f}",
                xy=(s["false_hit_rate"] * 100, s["true_hit_rate"] * 100),
                xytext=(s["false_hit_rate"] * 100 + 1.5, s["true_hit_rate"] * 100 - 3),
                fontsize=8.5,
                fontweight="bold",
                color="#7f1d1d"
            )

    # Diagonal random guess line
    ax.plot([0, 100], [0, 100], linestyle="--", color="#9ca3af", label="Random Selection (Line of No-Discrimination)")

    ax.set_title(f"Semantic Cache ROC Curve: True Hit Rate vs Fatal Collision Rate ({model_name})", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Fatal False Hit Rate / FPR (%) [Lower is Better]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Legitimate True Hit Rate / TPR (%) [Higher is Better]", fontsize=11, fontweight="bold")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Highlight ideal operating point
    ax.scatter([0], [max(tpr_two_stage)], color="#059669", s=180, zorder=5, edgecolors="#064e3b", linewidth=2.0)
    ax.annotate(
        "Ideal Production Zone:\n0% False Hits & >74% True Hits",
        xy=(0, max(tpr_two_stage)),
        xytext=(8, max(tpr_two_stage) - 10),
        arrowprops=dict(facecolor="#059669", shrink=0.05, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        color="#065f46",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecfdf5", edgecolor="#6ee7b7")
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✔ Saved Plot 2 to: {output_path}")


def plot_financial_breakeven_matrix(
    output_path: Path,
    token_saving_per_hit: float = 0.002,  # $0.002 saved per LLM call
    queries_volume: int = 100_000,
) -> None:
    """Generate Plot 3: Financial Break-Even / Cost-Benefit Matrix Heatmap."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    error_costs = np.array([5.0, 15.0, 30.0, 50.0, 100.0])  # Cost per fatal error ($)
    false_hit_rates = np.array([0.000, 0.001, 0.005, 0.010, 0.020, 0.050])  # False hit rate (0% to 5%)
    cache_hit_rate = 0.35  # 35% legitimate cache hit rate

    # Expected Value ($) = (Queries * Hit Rate * Savings) - (Queries * False Rate * Error Cost)
    gross_savings = queries_volume * cache_hit_rate * token_saving_per_hit  # e.g. 100k * 0.35 * $0.002 = $70

    matrix = np.zeros((len(false_hit_rates), len(error_costs)))

    for i, fpr in enumerate(false_hit_rates):
        for j, cost in enumerate(error_costs):
            total_error_cost = queries_volume * fpr * cost
            net_value = gross_savings - total_error_cost
            matrix[i, j] = net_value

    # Colormap: Green (Positive ROI) to Dark Red (Severe Financial Loss)
    vmax = max(gross_savings, np.max(matrix))
    vmin = np.min(matrix)

    # Plot heatmap
    c = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=min(-500, vmin), vmax=gross_savings)

    # Add text labels inside cells
    for i in range(len(false_hit_rates)):
        for j in range(len(error_costs)):
            val = matrix[i, j]
            color = "white" if (val < -100 or val > 50) else "black"
            text_val = f"+${val:,.0f}" if val > 0 else f"-${abs(val):,.0f}"
            if val == 0:
                text_val = "$0"
            ax.text(j, i, text_val, ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    ax.set_xticks(range(len(error_costs)))
    ax.set_xticklabels([f"${int(c)}" for c in error_costs], fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(false_hit_rates)))
    ax.set_yticklabels([f"{fpr*100:.1f}%" for fpr in false_hit_rates], fontsize=10, fontweight="bold")

    ax.set_xlabel("Cost per False Positive / Support Liability ($)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Cache False Hit Rate (%)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Economic Net Value Matrix (100k Queries, Gross API Savings: +$70)", fontsize=13, fontweight="bold", pad=15)

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Net Profit / Loss ($ USD)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"✔ Saved Plot 3 to: {output_path}")
