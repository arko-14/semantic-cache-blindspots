"""
Benchmark Dataset Generator & Validator.

Aggregates:
- 100 True Paraphrases (TP-001 .. TP-100)
- 75 Logical Negations (NEG-001 .. NEG-075)
- 75 Entity Substitutions (ENT-001 .. ENT-075)
- 50 Numerical / Temporal Inversions (NUM-001 .. NUM-050)
Total: 300 categorized query pairs.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import sys
import os

# Allow running directly from repo root or dataset/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.raw_true_paraphrases import TRUE_PARAPHRASES
from dataset.raw_negations import LOGICAL_NEGATIONS
from dataset.raw_entity_swaps import ENTITY_SWAPS
from dataset.raw_numerical_swaps import NUMERICAL_SWAPS


def get_all_benchmark_data() -> List[Dict[str, Any]]:
    dataset = []
    dataset.extend(TRUE_PARAPHRASES)
    dataset.extend(LOGICAL_NEGATIONS)
    dataset.extend(ENTITY_SWAPS)
    dataset.extend(NUMERICAL_SWAPS)
    return dataset


def validate_and_export_dataset(output_dir: Path) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_300.json"
    csv_path = output_dir / "benchmark_300.csv"

    dataset = get_all_benchmark_data()
    total_count = len(dataset)

    if total_count != 300:
        raise ValueError(f"Expected exactly 300 items, got {total_count}")

    category_counts = {}
    ids = set()
    required_fields = [
        "id", "category", "subcategory", "domain", 
        "query_a", "query_b", "response_a", "response_b", 
        "is_cache_safe", "hazard_severity"
    ]

    for item in dataset:
        item_id = item["id"]
        if item_id in ids:
            raise ValueError(f"Duplicate ID found: {item_id}")
        ids.add(item_id)

        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field '{field}' in item {item_id}")

        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    expected_counts = {
        "true_paraphrase": 100,
        "negation": 75,
        "entity_swap": 75,
        "numerical_temporal_swap": 50
    }
    for cat, exp_cnt in expected_counts.items():
        actual_cnt = category_counts.get(cat, 0)
        if actual_cnt != exp_cnt:
            raise ValueError(f"Category '{cat}' count mismatch: expected {exp_cnt}, got {actual_cnt}")

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=required_fields)
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)

    print(f"✅ Successfully validated and exported {total_count} benchmark queries:")
    print(f"   - JSON: {json_path}")
    print(f"   - CSV:  {csv_path}")
    print(f"   - Category Breakdown:")
    for cat, cnt in category_counts.items():
        print(f"     • {cat}: {cnt}")

    return dataset


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent
    validate_and_export_dataset(dataset_dir)
