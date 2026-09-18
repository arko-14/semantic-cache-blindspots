"""Tests for benchmark dataset schema, integrity, and balance."""

import json
from pathlib import Path
import pytest

from dataset.generate_dataset import get_all_benchmark_data, validate_and_export_dataset


def test_dataset_size_and_balance():
    data = get_all_benchmark_data()
    assert len(data) == 300, f"Expected 300 items, got {len(data)}"

    counts = {}
    for item in data:
        cat = item["category"]
        counts[cat] = counts.get(cat, 0) + 1

    assert counts["true_paraphrase"] == 100
    assert counts["negation"] == 75
    assert counts["entity_swap"] == 75
    assert counts["numerical_temporal_swap"] == 50


def test_dataset_id_uniqueness():
    data = get_all_benchmark_data()
    ids = [x["id"] for x in data]
    assert len(ids) == len(set(ids)), "Duplicate IDs detected in dataset!"


def test_dataset_field_integrity():
    data = get_all_benchmark_data()
    required = [
        "id", "category", "subcategory", "domain", 
        "query_a", "query_b", "response_a", "response_b", 
        "is_cache_safe", "hazard_severity"
    ]

    valid_severities = {"none", "low", "medium", "high", "critical"}

    for item in data:
        for r in required:
            assert r in item, f"Missing key {r} in item {item.get('id')}"
            assert item[r] is not None, f"Key {r} is None in item {item.get('id')}"

        assert item["hazard_severity"] in valid_severities

        # is_cache_safe must only be True for true_paraphrases
        if item["category"] == "true_paraphrase":
            assert item["is_cache_safe"] is True
            assert item["hazard_severity"] == "none"
        else:
            assert item["is_cache_safe"] is False
            assert item["hazard_severity"] != "none"


def test_dataset_export(tmp_path):
    exported = validate_and_export_dataset(tmp_path)
    assert len(exported) == 300
    assert (tmp_path / "benchmark_300.json").exists()
    assert (tmp_path / "benchmark_300.csv").exists()
