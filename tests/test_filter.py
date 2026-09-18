"""Unit tests for Stage-2 Micro-Verification Filter and TwoStageSemanticCache."""

import pytest
from src.filter import Stage2MicroFilter
from src.two_stage_cache import TwoStageSemanticCache


def test_polarity_and_negation_filter():
    f = Stage2MicroFilter()

    # Direct negation
    ok, reason = f.check_polarity_consistency("Is ibuprofen safe for dogs?", "Why is ibuprofen not safe for dogs?")
    assert ok is False
    assert "negation" in reason.lower()

    # Polar opposite / negation
    ok, reason = f.check_polarity_consistency("How to enable two-factor auth?", "How to disable two-factor auth?")
    assert ok is False
    assert "negation" in reason.lower() or "antonym" in reason.lower()

    # Explicit antonym pair without negation token
    ok, reason = f.check_polarity_consistency("Stocks that benefit from high interest rates", "Stocks that suffer from high interest rates")
    assert ok is False
    assert "antonym" in reason.lower()

    # True paraphrase
    ok, reason = f.check_polarity_consistency("How do I return a damaged item?", "What is the procedure for returning broken merchandise?")
    assert ok is True
    assert reason is None


def test_numerical_filter():
    f = Stage2MicroFilter()

    # Extreme dosage difference
    ok, reason = f.check_numerical_consistency("Single dose of 500mg acetaminophen", "Single dose of 5000mg acetaminophen")
    assert ok is False
    assert "numerical" in reason.lower()

    # Version difference
    ok, reason = f.check_numerical_consistency("New features in Python 3.12", "New features in Python 3.8")
    assert ok is False
    assert "numerical" in reason.lower()

    # Paraphrase with same numbers
    ok, reason = f.check_numerical_consistency("Order 1 pack of batteries", "Purchase 1 single battery pack")
    assert ok is True


def test_direction_role_filter():
    f = Stage2MicroFilter()

    # Role swap
    ok, reason = f.check_direction_and_entities(
        "What are the legal obligations of a landlord to a tenant?",
        "What are the legal obligations of a tenant to a landlord?"
    )
    assert ok is False
    assert "inversion" in reason.lower()

    # Direction swap
    ok, reason = f.check_direction_and_entities(
        "How to migrate a database from MySQL to PostgreSQL?",
        "How to migrate a database from PostgreSQL to MySQL?"
    )
    assert ok is False
    assert "inversion" in reason.lower()


def test_two_stage_semantic_cache_integration():
    cache = TwoStageSemanticCache(model_name="all-MiniLM-L6-v2", threshold=0.70, enable_stage2_filter=True)
    cache.clear()

    # Store a query
    cache.store(
        query="Is recording a phone conversation legal in California without consent?",
        response="No, California is a two-party consent state.",
        entry_id="LEGAL-01"
    )

    # Legitimate exact match
    res_exact = cache.query("Is recording a phone conversation legal in California without consent?")
    assert res_exact.is_hit is True
    assert res_exact.hit_type == "exact"

    # Dangerous inverted probe: "with consent"
    # Stage 1 vector cosine will be very high (>0.90), but Stage 2 must intercept and REJECT it!
    res_probe = cache.query("Is recording a phone conversation legal in California with consent?")
    assert res_probe.is_hit is False
    assert res_probe.hit_type == "rejected_by_filter"
    assert res_probe.stage2_verified is False
    assert res_probe.rejection_reason is not None
