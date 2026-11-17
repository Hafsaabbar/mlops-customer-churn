"""
Tests for the Quality Gate module.
Validates both nominal acceptance scenario and rejection scenario.
"""

from pathlib import Path
import json
import pytest

from src.lifecycle.gate import evaluate_quality_gate

def test_quality_gate_nominal_accepted(tmp_path):
    """Nominal scenario: a model satisfying all requirements is accepted."""
    good_metrics = {
        "f1": 0.58,
        "recall": 0.85,
        "roc_auc": 0.83,
        "precision": 0.45
    }
    result = evaluate_quality_gate(
        model_name="churn-model",
        candidate_version="1",
        candidate_metrics=good_metrics,
        data_validation_valid=True,
        output_dir=tmp_path
    )
    assert result["decision"] == "ACCEPTED"
    assert result["accepted"] is True
    assert len(result["reasons"]) == 0
    assert (tmp_path / "quality_gate_result.json").exists()

def test_quality_gate_failing_rejected(tmp_path):
    """Failure scenario: a candidate failing baseline or recall requirements is rejected."""
    bad_metrics = {
        "f1": 0.25,      # Below minimum 0.50
        "recall": 0.40,  # Below minimum 0.70
        "roc_auc": 0.60, # Below minimum 0.75
        "precision": 0.20
    }
    result = evaluate_quality_gate(
        model_name="churn-model",
        candidate_version="99",
        candidate_metrics=bad_metrics,
        data_validation_valid=True,
        output_dir=tmp_path
    )
    assert result["decision"] == "REJECTED"
    assert result["accepted"] is False
    assert len(result["reasons"]) > 0
    # Must report reasons
    assert any("Recall" in r for r in result["reasons"])
    assert any("F1" in r for r in result["reasons"])

def test_quality_gate_data_validation_failure_rejected(tmp_path):
    """Failure scenario: data validation failure immediately blocks promotion."""
    good_metrics = {
        "f1": 0.60,
        "recall": 0.85,
        "roc_auc": 0.84
    }
    result = evaluate_quality_gate(
        model_name="churn-model",
        candidate_version="2",
        candidate_metrics=good_metrics,
        data_validation_valid=False,  # Corrupted data
        output_dir=tmp_path
    )
    assert result["decision"] == "REJECTED"
    assert result["accepted"] is False
    assert any("Data validation" in r for r in result["reasons"])
