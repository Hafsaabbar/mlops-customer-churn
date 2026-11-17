"""
Tests for fairness analysis and group metrics calculation.
"""

from pathlib import Path
import json
import pytest
from src.lifecycle.fairness import analyze_fairness, compute_group_metrics
import numpy as np

def test_compute_group_metrics():
    """Verify group metrics calculation including FPR and FNR."""
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    # TP=1, FN=1, TN=2, FP=1
    metrics = compute_group_metrics(y_true, y_pred)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["fpr"] == pytest.approx(1 / 3, abs=1e-3)
    assert metrics["fnr"] == 0.5

def test_analyze_fairness_reports_generated():
    """Verify fairness evaluation executes and produces report files."""
    results = analyze_fairness()
    assert "gender_disparity" in results
    assert "senior_citizen_disparity" in results
    assert "Female" in results["gender_disparity"]
    assert "Male" in results["gender_disparity"]
    assert Path("reports/fairness/fairness_metrics.json").exists()
    assert Path("reports/fairness/fairness_analysis_report.md").exists()
    assert Path("reports/fairness/fairness_metrics.png").exists()
