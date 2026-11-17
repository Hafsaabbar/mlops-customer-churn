"""
Tests for offline delayed feedback replay and retrain candidate workflows.
"""

from pathlib import Path
import json
import pytest
from src.lifecycle.feedback import evaluate_stream_feedback, retrain_and_gate_candidate

def test_evaluate_stream_feedback():
    """Verify stream feedback calculates realized metrics on held-out stream data."""
    summary = evaluate_stream_feedback()
    assert "realized_metrics" in summary
    assert "precision" in summary["realized_metrics"]
    assert "recall" in summary["realized_metrics"]
    assert "f1" in summary["realized_metrics"]
    assert summary["n_stream_observations"] == 1057

def test_retrain_rejection_scenario():
    """Verify that when a retrained candidate fails Quality Gate, it is rejected and NOT promoted."""
    res = retrain_and_gate_candidate(force_failing=True)
    assert res["promotion_result"]["promoted"] is False
    assert res["promotion_result"]["gate_decision"] == "REJECTED"
