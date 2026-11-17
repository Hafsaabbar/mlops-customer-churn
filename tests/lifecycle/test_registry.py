"""
Tests for MLflow Model Registry alias lifecycle: candidate promotion, previous tracking, and rollback.
"""

from pathlib import Path
import pytest
from src.lifecycle.registry import (
    get_client, get_alias_version, promote_candidate,
    rollback_champion, get_registry_status, REGISTRY_MODEL_NAME
)

def test_registry_status_inspection():
    """Verify registry status returns structured state of aliases and versions."""
    status = get_registry_status(REGISTRY_MODEL_NAME)
    assert status["model_name"] == REGISTRY_MODEL_NAME
    assert status["exists"] is True
    assert "champion" in status
    assert "previous" in status

def test_rejection_does_not_modify_champion():
    """Verify that a rejected candidate does NOT alter the existing champion alias."""
    client = get_client()
    champ_before = get_alias_version(client, REGISTRY_MODEL_NAME, "champion")

    # Force reject a simulated candidate
    res = promote_candidate(
        candidate_version="1",
        model_name=REGISTRY_MODEL_NAME,
        run_gate=True,
        force_reject=True
    )
    assert res["promoted"] is False
    assert res["gate_decision"] == "REJECTED"

    champ_after = get_alias_version(client, REGISTRY_MODEL_NAME, "champion")
    assert champ_before == champ_after, "Champion alias must NOT change when gate rejects candidate."

def test_rollback_transitions():
    """Verify rollback swaps previous back to champion."""
    client = get_client()
    champ_before = get_alias_version(client, REGISTRY_MODEL_NAME, "champion")
    prev_before = get_alias_version(client, REGISTRY_MODEL_NAME, "previous")

    if prev_before is not None:
        rollback_res = rollback_champion(REGISTRY_MODEL_NAME)
        assert rollback_res["status"] == "SUCCESS"
        assert rollback_res["alias_after"]["champion"] == prev_before

        # Re-rollback to restore state
        rollback_champion(REGISTRY_MODEL_NAME)

