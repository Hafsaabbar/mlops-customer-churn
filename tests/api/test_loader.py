"""
Unit tests for ModelService loading lifecycle and MLflow registry fallback.
"""

from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import pytest

from src.api.model_loader import ModelService


def test_loader_default_use_registry_unset(metadata_dict: Dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    """Verify default behavior without USE_REGISTRY loads joblib and leaves registry unchecked."""
    monkeypatch.delenv("USE_REGISTRY", raising=False)
    service = ModelService()
    assert service.load() is True
    assert service.ready is True
    assert service.source == "joblib"
    assert service.registry_checked is False
    assert service.registry_consistent is None
    assert service.run_id == metadata_dict["run_id"]


def test_loader_use_registry_fallback_empty_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify fallback to joblib with registry_consistent=False when tracking URI points to empty database."""
    monkeypatch.setenv("USE_REGISTRY", "true")
    dummy_db = tmp_path / "empty_mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{dummy_db}")

    service = ModelService(mlflow_tracking_uri=f"sqlite:///{dummy_db}")
    assert service.load() is True
    assert service.ready is True
    assert service.source == "joblib"
    assert service.registry_checked is True
    assert service.registry_consistent is False


def test_loader_use_registry_success_matching_run_id(metadata_dict: Dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    """Verify registry model is loaded when champion alias run_id matches metadata run_id."""
    monkeypatch.setenv("USE_REGISTRY", "true")
    target_run_id = metadata_dict["run_id"]

    mock_version = MagicMock()
    mock_version.run_id = target_run_id
    mock_version.version = "2"

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_version

    mock_reg_model = MagicMock()

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client), \
         patch("mlflow.sklearn.load_model", return_value=mock_reg_model), \
         patch("mlflow.set_tracking_uri"):
        service = ModelService()
        assert service.load() is True
        assert service.ready is True
        assert service.source == "registry"
        assert service.registry_checked is True
        assert service.registry_consistent is True
        assert service.model == mock_reg_model


def test_loader_use_registry_fallback_mismatched_run_id(metadata_dict: Dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    """Verify fallback to joblib when champion alias has a different run_id than metadata."""
    monkeypatch.setenv("USE_REGISTRY", "true")

    mock_version = MagicMock()
    mock_version.run_id = "completely-different-run-id-9999"
    mock_version.version = "9"

    mock_client = MagicMock()
    mock_client.get_model_version_by_alias.return_value = mock_version

    with patch("mlflow.tracking.MlflowClient", return_value=mock_client), \
         patch("mlflow.set_tracking_uri"):
        service = ModelService()
        assert service.load() is True
        assert service.ready is True
        assert service.source == "joblib"
        assert service.registry_checked is True
        assert service.registry_consistent is False
