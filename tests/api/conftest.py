"""
Shared fixtures and test environment isolation for API test suite.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, model_service


@pytest.fixture(scope="session")
def metadata_dict() -> Dict[str, Any]:
    """Read artifacts/model_metadata.json dynamically without hardcoding."""
    path = Path("artifacts/model_metadata.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def optimal_threshold(metadata_dict: Dict[str, Any]) -> float:
    """Extract optimal threshold dynamically from metadata."""
    return float(metadata_dict["optimal_threshold"])


@pytest.fixture(autouse=True)
def isolate_monitoring_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect predictions log to a temporary file for complete isolation."""
    log_file = tmp_path / "test_predictions_log.jsonl"
    monkeypatch.setenv("PREDICTIONS_LOG_PATH", str(log_file))
    return log_file


@pytest.fixture(autouse=True)
def isolate_model_service(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Ensure model_service singleton state is restored after tests that mutate it."""
    orig_meta_path = model_service.metadata_path
    orig_model_path = model_service.model_path
    orig_uri = model_service.mlflow_tracking_uri
    orig_use_reg = os.environ.get("USE_REGISTRY")
    orig_model = model_service.model
    orig_threshold = model_service.threshold

    yield

    model_service.metadata_path = orig_meta_path
    model_service.model_path = orig_model_path
    model_service.mlflow_tracking_uri = orig_uri
    if orig_use_reg is None:
        monkeypatch.delenv("USE_REGISTRY", raising=False)
    else:
        monkeypatch.setenv("USE_REGISTRY", orig_use_reg)
    model_service.model = orig_model
    model_service.threshold = orig_threshold
    model_service.load()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a TestClient instance managing application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> Dict[str, Any]:
    """Provide the canonical valid customer payload for testing."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }
