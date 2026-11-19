"""
Integration and validation tests for Customer Churn inference API.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
import sklearn
import pytest
from fastapi.testclient import TestClient

from src.api import main


# -----------------------------------------------------------------------------
# A. Nominal Tests
# -----------------------------------------------------------------------------

def test_health_returns_ok(client: TestClient):
    """Verify GET /health returns 200 with status ok when service is ready."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_model_info_returns_expected_metadata(
    client: TestClient, metadata_dict: Dict[str, Any], optimal_threshold: float
):
    """Verify GET /model-info returns 200 and matches metadata file values."""
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()

    assert data["model_name"] == metadata_dict["model_name"]
    assert data["run_id"] == metadata_dict["run_id"]
    assert data["model_version"] == str(metadata_dict["registry_version"])
    assert data["threshold"] == optimal_threshold
    assert data["registry_checked"] is False
    assert data["registry_consistent"] is None
    assert data["sklearn_version"] == sklearn.__version__
    assert data["schema_version"] == "1.0.0"


def test_predict_valid_payload_returns_correct_keys_and_bounds(
    client: TestClient, valid_payload: Dict[str, Any], optimal_threshold: float
):
    """Verify POST /predict returns 200 with required response structure and valid value ranges."""
    res = client.post("/predict", json=valid_payload)
    assert res.status_code == 200
    data = res.json()

    expected_keys = {
        "churn_probability",
        "prediction",
        "threshold",
        "model_version",
        "model_source",
    }
    assert set(data.keys()) == expected_keys
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["prediction"] in {"Yes", "No"}
    assert data["threshold"] == optimal_threshold
    assert data["model_source"] in {"joblib", "registry"}


def test_predict_model_consistency_stream_rows(client: TestClient, optimal_threshold: float):
    """Verify API predictions match direct standalone pipeline predictions on 5 real stream rows."""
    stream_path = Path("data/splits/stream/X_stream.csv")
    df_stream = pd.read_csv(stream_path).head(5)
    pipeline = joblib.load("artifacts/champion_model.joblib")

    for _, row in df_stream.iterrows():
        row_dict = row.to_dict()
        row_dict["SeniorCitizen"] = int(row_dict["SeniorCitizen"])
        row_dict["tenure"] = int(row_dict["tenure"])
        row_dict["MonthlyCharges"] = float(row_dict["MonthlyCharges"])
        row_dict["TotalCharges"] = float(row_dict["TotalCharges"])

        res = client.post("/predict", json=row_dict)
        assert res.status_code == 200
        api_res = res.json()

        row_df = pd.DataFrame([{k: v for k, v in row_dict.items() if k != "customerID"}])
        raw_prob = float(pipeline.predict_proba(row_df)[0, 1])
        expected_prob = round(raw_prob, 4)
        expected_decision = "Yes" if raw_prob >= optimal_threshold else "No"

        assert api_res["churn_probability"] == expected_prob
        assert api_res["prediction"] == expected_decision


def test_predict_optional_customer_id(client: TestClient, valid_payload: Dict[str, Any]):
    """Verify inference succeeds with 200 when optional customerID field is omitted."""
    payload = dict(valid_payload)
    del payload["customerID"]
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "prediction" in res.json()


# -----------------------------------------------------------------------------
# B. Threshold Boundary and Precision Tests
# -----------------------------------------------------------------------------

class _StubProbabilityModel:
    def __init__(self, prob: float):
        self.prob = prob

    def predict_proba(self, X):
        return np.array([[1.0 - self.prob, self.prob]])


def test_decision_threshold_exact_boundary(
    client: TestClient,
    valid_payload: Dict[str, Any],
    optimal_threshold: float,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify decision is 'Yes' when raw probability is exactly equal to optimal threshold."""
    stub = _StubProbabilityModel(optimal_threshold)
    monkeypatch.setattr(main.model_service, "model", stub)

    res = client.post("/predict", json=valid_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "Yes"
    assert data["threshold"] == optimal_threshold


def test_decision_threshold_raw_probability_precision(
    client: TestClient,
    valid_payload: Dict[str, Any],
    optimal_threshold: float,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify decision is 'No' when raw probability is below threshold, even if rounded matches."""
    p_below = optimal_threshold - 0.00001
    stub = _StubProbabilityModel(p_below)
    monkeypatch.setattr(main.model_service, "model", stub)

    res = client.post("/predict", json=valid_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "No"
    assert data["churn_probability"] == round(p_below, 4)


# -----------------------------------------------------------------------------
# C. Invalid Contract Tests (422 Unprocessable Entity)
# -----------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@pytest.mark.parametrize("missing_field", REQUIRED_FIELDS)
def test_validation_missing_each_required_field(
    client: TestClient,
    valid_payload: Dict[str, Any],
    missing_field: str,
    isolate_monitoring_log: Path,
):
    """Verify 422 error and no prediction output when any single required field is missing."""
    payload = dict(valid_payload)
    del payload[missing_field]

    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "prediction" not in res.json()
    assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""


def test_validation_empty_body(client: TestClient, isolate_monitoring_log: Path):
    """Verify 422 error when POST /predict receives an empty JSON dictionary."""
    res = client.post("/predict", json={})
    assert res.status_code == 422
    assert "prediction" not in res.json()
    assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""


def test_validation_non_json_body(client: TestClient, isolate_monitoring_log: Path):
    """Verify 422 error when POST /predict receives a non-JSON payload."""
    res = client.post(
        "/predict",
        content="This is plain text and not valid JSON.",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 422
    assert "prediction" not in res.json()
    assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("tenure", "abc"),
        ("MonthlyCharges", "cher"),
        ("SeniorCitizen", "oui"),
    ],
)
def test_validation_invalid_types(
    client: TestClient,
    valid_payload: Dict[str, Any],
    key: str,
    bad_value: Any,
    isolate_monitoring_log: Path,
):
    """Verify 422 error when feature types violate schema expectations."""
    payload = dict(valid_payload, **{key: bad_value})
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "prediction" not in res.json()
    assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""


@pytest.mark.parametrize(
    "key,unallowed_value",
    [
        ("Contract", "Lifetime"),
        ("gender", "Other"),
        ("SeniorCitizen", 2),
        ("InternetService", "5G"),
    ],
)
def test_validation_unallowed_values(
    client: TestClient,
    valid_payload: Dict[str, Any],
    key: str,
    unallowed_value: Any,
    isolate_monitoring_log: Path,
):
    """Verify 422 error when categorical values fall outside the allowed set."""
    payload = dict(valid_payload, **{key: unallowed_value})
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "prediction" not in res.json()
    assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""


@pytest.mark.parametrize(
    "key,out_of_bound_value",
    [
        ("tenure", -1),
        ("tenure", 73),
        ("MonthlyCharges", 9.99),
        ("MonthlyCharges", 150.01),
        ("TotalCharges", -0.01),
        ("TotalCharges", 10000.01),
    ],
)
def test_validation_out_of_bounds(
    client: TestClient,
    valid_payload: Dict[str, Any],
    key: str,
    out_of_bound_value: Any,
    isolate_monitoring_log: Path,
):
    """Verify 422 error when numerical values exceed contracted bounds."""
    payload = dict(valid_payload, **{key: out_of_bound_value})
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "prediction" not in res.json()
    assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""


@pytest.mark.parametrize(
    "extra_dict",
    [
        {"Churn": "Yes"},
        {"foo": 1},
    ],
)
def test_validation_forbidden_extra_fields(
    client: TestClient,
    valid_payload: Dict[str, Any],
    extra_dict: Dict[str, Any],
    isolate_monitoring_log: Path,
):
    """Verify 422 error when unexpected extra attributes or target leakages are passed."""
    payload = dict(valid_payload, **extra_dict)
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "prediction" not in res.json()
    assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""


@pytest.mark.parametrize(
    "overrides",
    [
        {"PhoneService": "No", "MultipleLines": "Yes"},
        {"InternetService": "No", "OnlineSecurity": "Yes"},
        {"tenure": 0, "TotalCharges": 500.0},
    ],
)
def test_validation_cross_business_rules(
    client: TestClient,
    valid_payload: Dict[str, Any],
    overrides: Dict[str, Any],
    isolate_monitoring_log: Path,
):
    """Verify 422 error when inputs contradict cross-column business coherence rules."""
    payload = dict(valid_payload, **overrides)
    res = client.post("/predict", json=payload)
    assert res.status_code == 422
    assert "prediction" not in res.json()
    assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""


# -----------------------------------------------------------------------------
# D. Model Unavailable Tests (503 Service Unavailable)
# -----------------------------------------------------------------------------

def test_unavailable_model_missing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: Dict[str, Any],
    isolate_monitoring_log: Path,
):
    """Verify 503 response and zero readiness gauge when metadata JSON is missing."""
    absent_meta = tmp_path / "absent_metadata.json"
    monkeypatch.setattr(main.model_service, "metadata_path", str(absent_meta))

    with TestClient(main.app) as unavail_client:
        r_h = unavail_client.get("/health")
        assert r_h.status_code == 503
        assert r_h.json() == {"status": "unavailable"}

        r_m = unavail_client.get("/model-info")
        assert r_m.status_code == 503
        assert "unavailable" in r_m.json().get("detail", "").lower()

        r_p = unavail_client.post("/predict", json=valid_payload)
        assert r_p.status_code == 503
        assert "prediction" not in r_p.json()
        assert "unavailable" in r_p.json().get("detail", "").lower()

        assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""

        r_met = unavail_client.get("/metrics")
        assert "model_ready 0.0" in r_met.text


def test_unavailable_model_corrupted_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: Dict[str, Any],
    isolate_monitoring_log: Path,
):
    """Verify 503 response and zero readiness gauge when metadata JSON is corrupted."""
    corrupted_meta = tmp_path / "corrupted_metadata.json"
    corrupted_meta.write_text("{this is corrupt json: True", encoding="utf-8")
    monkeypatch.setattr(main.model_service, "metadata_path", str(corrupted_meta))

    with TestClient(main.app) as unavail_client:
        r_h = unavail_client.get("/health")
        assert r_h.status_code == 503
        assert r_h.json() == {"status": "unavailable"}

        r_m = unavail_client.get("/model-info")
        assert r_m.status_code == 503
        assert "unavailable" in r_m.json().get("detail", "").lower()

        r_p = unavail_client.post("/predict", json=valid_payload)
        assert r_p.status_code == 503
        assert "prediction" not in r_p.json()
        assert "unavailable" in r_p.json().get("detail", "").lower()

        assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""

        r_met = unavail_client.get("/metrics")
        assert "model_ready 0.0" in r_met.text


def test_unavailable_model_missing_joblib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: Dict[str, Any],
    isolate_monitoring_log: Path,
):
    """Verify 503 response and zero readiness gauge when joblib model file is missing."""
    absent_joblib = tmp_path / "absent_champion.joblib"
    monkeypatch.setattr(main.model_service, "model_path", str(absent_joblib))

    with TestClient(main.app) as unavail_client:
        r_h = unavail_client.get("/health")
        assert r_h.status_code == 503
        assert r_h.json() == {"status": "unavailable"}

        r_m = unavail_client.get("/model-info")
        assert r_m.status_code == 503
        assert "unavailable" in r_m.json().get("detail", "").lower()

        r_p = unavail_client.post("/predict", json=valid_payload)
        assert r_p.status_code == 503
        assert "prediction" not in r_p.json()
        assert "unavailable" in r_p.json().get("detail", "").lower()

        assert not isolate_monitoring_log.exists() or isolate_monitoring_log.read_text(encoding="utf-8").strip() == ""

        r_met = unavail_client.get("/metrics")
        assert "model_ready 0.0" in r_met.text


# -----------------------------------------------------------------------------
# E. Robustness & Logging Tests
# -----------------------------------------------------------------------------

def test_inference_exception_returns_generic_500(
    client: TestClient, valid_payload: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    """Verify inference exceptions return 500 with generic message and no internal leakage."""
    def mock_raise_error(X):
        raise RuntimeError("secret interne")

    monkeypatch.setattr(main.model_service.model, "predict_proba", mock_raise_error)

    res = client.post("/predict", json=valid_payload)
    assert res.status_code == 500
    assert res.json() == {"detail": "Inference failed."}
    assert "secret interne" not in res.text


def test_monitoring_log_structure_and_pseudonymization(
    client: TestClient, valid_payload: Dict[str, Any], isolate_monitoring_log: Path
):
    """Verify predictions log contains 1 JSON line with valid keys and pseudonymized customerID."""
    res = client.post("/predict", json=valid_payload)
    assert res.status_code == 200

    lines = isolate_monitoring_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    expected_entry_keys = {
        "timestamp",
        "customer_id",
        "model_version",
        "probability",
        "prediction",
        "features",
    }
    assert set(entry.keys()) == expected_entry_keys
    assert entry["customer_id"] != "7590-VHVEG"
    assert len(entry["customer_id"]) == 16
    assert all(c in "0123456789abcdef" for c in entry["customer_id"].lower())
    assert len(entry["features"]) == 19
    assert "7590-VHVEG" not in isolate_monitoring_log.read_text(encoding="utf-8")


def test_predict_succeeds_when_log_write_fails(
    client: TestClient, valid_payload: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Verify /predict returns 200 even if a filesystem error prevents writing the log."""
    blocking_file = tmp_path / "blocking_regular_file"
    blocking_file.write_text("file content", encoding="utf-8")
    blocked_log_path = blocking_file / "sub_dir" / "log.jsonl"
    monkeypatch.setenv("PREDICTIONS_LOG_PATH", str(blocked_log_path))

    res = client.post("/predict", json=valid_payload)
    assert res.status_code == 200
    assert "prediction" in res.json()
