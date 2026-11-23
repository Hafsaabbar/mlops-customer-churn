"""
Prometheus metrics exposure and request counter delta tests.
"""

import json
from pathlib import Path
from typing import Dict, Any
from fastapi.testclient import TestClient
from src.api.main import PREDICTIONS_TOTAL, CHURN_PREDICTION_PROBABILITY


def test_metrics_endpoint_exposes_standard_prometheus_collectors(client: TestClient):
    """Verify GET /metrics returns 200 and includes all core application metric families."""
    client.get("/health")
    res = client.get("/metrics")
    assert res.status_code == 200
    metrics_text = res.text

    assert "http_requests_total" in metrics_text
    assert "http_request_duration_seconds_bucket" in metrics_text
    assert "predictions_total" in metrics_text
    assert "model_ready" in metrics_text
    assert "model_info" in metrics_text


def test_metrics_unmatched_endpoint_labels_for_unknown_routes(client: TestClient):
    """Verify unknown route requests are labeled as unmatched without path leakage in metrics."""
    r1 = client.get("/x-1")
    assert r1.status_code == 404
    r2 = client.get("/x-2")
    assert r2.status_code == 404

    metrics_res = client.get("/metrics")
    assert metrics_res.status_code == 200
    metrics_text = metrics_res.text

    assert 'endpoint="unmatched"' in metrics_text
    assert "/x-1" not in metrics_text
    assert "/x-2" not in metrics_text


def test_metrics_predictions_counter_delta_on_success_and_validation_error(client: TestClient, valid_payload: Dict[str, Any]):
    """Verify predictions_total counter increments by 1 on 200 and delta is 0 on 422."""
    def get_count(label: str) -> float:
        try:
            return PREDICTIONS_TOTAL.labels(predicted_class=label)._value.get()
        except Exception:
            return 0.0

    # 1. Prediction valid
    res_pred = client.post("/predict", json=valid_payload)
    assert res_pred.status_code == 200
    pred_class = res_pred.json()["prediction"]

    before_success = get_count(pred_class)
    res_second = client.post("/predict", json=valid_payload)
    assert res_second.status_code == 200
    after_success = get_count(pred_class)
    assert after_success - before_success == 1.0

    # 2. Validation error (422)
    before_err_yes = get_count("Yes")
    before_err_no = get_count("No")

    res_err = client.post("/predict", json={"invalid_field": 123})
    assert res_err.status_code == 422

    after_err_yes = get_count("Yes")
    after_err_no = get_count("No")
    assert after_err_yes - before_err_yes == 0.0
    assert after_err_no - before_err_no == 0.0


def test_metrics_exposes_probability_histogram_and_threshold_gauge(
    client: TestClient,
    valid_payload: Dict[str, Any],
):
    """Verify /metrics exposes churn_prediction_probability_bucket and model_decision_threshold.

    The gauge value must equal the optimal_threshold from model_metadata.json.
    """
    metadata = json.loads(Path("artifacts/model_metadata.json").read_text(encoding="utf-8"))
    expected_threshold = float(metadata["optimal_threshold"])

    res = client.get("/metrics")
    assert res.status_code == 200
    metrics_text = res.text

    assert "churn_prediction_probability_bucket" in metrics_text
    assert "model_decision_threshold" in metrics_text

    # Parse the gauge value from the exposition text
    threshold_value = None
    for line in metrics_text.splitlines():
        if line.startswith("model_decision_threshold ") and not line.startswith("#"):
            threshold_value = float(line.split()[-1])
            break

    assert threshold_value is not None, "model_decision_threshold not found in /metrics output"
    assert abs(threshold_value - expected_threshold) < 1e-6


def test_metrics_probability_histogram_count_increments_on_success_not_on_422(
    client: TestClient,
    valid_payload: Dict[str, Any],
):
    """Verify churn_prediction_probability_count increments by 1 after /predict 200, not after 422."""

    def get_hist_count() -> float:
        res = client.get("/metrics")
        assert res.status_code == 200
        for line in res.text.splitlines():
            if line.startswith("churn_prediction_probability_count "):
                return float(line.split()[-1])
        return 0.0

    count_before = get_hist_count()

    res_ok = client.post("/predict", json=valid_payload)
    assert res_ok.status_code == 200

    count_after_ok = get_hist_count()
    assert count_after_ok - count_before == 1.0, (
        f"Expected histogram count to increase by 1 after successful /predict, "
        f"but delta was {count_after_ok - count_before}"
    )

    count_before_422 = count_after_ok

    res_422 = client.post("/predict", json={"invalid_field": 123})
    assert res_422.status_code == 422

    count_after_422 = get_hist_count()
    assert count_after_422 - count_before_422 == 0.0, (
        f"Expected histogram count NOT to increase after 422, "
        f"but delta was {count_after_422 - count_before_422}"
    )
