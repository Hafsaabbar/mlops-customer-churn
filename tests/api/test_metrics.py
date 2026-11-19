"""
Prometheus metrics exposure and request counter delta tests.
"""

from typing import Dict, Any
from fastapi.testclient import TestClient
from src.api.main import PREDICTIONS_TOTAL


def test_metrics_endpoint_exposes_standard_prometheus_collectors(client: TestClient):
    """Verify GET /metrics returns 200 and includes all core application metric families."""
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
