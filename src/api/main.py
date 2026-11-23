"""
FastAPI application for Customer Churn real-time prediction and observability.
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY

from src.api.schemas import (
    ChurnFeatures,
    PredictionResponse,
    ModelInfo,
    HealthResponse,
)
from src.api.model_loader import ModelService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Prometheus Metric Registration Helpers (safe against duplicate registration)
def _get_or_create_counter(name: str, doc: str, labelnames: tuple):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Counter(name, doc, labelnames)

def _get_or_create_histogram(name: str, doc: str, labelnames: tuple):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Histogram(name, doc, labelnames)

def _get_or_create_gauge(name: str, doc: str, labelnames: tuple):
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]
    return Gauge(name, doc, labelnames)

HTTP_REQUESTS_TOTAL = _get_or_create_counter(
    "http_requests_total",
    "Total count of HTTP requests processed by endpoint and status code.",
    ("method", "endpoint", "status_code"),
)

HTTP_REQUEST_DURATION_SECONDS = _get_or_create_histogram(
    "http_request_duration_seconds",
    "Latency of HTTP requests in seconds.",
    ("method", "endpoint"),
)

PREDICTIONS_TOTAL = _get_or_create_counter(
    "predictions_total",
    "Total count of inference predictions partitioned by predicted churn class.",
    ("predicted_class",),
)

MODEL_READY = _get_or_create_gauge(
    "model_ready",
    "Readiness state of the inference model pipeline (1 for ready, 0 for unavailable).",
    (),
)

MODEL_INFO = _get_or_create_gauge(
    "model_info",
    "Model version info and deployment indicator.",
    ("version",),
)

_PROB_BUCKETS = [round(i * 0.1, 1) for i in range(11)]

def _get_or_create_probability_histogram():
    _name = "churn_prediction_probability"
    if _name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[_name]
    return Histogram(
        _name,
        "Distribution of raw churn probability scores emitted by the model.",
        buckets=_PROB_BUCKETS,
    )

CHURN_PREDICTION_PROBABILITY = _get_or_create_probability_histogram()

MODEL_DECISION_THRESHOLD = _get_or_create_gauge(
    "model_decision_threshold",
    "Optimal classification threshold loaded from model metadata.",
    (),
)

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook: initialize model and update readiness gauges on startup."""
    model_service.load()
    if model_service.ready:
        MODEL_READY.set(1)
        if model_service.model_version:
            MODEL_INFO.labels(version=model_service.model_version).set(1)
        if model_service.threshold is not None:
            MODEL_DECISION_THRESHOLD.set(model_service.threshold)
    else:
        MODEL_READY.set(0)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production inference API for Telco customer churn probability scoring "
        "and decision gatekeeping."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """Intercept each HTTP request to measure execution latency and tally status codes."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time

    route = request.scope.get("route")
    endpoint = getattr(route, "path", None) or "unmatched"
    status_code = str(response.status_code)
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=status_code,
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        endpoint=endpoint,
    ).observe(duration)

    return response


def _pseudonymize_customer_id(customer_id: Optional[str]) -> Optional[str]:
    """Pseudonymize customerID using truncated SHA-256 hash or return None."""
    if not customer_id:
        return None
    return hashlib.sha256(customer_id.encode("utf-8")).hexdigest()[:16]


def _log_prediction(
    customer_id: Optional[str],
    model_version: str,
    probability: float,
    prediction: str,
    features: Dict[str, Any],
) -> None:
    """
    Append prediction audit record to monitoring log file in JSON lines format.
    Catches all exceptions to ensure logging never degrades inference availability.
    """
    log_file_path = os.getenv("PREDICTIONS_LOG_PATH", "reports/monitoring/predictions_log.jsonl")
    try:
        target_path = Path(log_file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_id": _pseudonymize_customer_id(customer_id),
            "model_version": model_version,
            "probability": probability,
            "prediction": prediction,
            "features": features,
        }

        with target_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.error("Failed to write prediction audit log to %s: %s", log_file_path, exc)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness probe",
    description="Returns 200 with status 'ok' if model is loaded and ready, 503 otherwise.",
)
def health_check():
    if not model_service.ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable"},
        )
    return {"status": "ok"}


@app.get(
    "/model-info",
    response_model=ModelInfo,
    summary="Model metadata and operational threshold",
    description="Provides algorithm details, version, run ID, and optimal decision threshold.",
)
def get_model_info():
    if not model_service.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is unavailable.",
        )

    return ModelInfo(
        model_name=model_service.model_name or "random_forest",
        model_version=model_service.model_version or "1",
        registry_version=model_service.registry_version,
        run_id=model_service.run_id,
        threshold=model_service.threshold,
        source=model_service.source,
        registry_checked=model_service.registry_checked,
        registry_consistent=model_service.registry_consistent,
        sklearn_version=model_service.sklearn_version,
        metrics_optimal=model_service.metrics_optimal,
        schema_version=model_service.schema_version,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Real-time churn risk prediction",
    description=(
        "Scores customer features against the champion model pipeline, "
        "applies optimal threshold, and logs the prediction."
    ),
)
def predict_churn(features: ChurnFeatures):
    if not model_service.ready or model_service.model is None or model_service.threshold is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is unavailable. Prediction cannot be performed.",
        )

    # Build DataFrame strictly with the 19 features in schema contract order
    features_dict = features.to_schema_ordered_dict()
    df_input = pd.DataFrame([features_dict])

    try:
        probabilities = model_service.model.predict_proba(df_input)
        churn_prob = float(probabilities[0, 1])
    except Exception as exc:
        logger.error("Model inference execution error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed.",
        )

    decision = "Yes" if churn_prob >= model_service.threshold else "No"
    churn_prob_rounded = round(churn_prob, 4)

    # Audit logging (safe against errors)
    _log_prediction(
        customer_id=features.customerID,
        model_version=model_service.model_version or "unknown",
        probability=churn_prob_rounded,
        prediction=decision,
        features=features_dict,
    )

    # Prometheus prediction counter increment
    PREDICTIONS_TOTAL.labels(predicted_class=decision).inc()
    CHURN_PREDICTION_PROBABILITY.observe(churn_prob)

    return PredictionResponse(
        churn_probability=churn_prob_rounded,
        prediction=decision,
        threshold=model_service.threshold,
        model_version=model_service.model_version or "unknown",
        model_source=model_service.source,
    )


@app.get(
    "/metrics",
    summary="Prometheus monitoring metrics",
    description="Exposes request counters, latency histogram, prediction counts, and readiness gauges.",
)
def prometheus_metrics():
    MODEL_READY.set(1 if model_service.ready else 0)
    if model_service.ready and model_service.model_version:
        MODEL_INFO.labels(version=model_service.model_version).set(1)
    if model_service.ready and model_service.threshold is not None:
        MODEL_DECISION_THRESHOLD.set(model_service.threshold)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
