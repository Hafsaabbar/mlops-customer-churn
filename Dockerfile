# syntax=docker/dockerfile:1
# Base image: python:3.12.10-slim chosen for minimal attack surface while
# matching the development Python version exactly (Python 3.12.10).
FROM python:3.12.10-slim

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PREDICTIONS_LOG_PATH=/app/reports/monitoring/predictions_log.jsonl

# Note: USE_REGISTRY is intentionally NOT set here.
# Default behaviour (false) means the container loads the joblib artefact
# only - no MLflow registry is consulted at runtime.

# ---------------------------------------------------------------------------
# Non-root user
# ---------------------------------------------------------------------------
RUN useradd --create-home --shell /bin/bash appuser

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies (no build deps left in image)
# ---------------------------------------------------------------------------
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# ---------------------------------------------------------------------------
# Build-time arguments (injected by docker build --build-arg / compose args)
# ---------------------------------------------------------------------------
ARG GIT_COMMIT=unknown
ARG MODEL_RUN_ID=unknown
ARG MODEL_VERSION=unknown

# ---------------------------------------------------------------------------
# OCI standard labels + project-specific labels
# ---------------------------------------------------------------------------
LABEL org.opencontainers.image.title="churn-api"
LABEL org.opencontainers.image.description="Customer churn prediction REST API"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL churn.model.run_id="${MODEL_RUN_ID}"
LABEL churn.model.version="${MODEL_VERSION}"

# ---------------------------------------------------------------------------
# Copy application source (only what is needed at runtime)
# ---------------------------------------------------------------------------
COPY src/__init__.py          src/__init__.py
COPY src/api/                 src/api/
COPY schema.json              schema.json
COPY artifacts/champion_model.joblib     artifacts/champion_model.joblib
COPY artifacts/model_metadata.json      artifacts/model_metadata.json

# ---------------------------------------------------------------------------
# Writable directory for prediction logs (bind-mounted in compose)
# ---------------------------------------------------------------------------
RUN mkdir -p /app/reports/monitoring \
    && chown -R appuser:appuser /app/reports

# ---------------------------------------------------------------------------
# Drop to non-root
# ---------------------------------------------------------------------------
USER appuser

EXPOSE 8000

# ---------------------------------------------------------------------------
# Healthcheck using Python urllib (curl is absent in slim)
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
