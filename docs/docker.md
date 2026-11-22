# Docker Deployment & Containerization Guide

This document describes the production-ready Docker packaging for the Customer Churn Prediction REST API.

---

## 1. Architecture & Design Rationale

### Base Image Choice: `python:3.12.10-slim`
- **Version Pinning**: Specifically pinned to `3.12.10-slim` to match the exact minor and patch version of Python used in development and local virtual environment (`.venv`), ensuring deterministic dependency resolution and preventing subtle C-extension or standard library incompatibilities.
- **Minimal Surface**: Debian slim provides a minimal Linux attack surface (~157 MB uncompressed base) without development packages, compilers, or extraneous system utilities.
- **No Extra System Dependencies**: `ca-certificates` is already pre-installed in the official `python:3.12.10-slim` image, avoiding unnecessary `apt-get` operations during image build.
- **No Build Tools Left Behind**: All Python dependencies are installed directly from wheels via `pip install --no-cache-dir`.

### Layer & Build Strategy
- **Layer Caching**: Dependencies from `requirements-serve.txt` are copied and installed before application code and build arguments, maximizing BuildKit cache reusability across commit changes.
- **Exclusions via `.dockerignore`**:
  - Training artifacts, raw/split data (`data/`)
  - Experiment tracking databases (`mlruns/`, `mlflow.db*`)
  - Test suites (`tests/`) and cache directories (`.pytest_cache`, `__pycache__`)
  - Development tools (`requirements.txt`, `requirements-api.txt`, `pytest`, `httpx2`)
  - Training and validation source code (`src/training`, `src/data`, `src/validation`, `src/lifecycle`)
  - All artifacts (`artifacts/*`) except `artifacts/champion_model.joblib` and `artifacts/model_metadata.json`
  - Only `src/__init__.py`, `src/api/`, `schema.json`, `artifacts/champion_model.joblib`, and `artifacts/model_metadata.json` are packaged into the container.

### Security & Least Privilege
- **Dedicated Non-root User**: The container runs under a dedicated `appuser` (UID/GID 1000). No root privileges are required or granted.
- **Docker Compose Security**: `security_opt: ["no-new-privileges:true"]` prevents privilege escalation attacks inside the container.
- **No Secrets**: No API keys, credentials, or secrets are embedded into the image or Dockerfile.

### OCI & Model Metadata Labels
Standard OCI annotations and project-specific labels are injected dynamically at build time:
- `org.opencontainers.image.title="churn-api"`
- `org.opencontainers.image.description="Customer churn prediction REST API"`
- `org.opencontainers.image.revision="${GIT_COMMIT}"`
- `churn.model.run_id="${MODEL_RUN_ID}"`
- `churn.model.version="${MODEL_VERSION}"`

---

## 2. Dockerfile Specification

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PREDICTIONS_LOG_PATH=/app/reports/monitoring/predictions_log.jsonl

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

ARG GIT_COMMIT=unknown
ARG MODEL_RUN_ID=unknown
ARG MODEL_VERSION=unknown

LABEL org.opencontainers.image.title="churn-api"
LABEL org.opencontainers.image.description="Customer churn prediction REST API"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL churn.model.run_id="${MODEL_RUN_ID}"
LABEL churn.model.version="${MODEL_VERSION}"

COPY src/__init__.py          src/__init__.py
COPY src/api/                 src/api/
COPY schema.json              schema.json
COPY artifacts/champion_model.joblib     artifacts/champion_model.joblib
COPY artifacts/model_metadata.json      artifacts/model_metadata.json

RUN mkdir -p /app/reports/monitoring \
    && chown -R appuser:appuser /app/reports

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. Docker Compose Configuration

The service is defined in `docker-compose.yml`:

```yaml
services:
  api:
    build:
      context: .
      args:
        GIT_COMMIT: ${GIT_COMMIT:-unknown}
        MODEL_RUN_ID: ${MODEL_RUN_ID:-unknown}
        MODEL_VERSION: ${MODEL_VERSION:-unknown}
    image: churn-api:local
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./reports/monitoring:/app/reports/monitoring
    environment:
      METADATA_PATH: ${METADATA_PATH:-artifacts/model_metadata.json}
      MODEL_PATH: ${MODEL_PATH:-artifacts/champion_model.joblib}
      # USE_REGISTRY intentionally absent -> defaults to false (no MLflow in container)
    healthcheck:
      test:
        - "CMD"
        - "python"
        - "-c"
        - "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"
      interval: 30s
      timeout: 5s
      start_period: 15s
      retries: 3
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
```

---

## 4. Usage Commands

Note for Windows environments: `make` is not required under Windows; the PowerShell commands documented below are the reference.

### Makefile Targets
Variables are dynamically retrieved from git and `artifacts/model_metadata.json`:
```bash
# Build the Docker image with commit and model metadata
make docker-build

# Start the service in detached mode
make up

# View real-time container logs
make logs

# Stop and remove containers
make down
```

### PowerShell Deployment Equivalent
On Windows PowerShell, build args are injected via environment variables:
```powershell
$meta = Get-Content artifacts\model_metadata.json -Raw | ConvertFrom-Json
$env:GIT_COMMIT = (git rev-parse HEAD); $env:MODEL_RUN_ID = $meta.run_id; $env:MODEL_VERSION = [string]$meta.registry_version
docker compose up -d --build
Remove-Item Env:GIT_COMMIT,Env:MODEL_RUN_ID,Env:MODEL_VERSION
```

> **Note on Revision Label**: The image should be rebuilt after creating a git commit so that the OCI `org.opencontainers.image.revision` label accurately matches the repository commit SHA of the running code.

### Direct Docker Commands
```powershell
$meta = Get-Content artifacts\model_metadata.json -Raw | ConvertFrom-Json

docker build `
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) `
  --build-arg MODEL_RUN_ID=$meta.run_id `
  --build-arg MODEL_VERSION=$meta.registry_version `
  -t churn-api:local .
```

---

## 5. Monitoring & Volume Persistence
- The REST API logs every incoming prediction request and model output to `$PREDICTIONS_LOG_PATH` (`/app/reports/monitoring/predictions_log.jsonl`).
- By bind-mounting the host's `./reports/monitoring` directory to `/app/reports/monitoring`, telemetry is preserved across container restarts and updates.
- The `predictions_log.jsonl` file is tracked in `.gitignore` so telemetry is never accidentally committed to version control.

---

## 6. Echec controle (demonstration)

Demonstration de defaillance controlee en PowerShell lorsque le fichier de metadonnees est manquant ou inaccessible:

```powershell
# 1. Simuler un chemin de metadonnees invalide
$env:METADATA_PATH = "/app/absent.json"
docker compose up -d --force-recreate

# 2. Verifier le comportement des routes
# GET /health renvoie 503 Service Unavailable: {"status":"unavailable"}
curl.exe -i http://127.0.0.1:8000/health

# POST /predict renvoie 503 Service Unavailable sans cle "prediction"
curl.exe -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{}"

# 3. Constat du statut du conteneur
# Apres environ 110 s (start_period de 15 s + 3 echecs espaces de 30 s de l'intervalle):
docker compose ps
# STATUS affiche "Up ... (unhealthy)"

# 4. Restauration de l'etat nominal
Remove-Item Env:METADATA_PATH
docker compose up -d --force-recreate
# Attendre environ 45 s pour que le premier controle reussisse:
docker compose ps
# STATUS repasse a "Up ... (healthy)"
```

Ce test prouve deux points essentiels:
- Le service refuse categoriquement d'effectuer une prediction sans modele valide charge (garantissant qu'aucune fausse prediction n'est retournee).
- L'orchestrateur Docker detecte la degradation via le healthcheck et marque le conteneur comme unhealthy.

---

## 7. Limites connues

1. **Permissions sur le volume de monitoring (uid 1000)**:
   - Le conteneur s'execute sous l'utilisateur dedie non-root `appuser` avec uid 1000.
   - Sous Windows avec Docker Desktop, le bind mount `reports/monitoring` est inscriptible grace a la virtualisation du systeme de fichiers.
   - Sous Linux, si le dossier de l'hote appartient a un autre utilisateur (ex. root), l'ecriture du journal `predictions_log.jsonl` echoue: l'API continue de repondre normalement (l'echec d'ecriture est capture et ne casse pas `/predict`), mais aucune ligne n'est journalisee. Contournement attendu (limite attendue, non testee sous Linux): executer `chown -R 1000:1000 reports/monitoring` sur l'hote avant le demarrage.

2. **Absence de registre MLflow dans l'image**:
   - Par conception, aucun registre MLflow n'est integre au conteneur (la base `mlflow.db` n'est pas versionnee).
   - La source du modele est exclusivement le fichier joblib embarque (`artifacts/champion_model.joblib`), et l'endpoint `/model-info` expose `registry_checked=false` et `registry_consistent=null`.

3. **Synchronisation du label OCI revision**:
   - Le label `org.opencontainers.image.revision` est injecte au build a partir de `git rev-parse HEAD`. Il ne correspond au code exact que si l'image est reconstruite apres le commit git.
