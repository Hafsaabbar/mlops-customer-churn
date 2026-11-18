"""
Model loading and lifecycle management service for inference.
Handles loading from local joblib artifacts or MLflow Model Registry with fallback.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import joblib
import sklearn

logger = logging.getLogger(__name__)


class ModelService:
    """
    Encapsulates model artifact loading, decision threshold extraction,
    and optional MLflow registry synchronization.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        self.model_path = model_path or os.getenv("MODEL_PATH", "artifacts/champion_model.joblib")
        self.metadata_path = metadata_path or os.getenv("METADATA_PATH", "artifacts/model_metadata.json")
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

        self.ready: bool = False
        self.model: Any = None
        self.threshold: Optional[float] = None
        self.model_version: Optional[str] = None
        self.registry_version: Optional[str] = None
        self.run_id: Optional[str] = None
        self.model_name: Optional[str] = None
        self.source: str = "joblib"
        self.registry_checked: bool = False
        self.registry_consistent: Optional[bool] = None
        self.sklearn_version: str = sklearn.__version__
        self.metrics_optimal: Optional[Dict[str, Any]] = None
        self.schema_version: Optional[str] = "1.0.0"

    def load(self) -> bool:
        """
        Load model artifact and associated metadata.
        Guarantees no unhandled exceptions during startup; sets ready=False on error.
        """
        self.ready = False
        self.model = None

        meta_path = Path(self.metadata_path)
        if not meta_path.is_file():
            logger.warning("Metadata file missing at: %s", self.metadata_path)
            return False

        try:
            raw_meta = meta_path.read_text(encoding="utf-8")
            meta = json.loads(raw_meta)
            self.threshold = float(meta["optimal_threshold"])
            self.run_id = meta.get("run_id")
            self.model_name = meta.get("model_name", "random_forest")
            self.registry_version = str(meta.get("registry_version", "1"))
            self.model_version = self.registry_version
            self.metrics_optimal = meta.get("metrics_optimal", {})
            registry_name = meta.get("registry_name", "churn-model")
        except Exception as exc:
            logger.error("Failed to parse metadata file %s: %s", self.metadata_path, exc)
            return False

        # Attempt to read schema version from schema.json if present
        schema_path = Path("schema.json")
        if schema_path.is_file():
            try:
                schema_json = json.loads(schema_path.read_text(encoding="utf-8"))
                self.schema_version = schema_json.get("version", "1.0.0")
            except Exception:
                self.schema_version = "1.0.0"

        use_registry = os.getenv("USE_REGISTRY", "false").strip().lower() in ("true", "1", "yes")

        if use_registry:
            self.registry_checked = True
            loaded_from_registry = False
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                client = MlflowClient()
                champ_mv = client.get_model_version_by_alias(registry_name, "champion")

                if champ_mv and champ_mv.run_id == self.run_id:
                    import mlflow.sklearn

                    model_uri = f"models:/{registry_name}@champion"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.source = "registry"
                    self.registry_consistent = True
                    self.model_version = str(champ_mv.version)
                    self.ready = True
                    loaded_from_registry = True
                    logger.info("Loaded champion model v%s directly from MLflow registry.", self.model_version)
                else:
                    found_run_id = getattr(champ_mv, "run_id", None) if champ_mv else None
                    logger.warning(
                        "MLflow champion run_id mismatch: registry has %s vs metadata %s. Falling back to joblib.",
                        found_run_id,
                        self.run_id,
                    )
                    self.registry_consistent = False
            except Exception as exc:
                logger.warning("MLflow registry query failed: %s. Falling back to joblib.", exc)
                self.registry_consistent = False

            if not loaded_from_registry:
                self._load_joblib()
        else:
            self.registry_checked = False
            self.registry_consistent = None
            self.source = "joblib"
            self._load_joblib()

        return self.ready

        return self.ready

    def _load_joblib(self) -> None:
        """Fallback or default: load standalone scikit-learn pipeline from joblib."""
        model_path = Path(self.model_path)
        if not model_path.is_file():
            logger.warning("Joblib model file missing at: %s", self.model_path)
            self.ready = False
            return

        try:
            self.model = joblib.load(self.model_path)
            self.source = "joblib"
            self.ready = True
            logger.info("Loaded model successfully from joblib file: %s", self.model_path)
        except Exception as exc:
            logger.error("Failed to load model from joblib file %s: %s", self.model_path, exc)
            self.ready = False
            self.model = None
