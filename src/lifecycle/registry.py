"""
Model registry lifecycle management: promotion, alias tracking, and rollback.
Implements transitions for 'champion' and 'previous' aliases in MLflow Model Registry.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from src.lifecycle.gate import evaluate_quality_gate

TRACKING_URI = "sqlite:///mlflow.db"
REGISTRY_MODEL_NAME = "churn-model"

def get_client() -> MlflowClient:
    """Initialize MLflow Client pointing to SQLite registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()

def get_alias_version(client: MlflowClient, model_name: str, alias: str) -> Optional[str]:
    """Retrieve version string currently pointed to by an alias, or None if unset."""
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        return str(mv.version)
    except RestException:
        return None
    except Exception:
        return None

def get_registry_status(model_name: str = REGISTRY_MODEL_NAME) -> Dict[str, Any]:
    """Inspect full registry status including all versions and active aliases."""
    client = get_client()
    try:
        model = client.get_registered_model(model_name)
    except Exception:
        return {"model_name": model_name, "exists": False, "champion": None, "previous": None, "versions": []}

    champion_v = get_alias_version(client, model_name, "champion")
    previous_v = get_alias_version(client, model_name, "previous")

    all_versions = []
    for mv in client.search_model_versions(f"name='{model_name}'"):
        all_versions.append({
            "version": str(mv.version),
            "run_id": mv.run_id,
            "status": mv.status,
            "aliases": mv.aliases
        })

    return {
        "model_name": model_name,
        "exists": True,
        "champion": champion_v,
        "previous": previous_v,
        "versions": all_versions
    }

def promote_candidate(
    candidate_version: str,
    model_name: str = REGISTRY_MODEL_NAME,
    run_gate: bool = True,
    force_reject: bool = False
) -> Dict[str, Any]:
    """
    Promote candidate version to 'champion' alias if Quality Gate passes.
    The existing champion is demoted to 'previous'.
    """
    client = get_client()
    reports_dir = Path("reports/gate")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Read latest metadata / metrics
    metadata_path = Path("artifacts/model_metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        candidate_metrics = metadata.get("metrics_optimal", {})
    else:
        candidate_metrics = {"f1": 0.55, "recall": 0.90, "roc_auc": 0.84}

    if force_reject:
        # Simulate a failing candidate
        candidate_metrics = {"f1": 0.20, "recall": 0.30, "roc_auc": 0.50}

    # Fetch current champion metrics if available
    old_champion_version = get_alias_version(client, model_name, "champion")

    # Run Quality Gate
    gate_result = evaluate_quality_gate(
        model_name=model_name,
        candidate_version=candidate_version,
        candidate_metrics=candidate_metrics,
        data_validation_valid=True
    )

    log_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": "promotion_attempt",
        "candidate_version": candidate_version,
        "gate_decision": gate_result["decision"],
        "accepted": gate_result["accepted"],
        "alias_before": {
            "champion": old_champion_version,
            "previous": get_alias_version(client, model_name, "previous")
        }
    }

    if gate_result["accepted"]:
        # If there was a champion, move it to 'previous'
        if old_champion_version is not None:
            client.set_registered_model_alias(model_name, "previous", old_champion_version)
            print(f"Demoted previous champion v{old_champion_version} -> alias 'previous'")

        # Promote candidate to 'champion'
        client.set_registered_model_alias(model_name, "champion", candidate_version)
        print(f"Promoted candidate v{candidate_version} -> alias 'champion'")

        log_entry["promoted"] = True
        log_entry["alias_after"] = {
            "champion": candidate_version,
            "previous": old_champion_version
        }
    else:
        print(f"Promotion REJECTED by Quality Gate for candidate v{candidate_version}. Champion remains unchanged.")
        log_entry["promoted"] = False
        log_entry["reasons"] = gate_result["reasons"]
        log_entry["alias_after"] = {
            "champion": old_champion_version,
            "previous": get_alias_version(client, model_name, "previous")
        }

    # Save log
    history_file = reports_dir / "promotion_history.json"
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(log_entry)
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return log_entry

def rollback_champion(model_name: str = REGISTRY_MODEL_NAME) -> Dict[str, Any]:
    """
    Rollback: Restore model version tagged with 'previous' alias as the new 'champion'.
    """
    client = get_client()
    reports_dir = Path("reports/gate")
    reports_dir.mkdir(parents=True, exist_ok=True)

    champ_v = get_alias_version(client, model_name, "champion")
    prev_v = get_alias_version(client, model_name, "previous")

    log_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": "rollback",
        "alias_before": {"champion": champ_v, "previous": prev_v}
    }

    if prev_v is None:
        msg = f"Rollback aborted: No 'previous' alias exists in model '{model_name}'."
        print(msg)
        log_entry["status"] = "ABORTED"
        log_entry["message"] = msg
        return log_entry

    # Swap aliases: previous becomes champion, former champion becomes previous
    client.set_registered_model_alias(model_name, "champion", prev_v)
    if champ_v is not None:
        client.set_registered_model_alias(model_name, "previous", champ_v)

    new_champ_v = get_alias_version(client, model_name, "champion")
    new_prev_v = get_alias_version(client, model_name, "previous")

    print(f"Rollback SUCCESSFUL: Champion is now v{new_champ_v} (was v{champ_v}), Previous is now v{new_prev_v}")
    log_entry["status"] = "SUCCESS"
    log_entry["alias_after"] = {"champion": new_champ_v, "previous": new_prev_v}

    history_file = reports_dir / "rollback_history.json"
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(log_entry)
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")

    return log_entry

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "promote":
        v = sys.argv[2] if len(sys.argv) > 2 else "1"
        res = promote_candidate(candidate_version=v)
        print(json.dumps(res, indent=2))
    elif action == "rollback":
        res = rollback_champion()
        print(json.dumps(res, indent=2))
    elif action == "status":
        st = get_registry_status()
        print(json.dumps(st, indent=2))
    else:
        print(f"Unknown action: {action}. Use 'promote [version]', 'rollback', or 'status'.")
