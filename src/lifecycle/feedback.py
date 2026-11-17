"""
Offline delayed feedback and retraining module.
Simulates delayed ground truth arrival by replaying stream features and labels separately.
Evaluates realized champion performance, triggers retraining (v2), and applies the Quality Gate.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.training.pipeline import get_model_pipelines
from src.training.train import evaluate_predictions, setup_mlflow, REGISTRY_MODEL_NAME, TRACKING_URI
from src.training.cost import find_optimal_threshold
from src.lifecycle.registry import get_alias_version, promote_candidate, get_client
from src.data.manifest import compute_sha256, get_git_commit_hash

def evaluate_stream_feedback(
    splits_dir: Path = Path("data/splits"),
    reports_dir: Path = Path("reports/training")
) -> dict:
    """
    Step 1 to 7: Replay delayed stream labels against current Champion model.
    Loads features and delayed ground-truth labels separately.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    x_stream_path = splits_dir / "stream" / "X_stream.csv"
    y_stream_path = splits_dir / "stream" / "y_stream.csv"

    if not x_stream_path.exists() or not y_stream_path.exists():
        raise FileNotFoundError("Stream split files not found.")

    # 1. Load stream features
    X_stream = pd.read_csv(x_stream_path)
    # 2. Load delayed ground truth labels separately
    y_stream = pd.read_csv(y_stream_path)["Churn"].map({"Yes": 1, "No": 0}).values

    # 3. Load currently champion model
    client = get_client()
    champ_version = get_alias_version(client, REGISTRY_MODEL_NAME, "champion")
    model_path = Path("artifacts/champion_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Champion model artifact not found.")
    champion_pipeline = joblib.load(model_path)

    # Read optimal threshold from metadata
    meta_path = Path("artifacts/model_metadata.json")
    threshold = 0.5
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        threshold = meta.get("optimal_threshold", 0.5)

    # 4. Perform predictions on stream
    stream_probs = champion_pipeline.predict_proba(X_stream)[:, 1]

    # 5 & 6. Calculate realized metrics on stream
    realized_metrics = evaluate_predictions(y_stream, stream_probs, threshold=threshold)

    feedback_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_type": "offline_delayed_feedback_replay",
        "champion_version": champ_version,
        "n_stream_observations": len(y_stream),
        "threshold_used": threshold,
        "realized_metrics": realized_metrics,
        "note": "Ground truth labels are replayed from isolated stream split; no real production stream telemetry."
    }

    # 7. Log results
    out_json = reports_dir / "feedback_stream_metrics.json"
    out_json.write_text(json.dumps(feedback_summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Rapport de Feedback Différé Hors-Ligne (Stream Evaluation)",
        "",
        f"- **Date de rejeu** : {feedback_summary['timestamp_utc']}",
        f"- **Version Champion Évaluée** : v{champ_version}",
        f"- **Échantillons Stream Rejoués** : {len(y_stream)} clients",
        f"- **Seuil de Décision Appliqué** : {threshold}",
        "",
        "## Performances Réelles Constatées sur le Stream",
        "",
        f"- **Precision** : {realized_metrics['precision']}",
        f"- **Recall** : {realized_metrics['recall']}",
        f"- **F1-Score** : {realized_metrics['f1']}",
        f"- **ROC-AUC** : {realized_metrics['roc_auc']}",
        f"- **PR-AUC** : {realized_metrics['pr_auc']}",
        f"- **Coût Total Estimé** : {realized_metrics['total_cost']}\\$",
        f"- **Coût Moyen / Client** : {realized_metrics['avg_cost_per_customer']}\\$",
        "",
        "> [!NOTE]",
        "> Ce mécanisme simule l'arrivée différée des labels réels après facturation mensuelle.",
        "> Les labels étaient tenus à l'écart du pipeline d'entraînement initial pour éviter tout biais.",
        ""
    ]
    (reports_dir / "stream_feedback_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Feedback evaluated. Stream F1: {realized_metrics['f1']}, Recall: {realized_metrics['recall']}")
    return feedback_summary

def retrain_and_gate_candidate(
    splits_dir: Path = Path("data/splits"),
    seed: int = 42,
    force_failing: bool = False
) -> dict:
    """
    Step 8 to 10: Retrain model candidate on accumulated data (Train + Stream),
    evaluate on held-out Test split, run Quality Gate, and promote if accepted.
    """
    setup_mlflow()
    client = get_client()

    # Load Train and Stream splits to simulate accumulated training data
    X_train = pd.read_csv(splits_dir / "train" / "X_train.csv")
    y_train = pd.read_csv(splits_dir / "train" / "y_train.csv")["Churn"].map({"Yes": 1, "No": 0}).values

    X_stream = pd.read_csv(splits_dir / "stream" / "X_stream.csv")
    y_stream = pd.read_csv(splits_dir / "stream" / "y_stream.csv")["Churn"].map({"Yes": 1, "No": 0}).values

    # Merge Train + Stream
    X_accumulated = pd.concat([X_train, X_stream], ignore_index=True)
    y_accumulated = np.concatenate([y_train, y_stream])

    X_test = pd.read_csv(splits_dir / "test" / "X_test.csv")
    y_test = pd.read_csv(splits_dir / "test" / "y_test.csv")["Churn"].map({"Yes": 1, "No": 0}).values

    # Choose model architecture (Random Forest or Logistic Regression with retrained data)
    pipelines = get_model_pipelines(seed=seed)
    model_name = "random_forest" if not force_failing else "baseline_always_no"
    pipeline = pipelines[model_name]

    with mlflow.start_run(run_name=f"retrain_candidate_{model_name}") as run:
        mlflow.log_param("retrain_mode", "accumulated_train_plus_stream")
        mlflow.log_param("total_train_samples", len(X_accumulated))
        mlflow.log_param("model_name", model_name)

        pipeline.fit(X_accumulated, y_accumulated)

        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            probs = pipeline.predict_proba(X_test)[:, 1]
            opt = find_optimal_threshold(y_accumulated, pipeline.predict_proba(X_accumulated)[:, 1])
            thresh = opt["optimal_threshold"]
        else:
            probs = np.zeros(len(y_test))
            thresh = 0.5

        metrics = evaluate_predictions(y_test, probs, threshold=thresh)

        mlflow.log_metric("retrain_test_f1", metrics["f1"])
        mlflow.log_metric("retrain_test_recall", metrics["recall"])
        mlflow.log_metric("retrain_test_roc_auc", metrics["roc_auc"])

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        candidate_mv = mlflow.register_model(model_uri, REGISTRY_MODEL_NAME)
        cand_v = str(candidate_mv.version)
        print(f"Retrained model registered as version {cand_v} under '{REGISTRY_MODEL_NAME}'")

    # Step 9 & 10: Run Quality Gate and promote or reject
    promotion_result = promote_candidate(
        candidate_version=cand_v,
        model_name=REGISTRY_MODEL_NAME,
        run_gate=True,
        force_reject=force_failing
    )

    if promotion_result["promoted"]:
        # Update local champion artifact
        joblib.dump(pipeline, Path("artifacts/champion_model.joblib"))
        meta = {
            "model_name": model_name,
            "registry_name": REGISTRY_MODEL_NAME,
            "registry_version": cand_v,
            "run_id": run.info.run_id,
            "optimal_threshold": thresh,
            "random_seed": seed,
            "metrics_optimal": metrics
        }
        Path("artifacts/model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "candidate_version": cand_v,
        "metrics": metrics,
        "promotion_result": promotion_result
    }

if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "feedback"
    if action == "feedback":
        evaluate_stream_feedback()
    elif action == "retrain":
        retrain_and_gate_candidate()
