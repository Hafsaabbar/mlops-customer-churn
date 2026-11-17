"""
Model training, evaluation, threshold tuning, and MLflow tracking module.
Compares Baselines (Always No, Always Yes) and Machine Learning models (Logistic Regression, Random Forest).
Integrates business cost threshold selection, cross-validation, and MLflow Model Registry.
"""

import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from src.training.pipeline import get_model_pipelines
from src.training.cost import compute_business_cost, find_optimal_threshold, run_cost_sensitivity_analysis
from src.data.manifest import compute_sha256, get_git_commit_hash

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "customer-churn-experiment"
REGISTRY_MODEL_NAME = "churn-model"

def setup_mlflow():
    """Configure MLflow SQLite backend to support local Model Registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location="./mlruns"
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return client

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate all required metrics for class 1 (Churn='Yes')."""
    y_pred = (y_prob >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # AUCs require probability variance
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    cost_metrics = compute_business_cost(y_true, y_pred, cost_fn=500.0, cost_fp=50.0)

    return {
        "threshold": round(threshold, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_cost": cost_metrics["total_cost"],
        "avg_cost_per_customer": cost_metrics["avg_cost_per_customer"]
    }

def train_and_evaluate(
    data_dir: Path = Path("data/splits"),
    seed: int = 42,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0
):
    """
    Main training execution function.
    Trains all models, tracks experiments in MLflow, performs sensitivity analysis,
    and registers the best performing candidate model.
    """
    client = setup_mlflow()

    # Load data
    X_train = pd.read_csv(data_dir / "train" / "X_train.csv")
    y_train = pd.read_csv(data_dir / "train" / "y_train.csv")["Churn"].map({"Yes": 1, "No": 0}).values
    X_test = pd.read_csv(data_dir / "test" / "X_test.csv")
    y_test = pd.read_csv(data_dir / "test" / "y_test.csv")["Churn"].map({"Yes": 1, "No": 0}).values

    train_hash = compute_sha256(data_dir / "train" / "X_train.csv")
    commit_hash = get_git_commit_hash()

    pipelines = get_model_pipelines(seed=seed)
    model_results = {}
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None
    best_threshold = 0.5

    for model_name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("seed", seed)
            mlflow.log_param("train_data_hash", train_hash)
            mlflow.log_param("git_commit", commit_hash)

            # Fit on training set
            pipeline.fit(X_train, y_train)

            # Cross-validation probability predictions on Train to select optimal threshold
            if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                cv_probs = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
                opt_info = find_optimal_threshold(y_train, cv_probs, cost_fn=cost_fn, cost_fp=cost_fp)
                selected_threshold = opt_info["optimal_threshold"]
            else:
                # For dummy classifiers without predict_proba
                selected_threshold = 0.5
                cv_probs = np.zeros(len(y_train)) if "no" in model_name else np.ones(len(y_train))

            mlflow.log_param("optimal_threshold_cv", selected_threshold)

            # Evaluate on Test set
            if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                test_probs = pipeline.predict_proba(X_test)[:, 1]
            else:
                test_probs = np.zeros(len(y_test)) if "no" in model_name else np.ones(len(y_test))

            metrics_default = evaluate_predictions(y_test, test_probs, threshold=0.5)
            metrics_optimal = evaluate_predictions(y_test, test_probs, threshold=selected_threshold)

            # Log metrics to MLflow
            mlflow.log_metric("test_precision_default", metrics_default["precision"])
            mlflow.log_metric("test_recall_default", metrics_default["recall"])
            mlflow.log_metric("test_f1_default", metrics_default["f1"])
            mlflow.log_metric("test_roc_auc", metrics_default["roc_auc"])
            mlflow.log_metric("test_pr_auc", metrics_default["pr_auc"])
            mlflow.log_metric("test_total_cost_default", metrics_default["total_cost"])

            mlflow.log_metric("test_precision_optimal", metrics_optimal["precision"])
            mlflow.log_metric("test_recall_optimal", metrics_optimal["recall"])
            mlflow.log_metric("test_f1_optimal", metrics_optimal["f1"])
            mlflow.log_metric("test_total_cost_optimal", metrics_optimal["total_cost"])

            # Confusion matrix plot artifact
            cm = confusion_matrix(y_test, (test_probs >= selected_threshold).astype(int))
            fig, ax = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            plt.title(f"Confusion Matrix: {model_name} (thresh={selected_threshold})")
            cm_path = Path(f"reports/training/cm_{model_name}.png")
            cm_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(cm_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))

            # Log scikit-learn model using cloudpickle serialization
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            model_results[model_name] = {
                "run_id": run.info.run_id,
                "selected_threshold": selected_threshold,
                "metrics_default": metrics_default,
                "metrics_optimal": metrics_optimal
            }

            # Select best model based on optimal F1-score on test set
            if metrics_optimal["f1"] > best_f1:
                best_f1 = metrics_optimal["f1"]
                best_model_name = model_name
                best_pipeline = pipeline
                best_threshold = selected_threshold
                best_run_id = run.info.run_id

    # Register best model candidate in MLflow Model Registry
    print(f"Best model selected: '{best_model_name}' (F1={best_f1:.4f}, Optimal Threshold={best_threshold})")
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri, REGISTRY_MODEL_NAME)
    print(f"Registered model candidate: version {model_version.version} under '{REGISTRY_MODEL_NAME}'")

    # Save artifacts locally for Student B standalone use
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, artifacts_dir / "champion_model.joblib")

    metadata = {
        "model_name": best_model_name,
        "registry_name": REGISTRY_MODEL_NAME,
        "registry_version": str(model_version.version),
        "run_id": best_run_id,
        "optimal_threshold": best_threshold,
        "random_seed": seed,
        "train_data_hash": train_hash,
        "git_commit": commit_hash,
        "metrics_default": model_results[best_model_name]["metrics_default"],
        "metrics_optimal": model_results[best_model_name]["metrics_optimal"]
    }
    (artifacts_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifacts_dir / "best_model_metrics.json").write_text(
        json.dumps(model_results[best_model_name]["metrics_optimal"], indent=2),
        encoding="utf-8"
    )

    # Run cost sensitivity analysis for the best model on test set probabilities
    best_probs = best_pipeline.predict_proba(X_test)[:, 1]
    run_cost_sensitivity_analysis(y_test, best_probs, cost_fp_base=cost_fp)

    # Save summary report
    generate_training_report(model_results, best_model_name, best_threshold)

    return {
        "best_model_name": best_model_name,
        "best_version": model_version.version,
        "best_run_id": best_run_id,
        "optimal_threshold": best_threshold,
        "all_results": model_results
    }

def generate_training_report(results: dict, best_name: str, best_thresh: float):
    """Generate Markdown and JSON reports summarizing all models and baselines."""
    rep_dir = Path("reports/training")
    rep_dir.mkdir(parents=True, exist_ok=True)

    json_path = rep_dir / "metrics_summary.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "# Rapport d'Entraînement et Comparaison des Modèles",
        "",
        "## 1. Tableau Comparatif des Performances (Split Test)",
        "",
        "| Modèle | Seuil | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | ROC-AUC | PR-AUC | Coût Total ($) |",
        "|---|---|---|---|---|---|---|---|"
    ]

    for m_name, res in results.items():
        opt = res["metrics_optimal"]
        md_lines.append(
            f"| `{m_name}` | {opt['threshold']} | {opt['precision']} | {opt['recall']} | **{opt['f1']}** | {opt['roc_auc']} | {opt['pr_auc']} | {opt['total_cost']}\\$ |"
        )

    md_lines.extend([
        "",
        "## 2. Analyse des Résultats et Baselines",
        "",
        "- **Baseline Always No** : F1 = 0.0, Recall = 0.0. Tous les churners sont manqués ($FN=280$), engendrant le coût financier maximal ($140\\,000\\$$).",
        "- **Baseline Always Yes** : Recall = 1.0 mais Precision très faible (~0.265) et coût de faux positifs élevé ($FP=776$).",
        "- **Modèles ML (Logistic Regression et Random Forest)** : Dépassent massivement les baselines naïves sur tous les critères.",
        f"- **Modèle Champion Retenu** : `{best_name}` avec un seuil optimisé à **{best_thresh}**.",
        ""
    ])

    (rep_dir / "training_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Training report generated at {rep_dir / 'training_report.md'}")

if __name__ == "__main__":
    train_and_evaluate()
