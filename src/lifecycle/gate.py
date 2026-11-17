"""
Quality Gate for Customer Churn Model Promotion.
Evaluates candidate model against explicit validation, baseline, metric, and champion rules.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Dict, Any, List, Optional

# Justified minimal thresholds founded on exploration and baseline performance:
# - Baseline Always Yes achieves F1=0.4192 on test. Candidate must significantly beat it.
# - High Recall on churners is critical to prevent lost customer revenue.
GATE_THRESHOLDS = {
    "min_recall": 0.70,
    "min_f1": 0.50,
    "min_roc_auc": 0.75,
    "baseline_f1_threshold": 0.42
}

def evaluate_quality_gate(
    model_name: str,
    candidate_version: str,
    candidate_metrics: Dict[str, float],
    data_validation_valid: bool = True,
    current_champion_metrics: Optional[Dict[str, float]] = None,
    output_dir: Path = Path("reports/gate")
) -> Dict[str, Any]:
    """
    Run Quality Gate checks on candidate model version.
    Returns structured decision dictionary and writes JSON reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    reasons = []

    # Check 1: Data validation status
    if data_validation_valid:
        checks.append({"name": "data_validation", "status": "PASSED", "message": "All data splits conform to Pandera schema."})
    else:
        checks.append({"name": "data_validation", "status": "FAILED", "message": "Data validation failed."})
        reasons.append("Data validation failed schema checks.")

    # Check 2: Beat naive baselines
    cand_f1 = candidate_metrics.get("f1", 0.0)
    baseline_threshold = GATE_THRESHOLDS["baseline_f1_threshold"]
    if cand_f1 > baseline_threshold:
        checks.append({"name": "beat_baselines", "status": "PASSED", "message": f"F1 {cand_f1} exceeds baseline threshold {baseline_threshold}."})
    else:
        checks.append({"name": "beat_baselines", "status": "FAILED", "message": f"F1 {cand_f1} fails to beat baseline threshold {baseline_threshold}."})
        reasons.append(f"F1-score ({cand_f1}) does not beat naive baselines ({baseline_threshold}).")

    # Check 3: Recall threshold
    cand_recall = candidate_metrics.get("recall", 0.0)
    if cand_recall >= GATE_THRESHOLDS["min_recall"]:
        checks.append({"name": "min_recall", "status": "PASSED", "message": f"Recall {cand_recall} >= {GATE_THRESHOLDS['min_recall']}."})
    else:
        checks.append({"name": "min_recall", "status": "FAILED", "message": f"Recall {cand_recall} < {GATE_THRESHOLDS['min_recall']}."})
        reasons.append(f"Recall ({cand_recall}) below required minimum ({GATE_THRESHOLDS['min_recall']}).")

    # Check 4: F1 threshold
    if cand_f1 >= GATE_THRESHOLDS["min_f1"]:
        checks.append({"name": "min_f1", "status": "PASSED", "message": f"F1 {cand_f1} >= {GATE_THRESHOLDS['min_f1']}."})
    else:
        checks.append({"name": "min_f1", "status": "FAILED", "message": f"F1 {cand_f1} < {GATE_THRESHOLDS['min_f1']}."})
        reasons.append(f"F1-score ({cand_f1}) below required minimum ({GATE_THRESHOLDS['min_f1']}).")

    # Check 5: ROC-AUC threshold
    cand_auc = candidate_metrics.get("roc_auc", 0.0)
    if cand_auc >= GATE_THRESHOLDS["min_roc_auc"]:
        checks.append({"name": "min_roc_auc", "status": "PASSED", "message": f"ROC-AUC {cand_auc} >= {GATE_THRESHOLDS['min_roc_auc']}."})
    else:
        checks.append({"name": "min_roc_auc", "status": "FAILED", "message": f"ROC-AUC {cand_auc} < {GATE_THRESHOLDS['min_roc_auc']}."})
        reasons.append(f"ROC-AUC ({cand_auc}) below required minimum ({GATE_THRESHOLDS['min_roc_auc']}).")

    # Check 6: Comparison with current champion (if one exists)
    if current_champion_metrics:
        champ_f1 = current_champion_metrics.get("f1", 0.0)
        # Allow candidate if within 0.01 margin or better
        if cand_f1 >= (champ_f1 - 0.01):
            checks.append({"name": "champion_comparison", "status": "PASSED", "message": f"Candidate F1 ({cand_f1}) is competitive with Champion F1 ({champ_f1})."})
        else:
            checks.append({"name": "champion_comparison", "status": "FAILED", "message": f"Candidate F1 ({cand_f1}) is significantly lower than Champion F1 ({champ_f1})."})
            reasons.append(f"Candidate F1 ({cand_f1}) is inferior to current Champion F1 ({champ_f1}).")

    accepted = len(reasons) == 0
    decision = "ACCEPTED" if accepted else "REJECTED"

    gate_result = {
        "model_name": model_name,
        "candidate_version": str(candidate_version),
        "decision": decision,
        "accepted": accepted,
        "metrics": candidate_metrics,
        "checks": checks,
        "reasons": reasons,
        "timestamp_utc": datetime.now(timezone.utc).isoformat()
    }

    # Save to reports/gate
    (output_dir / "quality_gate_result.json").write_text(json.dumps(gate_result, indent=2), encoding="utf-8")
    (output_dir / f"gate_decision_v{candidate_version}.json").write_text(json.dumps(gate_result, indent=2), encoding="utf-8")

    return gate_result

