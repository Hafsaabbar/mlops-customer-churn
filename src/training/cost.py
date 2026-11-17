"""
Business cost metrics, optimal threshold search, and sensitivity analysis.
Evaluates cost tradeoffs between False Negatives (lost churners) and False Positives (retention spend).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json

def compute_confusion_matrix_elements(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Compute TP, FP, TN, FN counts."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0
) -> Dict[str, float]:
    """
    Compute total business cost and cost per customer under explicit cost hypotheses:
    - cost_fn: Lost customer lifetime value / revenue when churner is missed ($500 hypothesis)
    - cost_fp: Cost of unnecessary discount / retention outreach to loyal customer ($50 hypothesis)
    """
    cm = compute_confusion_matrix_elements(y_true, y_pred)
    total_cost = cm["fn"] * cost_fn + cm["fp"] * cost_fp
    n = len(y_true)
    avg_cost = total_cost / n if n > 0 else 0.0

    return {
        "cost_fn_unit": cost_fn,
        "cost_fp_unit": cost_fp,
        "total_cost": round(total_cost, 2),
        "avg_cost_per_customer": round(avg_cost, 2),
        "tp": cm["tp"],
        "fp": cm["fp"],
        "tn": cm["tn"],
        "fn": cm["fn"]
    }

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0
) -> Dict[str, Any]:
    """
    Search grid of probability thresholds [0.05, 0.95] to minimize total business cost.
    Must be called on training / validation set, NEVER on the test set.
    """
    best_threshold = 0.5
    min_cost = float("inf")
    best_metrics = None

    thresholds = np.linspace(0.05, 0.95, 91)
    curve = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        res = compute_business_cost(y_true, y_pred, cost_fn, cost_fp)
        cost = res["total_cost"]
        curve.append({
            "threshold": round(float(t), 2),
            "total_cost": cost,
            "avg_cost": res["avg_cost_per_customer"]
        })
        if cost < min_cost:
            min_cost = cost
            best_threshold = float(t)
            best_metrics = res

    return {
        "optimal_threshold": round(best_threshold, 2),
        "min_total_cost": min_cost,
        "min_avg_cost": best_metrics["avg_cost_per_customer"],
        "metrics_at_optimal": best_metrics,
        "curve_sample": curve[::10]
    }

def run_cost_sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp_base: float = 50.0,
    cost_ratios: List[float] = [2.0, 5.0, 10.0, 15.0, 20.0],
    output_dir: Path = Path("reports/cost")
) -> Dict[str, Any]:
    """
    Sensitivity analysis: evaluate how optimal threshold varies across different cost ratios (C_FN / C_FP).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for ratio in cost_ratios:
        cost_fn = cost_fp_base * ratio
        opt = find_optimal_threshold(y_true, y_prob, cost_fn=cost_fn, cost_fp=cost_fp_base)
        results.append({
            "cost_ratio_fn_to_fp": ratio,
            "cost_fn": cost_fn,
            "cost_fp": cost_fp_base,
            "optimal_threshold": opt["optimal_threshold"],
            "min_total_cost": opt["min_total_cost"],
            "min_avg_cost": opt["min_avg_cost"]
        })

    # Generate markdown report
    md_lines = [
        "# Rapport d'Analyse des Coûts Métier et Sensibilité du Seuil",
        "",
        "## 1. Hypothèses Métier et Définition de la Fonction de Coût",
        "",
        "- **Faux Négatif (FN)** : Le modèle prédit qu'un client reste fidèle, mais celui-ci résilie. L'entreprise subit la perte nette de la valeur client (LTV) / revenus futurs estimés.",
        "- **Faux Positif (FP)** : Le modèle prédit une résiliation alors que le client reste fidèle. L'entreprise engage inutilement un coût de campagne de rétention (offre commerciale, remise, temps d'appel téléconseiller).",
        r"- **Formule de Coût Total** : $\text{Coût Total} = FN \times C_{FN} + FP \times C_{FP}$",
        r"- **Hypothèse de base** : $C_{FN} = 500\$$ et $C_{FP} = 50\$$ (ratio 10:1).",
        "",
        "## 2. Analyse de Sensibilité du Seuil selon le Ratio de Coûts",
        "",
        "| Ratio $C_{FN} / C_{FP}$ | Coût FN ($) | Coût FP ($) | Seuil Optimal Détecté | Coût Moyen / Client ($) |",
        "|---|---|---|---|---|"
    ]

    for r in results:
        md_lines.append(
            f"| {r['cost_ratio_fn_to_fp']}x | {r['cost_fn']}\\$ | {r['cost_fp']}\\$ | **{r['optimal_threshold']}** | {r['min_avg_cost']}\\$ |"
        )

    md_lines.extend([
        "",
        "## 3. Interprétation Métier",
        "",
        "1. Lorsque le coût d'un churn manqué ($C_{FN}$) est beaucoup plus élevé que le coût de rétention ($C_{FP}$), le seuil optimal de décision diminue significativement (en dessous de 0.50).",
        "2. Un seuil plus bas augmente le **Recall** sur les churners, capturant davantage de résiliations au prix de quelques faux positifs peu onéreux.",
        "3. Cela prouve rigoureusement que le seuil standard de 0.50 n'est pas optimal dans un contexte économique asymétrique.",
        ""
    ])

    report_path = output_dir / "cost_analysis_report.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")

    json_path = output_dir / "sensitivity_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return {"base_results": results}
