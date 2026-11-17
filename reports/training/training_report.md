# Rapport d'Entraînement et Comparaison des Modèles

## 1. Tableau Comparatif des Performances (Split Test)

| Modèle | Seuil | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | ROC-AUC | PR-AUC | Coût Total ($) |
|---|---|---|---|---|---|---|---|
| `baseline_always_no` | 0.05 | 0.0 | 0.0 | **0.0** | 0.5 | 0.2652 | 140000.0\$ |
| `baseline_always_yes` | 0.05 | 0.2652 | 1.0 | **0.4192** | 0.5 | 0.2652 | 38800.0\$ |
| `logistic_regression` | 0.2 | 0.3854 | 0.9607 | **0.5501** | 0.8449 | 0.6315 | 26950.0\$ |
| `random_forest` | 0.19 | 0.3816 | 0.9607 | **0.5462** | 0.8395 | 0.6304 | 27300.0\$ |

## 2. Analyse des Résultats et Baselines

- **Baseline Always No** : F1 = 0.0, Recall = 0.0. Tous les churners sont manqués ($FN=280$), engendrant le coût financier maximal ($140\,000\$$).
- **Baseline Always Yes** : Recall = 1.0 mais Precision très faible (~0.265) et coût de faux positifs élevé ($FP=776$).
- **Modèles ML (Logistic Regression et Random Forest)** : Dépassent massivement les baselines naïves sur tous les critères.
- **Modèle Champion Retenu** : `logistic_regression` avec un seuil optimisé à **0.2**.
