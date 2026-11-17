# Document de Relais (Handoff) — De l'Étudiante A vers l'Étudiant B

**Projet** : Customer Churn Prediction — Plateforme MLOps
**Auteure** : Étudiante A (Senior ML / MLOps Engineer)
**Destinataire** : Étudiant B (Ingénieur Backend & Production MLOps)
**Tag Git de Référence** : `handoff-A`

---

## 1. État d'Avancement et Ce qui a été Réalisé

La **Partie A** est entièrement achevée, testée, reproductible et validée par une suite de 25 tests automatisés avec 100% de réussite.

| Tâche | Statut | Résumé de Réalisation |
|---|---|---|
| **A1 — Squelette & Ingestion** | **TERMINÉ ET TESTÉ** | Dépôt initialisé, splits stratifiés reproductibles (seed=42) `Train` (70%), `Test` (15%), `Stream` (15%) avec séparation des features ($X$) et labels ($y$). `MANIFEST.json` avec hashes SHA-256 calculés. Makefile et Runner Python opérationnels. CI GitHub Actions configurée. |
| **A2 — Validation Pandera** | **TERMINÉ ET TESTÉ** | Schémas stricts Pandera (`feature_schema`, `target_schema`), contrôles de cohérence métier, tests de corruption volontaire prouvant la détection d'anomalies, export du contrat `schema.json`. |
| **A3 — Entraînement & MLflow** | **TERMINÉ ET TESTÉ** | Pipelines Scikit-learn avec `ColumnTransformer` (zéro fuite), baselines naïves, Logistic Regression, Random Forest. Traçabilité MLflow complète (`sqlite:///mlflow.db`). Fonction de coût métier asymétrique ($C_{FN}=500\$, C_{FP}=50\$$) et optimisation du seuil de décision. |
| **A4 — Quality Gate** | **TERMINÉ ET TESTÉ** | Quality Gate automatisé à règles explicites (validation, baselines, seuils min, comparaison champion). Sorties structurées JSON dans `reports/gate/`. |
| **A5 — Registre, Promotion & Rollback** | **TERMINÉ ET TESTÉ** | Gestion sous MLflow Model Registry (`churn-model`) avec alias `champion` et `previous`. Procédure de rollback testée et validée. |
| **A6 — Feedback Différé & Réentraînement** | **TERMINÉ ET TESTÉ** | Rejeu simulé hors-ligne sur le split stream, évaluation des métriques constatées, réentraînement sur données cumulées (v2), passage au Quality Gate et promotion/rejet. |
| **A7 — Analyse d'Équité** | **TERMINÉ ET TESTÉ** | Audit de biais sur `gender` et `SeniorCitizen`. Calculs de parité, FPR, FNR, graphiques et documentation des limites éthiques. |
| **A8 — Documentation & Relais** | **TERMINÉ ET TESTÉ** | Dictionnaire de données, guides d'exécution, rapport d'audit, simulation de prise en main en environnement vierge et création du tag `handoff-A`. |

---

## 2. Guide d'Installation Rapide pour l'Étudiant B

Pour cloner et initialiser le projet sur ta machine :

```bash
# 1. Cloner le dépôt et se placer sur le tag du livrable
git clone https://github.com/Hafsaabbar/mlops-customer-churn.git
cd mlops-customer-churn
git checkout handoff-A

# 2. Créer et activer l'environnement virtuel Python
python -m venv .venv
# Sous Windows :
.venv\Scripts\Activate.ps1
# Sous Linux / macOS :
source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Reconstruire la chaîne initiale et le registre local
make all
# ou sur Windows sans make :
python scripts/run.py all

# 5. Lancer la suite de tests pour vérifier l'intégrité
make test
# ou sur Windows :
python scripts/run.py test
```

---

## 3. Contrat d'Interface et Instructions pour la Partie B (API FastAPI & Docker)

Pour développer l'API d'inférence en Partie B, tu n'as pas besoin de deviner les colonnes ou de ré-entraîner les modèles. Tous les contrats et artefacts sont prêts :

### 3.1 Contrat des Caractéristiques d'Entrée : `schema.json`
Consulte le fichier [schema.json](file:///c:/Users/HP/mlops project/schema.json).
Il décrit rigoureusement les **19 caractéristiques prédictives obligatoires** et l'identifiant facultatif `customerID`.
La variable cible `Churn` en est **explicitement exclue**.

Exemple d'objet JSON attendu par ton futur endpoint `POST /predict` :

```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

### 3.2 Modèle Champion Autonome
Un modèle sérialisé autonome est disponible dans [artifacts/champion_model.joblib](file:///c:/Users/HP/mlops project/artifacts/champion_model.joblib).
Il intègre à la fois le prétraitement scikit-learn (`ColumnTransformer`) et le classifieur.
Tu peux le charger dans ton application FastAPI directement :

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/champion_model.joblib")

# Prédiction de probabilité :
# X_input est un DataFrame pandas contenant les 19 ou 20 colonnes du contrat
churn_prob = model.predict_proba(X_input)[:, 1]

# Seuil de décision optimal recommandé (issu de artifacts/model_metadata.json) :
THRESHOLD = 0.20  # Optimisé pour limiter les churns manqués (FN)
prediction = "Yes" if churn_prob[0] >= THRESHOLD else "No"
```

### 3.3 Utilisation via MLflow Model Registry
Si ton service souhaite interroger directement le registre MLflow local :
```python
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
champion_model = mlflow.sklearn.load_model("models:/churn-model@champion")
```

---

## 4. Emplacements des Livrables et Preuves

- **Données et splits** : `data/splits/train/`, `data/splits/test/`, `data/splits/stream/`
- **Empreintes cryptographiques** : [MANIFEST.json](file:///c:/Users/HP/mlops project/MANIFEST.json)
- **Contrat API** : [schema.json](file:///c:/Users/HP/mlops project/schema.json)
- **Modèle et métadonnées** : `artifacts/champion_model.joblib` et `artifacts/model_metadata.json`
- **Rapports de validation** : [reports/validation/validation_report.md](file:///c:/Users/HP/mlops project/reports/validation/validation_report.md)
- **Rapports d'entraînement** : [reports/training/training_report.md](file:///c:/Users/HP/mlops project/reports/training/training_report.md)
- **Décisions du Quality Gate** : [reports/gate/quality_gate_result.json](file:///c:/Users/HP/mlops project/reports/gate/quality_gate_result.json)
- **Historique de promotion et rollback** : `reports/gate/promotion_history.json` et `reports/gate/rollback_history.json`
- **Analyse des coûts métier** : [reports/cost/cost_analysis_report.md](file:///c:/Users/HP/mlops project/reports/cost/cost_analysis_report.md)
- **Audit d'équité** : [reports/fairness/fairness_analysis_report.md](file:///c:/Users/HP/mlops project/reports/fairness/fairness_analysis_report.md)
- **Suite de tests** : `tests/data/`, `tests/model/`, `tests/lifecycle/`
- **CI Workflow** : `.github/workflows/ci.yml`

---

## 5. Limites Connues et Éléments Simulés

1. **Rejeu Hors-Ligne du Stream (A6)** :
   Le split `stream` provient du dataset initial IBM Telco. Les étiquettes cibles ont été tenues à l'écart jusqu'au scénario de feedback. Il ne s'agit pas de vraies données temps-réel de production, mais d'une simulation industrielle fidèle de l'arrivée différée des factures.
2. **Attribut `SeniorCitizen` (A7)** :
   L'attribut est une variable binaire simplifiée présente dans le dataset d'origine. Les conclusions d'équité doivent tenir compte du déséquilibre d'effectif (~16% de seniors).
3. **Registre MLflow Local** :
   Le registre fonctionne sur base SQLite locale (`mlflow.db`). Il ne dépend d'aucun cloud tiers et est reconstructible sur toute machine en lançant `make all`.
