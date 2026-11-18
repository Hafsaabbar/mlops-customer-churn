# Documentation de l'API d'Inférence — Customer Churn Prediction

Service de scoring temps-réel et observabilité basé sur **FastAPI**, exposant le modèle champion de prédiction de churn client (Partie B).

---

## 1. Vue d'Ensemble de l'Architecture

L'API charge au démarrage (via le hook `lifespan`) les artefacts produits en Partie A :
- Le pipeline scikit-learn complet : `artifacts/champion_model.joblib` (prétraitement + modèle).
- Les métadonnées et le seuil de décision optimal : `artifacts/model_metadata.json`.
- *(Optionnel)* Le registre de modèles MLflow (`sqlite:///mlflow.db`) activable via `USE_REGISTRY=true`.

### Variables d'Environnement Configurables

| Variable | Valeur par défaut | Description |
|---|---|---|
| `MODEL_PATH` | `artifacts/champion_model.joblib` | Chemin vers le pipeline scikit-learn sérialisé. |
| `METADATA_PATH` | `artifacts/model_metadata.json` | Chemin vers le fichier JSON de métadonnées et seuil. |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | URI de la base MLflow locale. |
| `USE_REGISTRY` | `false` | Tente le chargement depuis le registre si `true` (fallback joblib garanti). |
| `PREDICTIONS_LOG_PATH` | `reports/monitoring/predictions_log.jsonl` | Chemin du fichier journal d'audit des prédictions. |

---

## 2. Contrat d'Interface des Caractéristiques d'Entrée (`schema.json`)

Le payload de prédiction attendu par `POST /predict` comporte **19 caractéristiques obligatoires** et un identifiant optionnel `customerID`.

### Spécification des Attributs

| Champ | Type | Contraintes & Valeurs Autorisées | Description |
|---|---|---|---|
| `customerID` | string | *Facultatif* (chaîne libre) | Identifiant client pseudonymisé |
| `gender` | string | `"Female"`, `"Male"` | Genre du client |
| `SeniorCitizen` | integer | `0`, `1` | Statut senior (1: Oui, 0: Non) |
| `Partner` | string | `"Yes"`, `"No"` | Présence d'un partenaire |
| `Dependents` | string | `"Yes"`, `"No"` | Personnes à charge |
| `tenure` | integer | Bornes : `0` à `72` | Ancienneté en mois |
| `PhoneService` | string | `"Yes"`, `"No"` | Abonnement téléphonique |
| `MultipleLines` | string | `"No"`, `"Yes"`, `"No phone service"` | Lignes multiples |
| `InternetService` | string | `"DSL"`, `"Fiber optic"`, `"No"` | Fournisseur d'accès internet |
| `OnlineSecurity` | string | `"No"`, `"Yes"`, `"No internet service"` | Option sécurité en ligne |
| `OnlineBackup` | string | `"No"`, `"Yes"`, `"No internet service"` | Option sauvegarde en ligne |
| `DeviceProtection` | string | `"No"`, `"Yes"`, `"No internet service"` | Protection des équipements |
| `TechSupport` | string | `"No"`, `"Yes"`, `"No internet service"` | Support technique dédié |
| `StreamingTV` | string | `"No"`, `"Yes"`, `"No internet service"` | Streaming de flux TV |
| `StreamingMovies` | string | `"No"`, `"Yes"`, `"No internet service"` | Streaming de films |
| `Contract` | string | `"Month-to-month"`, `"One year"`, `"Two year"` | Type d'engagement |
| `PaperlessBilling` | string | `"Yes"`, `"No"` | Facturation dématérialisée |
| `PaymentMethod` | string | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` | Mode de règlement |
| `MonthlyCharges` | float | Bornes : `10.0` à `150.0` | Montant mensuel facturé ($) |
| `TotalCharges` | float | Bornes : `0.0` à `10000.0` | Montant cumulé facturé ($) |

### Règles Métier Croisées (Validation Pydantic v2)

1. **Cohérence Téléphonie** : Si `PhoneService == "No"`, alors `MultipleLines` doit être `"No phone service"`.
2. **Cohérence Internet** : Si `InternetService == "No"`, alors les 6 options internet (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) doivent être `"No internet service"`.
3. **Cohérence Facturation Ancienneté Zéro** : Si `tenure == 0`, alors `TotalCharges` doit être `0.0` (tolérance $\le 0.01$).

Tout champ non prévu (`extra="forbid"`) ou non conforme déclenche un rejet automatique **422 Unprocessable Entity**.

---

## 3. Endpoints de l'API

### 3.1 `GET /health`

Sonde de disponibilité et de préparation opérationnelle (Readiness Probe).

- **200 OK** : Le modèle est correctement chargé et prêt à inférer.
  ```json
  {
    "status": "ok"
  }
  ```
- **503 Service Unavailable** : Modèle indisponible ou fichiers corrompus/absents.
  ```json
  {
    "status": "unavailable"
  }
  ```

---

### 3.2 `GET /model-info`

Fournit les métadonnées de versioning, le seuil optimal calculé hors-ligne, la version scikit-learn réellement chargée et le statut d'intégrité du registre.

- **200 OK** :
  ```json
  {
    "model_name": "random_forest",
    "model_version": "2",
    "registry_version": "2",
    "run_id": "32b1e0ce5859482f9d12058cf47fa1f9",
    "threshold": 0.38,
    "source": "joblib",
    "registry_checked": false,
    "registry_consistent": null,
    "sklearn_version": "1.9.1",
    "metrics_optimal": {
      "threshold": 0.38,
      "precision": 0.4749,
      "recall": 0.8786,
      "f1": 0.6165,
      "roc_auc": 0.8416,
      "pr_auc": 0.6366,
      "tp": 246,
      "fp": 272,
      "tn": 504,
      "fn": 34,
      "total_cost": 30600.0,
      "avg_cost_per_customer": 28.98
    },
    "schema_version": "1.0.0"
  }
  ```
  *Note* : Si `USE_REGISTRY` est désactivé (`false`), le registre n'est pas interrogé (`registry_checked: false`) et `registry_consistent` vaut `null`. Si `USE_REGISTRY` est activé (`true`), `registry_checked: true` et `registry_consistent` est évalué à un booléen strict (`true` si l'alias champion correspond au `run_id`, `false` en cas de divergence ou d'indisponibilité).
  ```
- **503 Service Unavailable** : Modèle non chargé.

---

### 3.3 `POST /predict`

Calcule la probabilité de churn $P(\text{Churn}=\text{"Yes"})$, applique le seuil optimal issu des métadonnées ($P \ge \text{seuil} \implies \text{"Yes"}$), journalise la prédiction et incrémente les métriques Prometheus.

- **Codes de Retour** :
  - **200 OK** : Prédiction calculée avec succès.
  - **422 Unprocessable Entity** : Payload invalide (borne dépassée, type erroné, champ manquant, règle croisée violée).
  - **503 Service Unavailable** : Modèle indisponible (aucune prédiction n'est émise).

#### Exemple de Réponse (200 OK) :
```json
{
  "churn_probability": 0.6282,
  "prediction": "Yes",
  "threshold": 0.38,
  "model_version": "2",
  "model_source": "joblib"
}
```

---

### 3.4 `GET /metrics`

Expose les métriques au format texte Prometheus standard :
- `http_requests_total{endpoint, method, status_code}` : Compteur de requêtes.
- `http_request_duration_seconds` : Histogramme de latence par endpoint.
- `predictions_total{predicted_class}` : Compteur de décisions de churn émises (`Yes` / `No`).
- `model_ready` : Jauge d'état du modèle (1 si prêt, 0 sinon).
- `model_info{version}` : Jauge de version active.

---

## 4. Exemples de Requêtes

### 4.1 Exemple avec PowerShell (`Invoke-RestMethod`)

#### Vérification de santé :
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

#### Consultation des informations du modèle :
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/model-info" -Method Get
```

#### Prédiction valide (Client 7590-VHVEG) :
```powershell
$body = @{
    customerID = "7590-VHVEG"
    gender = "Female"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "No"
    tenure = 1
    PhoneService = "No"
    MultipleLines = "No phone service"
    InternetService = "DSL"
    OnlineSecurity = "No"
    OnlineBackup = "Yes"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "No"
    StreamingMovies = "No"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 29.85
    TotalCharges = 29.85
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

#### Cas d'erreur 422 (Exemple tenure négative) :
```powershell
$badBody = @{
    gender = "Female"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "No"
    tenure = -3
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "DSL"
    OnlineSecurity = "No"
    OnlineBackup = "No"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "No"
    StreamingMovies = "No"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    MonthlyCharges = 29.85
    TotalCharges = 29.85
} | ConvertTo-Json

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $badBody
} catch {
    Write-Host "Code HTTP :" $_.Exception.Response.StatusCode.value__
}
```

---

### 4.2 Exemple avec `curl` (Linux / macOS / Bash)

#### Prédiction valide :
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Consultation des métriques :
```bash
curl -s "http://127.0.0.1:8000/metrics" | grep -E "predictions_total|model_ready"
```

---

## 5. Journalisation et Traçabilité

Toute prédiction réussie est auditée dans `reports/monitoring/predictions_log.jsonl` sous la forme :
```json
{
  "timestamp": "2026-11-18T22:00:00.000000+00:00",
  "customer_id": "a1b2c3d4e5f67890",
  "model_version": "2",
  "probability": 0.6282,
  "prediction": "Yes",
  "features": {
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
}
```
L'identifiant `customer_id` y est systématiquement pseudonymisé par les 16 premiers caractères hexadécimaux de son empreinte SHA-256 (ou `null` s'il n'était pas fourni).
