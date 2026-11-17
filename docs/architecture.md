# Architecture Système et Flux de Données — Partie A

Ce document décrit l'architecture modulaire et les flux de données composant la Partie A de la plateforme MLOps Customer Churn.

---

## 1. Diagramme d'Architecture MLOps

```mermaid
flowchart TD
    subgraph Ingestion_Splitting["1. Ingestion & Découpage Stratifié"]
        Raw["data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"] --> Clean["src/data/prepare.py"]
        Clean --> TrainSplit["data/splits/train/ (X_train, y_train - 70%)"]
        Clean --> TestSplit["data/splits/test/ (X_test, y_test - 15%)"]
        Clean --> StreamSplit["data/splits/stream/ (X_stream, y_stream - 15%)"]
        Clean --> Manifest["MANIFEST.json (SHA-256 Hashes)"]
    end

    subgraph Validation["2. Validation des Données (Pandera)"]
        TrainSplit --> ValEngine["src/validation/validate.py"]
        TestSplit --> ValEngine
        StreamSplit --> ValEngine
        ValEngine --> ValRep["reports/validation/validation_report.json"]
        ValEngine --> SchemaContract["schema.json (Contrat FastAPI)"]
    end

    subgraph Training_Tracking["3. Entraînement & Suivi MLflow"]
        TrainSplit --> Trainer["src/training/train.py"]
        Trainer --> Preproc["ColumnTransformer (Zero Leakage)"]
        Preproc --> Baselines["Baselines (Always No, Always Yes)"]
        Preproc --> Models["Modèles ML (LogisticRegression, RandomForest)"]
        Trainer --> CostEngine["src/training/cost.py (Optimisation Seuil)"]
        Trainer --> MLflowTrack["MLflow Tracking (sqlite:///mlflow.db)"]
        Trainer --> TrainingRep["reports/training/training_report.md"]
        Trainer --> CostRep["reports/cost/cost_analysis_report.md"]
    end

    subgraph Governance["4. Quality Gate & Cycle de Vie (MLflow Registry)"]
        Trainer --> Candidate["Modèle Candidat v1"]
        Candidate --> Gate["src/lifecycle/gate.py (Quality Gate)"]
        ValRep -.-> Gate
        Gate --> GateRep["reports/gate/quality_gate_result.json"]
        Gate -->|Si ACCEPTÉ| Registry["MLflow Registry 'churn-model'"]
        Registry --> ChampionAlias["Alias: champion"]
        Registry --> PreviousAlias["Alias: previous"]
        Registry --> RollbackEngine["src/lifecycle/registry.py (Rollback)"]
    end

    subgraph Feedback_Loop["5. Feedback Différé & Réentraînement"]
        StreamSplit --> FeedbackEngine["src/lifecycle/feedback.py"]
        ChampionAlias -.-> FeedbackEngine
        FeedbackEngine --> FeedbackRep["reports/training/stream_feedback_report.md"]
        FeedbackEngine --> Retrainer["Réentraînement (Train + Stream)"]
        Retrainer --> CandidateV2["Modèle Candidat v2"]
        CandidateV2 --> Gate
    end

    subgraph Fairness_Audit["6. Audit d'Équité & Parité Démographique"]
        TestSplit --> FairnessEngine["src/lifecycle/fairness.py"]
        ChampionAlias -.-> FairnessEngine
        FairnessEngine --> FairnessRep["reports/fairness/fairness_analysis_report.md"]
    end
```

---

## 2. Description des Composants

### `src/data/`
- `prepare.py` : Charge le dataset brut, nettoie les chaînes, impute `TotalCharges = 0.0` pour `tenure = 0`, découpe de manière stratifiée et reproductible (graine fixe 42), et génère le `MANIFEST.json`.
- `manifest.py` : Calcule les empreintes SHA-256 de chaque fichier et vérifie à tout moment l'intégrité du dépôt.

### `src/validation/`
- `schema.py` : Déclare les schémas stricts Pandera (`feature_schema`, `target_schema`) et exporte `schema.json` pour la future API de l'étudiant B.
- `validate.py` : Exécute les contrôles de schéma sur les trois splits et consigne le rapport JSON et Markdown.

### `src/training/`
- `pipeline.py` : Construit les pipelines Scikit-learn associant `ColumnTransformer` (StandardScaler pour numériques, OneHotEncoder pour catégorielles, exclusion automatique de `customerID`) et classifieurs.
- `cost.py` : Calcule les matrices de coûts asymétriques ($C_{FN} = 500\$, C_{FP} = 50\$$) et recherche le seuil optimal sur les données d'entraînement.
- `train.py` : Orchestre l'entraînement des baselines et modèles, enregistre les artefacts et métriques dans MLflow, et inscrit le vainqueur au registre.

### `src/lifecycle/`
- `gate.py` : Quality Gate automatique vérifiant la validité des données, le dépassement des baselines, les seuils minimaux (Recall $\ge 0.70$, F1 $\ge 0.50$, ROC-AUC $\ge 0.75$) et la comparaison au champion.
- `registry.py` : Gère les alias `champion` et `previous` dans MLflow Model Registry et orchestre la commande de `rollback`.
- `feedback.py` : Simule l'arrivée différée des étiquettes du stream, calcule les métriques de production constatées, et lance le réentraînement sur données cumulées (v2).
- `fairness.py` : Évalue les écarts de performance sur les attributs protégés `gender` et `SeniorCitizen` (calcul du FPR et du FNR).

---

## 3. Stockage des Données et Artefacts

| Chemin | Contenu | Versionné Git |
|---|---|---|
| `data/raw/` | Dataset source original | Non (protégé) |
| `data/splits/` | Splits `train`, `test`, `stream` (X et y séparés) | Oui |
| `MANIFEST.json` | Métadonnées et hashes SHA-256 | Oui |
| `schema.json` | Contrat JSON des caractéristiques d'entrée pour l'API | Oui |
| `artifacts/` | Modèle champion (.joblib) et métadonnées JSON | Oui |
| `reports/` | Rapports JSON, Markdown et graphiques générés | Oui |
| `mlflow.db` / `mlruns/` | Base de suivi d'expériences et registre de modèles local | Non (reconstructible via `make all`) |
