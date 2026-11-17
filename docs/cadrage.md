# Cadrage du Projet — Customer Churn Prediction (Partie A)

## 1. Contexte Métier et Enjeux

Dans l'industrie des télécommunications, l'acquisition d'un nouveau client est estimée entre **5 et 10 fois plus coûteuse** que la rétention d'un client existant. Identifier proactivement les clients en risque de résiliation (Churn) permet à l'entreprise d'engager des actions de rétention ciblées avant que la décision de départ ne soit irréversible.

Cependant, une modélisation purement prédictive ne suffit pas : dans un environnement de production, les modèles dérivent, les données évoluent, et des retours terrain (labels réels) arrivent avec un décalage temporel de facturation. Le but de ce projet est donc de bâtir une **chaîne MLOps industrielle complète**, modulaire, reproductible, vérifiable et gouvernée.

---

## 2. Périmètre des Responsabilités

Le projet est divisé en deux parties complémentaires :

### Étudiante A (Ce livrable — Partie A)
- **A1** : Squelette, ingestion, découpage stratifié (Train, Test, Stream), intégrité cryptographique SHA-256 (`MANIFEST.json`), automatisation Makefile/Runner, CI minimale.
- **A2** : Validation rigoureuse des données via schémas Pandera, cas d'échecs sur données corrompues, contrat d'interface `schema.json`.
- **A3** : Entraînement reproductible scikit-learn (`Pipeline`, `ColumnTransformer`), baselines, MLflow Tracking, métriques ciblées classe positive (`Yes`), métrique de coût métier et optimisation du seuil de décision.
- **A4** : Quality Gate automatisé avec règles strictes et journalisation JSON (`reports/gate/`).
- **A5** : Gestion du cycle de vie sous MLflow Model Registry (`churn-model`), promotion d'alias (`champion`, `previous`) et procédure de rollback vérifiée.
- **A6** : Simulation de feedback différé hors-ligne sur le split `stream`, réentraînement avec données accumulées (`v2`), contrôle et gating.
- **A7** : Audit d'équité et parité démographique (`gender`, `SeniorCitizen`), calcul FPR/FNR et documentation des limites éthiques.
- **A8** : Documentation, interface de relais `HANDOFF.md`, simulation de prise en main en environnement isolé et tag Git `handoff-A`.

### Étudiant B (Relais ultérieur — Partie B)
- Service d'inférence en ligne FastAPI / Uvicorn basé sur le contrat `schema.json` et le modèle champion exporté.
- Conteneurisation Docker et Docker Compose.
- Instrumentation de métriques de production (Prometheus & dashboards Grafana).
- Surveillance continue du drift (Evidently).
- Endpoints de prédiction unitaire et par lots.

---

## 3. Dataset Officiel et Gouvernance

- **Nom officiel** : IBM Telco Customer Churn.
- **Source** : Kaggle / IBM Sample Data (`blastchar/telco-customer-churn`).
- **Volume** : 7043 enregistrements clients, 21 colonnes.
- **Licence et Éthique** : Données publiées sous licence d'usage éducatif / échantillon démonstratif. Toutes les données sont pseudonymisées via `customerID` ; aucune information personnelle identifiable (PII - nom, adresse, coordonnées bancaires réelles) n'est présente.
- **Sécurité et intégrité** : Le fichier source original `WA_Fn-UseC_-Telco-Customer-Churn.csv` est préservé intact sans altération. Les splits sont régis par des empreintes SHA-256 réelles calculées dans `MANIFEST.json`. Aucun secret Kaggle ni clé d'API n'est consigné dans le code.

---

## 4. Choix Techniques Structurants

1. **Étanchéité des données (Zero Data Leakage)** :
   Le prétraitement (imputation, standardisation, encodage catégoriel One-Hot) est entièrement encapsulé au sein des `Pipeline` scikit-learn. L'identifiant `customerID` est exclu des variables prédictives.
2. **Isolation du Stream pour le feedback** :
   Le split `stream` (15%) sépare rigoureusement les variables explicatives `X_stream.csv` des étiquettes cibles `y_stream.csv`. Les labels ne sont jamais révélés au modèle avant le scénario de feedback différé en A6.
3. **MLflow Model Registry local** :
   Configuration de MLflow avec un backend SQLite (`sqlite:///mlflow.db`) permettant l'exécution locale complète des fonctionnalités de Model Registry, aliases (`champion`, `previous`) et transitions, sans dépendre d'un serveur distant payant.
4. **Optimisation du Seuil guidée par le Coût Métier** :
   Dans ce cas d'usage, le coût d'un faux négatif ($C_{FN} = 500\$$ - perte nette du client) dépasse largement le coût d'un faux positif ($C_{FP} = 50\$$ - offre de rétention). Le seuil de décision est ajusté en conséquence sur l'ensemble d'entraînement pour minimiser le coût global d'exploitation.

