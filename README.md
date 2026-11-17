# MLOps Platform — Customer Churn Prediction (Partie A)

[![MLOps CI Pipeline](https://github.com/Hafsaabbar/mlops-customer-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/Hafsaabbar/mlops-customer-churn/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.18.0-blueviolet.svg)](https://mlflow.org/)
[![Pandera](https://img.shields.io/badge/Pandera-Validated-brightgreen.svg)](https://pandera.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/Pytest-25%20Passed-success.svg)](https://pytest.org/)

Plateforme MLOps industrielle pour la prédiction de la résiliation client (**Customer Churn**) dans les télécommunications, basée sur le dataset officiel **IBM Telco Customer Churn**.

Ce dépôt concrétise l'intégralité de la **Partie A** du cycle de vie MLOps (tâches **A1 à A8**), réalisée avec un niveau d'exigence professionnel : code modulaire, typé, étanche aux fuites de données, validé par schémas Pandera, suivi sous MLflow, gouverné par Quality Gate, outillé pour le rollback et prêt pour l'intégration de la Partie B (API FastAPI & Docker).

---

## 🏗️ Architecture du Projet

```text
mlops-customer-churn/
│
├── data/
│   ├── raw/                 # Dataset source (WA_Fn-UseC_-Telco-Customer-Churn.csv - protégé)
│   ├── splits/              # Splits stratifiés disjoints (Train 70%, Test 15%, Stream 15%)
│   │   ├── train/           # X_train.csv, y_train.csv (4930 lignes)
│   │   ├── test/            # X_test.csv, y_test.csv   (1056 lignes)
│   │   └── stream/          # X_stream.csv, y_stream.csv (1057 lignes - labels isolés)
│   └── processed/           # Répertoire de traitement
│
├── src/
│   ├── data/                # Ingestion, nettoyage, découpage et vérification SHA-256
│   ├── validation/          # Schémas stricts Pandera et export du contrat schema.json
│   ├── training/            # Pipelines scikit-learn, matrice de coûts et traçabilité MLflow
│   └── lifecycle/           # Quality Gate, MLflow Registry (champion/previous), feedback et équité
│
├── tests/
│   ├── data/                # Tests de découpage, intégrité manifest et validation Pandera
│   ├── model/               # Tests de prétraitement, baselines, inférence et fonctions de coût
│   └── lifecycle/           # Tests Quality Gate, registry, promotion, rollback, feedback et équité
│
├── reports/
│   ├── validation/          # Audit initial et rapport de conformité des données
│   ├── training/            # Rapports comparatifs des baselines et modèles
│   ├── gate/                # Décisions structurées du Quality Gate et historiques de transition
│   ├── cost/                # Analyse de sensibilité économique du seuil de décision
│   └── fairness/            # Audit de biais et parité démographique (gender, SeniorCitizen)
│
├── artifacts/               # Modèle champion autonome (.joblib) et métadonnées JSON
├── docs/                    # Cadrage, dictionnaire des données, architecture, exécution et HANDOFF
├── .github/workflows/       # Workflow CI GitHub Actions (validation + pytest automatisés)
├── scripts/                 # Runner multiplateforme python scripts/run.py et scripts d'audit
├── requirements.txt         # Dépendances épinglées
├── Makefile                 # Automatisation GNU Make standard
├── schema.json              # Contrat JSON d'entrée pour la future API FastAPI (Étudiant B)
├── MANIFEST.json            # Empreintes cryptographiques SHA-256 de tous les fichiers de données
└── README.md                # Documentation principale
```

---

## ⚡ Démarrage Rapide

### 1. Installation de l'environnement

```bash
# Cloner le dépôt
git clone https://github.com/Hafsaabbar/mlops-customer-churn.git
cd mlops-customer-churn

# Créer et activer l'environnement virtuel
python -m venv .venv
# Sous Windows :
.venv\Scripts\Activate.ps1
# Sous Linux / macOS :
source .venv/bin/activate

# Installer les dépendances épinglées
pip install -r requirements.txt
```

### 2. Exécution de la chaîne complète MLOps

Une seule commande lance l'intégralité du cycle initial (Données $\to$ Validation Pandera $\to$ Entraînement MLflow $\to$ Quality Gate $\to$ Registre $\to$ Équité) :

```bash
make all
# ou sur Windows sans make :
python scripts/run.py all
```

### 3. Exécution des tests automatisés

```bash
make test
# ou sur Windows :
python scripts/run.py test
```

Résultat : **25 tests unitaires et d'intégration réussis (100% pass)**.

---

## 📊 Résultats des Modèles et Baselines

Évaluation sur le jeu de test tenu à l'écart (`data/splits/test/`, 1056 clients) ciblant la classe d'intérêt **`Yes` (Churn)** :

| Modèle | Seuil | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Coût Total ($) |
|---|---|---|---|---|---|---|---|
| `baseline_always_no` | 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2652 | 140 000 $ |
| `baseline_always_yes` | 0.50 | 0.2652 | 1.0000 | 0.4192 | 0.5000 | 0.2652 | 38 800 $ |
| `random_forest` | 0.19 | 0.3816 | 0.9607 | 0.5462 | 0.8395 | 0.6304 | 27 300 $ |
| **`logistic_regression` (Champion)** | **0.20** | **0.3854** | **0.9607** | **0.5501** | **0.8449** | **0.6315** | **26 950 $** |

> [!TIP]
> **Pourquoi un seuil à 0.20 plutôt que 0.50 ?**
> Dans notre modélisation économique, manquer un client qui résilie coûte $500\$$ (perte sèche de valeur client) alors qu'engager une campagne de rétention inutile sur un client fidèle ne coûte que $50\$$. En diminuant le seuil à 0.20, le **Recall bondit à 96.1%** (seulement 11 churns manqués sur 280), faisant chuter le coût total d'exploitation de $37\,100\$$ à $26\,950\$$.

---

## 🚦 Quality Gate et Cycle de Vie (MLflow Registry)

Le modèle final est gouverné sous le registre `churn-model` avec deux alias stricts :
- **`champion`** : Modèle en service validé par le Quality Gate.
- **`previous`** : Modèle de repli immédiat pour la procédure de rollback.

### Règles du Quality Gate
1. Intégrité des données vérifiée via schémas Pandera.
2. Dépassement obligatoire de la meilleure baseline naïve ($F_1 > 0.42$).
3. Seuils minimaux garantis : Recall $\ge 0.70$, $F_1 \ge 0.50$, ROC-AUC $\ge 0.75$.
4. Performance non inférieure au champion actif.

### Commandes de cycle de vie :
```bash
# Consulter le statut des versions et alias
python scripts/run.py status

# Exécuter le rollback vers le modèle précédent
python scripts/run.py rollback

# Rejouer le feedback différé (stream)
python scripts/run.py feedback

# Déclencher le réentraînement sur données cumulées (v2)
python scripts/run.py retrain
```

---

## 🤝 Relais vers l'Étudiant B (Partie B)

Pour la prise en main par le binôme (implémentation FastAPI, Docker, Evidently, Prometheus) :
- Consultez le document dédié : [docs/HANDOFF.md](docs/HANDOFF.md).
- Contrat Pydantic / JSON des caractéristiques d'entrée : [schema.json](schema.json).
- Modèle autonome sérialisé : `artifacts/champion_model.joblib`.
- Tag Git stable du livrable : **`handoff-A`**.

