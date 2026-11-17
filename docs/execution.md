# Guide d'Exécution et Commandes — Partie A

Ce guide fournit les instructions pas-à-pas pour installer, exécuter, tester et inspecter la Partie A du projet MLOps.

---

## 1. Prérequis et Environnement

- **Python** : Version $\ge 3.10$ (testé et validé sur Python 3.11, 3.12 et 3.13).
- **Git** : Installé sur la machine.

### Création et activation de l'environnement virtuel

```bash
# Création de l'environnement virtuel
python -m venv .venv

# Activation sous Windows (PowerShell) :
.venv\Scripts\Activate.ps1

# Activation sous Linux / macOS :
source .venv/bin/activate
```

---

## 2. Commandes Standard

Le projet propose une parité stricte entre les commandes `make` (Linux/macOS) et le script multi-plateforme `python scripts/run.py` (Windows/PowerShell/CMD).

| Action | Commande Make | Commande Python / Windows |
|---|---|---|
| **Installation des dépendances** | `make install` | `python scripts/run.py install` |
| **Exécution des tests unitaires** | `make test` | `python scripts/run.py test` |
| **Préparation des découpages** | `make prepare` | `python scripts/run.py prepare` |
| **Validation Pandera** | `make validate` | `python scripts/run.py validate` |
| **Entraînement des modèles** | `make train` | `python scripts/run.py train` |
| **Quality Gate et Promotion** | `make gate` | `python scripts/run.py gate` |
| **Rejeu du Feedback Différé** | `make feedback` | `python scripts/run.py feedback` |
| **Réentraînement sur Données Accumulées** | `make retrain` | `python scripts/run.py retrain` |
| **Audit d'Équité Algorithmique** | `make fairness` | `python scripts/run.py fairness` |
| **Rollback vers Version Antérieure** | `make rollback` | `python scripts/run.py rollback` |
| **Statut du Registre MLflow** | `make status` | `python scripts/run.py status` |
| **Chaîne Complète Initiale** | `make all` | `python scripts/run.py all` |

---

## 3. Détail de la Chaîne `make all`

L'exécution de `make all` ou `python scripts/run.py all` déclenche dans l'ordre strict :

1. **Préparation des Données** :
   Nettoyage des valeurs vides de `TotalCharges`, découpage stratifié (Train 70%, Test 15%, Stream 15%), calcul et enregistrement des empreintes SHA-256 dans `MANIFEST.json`.
2. **Validation Pandera** :
   Vérification des colonnes, types, intervalles, valeurs autorisées et règles de cohérence croisées. Génération de `reports/validation/validation_report.json` et `schema.json`.
3. **Entraînement et Traçabilité MLflow** :
   Entraînement des baselines et modèles, optimisation du seuil selon la matrice de coûts ($C_{FN}=500\$, C_{FP}=50\$$), calcul des métriques et enregistrement dans MLflow SQLite.
4. **Quality Gate et Promotion** :
   Évaluation de la candidate v1 selon les règles strictes. Si acceptée, promotion sous l'alias `champion`.
5. **Audit d'Équité** :
   Calcul des métriques segmentées par genre et statut senior. Génération du rapport Markdown et graphique PNG.
6. **Statut Final** :
   Affichage des versions enregistrées et des alias actifs.

---

## 4. Visualisation de l'Interface MLflow

Pour explorer les runs, métriques, artefacts et le Model Registry dans l'interface web MLflow :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrez ensuite votre navigateur sur `http://127.0.0.1:5000`.
Vous pouvez y observer l'expérience `customer-churn-experiment` ainsi que le modèle enregistré `churn-model` avec ses alias `champion` et `previous`.
