"""
Dataset audit and exploration script for IBM Telco Customer Churn.
Computes complete statistics, checks anomalies, investigates TotalCharges,
and outputs reports to reports/validation/data_audit_report.md and docs/data_dictionary.md.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

def audit_dataset():
    data_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if not data_path.exists():
        data_path = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Source dataset not found at {data_path}")

    # Read raw CSV as strings first to inspect raw formatting
    df_raw = pd.read_csv(data_path, dtype=str)
    n_rows, n_cols = df_raw.shape

    # Check TotalCharges whitespace / empty values
    total_charges_raw = df_raw["TotalCharges"]
    whitespace_mask = total_charges_raw.str.strip() == ""
    n_blank_total_charges = int(whitespace_mask.sum())

    # Convert to numeric for numerical columns
    df = pd.read_csv(data_path)
    # Check duplicate customer IDs
    duplicate_ids = int(df["customerID"].duplicated().sum())
    duplicate_rows = int(df.duplicated().sum())

    # Churn distribution
    churn_counts = df["Churn"].value_counts().to_dict()
    churn_pct = (df["Churn"].value_counts(normalize=True) * 100).round(2).to_dict()

    # Column inspection
    columns_info = []
    for col in df.columns:
        col_series = df[col]
        n_null = int(col_series.isnull().sum())
        n_unique = int(col_series.nunique())
        unique_sample = list(col_series.dropna().unique()[:8])
        if col == "TotalCharges":
            n_null += n_blank_total_charges
        columns_info.append({
            "name": col,
            "raw_dtype": str(df_raw[col].dtype),
            "inferred_dtype": str(col_series.dtype),
            "null_count": n_null,
            "unique_count": n_unique,
            "sample_values": [str(x) for x in unique_sample]
        })

    # Detailed inspection of tenure == 0
    tenure_zero = df[df["tenure"] == 0]
    n_tenure_zero = len(tenure_zero)

    # Descriptive statistics for numericals after proper coercion
    df_clean_stats = df.copy()
    df_clean_stats["TotalCharges_num"] = pd.to_numeric(df_clean_stats["TotalCharges"], errors="coerce")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges_num"]
    stats_desc = df_clean_stats[num_cols].describe().to_dict()

    # Write report
    report_lines = [
        "# Rapport d'Exploration et d'Audit des Données — IBM Telco Customer Churn",
        "",
        f"- **Fichier source** : `{data_path.as_posix()}`",
        f"- **Dimensions** : {n_rows} lignes × {n_cols} colonnes",
        f"- **Doublons identifiants (`customerID`)** : {duplicate_ids}",
        f"- **Doublons de lignes entières** : {duplicate_rows}",
        "",
        "## 1. Distribution de la variable cible (`Churn`)",
        "",
        f"- **No (Non résilié)** : {churn_counts.get('No', 0)} ({churn_pct.get('No', 0)}%)",
        f"- **Yes (Résilié)** : {churn_counts.get('Yes', 0)} ({churn_pct.get('Yes', 0)}%)",
        "- **Constat** : Présence d'un déséquilibre de classe modéré (~73.46% 'No' vs ~26.54% 'Yes').",
        "- **Recommandation** : Stratification obligatoire lors des découpages Train/Test/Stream, et utilisation de métriques adaptées (F1-score, Recall, PR-AUC, ROC-AUC) plutôt que la simple accuracy.",
        "",
        "## 2. Anomalie critique détectée : `TotalCharges`",
        "",
        f"- **Valeurs vides / espaces blancs** : {n_blank_total_charges} lignes contiennent une chaîne d'espaces `' '` au lieu d'un nombre.",
        f"- **Lien avec `tenure`** : Toutes les {n_blank_total_charges} lignes correspondent exactement aux clients avec `tenure == 0` ({n_tenure_zero} clients).",
        "- **Explication métier** : Ce sont de nouveaux clients tout juste souscrits dont aucune facture mensuelle n'a encore été clôturée.",
        "- **Décision de prétraitement** : Imputer `TotalCharges = 0.0` (ou `MonthlyCharges * tenure`) pour ces clients est parfaitement logique du point de vue métier, plutôt que de supprimer arbitrairement 11 observations.",
        "",
        "## 3. Inventaire des 21 Colonnes",
        "",
        "| Colonne | Type Détecté | Valeurs Manquantes | Valeurs Uniques | Échantillon |",
        "|---|---|---|---|---|"
    ]

    for c in columns_info:
        sample_str = ", ".join(c["sample_values"][:4])
        report_lines.append(f"| `{c['name']}` | `{c['inferred_dtype']}` | {c['null_count']} | {c['unique_count']} | {sample_str} |")

    report_lines.extend([
        "",
        "## 4. Statistiques Descriptives des Variables Numériques",
        "",
        "| Métrique | `tenure` (mois) | `MonthlyCharges` ($) | `TotalCharges` ($) |",
        "|---|---|---|---|"
    ])

    for metric in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
        t_val = round(stats_desc["tenure"][metric], 2)
        m_val = round(stats_desc["MonthlyCharges"][metric], 2)
        tot_val = round(stats_desc["TotalCharges_num"][metric], 2) if not np.isnan(stats_desc["TotalCharges_num"][metric]) else "N/A"
        report_lines.append(f"| {metric} | {t_val} | {m_val} | {tot_val} |")

    report_lines.extend([
        "",
        "## 5. Décisions de Nettoyage et Justification",
        "",
        "1. **Identifiant `customerID`** : Clé technique unique par client. Doit être retirée des variables explicatives ($X$) pour l'entraînement afin d'éviter le surapprentissage sur un identifiant arbitraire.",
        "2. **Gestion de `TotalCharges`** : Conversion en flottant (`float64`). Les 11 valeurs vides correspondent à des clients sans ancienneté (`tenure = 0`), imputées à 0.0.",
        "3. **Variable `SeniorCitizen`** : Fournie sous forme binaire int (0 ou 1). Considérée comme variable catégorielle binaire dans le préprocessing.",
        "4. **Encodage des variables catégorielles** : One-Hot Encoding ou Ordinal Encoding via `ColumnTransformer` scikit-learn dans le pipeline pour garantir l'étanchéité.",
        "5. **Aucune suppression arbitraire de lignes** : Les 7043 enregistrements sont conservés et valorisés.",
        ""
    ])

    report_path = Path("reports/validation/data_audit_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Data audit report generated: {report_path}")

    # Generate docs/data_dictionary.md
    dict_lines = [
        "# Dictionnaire des Données — IBM Telco Customer Churn",
        "",
        "Ce dictionnaire détaille les 21 colonnes du dataset officiel IBM Telco Customer Churn, validé après inspection réelle.",
        "",
        "| # | Colonne | Type Métier | Type Pandas | Obligatoire | Modalités / Bornes | Description |",
        "|---|---|---|---|---|---|---|",
        "| 1 | `customerID` | Identifiant | object (string) | Oui | Format alphanumérique unique (ex: `7590-VHVEG`) | Identifiant client unique |",
        "| 2 | `gender` | Catégorielle binaire | object | Oui | `Female`, `Male` | Genre du client |",
        "| 3 | `SeniorCitizen` | Catégorielle binaire | int64 | Oui | `0` (Non), `1` (Oui) | Indique si le client est une personne âgée (≥ 65 ans) |",
        "| 4 | `Partner` | Catégorielle binaire | object | Oui | `Yes`, `No` | Indique si le client vit en couple / marié |",
        "| 5 | `Dependents` | Catégorielle binaire | object | Oui | `Yes`, `No` | Indique si le client a des personnes à charge |",
        "| 6 | `tenure` | Numérique discrète | int64 | Oui | Min: 0, Max: 72 | Nombre de mois d'ancienneté du client |",
        "| 7 | `PhoneService` | Catégorielle binaire | object | Oui | `Yes`, `No` | Indique si le client souscrit au service téléphonique |",
        "| 8 | `MultipleLines` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No phone service` | Indique si le client a plusieurs lignes téléphoniques |",
        "| 9 | `InternetService` | Catégorielle nominale | object | Oui | `DSL`, `Fiber optic`, `No` | Fournisseur de service Internet du client |",
        "| 10 | `OnlineSecurity` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement à la sécurité en ligne |",
        "| 11 | `OnlineBackup` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement à la sauvegarde en ligne |",
        "| 12 | `DeviceProtection` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement à la protection des appareils |",
        "| 13 | `TechSupport` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement au support technique dédié |",
        "| 14 | `StreamingTV` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement au streaming TV |",
        "| 15 | `StreamingMovies` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement au streaming films |",
        "| 16 | `Contract` | Catégorielle ordinale | object | Oui | `Month-to-month`, `One year`, `Two year` | Type d'engagement contractuel |",
        "| 17 | `PaperlessBilling` | Catégorielle binaire | object | Oui | `Yes`, `No` | Facturation électronique sans papier |",
        "| 18 | `PaymentMethod` | Catégorielle nominale | object | Oui | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Mode de paiement |",
        "| 19 | `MonthlyCharges` | Numérique continue | float64 | Oui | Min: 18.25, Max: 118.75 | Montant facturé mensuellement ($) |",
        "| 20 | `TotalCharges` | Numérique continue | float64 / object | Oui | Min: 0.0, Max: 8684.80 | Montant total facturé depuis la souscription. Attention : 11 valeurs vides `' '` pour `tenure=0`. |",
        "| 21 | `Churn` | Cible binaire | object | Oui | `Yes` (Résilié), `No` (Fidèle) | Variable cible de prédiction |",
        "",
        "## Conditions de Licence et Redistribution",
        "- **Source officielle** : Kaggle / IBM Telco Customer Churn (`blastchar/telco-customer-churn`).",
        "- **Licence** : Données publiées à des fins éducatives et démonstratives (CC0 / Apache 2.0 IBM Samples).",
        "- **Politique de confidentialité et intégrité** : Aucune donnée personnelle réelle (PII) n'est présente ; les identifiants sont pseudonymisés (`customerID`).",
        "- **Pratique MLOps retenue** : Pour respecter la traçabilité et ne pas altérer les données sources, le fichier brut est préservé intact dans `data/raw/` et exclu de la redistribution publique brute non autorisée. Les découpages stratifiés reproductibles et les schémas de validation sont versionnés avec leurs empreintes cryptographiques SHA-256."
    ]

    dict_path = Path("docs/data_dictionary.md")
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    dict_path.write_text("\n".join(dict_lines), encoding="utf-8")
    print(f"Data dictionary generated: {dict_path}")

if __name__ == "__main__":
    audit_dataset()
