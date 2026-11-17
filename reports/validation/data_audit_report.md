# Rapport d'Exploration et d'Audit des Données — IBM Telco Customer Churn

- **Fichier source** : `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Dimensions** : 7043 lignes × 21 colonnes
- **Doublons identifiants (`customerID`)** : 0
- **Doublons de lignes entières** : 0

## 1. Distribution de la variable cible (`Churn`)

- **No (Non résilié)** : 5174 (73.46%)
- **Yes (Résilié)** : 1869 (26.54%)
- **Constat** : Présence d'un déséquilibre de classe modéré (~73.46% 'No' vs ~26.54% 'Yes').
- **Recommandation** : Stratification obligatoire lors des découpages Train/Test/Stream, et utilisation de métriques adaptées (F1-score, Recall, PR-AUC, ROC-AUC) plutôt que la simple accuracy.

## 2. Anomalie critique détectée : `TotalCharges`

- **Valeurs vides / espaces blancs** : 11 lignes contiennent une chaîne d'espaces `' '` au lieu d'un nombre.
- **Lien avec `tenure`** : Toutes les 11 lignes correspondent exactement aux clients avec `tenure == 0` (11 clients).
- **Explication métier** : Ce sont de nouveaux clients tout juste souscrits dont aucune facture mensuelle n'a encore été clôturée.
- **Décision de prétraitement** : Imputer `TotalCharges = 0.0` (ou `MonthlyCharges * tenure`) pour ces clients est parfaitement logique du point de vue métier, plutôt que de supprimer arbitrairement 11 observations.

## 3. Inventaire des 21 Colonnes

| Colonne | Type Détecté | Valeurs Manquantes | Valeurs Uniques | Échantillon |
|---|---|---|---|---|
| `customerID` | `str` | 0 | 7043 | 7590-VHVEG, 5575-GNVDE, 3668-QPYBK, 7795-CFOCW |
| `gender` | `str` | 0 | 2 | Female, Male |
| `SeniorCitizen` | `int64` | 0 | 2 | 0, 1 |
| `Partner` | `str` | 0 | 2 | Yes, No |
| `Dependents` | `str` | 0 | 2 | No, Yes |
| `tenure` | `int64` | 0 | 73 | 1, 34, 2, 45 |
| `PhoneService` | `str` | 0 | 2 | No, Yes |
| `MultipleLines` | `str` | 0 | 3 | No phone service, No, Yes |
| `InternetService` | `str` | 0 | 3 | DSL, Fiber optic, No |
| `OnlineSecurity` | `str` | 0 | 3 | No, Yes, No internet service |
| `OnlineBackup` | `str` | 0 | 3 | Yes, No, No internet service |
| `DeviceProtection` | `str` | 0 | 3 | No, Yes, No internet service |
| `TechSupport` | `str` | 0 | 3 | No, Yes, No internet service |
| `StreamingTV` | `str` | 0 | 3 | No, Yes, No internet service |
| `StreamingMovies` | `str` | 0 | 3 | No, Yes, No internet service |
| `Contract` | `str` | 0 | 3 | Month-to-month, One year, Two year |
| `PaperlessBilling` | `str` | 0 | 2 | Yes, No |
| `PaymentMethod` | `str` | 0 | 4 | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `MonthlyCharges` | `float64` | 0 | 1585 | 29.85, 56.95, 53.85, 42.3 |
| `TotalCharges` | `str` | 11 | 6531 | 29.85, 1889.5, 108.15, 1840.75 |
| `Churn` | `str` | 0 | 2 | No, Yes |

## 4. Statistiques Descriptives des Variables Numériques

| Métrique | `tenure` (mois) | `MonthlyCharges` ($) | `TotalCharges` ($) |
|---|---|---|---|
| count | 7043.0 | 7043.0 | 7032.0 |
| mean | 32.37 | 64.76 | 2283.3 |
| std | 24.56 | 30.09 | 2266.77 |
| min | 0.0 | 18.25 | 18.8 |
| 25% | 9.0 | 35.5 | 401.45 |
| 50% | 29.0 | 70.35 | 1397.47 |
| 75% | 55.0 | 89.85 | 3794.74 |
| max | 72.0 | 118.75 | 8684.8 |

## 5. Décisions de Nettoyage et Justification

1. **Identifiant `customerID`** : Clé technique unique par client. Doit être retirée des variables explicatives ($X$) pour l'entraînement afin d'éviter le surapprentissage sur un identifiant arbitraire.
2. **Gestion de `TotalCharges`** : Conversion en flottant (`float64`). Les 11 valeurs vides correspondent à des clients sans ancienneté (`tenure = 0`), imputées à 0.0.
3. **Variable `SeniorCitizen`** : Fournie sous forme binaire int (0 ou 1). Considérée comme variable catégorielle binaire dans le préprocessing.
4. **Encodage des variables catégorielles** : One-Hot Encoding ou Ordinal Encoding via `ColumnTransformer` scikit-learn dans le pipeline pour garantir l'étanchéité.
5. **Aucune suppression arbitraire de lignes** : Les 7043 enregistrements sont conservés et valorisés.
