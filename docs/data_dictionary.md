# Dictionnaire des Données — IBM Telco Customer Churn

Ce dictionnaire détaille les 21 colonnes du dataset officiel IBM Telco Customer Churn, validé après inspection réelle.

| # | Colonne | Type Métier | Type Pandas | Obligatoire | Modalités / Bornes | Description |
|---|---|---|---|---|---|---|
| 1 | `customerID` | Identifiant | object (string) | Oui | Format alphanumérique unique (ex: `7590-VHVEG`) | Identifiant client unique |
| 2 | `gender` | Catégorielle binaire | object | Oui | `Female`, `Male` | Genre du client |
| 3 | `SeniorCitizen` | Catégorielle binaire | int64 | Oui | `0` (Non), `1` (Oui) | Indique si le client est une personne âgée (≥ 65 ans) |
| 4 | `Partner` | Catégorielle binaire | object | Oui | `Yes`, `No` | Indique si le client vit en couple / marié |
| 5 | `Dependents` | Catégorielle binaire | object | Oui | `Yes`, `No` | Indique si le client a des personnes à charge |
| 6 | `tenure` | Numérique discrète | int64 | Oui | Min: 0, Max: 72 | Nombre de mois d'ancienneté du client |
| 7 | `PhoneService` | Catégorielle binaire | object | Oui | `Yes`, `No` | Indique si le client souscrit au service téléphonique |
| 8 | `MultipleLines` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No phone service` | Indique si le client a plusieurs lignes téléphoniques |
| 9 | `InternetService` | Catégorielle nominale | object | Oui | `DSL`, `Fiber optic`, `No` | Fournisseur de service Internet du client |
| 10 | `OnlineSecurity` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement à la sécurité en ligne |
| 11 | `OnlineBackup` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement à la sauvegarde en ligne |
| 12 | `DeviceProtection` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement à la protection des appareils |
| 13 | `TechSupport` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement au support technique dédié |
| 14 | `StreamingTV` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement au streaming TV |
| 15 | `StreamingMovies` | Catégorielle nominale | object | Oui | `No`, `Yes`, `No internet service` | Abonnement au streaming films |
| 16 | `Contract` | Catégorielle ordinale | object | Oui | `Month-to-month`, `One year`, `Two year` | Type d'engagement contractuel |
| 17 | `PaperlessBilling` | Catégorielle binaire | object | Oui | `Yes`, `No` | Facturation électronique sans papier |
| 18 | `PaymentMethod` | Catégorielle nominale | object | Oui | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Mode de paiement |
| 19 | `MonthlyCharges` | Numérique continue | float64 | Oui | Min: 18.25, Max: 118.75 | Montant facturé mensuellement ($) |
| 20 | `TotalCharges` | Numérique continue | float64 / object | Oui | Min: 0.0, Max: 8684.80 | Montant total facturé depuis la souscription. Attention : 11 valeurs vides `' '` pour `tenure=0`. |
| 21 | `Churn` | Cible binaire | object | Oui | `Yes` (Résilié), `No` (Fidèle) | Variable cible de prédiction |

## Conditions de Licence et Redistribution
- **Source officielle** : Kaggle / IBM Telco Customer Churn (`blastchar/telco-customer-churn`).
- **Licence** : Données publiées à des fins éducatives et démonstratives (CC0 / Apache 2.0 IBM Samples).
- **Politique de confidentialité et intégrité** : Aucune donnée personnelle réelle (PII) n'est présente ; les identifiants sont pseudonymisés (`customerID`).
- **Pratique MLOps retenue** : Pour respecter la traçabilité et ne pas altérer les données sources, le fichier brut est préservé intact dans `data/raw/` et exclu de la redistribution publique brute non autorisée. Les découpages stratifiés reproductibles et les schémas de validation sont versionnés avec leurs empreintes cryptographiques SHA-256.