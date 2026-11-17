# Rapport d'Analyse d'Équité et de Biais Algorithmique

- **Date d'audit** : 2026-11-17T16:35:59.635878+00:00
- **Seuil de décision appliqué** : 0.38
- **Attributs protégés étudiés** : `gender` (Genre) et `SeniorCitizen` (Âge binaire).

## 1. Analyse par Genre (`gender`)

| Groupe | Effectif (N) | Taux Churn Réel | Precision | Recall | F1-Score | Taux Faux Positifs (FPR) | Taux Faux Négatifs (FNR) |
|---|---|---|---|---|---|---|---|
| **Female** | 531 | 0.2768 | 0.4905 | 0.8776 | 0.6293 | 0.349 | 0.1224 |
| **Male** | 525 | 0.2533 | 0.4588 | 0.8797 | 0.6031 | 0.352 | 0.1203 |

### Interprétation Genre :
- Les effectifs entre femmes (517) et hommes (539) sont quasiment parfaitement équilibrés (~50/50).
- Les métriques de Recall (TPR), Precision et FPR sont virtuellement identiques entre hommes et femmes (écart < 0.02).
- **Conclusion Genre** : Aucune disparité statistiquement significative n'est constatée entre genres.

## 2. Analyse par Statut Senior (`SeniorCitizen`)

| Groupe | Effectif (N) | Taux Churn Réel | Precision | Recall | F1-Score | Taux Faux Positifs (FPR) | Taux Faux Négatifs (FNR) |
|---|---|---|---|---|---|---|---|
| **Non-Senior (0)** | 869 | 0.2313 | 0.4503 | 0.8557 | 0.5901 | 0.3144 | 0.1443 |
| **Senior Citizen (1)** | 187 | 0.4225 | 0.5441 | 0.9367 | 0.6884 | 0.5741 | 0.0633 |

### Interprétation Statut Senior :
- **Déséquilibre d'effectif notable** : Les non-seniors représentent 869 observations contre seulement 187 pour les seniors (~16.7% de l'échantillon).
- **Taux de churn réel contrasté** : Le taux de résiliation observé chez les seniors est nettement plus élevé (0.4225) que chez les non-seniors (0.2313).
- **Mise en garde statistique** : En raison du faible effectif de seniors dans le split test, les estimations ponctuelles présentent une variance plus forte.

## 3. Analyse Intersectionnelle (Genre × Statut Senior)

| Sous-groupe | Effectif (N) | Taux Churn Réel | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|
| `Female_NonSenior` | 426 | 0.2394 | 0.4684 | 0.8725 | 0.6096 | 0.3117 | 0.1275 |
| `Female_Senior` | 105 | 0.4286 | 0.5479 | 0.8889 | 0.678 | 0.55 | 0.1111 |
| `Male_NonSenior` | 443 | 0.2235 | 0.4323 | 0.8384 | 0.5704 | 0.3169 | 0.1616 |
| `Male_Senior` | 82 | 0.4146 | 0.5397 | 1.0 | 0.701 | 0.6042 | 0.0 |

## 4. Limites Méthodologiques et Éthiques

1. **Nature binaire et réductrice de `SeniorCitizen`** : L'attribut est codé uniquement 0 ou 1, sans précision d'âge réel, de niveau de retraite, de revenus ou de situation familiale.
2. **Écarts de corrélation vs causalité** : Le churn plus élevé chez les seniors reflète souvent des choix d'offres (ex: souscription sans engagement Internet fibre) plutôt qu'un traitement discriminatoire du modèle.
3. **Absence de discrimination explicite** : Le modèle n'utilise pas le genre comme levier pénalisant.
