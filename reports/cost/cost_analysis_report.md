# Rapport d'Analyse des Coûts Métier et Sensibilité du Seuil

## 1. Hypothèses Métier et Définition de la Fonction de Coût

- **Faux Négatif (FN)** : Le modèle prédit qu'un client reste fidèle, mais celui-ci résilie. L'entreprise subit la perte nette de la valeur client (LTV) / revenus futurs estimés.
- **Faux Positif (FP)** : Le modèle prédit une résiliation alors que le client reste fidèle. L'entreprise engage inutilement un coût de campagne de rétention (offre commerciale, remise, temps d'appel téléconseiller).
- **Formule de Coût Total** : $\text{Coût Total} = FN \times C_{FN} + FP \times C_{FP}$
- **Hypothèse de base** : $C_{FN} = 500\$$ et $C_{FP} = 50\$$ (ratio 10:1).

## 2. Analyse de Sensibilité du Seuil selon le Ratio de Coûts

| Ratio $C_{FN} / C_{FP}$ | Coût FN ($) | Coût FP ($) | Seuil Optimal Détecté | Coût Moyen / Client ($) |
|---|---|---|---|---|
| 2.0x | 100.0\$ | 50.0\$ | **0.73** | 14.82\$ |
| 5.0x | 250.0\$ | 50.0\$ | **0.37** | 20.41\$ |
| 10.0x | 500.0\$ | 50.0\$ | **0.22** | 25.0\$ |
| 15.0x | 750.0\$ | 50.0\$ | **0.14** | 27.08\$ |
| 20.0x | 1000.0\$ | 50.0\$ | **0.11** | 28.27\$ |

## 3. Interprétation Métier

1. Lorsque le coût d'un churn manqué ($C_{FN}$) est beaucoup plus élevé que le coût de rétention ($C_{FP}$), le seuil optimal de décision diminue significativement (en dessous de 0.50).
2. Un seuil plus bas augmente le **Recall** sur les churners, capturant davantage de résiliations au prix de quelques faux positifs peu onéreux.
3. Cela prouve rigoureusement que le seuil standard de 0.50 n'est pas optimal dans un contexte économique asymétrique.
