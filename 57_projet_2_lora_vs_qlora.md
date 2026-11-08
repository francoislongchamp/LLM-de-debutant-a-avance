# Module 57 — Projet 2 : mesurer LoRA et QLoRA sur la même tâche

## 1. Ce que tu vas construire

Deux conditions partent de la même base logique : l’une conserve une représentation non quantifiée, l’autre utilise une base 4 bits et des adapters LoRA.

Le projet doit produire un compromis mesuré mémoire/temps/qualité, pas répéter une économie théorique sur les seuls poids.

**Prérequis :** modules 20, 50, 54 et projet 56.

**Temps indicatif :** analyse courte sur CPU ; plusieurs heures pour des conditions réelles comparables.

**Matériel :** Python pour le laboratoire d’analyse. Le vrai chargement QLoRA exige un backend matériel/logiciel compatible ; le pilote CUDA du module 20 constitue un préalable.

## 2. Poser précisément la comparaison

Question d’exemple : « À données, cibles, rang, scaling et budget de training fixés, QLoRA permet-il de respecter une limite mémoire sans perte inacceptable de qualité ? »

Les deux méthodes peuvent avoir un coût et des dynamiques différents. Une première comparaison avec réglages contrôlés isole le changement de représentation ; une recherche de meilleure recette demande un budget de tuning comparable et annoncé.

## 3. La matrice d’expériences

| Élément | Condition A | Condition B |
| --- | --- | --- |
| Base/révision et tokenizer | Identiques | Identiques |
| Base chargée | Dtype choisi et documenté | Quantification, NF4/double quant et compute dtype documentés |
| LoRA | Rang, alpha, cibles, dropout fixés | Même convention de départ |
| Données et labels | Même version et masques | Même version et masques |
| Budget | Cibles/updates définies | Même convention |
| Mesure | Processus et phases définis | Même protocole |

Mesure aussi la base quantifiée **avant adaptation**. Tu peux ainsi distinguer la différence déjà introduite par la quantification de celle observée après training.

## 4. Laboratoire d’analyse — Des chiffres explicitement fictifs

Le script apprend à vérifier des unités et appliquer un critère de décision. Il ne charge aucun modèle et ne mesure aucun GPU. Remplace les fixtures par tes résultats archivés avant de tirer une conclusion réelle.

```python
fixtures = [
    {'condition': 'LoRA', 'peak_allocated_gib': 12.1, 'tokens_per_second': 900,
     'domain_score_percent': 84.2, 'general_score_percent': 77.0, 'adapter_mib': 65},
    {'condition': 'QLoRA', 'peak_allocated_gib': 7.4, 'tokens_per_second': 820,
     'domain_score_percent': 83.8, 'general_score_percent': 76.9, 'adapter_mib': 65},
]
base_general_percent = 80.0
memory_limit_gib = 8.0
minimum_domain_percent = 83.0
maximum_general_drop_points = 3.5

for row in fixtures:
    assert all(row[key] > 0 for key in ['peak_allocated_gib', 'tokens_per_second', 'adapter_mib'])
    assert 0 <= row['domain_score_percent'] <= 100
    assert 0 <= row['general_score_percent'] <= 100
    drop = base_general_percent - row['general_score_percent']
    accepted = (row['peak_allocated_gib'] <= memory_limit_gib
                and row['domain_score_percent'] >= minimum_domain_percent
                and drop <= maximum_general_drop_points)
    print(row['condition'], 'perte general en points=', drop, 'admissible=', accepted)

normal, quantized = fixtures
saving = 100 * (1 - quantized['peak_allocated_gib'] / normal['peak_allocated_gib'])
throughput_change = 100 * (quantized['tokens_per_second'] / normal['tokens_per_second'] - 1)
print('EXEMPLE FICTIF : economie de pic (%)=', saving)
print('EXEMPLE FICTIF : variation de debit (%)=', throughput_change)
```

Le gain mémoire fictif vaut environ **38.84 %** et le débit baisse d’environ **8.89 %**. Seule QLoRA respecte la limite fictive de 8 GiB dans cet exemple.

Ces deux nombres ne disent pas que QLoRA sera plus lent ou de même qualité sur ton matériel. Une différence de 0.4 point de domaine demande effectifs et variabilité avant de parler d’équivalence.

## 5. Réaliser les mesures réelles

1. Reprends les données, splits, baseline et benchmark du projet 56.
2. Vérifie séparément un step A et un step B avec la séquence et le micro-batch cibles.
3. Mesure après chargement, après injection, premier optimizer step et régime stabilisé.
4. Utilise des processus séparés et garde allocated/reserved, unité GiB et phases identiques.
5. Mesure cibles ou tokens réels par seconde, temps total et coût d’évaluation.
6. Choisis les checkpoints selon validation, puis évalue sur le même test réservé.
7. Archive qualité par catégorie, longueurs et erreurs, pas seulement un total.

Le processus séparé évite de garder des références de l’autre condition. `empty_cache()` n’est pas une preuve que le programme n’a plus de tenseurs vivants.

## 6. Comparaison fixe ou utilisation de la mémoire gagnée ?

Une seconde expérience peut augmenter séquence ou micro-batch grâce à QLoRA. C’est une question utile de faisabilité, mais elle change les formes et potentiellement le batch logique.

Ne mélange pas ce résultat avec l’ablation à réglages fixes. Conserve deux tableaux : effet de la quantification à conditions définies, puis meilleure utilisation du budget matériel.

À stockage d’adapter égal, base et activations peuvent encore demander des mémoires très différentes. Le poids compact de l’adapter ne mesure pas le coût du training.

## 7. Rechargement et déploiement

Teste le chargement dans les conditions où tu veux utiliser le résultat. Un adapter appris sur une base quantifiée peut être utilisé dans d’autres montages compatibles, mais changer la représentation de la base peut modifier les sorties.

Une fusion, conversion ou export demande sa propre validation numérique et comportementale. Conserve base/révision, tokenizer, configuration quantifiée et modules supplémentaires.

## 8. Livrables et conclusion attendue

Remets configurations A/B, versions matériel/logiciel, sources des données, mesures brutes horodatées ou indexées, résultats appariés, adapters et procédure de rechargement.

Le rapport doit répondre conditionnellement : quelle méthode satisfait les contraintes de mémoire, temps et qualité **mesurées ici** ? Une incompatibilité de backend est un résultat de faisabilité, pas un score nul inventé pour la méthode.

## 9. Exercices avec corrigés

### Exercice 1 — Pic réel

Pourquoi 14 GB contre 3.5 GB de poids idéaux ne donne-t-il pas forcément 75 % d’économie de pic training ?

<details>
<summary>Corrigé</summary>

Les autres postes ne suivent pas le même ratio : activations, métadonnées, états d’adapter, kernels et temporaires restent présents.

</details>

### Exercice 2 — Taille d’adapter

Pourquoi le même rang/ciblage peut-il donner des adapters de taille semblable ?

<details>
<summary>Corrigé</summary>

La base quantifiée ne change pas nécessairement les dimensions ou dtypes des A/B sauvegardées. Cela ne signifie pas même coût de base ou même qualité.

</details>

### Exercice 3 — Débit

Une économie mémoire prouve-t-elle une accélération ?

<details>
<summary>Corrigé</summary>

Non. Kernels, conversions, matériel et formes peuvent améliorer ou réduire le débit. Il faut le mesurer avec une convention temporelle annoncée.

</details>

### Exercice 4 — Séquence modifiée

QLoRA utilise un contexte deux fois plus long que LoRA. Compare-t-on seulement la quantification ?

<details>
<summary>Corrigé</summary>

Non. C’est une comparaison de recettes ou de faisabilité sous budget, à présenter séparément d’une ablation à formes fixes.

</details>

## 10. Critères de fin

- [ ] Les deux conditions sont identifiables et comparables selon la question annoncée.
- [ ] Les prérequis matériels sont vérifiés, ou la condition est marquée non exécutée.
- [ ] Pics, phases, débit et unités sont documentés.
- [ ] Baselines quantifiée/non quantifiée et résultats finaux sont distingués.
- [ ] Le rapport inclut qualité, régressions, répétitions utiles et rechargement.

## Fiche mémo

```text
Une économie mesurée et un compromis de qualité, sur un matériel et un protocole précis.
Les fixtures d’analyse ne sont jamais des résultats de training.
```

Le [projet 58](58_projet_3_lora_vs_full_fine_tuning.md) compare maintenant deux libertés de mise à jour.
