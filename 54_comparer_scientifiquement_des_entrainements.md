# Module 54 — Comparer des entraînements avec contrôles et incertitude

## 1. Ce que tu vas apprendre

Un gain de score peut venir de l’intervention, d’un budget différent, d’un tirage favorable ou d’une modification du protocole. La comparaison doit annoncer sa question et ses limites.

Tu vas distinguer variations entre seeds et entre exemples, calculer un bootstrap apparié et identifier des compromis non dominés.

**Prérequis :** modules 13, 18, 19 et 53.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python uniquement ; données du laboratoire fictives.

## 2. Poser une question falsifiable

Exemple : « À budget de cibles identique, r=32 améliore-t-il assez le score domaine par rapport à r=8 pour justifier le coût, sans dépasser deux points de perte générale ? »

Fixe base/révision, données, splits, tokenizer, génération, objectif, batch et budget pertinents. Si r change le scaling, applique la règle annoncée au module 18.

Une **ablation** retire ou modifie un composant pour étudier sa contribution. Elle ne remplace pas un benchmark indépendant ni l’examen des coûts.

## 3. Deux types de comparaison légitimes

| Question | Contrôle principal |
| --- | --- |
| Effet d’un changement isolé | Autres réglages aussi constants que possible |
| Meilleure recette par méthode | Budget de tuning comparable, réglages adaptés à chaque méthode |

Utiliser le même LR partout peut aider une ablation locale, mais défavoriser artificiellement une méthode qui demande un autre ordre de grandeur. À l’inverse, tuner une méthode cent fois et l’autre une fois ne compare pas leurs meilleures performances avec le même effort.

Même nombre de steps ne signifie pas nécessairement même nombre de tokens ni même compute. Annonce le budget fixé et rapporte les autres coûts.

## 4. Seeds et échantillon d’évaluation

Des seeds de training changent initialisation, ordre et sampling. Répéter permet d’observer cette variabilité, mais trois seeds ne décrivent pas parfaitement sa distribution.

L’incertitude liée aux exemples du benchmark est une autre question. Une comparaison appariée conserve les mêmes IDs avant/après ; un bootstrap peut rééchantillonner ces paires et recalculer le gain.

Si plusieurs questions sont liées au même document, rééchantillonne le groupe indépendant approprié. Un bootstrap de lignes corrélées peut sous-estimer l’incertitude.

## 5. Laboratoire autonome — Dispersion, bootstrap et Pareto

Les scores sont fictifs. L’intervalle bootstrap illustratif dépend de l’hypothèse que les paires représentent des unités indépendantes ; il ne corrige ni biais de sélection ni contamination.

```python
import random
import statistics

scores_a = [0.781, 0.779, 0.786]
scores_b = [0.792, 0.784, 0.790]
paired_seed_deltas = [b - a for a, b in zip(scores_a, scores_b)]
print('A moyenne/ecart-type :', statistics.mean(scores_a), statistics.stdev(scores_a))
print('B moyenne/ecart-type :', statistics.mean(scores_b), statistics.stdev(scores_b))
print('Deltas seeds moyenne/ecart-type :', statistics.mean(paired_seed_deltas), statistics.stdev(paired_seed_deltas))

before = [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
after =  [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
deltas = [new - old for old, new in zip(before, after)]
rng = random.Random(42)
bootstrap = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
interval = (bootstrap[int(0.025 * len(bootstrap))], bootstrap[int(0.975 * len(bootstrap))])
print('Gain observe :', statistics.mean(deltas))
print('Intervalle bootstrap percentile approximatif :', interval)

candidates = [{'name': 'A', 'score': 0.80, 'cost': 1.0},
              {'name': 'B', 'score': 0.81, 'cost': 2.0},
              {'name': 'C', 'score': 0.79, 'cost': 1.5}]
front = []
for candidate in candidates:
    dominated = any(other['score'] >= candidate['score'] and other['cost'] <= candidate['cost']
                    and (other['score'] > candidate['score'] or other['cost'] < candidate['cost'])
                    for other in candidates)
    if not dominated:
        front.append(candidate['name'])
print('Conditions non dominees sur score/cout :', front)
```

Le front doit contenir **A et B** : C est moins bon et plus coûteux que A. A et B demandent un compromis entre score et coût.

L’écart-type imprimé n’est pas un intervalle de confiance de la moyenne. L’intervalle bootstrap du petit benchmark est illustratif et peut inclure zéro ; il ne permet pas une déclaration de supériorité universelle.

## 6. Rapporter les résultats complets

Conserve toutes les seeds prévues, échecs techniques, checkpoints évalués et critères de sélection. Ne choisis pas la meilleure seed après coup pour représenter une méthode.

Rapporte moyenne, dispersion, effectifs, résultats par catégorie, coût de training, coût de tuning et conditions matérielles. Les métriques de coût peuvent être temps, cibles traitées, pic mémoire ou stockage ; elles ne sont pas interchangeables.

Un résultat négatif reste informatif si l’expérience permet de comprendre quelle hypothèse n’a pas été soutenue.

## 7. Éviter les conclusions plus fortes que les données

Un petit gain observé avec une variation de même ordre mérite des répétitions ou une conclusion limitée. Un intervalle large n’est pas la preuve que les méthodes sont équivalentes ; il peut surtout indiquer une mesure trop peu précise.

Comparer beaucoup de variantes puis ne montrer que la gagnante crée un biais de sélection. Utilise validation pour les choix et réserve le test final pour la procédure annoncée.

Une frontier Pareto ne choisit pas le produit à ta place. Des seuils minimaux de qualité générale ou des contraintes de mémoire peuvent éliminer une condition pourtant non dominée.

## 8. Expériences guidées

1. Compare r à scaling constant, puis documente une comparaison à alpha fixe. Écris les deux questions différentes.
2. Pour CPT, dessine contrôle base→SFT et traitement base→CPT→même SFT, avec coût supplémentaire annoncé.
3. Remplace le bootstrap de lignes par des groupes liés dans une copie du benchmark et compare l’incertitude.
4. Rédige une conclusion pour un gain de 0.3 point avec écart-type entre seeds de 0.5, sans transformer cela en test statistique automatique.

## 9. Exercices avec corrigés

### Exercice 1 — Même seed

Une seed identique garantit-elle une expérience identique sur deux machines ?

<details>
<summary>Corrigé</summary>

Non. Versions, kernels, ordre des opérations et matériel peuvent différer. Il faut conserver l’environnement et préciser le niveau de reproductibilité visé.

</details>

### Exercice 2 — Bootstrap apparié

Pourquoi rééchantillonner les différences sur les mêmes IDs ?

<details>
<summary>Corrigé</summary>

Pour conserver la relation entre les deux mesures de la même question. Rééchantillonner séparément peut perdre cette structure et répondre à une autre question.

</details>

### Exercice 3 — Pareto

C est moins précis et plus coûteux que A sur toutes les mesures considérées. Est-il non dominé dans ce tableau ?

<details>
<summary>Corrigé</summary>

Non. A le domine selon ces dimensions. Une autre dimension non mesurée pourrait modifier le problème, mais il faut alors l’ajouter explicitement.

</details>

### Exercice 4 — Tuning

Pourquoi un LR distinct peut-il être légitime pour LoRA et full FT ?

<details>
<summary>Corrigé</summary>

Les paramétrisations et dynamiques diffèrent. Pour comparer des recettes adaptées, tune-les sous un protocole et un budget de recherche comparables, puis annonce ce choix.

</details>

## 10. Validation — avant de continuer

- [ ] Je formule hypothèse et variables contrôlées.
- [ ] Je distingue budget de données, compute et tuning.
- [ ] Je sépare variabilité entre seeds et entre exemples.
- [ ] Je sais ce que le bootstrap suppose et ne corrige pas.
- [ ] Je rapporte toutes les conditions et les compromis de coût.

## Fiche mémo

```text
Une question → un protocole → des données appariées → répétitions utiles → limites.
Un score gagnant n’est pas une preuve sans son coût et son incertitude.
```

Le [module 55](55_catastrophic_forgetting.md) applique cette discipline aux capacités à conserver.
