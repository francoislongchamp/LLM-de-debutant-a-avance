# Module 43 — Lire une courbe et retrouver les données qui l’expliquent

## 1. Ce que tu vas apprendre

Une loss qui baisse ne dit pas seule pourquoi le modèle progresse. Un pic, un plateau ou un changement de masque peut modifier la courbe sans avoir la même signification.

Tu vas conserver la série brute, calculer un lissage causal et relier les anomalies à un batch, un LR et un compteur de cibles.

**Prérequis :** modules 13, 14, 22 et 42.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python uniquement pour l’analyse ; aucun entraînement.

## 2. Définir les axes avant la forme

| Axe horizontal | Utile pour | Attention |
| --- | --- | --- |
| Optimizer steps | Dynamique des updates | Un batch logique peut changer de taille |
| Cibles/tokens traités | Budget de données exposées | Convention et répétitions à annoncer |
| Temps réel | Coût de l’expérience | Inclut attentes, évaluation et matériel |
| Epochs | Parcours d’un corpus fixe | Peu comparable entre mixtures ou corpus différents |

Sur l’axe vertical, précise la loss, les cibles et la moyenne. Une nouvelle règle assistant-only peut réduire la loss affichée parce que l’on mesure autre chose, sans démontrer une amélioration instantanée.

## 3. Reconnaître des formes, pas inventer une cause

| Forme observée | Hypothèses de départ | Vérification |
| --- | --- | --- |
| Plateau | LR, capacité, données ou convergence | LR effectif, test un batch, validation |
| Pic isolé puis retour | Batch difficile, longueur, corruption ou instabilité passagère | IDs, source, labels et norme |
| Hausse durable | Dérive numérique ou updates inadaptées | Premier point anormal et état précédent |
| Train baisse, validation remonte | Suradaptation ou protocole différent | Splits, objectifs et checkpoints |
| Rupture au changement de source | Mixture ou difficulté modifiée | Source et compteurs par domaine |

Une corrélation temporelle aide à choisir quoi inspecter ; elle ne prouve pas la causalité. Un pic de grad_norm peut aussi dépendre de l’échelle de la loss ou du nombre de cibles.

## 4. Lisser sans effacer les faits

Une moyenne glissante sur les k dernières observations réduit le bruit visuel. Elle retarde aussi la lecture des changements et peut cacher des pics courts.

Conserve la série brute et indique fenêtre/convention. Une moyenne non pondérée de losses de batches n’est pas forcément la loss moyenne par cible si les effectifs varient.

Pour une courbe de corpus pondérée, conserve sommes NLL et nombres de cibles puis agrège leurs sommes. Le lissage est une aide de visualisation, pas une nouvelle preuve de qualité.

## 5. Laboratoire autonome — Retrouver un pic et son batch

Les données sont fictives et leur metadata fait partie de l’exercice. L’heuristique compare la loss courante à la moyenne des trois précédentes ; elle ne diagnostique pas automatiquement un bug.

```python
import statistics

losses = [9.7, 8.2, 7.3, 6.0, 5.4, 8.9, 5.1, 4.9]
rows = [{'step': i, 'loss': loss, 'lr': 0.001,
         'batch_id': f'batch-{i:02d}', 'targets': 100,
         'grad_norm': 20.0 if i == 5 else 1.0}
        for i, loss in enumerate(losses)]
window = 3
flagged = []
for index, row in enumerate(rows):
    recent = rows[max(0, index - window + 1):index + 1]
    smoothed = statistics.mean(item['loss'] for item in recent)
    previous = rows[max(0, index - window):index]
    deviation = row['loss'] - statistics.mean(item['loss'] for item in previous) if len(previous) == window else None
    if deviation is not None and deviation > 2.0:
        flagged.append(row)
    print(row['step'], 'brut=', row['loss'], 'lisse=', round(smoothed, 3),
          'ecart aux precedents=', None if deviation is None else round(deviation, 3))
print('A inspecter :', flagged)

batches = [{'mean_loss': 1.0, 'targets': 2}, {'mean_loss': 3.0, 'targets': 8}]
weighted = sum(row['mean_loss'] * row['targets'] for row in batches) / sum(row['targets'] for row in batches)
unweighted = statistics.mean(row['mean_loss'] for row in batches)
print('Moyenne par cible :', weighted, 'moyenne par batch :', unweighted)
```

Tu dois retrouver le **step 5**, `batch-05`, sa loss **8.9** et sa norme **20**. La mesure revient ensuite vers le niveau précédent : ce n’est pas la même histoire qu’une hausse persistante jusqu’à NaN.

Les deux moyennes finales valent **2.6** par cible et **2.0** par batch. Ce sont deux pondérations différentes, pas un arrondi.

## 6. Une procédure de diagnostic reproductible

1. Retrouve le premier événement suspect et les mesures brutes.
2. Vérifie configuration, code, données et éventuels changements à cet instant.
3. Rejoue un batch identifié dans des conditions contrôlées si les données peuvent être conservées.
4. Sépare effet de données, calcul numérique et update en examinant chaque étape.
5. Compare au checkpoint précédent et à la validation.
6. Change un facteur à la fois dans un pilote court.

Évite de modifier plusieurs réglages dans le run principal sans noter la rupture. Les courbes après intervention ne sont plus celles de la seule configuration initiale.

## 7. Expériences guidées

- Augmente la fenêtre de lissage à 5 et compare la visibilité du pic sans supprimer la série brute.
- Change seulement le nombre de cibles de certains points et recalcule une moyenne de corpus pondérée.
- Ajoute une colonne `source` et un changement de mixture au step 5. Décris ce qu’il faudrait tester pour attribuer le pic à la source.
- Sur le pilote 42, archive loss, LR utilisé, norme pré-clipping et cibles par update, puis compare train et validation selon des conventions compatibles.

## 8. Exercices avec corrigés

### Exercice 1 — Plateau

Une courbe plate prouve-t-elle que le modèle a atteint sa capacité maximale ?

<details>
<summary>Corrigé</summary>

Non. LR, données, masques, optimizer ou problème de pipeline peuvent aussi produire un plateau. Il faut des vérifications ciblées.

</details>

### Exercice 2 — Lissage

Pourquoi garder les points bruts si une moyenne est plus lisible ?

<details>
<summary>Corrigé</summary>

Le lissage peut cacher des anomalies brèves et déplacer visuellement leur début. Les valeurs brutes permettent de retrouver le batch et les circonstances exactes.

</details>

### Exercice 3 — Axe steps

Deux runs font 1 000 updates avec des batches effectifs différents. Ont-ils vu autant de données ?

<details>
<summary>Corrigé</summary>

Pas nécessairement. Il faut compter entrées et cibles réelles, avec la même convention, ainsi que les répétitions.

</details>

### Exercice 4 — Norme et clipping

La norme loggée vaut 20 alors que le clipping est à 1. Est-ce forcément un échec ?

<details>
<summary>Corrigé</summary>

Non. Le framework peut logger la norme avant clipping, comme `clip_grad_norm_` la renvoie. Vérifie le moment de mesure.

</details>

## 9. Validation — avant de continuer

- [ ] Je nomme axes, objectifs, masques et moyenne.
- [ ] Je distingue pic, plateau, rupture et divergence persistante.
- [ ] Je retrouve le batch suspect du laboratoire.
- [ ] Je garde série brute et lissage avec leur convention.
- [ ] Je propose une investigation plutôt qu’une cause certaine sans preuve.

## Fiche mémo

```text
Une courbe indique où regarder ; elle n’identifie pas seule la cause.
Relier chaque point aux cibles, au LR, aux gradients et aux données.
```

Le [module 44](44_perplexite_et_ses_limites.md) transforme la même NLL moyenne en perplexité.
