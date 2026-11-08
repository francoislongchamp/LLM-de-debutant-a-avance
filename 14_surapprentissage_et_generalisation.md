# Module 14 — Reconnaître le surapprentissage et choisir un checkpoint

## 1. Ce que tu vas apprendre

Apprendre les réponses d’un cahier ne garantit pas de réussir un nouvel exercice. De même, un modèle peut réduire sa loss train tout en devenant moins utile sur les données visées.

Tu vas lire des courbes train/validation, distinguer plusieurs hypothèses et appliquer une règle explicite d’arrêt anticipé.

**Prérequis :** modules 11 à 13 ; validation indépendante et métriques comparables.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python pour le laboratoire de courbes. Un entraînement réel est proposé en prolongement.

## 2. Définir les trois situations

| Terme | Sens concret | Exemple de signal à examiner |
| --- | --- | --- |
| Généralisation | Réussite sur de nouveaux exemples de la tâche visée | Validation et benchmark indépendants satisfaisants |
| Sous-apprentissage, underfitting | Signal utile insuffisamment appris | Train et validation restent mauvais |
| Surapprentissage, overfitting | Adaptation au train qui nuit à la généralisation visée | Train baisse pendant que validation se dégrade durablement |

Ces termes décrivent des comportements, pas une cause unique. Une mauvaise loss train peut venir d’un bug, d’un LR inadéquat, de données contradictoires, d’une capacité insuffisante ou d’un entraînement trop court.

Une hausse ponctuelle de validation peut être du bruit. Une mauvaise validation peut aussi venir d’une distribution différente ou d’un prétraitement incohérent. Lis la courbe avant de choisir un remède.

## 3. Calculer un écart, avec ses limites

Un indicateur simple est :

\[
Gap = L_{validation} - L_{train}
\]

Il est interprétable si les objectifs, masques et agrégations sont compatibles. La loss train mesurée pendant les updates et la loss validation mesurée après l’epoch ne correspondent pas toujours au même état du modèle.

Dropout, difficulté des exemples et longueur peuvent aussi contribuer à un écart. « Validation supérieure au train » ne prouve donc pas à lui seul un surapprentissage.

### Micro-exemple

| Epoch | Train | Validation |
| --- | --- | --- |
| 1 | 2.1 | 2.2 |
| 2 | 1.5 | 1.7 |
| 3 | 1.0 | 1.8 |
| 4 | 0.6 | 2.1 |

Selon la loss validation, le meilleur checkpoint observé est celui de **l’epoch 2**, même si la dernière loss train est la plus basse.

Si l’objectif produit dépend aussi du format ou des faits, vérifie le benchmark du module 13 : le checkpoint au minimum de loss n’est pas automatiquement optimal selon chaque critère.

## 4. Early stopping : définir une règle avant de regarder la fin

L’**early stopping**, ou arrêt anticipé, interrompt une expérience lorsque la validation ne s’améliore plus selon une règle.

Deux paramètres courants :

- **Patience** : nombre d’évaluations successives sans amélioration acceptées.
- **Min delta** : amélioration minimale pour remettre ce compteur à zéro.

La patience se compte en **évaluations**, pas nécessairement en epochs. Si tu évalues tous les dix steps, une patience de 3 concerne trois intervalles de dix steps.

Arrêter et sélectionner sont deux décisions : on peut s’arrêter à l’epoch 4 et recharger le checkpoint sauvegardé à l’epoch 2.

## 5. Laboratoire autonome — Appliquer la règle

Les courbes sont fictives ; le code teste une décision de suivi, sans entraîner de modèle.

```python
history = [
    {"epoch": 1, "train": 2.1, "validation": 2.2},
    {"epoch": 2, "train": 1.5, "validation": 1.7},
    {"epoch": 3, "train": 1.0, "validation": 1.8},
    {"epoch": 4, "train": 0.6, "validation": 2.1},
    {"epoch": 5, "train": 0.4, "validation": 2.3},
]
patience = 2
min_delta = 0.01
best_loss = float("inf")
best_epoch = None
bad_evaluations = 0
stop_epoch = None

for row in history:
    gap = row["validation"] - row["train"]
    print(row["epoch"], "train=", row["train"],
          "validation=", row["validation"], "gap=", round(gap, 3))
    if row["validation"] < best_loss - min_delta:
        best_loss = row["validation"]
        best_epoch = row["epoch"]
        bad_evaluations = 0
    else:
        bad_evaluations += 1
    if bad_evaluations >= patience:
        stop_epoch = row["epoch"]
        break

print("Arrêt :", stop_epoch)
print("Checkpoint à conserver selon la règle :", best_epoch)
print("Loss validation de ce checkpoint :", best_loss)
```

Tu dois obtenir **arrêt à l’epoch 4**, **checkpoint retenu à l’epoch 2**, loss **1.7**. L’epoch 5 n’a pas à être consultée pour décider cet arrêt.

Le code conserve une valeur et un numéro ; il ne sauvegarde pas de poids. Dans une vraie boucle, sauvegarde aussi le checkpoint au moment de l’amélioration, ou une copie indépendante de son état. Garder seulement une référence Python au modèle qui continue d’apprendre ne conserve pas ses anciens poids.

## 6. Choisir une action selon une hypothèse

| Observation | Hypothèse à vérifier | Expérience ciblée |
| --- | --- | --- |
| Train et validation ne baissent pas | Bug ou signal difficile à apprendre | Faire mémoriser un seul batch et inspecter labels/gradients |
| Train baisse, validation remonte durablement | Suradaptation au train | Moins de steps, checkpoint antérieur ou régularisation |
| Validation très irrégulière | Petit échantillon ou mesure instable | Examiner effectifs, catégories et mode eval |
| Domaine progresse, général régresse | Oubli de capacités antérieures | Benchmark général, adaptation moins forte ou mélange de données justifié |

L’**oubli catastrophique** concerne une perte de capacités précédemment acquises ; il peut accompagner le surapprentissage mais n’est pas son synonyme.

## 7. Ce que la régularisation peut changer

La **régularisation** limite certaines formes d’adaptation : dropout, weight decay, contraintes de capacité ou arrêt plus tôt. Plus de données pertinentes et variées peut aussi aider à apprendre un signal moins fragile.

Une augmentation de données mal contrôlée peut créer des variantes presque identiques ou des erreurs. Une baisse de LR peut ralentir l’adaptation sans corriger un dataset faux. LoRA limite la forme de la correction, mais peut lui aussi mémoriser ou dégrader les réponses.

Choisis une seule intervention selon ton hypothèse et mesure le résultat. Aucune technique ne garantit la généralisation à elle seule.

## 8. Expériences guidées

1. Dans la courbe du laboratoire, remplace l’epoch 3 par une petite amélioration inférieure à `min_delta`. Explique pourquoi le compteur peut continuer malgré un nombre légèrement plus bas.
2. Change la patience à 1, puis à 3. Distingue la date d’arrêt et le checkpoint choisi.
3. Reprends une miniature du module 12, évalue régulièrement et entraîne plus longtemps. Cherche un comportement observé sans supposer que l’overfitting se produira forcément sur ce petit essai.
4. Compare les générations du meilleur checkpoint et du dernier sur les mêmes prompts. Rapporte les cas où la loss et ta grille de qualité ne racontent pas exactement la même chose.

Ne change pas la composition du test après avoir vu les résultats. Si tu améliores la validation pour le développement, versionne cette nouvelle mesure et recommence les comparaisons concernées.

## 9. Exercices avec corrigés

### Exercice 1 — Choisir selon la validation

Train : `[2, 1, 0.5]`, validation : `[2.2, 1.8, 2.0]`. Quel checkpoint est le meilleur selon cette validation ?

<details>
<summary>Corrigé</summary>

Le deuxième, avec une validation de **1.8**. La loss train minimale du troisième ne compense pas automatiquement la dégradation mesurée sur les nouveaux exemples.

</details>

### Exercice 2 — Patience

On évalue tous les 20 steps, avec patience 2. Après une amélioration au step 40, les évaluations 60 et 80 échouent à améliorer. Quand s’arrête-t-on ?

<details>
<summary>Corrigé</summary>

Au step **80** selon cette règle. Le meilleur checkpoint reste celui de 40 s’il a bien été sauvegardé et qu’aucune autre métrique n’est utilisée pour la sélection.

</details>

### Exercice 3 — Une hypothèse trop rapide

Train et validation restent élevés. Pourquoi « ajoute du dropout » n’est-il pas un diagnostic suffisant ?

<details>
<summary>Corrigé</summary>

Il faut d’abord vérifier que le pipeline sait apprendre : labels, masques, gradients, updates, LR et test sur un batch. Ajouter une contrainte peut même rendre l’apprentissage plus difficile si le problème est ailleurs.

</details>

### Exercice 4 — LoRA protège-t-il toujours ?

Une adaptation LoRA utilise 1 % de paramètres entraînables. Peut-elle surapprendre ?

<details>
<summary>Corrigé</summary>

Oui. Un petit ensemble de paramètres peut suffire à mémoriser un petit corpus ou à modifier fortement le comportement. Le ratio n’est pas une garantie de qualité hors train.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue généralisation, sous-apprentissage, surapprentissage et oubli.
- [ ] Je lis une courbe sans diagnostiquer à partir d’un seul point.
- [ ] Je sais quand un gap est comparable et quand il ne l’est pas.
- [ ] Je retrouve arrêt 4 et checkpoint 2 dans le laboratoire.
- [ ] Je sais qu’il faut sauvegarder les poids, pas seulement leur numéro.
- [ ] Je propose une intervention et le test qui pourrait la réfuter.

## Fiche mémo

```text
Une meilleure loss train ne suffit pas à choisir le modèle final.
Validation guide la sélection ; benchmark vérifie le comportement.
Early stopping : règle explicite, patience en évaluations, meilleur état sauvegardé.
```

Le [module 15](15_peft_et_intuition_de_lora.md) introduit une correction des poids avec moins de paramètres entraînables : LoRA.
