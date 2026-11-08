# Module 44 — Calculer la perplexité avec le bon dénominateur

## 1. Ce que tu vas apprendre

La perplexité, ou **PPL**, est l’exponentielle d’une NLL moyenne dans un protocole de modèle de langage causal. Elle ne constitue pas une note universelle de chat.

Tu vas agréger des losses par cible, distinguer moyenne avant/après exponentielle et comprendre l’effet des fenêtres d’évaluation.

**Prérequis :** modules 6, 42 et 43.

**Temps indicatif :** 90 à 120 minutes.

**Matériel :** Python uniquement pour les calculs et fenêtres.

## 2. De la NLL à la perplexité

Avec N cibles actives et logarithme naturel :

\[
L = \frac{1}{N}\sum_t -\ln p(x_t|contexte_t),\qquad PPL=\exp(L)
\]

| NLL moyenne | PPL environ |
| --- | --- |
| 3 | 20.0855 |
| 2 | 7.3891 |
| 1 | 2.7183 |

L’exponentielle est monotone : PPL et loss ordonnent les checkpoints de la même façon **sur le même protocole**. PPL n’ajoute pas une seconde preuve indépendante.

Pour un vocabulaire uniforme de V tokens, la NLL vaut ln(V) et PPL=V. L’interprétation comme « choix effectifs » est une intuition, pas un comptage littéral de candidats à chaque token.

## 3. Agréger avant d’exponentier

Supposons un batch de 2 cibles avec loss moyenne 1 et un batch de 8 cibles avec loss moyenne 3 :

```text
NLL globale : (2×1 + 8×3)/10 = 2.6
PPL globale : exp(2.6) ≈ 13.4637
```

Ni la moyenne simple des losses ni la moyenne des perplexités de batches ne reproduisent cette valeur.

Exclus les cibles ignorées du dénominateur. Si aucune cible n’est active, une PPL de corpus n’a pas de définition utile pour ce calcul ; il faut signaler le problème.

## 4. Laboratoire A — Retrouver trois nombres différents

```python
import math

batches = [{'loss': 1.0, 'targets': 2}, {'loss': 3.0, 'targets': 8}]
count = sum(row['targets'] for row in batches)
if count == 0:
    raise ValueError('Aucune cible')
mean_nll = sum(row['loss'] * row['targets'] for row in batches) / count
ppl = math.exp(mean_nll)
wrong_unweighted = math.exp(sum(row['loss'] for row in batches) / len(batches))
mean_batch_ppl = sum(math.exp(row['loss']) for row in batches) / len(batches)
print('NLL et PPL globales :', mean_nll, ppl)
print('Exponentielle moyenne non ponderee :', wrong_unweighted)
print('Moyenne des PPL de batches :', mean_batch_ppl)
print('PPL pour loss 1.5 :', math.exp(1.5))
```

Les valeurs attendues sont environ **13.4637**, **7.3891** et **11.4019**, puis **4.4817** pour la dernière ligne.

Si l’exponentielle déborde sur une NLL très élevée, conserve la NLL et signale le débordement au lieu de présenter un nombre artificiellement plafonné comme la vraie PPL.

## 5. Le contexte d’évaluation change la mesure

Un modèle à contexte limité ne peut pas utiliser tout le passé d’un très long document. Le découper en blocs disjoints retire du contexte à certaines cibles et peut omettre des prédictions aux frontières.

Une stratégie glissante permet de fournir davantage de contexte, mais il faut éviter de compter les mêmes cibles plusieurs fois. La PPL doit annoncer longueur de fenêtre, stride, frontières de documents et traitement des tokens spéciaux.

## 6. Laboratoire B — Une cible par fenêtre glissante

Cette version pédagogique montre les indices. Elle ne calcule pas de logits ; dans un vrai modèle, seuls les labels de la dernière position resteraient actifs pour chaque fenêtre ci-dessous.

```python
tokens = list(range(9))
context_length = 4
counted_targets = []
for end in range(2, len(tokens) + 1):
    begin = max(0, end - context_length)
    input_ids = tokens[begin:end]
    labels = [-100] * (len(input_ids) - 1) + [tokens[end - 1]]
    counted_targets.append(end - 1)
    print('positions=', (begin, end), 'entree=', input_ids, 'labels=', labels)
assert counted_targets == list(range(1, len(tokens)))
print('Cibles comptees une fois :', counted_targets)
```

Les positions **1 à 8** sont chacune comptées une fois. La position 0 ne possède pas de préfixe fourni ici. La dernière cible de chaque fenêtre est prédite par la position précédente après décalage interne.

Cette stratégie fait beaucoup de forwards pour peu de cibles. Des implémentations à stride plus grand évaluent plusieurs nouvelles cibles à la fois ; elles doivent masquer celles déjà évaluées et utiliser le dénominateur réel.

## 7. Comparabilité entre modèles

Une PPL par token dépend du tokenizer. La même chaîne découpée en dix ou vingt unités ne définit pas le même nombre de termes ni la même tâche locale.

Une mesure en bits par octet peut aider certaines comparaisons de vraisemblance sur exactement les mêmes textes et conventions, mais elle demande elle aussi des détails de segmentation, contexte et normalisation. Ne convertis pas une PPL brute en classement universel.

Pour un assistant, ajoute respect des instructions, exactitude et benchmark de régression. Une bonne modélisation de documents ne certifie pas une réponse fidèle ou une citation correcte.

## 8. Expériences guidées

1. Recalcule la NLL de validation du module 42 à partir des sommes et nombres de cibles, puis exponentie.
2. Compare fenêtres disjointes et glissantes sur le même document. Liste d’abord les cibles et leurs contextes avant de comparer les valeurs.
3. Modifie le masque de loss d’une mesure et explique pourquoi la PPL obtenue ne porte plus exactement sur la même distribution de cibles.
4. Rapporte une PPL avec corpus/révision, tokenizer, contexte, stride, masque et unité de moyenne.

## 9. Exercices avec corrigés

### Exercice 1 — Interprétation

PPL=8 signifie-t-il 8 % d’erreurs ?

<details>
<summary>Corrigé</summary>

Non. C’est exp(NLL moyenne) dans le protocole décrit, pas un pourcentage d’accuracy ou d’erreur.

</details>

### Exercice 2 — Même classement

Sur la même validation, loss A=1.2 et B=1.4 : laquelle a la PPL la plus basse ?

<details>
<summary>Corrigé</summary>

A, car l’exponentielle est croissante. Cela ne rajoute pas une métrique indépendante à ces losses.

</details>

### Exercice 3 — Fenêtre

Pourquoi deux évaluations avec contexte 32 et 128 peuvent-elles donner des PPL différentes sur les mêmes textes ?

<details>
<summary>Corrigé</summary>

Les probabilités des cibles dépendent du contexte disponible. Le nombre de cibles et le traitement des frontières peuvent aussi différer.

</details>

### Exercice 4 — Tokenizers

Une PPL plus basse avec un autre tokenizer prouve-t-elle un meilleur assistant ?

<details>
<summary>Corrigé</summary>

Non. L’unité de prédiction et le protocole ont changé ; les qualités d’assistant demandent en plus des tâches et mesures appropriées.

</details>

## 10. Validation — avant de continuer

- [ ] Je calcule exp(NLL moyenne pondérée par les cibles).
- [ ] Je distingue trois agrégations du laboratoire A.
- [ ] Je vérifie qu’une cible glissante n’est comptée qu’une fois.
- [ ] Je décris contexte, frontières et tokenizer.
- [ ] Je sais ce qu’une PPL ne dit pas sur un assistant.

## Fiche mémo

```text
Additionner les NLL ; compter les cibles ; diviser ; exponentier.
Une PPL n’a de sens qu’avec son protocole de contexte et de tokenisation.
```

Le [module 45](45_pipeline_de_donnees_de_pre_entrainement.md) prépare les corpus qui alimentent ces mesures.

## Référence

[Transformers : perplexité et fenêtres fixes](https://huggingface.co/docs/transformers/en/perplexity).
