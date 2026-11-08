# Module 23 — Accumuler les gradients sans fausser leur moyenne

## 1. Ce que tu vas apprendre

Si un grand batch ne tient pas en mémoire, plusieurs petits calculs peuvent contribuer à une même mise à jour. Tu sais déjà que backward additionne les gradients ; nous allons utiliser cette propriété volontairement.

Tu vas compter micro-batches et updates, traiter une dernière fenêtre incomplète et vérifier une équivalence numérique dans un cas contrôlé.

**Prérequis :** modules 7, 8, 17 et 22.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et PyTorch sur CPU.

## 2. Compter ce qui contribue à un step

| Terme | Définition concrète |
| --- | --- |
| Micro-batch | Exemples traités dans un forward/backward local |
| Fenêtre d’accumulation | Micro-batches additionnés avant une update |
| Batch effectif | Ensemble des contributions à cette update |
| Optimizer step | Moment où les paramètres changent |

Quand les tailles sont régulières sur D appareils :

\[
B_{effectif} = B_{par\ appareil}\times D\times A
\]

Avec 2 exemples par appareil, 4 appareils et 8 accumulations : **64 exemples par update complète**.

Cette formule nominale ne décrit pas nécessairement le dernier groupe, ni les tokens évalués. Deux séquences peuvent avoir 10 et 1 000 cibles actives ; « deux exemples » ne décrit pas toute la normalisation.

## 3. Dessiner les compteurs

Sur un appareil, avec 5 exemples, micro-batch 2 et accumulation 2 :

```text
Micro-batch 1 : 2 exemples → backward
Micro-batch 2 : 2 exemples → backward → step 1 → zero_grad
Micro-batch 3 : 1 exemple  → backward → step 2 → zero_grad
```

Le batch effectif nominal vaut 4, mais la dernière update utilise 1 exemple. Oublier ce dernier step perd une contribution ; diviser sa loss par 2 comme si deux micro-batches étaient présents réduit son échelle différemment de la moyenne souhaitée.

Avec N exemples, micro-batch B et accumulation A, un appareil, dernier micro-batch conservé et fenêtre finale appliquée à chaque epoch :

\[
steps = \left\lceil\frac{\lceil N/B\rceil}{A}\right\rceil
\]

Les samplers distribués et stratégies de fin d’epoch peuvent modifier les comptes : mesure les contributions réelles.

## 4. Pourquoi la moyenne des moyennes peut être fausse

Deux micro-batches contiennent 2 et 1 exemples. Leurs losses moyennes valent 1 et 4.

```text
Moyenne des deux moyennes : (1+4)/2 = 2.5
Moyenne par exemple : (2×1+1×4)/3 = 2
```

Pour reproduire la moyenne par exemple, on peut calculer chaque somme d’erreurs puis diviser par **l’effectif total réel de la fenêtre** avant backward.

Pour une cross-entropy tokenisée, le dénominateur pertinent est souvent le nombre total de cibles actives, selon l’objectif choisi. La longueur en cases PAD n’est pas ce nombre.

## 5. Laboratoire autonome — Comparer deux calculs équivalents

Le script compare deux updates sur des batches logiques de tailles 4 puis 1 : l’un en une passe par batch logique, l’autre découpé en micro-batches de tailles 2,2 puis 1.

Nous utilisons une simple régression, une loss par exemple et SGD sans momentum. La double précision aide à observer l’équivalence malgré les arrondis.

```python
from copy import deepcopy
import torch
from torch import nn

torch.manual_seed(42)
reference = nn.Linear(1, 1).double()
accumulated = deepcopy(reference)
opt_reference = torch.optim.SGD(reference.parameters(), lr=0.01)
opt_accumulated = torch.optim.SGD(accumulated.parameters(), lr=0.01)
x = torch.arange(1, 6, dtype=torch.float64).reshape(-1, 1)
y = 2 * x + 1
micro_batches = [(x[i:i+2], y[i:i+2]) for i in range(0, len(x), 2)]
accumulation = 2
updates = 0

for start in range(0, len(micro_batches), accumulation):
    window = micro_batches[start:start + accumulation]
    total_examples = sum(len(x_batch) for x_batch, _ in window)

    # Référence : une moyenne sur le batch logique complet.
    full_x = torch.cat([x_batch for x_batch, _ in window])
    full_y = torch.cat([y_batch for _, y_batch in window])
    opt_reference.zero_grad(set_to_none=True)
    nn.functional.mse_loss(reference(full_x), full_y).backward()
    opt_reference.step()

    # Accumulation : chaque somme contribue à la même moyenne logique.
    opt_accumulated.zero_grad(set_to_none=True)
    for x_batch, y_batch in window:
        loss_sum = nn.functional.mse_loss(accumulated(x_batch), y_batch, reduction='sum')
        (loss_sum / total_examples).backward()
    opt_accumulated.step()
    updates += 1
    print('Update', updates, 'micro-batches=', len(window), 'exemples=', total_examples)

same = all(torch.allclose(left, right, atol=1e-12, rtol=1e-12)
           for left, right in zip(reference.parameters(), accumulated.parameters()))
print('Memes parametres finaux :', same)
```

Tu dois obtenir **2 updates**, avec **4 puis 1 exemple**, et une comparaison finale vraie.

Chaque backward reconstruit et libère son propre graphe. Le script conserve les gradients, pas tous les graphes précédents. Utiliser `retain_graph=True` n’est pas nécessaire pour cette accumulation.

## 6. Les endroits où placer les opérations

```text
Début de fenêtre : zero_grad
Chaque micro-batch : forward → contribution correctement normalisée → backward
Fin de fenêtre : clipping éventuel → optimizer step → scheduler step
```

Remettre les gradients à zéro à chaque micro-batch détruit l’accumulation. Faire un step à chaque micro-batch décrit un autre entraînement, car les paramètres changent avant les contributions suivantes.

Un trainer peut gérer lui-même les facteurs d’échelle et l’accumulation. Ne divise pas une seconde fois la loss sans connaître son implémentation. Vérifie notamment sa gestion des nombres de tokens actifs et de la dernière fenêtre.

## 7. Pourquoi « équivalent » demande des conditions

L’équivalence démontrée repose sur des poids fixes pendant la fenêtre, la même objective additive et la même normalisation.

Dropout peut modifier les tirages ; BatchNorm dépend des statistiques du micro-batch ; certains calculs et réductions distribuées introduisent des différences numériques. On ne promet donc pas une trajectoire bit à bit identique pour tous les réseaux.

En DDP, la réduction moyenne entre appareils et les effectifs locaux doivent aussi être pris en compte. Si les nombres de cibles diffèrent, une moyenne locale par appareil puis une moyenne de ces moyennes peut ne pas correspondre à la moyenne globale par token.

## 8. Mémoire et débit

L’accumulation évite de garder tous les exemples simultanément dans les activations d’un grand forward. Les gradients restent en mémoire entre micro-batches.

Elle ne supprime pas le coût des poids et de l’optimizer. Elle augmente le nombre de forwards/backwards entre updates, et le débit peut être moins bon qu’un micro-batch plus grand si celui-ci tient et utilise mieux le matériel.

Compare tokens réels par seconde, pic mémoire et résultat à budget défini. « Même nombre d’epochs » peut cacher un nombre différent d’updates.

## 9. Expériences guidées

1. Divise volontairement chaque moyenne de micro-batch par A, même dans le groupe final incomplet. Compare au résultat de référence et identifie le facteur erroné.
2. Passe à 7 exemples et des micro-batches de 3 avec A=2. Prédis les tailles logiques : 6 puis 1.
3. Calcule micro-batches et updates de ton run réel, puis compare au compteur du trainer.
4. Sur des réponses de longueurs différentes, compte les cibles actives par micro-batch et explique la moyenne que tu veux reproduire.

## 10. Exercices avec corrigés

### Exercice 1 — Batch nominal

1 exemple par appareil, 2 appareils et 32 accumulations : quel batch nominal ?

<details>
<summary>Corrigé</summary>

`1×2×32 = 64` exemples par update complète, sous ces hypothèses de tailles régulières.

</details>

### Exercice 2 — Fin incomplète

103 exemples, micro-batch 10, A=4, un appareil et fin appliquée : combien de micro-batches et d’updates ?

<details>
<summary>Corrigé</summary>

**11 micro-batches**, regroupés en 4,4,3, donc **3 updates**. La dernière update contient 23 exemples, puisque le dernier micro-batch n’en contient que 3.

</details>

### Exercice 3 — Mémoire du graphe

Faut-il garder tous les graphes des micro-batches pour accumuler ?

<details>
<summary>Corrigé</summary>

Non. Chaque backward ajoute son gradient puis son graphe peut être libéré. Ce sont les champs `.grad` qui restent pour l’update commune.

</details>

### Exercice 4 — Scheduler

À quel niveau avancent normalement clipping et scheduler ?

<details>
<summary>Corrigé</summary>

Au niveau de la fenêtre logique : clipping sur le gradient accumulé avant l’update, puis scheduler selon les optimizer steps réellement effectués.

</details>

## 11. Validation — avant de continuer

- [ ] Je distingue micro-batch, fenêtre et batch nominal/réel.
- [ ] Je traite le dernier groupe sans le perdre ni fausser sa moyenne.
- [ ] Je retrouve l’équivalence du laboratoire et ses conditions.
- [ ] Je sais quand compter des exemples ou des tokens actifs.
- [ ] Je place zero_grad, clipping et scheduler au bon endroit.
- [ ] Je ne normalise pas deux fois avec un framework.

## Fiche mémo

```text
Accumuler des contributions à une même moyenne, avec des poids fixes.
Remise à zéro par fenêtre ; backward par micro-batch ; step en fin de fenêtre.
La dernière fenêtre compte réellement ce qu’elle contient.
```

Le [module 24](24_gradient_checkpointing_et_memoire.md) économise une autre composante : les activations conservées.
