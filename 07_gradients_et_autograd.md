# Module 7 — Comprendre les gradients et les observer avec autograd

## 1. Ce que tu vas apprendre

Au [module 6](06_loss_et_prediction_du_prochain_token.md), tu as calculé une erreur. Cette valeur ne dit pas encore comment corriger chaque poids. Il faut mesurer **comment la loss varie quand un paramètre change légèrement**.

Nous allons commencer par un seul nombre, calculer sa dérivée à la main, puis laisser PyTorch retrouver le résultat. Nous passerons ensuite à deux paramètres et à l’accumulation des gradients.

À la fin, tu devrais pouvoir :

- interpréter le signe et l’amplitude locale d’une dérivée ;
- distinguer loss, gradient et mise à jour ;
- expliquer le graphe de calcul et la règle de chaîne ;
- utiliser `requires_grad`, `backward()` et `.grad` ;
- observer une accumulation volontaire et remettre les gradients à zéro ;
- distinguer gradient nul, gradient absent et calcul sans autograd ;
- expliquer pourquoi un pas trop grand peut faire monter la loss.

**Prérequis :** modules 1 à 6 ; variables, fonctions et boucles Python. Pas besoin de maîtriser un cours complet de calcul différentiel.

**Temps indicatif :** deux séances de 60 à 90 minutes, exercices compris.

**Matériel :** Python et PyTorch sur CPU, sans modèle à télécharger.

## 2. Commencer par une seule valeur à apprendre

Imaginons un paramètre `w` dont la valeur idéale serait 3. Nous choisissons une loss facile à inspecter :

\[
L(w) = (w - 3)^2
\]

| `w` | `w − 3` | Loss |
| --- | --- | --- |
| 1 | −2 | 4 |
| 2 | −1 | 1 |
| 3 | 0 | 0 |
| 4 | 1 | 1 |
| 5 | 2 | 4 |

La loss décrit une courbe en forme de bol. À gauche de 3, augmenter un peu `w` réduit l’erreur ; à droite, il faut le diminuer.

La **dérivée** mesure la pente locale de cette courbe. Pour notre fonction :

\[
\frac{dL}{dw} = 2(w - 3)
\]

Le symbole `dL/dw` se lit « dérivée de L par rapport à w ».

| Point de départ | Dérivée | Direction de correction avec un petit pas |
| --- | --- | --- |
| `w = 1` | −4 | Augmenter `w` |
| `w = 3` | 0 | Aucune correction suggérée par ce gradient |
| `w = 5` | +4 | Diminuer `w` |

Le signe indique une direction locale. L’amplitude mesure la sensibilité dans les unités de ce paramètre et de cette loss ; elle n’est pas une note de qualité du modèle.

## 3. Vérifier une pente avec une petite perturbation

Sans connaître la dérivée exacte, on peut l’approcher par :

\[
\frac{dL}{dw} \approx \frac{L(w+h)-L(w-h)}{2h}
\]

`h` est un petit déplacement positif. On compare la loss à droite et à gauche du point.

Pour `w = 1` et `h = 0.01` :

```text
L(1.01) = 3.9601
L(0.99) = 4.0401
Pente approchée = (3.9601 − 4.0401) / 0.02 = −4
```

Cette approximation par différences finies sert à vérifier un micro-exemple. Elle n’est pas la méthode utilisée par autograd pour chaque poids : perturber séparément des milliards de paramètres serait beaucoup trop coûteux.

Un `h` trop petit peut aussi rendre les différences sensibles aux arrondis numériques. Plus petit ne veut pas toujours dire plus précis.

## 4. Transformer un gradient en correction

La descente de gradient simple applique :

\[
w_{nouveau} = w_{ancien} - \eta \frac{dL}{dw}
\]

`η`, lettre grecque êta, représente le **learning rate**. Avec `w = 1`, un gradient de `−4` et `η = 0.1` :

```text
w nouveau = 1 − 0.1 × (−4) = 1.4
L nouvelle = (1.4 − 3)² = 2.56
```

La loss passe de **4 à 2.56**. Soustraire un gradient négatif augmente le paramètre.

Cette règle correspond à SGD sans momentum ni pénalisation. D’autres optimizers, comme AdamW, transforment les gradients avec des règles supplémentaires ; leurs mises à jour ne sont pas décrites entièrement par cette seule formule.

> **Pause compréhension.** Si le gradient vaut −4, pourquoi ne pas simplement ajouter 4 à `w` ?
>
> Le gradient décrit la pente au point actuel, pas la distance exacte jusqu’à la solution. Le learning rate règle l’amplitude du déplacement. Un déplacement trop grand peut dépasser la région où la direction locale est utile.

## 5. Laboratoire A — Loss, gradient et mise à jour séparés

Ce bloc est autonome. Prédis la valeur de `w` juste après backward, puis après la correction.

```python
import torch

w = torch.tensor(1.0, requires_grad=True)
loss = (w - 3.0) ** 2
print("Avant backward : w=", w.item(), "loss=", loss.item(), "grad=", w.grad)

loss.backward()
print("Après backward : w=", w.item(), "grad=", w.grad.item())

learning_rate = 0.1
# La mise à jour elle-même ne doit pas devenir une partie du prochain graphe.
with torch.no_grad():
    w -= learning_rate * w.grad
    new_loss = (w - 3.0) ** 2

print("Après correction : w=", w.item(), "nouvelle loss=", new_loss.item())
print("Ancien nombre de loss :", loss.item())
```

Tu dois retrouver, aux arrondis près :

```text
Avant backward : w=1, loss=4, grad=None
Après backward : w=1, grad=-4
Après correction : w=1.4, nouvelle loss=2.56
Ancien nombre de loss : 4
```

**Backward calcule les gradients ; il ne corrige pas les poids.** Ici, c’est la soustraction explicite qui réalise la mise à jour. Au module 8, un optimizer s’en chargera.

L’ancien tenseur `loss` ne recalcule pas sa valeur quand `w` change. Pour mesurer l’erreur après correction, il faut refaire le calcul, comme pour `new_loss`.

## 6. Comment autograd retrouve le résultat

### Le graphe de calcul

Dans le laboratoire A, PyTorch suit cette dépendance :

```text
w → soustraire 3 → élever au carré → loss
```

Avec le suivi activé, les opérations construisent un **graphe de calcul**. Il contient les relations et informations nécessaires pour remonter de la loss vers les paramètres.

`requires_grad=True` demande le suivi pour `w`. Un paramètre créé par les modules PyTorch est habituellement un `nn.Parameter`, entraînable par défaut.

### La règle de chaîne

Posons `u = w − 3`, puis `L = u²` :

\[
\frac{dL}{dw} = \frac{dL}{du} \times \frac{du}{dw}
= 2u \times 1 = 2(w-3)
\]

La **backpropagation**, ou rétropropagation, applique efficacement cette règle en remontant les opérations. Autograd calcule ainsi les gradients à partir du programme exécuté, sans essayer de petites variations de chaque paramètre.

### Lire les objets importants

| Objet | Rôle |
| --- | --- |
| `w.requires_grad` | Indique que ce tenseur demande un suivi des gradients |
| `loss.grad_fn` | Donne accès à une opération du graphe qui a produit la loss suivie |
| `loss.backward()` | Déclenche la rétropropagation depuis cette loss scalaire |
| `w.grad` | Reçoit le gradient accumulé pour notre paramètre feuille |
| `loss.item()` | Extrait une valeur Python, utile pour afficher, sans conserver le graphe |

Nous utilisons ici une **loss scalaire**, c’est-à-dire un seul nombre. Pour une sortie non scalaire, il faut préciser comment les différentes composantes contribuent à la dérivation, par exemple en prenant leur moyenne avant backward.

Les tenseurs intermédiaires suivis ne conservent pas tous automatiquement leur propre champ `.grad`. Nos exemples lisent les gradients des paramètres **feuilles** ; ce terme désigne ici des tenseurs créés directement plutôt que produits par une opération suivie.

## 7. Plusieurs paramètres : le gradient

Une dérivée décrit la sensibilité à une variable. Avec plusieurs paramètres, on obtient un **gradient** : l’ensemble de leurs dérivées partielles.

Exemple :

\[
\hat y = wx + b, \qquad L = (\hat y - y)^2
\]

Prenons `x = 2`, `y = 6`, `w = 1`, `b = 0` :

```text
Prédiction : 1×2 + 0 = 2
Erreur : 2 − 6 = −4
Loss : (−4)² = 16
```

Par la règle de chaîne :

\[
\frac{\partial L}{\partial w} = 2(\hat y-y)x = -16
\qquad
\frac{\partial L}{\partial b} = 2(\hat y-y) = -8
\]

Le symbole `∂` indique une dérivée partielle : on regarde l’effet d’un paramètre en maintenant les autres fixes.

### Laboratoire B — Deux paramètres, un backward

Ce bloc est autonome.

```python
import torch

x = torch.tensor(2.0)
y = torch.tensor(6.0)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

prediction = w * x + b
loss = (prediction - y) ** 2
loss.backward()

print("Prédiction :", prediction.item())
print("Loss :", loss.item())
print("Gradient w :", w.grad.item())
print("Gradient b :", b.grad.item())
print("Poids toujours inchangés :", w.item(), b.item())
```

Les gradients attendus sont **−16** et **−8**. Un seul backward a remonté toutes les branches nécessaires du graphe.

La différence entre leurs amplitudes vient ici notamment du facteur `x = 2`. Elle n’implique pas que `w` soit deux fois plus « intelligent » ou plus important dans toutes les situations.

## 8. Les gradients s’accumulent

PyTorch additionne les nouveaux gradients dans les champs `.grad` déjà présents. Il ne les remplace pas automatiquement à chaque backward.

### Laboratoire C — Observer l’addition

Ce bloc est autonome. Nous reconstruisons une nouvelle loss à chaque fois, sans mise à jour du paramètre.

```python
import torch

w = torch.tensor(1.0, requires_grad=True)

loss_1 = (w - 3.0) ** 2
loss_1.backward()
print("Premier gradient :", w.grad.item())

loss_2 = (w - 3.0) ** 2  # Nouveau graphe, même calcul, même w.
loss_2.backward()
print("Après addition du second :", w.grad.item())

w.grad = None          # Remise à zéro de l'accumulation pour ce paramètre.
loss_3 = (w - 3.0) ** 2
loss_3.backward()
print("Après remise à zéro et nouveau backward :", w.grad.item())
```

Résultat attendu : **−4 → −8 → −4**.

Dans une boucle avec optimizer, on utilise généralement `optimizer.zero_grad(set_to_none=True)` avant le nouveau calcul de gradients. Cela remet les gradients des paramètres de l’optimizer à `None`, sans effacer les poids.

L’accumulation peut être voulue pour combiner plusieurs micro-batches avant une mise à jour. Elle demande alors une agrégation et une normalisation appropriées ; le module 23 la détaillera.

### Nouveau graphe ou même graphe ?

Après un backward ordinaire, les intermédiaires sauvegardés pour la rétropropagation sont généralement libérés. Appeler une deuxième fois backward sur **la même loss** peut donc échouer.

Notre script calcule une nouvelle loss avant chaque backward. `retain_graph=True` sert à d’autres usages ; il ne corrige pas un oubli de remise à zéro et il n’est pas nécessaire ici.

## 9. Gradient nul, gradient absent et suivi désactivé

| Situation | Ce qu’il faut comprendre |
| --- | --- |
| `.grad` vaut `None` avant backward | Aucun gradient n’a encore été stocké, ou le champ a été remis à `None` |
| `.grad` vaut zéro après backward | La dérivée calculée vaut zéro pour cette composante ; elle peut aussi venir d’annulations ou de zones plates |
| `.grad` reste `None` après backward | Vérifier le suivi, la participation au graphe et le fait de lire un paramètre feuille |
| Calcul dans `no_grad()` | Les opérations usuelles de ce calcul ne construisent pas leur graphe autograd |
| Calcul détaché avec `detach()` | La branche détachée n’entretient plus le lien autograd avec son origine |

Dans notre fonction quadratique, `w = 3` donne un gradient nul et un minimum. Dans un réseau général, un gradient nul ne prouve pas que l’on a trouvé la meilleure solution : il peut exister des plateaux, des points selles ou d’autres causes.

### `eval()` et `no_grad()`

`model.eval()` change le comportement de couches comme dropout ; il ne désactive pas autograd. `torch.no_grad()` contrôle l’enregistrement du graphe pour les calculs du bloc ; il ne change pas le mode du modèle.

Pour évaluer ordinairement un réseau, on utilise les deux. Pour l’entraîner, il faut que le calcul reliant les paramètres à la loss reste suivi.

`loss.item()` sert à l’affichage : un nombre Python ne possède pas le graphe nécessaire à backward. Garde le tenseur `loss` pour la rétropropagation.

## 10. Pourquoi un pas trop grand peut aggraver l’erreur

Repartons toujours de `w = 1`, `L = 4`, gradient `−4` :

| Learning rate | Nouveau `w` | Nouvelle loss | Observation |
| --- | --- | --- | --- |
| 0.001 | 1.004 | 3.984016 | Très petite correction |
| 0.1 | 1.4 | 2.56 | Amélioration |
| 0.5 | 3.0 | 0 | Solution exacte de ce problème particulier |
| 1.0 | 5.0 | 4 | Dépassement ; un autre pas renverrait à 1 |
| 1.1 | 5.4 | 5.76 | L’erreur augmente |

`0.5` fonctionne exactement ici à cause de la forme de cette quadratique. Ce n’est pas un learning rate conseillé pour un LLM.

À un point où le gradient est non nul, une direction opposée au gradient réduit localement la loss pour des pas suffisamment petits dans ce problème lisse. Cela ne garantit pas qu’un pas arbitraire ni tous les pas d’un entraînement complexe feront baisser l’erreur mesurée.

## 11. Une première idée du gradient clipping

Des gradients très grands peuvent produire des mises à jour difficiles à contrôler. Le **gradient clipping** limite leur norme avant la mise à jour.

Ce bloc autonome montre une seule composante. Il ne représente pas un entraînement complet.

```python
import torch

parameter = torch.nn.Parameter(torch.tensor(1.0))
loss = 100.0 * parameter
loss.backward()
print("Gradient avant :", parameter.grad.item())
old_norm = torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.0)
print("Norme avant clipping :", old_norm.item())
print("Gradient après :", parameter.grad.item())
print("Paramètre inchangé :", parameter.item())
```

La norme passe d’environ **100 à 1**, tandis que le paramètre vaut toujours **1**. Le clipping agit ici sur le gradient stocké, pas sur le poids lui-même. Dans une boucle, il se place **après backward et avant step**.

Il ne remplace pas la recherche d’une erreur de labels, d’un learning rate mal adapté ou d’un calcul non fini. Les normes et les problèmes numériques seront repris dans les modules de diagnostic.

## 12. Expériences guidées

1. Refais le laboratoire A avec `w = 5`. Prédis un gradient de `+4`, puis un déplacement vers `4.6` avec le même LR.
2. Vérifie la pente de `L=(w−3)²` avec la formule de différences finies en `w = 1`, puis en `w = 5`. Compare avec autograd.
3. Compare les learning rates de la section 10, en recréant `w` à chaque essai. Note la loss avant et après une correction.
4. Dans le laboratoire B, remplace `x = 2` par `x = 0`, sans changer le reste. Prédis `grad_w = 0` et `grad_b = −12`, puis explique pourquoi modifier `w` ne change plus la prédiction pour cette entrée.
5. Dans une copie du laboratoire A, calcule la loss sous `no_grad()` puis tente backward. Observe l’erreur et explique le lien manquant ; replace ensuite le calcul suivi.

Pour chaque essai, écris la prédiction, l’observation et une phrase reliant les deux. Une erreur volontaire sert à comprendre un mécanisme précis.

## 13. Exercices avec corrigés

### Exercice 1 — Interpréter le signe

Pour `L=(w−10)²`, donne la dérivée en `w=3`, puis en `w=15`. Dans quel sens un petit pas de descente déplace-t-il `w` ?

<details>
<summary>Corrigé</summary>

La dérivée est `2(w−10)`. En 3 : **−14**, donc le petit pas augmente `w`. En 15 : **+10**, donc le petit pas le diminue.

</details>

### Exercice 2 — Lire une correction

Un poids vaut 2, son gradient vaut 5 et le LR vaut 0.1. Avec la règle simple de SGD, quelle est sa nouvelle valeur ? Est-ce backward qui réalise ce changement ?

<details>
<summary>Corrigé</summary>

`2 − 0.1 × 5 = 1.5`. Backward calcule le gradient ; la soustraction explicite ou l’optimizer step applique la correction.

</details>

### Exercice 3 — Comprendre l’accumulation

Deux backwards sur deux nouveaux graphes produisent chacun un gradient de 3 pour le même paramètre. Sans remise à zéro entre eux, que contient `.grad` après le second ?

<details>
<summary>Corrigé</summary>

Si le champ était initialement absent ou nul, il contient **6**. Aucun poids n’est nécessairement modifié ; l’accumulation concerne les gradients.

</details>

### Exercice 4 — Refaire la chaîne

Avec `x=2`, `y=6`, `w=1`, `b=0`, pourquoi le gradient de `w` vaut-il deux fois celui de `b` ?

<details>
<summary>Corrigé</summary>

La dérivée de la prédiction `wx+b` par rapport à `w` vaut `x=2`, tandis que celle par rapport à `b` vaut 1. La règle de chaîne multiplie la même dérivée d’erreur par ces deux facteurs : on obtient −16 et −8.

</details>

### Exercice 5 — Repérer une fausse conclusion

« Mon gradient est nul, donc mon modèle est parfait. » Pourquoi cette conclusion est-elle insuffisante ?

<details>
<summary>Corrigé</summary>

Un gradient nul est une information locale sur le calcul évalué. Il ne prouve ni un minimum global, ni une bonne généralisation, ni même que le graphe et les masques correspondent à l’objectif. Il faut examiner le contexte et les mesures pertinentes.

</details>

### Exercice 6 — Choisir le bon réglage

Tu veux mesurer la loss de validation sans enregistrer de graphe et avec le comportement d’évaluation des couches. Quel couple d’instructions utilises-tu ?

<details>
<summary>Corrigé</summary>

`model.eval()` puis le calcul dans `with torch.no_grad():`. L’un règle le comportement des couches, l’autre l’enregistrement autograd. Aucun optimizer step n’a lieu pendant cette validation.

</details>

## 14. Validation — avant de continuer

- [ ] Je peux refaire la dérivée de `(w−3)²` et interpréter ses signes.
- [ ] Je retrouve `w=1.4` et `loss=2.56` après la première correction.
- [ ] Je distingue backward et mise à jour des poids.
- [ ] Je peux expliquer la règle de chaîne du modèle à deux paramètres.
- [ ] Je retrouve l’accumulation `−4 → −8 → −4` avec des graphes reconstruits.
- [ ] Je distingue `.grad=None`, gradient nul et suivi désactivé.
- [ ] Je sais pourquoi `eval()` et `no_grad()` ne sont pas interchangeables.
- [ ] Je peux expliquer un dépassement dû à un pas trop grand et situer le clipping.

## Fiche mémo

```text
Loss : combien d’erreur selon la mesure choisie.
Gradient : sensibilité locale de cette erreur aux paramètres.
Backward : calculer et accumuler les gradients.
Mise à jour : utiliser les gradients pour modifier les paramètres.

Convention à conserver au module 8 :
zero_grad → forward → loss → backward → step
Clipping éventuel : après backward, avant step.
```

Le [module 8](08_premiere_boucle_dentrainement_pytorch.md) assemble ces opérations dans une boucle complète avec batches, epochs et validation. Refaire les micro-calculs de ce module te permettra d’expliquer chaque ligne de cette boucle.

## Références

- [PyTorch : différentiation automatique avec autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) — graphes et rétropropagation.
- [PyTorch : `Optimizer.zero_grad`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) — remise à zéro et gradients absents.
- [PyTorch : `clip_grad_norm_`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — limitation de norme des gradients.
