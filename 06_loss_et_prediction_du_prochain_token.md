# Module 6 — Calculer l’erreur de prédiction du prochain token

## 1. Ce que tu vas apprendre

Le modèle produit des logits. Pour l’entraîner, il faut mesurer si ces scores favorisent les bonnes cibles. Cette mesure est la **loss**, ou fonction de perte.

Nous allons partir d’un seul token à prédire, calculer sa pénalité, puis préparer plusieurs positions d’une séquence causale avec les bons décalages et masques.

À la fin, tu devrais pouvoir :

- relier logits, probabilités et pénalité du bon token ;
- expliquer la cross-entropy avec une cible unique ;
- aligner chaque sortie avec le token suivant ;
- calculer une loss PyTorch sans appliquer deux fois softmax ;
- exclure certaines cibles sans supprimer leur contexte ;
- interpréter une baisse de loss avec ses limites.

**Prérequis :** modules 1 à 5 ; formes `[B, L, V]`, masque causal et notion de probabilité.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python et PyTorch sur CPU. Aucun modèle à télécharger pour les trois laboratoires principaux.

## 2. Que veut-on mesurer ?

Contexte : `Le ciel est`. Cible observée dans notre exemple : ` bleu`.

Un **label**, ou cible, indique ici l’ID du token attendu. Le modèle attribue des probabilités à tous les candidats ; on examine celle du label.

| Modèle fictif | Probabilité donnée à ` bleu` | Ce que cela suggère pour cet exemple |
| --- | --- | --- |
| A | 0.80 | La cible est bien favorisée |
| B | 0.05 | Le modèle lui accorde peu de probabilité |

La mesure ne dépend pas du token éventuellement tiré par sampling. Elle évalue la distribution de scores par rapport à la cible connue.

Dans du texte naturel, plusieurs suites peuvent être acceptables. Notre exemple fournit pourtant un token observé comme cible : une pénalité élevée pour cette cible ne démontre pas que toute autre continuation serait absurde.

## 3. De logits à probabilités

Pour des logits `z_i`, softmax donne :

\[
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Le dénominateur tient compte de **tous les candidats du vocabulaire**. Augmenter le score d’un mauvais candidat peut donc réduire la probabilité de la cible, même sans toucher à son propre logit.

Pour trois candidats avec les logits `[2.0, 0.5, -1.0]`, softmax donne environ :

```text
Candidat 0 : 0.7856
Candidat 1 : 0.1753
Candidat 2 : 0.0391
```

Si la cible a l’ID 0, la probabilité qui nous intéresse est `0.7856`. Le nombre 2.0 n’est pas cette probabilité.

## 4. Transformer la probabilité en pénalité

Pour une cible unique d’ID `y`, la pénalité est :

\[
L = -\ln(p_y)
\]

`ln` est le logarithme naturel. Tu peux le voir comme une transformation qui rend les très petites probabilités fortement coûteuses.

| Probabilité de la cible | Pénalité `−ln(p)` environ |
| --- | --- |
| 1.00, limite idéale | 0 |
| 0.90 | 0.1054 |
| 0.50 | 0.6931 |
| 0.10 | 2.3026 |
| 0.01 | 4.6052 |

Passer de 0.01 à 0.10 réduit nettement la pénalité. Plus la probabilité tend vers 0, plus la pénalité augmente.

Cette pénalité est une **negative log-likelihood**, ou NLL. Avec des labels représentant une classe correcte unique, sans pondération ni lissage des labels, la **cross-entropy** utilisée ici correspond à cette NLL, moyennée sur les cibles retenues.

> **À toi.** Deux prédictions peuvent-elles avoir le même token le plus probable et des losses différentes ?
>
> Oui. `[0.9, 0.05, 0.05]` et `[0.4, 0.3, 0.3]` choisissent toutes deux l’ID 0 en greedy, mais si cet ID est la cible, leurs pénalités sont environ 0.1054 et 0.9163.

C’est pourquoi loss et **accuracy**, proportion de choix corrects, ne décrivent pas la même chose.

## 5. Laboratoire A — Une cible, trois candidats

Ce bloc est autonome. Prédis la probabilité de la cible et la pénalité avant de regarder les sorties.

```python
import torch
import torch.nn.functional as F

# Une prédiction, trois scores de classes.
logits = torch.tensor([[2.0, 0.5, -1.0]])
target = torch.tensor([0], dtype=torch.long)

probabilities = torch.softmax(logits, dim=-1)
manual_loss = -torch.log(probabilities[0, target[0]])
loss = F.cross_entropy(logits, target)

print("Probabilités :", probabilities)
print("Somme :", probabilities.sum().item())
print("Pénalité calculée à partir de p :", manual_loss.item())
print("Cross-entropy depuis les logits :", loss.item())
print("Même résultat :", torch.allclose(manual_loss, loss))
```

La loss doit être proche de **0.2413**, et les deux calculs doivent coïncider aux arrondis près.

### Pourquoi donner les logits à `cross_entropy` ?

Avec notre usage à cibles entières, `F.cross_entropy` attend les **scores bruts**, pas leur softmax. La fonction calcule une forme numériquement stable de log-softmax puis de la pénalité des cibles.

La formule directe `-log(softmax(...))` aide à comprendre, mais peut perdre en stabilité avec des valeurs extrêmes. Évite de passer `probabilities` comme si elles étaient des logits : cela changerait le calcul.

Les cibles sont des entiers de type `torch.long`, chacun étant un ID de classe, sans vecteur one-hot nécessaire dans cet usage.

## 6. Plusieurs tokens : aligner les positions

Supposons une séquence fictive de quatre IDs :

```text
IDs : [0, 1, 2, 3]
Texte illustratif : Le | chat | dort | .
```

Le modèle causal produit un vecteur de logits à chaque position. La sortie d’une position prédit **le token suivant**, pas le token déjà présent à cette position.

| Position de sortie | Préfixe visible | ID cible |
| --- | --- | --- |
| 0 | `[0]` | 1 |
| 1 | `[0, 1]` | 2 |
| 2 | `[0, 1, 2]` | 3 |
| 3 | `[0, 1, 2, 3]` | Pas de token suivant fourni ici |

Nous gardons donc les logits jusqu’à l’avant-dernière position et les labels à partir de la deuxième :

```text
logits utilisés : positions 0, 1, 2
labels utilisés : IDs       1, 2, 3
```

La dernière sortie ne peut pas être évaluée dans ce morceau sans cible suivante. Si un token EOS explicite est ajouté à la séquence, il peut devenir une cible lui aussi.

### Pourquoi le masque causal reste indispensable

Pendant l’entraînement, toute la séquence est disponible en mémoire. Sans masque causal, la sortie de position 0 pourrait consulter le token de position 1 qu’elle essaie de prédire. Le décalage des labels n’empêche pas cette fuite à lui seul.

## 7. Laboratoire B — Calculer la loss d’une séquence

Ce bloc autonome fournit des logits fictifs afin de voir exactement quelles sorties sont utilisées. Il ne calcule pas l’attention et n’entraîne aucun réseau.

```python
import torch
import torch.nn.functional as F

input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
labels = input_ids.clone()
logits = torch.tensor([[
    [0.0, 2.0, 0.0, 0.0],  # Position 0 : cible 1.
    [0.0, 0.0, 0.5, 0.0],  # Position 1 : cible 2, moins favorisée.
    [0.0, 0.0, 0.0, 3.0],  # Position 2 : cible 3, bien favorisée.
    [9.0, 0.0, 0.0, 0.0],  # Position 3 : sortie non évaluée ici.
]])

shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
vocab_size = logits.shape[-1]

# Pour cet appel, chaque ligne est une prédiction sur le vocabulaire.
flat_logits = shift_logits.reshape(-1, vocab_size)
flat_labels = shift_labels.reshape(-1)
per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
loss = F.cross_entropy(flat_logits, flat_labels)

print("Forme avant décalage :", tuple(logits.shape))
print("Forme après décalage :", tuple(shift_logits.shape))
print("Cibles :", flat_labels.tolist())
print("Loss par cible :", per_token.tolist())
print("Moyenne :", loss.item())
print("Moyenne vérifiée :", torch.allclose(loss, per_token.mean()))
```

Tu dois obtenir des logits de forme **`[1, 4, 4]`**, puis **`[1, 3, 4]`** après décalage. Les cibles sont **`[1, 2, 3]`**. La cible de position 2 dans la liste initiale, moins favorisée, reçoit la pénalité la plus élevée.

`reshape(-1, vocab_size)` rassemble batch et positions en un axe de prédictions. Pour `F.cross_entropy` avec ces scores 2D, le second axe est celui des classes.

Passer directement des logits `[B, L, V]` à cette fonction ne lui dit pas que le dernier axe est celui des classes : dans son usage multidimensionnel, les classes sont attendues sur l’axe 1. Notre mise à plat évite cette confusion.

### Et `labels=input_ids` dans Transformers ?

De nombreuses implémentations de modèles causaux dans Transformers, dont Qwen3 utilisé dans ce cours, effectuent le décalage dans leur calcul interne de loss. On leur fournit alors une copie **non décalée** des IDs comme labels, avec les masques nécessaires.

Dans le laboratoire ci-dessus, nous appelons directement `F.cross_entropy` : nous effectuons donc le décalage nous-mêmes. Ne le fais pas à la fois à l’extérieur et à l’intérieur du modèle.

Les noms `labels` et `input_ids` ne suffisent pas à déduire la convention de toute API : vérifie le modèle ou le trainer concerné.

## 8. Masquer des cibles avec `-100`

Dans notre utilisation de `cross_entropy`, `ignore_index=-100` signifie : **ne pas compter les cibles qui portent cette valeur**.

Ce n’est pas un token du vocabulaire. On le place dans les labels, pas dans les IDs d’entrée envoyés à l’embedding.

Cas courants :

- ignorer les positions de padding ;
- superviser seulement la réponse assistant, en gardant la question comme contexte.

### Laboratoire C — Superviser seulement la fin

Ce bloc est autonome. Les trois premiers IDs représentent notre prompt fictif et le dernier la réponse fictive.

```python
import torch
import torch.nn.functional as F

input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
logits = torch.tensor([[
    [0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.0, 3.0],
    [9.0, 0.0, 0.0, 0.0],
]])
labels = input_ids.clone()
labels[:, :3] = -100  # Exclure les cibles du prompt, garder la réponse d'ID 3.

shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
shift_labels = labels[:, 1:].reshape(-1)
active = shift_labels != -100
if not active.any().item():
    raise ValueError("Aucune cible active : impossible de calculer une moyenne utile.")

loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
print("Entrée conservée :", input_ids.tolist())
print("Labels avant décalage :", labels.tolist())
print("Labels après décalage :", shift_labels.tolist())
print("Nombre de cibles actives :", active.sum().item())
print("Loss sur la réponse seule :", loss.item())
```

Les labels décalés sont **`[-100, -100, 3]`**. Une seule cible contribue à la moyenne, avec une pénalité proche de **0.1392**.

Le token réponse de position 3 est évalué grâce aux logits de position 2. C’est ce décalage qui permet d’apprendre le premier token de la réponse à partir du prompt.

Les tokens du prompt restent dans `input_ids`. Dans un vrai forward causal, ils peuvent donc influencer la prédiction de la réponse même si leurs propres cibles ne sont pas supervisées.

### Choisir la moyenne correcte

Avec notre appel sans poids de classes, la réduction moyenne divise par le nombre de cibles **non ignorées**, pas par le nombre total de cases du batch.

Si toutes les cibles sont ignorées, il n’existe pas de moyenne utile à calculer ; une réduction moyenne peut produire `nan`. Le contrôle du script permet de repérer le problème.

Pour une loss globale sur plusieurs batches, additionne les pénalités actives et divise par le nombre total de cibles actives. Faire la moyenne des moyennes donne un poids identique aux batches, même si leurs effectifs de tokens évalués diffèrent.

## 9. Lire une loss sans lui faire dire trop

Une loss faible signifie que le modèle attribue en moyenne davantage de probabilité aux cibles retenues, **dans les conditions de cette mesure**.

Elle ne prouve pas seule :

- que le modèle généralisera à d’autres exemples ;
- que ses réponses sont factuellement vraies ;
- qu’il suit toutes les instructions ;
- qu’il obtient un bon score sur une autre tâche.

Comparer deux losses demande notamment de garder compatibles le corpus, le tokenizer, les cibles et masques, et la règle de moyenne. Changer le masque entre prompt et réponse change ce que tu mesures.

Une baisse de loss train accompagnée d’une hausse de validation peut indiquer du surapprentissage ; il faudra examiner les données et les courbes aux modules 11 et 14.

**Calculer la loss ne change toujours pas les poids.** Nous n’avons appelé ni backward ni optimizer dans ce module.

## 10. Expériences guidées

1. Dans le laboratoire A, remplace le score de la cible par 4.0, puis par −4.0. Prédis le sens du changement de loss et vérifie-le.
2. Repars du laboratoire A et ajoute 100 à tous les logits. La distribution et la loss doivent rester presque identiques : seuls les écarts de scores comptent.
3. Dans le laboratoire B, change uniquement la dernière ligne de logits. La loss doit rester identique, car cette ligne est exclue par le décalage.
4. Dans le laboratoire C, supervise tous les labels à la place de la seule réponse, puis compare. Le score change parce que les cibles évaluées changent ; ne conclus pas que le modèle a appris entre les deux appels.

Note pour chaque essai : la modification, le nombre de cibles actives et l’explication du résultat.

## 11. Exercices avec corrigés

### Exercice 1 — Comparer deux probabilités

Pour une même cible, laquelle donne la loss la plus faible : 0.7 ou 0.2 ? Donne les deux pénalités approximatives.

<details>
<summary>Corrigé</summary>

`−ln(0.7) ≈ 0.3567` contre `−ln(0.2) ≈ 1.6094`. La probabilité 0.7 donne la plus faible pénalité.

</details>

### Exercice 2 — Décaler sans se tromper

Une séquence d’IDs vaut `[5, 8, 2, 9]`. Quelles positions de logits et quelles cibles utilises-tu pour la loss causale manuelle ?

<details>
<summary>Corrigé</summary>

Les logits de positions **0, 1, 2** sont comparés aux cibles **8, 2, 9**. La première cible observée, 5, n’a pas de préfixe fourni ici ; la dernière sortie n’a pas de token suivant fourni.

</details>

### Exercice 3 — Loss uniforme

Le modèle attribue une probabilité de 1/4 à chaque token d’un vocabulaire de quatre entrées. Quelle est la loss d’une cible unique ?

<details>
<summary>Corrigé</summary>

`−ln(1/4) = ln(4) ≈ 1.3863`. C’est une référence pour une distribution uniforme dans ce vocabulaire précis, pas une valeur universelle de loss de LLM.

</details>

### Exercice 4 — Garder le contexte

Pourquoi ne faut-il pas remplacer par `-100` les IDs du prompt dans `input_ids` pour ne superviser que la réponse ?

<details>
<summary>Corrigé</summary>

`-100` est ici une valeur spéciale de label ignoré, pas un ID d’embedding valide. Il faut garder le prompt dans l’entrée et masquer ses cibles dans une copie utilisée comme labels.

</details>

### Exercice 5 — Agréger les batches

Un batch contient 2 cibles actives avec une loss moyenne de 1. Un autre en contient 8 avec une moyenne de 3. Quelle est la moyenne globale par cible ?

<details>
<summary>Corrigé</summary>

`(2×1 + 8×3) / (2+8) = 2.6`. La moyenne simple des deux losses donnerait 2, ce qui ne pondère pas correctement les dix cibles.

</details>

## 12. Validation — avant de continuer

- [ ] Je peux expliquer `logits → softmax → p(cible) → −ln(p)`.
- [ ] Je retrouve la loss de 0.2413 du premier laboratoire.
- [ ] Je sais pourquoi `cross_entropy` reçoit des logits dans cet usage.
- [ ] Je peux dessiner le décalage causal sans le faire deux fois.
- [ ] Je distingue ID d’entrée et label ignoré `-100`.
- [ ] Je sais compter les cibles actives et expliquer la moyenne.
- [ ] Je peux dire ce qu’une loss basse ne démontre pas.

## Fiche mémo

```text
Cible unique : pénalité = −ln(probabilité de cette cible).
Cross-entropy : calcul stable depuis les logits, puis agrégation.
Décalage manuel : logits[:, :-1] face à labels[:, 1:].
Masque de loss : ignorer des cibles sans effacer le contexte d’entrée.
Ne pas confondre bonne moyenne sur les cibles et qualité universelle du modèle.
```

Le [module 7](07_gradients_et_autograd.md) explique comment cette erreur permet de calculer une direction de correction des paramètres.

## Références

- [PyTorch : `functional.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) — formes, cibles, réduction et `ignore_index`.
- [Hugging Face : causal language modeling](https://huggingface.co/docs/transformers/en/tasks/language_modeling) — cibles et entraînement causal dans Transformers.
