# Module 17 — Auditer les paramètres et l’optimizer

## 1. Ce que tu vas apprendre

Après une injection LoRA ou un gel de couches, « peu de paramètres apprennent » doit devenir un ensemble de nombres et de vérifications.

Tu vas compter total, gelé et entraînable, vérifier ce que l’optimizer connaît, puis mesurer les gradients et ses états après une mise à jour.

**Prérequis :** modules 5, 7 et 16.

**Temps indicatif :** 90 à 120 minutes.

**Matériel :** Python et PyTorch sur CPU. Le laboratoire est autonome et ne dépend pas d’un `trainer` déjà créé.

## 2. Quatre ensembles à distinguer

| Ensemble | Comment le reconnaître | Ce qu’il ne garantit pas |
| --- | --- | --- |
| Paramètres présents | `model.parameters()` | Qu’ils reçoivent des gradients |
| Autorisés au suivi | `requires_grad=True` | Qu’ils participent à la loss |
| Inclus dans l’optimizer | Ses `param_groups` | Qu’ils reçoivent une mise à jour non nulle |
| Gradients effectivement calculés | `.grad` après backward | Qu’une amélioration de qualité ait lieu |

Un paramètre peut être entraînable mais absent de l’optimizer. Il peut aussi être inclus mais inutilisé dans ce forward. L’audit doit comparer ces informations plutôt que de regarder seulement un pourcentage.

Si tu gèles des poids après qu’ils ont reçu des gradients, traite aussi leurs éventuels gradients et les groupes d’optimizer. Le laboratoire crée les flags et l’optimizer dans le bon ordre, avant tout backward.

## 3. Refaire les calculs à la main

Notre réseau possède :

```text
Linear(4,4,bias=False) gelée     → 16 paramètres
Linear(4,2,bias=False) entraînée →  8 paramètres
Total                          → 24 paramètres
```

\[
ratio = 100\times\frac{N_{entraînable}}{N_{total}}
\]

Ici `100×8/24 ≈ 33.33 %`.

Tous les poids FP32 occupent `24×4 = 96` octets. Les gradients de la tête, si FP32, ajoutent `8×4 = 32` octets.

AdamW crée généralement deux moments pour chaque paramètre concerné et un compteur de step. Dans notre expérience dense FP32, les deux moments de la matrice de 8 poids occupent `2×8×4 = 64` octets, plus le compteur observé.

Le dtype et l’implémentation peuvent changer : mesure les tenseurs plutôt que d’imposer un multiplicateur universel.

## 4. Laboratoire autonome — Avant et après le premier step

```python
import torch
from torch import nn

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 2, bias=False))
model[0].requires_grad_(False)
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)


def tensor_bytes(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def audit(model, optimizer):
    parameters = list(model.parameters())
    total = sum(p.numel() for p in parameters)
    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    optimized_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and id(p) not in optimized_ids]
    grads = [p.grad for p in parameters if p.grad is not None]
    states = [value for state in optimizer.state.values() for value in state.values()
              if isinstance(value, torch.Tensor)]
    report = {'total': total, 'trainable': trainable, 'frozen': total - trainable,
              'ratio_percent': 100 * trainable / total,
              'weights_bytes': tensor_bytes(parameters),
              'gradient_bytes': tensor_bytes(grads), 'optimizer_tensor_bytes': tensor_bytes(states),
              'missing_from_optimizer': missing}
    return report


print('Avant :', audit(model, optimizer))
x = torch.eye(4)
target = torch.ones(4, 2)
optimizer.zero_grad(set_to_none=True)
loss = nn.functional.mse_loss(model(x), target)
loss.backward()
for name, parameter in model.named_parameters():
    print(name, 'requires_grad=', parameter.requires_grad,
          'gradient present=', parameter.grad is not None)
optimizer.step()
print('Apres :', audit(model, optimizer))
```

Tu dois retrouver **24 au total**, **8 entraînables**, **16 gelés**, **96 octets de poids** et **32 octets de gradients après backward**. La liste des paramètres entraînables absents de l’optimizer doit être vide.

Avant le premier step, les états AdamW sont encore vides. Après, leurs tenseurs apparaissent. Mesurer uniquement juste après la création de l’optimizer sous-estimerait donc son coût final.

Le code ne mesure ni les objets Python, ni toutes les allocations temporaires, ni les activations conservées pendant backward. Les octets obtenus ne sont pas la mémoire totale du processus.

## 5. Adapter l’audit à LoRA

Applique la fonction au modèle PEFT du module 16 et à son optimizer. Exporte les noms, formes, dtypes, flags et participation aux groupes.

Pour une couverture LoRA fixe sans biais ou modules supplémentaires :

\[
N_{ajouté} = \sum_{modules\ ciblés} r(in_i+out_i)
\]

Si la mesure diffère, examine les cibles réelles, les poids partagés, les biais, les embeddings ou les `modules_to_save`. Le modèle adapté contient aussi les paramètres de base : le dénominateur du ratio est donc généralement base **plus** adapter.

Exemple : base 600M plus adapter 6M donne `6/606 ≈ 0.9901 %`, et non exactement 1 %. Les ordres de grandeur restent utiles, mais annonce ta convention.

## 6. Pourquoi le ratio ne prédit pas toute la mémoire

Les activations nécessaires pour entraîner un adapter existent encore. Des gradients peuvent devoir traverser plusieurs blocs gelés pour atteindre les petites matrices entraînables.

Une configuration avec 1 % de paramètres entraînables ne promet donc pas 99 % de VRAM totale en moins. Elle réduit principalement les états associés aux mises à jour de paramètres, pas toutes les autres composantes.

La taille du fichier d’adapter inclut aussi format de sauvegarde et métadonnées. Des dtypes différents ou des modules supplémentaires empêchent parfois de la déduire d’un unique `nombre×2 octets`.

## 7. Expériences guidées

1. Ajoute un biais à la dernière couche. Recalcule total, entraînable, gradients et états attendus ; vérifie les mesures.
2. Construis l’optimizer seulement avec le poids de cette tête, en omettant volontairement son biais entraînable. L’audit doit signaler ce biais absent.
3. Sur la miniature LoRA du module 16, compare r=1 et r=2 à base et cibles fixes. Vérifie le coût théorique et les noms réels.
4. Mesure les gradients immédiatement après `zero_grad(set_to_none=True)`. Explique pourquoi leur mémoire de tenseurs peut alors être nulle sans que l’architecture ait changé.

## 8. Exercices avec corrigés

### Exercice 1 — Ratio exact

Un modèle adapté totalise 1.2 milliard de paramètres, dont 12 millions entraînables. Quel ratio ?

<details>
<summary>Corrigé</summary>

`12M/1200M = 0.01`, donc **1 %**. Ici le total annoncé inclut déjà tous les paramètres présents.

</details>

### Exercice 2 — Absence de gradient

Un paramètre a `requires_grad=True` mais `.grad=None` après backward. Quelles vérifications faire ?

<details>
<summary>Corrigé</summary>

Vérifier sa participation au calcul de loss, le suivi autograd et le fait que l’on lit un paramètre feuille. Vérifier aussi quand les gradients sont remis à zéro ; leur absence n’est pas une mesure de qualité.

</details>

### Exercice 3 — Optimizer vide

Pourquoi ses états sont-ils vides avant le premier step dans ce laboratoire ?

<details>
<summary>Corrigé</summary>

AdamW initialise ici ses états de manière différée pour les paramètres concernés lorsqu’une mise à jour les utilise. Il faut mesurer au bon moment du cycle.

</details>

### Exercice 4 — Poids gelés

La base est gelée. Pourquoi sa taille reste-t-elle incluse dans la mémoire des poids ?

<details>
<summary>Corrigé</summary>

Elle est encore stockée et nécessaire au forward. Le gel limite les mises à jour et leurs états ; il ne supprime pas la contribution de base.

</details>

## 9. Validation — avant de continuer

- [ ] Je calcule total, entraînable, gelé et ratio avec un dénominateur annoncé.
- [ ] Je compare les flags aux groupes de l’optimizer.
- [ ] Je distingue gradient absent et paramètre non optimisé.
- [ ] Je mesure les états après le premier step.
- [ ] Je compare les comptes LoRA théoriques et réels.
- [ ] Je peux expliquer les limites de ces octets et du ratio.

## Fiche mémo

```text
Compter → inspecter les noms → vérifier les groupes → backward → inspecter les états.
Le coût de l’optimizer ne se voit pas forcément avant sa première mise à jour.
Ratio entraînable et économie de mémoire totale sont deux mesures différentes.
```

Le [module 18](18_rank_lora_alpha_et_dropout.md) étudie les réglages qui changent la taille et la dynamique des adapters.

## Référence

[PyTorch : AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html) — règle de mise à jour et paramètres d’optimizer.
