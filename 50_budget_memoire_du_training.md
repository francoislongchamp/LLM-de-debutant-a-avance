# Module 50 — Estimer les postes et mesurer le pic mémoire

## 1. Ce que tu vas apprendre

Les poids ne sont qu’une partie du budget. Tu vas compter des tenseurs réels après une update, distinguer ces octets du pic de l’allocator et associer chaque technique mémoire au poste qu’elle vise.

**Prérequis :** modules 5, 17, 20, 23, 24 et 49.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** PyTorch sur CPU ; la mesure CUDA est conditionnelle à un GPU accessible.

## 2. Décomposer avant de multiplier

| Poste | Dépend principalement de |
| --- | --- |
| Poids | Paramètres uniques, représentation, partage, quantification |
| Gradients | Paramètres concernés et dtype des gradients |
| Optimizer | Algorithme, états créés, dtypes et éventuelles copies maîtresses |
| Activations | Batch, séquence, architecture et sauvegardes pour backward |
| Temporaires | Kernels, compilations, collectives et conversions |
| Cache | Mode de génération et politique de conservation |

Une hypothèse « 16 octets par paramètre » peut décrire un montage poids/gradients/moments/copie maîtresse donné ; elle ne décrit pas tous les entraînements mixtes ni les adapters.

Compter un tenseur partagé deux fois donne une autre erreur. Commence par les objets et dtypes effectivement utilisés.

## 3. Allocated, reserved et vue système

`memory_allocated` décrit les allocations de tenseurs suivies par l’allocator concerné. `memory_reserved` décrit son pool réservé ; il peut rester élevé après la libération de tenseurs.

La vue système GPU peut inclure contexte CUDA et allocations d’autres composants/processus. Une différence avec PyTorch n’est pas forcément une fuite.

`empty_cache()` ne libère pas des tenseurs encore référencés. Un pic mesuré après un simple forward d’inférence ne valide pas un premier step AdamW, dont les états sont souvent créés au premier usage.

## 4. Micro-budget de notre réseau

`Linear(16,32)` avec biais : 544 paramètres. `Linear(32,4)` avec biais : 132. Total **676**.

En FP32 : **2704 octets** de poids et, après backward, autant de gradients si tous participent.

Les deux moments AdamW ajoutent ici `2×676×4 = 5408` octets, plus les compteurs de step observés. Ce n’est pas toute la mémoire du processus : objets, activations et espaces de travail ne sont pas dans cette somme.

## 5. Laboratoire autonome — Compte réel et pic conditionnel

```python
import torch
from torch import nn


def size_bytes(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def run(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.randn(8, 16, device=device)
    y = torch.randn(8, 4, device=device)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    states = [value for state in optimizer.state.values() for value in state.values()
              if isinstance(value, torch.Tensor)]
    report = {'parameters': sum(p.numel() for p in model.parameters()),
              'weights_bytes': size_bytes(model.parameters()),
              'gradient_bytes': size_bytes(p.grad for p in model.parameters() if p.grad is not None),
              'optimizer_tensor_bytes': size_bytes(states)}
    if device.type == 'cuda':
        torch.cuda.synchronize()
        report.update(peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved())
    return report


torch.manual_seed(42)
cpu_report = run(torch.device('cpu'))
print('CPU :', cpu_report)
if torch.cuda.is_available():
    print('CUDA :', run(torch.device('cuda')))
else:
    print('Pic CUDA non mesure : aucun GPU accessible dans cet environnement.')
```

Le rapport CPU doit retrouver **676**, **2704** pour poids et gradients. Les moments et compteurs sont réellement parcourus plutôt qu’estimés à partir du seul nom d’optimizer.

Sur CUDA, les pics incluent davantage que les octets de paramètres listés. Ce petit réseau ne sert pas à comparer les performances de stratégies de LLM ; il montre les unités et le moment de mesure.

## 6. Associer les techniques aux postes

| Technique | Poste ciblé en premier | Ce qui reste |
| --- | --- | --- |
| LoRA | Paramètres optimisés, gradients et états associés | Base et chemins d’activations nécessaires |
| QLoRA | Stockage d’une base quantifiée + petite adaptation | Métadonnées, calcul et activations dans d’autres formats |
| Micro-batch plus petit | Activations simultanées | Poids et états persistants |
| Accumulation | Batch logique plus grand sans un grand forward | Gradients accumulés et calcul supplémentaire |
| Activation checkpointing | Intermédiaires conservés | Recomputation et entrées nécessaires |
| FSDP | Duplication d’états répartissables | Pics de matérialisation et communication |

Leurs effets peuvent se combiner, mais pas en multipliant naïvement leurs ratios idéaux.

## 7. Protocole d’une mesure utile

1. Fixe versions, matériel, modèle, longueur, micro-batch, précision et objectif.
2. Définis si le pic inclut chargement, première update, warmup, évaluation et sauvegarde.
3. Mesure le premier step qui initialise les états puis plusieurs steps stabilisés.
4. Synchronise pour mesurer le temps GPU et garde une convention identique.
5. Compare dans des processus séparés si nécessaire pour isoler les états vivants.
6. En distribué, rapporte chaque rank et le maximum, pas seulement la moyenne.
7. Garde une marge opérationnelle pour longueurs variables et autres phases du run.

La valeur doit être une mesure sur le batch et la séquence visés. Un petit cas qui tient ne prouve pas que le batch représentatif tiendra.

## 8. Expériences guidées

- Double batch ou longueur en changeant un seul facteur ; compare paramètres inchangés et pic d’activations.
- Mesure avant/après le premier AdamW step pour voir les états différés.
- Compare un même problème LoRA/QLoRA/full FT seulement dans des conditions explicites, avec leur qualité.
- Crée un tableau `poste / estimation / mesure / convention` et note ce qui n’est pas encore mesuré.

## 9. Exercices avec corrigés

### Exercice 1 — Unités

1 GiB égale-t-il un milliard d’octets ?

<details>
<summary>Corrigé</summary>

Non. **1 GiB=1024³ octets**, alors que 1 GB=10⁹ octets. Annonce l’unité pour éviter un écart de budget.

</details>

### Exercice 2 — Allocator

Reserved reste élevé après libération d’un tenseur. Est-ce automatiquement une fuite ?

<details>
<summary>Corrigé</summary>

Non. L’allocator peut garder un pool pour le réutiliser. Il faut distinguer références vivantes, allocated, reserved et vue système.

</details>

### Exercice 3 — Technique

Quelle méthode vise principalement les activations sauvegardées : QLoRA ou activation checkpointing ?

<details>
<summary>Corrigé</summary>

L’activation checkpointing. QLoRA vise principalement le stockage de la base et le coût des paramètres optimisés via adapters.

</details>

### Exercice 4 — Premier step

Pourquoi mesurer seulement après construction de l’optimizer sous-estime-t-il parfois sa mémoire ?

<details>
<summary>Corrigé</summary>

Certains états sont créés à la première mise à jour des paramètres concernés. Le step complet doit faire partie du test.

</details>

## 10. Validation — avant de continuer

- [ ] Je retrouve le compte et les octets du réseau miniature.
- [ ] Je distingue somme de tenseurs, allocated, reserved et mémoire système.
- [ ] Je mesure la phase qui initialise les états.
- [ ] Je sais quel poste vise chaque technique.
- [ ] Je rapporte longueur, batch, phase et unités avec le pic.

## Fiche mémo

```text
Estimer par poste, puis mesurer un step représentatif et les autres pics du run.
Aucune formule de poids seuls ne remplace ce budget complet.
```

Le [module 51](51_checkpoints_et_reprise.md) vérifie que l’état coûteux à produire peut réellement être restauré.
