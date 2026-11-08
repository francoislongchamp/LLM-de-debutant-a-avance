# Module 24 — Économiser des activations en recalculant

## 1. Ce que tu vas apprendre

Le gradient checkpointing réduit certaines valeurs intermédiaires conservées pour backward, en acceptant de refaire une partie du forward. Il ne compresse pas les poids et ne sauvegarde pas le modèle sur disque.

Tu vas observer la recomputation, comparer les gradients et distinguer une mesure pédagogique de tenseurs sauvegardés d’un vrai pic mémoire.

**Prérequis :** modules 5, 7, 17 et 23.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** PyTorch sur CPU pour le laboratoire. La mesure de VRAM nécessite ensuite un GPU accessible.

## 2. Pourquoi backward conserve-t-il des valeurs ?

Pour dériver une opération, autograd peut avoir besoin de son entrée, de sa sortie ou d’une autre information calculée pendant le forward. Ces valeurs sauvegardées contribuent au budget des **activations**.

```text
Sans recomputation : calculer → garder des intermédiaires → réutiliser au backward
Avec recomputation : garder des points utiles → rejouer un segment au backward
```

Les poids restent présents. Les gradients et états d’optimizer nécessaires ne disparaissent pas. Le gain vise principalement les intermédiaires du chemin différentiable.

## 3. Définir une frontière de checkpoint

On peut envelopper une fonction ou un bloc avec `torch.utils.checkpoint.checkpoint`. PyTorch garde ce qui permet de recalculer les valeurs requises, puis rejoue tout ou partie du segment au besoin.

L’option `use_reentrant=False` choisit ici la variante non réentrante, explicitement. Les variantes ont des contraintes différentes sur le suivi autograd, les entrées et la recomputation ; ne transpose pas leurs règles indistinctement.

Le segment rejoué doit conserver un comportement compatible avec son premier calcul. S’il dépend d’un état externe modifié entre-temps, les gradients peuvent être incorrects. Le dropout demande aussi une gestion cohérente des états aléatoires, prise en charge selon les options du mécanisme.

## 4. Micro-exemple : ce qui est conservé

Une suite de couches 32 → 64 → 32 produit notamment des entrées, sorties de linéaires et résultats d’activation. Sans checkpoint, plusieurs de ces tenseurs peuvent être sauvegardés.

Avec un checkpoint autour de cette suite, certains intermédiaires internes sont recalculés plutôt que conservés. L’entrée du segment reste nécessaire.

Le rapport mémoire/temps dépend de la taille du bloc, du batch, de la longueur et des kernels. Aucune règle « moitié de mémoire pour deux fois plus de temps » n’est universelle.

## 5. Laboratoire autonome — Voir les recomputations et vérifier les gradients

Le script utilise des hooks autograd pour compter les octets des tenseurs sauvegardés pendant le forward. C’est un **volume cumulé pédagogique**, pas une mesure de pic mémoire ou de stockages uniques : un même stockage peut être référencé plusieurs fois.

```python
from copy import deepcopy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.calls = 0

    def forward(self, x):
        self.calls += 1  # Compteur d'observation ; il ne modifie pas le calcul.
        return self.net(x)


torch.manual_seed(42)
base = Block()
input_data = torch.randn(8, 32)


def run(use_checkpoint):
    model = deepcopy(base)
    x = input_data.clone().requires_grad_(True)
    saved_sizes = []

    def pack(tensor):
        saved_sizes.append(tensor.numel() * tensor.element_size())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = checkpoint(model, x, use_reentrant=False) if use_checkpoint else model(x)
        loss = output.square().mean()
    forward_saved_bytes = sum(saved_sizes)
    loss.backward()
    gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    return loss.item(), gradients, forward_saved_bytes, model.calls, x.grad


plain = run(False)
checked = run(True)
print('Loss sans/avec :', plain[0], checked[0])
print('Volume de tenseurs sauvegardes au forward :', plain[2], checked[2])
print('Appels du bloc sans/avec :', plain[3], checked[3])
print('Gradients des parametres equivalents :',
      all(torch.allclose(a, b, atol=1e-6) for a, b in zip(plain[1], checked[1])))
print('Gradient entree equivalent :', torch.allclose(plain[4], checked[4], atol=1e-6))
```

Tu dois observer des losses identiques et des gradients équivalents. Le compteur montre un seul appel du bloc dans le chemin ordinaire et une recomputation avec checkpointing.

Le volume sauvegardé pendant le forward est plus faible pour cette configuration. Il ne faut pas le présenter comme « X octets de VRAM totale économisés » : les hooks ne mesurent pas l’allocator, les poids, les états d’optimizer et tous les temporaires.

La variante non réentrante peut s’arrêter de recalculer un segment dès que les intermédiaires nécessaires ont été obtenus. Compter des appels n’est donc pas mesurer exactement un doublement des FLOPs.

## 6. Mesurer le vrai compromis sur GPU

Pour une expérience réelle :

1. Fixe modèle, poids initiaux, données, batch, séquence, précision et objectif.
2. Fais une mise en route comparable avant les mesures, car les premières allocations et kernels peuvent être particuliers.
3. Réinitialise le compteur de pic mémoire au début de la région mesurée.
4. Synchronise l’accélérateur avant et après le chronométrage.
5. Mesure un step complet puis plusieurs steps représentatifs.
6. Rapporte pic alloué, mémoire réservée si utile, temps et tokens réels par seconde.

Réalise les deux configurations dans des conditions comparables, idéalement dans des processus séparés. Un simple coup d’œil à la mémoire après backward manque souvent le pic du forward ou du step.

Si tu profites du gain pour doubler la séquence, tu mesures un nouveau compromis. Ne l’utilise pas pour attribuer toute la différence au seul checkpointing.

## 7. Lien avec LoRA, QLoRA et le cache

Un adapter petit peut encore nécessiter des activations de nombreux blocs. Le checkpointing peut donc rester utile avec LoRA ou QLoRA.

Dans un causal LM, le KV cache de génération et la recomputation d’entraînement ont des objectifs différents. Des workflows désactivent `use_cache` pendant le training avec checkpointing. Suis les contraintes du modèle et du framework ; ne confonds pas ce cache et les intermédiaires conservés pour backward.

Une option `gradient_checkpointing=True` dans un trainer active un mécanisme du modèle compatible, pas une transformation universelle de n’importe quelle fonction Python.

## 8. Deux sens du mot checkpoint

| Terme | Ce qui est conservé | Objectif |
| --- | --- | --- |
| Checkpoint de modèle | Poids et éventuellement états sur disque | Sauvegarder/reprendre un run |
| Gradient/activation checkpointing | Points utiles du calcul en mémoire | Réduire certaines activations sauvegardées |

Tu peux avoir l’un sans l’autre. Un training avec recomputation et sans sauvegarde de poids reste impossible à reprendre après arrêt du processus.

## 9. Expériences guidées

1. Double le batch du laboratoire et compare les volumes pédagogiques. Ne déduis pas un pic VRAM d’une simple proportion.
2. Élargis le bloc ou ajoute plusieurs couches. Observe comment le placement de la frontière modifie le coût de recomputation et les intermédiaires conservés.
3. Sur un modèle compatible, compare 20 steps sans/avec à paramètres identiques, puis choisis un critère : tenir en mémoire, augmenter la séquence ou améliorer le débit global.
4. Ajoute du dropout dans une copie et vérifie les gradients avec une gestion contrôlée du hasard ; examine la documentation avant de changer les options de préservation RNG.

## 10. Exercices avec corrigés

### Exercice 1 — Quel poste ?

Le checkpointing divise-t-il le nombre de paramètres entraînables ?

<details>
<summary>Corrigé</summary>

Non. Il modifie ce qui est conservé ou recalculé pour backward. Les mêmes paramètres peuvent être optimisés, avec moins d’intermédiaires sauvegardés.

</details>

### Exercice 2 — Même fonction

Pourquoi un état global modifié entre forward et backward peut-il poser problème ?

<details>
<summary>Corrigé</summary>

La recomputation doit fournir des intermédiaires compatibles avec le calcul initial. Si la fonction rejouée ne fait plus le même calcul, les gradients peuvent ne plus correspondre à la loss évaluée.

</details>

### Exercice 3 — Mesure insuffisante

Tu compares la mémoire après les deux runs. Pourquoi cela ne mesure-t-il pas forcément le gain au pic ?

<details>
<summary>Corrigé</summary>

Des allocations temporaires ont pu être libérées avant la mesure. Il faut un compteur de pic sur une région définie et des conditions comparables.

</details>

### Exercice 4 — Deux checkpoints

Activer gradient checkpointing conserve-t-il un fichier de reprise ?

<details>
<summary>Corrigé</summary>

Non. La sauvegarde du modèle et des états d’entraînement est un mécanisme séparé, à configurer explicitement.

</details>

## 11. Validation — avant de continuer

- [ ] Je définis activation, frontière et recomputation.
- [ ] Je montre une équivalence des gradients du laboratoire.
- [ ] Je distingue volume sauvegardé pédagogique et pic réel.
- [ ] Je peux expliquer les contraintes d’un segment rejoué.
- [ ] Je sais mesurer mémoire et débit de façon comparable.
- [ ] Je distingue sauvegarde sur disque, KV cache et checkpointing d’activations.

## Fiche mémo

```text
Conserver moins d’intermédiaires ; refaire du calcul au backward.
Même objectif et paramètres, autre compromis mémoire/temps.
Comparer un pic mesuré, pas seulement la mémoire en fin de run.
```

Le [module 25](25_continued_pretraining.md) change le type de données et étudie le continued pretraining.

## Référence

[PyTorch : `torch.utils.checkpoint`](https://docs.pytorch.org/docs/stable/checkpoint.html) — variantes, recomputation et gestion des états aléatoires.
