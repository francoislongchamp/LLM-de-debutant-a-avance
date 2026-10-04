# Module 50 — Budget mémoire du training

## Pourquoi ce module arrive ici

Les erreurs OOM viennent souvent d’une estimation qui ne comptait que les poids. Pour dimensionner une expérience, il faut séparer poids, gradients, optimizer, activations, temporaires, quantification et sharding.

## Objectifs

Construire un tableau de budget mémoire, comprendre les composantes dominantes selon LoRA/full FT/contexte et mesurer peak allocated/reserved plutôt que deviner.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Training distribué

## Définitions concrètes

### Weights memory

**Définition concrète.** Mémoire des paramètres du modèle.


**Exemple simple.** Nparams×bytes selon stockage.

### Gradient memory

**Définition concrète.** Mémoire des gradients des paramètres entraînables.


**Exemple simple.** Grande en full FT, petite pour adapters.

### Optimizer memory

**Définition concrète.** États maintenus par l’optimizer pour paramètres entraînables.


**Exemple simple.** Moments Adam, éventuellement master weights selon stack.

### Activation memory

**Définition concrète.** Intermédiaires du forward/backward dépendant fortement de batch et séquence.


**Exemple simple.** Peut dominer les longs contextes.

### Temporary buffers

**Définition concrète.** Espaces de travail des kernels/collectives.


**Exemple simple.** FlashAttention, GEMM, all-gather, etc.

### Allocated vs reserved

**Définition concrète.** Mémoire réellement allouée à des tenseurs vs pool réservé par l’allocator.


**Exemple simple.** PyTorch peut réserver plus que ce qui est actif.

## Intuition simple

Pense à une valise : les poids ne sont qu’un type d’objet. Il faut aussi emporter gradients, carnets de l’optimizer et toutes les notes intermédiaires du forward. Une estimation qui compte uniquement les vêtements oublie le reste de la valise.

## Ce qui se passe réellement sous le capot

1. Calculer mémoire brute des poids.
2. Identifier paramètres entraînables et dtype gradients.
3. Identifier l’optimizer et ses états réels dans la stack choisie.
4. Mesurer activations via un run représentatif.
5. Ajouter overhead/temporaires.
6. Mesurer `max_memory_allocated` et `max_memory_reserved`.
7. Tester avec séquence/batch cible, pas un petit exemple irréaliste.
8. Introduire une marge opérationnelle plutôt que viser 100 % de VRAM.

## Exemple minimal à comprendre mentalement

Deux expériences sur le même modèle :

```text
QLoRA, seq=512, batch=1  → base compacte, petites gradients
Full FT, seq=4096,batch=2 → poids+grad+optimizer+grosses activations
```

Le second peut consommer un ordre de grandeur supplémentaire même si le nombre de paramètres du modèle n’a pas changé.

## Formules et notation utiles

Poids bruts : `N×bytes`. Mais le budget total est mieux représenté comme :

```text
M_total ≈ weights + gradients + optimizer + activations + temporaries + fragmentation
```

Les derniers termes sont souvent mesurés empiriquement.

## Code minimal observable

```python
import torch

torch.cuda.reset_peak_memory_stats()
# exécuter ici une étape complète forward/backward/step
print("peak allocated GiB:", torch.cuda.max_memory_allocated()/1024**3)
print("peak reserved  GiB:", torch.cuda.max_memory_reserved()/1024**3)
```

## Laboratoire guidé

1. Crée un tableau `component / estimate / measured`.  
2. Mesure peak pour inference, LoRA et full FT d’un petit modèle si possible.  
3. Double la sequence length à batch constant et observe.  
4. Active gradient checkpointing et observe.  
5. Active QLoRA et observe la composante base.  
6. Sur FSDP, mesure par rank plutôt que seulement le total système.

## Ce que tu dois observer

- Activations peuvent dominer aux longs contextes.
- Reserved est souvent supérieur à allocated.
- Une petite modification de séquence/batch peut déclencher un OOM proche de la limite.

## À ne pas confondre

- VRAM `nvidia-smi` ≠ `max_memory_allocated`.
- Poids 4-bit ≠ activations 4-bit.
- Paramètres gelés ≠ zéro mémoire.

## Erreurs fréquentes

- Dimensionner avec un forward inference alors qu’on veut backward.
- Ne pas inclure la longueur cible.
- Viser exactement 100 % de VRAM sans marge.

## Exercices

1. Crée un budget qualitatif pour LoRA vs full FT.
2. Explique quelle composante gradient checkpointing vise.
3. Explique quelle composante QLoRA vise principalement.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Lister toutes les grandes composantes mémoire.
- Mesurer peak memory.
- Expliquer l’effet de LoRA/QLoRA/checkpointing/FSDP sur des composantes distinctes.
- Prévoir une marge avant un long run.

## Fiche mémo

Les techniques mémoire sont complémentaires parce qu’elles ciblent **des composantes différentes** du budget.

## Lien avec le module suivant

Le module 51 garantit que ce calcul coûteux n’est pas perdu en cas d’arrêt : checkpoints et reprise.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
