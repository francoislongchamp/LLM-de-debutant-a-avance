# Module 24 — Gradient checkpointing et mémoire

## Pourquoi ce module arrive ici

Pendant le forward, le backward a besoin d’informations intermédiaires appelées activations. Sur de longs contextes, elles peuvent dominer la mémoire. Le gradient checkpointing échange du calcul supplémentaire contre moins d’activations conservées.

## Objectifs

Comprendre activation, recomputation, checkpoint boundary et compromis mémoire/temps. Mesurer l’effet réel du gradient checkpointing.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Optimisation mémoire

## Définitions concrètes

### Activation

**Définition concrète.** Valeur intermédiaire produite pendant le forward et nécessaire à certains gradients.


**Exemple simple.** Hidden states, sorties de sous-opérations, etc.

### Gradient checkpointing

**Définition concrète.** Stratégie qui ne conserve qu’une partie des activations et recalcule certaines autres pendant backward.


**Exemple simple.** Moins de VRAM, plus de compute.

### Recomputation

**Définition concrète.** Nouveau calcul d’une portion du forward pendant backward.


**Exemple simple.** Le bloc est exécuté une seconde fois au besoin.

### Peak memory

**Définition concrète.** Pic maximal de mémoire allouée durant l’étape.


**Exemple simple.** Plus utile que la mémoire observée à un instant arbitraire.

## Intuition simple

Au lieu de conserver toutes les étapes d’un calcul sur des feuilles, tu gardes seulement quelques points de contrôle. Quand tu as besoin du détail, tu le recalcules depuis le dernier point. Tu utilises moins de papier, mais tu refais du travail.

## Ce qui se passe réellement sous le capot

1. Sans checkpointing, le forward garde de nombreuses activations.
2. Avec checkpointing, certaines activations internes ne sont pas stockées.
3. Pendant backward, le framework rejoue des segments du forward.
4. Les gradients finaux restent calculables.
5. Le pic mémoire diminue généralement, tandis que le temps de step augmente.

## Exemple minimal à comprendre mentalement

Sans checkpointing :

```text
Forward: calcule A→B→C→D et conserve B,C,D utiles
Backward: réutilise les valeurs
```

Avec checkpointing :

```text
Forward: conserve seulement certains points
Backward: recalcule B→C avant d’en dériver les gradients
```

## Formules et notation utiles

Il n’existe pas un ratio universel mémoire/temps : il dépend de l’architecture, du placement des checkpoints, de la longueur de séquence et des kernels. La bonne pratique pédagogique est donc de **mesurer** peak VRAM et tokens/s.

## Code minimal observable

```python
from trl import SFTConfig

args = SFTConfig(
    output_dir="outputs/gc",
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
)
```

## Laboratoire guidé

1. Fixe modèle, batch, séquence et dataset.  
2. Mesure peak VRAM et temps pour 20 steps sans checkpointing.  
3. Recommence avec checkpointing.  
4. Compare perte de débit et gain mémoire.  
5. Si la mémoire gagnée permet d’augmenter la séquence ou micro-batch, teste ce nouveau compromis.

## Ce que tu dois observer

- Le gain mémoire dépend du modèle et de la longueur.
- Le training ralentit généralement par recomputation.
- Le gain peut permettre un contexte plus long ou un batch plus grand.

## À ne pas confondre

- Gradient checkpointing ≠ sauvegarder un checkpoint de modèle sur disque.
- Recomputation ≠ refaire tout le training.
- Moins de VRAM ≠ moins de FLOPs.

## Erreurs fréquentes

- Dire “checkpointing” sans préciser s’il s’agit de mémoire d’activation ou de sauvegarde du modèle.
- Comparer deux runs avec des longueurs de séquence différentes puis attribuer tout au checkpointing.

## Exercices

1. Explique pourquoi les activations augmentent avec la longueur de séquence.
2. Donne le compromis principal du checkpointing.
3. Propose une mesure objective de son utilité pour ton GPU.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir activation et recomputation.
- Expliquer pourquoi la VRAM baisse et le temps augmente.
- Mesurer peak VRAM et débit avant/après.
- Distinguer gradient checkpointing et model checkpoint.

## Fiche mémo

Gradient checkpointing économise **les activations** en acceptant de recalculer une partie du forward.

## Lien avec le module suivant

Le module 25 change maintenant le type de données : on revient au texte brut pour le continued pretraining.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
