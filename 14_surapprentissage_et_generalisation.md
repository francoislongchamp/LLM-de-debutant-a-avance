# Module 14 — Surapprentissage et généralisation

## Pourquoi ce module arrive ici

Un modèle peut mémoriser parfaitement ses exemples d’entraînement et devenir moins bon sur de nouveaux cas. Comprendre l’écart train/validation est indispensable avant de comparer LoRA et full FT.

## Objectifs

Définir underfitting, overfitting, généralisation, capacité, régularisation et early stopping. Lire des courbes train/validation et proposer des corrections adaptées.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning — évaluation

## Définitions concrètes

### Généralisation

**Définition concrète.** Capacité à réussir sur des exemples non vus mais provenant de la tâche visée.


**Exemple simple.** Répondre correctement à une formulation nouvelle.

### Overfitting

**Définition concrète.** Adaptation excessive au train au détriment de nouveaux exemples.


**Exemple simple.** Train loss baisse tandis que validation loss remonte.

### Underfitting

**Définition concrète.** Modèle/configuration incapable d’apprendre suffisamment le signal utile.


**Exemple simple.** Train et validation restent mauvaises.

### Regularization

**Définition concrète.** Technique réduisant certaines formes d’adaptation excessive.


**Exemple simple.** Weight decay, dropout, moins d’epochs, data augmentation contrôlée.

### Early stopping

**Définition concrète.** Arrêt lorsque la validation cesse de s’améliorer selon une règle.


**Exemple simple.** Conserver le checkpoint au meilleur eval loss.

## Intuition simple

Mémoriser les réponses d’un cahier d’exercices n’est pas la même chose que maîtriser la matière. La généralisation se teste sur de nouveaux exercices qui demandent les mêmes compétences.

## Ce qui se passe réellement sous le capot

1. Suivre train loss à chaque intervalle.
2. Évaluer périodiquement sur validation indépendante.
3. Observer l’écart train-validation et les métriques métier.
4. Si train continue de s’améliorer mais validation se dégrade, suspecter overfitting ou drift.
5. Réduire epochs/LR/capacité d’adaptation, améliorer données ou régulariser.
6. Revalider sur plusieurs seeds/configurations si le dataset est petit.

## Exemple minimal à comprendre mentalement

```text
Epoch 1: train 2.1, val 2.2
Epoch 2: train 1.5, val 1.7
Epoch 3: train 1.0, val 1.8
Epoch 4: train 0.6, val 2.1
```

Epoch 2 semble ici un meilleur compromis selon la validation, même si le train continue de baisser ensuite.

## Formules et notation utiles

Un “generalization gap” simple peut être suivi comme :

\[
gap=L_{val}-L_{train}
\]

Ce n’est pas une métrique universelle de qualité, mais une alerte utile lorsque les losses sont comparables.

## Code minimal observable

```python
history = [
    (1, 2.1, 2.2),
    (2, 1.5, 1.7),
    (3, 1.0, 1.8),
    (4, 0.6, 2.1),
]

best = min(history, key=lambda x: x[2])
print("best validation epoch:", best)
```

## Laboratoire guidé

1. Sur un petit dataset, entraîne volontairement trop longtemps.  
2. Enregistre train et eval loss.  
3. Repère le meilleur checkpoint selon validation.  
4. Compare ses générations avec le dernier checkpoint.  
5. Réduis les epochs ou le LR et observe.  
6. Ajoute quelques exemples de validation plus difficiles et vois si le diagnostic change.

## Ce que tu dois observer

- Train loss seule encourage à continuer même quand ce n’est plus utile.
- Les métriques métier peuvent se dégrader avant/après la validation loss selon la tâche.
- Un très petit jeu de validation produit des signaux instables.

## À ne pas confondre

- Overfitting ≠ catastrophic forgetting, même s’ils peuvent coexister.
- Validation plus mauvaise ≠ forcément bug de code.
- Régularisation ≠ garantie de généralisation.

## Erreurs fréquentes

- Choisir le dernier checkpoint par habitude.
- Utiliser un validation set contaminé.
- Modifier simultanément LR, dataset et epochs puis attribuer le résultat à un seul facteur.

## Exercices

1. Interprète trois courbes fictives : underfit, bon fit, overfit.
2. Propose quatre actions possibles contre overfitting.
3. Explique pourquoi LoRA peut parfois réduire le risque d’altérer tout le modèle sans empêcher tout overfitting.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Reconnaître un pattern d’overfitting.
- Expliquer généralisation vs mémorisation.
- Choisir un checkpoint à partir de validation.
- Proposer des corrections testables.

## Fiche mémo

Une loss de train qui baisse prouve seulement que le modèle s’adapte au train. La question utile est : **est-ce que cette adaptation se transfère aux nouveaux exemples ?**

## Lien avec le module suivant

Le module 15 introduit PEFT et LoRA : comment adapter un modèle en entraînant une petite fraction de paramètres.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
