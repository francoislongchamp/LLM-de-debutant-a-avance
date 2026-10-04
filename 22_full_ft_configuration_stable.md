# Module 22 — Full FT — configuration stable

## Pourquoi ce module arrive ici

Une configuration qui “tourne” n’est pas forcément stable. Full FT exige une attention particulière au learning rate, warmup, scheduler, clipping, précision et validation.

## Objectifs

Construire une configuration full FT prudente, comprendre warmup/scheduler/weight decay/clipping et diagnostiquer divergence, NaN ou mise à jour trop agressive.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning complet

## Définitions concrètes

### Warmup

**Définition concrète.** Période où le learning rate monte progressivement depuis une petite valeur.


**Exemple simple.** Les premiers 3 % des steps peuvent servir de warmup.

### Scheduler

**Définition concrète.** Règle faisant évoluer le learning rate pendant le training.


**Exemple simple.** Cosine ou linear decay.

### Weight decay

**Définition concrète.** Terme de régularisation appliqué à certains poids selon l’optimizer/configuration.


**Exemple simple.** AdamW sépare conceptuellement decay et gradient update.

### Gradient clipping

**Définition concrète.** Limitation de la norme des gradients avant l’update.


**Exemple simple.** `max_grad_norm=1.0`.

### Mixed precision

**Définition concrète.** Calcul avec des formats comme BF16/FP16 afin de réduire mémoire/augmenter débit.


**Exemple simple.** BF16 possède une plage d’exposant utile pour la stabilité sur matériel compatible.

## Intuition simple

Full FT ressemble à piloter un véhicule plus puissant : une commande trop forte modifie tout le système. Un LR plus prudent, un warmup et la surveillance des gradients réduisent les mises à jour brutales.

## Ce qui se passe réellement sous le capot

1. Commencer avec un petit modèle/dataset validés.
2. Choisir une précision supportée matériellement.
3. Utiliser un LR prudent et un warmup.
4. Suivre loss, learning rate et gradient norm.
5. Clipper les gradients si configuré.
6. Évaluer régulièrement sur validation.
7. Sauvegarder plusieurs checkpoints.
8. Arrêter/diagnostiquer si loss non finie, gradients explosifs ou validation se dégrade.

## Exemple minimal à comprendre mentalement

Deux expériences identiques sauf LR :

```text
LR 5e-6 : loss descend progressivement
LR 5e-4 : loss devient instable/NaN ou le modèle régresse fortement
```

Ce n’est pas une règle chiffrée universelle; c’est l’illustration qu’un ordre de grandeur peut complètement changer la dynamique.

## Formules et notation utiles

Clipping par norme globale : si `||g|| > C`, on redimensionne approximativement le gradient pour que sa norme soit limitée à `C`. Cela évite une update exceptionnellement énorme sans résoudre la cause fondamentale.

## Code minimal observable

```python
from trl import SFTConfig

args = SFTConfig(
    output_dir="outputs/fullft",
    learning_rate=5e-6,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    logging_steps=5,
    gradient_checkpointing=True,
)
```

## Laboratoire guidé

1. Lance une configuration prudente sur un petit modèle.  
2. Logge LR, loss et grad_norm.  
3. Duplique l’expérience avec un LR 10× plus grand pour observer, sur un test court et contrôlé, la sensibilité.  
4. Compare validation et benchmark, pas seulement train loss.  
5. Reviens à la config stable et documente ton choix.

## Ce que tu dois observer

- Warmup modifie surtout le début du training.
- Une loss non finie exige un diagnostic immédiat.
- Gradient clipping limite un symptôme mais n’est pas une solution universelle.
- La précision choisie dépend du GPU et de la pile logicielle.

## À ne pas confondre

- Warmup ≠ scheduler entier.
- Weight decay ≠ dropout.
- Gradient clipping ≠ baisse automatique du learning rate.

## Erreurs fréquentes

- Augmenter LR pour “aller plus vite”.
- Continuer un training avec NaN en espérant qu’il se répare.
- Choisir une précision non adaptée au matériel.

## Exercices

1. Dessine un LR avec warmup puis cosine decay.
2. Explique ce que signifie grad_norm=1000 comparé à 1, sans conclure automatiquement qu’un nombre est mauvais.
3. Propose un ordre de diagnostic pour une loss NaN.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer warmup, scheduler, weight decay et clipping.
- Lire une courbe learning rate/loss.
- Diagnostiquer au moins trois causes possibles d’instabilité.
- Construire une config prudente et mesurée.

## Fiche mémo

La stabilité vient d’une combinaison : données valides, LR adapté, précision compatible, gradients surveillés et validation régulière.

## Lien avec le module suivant

Le module 23 explique comment augmenter le batch effectif sans faire tenir plus d’exemples simultanément en VRAM.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
