# Module 23 — Gradient accumulation et batch effectif

## Pourquoi ce module arrive ici

La VRAM limite souvent le nombre de séquences simultanées. L’accumulation permet de faire plusieurs micro-batches avant une mise à jour et d’obtenir un batch effectif plus grand.

## Objectifs

Distinguer micro-batch, batch par device, gradient accumulation, nombre de devices et batch effectif. Calculer optimizer steps et comprendre l’effet sur la dynamique d’entraînement.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Optimisation mémoire

## Définitions concrètes

### Micro-batch

**Définition concrète.** Exemples traités dans un seul forward/backward local.


**Exemple simple.** 1 exemple par GPU.

### Accumulation steps

**Définition concrète.** Nombre de micro-batches dont les gradients sont accumulés avant `optimizer.step()`.


**Exemple simple.** 16 passages avant update.

### Global/effective batch

**Définition concrète.** Nombre total approximatif d’exemples contribuant à une update sur tous les devices.


**Exemple simple.** batch/device × devices × accumulation.

### Optimizer step

**Définition concrète.** Une mise à jour effective des paramètres.


**Exemple simple.** Se produit après le nombre configuré d’accumulations.

## Intuition simple

Si tu ne peux porter qu’une boîte à la fois, tu peux déposer 16 boîtes au même endroit avant de faire le voyage final. L’accumulation n’augmente pas la mémoire simultanée de 16×; elle retarde la mise à jour pour intégrer plusieurs micro-batches.

## Ce qui se passe réellement sous le capot

1. Forward/backward sur micro-batch 1.
2. Conserver les gradients.
3. Forward/backward sur micro-batch 2, etc.
4. Les frameworks normalisent généralement la loss/gradients selon leur implémentation pour reproduire un batch logique.
5. Après N accumulations, optimizer step.
6. Remise à zéro puis nouveau cycle.

## Exemple minimal à comprendre mentalement

```text
per_device_batch = 2
GPUs             = 4
accumulation     = 8

effective batch = 2×4×8 = 64 exemples/update
```

Avec 6400 exemples et un epoch, cela représente environ 100 optimizer steps si tout est divisible et sans subtilités de sampler.

## Formules et notation utiles

\[
B_{effective}=B_{device}\times N_{devices}\times A
\]

Nombre approximatif de steps/epoch :

\[
steps\approx \frac{N_{examples}}{B_{effective}}
\]

## Code minimal observable

```python
def effective_batch(per_device, devices, accumulation):
    return per_device * devices * accumulation

print(effective_batch(2, 4, 8))  # 64
```

## Laboratoire guidé

1. Calcule ton batch effectif actuel.  
2. Observe la fréquence de `optimizer.step()` ou du global step.  
3. Garde batch effectif constant avec deux combinaisons micro-batch/accumulation différentes si ta mémoire le permet.  
4. Compare débit et stabilité.  
5. Documente aussi le nombre de tokens par batch, car des séquences de longueurs très différentes rendent “64 exemples” moins informatif.

## Ce que tu dois observer

- Accumulation augmente le temps entre updates.
- Même nombre d’exemples ≠ même nombre de tokens.
- Des implémentations distribuées peuvent introduire des détails de synchronisation.

## À ne pas confondre

- Batch effectif ≠ micro-batch.
- Accumulation ≠ data parallel.
- Augmenter batch effectif ≠ toujours améliorer la qualité.

## Erreurs fréquentes

- Comparer learning rates sans considérer batch effectif.
- Oublier que les derniers batches peuvent ne pas remplir parfaitement la formule.
- Ne compter que les exemples pour des séquences très variables.

## Exercices

1. Calcule le batch effectif pour 1×2 GPUs×32 accumulation.
2. Avec 10 000 exemples et batch effectif 80, environ combien de steps/epoch ?
3. Explique pourquoi tokens/update peut être une métrique plus précise.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Calculer batch effectif.
- Distinguer micro-batch et optimizer step.
- Expliquer pourquoi accumulation aide la VRAM.
- Calculer approximativement les steps.

## Fiche mémo

L’accumulation échange du temps entre updates contre un batch logique plus grand, sans exiger que tous les exemples soient simultanément en mémoire.

## Lien avec le module suivant

Le module 24 économise une autre grosse composante de VRAM : les activations.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
