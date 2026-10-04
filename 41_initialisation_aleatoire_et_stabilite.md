# Module 41 — Initialisation aléatoire et stabilité

## Pourquoi ce module arrive ici

Un modèle from scratch commence sans structure apprise. Si les poids initiaux ont de mauvaises échelles, les activations ou gradients peuvent disparaître/exploser avant que l’apprentissage ne commence réellement.

## Objectifs

Comprendre pourquoi on initialise avec de petites distributions adaptées, vérifier statistiques des poids/activations et reconnaître NaN, explosion ou symétrie problématique.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — optimisation

## Définitions concrètes

### Initialization distribution

**Définition concrète.** Règle générant les valeurs initiales des poids.


**Exemple simple.** Normale de faible écart-type selon l’architecture.

### Symmetry breaking

**Définition concrète.** Différences initiales permettant à des neurones similaires d’apprendre des fonctions différentes.


**Exemple simple.** Tous les poids à zéro seraient problématiques dans de nombreux réseaux.

### Activation scale

**Définition concrète.** Ordre de grandeur des valeurs dans le forward.


**Exemple simple.** Valeurs qui explosent couche après couche signalent un problème.

### Gradient scale

**Définition concrète.** Ordre de grandeur des gradients.


**Exemple simple.** Très grands, très petits ou NaN sont à surveiller.

### Finite check

**Définition concrète.** Vérification que valeurs/loss ne sont ni NaN ni ±inf.


**Exemple simple.** `torch.isfinite(loss)`.

## Intuition simple

Au départ, le réseau est un système de transformations aléatoires. Les valeurs doivent être assez petites pour ne pas exploser, mais suffisamment variées pour que les unités ne soient pas toutes identiques.

## Ce qui se passe réellement sous le capot

1. Instancier le modèle avec l’initialisation de son implémentation.
2. Inspecter moyenne/écart-type de quelques poids.
3. Faire un forward sur un batch réel tokenisé.
4. Vérifier logits et loss finis.
5. Faire backward et inspecter gradient norms.
6. Tester quelques dizaines de steps avant tout long run.
7. Ne modifier l’initialisation qu’avec une hypothèse claire.

## Exemple minimal à comprendre mentalement

Deux extrêmes pédagogiques :

```text
Poids énormes → activations/logits énormes → softmax/loss instables
Poids tous identiques → faible diversité/symétrie problématique
```

Les architectures modernes possèdent leurs conventions; les réimplémenter “à la main” sans raison peut casser cette stabilité.

## Formules et notation utiles

Les méthodes Xavier/He et variantes cherchent à contrôler la variance à travers les couches. Pour les Transformers modernes, suivre l’initialisation prévue par l’architecture est un bon point de départ pédagogique; les détails avancés dépendent du design (résidus, profondeur, normes).

## Code minimal observable

```python
import torch

for name, p in list(model.named_parameters())[:20]:
    if p.ndim >= 2:
        print(name, "mean", p.float().mean().item(), "std", p.float().std().item())

# après calcul de loss
print("loss finite?", torch.isfinite(loss).item())
```

## Laboratoire guidé

1. Inspecte stats de 10 matrices.  
2. Fais forward/backward sur 3 batches.  
3. Logge loss et global grad norm.  
4. Fais volontairement une copie expérimentale où tu multiplies certains poids par 100 et observe l’instabilité; ne réutilise pas ce checkpoint.  
5. Reviens à l’initialisation normale et confirme la stabilité.

## Ce que tu dois observer

- La moyenne proche de zéro ne suffit pas à garantir une bonne initialisation.
- Les premières dizaines de steps sont un excellent smoke test.
- Les NaN peuvent venir des données, du LR, de la précision ou d’autres causes, pas seulement de l’initialisation.

## À ne pas confondre

- Initialisation ≠ seed; la seed rend l’aléatoire reproductible, l’initialisation définit sa distribution/règle.
- Small weights ≠ small model.
- Loss élevée au step 0 ≠ automatiquement instable.

## Erreurs fréquentes

- Modifier l’initialisation avant d’avoir testé la valeur par défaut.
- Lancer longtemps sans `isfinite`/logging.
- Attribuer toute NaN à CUDA sans inspecter données et gradients.

## Exercices

1. Explique pourquoi tous les poids à la même valeur sont risqués.
2. Liste quatre causes d’une loss NaN.
3. Explique seed vs initialization.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Inspecter stats de poids.
- Vérifier loss/gradients finis.
- Expliquer symmetry breaking.
- Faire un smoke test avant long training.

## Fiche mémo

Une bonne initialisation donne un point de départ **numériquement stable et non symétrique**, pas des connaissances.

## Lien avec le module suivant

Le module 42 écrit la vraie boucle de pré-entraînement causal sur tokens.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
