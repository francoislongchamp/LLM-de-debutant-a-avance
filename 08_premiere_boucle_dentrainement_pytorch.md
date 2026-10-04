# Module 8 — Première boucle d’entraînement PyTorch

## Pourquoi ce module arrive ici

Avant d’utiliser des trainers de haut niveau, tu dois savoir ce qu’ils automatisent. Une boucle minimale rend explicites forward, loss, backward et update.

## Objectifs

Écrire et expliquer une boucle d’entraînement, distinguer epoch, batch et step, utiliser train/eval, remettre les gradients à zéro et suivre une loss.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fondations

## Définitions concrètes

### Epoch

**Définition concrète.** Passage complet approximatif sur le dataset.


**Exemple simple.** 3 epochs ≈ chaque exemple vu trois fois.

### Batch

**Définition concrète.** Sous-ensemble traité ensemble.


**Exemple simple.** Batch size 8 = huit exemples dans une passe locale.

### Optimizer step

**Définition concrète.** Moment où les paramètres sont effectivement mis à jour.


**Exemple simple.** Plusieurs micro-batches peuvent précéder un step.

### Optimizer

**Définition concrète.** Algorithme transformant gradients en mises à jour.


**Exemple simple.** AdamW utilise des statistiques historiques des gradients.

### Learning rate

**Définition concrète.** Échelle globale de la mise à jour.


**Exemple simple.** 1e-3 est 100× 1e-5.

### Train/eval mode

**Définition concrète.** Modes influençant certains modules comme dropout.


**Exemple simple.** `eval()` ne désactive pas autograd.

## Intuition simple

Le cycle est : prédire → mesurer l’erreur → calculer la responsabilité des poids → corriger légèrement → recommencer.

## Ce qui se passe réellement sous le capot

1. Prendre un batch.
2. `model.train()`.
3. Forward.
4. Calcul de loss.
5. Remise à zéro des anciens gradients.
6. Backward.
7. Optionnel : clipping.
8. Optimizer step.
9. Logging.
10. Pour validation : `model.eval()` et sans gradient.

## Exemple minimal à comprendre mentalement

Apprendre `y=2x`. Au départ `x=4` peut produire 0,7 au lieu de 8. En répétant les corrections, le poids de la couche linéaire se rapproche de 2.

## Formules et notation utiles

Régression jouet : `ŷ=wx+b`, puis MSE :

\[
L=\frac1N\sum_i(\hat y_i-y_i)^2
\]

Le LLM utilise une autre architecture et une autre loss, mais le cycle d’optimisation reste le même.

## Code minimal observable

```python
import torch

x = torch.tensor([[1.],[2.],[3.],[4.]])
y = 2*x
model = torch.nn.Linear(1,1)
opt = torch.optim.AdamW(model.parameters(), lr=0.05)

for step in range(500):
    model.train()
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(step, loss.item())

model.eval()
with torch.no_grad():
    print("10 ->", model(torch.tensor([[10.]])).item())
```

## Laboratoire guidé

1. Affiche poids et biais tous les 100 steps.  
2. Teste un LR minuscule, raisonnable, puis trop grand.  
3. Sépare un exemple de validation.  
4. Remplace AdamW par SGD et compare.  
5. Réécris la boucle de mémoire.

## Ce que tu dois observer

- La loss peut osciller sur des problèmes plus complexes.
- LR trop faible ralentit; trop fort peut diverger.
- Train/eval et autograd sont deux axes distincts.

## À ne pas confondre

- Epoch ≠ optimizer step.
- Batch size ≠ gradient accumulation.
- `model.eval()` ≠ mesure de qualité.
- Optimizer ≠ loss.

## Erreurs fréquentes

- Oublier `zero_grad()` hors accumulation intentionnelle.
- Évaluer uniquement sur le train.
- Changer plusieurs hyperparamètres simultanément.

## Exercices

1. Ajoute un DataLoader batch size 2.
2. Compte les optimizer steps pour 100 exemples, batch 10, 3 epochs.
3. Explique chaque ligne de la boucle sans jargon.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Écrire l’ordre forward → loss → zero_grad → backward → step.
- Définir batch/epoch/step.
- Expliquer le learning rate.
- Écrire une boucle validation séparée.

## Fiche mémo

Les frameworks de fine-tuning orchestrent cette boucle; la comprendre permet de diagnostiquer les problèmes au lieu de seulement modifier des paramètres au hasard.

## Lien avec le module suivant

Le module 9 passe à la matière première du fine-tuning : les données.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
