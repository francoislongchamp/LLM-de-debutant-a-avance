# Module 7 — Gradients et autograd

## Pourquoi ce module arrive ici

La loss ne modifie aucun poids par elle-même. Il faut connaître l’effet local de chaque paramètre sur la loss : c’est le gradient.

## Objectifs

Comprendre dérivée, gradient, graphe de calcul, `requires_grad`, `backward`, accumulation et `no_grad`. Lire un exemple à un paramètre avant de généraliser aux LLM.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fondations

## Définitions concrètes

### Dérivée

**Définition concrète.** Variation locale d’une sortie lorsqu’une entrée scalaire change légèrement.


**Exemple simple.** Pour `L=(w-3)^2`, `dL/dw=2(w-3)`.

### Gradient

**Définition concrète.** Vecteur de dérivées partielles par rapport aux paramètres.


**Exemple simple.** [∂L/∂w1,∂L/∂w2,…].

### Graphe de calcul

**Définition concrète.** Enregistrement des opérations nécessaires à la règle de chaîne.


**Exemple simple.** multiplication → somme → carré → moyenne.

### Backpropagation

**Définition concrète.** Calcul des gradients en remontant le graphe depuis la loss.


**Exemple simple.** `loss.backward()`.

### requires_grad

**Définition concrète.** Indique qu’un tenseur doit participer au calcul de gradient.


**Exemple simple.** Les paramètres entraînables l’activent.

### Accumulation

**Définition concrète.** Addition des gradients de plusieurs backward avant leur remise à zéro.


**Exemple simple.** Peut être volontaire ou accidentelle.

## Intuition simple

La loss est une altitude sur une surface gigantesque. Le gradient indique la direction de montée la plus rapide; pour descendre, on avance en sens opposé. La surface réelle a des milliards de dimensions, mais l’idée reste la même.

## Ce qui se passe réellement sous le capot

1. Le forward construit un graphe pour les opérations suivies.
2. La loss scalaire dépend des paramètres.
3. `backward()` applique la règle de chaîne.
4. Chaque paramètre reçoit `param.grad`.
5. Les gradients s’additionnent par défaut.
6. L’optimizer lit les gradients lors de `step()`.
7. On remet les gradients à zéro avant le cycle suivant lorsque souhaité.

## Exemple minimal à comprendre mentalement

\[
L=(w-3)^2
\]

Pour `w=1`, `L=4` et `dL/dw=-4`. Avec `w←w-lr×grad`, soustraire un nombre négatif augmente `w`, donc le rapproche de 3.

## Formules et notation utiles

\[
\theta_{new}=\theta_{old}-\eta\nabla_\theta L
\]

Règle de chaîne :

\[
\frac{dL}{dx}=\frac{dL}{dy}\frac{dy}{dx}
\]

## Code minimal observable

```python
import torch

w = torch.tensor(1.0, requires_grad=True)
loss = (w - 3.0)**2
loss.backward()
print("loss", loss.item())
print("grad", w.grad.item())

lr = 0.1
with torch.no_grad():
    w -= lr * w.grad
print("new w", w.item())
```

## Laboratoire guidé

1. Teste `w=1` et `w=5`; compare le signe.  
2. Fais plusieurs backward sans `zero_()` et observe l’accumulation.  
3. Teste plusieurs learning rates.  
4. Sur un petit réseau, affiche la norme de quelques gradients.  
5. Provoque volontairement un gradient très grand dans un exemple jouet et réfléchis à l’utilité du gradient clipping.

## Ce que tu dois observer

- `backward()` ne met pas à jour les poids.
- Les gradients s’accumulent.
- `no_grad()` et `eval()` ne sont pas la même chose.
- Un gradient nul peut indiquer une zone plate, un masque, un paramètre gelé ou d’autres causes.

## À ne pas confondre

- Gradient ≠ update.
- Backward ≠ optimizer step.
- Accumulation voulue ≠ oubli de remettre à zéro.

## Erreurs fréquentes

- Oublier que les gradients peuvent devenir non finis.
- Tirer une conclusion sur la qualité à partir d’un seul gradient.

## Exercices

1. Pour `L=(w-10)^2`, quel signe de gradient à w=3 et w=15 ?
2. Explique pourquoi on soustrait le gradient.
3. Explique l’effet d’un learning rate trop grand dans l’exemple scalaire.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Interpréter le signe d’une dérivée simple.
- Expliquer forward/backward/step.
- Expliquer l’accumulation.
- Lire `param.grad`.

## Fiche mémo

Loss = combien d’erreur; gradient = dans quelle direction locale modifier les paramètres.

## Lien avec le module suivant

Le module 8 assemble tout cela dans une boucle d’entraînement réelle.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
