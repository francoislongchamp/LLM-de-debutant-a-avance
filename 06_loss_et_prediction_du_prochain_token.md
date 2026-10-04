# Module 6 — Loss et prédiction du prochain token

## Pourquoi ce module arrive ici

Un modèle apprend uniquement si une fonction quantifie son erreur. La cross-entropy relie les logits à la probabilité accordée au vrai prochain token.

## Objectifs

Comprendre labels décalés, logits, softmax, probabilité du bon token, negative log-likelihood et cross-entropy. Interpréter une loss sans la confondre avec une note universelle de qualité.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fondations

## Définitions concrètes

### Label

**Définition concrète.** Valeur cible à prédire.


**Exemple simple.** Après `Le ciel est`, la cible peut être ` bleu`.

### Logit

**Définition concrète.** Score brut avant normalisation.


**Exemple simple.** [2.0,0.5,-1.0].

### Softmax

**Définition concrète.** Transformation de logits en distribution positive qui somme à 1.


**Exemple simple.** [2,1] → environ [0,73,0,27].

### NLL

**Définition concrète.** Pénalité `-log(p_correct)`.


**Exemple simple.** p=0,5 → ≈0,693.

### Cross-entropy

**Définition concrète.** Agrégation de la pénalité des cibles correctes.


**Exemple simple.** Faible si le modèle donne beaucoup de probabilité aux bons tokens.

### Ignore index

**Définition concrète.** Valeur de label exclue du calcul de loss.


**Exemple simple.** `-100` est couramment utilisé dans PyTorch/Transformers.

## Intuition simple

La loss ne demande pas seulement “as-tu choisi le bon token ?”. Elle demande “combien de probabilité avais-tu donné au bon token ?”. Deux modèles qui ratent l’argmax peuvent donc être très différents en qualité d’apprentissage.

## Ce qui se passe réellement sous le capot

1. Le modèle produit des logits à chaque position.
2. La sortie d’une position sert conceptuellement à prédire la position suivante.
3. On regarde la probabilité attribuée au label correct.
4. On applique `-log(p_correct)`.
5. On ignore les positions masquées.
6. On moyenne/agrège sur les positions du batch.

## Exemple minimal à comprendre mentalement

```text
Modèle A : P(bleu)=0.80 → -ln(0.80)=0.223
Modèle B : P(bleu)=0.05 → -ln(0.05)=2.996
```

Le modèle A est beaucoup moins pénalisé, même si l’on ne regarde pas encore quel token aurait été échantillonné.

## Formules et notation utiles

\[
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad L=-\log p(y)
\]

Sur plusieurs positions :

\[
L=\frac1N\sum_t -\log p(y_t)
\]

## Dimensions / formes à savoir lire

```text
logits : [batch, sequence, vocab]
labels : [batch, sequence]
loss   : scalaire
```

## Code minimal observable

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0]])
target = torch.tensor([0])
probs = torch.softmax(logits, dim=-1)
loss = F.cross_entropy(logits, target)
print(probs)
print(loss.item())
print(-torch.log(probs[0,0]).item())
```

## Laboratoire guidé

1. Rends la bonne classe plus probable et observe la loss.  
2. Rends-la presque impossible et observe.  
3. Sur un causal LM, passe `labels=input_ids`.  
4. Masque certaines positions avec `-100`.  
5. Compare loss avec et sans positions masquées.

## Ce que tu dois observer

- `-log(p)` pénalise fortement une très faible probabilité du bon token.
- Train loss peut descendre tandis que validation se dégrade.
- La loss brute est surtout comparable à conditions compatibles.

## À ne pas confondre

- Loss ≠ accuracy.
- Train loss basse ≠ bonne généralisation.
- Logit ≠ probabilité.
- Masquer un label ≠ supprimer le token de l’entrée.

## Erreurs fréquentes

- Comparer directement des losses issues de tokenizers/datasets très différents.
- Superviser le prompt alors qu’on voulait uniquement la réponse assistant.

## Exercices

1. Pourquoi `-log(1)=0` ?
2. Dessine les cibles de `Le chat dort`.
3. Quelle probabilité donne une plus faible loss : 0,7 ou 0,2 ? Pourquoi ?

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Relier logit → softmax → probabilité → NLL → loss.
- Expliquer le décalage causal des labels.
- Expliquer `-100`.
- Dire ce qu’une loss ne prouve pas.

## Fiche mémo

La cross-entropy transforme la probabilité du **bon prochain token** en signal d’erreur.

## Lien avec le module suivant

Le module 7 transforme cette erreur en gradients.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
