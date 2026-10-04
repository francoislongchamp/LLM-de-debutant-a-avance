# Module 29 — Reward modeling

## Pourquoi ce module arrive ici

Certaines méthodes RL ont besoin d’un signal scalaire capable d’évaluer une réponse. Un reward model apprend à approximer des préférences à partir de données comparatives.

## Objectifs

Comprendre score scalar, reward head, pairwise loss et différence entre reward model, judge et policy. Entraîner un petit reward model et mesurer pairwise accuracy.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Post-training — reward

## Définitions concrètes

### Reward model

**Définition concrète.** Modèle produisant un score scalaire pour une réponse/context.


**Exemple simple.** R(prompt,response)=2.1.

### Reward head

**Définition concrète.** Tête produisant généralement un score au lieu d’un vocabulaire complet.


**Exemple simple.** Une projection vers 1 dimension.

### Pairwise loss

**Définition concrète.** Loss encourageant `reward(chosen)>reward(rejected)`.


**Exemple simple.** Différence de scores passée dans une fonction logistique.

### Reward accuracy

**Définition concrète.** Fraction de paires où le chosen reçoit le score supérieur.


**Exemple simple.** 75 % signifie 75 paires correctement ordonnées sur 100.

### Calibration

**Définition concrète.** Relation entre valeur numérique du score et signification probabiliste/absolue.


**Exemple simple.** Un reward 3.2 n’est pas automatiquement “deux fois meilleur” que 1.6.

## Intuition simple

Le reward model agit comme un correcteur appris : on lui montre des paires et il apprend à donner un nombre plus élevé à la réponse préférée. Ce nombre devient ensuite un signal d’optimisation possible.

## Ce qui se passe réellement sous le capot

1. Construire des paires chosen/rejected.
2. Encoder contexte+réponse.
3. Le reward model produit un score pour chaque séquence.
4. La loss pénalise les cas où chosen n’est pas supérieur.
5. Backward entraîne le reward model.
6. Validation mesure pairwise accuracy et éventuellement calibration/robustesse.
7. Le reward model est gelé lorsqu’il sert ensuite à scorer une policy RL, sauf méthode particulière.

## Exemple minimal à comprendre mentalement

```text
chosen reward   = 1.8
rejected reward = 0.2
```

Bon ordre. Si :

```text
chosen = -0.5
rejected = 0.7
```

la loss doit pousser les paramètres pour inverser cet ordre relatif.

## Formules et notation utiles

Une loss pairwise classique peut s’écrire :

\[
L=-\log\sigma(r_{chosen}-r_{rejected})
\]

Si la différence est fortement positive, la loss devient faible. Si elle est négative, la pénalité augmente.

## Code minimal observable

```python
import torch

r_chosen = torch.tensor([1.8])
r_rejected = torch.tensor([0.2])
loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
print(loss.item())
```

## Laboratoire guidé

1. Calcule la loss pour plusieurs couples de rewards.  
2. Entraîne un reward model sur un petit jeu pairwise avec l’outil adapté à ta version.  
3. Mesure pairwise accuracy train/validation.  
4. Cherche des exemples où le reward se trompe.  
5. Modifie superficiellement longueur/format d’une réponse et vérifie si le reward est facilement trompé.

## Ce que tu dois observer

- Un bon pairwise accuracy moyen peut cacher des faiblesses par catégorie.
- Les scores ne sont pas nécessairement calibrés en valeur absolue.
- Le reward model peut apprendre les mêmes raccourcis que le dataset.

## À ne pas confondre

- Reward model ≠ policy.
- Reward model ≠ référence DPO.
- Score élevé ≠ vérité garantie.

## Erreurs fréquentes

- Utiliser le reward model comme vérité absolue.
- Entraîner et évaluer sur des paires quasi identiques.
- Ignorer les attaques/raccourcis de longueur ou style.

## Exercices

1. Calcule qualitativement la loss lorsque chosen-rejected vaut +5, 0 et -5.
2. Explique pourquoi seul l’ordre peut être appris sans calibration absolue.
3. Propose trois tests de robustesse du reward model.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir reward model et reward head.
- Expliquer la pairwise loss.
- Mesurer pairwise accuracy.
- Identifier au moins un raccourci possible.

## Fiche mémo

Le reward model transforme des préférences en une **fonction de score apprise**, utile mais imparfaite.

## Lien avec le module suivant

Le module 30 introduit les concepts généraux du reinforcement learning avant de parler d’algorithmes LLM.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
