# Module 36 — KL divergence et contrôle de dérive

## Pourquoi ce module arrive ici

Une policy qui optimise fortement un reward peut s’éloigner du modèle de départ : style étrange, perte de diversité ou exploitation du reward. La KL divergence mesure une forme d’écart entre distributions.

## Objectifs

Comprendre KL comme mesure directionnelle entre distributions, la calculer sur un exemple discret, expliquer son utilisation comme pénalité/contrôle en post-training et ses limites.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Reinforcement learning

## Définitions concrètes

### Distribution

**Définition concrète.** Ensemble de probabilités sur les tokens/actions.


**Exemple simple.** P=[0.7,0.2,0.1].

### KL divergence

**Définition concrète.** Mesure directionnelle de différence informationnelle entre deux distributions.


**Exemple simple.** KL(P||Q) n’est généralement pas égal à KL(Q||P).

### Reference policy

**Définition concrète.** Distribution de référence avec laquelle on compare la policy courante.


**Exemple simple.** Modèle SFT gelé.

### KL penalty

**Définition concrète.** Terme réduisant le reward/objectif lorsque la policy s’éloigne de la référence.


**Exemple simple.** `R_total=R_task-β×KL` dans une intuition simplifiée.

### Drift

**Définition concrète.** Déplacement du comportement/distribution par rapport au point de départ.


**Exemple simple.** La policy devient très spécialisée ou étrange.

## Intuition simple

Imagine deux répartitions de votes sur les prochains tokens. Si la nouvelle policy déplace beaucoup de masse de probabilité vers des choix que la référence jugeait très improbables, la KL peut devenir grande.

## Ce qui se passe réellement sous le capot

1. Pour le même contexte, obtenir une distribution P de la policy et Q de la référence.
2. Comparer leurs probabilités sur les mêmes événements/tokens.
3. Calculer une KL ou approximation adaptée.
4. Ajouter la valeur au logging.
5. Dans certains algorithmes, soustraire une pénalité proportionnelle à la KL ou ajuster un coefficient.
6. Vérifier qu’un faible KL n’est pas l’objectif final : il faut aussi améliorer la tâche.

## Exemple minimal à comprendre mentalement

```text
Reference Q : [0.8, 0.1, 0.1]
Policy    P : [0.4, 0.5, 0.1]
```

La policy a déplacé beaucoup de probabilité du premier vers le deuxième token. La KL quantifie ce déplacement de manière pondérée par la distribution choisie dans la direction du calcul.

## Formules et notation utiles

\[
D_{KL}(P\|Q)=\sum_i P(i)\log\frac{P(i)}{Q(i)}
\]

Propriétés importantes : `D_KL≥0`, vaut 0 si distributions identiques (dans les conditions usuelles), et n’est pas symétrique.

## Code minimal observable

```python
import math

def kl(p, q):
    return sum(pi*math.log(pi/qi) for pi,qi in zip(p,q) if pi>0)

P=[0.4,0.5,0.1]
Q=[0.8,0.1,0.1]
print("KL(P||Q)=", kl(P,Q))
print("KL(Q||P)=", kl(Q,P))
```

## Laboratoire guidé

1. Calcule KL pour P=Q.  
2. Déplace progressivement la masse de probabilité et observe.  
3. Compare KL(P||Q) et KL(Q||P).  
4. Pendant un post-training disponible, logge la métrique KL si l’outil l’expose.  
5. Compare reward/task score et KL ensemble, jamais KL seule.

## Ce que tu dois observer

- KL est directionnelle.
- Une policy peut avoir un bon score avec dérive importante.
- Trop pénaliser la KL peut empêcher l’apprentissage utile.

## À ne pas confondre

- KL ≠ distance métrique symétrique.
- KL penalty ≠ PPO clipping.
- Faible KL ≠ bon modèle.

## Erreurs fréquentes

- Optimiser “KL faible” sans objectif de tâche.
- Comparer des KL calculées avec conventions différentes.
- Ignorer les tokens où la référence met une très faible probabilité.

## Exercices

1. Calcule KL de [0.5,0.5] vers [0.5,0.5].
2. Explique pourquoi KL est non symétrique avec un exemple intuitif.
3. Décris le compromis reward vs KL.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Écrire la formule discrète.
- Calculer une KL simple.
- Expliquer son rôle de contrôle de dérive.
- Distinguer KL, clipping PPO et gradient clipping.

## Fiche mémo

La KL ne dit pas si la nouvelle policy est meilleure; elle dit **à quel point sa distribution s’est déplacée** dans une direction définie.

## Lien avec le module suivant

Le module 37 quitte l’adaptation de modèles existants et commence le pré-entraînement from scratch.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
