# Module 44 — Perplexité et ses limites

## Pourquoi ce module arrive ici

La perplexité est une transformation intuitive de la loss de langage, souvent utilisée pour comparer la capacité prédictive sur un corpus. Elle est néanmoins facile à mal comparer entre tokenizers ou distributions.

## Objectifs

Calculer perplexité depuis une cross-entropy moyenne, comprendre son intuition et connaître les conditions qui rendent les comparaisons trompeuses.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — évaluation

## Définitions concrètes

### Perplexity / PPL

**Définition concrète.** Exponentielle de la cross-entropy moyenne en nats pour un modèle de langage.


**Exemple simple.** loss=2 → PPL≈7.39.

### Evaluation corpus

**Définition concrète.** Corpus fixe sur lequel la loss/PPL est calculée.


**Exemple simple.** Validation holdout du domaine.

### Tokenization dependence

**Définition concrète.** La loss est définie par token; changer la tokenisation change l’unité de mesure.


**Exemple simple.** Deux tokenizers rendent une PPL brute difficile à comparer directement.

### Domain dependence

**Définition concrète.** Une PPL est spécifique à une distribution.


**Exemple simple.** Un modèle code peut avoir mauvaise PPL sur poésie et inversement.

## Intuition simple

PPL est parfois décrite comme un “nombre effectif de choix” moyen, mais cette intuition est approximative. Mathématiquement, retiens surtout qu’elle est une transformation monotone de la loss : plus la cross-entropy baisse, plus la PPL baisse.

## Ce qui se passe réellement sous le capot

1. Calculer la cross-entropy moyenne sur le corpus d’évaluation sans gradient.
2. Exponentier la valeur si elle est exprimée avec logarithme naturel.
3. Comparer seulement des runs sur une base compatible.
4. Rapporter le corpus, tokenizer et protocole.
5. Compléter par des benchmarks de comportement si le but est un assistant.

## Exemple minimal à comprendre mentalement

```text
loss 3.0 → PPL ≈ 20.09
loss 2.0 → PPL ≈ 7.39
loss 1.0 → PPL ≈ 2.72
```

Le classement par PPL est le même que par loss, car exp est monotone.

## Formules et notation utiles

\[
PPL=e^{L}
\]

si `L` est la cross-entropy moyenne utilisant le logarithme naturel.

## Code minimal observable

```python
import math
for loss in [3.0,2.0,1.0]:
    print(loss, math.exp(loss))
```

## Laboratoire guidé

1. Calcule val loss et PPL à plusieurs checkpoints.  
2. Vérifie qu’elles ordonnent les checkpoints pareil.  
3. Évalue sur deux domaines différents.  
4. Si tu as deux tokenizers différents, résiste à la tentation de comparer directement la PPL brute; explique pourquoi.  
5. Compare PPL avec un benchmark d’instruction après SFT.

## Ce que tu dois observer

- PPL est très sensible au corpus.
- Une base model avec bonne PPL n’est pas automatiquement un bon assistant.
- La PPL n’ajoute pas d’information mathématique indépendante à la loss correspondante; elle change l’échelle.

## À ne pas confondre

- PPL ≠ pourcentage.
- PPL basse ≠ vérité factuelle garantie.
- PPL de tokenizers différents ≠ comparaison triviale.

## Erreurs fréquentes

- Annoncer une PPL sans corpus/protocole.
- Utiliser PPL comme seule métrique d’un chat model.
- Comparer des évaluations avec masking différent.

## Exercices

1. Calcule PPL pour loss 1.5.
2. Explique pourquoi deux tokenizers compliquent la comparaison.
3. Donne un cas où PPL domaine s’améliore mais instruction following ne change pas.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Calculer PPL.
- Expliquer son intuition et ses limites.
- Dire quelles informations accompagner d’une valeur de PPL.
- Ne pas la confondre avec une évaluation de chat.

## Fiche mémo

Perplexité est `exp(loss)` sur un protocole donné : utile pour la modélisation de langage, insuffisante pour résumer toutes les capacités.

## Lien avec le module suivant

Le module 45 revient en amont : comment transformer des sources brutes en corpus fiable à grande échelle.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
