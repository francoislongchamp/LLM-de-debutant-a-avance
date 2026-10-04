# Module 46 — Déduplication exacte et near-duplicate

## Pourquoi ce module arrive ici

Le même contenu peut apparaître de nombreuses fois sous forme exacte ou légèrement modifiée. Cela surpondère certains textes, augmente la mémorisation et peut contaminer les benchmarks.

## Objectifs

Comprendre hash exact, canonicalisation, shingles, Jaccard, MinHash/LSH au niveau conceptuel. Construire une déduplication exacte et un détecteur near-duplicate pédagogique.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — données

## Définitions concrètes

### Exact duplicate

**Définition concrète.** Contenu identique après une normalisation définie.


**Exemple simple.** Deux documents ont le même hash canonique.

### Canonicalisation

**Définition concrète.** Transformation déterministe avant comparaison.


**Exemple simple.** Normaliser espaces ou fins de ligne.

### Shingle

**Définition concrète.** N-gramme de tokens/mots utilisé pour représenter un document.


**Exemple simple.** Suite de 5 mots.

### Jaccard similarity

**Définition concrète.** Intersection/union de deux ensembles.


**Exemple simple.** 0=aucun recouvrement, 1=identiques.

### MinHash

**Définition concrète.** Technique probabiliste résumant des ensembles pour approximer Jaccard.


**Exemple simple.** Permet de comparer à grande échelle.

### LSH

**Définition concrète.** Indexation probabiliste regroupant des signatures susceptibles d’être similaires.


**Exemple simple.** Réduit le nombre de comparaisons paire-à-paire.

## Intuition simple

Les hashes répondent “exactement pareil ?”. Les shingles répondent “partagent-ils beaucoup de fragments ?”. MinHash/LSH rendent cette seconde question praticable à grande échelle.

## Ce qui se passe réellement sous le capot

1. Définir une canonicalisation qui ne détruit pas trop de sens.
2. Calculer un hash et retirer les doublons exacts.
3. Pour near-duplicate, produire des shingles.
4. Calculer/approximer la similarité entre documents candidats.
5. Regrouper les quasi-doublons selon un seuil/stratégie.
6. Choisir un représentant selon qualité/source.
7. Conserver les clusters et raisons de suppression pour audit.

## Exemple minimal à comprendre mentalement

```text
A: "le chat noir dort sur le canapé"
B: "le chat noir dort sur le grand canapé"
```

Les chaînes ne sont pas identiques, donc hashes différents. Leurs ensembles de shingles peuvent pourtant avoir un fort recouvrement.

## Formules et notation utiles

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

Pour ensembles `{a,b,c}` et `{b,c,d}` : intersection=2, union=4 → J=0.5.

## Code minimal observable

```python
import hashlib

def canonical(text):
    return " ".join(text.split()).strip().lower()

def sha(text):
    return hashlib.sha256(canonical(text).encode()).hexdigest()

print(sha("Bonjour   monde"))
print(sha("bonjour monde"))
```

## Laboratoire guidé

1. Crée 20 documents avec exact duplicates et versions légèrement modifiées.  
2. Retire les exacts par hash.  
3. Implémente des shingles de 3 mots et Jaccard.  
4. Affiche toutes les paires au-dessus de 0.7 sur ce petit corpus.  
5. Inspecte manuellement faux positifs/faux négatifs.  
6. Étudie ensuite MinHash/LSH comme optimisation d’échelle, sans remplacer la compréhension de Jaccard.

## Ce que tu dois observer

- Le seuil near-duplicate dépend du type de document.
- Canonicaliser trop agressivement peut fusionner des textes distincts.
- Dédupliquer peut modifier les proportions de sources.

## À ne pas confondre

- Hash exact ≠ similarité sémantique.
- Jaccard élevé ≠ documents nécessairement équivalents.
- Déduplication ≠ contamination benchmark entièrement résolue.

## Erreurs fréquentes

- Utiliser lowercasing comme canonicalisation universelle.
- Supprimer sans conserver cluster/source.
- Appliquer un seuil unique à tous les formats sans inspection.

## Exercices

1. Calcule Jaccard de deux petits ensembles.
2. Propose deux canonicalisations et leurs risques.
3. Explique pourquoi comparaison O(N²) devient impossible à grande échelle.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Dédupliquer exactement avec hash.
- Calculer Jaccard sur shingles.
- Expliquer MinHash/LSH conceptuellement.
- Auditer les suppressions.

## Fiche mémo

Exact dedup enlève les copies; near-dedup réduit la répétition masquée. Les deux influencent mémorisation, mixture et contamination.

## Lien avec le module suivant

Le module 47 choisit combien de tokens chaque source contribue : dataset mixture et sampling.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
