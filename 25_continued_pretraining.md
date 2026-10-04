# Module 25 — Continued pretraining

## Pourquoi ce module arrive ici

Le SFT apprend surtout des comportements à partir d’exemples entrée→sortie. Si le problème est plutôt que le modèle connaît mal le vocabulaire, le style ou la distribution d’un domaine, on peut continuer l’objectif de langage causal sur du texte brut : continued pretraining.

## Objectifs

Distinguer pré-entraînement initial, continued pretraining (CPT) et SFT. Construire un corpus brut, le tokeniser en blocs et expliquer ce que la next-token loss apprend dans un domaine.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Adaptation de domaine

## Définitions concrètes

### Pretraining

**Définition concrète.** Entraînement initial d’un modèle à partir de poids aléatoires ou quasi aléatoires sur un grand corpus.


**Exemple simple.** Le modèle apprend d’abord des régularités générales.

### Continued pretraining

**Définition concrète.** Continuation de l’objectif de langage sur un modèle déjà pré-entraîné.


**Exemple simple.** Texte technique brut sans structure question/réponse obligatoire.

### Domain-adaptive pretraining

**Définition concrète.** CPT centré sur une distribution de domaine particulière.


**Exemple simple.** Articles médicaux, juridiques, scientifiques, etc.

### Corpus brut

**Définition concrète.** Texte servant directement à la prédiction du prochain token.


**Exemple simple.** Documents nettoyés concaténés/découpés en séquences.

### Token budget

**Définition concrète.** Nombre total de tokens utilisés pendant l’entraînement.


**Exemple simple.** 10M tokens vus une fois = 10M token exposures environ.

## Intuition simple

SFT ressemble à apprendre avec un professeur qui montre des réponses attendues. CPT ressemble davantage à lire énormément de documents du domaine pour absorber sa langue et ses régularités, sans qu’on indique explicitement comment répondre à un utilisateur.

## Ce qui se passe réellement sous le capot

1. Collecter et nettoyer du texte du domaine.
2. Dédupliquer et contrôler provenance/licence/qualité.
3. Tokeniser le corpus.
4. Assembler les tokens en séquences de longueur choisie.
5. Utiliser les mêmes labels causaux que le pré-entraînement : prochain token.
6. Mettre à jour le modèle avec un LR généralement prudent.
7. Évaluer à la fois la perplexité/loss de domaine et les capacités générales pour détecter la dérive.
8. Après CPT, appliquer éventuellement SFT pour apprendre le format de réponse.

## Exemple minimal à comprendre mentalement

Corpus :

```text
TLS utilise des mécanismes cryptographiques pour protéger les communications...
Le handshake négocie des paramètres...
Un certificat X.509 contient...
```

Le modèle apprend des cooccurrences et structures du domaine. Il n’a pas reçu explicitement :

```text
Question: Explique TLS.
Réponse: ...
```

C’est pourquoi CPT et SFT ont des effets différents.

## Formules et notation utiles

L’objectif reste la cross-entropy causale :

\[
L=-\sum_t \log p_\theta(x_t|x_{<t})
\]

La différence essentielle avec SFT est surtout **la distribution et le masking des données**, pas une nouvelle formule magique.

## Code minimal observable

```python
from datasets import load_dataset

raw = load_dataset("text", data_files="data/domain.txt", split="train")
print(raw[0])

# Le pipeline complet tokenise ensuite ces textes et les groupe
# en blocs de tokens pour une loss de langage causal.
```

## Laboratoire guidé

1. Prépare un corpus de domaine suffisamment petit pour une expérience.  
2. Mesure le nombre de documents, caractères et tokens.  
3. Réserve une validation domaine.  
4. Fais un CPT court sur un petit modèle.  
5. Mesure loss/perplexité sur domaine avant/après.  
6. Réévalue aussi un petit benchmark général.  
7. Applique ensuite le même SFT à base et CPT pour comparer honnêtement.

## Ce que tu dois observer

- Le CPT peut améliorer la modélisation du domaine sans rendre le modèle meilleur en instruction following.
- Un corpus étroit peut provoquer du forgetting ou du style drift.
- La quantité de tokens et la mixture sont cruciales.

## À ne pas confondre

- CPT ≠ SFT.
- CPT ≠ full fine-tuning : CPT décrit l’objectif/données, full FT décrit les paramètres mis à jour.
- Texte brut ≠ données non nettoyées.

## Erreurs fréquentes

- Faire CPT sur un petit corpus très répétitif et conclure à une acquisition de connaissance générale.
- Évaluer uniquement sur la loss du corpus d’entraînement.
- Oublier un benchmark général.

## Exercices

1. Explique comment un CPT peut être fait en full FT ou potentiellement avec PEFT.
2. Donne un problème mieux adapté au CPT qu’au SFT.
3. Explique pourquoi “lire des documents” et “apprendre à répondre” sont deux objectifs distincts.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir CPT et le distinguer du pretraining from scratch et du SFT.
- Décrire un pipeline corpus brut→tokens→causal loss.
- Expliquer quels benchmarks utiliser avant/après.
- Identifier le risque de domain over-specialization.

## Fiche mémo

CPT adapte la **distribution de langage/connaissances du domaine**; SFT adapte surtout le **comportement supervisé**.

## Lien avec le module suivant

Le module 26 formalise le choix entre CPT et SFT au lieu de les empiler automatiquement.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
