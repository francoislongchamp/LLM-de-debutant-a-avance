# Module 2 — Tokenisation en profondeur

## Pourquoi ce module arrive ici

Le module 1 a montré que le modèle ne lit pas directement des mots : il reçoit des identifiants de tokens. Il faut maintenant comprendre comment ces tokens sont construits, car la tokenisation influence la longueur de contexte, le coût, la vitesse et la facilité d’apprentissage.

## Objectifs

Comprendre vocabulaire, token, token ID, sous-mot, pré-tokenisation, BPE, byte-level, tokens spéciaux et fertility. Savoir inspecter un tokenizer et expliquer pourquoi deux tokenizers peuvent représenter le même texte avec un nombre de tokens très différent.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fondations

## Définitions concrètes

### Token

**Définition concrète.** Une unité discrète choisie dans le vocabulaire du tokenizer. Un token n’est pas nécessairement un mot.


**Exemple simple.** `cybersécurité` peut devenir un token unique ou plusieurs fragments.

### Vocabulaire

**Définition concrète.** La table qui associe chaque token à un entier unique appelé token ID.


**Exemple simple.** Avec 50 000 entrées, il existe environ 50 000 IDs possibles.

### Sous-mot

**Définition concrète.** Un fragment réutilisable plus petit qu’un mot complet.


**Exemple simple.** `authentication` peut être représenté par plusieurs fragments fréquents.

### BPE

**Définition concrète.** Une famille d’algorithmes qui fusionne progressivement les paires fréquentes.


**Exemple simple.** Si `l` + `e` apparaît souvent, `le` peut devenir une unité.

### Byte-level

**Définition concrète.** Une représentation qui peut retomber sur des octets pour les caractères rares.


**Exemple simple.** Un symbole rare reste encodable même sans token dédié.

### Fertility

**Définition concrète.** Nombre moyen de tokens nécessaires par mot ou unité de texte.


**Exemple simple.** 100 mots représentés par 135 tokens donnent 1,35 token/mot.

### Token spécial

**Définition concrète.** Token réservé à une fonction structurelle.


**Exemple simple.** BOS, EOS, PAD, ou marqueurs de rôles de chat.

## Intuition simple

Imagine un jeu de pièces. Une pièce par mot rendrait le vocabulaire gigantesque; une pièce par caractère rendrait les séquences très longues. Les sous-mots sont un compromis : les fragments fréquents deviennent des pièces compactes, les mots rares restent assemblables.

## Ce qui se passe réellement sous le capot

1. Le texte peut être normalisé selon les règles du tokenizer.
2. Une pré-tokenisation peut repérer des frontières grossières.
3. L’algorithme applique son vocabulaire et ses règles de segmentation/fusion.
4. Chaque token est remplacé par son ID.
5. Des tokens spéciaux peuvent être ajoutés.
6. Le modèle reçoit finalement une matrice d’entiers `[batch, sequence]`.

## Exemple minimal à comprendre mentalement

Corpus jouet :

```text
bas bas basse
```

Au départ :

```text
b a s
b a s
b a s s e
```

La paire `b a` est fréquente et peut devenir `ba`, puis `ba s` peut devenir `bas`. Le mot fréquent `bas` coûte alors un token tandis que `basse` peut rester `bas` + `s` + `e`. BPE donne donc des unités plus compactes aux séquences fréquentes.

## Formules et notation utiles

Mesures simples :

\[
\text{tokens/mot}=\frac{N_{tokens}}{N_{mots}}
\]

\[
\text{tokens/caractère}=\frac{N_{tokens}}{N_{caractères}}
\]

Un ratio élevé signifie que la même information consomme davantage de positions de contexte.

## Code minimal observable

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
samples = ["chat", "anticonstitutionnellement", "cybersécurité", "cybersecurity", "CVE-2026-12345"]

for text in samples:
    ids = tok.encode(text, add_special_tokens=False)
    print("\nTEXT  :", text)
    print("TOKENS:", tok.convert_ids_to_tokens(ids))
    print("IDS   :", ids)
    print("COUNT :", len(ids))
    print("DECODE:", tok.decode(ids))
```

## Laboratoire guidé

1. Exécute le script.  
2. Ajoute dix termes de ton domaine et leurs équivalents anglais.  
3. Calcule tokens/mot sur deux paragraphes comparables.  
4. Vérifie `decode(encode(texte))`.  
5. Inspecte `tok.special_tokens_map`.  
6. Cherche les cinq chaînes les plus fragmentées de ton petit corpus.

## Ce que tu dois observer

- Les frontières de tokens ne suivent pas toujours les espaces.
- Un mot long peut être peu coûteux s’il est fréquent.
- Les nombres, identifiants et termes techniques peuvent être fortement fragmentés.
- La même idée exprimée dans deux langues peut consommer un nombre de tokens différent.

## À ne pas confondre

- Token ≠ mot.
- Token ID ≠ embedding.
- Vocabulaire du tokenizer ≠ connaissances du modèle.
- Ajouter un token au vocabulaire ≠ apprendre sa signification.

## Erreurs fréquentes

- Comparer uniquement la taille de contexte annoncée entre deux modèles sans comparer leurs tokenizers.
- Modifier le vocabulaire sans adapter les embeddings du modèle.
- Croire qu’un tokenizer comprend le sens : il segmente et mappe vers des IDs.

## Exercices

1. Explique pourquoi 256 tokens d’octets peuvent représenter tout texte mais seraient souvent inefficaces.
2. Trouve un mot fortement fragmenté et explique pourquoi cela peut coûter cher.
3. Calcule la fertility d’un échantillon de 50 phrases.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir token, token ID, vocabulaire, sous-mot et token spécial.
- Expliquer BPE avec un exemple de fusion simple.
- Expliquer pourquoi le tokenizer affecte coût et contexte.
- Lire une sortie tokens/IDs et repérer une forte fragmentation.

## Fiche mémo

`texte → segmentation → tokens → IDs`. Le tokenizer est l’interface discrète entre le texte et le réseau de neurones.

## Lien avec le module suivant

Le module 3 utilise ces tokens pour comprendre les séquences : padding, truncation, masques et fenêtre de contexte.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
