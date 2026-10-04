# Module 38 — Entraîner un tokenizer BPE

## Pourquoi ce module arrive ici

Un modèle from scratch ne peut pas réutiliser arbitrairement les IDs d’un tokenizer non choisi. Le tokenizer définit le vocabulaire et donc la taille/semantique des embeddings et de la tête de sortie.

## Objectifs

Entraîner un petit tokenizer BPE, comprendre initial alphabet, merges, vocab size, special tokens, unk/byte fallback selon implémentation et sauvegarder un tokenizer reproductible.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — tokenizer

## Définitions concrètes

### Initial alphabet

**Définition concrète.** Unités de départ avant les merges.


**Exemple simple.** Caractères ou octets selon l’implémentation.

### Merge

**Définition concrète.** Fusion d’une paire fréquente en nouvelle unité.


**Exemple simple.** `t`+`h`→`th`.

### Vocab size

**Définition concrète.** Nombre maximal/cible de tokens appris, incluant selon outil les tokens spéciaux.


**Exemple simple.** 16 000.

### UNK

**Définition concrète.** Token représentant une unité non représentable dans certains tokenizers.


**Exemple simple.** `<unk>`; byte-level peut souvent éviter le problème de couverture.

### Normalizer

**Définition concrète.** Étape optionnelle modifiant le texte avant tokenisation.


**Exemple simple.** Unicode normalization, lowercase selon design.

### Pre-tokenizer

**Définition concrète.** Découpage préliminaire avant l’algorithme subword.


**Exemple simple.** Whitespace ou ByteLevel.

## Intuition simple

BPE commence avec de petites pièces et observe quelles paires sont fréquentes. Chaque fusion crée une pièce plus grande. Après beaucoup de fusions, les motifs fréquents deviennent compacts sans avoir besoin d’un token par mot possible.

## Ce qui se passe réellement sous le capot

1. Choisir un corpus représentatif de la distribution cible.
2. Définir normalisation et pré-tokenisation.
3. Initialiser le modèle BPE.
4. Compter les paires fréquentes dans les unités actuelles.
5. Fusionner les paires selon l’algorithme jusqu’au budget de vocabulaire/critère.
6. Réserver les tokens spéciaux avec IDs stables.
7. Sauvegarder tokenizer et métadonnées.
8. Tester round-trip et couverture avant de créer le modèle.

## Exemple minimal à comprendre mentalement

Corpus : `bas bas basse`.

Départ : `b a s`. La paire `b a` apparaît 3 fois → fusion `ba`. Puis `ba s` apparaît 3 fois → fusion `bas`. Les merges appris dépendent du corpus et de ses fréquences.

## Formules et notation utiles

Le vocabulaire plus grand peut raccourcir les séquences, mais augmente notamment la matrice d’embedding et souvent le LM head :

\[
N_{embedding}=V\times H
\]

Passer V=16k à 64k avec H=1024 ajoute environ `48k×1024 ≈ 49M` valeurs si les poids ne sont pas autrement partagés/structurés.

## Code minimal observable

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

special = ["<pad>", "<bos>", "<eos>", "<unk>"]
tok = Tokenizer(BPE(unk_token="<unk>"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.decoder = ByteLevelDecoder()
trainer = BpeTrainer(vocab_size=16000, special_tokens=special)
tok.train(["data/pretrain.txt"], trainer)
tok.save("tokenizer/tokenizer.json")
```

## Laboratoire guidé

1. Commence avec un petit corpus afin que l’entraînement soit rapide.  
2. Entraîne vocab 2k, 8k, 16k.  
3. Compare la tokenisation de 100 phrases hors train.  
4. Vérifie les IDs des special tokens.  
5. Sauvegarde corpus hash/config/vocab size.  
6. Ne démarre pas le pré-entraînement tant que le tokenizer n’est pas figé pour l’expérience.

## Ce que tu dois observer

- Vocab plus grand raccourcit souvent certaines séquences mais augmente embeddings/head.
- Un corpus non représentatif donne une segmentation médiocre hors domaine.
- Les choix de normalisation peuvent perdre de l’information de façon irréversible.

## À ne pas confondre

- BPE ≠ byte-level BPE automatiquement; “byte-level” décrit aussi la représentation/pré-tokenisation.
- Vocab size ≠ nombre de mots connus.
- Special token text ≠ rôle automatique sans configuration correcte.

## Erreurs fréquentes

- Entraîner le tokenizer seulement sur un échantillon non représentatif.
- Changer le tokenizer après avoir commencé à entraîner les embeddings.
- Lowercase automatiquement un corpus où la casse porte du sens sans justification.

## Exercices

1. Calcule l’impact paramètres de V=32k→64k pour H=768.
2. Explique une conséquence d’un tokenizer trop fragmentant.
3. Explique pourquoi les IDs spéciaux doivent rester stables avec le modèle.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Entraîner/sauvegarder un tokenizer.
- Expliquer les merges BPE.
- Calculer l’impact du vocabulaire sur les embeddings.
- Vérifier special tokens et round-trip.

## Fiche mémo

Le tokenizer fixe l’alphabet discret du modèle. Une fois les embeddings entraînés, changer cette correspondance ID↔token casse le sens appris.

## Lien avec le module suivant

Le module 39 évalue le tokenizer avant de lui confier des centaines de millions de tokens de training.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
