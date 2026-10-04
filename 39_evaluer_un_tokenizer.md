# Module 39 — Évaluer un tokenizer

## Pourquoi ce module arrive ici

Un tokenizer qui s’entraîne sans erreur peut être mauvais : fragmentation excessive, mauvais round-trip, mauvaise couverture linguistique ou vocabulaire gaspillé. Il faut le benchmarker comme un composant à part entière.

## Objectifs

Mesurer tokens/mot, tokens/caractère, longueur par domaine/langue, round-trip, special tokens et fréquence d’utilisation du vocabulaire. Comparer plusieurs tokenizers sur un corpus holdout.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — tokenizer

## Définitions concrètes

### Compression ratio

**Définition concrète.** Rapport entre longueur du texte et nombre de tokens selon une convention choisie.


**Exemple simple.** caractères/token ou bytes/token.

### Fertility

**Définition concrète.** Tokens moyens par mot/unité.


**Exemple simple.** 1.2 vs 2.5 tokens/mot.

### Coverage

**Définition concrète.** Capacité à représenter correctement le texte cible.


**Exemple simple.** Byte-level offre une couverture pratique très large.

### Round-trip

**Définition concrète.** Encode puis decode restitue le texte selon la normalisation prévue.


**Exemple simple.** Détecte certaines pertes inattendues.

### Vocabulary utilization

**Définition concrète.** Part du vocabulaire réellement observée dans le corpus évalué.


**Exemple simple.** Des milliers de tokens jamais utilisés peuvent signaler un mauvais corpus ou simplement être rares; à interpréter.

## Intuition simple

Un tokenizer est un compresseur discret appris. Tu veux qu’il représente efficacement les textes importants sans perdre de caractères, tout en gardant un vocabulaire de taille raisonnable.

## Ce qui se passe réellement sous le capot

1. Préparer un corpus d’évaluation non identique au corpus de training tokenizer.
2. Encoder chaque document.
3. Mesurer tokens, caractères/bytes, mots et catégories.
4. Comparer distributions de longueur, pas seulement moyenne.
5. Tester chaînes Unicode, code, nombres, identifiants et langues importantes.
6. Vérifier round-trip selon les règles de normalisation.
7. Comparer plusieurs tailles de vocabulaire et choisir selon compromis.

## Exemple minimal à comprendre mentalement

Tokenizer A : 1.25 token/mot français, 1.10 anglais.  
Tokenizer B : 1.55 français, 1.05 anglais.  
Si ton corpus futur est majoritairement français, A peut offrir davantage de texte utile par fenêtre même si B est légèrement meilleur en anglais.

## Formules et notation utiles

\[
Fertility=\frac{N_{tokens}}{N_{mots}}
\]

Percentile 95 de longueur est souvent plus informatif que seulement la moyenne pour dimensionner les séquences.

## Code minimal observable

```python
def metrics(tok, texts):
    n_tokens=n_chars=n_words=0
    for text in texts:
        n_tokens += len(tok.encode(text).ids)
        n_chars += len(text)
        n_words += len(text.split())
    return {
        "tokens/word": n_tokens/max(n_words,1),
        "chars/token": n_chars/max(n_tokens,1),
    }
```

## Laboratoire guidé

1. Constitue un holdout de textes par catégorie.  
2. Compare au moins deux tailles de vocabulaire.  
3. Rapporte moyenne, médiane et P95 des longueurs.  
4. Teste accents, emojis, code, URLs, nombres et termes techniques.  
5. Liste les 20 textes les plus coûteux en tokens et inspecte-les.

## Ce que tu dois observer

- Les moyennes globales masquent les sous-domaines très fragmentés.
- Un vocab plus grand n’améliore pas uniformément toutes les langues.
- Un round-trip différent peut être normal si tu as explicitement choisi une normalisation destructive, mais ce choix doit être intentionnel.

## À ne pas confondre

- Meilleure compression ≠ meilleur modèle automatiquement.
- Utilisation rare d’un token ≠ token inutile automatiquement.
- Tokenizer benchmark ≠ LM benchmark.

## Erreurs fréquentes

- Choisir le tokenizer seulement sur le corpus d’entraînement.
- Ignorer P95/max et ne regarder que la moyenne.
- Évaluer une seule langue alors que le modèle sera multilingue.

## Exercices

1. Définis trois métriques importantes pour ton corpus.
2. Explique le compromis vocab large vs paramètres embeddings.
3. Propose un cas où un tokenizer plus compact n’est pas forcément préférable.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Mesurer fertility/compression.
- Comparer par catégories.
- Tester round-trip/special tokens.
- Choisir un vocabulaire avec une justification quantitative.

## Fiche mémo

Évaluer le tokenizer revient à vérifier qu’il offre une représentation **fidèle, efficace et adaptée à la distribution cible**.

## Lien avec le module suivant

Le module 40 dimensionne maintenant le Transformer qui consommera ces token IDs.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
