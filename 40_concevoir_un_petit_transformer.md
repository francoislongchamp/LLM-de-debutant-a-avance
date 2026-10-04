# Module 40 — Concevoir un petit Transformer

## Pourquoi ce module arrive ici

Le tokenizer donne `vocab_size`; il faut maintenant choisir hidden size, nombre de couches, heads, MLP et contexte. Ces dimensions déterminent paramètres, mémoire et capacité.

## Objectifs

Comprendre les hyperparamètres architecturaux principaux et leurs contraintes de formes. Construire plusieurs configs, compter les paramètres et choisir un petit modèle cohérent.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — architecture

## Définitions concrètes

### Hidden size H

**Définition concrète.** Dimension principale des hidden states.


**Exemple simple.** 384, 512, 768…

### Num layers L

**Définition concrète.** Nombre de blocs Transformer empilés.


**Exemple simple.** 6 ou 12 pour une petite expérience.

### Attention heads

**Définition concrète.** Nombre de têtes partageant généralement H en sous-dimensions.


**Exemple simple.** H=384, 6 heads → 64 dimensions/head.

### Intermediate size

**Définition concrète.** Largeur interne du MLP.


**Exemple simple.** Souvent plusieurs fois H selon l’architecture.

### Context length

**Définition concrète.** Longueur maximale de séquence utilisée/configurée.


**Exemple simple.** 512 pour une expérience rapide.

### Parameter tying

**Définition concrète.** Partage possible entre embeddings d’entrée et tête de sortie.


**Exemple simple.** Réduit les paramètres dans certaines architectures.

## Intuition simple

Hidden size définit la largeur de la “représentation” à chaque position; num layers définit la profondeur de transformation; contexte définit combien de positions interagissent; vocabulaire définit combien d’entrées/sorties discrètes existent.

## Ce qui se passe réellement sous le capot

1. Fixer vocab size depuis le tokenizer.
2. Choisir une longueur de contexte compatible avec les données et la mémoire.
3. Choisir H et un nombre de heads compatible avec la dimension/head.
4. Choisir nombre de blocs.
5. Choisir largeur du MLP et autres détails de l’architecture.
6. Instancier le modèle sans poids pré-entraînés.
7. Compter les paramètres et faire un forward aléatoire.
8. Réduire la taille si le smoke test est trop lent/OOM.

## Exemple minimal à comprendre mentalement

Config pédagogique :

```text
V=16000
H=384
layers=6
heads=6
context=512
```

Dimension/head : `384/6=64`. Un embedding seul contient `16000×384=6 144 000` paramètres avant éventuel partage avec la tête.

## Formules et notation utiles

Attention dense standard a une matrice de scores qui croît approximativement comme `L_seq²` par head pour une séquence. C’est pourquoi doubler la longueur de contexte peut augmenter fortement la mémoire/calcul d’attention, même si le nombre de paramètres du modèle ne change presque pas.

## Code minimal observable

```python
from transformers import GPT2Config, GPT2LMHeadModel

cfg = GPT2Config(
    vocab_size=16000,
    n_positions=512,
    n_embd=384,
    n_layer=6,
    n_head=6,
)
model = GPT2LMHeadModel(cfg)
print(model.config)
print("params:", sum(p.numel() for p in model.parameters()))
```

## Laboratoire guidé

1. Crée configs H=256/384/512 à profondeur fixe.  
2. Crée profondeurs 4/6/12 à H fixe.  
3. Compte les paramètres.  
4. Fais un forward sur `[1,128]`, `[1,256]`, `[1,512]` et mesure peak mémoire.  
5. Choisis une config finale en fonction du temps d’itération, pas seulement du nombre de paramètres.

## Ce que tu dois observer

- Augmenter H augmente plusieurs matrices quadratiquement.
- Augmenter contexte affecte surtout activations/attention, pas seulement paramètres.
- Deux modèles au même nombre de paramètres peuvent répartir profondeur/largeur différemment.

## À ne pas confondre

- Context length ≠ hidden size.
- Heads ≠ layers.
- Plus de heads ≠ plus de paramètres de façon simple si H reste constant.

## Erreurs fréquentes

- Choisir des heads incompatibles avec H dans une architecture donnée.
- Dimensionner seulement sur la mémoire des poids.
- Modifier trop de dimensions avant d’avoir un baseline.

## Exercices

1. Calcule dimension/head pour H=768, heads=12.
2. Calcule paramètres embedding V=32k,H=768.
3. Explique pourquoi contexte 4× plus long peut coûter beaucoup plus que 4× pour l’attention dense.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir H/layers/heads/intermediate/context.
- Instancier un modèle from config.
- Compter les paramètres.
- Justifier une petite architecture selon ton hardware.

## Fiche mémo

Architecture = compromis entre **largeur, profondeur, contexte, vocabulaire et compute**.

## Lien avec le module suivant

Le module 41 étudie l’état initial des poids et les premiers signaux de stabilité.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
