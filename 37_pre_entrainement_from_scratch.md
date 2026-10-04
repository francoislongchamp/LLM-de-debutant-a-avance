# Module 37 — Pré-entraînement from scratch

## Pourquoi ce module arrive ici

Jusqu’ici, tous les entraînements partaient de poids déjà appris. Le pré-entraînement from scratch change la situation : tokenizer, architecture, initialisation, corpus et optimisation doivent ensemble créer les représentations de base.

## Objectifs

Comprendre ce que “from scratch” signifie, les quatre budgets fondamentaux (paramètres, tokens, contexte, compute), pourquoi commencer petit, et distinguer une expérience pédagogique d’un pré-entraînement de production.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement

## Définitions concrètes

### From scratch

**Définition concrète.** Modèle initialisé sans poids pré-entraînés de la même tâche.


**Exemple simple.** Les matrices commencent avec une initialisation numérique, pas des connaissances linguistiques.

### Base model

**Définition concrète.** Modèle issu du pré-entraînement causal avant instruction tuning.


**Exemple simple.** Il complète du texte mais n’est pas forcément un bon assistant.

### Training token

**Définition concrète.** Position de token effectivement présentée au modèle pendant l’optimisation.


**Exemple simple.** Un corpus de 10M tokens vu 2 fois représente environ 20M token exposures.

### Compute budget

**Définition concrète.** Quantité de calcul disponible pour l’expérience.


**Exemple simple.** GPU-heures, FLOPs, tokens/s.

### Scaling law

**Définition concrète.** Relation empirique entre taille modèle, données, compute et loss.


**Exemple simple.** Utile pour planifier; ce n’est pas une règle unique à appliquer aveuglément.

### Initialization

**Définition concrète.** Distribution initiale des paramètres avant apprentissage.


**Exemple simple.** Poids petits, souvent selon une règle adaptée à l’architecture.

## Intuition simple

Fine-tuner, c’est modifier une maison déjà construite. Pré-entraîner from scratch, c’est choisir les briques, les dimensions, les fondations et construire la maison entière. Une erreur dans le tokenizer ou le pipeline de données peut coûter tout le run.

## Ce qui se passe réellement sous le capot

1. Définir le but scientifique/pédagogique.
2. Choisir ou entraîner un tokenizer.
3. Préparer et versionner le corpus.
4. Choisir une architecture et un nombre de paramètres compatible avec le compute.
5. Initialiser aléatoirement les poids.
6. Découper les tokens en séquences.
7. Optimiser la next-token loss pendant un budget défini.
8. Sauvegarder checkpoints et logs.
9. Évaluer loss/perplexité et générations.
10. Appliquer ensuite éventuellement SFT et post-training.

## Exemple minimal à comprendre mentalement

Expérience pédagogique réaliste :

```text
Tokenizer : 16k
Modèle    : ~30–100M paramètres
Contexte  : 512 ou 1024
Corpus    : quelques dizaines/centaines de millions de tokens si disponible
Objectif  : observer apprentissage et diagnostiquer le pipeline
```

Ce modèle ne vise pas à rivaliser avec un grand LLM moderne. Il vise à te permettre d’observer chaque mécanisme.

## Formules et notation utiles

Une approximation souvent utilisée pour le coût du training dense d’un Transformer est de l’ordre de :

\[
FLOPs \propto N_{params}\times N_{tokens}
\]

avec un facteur constant dépendant des conventions/architecture. Retenir surtout : doubler paramètres **ou** tokens augmente fortement le compute; on ne choisit pas ces nombres indépendamment du matériel.

## Code minimal observable

```python
# Le point pédagogique : from scratch = configuration -> modèle, pas from_pretrained.
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=16000,
    n_positions=512,
    n_embd=384,
    n_layer=6,
    n_head=6,
)
model = GPT2LMHeadModel(config)
print(sum(p.numel() for p in model.parameters()))
```

## Laboratoire guidé

1. Crée trois configurations de tailles différentes sans les entraîner.  
2. Compte les paramètres.  
3. Estime la mémoire des poids BF16.  
4. Écris un budget de tokens et calcule approximativement combien de steps il représente pour ton batch de tokens.  
5. Choisis la plus petite configuration qui permettra d’itérer rapidement sur ton matériel.

## Ce que tu dois observer

- Le choix de vocabulaire affecte embeddings et LM head.
- Le contexte affecte fortement les activations/attention.
- Un modèle trop grand ralentit tellement les itérations qu’il devient mauvais pour apprendre le pipeline.

## À ne pas confondre

- From scratch ≠ continued pretraining.
- Base model ≠ chat model.
- Plus gros ≠ meilleure expérience pédagogique.
- Nombre d’epochs sur texte ≠ seule manière de penser le budget; les tokens vus sont souvent plus utiles.

## Erreurs fréquentes

- Commencer par plusieurs milliards de paramètres.
- Ne pas faire de smoke test sur un petit sous-ensemble.
- Lancer un long run avant de vérifier tokenizer, loss, checkpoints et validation.

## Exercices

1. Explique pourquoi vocab_size augmente le nombre de paramètres.
2. Définis un budget de 20M tokens avec batch global de 16×512 tokens : combien de steps environ ?
3. Liste cinq artefacts à figer avant un long run.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer from scratch vs CPT.
- Construire un modèle aléatoire depuis une config.
- Relier paramètres, tokens, contexte et compute.
- Choisir une taille pédagogique justifiée.

## Fiche mémo

Le pré-entraînement from scratch est un **système complet** : données + tokenizer + architecture + optimisation + infrastructure + évaluation.

## Lien avec le module suivant

Le module 38 entraîne le premier composant irréversible de cette chaîne : le tokenizer BPE.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
