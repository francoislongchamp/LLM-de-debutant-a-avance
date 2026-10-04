# Module 56 — Projet 1 — LoRA spécialisé

## Pourquoi ce module arrive ici

Ce premier projet transforme les modules 9 à 19 en une expérience complète : données, baseline, LoRA, validation, benchmark et rapport. Le but n’est pas seulement d’obtenir un adapter, mais de pouvoir expliquer chaque décision.

## Objectifs

Construire un LoRA spécialisé sur un petit domaine, mesurer son coût et démontrer avec un benchmark indépendant ce qui s’améliore et ce qui ne s’améliore pas.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 4 à 8 heures selon le dataset et le GPU.  
**Niveau :** Projet intégrateur

## Définitions concrètes

### Hypothèse projet

**Définition concrète.** Phrase falsifiable décrivant le gain attendu.


**Exemple simple.** “LoRA améliore le respect du format X sans perdre plus de 2 points au benchmark général.”

### Acceptance criteria

**Définition concrète.** Conditions fixées avant le run pour considérer le projet réussi.


**Exemple simple.** +15 points domaine, régression général ≤2.

### Experiment card

**Définition concrète.** Résumé de dataset, modèle, config, hardware et résultats.


**Exemple simple.** Fichier `RESULTS.md`.

### Artifact

**Définition concrète.** Fichier produit et versionné.


**Exemple simple.** adapter, config, eval JSONL, logs.

## Intuition simple

Un projet réussi n’est pas “j’ai lancé LoRA”. C’est : j’avais une hypothèse, j’ai construit une expérience qui pouvait la réfuter, j’ai mesuré les effets et je peux reproduire le résultat.

## Ce qui se passe réellement sous le capot

1. Choisir un problème étroit et mesurable.
2. Construire/valider un dataset SFT.
3. Créer train/validation/test sans contamination.
4. Construire benchmark domaine + général.
5. Évaluer la base.
6. Choisir LoRA r/targets avec justification.
7. Faire un smoke test puis le run.
8. Évaluer le meilleur checkpoint.
9. Mesurer VRAM, temps, paramètres, checkpoint.
10. Écrire un rapport et conserver tous les artefacts.

## Exemple minimal à comprendre mentalement

Objectif pédagogique : améliorer des réponses au format structuré.

```text
Base: format exact 42/100
LoRA: format exact 89/100
General: base 78/100, LoRA 77/100
```

Conclusion : amélioration forte du comportement ciblé avec régression générale limitée selon ce benchmark. Cette conclusion reste bornée par le dataset et les tests utilisés.

## Formules et notation utiles

Rapporte au minimum :

```text
delta_domain = score_lora - score_base
regression_general = score_lora_general - score_base_general
trainable_ratio = trainable/total
```

Ajoute peak VRAM, durée, tokens/steps et taille adapter.

## Code minimal observable

```text
project_56/
├── README.md
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── configs/lora.yaml
├── train.py
├── evaluate.py
├── baseline.jsonl
├── results.jsonl
├── outputs/
└── RESULTS.md
```

## Laboratoire guidé

### Étape A — Spécification
Écris l’hypothèse et les seuils de succès.

### Étape B — Données
Vérifie 100 % des petits jeux à la main si possible; sinon fais un échantillonnage qualité documenté.

### Étape C — Baseline
Sauvegarde toutes les sorties du modèle de base.

### Étape D — Training
Commence par 10–20 steps smoke test. Vérifie loss, labels, checkpoint. Lance ensuite le run prévu.

### Étape E — Évaluation
Exécute exactement le même benchmark.

### Étape F — Rapport
Explique ce qui s’est amélioré, les régressions, les limites et le prochain test utile.

## Ce que tu dois observer

- Le dataset explique souvent plus le résultat que de petits changements de rank.
- Les gains sur prompts similaires au train sont plus faciles que les gains sur formulations nouvelles.
- La taille de l’adapter n’est pas une métrique de qualité.

## À ne pas confondre

- Projet fini ≠ training terminé.
- Meilleure train loss ≠ meilleur checkpoint.
- Réponse impressionnante ≠ benchmark.

## Erreurs fréquentes

- Changer le benchmark après avoir vu les sorties.
- Oublier le modèle/base revision exacte.
- Ne pas sauvegarder le config du tokenizer/chat template.

## Exercices

1. Écris avant le run trois raisons possibles d’échec.
2. Après le run, cherche activement cinq régressions.
3. Propose l’expérience suivante la plus informative, pas simplement “plus d’epochs”.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Reproduire le run depuis README/config.
- Montrer la baseline et les résultats bruts.
- Démontrer les trainable params et le coût.
- Écrire une conclusion limitée aux données observées.

## Fiche mémo

Livrable : un adapter **et** la preuve reproductible de ce qu’il apporte.

## Lien avec le module suivant

Le projet 2 garde le même problème et isole l’effet de la quantification : LoRA vs QLoRA.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
