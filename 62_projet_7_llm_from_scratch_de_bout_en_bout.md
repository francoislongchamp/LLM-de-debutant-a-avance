# Module 62 — Projet 7 — LLM from scratch de bout en bout

## Pourquoi ce module arrive ici

Ce projet final force à relier toutes les couches du système. Tu ne pars plus d’un modèle existant : tu contrôles tokenizer, architecture, données, training, checkpoints, évaluation et post-training.

## Objectifs

Construire un petit LM causal from scratch de manière reproductible, démontrer que le pré-entraînement apprend, effectuer un SFT, comparer base vs instruct et documenter les limites. DPO/RL sont des extensions seulement si les étapes précédentes sont solides.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** Projet long; commencer par une version miniature exécutable en quelques heures, puis augmenter seulement si tout est validé.  
**Niveau :** Projet final

## Définitions concrètes

### End-to-end reproducibility

**Définition concrète.** Capacité à reconstruire le résultat depuis sources/config/code/version.


**Exemple simple.** Raw data→tokenizer→model→checkpoints→eval.

### Smoke model

**Définition concrète.** Version très petite du modèle utilisée pour valider le pipeline.


**Exemple simple.** Quelques millions de paramètres.

### Scale-up gate

**Définition concrète.** Critère qui doit être satisfait avant d’augmenter taille/durée.


**Exemple simple.** Overfit one batch + resume checkpoint + val loss cohérente.

### Base checkpoint

**Définition concrète.** Résultat du pré-entraînement causal.


**Exemple simple.** Avant SFT.

### Instruct checkpoint

**Définition concrète.** Résultat après SFT du base model.


**Exemple simple.** Apprend le format conversationnel/comportement.

### Model card

**Définition concrète.** Document décrivant architecture, données, training, évaluation, limites et usage.


**Exemple simple.** Livrable final.

## Intuition simple

Le projet final ressemble à une fusée avec plusieurs étages. Si le tokenizer est faux, aucune optimisation plus tard ne le réparera proprement. Si le checkpoint ne reprend pas, augmenter le compute est risqué. On ne franchit un étage que lorsque le précédent est validé.

## Ce qui se passe réellement sous le capot

1. Définir objectif et limites du modèle.
2. Construire/versionner le corpus et holdout.
3. Entraîner/évaluer tokenizer.
4. Créer un smoke model très petit.
5. Réussir forward/backward/overfit-one-batch/checkpoint-resume.
6. Créer le modèle cible pédagogique.
7. Pré-entraîner sous budget de tokens défini.
8. Évaluer loss/PPL et générations sur prompts fixes.
9. Construire un SFT propre et benchmark instruct.
10. Fine-tuner le base model.
11. Comparer base vs instruct.
12. Optionnel : DPO sur préférence validée.
13. Optionnel : RL sur tâche vérifiable séparée.
14. Écrire model card et rapport complet.

## Exemple minimal à comprendre mentalement

Version miniature recommandée :

```text
Corpus         : suffisamment petit pour itérer, mais propre et séparé train/val
Tokenizer      : 8k–16k
Smoke model    : ~5–15M paramètres
Target model   : ~30–100M selon matériel
Context        : 256–1024
Pretrain       : budget de tokens explicite
SFT            : petit dataset de qualité
Benchmark      : LM holdout + instruction + regression
```

Les valeurs sont pédagogiques et doivent être adaptées à ton GPU/temps.

## Formules et notation utiles

Checklist quantitative minimale :

```text
N_params
N_train_tokens
context_length
batch_tokens/update
optimizer_steps
peak_VRAM
tokens_per_second
val_loss / PPL
benchmark_base
benchmark_SFT
```

Sans ces nombres, il est difficile d’expliquer ton expérience.

## Code minimal observable

```text
llm_from_scratch/
├── README.md
├── env/
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── splits/
│   └── manifests/
├── tokenizer/
├── configs/
│   ├── smoke.yaml
│   ├── pretrain.yaml
│   └── sft.yaml
├── src/
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── pretrain.py
│   ├── sft.py
│   └── evaluate.py
├── checkpoints/
├── logs/
├── eval/
└── MODEL_CARD.md
```

## Laboratoire guidé

### Phase 0 — Design
Écris l’objectif, la taille cible et les ressources disponibles.

### Phase 1 — Data
Construis raw→clean→dedup→split et un manifest de tokens.

### Phase 2 — Tokenizer
Entraîne au moins deux tailles et benchmarke-les. Fige celle choisie.

### Phase 3 — Smoke model
Un très petit modèle doit : charger un batch, produire loss finie, backward, overfit one batch, sauvegarder et reprendre.

### Phase 4 — Pretraining
Lance le modèle cible avec budget de tokens; valide régulièrement; conserve plusieurs checkpoints.

### Phase 5 — Base evaluation
Mesure val loss/PPL et prompts fixes. Documente ce que le base model sait et ne sait pas faire.

### Phase 6 — SFT
Construis un dataset conversationnel et entraîne le base model.

### Phase 7 — Instruct evaluation
Compare avec exactement la même suite de prompts/benchmarks.

### Phase 8 — Extensions
DPO puis RL seulement si un besoin mesuré le justifie.

### Phase 9 — Rapport
Rédige architecture, données, compute, courbes, résultats, échecs et limites.

## Ce que tu dois observer

- Le plus gros apprentissage vient souvent des bugs et diagnostics du pipeline, pas du score final.
- Un petit modèle from scratch sera limité; c’est normal.
- La qualité des données et du tokenizer est visible très tôt dans les métriques.
- Scale-up ne doit arriver qu’après une version miniature robuste.

## À ne pas confondre

- Projet final ≠ obligation de construire un modèle compétitif avec les grands LLM.
- Pré-entraînement réussi ≠ assistant réussi.
- SFT réussi ≠ alignement complet.
- Un long run ≠ un bon run.

## Erreurs fréquentes

- Sauter le smoke model.
- Changer tokenizer en cours de pré-entraînement.
- Ne pas pouvoir reprendre un checkpoint.
- Ne pas garder le base checkpoint avant SFT.
- Ajouter DPO/RL pour “faire complet” sans besoin démontré.

## Exercices

1. Écris ton scale-up gate en cinq critères binaires.
2. Calcule le nombre de steps pour ton budget de tokens.
3. Définis trois résultats qui te feraient arrêter le run tôt.
4. Écris les sections de ta future model card avant de commencer.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Reproduire tokenizer et dataset final.
- Réussir le smoke pipeline complet.
- Pré-entraîner avec logs/checkpoints/reprise.
- Évaluer le base model.
- Faire SFT et comparer base/instruct.
- Produire une model card honnête avec limites.

## Fiche mémo

Tu maîtrises réellement le parcours lorsque tu peux expliquer et reproduire : **données → tokenizer → architecture → pré-entraînement → évaluation → SFT → préférence/RL si nécessaire**.

## Lien avec le module suivant

Fin du parcours principal. Les annexes synthétisent maintenant les choix de méthode, les diagnostics et les commandes de référence.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
