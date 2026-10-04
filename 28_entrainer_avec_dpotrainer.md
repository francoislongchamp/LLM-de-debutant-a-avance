# Module 28 — Entraîner avec DPOTrainer

## Pourquoi ce module arrive ici

Après l’intuition, il faut rendre DPO observable : dataset accepté, tokenisation, loss, métriques de préférence et comparaison avec le modèle SFT de départ.

## Objectifs

Configurer un DPO minimal, comprendre `beta`, longueur de prompt/completion, policy/reference, et construire une évaluation pairwise avant/après.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Post-training — préférences

## Définitions concrètes

### Policy model

**Définition concrète.** Modèle que l’on met à jour.


**Exemple simple.** Le modèle SFT que l’on veut aligner davantage.

### Reference model

**Définition concrète.** Point d’ancrage non optimisé utilisé dans la loss DPO selon le setup.


**Exemple simple.** État de départ ou référence implicite/explicite.

### Beta

**Définition concrète.** Hyperparamètre contrôlant l’échelle du signal préférence/écart à la référence selon la formulation.


**Exemple simple.** À tuner, pas à interpréter comme une probabilité.

### Preference accuracy

**Définition concrète.** Fraction de paires où le modèle score le chosen au-dessus du rejected selon la métrique choisie.


**Exemple simple.** Un indicateur, pas une évaluation complète.

### Margin

**Définition concrète.** Écart entre scores/log-probabilités relatifs chosen et rejected.


**Exemple simple.** Plus positif peut signifier préférence plus nette, mais surveiller l’overoptimization.

## Intuition simple

DPOTrainer automatise beaucoup de plomberie, mais l’expérience reste : “est-ce que la policy préfère davantage les chosen du jeu de validation sans perdre d’autres capacités ?”.

## Ce qui se passe réellement sous le capot

1. Charger policy et dataset de préférences.
2. Préparer/tokeniser prompt et réponses selon le format attendu.
3. Calculer log-probabilités chosen/rejected de la policy et référence.
4. Calculer la loss DPO.
5. Backward/update de la policy.
6. Évaluer des métriques de préférence sur validation.
7. Comparer benchmark général et style pour détecter dérive.

## Exemple minimal à comprendre mentalement

Avant DPO, sur 100 paires validation :

```text
chosen préféré : 58/100
```

Après DPO :

```text
chosen préféré : 78/100
```

Mais si le benchmark général passe de 80 à 60, le gain de préférence n’est pas suffisant pour conclure que le modèle global est meilleur.

## Code minimal observable

```python
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer

prefs = load_dataset("json", data_files="data/preferences.jsonl", split="train")

args = DPOConfig(
    output_dir="outputs/dpo",
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    logging_steps=5,
)

trainer = DPOTrainer(
    model="path-or-model-id",
    args=args,
    train_dataset=prefs,
)
trainer.train()
```

## Laboratoire guidé

1. Évalue la policy de départ sur un split de préférences.  
2. Lance un DPO court.  
3. Observe les métriques chosen/rejected fournies par le trainer/version utilisée.  
4. Réévalue le split de préférence.  
5. Réévalue le benchmark général du module 13.  
6. Compare au simple SFT sur chosen pour comprendre que l’objectif n’est pas identique.

## Ce que tu dois observer

- La préférence peut s’améliorer plus vite que les capacités générales.
- Le choix de beta et LR influence fortement la dérive.
- Les API TRL évoluent; toujours vérifier la version installée et la documentation correspondante.

## À ne pas confondre

- DPO loss ≠ cross-entropy SFT standard.
- Preference accuracy ≠ exactitude factuelle universelle.
- Reference model ≠ reward model.

## Erreurs fréquentes

- Utiliser un LR de SFT sans justification.
- Évaluer seulement sur le dataset de préférence train.
- Ignorer les changements de longueur/style produits par l’optimisation.

## Exercices

1. Explique pourquoi la référence n’est pas un reward model.
2. Propose deux valeurs de beta à comparer et quelles métriques surveiller.
3. Explique pourquoi un DPO peut sur-optimiser un biais du dataset.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Lancer DPO sur un petit dataset.
- Expliquer policy/reference/beta.
- Mesurer préférence avant/après.
- Vérifier un benchmark général pour détecter régression.

## Fiche mémo

DPOTrainer est un outil; la compétence réelle est de contrôler **le signal de préférence, la dérive et l’évaluation hors préférence**.

## Lien avec le module suivant

Le module 29 apprend une autre approche : entraîner explicitement un modèle qui produit un score de récompense.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
