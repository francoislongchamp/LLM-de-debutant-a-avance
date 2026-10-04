# Module 13 — Baseline et évaluation avant/après

## Pourquoi ce module arrive ici

Un modèle fine-tuné peut sembler meilleur simplement parce que l’on choisit des exemples favorables. La baseline fixe l’état initial et rend les changements mesurables.

## Objectifs

Construire une baseline reproductible, fixer prompts et paramètres de génération, comparer avant/après sur plusieurs catégories et distinguer métriques automatiques et jugement qualitatif.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning — évaluation

## Définitions concrètes

### Baseline

**Définition concrète.** Point de référence avant modification.


**Exemple simple.** Le modèle original évalué sur exactement le même benchmark.

### Benchmark

**Définition concrète.** Ensemble d’exemples et procédure de scoring fixe.


**Exemple simple.** 50 questions réparties en cinq catégories.

### Metric

**Définition concrète.** Règle produisant une mesure.


**Exemple simple.** Accuracy, exact match, loss, score de rubric.

### Determinism

**Définition concrète.** Réduction/contrôle de la variabilité aléatoire pour rendre une comparaison reproductible.


**Exemple simple.** Greedy decoding ou seed lorsque pertinent.

### Paired comparison

**Définition concrète.** Comparaison de deux sorties sur exactement le même exemple.


**Exemple simple.** Base vs fine-tuned sur le même prompt.

## Intuition simple

Sans photo “avant”, une photo “après” ne montre pas l’amélioration. La baseline est cette photo, mais prise avec le même éclairage : mêmes prompts, même format, mêmes paramètres de génération et même méthode de scoring.

## Ce qui se passe réellement sous le capot

1. Créer un benchmark qui ne provient pas directement du train.
2. Évaluer le modèle de base et sauvegarder sorties + paramètres.
3. Entraîner.
4. Réévaluer sur les mêmes entrées avec la même procédure.
5. Comparer par catégorie, pas seulement moyenne globale.
6. Inspecter régressions et gains.
7. Conserver les artefacts bruts pour audit.

## Exemple minimal à comprendre mentalement

```text
                Base   SFT
Format JSON      40%   92%
Connaissance     71%   72%
Raisonnement     65%   59%
```

La moyenne peut cacher une régression en raisonnement. Une bonne évaluation ne demande donc pas seulement “le score moyen a-t-il monté ?”.

## Formules et notation utiles

Gain absolu :

\[
\Delta = score_{après}-score_{avant}
\]

Pour une accuracy, rapporter aussi le nombre d’exemples : 90 % sur 10 questions n’a pas la même stabilité que 90 % sur 1000.

## Code minimal observable

```python
import json

# Structure minimale d’un enregistrement reproductible
record = {
  "model": "base-model-id",
  "prompt_id": "q001",
  "generation": {"do_sample": False, "max_new_tokens": 128},
  "output": "...",
  "score": 1,
}
print(json.dumps(record, ensure_ascii=False, indent=2))
```

## Laboratoire guidé

1. Crée un benchmark de 30 prompts, 3 catégories minimum.  
2. Fixe paramètres de génération.  
3. Sauvegarde chaque sortie du modèle de base dans JSONL.  
4. Évalue le modèle fine-tuné avec le même script.  
5. Calcule scores par catégorie et total.  
6. Sélectionne aussi les plus grosses régressions et explique-les.

## Ce que tu dois observer

- Une moyenne globale peut cacher des gains et pertes opposés.
- Le sampling peut ajouter du bruit à la comparaison.
- Un benchmark trop proche du train surestime les gains utiles.

## À ne pas confondre

- Baseline ≠ “ancien modèle mauvais”.
- Validation loss ≠ benchmark de comportement.
- Un exemple impressionnant ≠ preuve statistique.

## Erreurs fréquentes

- Changer les paramètres de génération entre avant et après.
- Regarder seulement les réussites.
- Modifier le benchmark après avoir vu les résultats sans versionner ce changement.

## Exercices

1. Propose quatre catégories d’évaluation pour un assistant technique.
2. Explique pourquoi `temperature=1` peut compliquer une comparaison paired.
3. Écris une règle pour décider si une sortie est correcte sans regarder quel modèle l’a produite.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Produire une baseline sauvegardée.
- Comparer avant/après sur les mêmes prompts.
- Rapporter par catégorie.
- Identifier au moins une régression potentielle au lieu de chercher seulement des gains.

## Fiche mémo

L’entraînement n’est qu’une hypothèse d’amélioration; la baseline et le benchmark testent cette hypothèse.

## Lien avec le module suivant

Le module 14 étudie le cas où le train s’améliore mais le modèle généralise moins bien : le surapprentissage.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
