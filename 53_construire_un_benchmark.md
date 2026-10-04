# Module 53 — Construire un benchmark

## Pourquoi ce module arrive ici

Loss et reward sont des signaux d’entraînement. Un benchmark doit mesurer les comportements que tu veux réellement conserver/améliorer, sur des exemples indépendants et avec un scoring explicite.

## Objectifs

Définir tâches, catégories, prompts, rubrics, métriques déterministes, pairwise evaluation et jeu de régression. Construire un benchmark versionné.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Évaluation

## Définitions concrètes

### Benchmark suite

**Définition concrète.** Ensemble de tâches évaluées avec procédure fixe.


**Exemple simple.** Format, connaissance domaine, général, raisonnement.

### Rubric

**Définition concrète.** Critères explicites pour juger une sortie.


**Exemple simple.** Exactitude 0–2, complétude 0–2, format 0/1.

### Exact match

**Définition concrète.** La sortie doit égaler une valeur normalisée.


**Exemple simple.** Classification ou résultat arithmétique.

### Pass@tests

**Définition concrète.** Succès selon tests automatiques.


**Exemple simple.** Fonction de code valide tous les tests.

### Regression set

**Définition concrète.** Exemples ciblés sur des comportements qui ne doivent pas régresser.


**Exemple simple.** Questions générales avant spécialisation.

### Blind evaluation

**Définition concrète.** Évaluation où le juge ne sait pas quel modèle a produit la réponse.


**Exemple simple.** Réduit certains biais humains.

## Intuition simple

Le benchmark est ton contrat de réussite. S’il ne teste pas une capacité, ton training peut la casser sans que tu t’en rendes compte.

## Ce qui se passe réellement sous le capot

1. Lister les capacités prioritaires et les risques de régression.
2. Créer des catégories séparées.
3. Définir la métrique de chaque catégorie avant de voir les résultats.
4. Réserver les exemples hors train.
5. Fixer prompts/chat template/generation settings.
6. Exécuter base et variantes.
7. Rapporter scores par catégorie + intervalles/incertitude quand possible.
8. Conserver sorties brutes pour inspection.

## Exemple minimal à comprendre mentalement

```text
Category          N   Metric
format_json      100 exact schema
knowledge_domain 100 rubric factual
reasoning         50 exact answer
general           50 regression accuracy
```

Un modèle spécialisé n’est accepté que si domain gagne sans dépasser un seuil de régression sur general.

## Formules et notation utiles

Accuracy :

```text
correct / N
```

Pour petits N, rapporter le nombre brut en plus du pourcentage. Pour comparaisons pairwise, compter wins/ties/losses et randomiser l’ordre A/B afin de réduire le position bias.

## Code minimal observable

```python
import json

row = {
  "id": "fmt-001",
  "category": "format_json",
  "prompt": "Réponds en JSON avec la clé answer.",
  "expected": {"answer": 4},
}
print(json.dumps(row, ensure_ascii=False, indent=2))
```

## Laboratoire guidé

1. Choisis 4 catégories, 20–50 exemples chacune pour commencer.  
2. Définis le scorer avant le premier run.  
3. Ajoute au moins une catégorie de régression générale.  
4. Évalue la base et sauvegarde sorties.  
5. Versionne le benchmark (`v1`).  
6. Toute modification ultérieure doit produire `v2`, pas écraser silencieusement v1.

## Ce que tu dois observer

- Les benchmarks déterministes sont faciles à reproduire mais couvrent moins les qualités ouvertes.
- Les rubrics humaines/LLM judges apportent de la couverture mais introduisent du bruit/biais.
- Un benchmark interne peut finir “overfit” par les décisions répétées de développement.

## À ne pas confondre

- Benchmark ≠ dataset train.
- Rubric ≠ reward nécessairement.
- LLM-as-judge ≠ vérité objective.

## Erreurs fréquentes

- Changer les critères après avoir vu quel modèle gagne.
- Ne rapporter que le score global.
- Utiliser des prompts contaminés par le train.

## Exercices

1. Écris une rubrique à 3 critères pour une explication technique.
2. Propose un scorer déterministe pour format JSON.
3. Ajoute une règle de seuil de régression.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Créer un benchmark versionné.
- Définir scorers avant run.
- Rapporter par catégorie.
- Inclure des capacités de régression.

## Fiche mémo

Un bon benchmark transforme “je préfère ce modèle” en critères répétables et auditables.

## Lien avec le module suivant

Le module 54 applique cette discipline à la comparaison scientifique de plusieurs entraînements.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
