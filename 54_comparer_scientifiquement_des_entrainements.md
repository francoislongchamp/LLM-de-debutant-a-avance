# Module 54 — Comparer scientifiquement des entraînements

## Pourquoi ce module arrive ici

Si deux runs changent simultanément rank, dataset, seed, LR et nombre de steps, impossible de savoir ce qui explique le résultat. Une comparaison utile contrôle les variables et rapporte les coûts.

## Objectifs

Construire des ablations, contrôler variables, répéter avec seeds quand nécessaire, rapporter qualité+coût et éviter les conclusions à partir d’une seule run.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Évaluation

## Définitions concrètes

### Controlled experiment

**Définition concrète.** Comparaison où la variable d’intérêt change tandis que le reste reste autant que possible constant.


**Exemple simple.** r=8 vs r=32, même dataset/steps.

### Ablation

**Définition concrète.** Expérience retirant/modifiant un composant pour mesurer sa contribution.


**Exemple simple.** Avec vs sans CPT.

### Seed

**Définition concrète.** État pseudo-aléatoire influençant initialisation/sampling/order.


**Exemple simple.** 42, 43, 44.

### Variance

**Définition concrète.** Variabilité des résultats entre répétitions.


**Exemple simple.** Important sur petits datasets/runs courts.

### Pareto trade-off

**Définition concrète.** Solutions où améliorer une dimension exige sacrifier une autre.


**Exemple simple.** Qualité vs VRAM ou latence.

## Intuition simple

Une expérience n’est informative que si tu sais quelle question elle pose. “J’ai changé cinq choses et le score a monté” est une observation, pas une attribution causale.

## Ce qui se passe réellement sous le capot

1. Formuler une hypothèse.
2. Choisir une variable indépendante.
3. Fixer dataset, benchmark, budget de steps/tokens et procédure.
4. Exécuter plusieurs conditions.
5. Répéter avec plusieurs seeds si la variance peut être significative.
6. Rapporter moyenne et dispersion, plus coûts.
7. Inspecter exceptions/régressions.
8. Conclure uniquement à la portée des données.

## Exemple minimal à comprendre mentalement

Question : “rank 32 vaut-il le coût vs rank 8 ?”

```text
       score  VRAM  time  params
r=8    78.2   8GB   1h    4M
r=32   79.0   9GB   1.2h  16M
```

Le gain +0.8 doit être interprété avec variance et besoin métier, pas automatiquement comme “r=32 gagne”.

## Formules et notation utiles

Moyenne sur seeds : `mean = sum(scores)/n`. Écart-type ou intervalles permettent de représenter la dispersion. Avec très peu de runs, rester prudent : trois seeds ne transforment pas un petit benchmark en certitude.

## Code minimal observable

```python
import statistics

scores_r8=[78.1,77.9,78.6]
scores_r32=[79.2,78.4,79.0]
for name,s in [("r8",scores_r8),("r32",scores_r32)]:
    print(name, statistics.mean(s), statistics.stdev(s))
```

## Laboratoire guidé

1. Choisis une seule question (rank, LR, CPT, target modules…).  
2. Écris l’hypothèse avant les runs.  
3. Crée un tableau de variables contrôlées.  
4. Exécute au moins deux conditions.  
5. Si faisable, répète 3 seeds.  
6. Rapporte qualité, VRAM, temps, tokens et taille checkpoint.  
7. Écris une conclusion avec niveau de confiance/limites.

## Ce que tu dois observer

- Une différence plus petite que la variance peut ne pas être robuste.
- Le meilleur score peut coûter disproportionnellement plus.
- Les erreurs individuelles peuvent révéler des effets que la moyenne cache.

## À ne pas confondre

- Corrélation entre deux runs ≠ preuve causale forte.
- Même seed ≠ garantie de déterminisme absolu sur tous systèmes.
- Ablation ≠ benchmark.

## Erreurs fréquentes

- Changer plusieurs variables.
- Choisir la meilleure seed et cacher les autres.
- Omettre le coût dans une comparaison de méthodes d’efficacité.

## Exercices

1. Conçois une ablation CPT vs no-CPT.
2. Conçois une ablation target modules.
3. Écris une conclusion prudente pour un gain de 0.3 avec std=0.5.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Formuler une hypothèse testable.
- Contrôler les variables.
- Rapporter dispersion et coût.
- Écrire une conclusion qui ne dépasse pas l’expérience.

## Fiche mémo

La qualité du raisonnement expérimental détermine si tes trainings produisent de la connaissance ou seulement des checkpoints.

## Lien avec le module suivant

Le module 55 applique l’évaluation à un risque majeur de spécialisation : catastrophic forgetting.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
