# Module 58 — Projet 3 — LoRA vs full fine-tuning

## Pourquoi ce module arrive ici

La comparaison la plus instructive du cours est de voir ce que gagne réellement le full FT lorsque toutes les matrices peuvent changer, et ce qu’il coûte en mémoire, stockage et forgetting.

## Objectifs

Comparer LoRA et full FT sur un petit modèle avec budgets comparables, mesurer domaine/généralisation/forgetting, coût et taille checkpoint.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 4 à 10 heures; choisir un modèle suffisamment petit pour un full FT sûr.  
**Niveau :** Projet intégrateur

## Définitions concrètes

### Budget matched

**Définition concrète.** Tentative de rendre deux expériences comparables sur tokens/steps/données.


**Exemple simple.** Même nombre de tokens vus.

### Capacity difference

**Définition concrète.** Différence intrinsèque de paramètres modifiables entre méthodes.


**Exemple simple.** Full FT peut modifier tout W.

### Forgetting delta

**Définition concrète.** Régression sur benchmark général après training.


**Exemple simple.** Après-base.

### Checkpoint footprint

**Définition concrète.** Espace disque nécessaire pour l’artefact final.


**Exemple simple.** Adapter vs modèle complet.

## Intuition simple

Full FT a plus de liberté. La question expérimentale n’est pas “peut-il fitter le train ?”, mais “cette liberté supplémentaire produit-elle un gain utile assez grand pour justifier coût et régressions ?”.

## Ce qui se passe réellement sous le capot

1. Choisir un modèle qui tient en full FT avec marge.
2. Fixer datasets et benchmark.
3. Définir LoRA stable.
4. Définir full FT avec LR adapté, pas copié du LoRA.
5. Matcher le budget de tokens/steps autant que pertinent.
6. Évaluer domaine, validation et général.
7. Mesurer VRAM, temps, checkpoint.
8. Calculer forgetting.
9. Comparer le front qualité/coût.

## Exemple minimal à comprendre mentalement

```text
              LoRA     Full FT
Domain        85       87
General       78       71
Peak VRAM     9 GB     18 GB
Checkpoint    80 MB    1.2 GB
```

Le full FT gagne 2 domaine mais perd 7 général dans cet exemple fictif. Le “meilleur” dépend alors du cahier des charges.

## Formules et notation utiles

Rapporte : `domain_delta`, `general_delta`, `cost_ratio`, et éventuellement gain domaine par GB/heure uniquement comme indicateur secondaire.

## Code minimal observable

```text
Ablation matrix
---------------
Data        : identical
Base model  : identical revision
Tokens seen : matched
Benchmark   : identical
LoRA LR     : tuned for LoRA
Full FT LR  : tuned separately but documented
Seeds       : repeat if feasible
```

## Laboratoire guidé

1. Vérifie mémoire full FT sur 2–5 steps.  
2. Lance LoRA et full FT avec checkpoints réguliers.  
3. Choisis chaque meilleur checkpoint sur validation, pas le dernier automatiquement.  
4. Évalue test domaine et regression général.  
5. Compare l’évolution du forgetting par checkpoint si possible.  
6. Écris une recommandation selon trois scénarios de ressources.

## Ce que tu dois observer

- Full FT peut améliorer certaines tâches mais suradapter plus vite.
- Le LR optimal des deux méthodes diffère souvent.
- Le coût disque/opérationnel des checkpoints complets compte aussi.

## À ne pas confondre

- Même budget de steps ≠ mêmes paramètres modifiés.
- Meilleur train loss full FT ≠ meilleure généralisation.
- LoRA plus petit ≠ nécessairement moins bon.

## Erreurs fréquentes

- Full FT sur un modèle trop gros et OOM à mi-run.
- Utiliser le même LR “pour être fair” alors qu’il est inadapté à une condition.
- Évaluer uniquement le domaine.

## Exercices

1. Définis ce qui doit rester identique et ce qui peut être tune séparément.
2. Écris un critère d’acceptation incluant forgetting.
3. Explique quand un gain de 1 point peut ou non justifier 2× VRAM.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Exécuter les deux méthodes de façon sûre.
- Rapporter coût + qualité + forgetting.
- Choisir les checkpoints par validation.
- Écrire une recommandation conditionnelle.

## Fiche mémo

Comparer LoRA/full FT apprend à penser en **gain marginal contre liberté et coût supplémentaires**.

## Lien avec le module suivant

Le projet 4 teste si une étape de continued pretraining apporte quelque chose avant SFT.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
