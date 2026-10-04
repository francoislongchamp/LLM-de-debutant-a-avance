# Module 52 — Logging et TensorBoard

## Pourquoi ce module arrive ici

Sans séries temporelles de loss, LR, gradients, mémoire et débit, le diagnostic devient une suite d’impressions. Le logging transforme le run en expérience observable.

## Objectifs

Choisir les métriques essentielles, comprendre step vs wall time, utiliser TensorBoard et créer des conventions de run reproductibles.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fiabilité du training

## Définitions concrètes

### Scalar metric

**Définition concrète.** Valeur numérique enregistrée avec un step.


**Exemple simple.** train/loss.

### Throughput

**Définition concrète.** Quantité traitée par unité de temps.


**Exemple simple.** tokens/s.

### Wall time

**Définition concrète.** Temps réel écoulé.


**Exemple simple.** Utile pour coût/efficacité.

### Run metadata

**Définition concrète.** Configuration décrivant l’expérience.


**Exemple simple.** git commit, model, dataset, seed, hardware.

### TensorBoard

**Définition concrète.** Interface pour visualiser scalars, histogrammes et autres logs compatibles.


**Exemple simple.** Comparer plusieurs runs de loss/LR.

## Intuition simple

Un run sans métadonnées est comme une expérience de laboratoire sans cahier : même un bon résultat devient difficile à reproduire ou expliquer.

## Ce qui se passe réellement sous le capot

1. Donner un identifiant unique au run.
2. Sauvegarder config complète avant training.
3. Logger train loss, eval loss, LR, grad norm, tokens/s et mémoire utile.
4. Logger reward components pour RL.
5. Enregistrer global step/tokens seen/wall time.
6. Visualiser plusieurs runs avec axes comparables.
7. Conserver raw logs avec checkpoints et résultats benchmark.

## Exemple minimal à comprendre mentalement

Convention utile :

```text
runs/2026-10-04_qwen06b_lora_r16_seed42/
  config.yaml
  metrics/
  checkpoints/
  eval/
  environment.txt
```

Le nom aide, mais la vraie source de vérité reste la configuration stockée.

## Code minimal observable

```bash
tensorboard --logdir outputs/

# Puis ouvrir l’adresse locale indiquée par TensorBoard.
# Dans Trainer/SFTConfig, utiliser report_to="tensorboard" selon ta version/config.
```

## Laboratoire guidé

1. Logge au minimum train/loss, eval/loss, learning_rate.  
2. Ajoute grad_norm, tokens/s et peak memory si disponible.  
3. Lance deux runs ne différant que par un hyperparamètre.  
4. Superpose les courbes.  
5. Écris une conclusion fondée sur les courbes et le benchmark, pas seulement sur la dernière loss.

## Ce que tu dois observer

- Une fréquence de logging trop haute peut ajouter du bruit/overhead; trop basse masque les événements.
- Comparer des courbes en steps peut être trompeur si batch global diffère; tokens vus peut aider.
- Le meilleur throughput n’est pas forcément la meilleure qualité.

## À ne pas confondre

- Logging ≠ evaluation.
- TensorBoard ≠ stockage du modèle.
- Wall time ≠ GPU time exactement.

## Erreurs fréquentes

- Réutiliser le même output_dir et mélanger les runs.
- Ne pas logguer le LR réel.
- Oublier les versions package/CUDA/driver dans un bug difficile.

## Exercices

1. Définis 10 métriques par priorité essential/nice-to-have.
2. Explique quand utiliser steps vs tokens vus en axe x.
3. Crée une convention de noms de runs.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Lancer TensorBoard sur deux runs.
- Retrouver config et métriques d’un run.
- Comparer sur un axe cohérent.
- Produire une conclusion reproductible.

## Fiche mémo

Le logging transforme le training en expérience scientifique : on peut voir, comparer, expliquer et reproduire.

## Lien avec le module suivant

Le module 53 construit le benchmark qui mesure ce que la loss de training ne peut pas résumer.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
