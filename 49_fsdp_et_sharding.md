# Module 49 — FSDP et sharding

## Pourquoi ce module arrive ici

Quand une réplique complète du modèle et de ses états ne tient plus sur chaque GPU, il faut répartir ces états. FSDP sharde les paramètres, gradients et/ou optimizer states selon la stratégie utilisée.

## Objectifs

Comprendre sharding, all-gather, reduce-scatter, full shard, state dict sharded/full et différence conceptuelle avec DDP/ZeRO. Configurer FSDP via Accelerate au niveau de base.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Training distribué

## Définitions concrètes

### Sharding

**Définition concrète.** Division d’un tenseur/état entre plusieurs ranks.


**Exemple simple.** Chaque GPU conserve une fraction d’une grande matrice ou de l’état associé selon la stratégie.

### All-gather

**Définition concrète.** Collective réunissant temporairement des shards nécessaires sur les ranks.


**Exemple simple.** Reconstruire des paramètres pour un forward.

### Reduce-scatter

**Définition concrète.** Réduit des valeurs entre ranks puis distribue les morceaux du résultat.


**Exemple simple.** Utile pour gradients sharded.

### Full shard

**Définition concrète.** Stratégie shardant paramètres, gradients et optimizer states.


**Exemple simple.** Comparable conceptuellement à ZeRO-3 sur ces catégories.

### State dict sharded

**Définition concrète.** Checkpoint où chaque rank sauvegarde des shards plutôt qu’un gros tenseur complet.


**Exemple simple.** Doit être consolidé/chargé avec le bon mécanisme.

### Auto wrap

**Définition concrète.** Règle déterminant quels modules/blocs sont encapsulés/shardés.


**Exemple simple.** Souvent blocs Transformer.

## Intuition simple

DDP donne un livre complet à chaque personne. FSDP découpe le livre entre elles et rassemble temporairement les pages nécessaires lorsqu’un chapitre doit être lu/calculé, puis redistribue les résultats.

## Ce qui se passe réellement sous le capot

1. Au repos, chaque rank possède seulement des shards selon la stratégie.
2. Avant le calcul d’un bloc, les paramètres nécessaires sont all-gathered.
3. Le bloc effectue forward/backward.
4. Les gradients sont réduits et reshared via collectives adaptées.
5. Les optimizer states restent eux aussi distribués en full shard.
6. Le prochain bloc est traité.
7. Les checkpoints peuvent rester sharded ou être consolidés selon objectif.

## Exemple minimal à comprendre mentalement

Modèle avec 8 GB de paramètres bruts sur 4 GPU : idéalement, le stockage sharded des paramètres peut tendre vers ~2 GB/GPU pour cette composante, **mais** le pic réel inclut all-gathers, activations, buffers, fragmentation et autres états. On ne divise donc pas simplement toute la VRAM par 4.

## Formules et notation utiles

Dans un idéal simplifié, une composante entièrement shardée de taille `M` sur `N` ranks utilise environ `M/N` au repos par rank. Le peak est supérieur à cette valeur en raison des paramètres temporairement matérialisés et des activations.

## Code minimal observable

```yaml
# Extrait conceptuel d’une config Accelerate FSDP.
# Les options exactes dépendent de la version installée.
distributed_type: FSDP
mixed_precision: bf16
num_processes: 4
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_state_dict_type: SHARDED_STATE_DICT
```

## Laboratoire guidé

1. Lis la config FSDP générée par `accelerate config`.  
2. Identifie ce qui est sharded.  
3. Sur plusieurs GPU, compare mémoire/rank DDP vs FSDP pour un petit modèle suffisamment gros pour voir la différence.  
4. Sauvegarde un checkpoint sharded et documente la procédure de restauration.  
5. Étudie l’outil de merge/consolidation correspondant à ta version si tu as besoin d’un checkpoint unique.

## Ce que tu dois observer

- FSDP réduit la duplication de certains états mais augmente les communications.
- Le wrapping influence mémoire et performance.
- Les checkpoints distribués exigent une procédure explicite.
- Les versions récentes de PyTorch/Accelerate font évoluer les API FSDP; figer la version du cours/projet est important.

## À ne pas confondre

- FSDP ≠ DDP.
- Shard au repos ≠ aucune matérialisation temporaire.
- FSDP checkpoint ≠ forcément un fichier unique immédiatement chargeable comme un modèle standard.

## Erreurs fréquentes

- Conclure que 4 GPU donnent exactement 4× la capacité mémoire.
- Ignorer le réseau/NVLink/PCIe et le coût des collectives.
- Ne pas tester la restauration des checkpoints avant un long run.

## Exercices

1. Explique all-gather avec 4 shards.
2. Explique reduce-scatter.
3. Compare conceptuellement DDP, FSDP full shard et ZeRO-3.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer ce qui est sharded.
- Expliquer all-gather/reduce-scatter.
- Distinguer FSDP de DDP.
- Créer et comprendre une config Accelerate FSDP.

## Fiche mémo

FSDP échange **moins de duplication mémoire** contre **plus d’orchestration et de communication**.

## Lien avec le module suivant

Le module 50 rassemble toutes les composantes pour construire un vrai budget mémoire.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
