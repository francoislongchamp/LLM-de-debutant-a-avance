# Module 48 — Data Parallel et DDP

## Pourquoi ce module arrive ici

Avant de shard un modèle, il faut comprendre le cas le plus simple du multi-GPU : chaque GPU possède une copie du modèle et traite une partie différente du batch. Les gradients sont ensuite synchronisés.

## Objectifs

Comprendre process/rank, replica, local batch, global batch, all-reduce de gradients et DistributedDataParallel. Lancer un exemple multi-GPU si disponible ou simuler les calculs de batch sur une seule machine.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Training distribué

## Définitions concrètes

### Data parallel

**Définition concrète.** Même modèle répliqué, données différentes sur chaque device.


**Exemple simple.** GPU0 traite batch A, GPU1 batch B.

### Process / rank

**Définition concrète.** Processus distribué associé à un indice global.


**Exemple simple.** rank 0, rank 1…

### Local rank

**Définition concrète.** Indice du processus sur une machine donnée.


**Exemple simple.** Souvent utilisé pour choisir la GPU locale.

### Replica

**Définition concrète.** Copie complète du modèle sur un device.


**Exemple simple.** DDP garde une réplique par processus/GPU.

### All-reduce

**Définition concrète.** Collective combinant une valeur de tous les ranks et redistribuant le résultat.


**Exemple simple.** Synchroniser les gradients moyens/sommes.

### Distributed sampler

**Définition concrète.** Répartit les exemples entre les ranks pour éviter que tous lisent le même mini-batch.


**Exemple simple.** Chaque rank reçoit une partition différente.

## Intuition simple

Quatre personnes possèdent chacune la même feuille de modèle. Chacune travaille sur des exemples différents, calcule ses corrections, puis tout le monde met en commun les corrections avant de modifier sa copie de la même manière.

## Ce qui se passe réellement sous le capot

1. Lancer un process par GPU.
2. Chaque process charge une réplique du modèle.
3. Le sampler fournit des données différentes à chaque rank.
4. Chaque rank fait forward/backward local.
5. DDP synchronise les gradients, généralement via all-reduce/buckets.
6. Chaque optimizer local applique alors la même mise à jour synchronisée.
7. Les répliques restent cohérentes après chaque step.

## Exemple minimal à comprendre mentalement

```text
GPU0: 2 exemples
GPU1: 2 exemples
GPU2: 2 exemples
GPU3: 2 exemples
accumulation=4
```

Batch effectif approximatif = `2×4×4=32` exemples/update.

## Formules et notation utiles

Le modèle complet doit tenir sur **chaque GPU** en DDP classique. DDP augmente donc surtout le débit/batch global, mais ne résout pas le cas où les poids + états ne tiennent pas sur une seule GPU.

## Code minimal observable

```bash
# Avec Accelerate, après configuration multi-GPU :
accelerate config
accelerate launch train.py

# Le script de training peut ensuite être préparé par Accelerate
# ou utiliser torch.distributed/DDP directement.
```

## Laboratoire guidé

**Si tu as plusieurs GPU :**  
1. Lance un script simple avec 2 processes.  
2. Affiche rank/local_rank/device.  
3. Vérifie que chaque process reçoit des indices de données différents.  
4. Compare le débit 1 GPU vs 2 GPU.  
5. Vérifie que le modèle complet tient toujours sur chaque GPU.

**Si tu n’as qu’une GPU :** calcule sur papier les batchs locaux/globaux pour 2, 4 et 8 ranks et étudie les collectives avec un petit exemple CPU `torch.distributed` plus tard.

## Ce que tu dois observer

- Le speedup n’est jamais parfaitement linéaire à cause des communications et du pipeline de données.
- DDP réplique le modèle; sa mémoire par GPU n’est pas divisée par le nombre de GPU.
- Le global batch change si tu ajoutes des GPUs sans ajuster batch/accumulation.

## À ne pas confondre

- DDP ≠ model parallelism.
- Rank ≠ GPU model/name.
- All-reduce de gradients ≠ partager les activations de toutes les couches.

## Erreurs fréquentes

- Ajouter des GPUs sans ajuster le batch global et comparer comme si l’expérience était identique.
- Laisser chaque rank lire tous les mêmes exemples.
- Faire du logging/checkpoint concurrent depuis tous les ranks sans coordination.

## Exercices

1. Calcule batch effectif pour 8 GPUs×2 exemples×4 accumulation.
2. Explique pourquoi un 70B qui ne tient pas sur une GPU ne devient pas magiquement chargeable avec DDP pur.
3. Dessine le flux gradient local→all-reduce→optimizer.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer la réplication DDP.
- Calculer batch global.
- Expliquer all-reduce de gradients.
- Dire quand DDP aide et quand il n’aide pas la mémoire du modèle.

## Fiche mémo

DDP répartit **les données et le calcul**, pas les poids du modèle : chaque GPU garde une réplique complète.

## Lien avec le module suivant

Le module 49 shard les paramètres/gradients/optimizer states avec FSDP pour dépasser cette limite.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
