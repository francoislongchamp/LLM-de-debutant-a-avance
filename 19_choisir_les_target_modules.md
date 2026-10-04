# Module 19 — Choisir les target modules

## Pourquoi ce module arrive ici

LoRA n’a d’effet que dans les modules où il est injecté. Cibler Q/V seulement, toute l’attention ou toutes les couches linéaires change fortement le nombre de paramètres et la capacité d’adaptation.

## Objectifs

Identifier les couches linéaires d’une architecture, comprendre l’effet de cibler attention vs MLP, utiliser `all-linear` de manière consciente et comparer les stratégies expérimentalement.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning léger

## Définitions concrètes

### Target module

**Définition concrète.** Module dont la couche linéaire reçoit un adapter LoRA.


**Exemple simple.** `q_proj` ou `down_proj`.

### Attention projections

**Définition concrète.** Matrices Q/K/V/O du mécanisme d’attention.


**Exemple simple.** Cibler q/v est une stratégie historique courante.

### MLP projections

**Définition concrète.** Grandes matrices du feed-forward.


**Exemple simple.** `gate_proj`, `up_proj`, `down_proj` dans certaines familles.

### all-linear

**Définition concrète.** Sélecteur PEFT pour viser les couches linéaires pertinentes sans lister les noms architecture par architecture.


**Exemple simple.** Souvent utilisé pour un style QLoRA.

## Intuition simple

Le rank dit “combien de capacité” tu donnes à chaque correction; target_modules dit **où** le modèle a le droit d’apprendre cette correction.

## Ce qui se passe réellement sous le capot

1. Inspecter `named_modules()` et la structure imprimée du modèle.
2. Identifier les projections d’attention et MLP.
3. Construire une configuration minimale (ex. q/v).
4. Construire une configuration plus large (attention complète).
5. Construire éventuellement `all-linear`.
6. Mesurer paramètres entraînables et mêmes benchmarks.
7. Conserver la stratégie la plus simple qui satisfait les objectifs.

## Exemple minimal à comprendre mentalement

Pour 24 blocs, cibler seulement `q_proj,v_proj` injecte 2 modules par bloc. Cibler `q,k,v,o,gate,up,down` en injecte 7. À rank égal, le second choix peut donc utiliser plusieurs fois plus de paramètres LoRA.

## Formules et notation utiles

Le total LoRA est la somme sur chaque module ciblé :

\[
N_{total}=\sum_i r(in_i+out_i)
\]

Deux configurations ayant le même r peuvent donc avoir des tailles très différentes.

## Code minimal observable

```python
import torch

for name, mod in trainer.model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        print(name, mod.in_features, mod.out_features)
```

## Laboratoire guidé

1. Liste toutes les couches linéaires avant LoRA.  
2. Regroupe-les par suffixe (`q_proj`, `up_proj`, etc.).  
3. Calcule combien d’instances de chaque type existent.  
4. Compare q/v, attention complète et all-linear avec le même r.  
5. Mesure trainable params et benchmark.

## Ce que tu dois observer

- Les MLP peuvent ajouter beaucoup de paramètres LoRA.
- Les noms varient entre architectures; `all-linear` évite une partie de cette dépendance.
- Une configuration plus large n’est pas automatiquement meilleure sur validation.

## À ne pas confondre

- Target module ≠ layer number.
- `all-linear` ≠ full fine-tuning.
- Même r ≠ même taille d’adapter si les cibles changent.

## Erreurs fréquentes

- Copier une liste de targets d’une architecture différente.
- Supposer que Q/V est toujours optimal.
- Comparer des configurations avec des budgets de paramètres très différents sans le signaler.

## Exercices

1. Calcule le nombre total d’adapters pour 32 blocs et 7 targets/bloc.
2. Explique pourquoi cibler le MLP peut augmenter la capacité.
3. Propose une expérience à budget de paramètres approximativement constant entre deux stratégies.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Identifier les modules linéaires de ton modèle.
- Expliquer attention targets vs MLP targets.
- Calculer le total de paramètres LoRA par somme des modules.
- Justifier une stratégie plutôt que la copier.

## Fiche mémo

LoRA répond à deux questions : **combien** de capacité (rank) et **où** l’injecter (targets).

## Lien avec le module suivant

Le module 20 combine LoRA à une base quantifiée 4 bits : QLoRA.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
