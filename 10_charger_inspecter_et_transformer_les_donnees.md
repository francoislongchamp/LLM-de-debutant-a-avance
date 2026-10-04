# Module 10 — Charger, inspecter et transformer les données

## Pourquoi ce module arrive ici

Un dataset réel doit être chargé, filtré, transformé et parfois tokenisé. Chaque transformation peut supprimer ou altérer des exemples; il faut donc rendre ces opérations observables et reproductibles.

## Objectifs

Utiliser `datasets` pour charger JSON/JSONL, inspecter features, `map`, `filter`, `select`, `shuffle` et sauvegarder. Comprendre transformation paresseuse vs matérialisée au niveau conceptuel et éviter les mutations opaques.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning — données

## Définitions concrètes

### Dataset

**Définition concrète.** Collection structurée d’exemples accessible par colonnes et lignes.


**Exemple simple.** Une table de messages et catégories.

### Feature

**Définition concrète.** Description du type d’une colonne.


**Exemple simple.** `messages` peut être une liste de structures.

### map

**Définition concrète.** Transformation appliquée aux exemples ou batches.


**Exemple simple.** Ajouter `length_chars`.

### filter

**Définition concrète.** Conservation des exemples satisfaisant un prédicat.


**Exemple simple.** Retirer les réponses vides.

### shuffle

**Définition concrète.** Permutation pseudo-aléatoire contrôlable par seed.


**Exemple simple.** `seed=42` rend le résultat reproductible.

### Fingerprint

**Définition concrète.** Identifiant interne lié au contenu/transformation utilisé par la bibliothèque pour le cache.


**Exemple simple.** Une transformation change généralement l’empreinte.

## Intuition simple

Traite ton pipeline de données comme un programme de compilation : données brutes → étapes explicites → dataset final. À chaque étape, tu dois pouvoir dire combien d’exemples entrent, combien sortent et pourquoi.

## Ce qui se passe réellement sous le capot

1. Charger le fichier avec un builder approprié.
2. Inspecter colonnes, types et quelques exemples.
3. Ajouter des métriques simples avec `map`.
4. Filtrer les entrées invalides avec des conditions explicites.
5. Normaliser les champs de manière déterministe.
6. Sauvegarder ou versionner le résultat transformé.
7. Recalculer les statistiques après chaque étape majeure.

## Exemple minimal à comprendre mentalement

Avant nettoyage :

```text
1000 exemples
- 12 réponses vides
- 8 rôles invalides
- 40 doublons exacts
```

Après pipeline :

```text
940 exemples valides
```

Le nombre final n’est pas “magique” : le pipeline doit expliquer les 60 retraits.

## Code minimal observable

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="data/train.jsonl", split="train")
print(ds)
print(ds.features)
print(ds[0])

def add_len(row):
    row["chars"] = sum(len(m["content"]) for m in row["messages"])
    return row

ds2 = ds.map(add_len)
ds3 = ds2.filter(lambda r: r["chars"] > 0)
print(len(ds), len(ds3))
```

## Laboratoire guidé

1. Charge ton dataset.  
2. Affiche 5 exemples choisis à différents indices, pas seulement les premiers.  
3. Ajoute `chars`, nombre de tours et catégorie.  
4. Filtre les réponses vides ou invalides.  
5. `shuffle(seed=123)` deux fois et confirme le même ordre.  
6. Écris un petit rapport avant/après transformation.

## Ce que tu dois observer

- Les erreurs de schéma apparaissent plus tôt si tu valides dès le chargement.
- Une transformation simple peut changer fortement la distribution.
- Une seed contrôle le pseudo-aléatoire mais ne remplace pas la version des données.

## À ne pas confondre

- `map` ≠ entraînement.
- Shuffle ≠ split.
- Filtrer un exemple ≠ masquer certains de ses tokens dans la loss.

## Erreurs fréquentes

- Transformer des données sans mesurer les comptes avant/après.
- Cacher des règles métier complexes dans une lambda impossible à auditer.
- Inspecter uniquement `ds[0]`.

## Exercices

1. Ajoute un contrôle de rôles autorisés.
2. Produit un histogramme textuel des longueurs par tranches.
3. Explique ce que tu dois versionner pour reproduire le dataset final.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Charger et inspecter un dataset avec `datasets`.
- Expliquer `map`, `filter`, `shuffle` et seed.
- Montrer les statistiques avant/après nettoyage.
- Reproduire exactement le dataset transformé depuis les données brutes.

## Fiche mémo

Un pipeline de données sérieux rend chaque modification explicite, mesurable et reproductible.

## Lien avec le module suivant

Le module 11 sépare maintenant ce dataset en train, validation et test sans contaminer l’évaluation.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
