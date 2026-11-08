# Module 45 — Construire un pipeline de corpus traçable

## 1. Ce que tu vas apprendre

Le modèle ne voit pas les sources d’origine : il voit le résultat des extractions, filtres, mélanges et segmentations. Ces étapes font partie de l’expérience.

Tu vas conserver des étapes identifiables, expliquer leurs comptes, écrire des shards et vérifier leurs hashes dans un manifeste.

**Prérequis :** modules 9 à 11, 25 et 38 à 44.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python uniquement pour le pipeline miniature.

## 2. Un graphe de transformations

```text
Sources et provenance
→ extraction adaptée au format
→ contrôles et nettoyage versionnés
→ déduplication avec trace des copies
→ séparation des groupes liés
→ mixture puis tokenisation figée
→ séquences/shards + manifeste
```

L’ordre exact dépend du projet. La règle importante est de garder les groupes liés ensemble et d’éviter qu’une transformation destinée au train exploite le test pour choisir ses paramètres.

Un **split** définit un rôle d’évaluation. Un **shard** est une fraction physique d’un dataset. Cent shards de train ne font pas cent ensembles de validation indépendants.

## 3. Ce qui doit survivre au nettoyage

Chaque document a un ID stable, une source, un groupe pertinent, une version et un texte. Les conditions d’utilisation connues et décisions sur les informations à conserver doivent être documentées selon le corpus réel.

Une extraction HTML ou PDF peut perdre tableaux, formules ou ordre de lecture. Un filtre de langue ou qualité est un estimateur, pas une preuve de vérité. Un détecteur de données personnelles ou sensibles a des limites : une regex d’emails ne certifie pas un corpus dépourvu de telles informations.

Pour notre corpus artificiel de prose, nous normalisons simplement les espaces. Cette règle serait inadaptée à du code où indentation et sauts de ligne portent une information.

## 4. Compter plusieurs unités

À chaque étape, rapporte documents, octets/caractères, tokens et proportions par source quand ils sont disponibles.

Retirer une longue page peut réduire peu le nombre de documents mais beaucoup les tokens. Dédupliquer une source très répétitive modifie aussi la mixture naturelle.

Un taux de rétention `N_après/N_avant` doit annoncer son unité. Il ne mesure pas automatiquement une amélioration de qualité.

## 5. Laboratoire autonome — Six lignes vers des shards vérifiables

Le script produit ses propres données et écrit sous `outputs/module45/`. Il ne traite pas de vrais documents personnels. Sa tokenisation par octets est une convention pédagogique, distincte du BPE du module 38.

```python
import hashlib
import json
from pathlib import Path

root = Path('outputs/module45')
root.mkdir(parents=True, exist_ok=True)
raw = [
    {'id': 'a', 'source': 'alpha', 'group': 'doc-a', 'text': ' Le chat dort. '},
    {'id': 'b', 'source': 'alpha', 'group': 'doc-b', 'text': 'Le   chat dort.'},
    {'id': 'c', 'source': 'beta', 'group': 'doc-c', 'text': 'TLS protege un flux.'},
    {'id': 'd', 'source': 'beta', 'group': 'doc-d', 'text': '  '},
    {'id': 'e', 'source': 'beta', 'group': 'doc-e', 'text': 'DNS relie nom et adresse.'},
    {'id': 'f', 'source': 'alpha', 'group': 'doc-f', 'text': 'Un batch contient des exemples.'},
]


def write_jsonl(relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
    return relative


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


files = [write_jsonl('raw.jsonl', raw)]
clean = []
rejected = []
for row in raw:
    text = ' '.join(row['text'].split())
    if not text:
        rejected.append({'id': row['id'], 'reason': 'empty_after_space_normalization'})
    else:
        clean.append(dict(row, text=text))
files.append(write_jsonl('clean.jsonl', clean))

unique = []
seen = {}
duplicates = []
for row in clean:
    key = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
    if key in seen:
        duplicates.append({'removed': row['id'], 'kept': seen[key]})
    else:
        seen[key] = row['id']
        unique.append(row)
files.append(write_jsonl('deduplicated.jsonl', unique))

# Affectation explicite par groupe pour cette petite démonstration.
groups = sorted({row['group'] for row in unique})
validation_groups = {groups[-1]}
splits = {'train': [], 'validation': []}
for row in unique:
    split = 'validation' if row['group'] in validation_groups else 'train'
    ids = [value + 2 for value in row['text'].encode('utf-8')] + [1]
    splits[split].append(dict(row, input_ids=ids))

shards = []
for split, rows in splits.items():
    for index in range(0, len(rows), 2):
        content = rows[index:index + 2]
        name = write_jsonl(f'{split}/shard-{index // 2:03d}.jsonl', content)
        files.append(name)
        shards.append({'path': name, 'split': split, 'documents': len(content),
                       'input_tokens': sum(len(row['input_ids']) for row in content)})

manifest = {'pipeline_version': 'space-normalize_exact-dedup_bytes-v1',
            'counts': {'raw': len(raw), 'clean': len(clean), 'unique': len(unique)},
            'rejections': rejected, 'duplicates': duplicates,
            'split_groups': {name: sorted({row['group'] for row in rows}) for name, rows in splits.items()},
            'tokenizer': 'utf8_bytes_plus_2_eos_1_pad_0', 'shards': shards,
            'sha256': {name: digest(root / name) for name in files}}
(root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
assert all(digest(root / name) == expected for name, expected in manifest['sha256'].items())
print('Comptes :', manifest['counts'])
print('Splits :', {name: len(rows) for name, rows in splits.items()})
print('Shards :', shards)
print('Tous les hashes verifies')
```

Tu dois retrouver **6 → 5 → 4 documents**, un rejet vide et une copie retirée. La séparation donne **3 documents train**, **1 validation**, et **3 shards** au total avec notre limite de deux documents par shard.

L’affectation explicite est lisible mais ne prétend pas constituer un split représentatif sur quatre documents. Pour le projet réel, définis groupes, proportions et règles selon le module 11.

Les copies sont tracées dans `duplicates` plutôt que perdues silencieusement. Avec des sources multiples, conserve toutes les provenances pertinentes dans la représentation finale ou le manifeste.

## 6. Rendre le résultat reproductible

Le manifeste lie règles, groupes, tokenizer, comptes et fichiers. Un hash détecte un changement d’octets ; il ne garantit pas une absence de biais ou de contamination.

Pour reproduire, repars des mêmes sources et versions dans un nouveau dossier, puis compare les contenus attendus. Un pipeline distribué peut écrire les mêmes lignes dans un autre ordre : décide si l’ordre fait partie du contrat de reproductibilité.

Une reprise de streaming doit aussi savoir quels shards et quelles positions ont déjà été consommés. Le manifeste de données et le checkpoint d’entraînement se complètent.

## 7. Expériences guidées

1. Ajoute une longue copie et compare rétention en documents et en octets/tokens.
2. Conserve les sauts de ligne au lieu de normaliser tous les espaces ; documente l’effet sur les hashes et les comptes.
3. Modifie un shard après écriture et vérifie que la comparaison de hash détecte ce changement.
4. Remplace la tokenisation d’octets par le tokenizer figé du module 38 ; garde son hash et compare les compteurs.

## 8. Exercices avec corrigés

### Exercice 1 — Shard ou split ?

Un corpus train est découpé en dix fichiers. A-t-on dix validations ?

<details>
<summary>Corrigé</summary>

Non. Ce sont dix shards d’un même rôle train. L’indépendance d’évaluation vient de l’affectation des données, pas du nombre de fichiers.

</details>

### Exercice 2 — Nettoyage universel

Pourquoi remplacer tous les espaces et sauts de ligne par un espace peut-il abîmer du code ?

<details>
<summary>Corrigé</summary>

Indentation et frontières de lignes peuvent être syntaxiques ou sémantiques. La normalisation de prose du pilote ne doit pas être généralisée à tous les formats.

</details>

### Exercice 3 — Hash

Un fichier conserve son nombre de lignes mais une réponse change. Un compteur de lignes détecte-t-il cela ?

<details>
<summary>Corrigé</summary>

Pas nécessairement. Un hash de contenu peut détecter le changement d’octets ; il ne décide pas si la nouvelle réponse est meilleure.

</details>

### Exercice 4 — Mixture

Pourquoi recompter les sources après déduplication ?

<details>
<summary>Corrigé</summary>

Les taux de répétition diffèrent selon les sources. Retirer les copies change leurs proportions naturelles et donc le signal si l’on concatène simplement le résultat.

</details>

## 9. Validation — avant de continuer

- [ ] Je dessine les étapes et leurs entrées/sorties.
- [ ] Je retrouve 6/5/4, les splits et les shards du pilote.
- [ ] Je conserve IDs, groupes, sources et raisons de retrait.
- [ ] Je distingue hashes, provenance et qualité.
- [ ] Je reproduis une version sans écraser les données brutes réelles.

## Fiche mémo

```text
Chaque transformation modifie le corpus que voit le modèle.
Conserver étapes, comptes, provenance, groupes et tokenizer jusqu’aux shards.
```

Le [module 46](46_deduplication_exacte_et_near_duplicate.md) approfondit la détection des copies.
