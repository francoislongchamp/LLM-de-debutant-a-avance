# Module 10 — Charger, inspecter et transformer des données

## 1. Ce que tu vas apprendre

Au [module 9](09_construire_un_dataset_de_qualite.md), tu as vérifié les exemples un par un. Ici, tu vas transformer une collection avec Hugging Face Datasets tout en gardant des comptes explicables.

À la fin, tu pourras charger du JSONL, lire un schéma, ajouter des colonnes avec `map`, sélectionner avec `filter` et `select`, mélanger avec une seed et vérifier une sauvegarde.

**Prérequis :** module 9 ; listes, dictionnaires et fonctions Python.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et `datasets` ; CPU, sans téléchargement de modèle. Si nécessaire : `python -m pip install datasets` dans ton environnement du cours.

## 2. Une table d’exemples, pas un modèle

Un objet `Dataset` représente une collection structurée. Chaque ligne est un exemple et chaque colonne décrit un champ.

| Ligne | `id` | `question` | `answer` |
| --- | --- | --- | --- |
| 0 | a | Qu’est-ce qu’un batch ? | Un groupe d’exemples. |
| 1 | b | Que calcule backward ? | Les gradients. |

`dataset[0]` retourne une ligne. `dataset["answer"]` donne accès à une colonne. Les **features** décrivent les types des colonnes, par exemple texte ou liste de messages.

Charger une table ne lance aucun entraînement. Ajouter une colonne de longueurs ne tokenise pas automatiquement les réponses et ne modifie aucun poids.

## 3. Les opérations à distinguer

| Opération | Question traitée | Effet typique sur le nombre de lignes |
| --- | --- | --- |
| `map` | Que faut-il calculer ou transformer ? | Inchangé dans notre exemple |
| `filter` | Quels exemples conserver ? | Identique ou réduit |
| `select` | Quels indices précis examiner ? | Nombre d’indices sélectionnés |
| `shuffle` | Dans quel ordre parcourir ? | Inchangé |
| `save_to_disk` | Comment conserver cette version ? | Sauvegarde, pas sélection |

Avec `batched=True`, `map` peut aussi produire un nombre de lignes différent ; ce n’est donc pas une règle universelle de conservation de la taille.

Une fonction de transformation lisible retourne ici les champs à ajouter, sans modifier l’exemple reçu. Cela rend son intention plus facile à vérifier.

### Un compte à refaire à la main

Supposons 8 lignes, dont 2 réponses vides. Si ces deux lignes sont les seuls retraits, il doit rester 6 lignes. Une transformation « trim des espaces » peut modifier le contenu sans changer ce nombre.

Les catégories peuvent pourtant changer de proportion : si les deux retraits appartiennent au diagnostic, cette catégorie devient moins représentée. Recompter seulement le total est insuffisant.

## 4. Laboratoire autonome — Du JSONL à une version sauvegardée

Le script fabrique quatre lignes pédagogiques et place son cache dans le même dossier temporaire. Il utilise un dossier temporaire qui sera supprimé à la fin : aucun fichier de ton dataset personnel n’est modifié.

Avant de l’exécuter, prédis les IDs conservés et le nombre de lignes après chaque étape.

```python
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from datasets import load_dataset, load_from_disk

rows = [
    {"id": "a", "category": "definition", "answer": " Un batch regroupe des exemples. "},
    {"id": "b", "category": "comparaison", "answer": "backward calcule ; step met à jour."},
    {"id": "c", "category": "diagnostic", "answer": "   "},
    {"id": "d", "category": "diagnostic", "answer": "Vérifie les formes des tenseurs."},
]


def add_metrics(row):
    cleaned = row["answer"].strip()
    return {"answer": cleaned, "answer_chars": len(cleaned)}


def has_answer(row):
    return row["answer_chars"] > 0


with TemporaryDirectory(prefix="module10-") as directory:
    root = Path(directory)
    path = root / "brut.jsonl"
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    raw = load_dataset("json", data_files=str(path), split="train", cache_dir=str(root / "cache"))
    measured = raw.map(add_metrics)
    clean = measured.filter(has_answer)

    print("Schéma :", clean.features)
    print("Comptes :", len(raw), len(measured), len(clean))
    print("IDs conservés :", list(clean["id"]))
    print("Catégories :", dict(Counter(clean["category"])))
    indices = sorted({0, len(clean) // 2, len(clean) - 1})
    print("Inspection à différents indices :", clean.select(indices).to_list())

    first = clean.shuffle(seed=42)
    second = clean.shuffle(seed=42)
    print("Même ordre avec la même seed :", list(first["id"]) == list(second["id"]))

    destination = root / "version_nettoyee"
    clean.save_to_disk(str(destination))
    restored = load_from_disk(str(destination))
    print("Contenu restauré identique :", restored.to_list() == clean.to_list())
    print("Première réponse brute préservée :", repr(raw[0]["answer"]))
```

Tu dois retrouver **4 → 4 → 3 lignes**, les IDs **a, b, d**, une ligne par catégorie, et deux comparaisons finales vraies. L’objet `raw` conserve les espaces initiaux ; `map` retourne un autre dataset.

`split="train"` est ici le nom technique du split chargé par le builder. Il ne transforme pas ces données en un train indépendant d’une validation : cette séparation reste à construire au module 11.

### Ce que le code vérifie réellement

La règle `answer_chars > 0` vérifie la présence d’une réponse après suppression des espaces de bord. Elle ne juge ni sa vérité ni sa clarté. Les contrôles de contenu du module 9 restent nécessaires.

Les longueurs sont des caractères Python, pas des tokens. Une réponse courte en caractères peut encore être fortement fragmentée par un tokenizer.

## 5. Reprendre ton corpus du module 9

Remplace la création des lignes pédagogiques par un chargement de `data/module09/dataset.jsonl`. Le chemin dépend du dossier depuis lequel tu lances Python.

Ton schéma contient `messages`. Adapte donc la mesure : compte par exemple la longueur de `row["messages"][1]["content"]` pour notre format à un échange. Ne réutilise pas une fonction conçue pour une colonne `answer` absente de ce fichier.

Conserve les IDs, catégories et sources. Pour une vraie sortie persistante, remplace le dossier temporaire par un dossier de version explicite, par exemple `data/module10/v1`, puis documente les transformations.

## 6. Matérialisé, paresseux et cache

Un `Dataset` classique s’appuie généralement sur des données au format Arrow. Ses transformations peuvent être matérialisées et mises en cache ; toutes les données n’ont pas nécessairement à être copiées dans la RAM en une fois.

Un `IterableDataset`, notamment en streaming, est parcouru progressivement. Il ne propose pas les mêmes accès par indices ni les mêmes garanties de mélange global.

Le **fingerprint** est une empreinte interne utilisée notamment pour identifier des transformations et gérer le cache. C’est utile pour éviter du travail répété ; ce n’est pas une fiche de provenance complète ni une preuve de qualité.

Pour reproduire le résultat, conserve données sources, script, versions, décisions humaines et paramètres aléatoires. Une seed ne remplace pas un corpus versionné.

## 7. Expériences guidées

1. Ajoute une deuxième réponse vide dans la catégorie diagnostic. Prédis les nouvelles proportions après filtre ; compare les comptes par catégorie avant/après.
2. Remplace `has_answer` par un seuil de 40 caractères. Examine toutes les lignes supprimées : le seuil élimine-t-il des réponses pourtant utiles ?
3. Recommence le mélange avec une autre seed. Le contenu doit rester le même ; l’ordre peut changer. Sur un très petit dataset, deux seeds peuvent aussi donner le même ordre.
4. Ajoute une colonne `turns` à ton corpus conversationnel avec `map`, puis inspecte plusieurs indices. Écris la convention : nombre de messages ou nombre d’échanges ?

Conserve un rapport qui indique les règles, les comptes et au moins un exemple affecté par chaque règle.

## 8. Erreurs à savoir reconnaître

| Symptôme | Vérification |
| --- | --- |
| `KeyError: answer` | Inspecter `column_names` et une ligne ; utiliser le schéma réel |
| Une catégorie disparaît | Recompter avant/après chaque filtre |
| Un texte est modifié sans raison apparente | Relire la normalisation appliquée dans `map` |
| Une sortie semble ancienne | Vérifier sources, destination et cache au lieu de changer plusieurs paramètres |
| Les premiers exemples semblent tous semblables | Inspecter plusieurs indices et catégories |

## 9. Exercices avec corrigés

### Exercice 1 — Transformer ou sélectionner ?

Tu veux ajouter le nombre de caractères puis conserver les réponses non vides. Quelles opérations utilises-tu ?

<details>
<summary>Corrigé</summary>

`map` ajoute la longueur ; `filter` conserve les exemples selon une condition. Le second peut réduire le nombre de lignes. Aucun de ces appels n’entraîne un modèle.

</details>

### Exercice 2 — Lire un retrait

Un dataset passe de 100 à 85 lignes. Peut-on conclure que 15 doublons ont été supprimés ?

<details>
<summary>Corrigé</summary>

Non. Il faut connaître la règle et le rapport des retraits. Le même compte pourrait provenir de réponses vides, de longueurs excessives ou d’un filtre erroné.

</details>

### Exercice 3 — Reproduire

Tu conserves seulement `seed=42` et un fichier final. Qu’est-ce qui manque pour expliquer sa fabrication ?

<details>
<summary>Corrigé</summary>

Les sources versionnées, le script, les décisions de relecture, les versions des bibliothèques et les autres paramètres. La seed explique une partie du hasard, pas les modifications du contenu.

</details>

### Exercice 4 — Un champ vide caché

Une ligne contient une question longue mais une réponse vide. Pourquoi filtrer sur la somme des longueurs de tous les messages la conserverait-il ?

<details>
<summary>Corrigé</summary>

La longueur de la question suffit à rendre la somme positive. Il faut contrôler le contenu de la réponse attendue, pas seulement le texte total de la conversation.

</details>

## 10. Validation — avant de continuer

- [ ] Je charge un JSONL et inspecte son schéma réel.
- [ ] Je distingue `map`, `filter`, `select` et `shuffle`.
- [ ] Je retrouve `4 → 4 → 3` et les IDs attendus.
- [ ] Je rapporte les catégories avant/après une transformation.
- [ ] Je sais sauvegarder, recharger et comparer les contenus.
- [ ] Je peux expliquer les limites d’un filtre de longueur et du cache.

## Fiche mémo

```text
Charger → inspecter → mesurer → transformer → recompter → sauvegarder.
Chaque retrait doit correspondre à une règle explicite.
Dataset chargé n’est pas synonyme de split d’évaluation indépendant.
```

Le [module 11](11_train_validation_test_et_contamination.md) construit cette séparation.

## Référence

[Hugging Face Datasets : transformations](https://huggingface.co/docs/datasets/en/process) — opérations de sélection, transformation et sauvegarde.
