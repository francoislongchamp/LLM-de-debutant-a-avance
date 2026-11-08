# Module 11 — Séparer train, validation et test sans tricher involontairement

## 1. Ce que tu vas apprendre

Le modèle doit apprendre sur certains exemples et être évalué sur d’autres. Mélanger leurs rôles peut produire un score impressionnant sans démontrer la compétence recherchée.

Tu vas attribuer un rôle aux données, construire un split groupé, vérifier les recouvrements et expliquer ce qu’une seed ne garantit pas.

**Prérequis :** modules 9 et 10 ; ensembles Python et identifiants stables.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python uniquement pour le laboratoire ; aucun modèle à télécharger.

## 2. Trois usages différents

| Ensemble | À quoi sert-il ? | Influence sur les choix ? |
| --- | --- | --- |
| Train | Calculer les gradients et mettre à jour les paramètres | Directe |
| Validation | Comparer réglages, checkpoints et variantes | Oui, même sans gradient |
| Test | Estimer la performance finale du système choisi | À garder hors des choix de développement |

La validation n’entraîne pas directement les poids, mais elle guide tes décisions. Si tu consultes le test après chaque changement pour choisir la prochaine version, il joue progressivement le rôle d’une validation.

« Le modèle n’a pas fait backward sur ces exemples » n’est donc pas une preuve que le processus de sélection est indépendant de ces exemples.

## 3. Comprendre la contamination

La **fuite d’information**, ou *data leakage*, est un accès à une information qui ne devrait pas être disponible pour la prédiction ou pour son évaluation.

La **contamination** inclut la présence d’exemples d’évaluation, ou de variantes très proches, dans l’apprentissage.

| Situation | Pourquoi le score peut tromper |
| --- | --- |
| Même question/réponse copiée dans train et test | Mémorisation possible |
| Reformulations du même problème dans plusieurs splits | Indépendance affaiblie malgré des textes différents |
| Fragments voisins du même document répartis au hasard | Le contexte et les faits se recouvrent |
| Réponse finale incluse dans un champ d’entrée inaccessible en production | Le modèle reçoit un indice artificiel |

La déduplication exacte du module 9 détecte seulement une partie de ces situations. Un ID différent ne rend pas deux contenus indépendants.

### Définir l’unité de séparation

Si tes exemples sont des paragraphes d’un manuel, le groupe peut être le document ou le chapitre. Pour des dialogues, conserve une conversation entière dans le même split. Pour une généralisation à de nouveaux clients, un groupement par client peut être nécessaire.

Choisis le groupe selon ce que tu veux pouvoir affirmer : nouvelles formulations, nouveaux documents, nouveaux utilisateurs ou dates futures sont des objectifs différents.

## 4. Micro-exemple : groupes et proportions

Dix documents contiennent chacun deux fragments. Tu veux répartir les documents en 6/2/2 :

```text
6 documents train       → 12 fragments
2 documents validation  →  4 fragments
2 documents test        →  4 fragments
```

Le laboratoire utilisera cette répartition pour rendre les comptes visibles. Avec des documents de tailles inégales, 60 % des groupes ne donneront pas forcément 60 % des lignes ou des tokens.

Les proportions 80/10/10 ne sont pas une loi. Un test de trois exemples peut être trop petit même si son pourcentage semble conforme à une convention.

## 5. Laboratoire autonome — Séparation par document

Le script crée vingt lignes, effectue un mélange des groupes, vérifie les intersections et exporte les splits dans un dossier temporaire. Aucun fichier personnel n’est modifié.

```python
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

rows = [
    {"id": f"doc{document:02d}-p{part}",
     "document_id": f"doc{document:02d}",
     "text": f"Document {document}, fragment {part}."}
    for document in range(10)
    for part in range(2)
]


def split_by_document(rows, seed):
    groups = sorted({row["document_id"] for row in rows})
    random.Random(seed).shuffle(groups)
    assignments = {
        "train": set(groups[:6]),
        "validation": set(groups[6:8]),
        "test": set(groups[8:]),
    }
    return {name: [row for row in rows if row["document_id"] in group_ids]
            for name, group_ids in assignments.items()}


splits = split_by_document(rows, seed=42)
assert splits == split_by_document(rows, seed=42)

names = list(splits)
for index, left in enumerate(names):
    for right in names[index + 1:]:
        left_groups = {row["document_id"] for row in splits[left]}
        right_groups = {row["document_id"] for row in splits[right]}
        left_ids = {row["id"] for row in splits[left]}
        right_ids = {row["id"] for row in splits[right]}
        assert left_groups.isdisjoint(right_groups)
        assert left_ids.isdisjoint(right_ids)

all_ids = [row["id"] for split in splits.values() for row in split]
assert len(all_ids) == len(set(all_ids)) == len(rows)

with TemporaryDirectory(prefix="module11-") as directory:
    for name, split in splits.items():
        path = Path(directory) / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in split:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(name, "lignes=", len(split),
              "documents=", sorted({row["document_id"] for row in split}))

print("Aucun ID ni document partagé ; toutes les lignes sont affectées.")
```

Tu dois obtenir **12/4/4 lignes**, sans groupe ni ID partagé. La seed contrôle l’affectation des groupes pour cette entrée ordonnée ; elle ne corrige pas des groupes mal définis.

Les textes sont ici artificiels et distincts par construction. Pour ton corpus, ajoute une recherche de contenu identique entre splits : deux sources différentes peuvent contenir le même passage.

## 6. Audit avant d’entraîner

Pour chaque split, compte les lignes, groupes, catégories et tokens. Lis plusieurs exemples et vérifie :

1. Les IDs sont uniques et tous les exemples sont affectés.
2. Les groupes choisis ne traversent pas les splits.
3. Les copies exactes et variantes connues ne créent pas de fuite.
4. Les catégories utiles sont suffisamment représentées pour être mesurées.
5. Le schéma et le formatage sont compatibles entre train et évaluation.

Un équilibre statistique parfait ne garantit pas l’indépendance. À l’inverse, un split temporel réaliste peut avoir des distributions différentes : c’est parfois justement le changement que tu veux tester.

### Séparer avant de dériver des variantes

Affecte les sources ou problèmes aux splits, puis garde leurs augmentations dans le même split. Fabriquer dix paraphrases puis les répartir au hasard peut faire passer une même information dans les trois ensembles.

Ne choisis pas les données d’entraînement en inspectant les réponses du test final. Les transformations décidées à partir des données doivent elles aussi respecter la frontière d’évaluation.

## 7. Adapter à ton projet

Reprends les 30 exemples du module 9. Ajoute un `group_id` qui représente la source ou le problème, puis écris la raison de ce choix.

Avec un corpus aussi petit, une estimation finale restera fragile. Tu peux apprendre à construire les fichiers sans prétendre mesurer précisément une amélioration d’un point.

Pour les modules suivants, un dossier persistant peut contenir :

```text
data/splits_v1/
    train.jsonl
    validation.jsonl
    test.jsonl
    split_manifest.json
```

Le manifeste garde la seed, la version des sources, les IDs/groupes par split et la règle de séparation. Cette liste reste utile si l’ordre initial change plus tard.

## 8. Expériences guidées

1. Répartis d’abord les vingt fragments ligne par ligne avec une seed. Compte les documents qui traversent les splits ; compare au split groupé.
2. Donne dix fragments à un document et un seul aux autres. Refais le split groupé : observe pourquoi les proportions en lignes changent.
3. Ajoute une copie textuelle sous un nouvel ID et un autre document. Les assertions du laboratoire peuvent encore passer ; écris le contrôle de contenu qui manque.
4. Imagine des procédures mises à jour dans le temps. Décris une séparation qui évalue des documents futurs plutôt que des fragments aléatoires du passé.

## 9. Exercices avec corrigés

### Exercice 1 — Reconnaître le rôle d’un ensemble

Tu choisis le LR donnant la meilleure loss sur un ensemble, sans backward dessus. Est-ce un test final indépendant ?

<details>
<summary>Corrigé</summary>

Non : cet ensemble sert de validation, puisque ses résultats guident la sélection du LR. L’absence de gradient ne suffit pas à en faire un test final.

</details>

### Exercice 2 — Choisir un groupe

Cent conversations contiennent chacune dix échanges. Tu veux évaluer sur de nouvelles conversations. Quelle unité gardes-tu entière ?

<details>
<summary>Corrigé</summary>

La conversation, avec un identifiant stable. Des échanges d’une même conversation peuvent partager contexte, faits et formulations ; les séparer par ligne ne teste pas la même généralisation.

</details>

### Exercice 3 — Interpréter 90 %

Un modèle réussit 9 questions sur 10. Un autre réussit 900 sur 1 000. Pourquoi faut-il rapporter les effectifs malgré le même pourcentage ?

<details>
<summary>Corrigé</summary>

La sensibilité à un exemple et l’incertitude d’échantillonnage diffèrent. Une seule erreur supplémentaire fait passer le premier score à 80 %, contre 89.9 % pour le second. Les conditions et la diversité des questions comptent aussi.

</details>

### Exercice 4 — Limite des assertions

Aucun ID ni document ne traverse les splits. Peut-il encore exister de la contamination ?

<details>
<summary>Corrigé</summary>

Oui : copies sous d’autres IDs, mêmes problèmes reformulés, sources qui se recouvrent ou données présentes dans un entraînement antérieur. Les assertions vérifient seulement les conditions qu’elles expriment.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue les trois rôles et l’effet indirect de la validation.
- [ ] Je justifie le groupe selon la généralisation recherchée.
- [ ] Je retrouve 12/4/4 sans recouvrement dans le laboratoire.
- [ ] Je peux montrer une contamination que ces assertions ne détectent pas.
- [ ] Je conserve un manifeste reproductible et les versions des sources.
- [ ] Je sais expliquer les limites d’un tout petit test.

## Fiche mémo

```text
Train apprend ; validation guide les choix ; test estime le résultat final.
Définir l’unité indépendante avant de répartir les lignes.
Seed et absence d’IDs communs ne prouvent pas l’absence de contamination.
```

Le [module 12](12_premier_supervised_fine_tuning.md) utilise ces distinctions pour préparer un premier entraînement supervisé.

## Référence

[Hugging Face Datasets : séparation et sélection](https://huggingface.co/docs/datasets/en/process) — outils de partitionnement ; le choix des groupes reste une décision de protocole.
