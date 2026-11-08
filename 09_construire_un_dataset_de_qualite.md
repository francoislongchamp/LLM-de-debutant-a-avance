# Module 9 — Construire et vérifier un dataset de qualité

## 1. Ce que tu vas apprendre

Au [module 8](08_premiere_boucle_dentrainement_pytorch.md), tu as fait apprendre `y = 2x` à un modèle. Les cibles étaient faciles à vérifier : pour `x = 3`, la bonne réponse était 6.

Pour entraîner un assistant, une cible peut être une explication entière. Comment savoir si cette explication est correcte, utile et représentative des situations à traiter ? Ce choix fait partie du travail d’entraînement.

Dans ce module, tu vas construire un petit dataset pour un **assistant qui explique les bases de l’entraînement PyTorch à un débutant francophone**. Ce sujet te permet de réutiliser ce que tu viens d’apprendre.

À la fin, tu devrais pouvoir :

- définir le comportement que tes exemples doivent enseigner ;
- lire et écrire un exemple conversationnel dans un fichier JSONL ;
- distinguer erreur de structure et erreur de contenu ;
- repérer doublons, réponses potentiellement contradictoires et catégories sous-représentées ;
- produire un dataset vérifié accompagné d’un rapport et d’une fiche descriptive ;
- justifier pourquoi tu conserves, corriges ou rejettes un exemple.

**Prérequis :** modules 6 à 8 ; en Python, listes, dictionnaires, boucles et fonctions. Les parties du code liées aux fichiers sont expliquées ici.

**Temps indicatif :** deux séances de 60 à 90 minutes, laboratoire compris.

**Matériel :** Python uniquement. Aucun GPU, aucune bibliothèque à installer, aucun entraînement dans ce module.

Commence par les exemples à lire. Avant de lancer le laboratoire, essaie de prévoir ce que chaque contrôle pourra détecter.

## 2. Un dataset SFT, concrètement

Un **dataset** est une collection d’exemples. Le **SFT**, pour *Supervised Fine-Tuning*, consiste à poursuivre l’entraînement d’un modèle à partir d’exemples de comportement attendu.

Dans notre cas, un exemple contient une question et une réponse de référence :

```text
Utilisateur : Est-ce que backward() modifie les poids ?
Assistant   : Non. backward() calcule les gradients. L’optimizer
              utilise ensuite ces gradients pour modifier les paramètres
              lorsque tu appelles step().
```

La réponse de l’assistant est une **cible d’apprentissage**, pas une réponse que le modèle aurait forcément produite avant l’entraînement.

| Au module 8 | Dans notre dataset conversationnel |
| --- | --- |
| Entrée numérique `x` | Question de l’utilisateur et contexte éventuel |
| Cible numérique `y` | Réponse attendue de l’assistant |
| Couple `(x, y)` | Conversation de référence |
| Petite liste de couples | Collection de conversations |

Le modèle traite ensuite ces textes sous forme de tokens. Le formatage de la conversation et le choix des tokens qui contribuent à la loss seront précisés lors du SFT. Stocker une conversation ne définit pas, à lui seul, le calcul de la loss.

### Un exemple enseigne plus que son contenu factuel

Compare ces deux réponses à la même question :

| Réponse | Signal donné par l’exemple |
| --- | --- |
| « Non, évidemment. Relis la documentation. » | Réponse courte, ton méprisant, aucune explication |
| « Non. `backward()` calcule les gradients ; `step()` modifie les paramètres. » | Réponse directe, vocabulaire précis, distinction utile |

Les deux commencent par une réponse factuellement correcte. Une seule correspond au professeur patient que nous voulons entraîner.

Un exemple peut donc enseigner simultanément un fait, un ton, une longueur, une langue et une façon de raisonner ou de présenter la réponse. Un entraînement peut renforcer ces régularités sans les reproduire parfaitement dans tous les cas.

## 3. Définir le comportement avant de collecter

« Un assistant utile » est trop vague pour décider si un exemple est bon. Écrivons un objectif vérifiable :

> Pour une question sur les bases de l’entraînement PyTorch, répondre en français, directement, en deux à quatre phrases courtes. Expliquer le mécanisme au niveau débutant. Quand un diagnostic manque d’informations, demander une précision utile sans inventer la cause.

Voici notre grille de relecture :

| Critère | Question à poser à chaque exemple |
| --- | --- |
| Exactitude | Chaque affirmation technique est-elle correcte dans le contexte ? |
| Pertinence | La réponse traite-t-elle la question posée ? |
| Clarté | Un débutant peut-il comprendre les termes utilisés ? |
| Format et style | Le français, le ton et la longueur suivent-ils l’objectif ? |
| Gestion de l’incertitude | La réponse évite-t-elle d’affirmer une cause sans preuve ? |

Une contrainte de longueur aide à relire ; elle ne remplace pas l’exactitude. Ne supprime pas une précision nécessaire uniquement pour respecter un nombre de phrases.

> **À toi.** « Ma loss augmente : baisse toujours le learning rate » est-elle une bonne cible ?
>
> Non. Elle transforme une hypothèse possible en règle universelle. Une meilleure réponse demande la courbe, les réglages ou le contexte, puis propose une vérification.

## 4. Définir un schéma lisible

Un **schéma** décrit les champs attendus, leurs types et les règles qu’ils doivent respecter.

Voici un exemple de notre schéma pédagogique :

```json
{
  "id": "ex01",
  "category": "definition",
  "source": "redaction_pedagogique_module_09",
  "messages": [
    {"role": "user", "content": "Qu’est-ce qu’un batch ?"},
    {"role": "assistant", "content": "Un batch est un groupe d’exemples traités ensemble. Avec huit exemples et des batches de deux, un passage complet contient quatre batches."}
  ]
}
```

| Champ | Type | Utilité |
| --- | --- | --- |
| `id` | Chaîne non vide et unique | Retrouver un exemple et suivre les décisions |
| `category` | Chaîne parmi nos catégories autorisées | Compter les types de situations |
| `source` | Chaîne non vide | Retrouver l’origine déclarée du contenu |
| `messages` | Liste de deux dictionnaires | Contenir la question et sa réponse |
| `role` | `user`, puis `assistant` | Identifier qui parle |
| `content` | Chaîne non vide | Contenir le texte du message |

Nous limitons volontairement chaque conversation à **un échange utilisateur/assistant**. D’autres datasets peuvent contenir un message système, plusieurs échanges ou des outils. Notre validateur applique notre schéma ; il ne définit pas tous les formats possibles pour un assistant.

### Métadonnées et texte d’apprentissage

`id`, `category` et `source` sont des **métadonnées** : des informations qui décrivent l’exemple. Elles restent hors de `messages`.

Elles ne sont pas automatiquement du contexte pour le modèle. C’est la préparation des entrées qui décide quels champs lui sont transmis. Plus tard, vérifie que ton pipeline utilise les messages prévus sans concaténer accidentellement ces métadonnées à la question.

`source` est une trace d’origine déclarée, pas une preuve d’exactitude ni une autorisation de réutilisation. Pour des données externes, conserve aussi la référence précise et les conditions d’utilisation connues.

## 5. JSON et JSONL : comprendre le fichier que tu manipules

**JSON** est un format de données. Un objet utilise des accolades `{}`, une liste des crochets `[]`, et les noms de champs et chaînes utilisent des guillemets doubles.

L’objet de la section précédente est affiché sur plusieurs lignes pour faciliter sa lecture.

**JSONL**, pour *JSON Lines*, contient une valeur JSON complète par ligne. Dans notre projet, chaque ligne sera un objet représentant un exemple :

```jsonl
{"id":"demo01","category":"definition","source":"exemple_du_cours","messages":[{"role":"user","content":"Qu’est-ce qu’un batch ?"},{"role":"assistant","content":"Un batch est un groupe d’exemples traités ensemble."}]}
{"id":"demo02","category":"comparaison","source":"exemple_du_cours","messages":[{"role":"user","content":"Epoch et step, est-ce pareil ?"},{"role":"assistant","content":"Non. Une epoch parcourt les données d’entraînement ; un step met à jour les paramètres."}]}
```

Il n’y a ni liste englobante ni virgule entre ces deux lignes. Un retour à la ligne à l’intérieur d’un contenu est encodé dans la chaîne sous la forme `\n` ; il ne doit pas couper l’objet en plusieurs lignes physiques.

Dans le laboratoire :

- `json.dumps(row, ensure_ascii=False)` transforme un dictionnaire Python en texte JSON et conserve les accents lisibles ;
- `json.loads(line)` reconstruit une valeur Python à partir d’une ligne JSON ;
- `encoding="utf-8"` fixe l’encodage de lecture et d’écriture.

Le fait que `json.loads()` réussisse garantit seulement une syntaxe JSON valide. La valeur peut encore avoir le mauvais type, manquer de champs ou contenir une réponse fausse.

## 6. Les problèmes à distinguer

### 6.1 Structure incorrecte

Exemples : un champ `messages` absent, un rôle `bot` au lieu de `assistant`, une réponse vide ou une valeur numérique à la place d’un texte.

Ces problèmes sont détectables par des règles de code explicites.

### 6.2 Contenu incorrect

« `backward()` modifie directement les poids » est une réponse fausse dans la boucle du module 8. Pourtant, elle peut être stockée dans un objet JSON parfaitement valide.

Il faut relire le sens et vérifier les affirmations. Notre script ne comprend pas la vérité d’une explication PyTorch.

### 6.3 Doublon

Deux exemples peuvent avoir des identifiants différents et exactement les mêmes messages. Pour l’apprentissage, le même contenu est alors présenté plusieurs fois.

Si un dataset contient 10 exemples distincts, puis que tu ajoutes 90 copies d’un seul de ces exemples, ce contenu occupe **91 lignes sur 100**, soit 91 % des lignes.

Avec un échantillonnage uniforme des lignes, il sera donc présenté beaucoup plus souvent. Son poids exact dans la loss dépend aussi des longueurs, des masques et de la méthode d’agrégation ; 91 % des lignes ne signifie pas nécessairement 91 % du gradient.

### 6.4 Réponses différentes et contradiction

Deux réponses différentes à la même question ne sont pas forcément contradictoires : elles peuvent être deux explications correctes.

| Question et réponses | Décision |
| --- | --- |
| Même question, même réponse exacte | Doublon de contenu |
| Même question, deux reformulations correctes | Variantes à examiner |
| « backward modifie les poids » et « backward calcule les gradients sans les modifier » | Contradiction factuelle dans notre contexte |
| Réponses différentes à des questions qui précisent des contextes différents | Peut être cohérent |

Un script peut **signaler** plusieurs réponses pour une même question. Il faut ensuite décider si elles sont compatibles, si une réponse est fausse ou si le contexte manque.

### 6.5 Couverture et distribution

La **couverture** décrit les situations présentes : définitions, comparaisons, diagnostics, questions ambiguës, erreurs fréquentes…

La **distribution** décrit leurs proportions. Pour une catégorie `c` :

\[
p(c)=\frac{\text{nombre d’exemples de catégorie }c}{\text{nombre total d’exemples}}
\]

Si 24 exemples sur 30 sont des définitions, la proportion est `24 / 30 = 80 %`. Si seulement 3 sont des diagnostics, leur proportion est `3 / 30 = 10 %`.

Un dataset peut citer toutes les catégories et laisser très peu de place à certaines. Une distribution équilibrée n’est pas toujours idéale : elle doit correspondre à ton objectif et aux situations que tu veux évaluer.

## 7. Laboratoire autonome : voir les défauts avant de nettoyer

### 7.1 Le résultat à prévoir

Le script crée un jeu pédagogique de **10 lignes** :

- 6 exemples de référence, avec 2 exemples par catégorie ;
- 1 copie exacte des messages de `ex01`, avec un autre ID ;
- 1 réponse fausse à la question de `ex03` ;
- 1 réponse vide ;
- 1 rôle invalide.

Prédis les comptes : combien de lignes passeront le contrôle de structure ? Combien resteront après déduplication ? Le contrôle de structure rejettera-t-il la réponse fausse ?

### 7.2 Exécuter le script

Copie le bloc complet dans `09_dataset.py`, dans ton dossier de travail, puis lance `python 09_dataset.py` avec ton environnement Python.

Le dossier `data/module09/` sera créé **relativement au dossier depuis lequel tu lances la commande**. Le fichier brut est créé seulement s’il n’existe pas déjà, pour conserver tes modifications. Les sorties `dataset.jsonl` et `rapport.json` sont recalculées à chaque exécution ; ne les modifie pas pour corriger les données brutes.

```python
import json
from collections import Counter, defaultdict
from pathlib import Path

# Catégories choisies pour notre objectif pédagogique.
CATEGORIES = {"definition", "comparaison", "diagnostic"}
root = Path("data/module09")
root.mkdir(parents=True, exist_ok=True)
raw_path = root / "brut.jsonl"


def make_row(row_id, category, question, answer):
    return {
        "id": row_id,
        "category": category,
        "source": "redaction_pedagogique_module_09",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


# Création du jeu initial : aucune génération ni aucun téléchargement externe.
if not raw_path.exists():
    initial = [
        make_row("ex01", "definition", "Qu’est-ce qu’un batch ?",
                 "Un batch est un groupe d’exemples traités ensemble. "
                 "Huit exemples avec des batches de deux donnent quatre batches."),
        make_row("ex02", "definition", "Qu’est-ce que le learning rate ?",
                 "Le learning rate règle l’amplitude des mises à jour. "
                 "Avec SGD sans momentum, on soustrait le gradient multiplié par ce nombre."),
        make_row("ex03", "comparaison", "backward() et step(), est-ce pareil ?",
                 "Non. backward() calcule les gradients sans modifier les paramètres. "
                 "L’optimizer les modifie lorsque tu appelles step()."),
        make_row("ex04", "comparaison", "eval() et no_grad(), est-ce pareil ?",
                 "Non. eval() change le comportement de certaines couches. "
                 "no_grad() désactive l’enregistrement du graphe autograd dans le bloc."),
        make_row("ex05", "diagnostic", "Ma loss augmente : quelle est la cause ?",
                 "Une hausse de loss ne suffit pas à identifier la cause. "
                 "Peux-tu montrer la courbe et préciser le learning rate ainsi que les données utilisées ?"),
        make_row("ex06", "diagnostic", "Les poids ne changent pas : que vérifier ?",
                 "Vérifie qu’un backward calcule des gradients puis que step() est appelé. "
                 "Affiche un poids avant et après le step pour confirmer la mise à jour."),
    ]
    # Copie indépendante ; seules les métadonnées d'identification changent.
    duplicate = json.loads(json.dumps(initial[0]))
    duplicate["id"] = "ex07"
    initial.append(duplicate)
    initial.append(make_row(
        "ex08", "comparaison", "backward() et step(), est-ce pareil ?",
        "Oui. backward() modifie directement les poids. step() affiche seulement la loss.",
    ))
    initial.append(make_row("ex09", "definition", "Qu’est-ce qu’une epoch ?", "   "))
    bad_role = make_row("ex10", "definition", "Qu’est-ce qu’un gradient ?",
                        "Un gradient mesure la sensibilité locale de la loss aux paramètres.")
    bad_role["messages"][1]["role"] = "bot"
    initial.append(bad_role)
    write_jsonl(raw_path, initial)


def validate(row):
    """Appliquer notre schéma simple, sans juger la vérité des réponses."""
    if not isinstance(row, dict):
        return "la ligne doit contenir un objet"
    for field in ("id", "category", "source"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            return f"champ {field} absent, vide ou non textuel"
    if row["category"] not in CATEGORIES:
        return "catégorie inconnue"
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        return "messages doit contenir exactement deux messages"
    for message, expected_role in zip(messages, ("user", "assistant")):
        if not isinstance(message, dict):
            return "chaque message doit être un objet"
        if message.get("role") != expected_role:
            return f"rôle attendu : {expected_role}"
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return "contenu absent, vide ou non textuel"
    return None


valid = []
rejections = []
seen_ids = set()
line_count = 0
with raw_path.open(encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        line_count += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            rejections.append({"line": line_number, "reason": f"JSON invalide : {error.msg}"})
            continue
        reason = validate(row)
        if reason is None:
            if row["id"] in seen_ids:
                reason = "ID déjà utilisé par une ligne valide"
            else:
                seen_ids.add(row["id"])
        if reason is not None:
            rejections.append({"line": line_number, "reason": reason})
        else:
            valid.append(row)

# Déduplication exacte des messages, indépendamment de l'ID et de la catégorie.
# Conserver le premier exemplaire est une règle simple propre à ce laboratoire.
unique = []
seen_messages = {}
duplicates = []
for row in valid:
    key = tuple((message["role"], message["content"]) for message in row["messages"])
    if key in seen_messages:
        duplicates.append({"removed": row["id"], "kept": seen_messages[key]})
    else:
        seen_messages[key] = row["id"]
        unique.append(row)

# Signalement seulement : réponses différentes à une question exactement identique.
by_question = defaultdict(list)
for row in unique:
    by_question[row["messages"][0]["content"]].append(row)
review_groups = []
for question, group in by_question.items():
    answers = {row["messages"][1]["content"] for row in group}
    if len(answers) > 1:
        review_groups.append({"question": question, "ids": [row["id"] for row in group]})

print("Questions à relire :", review_groups)

# Décision HUMAINE fournie pour le jeu initial, pas détection automatique de vérité.
# Vérification : au module 8, backward calcule les gradients ; step met à jour les poids.
# On associe aussi la décision au texte pour ne pas rejeter un ex08 corrigé ensuite.
manual_rejections = {
    "ex08": {
        "answer": "Oui. backward() modifie directement les poids. step() affiche seulement la loss.",
        "reason": "Réponse factuellement fausse ; voir la correction observable du module 8.",
    }
}
clean = []
manual_log = []
for row in unique:
    decision = manual_rejections.get(row["id"])
    if decision and row["messages"][1]["content"] == decision["answer"]:
        manual_log.append({"id": row["id"], "reason": decision["reason"]})
    else:
        clean.append(row)

write_jsonl(root / "dataset.jsonl", clean)
counts = Counter(row["category"] for row in clean)
# Longueur de la réponse seulement, mesurée en caractères Python, pas en tokens.
lengths = [len(row["messages"][1]["content"]) for row in clean]
report = {
    "raw_lines": line_count,
    "schema_valid": len(valid),
    "structural_rejections": rejections,
    "duplicates": duplicates,
    "after_deduplication": len(unique),
    "questions_to_review_before_manual_decisions": review_groups,
    "manual_rejections": manual_log,
    "final_examples": len(clean),
    "categories": dict(sorted(counts.items())),
    "answer_chars_min": min(lengths) if lengths else None,
    "answer_chars_max": max(lengths) if lengths else None,
}
(root / "rapport.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Fichiers produits dans :", root.resolve())
```

### 7.3 Lire le résultat, étape par étape

Avec le jeu initial, tu dois retrouver :

```text
10 lignes brutes
 ├─ 2 rejets de structure : ex09 (réponse vide), ex10 (rôle bot)
 ↓
 8 exemples valides selon le schéma
 ├─ 1 doublon de messages : ex07, copie de ex01
 ↓
 7 exemples après déduplication
 ├─ 1 question à relire : réponses ex03 et ex08
 ├─ 1 rejet décidé par relecture : ex08, réponse fausse
 ↓
 6 exemples conservés
    definition : 2 ; comparaison : 2 ; diagnostic : 2
```

Chaque catégorie représente `2 / 6`, soit environ **33,3 %** du résultat final.

Observe surtout **ex08** : il passe le validateur. Le programme ne découvre pas qu’il est faux ; nous lui avons fourni une décision humaine documentée pour ce cas précis.

Le rapport conserve le signalement du conflit avant cette décision, afin de garder la trace de ce qui a été examiné. Si tu ajoutes d’autres réponses problématiques, leur examen restera à faire avant de considérer le fichier final comme prêt.

### 7.4 Comprendre le code sans tout mémoriser

| Partie | Pourquoi elle existe |
| --- | --- |
| `Path` et `mkdir` | Construire les chemins et créer le dossier de sortie |
| `make_row` | Fabriquer les exemples avec les mêmes champs |
| `validate` | Retourner une raison de rejet, ou `None` si la structure convient |
| `enumerate(..., start=1)` | Retrouver le numéro de ligne dans le fichier brut |
| `try / except` | Signaler une ligne JSON malformée et poursuivre l’inspection |
| `seen_ids` | Empêcher deux lignes structurellement valides de partager un ID |
| Clé construite à partir des messages | Comparer le contenu en ignorant les métadonnées |
| `defaultdict(list)` | Regrouper les exemples qui ont exactement la même question |
| `Counter` | Compter les exemples par catégorie |
| `manual_log` | Garder la raison des décisions humaines appliquées |

Dans la déduplication, le programme garde le premier contenu rencontré. Pour un vrai corpus, les copies peuvent avoir des sources ou catégories différentes : réconcilie leurs métadonnées au lieu de perdre leur provenance sans examen.

### 7.5 Ce que ces contrôles ne garantissent pas

- Deux paraphrases peuvent être des quasi-doublons sans avoir les mêmes caractères.
- Un espace ajouté suffit à faire échapper un doublon à notre comparaison exacte.
- Une contradiction entre deux questions reformulées n’est pas signalée par ce regroupement.
- Une catégorie déclarée peut être mal choisie : le code ne juge pas son sens.
- Une réponse fausse mais unique peut ne déclencher aucun signalement.
- Une longueur en caractères ne dit pas combien de tokens le modèle utilisera.

Pour ce petit dataset, relis **toutes les réponses**. Les contrôles automatiques aident à organiser la relecture ; ils ne valident pas le contenu à ta place.

## 8. Nettoyer sans effacer l’histoire

Un pipeline de nettoyage doit rendre les décisions compréhensibles :

```text
Données brutes conservées
        ↓
Contrôles de structure et rapport des rejets
        ↓
Recherche des doublons et des cas à examiner
        ↓
Relecture : conserver, corriger ou rejeter avec une raison
        ↓
Données vérifiées + rapport + fiche descriptive
```

Si tu corriges une réponse, conserve sa version initiale dans les données brutes ou dans un historique de versions. Si tu rejettes un exemple, note pourquoi. « Supprimé pour améliorer le dataset » n’explique pas la décision ; « réponse vide » ou « confond backward et mise à jour des poids » l’explique.

Le fichier `dataset.jsonl` du laboratoire est un **corpus vérifié pour l’exercice**, encore non séparé. Il n’est pas encore un ensemble d’entraînement accompagné d’une évaluation indépendante. Le [module 11](11_train_validation_test_et_contamination.md) expliquera les séparations train/validation/test et les risques de contamination.

Ne fabrique pas des reformulations d’un même exemple pour les répartir ensuite au hasard entre train et test : leurs liens devront être pris en compte lors de cette séparation.

## 9. Écrire la fiche du dataset

Crée un fichier `data/module09/DATASET_CARD.md`. Commence avec ce modèle, puis adapte-le à tes données réelles :

```markdown
# Dataset — Explications des bases de l’entraînement PyTorch

- Version : v1
- Objectif : répondre en français à un débutant sur la boucle d’entraînement.
- Usage prévu : apprendre à préparer un petit dataset SFT pédagogique.
- Provenance : exemples fournis dans le module 9, puis relus localement.
- Conditions de réutilisation : vérifier celles du cours avant redistribution.
- Schéma : id, category, source, messages (un échange user/assistant).
- Catégories : definition, comparaison, diagnostic.
- Nombre brut : 10.
- Nombre final : 6.
- Nettoyage : 2 rejets de structure, 1 doublon exact, 1 réponse fausse rejetée.
- Vérification du contenu : comparaison avec le module 8 pour les faits PyTorch.
- Rapport : rapport.json ; données sources : brut.jsonl.
- Séparation train/validation/test : non réalisée à ce stade.
- Limites : très petit corpus, français uniquement, questions courtes,
  couverture technique réduite, absence d’évaluation du modèle entraîné.
```

Pour reproduire une version, conserve ensemble le brut, le script, les décisions de relecture, la fiche et les sorties. Si tes données proviennent de plusieurs sources, documente chaque origine ; une seule étiquette générale ne suffit plus.

## 10. Expériences guidées : comprendre en modifiant les données

Le script préserve `brut.jsonl` lors des exécutions suivantes. Tu peux donc modifier ce fichier, relancer le script et comparer les rapports. Fais une copie du jeu initial avant de commencer, puis repars de cette copie pour chaque expérience.

### Expérience A — Un doublon avec un autre ID

Copie une ligne correcte dans `brut.jsonl`, change seulement son `id`, puis relance.

**Prédis :** le contrôle d’ID l’acceptera-t-il ? La déduplication la gardera-t-elle ?

L’ID est nouveau, donc cette règle l’accepte. Les messages sont identiques, donc la déduplication retire la copie. Un ID unique ne garantit pas un contenu unique.

### Expérience B — Un quasi-doublon

Ajoute une copie avec un nouvel ID, puis reformule légèrement la question et la réponse.

**Prédis :** notre script reconnaîtra-t-il la proximité de sens ?

Non. La comparaison porte sur les caractères exacts. Note manuellement le lien entre ces exemples ; une reformulation ne crée pas forcément une nouvelle compétence à apprendre.

### Expérience C — Une réponse fausse qui passe tous les contrôles

Ajoute une question nouvelle et une réponse non vide mais techniquement fausse, avec un ID unique et une catégorie autorisée.

**Prédis :** sera-t-elle automatiquement rejetée ?

Non. Elle peut arriver dans `dataset.jsonl`. Relis-la, explique l’erreur, puis ajoute une décision documentée ou prépare une correction traçable. N’oublie pas : la liste des décisions du script ne couvre initialement que `ex08`.

### Expérience D — Une erreur de syntaxe JSON

Dans une copie du fichier brut, retire un guillemet d’une ligne, puis utilise cette copie comme `raw_path` dans une copie du script.

**Prédis :** est-ce le contrôle du schéma ou le parseur JSON qui signalera le problème ?

`json.loads()` échouera d’abord. Le rapport indiquera le numéro de ligne et une erreur JSON ; le schéma ne peut pas être inspecté tant que la ligne ne peut pas être lue.

## 11. Projet personnel : passer de 6 à 30 exemples utiles

Construis maintenant **30 exemples finaux**, relus et distincts, pour le même objectif ou pour un domaine étroit que tu maîtrises.

1. Écris ton objectif et ta grille de qualité avant les réponses.
2. Prépare un plan de couverture : par exemple 10 définitions, 10 comparaisons et 10 diagnostics. Justifie les proportions si tu choisis un autre équilibre.
3. Rédige des questions variées : formulation simple, confusion courante, demande de diagnostic avec informations insuffisantes.
4. Écris les réponses de référence et relis chaque affirmation technique.
5. Utilise un fichier brut propre à ce projet. Dans une copie du script, adapte `raw_path`, les catégories et les décisions humaines ; ne conserve pas une règle de rejet `ex08` qui ne correspond plus à tes données.
6. Exécute les contrôles, examine les signalements et documente tes corrections.
7. Recompte les catégories **après** nettoyage et complète la fiche du dataset.

N’allonge pas artificiellement le corpus avec des copies. Trente exemples sont un bon exercice de préparation ; ils ne garantissent pas une spécialisation robuste après entraînement.

Ton livrable comprend le brut, le script, le corpus final, le rapport et `DATASET_CARD.md`. Une autre personne doit pouvoir comprendre comment tu es passé du premier au dernier.

## 12. Exercices avec corrigés

Réponds avant d’ouvrir le corrigé. Cherche à expliquer ta décision, pas seulement à nommer le problème.

### Exercice 1 — Qualité ou format ?

Une ligne JSON contient une question pertinente, une réponse non vide et des rôles corrects. La réponse affirme que `eval()` désactive autograd. Que conclure ?

<details>
<summary>Corrigé</summary>

La structure peut être valide, mais le contenu est faux. `eval()` change le comportement de certaines couches ; `no_grad()` désactive l’enregistrement autograd dans son bloc. Il faut corriger ou rejeter la réponse avec une raison documentée.

</details>

### Exercice 2 — Mesurer une distribution

Un corpus contient 30 exemples : 24 définitions, 3 comparaisons et 3 diagnostics. Quelles sont les proportions ? Toutes les catégories sont présentes : cela suffit-il à démontrer une bonne couverture ?

<details>
<summary>Corrigé</summary>

Définitions : **80 %** ; comparaisons : **10 %** ; diagnostics : **10 %**.

La présence des trois catégories ne suffit pas. Il faut examiner les situations couvertes à l’intérieur de chaque catégorie et vérifier que les proportions servent l’objectif. Trois diagnostics presque identiques couvrent peu de difficultés différentes.

</details>

### Exercice 3 — Compter les retraits sans compter deux fois

Sur 100 lignes, tu rejettes d’abord 5 lignes invalides. Parmi les 95 restantes, tu retires 8 copies exactes. Puis la relecture rejette 2 autres exemples. Combien reste-t-il d’exemples ? Pourquoi l’ordre des comptes est-il utile ?

<details>
<summary>Corrigé</summary>

`100 − 5 − 8 − 2 = 85` exemples. Chaque nombre désigne ici un retrait parmi les exemples encore présents à cette étape : les retraits sont distincts.

Si plusieurs contrôles portent indépendamment sur le brut, un même exemple peut avoir plusieurs défauts. Additionner leurs compteurs sans tenir compte des recouvrements donnerait alors un total trompeur.

</details>

### Exercice 4 — Différence ou contradiction ?

À « Qu’est-ce qu’un batch ? », une réponse dit « un groupe d’exemples traités ensemble » et une autre « un ensemble d’exemples utilisé dans une même passe du modèle ». Faut-il supprimer automatiquement la seconde ?

<details>
<summary>Corrigé</summary>

Non. Les formulations sont différentes mais compatibles dans notre contexte. Examine leur intérêt pédagogique et leur redondance avant de décider. Un signalement de plusieurs réponses n’établit pas une contradiction.

</details>

### Exercice 5 — Métadonnées et modèle

Tu ajoutes `category: diagnostic` hors de `messages`. Le modèle connaît-il automatiquement cette catégorie quand il reçoit la question ?

<details>
<summary>Corrigé</summary>

Non. Elle sert ici à analyser et sélectionner les exemples. Le modèle ne reçoit que ce que le pipeline transforme en entrées. Si la catégorie doit être un indice pour répondre, il faut concevoir explicitement ce contexte et pouvoir le fournir aussi lors de l’utilisation du modèle.

</details>

### Exercice 6 — Décider d’inclure un exemple

Propose trois questions qui enrichissent notre dataset de six exemples. Pour chacune, indique la catégorie, la compétence visée et un critère pour juger la réponse.

<details>
<summary>Exemple de corrigé</summary>

| Question | Catégorie | Compétence et critère |
| --- | --- | --- |
| « Une epoch contient-elle toujours un seul step ? » | comparaison | Distinguer passage sur les données et mise à jour, avec un exemple chiffré correct |
| « Pourquoi remettre les gradients à zéro ? » | definition | Expliquer l’accumulation sans prétendre que les poids sont effacés |
| « Ma validation est meilleure que mon train : est-ce forcément un bug ? » | diagnostic | Éviter un diagnostic certain et examiner les conditions de mesure |

Ce sont des propositions de couverture. Il reste à rédiger et vérifier les réponses, puis à examiner leur éventuelle redondance avec les exemples existants.

</details>

## 13. Validation — avant de continuer

Tu es prêt pour la suite lorsque tu peux :

- [ ] Décrire le comportement que ton dataset doit enseigner.
- [ ] Expliquer ce qu’une réponse enseigne au-delà des faits.
- [ ] Lire un exemple et distinguer messages et métadonnées.
- [ ] Expliquer la différence entre JSON valide, schéma valide et contenu correct.
- [ ] Retrouver les comptes `10 → 8 → 7 → 6` du laboratoire.
- [ ] Expliquer pourquoi `ex08` passe le validateur et comment son rejet est décidé.
- [ ] Calculer les proportions des catégories après nettoyage.
- [ ] Donner les limites d’une déduplication exacte.
- [ ] Produire un petit corpus relu et une fiche qui documente ses limites.

Si un point bloque, reviens à un exemple concret, indique ce que tu attendais et compare avec le rapport. L’objectif est de pouvoir défendre tes choix de données.

## Fiche mémo

```text
Définir le comportement cible
        ↓
Choisir un schéma et des catégories utiles
        ↓
Rédiger ou collecter des exemples avec leur provenance
        ↓
Vérifier syntaxe → structure → doublons → contenu → distribution
        ↓
Documenter chaque correction et chaque retrait
        ↓
Conserver corpus vérifié + rapport + fiche du dataset

Format valide : le programme sait lire les données.
Contenu correct : les réponses ont été vérifiées dans leur contexte.
Couverture : les situations représentées.
Distribution : leur fréquence dans le corpus.
```

Le [module 10](10_charger_inspecter_et_transformer_les_donnees.md) utilisera Hugging Face Datasets pour charger, inspecter et transformer les données. Pour reprendre le résultat de ce laboratoire, remplace son chemin d’exemple `data/train.jsonl` par `data/module09/dataset.jsonl`, en tenant compte de ton dossier de lancement. La séparation en train, validation et test viendra au module 11.

## Références pour approfondir

- [Documentation Python du module `json`](https://docs.python.org/3/library/json.html) — lecture et écriture du JSON.
- [Description du format JSON Lines](https://jsonlines.org/) — UTF-8 et une valeur JSON par ligne.
- [Formats de datasets conversationnels dans TRL](https://huggingface.co/docs/trl/en/dataset_formats) — exemples de structures `messages`, rôles et contenus ; notre schéma pédagogique ajoute ses propres métadonnées et restrictions.
