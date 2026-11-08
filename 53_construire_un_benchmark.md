# Module 53 — Transformer un objectif en benchmark versionné

## 1. Ce que tu vas apprendre

Un benchmark associe tâches, données réservées, procédure de génération et règles de scoring. Il doit pouvoir révéler une régression autant qu’un gain.

Tu vas écrire des cas à IDs stables, séparer format et contenu, compter les sorties manquantes et conserver résultats bruts et manifeste.

**Prérequis :** modules 11, 13, 31 et 52.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python uniquement pour le banc d’analyse.

## 2. Partir du contrat produit

Liste les comportements à améliorer et ceux à préserver. Une suite peut contenir format, exactitude de domaine, raisonnement vérifiable et régression générale.

Pour chaque catégorie, fixe avant génération : effectif, type de question, références, normalisations acceptées, métrique et seuils. Une règle de format ne prouve pas les faits contenus dans ce format.

Les cas utilisés pour régler prompt, données et hyperparamètres deviennent des données de développement. Garde un test final selon le protocole du module 11.

## 3. Plusieurs familles de scoring

| Type | Usage | Limite |
| --- | --- | --- |
| Exact match sous contrat | Classes, résultats ou formats stricts | Peut rejeter des variantes légitimes si le contrat est trop étroit |
| Tests fonctionnels | Comportement vérifiable | Couverture des tests et conditions d’exécution |
| Rubrique humaine | Qualités ouvertes | Bruit, formation des juges et biais |
| Juge LLM | Aide à la notation ouverte | Accord, biais de style/position et fiabilité à mesurer |

Pour des comparaisons qualitatives, masque l’identité du système, équilibre l’ordre A/B et annonce traitement des égalités. Les tests de code demandent un environnement d’exécution adapté ; ce laboratoire ne lance pas de code généré.

## 4. Micro-suite avec format et contenu séparés

Le laboratoire contient cinq cas artificiels. Le scorer exige un objet JSON aux champs et types attendus. La vérité est ici fournie par les références de l’exercice.

Les deux jeux de sorties sont **fictifs**. Leur rôle est de montrer l’agrégation et les régressions, pas d’attribuer ces scores à un modèle réel.

## 5. Laboratoire autonome — Une suite, deux systèmes, tous les cas comptés

```python
import hashlib
import json
from collections import defaultdict
from pathlib import Path

benchmark = [
    {'id': 'f1', 'category': 'format', 'prompt': '2+2, JSON answer.', 'expected': {'answer': 4}},
    {'id': 'f2', 'category': 'format', 'prompt': 'Derivee locale, JSON label.', 'expected': {'label': 'gradient'}},
    {'id': 'c1', 'category': 'calcul', 'prompt': '17x18, JSON answer.', 'expected': {'answer': 306}},
    {'id': 'c2', 'category': 'calcul', 'prompt': '1-1, JSON answer.', 'expected': {'answer': 0}},
    {'id': 'r1', 'category': 'regression', 'prompt': 'Groupe exemples, JSON label.', 'expected': {'label': 'batch'}},
]
outputs = {
    'before': {'f1': '4', 'f2': '{"label":"gradient"}', 'c1': '{"answer":306}',
               'c2': '{"answer":1}', 'r1': '{"label":"batch"}'},
    'after': {'f1': '{"answer":4}', 'f2': '{"label":"gradient"}', 'c1': '{"answer":308}',
              'c2': '{"answer":0}', 'r1': '{"label":"epoch"}'},
}


def unique_object(pairs):
    if len(dict(pairs)) != len(pairs):
        raise ValueError('Cle repetee')
    return dict(pairs)


def score(case, text):
    try:
        obj = json.loads(text, object_pairs_hook=unique_object)
    except ValueError:
        return False, False
    expected = case['expected']
    format_ok = (isinstance(obj, dict) and set(obj) == set(expected)
                 and all(type(obj[key]) is type(expected[key]) for key in expected))
    return format_ok, format_ok and obj == expected


ids = {case['id'] for case in benchmark}
assert len(ids) == len(benchmark)
records = []
for system, predictions in outputs.items():
    assert set(predictions) <= ids
    groups = defaultdict(list)
    for case in benchmark:
        present = case['id'] in predictions
        text = predictions.get(case['id'], '')
        format_ok, correct = score(case, text)
        record = {'system': system, 'id': case['id'], 'category': case['category'],
                  'present': present, 'output': text, 'format_ok': format_ok, 'correct': correct}
        records.append(record)
        groups[case['category']].append(record)
    groups['total'] = [row for row in records if row['system'] == system]
    for category, rows in groups.items():
        print(system, category, 'n=', len(rows), 'correct=', sum(row['correct'] for row in rows),
              'missing=', sum(not row['present'] for row in rows))

root = Path('outputs/module53')
root.mkdir(parents=True, exist_ok=True)
serialized = json.dumps(benchmark, ensure_ascii=False, sort_keys=True)
(root / 'benchmark.json').write_text(serialized, encoding='utf-8')
(root / 'results.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8')
manifest = {'benchmark_version': 'toy-v1', 'sha256': hashlib.sha256(serialized.encode()).hexdigest(),
            'outputs': 'fictitious', 'scoring': 'exact_keys_exact_types_exact_values-v1'}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
```

Les deux systèmes réussissent **3/5** cas. Pourtant le format passe de 1/2 à 2/2, le calcul reste à 1/2 avec un gain et une perte, et la régression passe de 1/1 à 0/1.

Les métriques par catégorie et les IDs appariés révèlent ce que le total cache. Une sortie manquante doit rester un échec ou une catégorie d’erreur explicite selon ton protocole, pas disparaître du dénominateur.

## 6. Versionner ce qui change le sens du score

Un benchmark contient aussi template, sources de contexte, limites de génération, paramètres de décodage, code de scoring et conventions de normalisation.

Si tu réécris une question ambiguë ou changes la tolérance d’un scorer, crée une version et recalcule les conditions comparées. Un hash de questions ne versionne pas à lui seul le juge ou les réglages de génération.

Conserve les sorties brutes pour pouvoir analyser une nouvelle règle sans régénérer et confondre deux sources de variation.

## 7. Taille, groupes et couverture

Cinq cas servent au micro-exemple, pas à une conclusion fiable sur un produit. Choisis les familles de situations, groupes indépendants et effectifs selon la précision de mesure nécessaire.

Un benchmark général peut être très nombreux mais manquer une compétence critique. Un jeu de régression ciblé peut être utile sans être représentatif de toute la distribution ; annonce son rôle.

Le module 54 ajoute répétitions et incertitude ; aucune méthode statistique ne répare un benchmark contaminé ou une grille qui mesure le mauvais objectif.

## 8. Expériences guidées

1. Supprime une prédiction et vérifie qu’elle reste dans les cinq cas avec une erreur visible.
2. Ajoute un booléen à la place d’un entier puis une clé répétée ; vérifie le contrat.
3. Prépare une rubrique d’explication technique avec critères indépendants : exactitude, pertinence, clarté. Définis comment traiter désaccords et égalités.
4. Sur un modèle réel, remplace seulement les sorties fictives par des générations archivées avec la même suite et le même protocole.

## 9. Exercices avec corrigés

### Exercice 1 — Total inchangé

Pourquoi 3/5 avant et après ne signifie-t-il pas absence de changement ?

<details>
<summary>Corrigé</summary>

Les identités des réussites et les catégories changent. Le total peut compenser des gains par des pertes importantes.

</details>

### Exercice 2 — JSON vrai ?

Un objet valide selon le schéma prouve-t-il que sa valeur répond à la question ?

<details>
<summary>Corrigé</summary>

Non. Le format et la correction du contenu sont deux tests distincts.

</details>

### Exercice 3 — Sortie manquante

Pourquoi exclure une génération qui a échoué techniquement peut-il embellir le score ?

<details>
<summary>Corrigé</summary>

Le dénominateur retire un cas où le système n’a pas fourni la réponse attendue. L’échec doit être rapporté selon une règle fixée à l’avance.

</details>

### Exercice 4 — Évaluation à l’aveugle

Que cherche-t-on à réduire en masquant l’identité des modèles ?

<details>
<summary>Corrigé</summary>

Les attentes et préférences liées au nom du système. Il faut aussi traiter biais de position, style et qualité de la grille ; l’aveugle ne supprime pas tous les biais.

</details>

## 10. Validation — avant de continuer

- [ ] Je définis capacités, références et scorers avant les sorties.
- [ ] Je garde tous les IDs, y compris erreurs et manquants.
- [ ] Je sépare format, contenu et catégories de régression.
- [ ] Je conserve sortie brute, protocole et version.
- [ ] Je ne généralise pas les résultats fictifs ou minuscules du laboratoire.

## Fiche mémo

```text
Le benchmark est un contrat de mesure avec ses données et son protocole.
Rapporter catégories, effectifs et cas individuels autant que le total.
```

Le [module 54](54_comparer_scientifiquement_des_entrainements.md) organise une comparaison contrôlée et son incertitude.
