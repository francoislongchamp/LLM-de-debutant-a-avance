# Module 31 — Construire une récompense vérifiable et testée

## 1. Ce que tu vas apprendre

Au [module 30](30_fondements_du_reinforcement_learning.md), une policy a appris à choisir une réponse récompensée. Si le correcteur accepte une mauvaise réponse, l’apprentissage peut renforcer cette erreur.

Tu vas écrire un contrat de sortie, séparer parser et oracle, tester les cas limites et mesurer ce que le score accepte réellement.

**Prérequis :** modules 9, 29 et 30 ; dictionnaires, exceptions et fonctions Python.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python uniquement. Aucun modèle ni service externe.

## 2. Définir le contrat avant le code

Tâche : multiplier deux entiers fournis dans le prompt. Sortie attendue : un objet JSON contenant uniquement `answer`, dont la valeur est un entier JSON.

| Sortie pour 17×18 | Décision de notre contrat |
| --- | --- |
| `{"answer":306}` | Acceptée, correcte |
| `{"answer":308}` | Structure acceptée, réponse fausse |
| `306` | Refusée : pas un objet |
| `{"answer":"306"}` | Refusée : chaîne |
| `{"answer":306.0}` | Refusée : nombre décimal |
| `{"answer":true}` | Refusée : booléen |
| Deux clés `answer` | Refusée : ambiguïté |
| Explication suivie du JSON | Refusée : texte hors objet |

Ces choix conviennent à cet exercice strict. Une tâche qui autorise des réponses numériques équivalentes demanderait un autre parser, versionné et testé.

## 3. Trois responsabilités

```text
Génération → parser → valeur valide ou raison d’échec
Entrées du problème → oracle → valeur de référence
Valeur parsée + référence → score
```

Le **parser** lit la réponse selon le contrat. L’**oracle** détermine la référence ; ici une multiplication Python. La **reward** combine ces résultats selon une règle explicite.

Un LLM juge n’est pas nécessaire pour cette tâche. Utiliser `eval()` pour lire la sortie serait inutile : JSON fournit une grammaire de données adaptée.

Une reward déterministe peut être reproductible tout en étant fausse : il faut tester son accord avec le contrat, pas seulement sa constance.

## 4. Deux pièges du JSON Python

`json.loads` accepte plusieurs types JSON : objet, liste, nombre, chaîne… Il faut vérifier le type de la valeur retournée avant d’utiliser ses champs.

En Python, `bool` est un sous-type de `int` et `True == 1`. Pour exiger un entier qui n’est pas un booléen, nous utilisons `type(value) is int`.

Par défaut, des clés répétées peuvent être acceptées en gardant une valeur. Le hook du laboratoire les refuse. Nous refusons aussi les constantes non standard comme `NaN`.

## 5. Laboratoire autonome — Un correcteur strict

Le script ne génère pas de réponses avec une policy. Il vérifie le correcteur sur des cas dont la décision est connue à l’avance.

```python
import json


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('Cle repetee')
        obj[key] = value
    return obj


def reject_constant(value):
    raise ValueError('Constante JSON non standard')


def parse_answer(text):
    if not isinstance(text, str) or len(text) > 4096:
        return None
    try:
        obj = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, RecursionError):
        return None
    if not isinstance(obj, dict) or set(obj) != {'answer'}:
        return None
    value = obj['answer']
    return value if type(value) is int else None


def oracle(a, b):
    return a * b


def score(text, expected):
    value = parse_answer(text)
    valid = value is not None
    correct = valid and value == expected
    return {'valid': valid, 'correct': correct, 'reward': float(correct)}


cases = [
    ('{"answer":306}', 306, 1),
    (' {"answer": 306}\n', 306, 1),
    ('{"answer":308}', 306, 0),
    ('306', 306, 0), ('null', 306, 0), ('[]', 306, 0),
    ('true', 1, 0), ('{"answer":true}', 1, 0),
    ('{"answer":false}', 0, 0), ('{"answer":306.0}', 306, 0),
    ('{"answer":"306"}', 306, 0), ('{"answer":null}', 306, 0),
    ('{"answer":[]}', 306, 0), ('{"answer":{}}', 306, 0),
    ('{}', 306, 0), ('{"result":306}', 306, 0),
    ('{"answer":306,"extra":0}', 306, 0),
    ('{"answer":306,"answer":308}', 306, 0),
    ('{"answer":308,"answer":306}', 306, 0),
    ('{"answer":NaN}', 306, 0), ('{"answer":Infinity}', 306, 0),
    ('{"answer":306} merci', 306, 0), ('', 306, 0),
    ('{"answer":0}', 0, 1), ('{"answer":-306}', -306, 1),
]
for text, expected, wanted in cases:
    result = score(text, expected)
    assert result['reward'] == wanted, (text, result)
print(len(cases), 'tests de contrat passes')
print('Oracle 17x18 :', oracle(17, 18))
print('Faux mais valide :', score('{"answer":308}', 306))
```

Tu dois obtenir **25 tests réussis**, l’oracle **306**, puis `valid=True`, `correct=False`, `reward=0.0` pour 308.

La valeur entière zéro reste valide : nous ne testons pas simplement `if value`, ce qui la confondrait avec l’absence de réponse.

La limite de longueur et le refus d’entrées trop profondes bornent certains cas du contrat ; ce n’est pas un exécuteur sécurisé de code arbitraire. Nous ne faisons exécuter aucun contenu généré.

## 6. Mesurer le signal initial

Sur des générations réelles, rapporte séparément : taux de format valide, exactitude globale, exactitude parmi les formats valides et distribution de reward.

Si 80 réponses sur 100 ont le bon format et 20 sont exactes, l’exactitude globale vaut 20 %, et l’exactitude conditionnelle au format valide vaut 25 %. Les deux dénominateurs ne répondent pas à la même question.

Une reward presque toujours nulle apporte peu de contraste dans certains algorithmes. Avant de changer la fonction, vérifie difficulté, parser et comportements de la policy. Un curriculum ou quelques démonstrations peuvent être des interventions à comparer, sans ajouter discrètement la cible au prompt.

## 7. Expériences guidées

1. Transforme le contrat en classification parmi quatre chaînes autorisées. Écris les décisions pour casse, espaces, clés supplémentaires et type avant le parser.
2. Ajoute vingt cas : réponses négatives, grands entiers, Unicode, chaînes échappées et objets imbriqués. Chaque test doit exprimer une règle utile, pas recopier l’implémentation.
3. Score cent sorties initiales d’une policy puis relis des succès et échecs. Un correcteur erroné peut embellir ou dégrader le taux mesuré.
4. Versionne le contrat et les tests. Si tu autorises désormais `306.0`, crée une nouvelle version et recalcule les comparaisons concernées.

## 8. Exercices avec corrigés

### Exercice 1 — Parser ou oracle ?

Quelle fonction calcule le résultat de 17×18 et laquelle décide si une chaîne est un JSON acceptable ?

<details>
<summary>Corrigé</summary>

L’oracle calcule la référence. Le parser vérifie et extrait la réponse selon le contrat. Le scoring compare ensuite les deux.

</details>

### Exercice 2 — Booléen

Pourquoi `isinstance(True, int)` ne suffit-il pas à refuser un booléen ?

<details>
<summary>Corrigé</summary>

Parce que bool hérite de int en Python. Notre contrat exige le type exact int, distinct de bool.

</details>

### Exercice 3 — Deux exactitudes

Sur 200 réponses, 100 sont valides et 40 exactes. Quels taux globaux et conditionnels ?

<details>
<summary>Corrigé</summary>

Format valide : 50 %. Exactitude globale : 20 %. Exactitude parmi les valides : 40 %. Annonce toujours le dénominateur.

</details>

### Exercice 4 — Reward nulle

Tous les scores valent zéro. Est-ce une preuve que le modèle ne comprend rien ?

<details>
<summary>Corrigé</summary>

Non. Le contrat, le parser, la troncature ou le niveau de difficulté peuvent être en cause. Inspecte les textes et les raisons d’échec avant d’interpréter la compétence.

</details>

## 9. Validation — avant de continuer

- [ ] Je spécifie types, format et équivalences acceptées.
- [ ] Je sépare parser, oracle et scoring.
- [ ] Je retrouve les 25 tests et sais ajouter des contre-exemples.
- [ ] Je distingue format valide et réponse correcte.
- [ ] Je mesure les bons dénominateurs et versionne le correcteur.

## Fiche mémo

```text
Contrat → parser testé + oracle → scoring → audit sur sorties réelles.
Le reward est une spécification exécutable ; sa fiabilité se vérifie.
```

Le [module 32](32_rewards_composites_et_normalisation.md) combine plusieurs signaux sans perdre leurs priorités.

## Référence

[Python : module json](https://docs.python.org/3/library/json.html) — types acceptés, hooks et erreurs de parsing.
