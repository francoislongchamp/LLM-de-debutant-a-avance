# Module 13 — Mesurer une amélioration avant de la déclarer

## 1. Ce que tu vas apprendre

Une réponse plus jolie après entraînement n’est pas une preuve d’amélioration générale. Une **baseline** est un résultat de référence, produit avant modification avec un protocole conservé.

Tu vas fixer ce protocole, comparer les mêmes questions avant/après, compter gains et régressions, puis interpréter les résultats par catégorie.

**Prérequis :** modules 11 et 12 ; split indépendant et distinction loss/qualité.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python uniquement pour le laboratoire. Les sorties fournies sont fictives et ne nécessitent pas de modèle.

## 2. Que faut-il garder identique ?

Un **benchmark** est un ensemble d’exemples accompagné d’une procédure de mesure. Un **score** est le résultat de cette procédure, pas une propriété absolue du modèle.

| Élément à fixer | Pourquoi |
| --- | --- |
| Questions et contexte | Pour comparer la même tâche |
| Template et tokenisation | Pour ne pas modifier silencieusement l’entrée |
| Limites de génération | Une réponse coupée peut échouer artificiellement |
| Décodage | Le sampling ajoute une source de variation |
| Grille de notation | Le sens du score doit rester constant |
| Modèle et révision | Identifier précisément le point de départ |

`do_sample=False` réduit la variabilité du choix des tokens, mais ne promet pas une identité numérique entre toutes les plateformes. Avec du sampling, conserve seeds et réglages, et compare plusieurs répétitions si la variabilité influence la conclusion.

La température n’a d’effet que si la stratégie l’utilise : afficher `temperature=1` dans une config ne prouve pas qu’un sampling est activé.

## 3. Définir une règle avant de lire les résultats

Pour un exercice de format JSON, une grille pourrait demander :

1. Le texte est-il un objet JSON valide ?
2. Les champs attendus sont-ils présents avec les bons types ?
3. Les valeurs répondent-elles correctement à la question ?

Un JSON valide peut être factuellement faux. Une métrique de syntaxe ne doit pas être présentée comme un score de vérité.

Pour des explications ouvertes, utilise une rubrique explicite : exactitude, pertinence, clarté, respect des contraintes. Si un juge humain ou un autre modèle note les sorties, masque l’identité de leur producteur et varie l’ordre de présentation pour limiter les biais de position.

## 4. Un exemple où la moyenne cache un problème

Sur huit questions fictives :

```text
Avant : 4 bonnes réponses sur 8 = 50 %
Après : 5 bonnes réponses sur 8 = 62.5 %
Gain global : +12.5 points de pourcentage
```

Mais le détail peut montrer une forte amélioration de format et une baisse sur le contenu. Il faut examiner **les mêmes exemples**, pas comparer des succès choisis séparément.

Une **comparaison appariée** associe chaque sortie avant à la sortie après du même prompt. Elle permet de distinguer les questions améliorées, dégradées et inchangées.

## 5. Laboratoire autonome — Compter les gains et les pertes

Les notes ci-dessous sont inventées pour apprendre l’analyse. `1` signifie réussite selon une règle déjà fixée, `0` échec. Le code ne prétend pas noter automatiquement la vérité des réponses.

```python
from collections import defaultdict

rows = [
    {"id": "f1", "category": "format", "before": 0, "after": 1},
    {"id": "f2", "category": "format", "before": 0, "after": 1},
    {"id": "f3", "category": "format", "before": 1, "after": 1},
    {"id": "f4", "category": "format", "before": 0, "after": 1},
    {"id": "c1", "category": "contenu", "before": 1, "after": 0},
    {"id": "c2", "category": "contenu", "before": 1, "after": 0},
    {"id": "c3", "category": "contenu", "before": 1, "after": 1},
    {"id": "c4", "category": "contenu", "before": 0, "after": 0},
]
assert len({row["id"] for row in rows}) == len(rows)
groups = defaultdict(list)
for row in rows:
    groups[row["category"]].append(row)
groups["total"] = rows

for category, examples in groups.items():
    before = sum(row["before"] for row in examples) / len(examples)
    after = sum(row["after"] for row in examples) / len(examples)
    print(category, "n=", len(examples), "avant=", before,
          "après=", after, "gain en points=", 100 * (after - before))

improved = [row["id"] for row in rows if row["after"] > row["before"]]
regressed = [row["id"] for row in rows if row["after"] < row["before"]]
unchanged = [row["id"] for row in rows if row["after"] == row["before"]]
print("Gains :", improved)
print("Régressions :", regressed)
print("Inchangés :", unchanged)
```

Tu dois retrouver :

| Catégorie | Avant | Après | Variation |
| --- | --- | --- | --- |
| Format | 25 % | 100 % | +75 points |
| Contenu | 75 % | 25 % | −50 points |
| Total | 50 % | 62.5 % | +12.5 points |

Il y a **3 gains**, **2 régressions** et **3 résultats inchangés**. Le modèle fictif s’améliore sur la moyenne retenue tout en perdant une compétence importante.

Le gain relatif global serait `(0.625−0.5)/0.5 = 25 %`. Ce n’est pas la même unité que **12.5 points de pourcentage**.

## 6. Construire tes artefacts réels

Pour chaque génération, conserve au minimum : ID du prompt, catégorie, entrée complète ou sa version, identifiant/révision du modèle, template, réglages de décodage, sortie brute et score avec sa justification.

Conserve aussi la version du script d’évaluation et les erreurs d’exécution. Écarter silencieusement les sorties invalides du dénominateur peut embellir le score.

La loss de validation et le benchmark de comportement se complètent. Une loss plus basse sur les cibles de référence ne garantit pas que le texte généré respecte mieux le format ou les faits.

### Ne pas transformer le test en outil de réglage

Développe ta grille et tes choix sur validation. Une fois le système choisi, utilise le test final selon le protocole du module 11. Si ses erreurs guident les changements suivants, reconnais qu’il est désormais réutilisé pour le développement.

## 7. Expériences guidées

1. Change seulement le nombre de questions de format en dupliquant leur poids dans le calcul. Observe comment le total peut monter sans amélioration supplémentaire sur le contenu ; discute le choix de pondération.
2. Réalise un benchmark de 30 prompts, avec au moins trois catégories et une grille écrite avant génération.
3. Évalue la base puis le checkpoint du module 12 avec la même procédure. Conserve aussi toutes les sorties dégradées.
4. Renote quelques sorties en masquant les noms des modèles. Si ton jugement change, note la source possible de biais.

Une moyenne par ligne donne davantage de poids aux catégories plus nombreuses. Une moyenne des catégories leur donne un poids égal : choisis et annonce ta convention.

## 8. Exercices avec corrigés

### Exercice 1 — Unité du gain

Un score passe de 40 % à 50 %. Quel est le gain en points ? Et le gain relatif ?

<details>
<summary>Corrigé</summary>

**+10 points de pourcentage** ; gain relatif de `(50−40)/40 = 25 %`. Écrire seulement « +25 % » sans préciser l’unité peut créer une confusion.

</details>

### Exercice 2 — Régression cachée

La moyenne augmente mais les diagnostics deviennent faux. Quelle analyse manque ?

<details>
<summary>Corrigé</summary>

Les scores par catégorie, les effectifs et l’inspection appariée des sorties. Une moyenne globale ne remplace pas une exigence minimale sur une compétence critique.

</details>

### Exercice 3 — Comparaison injuste

Avant : réponse limitée à 20 tokens. Après : 200 tokens. Peut-on attribuer tout le gain au fine-tuning ?

<details>
<summary>Corrigé</summary>

Non. La limite de sortie a aussi changé et peut supprimer des coupures. Il faut comparer à réglages compatibles ou mesurer séparément l’effet de cette modification.

</details>

### Exercice 4 — Petite différence

Tu gagnes une question sur dix. Peux-tu annoncer une amélioration robuste de 10 points ?

<details>
<summary>Corrigé</summary>

Tu peux rapporter le résultat observé et son effectif, mais sa robustesse reste à établir sur davantage de cas et, si pertinent, de répétitions. Les questions ne représentent peut-être qu’une petite partie de la tâche.

</details>

## 9. Validation — avant de continuer

- [ ] Je définis une baseline et une grille avant le training.
- [ ] Je fixe les conditions de génération et conserve les sorties brutes.
- [ ] Je distingue gain relatif et points de pourcentage.
- [ ] Je retrouve les résultats globaux et par catégorie du laboratoire.
- [ ] Je rapporte les régressions et les effectifs.
- [ ] Je sais ce qu’une moyenne ou un petit benchmark ne démontrent pas.

## Fiche mémo

```text
Même question + même procédure + règle fixée = comparaison interprétable.
Mesurer total ET catégories ; conserver gains ET régressions.
Le score évalue un protocole précis, pas une qualité universelle.
```

Le [module 14](14_surapprentissage_et_generalisation.md) examine ce qui arrive lorsque le train continue de progresser mais la validation se dégrade.
