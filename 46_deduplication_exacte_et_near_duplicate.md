# Module 46 — Détecter des copies sans confondre ressemblance et équivalence

## 1. Ce que tu vas apprendre

Deux textes peuvent être identiques après une normalisation ou partager de nombreux fragments sans être équivalents. La déduplication influence exposition, mémorisation et contamination.

Tu vas construire des hashes, des shingles et une similarité Jaccard, puis examiner les limites d’un seuil.

**Prérequis :** modules 9, 11 et 45.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python uniquement.

## 2. Définir ce que signifie « identique »

La **canonicalisation** transforme le texte avant comparaison. Normaliser les fins de ligne ou certains espaces peut être adapté à un corpus ; supprimer casse, accents ou ponctuation peut fusionner des contenus distincts.

Un hash SHA-256 identifie une représentation d’octets. Il ne fournit pas une distance sémantique. En pratique, conserve aussi le contenu ou une vérification d’égalité pour confirmer les candidats exacts selon le contrat.

Exemples de risque : `US` et `us`, du code indenté et désindenté, `résumé` et `resume`. Une règle « tout en minuscules » doit être justifiée par le domaine.

## 3. Shingles et Jaccard

Un **shingle de k mots** est un fragment de k mots consécutifs. Pour `le chat noir dort` et k=2 :

```text
{('le','chat'), ('chat','noir'), ('noir','dort')}
```

On compare les ensembles par :

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

Pour `{a,b,c}` et `{b,c,d}`, J=2/4=0.5.

La représentation en ensemble oublie combien de fois chaque shingle est répété. Un score élevé révèle un recouvrement de fragments, pas forcément une paraphrase exacte ni une vérité commune.

## 4. Laboratoire autonome — Exact et proche

```python
import hashlib
from itertools import combinations


def canonical(text):
    return ' '.join(text.split())  # La casse est préservée dans cet exercice.


def fingerprint(text):
    return hashlib.sha256(canonical(text).encode('utf-8')).hexdigest()


def shingles(text, size=3):
    words = canonical(text).split()
    return {tuple(words[index:index + size]) for index in range(len(words) - size + 1)}


def jaccard(left, right):
    union = left | right
    if not union:
        return None  # Absence de représentation informative pour nos textes courts.
    return len(left & right) / len(union)


documents = {
    'a': 'le chat noir dort sur le canape',
    'b': 'le  chat noir dort sur le canape ',
    'c': 'le chat noir dort sur le grand canape',
    'd': 'le chat noir dort sur le canape rarement',
    'e': 'bonjour',
    'f': 'autre',
}
seen = {}
kept = {}
removed = []
for doc_id, text in documents.items():
    key = fingerprint(text)
    if key in seen:
        removed.append((doc_id, seen[key]))
    else:
        seen[key] = doc_id
        kept[doc_id] = text

candidates = []
for (left_id, left), (right_id, right) in combinations(kept.items(), 2):
    similarity = jaccard(shingles(left), shingles(right))
    if similarity is not None and similarity >= 0.55:
        candidates.append((left_id, right_id, similarity))
print('Copies exactes retirees :', removed)
print('Candidats proches :', candidates)
print('Textes courts sans shingles :', jaccard(shingles('bonjour'), shingles('autre')))
```

La copie **b** de **a** doit être retirée par normalisation des espaces. Les variantes c et d peuvent être signalées comme proches. Le seuil 0.55 est choisi pour l’exercice, pas recommandé universellement.

Les textes de moins de trois mots donnent ici des ensembles vides. Les déclarer automatiquement similaires à 1 parce que les deux ensembles sont vides serait une décision dangereuse : le code retourne `None` et demande un autre traitement.

## 5. Décider après le signalement

Un détecteur de proximité produit des candidats. Pour un petit corpus, relis les paires et décide : conserver deux variantes utiles, fusionner leurs provenances ou choisir un représentant selon une règle.

Le mot ajouté « rarement » modifie le sens. Une similarité élevée ne suffit pas à déclarer les phrases interchangeables.

La proximité n’est pas nécessairement transitive : A ressemble à B et B à C n’implique pas le même score A/C. Un clustering par composantes connexes peut donc créer des groupes plus larges que le seuil appliqué paire par paire. Documente ta règle de regroupement et la sélection du représentant.

## 6. Passer à l’échelle : MinHash et LSH

Comparer toutes les paires demande `N(N−1)/2` comparaisons. Pour un million de documents, cela représente environ 500 milliards de paires.

**MinHash** produit des signatures probabilistes d’ensembles qui permettent d’estimer Jaccard. **LSH** indexe des signatures pour obtenir des candidats susceptibles d’être proches, réduisant les comparaisons détaillées.

Ces mécanismes introduisent des paramètres et faux positifs/négatifs. Ils ne remplacent pas une validation sur un échantillon annoté, ni la définition de shingles, normalisation et seuil adaptés.

## 7. Déduplication et splits

Conserve les clusters et sources pour les affecter sans fuite aux rôles train/validation/test. Une copie sous un autre document ou une paraphrase peut traverser des splits même si les IDs sont disjoints.

Dédupliquer ne résout pas toute contamination sémantique ou historique d’un modèle préentraîné. Le résultat dépend des détecteurs et des données que tu peux inspecter.

Après retrait, recompte documents et tokens par source. Une source répétée peut perdre une part importante de son exposition.

## 8. Expériences guidées

1. Compare k=2,3,5 et inspecte les candidats, notamment les textes courts.
2. Ajoute une négation et des nombres changés dans des documents longs. Cherche les faux équivalents malgré le recouvrement lexical.
3. Fais varier le seuil et annote faux positifs/faux négatifs sur un jeu réservé.
4. Écris un rapport avec copie retirée, représentant, source et justification ; ne supprime pas seulement une ligne sans trace.

## 9. Exercices avec corrigés

### Exercice 1 — Jaccard

A={a,b,c,d}, B={c,d,e} : quel score ?

<details>
<summary>Corrigé</summary>

Intersection de taille 2, union de taille 5 : **0.4**.

</details>

### Exercice 2 — Ensembles vides

Pourquoi le pilote ne renvoie-t-il pas 1 pour deux textes de deux mots avec k=3 ?

<details>
<summary>Corrigé</summary>

Ils ne disposent d’aucun shingle sous cette représentation. Leur absence d’information ne prouve pas leur identité ; il faut une règle de repli adaptée.

</details>

### Exercice 3 — Sémantique

Un Jaccard de 0.9 garantit-il le même sens ?

<details>
<summary>Corrigé</summary>

Non. Une petite différence, notamment nombre ou négation, peut changer une information importante tout en laissant la plupart des fragments communs.

</details>

### Exercice 4 — Échelle

Pourquoi utiliser un index de candidats avant de comparer un très grand corpus ?

<details>
<summary>Corrigé</summary>

Le nombre de paires croît quadratiquement. Un index limite les vérifications coûteuses, avec une qualité de rappel/précision à évaluer.

</details>

## 10. Validation — avant de continuer

- [ ] Je définis la canonicalisation et ses risques.
- [ ] Je distingue hash exact, shingles et similarité sémantique.
- [ ] Je traite les textes courts sans inventer une identité.
- [ ] Je conserve candidats, clusters, représentants et provenance.
- [ ] Je comprends le rôle et les limites de MinHash/LSH.

## Fiche mémo

```text
Comparer exactement selon une règle ; signaler les ressemblances selon une autre.
Une proximité numérique n’autorise pas automatiquement une suppression.
```

Le [module 47](47_dataset_mixture_et_sampling.md) choisit l’exposition des sources après ces transformations.
