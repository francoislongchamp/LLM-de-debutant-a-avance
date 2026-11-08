# Module 47 — Choisir la proportion de tokens réellement vus

## 1. Ce que tu vas apprendre

Concaténer des sources laisse leur taille naturelle décider de l’exposition. Une **mixture** choisit une distribution d’entraînement, mais tirer 50 % des documents d’une source ne donne pas forcément 50 % des tokens.

Tu vas calculer les tokens cibles, les répétitions implicites et une probabilité de tirage adaptée à des longueurs différentes.

**Prérequis :** modules 25, 45 et 46.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python uniquement.

## 2. Fixer l’unité de mixture

| Proportion souhaitée | Unité à suivre |
| --- | --- |
| Nombre de documents | Documents tirés |
| Positions d’entrée | Tokens non PAD ou cases selon la convention |
| Cibles de loss | Tokens après shift et masques |
| Temps de calcul | Mesure de durée, dépendante des formes et kernels |

Un même corpus peut avoir ces quatre proportions différentes. Annonce celle qui guide ton sampler et mesure les autres si elles influencent le coût ou l’objectif.

Après déduplication, les tailles disponibles changent. Recalcule le budget à partir du corpus réellement éligible, pas de sa version brute.

## 3. Calculer les expositions par source

Pour un budget T et des poids de tokens w_i dont la somme vaut 1 :

\[
T_i=w_iT,\qquad passages_i\approx T_i/T_{disponible,i}
\]

Budget de 1B tokens, dont 20 % de documents de domaine : 200M tokens exposés. Si la source contient 50M tokens disponibles, cela représente environ **quatre passages moyens**.

Cela ne crée pas 200M tokens distincts. Un sampling avec remise peut répéter certaines unités plus souvent et ne pas en visiter d’autres, même si le ratio moyen vaut quatre.

## 4. Documents longs : un biais facile à manquer

Deux sources A et B fournissent des documents de 10 et 100 tokens. Avec une probabilité de source 0.5 chacune, leur contribution moyenne vaut :

```text
A : 0.5 × 10  = 5 tokens par tirage
B : 0.5 × 100 = 50 tokens par tirage
```

La part token de B tend vers **50/55 ≈ 90.9 %**.

Pour viser une mixture de tokens w_i en tirant des documents de longueur moyenne m_i, une correction indicative est `p_i ∝ w_i/m_i`. Les longueurs variables, découpes et règles de fin de flux demandent ensuite une mesure réelle.

## 5. Laboratoire autonome — Probabilité de document et part token

```python
import random
from collections import Counter

lengths = {'short': 10, 'long': 100}


def simulate(weights, seed=42, draws=10000):
    sources = list(weights)
    rng = random.Random(seed)
    chosen = rng.choices(sources, weights=[weights[source] for source in sources], k=draws)
    docs = Counter(chosen)
    tokens = {source: docs[source] * lengths[source] for source in sources}
    return {'document_share': {source: docs[source] / draws for source in sources},
            'token_share': {source: tokens[source] / sum(tokens.values()) for source in sources}}


uniform_documents = simulate({'short': 0.5, 'long': 0.5})
corrected = {source: 0.5 / length for source, length in lengths.items()}
corrected_tokens = simulate(corrected)
print('Tirage uniforme des sources :', uniform_documents)
print('Poids corriges par longueur :', corrected_tokens)

budget = 1_000_000_000
mixture = {'web': 0.5, 'code': 0.2, 'domain': 0.2, 'books': 0.1}
available = {'web': 2_000_000_000, 'code': 300_000_000,
             'domain': 50_000_000, 'books': 200_000_000}
assert abs(sum(mixture.values()) - 1) < 1e-12
for source, weight in mixture.items():
    target = budget * weight
    print(source, 'tokens cibles=', int(target), 'passages moyens=', target / available[source])
```

La première simulation doit approcher 50/50 en documents et 9.1/90.9 en tokens. La seconde approche 50/50 en tokens, avec beaucoup plus de documents courts.

Les proportions ne sont pas exactement celles attendues sur un nombre fini de tirages. La seed rend la simulation reproductible dans cet environnement, pas les fréquences mathématiquement exactes.

## 6. Température de sampling : annoncer la convention

Une convention utilise `p_i ∝ n_i^a`, avec n_i une taille de source. Pour a=1, les probabilités suivent les tailles ; a=0 donne des sources équiprobables ; 0<a<1 aplatit la distribution.

D’autres outils utilisent un paramètre T et un exposant 1/T. Ne compare pas deux valeurs appelées « température » sans lire leur formule et leur unité de taille.

Aplatir la distribution surpondère généralement les petites sources par rapport à leur taille naturelle. Calcule leurs passages moyens pour repérer les répétitions excessives.

## 7. Fin d’une source et reprise

Décide ce qui arrive lorsqu’une petite source est épuisée : arrêter le mélange, la répéter, rééchantillonner ailleurs ou respecter un quota de tokens restant.

Ces choix peuvent modifier la mixture et le sens d’une epoch. Conserve état du sampler, RNG, positions dans les sources et compteurs pour une reprise cohérente.

En distribué, évite que tous les ranks tirent involontairement les mêmes documents avec le même flux aléatoire. La partition et la synchronisation doivent préserver le protocole choisi.

## 8. Expériences guidées

1. Remplace les longueurs fixes par des distributions et compare probabilités de source prévues et tokens observés.
2. Diminue fortement la taille disponible du domaine : calcule le nombre moyen de passages nécessaire avant tout training.
3. Sur ton corpus, logge source et cibles réellement actives par update. Compare au manifeste de mixture.
4. Évalue par domaine et langue, pas seulement la loss totale ; une meilleure moyenne peut masquer une perte sur une petite source.

## 9. Exercices avec corrigés

### Exercice 1 — Tokens cibles

Budget 500M, poids code 0.3 : combien de tokens code ?

<details>
<summary>Corrigé</summary>

**150M tokens** sous cette convention de mixture.

</details>

### Exercice 2 — Répétitions

La source code contient 50M tokens disponibles. Combien de passages moyens pour ce budget ?

<details>
<summary>Corrigé</summary>

**Trois** en moyenne. Cela ne garantit pas exactement trois visites de chaque document sous un sampling avec remise.

</details>

### Exercice 3 — Unité

50 % de documents d’une source garantit-il 50 % des cibles ?

<details>
<summary>Corrigé</summary>

Non. Longueurs, frontières, padding et masques peuvent modifier la contribution réelle à la loss.

</details>

### Exercice 4 — Upsampling

Peut-on dire qu’upsampler 20 fois un corpus multiplie par 20 l’information indépendante disponible ?

<details>
<summary>Corrigé</summary>

Non. Cela augmente l’exposition à son contenu et potentiellement la mémorisation, pas sa diversité intrinsèque.

</details>

## 10. Validation — avant de continuer

- [ ] Je définis l’unité des poids de mixture.
- [ ] Je calcule tokens cibles et passages moyens.
- [ ] Je retrouve le biais 90.9 % du cas long.
- [ ] Je vérifie la mixture observée et la convention de température.
- [ ] Je définis épuisement, RNG et reprise des sources.

## Fiche mémo

```text
Poids de tirage ≠ nécessairement part token.
Calculer l’exposition et la répétition, puis mesurer la distribution réellement consommée.
```

Le [module 48](48_data_parallel_et_ddp.md) répartit les données entre plusieurs processus.
