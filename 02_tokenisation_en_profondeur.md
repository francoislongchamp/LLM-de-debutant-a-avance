# Module 2 — Comprendre comment le texte devient des tokens

## 1. Ce que tu vas apprendre

Le [module 1](01_pipeline_complet_dun_llm.md) t’a montré que le réseau reçoit des IDs. Ici, nous ouvrons l’outil qui les produit : le **tokenizer**.

À la fin, tu devrais pouvoir :

- distinguer caractère, octet, mot, token et identifiant ;
- appliquer quelques fusions BPE à la main ;
- inspecter un tokenizer réel sans charger les poids du modèle ;
- mesurer la fragmentation d’un texte et en expliquer les limites ;
- distinguer entraîner un tokenizer et entraîner un modèle de langage.

**Prérequis :** module 1 ; listes, chaînes et boucles Python.

**Temps indicatif :** 90 à 150 minutes, expériences comprises.

**Matériel :** Python pour le premier laboratoire ; Transformers pour le second. Aucun GPU, aucun poids de LLM à charger.

## 2. Cinq objets à ne pas confondre

Prenons `café` :

| Objet | Ce que l’on compte | Pour ce texte |
| --- | --- | --- |
| Mot | Une unité linguistique selon une convention | Un mot |
| Caractère Python | Un point de code Unicode | 4 dans cette écriture |
| Octet UTF-8 | Une unité du codage en mémoire/fichier | 5 : `é` utilise deux octets |
| Token | Une unité choisie par un tokenizer donné | Nombre à mesurer |
| ID | L’adresse numérique d’un token dans son vocabulaire | Dépend du tokenizer |

Un caractère visible peut parfois être composé de plusieurs points de code : un `é` peut être représenté de différentes manières en Unicode. Ne généralise donc pas « un signe visible = une case Python = un octet ».

Un tokenizer peut représenter un mot entier, plusieurs fragments ou des unités reposant sur des octets. Le découpage n’est pas une analyse grammaticale obligatoire.

### Pourquoi les sous-mots ?

Un vocabulaire contenant chaque mot possible serait difficile à maintenir : noms propres, variantes, mots nouveaux et identifiants se multiplient. À l’inverse, représenter tout texte caractère par caractère produit des séquences longues.

Les sous-mots offrent un compromis : conserver des fragments fréquents tout en permettant d’assembler d’autres chaînes. Leur découpage ne suit pas nécessairement les syllabes ni les racines linguistiques.

## 3. Le pipeline du tokenizer

```text
Texte
  ↓ normalisation éventuelle
Texte normalisé
  ↓ pré-tokenisation éventuelle
Morceaux de départ
  ↓ algorithme de segmentation : BPE, Unigram, etc.
Tokens et IDs
  ↓ post-traitement éventuel
Séquence finale avec marqueurs spéciaux
```

| Étape | Exemple possible | Effet à examiner |
| --- | --- | --- |
| Normalisation | Passer en minuscules ou normaliser Unicode | Peut modifier la forme du texte |
| Pré-tokenisation | Repérer espaces et ponctuation | Peut fixer des frontières que les fusions ne traverseront pas |
| Segmentation | Représenter un mot en sous-unités | Détermine les tokens ordinaires |
| Post-traitement | Ajouter BOS ou EOS | Consomme des positions supplémentaires |

Toutes les configurations n’utilisent pas les mêmes étapes. La normalisation n’est pas forcément activée, et un espace peut faire partie de la représentation d’un token.

**Encoder** signifie convertir du texte en IDs. **Décoder** signifie reconstruire du texte à partir des IDs. Si une étape normalise ou supprime de l’information, le texte reconstruit peut différer de l’original.

## 4. BPE : faire les fusions à la main

**BPE**, pour *Byte Pair Encoding*, apprend une suite de fusions de paires adjacentes fréquentes. Commençons avec un exemple volontairement simplifié : on découpe chaque mot en caractères, et on interdit les fusions entre mots.

Corpus :

```text
bas bas basse
```

Au départ :

```text
b a s | b a s | b a s s e
```

La paire `b a` apparaît trois fois. La paire `a s` aussi : il y a une égalité. Supposons que notre règle de départage choisisse `b a`.

| Étape | Représentation | Nombre total d’unités |
| --- | --- | --- |
| Départ | `b a s` / `b a s` / `b a s s e` | 11 |
| Fusion `b` + `a` | `ba s` / `ba s` / `ba s s e` | 8 |
| Fusion `ba` + `s` | `bas` / `bas` / `bas s e` | 5 |

La paire `ba s` apparaît trois fois après la première fusion : elle peut être fusionnée à son tour.

L’ordre des fusions fait partie du tokenizer appris. À l’encodage d’un nouveau texte, un tokenizer BPE utilise ses règles déjà apprises ; il ne réapprend pas les fréquences à chaque phrase.

> **À toi.** Après ces deux fusions, combien d’unités faut-il pour `basse` ?
>
> Trois : `bas`, `s`, `e`. Ce découpage vient de notre règle simplifiée, pas d’une affirmation sur un tokenizer réel.

### Laboratoire A — Reproduire ces fusions

Ce bloc est autonome et utilise seulement Python. Il applique les deux règles choisies ; il ne cherche pas automatiquement les meilleures paires et ne constitue pas un tokenizer BPE complet.

```python
words = [list(word) for word in "bas bas basse".split()]


def merge_pair(symbols, pair):
    result = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            result.append(symbols[index] + symbols[index + 1])
            index += 2
        else:
            result.append(symbols[index])
            index += 1
    return result


print("Départ :", words, "unités :", sum(len(word) for word in words))
for pair in [("b", "a"), ("ba", "s")]:
    words = [merge_pair(word, pair) for word in words]
    print("Fusion :", pair, "résultat :", words,
          "unités :", sum(len(word) for word in words))
```

Tu dois retrouver **11 → 8 → 5 unités**. Modifie le corpus et prédis le résultat avec les mêmes règles : elles ne changent pas automatiquement lorsque le texte change.

## 5. Byte-level, byte fallback et Unigram

Ces termes décrivent des choix qu’il faut distinguer :

- **Byte-level** : travailler sur une représentation fondée sur les octets, avec des fusions possibles. Une base couvrant les 256 valeurs d’octets permet de représenter les chaînes UTF-8 sans réserver un token par caractère Unicode.
- **Byte fallback** : retomber sur des unités d’octets lorsqu’un fragment ne dispose pas d’une représentation ordinaire adaptée.
- **Unigram** : une autre famille de segmentation, qui attribue des scores à des unités candidates et recherche une segmentation selon ces scores. Son apprentissage ne se décrit pas comme la suite de fusions BPE ci-dessus.

Un tokenizer byte-level peut avoir bien plus de 256 tokens grâce aux fusions. Des tokens d’octets isolés ne sont pas nécessairement des caractères Unicode complets ; les décoder séparément peut afficher un caractère de remplacement.

À ce stade, retiens surtout qu’un choix d’algorithme et un vocabulaire appris déterminent les unités. Le [module 38](38_entrainer_un_tokenizer_bpe.md) abordera l’entraînement d’un tokenizer.

## 6. Tokenizer et modèle doivent rester compatibles

L’ID 42 d’un tokenizer n’a pas nécessairement le même contenu que l’ID 42 d’un autre. Le modèle a appris à exploiter les lignes d’embedding correspondant à **son** tokenizer.

Changer de tokenizer tout en gardant le même modèle peut donc lui fournir des IDs de sens différent, même si le programme accepte les dimensions.

Ajouter un token ne lui apprend pas automatiquement sa signification. Selon l’opération, il faut aussi adapter les embeddings et la sortie du réseau, puis entraîner les paramètres concernés.

Le **vocabulaire du tokenizer** décrit ce qu’il peut encoder directement. Les **connaissances du modèle** dépendent de ses paramètres appris. Un terme absent comme token unique peut malgré tout être représenté et utilisé via plusieurs tokens.

## 7. Mesurer la fragmentation

Pour un échantillon donné :

\[
\text{tokens par mot} = \frac{N_{tokens}}{N_{mots}}
\qquad
\text{tokens par caractère} = \frac{N_{tokens}}{N_{caractères}}
\]

Le premier ratio est parfois appelé **fertility**. Il faut préciser comment on compte les mots et si les tokens spéciaux sont inclus.

Exemple : 100 mots selon une convention, encodés en 135 tokens ordinaires, donnent `1.35 token par mot`.

Si le même texte prend 200 tokens avec un autre tokenizer, il utilise davantage de positions. Cela ne prouve pas à lui seul que le deuxième modèle est moins bon : la qualité du réseau, ses données et sa tâche comptent aussi.

Comparer deux langues avec des mots comptés uniquement par espaces peut être trompeur, notamment pour les langues qui n’utilisent pas ces séparateurs de la même façon. Nos ratios seront des mesures locales, pas un classement universel des langues.

## 8. Laboratoire B — Inspecter un tokenizer réel

Ce bloc est autonome. Il nécessite Transformers ; si besoin, `python -m pip install transformers`. Le premier lancement peut télécharger les fichiers du tokenizer Qwen3. Il ne charge pas les poids du modèle.

Prédis si un identifiant technique, un emoji et un mot courant seront segmentés de la même manière. Observe ensuite sans supposer les IDs à l’avance.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
samples = [
    "chat", " chat", "Chat", "café", "🙂",
    "anticonstitutionnellement", "cybersécurité", "cybersecurity",
    "CVE-2026-12345", "Le chat dort sur le canapé.",
]

for text in samples:
    ids = tokenizer.encode(text, add_special_tokens=False)
    # split() est notre convention simple de comptage, pas une analyse linguistique.
    word_count = len(text.split())
    print("\nTexte :", repr(text))
    print("Unités internes :", tokenizer.convert_ids_to_tokens(ids))
    print("IDs :", ids)
    print("Tokens :", len(ids), "caractères :", len(text),
          "octets UTF-8 :", len(text.encode("utf-8")))
    print("Tokens/mot :", round(len(ids) / word_count, 3) if word_count else None)
    decoded = tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    print("Décodé :", repr(decoded), "identique :", decoded == text)

print("\nTokens spéciaux :", tokenizer.special_tokens_map)
print("Nombre d’entrées avec tokens ajoutés :", len(tokenizer))
```

### Lire les observations

`repr()` rend les espaces et caractères échappés plus visibles. Les chaînes renvoyées par `convert_ids_to_tokens()` peuvent utiliser une notation interne pour les espaces ou les octets : elles ne sont pas toujours des fragments directement lisibles.

Compare `chat`, ` chat` et `Chat`. Le nombre de tokens peut rester identique alors que les IDs changent. Il peut aussi changer : les frontières et les fréquences apprises comptent.

Le décodage de la **liste complète** est plus pertinent que la concaténation de décodages isolés, notamment quand plusieurs tokens correspondent ensemble à un caractère UTF-8.

## 9. Expériences guidées

Pour chaque expérience, note le texte exact, le tokenizer, les IDs et le nombre de tokens.

### A — Examiner les espaces

Compare `"chat"`, `" chat"`, `"chat "` et `"chat\n"`. Prédis ce qui sera visible avec `repr()`, puis examine les unités et le décodage. Un espace n’est pas une décoration gratuite : il peut changer la séquence d’entrée.

### B — Examiner ton domaine

Ajoute dix termes de ton domaine, avec nombres, noms composés ou code. Trie-les par nombre de tokens. Ne conclus pas qu’un mot très fragmenté est incompréhensible pour le réseau ; constate seulement son coût en positions.

### C — Comparer deux paragraphes

Choisis deux paragraphes comparables en français et en anglais, puis calcule tokens/mot et tokens/caractère. Indique ta convention de comptage et les limites de cette comparaison.

### D — Observer les tokens spéciaux

Compare un encodage avec `add_special_tokens=False` et un avec `True`. Certains tokenizers n’ajoutent rien dans cet appel : c’est un résultat possible. Compare ensuite avec un texte construit par `apply_chat_template` comme au module 1 ; le format conversationnel est une autre étape.

## 10. Exercices avec corrigés

### Exercice 1 — Compter les unités

Pourquoi `len("café")` et `len("café".encode("utf-8"))` ne valent-ils pas la même chose ? Lequel donne le nombre de tokens ?

<details>
<summary>Corrigé</summary>

Dans cette écriture, le premier compte 4 points de code et le second 5 octets UTF-8. Aucun ne donne le nombre de tokens : il faut utiliser le tokenizer choisi et compter les IDs produits.

</details>

### Exercice 2 — Refaire BPE

Avec nos deux règles `b+a → ba`, puis `ba+s → bas`, comment représenter `bas` et `basse` ? Le tokenizer réapprend-il ses règles pour chaque mot ?

<details>
<summary>Corrigé</summary>

`bas` devient `["bas"]` et `basse` devient `["bas", "s", "e"]`. Les règles sont appliquées telles quelles ; les apprendre à partir d’un corpus est une autre opération.

</details>

### Exercice 3 — Calculer la fertility

Un texte contient 80 mots selon `split()` et produit 120 tokens hors marqueurs spéciaux. Quel ratio obtiens-tu ? Que faut-il préciser dans un rapport ?

<details>
<summary>Corrigé</summary>

`120 / 80 = 1.5 token par mot`. Il faut préciser le tokenizer et sa version ou révision, le texte ou corpus, la convention `split()` et l’exclusion des marqueurs spéciaux.

</details>

### Exercice 4 — Comprendre un ajout de token

Tu ajoutes un token unique pour le nom d’une entreprise. Le modèle connaît-il immédiatement son activité ?

<details>
<summary>Corrigé</summary>

Non. Créer une unité et son ID ne fournit pas ses connaissances au réseau. Les paramètres associés doivent être compatibles et apprendre à exploiter cette unité. Le nom pouvait déjà être représenté par plusieurs sous-tokens auparavant.

</details>

### Exercice 5 — Expliquer un budget

Deux tokenizers encodent le même document en 900 et 1 200 tokens. Dans une fenêtre de 1 024 positions, que peux-tu conclure avant d’ajouter les marqueurs et une réponse ?

<details>
<summary>Corrigé</summary>

La première représentation laisse théoriquement 124 positions ; la seconde dépasse déjà la fenêtre de 176 positions. Il faut encore compter les marqueurs et la sortie prévue. Cela ne compare pas, à lui seul, la qualité des deux modèles.

</details>

## 11. Validation — avant de continuer

- [ ] Je distingue mot, caractère, octet, token et ID.
- [ ] Je peux refaire les deux fusions et retrouver `11 → 8 → 5`.
- [ ] Je sais qu’un tokenizer déjà appris ne réapprend pas pendant chaque encodage.
- [ ] Je peux expliquer le rôle possible de la normalisation et des marqueurs spéciaux.
- [ ] Je sais inspecter les IDs et décoder une liste complète.
- [ ] Je peux calculer un ratio de fragmentation en indiquant ses limites.
- [ ] Je peux expliquer pourquoi remplacer le tokenizer n’est pas une modification anodine.

## Fiche mémo

```text
Texte → normalisation éventuelle → pré-tokenisation
→ segmentation → tokens/IDs → marqueurs éventuels

BPE : apprendre puis appliquer des règles de fusion.
Encoder : texte vers IDs ; décoder : IDs vers texte.
Nombre de tokens : à mesurer avec le tokenizer réellement utilisé.
```

Le [module 3](03_fenetre_de_contexte_et_sequences.md) utilisera ces IDs pour préparer des séquences, du padding et des masques.

## Références

- [Hugging Face Tokenizers : pipeline](https://huggingface.co/docs/tokenizers/en/pipeline) — normalisation, pré-tokenisation, segmentation et post-traitement.
- [Hugging Face : résumé des tokenizers](https://huggingface.co/docs/transformers/en/tokenizer_summary) — familles de tokenisation, dont BPE et Unigram.
