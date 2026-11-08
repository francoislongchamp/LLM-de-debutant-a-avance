# Module 39 — Évaluer la fidélité et le coût d’un tokenizer

## 1. Ce que tu vas apprendre

Un tokenizer entraîné sans erreur peut fragmenter excessivement certains domaines ou perdre une information que tu voulais préserver.

Tu vas comparer deux budgets sur le même holdout, calculer des métriques par catégorie et examiner les textes les plus coûteux.

**Prérequis :** modules 2 et 38 ; comptage de tokens et round-trip.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et Tokenizers, CPU ; le script entraîne ses propres tokenizers.

## 2. Choisir des unités explicites

| Mesure | Calcul | Limite |
| --- | --- | --- |
| Fertility | tokens / mots selon une convention | Le mot n’est pas défini uniformément entre langues |
| Bytes/token | octets UTF-8 / tokens | Compression de représentation, pas taille exacte d’un fichier |
| Longueur médiane/P95 | Distribution des nombres de tokens par texte | Dépend des longueurs et du corpus |
| Round-trip | decode(encode(texte)) comparé à la règle attendue | Une normalisation voulue change la référence |
| UNK | Occurrences du marqueur inconnu | Zéro UNK ne garantit pas une bonne segmentation |
| Utilisation du vocabulaire | IDs distincts observés / vocabulaire | Un token rare n’est pas nécessairement inutile |

Nous excluons les marqueurs ajoutés du nombre de tokens ordinaires. Pour les mots, nous utilisons `split()`, convention simple qui doit rester annoncée.

## 3. Moyenne globale ou par texte ?

Pour 10 mots donnant 20 tokens et 100 mots donnant 100 tokens :

```text
Ratio corpus : 120/110 ≈ 1.091 token/mot
Moyenne des deux ratios : (2+1)/2 = 1.5
```

Les deux nombres pondèrent différemment les textes. Pour une question de budget de corpus, le ratio des sommes est souvent plus directement interprétable ; annonce ton choix.

De même, un P95 sur cinq documents est un indicateur très grossier, parfois égal au maximum. Il ne devient pas robuste parce qu’on lui donne un nom de percentile.

## 4. Laboratoire autonome — Deux BPE, un holdout catégorisé

Le corpus d’entraînement n’est pas le holdout. Les quelques chaînes réservées servent à apprendre le protocole, pas à classer scientifiquement toutes les langues.

```python
import math
import statistics
from collections import defaultdict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteDecoder

train_texts = ['Le chat dort. Le chien mange.',
               'Un modele traite des tokens. Le modele apprend des regularites.',
               'Un gradient mesure une variation. Les donnees guident le modele.']
holdout = [
    ('fr', 'Le réseau traite un café et un emoji 🙂.'),
    ('fr', 'Une question différente demande une explication précise.'),
    ('code', 'def add(x, y):\n    return x + y\n'),
    ('code', 'https://example.test/a?id=12345'),
    ('en', 'A different sentence tests the tokenizer.'),
]


def train_tokenizer(size):
    tokenizer = Tokenizer(BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteDecoder()
    tokenizer.train_from_iterator(train_texts, BpeTrainer(
        vocab_size=size, min_frequency=1, initial_alphabet=ByteLevel.alphabet(),
        special_tokens=['<pad>', '<eos>', '<unk>'], show_progress=False))
    return tokenizer


def report(tokenizer):
    groups = defaultdict(list)
    used = set()
    expensive = []
    for category, text in holdout:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        assert tokenizer.decode(ids, skip_special_tokens=False) == text
        used.update(ids)
        groups[category].append((len(ids), len(text.split()), len(text.encode('utf-8'))))
        expensive.append((len(ids), category, text))
    for category, values in groups.items():
        lengths = sorted(value[0] for value in values)
        words = sum(value[1] for value in values)
        total = sum(lengths)
        p95 = lengths[math.ceil(0.95 * len(lengths)) - 1]  # nearest rank
        print(category, 'n=', len(values), 'tokens/mot=', total / words,
              'bytes/token=', sum(value[2] for value in values) / total,
              'mediane=', statistics.median(lengths), 'P95=', p95)
    print('IDs observes / vocabulaire :', len(used), '/', tokenizer.get_vocab_size())
    print('Textes les plus couteux :', sorted(expensive, reverse=True)[:2])


for size in [280, 340]:
    tokenizer = train_tokenizer(size)
    print('\nBudget=', size, 'taille reelle=', tokenizer.get_vocab_size())
    report(tokenizer)
```

Les round-trips doivent réussir pour ces chaînes. Les métriques numériques dépendront des merges réellement appris ; observe où le plus grand vocabulaire aide et où il apporte peu.

Le P95 utilise ici la convention *nearest rank*. Dans les catégories à un ou deux exemples, il correspond au maximum. Le code l’affiche pour apprendre le calcul, pas pour prétendre à une estimation précise des queues.

## 5. Interpréter une meilleure compacité

Un tokenizer plus compact peut laisser davantage de texte dans la même fenêtre. Il peut aussi augmenter le coût des embeddings et du LM head.

À architecture égale hors vocabulaire, le meilleur ratio tokens/mot n’est donc pas automatiquement le meilleur compromis temps/mémoire/qualité. À l’inverse, une bonne couverture par octets ne rend pas la fragmentation gratuite.

Une différence de langue peut refléter corpus d’apprentissage, conventions et types de documents. Compare des échantillons représentatifs de l’usage, sans généraliser depuis deux traductions.

## 6. Ne pas choisir sur le test final

Utilise un corpus de développement pour régler normalisation, vocabulaire et pré-tokenisation. Garde un autre échantillon pour l’estimation finale si tu multiplies les choix.

Le tokenizer est un composant appris : ses données et sa sélection font partie du protocole. Après chaque choix, conserve la configuration et les résultats plutôt que de ne garder que le vainqueur.

## 7. Expériences guidées

1. Ajoute nombres, identifiants, caractères combinants, tabulations et plusieurs langues importantes pour ton usage.
2. Calcule ratio des sommes et moyenne des ratios ; explique les différences de pondération.
3. Liste les textes les plus longs en tokens, puis inspecte ce qui provoque leur fragmentation.
4. Compare le surcoût V×H des deux tailles et le gain de positions sur ton corpus réservé.

## 8. Exercices avec corrigés

### Exercice 1 — Ratio

50 mots et 75 tokens hors marqueurs donnent quelle fertility ?

<details>
<summary>Corrigé</summary>

**1.5 token/mot**, sous la convention de mots retenue. Il faut annoncer tokenizer, corpus et inclusion des marqueurs.

</details>

### Exercice 2 — UNK

Aucun UNK dans un texte signifie-t-il que son encodage est compact ?

<details>
<summary>Corrigé</summary>

Non. Un alphabet d’octets peut représenter le texte sans UNK tout en produisant de nombreuses unités.

</details>

### Exercice 3 — P95 minuscule

Avec deux longueurs [10,100], que donne le nearest-rank P95 ?

<details>
<summary>Corrigé</summary>

**100**, car ceil(0.95×2)=2. Cela reste un très petit échantillon, pas une description précise d’une distribution future.

</details>

### Exercice 4 — Choix final

Pourquoi ne pas choisir uniquement la taille qui minimise le nombre de tokens ?

<details>
<summary>Corrigé</summary>

Elle peut coûter davantage en paramètres et calcul de sortie. Il faut aussi fidélité, couverture, coûts et résultats du modèle dans un protocole compatible.

</details>

## 9. Validation — avant de continuer

- [ ] Je définis unités, marqueurs et convention de percentile.
- [ ] Je compare les mêmes textes et catégories.
- [ ] Je vérifie le round-trip selon la normalisation prévue.
- [ ] Je distingue ratio corpus et moyenne des ratios.
- [ ] Je prends une décision qui inclut compacité et coût du vocabulaire.

## Fiche mémo

```text
Évaluer fidélité, fragmentation, queues de longueur et coût V×H.
Un benchmark de tokenizer n’est pas un benchmark de qualité linguistique du modèle.
```

Le [module 40](40_concevoir_un_petit_transformer.md) dimensionne le réseau qui consommera ces IDs.
