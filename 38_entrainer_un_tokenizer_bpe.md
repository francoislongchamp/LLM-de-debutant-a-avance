# Module 38 — Entraîner et figer un tokenizer BPE

## 1. Ce que tu vas apprendre

Le tokenizer fixe les unités et leurs IDs. Une fois les embeddings entraînés, remplacer cette correspondance change le sens des lignes de poids.

Tu vas entraîner un BPE local, expliciter son alphabet initial, tester les caractères absents du corpus et sauvegarder une version rechargeable.

**Prérequis :** modules 2, 9, 37.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** `tokenizers` et `transformers`, CPU. Aucun modèle ou corpus externe.

## 2. Les décisions avant les fusions

| Choix | Question |
| --- | --- |
| Corpus de train | Représente-t-il les textes à traiter, sans utiliser le test pour les merges ? |
| Normalisation | Préserver casse, accents, espaces et formes Unicode ? |
| Pré-tokenisation | Quelles frontières et quelle représentation de départ ? |
| Alphabet initial | Comment encoder un caractère non observé ? |
| Tokens spéciaux | Quels IDs et quelle fonction dans le pipeline ? |
| Budget de vocabulaire | Quel coût de séquence contre quel coût d’embedding ? |

BPE fusionne des paires selon son apprentissage. **Byte-level** décrit ici la représentation basée sur les octets ; cela ne signifie pas que tous les tokens finaux sont des octets isolés.

Une configuration ByteLevel sans alphabet initial complet peut dépendre davantage des octets observés à l’apprentissage. Nous fournissons les 256 unités de départ pour couvrir les chaînes UTF-8, sous les règles de ce pipeline.

## 3. Refaire le compromis de vocabulaire

Un embedding de vocabulaire V et largeur H contient `V×H` nombres. Si sortie et embedding ne sont pas partagés, la tête peut ajouter une matrice de taille comparable.

Passer de V=32 000 à 64 000, avec H=768, ajoute **24 576 000** valeurs dans la table d’entrée seule. Avec une tête indépendante de même taille, le surcoût est encore augmenté.

Un grand vocabulaire peut compacter certains textes, mais il faut comparer le corpus réservé et les coûts réels. Il n’apprend pas à lui seul la signification des nouveaux tokens.

## 4. Tokens spéciaux : distinguer réservation et insertion

Nous réservons `<pad>`, `<bos>`, `<eos>` et `<unk>`. Leur présence dans le vocabulaire ne demande pas automatiquement de les ajouter à chaque séquence.

Le post-traitement ou le code d’assemblage doit décider où BOS/EOS apparaissent. Pour nos documents, nous ajouterons explicitement EOS quand l’objectif le demande.

La valeur de PAD n’est pas universellement zéro ; lis l’ID sauvegardé. Le modèle et le collator doivent connaître les mêmes IDs.

## 5. Laboratoire autonome — Entraîner, tester, recharger

Le corpus est minuscule et sert à valider le mécanisme. Les sorties pédagogiques sont écrites dans `outputs/module38/tokenizer/`.

Prédis : un emoji absent des trois phrases peut-il être encodé sans UNK grâce à notre alphabet ?

```python
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

corpus = [
    'Le chat dort. Le chat mange. Le chien dort.',
    'Un batch regroupe des exemples. Un gradient depend de la loss.',
    'Les tokens representent le texte. Le texte contient des mots et des espaces.',
]
special = ['<pad>', '<bos>', '<eos>', '<unk>']
backend = Tokenizer(BPE(unk_token='<unk>'))
backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
backend.decoder = ByteLevelDecoder()
trainer = BpeTrainer(vocab_size=320, min_frequency=1,
                     initial_alphabet=ByteLevel.alphabet(), special_tokens=special,
                     show_progress=False)
backend.train_from_iterator(corpus, trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend, pad_token='<pad>', bos_token='<bos>',
    eos_token='<eos>', unk_token='<unk>', model_max_length=128,
)
root = Path('outputs/module38/tokenizer')
root.mkdir(parents=True, exist_ok=True)
tokenizer.save_pretrained(root)
restored = PreTrainedTokenizerFast.from_pretrained(root)

for text in ['Bonjour café 🙂', 'code = 42\n', 'é', 'e\u0301']:
    ids = restored.encode(text, add_special_tokens=False)
    decoded = restored.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print(repr(text), 'tokens=', len(ids), 'round-trip=', decoded == text,
          'UNK=', restored.unk_token_id in ids)
    assert decoded == text
    assert restored.unk_token_id not in ids

manifest = {
    'requested_vocab': 320, 'actual_vocab': len(restored),
    'special_ids': {token: restored.convert_tokens_to_ids(token) for token in special},
    'corpus_sha256': hashlib.sha256(json.dumps(corpus, ensure_ascii=False).encode()).hexdigest(),
    'tokenizer_sha256': hashlib.sha256((root / 'tokenizer.json').read_bytes()).hexdigest(),
    'normalization': 'none', 'pre_tokenizer': 'ByteLevel(add_prefix_space=False)',
}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(manifest)
```

Les round-trips doivent être vrais et UNK absent sur ces tests. Les IDs spéciaux attendus sont 0,1,2,3 dans l’ordre réservé par ce trainer. Vérifie-les réellement au lieu de généraliser cet ordre à tous les tokenizers.

La taille demandée est une cible/budget ; selon le corpus et les contraintes, la taille obtenue peut différer. Compte `len(tokenizer)` pour dimensionner le modèle utilisé avec ce tokenizer.

## 6. Pourquoi les deux écritures de é comptent

Notre pipeline n’impose pas de normalisation Unicode. Le `é` précomposé et la suite `e` + accent combinant peuvent avoir des IDs différents tout en restant visuellement proches.

Une normalisation NFC pourrait les rapprocher ; un lowercasing ou une suppression d’accents ferait d’autres transformations. Ces choix peuvent modifier du code, des noms ou des données sensibles à la casse. Définis leur utilité au lieu de les appliquer comme un nettoyage universel.

Décoder les tokens un par un n’est pas toujours un test pertinent avec les octets : plusieurs unités peuvent former ensemble un caractère UTF-8. Vérifie la séquence complète.

## 7. Figer une version utilisable

Conserve corpus de train et groupes d’évaluation, code, versions de la bibliothèque, paramètres du trainer, fichiers du tokenizer, hash et IDs spéciaux.

Le hash identifie des octets ; il ne prouve pas la qualité ni les droits d’utilisation. La provenance du corpus et les décisions de nettoyage restent à documenter.

Après apprentissage des embeddings, changer le tokenizer exige une stratégie explicite de compatibilité et d’entraînement. « Refaire un meilleur vocabulaire » n’est pas un remplacement transparent.

## 8. Expériences guidées

1. Entraîne des budgets 280,320,400 sur le même corpus. Mesure les tailles réellement obtenues et le nombre de tokens d’un holdout fixe.
2. Retire l’alphabet complet dans une copie et teste des caractères absents. Ne suppose pas le comportement : observe UNK et round-trip.
3. Ajoute une normalisation NFC puis compare les deux écritures de é. Documente le changement voulu.
4. Recharge le tokenizer dans un nouveau processus et compare les mêmes IDs. La sauvegarde doit contenir les règles nécessaires, pas seulement une liste de mots.

## 9. Exercices avec corrigés

### Exercice 1 — Couverture

Pourquoi fournir un alphabet d’octets complet aide-t-il pour un emoji non présent au train ?

<details>
<summary>Corrigé</summary>

La chaîne UTF-8 reste décomposable en octets représentés dans les unités de départ. Cela n’impose pas une segmentation compacte ni une connaissance sémantique de l’emoji.

</details>

### Exercice 2 — Vocabulaire et matrice

Combien de valeurs dans V=16 000 et H=384 ?

<details>
<summary>Corrigé</summary>

**6 144 000** valeurs pour l’embedding d’entrée. Partage ou indépendance de la tête doivent être comptés séparément.

</details>

### Exercice 3 — EOS réservé

EOS dans le vocabulaire signifie-t-il qu’il est automatiquement présent à la fin de chaque document ?

<details>
<summary>Corrigé</summary>

Non. L’insertion dépend du post-traitement et du pipeline. Le laboratoire configure sa signification, puis les futurs scripts doivent décider où l’ajouter.

</details>

### Exercice 4 — Changement tardif

Pourquoi un nouveau mapping ID/token peut-il casser un modèle même si le vocabulaire garde la même taille ?

<details>
<summary>Corrigé</summary>

Les lignes d’embeddings et de sortie ont appris l’ancien mapping. Une dimension compatible n’assure pas une signification compatible.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue BPE, ByteLevel, alphabet et normalisation.
- [ ] Je peux entraîner puis recharger le tokenizer sans téléchargement.
- [ ] Je vérifie taille réelle, IDs spéciaux, UNK et round-trip.
- [ ] Je calcule le coût V×H.
- [ ] Je conserve fichiers, hashes, corpus et configuration.

## Fiche mémo

```text
Choix du corpus → alphabet/normalisation → merges → tests → sauvegarde figée.
Réserver un marqueur ne signifie pas l’insérer automatiquement.
```

Le [module 39](39_evaluer_un_tokenizer.md) compare l’efficacité sur des textes réservés.

## Référence

[Tokenizers : entraînement BPE](https://huggingface.co/docs/tokenizers/en/quicktour) — composants, trainer et sauvegarde.
