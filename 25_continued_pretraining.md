# Module 25 — Continuer l’objectif de langage sur un domaine

## 1. Ce que tu vas apprendre

Le **continued pretraining**, ou CPT, reprend un modèle déjà entraîné et poursuit un objectif de langage sur un corpus choisi. Il peut adapter la modélisation d’un domaine sans fournir des démonstrations question/réponse.

Tu vas préparer des documents en séquences, respecter leurs frontières, mesurer les tokens vus et effectuer une continuation miniature depuis un premier état appris.

**Prérequis :** modules 6, 11, 12 et 21 à 24.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch et Transformers sur CPU pour la miniature. Pas de téléchargement de modèle.

## 2. Distinguer départ, données et mise à jour

| Terme | Ce qu’il précise |
| --- | --- |
| Pré-entraînement from scratch | Départ aléatoire et apprentissage initial |
| CPT | Continuation d’un objectif de pré-entraînement à partir d’un état déjà appris |
| Adaptation de domaine | Corpus et évaluation centrés sur une distribution particulière |
| SFT | Démonstrations de réponses ou comportements attendus |
| Full FT ou PEFT | Paramètres autorisés à changer |

Un CPT peut modifier tous les poids ou utiliser une adaptation compatible. SFT et CPT peuvent partager la cross-entropy causale ; leur différence vient notamment des données, du format et des cibles supervisées.

« Texte brut » signifie ici texte documentaire, pas données laissées sales ou sans provenance.

## 3. Que peut apprendre l’objectif ?

Le modèle est encouragé à attribuer davantage de probabilité aux suites observées dans le corpus. Cela peut concerner vocabulaire, style, syntaxe et relations présentes dans les documents.

Ce n’est pas une base documentaire consultable de façon garantie : apprendre sur un fait ne garantit ni sa restitution exacte ni une citation traçable. Une loss de domaine plus basse ne prouve pas une meilleure réponse à une instruction.

Après CPT, un SFT peut être utile pour des comportements, mais l’enchaînement n’est pas obligatoire. Chaque étape doit répondre à un problème mesuré.

## 4. Préparer les séquences sans perdre le compte

Une chaîne de préparation possible :

```text
Sources → nettoyage/provenance → groupes train/validation
→ tokenisation → frontières de document → séquences/blocs → labels causaux
```

Sépare les sources avant de fabriquer des blocs voisins. Un split aléatoire de blocs issus du même document peut affaiblir l’évaluation.

### Frontières et blocs

Ajouter EOS marque une fin, mais ne crée pas à lui seul un masque qui interdit toute attention vers le document précédent si on concatène plusieurs documents dans une séquence.

Trois choix à documenter : conserver les documents séparés avec padding, concaténer avec EOS et contexte partagé, ou utiliser une stratégie de packing avec masques/positions adaptés.

Si tu découpes un flux en blocs non chevauchants, les prédictions à certaines frontières changent ou disparaissent. Si tu abandonnes systématiquement le reste de chaque petit lot de tokenisation, tu peux perdre davantage de données que prévu.

Notre laboratoire garde chaque document dans une ligne séparée : il rend les comptes simples sans prétendre être un pipeline de corpus massif.

## 5. Laboratoire autonome — Un premier état, puis une continuation

Le modèle est minuscule. Nous lui faisons d’abord apprendre quelques séquences générales, puis poursuivons sur des séquences de domaine. Cette expérience reproduit la **succession des états**, pas la qualité d’un LLM préentraîné à grande échelle.

Les données et le tokenizer par espaces sont artificiels ; les petites variantes de validation ne constituent pas un benchmark indépendant robuste.

```python
from copy import deepcopy
import torch
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(42)
words = ['<pad>', '<eos>', 'le', 'chat', 'dort', 'mange', '.',
         'dns', 'relie', 'nom', 'adresse', 'tls', 'protege', 'flux', 'reseau', 'machine']
vocab = {word: index for index, word in enumerate(words)}


def batch(texts):
    sequences = [[vocab[word] for word in text.split()] + [vocab['<eos>']] for text in texts]
    length = max(map(len, sequences))
    ids = torch.tensor([sequence + [0] * (length - len(sequence)) for sequence in sequences])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def fit(model, data, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**data).loss
        loss.backward()
        optimizer.step()


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        return model(**data).loss.item()


general_train = batch(['le chat dort .', 'le chat mange .'])
domain_train = batch(['dns relie nom adresse .', 'tls protege flux reseau .'])
domain_validation = batch(['dns relie machine adresse .', 'tls protege flux .'])
model = GPT2LMHeadModel(GPT2Config(
    vocab_size=len(vocab), n_positions=32, n_embd=24, n_layer=1, n_head=2,
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    bos_token_id=None, eos_token_id=1, pad_token_id=0, use_cache=False,
))
fit(model, general_train, steps=20)
initial_pretrained = deepcopy(model.state_dict())
before_domain = evaluate(model, domain_validation)
before_general = evaluate(model, general_train)
fit(model, domain_train, steps=20)
after_domain = evaluate(model, domain_validation)
after_general = evaluate(model, general_train)
active_targets_per_step = (domain_train['labels'][:, 1:] != -100).sum().item()
print('Loss domaine avant/apres :', before_domain, after_domain)
print('Loss ancien train general avant/apres :', before_general, after_general)
print('Cibles de domaine par step :', active_targets_per_step)
print('Expositions de cibles de domaine :', 20 * active_targets_per_step)
print('Au moins un poids modifie pendant la continuation :',
      any(not torch.equal(value, initial_pretrained[name]) for name, value in model.state_dict().items()))
```

Chaque document domaine contient ici six IDs, EOS compris, donc cinq cibles après décalage. Avec deux documents et vingt steps : **10 cibles/step**, soit **200 expositions de cibles**.

Le nombre de tokens d’entrée traités serait une autre convention : ici 12 positions réelles par step, soit 240. Annonce ce que ton « budget de tokens » compte, surtout quand il y a padding, masques ou répétitions.

La mesure générale porte explicitement sur **l’ancien train**, pas sur un benchmark de généralisation. Elle montre seulement un éventuel déplacement de la fonction. Une vraie évaluation de conservation des capacités doit utiliser des exemples réservés.

## 6. Transposer à un vrai CPT

Remplace le petit état initial par une base préentraînée identifiée et son tokenizer. Conserve sa révision et une baseline avant toute update.

Mesure documents, tokens uniques après nettoyage, expositions et catégories/sources. Dix millions de tokens répétés dix fois correspondent à environ cent millions d’expositions, pas à cent millions de tokens distincts.

Réserve une validation de domaine par source ou groupe pertinent et un benchmark général indépendant. Compare loss/perplexité de domaine dans des conditions compatibles, mais aussi tâches concrètes et comportements conversationnels si tu pars d’un modèle instruct.

Dans le laboratoire, nous recréons un optimizer pour la phase domaine : c’est une continuation depuis des **poids** appris, pas une reprise exacte des états d’optimizer de la première phase. Pour reprendre exactement un run, restaure les états supplémentaires.

## 7. Expériences guidées

1. Change seulement le nombre de steps domaine et compte les expositions. Compare les deux losses ; n’assimile pas davantage de répétitions à davantage de diversité.
2. Ajoute un troisième document, puis vérifie EOS, labels PAD et nombre de cibles après décalage.
3. Découpe un document long en blocs dans une copie. Affiche chaque bloc décodé, les frontières et les tokens perdus.
4. Sur un projet réel, compare base→SFT et base→CPT→même SFT avec un benchmark fixé, en annonçant le budget supplémentaire de CPT.

## 8. Exercices avec corrigés

### Exercice 1 — Même loss, autre signal

Pourquoi CPT et SFT peuvent-ils partager une formule de cross-entropy sans être la même expérience ?

<details>
<summary>Corrigé</summary>

Le corpus, le contexte, le format et les tokens cibles diffèrent. Des documents montrent des suites de langage ; des démonstrations SFT montrent des sorties attendues sous instructions.

</details>

### Exercice 2 — Répétitions

Un corpus de 2M tokens est parcouru trois fois. Combien d’expositions approximatives, hors troncature/padding ? Combien de tokens distincts garantis ?

<details>
<summary>Corrigé</summary>

Environ **6M expositions**. Cela ne crée pas 6M tokens de contenu distinct : le corpus reste celui de 2M, qui peut déjà contenir des répétitions internes.

</details>

### Exercice 3 — EOS et visibilité

Deux documents sont concaténés avec EOS. Le second ne peut-il plus voir le premier ?

<details>
<summary>Corrigé</summary>

Pas nécessairement. EOS est un marqueur ; un masque causal ordinaire permet encore de voir les positions précédentes. Il faut une stratégie de visibilité explicite si l’on veut des segments indépendants.

</details>

### Exercice 4 — Répondre ou citer

Une loss domaine baisse. Le modèle peut-il désormais citer de façon garantie la source de chaque fait appris ?

<details>
<summary>Corrigé</summary>

Non. L’objectif optimise les probabilités des suites, pas un index de sources récupérables avec garantie. Il faut mesurer les tâches voulues et envisager une récupération documentaire si la traçabilité est nécessaire.

</details>

## 9. Validation — avant de continuer

- [ ] Je distingue CPT, SFT, full FT et départ aléatoire.
- [ ] Je prépare documents, EOS, padding et labels causalement alignés.
- [ ] Je retrouve 10 cibles/step et 200 expositions dans le laboratoire.
- [ ] Je distingue contenu distinct, entrée traitée et cibles évaluées.
- [ ] Je sais pourquoi la validation doit être séparée avant les blocs.
- [ ] Je mesure domaine et conservation des capacités sans promettre une acquisition garantie.

## Fiche mémo

```text
CPT : poursuivre un objectif de langage depuis des poids déjà appris.
Choisir et nettoyer le corpus ; contrôler frontières, pertes et répétitions.
Évaluer la distribution de domaine ET les capacités à conserver.
```

Le [module 26](26_cpt_vs_sft_savoir_choisir.md) apprend à décider si CPT, SFT ou une autre intervention répond au problème observé.
