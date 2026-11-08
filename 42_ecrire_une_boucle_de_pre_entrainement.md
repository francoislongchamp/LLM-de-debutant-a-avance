# Module 42 — Écrire une boucle causale complète et comptable

## 1. Ce que tu vas apprendre

La boucle du module 8 apprend maintenant des prochains tokens. La difficulté supplémentaire vient notamment des labels masqués, des nombres de cibles variables et des compteurs.

Tu vas assembler un modèle aléatoire, un collator, une accumulation pondérée, un scheduler, une validation et une sauvegarde.

**Prérequis :** modules 6 à 8, 22 à 24 et 37 à 41.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch et Transformers sur CPU. Les IDs du pilote sont fictifs et déjà préparés ; le projet 62 intégrera le tokenizer.

## 2. Le contrat d’un batch causal

| Champ | Forme | Règle |
| --- | --- | --- |
| `input_ids` | [B,T] | IDs valides du vocabulaire |
| `attention_mask` | [B,T] | Contenu réel ou padding selon cette API |
| `labels` | [B,T] | Copie des cibles, PAD remplacé par −100 |

Le modèle GPT-2 choisi effectue le décalage interne de loss. Le nombre de cibles effectivement évaluées est donc `count(labels[:,1:] != -100)`.

Les labels gardent les tokens du document, EOS compris. Nous ne faisons pas une loss assistant-only : l’objectif est le langage causal sur les documents préparés.

## 3. Une fenêtre d’accumulation, un dénominateur

Avec des micro-batches contenant n1 et n2 cibles, leur moyenne globale est :

\[
L = \frac{n_1L_1+n_2L_2}{n_1+n_2}
\]

Nous pondérons chaque backward par son nombre de cibles sur la fenêtre réelle. Cela traite aussi le dernier groupe incomplet.

Dans ce petit pilote, on connaît la fenêtre en prenant quelques micro-batches avant leurs forwards. On ne conserve pas tous leurs graphes : les backward sont réalisés un par un.

## 4. Laboratoire autonome — Pipeline complet sur cinq séquences

Le script inclut un test de mémorisation d’un batch avec une **copie** du modèle. Le vrai pilote repart ensuite de l’état initial, pas des poids de ce test.

Les données sont artificielles et trop petites pour estimer une compétence de langage. Les sorties pédagogiques sont sauvegardées sous `outputs/module42/`.

```python
from copy import deepcopy
from itertools import islice
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup

torch.manual_seed(42)
train_docs = [[2,3,4,1], [2,3,5,1], [3,4,6,1], [4,5,6,1], [5,6,1]]
val_docs = [[2,4,6,1], [3,5,1]]


def collate(docs):
    length = max(map(len, docs))
    ids = torch.tensor([doc + [0] * (length - len(doc)) for doc in docs])
    mask = ids.ne(0).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def targets(batch):
    return int((batch['labels'][:, 1:] != -100).sum().item())


def evaluate(model, loader):
    model.eval()
    total_nll = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            n = targets(batch)
            if n:
                total_nll += model(**batch).loss.item() * n
                count += n
    if count == 0:
        raise ValueError('Validation sans cibles')
    return total_nll / count


model = GPT2LMHeadModel(GPT2Config(
    vocab_size=8, n_positions=16, n_embd=24, n_layer=1, n_head=2,
    bos_token_id=None, eos_token_id=1, pad_token_id=0, use_cache=False,
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
))
smoke = deepcopy(model)
smoke_optimizer = torch.optim.AdamW(smoke.parameters(), lr=0.01, weight_decay=0.0)
fixed_batch = collate([train_docs[0]])
smoke_before = evaluate(smoke, [fixed_batch])
for _ in range(40):
    smoke.train()
    smoke_optimizer.zero_grad(set_to_none=True)
    smoke(**fixed_batch).loss.backward()
    smoke_optimizer.step()
print('Test un batch avant/apres :', smoke_before, evaluate(smoke, [fixed_batch]))
del smoke, smoke_optimizer

train_loader = DataLoader(train_docs, batch_size=2, shuffle=True,
                          generator=torch.Generator().manual_seed(42), collate_fn=collate)
val_loader = DataLoader(val_docs, batch_size=2, collate_fn=collate)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=12)
global_step = seen_inputs = seen_targets = 0
print('Validation initiale :', evaluate(model, val_loader))

for epoch in range(1, 7):
    model.train()
    iterator = iter(train_loader)
    while True:
        window = list(islice(iterator, 2))
        if not window:
            break
        n_total = sum(targets(batch) for batch in window)
        if n_total == 0:
            raise ValueError('Fenetre sans cibles')
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            n = targets(batch)
            loss = model(**batch).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError('Loss non finie')
            (loss * n / n_total).backward()
            seen_inputs += int(batch['attention_mask'].sum().item())
            seen_targets += n
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError('Gradient non fini')
        used_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        global_step += 1
    print('epoch=', epoch, 'updates=', global_step, 'cibles=', seen_targets,
          'validation=', evaluate(model, val_loader), 'dernier LR utilise=', used_lr)

root = Path('outputs/module42')
root.mkdir(parents=True, exist_ok=True)
torch.save({'model': model.state_dict(), 'config': model.config.to_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'step': global_step, 'seen_inputs': seen_inputs, 'seen_targets': seen_targets,
            'rng': torch.get_rng_state()}, root / 'checkpoint.pt')
print('Compteurs finaux :', global_step, seen_inputs, seen_targets)
```

## 5. Retrouver les compteurs à la main

Les cinq documents contiennent **19 IDs réels**, dont **14 cibles après décalage**, par epoch.

Chaque epoch a trois micro-batches, regroupés en deux fenêtres : une fenêtre de deux micro-batches et une fenêtre d’un seul. Sur six epochs :

```text
12 optimizer steps
114 positions d’entrée réelles traitées
84 cibles présentées à la loss
```

Les répétitions ne créent pas 84 cibles distinctes de corpus. Le test préalable de mémorisation est une expérience séparée, exclue de ces compteurs du pilote.

La loss du batch fixe doit pouvoir fortement baisser. La validation, elle, peut évoluer autrement ; sa petite taille ne permet pas une conclusion générale.

## 6. Ce que la sauvegarde couvre et ne couvre pas

Le fichier conserve poids, optimizer, scheduler, config, RNG et compteurs. Il est enregistré à la fin d’un parcours d’epoch.

Une reprise exacte d’un pipeline réel doit aussi restaurer le générateur du DataLoader, la position du sampler/flux, les RNG supplémentaires, la précision mixte et les états distribués concernés. Notre fichier ne prétend donc pas résoudre toutes les reprises : le module 51 réalise un test contrôlé complet sur un cas défini.

Les IDs du pilote ont une signification fictive définie par le code. Pour un texte réel, sauvegarde le tokenizer et ses IDs spéciaux avec la configuration, comme dans le projet 62.

## 7. Expériences guidées

1. Mets toutes les cibles d’un batch à −100 dans une copie. Vérifie que le contrôle de comptage détecte un problème avant d’interpréter un NaN.
2. Change micro-batch et accumulation en gardant la moyenne logique voulue. Recalcule les fenêtres et steps avant le run.
3. Ajoute une séquence plus longue puis vérifie compteurs, padding et budget `n_positions`.
4. Change la convention de compteur pour afficher aussi les cases PAD traitées. Explique pourquoi cette valeur diffère des entrées réelles et des cibles.

## 8. Exercices avec corrigés

### Exercice 1 — Shift

Une ligne de quatre IDs réels, EOS compris, fournit combien de cibles dans ce modèle sans autre cible initiale ?

<details>
<summary>Corrigé</summary>

**Trois**, car la sortie de chaque position prédit le token suivant et le premier label de la ligne n’a pas de sortie précédente fournie ici.

</details>

### Exercice 2 — Moyenne

Deux micro-batches ont 3 et 9 cibles avec losses 2 et 1. Quelle moyenne ?

<details>
<summary>Corrigé</summary>

`(3×2+9×1)/12 = 1.25`. La moyenne simple des deux losses donnerait 1.5 et pondérerait mal les cibles.

</details>

### Exercice 3 — Validation après évaluation

Pourquoi appelle-t-on `model.train()` à chaque nouvelle epoch ?

<details>
<summary>Corrigé</summary>

La fonction de validation a mis le réseau en mode eval. L’apprentissage doit réactiver le comportement train des couches concernées avant de reprendre les updates.

</details>

### Exercice 4 — Smoke test

Pourquoi ne pas utiliser les poids qui ont déjà mémorisé le batch pour prétendre repartir d’un modèle aléatoire ?

<details>
<summary>Corrigé</summary>

Ils ont reçu des updates. Il faut une copie ou une restauration de l’état initial pour distinguer diagnostic et entraînement annoncé.

</details>

## 9. Validation — avant de continuer

- [ ] Je lis et construis les trois champs d’un batch.
- [ ] Je compte les cibles après le décalage du modèle.
- [ ] Je retrouve 12/114/84 pour le pilote.
- [ ] Je traite les fenêtres incomplètes avec le bon dénominateur.
- [ ] Je distingue smoke test, validation et état de reprise complet.

## Fiche mémo

```text
Compter ce que voit l’entrée, ce qu’évalue la loss et ce qui déclenche un step.
Les cibles actives gouvernent leur moyenne ; les checkpoints doivent être restaurables.
```

Le [module 43](43_lire_les_courbes_de_loss.md) interprète les métriques de cette boucle.
