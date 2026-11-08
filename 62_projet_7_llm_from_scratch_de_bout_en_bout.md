# Module 62 — Projet final : construire et vérifier toute la chaîne

## 1. Ce que tu vas construire

Le projet relie corpus, tokenizer BPE, modèle aléatoire, loss causale, checkpoint/reprise, SFT et génération. Tu contrôles la signification des IDs et les transitions entre états.

Le premier livrable est une miniature locale exécutable. Le second, si tu veux aller plus loin, est une expérience à plus grande échelle avec données et évaluation adaptées à ton objectif.

**Prérequis :** modules 37 à 55 ; modules 9 à 14 pour les données et le SFT. DPO/RL restent des extensions motivées, pas des obligations.

**Temps indicatif :** pilote court sur CPU, lecture et analyse en plusieurs séances ; le projet cible dépend du corpus, du matériel et du budget choisi.

**Matériel :** PyTorch, Tokenizers et Transformers. La miniature n’utilise aucun checkpoint préentraîné ni téléchargement.

## 2. Fixer ce que le projet doit démontrer

Une bonne hypothèse pédagogique : « Je peux reconstruire les composants, apprendre une distribution simple, reprendre le même training contrôlé et superviser seulement les réponses. »

Une hypothèse de qualité linguistique demande en plus un corpus représentatif, des groupes d’évaluation suffisants et des tâches concrètes. Les six phrases du pilote ne permettent pas de revendiquer un assistant général.

Définis avant le run : architecture, contexte, tokens/cibles, temps, mémoire, critères d’arrêt, budget de tuning et limites de régression.

## 3. Les portes à franchir

| Étape | Preuve demandée |
| --- | --- |
| Données | Sources, groupes, nettoyage et comptes |
| Tokenizer | Train réservé, couverture, round-trip et fichiers stables |
| Modèle | Dimensions compatibles et paramètres comptés |
| Boucle | Cibles et gradients finis, apprentissage d’un batch possible |
| Reprise | Trajectoire contrôlée restaurée selon le contrat |
| Base | Mesures de langage et générations archivées |
| SFT | Masque prompt/réponse vérifié, base conservée séparément |
| Résultat | Reload, benchmark, coûts et limites documentés |

Une porte technique réussie n’est pas une preuve de qualité d’usage. Une expérience négative sur validation reste à rapporter.

## 4. Format du pilote

Le BPE est appris seulement sur les phrases de train, avec alphabet d’octets complet. Le modèle GPT-2 utilise de nouveaux poids aléatoires et une fenêtre de 128 positions.

La convention conversationnelle locale est `<q>question<a>réponse<eos>`. Les marqueurs sont réservés avant le pré-entraînement et sauvegardés avec le tokenizer. Le template du pilote gère uniquement user/assistant ; il ne prétend pas être celui de Qwen ni un assistant-only template pour tous les trainers.

Le SFT masque explicitement le préfixe dans les labels. Le contrôle de frontière évite de supposer que deux encodages séparés correspondent toujours à l’encodage complet.

## 5. Laboratoire autonome — Du BPE à une génération après SFT

Copie le bloc entier dans un fichier Python. Il écrit sous `outputs/project62/` et peut remplacer ses sorties pédagogiques lors d’une nouvelle exécution.

Le pré-entraînement utilise un batch fixe pour rendre la reprise parfaitement contrôlable. Il n’y a pas de sampler externe, de CUDA ou de précision mixte à restaurer dans ce contrat miniature.

```python
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteDecoder
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel

torch.manual_seed(42)
root = Path('outputs/project62')
root.mkdir(parents=True, exist_ok=True)
train_texts = [
    'Un batch regroupe plusieurs exemples.',
    'Un gradient mesure une variation de la loss.',
    'Le modele lit des tokens et predit la suite.',
    'Une validation mesure le modele sans mise a jour.',
    'Un optimiseur modifie les parametres avec des gradients.',
    'Le contexte contient les tokens precedents.',
]
validation_texts = ['Un modele apprend sur des exemples.',
                    'Les tokens precedents donnent du contexte.']
backend = Tokenizer(BPE(unk_token='<unk>'))
backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
backend.decoder = ByteDecoder()
backend.train_from_iterator(train_texts, BpeTrainer(
    vocab_size=320, min_frequency=1, initial_alphabet=ByteLevel.alphabet(),
    special_tokens=['<pad>', '<eos>', '<unk>', '<q>', '<a>'], show_progress=False))
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
                                   eos_token='<eos>', unk_token='<unk>',
                                   additional_special_tokens=['<q>', '<a>'], model_max_length=128)
tokenizer.chat_template = ("{% for m in messages %}{% if m['role'] == 'user' %}{{ '<q>' + m['content'] }}"
                           "{% elif m['role'] == 'assistant' %}{{ '<a>' + m['content'] + eos_token }}"
                           "{% else %}{{ raise_exception('Roles user/assistant seulement') }}{% endif %}{% endfor %}"
                           "{% if add_generation_prompt %}{{ '<a>' }}{% endif %}")
tokenizer.save_pretrained(root / 'tokenizer')
tokenizer_hash = hashlib.sha256((root / 'tokenizer' / 'tokenizer.json').read_bytes()).hexdigest()


def make_batch(rows):
    sequences, labels = [], []
    for row in rows:
        prefix = tokenizer.encode(row['prefix'], add_special_tokens=False)
        full = tokenizer.encode(row['prefix'] + row['completion'], add_special_tokens=False) + [tokenizer.eos_token_id]
        if full[:len(prefix)] != prefix or len(full) > 128:
            raise ValueError('Frontiere ou longueur incompatible')
        sequences.append(full)
        labels.append([-100] * len(prefix) + full[len(prefix):])
    length = max(map(len, sequences))
    return {'input_ids': torch.tensor([ids + [tokenizer.pad_token_id] * (length - len(ids)) for ids in sequences]),
            'attention_mask': torch.tensor([[1] * len(ids) + [0] * (length - len(ids)) for ids in sequences]),
            'labels': torch.tensor([target + [-100] * (length - len(target)) for target in labels])}


def measure(model, batch):
    model.eval()
    with torch.no_grad():
        return model(**batch).loss.item()


def create_optimizer(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return optimizer, scheduler


def update(model, optimizer, scheduler, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    if not torch.isfinite(loss).item():
        raise RuntimeError('Loss non finie')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    return loss.item()


pretrain = make_batch([{'prefix': '', 'completion': text} for text in train_texts])
validation = make_batch([{'prefix': '', 'completion': text} for text in validation_texts])
config = GPT2Config(vocab_size=len(tokenizer), n_positions=128, n_embd=32,
                    n_layer=2, n_head=4, bos_token_id=None,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False)
model = GPT2LMHeadModel(config)
optimizer, scheduler = create_optimizer(model)
initial_train = measure(model, pretrain)
initial_validation = measure(model, validation)
for step in range(1, 21):
    update(model, optimizer, scheduler, pretrain)
    if step == 10:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng': torch.get_rng_state(),
                    'step': step, 'tokenizer_hash': tokenizer_hash}, root / 'checkpoint-pretrain-10.pt')

base_train = measure(model, pretrain)
base_validation = measure(model, validation)
model.save_pretrained(root / 'base')
resumed = GPT2LMHeadModel(config)
resumed_optimizer, resumed_scheduler = create_optimizer(resumed)
state = torch.load(root / 'checkpoint-pretrain-10.pt', weights_only=True)
assert state['tokenizer_hash'] == tokenizer_hash
resumed.load_state_dict(state['model'])
resumed_optimizer.load_state_dict(state['optimizer'])
resumed_scheduler.load_state_dict(state['scheduler'])
torch.set_rng_state(state['rng'])
for step in range(state['step'] + 1, 21):
    update(resumed, resumed_optimizer, resumed_scheduler, pretrain)
resume_equal = all(torch.equal(value, resumed.state_dict()[name]) for name, value in model.state_dict().items())
assert resume_equal

sft_rows = [{'prefix': '<q>Qu est ce qu un batch ?<a>', 'completion': train_texts[0]},
            {'prefix': '<q>Qu est ce qu un gradient ?<a>', 'completion': train_texts[1]}]
sft_validation_rows = [{'prefix': '<q>Explique un batch.<a>', 'completion': 'Un batch regroupe des exemples.'},
                       {'prefix': '<q>Explique un gradient.<a>', 'completion': 'Un gradient mesure une variation.'}]
sft_batch = make_batch(sft_rows)
sft_validation = make_batch(sft_validation_rows)
before_sft = measure(model, sft_validation)
sft_optimizer, sft_scheduler = create_optimizer(model)
for _ in range(20):
    update(model, sft_optimizer, sft_scheduler, sft_batch)
after_sft = measure(model, sft_validation)
model.save_pretrained(root / 'instruct')
model.eval()
prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Explique un batch.'}],
                                        tokenize=False, add_generation_prompt=True)
assert prompt == sft_validation_rows[0]['prefix']
inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
answer = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
report = {'parameters': sum(p.numel() for p in model.parameters()), 'tokenizer_hash': tokenizer_hash,
          'train_nll_initial': initial_train, 'train_nll_base': base_train,
          'validation_nll_initial': initial_validation, 'validation_nll_base': base_validation,
          'pretrain_updates': 20, 'pretrain_target_exposures': 20 * int((pretrain['labels'][:,1:] != -100).sum()),
          'resume_equal': resume_equal, 'sft_updates': 20,
          'sft_validation_before': before_sft, 'sft_validation_after': after_sft,
          'lm_validation_after_sft': measure(model, validation),
          'example_generation': answer, 'scope': 'tiny_synthetic_pipeline_only'}
(root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(root / 'MODEL_CARD.md').write_text('# Miniature pedagogique\n\nCorpus synthetique tres petit ; aucune performance generale revendiquee.\n', encoding='utf-8')
print(report)
```

## 6. Lire les résultats sans surinterpréter

Le script doit produire un tokenizer rechargeable, un état **base**, un état **instruct**, un checkpoint au step 10, un rapport et une première model card minimale.

Le test de reprise doit être vrai : les steps 11 à 20 restaurés reproduisent les poids de la trajectoire continue dans ce contexte CPU déterministe. La loss train doit baisser sur le batch répété.

Le rapport conserve la validation de langage initiale/base/après SFT et la validation de completion avant/après SFT. Leurs objectifs et dénominateurs diffèrent : ne compare pas directement leurs valeurs comme si elles notaient la même chose.

La génération peut être répétitive, courte ou incohérente. Ne la cache pas : ce résultat révèle les limites d’un minuscule corpus et d’un budget de 20+20 updates. La miniature prouve une chaîne fonctionnelle, pas la maîtrise de la langue.

## 7. Ce que la reprise couvre ici

Le checkpoint conserve poids, optimizer, scheduler, RNG PyTorch et identité du tokenizer. Le prochain batch est toujours le même, le dropout est nul et il n’existe pas de worker/sampler à repositionner.

À plus grande échelle, le contrat doit inclure RNG des autres bibliothèques/appareils, position des données, précision mixte et états distribués. Le test du module 51 et les exigences du framework restent à appliquer au nouveau pipeline.

SFT commence intentionnellement une nouvelle phase avec un nouvel optimizer/scheduler. Ce n’est pas la reprise du même objectif de pré-entraînement.

## 8. Passer de la miniature au projet cible

1. **Données :** remplace les phrases artificielles par un corpus autorisé, tracé et séparé par groupes. Conserve les comptes et raisons de retrait.
2. **Tokenizer :** compare plusieurs configurations sur développement, puis fige la version choisie. Conserve un autre holdout si tu multiplies les choix.
3. **Smoke model :** valide formes, labels, overfit un batch et reprise avec une taille qui itère vite.
4. **Architecture :** choisis une taille à partir du step complet et du débit mesurés, pas uniquement des poids bruts.
5. **Pré-entraînement :** fixe budget et mixture ; logge NLL, cibles, LR, gradients, mémoire et temps.
6. **Base :** évalue langage sur corpus réservé et générations à prompts fixés.
7. **SFT :** prépare des démonstrations et une validation distinctes ; inspecte les masques.
8. **Comparaison :** évalue base et instruct sur comportement et régressions avec le même protocole approprié.
9. **Extensions :** n’ajoute DPO ou RL que pour une hypothèse restante mesurable et des données/rewards contrôlées.

Augmenter simultanément vocabulaire, couches, contexte et corpus rend l’origine d’un problème difficile à identifier. Change d’échelle après des mesures intermédiaires.

## 9. Livrables du projet final

```text
project62/
  README.md                 reconstruction et commandes dans l’ordre
  environment/              versions, matériel et conventions
  data/                     sources, groupes, manifests et splits
  tokenizer/                fichiers et hash figés
  configs/                  smoke, pré-entraînement, SFT
  src/                      préparation, training, évaluation
  checkpoints/              états base et phases/reprise
  logs/                     métriques brutes et configurations
  eval/                     sorties, scores et cas d’échec
  MODEL_CARD.md             portée, données, mesures et limites
```

La model card minimale générée par le pilote est un point de départ. Complète-la avec paramètres exacts, corpus/provenance, tokenizer, budget, protocole, résultats, échecs et conditions d’usage réellement évaluées.

Les fichiers d’export et de reprise n’ont pas nécessairement le même rôle. Indique lesquels servent à générer, à continuer le training et à reproduire l’évaluation.

## 10. Exercices avec corrigés

### Exercice 1 — Nouvelle base

Pourquoi ce projet est-il from scratch même s’il utilise la classe GPT2LMHeadModel ?

<details>
<summary>Corrigé</summary>

La classe fournit l’architecture, mais nous construisons ses poids à partir d’une configuration et d’une initialisation. Aucun checkpoint linguistique préentraîné n’est rechargé.

</details>

### Exercice 2 — Tokenizer figé

Pourquoi comparer son hash avant de reprendre ?

<details>
<summary>Corrigé</summary>

Les embeddings et sorties correspondent à des IDs précis. Un autre mapping rendrait les mêmes poids incompatibles avec le sens attendu des entrées et cibles.

</details>

### Exercice 3 — Deux pertes

La completion NLL après SFT est plus basse que la NLL documentaire. Est-ce une comparaison de qualité directe ?

<details>
<summary>Corrigé</summary>

Non. Corpus, cibles, contexte et masques diffèrent. Compare chaque mesure à sa baseline compatible et ajoute le benchmark de comportement.

</details>

### Exercice 4 — Réponse médiocre

Une génération incohérente malgré la baisse de train loss annule-t-elle toutes les vérifications techniques ?

<details>
<summary>Corrigé</summary>

Non. Elle peut coexister avec un pipeline correct et un corpus/budget insuffisant. Les contrôles techniques ne promettent pas la qualité linguistique.

</details>

### Exercice 5 — Extensions

Pourquoi ne pas appliquer DPO puis RL uniquement pour « finir toutes les méthodes » ?

<details>
<summary>Corrigé</summary>

Chaque étape a besoin d’un problème identifié, de données ou rewards adaptés, d’un budget et d’une évaluation. L’empilement sans hypothèse ajoute des risques et empêche l’attribution des résultats.

</details>

## 11. Critères de fin du parcours

- [ ] Je reconstruis le corpus et le tokenizer versionnés.
- [ ] Je peux expliquer chaque forme et chaque cible de la boucle.
- [ ] Je montre apprentissage d’un batch et reprise du contrat défini.
- [ ] Je conserve base et instruct avec leur tokenizer compatible.
- [ ] Je rapporte langage, comportement, régressions et coût.
- [ ] Je peux présenter une limite ou un résultat négatif avec ses preuves.
- [ ] Une autre personne peut refaire le pilote depuis le README.

## Fiche mémo

```text
Données → tokenizer → architecture → pré-entraînement → reprise → évaluation
→ SFT → comparaison → préférence/RL seulement si un besoin le justifie.
```

Fin du parcours principal. Utilise l’[index](INDEX.md), le [glossaire](GLOSSAIRE.md), l’[annexe de choix](ANNEXE_CHOISIR_LA_BONNE_TECHNIQUE.md) et l’[annexe de diagnostic](ANNEXE_DIAGNOSTIC_ENTRAINEMENT.md) pour revenir au mécanisme qui manque.
