# Module 12 — Comprendre et réaliser un premier SFT

## 1. Ce que tu vas apprendre

Le **Supervised Fine-Tuning**, ou SFT, adapte un modèle à partir de sorties de référence. La question devient du contexte ; la réponse fournit les tokens que l’on veut rendre plus probables.

Tu vas construire les labels, inspecter le batch reçu par un trainer, effectuer de vraies mises à jour et sauvegarder le résultat.

**Prérequis :** modules 6 à 11 ; décalage causal, `-100`, backward, validation indépendante.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** `torch`, `transformers`, `tokenizers`, `datasets`, `trl` et `accelerate`. Le laboratoire miniature fonctionne sur CPU, sans téléchargement.

## 2. Quelle erreur le SFT mesure-t-il ?

Une conversation contient ici une question et une réponse de référence. Le modèle voit la question et apprend à prédire les tokens de la réponse, un à un, en utilisant les tokens précédents de référence pendant l’entraînement : c’est le *teacher forcing*.

```text
Entrée : [USER, question, ASSISTANT, réponse_1, réponse_2, EOS]
Labels : [-100, -100,     -100,     réponse_1, réponse_2, EOS]
```

Les labels sont montrés **avant le décalage interne** du modèle causal. La sortie juste avant `réponse_1` apprend à prédire ce premier token.

\[
L = -\frac{1}{N_{cibles}}\sum_{t\in cibles}\log p_\theta(y_t\mid contexte, y_{<t})
\]

Le masque décide quelles positions sont des cibles. Le prompt reste visible même lorsqu’il est exclu de la loss.

### Trois configurations différentes

| Configuration | Cibles retenues |
| --- | --- |
| Séquence complète | Texte entier hors padding et positions non évaluables |
| Completion-only | Partie completion d’un dataset prompt/completion |
| Assistant-only | Réponses assistant identifiées dans un format conversationnel compatible |

`assistant_only_loss=True` exige un template qui permet de repérer les segments assistant, notamment avec les mécanismes de masque attendus par TRL. Un template affichant simplement les rôles ne garantit pas cette possibilité.

Notre premier script utilise donc un dataset **prompt/completion** et `completion_only_loss=True` : la frontière est explicite et nous vérifions les labels effectivement produits.

## 3. Ce que le trainer automatise

```text
Dataset → tokenisation → collator → batch avec labels
        → forward → cross-entropy → backward → optimizer step
        → logs, évaluation et sauvegarde
```

Le **collator** assemble les exemples en batches, gère le padding et prépare les champs attendus. Le trainer ne garantit pas que les cibles sont pertinentes : cela reste ton travail.

Le **packing** remplit des séquences avec plusieurs exemples pour réduire le padding. Les frontières, positions et masques dépendent de la stratégie. Sans précautions adaptées, des exemples empaquetés peuvent partager du contexte : le packing n’est pas une garantie d’indépendance. Nous le désactivons d’abord.

## 4. Pourquoi commencer avec un modèle minuscule ?

Le script construit un GPT-2 de démonstration à partir d’une configuration, avec un vocabulaire local. Il réalise le même passage par `SFTTrainer`, mais **ses poids sont initialement aléatoires**.

Ce premier laboratoire valide la mécanique du SFT ; il ne démontre pas l’adaptation d’un assistant préentraîné ni une compétence linguistique. La seconde étape consiste à remplacer ce modèle et son tokenizer par une base préentraînée compatible.

Nos quatre exemples d’apprentissage et deux reformulations de validation sont trop petits et trop proches pour servir de benchmark sérieux. Leur rôle est de rendre la plomberie observable.

## 5. Laboratoire autonome — Inspecter, entraîner, sauvegarder

Copie le bloc entier dans un fichier, active ton environnement puis lance-le. Les fichiers générés sont écrits dans `outputs/module12/`, relativement au dossier de lancement. Une nouvelle exécution peut remplacer ces sorties pédagogiques.

Prédis : les poids vont-ils changer ? Les tokens de `question ... reponse` doivent-ils être des cibles actives ?

```python
from pathlib import Path
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

torch.manual_seed(42)
words = ['<pad>', '<eos>', '<unk>', 'question', 'batch', 'gradient', 'lr', 'step',
         'reponse', 'groupe', 'exemples', 'derivee', 'loss', 'amplitude', 'mise',
         'jour', 'signifie', 'quoi']
backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token='<unk>'))
backend.pre_tokenizer = WhitespaceSplit()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
                                   eos_token='<eos>', unk_token='<unk>')
config = GPT2Config(vocab_size=len(tokenizer), n_positions=64, n_embd=32,
                    n_layer=1, n_head=2, resid_pdrop=0.0, embd_pdrop=0.0,
                    attn_pdrop=0.0, bos_token_id=None, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, use_cache=False)
model = GPT2LMHeadModel(config)
train = Dataset.from_list([
    {'prompt': 'question batch reponse ', 'completion': 'groupe exemples <eos>'},
    {'prompt': 'question gradient reponse ', 'completion': 'derivee loss <eos>'},
    {'prompt': 'question lr reponse ', 'completion': 'amplitude mise jour <eos>'},
    {'prompt': 'question step reponse ', 'completion': 'mise jour <eos>'},
])
validation = Dataset.from_list([
    {'prompt': 'question batch signifie quoi reponse ', 'completion': 'groupe exemples <eos>'},
    {'prompt': 'question gradient signifie quoi reponse ', 'completion': 'derivee loss <eos>'},
])
args = SFTConfig(output_dir='outputs/module12', use_cpu=True, bf16=False, fp16=False,
                 max_steps=20, per_device_train_batch_size=2, learning_rate=1e-3,
                 max_length=64, completion_only_loss=True, assistant_only_loss=False,
                 packing=False, gradient_checkpointing=False, report_to='none',
                 save_strategy='no', logging_steps=10, seed=42, disable_tqdm=True)
trainer = SFTTrainer(model=model, processing_class=tokenizer, args=args,
                     train_dataset=train, eval_dataset=validation)
batch = next(iter(trainer.get_train_dataloader()))
print('IDs :', batch['input_ids'][0].tolist())
print('Labels :', batch['labels'][0].tolist())
assert (batch['labels'] != -100).any()
assert (batch['labels'] == -100).any()
reference = next(model.parameters()).detach().clone()
before = trainer.evaluate()['eval_loss']
trainer.train()
after = trainer.evaluate()['eval_loss']
changed = not torch.equal(reference, next(model.parameters()).detach())
print('Validation avant/apres :', before, after)
print('Parametres modifies :', changed)
trainer.save_model('outputs/module12/final')
tokenizer.save_pretrained('outputs/module12/final')
print('Modele et tokenizer :', Path('outputs/module12/final').resolve())
```

### Lire les résultats

Le script affiche de vrais labels produits par le collator. Les cibles du prompt et du padding sont ignorées, tandis que la completion possède des cibles actives. Vérifie les IDs avec le tokenizer au lieu de te contenter des assertions « au moins un masque ».

Le vocabulaire local se découpe aux espaces ; il ne reproduit pas BPE. `GPT2Config` définit ici une fenêtre de 64 positions, un bloc et des vecteurs de taille 32. `max_length=64` est un budget choisi pour cette miniature.

Les paramètres doivent changer après les 20 steps. La loss peut baisser, mais une baisse sur ces données ne constitue pas une validation d’un vrai assistant.

Les poids et le tokenizer sont sauvegardés ensemble : sans le même vocabulaire et le même mapping, les IDs n’ont plus la même signification.

## 6. Passer à un modèle préentraîné

Reprends ensuite un modèle déjà utilisé dans le cours, par exemple Qwen3-0.6B, avec **son propre tokenizer**. Ne conserve pas le tokenizer local ni la configuration aléatoire du laboratoire.

1. Charge le checkpoint de départ et fixe sa révision.
2. Prépare tes splits du module 11 et sauvegarde une baseline avant toute mise à jour.
3. Utilise le chat template attendu. Pour un format prompt/completion, assure-toi que leur concaténation représente exactement la conversation et sa réponse ; vérifie la frontière tokenisée.
4. Inspecte au moins un batch : contenu décodé, labels, nombre de cibles, EOS et éventuelle troncature.
5. Commence par un ou deux steps sur quelques exemples. Mesure la mémoire avant d’augmenter le volume.
6. Choisis une précision et un LR adaptés à ce modèle et à la méthode de mise à jour ; le LR `1e-3` de la miniature aléatoire n’est pas une recommandation de full FT.
7. Évalue sur validation puis sur le benchmark fixé, et recharge le résultat sauvegardé pour vérifier son utilisation.

Le SFT peut être réalisé en full FT ou avec LoRA : SFT décrit le **signal d’apprentissage**, pas le nombre de paramètres modifiés.

## 7. Expériences guidées

- **Masking :** remplace seulement `completion_only_loss=True` par `False`. Inspecte le batch et compte les cibles supplémentaires ; tu changes l’objectif évalué.
- **Budget :** baisse `max_length` jusqu’à retirer une partie de la réponse. Repère les exemples sans cibles utiles avant d’entraîner ; ne confonds pas succès de tokenisation et qualité de l’exemple.
- **Mémorisation :** compare une question train et une reformulation réservée. Note que la première ne suffit pas à mesurer une généralisation.
- **Rechargement :** charge le dossier `final` avec les classes Auto de Transformers et vérifie les dimensions, le tokenizer et un forward en mode eval.

Change une seule chose par expérience et recrée le modèle initial avec la même seed.

## 8. Diagnostiquer les premiers problèmes

| Symptôme | Question à examiner |
| --- | --- |
| Toutes les cibles sont `-100` | Frontière prompt/réponse ou troncature incorrecte ? |
| Le prompt participe malgré l’objectif annoncé | Mauvais format ou masque non produit ? |
| OOM au premier step | Poids, gradients, optimizer et activations ont-ils été budgétés ? |
| Belle réponse sur train seulement | Mémorisation possible ; que dit la validation ? |
| Réponse sauvegardée incohérente | Même tokenizer, template, base et configuration au rechargement ? |

## 9. Exercices avec corrigés

### Exercice 1 — Premier token de réponse

À quelle position de logits compare-t-on le premier token assistant ?

<details>
<summary>Corrigé</summary>

Aux logits de la position qui le précède dans la séquence causale. Les labels non décalés restent placés sur les tokens cibles ; le modèle ou le calcul de loss réalise ensuite le décalage.

</details>

### Exercice 2 — Template et masque

Un template affiche `user` et `assistant`. Peut-on activer assistant-only sans rien vérifier ?

<details>
<summary>Corrigé</summary>

Non. Il faut que la préparation fournisse un masque assistant compatible avec le trainer et vérifier les labels finaux. La présence de marqueurs textuels ne prouve pas que le masque existe.

</details>

### Exercice 3 — SFT ou full FT ?

Peut-on faire du SFT en entraînant seulement des adapters ?

<details>
<summary>Corrigé</summary>

Oui. Le SFT décrit les cibles supervisées. LoRA décrit une manière de paramétrer les mises à jour. Les deux notions répondent à des questions différentes.

</details>

### Exercice 4 — Résultat utile ?

La miniature atteint une loss faible. A-t-on entraîné un professeur de français ?

<details>
<summary>Corrigé</summary>

Non. On a vérifié un pipeline sur quelques séquences et un petit vocabulaire. Il faut une base adaptée, des données pertinentes et une évaluation indépendante pour conclure sur un usage réel.

</details>

## 10. Validation — avant de continuer

- [ ] Je peux dessiner IDs et labels avant/après décalage.
- [ ] Je distingue completion-only et assistant-only.
- [ ] J’inspecte un batch effectivement fourni au modèle.
- [ ] Je peux montrer une vraie modification de poids.
- [ ] Je sais ce que le laboratoire aléatoire ne démontre pas.
- [ ] Je conserve modèle, tokenizer, configuration et baseline.

## Fiche mémo

```text
SFT : rendre les sorties de référence plus probables dans leur contexte.
Le point à vérifier : quels tokens deviennent réellement des cibles ?
Un trainer automatise la boucle ; il ne valide pas tes données.
```

Le [module 13](13_baseline_et_evaluation_avant_apres.md) rend la comparaison avant/après mesurable.

## Référence

[TRL : SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer) — formats de données, completion-only et assistant-only. Les scripts du cours sont vérifiés avec les versions de l’environnement ; enregistre aussi les tiennes.

Compatibilité vérifiée dans l’environnement du cours : PyTorch 2.6.0, Transformers 5.18.0, Datasets 5.0.1, TRL 1.14.1 et Accelerate 1.15.0. Les laboratoires PEFT utilisent PEFT 0.21.2. Enregistre tes versions avant de comparer ou de reprendre une expérience.
