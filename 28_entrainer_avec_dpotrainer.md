# Module 28 — Exécuter DPO et interpréter ses métriques

## 1. Ce que tu vas apprendre

Le module 27 a calculé une loss sur deux log-probabilités. Ici, elles seront calculées par de vrais réseaux et DPOTrainer effectuera les updates de la policy.

Tu vas construire une référence figée, entraîner quelques steps, calculer une marge sur des paires réservées et sauvegarder les poids nécessaires à l’expérience.

**Prérequis :** modules 12, 13 et 27 ; masques de réponse et marge relative DPO.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch, Transformers, Tokenizers, Datasets, TRL et Accelerate. Miniature sur CPU, sans téléchargement.

## 2. Le trajet d’une paire

```text
Même prompt + chosen   → policy → logp chosen
Même prompt + rejected → policy → logp rejected
                       → référence fixe pour les mêmes réponses
                       → marge relative → loss DPO → backward policy → step
```

La référence ne reçoit pas de mises à jour. Ce n’est pas un évaluateur qui découvre la bonne réponse ; son rôle est de fournir les log-probabilités d’ancrage.

Avec deux modèles complets, le budget mémoire augmente. D’autres configurations peuvent utiliser des adapters ou pré-calculer des scores de référence, mais il faut vérifier leur équivalence et conserver la bonne identité de référence.

## 3. Définir les métriques avant de lire les logs

Pour le DPO sigmoid standard :

\[
r^+ = \beta(\log\pi_\theta(y^+|x)-\log\pi_{ref}(y^+|x))
\]

et de même pour `r−`. La **reward margin** est `r+−r−`, donc `beta × m` avec la marge m du module 27.

La **reward accuracy** courante compte les paires où cette marge relative est positive. Cela diffère d’une accuracy calculée sur le simple ordre des log-probabilités brutes chosen/rejected.

Quand policy=référence, les marges relatives valent zéro. Avec une règle stricte `>0`, les égalités ne sont pas des succès : une accuracy initiale de zéro peut donc être normale pour cette métrique. Elle ne veut pas dire que le modèle est incapable de produire des réponses utiles.

Les noms et agrégations exacts dépendent de la version et de la variante. Lis ce qui est réellement calculé avant de comparer deux logs.

## 4. Laboratoire autonome — Une policy, sa référence et dix steps DPO

Nous créons une miniature locale, effectuons quelques mises à jour supervisées puis copions cet état pour la référence. Il ne s’agit pas d’un assistant préentraîné : le but est de rendre DPOTrainer observable.

Les deux paires train et leurs reformulations réservées sont trop petites et trop proches pour démontrer une généralisation réelle. Les fichiers sont écrits sous `outputs/module28/`.

Le tokenizer par espaces rend la frontière prompt/réponse explicite. Cette propriété doit être revérifiée avec un tokenizer et un template réels.

```python
from copy import deepcopy
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import DPOConfig, DPOTrainer

torch.manual_seed(42)
words = ['<pad>', '<eos>', '<unk>', 'question', 'batch', 'gradient', 'reponse',
         'groupe', 'exemples', 'poids', 'derivee', 'loss', 'change', 'valeur', 'signifie', 'quoi']
backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token='<unk>'))
backend.pre_tokenizer = WhitespaceSplit()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
                                   eos_token='<eos>', unk_token='<unk>')
policy = GPT2LMHeadModel(GPT2Config(
    vocab_size=len(tokenizer), n_positions=64, n_embd=24, n_layer=1, n_head=2,
    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    bos_token_id=None, eos_token_id=1, pad_token_id=0, use_cache=False,
))
rows = [
    {'prompt': 'question batch reponse ', 'chosen': 'groupe exemples <eos>', 'rejected': 'poids <eos>'},
    {'prompt': 'question gradient reponse ', 'chosen': 'derivee loss <eos>', 'rejected': 'change valeur <eos>'},
]
validation_rows = [dict(row, prompt=row['prompt'].replace('reponse', 'signifie quoi reponse')) for row in rows]

# Quelques mises a jour supervisees pour construire un etat de depart appris.
sft_optimizer = torch.optim.AdamW(policy.parameters(), lr=0.003, weight_decay=0.0)
policy.train()
for step in range(12):
    row = rows[step % len(rows)]
    encoded = tokenizer(row['prompt'] + row['chosen'], return_tensors='pt', add_special_tokens=False)
    prompt_length = len(tokenizer.encode(row['prompt'], add_special_tokens=False))
    labels = encoded['input_ids'].clone()
    labels[:, :prompt_length] = -100
    sft_optimizer.zero_grad(set_to_none=True)
    policy(**encoded, labels=labels).loss.backward()
    sft_optimizer.step()
del sft_optimizer
reference = deepcopy(policy).eval()
reference.requires_grad_(False)
reference_snapshot = {name: tensor.detach().clone() for name, tensor in reference.state_dict().items()}


def response_logp(model, prompt, answer):
    encoded = tokenizer(prompt + answer, return_tensors='pt', add_special_tokens=False)
    start = len(tokenizer.encode(prompt, add_special_tokens=False))
    with torch.no_grad():
        logits = model(**encoded).logits[:, :-1].float()
        target = encoded['input_ids'][:, 1:]
        logps = logits.log_softmax(-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return logps[:, start - 1:].sum().item()


def margins(model):
    model.eval()
    result = []
    for row in validation_rows:
        chosen = response_logp(model, row['prompt'], row['chosen'])
        rejected = response_logp(model, row['prompt'], row['rejected'])
        ref_chosen = response_logp(reference, row['prompt'], row['chosen'])
        ref_rejected = response_logp(reference, row['prompt'], row['rejected'])
        result.append((chosen - ref_chosen) - (rejected - ref_rejected))
    return result


before = margins(policy)
args = DPOConfig(output_dir='outputs/module28', use_cpu=True, bf16=False, fp16=False,
                 beta=0.1, max_steps=10, learning_rate=1e-3, max_length=64,
                 per_device_train_batch_size=2, gradient_checkpointing=False,
                 loss_type=['sigmoid'], report_to='none', logging_steps=5,
                 save_strategy='no', seed=42, disable_tqdm=True)
trainer = DPOTrainer(model=policy, ref_model=reference, processing_class=tokenizer,
                     args=args, train_dataset=Dataset.from_list(rows),
                     eval_dataset=Dataset.from_list(validation_rows))
trainer.train()
after = margins(policy)
print('Marges relatives validation avant/apres :', before, after)
print('Reference inchangee :', all(torch.equal(tensor, reference.state_dict()[name])
                                 for name, tensor in reference_snapshot.items()))
trainer.save_model('outputs/module28/final')
tokenizer.save_pretrained('outputs/module28/final')
reference.save_pretrained('outputs/module28/reference')
```

### Ce qu’il faut retrouver

Les marges relatives avant DPO doivent être **`[0,0]`**, à l’arrondi près. Après les dix steps, elles doivent devenir positives sur cette miniature facile. La comparaison de référence doit rester vraie.

Les valeurs exactes dépendent des versions et calculs ; le point à vérifier est le mécanisme. Les losses train et marges ne remplacent pas un benchmark indépendant.

`response_logp` sélectionne les scores des tokens effectivement présents, puis ne somme que ceux de la réponse, EOS compris ici. `start−1` vient du décalage causal : le premier token de réponse est prédit par la position précédente.

Notre calcul affiche **m**, sans beta. Un log du trainer nommé `rewards/margins` peut afficher **beta×m** et une moyenne sur d’autres batches. Une différence de valeur ne signifie pas automatiquement un bug.

## 5. Ce que les contrôles prouvent et ce qu’ils ne prouvent pas

Le script confirme qu’un entraînement DPO peut calculer une loss et changer la policy avec une référence conservée. La comparaison exacte affichée porte sur tous les tenseurs de l’état de référence, copiés avant l’entraînement.

Une marge accrue ne prouve pas que chosen devient plus probable en valeur absolue sur chaque paire, ni que les générations sont plus vraies ou plus claires. Relis les scores chosen et rejected séparément.

Les paires de notre miniature sont lexicales et simples. Leur succès ne valide pas une préférence humaine complexe. Sur un vrai corpus, respecte les groupes du module 11 et les audits du module 27.

## 6. Passer à une policy réelle

1. Choisis le checkpoint SFT et sa révision ; conserve une baseline.
2. Charge son tokenizer/template. Pour du conversationnel, utilise le format de paires pris en charge plutôt que de concaténer arbitrairement des marqueurs.
3. Vérifie que les deux branches partagent exactement le prompt attendu, et inspecte frontières et cibles après tokenisation.
4. Choisis une référence explicite ou un mécanisme équivalent documenté. Vérifie qu’elle ne change pas.
5. Commence avec quelques steps et un LR justifié, pas une copie du LR de SFT ou de la miniature.
6. Mesure marges sur validation, sorties générées et benchmark général.
7. Sauvegarde policy, tokenizer, identité de référence et configuration. Une reprise exacte exige aussi optimizer/scheduler et état du run.

Si tu pré-calcules les log-probabilités de référence, garde la même référence, les mêmes tokens, masques et conventions. Modifier ensuite le tokenizer rend ces valeurs incompatibles.

## 7. Longueurs et troncature

Les probabilités de séquence standard sont des produits de probabilités de tokens, donc leurs logarithmes sont des sommes. La longueur influence leur échelle ; ne remplace pas silencieusement les sommes par des moyennes si tu prétends reproduire la loss standard.

Une troncature peut supprimer la partie où chosen et rejected diffèrent ou enlever toute la réponse. Inspecte les exemples effectivement conservés et compte les cibles actives dans les deux branches.

Augmenter `max_length` change le budget mémoire ; diminuer ce budget peut changer le signal d’apprentissage. Le nom de l’option ne dispense pas d’examiner les tokens.

## 8. Expériences guidées

1. Compare beta=0.05 et beta=0.2 en recréant le même état SFT initial. Observe marge, losses, longueurs et résultats ; ne prédis pas la dérive par une proportion magique.
2. Entraîne séparément un simple SFT sur chosen depuis le même départ, puis évalue les mêmes paires. Explique la différence d’objectifs et le budget utilisé.
3. Inverse volontairement une annotation dans une copie. Observe que le trainer peut apprendre un signal erroné sans savoir que la préférence est fausse.
4. Sur le corpus réel, compare les générations avant/après avec une grille à l’aveugle et conserve aussi les régressions générales.

## 9. Diagnostic des premiers échecs

| Symptôme | Vérification |
| --- | --- |
| Toutes les marges valent zéro après plusieurs steps | Policy modifiée ? LR effectif ? Référence accidentellement partagée ? |
| Aucune cible de réponse | Frontière ou troncature incorrecte |
| Référence change | Même objet que policy, optimizer mal construit ou mauvais mécanisme de référence |
| Marge train excellente, sorties moins bonnes | Suradaptation au signal ou biais des paires |
| Métriques incompatibles entre scripts | Somme/moyenne, beta, masques, split ou variante différents |

## 10. Exercices avec corrigés

### Exercice 1 — Marge affichée

Ton calcul donne m=2 et beta=0.1. Quelle reward margin standard correspondante ?

<details>
<summary>Corrigé</summary>

**0.2**, car la marge de rewards relative est beta×m. Il faut aussi comparer des agrégations sur les mêmes exemples.

</details>

### Exercice 2 — Zéro initial

Pourquoi une accuracy relative stricte peut-elle valoir zéro quand la policy égale la référence ?

<details>
<summary>Corrigé</summary>

Chaque marge vaut zéro et la règle compte seulement `m>0`. Les égalités ne sont pas des succès selon cette convention ; cela ne décrit pas directement la qualité des générations.

</details>

### Exercice 3 — Référence ou juge ?

La référence donne-t-elle une note de vérité à chaque réponse ?

<details>
<summary>Corrigé</summary>

Non. Elle fournit des log-probabilités utilisées comme ancrage dans l’objectif. La préférence vient des annotations chosen/rejected, pas d’une évaluation factuelle par la référence.

</details>

### Exercice 4 — Vérifier une régression

La préférence validation progresse mais l’exactitude générale baisse. Quelle conclusion tirer ?

<details>
<summary>Corrigé</summary>

L’intervention améliore cette métrique de préférence dans ces conditions mais crée une régression sur l’autre mesure. Il faut rapporter les deux et décider selon les exigences, pas déclarer un progrès universel.

</details>

## 11. Validation — avant de continuer

- [ ] Je construis policy et référence distinctes depuis le même départ.
- [ ] Je calcule les log-probabilités de réponse avec le bon décalage.
- [ ] Je distingue marge brute, marge relative et reward margin.
- [ ] Je peux exécuter DPOTrainer puis vérifier la référence.
- [ ] Je connais les limites statistiques et pédagogiques de la miniature.
- [ ] Je garde un benchmark général et les informations de rechargement.

## Fiche mémo

```text
DPOTrainer orchestre une comparaison de log-probabilités relative à une référence.
Inspecter les deux branches, les cibles, les marges et les régressions.
Conserver la référence exacte aussi bien que la policy entraînée.
```

Le [module 29](29_reward_modeling.md) entraîne une fonction de score explicite, différente de la référence DPO.

## Référence

[TRL : DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer) — datasets, paramètres et métriques de l’entraînement DPO.
