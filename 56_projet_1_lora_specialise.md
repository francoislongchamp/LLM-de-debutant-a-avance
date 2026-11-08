# Module 56 — Projet 1 : livrer un adapter LoRA et sa preuve d’évaluation

## 1. Ce que tu vas construire

Un projet complet relie besoin, données, baseline, entraînement, évaluation et rechargement. Le livrable n’est pas seulement un dossier d’adapter.

Tu vas réaliser un pilote local sur la miniature du module 12, puis définir une expérience de spécialisation sur un domaine maîtrisé.

**Prérequis :** modules 9 à 19, 51 à 55.

**Temps indicatif :** pilote court sur CPU ; 4 à 8 heures ou davantage pour le projet réel selon données, matériel et itérations.

**Matériel :** environnement du module 12 + PEFT. Le pilote réutilise `outputs/module12/final`, donc exécute d’abord le bloc du module 12 depuis le même dossier de travail.

## 2. Écrire l’hypothèse avant les réponses

Exemple de cahier des charges : améliorer des explications PyTorch au format défini, sans dépasser une perte générale fixée à l’avance.

Définis quatre éléments : comportement voulu, catégories de données, mesure de réussite et limites acceptables de régression/coût. Les seuils chiffrés d’exemple ne deviennent pas automatiquement tes exigences.

Pour un petit corpus, relis toutes les réponses et conserve les raisons de corrections. Sépare les groupes avant les reformulations, puis garde un test final hors du tuning.

## 3. Le pilote local et ses limites

Le checkpoint du module 12 est un petit GPT-2 à vocabulaire artificiel, entraîné sur quelques séquences. Il fournit ici un état appris local à adapter, sans télécharger un grand modèle.

Les deux reformulations réservées du pilote sont proches du train : elles vérifient la chaîne technique, pas une généralisation robuste ni la qualité d’un professeur de français.

Nous ciblons `c_attn`, une projection fusionnée de GPT-2, avec la convention `fan_in_fan_out` correspondante. Ce choix ne doit pas être copié sur Qwen sans inspection de ses projections.

## 4. Laboratoire guidé — Baseline, LoRA, rechargement

Le script écrit sous `outputs/project56/` : sorties avant/après, adapter, tokenizer et rapport.

```python
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer

source = Path('outputs/module12/final')
if not source.exists():
    raise FileNotFoundError('Executer le laboratoire du module 12 depuis le meme dossier de travail.')
root = Path('outputs/project56')
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
base = AutoModelForCausalLM.from_pretrained(source, local_files_only=True)
rows = [
    {'prompt': 'question batch signifie reponse ', 'completion': 'groupe exemples <eos>'},
    {'prompt': 'question gradient signifie reponse ', 'completion': 'derivee loss <eos>'},
]
holdout = [
    {'prompt': 'question batch quoi signifie reponse ', 'expected': 'groupe exemples'},
    {'prompt': 'question gradient quoi signifie reponse ', 'expected': 'derivee loss'},
]


def generate_records(model):
    model.eval()
    records = []
    with torch.no_grad():
        for index, case in enumerate(holdout):
            inputs = tokenizer(case['prompt'], return_tensors='pt', add_special_tokens=False)
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            records.append({'id': f'h{index}', 'prompt': case['prompt'], 'output': answer,
                            'expected': case['expected'], 'correct': answer == case['expected']})
    return records


before = generate_records(base)
model = get_peft_model(base, LoraConfig(
    r=2, lora_alpha=4, lora_dropout=0.0, target_modules=['c_attn'],
    fan_in_fan_out=True, bias='none', task_type='CAUSAL_LM',
))
frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
          if not parameter.requires_grad}
trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
args = SFTConfig(output_dir=str(root / 'training'), use_cpu=True, bf16=False, fp16=False,
                 max_steps=30, learning_rate=0.01, per_device_train_batch_size=2,
                 max_length=64, completion_only_loss=True, packing=False,
                 gradient_checkpointing=False, report_to='none', save_strategy='no',
                 logging_steps=15, seed=42, disable_tqdm=True)
trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                     train_dataset=Dataset.from_list(rows))
trainer.train()
after = generate_records(model)
base_unchanged = all(torch.equal(parameter, frozen[name]) for name, parameter in model.named_parameters()
                     if name in frozen)
model.save_pretrained(root / 'adapter')
tokenizer.save_pretrained(root / 'tokenizer')
restored = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(source, local_files_only=True), root / 'adapter')
restored.eval()
model.eval()
probe = tokenizer('question batch reponse ', return_tensors='pt', add_special_tokens=False)
with torch.no_grad():
    reload_equal = torch.allclose(model(**probe).logits, restored(**probe).logits, atol=1e-6)
for name, records in [('baseline', before), ('after', after)]:
    (root / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
report = {'base': str(source), 'trainable_parameters': trainable,
          'base_unchanged': base_unchanged, 'reload_equal': reload_equal,
          'before_correct': sum(row['correct'] for row in before),
          'after_correct': sum(row['correct'] for row in after), 'holdout_n': len(holdout),
          'scope': 'synthetic_miniature_not_a_language_quality_claim'}
(root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(report)
```

Le rapport doit montrer des paramètres LoRA entraînables, `base_unchanged=True` et `reload_equal=True`. Les scores avant/après sont mesurés, pas imposés : un gain nul ou une régression doit rester visible.

`SFTTrainer` reçoit un modèle déjà adapté ; aucune deuxième `peft_config` n’est fournie. Les générations utilisent les mêmes prompts et réglages, en mode eval.

Le pilote n’inclut pas une sélection de checkpoint sur validation indépendante. Cette étape est obligatoire pour la conclusion du projet réel, et distincte du test final.

## 5. Passer au projet de domaine

1. Choisis la base et sa révision, son tokenizer et son template.
2. Rédige un corpus supervisé vérifié et des splits groupés.
3. Fixe le benchmark domaine, les régressions et la procédure de scoring.
4. Archive une baseline avant toute adaptation.
5. Choisis rang, scaling, dropout et cibles à partir des modules 18 et 19.
6. Inspecte labels, gradients et paramètres sur un smoke test court.
7. Entraîne sous un budget de cibles/updates annoncé et choisis sur validation.
8. Évalue le checkpoint sélectionné sur le test réservé et recharge l’adapter.

Le LR de 0.01 du petit pilote n’est pas une recommandation pour une grande base préentraînée.

## 6. Les artefacts à remettre

```text
project56/
  README.md              commandes et dossier de lancement
  data/                  sources/version, splits et manifeste
  configs/               base, tokenizer, LoRA et training
  baseline.jsonl         sorties brutes avant
  results.jsonl          sorties brutes après, IDs appariés
  logs/                  compteurs, LR, losses et coût
  adapter/               poids PEFT et configuration
  tokenizer/             fichiers compatibles
  RESULTS.md             gains, régressions, échecs et limites
```

Un rapport utile inclut total/entraînable, cibles exposées, temps, pic mémoire si mesuré, taille d’adapter et protocole de rechargement. Une mesure non effectuée reste indiquée comme telle.

## 7. Jalons et décisions

| Jalon | Preuve attendue | Si le jalon échoue |
| --- | --- | --- |
| Données | Schéma, relecture et groupes cohérents | Corriger avant le run |
| Baseline | Sorties et scores archivés | Pas de conclusion avant/après |
| Pipeline | Labels actifs, gradients finis, gel vérifié | Diagnostiquer un petit cas |
| Checkpoint | Sorties rechargées équivalentes | Corriger base/tokenizer/format |
| Qualité | Domaine et régressions selon protocole | Rapport négatif ou nouvelle hypothèse |

Une expérience techniquement valide peut réfuter l’hypothèse d’amélioration. Le projet reste informatif si ses données et sa conclusion sont honnêtes.

## 8. Exercices avec corrigés

### Exercice 1 — Trois premières vérifications

Quels résultats techniques vérifier avant d’interpréter le score du pilote ?

<details>
<summary>Corrigé</summary>

Présence des paramètres LoRA attendus, base inchangée et équivalence au rechargement. Il faut aussi vérifier labels, sorties et convention du score.

</details>

### Exercice 2 — Zéro gain

Le score après LoRA est identique. Faut-il retirer ce résultat du rapport ?

<details>
<summary>Corrigé</summary>

Non. Conserve le résultat et ses limites : taille du benchmark, compétence ciblée, budget, données et éventuelles régressions. Un nouveau test doit partir d’une hypothèse précise.

</details>

### Exercice 3 — Changer de base

Pourquoi `c_attn` et fan_in_fan_out ne sont-ils pas des réglages universels ?

<details>
<summary>Corrigé</summary>

Ils correspondent à des projections et conventions de stockage de l’architecture choisie. Une autre famille peut utiliser q_proj/v_proj ou d’autres classes et formes.

</details>

### Exercice 4 — Généralisation

Deux reformulations proches prouvent-elles la qualité du modèle sur tout le domaine ?

<details>
<summary>Corrigé</summary>

Non. Elles servent ici au test technique. Le projet réel demande couverture, effectifs et groupes indépendants adaptés à la compétence visée.

</details>

## 9. Critères de fin

- [ ] Une autre personne peut retrouver données, base et configuration.
- [ ] Les sorties avant/après sont appariées et conservées.
- [ ] Les paramètres optimisés et la base gelée sont vérifiés.
- [ ] Le rechargement est testé.
- [ ] Domaine, régressions, coût et limites sont rapportés.
- [ ] La conclusion peut être positive ou négative selon la mesure.

## Fiche mémo

```text
Un adapter + son identité de base + ses données + la mesure de ce qu’il change.
```

Le [projet 57](57_projet_2_lora_vs_qlora.md) isole ensuite l’effet du stockage quantifié de la base.
