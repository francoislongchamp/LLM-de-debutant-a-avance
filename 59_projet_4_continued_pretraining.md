# Module 59 — Projet 4 : isoler la valeur ajoutée d’un CPT

## 1. Ce que tu vas construire

Le projet compare deux chemins depuis la même base : **base→SFT** et **base→CPT→même SFT**. Il doit distinguer progrès de modélisation documentaire et valeur finale pour la tâche.

**Prérequis :** modules 25, 26, 45 à 47, 54 à 56.

**Temps indicatif :** pilote court sur CPU ; expérience réelle de plusieurs heures ou davantage selon le corpus.

**Matériel :** environnement TRL du module 12. Le pilote nécessite `outputs/module12/final`, créé depuis le même dossier de lancement.

## 2. Dessiner les points de mesure

```text
Base ───────────→ SFT contrôle ───→ évaluation finale contrôle
  └→ CPT → mesure intermédiaire → même SFT → évaluation finale traitement
```

Évalue la base, l’état CPT intermédiaire et les deux modèles finaux. Sans l’intermédiaire, une variation finale est plus difficile à relier aux phases.

Le coût du CPT est supplémentaire. Ce protocole mesure sa valeur marginale dans ce pipeline ; il n’est pas une égalité de compute total entre les branches. Une autre expérience à compute total égal poserait une autre question.

## 3. Données à préparer

Le corpus documentaire CPT et les démonstrations SFT ont des rôles différents. Ils peuvent concerner le même domaine, mais tu dois tracer leurs sources et éviter les fuites vers les évaluations.

Sépare les documents liés avant le blocage de tokens. Garde un holdout de langage de domaine, un benchmark métier et des catégories générales à préserver.

Une baisse de PPL domaine ne prouve ni suivi d’instruction ni restitution fiable des faits. Les résultats de la tâche finale comptent pour décider si le CPT répond à ton besoin.

## 4. Laboratoire guidé — Deux chemins réellement entraînés

Le pilote utilise le petit checkpoint appris du module 12 et des séquences de mots artificielles. Les variantes réservées sont proches et peu nombreuses ; il vérifie le graphe expérimental, pas une spécialisation linguistique robuste.

Les deux SFT ont mêmes données, seed, steps et configuration. Les LR de miniature ne sont pas des recommandations pour un LLM réel.

```python
import json
from pathlib import Path
import time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

source = Path('outputs/module12/final')
if not source.exists():
    raise FileNotFoundError('Executer le laboratoire 12 depuis le meme dossier.')
root = Path('outputs/project59')
root.mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
train = Dataset.from_list([
    {'prompt': 'question batch signifie reponse ', 'completion': 'groupe exemples <eos>'},
    {'prompt': 'question gradient signifie reponse ', 'completion': 'derivee loss <eos>'},
])
validation = Dataset.from_list([
    {'prompt': 'question batch quoi signifie reponse ', 'completion': 'groupe exemples <eos>'},
    {'prompt': 'question gradient quoi signifie reponse ', 'completion': 'derivee loss <eos>'},
])
domain_train = tokenizer(['gradient derivee loss <eos>', 'batch groupe exemples <eos>'],
                         return_tensors='pt', padding=True, add_special_tokens=False)
domain_validation = tokenizer(['gradient loss derivee <eos>', 'batch exemples groupe <eos>'],
                              return_tensors='pt', padding=True, add_special_tokens=False)


def labels_for(inputs):
    labels = inputs['input_ids'].clone()
    labels[inputs['attention_mask'] == 0] = -100
    return labels


def lm_loss(model, inputs):
    model.eval()
    with torch.no_grad():
        return model(**inputs, labels=labels_for(inputs)).loss.item()


reports = []
for name, cpt_steps in [('sft_only', 0), ('cpt_then_sft', 20)]:
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True)
    before = lm_loss(model, domain_validation)
    start = time.perf_counter()
    if cpt_steps:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
        model.train()
        for _ in range(cpt_steps):
            optimizer.zero_grad(set_to_none=True)
            model(**domain_train, labels=labels_for(domain_train)).loss.backward()
            optimizer.step()
        del optimizer
        model.save_pretrained(root / 'intermediate_cpt')
    cpt_seconds = time.perf_counter() - start
    after_cpt = lm_loss(model, domain_validation)
    args = SFTConfig(output_dir=str(root / name), use_cpu=True, bf16=False, fp16=False,
                     max_steps=20, learning_rate=0.001, per_device_train_batch_size=2,
                     max_length=64, completion_only_loss=True, packing=False,
                     gradient_checkpointing=False, report_to='none', save_strategy='no',
                     logging_steps=20, seed=42, disable_tqdm=True)
    trainer = SFTTrainer(model=model, processing_class=tokenizer, args=args,
                         train_dataset=train, eval_dataset=validation)
    before_sft = trainer.evaluate()['eval_loss']
    trainer.train()
    after_sft = trainer.evaluate()['eval_loss']
    trainer.save_model(str(root / name / 'final'))
    reports.append({'path': name, 'cpt_steps': cpt_steps, 'cpt_seconds_cpu': cpt_seconds,
                    'domain_nll_before': before, 'domain_nll_after_cpt': after_cpt,
                    'completion_nll_before_sft': before_sft, 'completion_nll_after_sft': after_sft,
                    'sft_steps': 20})
(root / 'report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
print(reports)
```

Le rapport doit contenir **deux branches**, avec **0 et 20 steps CPT**, puis **20 steps SFT dans chacune**. L’état intermédiaire est conservé séparément.

Les gains sont mesurés, pas garantis. Si la completion NLL finale ne s’améliore pas grâce au CPT, le pilote ne démontre pas son intérêt pour cette mesure. Un résultat négatif n’autorise pas à changer le benchmark après coup.

`lm_loss` utilise ici un batch documentaire avec une même longueur et des labels PAD masqués. Pour plusieurs batches variables, agrège leurs sommes NLL et leurs effectifs actifs comme au module 44.

## 5. Passer à un corpus réel

1. Fige base, tokenizer, corpus, groupes et rapports de nettoyage.
2. Définis un budget de tokens CPT, la mixture et les répétitions prévues.
3. Archive les mesures initiales domaine et générales.
4. Réalise le CPT avec logs, contrôles de mémoire et reprise testée.
5. Mesure l’état intermédiaire et les éventuelles régressions.
6. Applique le même protocole SFT de contrôle et de traitement ; annonce toute adaptation de tuning.
7. Sélectionne sur validation et évalue les deux finales selon le même test.
8. Rapporte coût supplémentaire et gain marginal par catégorie.

Pour séparer effet d’étape et meilleure recette, tu peux prévoir une étude secondaire de tuning comparable. Ne la mélange pas avec l’ablation principale à SFT fixé.

## 6. Tableau et livrables

Conserve pour chaque état : loss/PPL domaine avec protocole, benchmark de tâche, régression générale, tokens exposés, temps, mémoire et identité de checkpoint.

Remets corpus/manifeste, config CPT, config SFT commune, splits, état intermédiaire, checkpoints finaux, sorties brutes et `RESULTS.md`.

Une conclusion utile dit : « Dans ces données et ce budget, l’étape CPT apporte X sur la tâche finale pour Y de coût et Z de régression », ou indique que ce bénéfice n’est pas démontré.

## 7. Exercices avec corrigés

### Exercice 1 — Gain marginal

SFT seul obtient 72 %, CPT→SFT 81 %. Quel gain ?

<details>
<summary>Corrigé</summary>

**9 points de pourcentage**, avec effectifs et incertitude à rapporter. Le coût CPT supplémentaire reste à intégrer à la décision.

</details>

### Exercice 2 — Deux datasets SFT

Pourquoi changer aussi les démonstrations SFT empêche-t-il d’isoler le CPT ?

<details>
<summary>Corrigé</summary>

Le résultat peut alors venir des nouvelles données supervisées autant que de la phase documentaire. Il faut une comparaison qui distingue ces interventions.

</details>

### Exercice 3 — PPL intermédiaire

Une PPL domaine plus basse après CPT suffit-elle à valider le projet ?

<details>
<summary>Corrigé</summary>

Non. Il faut mesurer la tâche finale après SFT et les capacités à préserver, selon l’objectif du projet.

</details>

### Exercice 4 — Compute

Les branches sont-elles à compute total égal simplement parce qu’elles font le même SFT ?

<details>
<summary>Corrigé</summary>

Non. La branche CPT a une phase supplémentaire. Le protocole mesure une valeur ajoutée avec coût marginal, pas un match automatique de calcul total.

</details>

## 8. Critères de fin

- [ ] Contrôle, traitement et état intermédiaire sont distincts.
- [ ] Corpus documentaire, démonstrations et évaluations sont tracés.
- [ ] Le SFT final comparé est défini selon la même convention.
- [ ] Les mesures intermédiaires et finales sont conservées.
- [ ] Coût marginal, régressions et absence éventuelle de gain sont rapportés.

## Fiche mémo

```text
La question n’est pas seulement « CPT baisse-t-il sa loss ? »
Elle est : apporte-t-il une valeur au pipeline final, pour quel coût ?
```

Le [projet 60](60_projet_5_dpo.md) étudie une préférence au-delà du SFT.
