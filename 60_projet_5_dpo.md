# Module 60 — Projet 5 : vérifier la préférence apprise par DPO

## 1. Ce que tu vas construire

Une policy SFT sert de départ à une optimisation de préférences. Le projet doit montrer si la préférence voulue progresse sans se réduire à un indice de style ou à une marge sur le train.

Tu vas auditer les paires, relire les marges et générations d’un pilote existant, puis définir une vraie comparaison indépendante.

**Prérequis :** modules 27 à 29, 53 à 56.

**Temps indicatif :** analyse courte ; plusieurs heures pour annotation, training et évaluation réelle.

**Matériel :** PyTorch et Transformers pour le script. Exécute auparavant le laboratoire 28 depuis le même dossier : il crée `outputs/module28/final` et `reference`.

## 2. Fixer une préférence principale

« Plus clair sans perdre d’exactitude » demande des critères : répondre à la question, expliquer les termes utiles, éviter les détours inutiles, conserver les précisions nécessaires.

Une réponse plus courte n’est pas toujours meilleure. Si chosen est systématiquement court et rejected toujours long, vérifie si les paires enseignent un raccourci.

Conserve contexte commun, justification, source/groupe, désaccords et décision pour les deux mauvaises réponses. Sépare les groupes avant les reformulations.

## 3. Deux évaluations à ne pas confondre

| Mesure | Question |
| --- | --- |
| Marge DPO sur paires réservées | La policy a-t-elle changé l’ordre relatif à la référence sur ces réponses fixes ? |
| Comparaison de générations | Les nouvelles réponses sont-elles préférées selon la rubrique, à prompts fixés ? |

Une reward accuracy relative n’est pas un win rate de deux textes générés à l’aveugle. Les longueurs et log-probabilités brutes apportent encore d’autres informations.

## 4. Laboratoire guidé — Recharger policy et référence du module 28

Le script n’effectue pas de nouveaux updates : il audite les artefacts d’un entraînement déjà réalisé. La miniature et ses deux reformulations ne constituent pas un benchmark humain.

```python
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path('outputs/module28')
if not (root / 'reference').exists() or not (root / 'final').exists():
    raise FileNotFoundError('Executer le laboratoire du module 28 depuis ce dossier.')
tokenizer = AutoTokenizer.from_pretrained(root / 'final', local_files_only=True)
reference = AutoModelForCausalLM.from_pretrained(root / 'reference', local_files_only=True).eval()
policy = AutoModelForCausalLM.from_pretrained(root / 'final', local_files_only=True).eval()
cases = [
    {'id': 'p1', 'prompt': 'question batch signifie quoi reponse ',
     'chosen': 'groupe exemples <eos>', 'rejected': 'poids <eos>'},
    {'id': 'p2', 'prompt': 'question gradient signifie quoi reponse ',
     'chosen': 'derivee loss <eos>', 'rejected': 'change valeur <eos>'},
]


def logp(model, prompt, answer):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    inputs = tokenizer(prompt + answer, return_tensors='pt', add_special_tokens=False)
    if inputs['input_ids'][0, :len(prefix)].tolist() != prefix:
        raise ValueError('Frontiere tokenisee non compatible avec cette methode de masque')
    with torch.no_grad():
        logits = model(**inputs).logits[:, :-1].float()
        ids = inputs['input_ids'][:, 1:]
        token_logps = logits.log_softmax(-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return token_logps[:, len(prefix)-1:].sum().item()


records = []
for case in cases:
    pc = logp(policy, case['prompt'], case['chosen'])
    pr = logp(policy, case['prompt'], case['rejected'])
    rc = logp(reference, case['prompt'], case['chosen'])
    rr = logp(reference, case['prompt'], case['rejected'])
    record = {'id': case['id'], 'raw_policy_margin': pc - pr,
              'relative_margin': (pc - rc) - (pr - rr),
              'chosen_tokens': len(tokenizer.encode(case['chosen'], add_special_tokens=False)),
              'rejected_tokens': len(tokenizer.encode(case['rejected'], add_special_tokens=False))}
    records.append(record)
    print(record)
print('Preference relative stricte :', sum(row['relative_margin'] > 0 for row in records), '/', len(records))
output = Path('outputs/project60')
output.mkdir(parents=True, exist_ok=True)
(output / 'pair_audit.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
```

Les marges relatives du pilote 28 doivent être positives, avec les limites déjà indiquées. La marge brute peut raconter un ordre différent car elle ne retire pas la référence.

Le contrôle de préfixe est important : avec un vrai tokenizer, encoder prompt puis concaténer une réponse ne garantit pas toujours une frontière identique à l’encodage complet. Utilise la préparation du trainer/template et vérifie les cibles réelles.

## 5. Réaliser le projet complet

1. Choisis le checkpoint SFT et une référence identifiée.
2. Fixe une grille, construis des paires utiles et audite les raccourcis.
3. Prépare splits groupés et benchmark général indépendant.
4. Archive avant DPO scores sur paires et générations sur prompts fixés.
5. Inspecte les deux branches après tokenisation et leurs cibles actives.
6. Lance un budget court puis une expérience avec beta, LR, variante et arrêt documentés.
7. Évalue paires réservées, générations à l’aveugle et régressions.
8. Teste la sauvegarde/reprise et conserve référence ou identité nécessaire.

Une comparaison supplémentaire SFT sur chosen peut aider à attribuer ce qui vient de l’objectif pairwise, sous un budget clairement annoncé.

## 6. Juger les générations indépendamment

Randomise l’ordre A/B et masque les noms des modèles. La grille doit pouvoir noter égalités et deux mauvaises réponses. Si un juge LLM est utilisé, mesure son accord sur un sous-ensemble humain et cherche biais de style, longueur et position.

Rapporte wins/ties/losses, effectifs, métriques factuelles et distributions de longueur. Un gain de win rate n’est pas une preuve universelle de vérité.

Si le même juge a produit les labels et fournit l’unique évaluation finale, il peut surtout confirmer ses propres biais. Prévois un contrôle indépendant pertinent.

## 7. Livrables et décision

Remets paires avec audit hors champs d’apprentissage, configurations, référence, policy/adapters, tokenizer, logs, sorties brutes et tableau par critère.

Fixe un seuil de préférence et des limites de régression avant le run. Conserve aussi une expérience qui échoue : elle peut révéler une annotation incohérente, une troncature ou un objectif trop étroit.

## 8. Exercices avec corrigés

### Exercice 1 — Deux marges

Pourquoi une marge relative positive ne signifie-t-elle pas forcément logp(chosen)>logp(rejected) ?

<details>
<summary>Corrigé</summary>

Elle mesure le changement d’écart par rapport à la référence. La préférence brute de départ peut être différente et ne pas avoir encore été inversée.

</details>

### Exercice 2 — Longueur

Chosen toujours plus long est-il un bon corpus pour apprendre automatiquement la précision ?

<details>
<summary>Corrigé</summary>

Pas sans audit : la longueur peut devenir un raccourci. Construis des cas où elle varie sans déterminer mécaniquement la préférence.

</details>

### Exercice 3 — Win rate

La reward accuracy DPO est-elle le taux de victoire de générations nouvelles ?

<details>
<summary>Corrigé</summary>

Non. Elle porte couramment sur des scores relatifs de réponses fixes ; la comparaison de nouvelles générations utilise un autre protocole.

</details>

### Exercice 4 — Référence

Pourquoi conserver son identité exacte dans les résultats ?

<details>
<summary>Corrigé</summary>

Les marges relatives et l’objectif dépendent de ses log-probabilités. Une autre référence change leur sens même à policy identique.

</details>

## 9. Critères de fin

- [ ] La préférence et les cas ambigus sont définis.
- [ ] Les paires et frontières tokenisées sont auditées.
- [ ] Les scores relatifs et générations sont évalués séparément.
- [ ] Les métriques de longueur et régression accompagnent la préférence.
- [ ] Policy, référence, config et sorties sont rechargeables et traçables.

## Fiche mémo

```text
Apprendre la préférence visée, puis la vérifier sur autre chose que la loss optimisée.
```

Le [projet 61](61_projet_6_rl_verifiable.md) ajoute une collecte online avec oracle.
