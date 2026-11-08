# Module 61 — Projet 6 : une boucle online avec correcteur vérifié

## 1. Ce que tu vas construire

Un projet RL vérifiable relie générateur de problèmes, oracle, parser, policy, collecte online et évaluation réservée. Le score optimisé doit correspondre à la réussite voulue.

Tu vas d’abord exécuter une miniature complète qui choisit entre deux procédures fournies, puis définir l’expérience où un LLM produit réellement les réponses.

**Prérequis :** modules 30 à 36, 51 à 55.

**Temps indicatif :** pilote court ; plusieurs heures ou davantage pour un vrai modèle avec génération.

**Matériel :** Python et PyTorch sur CPU pour la miniature. GRPO de LLM demande ensuite l’environnement et les ressources adaptés.

## 2. Contrat de tâche et séparation

Tâche choisie : addition d’entiers positifs, réponse JSON stricte `{"answer": entier}`. Le générateur conserve vérité et métadonnées hors du prompt transmis à la policy.

Pour ne pas répartir `a+b` et `b+a` entre train et test comme deux problèmes indépendants, nous groupons par paire canonique d’opérandes. Des seeds différentes seules ne garantissent pas des ensembles disjoints.

Les groupes ne doivent pas traverser les splits. Si tu veux tester une généralisation à des nombres plus grands ou à un autre type d’opération, construis aussi cette séparation et annonce qu’elle change la distribution.

## 3. Ce que le pilote apprend réellement

La miniature reçoit deux actions possibles : utiliser une procédure d’addition déjà codée, ou une procédure de soustraction. Elle apprend à favoriser l’action récompensée.

Le programme contient donc déjà la procédure correcte parmi les candidats. Le pilote ne démontre **ni apprentissage de l’addition, ni raisonnement d’un LLM**. Il valide groupes, vérificateur, collecte, gradients et archivage.

La réussite attendue est calculable exactement sur ces deux actions. Elle est distincte d’un échantillonnage de réponses de texte libre.

## 4. Laboratoire autonome — Tests, splits, collecte et policy gradient

```python
import json
import random
from itertools import combinations_with_replacement
from pathlib import Path
import torch


def parse_object(pairs):
    if len(dict(pairs)) != len(pairs):
        raise ValueError('Cle dupliquee')
    return dict(pairs)


def reward(text, a, b):
    try:
        obj = json.loads(text, object_pairs_hook=parse_object)
    except ValueError:
        return 0.0
    if not isinstance(obj, dict) or set(obj) != {'answer'} or type(obj['answer']) is not int:
        return 0.0
    return float(obj['answer'] == a + b)


def candidate(a, b, action):
    # Deux procedures deja fournies. La policy apprend seulement a choisir entre elles.
    return json.dumps({'answer': a + b if action == 0 else a - b})


tests = [(json.dumps({'answer': a + b}), a, b, 1.0) for a in range(5) for b in range(5)]
tests += [('185', 127, 58, 0.0), ('{"answer":184}', 127, 58, 0.0),
          ('{"answer":true}', 1, 0, 0.0), ('{"answer":"185"}', 127, 58, 0.0),
          ('{"answer":185,"answer":184}', 127, 58, 0.0), ('[]', 1, 2, 0.0)]
for text, a, b, expected in tests:
    assert reward(text, a, b) == expected

pool = list(combinations_with_replacement(range(1, 21), 2))
random.Random(42).shuffle(pool)
train, validation, test = pool[:100], pool[100:130], pool[130:]
assert set(train).isdisjoint(validation) and set(train).isdisjoint(test) and set(validation).isdisjoint(test)
torch.manual_seed(42)
rng = random.Random(123)
logits = torch.nn.Parameter(torch.zeros(2))
optimizer = torch.optim.Adam([logits], lr=0.05)


def expected_success(tasks):
    probabilities = logits.detach().softmax(-1).tolist()
    return sum(sum(probabilities[action] * reward(candidate(a, b, action), a, b)
                   for action in range(2)) for a, b in tasks) / len(tasks)


before_validation = expected_success(validation)
records = []
for step in range(1, 81):
    tasks = rng.choices(train, k=32)
    distribution = torch.distributions.Categorical(logits=logits)
    actions = distribution.sample((len(tasks),))
    answers = [candidate(a, b, action) for (a, b), action in zip(tasks, actions.tolist())]
    rewards = torch.tensor([reward(answer, a, b) for (a, b), answer in zip(tasks, answers)])
    advantages = rewards - 0.5
    loss = -(advantages.detach() * distribution.log_prob(actions)).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    for (a, b), action, answer, value in zip(tasks, actions.tolist(), answers, rewards.tolist()):
        records.append({'step': step, 'group': [a, b], 'action': action, 'answer': answer, 'reward': value})
    if step % 20 == 0:
        print('step=', step, 'train reward=', rewards.mean().item(), 'validation attendue=', expected_success(validation))

root = Path('outputs/project61')
root.mkdir(parents=True, exist_ok=True)
(root / 'samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
(root / 'split_manifest.json').write_text(json.dumps({'train': train, 'validation': validation, 'test': test}), encoding='utf-8')
torch.save({'logits': logits.detach(), 'reward_version': 'strict_integer_sum-v1'}, root / 'selector.pt')
print('Tests reward :', len(tests), 'groupes train/val/test :', len(train), len(validation), len(test))
print('Validation avant/apres :', before_validation, expected_success(validation))
print('Test final attendu :', expected_success(test))
```

Le script doit passer **31 tests de reward**, séparer **100/30/80 groupes** et collecter **2560 actions** au cours de 80 updates.

La réussite attendue initiale vaut 0.5 et doit augmenter nettement dans ce pilote. Les actions ne modifient pas la définition de l’oracle ; c’est la probabilité de choisir la procédure qui change.

Les cas du correcteur couvrent des erreurs utiles, mais pas tous les cas limites du module 31. Pour ton projet réel, reprends et étends ce contrat au lieu de prétendre qu’un nombre de tests garantit son exhaustivité.

## 5. Passer à une vraie génération de LLM

1. Charge la policy préentraînée/SFT et son template exact.
2. Donne seulement le problème et le contrat de sortie, pas la cible dans le contexte.
3. Échantillonne plusieurs réponses et archive le texte avant parsing.
4. Applique le correcteur testé, avec raisons d’échec séparées.
5. Mesure succès initial, format, longueur et groupes sans contraste.
6. Choisis GRPO ou une autre optimisation compatible, avec interface de reward vérifiée.
7. Fais un budget court avant l’échelle cible ; mesure coût de collecte et coût d’update séparément.
8. Évalue validation sans updates, sélectionne selon le protocole puis utilise le test final.

Le noyau du pilote est un policy gradient à une étape, pas une implémentation complète de GRPOTrainer. Les groupes, ratios, masques de tokens et contraintes de l’algorithme choisi doivent être traités comme au module 34.

## 6. Si la reward ne donne aucun contraste

Avec presque tous les scores à zéro : vérifie d’abord parser, format, troncature et oracle sur les sorties réelles. La tâche peut aussi être trop difficile pour le modèle de départ.

Un curriculum ou un petit SFT peuvent augmenter la réussite initiale ; ce sont des interventions à mesurer et à compter dans le budget. Si toutes les réponses réussissent déjà, augmente la difficulté selon une règle définie avant de déclarer un gain.

Le shaping de format peut enseigner seulement le format. Conserve l’exactitude globale et la mesure indépendante, pas seulement la somme des rewards.

## 7. Audit et livrables

Remets générateur et version, splits/groupes, correcteur et tests, baseline, configuration de collecte/training, échantillons bruts, composantes de reward, longueur, métriques KL si utilisées, checkpoints et procédure de reprise.

Inspecte les hauts rewards pour les exploits du contrat. Un juge indépendant ou un vérificateur de validation distinct peut aider à détecter une réussite artificielle, sans garantir seul une vérité universelle.

Le rapport doit distinguer réussite sur problèmes vus, nouvelles instances du même régime et changements de distribution. Le gain sur une famille ne prouve pas un raisonnement général.

## 8. Exercices avec corrigés

### Exercice 1 — Seeds

Pourquoi deux générateurs avec seeds différentes peuvent-ils produire les mêmes additions ?

<details>
<summary>Corrigé</summary>

Leurs tirages peuvent se recouvrir dans le même espace fini. Il faut vérifier et partitionner les groupes, pas seulement changer la graine.

</details>

### Exercice 2 — Procédure fournie

Pourquoi le succès du sélecteur ne prouve-t-il pas l’apprentissage de l’arithmétique ?

<details>
<summary>Corrigé</summary>

L’action correcte appelle déjà une addition programmée. La policy apprend laquelle des procédures fournies choisir, pas à produire la procédure ni le calcul elle-même.

</details>

### Exercice 3 — Sparse reward

Que vérifier avant d’assouplir le correcteur quand tout vaut zéro ?

<details>
<summary>Corrigé</summary>

Sorties brutes, parsing, format, longueurs/troncature, vérité de référence et difficulté. Assouplir aveuglément peut accepter des réponses fausses.

</details>

### Exercice 4 — Test final

Les résultats du test guident tes prochains réglages. Quel rôle ce test a-t-il commencé à jouer ?

<details>
<summary>Corrigé</summary>

Un rôle de développement/validation. Il faut reconnaître cette réutilisation et prévoir une estimation finale adaptée.

</details>

## 9. Critères de fin

- [ ] Correcteur et générateur sont versionnés et testés.
- [ ] Groupes et vérités ne fuient pas entre splits ni dans le prompt.
- [ ] Le taux initial et le contraste de groupes sont mesurés.
- [ ] Les générations online et leurs scores restent auditables.
- [ ] Succès réservé, format, longueur, coût et limites sont rapportés.
- [ ] Le pilote de sélection n’est pas présenté comme un LLM qui raisonne.

## Fiche mémo

```text
Un oracle clair, des groupes réservés et des sorties brutes avant le grand run.
Optimiser le reward ne suffit pas : vérifier la tâche réellement apprise.
```

Le [projet 62](62_projet_7_llm_from_scratch_de_bout_en_bout.md) assemble les composants du modèle de langage lui-même.
