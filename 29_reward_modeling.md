# Module 29 — Apprendre une fonction de récompense et tester ses limites

## 1. Ce que tu vas apprendre

Un **reward model** attribue un score à une réponse dans son contexte. Il peut apprendre à ordonner des réponses à partir de préférences, puis servir de signal pour une autre optimisation.

Tu vas calculer une loss pairwise, entraîner une petite tête de score, mesurer l’ordre sur des paires et comprendre pourquoi un score élevé n’est pas une vérité garantie.

**Prérequis :** modules 7, 13 et 27 à 28.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python et PyTorch sur CPU. Aucun modèle à télécharger pour le laboratoire.

## 2. Ne pas confondre les trois modèles

| Modèle | Produit principalement | Rôle dans les expériences étudiées |
| --- | --- | --- |
| Policy | Distribution de tokens/réponses | Générer et être adaptée |
| Référence DPO | Log-probabilités de réponses | Ancrer une comparaison relative |
| Reward model | Un score scalaire par contexte/réponse | Approximer un critère de préférence |

Un reward model textuel utilise souvent un réseau qui encode prompt et réponse puis une **reward head**, par exemple une projection vers une valeur.

Il faut choisir quelle représentation de séquence est notée. Une dernière position valide n’est pas la dernière case PAD d’un batch. Le template, le padding, EOS et les longueurs doivent être cohérents au train comme au scoring.

Un juge qui génère un commentaire et une note est un autre protocole possible ; sa sortie n’est pas automatiquement une reward head ni un score calibré.

## 3. La loss pairwise

Pour une réponse préférée `y+` et une moins préférée `y−` au même contexte :

\[
d = r_\phi(x,y^+) - r_\phi(x,y^-)
\]
\[
L = -\log\sigma(d)
\]

`φ` désigne les paramètres du reward model. La loss encourage un score plus élevé pour chosen.

| Différence d | Loss environ |
| --- | --- |
| +5 | 0.0067 |
| 0 | 0.6931 |
| −5 | 5.0067 |

Une paire déjà correctement ordonnée peut encore avoir une loss positive. L’accuracy ne change plus une fois le bon ordre atteint, tandis que la loss peut encore pousser la marge à grandir.

Sur des paires parfaitement séparables, les scores peuvent croître sans limite utile si aucune contrainte ne les stabilise. Évalue les marges et la généralisation, pas seulement la loss train.

## 4. Pourquoi les valeurs absolues sont ambiguës

Ajouter une même constante c aux deux scores conserve leur différence :

```text
chosen=1.8, rejected=0.2 → difference=1.6
chosen=101.8, rejected=100.2 → difference=1.6
```

La loss pairwise seule ne fixe donc pas une origine absolue du score. Multiplier tous les scores n’est en revanche pas une invariance de cette loss : les marges et pénalités changent.

Un reward de 4 ne signifie pas « deux fois meilleur que 2 ». `sigmoid(différence)` peut être interprétée comme une probabilité de préférence **selon ce modèle de comparaison**, pas comme une probabilité universelle de vérité.

La **calibration** examine l’accord entre probabilités annoncées et fréquences observées sur un jeu réservé. Elle demande des données et une procédure distinctes de la seule accuracy d’ordre.

## 5. Laboratoire autonome — Une tête de score sur des caractéristiques contrôlées

Pour pouvoir tout inspecter, nous fournissons deux caractéristiques numériques : un indicateur de conformité déterminé pour l’exercice et une mesure artificielle de longueur.

Ces indicateurs sont connus par construction. Le réseau ne lit pas le français et ne découvre pas l’exactitude d’une réponse ; il apprend seulement à pondérer ces entrées. C’est une miniature du mécanisme de reward modeling, pas un évaluateur de vérité textuelle.

```python
import torch
import torch.nn.functional as F
from torch import nn

for difference in [5.0, 0.0, -5.0]:
    print('Difference=', difference, 'loss=', -F.logsigmoid(torch.tensor(difference)).item())

# Chaque ligne choisie est comparee a la ligne rejetee du meme indice.
chosen = torch.tensor([[1., 0.], [1., 1.], [1., 0.5]])
rejected = torch.tensor([[0., 1.], [0., 0.], [0., 0.5]])
validation_chosen = torch.tensor([[1., 2.], [1., 0.]])
validation_rejected = torch.tensor([[0., 0.], [0., 2.]])
model = nn.Linear(2, 1)
nn.init.zeros_(model.weight)
nn.init.zeros_(model.bias)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)


def compare(left, right):
    with torch.no_grad():
        difference = model(left).squeeze(-1) - model(right).squeeze(-1)
        return (-F.logsigmoid(difference).mean().item(),
                (difference > 0).float().mean().item(), difference)


before = compare(validation_chosen, validation_rejected)
for step in range(100):
    optimizer.zero_grad(set_to_none=True)
    difference = model(chosen).squeeze(-1) - model(rejected).squeeze(-1)
    loss = -F.logsigmoid(difference).mean()
    loss.backward()
    optimizer.step()
after = compare(validation_chosen, validation_rejected)
print('Validation avant : loss, accuracy, marges =', before)
print('Validation apres :', after)
print('Poids appris :', model.weight.detach().tolist())

with torch.no_grad():
    left = model(validation_chosen)
    right = model(validation_rejected)
    original = -F.logsigmoid(left - right).mean()
    shifted = -F.logsigmoid((left + 100) - (right + 100)).mean()
print('Meme loss apres decalage commun :', torch.allclose(original, shifted, atol=1e-6))
```

L’accuracy initiale vaut **0** avec la règle stricte `d>0`, car les scores initialisés à zéro sont ex æquo. Après l’apprentissage, les deux paires de validation contrôlées doivent être correctement ordonnées, avec une loss plus basse.

La construction des paires de train équilibre ici l’indice de longueur : il n’est pas toujours favorable à chosen. La tête peut apprendre la caractéristique de conformité. Un vrai modèle ne reçoit pas gratuitement un indicateur fiable de vérité : c’est précisément une difficulté masquée par cette miniature.

## 6. Passer aux réponses textuelles

Pour un reward model réel :

1. Prépare les paires avec la grille et les groupes du module 27.
2. Charge un encodeur ou modèle de séquence avec une tête scalaire adaptée. Vérifie la dimension de sortie et la position notée.
3. Formate **contexte et réponse** de manière identique dans les deux branches.
4. Calcule les deux scores puis la loss pairwise, ou utilise un trainer compatible avec ce format et cette loss.
5. Mesure accuracy, marges et résultats par catégorie sur des groupes réservés.
6. Inspecte les paires mal classées et les raccourcis avant d’utiliser ce score pour optimiser une policy.

Les trainers de reward modeling proposent plusieurs objectifs et formats selon les versions. Le nom `RewardTrainer` ne dispense pas de vérifier qu’on exécute bien la loss pairwise expliquée ici.

## 7. Audit de robustesse

Un bon score moyen peut cacher un échec systématique sur un type de réponse. Teste notamment :

| Perturbation | Question à poser |
| --- | --- |
| Réponse correcte raccourcie | Le modèle pénalise-t-il la concision utile ? |
| Réponse fausse mais plus longue | Confond-il détails et qualité ? |
| Changement de ponctuation/style | Le score dépend-il d’indices superficiels ? |
| Ajout d’une affirmation de confiance | Confond-il assurance et exactitude ? |
| Changement de domaine | La préférence apprise se transfère-t-elle ? |

Définis les transformations qui devraient préserver l’ordre et celles qui changent réellement la qualité. Toute variation du score n’est pas une faiblesse : certaines modifications modifient la réponse utile.

## 8. Utiliser le reward après son apprentissage

Dans un pipeline RL classique, on gèle le reward model lorsqu’il fournit un signal à la policy. La policy peut alors trouver des réponses que le reward note haut tout en manquant l’intention : c’est un risque de **reward hacking**.

Des rewards plus hauts pendant l’optimisation ne prouvent donc pas seuls une amélioration réelle. Il faut des évaluations distinctes de la fonction optimisée et des tests hors des paires initiales.

Un vérificateur déterministe peut remplacer ou compléter un reward appris sur des tâches vérifiables. Il possède aussi ses erreurs possibles, notamment de parsing ou de définition de réussite.

## 9. Expériences guidées

1. Dans le laboratoire, rends chosen toujours plus long et retire l’indicateur de conformité. Observe le raccourci appris puis teste une réponse préférée plus courte.
2. Inverse une paire et mesure l’effet sur les marges. Le modèle suit les annotations, même si elles sont incohérentes.
3. Continue davantage de steps et compare accuracy et marge. Explique pourquoi elles peuvent évoluer différemment.
4. Sur un corpus textuel, conserve dix paires difficiles et les raisons d’erreur ; propose une correction de données plutôt qu’un simple entraînement plus long.

## 10. Exercices avec corrigés

### Exercice 1 — Loss

Quelle différence donne la plus faible loss : −2, 0 ou +2 ?

<details>
<summary>Corrigé</summary>

**+2**, car sigmoid(d) est alors plus grande et sa log-pénalité plus petite. La loss veut chosen au-dessus de rejected.

</details>

### Exercice 2 — Origine du score

Pourquoi peut-on ajouter 100 aux deux scores sans changer la loss ?

<details>
<summary>Corrigé</summary>

Seule leur différence intervient : `(r+ +100)−(r− +100)=r+−r−`. L’origine absolue n’est pas identifiée par cet objectif seul.

</details>

### Exercice 3 — Accuracy parfaite

100 % d’accuracy train prouve-t-il que les scores sont calibrés et robustes ?

<details>
<summary>Corrigé</summary>

Non. Cela montre seulement le bon ordre sur ces paires avec la convention choisie. Calibration, généralisation et résistance aux raccourcis demandent d’autres mesures.

</details>

### Exercice 4 — Référence DPO

Pourquoi ne faut-il pas utiliser « reward model » et « référence DPO » comme synonymes ?

<details>
<summary>Corrigé</summary>

Le premier apprend un score de préférence pour une réponse. La seconde fournit des log-probabilités d’ancrage d’une policy de référence. Leur rôle et leurs sorties diffèrent.

</details>

## 11. Validation — avant de continuer

- [ ] Je distingue policy, référence et reward model.
- [ ] Je calcule `−logsigmoid(r_chosen−r_rejected)`.
- [ ] Je sais expliquer l’invariance à une constante et ses limites.
- [ ] Je retrouve l’apprentissage de la tête contrôlée sans lui attribuer de compréhension textuelle.
- [ ] Je mesure ordre, marges et erreurs par catégorie sur validation.
- [ ] Je propose des tests de raccourcis et une évaluation indépendante du reward optimisé.

## Fiche mémo

```text
Un reward model apprend une fonction de score à partir de comparaisons.
Bon ordre sur train, calibration et qualité réelle sont trois questions différentes.
Un score optimisable reste un signal imparfait à auditer.
```

Le [module 30](30_fondements_du_reinforcement_learning.md) montre comment une policy peut apprendre à partir des scores de ses actions.

## Références

[TRL : reward modeling](https://huggingface.co/docs/trl/en/reward_trainer), [apprentissage avec feedback humain](https://arxiv.org/abs/2203.02155).
