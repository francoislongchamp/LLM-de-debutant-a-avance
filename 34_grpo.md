# Module 34 — Comprendre et observer une optimisation relative par groupes

## 1. Ce que tu vas apprendre

**GRPO**, *Group Relative Policy Optimization*, construit un signal à partir de plusieurs réponses au même prompt. Les réponses sont notées, puis comparées à leur groupe pour former des avantages.

Tu vas calculer ces avantages, distinguer ancienne policy et référence, puis entraîner une miniature avec groupes, ratio et clipping.

**Prérequis :** modules 30 à 33 ; log-probabilités, reward et moyenne de groupe.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python et PyTorch, CPU. Le pilote ne télécharge ni LLM ni corpus.

## 2. Pourquoi plusieurs réponses au même prompt ?

Un groupe de rewards `[0,0,1,0.5]` a une moyenne de 0.375. Les deux premiers résultats sont en dessous ; les deux derniers au-dessus.

Une version de l’avantage est :

\[
\widehat A_i = \frac{r_i-\mu_{groupe}}{\sigma_{groupe}+\epsilon}
\]

La standardisation et ses variantes doivent être annoncées. Des implémentations proposent d’autres échelles ou agrégations ; « GRPO » ne désigne pas une unique configuration immuable.

Un groupe uniformément faux et un groupe uniformément correct donnent tous deux un avantage centré nul. Ils n’ont pourtant pas la même réussite absolue : logge cette réussite séparément.

## 3. Trois distributions à ne pas fusionner

| Distribution | Rôle |
| --- | --- |
| Policy entraînée | Celle dont les paramètres changent |
| Ancienne policy de collecte | Celle qui a produit les réponses du lot ; dénominateur du ratio |
| Référence | Ancre fixe pour une éventuelle pénalité de dérive |

Le ratio de l’objectif proximal compare nouvelle et ancienne policy **sur les mêmes actions observées**. La référence fixe n’est pas interchangeable avec l’ancienne policy qui est renouvelée à la collecte suivante.

GRPO ne nécessite pas le critic appris utilisé par un PPO classique avec value baseline ; il utilise une information relative au groupe. Une reward vérifiable peut fournir les scores sans reward model.

## 4. Lire l’objectif simplifié

Pour une action par réponse, un surrogate courant utilise :

\[
\rho_i = \exp(\log\pi_\theta(a_i|x)-\log\pi_{old}(a_i|x))
\]
\[
J = mean\left[\min(\rho_i\widehat A_i,
clip(\rho_i,1-\varepsilon,1+\varepsilon)\widehat A_i)\right]
-\beta D_{KL}(\pi_\theta\|\pi_{ref})
\]

On minimise `−J`. Le clipping limite une incitation du surrogate ; il n’impose pas une borne dure sur toutes les probabilités. La KL est un terme distinct, détaillé au module 36.

Avec un LLM, chaque réponse contient plusieurs actions de tokens. Il faut préciser masques, agrégation token/séquence et traitement des fins tronquées : notre cas à une action supprime ces difficultés pour voir le noyau du calcul.

## 5. Laboratoire autonome — Une miniature groupée à deux contextes

La policy choisit parmi trois réponses déjà fournies. Pour le contexte 0, l’action correcte est 0 ; pour le contexte 1, c’est 1. Elle n’apprend pas le français ni une procédure de raisonnement.

Chaque tour collecte de nouveaux groupes, puis fait deux updates sur ces actions avec l’ancienne log-probabilité figée. Une référence uniforme sert uniquement à illustrer la pénalité KL exacte dans ce petit espace.

```python
import torch

torch.manual_seed(42)
logits = torch.nn.Parameter(torch.zeros(2, 3))
optimizer = torch.optim.Adam([logits], lr=0.05)
correct = torch.tensor([0, 1])
reference = torch.full((2, 3), 1 / 3)
group_size = 16
beta = 0.02
clip_epsilon = 0.2

for iteration in range(1, 81):
    with torch.no_grad():
        old_distribution = torch.distributions.Categorical(logits=logits.detach())
        actions = old_distribution.sample((group_size,)).T
        old_logp = old_distribution.log_prob(actions.T).T
        rewards = (actions == correct[:, None]).float()
        std = rewards.std(dim=1, correction=0, keepdim=True)
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (std + 1e-8)

    for _ in range(2):
        log_probs = logits.log_softmax(dim=-1)
        new_logp = log_probs.gather(1, actions)
        ratio = (new_logp - old_logp).exp()
        unclipped = ratio * advantages
        clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
        probs = log_probs.exp()
        kl = (probs * (log_probs - reference.log())).sum(dim=-1).mean()
        loss = -torch.minimum(unclipped, clipped).mean() + beta * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if iteration == 1 or iteration % 20 == 0:
        success = logits.detach().softmax(-1).gather(1, correct[:, None]).mean().item()
        print('tour=', iteration, 'reward collecte=', rewards.mean().item(),
              'reussite attendue=', round(success, 4),
              'groupes sans contraste=', int((std.squeeze(-1) == 0).sum().item()))

final_success = logits.detach().softmax(-1).gather(1, correct[:, None]).mean().item()
print('Reussite finale attendue :', final_success)
```

La réussite attendue commence à 1/3 et doit augmenter nettement sur cette tâche facile. Les groupes sans contraste deviennent possibles même quand la policy réussit bien : presque toutes les réponses d’un groupe peuvent être correctes.

Le nombre de générations est `80 × 2 × 16 = 2560` actions collectées, et le nombre d’updates est 160. Ce ne sont pas des tokens de LLM ni des exemples distincts de connaissance.

## 6. Ce que le pilote ne remplace pas

Il s’agit d’une miniature inspirée du mécanisme groupé proximal, pas d’une reproduction complète d’un entraînement GRPO de LLM ni de toutes ses variantes.

Pour un vrai `GRPOTrainer`, vérifie notamment : format des prompts, interface des reward functions, nombre de générations par prompt, contraintes de batch, longueur maximale, variante de loss, règle de normalisation, référence/KL et mode de génération.

N’importe quelle fonction `reward(text)` n’est pas automatiquement compatible avec la signature attendue par ta version du trainer. Inspecte quelques completions et leurs métadonnées avant de lancer le long run.

## 7. Protocole pour une tâche vérifiable réelle

1. Réutilise le parser, oracle et tests du module 31.
2. Sépare les familles de problèmes et mesure la policy initiale.
3. Conserve les groupes complets avec prompt, réponses, scores et longueurs.
4. Mesure taux de réussite absolu et part de groupes sans contraste.
5. Lance un petit budget online ; vérifie les métriques par groupe et le coût de génération.
6. Évalue un holdout fixe avec un critère indépendant et inspecte les hauts rewards.

Plus de générations peut donner davantage de contraste ou de couverture, au prix du temps et de la mémoire. Ce n’est pas une garantie de meilleure qualité à compute égal.

## 8. Expériences guidées

- Recalcule les avantages de `[0,1,1,0]` puis `[1,1,1,1]` avec l’écart-type population.
- Compare group_size=4 et 16 sur plusieurs seeds, à nombre de générations annoncé ; ne confonds pas nombre de tours et coût total.
- Retire seulement la standardisation par l’écart-type et explique le changement d’échelle.
- Change beta en gardant le reste fixe, puis mesure réussite et KL au lieu de juger seulement la loss.

## 9. Exercices avec corrigés

### Exercice 1 — Avantages

Pour `[0,1,1,0]`, quels avantages standardisés avec sigma population ?

<details>
<summary>Corrigé</summary>

Moyenne 0.5 et écart-type 0.5 : avantages environ **[-1,1,1,-1]**, aux effets de l’epsilon près.

</details>

### Exercice 2 — Deux ancrages

Pourquoi la référence fixe n’est-elle pas toujours le bon dénominateur du ratio proximal ?

<details>
<summary>Corrigé</summary>

Le ratio porte sur la policy qui a réellement collecté ce batch. La référence peut rester inchangée pendant que les collectes utilisent des policies successives.

</details>

### Exercice 3 — Groupe parfait

Tous les rewards valent 1 et les avantages valent zéro. Est-ce un échec de la tâche ?

<details>
<summary>Corrigé</summary>

Non : le groupe réussit, mais cette règle relative ne distingue pas ses réponses. Il faut logguer réussite absolue et contraste séparément.

</details>

### Exercice 4 — Online ou offline ?

Quelle différence centrale avec DPO standard sur des paires fixes ?

<details>
<summary>Corrigé</summary>

La boucle étudiée ici collecte et score de nouvelles actions de la policy au cours du training. DPO standard optimise des paires précollectées sans cette collecte online obligatoire dans la boucle.

</details>

## 10. Validation — avant de continuer

- [ ] Je calcule un avantage relatif avec une convention d’écart-type annoncée.
- [ ] Je distingue policy, ancienne policy et référence.
- [ ] Je peux suivre collecte, updates et groupes sans contraste dans le pilote.
- [ ] Je sais ce qui manque pour passer à des séquences de LLM.
- [ ] Je mesure coût de génération et succès holdout indépendamment du reward train.

## Fiche mémo

```text
Plusieurs réponses au même prompt → scores → avantages de groupe
→ ratio sur les actions collectées → update → nouvelle collecte.
Reward absolu et contraste relatif sont deux informations différentes.
```

Le [module 35](35_ppo.md) introduit une value baseline apprise et détaille le clipping PPO.

## Références

[DeepSeekMath : introduction de GRPO](https://arxiv.org/abs/2402.03300), [TRL : GRPOTrainer et variantes](https://huggingface.co/docs/trl/en/grpo_trainer).
