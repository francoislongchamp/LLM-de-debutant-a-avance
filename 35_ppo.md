# Module 35 — Comprendre le clipping PPO et le rôle du critic

## 1. Ce que tu vas apprendre

**PPO**, *Proximal Policy Optimization*, collecte des trajectoires puis optimise un objectif qui modère certaines incitations à changer la policy trop fortement sur ce lot.

Tu vas calculer les deux branches du surrogate, distinguer reward et value, puis entraîner une petite policy avec un critic.

**Prérequis :** modules 30, 34 et notions de gradients.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch sur CPU ; aucun téléchargement.

## 2. Quatre rôles dans un pipeline RLHF classique

| Élément | Produit ou rôle |
| --- | --- |
| Actor/policy | Actions ou tokens |
| Critic/value | Return attendu depuis une situation |
| Reward function/model | Note des conséquences ou réponses |
| Ancienne policy | Probabilité des actions au moment de leur collecte |

Une référence fixe pour une pénalité KL peut aussi être présente. Elle est encore distincte de l’ancienne policy, renouvelée à chaque lot de trajectoires.

Le critic estime une espérance de return, pas la vérité d’une réponse donnée. Le reward et la value peuvent produire chacun un scalaire sans représenter la même quantité.

## 3. Refaire le ratio

Si une action avait une probabilité 0.20 et possède maintenant 0.30 :

\[
\rho = \frac{0.30}{0.20}=1.5
\]

Pour la stabilité numérique, on calcule souvent `exp(logp_new−logp_old)` sur la même action et le même contexte.

L’ancienne log-probabilité doit rester fixe pendant les updates qui réutilisent ces données. La recalculer à chaque mini-update avec les poids courants rendrait le ratio artificiellement proche de 1.

## 4. Le minimum des deux branches

\[
J_{clip}=mean\left[\min(\rho\widehat A,
clip(\rho,1-\varepsilon,1+\varepsilon)\widehat A)\right]
\]

On maximise J, ou minimise −J. Avec epsilon=0.2 :

| Ratio | Advantage | rho×A | clip(rho)×A | Minimum |
| --- | --- | --- | --- | --- |
| 1.5 | +1 | 1.5 | 1.2 | 1.2 |
| 1.5 | −1 | −1.5 | −1.2 | −1.5 |
| 0.5 | −1 | −0.5 | −0.8 | −0.8 |
| 0.9 | +1 | 0.9 | 0.9 | 0.9 |

Pour un avantage positif, l’incitation directe cesse de croître au-delà du bord haut. Pour un avantage négatif, le côté utile du clipping est inversé.

Il ne s’agit pas de forcer toutes les probabilités dans une boîte. Paramètres partagés, autres actions et autres termes de loss peuvent continuer à changer la distribution. Le clipping n’est pas une garantie de KL bornée.

## 5. Une value loss distincte

Un critic peut être ajusté avec une erreur entre sa prédiction `V(s)` et une estimation du return. Une forme simple est une MSE.

Le signal d’actor peut utiliser `G−V(s)` traité comme constant pendant son calcul de gradient. Dans des trajectoires longues, des estimateurs comme GAE ajoutent des détails de bootstrap et d’actualisation ; notre tâche à une seule action n’en a pas besoin.

Une mauvaise value baseline peut augmenter la variance ou dégrader l’apprentissage. Elle ajoute aussi calcul, mémoire et hyperparamètres à vérifier.

## 6. Laboratoire autonome — PPO miniature sur une seule question

La policy choisit entre deux actions : la première reçoit 1, la seconde 0. Le critic est un unique scalaire car il n’y a qu’un état. Ce réseau ne génère pas de texte et ne démontre pas un RLHF complet.

```python
import torch

torch.manual_seed(42)
actor_logits = torch.nn.Parameter(torch.zeros(2))
critic = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.Adam([actor_logits, critic], lr=0.03)
epsilon = 0.2

for rollout in range(1, 51):
    with torch.no_grad():
        old = torch.distributions.Categorical(logits=actor_logits.detach())
        actions = old.sample((128,))
        old_logp = old.log_prob(actions)
        returns = (actions == 0).float()
        advantages = returns - critic.detach()

    for _ in range(4):
        distribution = torch.distributions.Categorical(logits=actor_logits)
        ratio = (distribution.log_prob(actions) - old_logp).exp()
        surrogate = torch.minimum(ratio * advantages,
                                  ratio.clamp(1 - epsilon, 1 + epsilon) * advantages)
        actor_loss = -surrogate.mean()
        value_loss = (critic - returns).square().mean()
        loss = actor_loss + 0.5 * value_loss - 0.01 * distribution.entropy()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if rollout == 1 or rollout % 10 == 0:
        print('collecte=', rollout, 'reward lot=', returns.mean().item(),
              'probabilite action correcte=', actor_logits.detach().softmax(-1)[0].item(),
              'value=', critic.item())

ratios = torch.tensor([1.5, 1.5, 0.5, 0.9])
advantages_example = torch.tensor([1.0, -1.0, -1.0, 1.0])
objective = torch.minimum(ratios * advantages_example,
                         ratios.clamp(0.8, 1.2) * advantages_example)
print('Table du surrogate :', objective.tolist())
```

Tu dois retrouver la table **[1.2,−1.5,−0.8,0.9]**, aux arrondis près. La probabilité de l’action récompensée doit augmenter nettement depuis 0.5.

Il y a **50 collectes**, **6 400 actions échantillonnées** et **200 updates**. Le critic cherche une valeur proche du reward attendu de la policy, mais son ajustement ne sera pas exact à chaque instant.

## 7. Ce qui a été volontairement simplifié

Le pilote utilise un seul état, une action par épisode, aucun bootstrap, pas de modèle de reward appris et pas de référence KL. Il vérifie le surrogate, la value loss et la réutilisation limitée des collectes.

Un PPO de LLM doit traiter les fins d’épisode, tokens de padding, retours par token, value head, génération, distribution de référence éventuelle et coûts de plusieurs réseaux.

Les API PPO des frameworks ont évolué ; le noyau mathématique ci-dessus reste inspectable sans dépendre d’une ancienne signature de `PPOTrainer`. Pour une mise à l’échelle, fige une version et suis sa documentation d’ensemble, pas un mélange d’extraits de versions différentes.

## 8. Expériences guidées

1. Calcule la table avec un avantage de magnitude 2. Vérifie le sens de chaque branche.
2. Remplace la value baseline par une constante 0. Compare les résultats sur plusieurs seeds, sans conclure depuis un seul run.
3. Augmente le nombre d’updates sur une même collecte. Logge les ratios et la dérive ; le clipping ne rend pas une réutilisation illimitée sans risque.
4. Supprime l’entropy bonus dans une copie. Observe la concentration de la policy et distingue ce bonus du clipping.

## 9. Exercices avec corrigés

### Exercice 1 — Ratio

p_old=0.1 et p_new=0.08 : quel ratio ?

<details>
<summary>Corrigé</summary>

**0.8**. Il mesure le changement relatif de probabilité de cette action dans ce contexte.

</details>

### Exercice 2 — Advantage négatif

Pourquoi min(1.5×−1,1.2×−1) vaut-il −1.5 et non −1.2 ?

<details>
<summary>Corrigé</summary>

Le minimum se prend après multiplication. Augmenter une action défavorable reste pénalisé ; le clipping ne doit pas protéger ce déplacement défavorable de la même manière qu’un gain excessif.

</details>

### Exercice 3 — Deux clippings

PPO clipping et gradient clipping limitent-ils la même quantité ?

<details>
<summary>Corrigé</summary>

Non. Le premier intervient dans le surrogate fondé sur un ratio de probabilités ; le second limite une norme de gradients avant l’optimizer step.

</details>

### Exercice 4 — Critic ou correcteur ?

Le critic prédit 0.8 pour un préfixe. Est-ce un score certifiant que la réponse finale est vraie ?

<details>
<summary>Corrigé</summary>

Non. Il estime un return attendu sous la tâche et la policy. Ce n’est pas un oracle factuel ni nécessairement le score d’une réponse terminée.

</details>

## 10. Validation — avant de continuer

- [ ] Je calcule les deux branches pour des avantages positifs et négatifs.
- [ ] Je distingue critic, reward, ancienne policy et référence.
- [ ] Je conserve l’ancienne log-probabilité pendant les updates d’un lot.
- [ ] Je peux commenter les compteurs et la value du pilote.
- [ ] Je sais pourquoi clipping et proximité globale ne sont pas synonymes.

## Fiche mémo

```text
Collecter → estimer returns/advantages → figer old_logp
→ quelques updates actor/critic → nouvelle collecte.
Le surrogate clippe une incitation, pas toute la distribution.
```

Le [module 36](36_kl_divergence_et_controle_de_derive.md) mesure une divergence explicite entre distributions.

## Référence

[Article PPO](https://arxiv.org/abs/1707.06347) — objectifs proximaux et optimisation sur collectes.
