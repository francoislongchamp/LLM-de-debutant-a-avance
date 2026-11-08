# Module 30 — Comprendre une première boucle de reinforcement learning

## 1. Ce que tu vas apprendre

Le SFT fournit une sortie à imiter. Le reinforcement learning, ou **apprentissage par renforcement**, optimise une policy à partir de récompenses liées à ses actions et à leurs conséquences.

Tu vas définir état, action, trajectoire, reward, return, value et advantage, puis entraîner réellement une petite policy à deux actions avec un gradient de politique.

**Prérequis :** modules 6 à 8 et 27 à 29 ; log-probabilités, gradients et distinction policy/reward.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python et PyTorch sur CPU. Aucun LLM ni environnement externe à télécharger.

## 2. Décrire une interaction sans anthropomorphisme

| Terme | Sens concret | Lecture token par token pour un LLM |
| --- | --- | --- |
| Policy `πθ` | Distribution d’actions conditionnée par ce qui est observé | Probabilités du prochain token |
| Observation | Information disponible pour choisir | Prompt et préfixe généré |
| État | Description de la situation dans le modèle de l’environnement | Le préfixe peut suffire dans une tâche simple ; des outils peuvent ajouter un état externe |
| Action | Choix effectué | Token choisi |
| Transition | Changement après l’action | Ajout du token ou réponse d’un outil |
| Trajectoire | Suite d’états/observations et d’actions | Génération jusqu’à la fin |
| Épisode | Interaction jusqu’à une condition terminale | Réponse terminée ou budget atteint |
| Reward | Score fourni par l’environnement ou l’évaluateur | Score de token, d’étape ou de réponse finale |

On peut aussi traiter une réponse entière comme une action dans un problème à une étape. Il faut annoncer le niveau d’analyse au lieu de mélanger ses compteurs.

Un état n’est pas toujours entièrement observable. Notre laboratoire choisit le cas le plus simple : une seule question et deux réponses possibles.

## 3. Reward et return

Un reward est un signal à un moment donné. Le **return** cumule des rewards futures, éventuellement actualisées :

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\]

`γ`, gamma, est le facteur d’actualisation. Avec `γ=1`, aucune réduction liée au délai n’est appliquée dans un épisode fini.

Exemple de rewards `[0,0,1]`, avec notre convention où `r_t` est celui associé à l’action t :

```text
Gamma 1.0 : return a t=0 = 1
Gamma 0.9 : return a t=0 = 0.81
```

Une récompense terminale unique peut informer toutes les actions de la réponse via leur return. Il n’est pas nécessaire de fabriquer une récompense indépendante pour chaque token, mais attribuer le mérite aux actions utiles reste difficile.

Un reward **sparse** apparaît rarement ou seulement à la fin. Un reward **dense** apporte davantage de signaux intermédiaires. Plus dense ne signifie pas automatiquement meilleur : il peut introduire des incitations incorrectes.

## 4. Value et advantage

La **value** `Vπ(s)` est l’espérance du return depuis l’état s en suivant la policy. Un critic peut chercher à l’estimer.

La valeur d’action `Qπ(s,a)` est le return attendu si l’on choisit a puis suit la policy. L’**advantage** compare cette valeur à la baseline de l’état :

\[
A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s)
\]

Une estimation simple peut utiliser `G−V(s)`. Ce n’est pas toujours « reward immédiat moins une moyenne » : dans un problème séquentiel, le return inclut les conséquences futures.

Dans un problème à une seule action, return et reward coïncident. Un reward de 1 comparé à une baseline 0.5 donne alors un signal positif de 0.5.

## 5. L’objectif et le détour par les log-probabilités

On veut augmenter le reward attendu :

\[
J(\theta)=\mathbb{E}_{y\sim\pi_\theta(.|x)}[R(x,y)]
\]

Le tirage discret d’un token ou d’une action ne se dérive pas comme une simple multiplication continue. Un estimateur de **policy gradient** utilise notamment :

\[
\nabla_\theta J \approx \frac1N\sum_i (G_i-b_i)\nabla_\theta\log\pi_\theta(a_i|s_i)
\]

Pour utiliser une descente de gradient, on minimise une loss avec le signe opposé :

\[
L = -\frac1N\sum_i \widehat A_i\log\pi_\theta(a_i|s_i)
\]

Ici, l’avantage sert de poids traité comme constant dans ce calcul. Une baseline appropriée, indépendante de l’action choisie conditionnellement à l’état, peut réduire la variance sans changer l’espérance du gradient de politique.

Si le poids est positif, la descente favorise l’action échantillonnée. S’il est négatif, elle la défavorise. Le mécanisme n’exige pas que le vérificateur soit dérivable.

## 6. Laboratoire autonome — Une policy qui choisit entre deux réponses

Question fixe : `17 × 18 ?`. Deux actions sont possibles : répondre `306` ou `308`. Le vérificateur accepte exactement la chaîne `306` après suppression des espaces de bord.

C’est un **bandit à une étape** : aucun réseau de langage, aucune généralisation à d’autres multiplications et aucun raisonnement ne sont appris. La policy apprend à choisir entre deux chaînes déjà fournies.

Prédis la probabilité initiale de chaque action et le sens attendu des updates.

```python
import torch


def reward(answer):
    return 1.0 if answer.strip() == '306' else 0.0


torch.manual_seed(42)
answers = ['306', '308']
logits = torch.nn.Parameter(torch.zeros(2))
optimizer = torch.optim.Adam([logits], lr=0.05)
print('Probabilites initiales :', logits.detach().softmax(-1).tolist())

for step in range(1, 81):
    distribution = torch.distributions.Categorical(logits=logits)
    actions = distribution.sample((64,))
    rewards = torch.tensor([reward(answers[index]) for index in actions.tolist()])
    # Baseline constante, independante de l'action. Pas de critic dans cette miniature.
    advantages = rewards - 0.5
    loss = -(advantages.detach() * distribution.log_prob(actions)).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step == 1 or step % 20 == 0:
        expected_reward = logits.detach().softmax(-1)[0].item()
        print('step=', step, 'reward echantillonne=', round(rewards.mean().item(), 4),
              'reward attendu exact=', round(expected_reward, 4), 'loss=', round(loss.item(), 4))

print('Probabilites finales :', logits.detach().softmax(-1).tolist())
print('Tests du verificateur :', [(text, reward(text)) for text in ['306', '308', '306 ', '306.0']])
```

Les probabilités commencent à **`[0.5,0.5]`**. Dans cette expérience, la probabilité de `306` doit augmenter nettement, généralement au-delà de 0.95 après 80 updates.

Le reward moyen échantillonné peut fluctuer. Le reward attendu exact vaut ici simplement `P(action 306)`, car cette action vaut 1 et l’autre 0. Nous pouvons le calculer exactement uniquement parce que le problème est minuscule et connu.

La loss de policy gradient n’est pas un score de qualité à comparer directement à une cross-entropy SFT. Elle peut être négative ; regarde le reward, la distribution et les comportements pertinents.

## 7. Pourquoi ce n’est pas un SFT déguisé dans le code

L’update utilise les **actions tirées par la policy** et les notes du vérificateur. Elle ne reçoit pas un tenseur de labels cibles à comparer directement avec `cross_entropy`.

La bonne réponse est connue par le vérificateur de notre exercice, mais elle intervient dans le calcul du score, pas comme une cible supervisée à chaque position de la policy.

Dans le RL online étudié ici, de nouvelles données proviennent de la policy courante. Il existe aussi du RL offline à partir de données précollectées : « RL » ne signifie donc pas universellement qu’il faut générer de nouvelles données à chaque update.

DPO standard, lui, utilise directement des paires offline et une autre loss, sans imposer cette boucle d’échantillonnage et de scoring pendant son entraînement.

## 8. Ce qui change avec un LLM

Une réponse peut contenir des centaines de décisions de tokens. Sa log-probabilité est une somme des log-probabilités des tokens, avec les bons masques et contextes.

Le reward peut venir d’un test automatique, d’un environnement avec outils ou d’un modèle appris. La policy peut modifier la distribution des réponses et découvrir des cas que le reward ne juge pas correctement.

Des algorithmes comme PPO ou GRPO ajoutent leurs propres estimateurs, comparaisons, contraintes et détails d’optimisation. Le laboratoire n’est ni une implémentation de PPO ni une implémentation de GRPO.

L’usage d’une référence, de pénalités de dérive ou de contraintes supplémentaires doit être compris dans l’objectif concerné, pas ajouté comme une formule universelle à tout RL.

## 9. Auditer le signal avant de choisir l’algorithme

| Signal observé | Problème possible |
| --- | --- |
| Tous les rewards sont 0 | Pas de contraste utile dans cet échantillon ; tâche trop dure ou vérificateur incorrect |
| Tous sont 1 | Tâche trop facile ou vérificateur trop permissif |
| Reward monte mais qualité baisse | Exploitation d’un raccourci ou objectif mal défini |
| Une réponse équivalente est rejetée | Contrat de format ou parsing à revoir |
| Un exploit reçoit 1 | Vérificateur manipulable |

Un batch uniforme peut encore produire du bruit avec certains estimateurs ; ne confonds pas mouvement des poids et signal utile. Avec des avantages centrés sur la moyenne du groupe, tous les rewards identiques peuvent donner des avantages nuls.

Pour `306.0`, notre reward vaut 0 : c’est cohérent avec un contrat de chaîne entière stricte, mais pas avec toutes les définitions d’équivalence numérique. Écris le contrat avant d’appeler le résultat « correct ».

## 10. Expériences guidées

1. Mets le reward à zéro pour toutes les actions et utilise une baseline nulle. Les gradients de cet objectif doivent être nuls : aucune information ne distingue les réponses.
2. Inverse volontairement le vérificateur pour récompenser `308`. La policy doit apprendre le mauvais choix selon l’intention initiale : elle suit le signal fourni.
3. Compare baseline 0 et 0.5 sur plusieurs seeds. Mesure la variabilité au lieu de conclure à partir d’une seule courbe.
4. Ajoute une troisième action contenant une réponse fausse mais reconnue par un vérificateur trop permissif. Décris comment le score peut être exploité.

## 11. Exercices avec corrigés

### Exercice 1 — Return

Rewards `[0,0,1]`, gamma=0.5 : quel return au début selon notre convention ?

<details>
<summary>Corrigé</summary>

`0 + 0.5×0 + 0.5²×1 = 0.25`. Avec gamma=1, il vaudrait 1.

</details>

### Exercice 2 — Advantage

Dans une tâche à une étape, reward=0 et baseline=0.5. Quel avantage et quelle direction pour l’action échantillonnée ?

<details>
<summary>Corrigé</summary>

Avantage **−0.5**. Dans la loss étudiée, la mise à jour tend à diminuer sa log-probabilité, sous les interactions avec les autres paramètres de la policy.

</details>

### Exercice 3 — Vérification ou loss ?

Pourquoi le reward 0/1 n’a-t-il pas besoin d’être dérivable ici ?

<details>
<summary>Corrigé</summary>

Le gradient passe par `log_prob` de l’action sous la policy. Le reward, traité comme un poids constant pour l’update, fournit le signal de préférence sur les actions échantillonnées.

</details>

### Exercice 4 — Généralisation

La policy choisit presque toujours `306`. A-t-elle appris à multiplier ?

<details>
<summary>Corrigé</summary>

Non. Elle a appris une distribution entre deux réponses fixes pour une question fixe. Apprendre une procédure ou généraliser à d’autres problèmes demanderait une autre tâche et une évaluation correspondante.

</details>

### Exercice 5 — Reward hacking

Le reward augmente mais un juge indépendant observe plus d’erreurs. Quelle hypothèse examiner ?

<details>
<summary>Corrigé</summary>

Le score optimisé peut être imparfait ou exploité. Il faut auditer sa définition, le vérificateur ou reward model et les nouvelles réponses, sans confondre reward optimisé et réussite réelle.

</details>

## 12. Validation — avant de continuer

- [ ] Je relie observation, action, trajectoire et épisode à un LLM.
- [ ] Je distingue reward, return, value et advantage.
- [ ] Je peux expliquer le signe de la loss de policy gradient.
- [ ] Je montre un changement de probabilité dans le bandit sans lui attribuer du raisonnement.
- [ ] Je distingue boucle online, données offline, SFT et DPO.
- [ ] Je définis le contrat d’un reward et ses modes d’échec.
- [ ] Je sais pourquoi reward moyen et qualité réelle peuvent diverger.

## Fiche mémo

```text
Policy → actions échantillonnées → conséquences et rewards
→ estimateur de gradient → update → nouvelles actions.
Le score optimisé doit être vérifié indépendamment.
Un exemple à une étape enseigne le mécanisme, pas toutes les difficultés d’un LLM.
```

Le [module 31](31_construire_une_reward_verifiable.md) approfondit la construction d’un vérificateur utile avant le choix d’un algorithme plus complexe.

## Références

[Spinning Up : gradient de politique](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) — estimateurs, returns et baselines ; [feedback humain pour modèles de langage](https://arxiv.org/abs/2203.02155).
