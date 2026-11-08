# Module 36 — Calculer une KL et savoir quelle dérive elle mesure

## 1. Ce que tu vas apprendre

Le clipping proximal, une pénalité KL et une limite de norme des gradients sont trois mécanismes différents. Ici, tu vas calculer une divergence entre distributions et interpréter ses estimateurs.

**Prérequis :** modules 6, 27, 34 et 35 ; logarithmes et probabilités.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et PyTorch, CPU.

## 2. Choisir les événements et la direction

Pour des distributions P et Q sur les **mêmes événements** :

\[
D_{KL}(P\|Q)=\sum_i P_i\ln(P_i/Q_i)
\]

Dans une comparaison de prochain token, les événements sont les tokens du même vocabulaire au même contexte. Comparer l’ID 42 de deux tokenizers incompatibles ne compare pas forcément le même événement.

P détermine la pondération. Échanger P et Q change généralement le résultat. La KL n’est donc pas une distance symétrique au sens d’une métrique.

Pour la KL exacte, les termes avec P=0 contribuent zéro. Si P>0 et Q=0, la divergence est infinie. Ajouter un epsilon change la mesure : il faut annoncer cette approximation plutôt que masquer le problème de support.

## 3. Un calcul discret

```text
Policy    P = [0.4, 0.5, 0.1]
Reference Q = [0.8, 0.1, 0.1]
```

Les trois contributions à `KL(P||Q)` sont :

```text
0.4 × ln(0.4/0.8) ≈ -0.2773
0.5 × ln(0.5/0.1) ≈  0.8047
0.1 × ln(0.1/0.1) =   0
Total ≈ 0.5275 nats
```

Une contribution peut être négative. C’est la somme exacte qui est non négative sous les conditions usuelles de distributions normalisées.

La direction inverse vaut environ 0.3936. Une valeur nulle correspond à des distributions identiques, pas à la réussite d’une tâche.

## 4. Laboratoire autonome — Exact et échantillonné

```python
import math
import torch


def kl(p, q):
    if len(p) != len(q) or any(value < 0 for value in p + q):
        raise ValueError('Supports ou probabilites invalides')
    if not math.isclose(sum(p), 1.0) or not math.isclose(sum(q), 1.0):
        raise ValueError('Probabilites non normalisees')
    total = 0.0
    for pi, qi in zip(p, q):
        if pi == 0:
            continue
        if qi == 0:
            return math.inf
        total += pi * math.log(pi / qi)
    return total


p = [0.4, 0.5, 0.1]
q = [0.8, 0.1, 0.1]
print('KL P||Q :', kl(p, q))
print('KL Q||P :', kl(q, p))
print('Identiques :', kl(p, p))
print('Support incompatible :', kl([1., 0.], [0., 1.]))

torch.manual_seed(42)
p_tensor = torch.tensor(p, dtype=torch.float64)
q_tensor = torch.tensor(q, dtype=torch.float64)
actions = torch.multinomial(p_tensor, num_samples=100000, replacement=True)
log_p_over_q = p_tensor[actions].log() - q_tensor[actions].log()
q_over_p = q_tensor[actions] / p_tensor[actions]
k3 = q_over_p - q_over_p.log() - 1
print('Estimateur log-ratio moyen :', log_p_over_q.mean().item())
print('Log-ratios individuels negatifs :', int((log_p_over_q < 0).sum().item()))
print('Estimateur non negatif moyen :', k3.mean().item())
```

Tu dois retrouver les deux KL exactes, zéro pour des distributions identiques et `inf` dans le cas de support incompatible. Les deux moyennes échantillonnées doivent être proches de 0.5275, avec une erreur de Monte-Carlo.

Le premier estimateur utilise `a~P` et moyenne `log(P(a)/Q(a))`. Un échantillon isolé peut être négatif ; un petit lot peut aussi donner une moyenne négative sans contredire la propriété de la KL exacte.

Le second calcule `u−ln(u)−1` avec `u=Q(a)/P(a)`. Il est non négatif par échantillon. Son espérance correspond ici à la même KL grâce aux distributions strictement positives sur le même support ; ne transpose pas la formule sans ses hypothèses.

## 5. Référence, ancienne policy et pénalité

Une règle de récompense régularisée peut inclure :

\[
J = E[R_{tâche}] - \beta D_{KL}(\pi_\theta\|\pi_{ref})
\]

La référence fixe protège une proximité choisie ; l’ancienne policy d’une collecte sert au ratio proximal. Une KL vers chacune ne raconte pas forcément le même changement.

Un beta trop contraignant peut limiter le gain utile ; trop faible peut laisser une dérive importante. Il ne possède pas une interprétation universelle en pourcentage de sécurité ou de qualité.

DPO encode un rapport à une référence dans son objectif dérivé ; ce n’est pas la preuve qu’une KL explicite calculée à chaque step est plafonnée à une valeur donnée.

## 6. Token, réponse ou corpus ?

Une KL conditionnelle sur le prochain token se calcule pour un contexte précis. Pour des séquences, une décomposition en contributions conditionnelles dépend aussi de la distribution des préfixes.

Une somme par réponse, une moyenne par token actif et une moyenne par exemple ont des échelles différentes. Padding, EOS, réponses tronquées et longueur modifient les agrégations.

Pour comparer des logs, conserve direction, distributions utilisées, contexte/collecte, estimateur, masques et unité de moyenne. Une « KL=0.1 » sans cette convention reste difficile à interpréter.

## 7. Expériences guidées

1. Déplace progressivement P depuis Q vers une distribution concentrée et mesure tâche et KL ensemble.
2. Réduis le nombre de tirages à 4 puis 16 ; observe l’incertitude de l’estimateur.
3. Compare les deux directions sans changer les vecteurs ; explique leur pondération différente.
4. Sur un run RL, rapproche les logs KL de reward, longueurs et benchmark indépendant. Une faible KL n’est pas l’objectif produit final.

## 8. Exercices avec corrigés

### Exercice 1 — Identité

KL([0.5,0.5] || [0.5,0.5]) vaut combien ?

<details>
<summary>Corrigé</summary>

Zéro : chaque rapport vaut 1 et ln(1)=0.

</details>

### Exercice 2 — Échantillon négatif

Une action a P=0.4 et Q=0.8. Son log-ratio négatif prouve-t-il que la KL exacte est négative ?

<details>
<summary>Corrigé</summary>

Non. La KL somme les contributions pondérées sur tous les événements. Un terme ou une estimation bruitée n’a pas toutes les propriétés de la somme exacte.

</details>

### Exercice 3 — Trois limites

Associe ratio PPO, clipping global et pénalité KL aux quantités qu’ils traitent.

<details>
<summary>Corrigé</summary>

PPO : une incitation fondée sur le changement relatif des probabilités observées. Clipping global : norme du gradient. KL : divergence entre distributions avec une direction et une agrégation définies.

</details>

### Exercice 4 — Qualité

Pourquoi conserver exactement la base donne une KL nulle sans démontrer d’amélioration ?

<details>
<summary>Corrigé</summary>

La distribution n’a pas changé. Cela décrit la proximité, pas un gain de tâche ou de comportement.

</details>

## 9. Validation — avant de continuer

- [ ] Je calcule la KL discrète et traite P=0 ou Q=0.
- [ ] Je distingue direction, terme individuel et estimateur.
- [ ] Je retrouve environ 0.5275 et 0.3936.
- [ ] Je précise contexte, référence, masques et moyenne d’un log KL.
- [ ] Je distingue proximité et amélioration utile.

## Fiche mémo

```text
Même événement + même contexte + direction annoncée.
La KL exacte est non négative ; un log-ratio échantillonné peut être négatif.
Proximité de la référence, ratio proximal et norme de gradient sont distincts.
```

Le [module 37](37_pre_entrainement_from_scratch.md) quitte l’adaptation d’un état déjà appris.
