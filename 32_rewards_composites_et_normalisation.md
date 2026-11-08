# Module 32 — Combiner des rewards sans cacher les compromis

## 1. Ce que tu vas apprendre

Une réponse peut être correcte, bien formatée et concise. Additionner ces mesures demande pourtant de décider quelle importance leur donner et si certaines exigences sont obligatoires.

Tu vas calculer une somme pondérée, distinguer préférence et contrainte, puis observer une standardisation sur un groupe.

**Prérequis :** modules 30 et 31 ; moyenne et dispersion.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et PyTorch, CPU.

## 2. Une composante, une définition et une échelle

Une **composante de reward** mesure un aspect du contrat. Avant sa pondération, précise sa plage et sa signification.

| Composante | Exemple de plage | Risque |
| --- | --- | --- |
| Exactitude | 0 ou 1 | Oracle ou extraction incorrects |
| Format | 0 ou 1 | Bien formé mais faux |
| Longueur | 0 à plusieurs milliers de caractères | Dominer la somme par son échelle |
| Score appris | Non borné selon le modèle | Comparabilité et calibration incertaines |

Dans `correctness + length`, 500 caractères peuvent écraser un point d’exactitude. Mettre des coefficients ne suffit pas si tu ignores les distributions réelles.

## 3. Refaire une somme à la main

Supposons deux indicateurs séparés, correctness et format, chacun dans {0,1}. Le premier peut être jugé avec une extraction permissive, le second selon le contrat strict : les quatre combinaisons ont alors un sens pour l’analyse.

\[
R = 0.8\,correctness + 0.2\,format
\]

| Correct | Format | Reward |
| --- | --- | --- |
| Oui | Oui | 1.0 |
| Oui | Non | 0.8 |
| Non | Oui | 0.2 |
| Non | Non | 0.0 |

Cette règle autorise qu’une réponse hors format reçoive une bonne note. Si le format est une **contrainte obligatoire**, il faut exprimer autre chose, par exemple `R = format × correctness`.

Une somme pondérée accepte des compensations ; une contrainte ou un filtre peut les interdire. Choisis selon l’usage avant d’optimiser.

## 4. Une prime de longueur peut inverser l’ordre

Avec une somme `0.2×correctness + 0.8×concision`, une réponse fausse très courte peut battre une réponse correcte plus longue. Ce n’est pas une erreur de multiplication : c’est le compromis écrit dans les poids.

La concision doit aussi être définie. Un texte vide n’est pas une bonne explication concise. Une longueur maximale peut être une contrainte de format plutôt qu’un bonus à maximiser.

Le **reward shaping** ajoute des signaux pour guider l’apprentissage. Un shaping arbitraire peut changer la solution favorisée ; il ne conserve pas automatiquement l’objectif initial.

## 5. Laboratoire autonome — Table de cas et normalisation

```python
import torch


def weighted(correctness, format_ok):
    return 0.8 * correctness + 0.2 * format_ok


def gated(correctness, format_ok):
    return correctness * format_ok


for correctness, format_ok in [(1, 1), (1, 0), (0, 1), (0, 0)]:
    print(correctness, format_ok, 'somme=', weighted(correctness, format_ok),
          'contrainte=', gated(correctness, format_ok))

rewards = torch.tensor([0.0, 0.0, 1.0, 0.5])
mean = rewards.mean()
std = rewards.std(correction=0)
advantages = (rewards - mean) / (std + 1e-8)
print('Moyenne :', mean.item(), 'ecart-type population :', std.item())
print('Scores centres reduits :', advantages.tolist())
print('Moyenne des scores transformes :', advantages.mean().item())

uniform = torch.ones(4)
no_contrast = (uniform - uniform.mean()) / (uniform.std(correction=0) + 1e-8)
print('Groupe uniforme :', no_contrast.tolist())
```

Tu dois retrouver **1, 0.8, 0.2, 0** pour la somme, puis **1,0,0,0** pour la règle de contrainte. Le groupe a une moyenne **0.375** ; ses scores centrés ont une moyenne proche de zéro.

Le groupe uniforme donne uniquement des zéros. L’epsilon évite une division par zéro, mais ne crée pas une différence de qualité inexistante.

Nous utilisons l’écart-type **population** avec `correction=0`. Un estimateur avec correction d’échantillon donne d’autres nombres : indique la convention, surtout pour les petits groupes.

## 6. Normaliser quoi, et sur quelles données ?

Deux opérations différentes sont souvent appelées normalisation :

- Transformer des **composantes** avant leur somme, pour contrôler leurs échelles.
- Centrer/réduire une **reward totale** dans un batch ou un groupe afin de former un signal d’apprentissage relatif.

Une transformation dépendant du groupe rend le score d’une réponse dépendant des autres réponses présentes. Cela peut être voulu, comme dans certaines variantes GRPO, mais ne constitue plus une note absolue stable.

Une normalisation min-max calculée sur les observations peut être très sensible aux extrêmes. Une z-standardisation d’une composante rare peut amplifier ses fluctuations. Examine histogrammes, valeurs extrêmes et fréquences, pas seulement les moyennes.

## 7. Instrumenter avant d’additionner

Pour chaque génération, garde les composantes brutes, leurs transformations, leurs poids, le total, la longueur et les raisons de parsing.

Rapporte notamment : exactitude globale, format valide, reward total, part de groupes sans contraste et score indépendant de la fonction optimisée.

Un total qui monte pendant que l’exactitude baisse doit être explicable par les composantes. Sans elles, tu ne peux pas savoir quel compromis la policy exploite.

## 8. Expériences guidées

1. Essaie les quatre combinaisons avec d’autres poids. Écris à l’avance les ordres qui doivent être respectés.
2. Ajoute une pénalité de longueur bornée et cherche une sortie vide ou fausse qui la maximise.
3. Recalcule le score standardisé d’une même réponse dans deux groupes différents. Montre pourquoi ce score n’est pas une note absolue.
4. Sur des sorties initiales réelles, logge chaque composante séparément avant de décider des poids.

## 9. Exercices avec corrigés

### Exercice 1 — Dominance numérique

Pourquoi ajouter un score 0–1 à un nombre de caractères 0–500 est-il problématique ?

<details>
<summary>Corrigé</summary>

La variation du second peut dominer le total. Le modèle est incité à produire beaucoup de caractères, même si l’exactitude ne progresse pas.

</details>

### Exercice 2 — Contrainte obligatoire

Tu ne peux accepter aucun JSON invalide. Une prime de 0.1 pour format valide exprime-t-elle cette contrainte ?

<details>
<summary>Corrigé</summary>

Non. Une somme autorise d’autres composantes à compenser l’absence du format. Il faut une règle de validation, une porte ou une décision qui représente explicitement le caractère obligatoire.

</details>

### Exercice 3 — Groupe identique

Tous les rewards d’un groupe valent 1. Que donne le centrage-réduction du laboratoire ?

<details>
<summary>Corrigé</summary>

Tous les avantages sont nuls. La réussite absolue est élevée, mais le groupe ne fournit pas de contraste relatif avec cette règle.

</details>

### Exercice 4 — Comparaison entre runs

Pourquoi conserver seulement le total ne suffit-il pas après un changement de poids ?

<details>
<summary>Corrigé</summary>

Le nombre ne représente plus le même compromis. Il faut versionner la règle et comparer les composantes ou rescorrer les sorties sous une convention commune.

</details>

## 10. Validation — avant de continuer

- [ ] Je définis chaque composante et sa plage.
- [ ] Je retrouve la table des quatre cas et distingue somme et contrainte.
- [ ] Je distingue normalisation de composantes et signal relatif de groupe.
- [ ] Je sais ce que fait un epsilon sans inventer du contraste.
- [ ] Je logge composantes et total avec leur version.

## Fiche mémo

```text
Définir les priorités → mesurer les échelles → choisir contraintes ou compromis.
La somme doit rester explicable par ses composantes.
```

Le [module 33](33_reward_hacking_et_robustesse.md) met ces règles à l’épreuve avec des contre-exemples.
