# Module 27 — Construire des préférences et comprendre l’objectif DPO

## 1. Ce que tu vas apprendre

Le SFT fournit une réponse à imiter. Une préférence compare deux réponses au **même contexte** : l’une est choisie selon une grille, l’autre moins bien classée.

Tu vas annoter ce type de paire, calculer une marge relative à une référence et expliquer pourquoi DPO n’est pas simplement un SFT sur la réponse préférée.

**Prérequis :** modules 6, 7, 11, 13 et 26 ; logarithmes de probabilités et gradients.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python et PyTorch pour le calcul miniature. Aucun LLM à télécharger.

## 2. Définir une paire utile

| Champ | Sens |
| --- | --- |
| `prompt` | Contexte commun, consigne et informations disponibles |
| `chosen` | Réponse préférée selon la règle d’annotation |
| `rejected` | Réponse moins préférée, pas forcément entièrement fausse |
| Métadonnées d’audit | Critère, justification, source, groupe et éventuel désaccord |

Exemple :

```json
{
  "prompt": "Explique le rôle de backward en deux phrases courtes.",
  "chosen": "backward calcule les gradients de la loss. L’optimizer les utilise ensuite pour mettre à jour les paramètres.",
  "rejected": "backward change directement tous les poids et choisit automatiquement le learning rate."
}
```

La préférence est ici motivée par une erreur technique. Il faut aussi des paires difficiles : deux réponses plausibles qui diffèrent sur précision, pertinence ou clarté selon une grille explicite.

`chosen` signifie « préféré dans cette comparaison », pas « vérité absolue ». Si les deux réponses sont mauvaises, la paire peut enseigner un ordre relatif sans montrer une bonne solution. Prévois une règle pour corriger, écarter ou annoter les cas où aucune réponse n’est acceptable.

## 3. Éviter les raccourcis d’annotation

Si chosen est toujours plus long, commence toujours par « Bien sûr » ou possède toujours une ponctuation particulière, le modèle peut apprendre ces indices plutôt que le critère voulu.

Relis des paires où la réponse préférée est courte et d’autres où une explication plus longue est nécessaire. Inverser l’ordre d’affichage aux annotateurs aide à repérer un biais de position ; cela ne signifie pas inverser arbitrairement les labels enregistrés.

Conserve hors du texte d’apprentissage la raison du choix et les informations d’audit. Les groupes d’un même prompt ou de variantes proches doivent rester dans le même split.

## 4. DPO : comparer policy et référence

La **policy** est le modèle entraîné. La **référence** est un modèle ou état fixé qui sert d’ancrage ; ce n’est pas un reward model qui note la qualité.

Pour une réponse y, définissons :

\[
\ell_\theta(y|x) = \log \pi_\theta(y|x)
\]

Dans le DPO standard étudié ici, cette log-probabilité de réponse est la somme des log-probabilités de ses tokens retenus, conditionnés au prompt et aux tokens précédents. Le prompt n’est pas ajouté comme cible de réponse.

La marge relative est :

\[
m = (\ell_\theta(y^+|x)-\ell_{ref}(y^+|x))
  - (\ell_\theta(y^-|x)-\ell_{ref}(y^-|x))
\]

`+` désigne chosen, `−` rejected. On veut que la policy favorise davantage chosen **relativement à la référence**.

## 5. Calculer la loss sur une paire

Pour la version sigmoid standard :

\[
L_{DPO} = -\log\sigma(\beta m)
\]

`σ` est la fonction logistique et `β > 0` une échelle de l’objectif liée à la formulation avec contrôle de dérive.

Exemple :

```text
Policy : chosen -3, rejected -5
Reference : chosen -4, rejected -4
Marge : (-3+4) - (-5+4) = 2
Avec beta=0.1 : loss = -log(sigmoid(0.2)) ≈ 0.5981
```

Si policy et référence sont identiques, m=0 et la loss vaut `ln(2) ≈ 0.6931`, même si la policy préférait déjà chosen en probabilité brute.

Une marge relative positive ne signifie pas toujours que chosen a une probabilité brute supérieure à rejected. Elle mesure un **changement relatif** de préférence par rapport à l’ancre.

### Comment lire beta sans slogan trompeur

Dans la dérivation standard, beta est associé au compromis de l’objectif régularisé. Dans la loss ci-dessus, il multiplie aussi la marge et modifie les gradients pour des poids donnés. On ne peut donc pas annoncer « doubler beta divise exactement la dérive par deux ».

Le résultat dépend de l’optimisation, du LR, des données et de la durée. Mesure marge, qualité et dérive au lieu d’assimiler beta à un pourcentage de préférence.

## 6. Laboratoire autonome — Loss et signes des gradients

```python
import torch
import torch.nn.functional as F

chosen_logp = torch.tensor(-3.0, requires_grad=True)
rejected_logp = torch.tensor(-5.0, requires_grad=True)
ref_chosen_logp = torch.tensor(-4.0)
ref_rejected_logp = torch.tensor(-4.0)
beta = 0.1
margin = (chosen_logp - ref_chosen_logp) - (rejected_logp - ref_rejected_logp)
loss = -F.logsigmoid(beta * margin)
loss.backward()
print('Marge :', margin.item())
print('Loss :', loss.item())
print('Gradient logp chosen :', chosen_logp.grad.item())
print('Gradient logp rejected :', rejected_logp.grad.item())
print('Policy identique a reference : loss=', -F.logsigmoid(torch.tensor(0.0)).item())
```

Tu dois retrouver **m=2**, **loss≈0.5981**, un gradient chosen négatif et rejected positif. Une descente indépendante sur ces deux variables pousserait la première vers le haut et la seconde vers le bas.

Dans un LLM, ces log-probabilités dépendent de paramètres partagés : on ne les règle pas comme deux boutons indépendants. Ce laboratoire vérifie l’objectif, pas une génération ou un entraînement complet.

## 7. Ce que DPO fait différemment du SFT

Un SFT sur chosen augmente sa probabilité selon la cross-entropy des cibles. DPO utilise aussi rejected et la référence dans une comparaison relative.

La préférence peut s’améliorer par différents changements de log-probabilités ; la loss ne garantit pas que la probabilité absolue de chosen monte sur chaque paire. Observe séparément les scores chosen/rejected et la marge.

DPO standard utilise des paires offline et ne nécessite pas d’entraîner explicitement un reward model puis de lancer PPO. Il appartient à l’optimisation de préférences ; toutes les méthodes de post-training ne sont pas du RL online.

## 8. Projet guidé — Construire un petit corpus de préférences

1. Définis trois critères, par exemple exactitude, format et pertinence, avec priorité si plusieurs critères s’opposent.
2. Constitue des paires sur un même prompt complet et conserve leur source/groupe.
3. Note pourquoi chosen est préféré et si les deux réponses sont acceptables.
4. Examine les longueurs, formules répétées, marques de style et éventuels indices triviaux.
5. Relis un sous-ensemble à l’aveugle, idéalement avec un second jugement pour repérer les ambiguïtés.
6. Sépare les groupes avant les variantes ; réserve validation et test.
7. Prépare un benchmark général, car l’optimisation de préférence peut créer des régressions.

Un corpus de 30 à 50 paires suffit pour apprendre cette préparation, pas pour garantir un alignement robuste.

## 9. Expériences guidées

- Remplace la marge du calcul par −2 puis 0 : retrouve une pénalité plus forte puis `ln(2)`.
- Garde les log-probabilités policy fixes et change la référence. Explique pourquoi la marge peut changer alors que les sorties de la policy n’ont pas bougé.
- Compare plusieurs beta sur le même calcul, puis distingue cette observation locale d’un entraînement complet.
- Pour ton corpus, compte la proportion de chosen plus longs que rejected et inspecte des contre-exemples. Une corrélation n’établit pas seule un biais appris, mais guide l’audit.

## 10. Exercices avec corrigés

### Exercice 1 — Calcul de marge

Policy chosen/rejected : −2 et −4 ; référence : −3 et −5. Quelle marge relative ?

<details>
<summary>Corrigé</summary>

`(-2+3)-(-4+5)=1−1=0`. La policy donne davantage de probabilité aux deux réponses de la même quantité en log ; sa préférence relative à la référence est inchangée.

</details>

### Exercice 2 — Référence identique

La référence égale la policy au départ. Quelle loss initiale pour cette version DPO ?

<details>
<summary>Corrigé</summary>

`−log(sigmoid(0))=ln(2)≈0.6931`, hors autres variantes ou termes d’objectif. Cela ne mesure pas directement la qualité absolue de la policy.

</details>

### Exercice 3 — Prompt commun

Pourquoi une paire avec deux questions différentes est-elle problématique ?

<details>
<summary>Corrigé</summary>

La préférence ne compare plus deux réponses dans le même contexte. La différence peut venir des questions et des informations accessibles, pas de la qualité relative des réponses.

</details>

### Exercice 4 — Deux réponses mauvaises

Peut-on présenter chosen comme cible idéale si les deux sorties contiennent des erreurs ?

<details>
<summary>Corrigé</summary>

Non. Une préférence relative n’implique pas l’acceptabilité absolue. Il faut suivre la politique d’annotation : correction, exclusion ou indication explicite du type de comparaison.

</details>

## 11. Validation — avant de continuer

- [ ] Je définis prompt, chosen, rejected et une grille d’annotation.
- [ ] Je calcule la marge relative et la loss de l’exemple.
- [ ] Je distingue préférence brute et relative à la référence.
- [ ] Je sais pourquoi la référence n’est pas un reward model.
- [ ] Je distingue DPO et SFT sur chosen.
- [ ] Je prépare des splits groupés et audite les raccourcis.

## Fiche mémo

```text
Même prompt, deux réponses, préférence justifiée.
DPO standard : comparaison policy/référence sur les log-probabilités de réponse.
Une meilleure marge sur des paires n’est pas une garantie de qualité générale.
```

Le [module 28](28_entrainer_avec_dpotrainer.md) exécute réellement ce calcul avec DPOTrainer.

## Références

[Article DPO](https://arxiv.org/abs/2305.18290), [TRL : formats de données](https://huggingface.co/docs/trl/en/dataset_formats).
