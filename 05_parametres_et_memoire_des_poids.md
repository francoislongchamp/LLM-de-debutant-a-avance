# Module 5 — Compter les paramètres et estimer la mémoire

## 1. Ce que tu vas apprendre

Au [module 4](04_anatomie_du_transformer.md), tu as vu les matrices du réseau. Ce sont des tableaux de nombres ; beaucoup de ces nombres sont appris pendant l’entraînement.

Quand on dit « modèle de 7B », on parle d’environ **7 milliards de paramètres**, pas de 7 Go de mémoire. Pour passer de l’un à l’autre, il faut connaître la représentation numérique et préciser ce que l’on compte.

À la fin, tu devrais pouvoir :

- calculer le nombre de paramètres d’un embedding et d’une couche linéaire ;
- distinguer total, entraînable et partagé ;
- convertir un nombre de paramètres en mémoire brute des poids ;
- lire `dtype`, `numel()` et `element_size()` ;
- expliquer pourquoi les poids ne suffisent pas à dimensionner un entraînement ;
- situer ce que le gel des poids, LoRA et la quantification peuvent changer.

**Prérequis :** modules 1 à 4 ; multiplication, divisions et formes des matrices.

**Temps indicatif :** 90 à 150 minutes, laboratoire compris.

**Matériel :** Python et PyTorch sur CPU. Aucun gros modèle à télécharger.

## 2. Qu’est-ce qu’un paramètre ?

Un **paramètre** est une valeur stockée dans le réseau que l’on peut ajuster pendant l’apprentissage : poids de matrices, biais ou coefficients de certaines normalisations.

Un **hyperparamètre** est un réglage choisi pour l’expérience, par exemple le learning rate. Ce n’est pas une valeur de matrice apprise par backward.

Une **activation** est une valeur intermédiaire calculée à partir des entrées et des paramètres. Elle dépend des données traitées.

| Objet | Exemple | Change avec l’entrée ? | Appris par l’optimizer ? |
| --- | --- | --- | --- |
| Poids | Matrice de projection Q | Pas pendant une inférence ordinaire | Oui s’il est entraînable et optimisé |
| Activation | Hidden state d’un token | Oui | Calculée, pas stockée comme paramètre à optimiser |
| Hyperparamètre | Learning rate | Réglage de l’expérience | Pas par la boucle ordinaire |

Augmenter la longueur d’un batch ne crée pas de nouvelles matrices de poids. Cela augmente en revanche certaines activations et les calculs nécessaires.

## 3. Compter à la main

### Une table d’embeddings

Un vocabulaire de `V` entrées avec des vecteurs de taille `H` contient :

\[
N_{embedding} = V \times H
\]

Pour `V = 10`, `H = 4`, cela donne **40 paramètres**.

### Une couche linéaire

`nn.Linear(in_features, out_features)` contient une matrice de poids et, par défaut, un biais par sortie :

\[
N_{linear} = H_{entrée}\times H_{sortie} + H_{sortie}
\]

Pour `Linear(4, 4)`, cela donne `16 + 4 = 20`. Avec `bias=False`, il n’en reste que 16.

### Un petit modèle complet

Nous allons mesurer ce réseau de démonstration :

```text
Embedding(10, 4) → Linear(4, 4) → Tanh → Linear(4, 10)
```

Il n’est pas un Transformer et n’est pas entraîné. Nous le choisissons pour pouvoir tout compter.

| Composant | Calcul | Paramètres |
| --- | --- | --- |
| Embedding | `10 × 4` | 40 |
| Première couche linéaire | `4 × 4 + 4` | 20 |
| Tanh | Aucun poids appris | 0 |
| Couche de sortie | `4 × 10 + 10` | 50 |
| Total | `40 + 20 + 50` | **110** |

> **À toi.** Si le batch passe d’une à huit séquences, le total devient-il 880 ?
>
> Non. Le réseau réutilise les mêmes 110 paramètres pour chaque entrée.

## 4. Du nombre de valeurs au nombre d’octets

Un **dtype** décrit le format numérique d’un tenseur. Pour les poids denses usuels :

| Format | Bits par valeur | Octets par valeur |
| --- | --- | --- |
| FP32, `torch.float32` | 32 | 4 |
| FP16, `torch.float16` | 16 | 2 |
| BF16, `torch.bfloat16` | 16 | 2 |
| Entier 8 bits | 8 | 1 |
| Poids compactés en 4 bits, idéal brut | 4 | 0.5 |

FP16 et BF16 occupent autant de place par élément mais ne représentent pas les nombres de la même manière : leurs compromis de précision et de plage sont différents.

La quantification n’est pas seulement une conversion avec `.to(torch.int8)`. Elle utilise une représentation et des calculs adaptés, avec par exemple des échelles et des métadonnées.

Pour un stockage uniforme idéal :

\[
M_{poids} = N_{paramètres} \times \frac{bits}{8}
\]

Nos 110 paramètres en FP32 occupent `110 × 4 = 440 octets` de données de poids. Ce chiffre n’inclut pas les objets Python, buffers ou autres allocations du programme.

### GB et GiB ne sont pas la même unité

```text
1 GB  = 1 000 000 000 octets
1 GiB = 1 073 741 824 octets = 1024³ octets
1 MiB = 1024² octets
1 KiB = 1024 octets
```

Pour 7 milliards de paramètres :

| Représentation idéale | Octets | GB | GiB environ |
| --- | --- | --- | --- |
| FP32 | 28 000 000 000 | 28 | 26.08 |
| BF16 / FP16 | 14 000 000 000 | 14 | 13.04 |
| 8 bits | 7 000 000 000 | 7 | 6.52 |
| 4 bits compactés | 3 500 000 000 | 3.5 | 3.26 |

Les lignes quantifiées sont des estimations brutes avant surcoûts et parties non quantifiées. Un modèle annoncé « 7B » peut aussi avoir un total exact différent de 7 000 000 000.

## 5. Laboratoire A — Mesurer un modèle que tu peux compter

Ce bloc est autonome. Prédis d’abord les totaux et la mémoire, puis compare.

```python
import torch
from torch import nn


def make_model():
    return nn.Sequential(
        nn.Embedding(10, 4),
        nn.Linear(4, 4),
        nn.Tanh(),
        nn.Linear(4, 10),
    )


def measure(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    weight_bytes = sum(parameter.numel() * parameter.element_size()
                       for parameter in model.parameters())
    return total, trainable, weight_bytes


torch.manual_seed(42)
model = make_model()
for name, parameter in model.named_parameters():
    print(name, "forme=", tuple(parameter.shape), "dtype=", parameter.dtype,
          "valeurs=", parameter.numel(), "octets/valeur=", parameter.element_size())

print("Avant gel (total, entraînable, octets) :", measure(model))

# Geler seulement les poids de l'embedding avant tout entraînement.
model[0].weight.requires_grad_(False)
print("Après gel :", measure(model))

# Deux nouvelles copies : aucune modification de la copie précédente.
model_fp16 = make_model().to(dtype=torch.float16)
model_bf16 = make_model().to(dtype=torch.bfloat16)
print("FP16 :", measure(model_fp16))
print("BF16 :", measure(model_bf16))
```

Tu dois retrouver :

```text
Avant gel : (110, 110, 440)
Après gel : (110, 70, 440)
FP16      : (110, 110, 220)
BF16      : (110, 110, 220)
```

### Comprendre les trois outils

- `numel()` compte les éléments du tenseur : `[10, 4]` en contient 40.
- `element_size()` donne les octets par élément dans sa représentation actuelle.
- `requires_grad` indique ici si le paramètre est autorisé à participer au calcul de gradients.

Un paramètre marqué entraînable ne sera pas nécessairement modifié : il doit contribuer au calcul, recevoir un gradient pertinent et être pris en charge par la mise à jour. Le module 7 précisera ces conditions.

Geler des paramètres **ne les retire pas de la mémoire**. Ils restent nécessaires au forward. Cette mesure compte les données des paramètres parcourus, pas la mémoire totale du processus ou du GPU.

## 6. Paramètres partagés : ne pas compter deux fois

Certains modèles partagent la table d’embeddings avec les poids du LM head : on parle de **weight tying**. Les deux usages renvoient alors au même paramètre.

### Laboratoire B — Observer un partage

Ce bloc est autonome. Il utilise le même petit réseau pour rendre le calcul visible.

```python
from torch import nn

model = nn.Sequential(
    nn.Embedding(10, 4),
    nn.Linear(4, 4),
    nn.Tanh(),
    nn.Linear(4, 10),
)
print("Avant partage :", sum(parameter.numel() for parameter in model.parameters()))
model[3].weight = model[0].weight
print("Même objet :", model[3].weight is model[0].weight)
print("Après partage :", sum(parameter.numel() for parameter in model.parameters()))
print("Éléments de toutes les entrées du state_dict :",
      sum(tensor.numel() for tensor in model.state_dict().values()))
```

Le total de paramètres uniques passe de **110 à 70** : les 40 poids de la sortie ne sont plus indépendants des 40 poids d’embedding. Le biais de sortie reste présent.

Le `state_dict` peut contenir plusieurs noms associés au même stockage. Additionner aveuglément ses entrées donne ici 110 éléments nommés, pas 110 paramètres indépendants. `model.parameters()` évite les répétitions du même objet paramètre dans ce cas courant.

Les vues ou stockages partagés plus complexes demandent une inspection supplémentaire : notre calcul n’est pas une mesure universelle des allocations physiques.

## 7. Pourquoi les poids ne donnent pas toute la mémoire

Pour l’inférence et l’entraînement, les poids ne sont qu’une partie du budget :

| Composante | Ce qu’elle contient | Quand la prévoir |
| --- | --- | --- |
| Poids | Paramètres du modèle | Inférence et entraînement |
| Activations | Résultats intermédiaires des calculs | Les deux, avec davantage de conservation possible pour backward |
| KV cache | K/V des tokens précédents | Génération avec cache |
| Gradients | Dérivées par rapport aux paramètres concernés | Entraînement |
| États d’optimizer | Historique nécessaire à l’algorithme | Entraînement, selon l’optimizer |
| Temporaires et buffers | Espaces de travail et données auxiliaires | Les deux |

Les activations dépendent notamment du batch, des longueurs et de l’architecture. Les états d’optimizer dépendent des paramètres optimisés et de l’algorithme.

### Un budget d’entraînement hypothétique

Supposons un milliard de paramètres, avec une organisation qui conserve :

```text
Poids BF16                         2 octets/paramètre
Gradients BF16                     2 octets/paramètre
Deux états Adam en FP32            8 octets/paramètre
Copie maîtresse des poids en FP32  4 octets/paramètre
Total de ces éléments             16 octets/paramètre
```

Cela ferait **16 GB avant activations et temporaires**, contre 2 GB pour les seuls poids BF16.

Ce budget décrit **une hypothèse de stockage**, pas une règle de toutes les implémentations d’Adam ou de précision mixte. Vérifie les dtypes et les états réellement utilisés. Le [module 50](50_budget_memoire_du_training.md) détaillera le budget d’entraînement.

Un modèle peut donc tenir en inférence et provoquer un manque de mémoire pendant son entraînement complet. Le message **OOM**, pour *Out Of Memory*, signifie que l’allocation demandée ne peut pas être satisfaite.

## 8. Que changent LoRA et la quantification ?

**LoRA** entraîne de petites matrices ajoutées à certaines projections, en gelant généralement la base. Cela réduit fortement le nombre de paramètres à optimiser, leurs gradients et leurs états d’optimizer.

La base reste nécessaire au calcul. Il faut aussi conserver les activations nécessaires aux gradients des adaptateurs. La mémoire totale ne baisse donc pas forcément dans la même proportion que le nombre de paramètres entraînables.

La **quantification** réduit la représentation de certains poids. Des échelles, des groupes de quantification et des couches restées dans d’autres formats créent des surcoûts. Le calcul peut décompacter ou convertir les poids : « poids 4 bits » ne signifie pas « tous les calculs et tous les tenseurs en 4 bits ».

Tu pratiqueras ces méthodes dans les modules 15 à 20. Ici, l’objectif est de savoir quel poste du budget elles visent.

## 9. Expériences guidées

### A — Prédire avant de changer une largeur

Dans le laboratoire A, remplace la largeur interne 4 par 8 dans toutes les couches concernées. Calcule d’abord le total : `10×8 + (8×8+8) + (8×10+10) = 242`. Vérifie ensuite avec le script.

### B — Désactiver les biais

Repars de la largeur 4 et ajoute `bias=False` aux deux `Linear`. Prédis combien de paramètres disparaissent : `4 + 10 = 14`. Le total attendu devient **96**.

### C — Faire un tableau mémoire

Pour 100 millions, un milliard et sept milliards de paramètres, calcule FP32 et BF16 en GB puis GiB. Nomme clairement le résultat « poids bruts » et non « mémoire totale nécessaire ».

### D — Examiner un vrai modèle déjà chargé

Dans la session du laboratoire réel du module 1, utilise `model.named_parameters()` pour classer les dix plus gros tenseurs selon `numel()`. Calcule leur mémoire avec leur **dtype effectif**, sans déduire celui-ci uniquement du nom du modèle ou de sa fiche.

## 10. Exercices avec corrigés

### Exercice 1 — Une grosse matrice

Combien de poids contient une matrice `4096 × 11008`, hors biais ? Combien de MiB en BF16 ?

<details>
<summary>Corrigé</summary>

`4096 × 11008 = 45 088 768` poids. En BF16 : `90 177 536` octets, soit **86 MiB** en divisant par `1024²`.

</details>

### Exercice 2 — Un milliard de paramètres

Donne la mémoire brute de 1B paramètres en FP32, BF16 et 4 bits compactés, en GB.

<details>
<summary>Corrigé</summary>

FP32 : **4 GB** ; BF16 : **2 GB** ; 4 bits compactés : **0.5 GB** avant surcoûts. Ces chiffres ne donnent ni la mémoire des activations ni celle d’un entraînement.

</details>

### Exercice 3 — Geler sans supprimer

Le total reste 110 mais le nombre entraînable passe à 70. Pourquoi les poids occupent-ils encore 440 octets en FP32 ?

<details>
<summary>Corrigé</summary>

Les 40 paramètres gelés sont toujours stockés et utilisés dans le forward. Le gel change leur participation aux gradients, pas leur présence dans le réseau.

</details>

### Exercice 4 — Diagnostiquer un budget

Un modèle a 14 GB de poids bruts. Peut-on affirmer qu’un appareil offrant exactement 14 GB disponibles suffit pour le générer sur un long prompt ?

<details>
<summary>Corrigé</summary>

Non. Il faut aussi prévoir activations, temporaires, éventuel cache et autres allocations. La longueur et le batch influencent ces besoins ; le poids brut n’est qu’un poste du budget.

</details>

### Exercice 5 — Taille du fichier ou RAM ?

Pourquoi la taille du fichier de poids, la somme `numel × element_size` et la mémoire du processus peuvent-elles différer ?

<details>
<summary>Corrigé</summary>

Le fichier a un format de sérialisation et peut enregistrer les poids dans un autre dtype que celui utilisé en mémoire. La somme mesure ici les tenseurs de paramètres. Le processus contient aussi objets, buffers, activations et autres allocations. Les paramètres partagés demandent également d’éviter les doubles comptes.

</details>

## 11. Validation — avant de continuer

- [ ] Je distingue paramètres, activations et hyperparamètres.
- [ ] Je compte un embedding et une couche linéaire avec ou sans biais.
- [ ] Je retrouve les totaux 110, 70 après gel, et 70 paramètres uniques après partage.
- [ ] Je convertis paramètres et dtype en octets, GB et GiB.
- [ ] Je sais expliquer les limites de `numel × element_size`.
- [ ] Je peux nommer les postes ajoutés par l’entraînement.
- [ ] Je sais ce que LoRA et la quantification réduisent en premier lieu.

## Fiche mémo

```text
Embedding : V × H paramètres.
Linear : entrée × sortie + sortie si biais.
Poids bruts : somme des numel × octets par élément.
Geler des poids ne les supprime pas.
Partager des poids exige de ne pas les compter deux fois.
Mémoire totale : poids + activations + cache éventuel + autres allocations.
Training : ajouter aussi gradients et états d’optimizer selon la configuration.
```

Le [module 6](06_loss_et_prediction_du_prochain_token.md) explique comment mesurer l’erreur qui guidera les modifications de ces paramètres.

## Références

- [PyTorch : `Tensor.element_size`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.element_size.html) — taille en octets d’un élément.
- [PyTorch : `Module.parameters`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters) — parcours des paramètres.
- [LoRA](https://arxiv.org/abs/2106.09685) — adaptation par matrices de faible rang.
