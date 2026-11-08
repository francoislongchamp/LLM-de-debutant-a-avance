# Module 15 — Comprendre la correction basse-rang de LoRA

## 1. Ce que tu vas apprendre

Le full fine-tuning peut modifier chaque élément d’une grande matrice. **LoRA**, pour *Low-Rank Adaptation*, garde la matrice de base et apprend une correction factorisée.

Tu vas dessiner les matrices, calculer leur coût, appliquer une correction numérique et comprendre ce que le gel économise réellement.

**Prérequis :** modules 4, 5, 7 et 14 ; couche linéaire, gradients et généralisation.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et PyTorch, CPU. Pas de modèle à télécharger.

## 2. PEFT, LoRA et base gelée

**PEFT**, *Parameter-Efficient Fine-Tuning*, désigne une famille de méthodes qui adaptent un modèle avec un nombre réduit de paramètres entraînables. LoRA est une méthode de cette famille ; PEFT est aussi le nom d’une bibliothèque qui en implémente plusieurs.

| Objet | Ce qui arrive pendant un LoRA classique |
| --- | --- |
| Poids de base `W` | Conservés et gelés |
| Matrices ajoutées `A` et `B` | Entraînées |
| Entrée du modèle | Toujours traitée par la base et la branche d’adaptation |
| Loss | Peut rester la même cross-entropy qu’un SFT full FT |

Geler un poids ne le retire pas de la mémoire. Le forward l’utilise et le gradient peut traverser son calcul pour atteindre d’autres paramètres, même sans stocker de gradient pour ce poids.

## 3. Dessiner les formes avant les formules

Pour une projection `in → out`, PyTorch stocke `W` sous la forme `[out, in]`. La sortie d’un vecteur d’entrée vaut `W x` dans la notation colonne.

LoRA ajoute :

\[
W' = W + sBA
\]

```text
W : [out, in]
A : [r, in]
B : [out, r]
BA: [out, in]
```

`r` est le **rang interne** choisi. Le produit `BA` a un rang au plus `r` : ses colonnes sont construites à partir d’au plus `r` directions de sortie. La grande correction n’est donc pas libre de prendre n’importe quelle valeur quand `r` est petit.

Avec des exemples en lignes dans un batch :

```text
sortie = x @ W.T + s × (x @ A.T) @ B.T
```

Pas besoin de former explicitement la grande matrice `BA` à chaque forward. La branche passe par les petits espaces intermédiaires.

## 4. Micro-exemple de rang 1

Prenons une projection 2 → 2 :

```text
W = [[1, 0],        A = [[1, 1]]        B = [[2],
     [0, 1]]                                [-1]]
```

Alors :

```text
BA = [[ 2,  2],
      [-1, -1]]
```

Avec `s = 1` et `x = [1, 2]`, la base produit `[1, 2]`. La branche LoRA produit `[6, -3]`, et la sortie totale vaut **`[7, -1]`**.

Cet exemple utilise des valeurs imposées pour comprendre le calcul ; il ne montre pas un adapter entraîné ni un bon réglage d’amplitude.

### Laboratoire A — Retrouver les deux chemins

```python
import torch

x = torch.tensor([[1.0, 2.0]])
w = torch.eye(2)
a = torch.tensor([[1.0, 1.0]])
b = torch.tensor([[2.0], [-1.0]])
scale = 1.0

branch = (x @ a.T) @ b.T
separate = x @ w.T + scale * branch
merged = x @ (w + scale * b @ a).T
print('Correction :\n', b @ a)
print('Branche LoRA :', branch)
print('Sortie :', separate)
print('Deux calculs equivalents :', torch.allclose(separate, merged))
```

Tu dois retrouver la correction, la sortie **`[7, -1]`** et une comparaison vraie.

## 5. Calculer les paramètres ajoutés

Pour une couche sans biais supplémentaire :

\[
N_{LoRA} = r\times in + out\times r = r(in+out)
\]

Pour une matrice carrée 4096 × 4096 :

| Méthode | Paramètres modifiables de cette projection |
| --- | --- |
| Matrice complète | 16 777 216 |
| LoRA r=4 | 32 768 |
| LoRA r=8 | 65 536 |
| LoRA r=16 | 131 072 |

Avec `r=8`, la correction utilise ici **256 fois moins** de paramètres que la matrice complète. La base n’a pourtant pas disparu : la projection adaptée contient la base plus l’adapter.

Dans le petit exemple 2 × 2 avec r=1, `A+B` utilise 4 nombres, autant que W. L’intérêt en nombre de paramètres vient surtout d’un rang petit par rapport aux grandes dimensions.

### Laboratoire B — Vérifier les coûts

Ce bloc est autonome et utilise seulement Python.

```python
in_features = 4096
out_features = 4096
full = in_features * out_features
for rank in [4, 8, 16, 64]:
    added = rank * (in_features + out_features)
    print('r=', rank, 'ajoutes=', added, 'rapport full/adaptation=', full / added)
```

Le coût croît linéairement avec `r` si les modules ciblés restent identiques. Le ratio entraînable sur le modèle complet demande aussi de compter toutes les couches et les éventuels paramètres supplémentaires.

## 6. Alpha, initialisation et gradients

Dans LoRA classique, le facteur est souvent `s = alpha/r`. D’autres variantes utilisent une autre règle ; le module 18 les distinguera.

Une initialisation courante prend `A` aléatoire et `B` nulle. Alors `BA=0` et le module commence avec la même sortie que la base, hors autres changements.

Au premier backward, le gradient de A peut être nul parce qu’il est multiplié par B nulle. B peut recevoir un gradient non nul et commencer à bouger ; A pourra ensuite apprendre. Ce comportement ne prouve pas un bug.

Si A et B étaient toutes deux nulles dans cette paramétrisation simple, les deux gradients pourraient rester nuls : il n’y aurait pas le même démarrage utile.

## 7. Adapter séparé ou fusionné

Un **adapter** contient les paramètres et la configuration de l’adaptation. Il peut être sauvegardé séparément de la base. Il faut alors recharger la base compatible : mêmes poids initiaux et architecture, avec le même tokenizer pour un LLM.

Dans certains formats et configurations, on peut fusionner `sBA` dans W pour l’inférence. La fusion modifie la représentation des poids ; sa compatibilité et ses effets numériques dépendent notamment de la précision et d’une éventuelle quantification.

Plusieurs adapters peuvent partager un même checkpoint de base stocké sur disque. Cela ne signifie pas qu’ils sont tous activés en même temps ni qu’on peut additionner leurs comportements sans mesure.

## 8. Ce que LoRA économise et ce qu’il ne garantit pas

LoRA réduit surtout les gradients et états d’optimizer associés aux paramètres optimisés, ainsi que la taille de l’adapter sauvegardé. La base, des activations, buffers et temporaires restent nécessaires.

LoRA n’est pas une quantification. Il ne garantit ni une division de la VRAM totale par le ratio des paramètres entraînables ni une qualité égale au full FT. Le choix des cibles, du rang, des données et des réglages reste expérimental.

## 9. Expériences guidées

1. Dans le laboratoire A, change seulement le facteur `scale`. Vérifie que la base reste identique tandis que la correction change.
2. Mets B à zéro. Prédis la sortie et compare avec le chemin de base.
3. Calcule le coût LoRA d’une projection 4096 → 11008 avec r=16 avant de lancer le calcul Python.
4. Dessine deux adapters A/B différents reliés à la même base. Liste les fichiers et identifiants nécessaires pour les utiliser correctement.

## 10. Exercices avec corrigés

### Exercice 1 — Formes

Pour `in=6`, `out=10`, `r=2`, quelles sont les formes de A, B et BA ?

<details>
<summary>Corrigé</summary>

A : `[2,6]`, B : `[10,2]`, BA : `[10,6]`. Il y a `2×(6+10)=32` paramètres LoRA, contre 60 dans W hors biais.

</details>

### Exercice 2 — Matrice non carrée

Calcule le coût de 4096 → 11008 avec r=16.

<details>
<summary>Corrigé</summary>

`16 × (4096+11008) = 241 664` paramètres ajoutés, hors biais ou modules supplémentaires.

</details>

### Exercice 3 — Base gelée

Pourquoi faut-il encore charger la base pour utiliser un adapter séparé ?

<details>
<summary>Corrigé</summary>

L’adapter décrit une correction, pas les poids complets du modèle. Le forward a besoin de la contribution W aussi bien que de la branche A/B.

</details>

### Exercice 4 — Gradient A nul au départ

B est initialisée à zéro et le premier gradient de A vaut zéro. Faut-il conclure que LoRA n’apprend pas ?

<details>
<summary>Corrigé</summary>

Non. Vérifie B et les steps suivants : B peut commencer à changer, puis permettre un gradient non nul vers A. Il faut observer le mécanisme et les mises à jour, pas un seul gradient isolé.

</details>

## 11. Validation — avant de continuer

- [ ] Je dessine W, A, B et leurs dimensions.
- [ ] Je retrouve `[7,-1]` et l’égalité des deux chemins.
- [ ] Je calcule `r(in+out)` et le ratio 256 de l’exemple.
- [ ] Je distingue PEFT, LoRA, base gelée et quantification.
- [ ] Je sais pourquoi B nulle peut préserver la sortie initiale.
- [ ] Je peux expliquer le besoin d’une base compatible.

## Fiche mémo

```text
W_effective = W + scale × B @ A.
Base conservée ; petites matrices entraînées.
Moins de paramètres à optimiser ne signifie pas autant de VRAM totale en moins.
```

Le [module 16](16_injecter_et_entrainer_lora.md) injecte réellement ces branches avec PEFT et vérifie leur sauvegarde.

## Références

[Article LoRA](https://arxiv.org/abs/2106.09685), [configuration LoRA dans PEFT](https://huggingface.co/docs/peft/en/package_reference/lora).
