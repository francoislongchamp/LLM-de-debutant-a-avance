# Module 4 — Ouvrir un bloc Transformer et suivre ses calculs

## 1. Ce que tu vas apprendre

Tu sais maintenant préparer des IDs et des masques. Mais comment le réseau transforme-t-il ces entrées en scores ? Nous allons ouvrir un **Transformer causal**, en commençant par une seule tête d’attention et de très petits vecteurs.

À la fin, tu devrais pouvoir :

- suivre IDs → embeddings → bloc → logits ;
- expliquer le rôle de Q, K, V et de la projection de sortie O ;
- lire les dimensions d’une matrice d’attention ;
- distinguer échange entre positions et transformation par position ;
- expliquer résidus, normalisation et information de position ;
- vérifier par une expérience que le futur ne change pas les sorties du passé.

**Prérequis :** modules 1 à 3 ; lecture des formes de tenseurs. Le produit matriciel est expliqué avec ses dimensions.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python et PyTorch, sur CPU. Le laboratoire principal ne télécharge aucun modèle.

## 2. Représenter chaque token par un vecteur

Un ID est une catégorie. Une **table d’embeddings** lui associe une ligne de nombres que le réseau peut transformer.

Avec un vocabulaire `V = 7` et une largeur interne `H = 4`, la table a la forme `[7, 4]`. Trois IDs sélectionnent trois lignes : le résultat a la forme `[3, 4]`, ou `[1, 3, 4]` en conservant l’axe batch.

| Objet | Forme dans notre laboratoire |
| --- | --- |
| IDs | `[1, 3]` |
| Table d’embeddings | `[7, 4]` |
| Vecteurs sélectionnés | `[1, 3, 4]` |

Le vecteur de départ d’un même ID est le même chaque fois que la table est consultée. Après les blocs, son **hidden state** dépend aussi du contexte autorisé.

Il n’existe pas de règle générale disant « la première coordonnée mesure le genre, la deuxième le sujet ». Les coordonnées sont apprises conjointement et leur interprétation n’est pas imposée ici.

## 3. Vue d’un bloc pré-normalisé

Voici une structure courante, simplifiée et causale :

```text
X ────────────────────────┐
│                         │
normalisation             │
│                         │
attention causale         │
│                         │
addition résiduelle ←─────┘
│
X' ───────────────────────┐
│                         │
normalisation             │
│                         │
MLP                       │
│                         │
addition résiduelle ←─────┘
│
X'' → bloc suivant, ou normalisation finale puis LM head
```

On peut écrire :

\[
X' = X + Attention(Norm_1(X))
\]
\[
X'' = X' + MLP(Norm_2(X'))
\]

Il existe d’autres placements de normalisation. Cette structure est notre exemple de travail, pas la définition de toutes les architectures Transformer.

## 4. L’attention : combiner des informations entre positions

### 4.1 Q, K et V sont des projections apprises

Chaque vecteur d’entrée est transformé en trois vecteurs :

| Nom | Intuition utile | Calcul simplifié |
| --- | --- | --- |
| Q, query | Ce que cette position va comparer | `Q = X @ Wq` |
| K, key | Ce à quoi les autres positions comparent leurs queries | `K = X @ Wk` |
| V, value | L’information qui sera combinée | `V = X @ Wv` |

Ce sont des calculs numériques, pas des questions et réponses en français. Les matrices `Wq`, `Wk`, `Wv` sont apprises pendant l’entraînement.

Un produit scalaire compare une query à une key. Par exemple `[1, 0] · [1, 0] = 1`, tandis que `[1, 0] · [0, 1] = 0`.

### 4.2 Pourquoi `Q @ K.T` relie les positions

Sans l’axe batch, pour trois positions et des vecteurs de taille 4 :

```text
Q          [3, 4]
K.T        [4, 3]
Q @ K.T    [3, 3]
```

Chaque ligne correspond à une query ; chaque colonne à une key. On obtient un score pour chaque paire de positions.

On divise les scores par `sqrt(d_k)`, où `d_k` est la dimension d’une key. Cette mise à l’échelle aide à contrôler leur amplitude avant softmax.

### 4.3 Masquer, puis normaliser, puis combiner

\[
A = softmax\left(\frac{QK^T}{\sqrt{d_k}} + M\right)
\qquad O_{tête} = AV
\]

Le masque additif `M` vaut ici 0 pour les relations permises et `−∞` pour les relations interdites. Softmax est appliquée **sur les positions des clés**, dans chaque ligne.

Les relations interdites ont alors un poids nul. Les poids des clés autorisées totalisent 1 pour chaque query, à condition qu’elle dispose d’au moins une clé autorisée.

`A @ V` calcule une combinaison pondérée des values. Ce n’est pas une sélection obligatoire d’un seul token.

### 4.4 Un calcul à deux positions

Imaginons que la deuxième query vaille `[1, 0]`, et que les deux keys soient `[1, 0]` et `[0, 1]`.

Les scores après division par `sqrt(2)` sont environ `[0.7071, 0]`. Softmax donne environ `[0.6698, 0.3302]`.

Si les values sont `[2, 0]` et `[0, 4]`, la combinaison vaut :

```text
0.6698 × [2, 0] + 0.3302 × [0, 4] ≈ [1.3395, 1.3210]
```

La deuxième position utilise donc de l’information des deux positions. La première, avec un masque causal, n’aurait accès qu’à sa propre value.

> **À toi.** Pourquoi appliquer le masque avant softmax ?
>
> Pour répartir la masse uniquement entre les clés autorisées. Mettre simplement certains poids à zéro après softmax ne les renormalise pas automatiquement.

## 5. Plusieurs têtes et projection O

La **multi-head attention** calcule plusieurs attentions dans différents espaces appris, puis combine leurs résultats avec une projection de sortie, souvent nommée `o_proj`.

Dans une configuration classique où `H = nombre_de_têtes × dimension_par_tête` :

```text
Hidden states      [B, L, H]
Q, K, V par tête   [B, têtes, L, dimension_par_tête]
Poids d’attention  [B, têtes, L, L]
Sortie combinée    [B, L, H]
```

Cette égalité des largeurs est une convention fréquente, pas une obligation de toutes les architectures. Avec **GQA**, plusieurs têtes de queries partagent des têtes de keys/values ; les nombres de têtes Q et KV diffèrent. Consulte aussi `head_dim` lorsqu’il est défini explicitement.

Une tête d’attention n’est pas le LM head. On ne peut pas non plus attribuer d’avance une fonction fixe comme « tête de grammaire » à chaque tête.

## 6. MLP, résidus et normalisation

### Le MLP transforme chaque position

Un MLP simple applique :

```text
vecteur de taille H → couche linéaire → activation → couche linéaire → taille H
```

L’activation, par exemple GELU, introduit une transformation non linéaire. Sans elle, deux couches linéaires successives resteraient une transformation affine.

Les mêmes paramètres du MLP sont utilisés pour toutes les positions, mais son entrée est déjà contextualisée par l’attention. Des variantes à porte utilisent notamment `gate_proj`, `up_proj` et `down_proj`.

### Le résidu additionne deux vecteurs de même forme

`x + f(x)` conserve un chemin direct pour l’information et les gradients. Cela facilite l’entraînement de réseaux profonds. Il faut que les formes soient compatibles ; le résidu ne concatène pas les deux vecteurs.

### La normalisation contrôle l’échelle des représentations

LayerNorm agit sur les coordonnées de chaque vecteur, ici selon son dernier axe. Elle ne normalise pas une phrase pour la rendre grammaticalement correcte.

LayerNorm et RMSNorm sont deux opérations différentes : RMSNorm utilise une échelle fondée sur la moyenne des carrés sans soustraire la moyenne comme le fait LayerNorm. Le laboratoire utilise LayerNorm pour rester simple.

## 7. Où intervient la position ?

Dans notre laboratoire, une table de positions apprise fournit un vecteur par position, ajouté à l’embedding du token.

RoPE, utilisé par d’autres architectures, agit différemment : il effectue des rotations dépendant de la position sur des composantes de Q et K. Les scores d’attention peuvent ainsi dépendre des positions relatives. On ne doit donc pas décrire tous les modèles comme « embedding de token + embedding de position ».

Tu n’as pas besoin de dériver RoPE pour ce laboratoire ; tu dois savoir quel mécanisme de position notre petit réseau utilise et que ce choix varie.

## 8. Laboratoire — Un Transformer causal miniature

Ce script autonome construit un petit réseau **non entraîné**, avec une seule tête, un seul bloc, une table de positions et aucun padding ni dropout. Ses scores ne sont pas des réponses linguistiques utiles.

Il montre les opérations et les formes. Prédis : si on modifie seulement le dernier token, les sorties des deux premières positions doivent-elles changer ?

```python
import math
import torch
from torch import nn


def causal_attention(q, k, v):
    length = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    future = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(7, 4)
        self.position_embedding = nn.Embedding(3, 4)
        self.norm1 = nn.LayerNorm(4)
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.norm2 = nn.LayerNorm(4)
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        self.final_norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 7, bias=False)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1])
        x = self.token_embedding(ids) + self.position_embedding(positions)
        normalized = self.norm1(x)
        q = self.q_proj(normalized)
        k = self.k_proj(normalized)
        v = self.v_proj(normalized)
        attended, weights = causal_attention(q, k, v)
        x = x + self.o_proj(attended)
        x = x + self.mlp(self.norm2(x))
        logits = self.lm_head(self.final_norm(x))
        return logits, weights


torch.manual_seed(42)
model = TinyTransformer()
model.eval()
ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
changed_ids = torch.tensor([[1, 2, 6]], dtype=torch.long)

with torch.no_grad():
    logits, weights = model(ids)
    changed_logits, _ = model(changed_ids)

print("Forme des logits :", tuple(logits.shape))
print("Poids d’attention :\n", weights[0])
print("Sommes des lignes :", weights[0].sum(dim=-1))
print("Sorties du préfixe inchangées :",
      torch.allclose(logits[:, :2], changed_logits[:, :2], atol=1e-6))
print("Écart sur la dernière position :",
      (logits[:, 2] - changed_logits[:, 2]).abs().max().item())

for name, parameter in model.named_parameters():
    print(name, tuple(parameter.shape))
```

### Résultats à comprendre

- Les logits ont la forme **`[1, 3, 7]`**.
- Les poids d’attention ont ici la forme **`[1, 3, 3]`**, car nous n’avons qu’une tête et n’affichons pas d’axe tête séparé.
- Les poids strictement au-dessus de la diagonale sont nuls.
- Les poids de chaque ligne totalisent environ 1.
- Modifier le dernier token laisse les logits du préfixe inchangés ; la dernière position change.

Cette expérience teste une propriété précise de notre réseau : le futur n’influence pas les sorties passées. Elle ne prouve pas sa capacité à comprendre du texte.

### Lire les noms de matrices

PyTorch stocke les poids d’une couche `Linear(in_features, out_features)` sous la forme `[out_features, in_features]`. Le calcul correspond à `x @ weight.T`, avec un biais si activé.

Ainsi, le premier MLP `Linear(4, 8)` a une matrice `[8, 4]`, même si nous décrivons sa transformation comme « 4 vers 8 ». Au module 5, tu compteras les nombres stockés dans ces matrices.

## 9. Expériences guidées

### A — Vérifier le calcul à deux positions

Dans un nouveau script, recopie la définition de `causal_attention`, puis crée trois tenseurs : Q vaut `[[1.0, 0.0], [1.0, 0.0]]`, K vaut `[[1.0, 0.0], [0.0, 1.0]]` et V vaut `[[2.0, 0.0], [0.0, 4.0]]`. Appelle la fonction avec ces tenseurs et retrouve, sur sa deuxième ligne de sortie, les poids et la combinaison de la section 4.4.

### B — Retirer le masque dans une copie

Dans une copie du laboratoire, enlève seulement la ligne `scores.masked_fill(...)`. Prédis si le test « préfixe inchangé » passera encore.

Avec les paramètres du laboratoire, le futur peut alors influencer le passé et le test doit devenir faux. Cela montre pourquoi le masque sert aussi pendant l’entraînement sur une séquence déjà entièrement disponible.

### C — Reconnaître les composants d’un vrai modèle

Si tu as exécuté le laboratoire réel du module 1 dans la même session, affiche `model` et parcours `model.named_parameters()`. Repère embeddings, projections Q/K/V/O, MLP, normalisations et LM head.

Les noms et les formes peuvent différer de notre miniature. Note la configuration réelle au lieu de supposer que Q, K et V ont toujours la même largeur. Ce prolongement nécessite les poids du modèle réel ; le laboratoire principal reste autonome.

## 10. Exercices avec corrigés

### Exercice 1 — Retrouver les axes

Pour quatre positions et des keys de taille 8, quelles sont les formes de Q, K transposée et `Q @ K.T`, sans batch ni tête ?

<details>
<summary>Corrigé</summary>

Q : `[4, 8]` ; K transposée : `[8, 4]` ; scores : **`[4, 4]`**. Les deux derniers axes des scores correspondent aux positions queries et keys.

</details>

### Exercice 2 — Compter une matrice

Une matrice de poids mesure `1024 × 1024`. Combien de nombres contient-elle, hors biais ?

<details>
<summary>Corrigé</summary>

`1024 × 1024 = 1 048 576` poids. Le nombre de positions d’entrée ne multiplie pas ce nombre de paramètres : les mêmes poids sont réutilisés.

</details>

### Exercice 3 — Expliquer l’échange

Pourquoi dit-on que l’attention mélange l’information entre positions tandis que le MLP agit par position ?

<details>
<summary>Corrigé</summary>

L’attention construit des poids entre queries et keys de différentes positions et combine leurs values. Le MLP applique la même transformation à chaque vecteur séparément ; ce vecteur peut déjà contenir de l’information provenant d’autres positions.

</details>

### Exercice 4 — Lire un résidu

`x` et `f(x)` ont la forme `[2, 5, 16]`. Quelle est la forme de `x + f(x)` ? Est-ce la même opération qu’une concaténation ?

<details>
<summary>Corrigé</summary>

La forme reste **`[2, 5, 16]`** : les nombres correspondants sont additionnés. Une concaténation sur le dernier axe donnerait `[2, 5, 32]`.

</details>

### Exercice 5 — Attention head ou LM head ?

Lequel construit des relations entre positions ? Lequel produit des scores sur le vocabulaire ?

<details>
<summary>Corrigé</summary>

Une tête d’attention construit des relations entre positions et combine leurs values. Le LM head projette les représentations finales vers les logits du vocabulaire. Le mot « head » ne désigne pas le même rôle dans les deux expressions.

</details>

## 11. Validation — avant de continuer

- [ ] Je peux dessiner un bloc pré-normalisé avec ses deux résidus.
- [ ] Je sais relier Q, K et V aux scores et à la combinaison pondérée.
- [ ] Je sais sur quel axe softmax agit dans l’attention.
- [ ] Je peux expliquer le test de modification du dernier token.
- [ ] Je distingue attention, MLP, normalisation, position et LM head.
- [ ] Je sais que les formes Q/K/V peuvent varier selon l’architecture.
- [ ] Je peux lire la forme de poids d’un `nn.Linear` PyTorch.

## Fiche mémo

```text
IDs → embeddings + information de position
→ blocs : normalisation → attention → résidu → normalisation → MLP → résidu
→ normalisation finale → LM head → logits

Q compare aux K ; les poids obtenus combinent les V.
O projette le résultat de l’attention vers la largeur de sortie.
Le masque causal interdit le futur avant softmax.
```

Le [module 5](05_parametres_et_memoire_des_poids.md) traduit les matrices en nombres de paramètres et en octets.

## Références

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — attention et architecture Transformer originale ; notre exemple causal pré-normalisé est une variante pédagogique.
- [RoFormer](https://arxiv.org/abs/2104.09864) — rotations de position RoPE.
- [PyTorch : `nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) — convention de stockage des poids et biais.
