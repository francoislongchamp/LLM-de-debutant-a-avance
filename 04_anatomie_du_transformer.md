# Module 4 — Anatomie du Transformer

## Pourquoi ce module arrive ici

Le fine-tuning modifie des paramètres situés dans des couches précises. Sans comprendre un bloc Transformer, les noms `q_proj`, `v_proj`, `gate_proj` ou `lm_head` restent opaques.

## Objectifs

Comprendre le trajet d’un tenseur dans un Transformer causal : embeddings, position/RoPE, normalisation, self-attention, Q/K/V/O, MLP, résidus et LM head.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fondations

## Définitions concrètes

### Embedding

**Définition concrète.** Vecteur appris associé à un token ID.


**Exemple simple.** ID 42 → vecteur de 1024 nombres.

### Hidden state

**Définition concrète.** Représentation contextuelle d’un token à une couche donnée.


**Exemple simple.** Le vecteur de `banque` change selon le contexte.

### Self-attention

**Définition concrète.** Mécanisme permettant à une position de combiner de l’information d’autres positions autorisées.


**Exemple simple.** Un pronom peut recevoir de l’information d’un nom précédent.

### Attention head

**Définition concrète.** Sous-espace d’attention calculé en parallèle.


**Exemple simple.** 16 heads produisent plusieurs vues de l’attention.

### MLP / FFN

**Définition concrète.** Réseau feed-forward qui transforme chaque position.


**Exemple simple.** Souvent `gate_proj`, `up_proj`, activation, `down_proj`.

### Résidu

**Définition concrète.** Addition de l’entrée au résultat d’un sous-bloc.


**Exemple simple.** `x_new=x+f(x)`.

### Normalisation

**Définition concrète.** Opération stabilisant l’échelle des activations.


**Exemple simple.** RMSNorm est courant dans les LLM modernes.

### LM head

**Définition concrète.** Projection finale vers un score pour chaque token du vocabulaire.


**Exemple simple.** hidden 1024 → vocab 50k produit 50k logits par position.

## Intuition simple

L’attention sert surtout à **échanger l’information entre positions**. Le MLP sert surtout à **transformer l’information de chaque position**. Les résidus conservent une route directe. Répéter ces blocs construit des représentations de plus en plus contextuelles.

## Ce qui se passe réellement sous le capot

1. Les IDs indexent la matrice d’embedding.
2. L’information de position est injectée, souvent via RoPE dans l’attention.
3. Une normalisation prépare les activations.
4. Des projections créent Q, K et V.
5. `QKᵀ` produit des scores entre positions; le causal mask supprime le futur.
6. Softmax transforme ces scores en poids d’attention.
7. Les V sont combinées puis projetées via O.
8. Le résultat est ajouté par résidu.
9. Le MLP transforme ensuite chaque position et un second résidu est ajouté.
10. Les blocs se répètent.
11. La normalisation finale et le LM head produisent les logits.

## Exemple minimal à comprendre mentalement

Pour 3 tokens et hidden size 4 :

```text
X       [3,4]
Q,K,V   [3,4]
Q @ K.T [3,3]
```

La matrice 3×3 contient un score de relation pour chaque paire de positions. Avec un masque causal, les cellules correspondant au futur sont interdites avant softmax.

## Formules et notation utiles

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
\]

`M` est le masque. Le facteur `1/√d_k` contrôle l’échelle des produits scalaires.

## Dimensions / formes à savoir lire

```text
hidden states : [B,L,H]
Q,K,V         : [B,heads,L,head_dim]
attention     : [B,heads,L,L]
output        : [B,L,H]
logits        : [B,L,vocab]
```

## Code minimal observable

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
print(model)

for name, p in list(model.named_parameters())[:50]:
    print(name, tuple(p.shape))
```

## Laboratoire guidé

1. Repère embeddings, q/k/v/o projections, MLP, normes et LM head.  
2. Note la forme de trois matrices.  
3. Compte les blocs Transformer.  
4. Calcule les paramètres d’une matrice `H×H`.  
5. Dessine le trajet d’un hidden state dans un bloc.

## Ce que tu dois observer

- De grandes matrices linéaires concentrent beaucoup de paramètres.
- Les noms exacts diffèrent selon l’architecture.
- LoRA ciblera plus tard certaines de ces matrices.

## À ne pas confondre

- Attention ≠ compréhension humaine.
- Attention head ≠ LM head.
- Hidden state ≠ embedding statique.
- Un bloc Transformer ≠ une seule couche linéaire.

## Erreurs fréquentes

- Mémoriser des noms de modules sans comprendre leur rôle.
- Croire que l’attention contient tous les paramètres importants : les MLP sont aussi massifs.

## Exercices

1. Combien de poids contient une matrice 1024×1024 ?
2. Dessine `norm → attention → résidu → norm → MLP → résidu`.
3. Explique pourquoi `QKᵀ` relie des positions entre elles.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Dessiner un bloc Transformer causal simplifié.
- Expliquer Q, K, V et O en termes simples.
- Distinguer attention, MLP, normalisation, résidu et LM head.
- Lire les formes principales d’un modèle PyTorch.

## Fiche mémo

Un Transformer alterne **communication entre positions** et **transformation par position**, à travers plusieurs blocs.

## Lien avec le module suivant

Le module 5 traduit ces matrices en nombre de paramètres et en mémoire.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
