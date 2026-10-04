# Module 1 — Pipeline complet d’un LLM

> **Version pédagogique approfondie.** Ce module sert de référence de profondeur pour tout le cours : définition concrète, intuition, mécanisme, micro-exemples, laboratoire et validation.


## Objectif du module

À la fin de ce module, tu dois comprendre **ce qui se passe réellement entre le texte que tu écris et le prochain token produit par un LLM**.

Tu dois pouvoir expliquer ce pipeline sans réciter du code :

```text
Texte / messages
      ↓
Chat template
      ↓
Tokenizer
      ↓
Token IDs
      ↓
Embeddings
      ↓
Information de position
      ↓
Blocs Transformer
      ↓
Hidden states
      ↓
LM Head
      ↓
Logits
      ↓
Décodage
      ↓
Prochain token
      ↓
La boucle recommence
```

Tu dois également comprendre la différence entre :

```text
inférence
≠
génération
≠
entraînement
```

---

# 1. Qu’est-ce qu’un LLM ?

## Définition concrète

Un **LLM** (*Large Language Model*) est un réseau de neurones entraîné à modéliser des séquences de tokens.

Dans le cas d’un modèle causal utilisé pour générer du texte, sa tâche fondamentale est :

> À partir des tokens déjà présents, attribuer un score à chaque token possible pour déterminer lequel pourrait venir ensuite.

Il ne rédige donc pas une réponse complète en une seule opération.

Il répète essentiellement :

```text
contexte actuel
→ prédire le prochain token
→ ajouter ce token au contexte
→ recommencer
```

### Exemple très simple

Entrée :

```text
Le ciel est
```

Le modèle pourrait produire une distribution ressemblant à :

```text
bleu       0.61
gris       0.13
clair      0.08
grand      0.01
route      0.0002
...
```

Il sélectionne ensuite un token selon la stratégie de génération utilisée.

Si le token choisi est ` bleu`, le nouveau contexte devient :

```text
Le ciel est bleu
```

Puis le modèle recommence pour prédire le token suivant.

---

# 2. Token ≠ mot

## Définition concrète

Un **token** est une unité appartenant au vocabulaire du tokenizer.

Un token peut représenter :

- un mot entier ;
- une partie de mot ;
- un espace suivi d’un mot ;
- un signe de ponctuation ;
- quelques caractères ;
- un octet ou une séquence d’octets selon le tokenizer ;
- un marqueur spécial.

Il ne faut donc jamais supposer :

```text
1 mot = 1 token
```

### Exemple conceptuel

Le texte :

```text
anticonstitutionnellement
```

pourrait être représenté comme plusieurs fragments :

```text
anti
constitution
nel
lement
```

Le découpage exact dépend du tokenizer.

### Pourquoi c’est important

Le nombre de tokens influence directement :

- la longueur de contexte utilisée ;
- le coût de calcul ;
- la mémoire ;
- la vitesse d’inférence ;
- la quantité de données traitées pendant l’entraînement.

---

# 3. Prompt

## Définition concrète

Le **prompt** est l’information fournie au modèle pour conditionner sa génération.

Dans sa forme la plus simple :

```text
Explique HTTP.
```

Mais avec un modèle conversationnel, le prompt réel peut contenir plusieurs messages :

```text
system: Tu es un assistant pédagogique.
user: Explique HTTP.
assistant:
```

Le modèle ne reçoit pas nécessairement ces trois lignes exactement telles quelles.

Elles passent généralement par un **chat template**.

---

# 4. Chat template

## Définition concrète

Un **chat template** est une règle de formatage qui transforme une conversation structurée en la séquence de texte/tokens attendue par le modèle.

### Exemple conceptuel

Conversation logique :

```text
system: Tu es un assistant pédagogique.
user: Explique HTTP.
```

Peut être convertie en quelque chose ressemblant à :

```text
<|system|>
Tu es un assistant pédagogique.
<|user|>
Explique HTTP.
<|assistant|>
```

Les marqueurs exacts varient selon les modèles.

## Pourquoi c’est important

Un modèle de chat a appris durant son post-entraînement qu’un certain motif représente :

```text
instruction système
message utilisateur
réponse assistant
```

Utiliser un mauvais template peut donc dégrader fortement son comportement même si le modèle lui-même est intact.

### À ne pas confondre

```text
prompt
≠
chat template
```

Le prompt est le contenu/logique de l’entrée.

Le chat template est la manière de l’encoder dans le format attendu par le modèle.

---

# 5. Tokenizer

## Définition concrète

Le **tokenizer** convertit du texte en tokens puis en identifiants numériques.

Il réalise donc essentiellement :

```text
texte
→ tokens
→ token IDs
```

et peut effectuer l’opération inverse :

```text
token IDs
→ texte
```

### Exemple conceptuel

Texte :

```text
Le chat dort.
```

Tokens possibles :

```text
"Le"
" chat"
" dort"
"."
```

Puis IDs :

```text
[2145, 7821, 11934, 13]
```

Ces nombres sont des exemples fictifs.

---

# 6. Token ID

## Définition concrète

Un **token ID** est simplement l’indice numérique d’un token dans le vocabulaire.

Supposons un vocabulaire minuscule :

```text
ID    token
0     <pad>
1     <eos>
2     le
3      chat
4      dort
5     .
```

Alors :

```text
le chat dort.
```

pourrait devenir :

```text
[2, 3, 4, 5]
```

Le nombre `3` ne signifie pas mathématiquement « chat ».

C’est uniquement une adresse dans une table.

---

# 7. Vocabulaire

## Définition concrète

Le **vocabulaire** est l’ensemble des tokens que le tokenizer sait représenter directement, chacun associé à un ID.

Exemple minuscule :

```text
0   <pad>
1   <eos>
2   bon
3   jour
4   bonjour
5   !
```

Un vrai modèle possède généralement des dizaines de milliers de tokens ou davantage.

## Pourquoi sa taille compte

Si le vocabulaire contient `V` tokens, la couche de sortie doit pouvoir produire approximativement :

```text
V scores
```

pour chaque position.

Donc un vocabulaire plus grand affecte notamment :

- la matrice d’embeddings ;
- le LM head ;
- la mémoire ;
- le coût de certains calculs.

---

# 8. Tokens spéciaux

## Définition concrète

Les **tokens spéciaux** ont une fonction structurelle plutôt que linguistique ordinaire.

Exemples courants :

```text
BOS = Beginning Of Sequence
EOS = End Of Sequence
PAD = Padding
UNK = Unknown, selon le tokenizer
```

Les modèles de chat peuvent également avoir des marqueurs de rôle.

## Exemple

```text
<bos>
<|user|>
Bonjour
<|assistant|>
Bonjour !
<eos>
```

Ces tokens font partie de la séquence que le modèle traite.

---

# 9. Padding

## Définition concrète

Le **padding** consiste à compléter des séquences plus courtes avec un token artificiel afin de pouvoir les regrouper dans un même batch rectangulaire.

Supposons :

```text
A = [10, 20, 30, 40]
B = [11, 21]
```

Pour créer un tenseur de même longueur :

```text
A = [10, 20, 30, 40]
B = [11, 21, PAD, PAD]
```

Le padding n’est pas une information linguistique réelle.

---

# 10. Attention mask

## Définition concrète

L’**attention mask** indique quelles positions doivent être considérées comme du contenu valide dans certaines opérations.

Pour l’exemple précédent :

```text
A IDs   = [10, 20, 30, 40]
A mask  = [ 1,  1,  1,  1]

B IDs   = [11, 21, PAD, PAD]
B mask  = [ 1,  1,   0,   0]
```

Les `0` indiquent les positions de padding.

### À ne pas confondre

```text
padding / attention mask
≠
causal mask
```

Le causal mask sert à empêcher un token de voir le futur pendant un modèle causal.

Nous y revenons plus bas.

---

# 11. Pourquoi le modèle ne peut pas utiliser directement les IDs

Un ID comme :

```text
48372
```

est une catégorie, pas une mesure.

On ne veut pas que le modèle interprète automatiquement :

```text
48372 > 120
```

comme si le premier token était « plus grand » linguistiquement que le second.

Il faut donc transformer chaque ID en un vecteur appris.

C’est le rôle des **embeddings**.

---

# 12. Embedding

## Définition concrète

Un **embedding** est un vecteur de nombres réels appris par le modèle pour représenter un token dans un espace continu.

Exemple fictif en 4 dimensions :

```text
chat
→ [0.21, -0.83, 0.14, 1.07]
```

Dans un vrai modèle, ce vecteur peut contenir des centaines ou milliers de dimensions.

### Vue matricielle

Si :

```text
vocab_size = 50 000
hidden_size = 1024
```

la table d’embeddings ressemble conceptuellement à une matrice :

```text
[50 000, 1024]
```

Chaque token ID sélectionne une ligne de cette matrice.

### À ne pas confondre

```text
token ID
≠
embedding
```

Le token ID est un entier servant d’indice.

L’embedding est un vecteur appris.

---

# 13. Information de position

## Pourquoi elle est nécessaire

Considère :

```text
Le chien mord l’homme.
```

et :

```text
L’homme mord le chien.
```

Les tokens sont similaires, mais leur ordre change le sens.

Le modèle doit donc intégrer une information relative ou absolue sur la position.

Les architectures modernes peuvent utiliser des mécanismes comme **RoPE** (*Rotary Position Embedding*).

À ce stade, retiens surtout :

> Le Transformer doit connaître l’ordre relatif des tokens ; sinon une séquence serait beaucoup trop proche d’un simple sac de tokens.

Nous détaillerons RoPE dans un module consacré à l’architecture.

---

# 14. Transformer

## Définition concrète

Le **Transformer** est l’architecture de réseau qui transforme les embeddings en représentations contextualisées.

Il est composé d’une pile de blocs.

```text
embeddings
   ↓
Transformer block 1
   ↓
Transformer block 2
   ↓
Transformer block 3
   ↓
...
   ↓
Transformer block N
```

Chaque bloc contient principalement :

- un mécanisme d’attention ;
- un MLP ;
- des normalisations ;
- des connexions résiduelles.

---

# 15. Que signifie « contextualiser » ?

Le token :

```text
banc
```

peut apparaître dans :

```text
Je m’assois sur un banc.
```

ou :

```text
Le banc de poissons se déplace.
```

Son embedding de départ correspond au token.

Mais après passage dans les blocs Transformer, sa représentation interne dépend du contexte autour de lui.

Cette représentation contextualisée est un **hidden state**.

---

# 16. Hidden state

## Définition concrète

Un **hidden state** est la représentation numérique interne d’un token à une étape donnée du réseau.

Supposons un hidden size de 4 uniquement pour illustrer :

```text
embedding initial de "banc"
[0.2, -0.4, 0.1, 0.9]
```

Après plusieurs couches dans un contexte donné :

```text
hidden state
[-0.7, 1.2, 0.5, -0.1]
```

Le hidden state n’est pas un mot ni une probabilité.

C’est une représentation interne.

---

# 17. Self-attention

## Définition concrète

La **self-attention** permet à chaque position d’utiliser l’information d’autres positions pertinentes de la même séquence.

### Exemple intuitif

Phrase :

```text
Le serveur ne répond plus parce qu’il est surchargé.
```

Pour construire une bonne représentation du token `il`, le modèle doit tenir compte du contexte précédent.

L’attention fournit un mécanisme différentiable permettant de pondérer l’importance des autres positions.

---

# 18. Q, K et V

Chaque hidden state est projeté vers plusieurs représentations.

Les trois noms classiques sont :

```text
Q = Query
K = Key
V = Value
```

## Intuition

Une intuition utile, sans la prendre au pied de la lettre :

```text
Query  = qu’est-ce que cette position cherche ?
Key    = quel type d’information cette autre position peut offrir ?
Value  = quelle information transporter si elle est pertinente ?
```

Mathématiquement, une forme classique est :

\[
Attention(Q,K,V)
=
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

### Ce que fait grossièrement cette formule

1. comparer les queries aux keys ;
2. obtenir des scores de compatibilité ;
3. normaliser ces scores ;
4. utiliser ces poids pour combiner les values.

Les détails mathématiques seront repris dans le module Transformer.

---

# 19. Causal mask

## Définition concrète

Dans un modèle causal, le **causal mask** empêche une position d’utiliser des tokens futurs lorsqu’elle prédit le prochain token.

Pour :

```text
Le chat mange une souris
```

quand le modèle apprend à prédire le token `mange`, il ne doit pas déjà regarder :

```text
une souris
```

Sinon la tâche serait trichée.

### Vue simple

```text
Position 1 peut voir : 1
Position 2 peut voir : 1 2
Position 3 peut voir : 1 2 3
Position 4 peut voir : 1 2 3 4
```

mais pas les positions suivantes.

---

# 20. Multi-head attention

## Définition concrète

Au lieu d’effectuer une seule opération d’attention, le modèle utilise plusieurs **têtes d’attention** en parallèle.

Chaque tête travaille dans un sous-espace de représentation.

Cela permet au réseau d’apprendre plusieurs types de relations simultanément.

Il est tentant de dire :

```text
head 1 = grammaire
head 2 = sujet/verbe
head 3 = mémoire longue
```

mais ce serait trop simpliste.

Certaines têtes peuvent présenter des comportements interprétables, mais leur rôle n’est pas garanti ni fixe.

---

# 21. MLP

## Définition concrète

Le **MLP** (*multi-layer perceptron*) d’un bloc Transformer est un petit réseau feed-forward appliqué aux représentations de chaque position.

Dans plusieurs architectures modernes, tu verras des couches nommées :

```text
gate_proj
up_proj
down_proj
```

L’attention permet surtout aux positions d’échanger de l’information.

Le MLP transforme ensuite fortement cette information localement dans l’espace des features.

Cette distinction est simplifiée mais utile pour commencer.

---

# 22. Connexion résiduelle

## Définition concrète

Une **connexion résiduelle** ajoute l’entrée d’un sous-bloc à sa sortie transformée.

Conceptuellement :

\[
y = x + f(x)
\]

Cela permet notamment au réseau de conserver un chemin direct pour l’information et facilite l’entraînement de réseaux profonds.

---

# 23. Normalisation

Les Transformers modernes utilisent généralement une forme de normalisation comme :

```text
LayerNorm
```

ou :

```text
RMSNorm
```

## Rôle concret

La normalisation aide à contrôler l’échelle des activations et améliore la stabilité de l’optimisation.

On approfondira sa position exacte et ses variantes dans le module Transformer.

---

# 24. LM Head

## Définition concrète

Le **LM Head** est la projection finale qui transforme la représentation interne du modèle en un score pour chaque token du vocabulaire.

Supposons :

```text
hidden_size = 1024
vocab_size  = 50 000
```

Le dernier hidden state pertinent possède environ :

```text
1024 valeurs
```

Le LM head le transforme en :

```text
50 000 scores
```

un par token du vocabulaire.

Ces scores sont les **logits**.

---

# 25. Logit

## Définition concrète

Un **logit** est un score brut produit par le modèle avant conversion en probabilité.

Exemple fictif :

```text
Token       Logit
Paris       10.8
Lyon         4.2
Tokyo        0.3
fromage     -1.1
```

Les logits :

- peuvent être négatifs ;
- ne sont pas limités entre 0 et 1 ;
- ne doivent pas être interprétés directement comme des probabilités.

### À ne pas confondre

```text
logit
≠
probabilité
```

---

# 26. Softmax

## Définition concrète

La fonction **softmax** transforme une liste de scores en une distribution normalisée positive dont la somme vaut 1.

\[
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

où :

```text
z_i = logit du token i
P_i = probabilité normalisée associée
```

### Exemple conceptuel

Avant :

```text
Paris   10.8
Lyon     4.2
Tokyo    0.3
```

Après softmax, on pourrait obtenir approximativement :

```text
Paris   99.x %
Lyon     0.x %
Tokyo    très faible
```

La valeur exacte dépend de tous les logits du vocabulaire, pas seulement de ceux affichés.

---

# 27. Pourquoi on ne peut pas interpréter les top logits isolément

Softmax utilise :

```text
tous les logits
```

Donc si tu affiches seulement les 10 meilleurs tokens, leur somme de probabilités peut être inférieure à 100 %.

Le reste est réparti entre tous les autres tokens.

---

# 28. Temperature

## Définition concrète

La **temperature** modifie la forme de la distribution avant échantillonnage.

Une formulation courante est :

\[
z'_i = \frac{z_i}{T}
\]

Puis on applique softmax.

### Si `T < 1`

Les différences entre logits sont amplifiées.

La distribution devient plus concentrée.

### Si `T > 1`

Les différences sont réduites.

La distribution devient plus plate.

### Ce que temperature ne fait pas

Elle n’ajoute :

- aucune connaissance ;
- aucun raisonnement ;
- aucun nouveau token au vocabulaire.

Elle modifie seulement la distribution utilisée pour le décodage.

---

# 29. Greedy decoding

## Définition concrète

Le **greedy decoding** sélectionne toujours le token ayant le score/probabilité le plus élevé à chaque étape.

Exemple :

```text
souris   0.70
poisson  0.20
viande   0.10
```

Greedy choisit :

```text
souris
```

à chaque fois que cette distribution se présente.

### Avantage

Simple et déterministe.

### Limite

Le meilleur choix local à chaque token n’est pas forcément la meilleure séquence globale ni la plus naturelle.

---

# 30. Sampling

## Définition concrète

Le **sampling** choisit aléatoirement un token en respectant une distribution de probabilité.

Avec :

```text
souris   70 %
poisson  20 %
viande   10 %
```

`souris` reste le résultat le plus probable, mais `poisson` ou `viande` peuvent parfois être sélectionnés.

Le sampling introduit donc de la diversité.

---

# 31. Top-k

## Définition concrète

**Top-k** limite les candidats aux `k` tokens ayant les meilleurs scores.

Si :

```text
top_k = 5
```

on élimine tous les tokens sauf les cinq meilleurs avant l’échantillonnage.

### But

Éviter d’échantillonner dans une longue traîne de tokens extrêmement improbables.

---

# 32. Top-p

## Définition concrète

**Top-p**, aussi appelé *nucleus sampling*, garde le plus petit ensemble de tokens dont la probabilité cumulée atteint un seuil `p`.

Exemple :

```text
A  0.50
B  0.25
C  0.12
D  0.05
E  0.03
...
```

Avec :

```text
top_p = 0.80
```

on garde au minimum les tokens nécessaires pour atteindre environ 80 % cumulés.

Le nombre de candidats varie donc selon la distribution.

---

# 33. Génération autoregressive

## Définition concrète

Une génération est **autoregressive** lorsque chaque nouvelle prédiction dépend de la séquence contenant les tokens précédemment générés.

Exemple :

```text
Le chat
```

→ génère :

```text
 dort
```

Nouveau contexte :

```text
Le chat dort
```

→ génère :

```text
 sur
```

Nouveau contexte :

```text
Le chat dort sur
```

et ainsi de suite.

---

# 34. Condition d’arrêt

La génération doit finir à un moment donné.

Elle peut s’arrêter lorsque :

- un token EOS est généré ;
- `max_new_tokens` est atteint ;
- une séquence d’arrêt configurée apparaît ;
- une logique spécifique de stopping criteria s’applique.

---

# 35. KV cache

## Définition concrète

Le **KV cache** conserve les Keys et Values déjà calculées pour les tokens précédents pendant l’inférence autoregressive.

## Pourquoi il existe

Sans cache, à chaque nouveau token, une grande partie du calcul d’attention sur les anciens tokens devrait être refaite.

Avec cache :

```text
anciens K/V → réutilisés
nouveau token → nouveaux K/V seulement
```

### Compromis

Le KV cache :

```text
accélère la génération
```

mais :

```text
consomme de la mémoire
```

et sa taille augmente avec la longueur de séquence, le batch et l’architecture.

---

# 36. Inférence

## Définition concrète

L’**inférence** est l’utilisation d’un modèle déjà entraîné pour calculer des sorties à partir d’entrées.

Pendant l’inférence :

```text
les poids restent fixes
```

On peut faire :

```text
input
→ forward pass
→ logits
```

sans forcément générer plusieurs tokens.

---

# 37. Génération

## Définition concrète

La **génération** est un type d’inférence dans lequel le modèle produit un ou plusieurs nouveaux tokens, généralement de manière autoregressive.

Donc :

```text
génération ⊂ inférence
```

L’inférence peut aussi servir à :

- calculer des logits ;
- obtenir des hidden states ;
- calculer une loss ;
- scorer une séquence.

---

# 38. Entraînement

## Définition concrète

L’**entraînement** modifie les paramètres du modèle pour réduire une fonction de perte.

Le pipeline devient :

```text
entrée
↓
forward pass
↓
logits
↓
loss
↓
backpropagation
↓
gradients
↓
optimizer
↓
nouveaux paramètres
```

C’est la différence fondamentale avec l’inférence.

---

# 39. Next-token prediction pendant l’entraînement

Prenons :

```text
Le chat dort bien
```

Le modèle doit apprendre les relations :

```text
Le        → chat
Le chat   → dort
Le chat dort → bien
```

Dans un Transformer causal, ces positions peuvent être calculées efficacement en parallèle grâce au causal mask.

---

# 40. Labels

## Définition concrète

Les **labels** représentent les cibles correctes utilisées pour calculer la loss.

Pour un modèle causal, la cible est généralement la séquence décalée d’un token.

Conceptuellement :

```text
Input position     cible
Le                 chat
chat               dort
dort               bien
```

L’implémentation exacte est gérée par le modèle/trainer, mais comprendre ce décalage est essentiel.

---

# 41. Loss

## Définition concrète

La **loss** est une valeur numérique qui mesure à quel point les prédictions du modèle diffèrent des cibles d’entraînement selon une fonction définie.

Pour du causal language modeling, on utilise couramment la cross-entropy.

Si le bon token est `Paris` et que le modèle lui attribue une forte probabilité, la loss est plus faible.

S’il lui attribue une très faible probabilité, la loss est plus élevée.

Pour une cible unique, l’intuition est :

\[
L = -\log(P_{correct})
\]

### Exemple

Si :

```text
P(correct) = 0.9
```

la loss associée est petite.

Si :

```text
P(correct) = 0.01
```

la loss est beaucoup plus grande.

---

# 42. Backpropagation

## Définition concrète

La **backpropagation** calcule comment la loss varie par rapport aux paramètres du réseau.

Elle applique efficacement la règle de dérivation en chaîne à travers le graphe de calcul.

Elle produit les **gradients**.

---

# 43. Gradient

## Définition concrète

Le **gradient** d’un paramètre indique la dérivée de la loss par rapport à ce paramètre.

Intuition :

> Si je modifie légèrement ce poids, dans quelle direction la loss tend-elle à changer et avec quelle sensibilité locale ?

Ce n’est pas encore la mise à jour du poids.

Le gradient fournit l’information utilisée par l’optimizer.

---

# 44. Optimizer

## Définition concrète

L’**optimizer** transforme les gradients en mises à jour de paramètres selon un algorithme donné.

Exemple courant :

```text
AdamW
```

Conceptuellement :

```text
poids actuels
+ mise à jour calculée à partir des gradients
→ nouveaux poids
```

La vraie formule d’AdamW est plus complexe qu’une simple soustraction du gradient ; elle sera vue plus tard.

---

# 45. Learning rate

## Définition concrète

Le **learning rate** contrôle l’échelle des mises à jour effectuées pendant l’optimisation.

Trop grand :

```text
instabilité
overshoot
divergence
perte rapide de capacités
```

Trop petit :

```text
apprentissage très lent
modification insuffisante
```

Il ne faut pas apprendre une « bonne valeur universelle » : elle dépend de la méthode, du modèle, du batch, de la durée et d’autres facteurs.

---

# 46. Vue complète : inférence

```text
Messages
  ↓
Chat template
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Embeddings + position
  ↓
Transformer blocks
  ↓
Hidden states
  ↓
LM head
  ↓
Logits
  ↓
Décodage
  ↓
Next token
  ↓
Ajouter au contexte
  ↓
Recommencer
```

---

# 47. Vue complète : entraînement

```text
Séquence de tokens
  ↓
Forward pass
  ↓
Logits pour plusieurs positions
  ↓
Comparaison aux labels
  ↓
Loss
  ↓
Backward
  ↓
Gradients
  ↓
Optimizer
  ↓
Poids mis à jour
```

---

# 48. Les distinctions essentielles

Tu dois savoir expliquer chacune de ces lignes :

```text
token          ≠ mot
```

```text
token ID       ≠ embedding
```

```text
embedding      ≠ hidden state contextualisé
```

```text
hidden state   ≠ logit
```

```text
logit          ≠ probabilité
```

```text
probabilité max ≠ token forcément choisi avec sampling
```

```text
attention mask ≠ causal mask
```

```text
inférence      ≠ entraînement
```

```text
génération     = un cas particulier d’inférence
```

---

# 49. Mini-laboratoire A — Voir les tokens

Installe au minimum :

```bash
pip install torch transformers
```

Créer :

```text
labs/01_tokens.py
```

```python
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "La capitale de la France est Paris."

encoded = tokenizer(text, add_special_tokens=False)

tokens = tokenizer.convert_ids_to_tokens(encoded)

print("Texte :", text)
print("IDs   :", encoded)
print("Tokens:", tokens)
print("Nombre de tokens:", len(encoded))
print("Décodé:", tokenizer.decode(encoded))
```

## Ce que tu dois observer

1. Le texte devient une liste d’entiers.
2. Le nombre de tokens n’est pas nécessairement égal au nombre de mots.
3. `decode()` reconstruit approximativement le texte original selon les règles du tokenizer.

---

# 50. Mini-laboratoire B — Voir les logits

Créer :

```text
labs/02_logits.py
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "La capitale de la France est"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("Shape complète des logits:", outputs.logits.shape)

next_logits = outputs.logits[0, -1]

print("Nombre de logits pour la prochaine position:", next_logits.shape[0])
print("Vocabulaire du tokenizer:", len(tokenizer))
```

## Ce que tu dois comprendre

Si la forme est :

```text
[batch, sequence_length, vocab_size]
```

alors :

```python
outputs.logits[0, -1]
```

signifie :

```text
premier élément du batch
+
dernière position de la séquence
+
tous les tokens du vocabulaire
```

---

# 51. Mini-laboratoire C — Logits → probabilités

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

prompt = "La capitale de la France est"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

next_logits = outputs.logits[0, -1]
probabilities = torch.softmax(next_logits, dim=-1)

values, ids = torch.topk(probabilities, k=10)

for probability, token_id in zip(values, ids):
    token_id = token_id.item()
    token = tokenizer.decode([token_id])
    logit = next_logits[token_id].item()

    print(
        repr(token),
        "id=", token_id,
        "logit=", round(logit, 3),
        "prob=", round(probability.item() * 100, 3), "%"
    )
```

## Ce que tu dois observer

Pour chaque candidat, tu peux maintenant voir trois choses différentes :

```text
token
ID
logit
probabilité
```

Ne les confonds jamais.

---

# 52. Mini-laboratoire D — Effet de la temperature

Ajouter :

```python
for temperature in [0.25, 0.5, 1.0, 2.0]:
    scaled_logits = next_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    values, ids = torch.topk(probs, 5)

    print("\nTemperature:", temperature)

    for p, token_id in zip(values, ids):
        print(
            repr(tokenizer.decode([token_id.item()])),
            round(p.item() * 100, 2), "%"
        )
```

## Ce que tu dois observer

À faible température, le meilleur token domine davantage.

À température élevée, la distribution devient moins concentrée.

---

# 53. Mini-laboratoire E — Choisir manuellement un token

Après avoir calculé `probabilities` :

```python
next_id = torch.multinomial(
    probabilities,
    num_samples=1
)

print("ID choisi:", next_id.item())
print("Token choisi:", repr(tokenizer.decode([next_id.item()])))
```

Tu viens d’effectuer toi-même une étape de sampling.

---

# 54. Mini-laboratoire F — Comparer greedy et sampling

## Greedy

```python
greedy_id = torch.argmax(probabilities)
print("Greedy:", repr(tokenizer.decode([greedy_id.item()])))
```

## Sampling répété

```python
for i in range(10):
    sampled_id = torch.multinomial(probabilities, 1)
    print(i, repr(tokenizer.decode([sampled_id.item()])))
```

## Ce que tu dois observer

Greedy retourne toujours le meilleur candidat pour la même distribution.

Sampling peut varier.

---

# 55. Mini-laboratoire G — Une étape d’entraînement minuscule

Ce laboratoire ne fine-tune pas correctement le modèle ; il montre seulement la mécanique `loss → backward → gradient`.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.train()

text = "Paris est la capitale de la France."
inputs = tokenizer(text, return_tensors="pt")

outputs = model(
    **inputs,
    labels=inputs["input_ids"]
)

loss = outputs.loss

print("Loss:", loss.item())

model.zero_grad()
loss.backward()

for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        print("Premier gradient trouvé:", name)
        print("Norme du gradient:", parameter.grad.norm().item())
        break
```

## Ce que tu dois observer

Après :

```python
loss.backward()
```

certains paramètres possèdent un tenseur `grad`.

Mais aucun optimizer n’a encore mis à jour les poids.

Donc :

```text
backward
≠
optimizer.step()
```

---

# 56. Erreurs conceptuelles fréquentes

## Erreur 1

> Le modèle choisit toujours le token ayant la plus grande probabilité.

Faux avec sampling.

---

## Erreur 2

> Temperature rend le modèle plus intelligent.

Non.

Elle modifie le décodage.

---

## Erreur 3

> Un token est un mot.

Non.

---

## Erreur 4

> Les logits sont des probabilités.

Non.

---

## Erreur 5

> Le tokenizer apprend pendant chaque conversation.

Non pendant une inférence normale.

Le tokenizer utilisé est déjà défini.

---

## Erreur 6

> `loss.backward()` met à jour les poids.

Non.

Il calcule les gradients.

Une étape d’optimizer est nécessaire pour modifier les paramètres.

---

## Erreur 7

> Le modèle génère toute la phrase en parallèle.

Pendant une génération causale standard, les nouveaux tokens sont produits autoregressivement.

Pendant l’entraînement, en revanche, plusieurs positions de la séquence peuvent être évaluées parallèlement grâce au causal mask.

---

# 57. Exercices de compréhension

## Exercice 1

Explique pourquoi ce pipeline est incorrect :

```text
texte
→ modèle
→ mot suivant
```

Réécris toutes les étapes principales manquantes.

---

## Exercice 2

Explique la différence entre :

```text
Token ID = 483
```

et :

```text
Embedding = [0.21, -0.71, ...]
```

---

## Exercice 3

Un vocabulaire contient 100 000 tokens.

Quelle est approximativement la taille du dernier axe des logits pour une position ?

Réponse attendue :

```text
100 000
```

---

## Exercice 4

Explique pourquoi softmax est appliquée sur la dimension du vocabulaire lorsqu’on veut une distribution sur le prochain token.

---

## Exercice 5

Pourquoi le causal mask est-il indispensable pendant l’entraînement d’un modèle causal ?

---

## Exercice 6

Quelle différence entre :

```text
attention mask
```

et :

```text
causal mask
```

---

## Exercice 7

Explique la différence entre :

```text
forward pass
backward pass
optimizer step
```

---

# 58. Expériences à faire toi-même

Teste au minimum ces prompts :

```text
La capitale de la France est
```

```text
La capitale du Canada est
```

```text
2 + 2 =
```

```text
HTTP est un protocole
```

```text
Le chat mange une
```

Pour chacun, conserve les cinq tokens les plus probables.

Compare :

- leurs logits ;
- leurs probabilités ;
- l’effet de la temperature.

---

# 59. Ce que tu n’as pas encore besoin de maîtriser

À ce stade, tu n’as pas besoin de savoir dériver entièrement :

- la formule de l’attention ;
- RoPE ;
- AdamW ;
- LayerNorm/RMSNorm ;
- FlashAttention ;
- GQA/MQA ;
- le calcul mémoire exact du KV cache.

Ces sujets seront traités dans les modules appropriés.

L’objectif ici est d’avoir une **carte mentale correcte du pipeline complet**.

---

# 60. Validation finale du module 1

Avant de passer au module 2, tu dois pouvoir répondre clairement aux questions suivantes.

### Entrée et tokenisation

1. Qu’est-ce qu’un prompt ?
2. Qu’est-ce qu’un chat template ?
3. Qu’est-ce qu’un token ?
4. Pourquoi un token n’est-il pas nécessairement un mot ?
5. Qu’est-ce qu’un token ID ?
6. Qu’est-ce qu’un vocabulaire ?
7. À quoi servent BOS, EOS et PAD ?
8. À quoi sert une attention mask ?

### Représentation interne

9. Pourquoi ne donne-t-on pas simplement les IDs comme valeurs numériques continues au Transformer ?
10. Qu’est-ce qu’un embedding ?
11. Pourquoi faut-il représenter l’ordre des tokens ?
12. Qu’est-ce qu’un hidden state ?

### Transformer

13. Quel est le rôle général de la self-attention ?
14. Que signifient Q, K et V ?
15. Pourquoi utilise-t-on un causal mask ?
16. Quel est le rôle général du MLP ?
17. À quoi servent les connexions résiduelles ?

### Sortie

18. Qu’est-ce que le LM head ?
19. Qu’est-ce qu’un logit ?
20. Pourquoi un logit n’est-il pas une probabilité ?
21. Que fait softmax ?
22. Que fait la temperature ?
23. Quelle différence entre greedy et sampling ?
24. Quelle différence entre top-k et top-p ?

### Génération

25. Pourquoi dit-on qu’un LLM causal génère de manière autoregressive ?
26. À quoi sert le KV cache ?
27. Pourquoi le KV cache consomme-t-il de la mémoire ?

### Entraînement

28. Quelle différence entre inférence et entraînement ?
29. Qu’est-ce qu’un label ?
30. Qu’est-ce qu’une loss ?
31. Que fait `backward()` ?
32. Qu’est-ce qu’un gradient ?
33. Que fait l’optimizer ?
34. À quoi sert le learning rate ?

---

# Résumé en une phrase

Un LLM causal transforme une séquence de tokens en représentations contextualisées, projette la dernière représentation vers un score pour chaque token possible, sélectionne le prochain token selon une stratégie de décodage et répète cette opération ; pendant l’entraînement, une loss compare les prédictions aux cibles et les gradients servent à modifier les paramètres.

---

# Passage au module 2

Le module 2 approfondira le premier maillon essentiel de cette chaîne :

```text
texte
→ tokenizer
→ tokens
→ IDs
```

Tu y verras notamment :

- BPE ;
- byte-level tokenization ;
- Unigram ;
- vocabulaire ;
- merge rules ;
- fertility ;
- tokens spéciaux ;
- entraînement d’un tokenizer ;
- impact réel du tokenizer sur contexte, mémoire et coût.