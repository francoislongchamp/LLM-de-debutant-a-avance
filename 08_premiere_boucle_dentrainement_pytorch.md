# Module 8 — Comprendre et écrire sa première boucle d’entraînement PyTorch

## 1. Ce que tu vas apprendre

Au module 7, tu as vu comment calculer un gradient. Ici, tu vas assembler les opérations qui permettent à un modèle **d’apprendre à partir d’exemples**.

Nous allons lui faire apprendre la relation `y = 2x`. Tu connais déjà la réponse, ce qui te permettra de vérifier chaque étape. Le modèle, lui, recevra seulement des entrées et leurs réponses attendues : son calcul ne contiendra pas la règle `2x`.

À la fin, tu devrais pouvoir :

- expliquer ce qui change pendant un entraînement ;
- calculer une première correction à la main ;
- écrire une boucle avec `zero_grad`, forward, loss, backward et `step` ;
- distinguer exemple, batch, epoch et mise à jour ;
- mesurer une erreur sur des exemples réservés à la validation ;
- repérer quelques erreurs courantes dans une boucle.

**Prérequis :** comprendre les tenseurs, la loss et l’idée de gradient des [modules 6](06_loss_et_prediction_du_prochain_token.md) et [7](07_gradients_et_autograd.md). En Python : variables, fonctions, boucles `for` et indentation.  
**Temps indicatif :** deux séances de 60 à 90 minutes, exercices compris.  
**Matériel :** Python et PyTorch ; un CPU suffit, aucun modèle à télécharger.

Lis les sections 2 à 5 avant d’exécuter du code. Pour chaque expérience, prédis d’abord ce qui va se passer, puis compare avec le résultat.

## 2. Qui fait quoi dans un entraînement ?

Imaginons ces exemples :

| Entrée `x` | Réponse attendue `y` |
| --- | --- |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |

Un **exemple** est ici un couple entrée/réponse. La réponse attendue est aussi appelée **cible**, ou *target* en anglais.

Notre modèle sera une droite :

\[
\hat y = wx + b
\]

- `ŷ` se lit « y chapeau » : c’est la prédiction.
- `w` est le poids : il multiplie l’entrée.
- `b` est le biais : il ajoute un décalage.

`w` et `b` sont les **paramètres entraînables**. Apprendre signifie modifier leurs valeurs. La solution recherchée est `w = 2` et `b = 0`.

| Élément | Rôle concret | Dans notre exemple |
| --- | --- | --- |
| Données | Fournir des entrées et les cibles | `x = 3`, `y = 6` |
| Modèle | Calculer une prédiction | `ŷ = w × 3 + b` |
| Loss | Mesurer l’écart à la cible | Erreur quadratique moyenne |
| Autograd | Calculer les gradients de la loss | Sensibilité de l’erreur à `w` et `b` |
| Optimizer | Modifier les paramètres à partir des gradients | SGD |
| Learning rate | Régler l’amplitude des corrections | `lr = 0.1` |

Le learning rate, la taille des batches et le nombre d’epochs sont des **hyperparamètres** : tu les règles avant l’expérience. Ils ne sont pas appris comme `w` et `b`.

> **Pause compréhension.** Si le modèle prédit 5 pour une cible de 6, qu’est-ce qui doit changer pour qu’il progresse ?
>
> Ce sont ses paramètres. La cible reste 6 : on ne change pas la réponse attendue pour donner raison au modèle.

## 3. Le cycle complet, sans code

```text
Entrées x ──→ modèle avec ses paramètres actuels ──→ prédictions ŷ
                                                          │
Cibles y ─────────────────────────────────────────────────┤
                                                          ↓
                                                        loss
                                                          │
                                                       backward
                                                          ↓
                                                      gradients
                                                          │
                                                    optimizer.step
                                                          ↓
                                                nouveaux paramètres
                                                          │
                                                   prochain batch
```

Avant de calculer les nouveaux gradients, on efface ceux de la correction précédente avec `zero_grad()`.

Les opérations ont des responsabilités distinctes :

1. **Forward** : le modèle calcule ses prédictions.
2. **Loss** : on mesure leur erreur.
3. **Backward** : autograd calcule les gradients.
4. **Step** : l’optimizer met à jour les paramètres.

**Calculer la loss ne modifie pas les poids. Appeler `backward()` non plus.** Dans cette boucle, c’est `optimizer.step()` qui les modifie.

## 4. Une correction calculée à la main

Prenons seulement deux exemples : `(1, 2)` et `(2, 4)`. Nous choisissons volontairement `w = 0` et `b = 0` au départ.

### 4.1 Prédire et mesurer l’erreur

Les deux prédictions valent 0. Nous utilisons la **MSE**, pour *Mean Squared Error*, ou erreur quadratique moyenne :

\[
L = \frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
\]

`N` est ici le nombre d’exemples du batch. On soustrait la cible, on élève au carré, puis on fait la moyenne. Le carré empêche les erreurs positives et négatives de s’annuler.

| `x` | `y` | `ŷ` | `ŷ − y` | Erreur au carré |
| --- | --- | --- | --- | --- |
| 1 | 2 | 0 | −2 | 4 |
| 2 | 4 | 0 | −4 | 16 |

Donc `L = (4 + 16) / 2 = 10`.

### 4.2 Calculer les gradients

Pour ce modèle et cette loss :

\[
\frac{\partial L}{\partial w}
= \frac{2}{N}\sum_i(\hat y_i-y_i)x_i
\qquad
\frac{\partial L}{\partial b}
= \frac{2}{N}\sum_i(\hat y_i-y_i)
\]

Tu peux d’abord lire ces formules comme une recette ; autograd fera ce calcul dans le code.

- Gradient de `w` : `(2 / 2) × ((−2) × 1 + (−4) × 2) = −10`.
- Gradient de `b` : `(2 / 2) × ((−2) + (−4)) = −6`.

Un gradient négatif indique qu’une petite augmentation du paramètre fait localement baisser la loss.

### 4.3 Appliquer la correction

Nous utilisons **SGD**, la descente de gradient stochastique, sans momentum ni autre option. Sa règle de mise à jour est :

\[
\text{paramètre nouveau}
= \text{paramètre ancien} - \text{learning rate} \times \text{gradient}
\]

Avec un learning rate de `0.1` :

- `w = 0 − 0.1 × (−10) = 1` ;
- `b = 0 − 0.1 × (−6) = 0.6`.

Les nouvelles prédictions sont `1.6` et `2.6`. La nouvelle loss vaut :

\[
L_{après} = \frac{(1.6-2)^2+(2.6-4)^2}{2} = 1.06
\]

**L’erreur est passée de 10 à 1.06 après une seule correction.** Le modèle n’a pas encore trouvé `w = 2`, `b = 0` ; il doit recommencer.

> **À toi.** Pourquoi le biais a-t-il augmenté alors que sa valeur idéale est 0 ?
>
> Parce que le gradient décrit l’effet local d’un paramètre avec les autres paramètres dans leur état actuel. Au départ, augmenter le biais aide à corriger des prédictions trop petites. Les corrections suivantes peuvent le ramener vers 0.

## 5. Observer exactement cette correction avec PyTorch

Ce bloc est autonome. Exécute-le dans ton environnement du cours, où `import torch` fonctionne. Dans un notebook, colle le bloc dans une cellule ; dans un fichier Python, lance ce fichier avec le Python de ton environnement.

```python
import torch
from torch import nn

# Deux exemples, chacun avec une seule caractéristique numérique.
x = torch.tensor([[1.0], [2.0]])
y = torch.tensor([[2.0], [4.0]])

# Une entrée, une sortie : le calcul est y_pred = w*x + b.
model = nn.Linear(in_features=1, out_features=1)

# On impose le même départ que dans le calcul à la main.
# Cette initialisation ne doit pas construire de graphe de gradients.
with torch.no_grad():
    model.weight.fill_(0.0)
    model.bias.fill_(0.0)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

model.train()
optimizer.zero_grad(set_to_none=True)
pred = model(x)                 # Forward : calculer les prédictions.
loss = loss_fn(pred, y)         # Comparer les prédictions aux cibles.

print("Loss avant :", loss.item())
loss.backward()                # Calculer les gradients, sans modifier w et b.
print("Gradient w :", model.weight.grad.item())
print("Gradient b :", model.bias.grad.item())
print("w avant step :", model.weight.item())

optimizer.step()               # Modifier w et b à partir des gradients.
print("w après step :", model.weight.item())
print("b après step :", model.bias.item())

model.eval()
with torch.no_grad():
    loss_apres = loss_fn(model(x), y)
print("Loss après :", loss_apres.item())
```

Tu dois retrouver, aux arrondis près :

```text
Loss avant : 10.0
Gradient w : -10.0
Gradient b : -6.0
w avant step : 0.0
w après step : 1.0
b après step : 0.6
Loss après : 1.06
```

### 5.1 Pourquoi les doubles crochets ?

`[[1.0], [2.0]]` a la forme `[2, 1]` : **deux exemples, une caractéristique par exemple**. `nn.Linear(1, 1)` produit ici une sortie de même forme `[2, 1]`.

Pour notre MSE, les prédictions et les cibles doivent avoir la même forme. Une cible de forme `[2]` peut déclencher une diffusion automatique (*broadcasting*) et comparer des valeurs d’une manière différente de celle souhaitée.

### 5.2 Que signifient les appels importants ?

| Code | Ce qu’il fait |
| --- | --- |
| `model.parameters()` | Donne à l’optimizer les paramètres à modifier |
| `model(x)` | Appelle le calcul du modèle, son forward |
| `loss_fn(pred, y)` | Produit un tenseur scalaire représentant l’erreur |
| `loss.backward()` | Calcule et accumule les gradients dans les champs `.grad` |
| `optimizer.step()` | Utilise ces gradients pour modifier les paramètres |
| `loss.item()` | Extrait un nombre Python pour l’affichage ou le journal |

Garde le tenseur `loss` pour appeler `backward()`. Un nombre Python issu de `loss.item()` ne conserve pas le graphe nécessaire à ce calcul.

### 5.3 Pourquoi effacer les gradients ?

PyTorch **additionne** les gradients aux gradients déjà présents. Pour que chaque correction utilise seulement son batch, nous appelons `optimizer.zero_grad(set_to_none=True)` au début de chaque itération.

Avec `set_to_none=True`, les champs `.grad` sont remis à `None` plutôt qu’à des tenseurs remplis de zéros. Le prochain backward les renseignera. Cela n’efface pas les poids appris.

On peut aussi remettre les gradients à zéro après un `step`, mais il faut le faire avant le prochain backward, hors accumulation volontaire. Dans ce module, garde toujours la même convention : **remise à zéro au début du batch**.

## 6. Batch, epoch et step : compter ce qui se passe

Un **batch** est un groupe d’exemples traités ensemble. Une **epoch** est un passage sur tout le jeu d’entraînement. Un **optimizer step** est une mise à jour des paramètres.

Avec 8 exemples et des batches de 2 :

```text
Epoch 1 : batch 1 → step 1 ; batch 2 → step 2 ;
          batch 3 → step 3 ; batch 4 → step 4
Epoch 2 : batch 1 → step 5 ; ... ; batch 4 → step 8
```

Dans notre boucle, il y aura **un step par batch d’entraînement**. Avec un seul appareil, sans accumulation, et en gardant le dernier batch :

\[
\text{steps par epoch} = \left\lceil \frac{N}{B} \right\rceil
\]

`N` est le nombre d’exemples d’entraînement, `B` la taille de batch, et `⌈ ⌉` signifie « arrondir à l’entier supérieur ».

Pour 10 exemples et un batch de 4, les tailles sont `4, 4, 2` : cela fait 3 steps. Avec `drop_last=True`, le dernier batch incomplet serait supprimé et il resterait 2 steps.

L’accumulation de gradients permet plusieurs backwards avant un step ; tu la verras au [module 23](23_gradient_accumulation_et_batch_effectif.md). Ici, nous n’en utilisons pas.

## 7. Construire une vraie boucle, avec validation

### 7.1 Préparer les données et leurs batches

Un **Dataset** représente les exemples. Un **DataLoader** les parcourt et les regroupe en batches. `TensorDataset(x, y)` associe chaque ligne de `x` à la ligne correspondante de `y`.

`shuffle=True` mélange l’ordre des exemples d’entraînement à chaque nouveau parcours du DataLoader, en conservant les couples entrée/cible.

Nous réservons deux entrées à la **validation** : le modèle sera évalué dessus, mais elles ne serviront jamais à ses corrections. Nous calculons leurs cibles avec `2x` parce que nous fabriquons les données ; le modèle lui-même ne reçoit toujours pas cette formule.

### 7.2 Script complet à exécuter

Ce deuxième script est autonome : commence dans un nouveau fichier ou relance-le en entier. Il ne dépend pas des variables du bloc précédent.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Stabiliser l'initialisation et le mélange pour comparer nos expériences.
torch.manual_seed(42)

# Huit exemples d'entraînement, deux exemples de validation distincts.
x_train = torch.tensor([
    [-2.0], [-1.5], [-1.0], [-0.5],
    [0.5], [1.0], [1.5], [2.0],
])
y_train = 2 * x_train
x_val = torch.tensor([[-1.25], [1.25]])
y_val = 2 * x_val

train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)


def evaluate(model, loader):
    """Mesurer la MSE moyenne par exemple, avec des paramètres fixes."""
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            pred = model(x_batch)
            assert pred.shape == y_batch.shape
            loss = loss_fn(pred, y_batch)
            batch_size = x_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples


print(f"Validation avant apprentissage : {evaluate(model, val_loader):.6f}")
global_step = 0

for epoch in range(1, 101):
    # evaluate() met le modèle en mode eval : on réactive train à chaque epoch.
    model.train()
    total_loss = 0.0
    total_examples = 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_batch)
        assert pred.shape == y_batch.shape
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()

        global_step += 1
        batch_size = x_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    train_loss = total_loss / total_examples
    val_loss = evaluate(model, val_loader)

    if epoch == 1 or epoch % 20 == 0:
        print(
            f"epoch={epoch:3d} | steps={global_step:3d} | "
            f"train_mse={train_loss:.6f} | val_mse={val_loss:.6f} | "
            f"w={model.weight.item():.4f} | b={model.bias.item():.4f}"
        )

model.eval()
with torch.no_grad():
    prediction = model(torch.tensor([[3.0]])).item()
print(f"Pour x=3, prédiction={prediction:.4f}, cible=6")
```

### 7.3 Ce que tu dois observer

- `len(train_dataset)` vaut 8 et `len(train_loader)` vaut 4.
- Après 100 epochs, le compteur atteint **400 steps**.
- Le poids se rapproche de **2** et le biais de **0**.
- Les MSE d’entraînement et de validation deviennent très petites.
- La prédiction pour `x = 3` se rapproche de **6**.

Une valeur affichée `0.000000` peut être une petite valeur arrondie, sans être exactement zéro. La seed aide à reproduire l’expérience ; elle ne garantit pas des résultats identiques entre toutes les versions et plateformes.

Ce problème est volontairement facile : données exactes, modèle adapté, deux paramètres seulement. Réussir ici ne prouve pas qu’un LLM généralisera. Prédire pour `x = 3`, hors de l’intervalle d’entraînement, fonctionne ici parce que la vraie relation est linéaire et que notre modèle l’est aussi.

### 7.4 Pourquoi multiplier la loss par la taille du batch ?

`MSELoss()` renvoie par défaut une moyenne. Pour calculer la moyenne sur tous les exemples, on additionne les erreurs pondérées par les effectifs, puis on divise par le nombre total d’exemples.

Supposons un batch de 4 exemples avec une loss de 1, puis un batch de 2 avec une loss de 4 :

```text
Moyenne des deux losses : (1 + 4) / 2 = 2.5
Moyenne par exemple :    (4 × 1 + 2 × 4) / 6 = 2.0
```

La seconde donne le même poids à chaque exemple. Notre calcul convient ici car chaque exemple a exactement une valeur cible. Pour des séquences de tokens de longueurs différentes, il faudra tenir compte des positions réellement évaluées.

### 7.5 Comment lire train et validation ?

La loss `train_mse` est calculée pendant l’apprentissage : chaque batch utilise les paramètres disponibles **avant sa propre correction**. Les paramètres changent donc au fil de l’epoch.

La loss `val_mse` utilise les paramètres de fin d’epoch, fixes pendant toute la validation. Les deux nombres ne mesurent donc pas exactement le même état du modèle sur les mêmes données. Une validation légèrement inférieure à la loss train n’est pas forcément un bug.

Pour mesurer aussi le train avec les paramètres fixes de fin d’epoch, appelle `evaluate(model, train_loader)` après l’epoch. Pense à réactiver `model.train()` avant de reprendre l’apprentissage.

## 8. `train()`, `eval()` et `no_grad()` : deux réglages différents

| Instruction | Effet | Modifie les poids à elle seule ? |
| --- | --- | --- |
| `model.train()` | Active le comportement d’entraînement des couches concernées | Non |
| `model.eval()` | Active leur comportement d’évaluation | Non |
| `with torch.no_grad():` | Désactive l’enregistrement autograd dans le bloc pour ces calculs | Non |
| `optimizer.step()` | Applique la mise à jour à partir des gradients présents | Oui, dans notre entraînement |

Par exemple, le **dropout** masque aléatoirement certaines activations en mode train et cesse de le faire en mode eval. Notre couche `Linear` seule n’a pas de comportement différent entre ces modes ; nous les utilisons pour apprendre une structure qui restera correcte avec des réseaux plus complexes.

**`eval()` ne désactive pas les gradients. `no_grad()` ne passe pas le modèle en mode eval.** Pour une validation ordinaire, on utilise les deux : `model.eval()` puis un bloc `with torch.no_grad():`.

La validation mesure une performance ; elle ne doit appeler ni `backward()` ni `optimizer.step()`.

## 9. Laboratoire : apprendre en changeant une seule chose

Avant chaque expérience, repars du script complet : recrée le modèle **et** l’optimizer, avec la même seed. Continuer avec un modèle déjà entraîné fausserait la comparaison.

Conserve ce journal, même si tu ne remplis que quelques lignes :

| Expérience | LR | Batch | Epochs | Steps | MSE validation avant/après | `w`, `b` finaux | Ton explication |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Référence | 0.05 | 2 | 100 | 400 | À remplir | À remplir | À remplir |

### Expérience A — Faire varier le learning rate

Exécute séparément le script avec `lr=0.00001`, `lr=0.05`, puis `lr=1.0`.

**Prédis :** lequel devrait progresser lentement ? Lequel risque de dépasser les bonnes valeurs ?

**Observe :** validation initiale et finale, poids, biais, éventuelles valeurs très grandes ou non finies (`inf`, `nan`). Pour ce problème, le LR très faible ralentit l’apprentissage et le LR de 1.0 provoque une instabilité. Un LR adapté dépend aussi des données, du modèle et de l’optimizer : ces valeurs ne sont pas des recommandations pour un LLM.

### Expérience B — Observer l’accumulation accidentelle

Dans le script de la section 5, remplace l’unique `loss.backward()` par :

```python
loss.backward()
print("Après premier backward :", model.weight.grad.item())

# Nouveau forward, mêmes données et mêmes paramètres : aucun step entre les deux.
loss_2 = loss_fn(model(x), y)
loss_2.backward()
print("Après second backward :", model.weight.grad.item())
```

**Prédis :** le second gradient sera-t-il `−10` ou `−20` ?

Tu dois obtenir `−10`, puis `−20`. Nous recréons le calcul pour le second backward : réutiliser le même graphe déjà parcouru pose un autre problème. Replace ensuite la version d’origine.

### Expérience C — Changer la taille des batches

Compare `batch_size=2` et `batch_size=8` sur le train, avec 100 epochs.

**Prédis :** combien de steps chaque expérience effectuera-t-elle ?

Tu obtiendras 400 contre 100 steps. Compare les erreurs, mais explique que tu compares aussi des nombres différents de mises à jour. Même à nombre de steps égal, les batches ne fournissent pas nécessairement les mêmes gradients.

### Expérience D — Remplacer SGD par AdamW

Remplace seulement la création de l’optimizer :

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
```

AdamW utilise notamment un historique des gradients pour adapter ses mises à jour. La formule simple de SGD de la section 4 ne décrit donc plus son calcul. Ici, `weight_decay=0.0` désactive la pénalisation des poids pour garder l’expérience centrée sur l’optimizer.

Compare les résultats, sans conclure qu’un optimizer est toujours meilleur. Le même LR ne représente pas une correction équivalente pour deux algorithmes différents.

## 10. Diagnostiquer les erreurs courantes

| Symptôme | Hypothèse à vérifier | Vérification concrète |
| --- | --- | --- |
| Les poids restent identiques | `step()` absent ou gradients absents | Afficher un poids avant/après step et son `.grad` après backward |
| La loss baisse très lentement | LR très faible | Refaire l’expérience de référence avec `lr=0.05` |
| La loss explose ou devient `nan` | Mise à jour trop grande | Repartir d’un modèle neuf et réduire le LR |
| Les corrections semblent anormales | Gradients accumulés par oubli | Vérifier `zero_grad()` à chaque batch |
| Avertissement sur les tailles | Formes incompatibles | Afficher `pred.shape` et `y_batch.shape` |
| `backward()` échoue après un forward | Calcul sous `no_grad()` ou loss détachée | Vérifier le contexte et `loss.requires_grad` |
| Le train baisse, la validation monte | Possible surapprentissage ou décalage des données | Vérifier les données et les courbes ; voir les modules 11 et 14 |

Une hypothèse est un point de départ pour vérifier, pas une preuve. Dans ce module, commence par retrouver le résultat de référence avant d’ajouter des techniques plus avancées.

## 11. Exercices de compréhension, avec corrigés

Essaie de répondre sans relire le code. Ouvre les corrigés après ton essai.

### Exercice 1 — Compter

Tu as 100 exemples, des batches de 10 et 3 epochs. Un step suit chaque batch. Combien de steps effectues-tu ? Et avec 103 exemples, si tu gardes le dernier batch ?

<details>
<summary>Corrigé</summary>

Pour 100 exemples : `100 / 10 = 10` steps par epoch, donc **30 steps**.

Pour 103 exemples : 10 batches de 10 et un batch de 3, donc 11 steps par epoch et **33 steps**. Les exemples de validation ne comptent pas dans les steps d’apprentissage.

</details>

### Exercice 2 — Retrouver l’ordre

Remets dans l’ordre : `step`, calcul de loss, remise à zéro des gradients, `backward`, forward. Quelle instruction change les paramètres ?

<details>
<summary>Corrigé</summary>

La convention de ce module est : **zero_grad → forward → loss → backward → step**. C’est `step` qui change les paramètres. Le gradient doit être calculé avant que l’optimizer l’utilise.

</details>

### Exercice 3 — Comprendre le signe

Un poids vaut `1.0`, son gradient vaut `−3.0` et le LR vaut `0.1`. Avec SGD sans momentum, quelle sera sa nouvelle valeur ?

<details>
<summary>Corrigé</summary>

`1.0 − 0.1 × (−3.0) = 1.3`. Soustraire un gradient négatif augmente le poids. Cela correspond à la direction locale de descente ; un pas trop grand peut quand même augmenter la loss.

</details>

### Exercice 4 — Réparer une validation

Un camarade écrit : « J’ai appelé `model.eval()`, donc mes calculs ne suivent plus les gradients. » Est-ce correct ? Que faut-il ajouter ?

<details>
<summary>Corrigé</summary>

Non. `eval()` règle le comportement de certaines couches. Il faut placer le forward et le calcul de loss de validation dans `with torch.no_grad():` pour ne pas enregistrer leur graphe autograd. Aucun backward ni step ne doit être exécuté pendant cette validation.

</details>

### Exercice 5 — Reconstruire la boucle

Dans un nouveau fichier, écris de mémoire une boucle qui apprend `y = 3x + 1`. Réserve deux entrées à la validation, affiche les erreurs avant/après et les paramètres finaux.

<details>
<summary>Indices et critères de réussite</summary>

Reprends la structure du script complet, mais change la fabrication des cibles d’entraînement **et** de validation en `3 * x + 1`. Recrée le modèle et l’optimizer avant l’entraînement.

Les valeurs attendues sont `w ≈ 3`, `b ≈ 1`. Sur ce problème simple, avec les réglages de référence, vise une erreur absolue inférieure à `0.01` sur chaque paramètre et une MSE de validation inférieure à `0.001`. Pour `x = 3`, la prédiction doit être proche de 10.

Explique aussi pourquoi écrire `pred = 3 * x_batch + 1` dans le forward contournerait l’apprentissage : tu aurais fourni directement la réponse au lieu d’entraîner les paramètres.

</details>

## 12. Ce que cette boucle devient pour un LLM

| Notre exemple | Entraînement d’un LLM |
| --- | --- |
| Nombres en entrée | Séquences d’identifiants de tokens |
| Une couche `Linear` | Un réseau Transformer |
| Deux paramètres | De nombreux paramètres, parfois des milliards |
| Cibles numériques | Tokens attendus aux positions évaluées |
| MSE | Généralement cross-entropy pour la prédiction de tokens |
| Batches de couples `x, y` | Batches de séquences et de masques |

Le mécanisme **prédire → mesurer → backward → mettre à jour** reste central. Les masques, la mémoire et les détails de préparation des cibles demandent ensuite des précautions supplémentaires.

Un trainer de haut niveau automatise une grande partie de ce travail. Comprendre la boucle te permet de savoir ce qu’il doit accomplir et de rechercher l’origine d’un problème.

## 13. Validation — avant de continuer

Tu es prêt pour la suite lorsque tu peux, sans recopier le cours :

- [ ] Expliquer ce que représentent `x`, `y`, `pred`, `w` et `b`.
- [ ] Retrouver la loss initiale de 10 et les paramètres après la première correction.
- [ ] Dire pourquoi `backward()` et `step()` sont deux opérations différentes.
- [ ] Expliquer ce que remet à zéro `zero_grad()`.
- [ ] Compter les batches et les steps, y compris un dernier batch incomplet.
- [ ] Écrire une validation avec `eval()` et `no_grad()`.
- [ ] Expliquer pourquoi une baisse de loss train seule ne suffit pas à prouver la généralisation.
- [ ] Faire apprendre `y = 3x + 1` et commenter tes résultats.

Si un point bloque, reviens au micro-exemple correspondant, prédis son résultat et refais-le. La capacité à expliquer ton expérience compte davantage que le nombre de lignes exécutées.

## Fiche mémo

```text
Préparer : données → batches → modèle → loss → optimizer

Pour chaque epoch :
    model.train()
    Pour chaque batch d'entraînement :
        zero_grad → forward → loss → backward → step
    model.eval()
    Sous no_grad : mesurer la validation, sans correction

Loss = mesure de l'erreur
Gradient = sensibilité locale de la loss aux paramètres
Optimizer step = mise à jour des paramètres
```

Le [module 9](09_construire_un_dataset_de_qualite.md) passe à la matière première de cet apprentissage : les données. Une boucle correcte apprend à partir du signal qu’on lui donne ; il faut donc aussi examiner la qualité des exemples.

## Pour vérifier les API ou approfondir

Ces références officielles complètent les explications du module :

- [Tutoriel PyTorch : optimiser les paramètres d’un modèle](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — structure générale d’une boucle d’apprentissage.
- [Documentation de `nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) — comportements de `train()` et `eval()`.
- [Documentation de `Optimizer.zero_grad`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) — remise à zéro et option `set_to_none`.
