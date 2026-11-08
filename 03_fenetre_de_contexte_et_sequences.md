# Module 3 — Préparer des séquences et respecter la fenêtre de contexte

## 1. Ce que tu vas apprendre

Au [module 2](02_tokenisation_en_profondeur.md), tu as transformé des textes en listes d’IDs. Ces listes n’ont généralement pas la même longueur. Comment les regrouper ? Que faire lorsqu’un texte dépasse la limite ? Comment empêcher le réseau de consulter des positions interdites ?

À la fin, tu devrais pouvoir :

- lire les axes `[batch, sequence]` d’un tenseur ;
- calculer un budget de contexte qui inclut l’entrée et la sortie ;
- préparer un batch avec du padding et mesurer sa proportion ;
- dessiner un masque causal ;
- distinguer visibilité dans l’attention et participation à la loss ;
- examiner ce qu’une troncature supprime.

**Prérequis :** modules 1 et 2 ; listes et boucles Python. Un tenseur sera ici un tableau rectangulaire de nombres.

**Temps indicatif :** 90 à 150 minutes, laboratoire compris.

**Matériel :** Python et PyTorch, sur CPU. Aucun modèle à télécharger.

## 2. Longueur de séquence et fenêtre de contexte

Une **séquence** est une liste ordonnée de tokens. Sa **longueur** est son nombre de positions.

```text
IDs = [11, 12, 13, 14] → longueur 4
```

La **fenêtre de contexte** est le budget de positions utilisables dans une configuration donnée du modèle. Elle se compte en tokens, pas en mots ni en caractères.

Pour un modèle causal standard dont la séquence reste dans une fenêtre de taille `C`, il faut prévoir :

\[
L_{entrée\ complète} + L_{nouveaux\ tokens} \leq C
\]

L’entrée complète inclut les marqueurs, instructions, historique et documents effectivement transmis au modèle.

### Exemple chiffré

Supposons une fenêtre de 4 096 positions :

| Élément | Tokens |
| --- | --- |
| Instructions et marqueurs | 96 |
| Historique | 900 |
| Document et question | 2 500 |
| Total d’entrée | 3 496 |
| Place restante pour générer | 600 |

Demander 800 nouveaux tokens dépasse le budget de 200. Il faut réduire l’entrée, la sortie ou changer de configuration adaptée ; changer seulement une variable ne donne pas magiquement plus de capacité au réseau.

La limite annoncée ne garantit pas une qualité constante jusqu’au dernier token. La mémoire nécessaire et le comportement sur des contextes longs comptent aussi. Les architectures à fenêtres glissantes et autres stratégies demandent des règles plus spécifiques ; nous raisonnons ici avec une fenêtre fixe simple.

> **Pause compréhension.** Si tu demandes `max_new_tokens=600`, le modèle produira-t-il forcément 600 tokens ?
>
> Non. C’est un maximum ; un token de fin peut arrêter la génération avant.

## 3. Former un batch rectangulaire

Un **batch** regroupe plusieurs séquences traitées ensemble. Avec des tenseurs denses ordinaires, elles doivent avoir la même longueur pour former un tableau rectangulaire.

Prenons deux séquences fictives :

```text
A = [11, 12, 13, 14]
B = [21, 22]
```

Le **padding** ajoute des positions de remplissage à la plus courte. Choisissons l’ID fictif `0` pour PAD :

```text
input_ids = [[11, 12, 13, 14],
             [21, 22,  0,  0]]
```

La forme est `[2, 4]` : deux séquences, quatre positions chacune. Ce n’est ni « deux tokens » ni « quatre exemples ».

Les 0 ne sont du padding que parce que nous avons défini cette convention. Un ID nul n’est pas universellement un token PAD.

### Padding dynamique ou longueur fixe

- **Dynamique** : compléter jusqu’à la plus longue séquence du batch courant.
- **À longueur fixe** : compléter jusqu’à une limite choisie, par exemple 64.

Le padding ne doit pas servir à couper les textes trop longs : cette opération distincte est la troncature.

## 4. Trois questions différentes, trois mécanismes

### 4.1 Où sont les positions réelles ?

Un masque de padding courant, souvent nommé `attention_mask` dans Transformers, utilise :

```text
A mask = [1, 1, 1, 1]
B mask = [1, 1, 0, 0]
```

Les positions à 0 sont du remplissage. Ce masque aide l’attention à ne pas utiliser les clés de padding comme contenu pertinent.

Il ne garantit pas que tous les vecteurs calculés aux positions PAD deviennent nuls. Il ne remplace pas non plus le masque de loss.

### 4.2 Quelles positions peuvent se consulter ?

Le **masque causal** empêche une position d’utiliser les tokens futurs.

Pour une séquence de quatre positions, voici la matrice des relations autorisées. Chaque **ligne est la position qui consulte**, chaque **colonne la position consultée** :

```text
             clé 0  clé 1  clé 2  clé 3
requête 0      1      0      0      0
requête 1      1      1      0      0
requête 2      1      1      1      0
requête 3      1      1      1      1
```

La position 1 peut utiliser les positions 0 et 1. Sa sortie prédit le token suivant, à la position 2 ; elle ne doit donc pas voir ce token à l’avance.

Grâce à cette contrainte, on peut calculer les sorties de plusieurs positions en parallèle pendant l’entraînement sans leur donner le futur. Pendant la génération, les nouveaux tokens n’existent pas encore et sont produits progressivement.

### 4.3 Quelles cibles contribuent à l’erreur ?

Un **masque de loss** exclut certaines cibles du calcul de l’erreur. Au module 6, nous utiliserons la valeur `-100` dans des labels avec `cross_entropy`.

| Mécanisme | Question traitée |
| --- | --- |
| Masque de padding pour l’attention | Où se trouve le contenu réel ? |
| Masque causal | Quelles relations entre positions sont autorisées ? |
| Masque de loss | Quelles cibles doivent compter dans l’erreur ? |

Un prompt peut être visible pour produire une réponse tout en étant exclu des cibles supervisées. Masquer sa loss ne le supprime pas de l’entrée.

Les conventions des API diffèrent : une valeur booléenne `True` peut signifier « autorisé » ou « interdit » selon la fonction. Dans ce module, nos matrices sont explicitement appelées `allowed` et `True` signifie **autorisé**. Nous ne les passons pas telles quelles à une API d’attention.

## 5. Laboratoire A — Construire et examiner le batch

Ce script autonome utilise des IDs fictifs. Il ne fait pas d’inférence et ne charge aucun tokenizer. Prédis les formes et la fraction de padding avant de l’exécuter.

```python
import torch

sequences = [[11, 12, 13, 14], [21, 22]]
pad_id = 0
length = max(len(sequence) for sequence in sequences)

padded = []
masks = []
for sequence in sequences:
    missing = length - len(sequence)
    padded.append(sequence + [pad_id] * missing)
    masks.append([1] * len(sequence) + [0] * missing)

input_ids = torch.tensor(padded, dtype=torch.long)
attention_mask = torch.tensor(masks, dtype=torch.long)
causal_allowed = torch.tril(torch.ones(length, length, dtype=torch.bool))

print("IDs :\n", input_ids)
print("Masque de padding :\n", attention_mask)
print("Forme :", tuple(input_ids.shape))
print("Relations causales autorisées :\n", causal_allowed.int())

real_positions = attention_mask.sum().item()
total_positions = input_ids.numel()
padding_fraction = 1 - real_positions / total_positions
print("Positions réelles :", real_positions, "/", total_positions)
print("Fraction de padding :", padding_fraction)

# Pour B seulement : une clé doit être réelle ET ne pas être dans le futur.
valid_keys = attention_mask[1].bool()
allowed_for_b = causal_allowed & valid_keys.unsqueeze(0)
print("Relations autorisées pour B :\n", allowed_for_b.int())
```

Tu dois obtenir :

- une forme **`[2, 4]`** ;
- **6** positions réelles sur **8** ;
- une fraction de padding de **0.25**, soit **25 %** ;
- aucune relation autorisée vers les colonnes de padding de B.

Les lignes correspondant aux requêtes PAD peuvent encore contenir des relations autorisées vers de vraies clés. Leurs sorties ne deviennent pas pour autant des cibles utiles : c’est une raison de traiter explicitement les labels de padding lors du calcul de loss.

### D’où vient le calcul de padding ?

\[
\text{fraction de padding}
= \frac{B L - \sum_i L_i}{B L}
\]

`L_i` est la longueur réelle de la séquence `i`, et `L` la longueur après padding. Sur ce batch : `(2 × 4 − (4 + 2)) / 8 = 0.25`.

Avec des calculs denses, un masque ne supprime pas automatiquement tout le travail associé aux cases PAD. Les implémentations optimisées peuvent traiter ce problème autrement ; le ratio reste ici un premier indicateur de remplissage.

## 6. Troncature : voir ce qui est perdu

La **troncature** supprime des tokens pour respecter une limite. Contrairement au padding, elle fait perdre du contenu.

### Laboratoire B — Inspecter les deux côtés

Ce bloc est autonome, en Python seulement. Nos mots représentent ici des unités fictives, pour que la perte soit immédiatement visible ; un vrai tokenizer peut découper chaque mot en plusieurs tokens.

```python
units = ["Explique", "ce", "code", "sans", "omettre", "la", "condition", "finale"]
max_length = 5

kept_right_cut = units[:max_length]
removed_right_cut = units[max_length:]
kept_left_cut = units[-max_length:]
removed_left_cut = units[:-max_length]

print("Original :", units)
print("Couper à droite :", kept_right_cut)
print("Supprimé :", removed_right_cut)
print("Couper à gauche :", kept_left_cut)
print("Supprimé :", removed_left_cut)
```

Couper à droite supprime ici la fin de l’instruction. Couper à gauche conserve la fin mais supprime le début. Aucun côté n’est toujours sûr : il faut examiner le contenu et l’objectif.

Pour un document long, on peut envisager de choisir un passage pertinent, découper en segments ou utiliser une autre configuration. Découper demande aussi de décider du chevauchement et du contexte nécessaire ; des fragments voisins ne deviennent pas automatiquement indépendants pour l’évaluation.

## 7. Padding à gauche ou à droite

```text
À droite : [21, 22, PAD, PAD]
À gauche : [PAD, PAD, 21, 22]
```

Le choix dépend du modèle et du mode d’utilisation. Pour la génération par lots avec de nombreux modèles *decoder-only* dans Transformers, le padding à gauche est recommandé : la dernière colonne correspond alors au dernier token réel de chaque entrée.

Cela explique une limite de l’expression `logits[:, -1]` : avec du padding à droite, elle sélectionnerait des positions PAD pour les lignes courtes. Un masque d’attention ne déplace pas les logits vers la dernière colonne.

Le tokenizer et le modèle doivent aussi disposer d’une configuration cohérente des positions et du padding. Réutiliser EOS comme PAD peut fonctionner dans certaines configurations, mais il faut conserver la distinction via le masque et ne pas supprimer aveuglément les vrais EOS des cibles.

## 8. Laboratoire C — Regrouper par longueur

Ce script autonome compare deux façons de former des batches de deux séquences. Il compte les cases, sans mesurer directement un temps d’exécution.

```python
lengths = [2, 8, 3, 7]


def summarize(order):
    real = sum(order)
    slots = 0
    for index in range(0, len(order), 2):
        batch_lengths = order[index:index + 2]
        slots += len(batch_lengths) * max(batch_lengths)
    return slots, (slots - real) / slots


print("Ordre initial :", summarize(lengths))
print("Longueurs proches :", summarize(sorted(lengths)))
```

Tu dois obtenir **30 positions, environ 33,3 % de padding** dans le premier cas, contre **22 positions, environ 9,1 %** dans le second. Le nombre de tokens réels reste **20**.

C’est l’idée du regroupement par longueur. Cela peut économiser du calcul dense, mais ne signifie pas qu’il faut trier toutes les données dans un ordre fixe à chaque epoch : l’ordre et la stratégie d’échantillonnage restent à concevoir.

## 9. Exercices avec corrigés

### Exercice 1 — Calculer un budget

Un prompt complet occupe 3 000 tokens. Tu veux autoriser 1 500 nouveaux tokens. Quelle longueur totale faut-il prévoir ? Une fenêtre de 4 096 suffit-elle ?

<details>
<summary>Corrigé</summary>

Il faut prévoir **4 500 positions**. La fenêtre de 4 096 est trop courte de **404 positions**. Elle laisserait au maximum 1 096 nouveaux tokens avec cette entrée dans notre modèle de fenêtre fixe.

</details>

### Exercice 2 — Compter les cases

Trois séquences de longueurs 2, 3 et 5 sont paddées dynamiquement dans le même batch. Quelle forme obtient-on et quelle fraction est du padding ?

<details>
<summary>Corrigé</summary>

Forme **`[3, 5]`** : 15 cases, dont 10 réelles et 5 de padding. La fraction est `5/15`, soit environ **33,3 %**.

</details>

### Exercice 3 — Comprendre le futur

À la position 2 d’une séquence indexée depuis 0, quelles positions sont autorisées par un masque causal ? Quel token sa sortie essaie-t-elle de prédire ?

<details>
<summary>Corrigé</summary>

Les positions **0, 1 et 2** sont accessibles. La sortie sert à prédire le token de position **3**. Lui permettre de voir ce token fournirait directement une partie de la réponse attendue.

</details>

### Exercice 4 — Distinguer les masques

Tu exclus la question utilisateur des cibles de loss pour superviser seulement la réponse assistant. Le modèle peut-il encore utiliser la question ?

<details>
<summary>Corrigé</summary>

Oui, si elle reste dans l’entrée et qu’elle est visible dans l’attention. Le masque de loss choisit les cibles évaluées ; il ne supprime pas le contexte.

</details>

### Exercice 5 — Diagnostiquer une dernière position

Un batch est paddé à droite et le programme choisit toujours `logits[:, -1]` pour démarrer la génération. Quel problème rechercher pour les séquences courtes ?

<details>
<summary>Corrigé</summary>

La dernière colonne peut correspondre au padding. Le programme utilise alors la sortie d’une position artificielle au lieu du dernier token réel. Il faut suivre la préparation recommandée pour ce modèle et cette API, notamment le padding à gauche pour les cas decoder-only concernés.

</details>

## 10. Validation — avant de continuer

- [ ] Je calcule un budget d’entrée complète et de sortie.
- [ ] Je sais lire `[B, L]` et calculer la fraction de padding.
- [ ] Je dessine les relations d’un masque causal 4 × 4.
- [ ] Je distingue masque de padding, masque causal et masque de loss.
- [ ] Je sais que les conventions booléennes des API doivent être vérifiées.
- [ ] Je peux montrer ce qu’une troncature supprime.
- [ ] Je peux expliquer le gain de cases du regroupement par longueur.

## Fiche mémo

```text
Padding : ajouter des cases artificielles pour former le batch.
Troncature : retirer du contenu ; inspecter ce qui disparaît.
Masque de padding : indiquer le contenu réel.
Masque causal : empêcher l’accès au futur.
Masque de loss : choisir les cibles évaluées.
Budget de contexte : compter l’entrée complète et la sortie prévue.
```

Le [module 4](04_anatomie_du_transformer.md) montre comment les vecteurs et les masques interviennent à l’intérieur du réseau.

## Références

- [Hugging Face : padding et troncature](https://huggingface.co/docs/transformers/main/en/pad_truncation) — préparation de séquences de longueurs différentes.
- [Hugging Face : génération de texte](https://huggingface.co/docs/transformers/en/llm_tutorial) — précautions pour la génération par lots et le padding.
