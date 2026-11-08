# Module 18 — Régler le rang, alpha et le dropout LoRA

## 1. Ce que tu vas apprendre

Trois réglages LoRA peuvent changer simultanément la capacité, l’amplitude et le bruit de l’adaptation. Pour comprendre une différence de résultat, il faut les distinguer.

Tu vas calculer leurs effets, observer le dropout et concevoir une comparaison de rangs qui annonce son facteur d’échelle.

**Prérequis :** modules 15 à 17 ; correction BA et comptage des paramètres.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et PyTorch sur CPU. Le laboratoire ne nécessite pas de modèle de base.

## 2. Trois rôles différents

| Réglage | Ce qu’il contrôle principalement | Ce qu’il n’est pas |
| --- | --- | --- |
| Rang `r` | Dimension interne, donc contrainte de capacité et coût | Nombre de blocs adaptés |
| `alpha` avec la règle de scaling | Amplitude de la branche calculée | Learning rate |
| Dropout | Masquage aléatoire d’entrées de la branche en mode train | Quantification des poids |

Le rang est une borne sur le rang de BA. Un rang autorisé élevé ne garantit pas que la correction apprise utilise toutes ses directions.

Pour une projection 4096 → 4096, `r=4`, `16`, `64` ajoutent respectivement **32 768**, **131 072**, **524 288** paramètres.

Un rang plus grand offre une correction moins contrainte, mais coûte davantage et peut suradapter. La loss train ne suffit pas à le choisir.

## 3. Le piège du scaling quand on compare les rangs

Avec LoRA classique :

\[
s=\frac{\alpha}{r},\qquad \Delta W=sBA
\]

Si alpha reste 16 et r passe de 4 à 16, le facteur passe de 4 à 1. Tu modifies donc **à la fois** la dimension de la correction et son facteur multiplicatif.

Pour comparer les rangs à scaling classique constant `s=2`, prends par exemple :

| r | alpha | alpha/r |
| --- | --- | --- |
| 4 | 8 | 2 |
| 16 | 32 | 2 |
| 64 | 128 | 2 |

Cela contrôle un facteur évident, sans rendre les dynamiques d’initialisation et d’optimisation identiques. Une comparaison reste une expérience à documenter.

Avec **rsLoRA**, PEFT utilise `alpha/sqrt(r)` lorsque cette variante est activée. Les mêmes nombres alpha/r ne décrivent alors plus le facteur réel. Annonce la variante et inspecte la configuration.

### Alpha et LR ne sont pas interchangeables

Alpha agit dans le forward et influence les gradients de la branche. Le LR agit dans la règle de mise à jour, avec l’historique éventuel de l’optimizer. Modifier l’un ne reproduit pas généralement l’effet de l’autre, surtout après plusieurs steps.

## 4. Comprendre le dropout de la branche

Une forme courante du chemin LoRA est :

```text
base(x) + scale × B(A(dropout(x)))
```

En mode train, `Dropout(p)` met certaines composantes à zéro et multiplie les autres par `1/(1−p)`. Pour `p=0.5`, les composantes conservées sont doublées.

C’est un masquage aléatoire à chaque appel, pas une suppression permanente de la moitié des poids. En mode eval, ce dropout laisse passer l’entrée.

Sa contribution moyenne et ses effets d’apprentissage ne garantissent pas une meilleure généralisation : sur un petit adapter ou un problème difficile, trop de bruit peut aussi nuire.

## 5. Laboratoire autonome — Coût, scaling et dropout

```python
import math
import torch
from torch import nn

for rank in [4, 16, 64]:
    parameters = rank * (4096 + 4096)
    alpha = 16
    print('r=', rank, 'parametres=', parameters,
          'scale classique=', alpha / rank,
          'scale rsLoRA=', alpha / math.sqrt(rank))

print('Scaling classique constant :')
for rank in [4, 16, 64]:
    alpha = 2 * rank
    print('r=', rank, 'alpha=', alpha, 'scale=', alpha / rank)

torch.manual_seed(42)
x = torch.ones(16)
dropout = nn.Dropout(p=0.5)
dropout.train()
train_a = dropout(x)
train_b = dropout(x)
dropout.eval()
eval_output = dropout(x)
print('Train A :', train_a.tolist())
print('Train B :', train_b.tolist())
print('Valeurs possibles en train :', sorted(set(train_a.tolist())))
print('Eval identique a x :', torch.equal(eval_output, x))
```

Tu dois retrouver les comptes précédents. Le scaling classique à alpha fixe vaut **4, 1, 0.25** ; le second tableau reste à **2**.

Les valeurs de dropout sur cette entrée sont 0 ou 2. Les deux tirages ne sont pas censés contenir exactement huit zéros, ni être toujours différents par principe. Le résultat eval doit être identique à l’entrée.

## 6. Concevoir une expérience lisible

Avant d’entraîner, écris ce qui reste fixe : base/révision, dataset et splits, tokenizer/template, cibles LoRA, nombre de tokens ou steps, batch effectif, LR/scheduler, règle de scaling et protocole d’évaluation.

Puis compare r=4,16,64 en documentant :

| Mesure | Pourquoi la conserver |
| --- | --- |
| Paramètres entraînables | Coût de l’adaptation |
| Validation et benchmark | Généralisation et comportement |
| Pic mémoire, temps et débit | Faisabilité réelle |
| Taille de l’adapter | Coût de stockage et de distribution |
| Régressions par catégorie | Effets cachés par la moyenne |

Choisis un compromis selon un seuil écrit avant la comparaison. Une différence d’une question sur un petit benchmark ne justifie pas automatiquement un adapter quatre fois plus gros.

## 7. Expériences guidées

1. Fais un tableau de rangs à alpha fixe puis à scaling fixe. Explique les deux questions expérimentales différentes.
2. Sur le laboratoire du module 16, compare r=1 et r=2 avec une règle de scaling annoncée. Recrée base et optimizer entre essais.
3. Fixe r et alpha, puis compare dropout 0 et 0.1 sur les mêmes données. Évalue en mode eval et garde la validation indépendante.
4. Ajoute une répétition avec une autre seed. Vérifie si l’écart entre configurations dépasse la variation entre répétitions.

## 8. Erreurs à éviter

- Comparer les rangs avec des cibles ou nombres de steps différents sans le signaler.
- Dire que seul r change alors que alpha/r change aussi.
- Assimiler alpha à un LR universel.
- Mesurer des sorties avec dropout actif puis attribuer la variation au checkpoint.
- Choisir le rang à partir de la seule loss train.

## 9. Exercices avec corrigés

### Exercice 1 — Une projection non carrée

Quel est le coût LoRA pour 4096 → 11008 et r=32 ?

<details>
<summary>Corrigé</summary>

`32×(4096+11008) = 483 328` paramètres, hors biais et autres modules.

</details>

### Exercice 2 — Facteur constant

Tu veux `s=2` en LoRA classique avec r=8 et r=32. Quels alpha ?

<details>
<summary>Corrigé</summary>

Alpha **16** puis **64**, car `alpha=2r`. Avec rsLoRA, il faudrait utiliser la règle correspondant à `alpha/sqrt(r)`.

</details>

### Exercice 3 — Rang très élevé

Pour une matrice H×H, r=H conserve-t-il une économie de paramètres ajoutés ?

<details>
<summary>Corrigé</summary>

A et B contiennent ensemble `2H²` paramètres, contre `H²` pour une correction complète. La contrainte de faible rang et son intérêt en nombre de paramètres disparaissent dans ce cas.

</details>

### Exercice 4 — Dropout

Avec p=0.25, quel facteur s’applique aux composantes conservées en mode train ? Et en eval ?

<details>
<summary>Corrigé</summary>

`1/(1−0.25)=4/3` en train pour les composantes conservées. En eval, le dropout ne masque ni ne multiplie ces composantes.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue capacité, scaling, LR et dropout.
- [ ] Je calcule les paramètres pour plusieurs rangs.
- [ ] J’annonce la règle classique ou rsLoRA utilisée.
- [ ] Je sais conserver un scaling classique constant.
- [ ] Je comprends la différence train/eval du dropout.
- [ ] Je construis un tableau coût/qualité avec validation et répétitions utiles.

## Fiche mémo

```text
r : forme de la correction et coût.
alpha + variante : facteur dans le forward.
LR : dynamique de mise à jour.
dropout : bruit de la branche pendant le train.
```

Le [module 19](19_choisir_les_target_modules.md) décide où placer cette capacité.

## Référence

[PEFT : LoRA et variantes](https://huggingface.co/docs/peft/en/package_reference/lora) — scaling, rsLoRA et dropout.
