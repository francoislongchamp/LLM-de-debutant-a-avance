# Module 21 — Comprendre le full fine-tuning

## 1. Ce que tu vas apprendre

Avec LoRA, la mise à jour passe par une correction factorisée. Avec le **full fine-tuning**, on autorise directement l’ensemble, ou presque, des paramètres du modèle à être optimisés.

Tu vas vérifier cette autorisation, observer les gradients et les changements réels, puis distinguer liberté de mise à jour, coût et qualité.

**Prérequis :** modules 5, 8, 13 et 15 à 20.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et PyTorch sur CPU. La miniature ne charge pas de LLM.

## 2. Deux questions indépendantes

| Question | Exemples de réponse |
| --- | --- |
| Quelles données et quel objectif ? | Texte brut causal, SFT, préférence |
| Quels paramètres peuvent changer ? | Full FT, LoRA, certaines couches seulement |

Le full FT n’est donc pas synonyme de CPT ni de SFT. On peut faire un SFT en full FT ou avec LoRA. On peut continuer un objectif de langage avec différents ensembles de paramètres entraînables.

Le **from scratch** part de poids aléatoires ; le fine-tuning part d’un état déjà appris. Notre laboratoire utilise une miniature aléatoire pour inspecter une mise à jour complète ; il ne démontre pas une adaptation linguistique de poids préentraînés.

## 3. Ce qui devient modifiable

Dans un LLM, embeddings, projections d’attention, MLP, normalisations et tête peuvent être entraînables. Les poids partagés restent un même ensemble de paramètres, même s’ils servent à plusieurs endroits.

```text
Full FT : W → W + mise à jour directe
LoRA    : W fixe ; correction représentée par sBA
```

Le full FT n’impose pas la contrainte de faible rang de LoRA à chaque update. Cette liberté supplémentaire peut être utile, mais ne garantit pas une meilleure généralisation ni une meilleure efficacité à budget fixé.

Tous les paramètres autorisés ne changent pas forcément à chaque step : certains gradients peuvent être nuls, une branche peut ne pas être utilisée ou une mise à jour peut être trop petite pour se voir à la précision courante.

## 4. Revenir au budget de mémoire

Pour les paramètres optimisés, il faut considérer les gradients et les états de l’optimizer. Full FT les ajoute à une beaucoup plus grande partie du réseau que LoRA.

| Poste | Full FT | LoRA classique |
| --- | --- | --- |
| Base | Présente | Présente |
| Gradients des poids principaux | Généralement oui | Base généralement gelée |
| États de mise à jour principaux | Généralement oui | Concentrés sur adapters et parties prévues |
| Activations pour backward | Présentes | Encore présentes selon les chemins nécessaires |
| Sauvegarde finale | Modèle complet | Souvent adapter séparé |

Une base 600M ne devient pas un « training de 1.2 GB » parce que ses poids BF16 occupent ce budget brut. Le module 5 a montré pourquoi les autres postes changent l’ordre de grandeur.

## 5. Laboratoire autonome — Une mise à jour complète

Prédis le nombre de paramètres : `Linear(2,4)` donne 12, `Linear(4,1)` donne 5, donc **17** au total.

```python
import torch
from torch import nn

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
model.requires_grad_(True)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
before = {name: p.detach().clone() for name, p in model.named_parameters()}
x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
y = x.sum(dim=-1, keepdim=True)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
optimizer.zero_grad(set_to_none=True)
loss = nn.functional.mse_loss(model(x), y)
loss.backward()
print('Total et entrainable :', total, trainable)
for name, p in model.named_parameters():
    print(name, 'gradient present=', p.grad is not None)
optimizer.step()
changed = [name for name, p in model.named_parameters()
           if not torch.equal(p.detach(), before[name])]
with torch.no_grad():
    after_loss = nn.functional.mse_loss(model(x), y).item()
print('Tenseurs modifies :', changed)
print('Loss avant/apres :', loss.item(), after_loss)
```

Les 17 paramètres sont entraînables et les quatre tenseurs de poids/biais participent au calcul. Pour cette configuration, ils doivent recevoir des gradients et des changements sont observables.

Une diminution de la loss sur ce batch montre une adaptation locale. Elle ne démontre pas la généralisation ni une conservation de capacités antérieures.

## 6. Que sauvegarde-t-on ?

Un modèle final contient les poids nécessaires à l’inférence, accompagnés de sa configuration et du tokenizer pour un LLM.

Un **checkpoint de reprise** doit aussi permettre de restaurer optimizer, scheduler, position dans l’entraînement et, si l’objectif est de reproduire la trajectoire, états aléatoires et données correspondantes.

Sauvegarder seulement les poids peut permettre de repartir de cette fonction, sans reprendre exactement la dynamique précédente d’AdamW. Les gros checkpoints peuvent être répartis en shards ; leur nombre ne change pas le sens de l’objectif.

## 7. Capacité, coût et risques à évaluer

Le full FT peut se justifier si les corrections souhaitées sont difficiles à obtenir avec une adaptation plus contrainte et si les données et le budget permettent une comparaison sérieuse. Cela reste une hypothèse expérimentale.

Un corpus étroit et des updates agressives peuvent dégrader des comportements auparavant utiles. C’est un risque d’**oubli**, à mesurer sur un benchmark général en plus du domaine.

LoRA n’est pas automatiquement immunisé contre ce risque ; full FT n’est pas automatiquement mauvais. Compare qualité, coût, facilité de sauvegarde et régressions avec des hyperparamètres adaptés à chaque méthode.

Copier le LR d’un adapter vers tous les poids peut modifier fortement la dynamique. Le LR `0.01` de notre petit réseau n’est pas un réglage conseillé pour un LLM.

## 8. Expériences guidées

1. Gèle seulement la première couche avant de créer l’optimizer. Recompte : il reste 5 paramètres entraînables. Vérifie que la première couche ne change pas.
2. Repars du même état initial et compare AdamW complet et un entraînement de la tête seule. Mesure loss train et résultats sur d’autres entrées, sans confondre budget et liberté.
3. Pour un petit modèle préentraîné déjà disponible, fais un seul step avec une courte séquence et mesure le pic mémoire. Compare ensuite LoRA dans un processus séparé.
4. Rédige le manifeste de sauvegarde et de reprise nécessaire pour ce modèle, en distinguant les deux usages.

## 9. Exercices avec corrigés

### Exercice 1 — Objectif ou paramètres ?

« Je fais du CPT, donc je ne peux pas utiliser LoRA. » Est-ce correct ?

<details>
<summary>Corrigé</summary>

Non. CPT décrit la continuation d’un objectif sur des données, LoRA la paramétrisation des mises à jour. Leur compatibilité et leur efficacité doivent être étudiées, mais les termes ne sont pas opposés par définition.

</details>

### Exercice 2 — Tous modifiés ?

Tous les paramètres ont `requires_grad=True`. Peut-on affirmer qu’ils changent tous d’une quantité non nulle à chaque step ?

<details>
<summary>Corrigé</summary>

Non. Il faut vérifier participation au graphe, gradient, optimizer et valeur effectivement modifiée. Le flag autorise le suivi ; il ne mesure pas l’amplitude de l’update.

</details>

### Exercice 3 — Poids seuls

Pourquoi un modèle peut-il tenir pour générer mais pas pour être entraîné en full FT ?

<details>
<summary>Corrigé</summary>

L’entraînement ajoute notamment gradients, états d’optimizer et activations conservées pour backward. La mémoire des poids seuls ne budgète pas le step complet.

</details>

### Exercice 4 — Checkpoint final ou reprise ?

Que manque-t-il à une sauvegarde de poids pour reprendre exactement AdamW ?

<details>
<summary>Corrigé</summary>

Ses moments et compteurs, ainsi que le scheduler et les autres états du run nécessaires à la reprise. Charger les poids puis créer un nouvel optimizer redémarre une autre dynamique.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue full FT, objectif SFT/CPT et départ from scratch.
- [ ] Je retrouve les 17 paramètres de la miniature.
- [ ] Je compare autorisation, gradients et modifications effectives.
- [ ] Je sais budgéter les postes supplémentaires d’un entraînement complet.
- [ ] Je distingue sauvegarde d’inférence et checkpoint de reprise.
- [ ] Je propose un benchmark de conservation des capacités.

## Fiche mémo

```text
Full FT : autoriser directement les paramètres principaux à changer.
Plus de liberté n’est pas une garantie de meilleure qualité.
Mesurer update, mémoire et régressions ; conserver une base de comparaison.
```

Le [module 22](22_full_ft_configuration_stable.md) construit une configuration surveillée.
