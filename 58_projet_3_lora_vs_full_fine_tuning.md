# Module 58 — Projet 3 : comparer LoRA et full fine-tuning sans cacher les coûts

## 1. Ce que tu vas construire

Deux branches partent du même état appris. L’une utilise une correction LoRA, l’autre autorise directement les poids principaux à changer.

Tu vas réaliser un pilote numérique contrôlé, puis construire la comparaison sur un petit LLM avec domaine, capacités à préserver et budget explicites.

**Prérequis :** modules 15 à 24, 50 à 56.

**Temps indicatif :** pilote court sur CPU ; projet réel de plusieurs heures selon le modèle et le tuning.

**Matériel :** PyTorch et PEFT pour le pilote ; ressources adaptées au step complet du LLM pour la suite.

## 2. La bonne question expérimentale

Full FT dispose d’une liberté différente de LoRA, pas d’une qualité automatiquement supérieure. La question est de savoir si un gain utile justifie le coût et les régressions sous les contraintes choisies.

Deux protocoles sont possibles : réglages aussi fixes que possible pour une ablation locale, ou réglages adaptés à chaque méthode avec un budget de tuning comparable.

Le même LR n’est pas une garantie d’équité. Les données, splits, scoring et identité de base doivent en revanche être cohérents avec la question.

## 3. Pilote contrôlé : correction connue

Notre petite matrice 8×8 démarre du même état dans les deux branches. Une correction de rang 1 est ajoutée à la cible ; LoRA r=2 peut donc la représenter.

Le « vieux comportement » et le nouveau sont volontairement en conflit sur les mêmes entrées. L’erreur sur l’ancienne sortie rend le déplacement visible, sans prétendre mesurer le catastrophic forgetting d’un vrai LLM.

Les entrées de validation sont différentes du train. Le montage reste artificiel et beaucoup plus simple qu’un corpus linguistique.

## 4. Laboratoire autonome — Deux branches, mêmes données

```python
from copy import deepcopy
import time
import torch
from torch import nn
from peft import LoraConfig, get_peft_model

class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)

torch.manual_seed(42)
initial = Base()
train_x = torch.randn(32, 8)
val_x = torch.randn(16, 8)
with torch.no_grad():
    train_y = initial(train_x) + train_x.sum(-1, keepdim=True).expand(-1, 8) / 8
    val_y = initial(val_x) + val_x.sum(-1, keepdim=True).expand(-1, 8) / 8
    old_val_y = initial(val_x).clone()
reports = []
for method, lr in [('LoRA', 0.05), ('FullFT', 0.03)]:
    torch.manual_seed(123)
    model = deepcopy(initial)
    if method == 'LoRA':
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0,
                                               target_modules=['proj'], bias='none'))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=0.0)
    model.train()
    start = time.perf_counter()
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        domain_loss = nn.functional.mse_loss(model(val_x), val_y).item()
        old_loss = nn.functional.mse_loss(model(val_x), old_val_y).item()
    report = {'method': method, 'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'steps': 100, 'example_exposures': 3200, 'domain_validation_mse': domain_loss,
              'old_behavior_mse': old_loss, 'training_seconds_cpu': elapsed, 'lr': lr}
    reports.append(report)
    print(report)
```

Le nombre entraînable doit être **32 pour LoRA** et **64 pour FullFT**. Chaque branche effectue **100 updates** et **3 200 expositions d’exemples**.

La loss de domaine doit devenir petite pour cette correction facile. L’ancienne sortie peut se dégrader parce que nous demandons un changement incompatible avec son maintien exact. Il n’y a pas de vainqueur universel prédéfini.

Les temps CPU du pilote incluent des effets de première utilisation et de petite taille. Ils montrent comment collecter une durée, pas une comparaison de performances GPU robuste. Les deux LR sont annoncés, mais n’ont pas été optimisés selon une étude de tuning : la portée de ce pilote reste limitée.

## 5. Construire la vraie comparaison

1. Choisis une base qui tient en full FT avec les états et longueurs cibles.
2. Fixe dataset, groupes, tokenizer, masque de loss et benchmark.
3. Sauvegarde la baseline domaine et générale.
4. Définis les réglages LoRA et FullFT avec un protocole de tuning annoncé.
5. Lance des smoke tests et une vraie reprise de checkpoint dans chaque branche.
6. Fixe le budget principal : cibles, tokens, temps ou compute selon la question ; rapporte les autres.
7. Choisis les checkpoints sur validation selon les mêmes contraintes de résultat.
8. Évalue le test réservé, les régressions, la mémoire, le temps et le stockage.

Le budget de la comparaison inclut les recherches d’hyperparamètres si tu présentes des recettes optimisées, pas seulement le meilleur run conservé.

## 6. Tableau de livraison

| Dimension | À rapporter pour chaque branche |
| --- | --- |
| Qualité | Domaine, catégories générales, effectifs et incertitude utile |
| Updates | Cibles exposées, batch logique, nombre de steps, scheduler |
| Paramètres | Total, entraînable et liste des modules |
| Ressources | Pic par appareil/rank, durée, débit et matériel |
| Artefacts | Adapter/base requis ou poids complets, tokenizer, config |
| Reprise | États conservés et procédure testée |

La taille d’un adapter ne doit pas être comparée au dossier complet d’un modèle sans indiquer que la base reste nécessaire à son usage.

## 7. Interpréter un résultat

Un gain d’un point domaine peut être utile ou négligeable selon variance, catégorie, coût et exigences. Une meilleure loss train ne remplace ni validation ni benchmark de régression.

Un FullFT qui ne tient pas au budget prévu donne une limite de faisabilité. Ne présente pas son absence de run comme une mesure de qualité inférieure ; adapte la taille ou rapporte la condition non exécutée.

## 8. Exercices avec corrigés

### Exercice 1 — Nombre de paramètres

Pourquoi LoRA ajoute-t-il 32 poids pour 8→8 et r=2 ?

<details>
<summary>Corrigé</summary>

`2×(8+8)=32`, tandis que la matrice complète contient 64 poids hors biais.

</details>

### Exercice 2 — Cible facile

Le succès de LoRA sur cette correction de rang 1 prouve-t-il son égalité au full FT sur toutes les tâches ?

<details>
<summary>Corrigé</summary>

Non. Nous avons construit un cas que sa capacité peut représenter. Une autre correction, les données ou l’optimisation peuvent changer le résultat.

</details>

### Exercice 3 — Équité

Pourquoi autoriser des LR distincts dans une comparaison de recettes ?

<details>
<summary>Corrigé</summary>

Les méthodes ont des dynamiques différentes. Il faut annoncer un budget de tuning comparable plutôt qu’imposer un réglage inadapté à l’une.

</details>

### Exercice 4 — Conclusion

FullFT gagne deux points mais dépasse un seuil fixé de régression générale. Le meilleur score domaine suffit-il à l’accepter ?

<details>
<summary>Corrigé</summary>

Non, si ce seuil est une exigence du projet. Rapporte le gain et l’échec de contrainte, puis décide ou révise le besoin explicitement.

</details>

## 9. Critères de fin

- [ ] Les branches partent du même état et les budgets sont annoncés.
- [ ] La question ablation/recette optimisée est claire.
- [ ] Les paramètres et coûts complets sont mesurés.
- [ ] Le choix de checkpoint ne dépend pas du test final.
- [ ] Domaine et capacités à préserver sont rapportés ensemble.
- [ ] La recommandation reste conditionnelle aux données et ressources observées.

## Fiche mémo

```text
Comparer le gain marginal à la liberté, au coût et aux régressions supplémentaires.
```

Le [projet 59](59_projet_4_continued_pretraining.md) isole une étape de continued pretraining.
