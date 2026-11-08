# Module 16 — Injecter, entraîner et recharger un adapter LoRA

## 1. Ce que tu vas apprendre

Le module 15 a décrit la correction `sBA`. Tu vas maintenant laisser **PEFT** l’ajouter à une vraie couche PyTorch, vérifier les gradients et recharger l’adapter sans changer la base.

L’objectif est de pouvoir montrer les paramètres A/B, les poids gelés et les fichiers nécessaires, plutôt que de faire confiance à une configuration.

**Prérequis :** modules 7, 8 et 15 ; gradients, optimizer et formes LoRA.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** Python, PyTorch et PEFT. Laboratoire sur CPU, sans téléchargement de modèle.

## 2. Ce que signifie « injecter »

Avant injection, une projection calcule `x @ W.T`. Après injection, le module contient la base et une branche LoRA qui ajoute sa contribution.

Les noms peuvent ressembler à :

```text
...proj.base_layer.weight
...proj.lora_A.default.weight
...proj.lora_B.default.weight
```

`requires_grad` doit être faux pour la base gelée et vrai pour les A/B attendues. Le nom ne suffit pas : vérifie les flags et les gradients après backward.

### Les réglages à lire

| Paramètre PEFT | Rôle |
| --- | --- |
| `r` | Dimension interne de la correction |
| `lora_alpha` | Échelle, classiquement alpha/r |
| `lora_dropout` | Dropout dans la branche LoRA pendant le train |
| `target_modules` | Modules recevant la branche |
| `bias` | Gestion des biais entraînables selon la configuration |
| `modules_to_save` | Autres modules à entraîner/sauvegarder dans certains usages |

`all-linear` est un sélecteur de couches, pas une demande de full FT. Les détails de ciblage viendront au module 19.

## 3. Un test dont on connaît la correction

Notre base contient une matrice 4 × 4. Pour les entrées de la matrice identité, nous construisons une cible égale à la sortie de base plus une correction de rang 1.

LoRA utilise r=2 : il dispose donc d’une capacité suffisante pour cette correction simple. La base a 16 poids, l’adapter `2×(4+4)=16` paramètres.

Dans cette miniature, l’adapter est aussi gros que la base. Nous choisissons ces dimensions pour les voir ; l’économie typique apparaît sur les grandes matrices du module 15.

Ce n’est pas un LLM ni un SFT linguistique : c’est un test complet du cycle PEFT, indépendant des problèmes de tokenizer et de chat template.

## 4. Laboratoire autonome — Vérifier le cycle entier

Le script génère uniquement ses sorties pédagogiques sous `outputs/module16/`. Il sauvegarde aussi l’état **initial** de la base pour que le rechargement soit reproductible.

Avant exécution, prédis le gradient de A au premier step lorsque B est initialement nulle, et demande-toi pourquoi il faut préserver les poids initiaux.

```python
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from peft import LoraConfig, get_peft_model, PeftModel

class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)

torch.manual_seed(42)
base = Base()
initial_state = deepcopy(base.state_dict())
x = torch.eye(4)
with torch.no_grad():
    target = base(x) + x.sum(dim=-1, keepdim=True).expand(-1, 4)

model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0,
                                     target_modules=['proj'], bias='none'))
model.print_trainable_parameters()
base_before = {name: p.detach().clone() for name, p in model.named_parameters()
               if not p.requires_grad}
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=0.05, weight_decay=0.0)
model.train()
first_loss = nn.functional.mse_loss(model(x), target).item()
for step in range(100):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), target)
    loss.backward()
    if step == 0:
        for name, p in model.named_parameters():
            if p.requires_grad:
                print('Premier gradient :', name, p.grad.norm().item())
    optimizer.step()

model.eval()
with torch.no_grad():
    final_output = model(x)
    final_loss = nn.functional.mse_loss(final_output, target).item()
assert all(torch.equal(p, base_before[name]) for name, p in model.named_parameters()
           if name in base_before)
print('Loss avant/apres :', first_loss, final_loss)
root = Path('outputs/module16')
root.mkdir(parents=True, exist_ok=True)
torch.save(initial_state, root / 'base_initiale.pt')
model.save_pretrained(root / 'adapter')
restored_base = Base()
restored_base.load_state_dict(torch.load(root / 'base_initiale.pt', weights_only=True))
restored = PeftModel.from_pretrained(restored_base, root / 'adapter')
restored.eval()
with torch.no_grad():
    print('Rechargement equivalent :', torch.allclose(final_output, restored(x), atol=1e-6))
```

### Lire les résultats

- Le modèle adapté contient **32 paramètres**, dont **16 entraînables**, soit 50 % pour cette miniature.
- Les noms LoRA sont visibles et les premiers gradients A peuvent être nuls tandis que B reçoit un signal.
- La loss finale doit être nettement plus basse que la loss initiale sur cette correction facile.
- L’assertion vérifie que les poids gelés sont identiques avant et après entraînement.
- Le rechargement doit afficher **True** pour l’équivalence des sorties en mode eval.

`deepcopy` et `.clone()` évitent de conserver simplement des références aux objets qui pourraient ensuite changer. `torch.equal` compare exactement les poids gelés ; `allclose` autorise de petits écarts numériques pour les sorties.

Le fichier `base_initiale.pt` n’est pas créé par la sauvegarde d’adapter : nous l’enregistrons explicitement parce que notre base aléatoire n’a pas de checkpoint externe à retrouver.

## 5. Pourquoi un adapter ne suffit pas seul

`save_pretrained` sur un modèle PEFT sauvegarde principalement la configuration et les tenseurs d’adaptation attendus par ce format. Le dossier ne remplace pas automatiquement la base complète.

Pour un LLM réel, conserve :

1. L’identifiant et la révision exacte de la base.
2. L’adapter et sa configuration.
3. Le tokenizer, les éventuels tokens ajoutés et le chat template utilisé.
4. Les modules supplémentaires entraînés, s’il y en a.
5. Les versions logicielles et les réglages nécessaires au chargement.

Recharger A/B sur une autre initialisation de W calcule une autre fonction. Le programme peut réussir son chargement sans reproduire le modèle que tu avais évalué.

`PeftModel.from_pretrained` recharge ici l’adapter pour l’inférence. Si tu veux reprendre son entraînement, vérifie l’option appropriée, notamment `is_trainable=True`, et recrée ou restaure aussi optimizer/scheduler selon l’objectif de reprise.

## 6. Transposer au SFT d’un LLM

Le module 12 a séparé modèle, dataset, tokenizer et trainer. Pour passer à LoRA :

- Charge la base préentraînée et inspecte ses modules avant de choisir les cibles.
- Construis une `LoraConfig` avec `task_type="CAUSAL_LM"` pour le cas causal concerné.
- Injecte avec `get_peft_model`, **ou** laisse `SFTTrainer` le faire avec `peft_config`. Ne fais pas les deux sans raison.
- Réutilise les mêmes vérifications de labels et de validation.
- Compte les paramètres, puis fais un seul backward et inspecte gradients de base et adapters.
- Sauvegarde et recharge sur la même base avant d’évaluer les générations finales.

Dans notre réseau personnalisé, nous n’avons pas déclaré `task_type="CAUSAL_LM"`, car son forward reçoit simplement un tenseur `x`, pas des `input_ids`. Le wrapper doit correspondre à l’interface du modèle.

## 7. Expériences guidées

1. Recharge l’adapter sur une base aléatoire différente, dans une copie de l’expérience. Compare les sorties et explique pourquoi la même A/B ne suffit pas.
2. Change uniquement r de 2 à 1. Recalcule le nombre ajouté puis vérifie si la correction reste apprenable.
3. Mets `lora_dropout=0.2`. Compare deux forwards en mode train puis en mode eval ; pour comparer des checkpoints, utilise eval.
4. Ajoute une couche qui n’est pas ciblée. Vérifie par noms et flags si elle est gelée ; ne suppose pas que tous les modules linéaires sont adaptés.

Recrée la base et les cibles initiales à chaque essai pour ne pas comparer des historiques différents.

## 8. Erreurs à savoir diagnostiquer

| Symptôme | Vérification |
| --- | --- |
| Aucun module ciblé | Noms réels et règle de correspondance |
| Gradient A nul au premier step | Initialisation B nulle ; inspecter B et le step suivant |
| Un gros poids de base change | Flags, paramètres de l’optimizer, biais ou modules supplémentaires |
| Rechargement différent | Base exacte, tokenizer, mode eval et adapter activé |
| Reprise qui n’apprend pas | Adapter rechargé entraînable et optimizer correctement construit ? |

## 9. Exercices avec corrigés

### Exercice 1 — Compter la miniature

Pourquoi obtient-on 32 paramètres au total et 16 entraînables ?

<details>
<summary>Corrigé</summary>

La matrice de base possède 16 poids gelés. A et B possèdent `2×4 + 4×2 = 16` poids entraînables. Le total est 32 ; le ratio vaut 50 % dans cette miniature.

</details>

### Exercice 2 — Sauvegarder

Le dossier adapter est petit. Est-ce une preuve que l’on n’a plus besoin de la base ?

<details>
<summary>Corrigé</summary>

Non. Sa petite taille vient justement du fait qu’il stocke la correction séparée. Il faut la base compatible pour calculer la contribution W.

</details>

### Exercice 3 — Deux injections

Tu appelles `get_peft_model` puis passes encore une nouvelle `peft_config` au trainer. Pourquoi revoir ce montage ?

<details>
<summary>Corrigé</summary>

Il faut savoir qui crée les adapters pour éviter une injection supplémentaire ou une configuration ambiguë. Choisis un seul lieu d’injection et vérifie les modules résultants.

</details>

### Exercice 4 — Vérifier plutôt que supposer

Quelle observation montre que la base n’a pas changé pendant ce laboratoire ?

<details>
<summary>Corrigé</summary>

La comparaison exacte entre chaque paramètre gelé après entraînement et sa copie avant entraînement. Le flag `requires_grad=False` exprime une intention ; comparer les valeurs vérifie le résultat du programme exécuté.

</details>

## 10. Validation — avant de continuer

- [ ] Je repère A/B et les poids de base dans le modèle injecté.
- [ ] Je retrouve les comptes théoriques et mesurés.
- [ ] Je peux expliquer les premiers gradients.
- [ ] Je montre une diminution de loss avec base inchangée.
- [ ] Je sauvegarde puis recharge sur la même base.
- [ ] Je distingue le réseau personnalisé d’un causal LM et leur wrapper.

## Fiche mémo

```text
Inspecter → configurer → injecter une fois → compter → backward de contrôle.
Entraîner → sauvegarder adapter + identité de la base → recharger → vérifier.
```

Le [module 17](17_mesurer_les_parametres_entrainables.md) transforme ces vérifications en audit de paramètres et d’états d’optimizer.

## Référence

[PEFT : configuration LoRA](https://huggingface.co/docs/peft/en/package_reference/lora) — paramètres, initialisations et chargement des adapters.
