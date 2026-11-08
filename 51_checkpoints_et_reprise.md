# Module 51 — Tester une vraie reprise, pas seulement un fichier de poids

## 1. Ce que tu vas apprendre

Un checkpoint utile doit se recharger et permettre de continuer selon un contrat défini. Reprendre les poids seuls ne restaure pas automatiquement les moments AdamW, le LR ou l’ordre des données.

Tu vas comparer huit updates continues à quatre updates suivies d’une sauvegarde et d’une reprise de quatre autres, avec dropout et tirage aléatoire des exemples.

**Prérequis :** modules 7, 22, 42 et 50.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch sur CPU. Le test est autonome et utilise un dossier temporaire.

## 2. Définir le contrat de reprise

| État | Pourquoi |
| --- | --- |
| Poids et buffers | Reconstituer la fonction du modèle |
| Optimizer | Moments, compteurs et groupes |
| Scheduler | Position du planning de LR |
| Scaler éventuel | Dynamique de précision mixte |
| Compteurs | Updates, tokens, epoch et position de données |
| RNG | Dropout, sampling et autres tirages |
| Données/sampler | Prochain exemple, shard ou curseur consommé |
| Config/code/tokenizer | Même calcul et même signification des IDs |

Le pilote n’utilise pas de DataLoader, CUDA ou précision mixte. La sélection d’exemples dépend uniquement du RNG Python, et le dropout du RNG PyTorch CPU. Cela rend son contrat complet et testable dans ce cadre limité.

## 3. Sauvegarder à une frontière explicite

Nous sauvegardons après optimizer et scheduler, entre deux updates. Le prochain step remettra les gradients à zéro : il n’est donc pas nécessaire de sauvegarder une accumulation en cours.

Si tu sauvegardes au milieu d’une fenêtre d’accumulation, il faut définir ce qu’il arrive aux gradients partiels et aux micro-batches déjà traités. Attendre une frontière d’update simplifie le contrat.

Une écriture dans un fichier temporaire suivie d’un remplacement dans le même système de fichiers évite de présenter un fichier partiellement écrit comme dernier checkpoint finalisé. Cela ne garantit pas à lui seul la durabilité face à toute panne matérielle.

## 4. Laboratoire autonome — Continu contre interrompu/repris

```python
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import torch
from torch import nn

x = torch.tensor([[1.,0.], [0.,1.], [1.,1.], [2.,1.]])
y = x.sum(dim=-1, keepdim=True)


def create():
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Dropout(0.2), nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler):
    index = random.randrange(len(x))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x[index:index+1]), y[index:index+1])
    loss.backward()
    optimizer.step()
    scheduler.step()
    return index, loss.item()


torch.manual_seed(42)
random.seed(42)
model, optimizer, scheduler = create()
continuous_trace = []
with TemporaryDirectory(prefix='module51-') as directory:
    checkpoint = Path(directory) / 'checkpoint.pt'
    for step in range(1, 9):
        continuous_trace.append(update(model, optimizer, scheduler))
        if step == 4:
            temporary = checkpoint.with_suffix('.tmp')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'step': step,
                        'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate()}, temporary)
            temporary.replace(checkpoint)
    final_state = deepcopy(model.state_dict())
    final_lr = optimizer.param_groups[0]['lr']

    restored_model, restored_optimizer, restored_scheduler = create()
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    restored_model.load_state_dict(state['model'])
    restored_optimizer.load_state_dict(state['optimizer'])
    restored_scheduler.load_state_dict(state['scheduler'])
    # Restaurer les RNG APRES les constructions qui ont pu les consommer.
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    resumed_trace = []
    for step in range(state['step'] + 1, 9):
        resumed_trace.append(update(restored_model, restored_optimizer, restored_scheduler))

    same_weights = all(torch.equal(tensor, restored_model.state_dict()[name])
                       for name, tensor in final_state.items())
    print('Prochains indices et losses identiques :', continuous_trace[4:] == resumed_trace)
    print('Poids finaux identiques :', same_weights)
    print('LR finaux :', final_lr, restored_optimizer.param_groups[0]['lr'])
    print('Step sauvegarde et fin :', state['step'], step)
```

Tu dois obtenir deux comparaisons vraies, un checkpoint au **step 4**, une fin au **step 8** et le même LR final. Le test montre une identité dans cette expérience CPU contrôlée, pas une promesse bit à bit entre tous les matériels et bibliothèques.

## 5. Pourquoi restaurer les RNG après construction ?

Créer les nouvelles couches consomme des nombres aléatoires. Si tu restaures le RNG puis construis le modèle, tu changes le point à partir duquel le prochain dropout tirera ses masques.

De même, un DataLoader avec son propre générateur, des workers ou une source streaming nécessite leurs états ou une procédure de repositionnement. Conserver seulement `seed=42` ne signifie pas reprendre au tirage 10 001 d’un run interrompu.

## 6. Poids seuls : utile, mais autre expérience

Charger les poids puis créer un optimizer neuf peut servir à une nouvelle phase d’adaptation. Ce n’est pas une reprise exacte de l’ancien training.

Un dossier final d’inférence doit contenir les poids nécessaires, configuration et tokenizer. Un adapter PEFT nécessite sa base compatible. Un checkpoint distribué peut nécessiter un chargement collectif et les outils de sa génération FSDP.

Le format du fichier n’indique pas à lui seul quels états sont présents : inspecte le contenu et teste l’usage visé.

## 7. Fréquence et rotation

Sauvegarder toutes les K updates peut faire perdre jusqu’à environ K updates depuis le dernier état valide. Des sauvegardes fréquentes coûtent temps et stockage.

Garde plusieurs générations d’états valides selon ton besoin : dernier, précédent, meilleur sur validation et jalons utiles. Évite de supprimer le seul checkpoint testé avant d’avoir vérifié le nouveau.

Pour une comparaison scientifique, garde aussi les versions de données et résultats d’évaluation liés à chaque état.

## 8. Expériences guidées

1. Oublie seulement l’état optimizer dans une copie. Compare la trace après reprise, même avec les mêmes poids et RNG.
2. Oublie seulement le RNG torch. Le dropout peut changer malgré le même ordre d’exemples.
3. Redémarre le scheduler au step zéro et observe le LR ; explique pourquoi ce n’est plus la même trajectoire.
4. Sur le pilote 42, définis les états supplémentaires du DataLoader avant de prétendre reprendre au milieu d’une epoch.

## 9. Exercices avec corrigés

### Exercice 1 — Seed et état

Pourquoi réappliquer la seed initiale ne suffit-il pas au milieu du run ?

<details>
<summary>Corrigé</summary>

La seed redémarre un flux, alors que la reprise doit retrouver sa position après les tirages déjà consommés. L’état du générateur encode cette position.

</details>

### Exercice 2 — Frontière

Pourquoi une sauvegarde entre updates simplifie-t-elle l’accumulation ?

<details>
<summary>Corrigé</summary>

Il n’y a pas de gradient partiel à restaurer pour une update non terminée. Il faut encore conserver les compteurs et le prochain état des données.

</details>

### Exercice 3 — Fichier créé

Un `torch.save` réussi prouve-t-il une reprise correcte ?

<details>
<summary>Corrigé</summary>

Non. Il faut recharger, continuer et vérifier les états ou sorties attendus. Un fichier peut manquer d’informations indispensables.

</details>

### Exercice 4 — Nouveau training

Est-il toujours incorrect de reprendre seulement les poids ?

<details>
<summary>Corrigé</summary>

Non. C’est un choix possible pour commencer une nouvelle phase, à annoncer comme tel. Il ne faut simplement pas l’appeler une reprise exacte de la dynamique précédente.

</details>

## 10. Validation — avant de continuer

- [ ] Je définis un contrat d’état correspondant au pipeline réel.
- [ ] Je retrouve l’identité de la trace et des poids du pilote.
- [ ] Je restaure les RNG après les constructions concernées.
- [ ] Je distingue poids seuls et continuation exacte.
- [ ] Je teste reprise, rotation et éventuel format distribué.

## Fiche mémo

```text
Un checkpoint est une promesse de restauration à vérifier.
Poids + optimizer + scheduler + RNG + position des données, selon le contrat réel.
```

Le [module 52](52_logging_et_tensorboard.md) relie ces états aux métriques du run.

## Référence

[PyTorch : sauvegarde et rechargement](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
