# Module 49 — Répartir les états avec FSDP

## 1. Ce que tu vas apprendre

DDP réplique les états du modèle. **FSDP**, *Fully Sharded Data Parallel*, répartit certains de ces états entre ranks et matérialise temporairement ce qui est nécessaire au calcul.

Tu vas suivre shards, all-gather et reduce-scatter, puis lire un pilote FSDP2 distinct de la simulation CPU.

**Prérequis :** modules 5, 23, 24 et 48.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch sur CPU pour le laboratoire A. Le B cible deux processus CUDA compatibles avec FSDP2 ; sans ce matériel, son exécution reste à valider.

## 2. Qu’est-ce qui est réparti ?

Dans une stratégie complètement shardée, paramètres, gradients et états d’optimizer peuvent être distribués. Le modèle logique reste le même ; chaque rank n’en conserve pas toujours une réplique complète au repos.

| Opération | Rôle simplifié |
| --- | --- |
| Sharding | Répartir les morceaux d’un état |
| All-gather | Réunir les morceaux nécessaires pour calculer un module |
| Reduce-scatter | Réduire des contributions puis distribuer les morceaux du résultat |
| Reshard | Revenir à une représentation répartie selon le planning |

Le détail de conservation après forward/backward dépend de la stratégie et de l’API. L’all-gather peut créer un pic bien supérieur au seul stockage des shards.

## 3. Un budget au repos n’est pas un pic

8 GiB de poids répartis idéalement sur quatre ranks donnent 2 GiB de cette composante par rank. Il faut encore prévoir paramètres temporairement réunis, gradients, optimizer, activations, buffers et communications.

FSDP ne garantit donc ni une division exacte de toute la mémoire par quatre ni la possibilité de traiter un bloc trop grand pour ses matérialisations transitoires.

Le choix des frontières de wrapping influence mémoire et communications : un seul énorme groupe et une multitude de minuscules groupes n’ont pas les mêmes compromis.

## 4. Laboratoire A — Simuler les morceaux et leur réduction

Ce bloc est autonome. Il n’effectue aucune communication distribuée ; il montre seulement les valeurs que les collectives doivent produire.

```python
import torch

full = torch.arange(8, dtype=torch.float64)
shards = list(full.chunk(4))
reconstructed = torch.cat(shards)
print('Shards au repos :', [shard.tolist() for shard in shards])
print('All-gather simule correct :', torch.equal(full, reconstructed))

local_gradients = [torch.arange(8, dtype=torch.float64) + rank for rank in range(4)]
mean_gradient = torch.stack(local_gradients).mean(dim=0)
reduced_shards = list(mean_gradient.chunk(4))
print('Gradient moyen :', mean_gradient.tolist())
print('Reduce-scatter moyenne simule :', [shard.tolist() for shard in reduced_shards])
print('Poids ideaux au repos, GiB/rank :', 8 / 4)
```

Les shards initiaux sont `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`. Le gradient moyen commence à 1.5 et finit à 8.5. La concaténation de ses shards doit restituer cette moyenne.

La simulation affiche une **moyenne**. Les collectives de bas niveau peuvent sommer ; la division et sa place dépendent du mécanisme utilisé. Annonce la convention au lieu de supposer que « reduce » signifie toujours moyenne.

## 5. Laboratoire B — Structure d’un pilote FSDP2 CUDA

Enregistre le bloc dans `fsdp2_demo.py`, puis lance sur une machine adaptée :

```bash
torchrun --standalone --nproc_per_node=2 fsdp2_demo.py
```

Ce code cible l’API composable FSDP2 présente dans l’environnement PyTorch 2.6 du cours. Le lancement simple ou sans CUDA affiche une limitation ; il ne constitue pas une validation de sharding.

```python
import os
import torch
from torch import nn
import torch.distributed as dist


def main():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world < 2 or not torch.cuda.is_available():
        print('Pilote FSDP2 non execute : deux processus CUDA compatibles sont requis.')
        return
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(42)
        mesh = init_device_mesh('cuda', (world,))
        model = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)).to(local_rank)
        fully_shard(model[0], mesh=mesh)
        fully_shard(model[2], mesh=mesh)
        fully_shard(model, mesh=mesh)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        torch.manual_seed(100 + dist.get_rank())
        x = torch.randn(8, 64, device=local_rank)
        y = x.sum(dim=-1, keepdim=True)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        print('rank=', dist.get_rank(), 'loss locale=', loss.item(), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

L’optimizer est créé **après** la transformation des paramètres. Le wrapping se fait ici depuis les sous-modules vers le module parent. Le petit réseau sert à tester le mécanisme, pas à démontrer un gain mémoire mesurable d’un gros LLM.

## 6. FSDP1, FSDP2 et ZeRO

Les générations d’API ne se configurent pas indistinctement. Une ancienne classe `FullyShardedDataParallel`, une config Accelerate FSDP1 et la fonction composable `fully_shard` n’ont pas toutes les mêmes options et formats d’état.

ZeRO-3 partage l’idée de répartir paramètres, gradients et optimizer ; les détails d’orchestration, d’offload et de checkpoint diffèrent selon les outils.

Pour Accelerate, génère une configuration avec la version utilisée et vérifie la génération FSDP ciblée. Copier un YAML ancien contenant quelques noms plausibles ne suffit pas à définir une expérience compatible.

## 7. Sauvegarder et restaurer des états répartis

Un checkpoint shardé n’est pas « incomplet » parce qu’il n’est pas un fichier monolithique. Il requiert un protocole de sauvegarde/chargement correspondant aux états distribués, souvent avec participation collective.

Pour une exportation standard, une consolidation ou une conversion peut être nécessaire. Vérifie le coût mémoire de cette étape : réunir tous les poids sur un rank peut réintroduire un pic important.

Teste restauration et éventuelle conversion sur le petit pilote avant un gros entraînement. Le module 51 expose les états logiques à conserver ; le format distribué doit en plus être traité selon le framework.

## 8. Expériences guidées

1. Sur CPU, change le nombre de shards et vérifie les tailles, y compris pour une longueur non divisible.
2. Sur matériel adapté, compare DDP et FSDP à modèle, batch, contexte et précision fixés. Mesure le pic de **chaque rank**.
3. Compare deux frontières de wrapping et rapporte mémoire, débit et communication.
4. Sauvegarde un petit état shardé puis recharge-le avec la procédure de la version utilisée ; ne te contente pas de l’existence des fichiers.

## 9. Exercices avec corrigés

### Exercice 1 — All-gather

Quatre ranks gardent chacun deux valeurs d’un vecteur de huit. Que reconstruit l’all-gather simplifié ?

<details>
<summary>Corrigé</summary>

Les huit valeurs dans l’ordre défini par le partitionnement, là où elles sont nécessaires au calcul. Cette matérialisation demande une mémoire temporaire.

</details>

### Exercice 2 — Différence avec DDP

Quel stockage DDP conserve-t-il que le full sharding cherche à répartir ?

<details>
<summary>Corrigé</summary>

DDP garde une réplique complète des paramètres et habituellement des états locaux associés. Le full sharding distribue ces composantes selon sa stratégie.

</details>

### Exercice 3 — Pic mémoire

Pourquoi M/N n’est-il pas le budget complet par rank ?

<details>
<summary>Corrigé</summary>

Cette division décrit une composante shardée au repos. Activations, all-gathers, temporaires, gradients et autres états contribuent au pic.

</details>

### Exercice 4 — Checkpoint

Pourquoi tester une conversion vers un modèle monolithique avant un long run ?

<details>
<summary>Corrigé</summary>

Le format et le pic mémoire de consolidation peuvent poser des problèmes différents du training. Un ensemble de shards créé sans erreur ne prouve pas une exportation restaurable dans l’usage visé.

</details>

## 10. Validation — avant de continuer

- [ ] Je suis les valeurs du laboratoire simulé sans le confondre avec une collective réelle.
- [ ] Je distingue réplication, all-gather, reduce-scatter et reshard.
- [ ] Je sais quelle API/génération mon pilote cible.
- [ ] Je distingue stockage au repos et pic par rank.
- [ ] Je documente restauration et conversion du checkpoint.

## Fiche mémo

```text
Moins de duplication permanente, davantage de matérialisations et communications.
La version d’API et les frontières de wrapping font partie de la configuration.
```

Le [module 50](50_budget_memoire_du_training.md) rassemble ces coûts dans un budget mesuré.

## Référence

[PyTorch : tutoriel FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).
