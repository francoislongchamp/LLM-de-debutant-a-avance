# Module 48 — Répliquer le modèle et synchroniser les gradients avec DDP

## 1. Ce que tu vas apprendre

**Distributed Data Parallel**, ou DDP, donne une copie du modèle à chaque processus et répartit les données. Les gradients synchronisés permettent aux copies d’effectuer des updates cohérentes.

Tu vas calculer une moyenne de gradients, lire les ranks et lancer un pilote réel à deux processus CPU si l’environnement autorise les communications locales.

**Prérequis :** modules 8, 23 et 47.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch avec backend Gloo pour le pilote CPU ; aucun GPU requis. Le lancement multiprocessus demande un fichier Python et des communications locales autorisées.

## 2. Définir les acteurs du calcul distribué

| Terme | Sens |
| --- | --- |
| Processus/rank | Travailleur identifié globalement |
| World size | Nombre total de processus du groupe |
| Local rank | Indice local à une machine, utile notamment pour son GPU |
| Réplique | Copie du modèle dans chaque travailleur |
| All-reduce | Réduction d’une quantité entre ranks puis partage du résultat |
| Sampler distribué | Affectation des indices de données aux ranks |

DDP synchronise les gradients, mais ne découpe pas automatiquement ton dataset à ta place. Il faut un sampler ou une autre stratégie cohérente.

## 3. Refaire une moyenne à la main

Avec deux gradients locaux moyens g0=2 et g1=6, DDP standard produit une moyenne de 4 sous les conventions habituelles. Si chaque rank a le même nombre de cibles, cette moyenne correspond à celle du batch global.

Avec 1 cible sur le premier et 9 sur le second, la bonne moyenne par cible serait `(1×2+9×6)/10=5.6`. Moyenne de gradients locaux et moyenne globale par token ne coïncident plus automatiquement.

Conserve les effectifs réels, notamment pour des réponses de longueurs variables et les fins de dataset. Le pilote utilise des effectifs égaux pour isoler la synchronisation.

## 4. Mémoire et batch effectif

Chaque réplique doit tenir sur son appareil, avec ses états selon la configuration. DDP ne divise pas les poids d’un gros modèle par le nombre de GPU.

Pour micro-batch b, D ranks et A accumulations régulières, le batch nominal vaut `b×D×A`. Ajouter des ranks sans ajuster b/A change donc l’expérience autant que son débit.

Les communications, transferts et données empêchent de promettre une accélération linéaire. Mesure débit global et efficacité, pas seulement utilisation apparente d’un appareil.

## 5. Laboratoire — DDP CPU et référence en un processus

Enregistre ce bloc dans `ddp_cpu.py`. `python ddp_cpu.py` affiche seulement le mode de lancement. Pour exécuter réellement deux ranks :

```bash
torchrun --standalone --nproc_per_node=2 ddp_cpu.py
```

Le script compare les updates DDP à celles d’une régression sur le même batch global dans chaque processus. Il utilise précisément deux ranks et quatre exemples pour éviter padding de sampler et effectifs inégaux.

```python
import os
from copy import deepcopy
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def main():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if world == 1:
        print('Lancer : torchrun --standalone --nproc_per_node=2 ddp_cpu.py')
        return
    if world != 2:
        raise ValueError('Ce pilote utilise exactement deux ranks')
    dist.init_process_group(backend='gloo')
    try:
        rank = dist.get_rank()
        torch.set_num_threads(1)
        torch.manual_seed(42)
        local_model = nn.Linear(1, 1).double()
        reference = deepcopy(local_model)
        model = DistributedDataParallel(local_model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        x = torch.arange(1, 5, dtype=torch.float64).reshape(-1, 1)
        y = 2 * x + 1
        dataset = TensorDataset(x, y)
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                     shuffle=True, seed=42, drop_last=False)
        loader = DataLoader(dataset, batch_size=2, sampler=sampler)
        for epoch in range(2):
            sampler.set_epoch(epoch)
            print('rank=', rank, 'epoch=', epoch, 'indices=', list(sampler), flush=True)
            for local_x, local_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(local_x), local_y)
                loss.backward()  # DDP synchronise les gradients ici.
                optimizer.step()

                reference_optimizer.zero_grad(set_to_none=True)
                nn.functional.mse_loss(reference(x), y).backward()
                reference_optimizer.step()
                assert all(torch.allclose(left, right, atol=1e-12, rtol=1e-12)
                           for left, right in zip(local_model.parameters(), reference.parameters()))
        if rank == 0:
            print('Deux updates DDP equivalentes a la reference globale.', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

Les indices doivent se répartir entre les ranks, et les deux updates doivent être équivalentes à la référence sous la tolérance annoncée. `set_epoch` renouvelle le mélange ; sans lui, un sampler à seed fixe peut répéter le même ordre chaque epoch.

Un lancement en un processus qui affiche la commande n’est **pas** un test réussi de synchronisation. Distingue ce cas du vrai lancement à deux ranks.

## 6. Fin de corpus et métriques globales

Un sampler peut ajouter des indices pour rendre les partitions compatibles, ou supprimer une fin selon ses options. Ces duplications peuvent biaiser une évaluation si tu les comptes sans correction.

Pour une loss globale, réduis les sommes NLL et les nombres de cibles puis divise ; ne moyenne pas aveuglément des moyennes locales de tailles différentes.

Pour des prédictions et benchmarks, rassemble les IDs et sorties puis détecte doublons et manquants. Un seul rank qui écrit les scalaires simplifie certains logs, mais les collectives nécessaires doivent être appelées par tous les ranks concernés.

## 7. Passer au multi-GPU

Un montage courant utilise un processus par GPU, `local_rank` pour choisir l’appareil et NCCL pour les communications GPU adaptées. Les entrées et la réplique doivent être sur le bon appareil.

Le pilote CPU n’est pas un test des performances, kernels ou communications multi-GPU. L’architecture matérielle, le réseau et les versions deviennent des paramètres de l’expérience.

Avec accumulation, `no_sync()` peut éviter certaines synchronisations intermédiaires si utilisé correctement. Le dernier backward de la fenêtre doit encore synchroniser les contributions nécessaires.

## 8. Exercices et expériences

- Calcule le batch effectif de 8 ranks × 2 exemples × 4 accumulations.
- Oublie volontairement le sampler dans une copie et affiche les indices : montre la répétition des mêmes données.
- Augmente le dataset à cinq exemples et inspecte le comportement de fin de sampler au lieu de supposer une partition parfaite.
- Compare débit 1/2 ranks avec même budget global et même mesure, puis explique la différence avec un changement de batch.

## 9. Exercices avec corrigés

### Exercice 1 — Batch global

Quel résultat pour 8×2×4 ?

<details>
<summary>Corrigé</summary>

**64 exemples** par update nominale complète.

</details>

### Exercice 2 — Mémoire

Un modèle ne tient pas comme réplique sur un GPU. Deux GPU en DDP pur résolvent-ils forcément cela ?

<details>
<summary>Corrigé</summary>

Non. Chaque rank garde une réplique complète. Il faut une autre stratégie de mémoire, de taille ou de répartition des états.

</details>

### Exercice 3 — Pondération

Pourquoi 1 et 9 cibles locales rendent-ils la moyenne de deux gradients moyens insuffisante ?

<details>
<summary>Corrigé</summary>

Elle donne le même poids aux deux groupes. Une moyenne globale par cible doit pondérer par leurs effectifs réels.

</details>

### Exercice 4 — Ranks

Rank global et local rank sont-ils toujours identiques ?

<details>
<summary>Corrigé</summary>

Non. Sur plusieurs machines, les indices locaux recommencent sur chaque machine, tandis que le rank global identifie le processus dans le groupe total.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue réplication et partition des données.
- [ ] Je sais calculer batch global et moyenne pondérée.
- [ ] Je distingue lancement simple et vraie exécution à deux processus.
- [ ] Je vérifie indices, fins de sampler et cohérence des poids.
- [ ] Je sais ce que le pilote CPU ne valide pas sur GPU.

## Fiche mémo

```text
Données différentes, copies du modèle cohérentes grâce aux gradients synchronisés.
DDP répartit le travail ; il ne sharde pas les poids d’une réplique.
```

Le [module 49](49_fsdp_et_sharding.md) répartit justement les états du modèle.

## Référence

[PyTorch : DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).
