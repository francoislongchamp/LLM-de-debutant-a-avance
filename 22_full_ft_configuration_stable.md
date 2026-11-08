# Module 22 — Construire une configuration de training surveillée

## 1. Ce que tu vas apprendre

Une configuration qui ne plante pas peut tout de même modifier trop fortement un modèle. La stabilité se juge avec des données valides, des mesures finies et des résultats utiles sur validation.

Tu vas suivre un learning rate variable, observer le clipping et relier précision, optimizer et ordre des opérations.

**Prérequis :** modules 7, 8, 13 et 21.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch et Transformers sur CPU pour le laboratoire ; une vraie précision mixte dépend du matériel et du backend.

## 2. Les réglages et leur rôle

| Réglage | Rôle | Limite |
| --- | --- | --- |
| Learning rate | Échelle des mises à jour | Aucune valeur universelle |
| Warmup | Montée initiale du LR | Ne corrige pas des labels faux |
| Scheduler | Évolution du LR au fil des updates | Doit suivre le bon compteur |
| Weight decay | Réduction de certains poids selon l’optimizer | Ne remplace pas la qualité des données |
| Clipping | Limite de norme des gradients | Ne répare pas un gradient NaN |
| Précision | Formats des calculs et stockages | Compatibilité matérielle et stabilité à vérifier |

Le warmup est une partie d’un scheduler. Le nombre de ses étapes se compte habituellement en **optimizer steps**, pas en forwards ou tokens.

Pour 1 000 updates et un warmup de 3 %, cela donne 30 updates. Pour une expérience de 5 updates, un pourcentage peut donner une phase très courte : regarde le compte réel.

## 3. Clipping et weight decay ne font pas la même chose

Le clipping par norme globale remplace approximativement g par `g × C/||g||` si la norme dépasse C. Il limite cette norme sans choisir automatiquement un nouveau LR.

AdamW applique un weight decay découplé des moments du gradient. Le choix des paramètres auxquels l’appliquer compte : des configurations excluent par exemple certains biais et coefficients de normalisation. Notre miniature l’applique simplement au paramètre fourni.

Un gradient de norme 100 n’est pas universellement « mauvais » : son échelle dépend du modèle, de la loss et de l’agrégation. Les changements brutaux et leur effet sur l’update sont souvent plus instructifs qu’un seuil isolé.

## 4. Laboratoire autonome — Voir le LR utilisé à chaque update

Le script optimise un scalaire. Le scheduler démarre ici à zéro, monte en deux steps puis décroît linéairement sur un total de six updates.

```python
import torch
from transformers import get_linear_schedule_with_warmup

parameter = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=2, num_training_steps=6,
)
records = []
for step in range(1, 7):
    optimizer.zero_grad(set_to_none=True)
    loss = (parameter - 3.0) ** 2
    if not torch.isfinite(loss).item():
        raise RuntimeError('Loss non finie : diagnostiquer avant de continuer.')
    loss.backward()
    if not torch.isfinite(parameter.grad).all().item():
        raise RuntimeError('Gradient non fini.')
    norm_before = torch.nn.utils.clip_grad_norm_([parameter], max_norm=1.0)
    used_lr = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    records.append((step, used_lr, norm_before.item(), parameter.item()))

for step, lr, norm, value in records:
    print('step=', step, 'LR utilise=', round(lr, 4),
          'norme avant clipping=', round(norm, 4), 'parametre=', round(value, 4))
```

Les LR utilisés doivent être **0, 0.05, 0.1, 0.075, 0.05, 0.025**. Le LR réglé après le dernier appel du scheduler vaut zéro.

La première update avec LR nul ne déplace pas le poids, même si l’optimizer peut mettre à jour son état. C’est la convention de ce scheduler, pas une règle à généraliser à tous les warmups.

`clip_grad_norm_` renvoie ici la norme **avant** clipping. Une norme affichée supérieure à 1 ne prouve donc pas que le clipping a échoué. Pour connaître la norme après, mesure de nouveau le gradient.

## 5. Où intervient la précision ?

FP16 et BF16 occupent 16 bits par valeur, avec des plages et précisions différentes. BF16 a une plage d’exposants plus large que FP16, mais moins de précision de mantisse. Cela ne garantit pas qu’un calcul sera toujours stable.

Une stratégie de précision mixte peut combiner poids, activations et états dans différents formats. FP16 peut nécessiter un mécanisme de loss scaling ; le framework doit gérer correctement scaling, dénormalisation, clipping et steps sautés en cas de débordement.

Le laboratoire utilise FP32 pour isoler la mécanique. Pour un LLM, choisis une précision prise en charge par le matériel et les kernels, puis vérifie loss, gradients et mémoire réels.

## 6. Ordre des opérations à conserver

```text
zero_grad → forward → loss → backward
→ dénormalisation si précision mixte avec scaling
→ contrôle/clipping → optimizer.step → scheduler.step
```

Un trainer réalise une grande partie de cette orchestration. En accumulation, scheduler et clipping se placent normalement au niveau de la **mise à jour logique**, pas après chaque micro-batch.

Si une update est sautée pour un gradient non fini, il faut aussi que l’avancement du scheduler reste cohérent avec l’implémentation choisie.

## 7. Procédure pour un premier full FT de LLM

1. Reproduis un forward et une validation de base avant training.
2. Inspecte les labels, le nombre de cibles et la troncature.
3. Vérifie le nombre de paramètres optimisés et le budget du premier step complet.
4. Commence avec une courte expérience et un LR justifié pour cette base, cette méthode et ce batch.
5. Logge loss, LR utilisé, norme pré-clipping, tokens traités et pic mémoire.
6. Évalue périodiquement, sauvegarde les états utiles et compare les régressions au benchmark général.

Le warmup de 3 % ou le clipping à 1 sont des exemples fréquents, pas une garantie de stabilité. Une expérience trop courte ne permet pas de juger la qualité finale.

## 8. Diagnostiquer une valeur non finie

| Étape de recherche | Ce qu’elle peut révéler |
| --- | --- |
| Entrées et labels | Valeurs invalides, indices hors vocabulaire |
| Nombre de cibles actives | Moyenne sur aucune cible |
| Forward en FP32 sur un petit cas | Sensibilité numérique ou opération incorrecte |
| Normes et LR avant l’échec | Update trop grande, échelle anormale |
| Premier step sur un batch fixe | Problème de pipeline reproductible |

N’essaie pas de rendre silencieusement une loss NaN égale à zéro : cela masque la cause. Réduire le LR est une hypothèse à tester, pas une explication universelle.

## 9. Expériences guidées

1. Mets le warmup à zéro et trace les LR utilisés. Compare l’amorce de la loss, sans attendre une meilleure qualité par principe.
2. Remplace le scheduler linéaire par un cosine avec les mêmes comptes. Dessine la courbe avant de l’exécuter.
3. Mesure la norme avant et après clipping sur un step. Vérifie la différence entre valeur loggée et gradient utilisé.
4. Dans la miniature, compare LR 0.01 et 0.1 depuis le même départ. Pour un LLM, n’extrapole pas ces valeurs ; refais une expérience courte adaptée.

## 10. Exercices avec corrigés

### Exercice 1 — Warmup

2 000 updates avec un warmup de 5 % : combien d’updates de montée ?

<details>
<summary>Corrigé</summary>

**100 updates**. Si tu utilises une accumulation, il ne s’agit pas de 100 micro-batches sauf si une update correspond à un seul micro-batch.

</details>

### Exercice 2 — Clipping

La norme avant clipping vaut 10 et C=2. Quel facteur approximatif est appliqué au gradient ?

<details>
<summary>Corrigé</summary>

Environ **0.2**. La direction du gradient est conservée et sa norme ramenée vers 2, avec les précautions numériques de l’implémentation.

</details>

### Exercice 3 — Compteur erroné

Tu avances le scheduler à chaque micro-batch avec une accumulation de 8. Quel risque ?

<details>
<summary>Corrigé</summary>

Il peut parcourir sa courbe huit fois plus vite en nombre d’updates que prévu. Son compteur doit correspondre à la définition du planning.

</details>

### Exercice 4 — Faux remède

Une loss NaN vient de labels tous ignorés. Le warmup corrige-t-il cette cause ?

<details>
<summary>Corrigé</summary>

Non. Il faut corriger la préparation ou le filtrage des exemples pour qu’il existe des cibles actives. Changer la trajectoire du LR ne crée pas ces cibles.

</details>

## 11. Validation — avant de continuer

- [ ] Je distingue warmup, scheduler, LR, weight decay et clipping.
- [ ] Je retrouve les six LR réellement utilisés.
- [ ] Je sais interpréter une norme renvoyée avant clipping.
- [ ] Je place les opérations au bon niveau en accumulation.
- [ ] Je vérifie précision, cibles et gradients avant de diagnostiquer une divergence.
- [ ] Je ne confonds pas stabilité numérique et amélioration du benchmark.

## Fiche mémo

```text
Mesurer le LR utilisé, le gradient utilisé et les cibles réellement actives.
Planifier en optimizer steps.
Données correctes + calcul fini + updates contrôlées + validation régulière.
```

Le [module 23](23_gradient_accumulation_et_batch_effectif.md) détaille ce qui contribue à une update logique.

## Références

[Transformers : schedulers](https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules), [PyTorch : clipping](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html).
