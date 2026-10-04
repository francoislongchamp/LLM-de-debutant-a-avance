# Annexe — Diagnostic rapide d'un entraînement

Cette annexe ne remplace pas l'analyse. Elle donne des **hypothèses de départ**.

## 1. Out Of Memory (OOM)

Premiers leviers à examiner :

```text
réduire batch/device
réduire sequence length
activer gradient checkpointing
utiliser BF16/FP16 si approprié
utiliser LoRA/QLoRA
sharder avec FSDP/ZeRO en multi-GPU
```

Ne confonds pas mémoire des poids et mémoire totale : activations, gradients, optimizer states et buffers peuvent dominer.

## 2. Loss = NaN

Hypothèses fréquentes :

- learning rate excessif ;
- overflow/underflow numérique ;
- données ou labels invalides ;
- opération instable ;
- gradient norm qui explose ;
- mélange de précision mal configuré.

À inspecter :

```text
premier step où NaN apparaît
grad_norm
learning_rate
batch fautif
logits / activations non finis
labels
```

## 3. Train loss baisse, validation loss monte

Hypothèse principale : surapprentissage ou distribution train/validation différente.

Questions :

- Le split est-il propre ?
- Le train contient-il des duplicats ?
- As-tu trop d'epochs ?
- Le dataset est-il trop petit ?
- Le learning rate est-il trop agressif ?

## 4. Loss ne baisse presque pas

Vérifie :

- paramètres réellement `requires_grad=True` ;
- optimizer connecté aux bons paramètres ;
- labels non tous masqués ;
- learning rate non nul/trop faible ;
- backward réellement exécuté ;
- `optimizer.step()` exécuté au bon moment ;
- données apprenables.

Le test « overfit one batch » est particulièrement utile.

## 5. SFT répond moins bien qu'avant

Inspecte :

- chat template ;
- masking de loss ;
- taux d'apprentissage ;
- qualité des réponses de référence ;
- répétitions ;
- équilibre des tâches ;
- catastrophic forgetting ;
- évaluation faite avec le bon template.

## 6. Reward monte mais qualité réelle baisse

C'est un signal classique de reward hacking ou de métrique incomplète.

Inspecte les exemples qui obtiennent **les rewards les plus élevés**, pas seulement la moyenne.

## 7. GPU peu utilisé

Possibilités :

- dataloader trop lent ;
- CPU/tokenisation bloque ;
- transferts host→device ;
- batch trop petit ;
- synchronisations fréquentes ;
- communication distribuée ;
- génération/autoregression ;
- modèle partiellement offloadé sur CPU.

Mesure avant d'optimiser.

## 8. Checklist minimale avant un long run

```text
[ ] un batch charge correctement
[ ] forward donne une loss finie
[ ] backward donne des gradients finis
[ ] overfit one batch fonctionne
[ ] train/validation sont séparés
[ ] checkpoint save fonctionne
[ ] checkpoint resume fonctionne
[ ] logs contiennent loss/LR/grad norm
[ ] benchmark baseline est sauvegardé
[ ] config et seed sont enregistrés
```
