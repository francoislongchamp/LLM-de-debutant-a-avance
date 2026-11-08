# Annexe — Diagnostiquer un entraînement à partir de preuves

Cette annexe donne des hypothèses et des tests courts. Un symptôme n’identifie pas automatiquement sa cause. Conserve le premier état anormal, la configuration et les données nécessaires à sa reproduction.

## 1. Ordre de vérification

```text
Contrat et données → tokenisation/labels → forward → loss
→ backward/gradients → optimizer/scheduler → validation → rechargement.
```

Commence par une petite reproduction et le dernier état stable. Changer plusieurs facteurs à la fois peut résoudre le symptôme sans expliquer ce qui l’a causé.

## 2. Loss non finie ou sans amélioration

| Symptôme | Vérifier d’abord | Petit test | Renvoi |
| --- | --- | --- | --- |
| Loss NaN dès le départ | Cibles actives après shift, indices, données et calcul | Un batch, compte des `labels[:,1:] != -100`, forward puis backward finis | [6](06_loss_et_prediction_du_prochain_token.md), [42](42_ecrire_une_boucle_de_pre_entrainement.md) |
| NaN après une update | LR utilisé, gradients, dtype, états optimizer | Rejouer depuis le checkpoint précédent et localiser la première valeur non finie | [22](22_full_ft_configuration_stable.md), [41](41_initialisation_aleatoire_et_stabilite.md) |
| Loss plate | Paramètres suivis, graphe, groupes optimizer, LR effectif | Faire apprendre un batch fixe et comparer les poids avant/après | [7](07_gradients_et_autograd.md), [8](08_premiere_boucle_dentrainement_pytorch.md), [17](17_mesurer_les_parametres_entrainables.md) |
| Forte perte seulement sur certains batches | Longueur, source, masque, nombre de cibles | Retrouver IDs et distributions du batch, conserver brut et moyenne pondérée | [43](43_lire_les_courbes_de_loss.md), [47](47_dataset_mixture_et_sampling.md) |

Une loss nulle obtenue en supprimant tous les labels ne démontre pas un modèle parfait. Une division sur zéro cible n’a pas de moyenne utile. Ne remplace pas silencieusement NaN par zéro pour continuer.

## 3. Gradients et LoRA

| Observation | Hypothèses à tester |
| --- | --- |
| `.grad=None` | Pas de backward, paramètre hors graphe, suivi désactivé ou gradient déjà effacé |
| Gradient nul | Dérivée locale nulle, masques, annulations, saturation ou montage bloqué |
| Paramètre entraînable mais inchangé | Absent de l’optimizer, LR nul, gradient nul ou update trop petite à la précision courante |
| Premier gradient A LoRA nul | B initialement nulle ; vérifier B et les steps suivants avant de conclure à un bug |
| Gros poids de base modifié | Flags, groupes, biais/modules supplémentaires ou mauvaise comparaison des snapshots |

Copie les valeurs avec `.detach().clone()` pour un avant/après. Garder une référence au même tenseur ne conserve pas son ancienne valeur.

Voir [15](15_peft_et_intuition_de_lora.md) à [19](19_choisir_les_target_modules.md).

## 4. OOM et mémoire

1. Identifie la phase : chargement, forward, backward, premier optimizer step, génération, validation ou sauvegarde/consolidation.
2. Sépare poids, gradients, états, activations et temporaires.
3. Mesure le pic du batch et de la longueur réellement visés, sur chaque rank si distribué.
4. Choisis le levier correspondant : micro-batch, contexte, checkpointing, PEFT, quantification, taille ou sharding.
5. Recontrôle débit et qualité après la modification.

`reserved` élevé n’est pas automatiquement une fuite. `empty_cache()` ne retire pas les tenseurs encore référencés. DDP ne divise pas une réplique du modèle par le nombre de GPU.

Renvois : [5](05_parametres_et_memoire_des_poids.md), [20](20_qlora_et_quantification_4_bits.md), [24](24_gradient_checkpointing_et_memoire.md), [49](49_fsdp_et_sharding.md), [50](50_budget_memoire_du_training.md).

## 5. Train progresse, usage se dégrade

| Observation | Enquête utile |
| --- | --- |
| Train baisse, validation remonte durablement | Objectifs comparables, split, répétitions, suradaptation et sélection de checkpoint |
| Bonne validation domaine mais général en baisse | Benchmark de régression, oubli, mixture et amplitude d’adaptation |
| Génération incohérente après rechargement | Même base, tokenizer, template, adapter activé et mode eval |
| Comparaison avant/après spectaculaire | Même prompts, limites, décodage, scoring et dénominateurs ? |
| PPL meilleure avec autre tokenizer | Unité changée ; reprendre une comparaison de protocole compatible |

Voir [11](11_train_validation_test_et_contamination.md), [13](13_baseline_et_evaluation_avant_apres.md), [14](14_surapprentissage_et_generalisation.md), [44](44_perplexite_et_ses_limites.md), [55](55_catastrophic_forgetting.md).

## 6. Accumulation et compteurs

Vérifie séparément micro-batches, fenêtres logiques, updates et cibles exposées. Inspecte le dernier groupe incomplet.

Une moyenne de moyennes locales n’est pas toujours la moyenne par cible. Le scheduler ne doit pas avancer comme si chaque micro-batch était une update si son planning a été défini en optimizer steps.

Si le trainer normalise déjà l’accumulation, une division supplémentaire peut réduire artificiellement les gradients. Ne change pas l’échelle sans lire le contrat de la version utilisée.

Voir [23](23_gradient_accumulation_et_batch_effectif.md) et [42](42_ecrire_une_boucle_de_pre_entrainement.md).

## 7. Rewards et préférences

| Symptôme | Vérification |
| --- | --- |
| Reward toujours zéro | Sorties brutes, parser, oracle, troncature et difficulté initiale |
| Reward toujours maximale | Tâche déjà résolue ou correcteur trop permissif |
| Avantages groupés tous nuls | Rewards identiques ; regarder réussite absolue et contraste séparément |
| Reward monte, audit baisse | Proxy exploité, raccourcis et changements de distribution |
| Marge DPO nulle au départ | Policy identique à référence ; les égalités peuvent être normales |
| Deux logs de marge différents | Marge brute/relative, facteur beta, longueur, masque et moyenne |

Ne confonds pas référence DPO, ancienne policy de collecte et reward model. Un critic estime un return, pas une note de vérité.

Renvois : [27](27_donnees_de_preference_et_dpo.md) à [36](36_kl_divergence_et_controle_de_derive.md).

## 8. Distribué : blocage, répétitions et états

- **Blocage collectif :** vérifier que les ranks exécutent les mêmes collectives, étapes et frontières ; repérer une exception sur un seul rank.
- **Données répétées :** vérifier sampler, seed/rank, `set_epoch` et traitement des fins/duplications de padding.
- **Métrique locale présentée globale :** réduire les sommes et comptes réels avant d’écrire.
- **FSDP incompatible :** génération d’API, wrapping, optimizer construit au bon moment et format de checkpoint.
- **Pic malgré sharding :** matérialisations all-gather, activations et plus gros groupe de paramètres.

Le pilote [DDP 48](48_data_parallel_et_ddp.md) se teste avec deux processus ; un simple lancement qui affiche la commande ne vérifie pas les communications. Le pilote [FSDP 49](49_fsdp_et_sharding.md) cible CUDA et reste matériellement conditionnel.

## 9. Reprise et logs

Si la reprise diverge immédiatement, compare poids, moments optimizer, LR/scheduler, RNG restaurés **après construction**, prochain batch et position dans les données.

Une sauvegarde d’inférence n’est pas forcément un état de reprise complet. Un checkpoint de milieu d’accumulation demande aussi un contrat pour les gradients partiels.

Si TensorBoard ne montre rien, vérifie dossier, tags, steps, fermeture/flush et lecture du fichier d’événements. Un logger ne sauvegarde pas les poids.

Voir [51](51_checkpoints_et_reprise.md) et [52](52_logging_et_tensorboard.md).

## 10. Débit faible

Mesure les temps de lecture/tokenisation, transfert, forward, backward, synchronisation, génération et logging. Une faible utilisation GPU peut venir de données lentes, petits batches, offload, synchronisations ou communications.

Définis la région chronométrée et synchronise l’accélérateur lorsque nécessaire. Le temps total d’une expérience inclut aussi évaluation et sauvegardes ; le débit du seul step ne le représente pas.

## 11. Fiche d’incident

```text
Version du code et de l’environnement :
Configuration, base/tokenizer/données :
Dernier état stable et premier événement suspect :
Batch/source/compteurs impliqués :
Loss, LR utilisé, gradients, cibles, mémoire :
Hypothèses et reproduction minimale :
Un changement testé :
Résultat et conclusion limitée :
```

Retour à l’[index](INDEX.md), au [glossaire](GLOSSAIRE.md) ou à l’[annexe de choix](ANNEXE_CHOISIR_LA_BONNE_TECHNIQUE.md).
