# Index — les 62 modules et leurs preuves d’apprentissage

Pour un premier passage, avance dans l’ordre en réalisant les exercices. Le résultat attendu est une explication et une vérification ; la vitesse de lecture est secondaire.

## Préparer et retrouver

- [Module 00 — méthode, environnement et exécution](00_METHODE_ET_ENVIRONNEMENT.md)
- [README — portée du cours et parcours](README.md)
- [Glossaire](GLOSSAIRE.md)
- [Annexe — choisir une technique](ANNEXE_CHOISIR_LA_BONNE_TECHNIQUE.md)
- [Annexe — diagnostiquer un entraînement](ANNEXE_DIAGNOSTIC_ENTRAINEMENT.md)
- [Sources et API — versions et vérifications](SOURCES_ET_API.md)

**Lecture de la colonne « mode » :** CPU désigne la miniature proposée, pas le coût d’un projet réel à grande échelle. « Fictif » indique des valeurs construites pour apprendre une analyse. Les modules donnent les prérequis précis et les extensions possibles.

## Modules 1 à 8 — Fondations : suivre le calcul

| Module | Ce que tu dois savoir vérifier | Mode du laboratoire |
| --- | --- | --- |
| [Module 1 — Suivre le trajet d’un texte dans un LLM](01_pipeline_complet_dun_llm.md) | Suivre texte, IDs, logits et sortie. | CPU ; modèle externe selon l’exemple |
| [Module 2 — Comprendre comment le texte devient des tokens](02_tokenisation_en_profondeur.md) | Comparer texte, octets et tokens. | CPU ; modèle externe selon l’exemple |
| [Module 3 — Préparer des séquences et respecter la fenêtre de contexte](03_fenetre_de_contexte_et_sequences.md) | Inspecter padding, troncature et contexte. | CPU |
| [Module 4 — Ouvrir un bloc Transformer et suivre ses calculs](04_anatomie_du_transformer.md) | Suivre les dimensions d’un bloc. | CPU |
| [Module 5 — Compter les paramètres et estimer la mémoire](05_parametres_et_memoire_des_poids.md) | Compter poids uniques et octets. | CPU |
| [Module 6 — Calculer l’erreur de prédiction du prochain token](06_loss_et_prediction_du_prochain_token.md) | Vérifier shift causal et cibles. | CPU |
| [Module 7 — Comprendre les gradients et les observer avec autograd](07_gradients_et_autograd.md) | Observer les dérivées et leur accumulation. | CPU |
| [Module 8 — Comprendre et écrire sa première boucle d’entraînement PyTorch](08_premiere_boucle_dentrainement_pytorch.md) | Relier forward, backward et update. | CPU |

## Modules 9 à 14 — Données, SFT et premières mesures

| Module | Ce que tu dois savoir vérifier | Mode du laboratoire |
| --- | --- | --- |
| [Module 9 — Construire et vérifier un dataset de qualité](09_construire_un_dataset_de_qualite.md) | Définir et contrôler un contrat de données. | CPU |
| [Module 10 — Charger, inspecter et transformer des données](10_charger_inspecter_et_transformer_les_donnees.md) | Transformer sans perdre la traçabilité. | CPU |
| [Module 11 — Séparer train, validation et test sans tricher involontairement](11_train_validation_test_et_contamination.md) | Séparer par unités qui évitent les fuites. | CPU |
| [Module 12 — Comprendre et réaliser un premier SFT](12_premier_supervised_fine_tuning.md) | Inspecter les labels et produire un modèle SFT local. | CPU |
| [Module 13 — Mesurer une amélioration avant de la déclarer](13_baseline_et_evaluation_avant_apres.md) | Mesurer une baseline et un après comparables. | CPU |
| [Module 14 — Reconnaître le surapprentissage et choisir un checkpoint](14_surapprentissage_et_generalisation.md) | Distinguer mémorisation et validation. | CPU |

## Modules 15 à 24 — Paramètres entraînés et compromis mémoire

| Module | Ce que tu dois savoir vérifier | Mode du laboratoire |
| --- | --- | --- |
| [Module 15 — Comprendre la correction basse-rang de LoRA](15_peft_et_intuition_de_lora.md) | Calculer une correction basse-rang. | CPU |
| [Module 16 — Injecter, entraîner et recharger un adapter LoRA](16_injecter_et_entrainer_lora.md) | Injecter puis recharger un adapter. | CPU |
| [Module 17 — Auditer les paramètres et l’optimizer](17_mesurer_les_parametres_entrainables.md) | Auditer gradients et groupes optimizer. | CPU |
| [Module 18 — Régler le rang, alpha et le dropout LoRA](18_rank_lora_alpha_et_dropout.md) | Tester scaling et initialisation LoRA. | CPU |
| [Module 19 — Choisir où injecter les corrections LoRA](19_choisir_les_target_modules.md) | Relier modules ciblés et architecture. | CPU |
| [Module 20 — Comprendre QLoRA et distinguer stockage et calcul](20_qlora_et_quantification_4_bits.md) | Séparer stockage 4 bits et calcul. | CPU pour les principes ; QLoRA sous conditions GPU |
| [Module 21 — Comprendre le full fine-tuning](21_full_fine_tuning_principe.md) | Mesurer les changements en full FT. | CPU |
| [Module 22 — Construire une configuration de training surveillée](22_full_ft_configuration_stable.md) | Surveiller une configuration avant de l’étendre. | CPU |
| [Module 23 — Accumuler les gradients sans fausser leur moyenne](23_gradient_accumulation_et_batch_effectif.md) | Pondérer correctement des microbatches. | CPU |
| [Module 24 — Économiser des activations en recalculant](24_gradient_checkpointing_et_memoire.md) | Mesurer le compromis recalcul/activations. | CPU |

## Modules 25 à 36 — Objectifs, préférences et reinforcement learning

| Module | Ce que tu dois savoir vérifier | Mode du laboratoire |
| --- | --- | --- |
| [Module 25 — Continuer l’objectif de langage sur un domaine](25_continued_pretraining.md) | Poursuivre l’objectif de langage. | CPU |
| [Module 26 — Choisir une intervention à partir des erreurs observées](26_cpt_vs_sft_savoir_choisir.md) | Choisir à partir d’erreurs observées. | CPU |
| [Module 27 — Construire des préférences et comprendre l’objectif DPO](27_donnees_de_preference_et_dpo.md) | Auditer les paires et la marge DPO. | CPU |
| [Module 28 — Exécuter DPO et interpréter ses métriques](28_entrainer_avec_dpotrainer.md) | Entraîner et sauvegarder politique et référence. | CPU |
| [Module 29 — Apprendre une fonction de récompense et tester ses limites](29_reward_modeling.md) | Tester les limites d’un score appris. | CPU |
| [Module 30 — Comprendre une première boucle de reinforcement learning](30_fondements_du_reinforcement_learning.md) | Identifier état, action et signal de retour. | CPU |
| [Module 31 — Construire une récompense vérifiable et testée](31_construire_une_reward_verifiable.md) | Tester un correcteur JSON strict : 25 cas. | CPU |
| [Module 32 — Combiner des rewards sans cacher les compromis](32_rewards_composites_et_normalisation.md) | Comparer somme pondérée et gate. | CPU |
| [Module 33 — Chercher les réponses qui trompent le reward](33_reward_hacking_et_robustesse.md) | Trouver des réponses qui exploitent un proxy. | CPU |
| [Module 34 — Comprendre et observer une optimisation relative par groupes](34_grpo.md) | Optimiser une politique miniature par groupes. | CPU |
| [Module 35 — Comprendre le clipping PPO et le rôle du critic](35_ppo.md) | Observer ratio clippé et critic sur un bandit. | CPU |
| [Module 36 — Calculer une KL et savoir quelle dérive elle mesure](36_kl_divergence_et_controle_de_derive.md) | Comparer KL exactes et estimateurs. | CPU |

## Modules 37 à 47 — Construire le pré-entraînement

| Module | Ce que tu dois savoir vérifier | Mode du laboratoire |
| --- | --- | --- |
| [Module 37 — Planifier un pré-entraînement depuis des poids aléatoires](37_pre_entrainement_from_scratch.md) | Budgéter paramètres, positions et updates. | CPU |
| [Module 38 — Entraîner et figer un tokenizer BPE](38_entrainer_un_tokenizer_bpe.md) | Entraîner et sauvegarder un BPE local. | CPU |
| [Module 39 — Évaluer la fidélité et le coût d’un tokenizer](39_evaluer_un_tokenizer.md) | Comparer fidélité et longueur hors train. | CPU |
| [Module 40 — Dimensionner un petit Transformer cohérent](40_concevoir_un_petit_transformer.md) | Vérifier une formule de paramètres : 39 808. | CPU |
| [Module 41 — Vérifier l’initialisation avant les premières updates](41_initialisation_aleatoire_et_stabilite.md) | Observer symétrie, gradients et saturation. | CPU |
| [Module 42 — Écrire une boucle causale complète et comptable](42_ecrire_une_boucle_de_pre_entrainement.md) | Compter updates, entrées et cibles réelles. | CPU |
| [Module 43 — Lire une courbe et retrouver les données qui l’expliquent](43_lire_les_courbes_de_loss.md) | Relier un pic aux logs et aux données. | CPU, résultats fictifs à analyser |
| [Module 44 — Calculer la perplexité avec le bon dénominateur](44_perplexite_et_ses_limites.md) | Pondérer la NLL et compter les cibles une fois. | CPU |
| [Module 45 — Construire un pipeline de corpus traçable](45_pipeline_de_donnees_de_pre_entrainement.md) | Tracer nettoyage, splits et shards. | CPU |
| [Module 46 — Détecter des copies sans confondre ressemblance et équivalence](46_deduplication_exacte_et_near_duplicate.md) | Séparer copie exacte et similarité. | CPU |
| [Module 47 — Choisir la proportion de tokens réellement vus](47_dataset_mixture_et_sampling.md) | Relier tirage de documents et part de tokens. | CPU |

## Modules 48 à 52 — Distribution, ressources et exploitation

| Module | Ce que tu dois savoir vérifier | Mode du laboratoire |
| --- | --- | --- |
| [Module 48 — Répliquer le modèle et synchroniser les gradients avec DDP](48_data_parallel_et_ddp.md) | Comparer deux ranks à une référence globale. | Deux processus CPU pour le test DDP |
| [Module 49 — Répartir les états avec FSDP](49_fsdp_et_sharding.md) | Simuler le partage puis étudier FSDP2. | Simulation CPU ; FSDP2 réel sur deux GPU CUDA |
| [Module 50 — Estimer les postes et mesurer le pic mémoire](50_budget_memoire_du_training.md) | Comparer budget théorique et allocations. | CPU ; mesure CUDA conditionnelle |
| [Module 51 — Tester une vraie reprise, pas seulement un fichier de poids](51_checkpoints_et_reprise.md) | Comparer trajectoire continue et reprise. | CPU |
| [Module 52 — Enregistrer des métriques que l’on peut relire](52_logging_et_tensorboard.md) | Écrire puis relire JSONL et événements. | CPU |

## Modules 53 à 55 — Évaluation et conservation des capacités

| Module | Ce que tu dois savoir vérifier | Mode du laboratoire |
| --- | --- | --- |
| [Module 53 — Transformer un objectif en benchmark versionné](53_construire_un_benchmark.md) | Scorer aussi les cas manquants et invalides. | CPU, résultats fictifs à analyser |
| [Module 54 — Comparer des entraînements avec contrôles et incertitude](54_comparer_scientifiquement_des_entrainements.md) | Contrôler facteurs, incertitude et coûts. | CPU, résultats fictifs à analyser |
| [Module 55 — Mesurer ce qu’une spécialisation fait perdre](55_catastrophic_forgetting.md) | Mesurer conflit de tâches et replay. | CPU |

## Modules 56 à 62 — Projets de synthèse

| Module | Ce que tu dois savoir vérifier | Mode du laboratoire |
| --- | --- | --- |
| [Module 56 — Projet 1 : livrer un adapter LoRA et sa preuve d’évaluation](56_projet_1_lora_specialise.md) | Entraîner, figer la base et recharger LoRA. | CPU ; artefact du module 12 requis |
| [Module 57 — Projet 2 : mesurer LoRA et QLoRA sur la même tâche](57_projet_2_lora_vs_qlora.md) | Analyser des chiffres fictifs puis définir une mesure réelle. | Analyse fictive CPU ; comparaison réelle sous conditions GPU |
| [Module 58 — Projet 3 : comparer LoRA et full fine-tuning sans cacher les coûts](58_projet_3_lora_vs_full_fine_tuning.md) | Comparer full FT et LoRA sur une correction contrôlée. | CPU |
| [Module 59 — Projet 4 : isoler la valeur ajoutée d’un CPT](59_projet_4_continued_pretraining.md) | Isoler une branche CPT avant un SFT commun. | CPU ; artefact du module 12 requis |
| [Module 60 — Projet 5 : vérifier la préférence apprise par DPO](60_projet_5_dpo.md) | Auditer des marges DPO sur des artefacts rechargés. | CPU ; artefacts du module 28 requis |
| [Module 61 — Projet 6 : une boucle online avec correcteur vérifié](61_projet_6_rl_verifiable.md) | Apprendre à sélectionner une procédure avec un correcteur testé. | CPU |
| [Module 62 — Projet final : construire et vérifier toute la chaîne](62_projet_7_llm_from_scratch_de_bout_en_bout.md) | Assembler tokenizer, pré-entraînement, reprise, SFT et génération. | CPU |

## Dépendances pratiques et parcours de retour

Les chemins de sortie sont relatifs au **dossier de lancement**. Les projets 56 et 59 attendent `outputs/module12/final`. Le projet 60 attend `outputs/module28/final` et `outputs/module28/reference`. Exécute les laboratoires producteurs dans la même racine avant les projets consommateurs.

Le projet 62 est une chaîne locale autonome. Son modèle minuscule permet de vérifier les étapes et la reprise ; ses textes générés ne prouvent pas une maîtrise linguistique.

Si tu bloques :

- **Loss et labels :** reviens à la phase 1, puis aux modules 12, 23, 42 et 44.
- **Choix d’objectif :** relis le module 26 et l’annexe de choix ; LoRA est un choix de paramètres, CPT/SFT un choix de signal.
- **Reward et dérive :** repars des modules 30 à 36 avant le projet 61.
- **Mémoire et reprise :** relis les modules 24 et 48 à 51 ; distingue le checkpoint sur disque du recalcul d’activations.
- **Amélioration incertaine :** repars de la baseline, du split et des modules 53 à 55.

Le détail des exercices, observations attendues et limites se trouve dans chaque module. Les résultats matériels et la couverture de validation sont documentés dans Sources et API.
