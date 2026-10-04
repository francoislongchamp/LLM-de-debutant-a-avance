# Index — Formation complète LLM

Ce fichier est le point d’entrée du cours. Suis les modules dans l’ordre lors du premier passage. Les retours en arrière sont normaux : le cours est volontairement cumulatif.

## Avant les modules

- [Méthode de travail et environnement](00_METHODE_ET_ENVIRONNEMENT.md)
- [Choisir la bonne technique](ANNEXE_CHOISIR_LA_BONNE_TECHNIQUE.md)
- [Diagnostic d’entraînement](ANNEXE_DIAGNOSTIC_ENTRAINEMENT.md)
- [Glossaire](GLOSSAIRE.md)
- [Sources et API](SOURCES_ET_API.md)

## Phase 1 — Fondations du fonctionnement

- [Module 1 — Pipeline complet d’un LLM](01_pipeline_complet_dun_llm.md)
- [Module 2 — Tokenisation en profondeur](02_tokenisation_en_profondeur.md)
- [Module 3 — Fenêtre de contexte et séquences](03_fenetre_de_contexte_et_sequences.md)
- [Module 4 — Anatomie du Transformer](04_anatomie_du_transformer.md)
- [Module 5 — Paramètres et mémoire des poids](05_parametres_et_memoire_des_poids.md)
- [Module 6 — Loss et prédiction du prochain token](06_loss_et_prediction_du_prochain_token.md)
- [Module 7 — Gradients et autograd](07_gradients_et_autograd.md)
- [Module 8 — Première boucle d’entraînement PyTorch](08_premiere_boucle_dentrainement_pytorch.md)

## Phase 2 — Données et SFT

- [Module 9 — Construire un dataset de qualité](09_construire_un_dataset_de_qualite.md)
- [Module 10 — Charger, inspecter et transformer les données](10_charger_inspecter_et_transformer_les_donnees.md)
- [Module 11 — Train, validation, test et contamination](11_train_validation_test_et_contamination.md)
- [Module 12 — Premier Supervised Fine-Tuning](12_premier_supervised_fine_tuning.md)
- [Module 13 — Baseline et évaluation avant/après](13_baseline_et_evaluation_avant_apres.md)
- [Module 14 — Surapprentissage et généralisation](14_surapprentissage_et_generalisation.md)

## Phase 3 — PEFT, LoRA, QLoRA et Full FT

- [Module 15 — PEFT et intuition de LoRA](15_peft_et_intuition_de_lora.md)
- [Module 16 — Injecter et entraîner LoRA](16_injecter_et_entrainer_lora.md)
- [Module 17 — Mesurer les paramètres entraînables](17_mesurer_les_parametres_entrainables.md)
- [Module 18 — Rank LoRA, alpha et dropout](18_rank_lora_alpha_et_dropout.md)
- [Module 19 — Choisir les target modules](19_choisir_les_target_modules.md)
- [Module 20 — QLoRA et quantification 4 bits](20_qlora_et_quantification_4_bits.md)
- [Module 21 — Full fine-tuning — principe](21_full_fine_tuning_principe.md)
- [Module 22 — Full FT — configuration stable](22_full_ft_configuration_stable.md)
- [Module 23 — Gradient accumulation et batch effectif](23_gradient_accumulation_et_batch_effectif.md)
- [Module 24 — Gradient checkpointing et mémoire](24_gradient_checkpointing_et_memoire.md)

## Phase 4 — Continued pretraining et post-training

- [Module 25 — Continued pretraining](25_continued_pretraining.md)
- [Module 26 — CPT vs SFT — savoir choisir](26_cpt_vs_sft_savoir_choisir.md)
- [Module 27 — Données de préférence et DPO](27_donnees_de_preference_et_dpo.md)
- [Module 28 — Entraîner avec DPOTrainer](28_entrainer_avec_dpotrainer.md)
- [Module 29 — Reward modeling](29_reward_modeling.md)
- [Module 30 — Fondements du reinforcement learning](30_fondements_du_reinforcement_learning.md)
- [Module 31 — Construire une reward vérifiable](31_construire_une_reward_verifiable.md)
- [Module 32 — Rewards composites et normalisation](32_rewards_composites_et_normalisation.md)
- [Module 33 — Reward hacking et robustesse](33_reward_hacking_et_robustesse.md)
- [Module 34 — GRPO](34_grpo.md)
- [Module 35 — PPO](35_ppo.md)
- [Module 36 — KL divergence et contrôle de dérive](36_kl_divergence_et_controle_de_derive.md)

## Phase 5 — Pré-entraînement from scratch

- [Module 37 — Pré-entraînement from scratch](37_pre_entrainement_from_scratch.md)
- [Module 38 — Entraîner un tokenizer BPE](38_entrainer_un_tokenizer_bpe.md)
- [Module 39 — Évaluer un tokenizer](39_evaluer_un_tokenizer.md)
- [Module 40 — Concevoir un petit Transformer](40_concevoir_un_petit_transformer.md)
- [Module 41 — Initialisation aléatoire et stabilité](41_initialisation_aleatoire_et_stabilite.md)
- [Module 42 — Écrire une boucle de pré-entraînement](42_ecrire_une_boucle_de_pre_entrainement.md)
- [Module 43 — Lire les courbes de loss](43_lire_les_courbes_de_loss.md)
- [Module 44 — Perplexité et ses limites](44_perplexite_et_ses_limites.md)
- [Module 45 — Pipeline de données de pré-entraînement](45_pipeline_de_donnees_de_pre_entrainement.md)
- [Module 46 — Déduplication exacte et near-duplicate](46_deduplication_exacte_et_near_duplicate.md)
- [Module 47 — Dataset mixture et sampling](47_dataset_mixture_et_sampling.md)

## Phase 6 — Training distribué et observabilité

- [Module 48 — Data Parallel et DDP](48_data_parallel_et_ddp.md)
- [Module 49 — FSDP et sharding](49_fsdp_et_sharding.md)
- [Module 50 — Budget mémoire du training](50_budget_memoire_du_training.md)
- [Module 51 — Checkpoints et reprise](51_checkpoints_et_reprise.md)
- [Module 52 — Logging et TensorBoard](52_logging_et_tensorboard.md)

## Phase 7 — Évaluation scientifique

- [Module 53 — Construire un benchmark](53_construire_un_benchmark.md)
- [Module 54 — Comparer scientifiquement des entraînements](54_comparer_scientifiquement_des_entrainements.md)
- [Module 55 — Catastrophic forgetting](55_catastrophic_forgetting.md)

## Phase 8 — Projets de synthèse

- [Module 56 — Projet 1 — LoRA spécialisé](56_projet_1_lora_specialise.md)
- [Module 57 — Projet 2 — LoRA vs QLoRA](57_projet_2_lora_vs_qlora.md)
- [Module 58 — Projet 3 — LoRA vs full fine-tuning](58_projet_3_lora_vs_full_fine_tuning.md)
- [Module 59 — Projet 4 — Continued pretraining](59_projet_4_continued_pretraining.md)
- [Module 60 — Projet 5 — DPO](60_projet_5_dpo.md)
- [Module 61 — Projet 6 — RL vérifiable](61_projet_6_rl_verifiable.md)
- [Module 62 — Projet 7 — LLM from scratch de bout en bout](62_projet_7_llm_from_scratch_de_bout_en_bout.md)

## Contrat de progression

Pour considérer un module acquis, tu dois pouvoir **définir**, **expliquer**, **faire le micro-exemple**, **exécuter le lab** et **interpréter le résultat**. Un script qui s’exécute sans erreur ne suffit pas.
