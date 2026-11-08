# Annexe — Choisir une technique à partir du problème observé

Cette annexe résume des hypothèses de départ. Elle ne transforme pas une erreur en diagnostic certain et ne remplace pas une baseline.

Pour la démarche complète, lis le [module 26](26_cpt_vs_sft_savoir_choisir.md). Pour les mesures, utilise les [modules 53](53_construire_un_benchmark.md) et [54](54_comparer_scientifiquement_des_entrainements.md).

## 1. Séparer quatre décisions

| Question | Choix possibles | Ce qu’elle ne décide pas seule |
| --- | --- | --- |
| Quelle information fournir à l’usage ? | Prompt, contexte, récupération documentaire, outils | Comment entraîner les poids |
| Quel signal apprendre ? | Langage documentaire/CPT, SFT, préférences, reward/RL | Quels paramètres changer |
| Quels paramètres modifier ? | Full FT, LoRA, certaines couches | Type de données ou vérité des réponses |
| Comment faire tenir/exécuter ? | Dtypes, quantification, accumulation, checkpointing, DDP/FSDP | Qualité finale |

Exemples compatibles : SFT avec LoRA, CPT en full FT, préférence avec adapters. QLoRA combine représentation quantifiée de la base et petites corrections entraînées.

Le nom le plus avancé n’est pas un critère de choix. Il faut préciser besoin, mesure, données disponibles et coût.

## 2. Partir d’un contre-test simple

| Observation | Premier test informatif | Hypothèse à étudier si le test la soutient |
| --- | --- | --- |
| Information absente ou récente | Fournir explicitement un passage correct et pertinent | Récupération documentaire/RAG si ce contexte aide |
| Bonne information mais format incorrect | Consigne précise, exemple ou contrainte de sortie adaptée | SFT de comportement si le besoin reste stable |
| Difficulté persistante sur un domaine | Contrôler le contexte, le vocabulaire et les tâches de domaine | CPT et/ou SFT ciblé selon les échecs |
| Deux réponses acceptables mais préférence claire | Écrire une grille et annoter des paires difficiles | DPO/autre préférence avec validation indépendante |
| Réussite calculable mais pas toutes les réponses connues | Tester oracle, parser et contraste initial | RL vérifiable si la boucle est justifiée |
| Besoin scientifique d’un tokenizer/architecture nouveaux | Prototype complet très petit | From scratch sous budget explicite |

Fournir un document ne suffit pas à prouver un bon système RAG : récupération, version du passage, contexte et génération doivent être évalués. Le cours traite ce choix, sans constituer un tutoriel complet de moteur de recherche documentaire.

## 3. Choisir les paramètres à optimiser

| Méthode | Quand la comparer | Coût/limite à vérifier | Modules |
| --- | --- | --- | --- |
| LoRA | Une correction factorisée peut suffire et l’on veut limiter les états optimisés | Base encore nécessaire, choix de rang/cibles, régressions possibles | [15](15_peft_et_intuition_de_lora.md) à [19](19_choisir_les_target_modules.md) |
| QLoRA | Le stockage de base est une limite et le backend est compatible | Erreur de quantification, compute dtype, activations et surcoûts | [20](20_qlora_et_quantification_4_bits.md), [57](57_projet_2_lora_vs_qlora.md) |
| Full FT | La liberté supplémentaire peut apporter un gain utile | Gradients/optimizer à grande échelle, coût et conservation générale | [21](21_full_fine_tuning_principe.md), [58](58_projet_3_lora_vs_full_fine_tuning.md) |
| Gel partiel | On veut étudier une partie du réseau | Participation réelle à l’optimizer et au graphe | [17](17_mesurer_les_parametres_entrainables.md) |

Ne copie pas un LR LoRA vers full FT comme garantie d’équité. Pour une comparaison de recettes, accorde un budget de tuning annoncé et adapté à chaque méthode.

## 4. Identifier le poste mémoire qui limite

| Limite observée | Levier à tester | Ce qu’il ne fait pas |
| --- | --- | --- |
| Poids de base | Quantification compatible, modèle plus petit ou sharding | Réduire toutes les activations au même ratio |
| États d’optimizer/gradients | PEFT, autre optimizer adapté, sharding | Supprimer les poids gelés du forward |
| Activations longues | Micro-batch/séquence, regroupement, checkpointing | Réduire le nombre de paramètres |
| Batch logique souhaité | Accumulation pondérée correctement | Éviter le calcul des micro-batches |
| Débit limité avec réplique possible | DDP et pipeline de données | Faire tenir une réplique trop grande par division des poids |
| Réplique trop coûteuse par appareil | FSDP/autre sharding approprié | Garantir M/N au pic sans communication |

Mesure le premier step complet et les autres phases du run avec les conventions du [module 50](50_budget_memoire_du_training.md).

## 5. CPT, SFT, DPO et RL : quelles données faut-il ?

- **CPT :** documents pertinents, nettoyés, dédupliqués et séparés par groupes, avec budget de tokens et évaluation domaine/générale.
- **SFT :** sorties de référence correctes, contexte cohérent, format attendu et cibles actives inspectées.
- **DPO :** paires au même prompt, préférence justifiée, référence identifiée et audit des raccourcis.
- **RL :** policy qui peut explorer utilement, oracle/reward testé, données collectées et évaluation réservée.

Un petit jeu de démonstrations peut être un meilleur premier test qu’un pipeline RL complexe. À l’inverse, connaître une procédure de vérification ne signifie pas toujours disposer facilement de toutes les bonnes solutions à superviser.

Pour les projets : [56 LoRA](56_projet_1_lora_specialise.md), [59 CPT](59_projet_4_continued_pretraining.md), [60 DPO](60_projet_5_dpo.md), [61 RL](61_projet_6_rl_verifiable.md), [62 from scratch](62_projet_7_llm_from_scratch_de_bout_en_bout.md).

## 6. Quatre décisions expliquées

**Procédure mise à jour hier.** Tester la réponse avec le bon passage fourni. Si cela suffit, étudier une récupération/version des sources avant de réentraîner des faits changeants.

**Réponse correcte mais JSON invalide.** Vérifier contrat, prompt et décodage disponible ; tester un SFT de format sans perdre le contrôle du contenu.

**Jargon mal modélisé malgré un contexte adéquat.** Définir des tâches concrètes et un holdout documentaire ; comparer une intervention CPT/SFT plutôt que déclarer automatiquement le modèle « sans connaissances ».

**Reward en hausse mais réponses moins utiles.** Auditer le scorer et ses extrêmes, pas choisir immédiatement un autre algorithme RL. Voir [31](31_construire_une_reward_verifiable.md) à [33](33_reward_hacking_et_robustesse.md).

## 7. Fiche de décision à remplir

```text
Échec observé et exemples :
Informations réellement disponibles :
Hypothèses concurrentes :
Contre-test le plus simple :
Intervention principale :
Données et paramètres modifiés :
Baseline et mesures réservées :
Budget et contraintes :
Critère qui réfuterait l’hypothèse :
Résultat, régressions et prochain test :
```

Une décision argumentée peut être « ne pas entraîner pour l’instant ». Elle reste utile si elle découle d’un test et de contraintes explicites.

Retour à l’[index](INDEX.md) ou à l’[annexe de diagnostic](ANNEXE_DIAGNOSTIC_ENTRAINEMENT.md).
