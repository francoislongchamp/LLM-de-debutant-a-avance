# Glossaire — comprendre les termes et leurs différences

Utilise ce fichier pour retrouver une définition, puis le module indiqué pour le raisonnement, les calculs et les expériences. Les termes sont regroupés par mécanisme : plusieurs mots proches désignent des objets différents.

## 1. Texte, tokenizer et séquences

À approfondir : [1](01_pipeline_complet_dun_llm.md), [2](02_tokenisation_en_profondeur.md), [3](03_fenetre_de_contexte_et_sequences.md), [38](38_entrainer_un_tokenizer_bpe.md), [39](39_evaluer_un_tokenizer.md).

| Terme | Définition et point d’attention |
| --- | --- |
| Octet / caractère / token | Unité de stockage, unité textuelle Unicode selon la convention, et unité du tokenizer. Un caractère peut occuper plusieurs octets et plusieurs tokens ; leur comptage n’est pas interchangeable. |
| Token | Élément du vocabulaire identifié par un entier. Un token n’est pas nécessairement un mot. |
| Tokenizer | Ensemble de règles et d’artefacts pour convertir texte en IDs et inversement. Normalisation, pré-tokenisation et tokens spéciaux font partie de son contrat. |
| Vocabulaire | Association entre éléments et IDs. Une même taille de vocabulaire ne rend pas deux tokenizers compatibles avec les mêmes embeddings. |
| BPE | Apprentissage de fusions fréquentes entre unités initiales. Le vocabulaire et les fusions dépendent du corpus d’apprentissage et de leur configuration. |
| Byte-level BPE | BPE partant d’une représentation des octets. Avec un alphabet initial complet, un texte nouveau peut être encodé sans remplacer ses caractères par UNK ; le round-trip dépend aussi de la normalisation et du décodage. |
| Normalisation textuelle | Transformation préalable : casse, Unicode, espaces, etc. Une transformation destructive peut empêcher de retrouver exactement le texte d’origine. |
| Pré-tokenisation | Découpage avant les fusions ou la segmentation du modèle de tokenizer. Il influence les frontières possibles. |
| Token spécial | Marqueur réservé : PAD, EOS, BOS, rôles, etc. Son ID et son interprétation doivent être cohérents entre tokenizer et modèle. |
| BOS / EOS / PAD / UNK | Début, fin de séquence, remplissage et unité inconnue. Les partager ou les confondre peut modifier les masques et l’arrêt de génération. |
| Round-trip | Test `decode(encode(texte)) == texte`, sous un contrat de normalisation précisé. Il ne mesure pas l’efficacité de compression. |
| Fertilité | Nombre de tokens par unité textuelle choisie, par exemple mot ou caractère. Il faut annoncer le dénominateur et le corpus. |
| P95 | Valeur de percentile 95 sous une convention de calcul précisée. Utile pour voir les séquences longues qu’une moyenne cache. |
| Fenêtre de contexte | Limite de positions que le modèle peut traiter dans la configuration utilisée. Prompt et génération doivent partager ce budget. |
| Padding / troncature | Remplir pour former un batch / retirer du contenu pour respecter une limite. La troncature peut supprimer la réponse cible entière. |
| Chat template | Règle transformant rôles et contenus en séquence. Son texte, ses tokens et ses masques font partie du protocole. |
| Packing | Remplissage de séquences avec plusieurs exemples. L’indépendance entre exemples dépend des masques, positions et frontières employés. |

## 2. Modèle et prédiction

À approfondir : [4](04_anatomie_du_transformer.md), [5](05_parametres_et_memoire_des_poids.md), [6](06_loss_et_prediction_du_prochain_token.md), [37](37_pre_entrainement_from_scratch.md), [40](40_concevoir_un_petit_transformer.md), [41](41_initialisation_aleatoire_et_stabilite.md).

| Terme | Définition et point d’attention |
| --- | --- |
| Paramètre / poids | Valeur du modèle, éventuellement apprise. Un buffer enregistré n’est pas automatiquement un paramètre optimisé. |
| Embedding | Vecteur appris associé à un ID, ou mécanisme de représentation des positions. Les IDs sont des indices, pas des quantités continues à comparer numériquement. |
| Activation | Valeur intermédiaire produite pendant le forward. Certaines sont conservées pour le backward. |
| Attention Q/K/V | Calcul où queries et keys déterminent des poids utilisés pour combiner les values. Les projections Q, K et V sont des paramètres ; leurs sorties sont des activations. |
| Tête d’attention | Sous-espace et calcul d’attention d’un bloc. Le nombre de têtes doit respecter les contraintes de largeur de l’architecture. |
| Causal mask | Contrainte empêchant une position d’utiliser des tokens futurs dans la prédiction causale. |
| Attention mask | Champ indiquant des positions ou relations à prendre en compte selon l’API. Il ne remplace pas automatiquement le masque des labels. |
| MLP / connexion résiduelle / LayerNorm | Transformation non linéaire, ajout d’un chemin direct, et normalisation des activations au sein d’un bloc. |
| Logits | Scores réels avant normalisation en probabilités. Leur dimension finale correspond généralement au vocabulaire. |
| Softmax / log-softmax | Transformation de scores en probabilités / logarithmes de probabilités. Le log-softmax évite de calculer naïvement un logarithme après un arrondi à zéro. |
| Autoregressif | Modèle factorisant une séquence en prédictions conditionnées par le préfixe. À la génération, les tokens produits deviennent du contexte. |
| Teacher forcing | En entraînement, utilisation des tokens précédents de référence comme contexte. La génération libre peut se comporter différemment. |
| Weight tying | Partage d’un même poids, souvent embeddings d’entrée et projection de sortie. Le compter deux fois fausse le budget de paramètres uniques. |
| From scratch | Entraînement depuis des poids initialisés, sans recharger les connaissances d’une base préentraînée. Un tokenizer réutilisé ne fournit pas ces poids. |
| Base model / modèle instruct | Modèle de langage de base / modèle adapté à un format et à des comportements d’instruction. Les noms ne garantissent aucune qualité donnée. |
| Température / greedy | Réglage de la distribution de génération / choix du token le plus probable. Les paramètres de décodage font partie de l’évaluation. |

## 3. Loss, gradients et boucle d’entraînement

À approfondir : [6](06_loss_et_prediction_du_prochain_token.md), [7](07_gradients_et_autograd.md), [8](08_premiere_boucle_dentrainement_pytorch.md), [12](12_premier_supervised_fine_tuning.md), [22](22_full_ft_configuration_stable.md), [23](23_gradient_accumulation_et_batch_effectif.md), [42](42_ecrire_une_boucle_de_pre_entrainement.md), [44](44_perplexite_et_ses_limites.md).

| Terme | Définition et point d’attention |
| --- | --- |
| Loss | Objectif scalaire minimisé pendant l’entraînement. Sa baisse ne prouve pas à elle seule une amélioration sur la tâche réelle. |
| Cross-entropy / NLL | Pour une cible unique, opposé du logarithme de sa probabilité prédite. La réduction en moyenne ou somme et ses unités doivent être précisées. |
| Labels | Cibles de supervision. En LM causal, beaucoup d’API décalent logits et labels à l’intérieur du modèle. |
| Shift causal | Alignement de la sortie en position t sur la cible en position t+1. Le faire deux fois entraîne sur les mauvaises cibles. |
| `-100` / ignore index | Valeur ignorée par certaines losses et collators. Elle doit être reconnue par l’implémentation ; ce n’est pas un token du vocabulaire. |
| Cible active | Position qui contribue effectivement à la loss, après décalage et masquage. Le nombre de cibles diffère des positions d’entrée non PAD. |
| Completion-only / assistant-only | Supervision de la completion / des segments assistant. Le second cas exige que le template et l’API identifient correctement ces segments. |
| Forward / backward | Calcul des sorties et de la loss / calcul de ses gradients via le graphe. Un forward réussi ne vérifie pas le backward. |
| Autograd | Différentiation automatique de PyTorch. Il faut conserver les dépendances nécessaires au graphe. |
| Gradient | Dérivée locale de la loss par rapport à un paramètre. Il ne représente pas une garantie d’amélioration après une update arbitrairement grande. |
| `requires_grad` / paramètre gelé | Drapeau de suivi des gradients / paramètre exclu de l’apprentissage prévu. Vérifie aussi la composition de l’optimizer et les changements réels de poids. |
| Optimizer / AdamW | Règle d’update / famille utilisant moments et weight decay découplé. L’état de l’optimizer fait partie d’une vraie reprise. |
| Learning rate / scheduler | Échelle de mise à jour / règle d’évolution de cette échelle. Précise si le scheduler avance par update ou par microbatch. |
| Gradient clipping | Réduction d’un gradient dépassant un seuil selon une norme choisie. Il ne corrige pas des labels erronés. |
| Gradient norm | Norme des gradients selon un périmètre explicite, avant ou après clipping. L’ordre de mesure change l’interprétation. |
| Microbatch / accumulation | Batch d’un passage local / regroupement de plusieurs backward avant une update. Les moyennes doivent être pondérées par les cibles de l’objectif. |
| Batch global effectif | Avec des batches pleins de même taille : microbatch par processus × processus × accumulation. Les dernières fenêtres et les nombres de tokens peuvent différer. |
| Step / global step | Compteur dont il faut définir l’unité. Dans les laboratoires de boucle, une update optimizer incrémente le global step, pas chaque microbatch. |
| Epoch / exposition | Passage sur un ensemble selon le sampler / présentation d’exemples ou tokens au modèle. Sampling avec remise et répétitions changent le sens d’une epoch. |
| Perplexité | Exponentielle de la NLL moyenne en logarithme naturel sur les cibles évaluées. Elle dépend du tokenizer, du corpus et du protocole de contexte. |
| Overfitting / généralisation | Amélioration sur les données vues sans progrès équivalent sur les données indépendantes / capacité à réussir au-delà des exemples d’apprentissage. |

## 4. Données et évaluation

À approfondir : [9](09_construire_un_dataset_de_qualite.md), [11](11_train_validation_test_et_contamination.md), [13](13_baseline_et_evaluation_avant_apres.md), [45](45_pipeline_de_donnees_de_pre_entrainement.md), [46](46_deduplication_exacte_et_near_duplicate.md), [47](47_dataset_mixture_et_sampling.md), [53](53_construire_un_benchmark.md), [54](54_comparer_scientifiquement_des_entrainements.md), [55](55_catastrophic_forgetting.md).

| Terme | Définition et point d’attention |
| --- | --- |
| Train / validation / test | Apprentissage / choix de configuration / estimation finale sur un ensemble tenu à l’écart. Consulter le test pour choisir une variante le transforme en validation. |
| Contamination / leakage | Information d’évaluation utilisée dans les données ou décisions d’apprentissage. Des reformulations d’un même document peuvent fuir entre splits. |
| Split par groupe | Affectation d’un document source, client, famille ou autre groupe entier à un seul split. La bonne unité dépend du déploiement visé. |
| Déduplication exacte | Retrait des copies selon une représentation normalisée définie. Cette normalisation peut aussi effacer des différences importantes. |
| Near duplicate | Paire très ressemblante selon un critère. Ce n’est pas nécessairement la même information ni la même réponse correcte. |
| Shingle / Jaccard | Fragment contigu d’unités / rapport intersection sur union d’ensembles. Une union vide demande une convention explicite. |
| MinHash / LSH | Estimation de similarité d’ensembles / recherche approximative de candidats proches. Les candidats demandent encore un traitement adapté. |
| Shard / manifest / hash | Fragment de corpus / inventaire de données et métadonnées / empreinte d’un contenu. Un hash prouve une identité de contenu, pas sa qualité ou ses droits d’usage. |
| Mixture / sampling | Mélange de sources / règle de tirage. 50 % des documents d’une source ne donnent pas forcément 50 % des tokens. |
| Baseline | Point de comparaison mesuré avec le même protocole. Un résultat final seul ne mesure pas un progrès. |
| Benchmark | Ensemble de cas, règles de score et procédure d’exécution versionnés. Il doit représenter l’usage et compter aussi les sorties manquantes ou invalides. |
| Régression / forgetting | Perte de performance sur une capacité ou catégorie / perte après un nouvel apprentissage. Une moyenne stable peut masquer cette perte. |
| Replay | Réutilisation de données d’anciennes tâches pendant une spécialisation. Sa proportion et son coût font partie du protocole. |
| Ablation / contrôle | Retrait ou variation isolée d’un facteur / branche permettant de tester une explication. Garder le nombre d’updates n’égalise pas toujours le calcul. |
| Comparaison appariée | Comparaison des variantes sur les mêmes cas ou seeds. Conserve les paires lors du calcul d’incertitude. |
| Bootstrap / intervalle | Rééchantillonnage pour estimer la variabilité / plage calculée sous des hypothèses. Cela ne corrige pas un benchmark non représentatif. |
| Front de Pareto | Variantes pour lesquelles aucune autre ne fait au moins aussi bien sur tous les critères et strictement mieux sur l’un. Le choix dépend ensuite des contraintes. |
| Fixture / résultat fictif | Données construites pour exercer un calcul. Elles ne constituent pas une mesure réelle de qualité, débit ou mémoire. |

## 5. Adaptation et paramètres entraînés

À approfondir : [15](15_peft_et_intuition_de_lora.md), [18](18_rank_lora_alpha_et_dropout.md), [20](20_qlora_et_quantification_4_bits.md), [21](21_full_fine_tuning_principe.md), [25](25_continued_pretraining.md), [26](26_cpt_vs_sft_savoir_choisir.md), [56](56_projet_1_lora_specialise.md) à [59](59_projet_4_continued_pretraining.md).

| Terme | Définition et point d’attention |
| --- | --- |
| SFT | Apprentissage supervisé à partir de sorties de référence. Le masque définit quelles parties de la séquence sont apprises. |
| CPT | Poursuite de l’objectif de langage sur un corpus, souvent de domaine. Ce choix d’objectif est indépendant du choix full FT ou LoRA. |
| Full fine-tuning | Adaptation de l’ensemble des paramètres visés du modèle. Les états optimizer et gradients peuvent coûter plus que les poids seuls. |
| PEFT | Famille d’adaptations où une fraction des paramètres est entraînée. PEFT est aussi le nom d’une bibliothèque. |
| Adapter | Ensemble de paramètres d’adaptation à charger avec la bonne base et la bonne configuration. Tous les adapters ne sont pas des LoRA. |
| LoRA | Correction matricielle `ΔW = (alpha/r) BA` dans la variante usuelle. Les poids de base restent gelés dans l’usage standard. |
| Rank `r` / alpha | Dimension intermédiaire de la correction / facteur de mise à l’échelle. D’autres variantes, dont rsLoRA, emploient une autre règle de scaling. |
| Initialisation LoRA | Souvent une matrice aléatoire et une nulle, rendant la correction initiale nulle. Mettre les deux à zéro bloque les gradients de cette branche. |
| `target_modules` | Modules où installer les corrections. Les noms et orientations des poids dépendent de l’architecture. |
| LoRA dropout / merge | Régularisation de la branche adaptée / incorporation de sa correction dans un poids compatible. La fidélité du merge dépend de la représentation et de l’implémentation. |
| Quantification / NF4 | Représentation de poids avec moins de bits / format 4 bits utilisé dans certaines configurations QLoRA. Échelles et métadonnées prennent aussi de la place. |
| QLoRA | Adaptation de paramètres LoRA avec une base quantifiée gelée. Cela ne signifie pas entraîner tous les poids avec des gradients 4 bits. |
| Storage dtype / compute dtype | Représentation de stockage / type utilisé pour les calculs. Leurs tailles et leurs coûts sont différents. |

## 6. Préférences, récompenses et RL

À approfondir : [27](27_donnees_de_preference_et_dpo.md) à [36](36_kl_divergence_et_controle_de_derive.md), [60](60_projet_5_dpo.md), [61](61_projet_6_rl_verifiable.md).

| Terme | Définition et point d’attention |
| --- | --- |
| Chosen / rejected | Réponses préférée et moins préférée pour le même contexte sous un critère annoncé. La préférence n’établit pas une vérité universelle. |
| DPO | Objectif de préférence augmentant la marge de log-probabilité de la politique relativement à une référence. Il ne requiert pas d’entraîner explicitement un reward model séparé. |
| Reward model | Modèle apprenant un score, souvent à partir de préférences. Ce score peut mal généraliser et être exploité. |
| Reward vérifiable | Score calculé par un correcteur aux règles explicites. Le parseur, les types acceptés et les cas adversariaux font partie de sa validité. |
| Reward hacking | Comportement obtenant un bon score tout en manquant l’intention du concepteur. Optimiser un proxy peut amplifier sa faille. |
| Reward composite / gate | Combinaison de composantes / condition bloquante. Une somme pondérée autorise des compensations qu’une gate peut interdire. |
| État / action / politique | Information utilisée pour décider / décision / distribution des décisions conditionnée par l’information. Pour un LLM, le préfixe peut être l’état et le prochain token l’action. |
| Rollout / trajectoire | Séquence d’interactions produite par une politique et utilisée pour l’apprentissage ou l’évaluation. |
| Reward / retour | Signal reçu à une étape / somme, éventuellement actualisée, de rewards à partir d’un point. |
| Value / critic | Estimation du retour attendu / modèle qui l’estime. Il faut préciser l’état, la politique et la cible. |
| Advantage | Estimation du retour relativement à une baseline. Un avantage positif rend l’action relativement préférable selon cet estimateur. |
| On-policy / offline | Données issues de la politique de collecte pertinente pour l’update / données déjà constituées. Le nombre de réutilisations et la dérive doivent être contrôlés. |
| Ancienne politique | Politique ayant collecté les actions ; son log-probabilités sert au ratio proximal. Ce rôle diffère de celui de la référence. |
| Politique de référence | Distribution servant d’ancre dans une pénalité ou une comparaison. Elle peut rester figée sur de nombreuses updates. |
| Ratio d’importance | `π_new(action\|état) / π_old(action\|état)`. L’ordre du numérateur et la politique de collecte comptent. |
| PPO / clipping | Méthode utilisant notamment un objectif proximal dont une branche borne le ratio. Le clipping ne garantit pas une borne dure sur la KL réelle. |
| GRPO | Famille d’optimisations utilisant des récompenses relatives dans des groupes de réponses. Les variantes diffèrent sur normalisation, clipping et KL ; la miniature du module 34 en isole des mécanismes. |
| Normalisation de groupe | Centrage, souvent réduction, des rewards de réponses au même contexte. Elle n’est pas une calibration globale des composantes d’un reward composite. |
| KL `D(P∥Q)` | Espérance sous P de `log(P/Q)`. Asymétrique ; infinie si P attribue une masse positive là où Q est nulle. |
| KL exacte / estimateur | Somme sur toute la distribution / valeur estimée à partir d’échantillons et d’hypothèses. Un log-ratio individuel peut être négatif même si la KL exacte est non négative. |
| Beta | Hyperparamètre dont le sens dépend de l’objectif : coefficient de KL ou échelle de préférence, par exemple. Il faut lire sa formule avant de le comparer. |

## 7. Ressources, distribution et reprise

À approfondir : [24](24_gradient_checkpointing_et_memoire.md), [48](48_data_parallel_et_ddp.md) à [52](52_logging_et_tensorboard.md), [62](62_projet_7_llm_from_scratch_de_bout_en_bout.md).

| Terme | Définition et point d’attention |
| --- | --- |
| FP32 / FP16 / BF16 | Formats flottants aux précisions et plages différentes. BF16 garde une large plage d’exposant ; le support matériel et les kernels comptent. |
| Mixed precision / loss scaling | Usage coordonné de plusieurs précisions / mise à l’échelle de la loss pour limiter certains problèmes de sous-flux, surtout en FP16. |
| Mémoire des poids / pic de training | Taille des paramètres stockés / maximum incluant selon le périmètre gradients, états optimizer, activations, temporaires et allocations. |
| GiB / GB | 2³⁰ octets / 10⁹ octets. L’unité doit accompagner le nombre. |
| Gradient checkpointing | Recalcul d’activations pendant le backward pour réduire leur stockage. Il échange généralement mémoire et calcul ; il ne sauvegarde pas un run sur disque. |
| DDP | Réplication du modèle par processus et synchronisation des gradients. Les données doivent être réparties et les moyennes locales correctement pondérées. |
| Rank / world size | Identité d’un processus / nombre de processus du groupe distribué. Le rank n’est pas ici le rang LoRA. |
| DistributedSampler | Sampler répartissant les indices entre processus. `set_epoch` change le mélange par epoch ; padding ou `drop_last` changent les expositions. |
| FSDP / sharding | Distribution de certains états entre processus pour réduire leur stockage local. Des all-gathers et activations peuvent encore produire des pics. |
| FSDP1 / FSDP2 | APIs et modes d’implémentation distincts. Les exemples du module 49 utilisent `fully_shard` de FSDP2, pas un simple renommage du wrapper FSDP1. |
| All-reduce / all-gather / reduce-scatter | Réduction avec résultat partagé / rassemblement des fragments / réduction puis répartition du résultat. |
| Checkpoint | Sauvegarde dont le contenu dépend de l’usage : poids seuls, adapter ou état complet de reprise. Le nom du fichier ne prouve pas son contenu. |
| RNG / seed | Générateur aléatoire et son état / valeur d’initialisation. Reprendre une trajectoire demande l’état courant, pas seulement la seed initiale. |
| Reprise exacte | Retour à la même trajectoire dans un périmètre défini incluant données, RNG, optimizer et scheduler. Le matériel et les opérations peuvent limiter cette égalité. |
| Sauvegarde atomique | Remplacement du fichier final seulement après écriture d’un fichier temporaire complet, sous les garanties du système de fichiers utilisé. |
| JSONL / TensorBoard | Fichier d’un objet JSON par ligne / outil de visualisation d’événements. Ils doivent porter des steps et unités cohérents et pouvoir être relus. |
| Model card | Fiche décrivant modèle, données, entraînement, évaluation, usages et limites. Elle doit distinguer les preuves des intentions. |

Pour choisir une intervention, consulte l’[annexe technique](ANNEXE_CHOISIR_LA_BONNE_TECHNIQUE.md). Pour une erreur concrète, consulte l’[annexe de diagnostic](ANNEXE_DIAGNOSTIC_ENTRAINEMENT.md). Les contrats des bibliothèques sont référencés dans [Sources et API](SOURCES_ET_API.md).
