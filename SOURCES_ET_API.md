# Sources techniques, API et portée des vérifications

Les modules expliquent les mécanismes avant d’utiliser une bibliothèque. Ce fichier permet de retrouver les sources primaires, d’identifier la pile observée et de distinguer un laboratoire vérifié d’une extension qui demande encore une mesure.

**État de cette révision : 8 novembre 2026.** Une documentation en ligne, notamment une branche `main`, peut décrire une API différente de celle installée. Vérifie la version locale et le sélecteur de version de la documentation avant de transposer un exemple.

## 1. Environnement observé

Les vérifications ont utilisé la pile suivante, déjà présente dans l’environnement de travail :

| Composant | Version observée |
| --- | --- |
| Python | 3.12.12 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.18.0 |
| Tokenizers | 0.23.2 |
| Datasets | 5.0.1 |
| Accelerate | 1.15.0 |
| PEFT | 0.21.2 |
| TRL | 1.14.1 |
| bitsandbytes | 0.50.2 |
| TensorBoard | 2.21.0 |

Ces versions décrivent une installation observée, **pas une installation neuve validée sur toutes les plateformes**. Le suffixe CUDA de PyTorch n’implique pas qu’un GPU soit accessible. Les tests de cette révision ont principalement utilisé le CPU.

Pour inspecter ton installation, utilise le script du [module 00](00_METHODE_ET_ENVIRONNEMENT.md), puis conserve `pip freeze` avec tes résultats. Pour choisir une roue PyTorch, consulte l’[installation officielle](https://pytorch.org/get-started/locally/) ou les [versions précédentes](https://pytorch.org/get-started/previous-versions/). Évite de remplacer une pile fonctionnelle au milieu d’une expérience comparée.

## 2. Ce qui a été vérifié

| Périmètre | Vérification effectuée | Limite de la preuve |
| --- | --- | --- |
| Modules 1–30 | Laboratoires locaux vérifiés lors de leur révision, avec modèles miniatures et bases mises en cache selon les exemples | Cela ne valide pas chaque modèle externe ni les extensions matérielles |
| Modules 31–41 | Exécution CPU des blocs Python ; tests du correcteur, calculs de rewards/KL, optimisation miniature, tokenizer et comptage des paramètres | GRPO/PPO sont des expériences réduites, pas une validation d’un training RLHF de LLM |
| Modules 42–53 | Exécution CPU : boucle causale, cibles, corpus, mémoire CPU, reprise exacte, relecture des événements et scoring | Les blocs CUDA conditionnels ne deviennent pas validés lorsqu’ils sont ignorés |
| Module 48 | Véritable lancement DDP à deux processus CPU ; chaque update comparée à une référence globale | Ne mesure ni réseau multi-machine ni débit GPU |
| Module 49 | Simulation CPU vérifiée ; présence de l’API `fully_shard` constatée | Le laboratoire FSDP2 réel sur deux GPU CUDA n’a pas été exécuté |
| Modules 54–60 | Analyses CPU et projets miniatures exécutés ; artefacts des modules 12 et 28 produits avant leurs consommateurs ; adapter rechargé et base gelée contrôlée | Le tableau LoRA/QLoRA du module 57 reste fictif ; aucun débit GPU n’en découle |
| Module 61 | 31 tests de correcteur, splits disjoints et apprentissage d’une sélection de procédures ; 2 560 actions journalisées | Les procédures arithmétiques sont déjà codées ; aucun raisonnement textuel n’est appris |
| Module 62 | BPE local, modèle de 39 808 paramètres, pré-entraînement, reprise exacte CPU, SFT, rechargement et génération | Une baisse de NLL sur ce petit corpus ne donne pas une qualité d’assistant ; la génération reste limitée |

Les projets 56 et 59 rechargent `outputs/module12/final`. Le projet 60 recharge `outputs/module28/final` et `outputs/module28/reference`. Les mêmes scripts doivent être lancés depuis la même racine pour résoudre ces chemins.

La comparaison CUDA/QLoRA réelle, les pics mémoire GPU et FSDP multi-GPU restent à vérifier sur le matériel cible. Les valeurs fictives sont signalées dans les modules ; elles ne doivent pas être recopiées comme des benchmarks obtenus.

## 3. Contrats d’API à contrôler

### Labels et SFT

Les modèles causaux Transformers usuels calculent le décalage causal dans leur loss : fournir des labels déjà décalés peut décaler deux fois l’objectif. Inspecte les labels réellement reçus, puis compte les cibles actives **après** ce décalage. L’`attention_mask` ne suffit pas à exclure le prompt de la supervision.

Le module 12 utilise un dataset prompt/completion et `completion_only_loss=True`. L’option `assistant_only_loss=True` exige un template et une version compatibles avec l’identification des segments assistant ; des noms de rôles visibles ne suffisent pas. Consulte le [contrat SFTTrainer](https://huggingface.co/docs/trl/sft_trainer).

### DPO, scores et référence

Le script du module 28 utilise l’API TRL observée, notamment la configuration de `loss_type`. Dans cette pile, l’exemple fournit une liste de types ; des tutoriels plus anciens peuvent employer une chaîne. Vérifie la signature et les valeurs acceptées dans ta version avant de recopier une configuration. La log-probabilité de réponse, sa moyenne par token et la marge relative à une référence sont trois mesures différentes. Voir [DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer).

### GRPO et PPO

La normalisation des rewards, la définition de la loss, le nombre de réutilisations et la pénalité KL dépendent de la variante. Le module 34 isole une optimisation relative par groupes ; le module 35 isole le mécanisme PPO sur un bandit. Ils ne se présentent pas comme des reproductions complètes des trainers TRL. Compare les formules aux descriptions du [GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) et du [PPOTrainer, branche main](https://huggingface.co/docs/trl/main/en/ppo_trainer) de la version utilisée.

Distingue toujours politique de collecte, politique mise à jour, référence et éventuel critic. Une pénalité KL n’est pas automatiquement la même quantité qu’un estimateur de dérive entre deux updates.

### LoRA et quantification

Les noms des modules, l’orientation des matrices et le scaling varient selon l’architecture et la configuration. L’usage de `fan_in_fan_out` avec les projections GPT-2 doit être cohérent avec leurs poids. Audite les paramètres entraînables et le rechargement au lieu de déduire la configuration des seuls noms. Voir [LoRA dans PEFT](https://huggingface.co/docs/peft/developer_guides/lora).

Une base 4 bits ne signifie pas activations, gradients et optimizer tous en 4 bits. Le backend, le matériel et les dtypes doivent être compatibles. Voir l’[intégration bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes) et la [compatibilité matérielle bitsandbytes](https://huggingface.co/docs/bitsandbytes/installation).

### DDP et FSDP

DDP synchronise les gradients ; il ne choisit pas à ta place une partition de données correcte. Les nombres de cibles et les dernières fenêtres d’accumulation déterminent la pondération. Voir le [tutoriel DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).

Le module 49 emploie **FSDP2**, via `torch.distributed.fsdp.fully_shard`. Un exemple FSDP1 avec wrapper ou configuration Accelerate ne se transpose pas par simple changement de nom. Voir le [tutoriel FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) et le [guide FSDP Accelerate](https://huggingface.co/docs/accelerate/usage_guides/fsdp).

### Tokenizer, perplexité et reprise

Un changement de tokenizer ou de template peut changer les IDs et frontières sans changer le texte apparent. Le cours utilise Tokenizers 0.23.2 ; vérifie que le [quick tour Tokenizers 0.23.2](https://huggingface.co/docs/tokenizers/v0.23.2/quicktour) consulté correspond à l’API installée. Sauvegarde les artefacts, pas seulement la taille du vocabulaire.

Pour la perplexité, fixe tokenizer, corpus, cibles et stratégie de contexte. Une moyenne arithmétique des perplexités de batches n’est généralement pas la perplexité du corpus. Voir le [guide Transformers sur la perplexité](https://huggingface.co/docs/transformers/perplexity).

Pour reprendre un run, conserve plus que les poids : optimizer, scheduler, états aléatoires et position dans les données selon le périmètre. Voir [sauvegarder et recharger avec PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html). Les modules 51 et 62 testent leur contrat de reprise dans une miniature précise ; ils ne prétendent pas restaurer un dataloader distribué arbitraire.

## 4. Documentation primaire par outil

| Outil ou contrat | Référence | À utiliser pour |
| --- | --- | --- |
| PyTorch | [Documentation](https://docs.pytorch.org/docs/stable/index.html) | Tenseurs, autograd, optimizer, distribution |
| Checkpointing d’activations | [API checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html) | Recalcul, RNG et variantes d’API |
| TensorBoard avec PyTorch | [SummaryWriter](https://docs.pytorch.org/docs/stable/tensorboard.html) | Événements, scalaires et fermeture du writer |
| Transformers | [Documentation](https://huggingface.co/docs/transformers/) | Modèles, génération et configuration |
| Datasets | [Documentation](https://huggingface.co/docs/datasets/) | Schémas, chargement et transformations |
| Tokenizers | [Documentation](https://huggingface.co/docs/tokenizers/) | Modèle BPE, alphabet et sérialisation |
| PEFT | [Documentation](https://huggingface.co/docs/peft/) | Injection, configuration et sauvegarde d’adapters |
| TRL | [Documentation](https://huggingface.co/docs/trl/) | SFT, préférences, reward modeling et RL |
| Accelerate | [Documentation](https://huggingface.co/docs/accelerate/) | Lancement, préparation et distribution |
| Python JSON | [Documentation `json`](https://docs.python.org/3/library/json.html) | Doublons de clés, constantes non standard et hooks du correcteur |

La documentation en ligne peut présenter un comportement plus récent. Les signatures installées et un test minimal départagent les exemples contradictoires. Si tu modifies une version, note-la avec le résultat plutôt que de garder l’étiquette d’une ancienne expérience.

## 5. Articles fondamentaux et questions de lecture

Lis les articles après les miniatures correspondantes. Recherche l’objectif, les hypothèses et le protocole expérimental avant les chiffres de performance.

| Article primaire | Pourquoi le lire | Modules |
| --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Attention, blocs et architecture Transformer ; le modèle du cours reste une architecture causale particulière | 4, 40 |
| [LoRA](https://arxiv.org/abs/2106.09685) | Paramétrisation basse-rang et protocole d’adaptation | 15–19, 56, 58 |
| [QLoRA](https://arxiv.org/abs/2305.14314) | Quantification et adaptation ; distinguer résultats de l’article et matériel local | 20, 57 |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | Démonstrations, préférences, reward model et optimisation | 12, 29, 35 |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | Dérivation de l’objectif relatif à la référence | 27, 28, 60 |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | Surrogate proximal, clipping et variantes | 35 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | Présentation du mécanisme GRPO dans son cadre expérimental | 34 |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Compromis paramètres/données/calcul et limites de transposition des lois de dimensionnement | 37, 40 |

Ces articles décrivent des systèmes et budgets qui dépassent les miniatures du cours. Une expérience pédagogique peut confirmer un calcul local sans reproduire leurs résultats à grande échelle.

## 6. Quand une API ne correspond pas

1. Relève l’interpréteur et les versions effectivement chargées.
2. Identifie le champ ou l’appel qui diffère, dans un script minimal.
3. Consulte la documentation de cette version et la signature locale.
4. Vérifie les entrées et sorties réelles : labels, paramètres, métriques et fichiers.
5. Note la modification et relance les contrôles concernés.

Si l’erreur est une absence d’artefact, reviens aux dépendances du [module 00](00_METHODE_ET_ENVIRONNEMENT.md). Si elle concerne le mécanisme, consulte l’[annexe de diagnostic](ANNEXE_DIAGNOSTIC_ENTRAINEMENT.md). Ne masque pas une incompatibilité en retirant un contrôle dont tu n’as pas compris le rôle.
