# Apprendre à entraîner et à évaluer un LLM

Ce cours en français comprend **62 modules progressifs**, un module 00 pour préparer le travail, deux annexes et un glossaire. Il part du trajet d’un texte dans un modèle et va jusqu’aux expériences de SFT, LoRA, QLoRA, full fine-tuning, CPT, DPO, RL et pré-entraînement.

**Commence par la [méthode et l’environnement](00_METHODE_ET_ENVIRONNEMENT.md), puis suis l’[index des 62 modules](INDEX.md).**

## Comment apprendre avec ce cours

Chaque module construit une intuition, détaille le mécanisme, propose des observations ou un laboratoire, puis des exercices corrigés et une checklist. Avant d’exécuter un bloc, prédis sa sortie. Après l’exécution, explique ce qui a changé, quelle mesure le prouve et ce que l’expérience ne permet pas de conclure.

Les corrections sont repliées pour te laisser chercher. Les temps indiqués sont des repères de travail, auxquels s’ajoutent les reprises et les expériences personnelles.

## Ce que les laboratoires démontrent

La plupart des laboratoires utilisent de petits tenseurs, un tokenizer local ou un Transformer initialisé aléatoirement. Ils permettent de vérifier les labels, gradients, masques, objectifs, sauvegardes et mesures sur CPU. Une loss qui baisse sur quelques exemples ne transforme pas ces modèles en assistants utiles.

Certains modules analysent des **données fictives explicitement annoncées** : une comparaison de scores ou de mémoire dans ces tableaux ne constitue pas un résultat matériel mesuré. Les projets proposent ensuite un protocole pour mener une expérience sur ses propres données.

Les laboratoires QLoRA réels et FSDP sur plusieurs GPU demandent un environnement adapté. Le module 49 fournit aussi une simulation CPU du partage des états. Le module 48 permet une véritable exécution DDP avec deux processus CPU. Les versions et la portée des vérifications figurent dans [Sources et API](SOURCES_ET_API.md).

## Parcours

| Modules | Compétence à construire |
| --- | --- |
| 1–8 | Suivre les tokens, la loss, les gradients et une update |
| 9–14 | Construire les données, faire un SFT et évaluer sans contamination |
| 15–24 | Comprendre LoRA, QLoRA, full FT et les compromis mémoire |
| 25–36 | Choisir un objectif et auditer préférences, rewards et RL |
| 37–47 | Construire tokenizer, architecture et pipeline de pré-entraînement |
| 48–55 | Distribuer, reprendre, observer et comparer les expériences |
| 56–62 | Livrer des projets avec artefacts, contrôles et limites explicites |

## Exécution et dépendances entre projets

Crée un dossier de travail et lance les scripts depuis **le même dossier racine** : les chemins `outputs/...` sont relatifs au répertoire courant, pas au fichier Markdown. Le [module 00](00_METHODE_ET_ENVIRONNEMENT.md) explique l’installation progressive et cette convention.

- Les projets **56 et 59** rechargent `outputs/module12/final` : exécute d’abord le laboratoire du module 12 dans cette même racine.
- Le projet **60** recharge `outputs/module28/final` et `outputs/module28/reference` : exécute d’abord le module 28.
- Les autres miniatures de projet précisent leurs propres données et artefacts. Le projet 62 reconstruit sa chaîne locale, du tokenizer à la génération, et teste une reprise intermédiaire.

Conserve une copie des résultats que tu veux comparer avant de relancer un script qui réécrit son dossier de sortie.

## Outils pour avancer

- [Glossaire](GLOSSAIRE.md) : retrouver une définition et son module.
- [Choisir une technique](ANNEXE_CHOISIR_LA_BONNE_TECHNIQUE.md) : relier une intervention à un problème observé.
- [Diagnostiquer un entraînement](ANNEXE_DIAGNOSTIC_ENTRAINEMENT.md) : transformer un symptôme en hypothèses testables.
- [Sources et API](SOURCES_ET_API.md) : documentation officielle, articles, versions observées et limites de validation.

Ton résultat final doit comprendre une expérience explicable : des données identifiées, un objectif vérifié, des mesures comparables et un artefact que tu sais recharger.
