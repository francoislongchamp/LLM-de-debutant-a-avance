# Avant de commencer — méthode de travail et environnement

Tu vas apprendre à expliquer et à vérifier un entraînement. Exécuter un script est une étape ; comprendre ses cibles, ses gradients et les limites de ses résultats est le véritable objectif.

Ce module prépare la lecture des [62 modules](INDEX.md). Aucun GPU n’est nécessaire pour commencer.

## 1. Une séance en six gestes

1. **Formule la question.** Exemple : « Le padding participe-t-il à la loss ? »
2. **Dessine un cas minuscule.** Trois tokens et leurs labels suffisent souvent.
3. **Prédis le résultat.** Note ce qui devrait changer et ce qui devrait rester identique.
4. **Exécute le laboratoire complet.** Garde les sorties utiles et les messages d’erreur.
5. **Change une seule chose.** Compare à la version de départ avec la même mesure.
6. **Explique sans regarder la correction.** Réponds aux exercices avant d’ouvrir leurs solutions.

Lire une formule sans savoir nommer ses termes est un signal pour ralentir. Une loss qui descend sans labels inspectés est un résultat encore incomplet.

## 2. Reconnaître, expliquer, diagnostiquer

| Niveau | Ce que tu dois pouvoir faire | Exemple |
| --- | --- | --- |
| Reconnaître | Définir un terme | Un gradient décrit une variation locale de la loss |
| Expliquer | Relier deux mécanismes | L’optimizer utilise ces gradients pour modifier les paramètres |
| Diagnostiquer | Tester une hypothèse | Vérifier que les paramètres attendus reçoivent des gradients avant d’accuser le learning rate |

La checklist de chaque module sert à décider si tu peux continuer. Si une expérience échoue, utilise l’[annexe de diagnostic](ANNEXE_DIAGNOSTIC_ENTRAINEMENT.md) et conserve le cas qui reproduit l’erreur.

## 3. Comprendre le statut d’une expérience

**Miniature de mécanisme :** de vrais calculs et parfois de vraies updates, sur un problème volontairement réduit. Une politique choisissant entre deux procédures déjà écrites n’apprend pas à inventer ces procédures. Un petit GPT-2 aléatoire ne possède pas les connaissances d’une base préentraînée.

**Analyse de résultats fictifs :** des tableaux construits pour apprendre un calcul ou une décision. Ils ne mesurent aucune carte graphique ni qualité réelle de modèle. C’est notamment le cas des valeurs LoRA/QLoRA du module 57.

**Expérience sur une tâche réelle :** données représentatives, base identifiée, budget mesuré, validation indépendante et évaluation finale. Les projets donnent un protocole pour passer de la miniature à ce niveau ; leurs sorties locales ne remplacent pas ce travail.

Note toujours ce statut dans ton journal. « Le code s’exécute », « les poids changent », « la validation s’améliore » et « le produit est utile » sont quatre affirmations qui demandent des preuves différentes.

## 4. Préparer un dossier stable

Travaille dans un dossier personnel, par exemple `llm-learning/` :

```text
llm-learning/
├── .venv/
├── labs/          # scripts copiés depuis les modules
├── data/
│   ├── raw/       # données sources conservées
│   ├── processed/
│   └── eval/
├── outputs/       # artefacts produits par les laboratoires
├── experiments/   # configurations et résultats à comparer
└── notes/         # prédictions, observations, erreurs et explications
```

Lance tous les laboratoires depuis cette racine. `outputs/module12/final` sera ainsi au même endroit pour le module 12 et les projets qui le rechargent. Le dossier contenant le script ne décide pas de la résolution des chemins relatifs.

Certains scripts remplacent leurs anciennes sorties pédagogiques. Archive les runs que tu veux conserver sous un nom distinct. Garde tes données sources séparées des fichiers générés.

## 5. Créer l’environnement

Exemple Linux/macOS, depuis le dossier de travail choisi :

```bash
mkdir -p labs data/raw data/processed data/eval outputs experiments notes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Sous PowerShell, l’activation habituelle est `.venv\Scripts\Activate.ps1`. Dans tous les cas, vérifie que ton éditeur et ton terminal utilisent le même interpréteur. Si un environnement est déjà prêt, inspecte-le avant d’en installer un autre.

Les versions observées pendant la vérification du cours sont données dans [Sources et API](SOURCES_ET_API.md). Elles constituent une référence de compatibilité testée dans cet environnement, pas une garantie qu’une installation neuve sur tout système reproduise la pile.

### Installer progressivement

| Étape | Bibliothèques concernées | Pourquoi |
| --- | --- | --- |
| Tenseurs et gradients | PyTorch | Calculs des premiers laboratoires |
| Tokenizer et modèle causal | Transformers, Tokenizers | Modèles locaux et tokenisation |
| Données et trainers | Datasets, Accelerate, TRL | Laboratoires SFT/DPO et traitement des données |
| Adapters | PEFT | LoRA et audit des paramètres |
| Journalisation | TensorBoard | Module 52 |
| Quantification du laboratoire GPU | bitsandbytes et pile compatible | Module 20 et expérience réelle du projet 57 |

Choisis la commande PyTorch adaptée à ton système depuis la [page officielle](https://pytorch.org/get-started/locally/). Une roue CPU suffit pour la majorité des miniatures. Une installation CUDA doit correspondre à une pile réellement compatible avec le pilote et le matériel disponibles.

Pour ajouter les bibliothèques Python après PyTorch, un exemple de commande est :

```bash
python -m pip install transformers tokenizers datasets accelerate trl peft tensorboard
```

Cette commande résout les versions disponibles au moment de son exécution. Compare le résultat à la pile de référence, puis valide les premiers laboratoires avant un long run. Sauvegarde l’environnement obtenu :

```bash
python -m pip freeze > experiments/requirements-observed.txt
```

Crée d’abord le dossier `experiments` si tu n’as pas suivi l’arborescence. L’installation de bitsandbytes est réservée à l’étape qui en a besoin ; les exemples CPU n’exigent pas ce package.

### Inspecter l’interpréteur et le matériel

```python
import sys
from importlib.metadata import version, PackageNotFoundError
import torch

print('Python :', sys.version.split()[0])
print('Executable :', sys.executable)
for package in ['torch', 'transformers', 'tokenizers', 'datasets',
                'accelerate', 'trl', 'peft', 'tensorboard', 'bitsandbytes']:
    try:
        print(package, version(package))
    except PackageNotFoundError:
        print(package, 'non installe')
print('CUDA accessible :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU :', torch.cuda.get_device_name(0))
    print('CUDA de la roue PyTorch :', torch.version.cuda)
```

L’absence de CUDA n’empêche pas d’apprendre. DDP peut s’exécuter sur deux processus CPU au module 48. Le véritable laboratoire FSDP du module 49 demande deux GPU CUDA ; sa simulation de partage reste exécutable sur CPU. Une branche conditionnelle ignorée ne compte pas comme un test GPU réussi.

## 6. Copier et exécuter les blocs

Copie **le bloc Python complet** dans un fichier, par exemple `labs/12_sft.py`, puis lance depuis la racine :

```bash
python labs/12_sft.py
```

Les blocs d’un même module peuvent se suivre ou être indépendants : lis leur introduction. Les tableaux de résultats, commandes de lancement et fragments conceptuels ne sont pas tous des scripts autonomes.

Les miniatures locales créent leurs données et leurs poids sans télécharger de gros modèle. Les exemples utilisant une base préentraînée peuvent demander un accès réseau initial, un cache, davantage de RAM et l’acceptation des conditions de la source. Le module concerné précise ce qu’il utilise.

### Trois dépendances d’artefacts à respecter

| Projet | Exécuter d’abord | Fichiers attendus dans la même racine |
| --- | --- | --- |
| 56 — LoRA spécialisé | [12](12_premier_supervised_fine_tuning.md) | `outputs/module12/final/` |
| 59 — CPT puis SFT | [12](12_premier_supervised_fine_tuning.md) | `outputs/module12/final/` |
| 60 — Audit DPO | [28](28_entrainer_avec_dpotrainer.md) | `outputs/module28/final/` et `outputs/module28/reference/` |

Un chemin absent se corrige en produisant l’artefact prévu, pas en remplaçant silencieusement le modèle par une autre base.

## 7. Tenir un journal proportionné

Dès le premier laboratoire, note la question, la prédiction, la configuration modifiée, le résultat observé et ton explication. Pour un entraînement à comparer, ajoute progressivement :

| Rubrique | Informations utiles |
| --- | --- |
| Identité | Nom du run, statut miniature/fictif/réel, date, script et versions |
| Données | Source, droits d’usage, version/hash, groupes train/validation/test, nettoyage |
| Modèle | Configuration ou identifiant et révision, tokenizer et template, paramètres totaux/entraînables |
| Objectif | Labels, masques, nombre réel de cibles après shift, définition du reward ou des préférences |
| Optimisation | Seed, optimizer, learning rate, scheduler, dtype, clipping, accumulation et nombre de processus |
| Budget réalisé | Updates, exemples et positions vus, cibles actives, temps, répétitions |
| Évaluation | Baseline, protocole et dénominateurs, scores par catégorie, dispersion et cas d’échec |
| Ressources | Matériel, pic mémoire mesuré, unité et périmètre, coût des sauvegardes/évaluations |
| Artefact | Chemin, hash, procédure de rechargement et test de reprise effectué |

Une mesure absente doit rester « non mesurée », et non devenir zéro. Indique si un temps comprend l’initialisation, la génération ou l’évaluation. Conserve les erreurs qui expliquent un échec, sans exporter de données sensibles dans des logs partagés.

## 8. Les portes avant d’augmenter le budget

1. Inspecte un exemple brut, ses IDs, son masque et ses labels après préparation.
2. Fais un forward : loss finie et au moins une cible active après le décalage causal.
3. Fais un backward : gradients attendus présents et finis ; paramètres gelés correctement exclus.
4. Surapprends volontairement un petit batch pour tester la mécanique, puis reviens à une validation indépendante.
5. Vérifie ce que contient réellement un fichier sauvegardé et recharge-le.
6. Mesure une baseline, puis un changement avec un protocole fixé.

Le décalage des labels est souvent réalisé à l’intérieur du modèle causal : vérifie l’API avant de le faire toi-même. Une moyenne de losses de batches de tailles différentes peut fausser la métrique. Ces deux points reviendront souvent.

## 9. Ce que veut dire « reproductible »

**Recharger un artefact** signifie retrouver un modèle utilisable. **Reprendre un entraînement** exige aussi optimizer, scheduler, compteurs et états aléatoires pertinents, ainsi que la position dans les données. **Retrouver bit pour bit une trajectoire** exige des conditions plus strictes encore.

Une seed seule ne garantit pas l’égalité entre matériels, versions, processus ou kernels. Les modules 51 et 62 testent une égalité exacte dans un périmètre CPU défini. Pour un projet plus large, annonce ce qui est reproduit : artefact, trajectoire exacte ou distribution de résultats sur plusieurs seeds.

## 10. Première vérification personnelle

Avant le module 1, réponds avec tes mots :

- Depuis quel dossier ton script sera-t-il lancé et où écrira-t-il ses sorties ?
- Comment sauras-tu si une valeur est fictive, mesurée ou attendue ?
- Quelle différence fais-tu entre vérifier un mécanisme et démontrer une qualité d’assistant ?
- Quel petit test lanceras-tu avant une expérience coûteuse ?

Garde ces réponses dans tes notes. Puis ouvre le [module 1](01_pipeline_complet_dun_llm.md). Le [glossaire](GLOSSAIRE.md) sert de rappel, et les [sources](SOURCES_ET_API.md) permettent de vérifier le contrat des outils.
