# Avant de commencer — méthode de travail et environnement

Ce cours est conçu pour être **compris**, pas seulement exécuté. Le but n'est pas de mémoriser des appels à `Trainer`, PEFT ou TRL : tu dois pouvoir expliquer ce que fait l'outil et reconnaître quand il fait quelque chose d'inadapté à ton expérience.

## 1. La règle pédagogique du cours

Pour chaque notion, suis toujours le même ordre :

```text
Définition concrète
→ intuition
→ micro-exemple
→ mécanisme réel
→ formule utile
→ code minimal
→ laboratoire
→ observation
→ explication avec tes propres mots
```

Ne saute pas directement au laboratoire. Si le vocabulaire n'est pas clair, le code donnera l'illusion de comprendre.

## 2. Les trois niveaux de compréhension

### Niveau A — reconnaître

Tu peux dire ce qu'est le concept.

Exemple : « un gradient mesure comment la loss varie localement quand un paramètre change ».

### Niveau B — expliquer

Tu peux expliquer pourquoi il existe et comment il intervient dans le pipeline.

Exemple : « l'optimizer utilise les gradients pour décider comment modifier les poids ».

### Niveau C — diagnostiquer

Tu peux reconnaître les conséquences d'un mauvais réglage.

Exemple : « si la loss devient NaN après une hausse brutale du gradient norm, je soupçonne une instabilité numérique ou un learning rate trop élevé ».

L'objectif du cours est d'atteindre progressivement le niveau C.

## 3. Journal d'expérience obligatoire

Pour chaque entraînement, conserve au minimum :

```yaml
experiment: nom-court
model: nom-ou-config
seed: 42
dataset_version: v1
train_examples: 0
validation_examples: 0
max_length: 0
precision: bf16
learning_rate: 0.0
batch_per_device: 0
gradient_accumulation: 0
epochs_or_steps: 0
trainable_parameters: 0
peak_vram_gb: 0
training_time: 0
final_train_loss: null
final_validation_loss: null
benchmark_score: null
notes: ""
```

Sans journal, tu peux obtenir un résultat. Avec un journal, tu peux **expliquer et reproduire** le résultat.

## 4. Arborescence recommandée

```text
llm-learning/
├── .venv/
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
├── labs/
├── configs/
├── outputs/
├── checkpoints/
├── logs/
├── experiments/
└── notes/
```

## 5. Environnement Python

Exemple avec `venv` :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Installe ensuite les bibliothèques nécessaires **au fur et à mesure du cours** plutôt que d'empiler des dépendances inutiles.

Pile principale :

```text
PyTorch
Transformers
Datasets
Tokenizers
PEFT
TRL
Accelerate
TensorBoard
```

`bitsandbytes` est surtout introduit lors de QLoRA/quantification et dépend de la plateforme.

## 6. Vérifier PyTorch et le GPU

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime PyTorch:", torch.version.cuda)
```

Important : le pilote NVIDIA, la version CUDA prise en charge par ce pilote et la build PyTorch doivent être compatibles. Installer simplement le Toolkit CUDA le plus récent ne corrige pas forcément une incompatibilité de pilote.

## 7. Commencer petit

Avant une expérience coûteuse :

```text
1 batch
→ 10 steps
→ 100 steps
→ petit subset
→ expérience complète
```

Et avant un modèle de 7B paramètres :

```text
petit modèle
→ pipeline validé
→ modèle cible
```

Cela transforme les erreurs de plusieurs heures en erreurs de quelques secondes ou minutes.

## 8. Le test « overfit one batch »

Avant un pré-entraînement ou une boucle custom, essaie volontairement de faire mémoriser **un seul petit batch**.

Si la loss ne peut pas descendre fortement sur un batch fixe, il existe probablement un problème dans :

- les labels ;
- le masking ;
- le forward ;
- le backward ;
- l'optimizer ;
- le learning rate ;
- la précision numérique.

Ce test ne prouve pas que le modèle généralisera. Il vérifie que la plomberie d'apprentissage fonctionne.

## 9. Quand passer au module suivant

Ne te base pas sur le fait que le script « fonctionne ». Passe au module suivant lorsque tu peux :

1. définir les termes sans regarder le texte ;
2. refaire le micro-exemple ;
3. prédire qualitativement le résultat d'un changement ;
4. exécuter le laboratoire ;
5. expliquer ce que tu as observé ;
6. répondre aux critères de validation.

## 10. Comment utiliser une IA pendant ce cours

Utilise l'IA comme professeur ou relecteur, pas comme bouton « produire le script final ».

Questions utiles :

- « Fais-moi calculer ce tenseur à la main. »
- « Pose-moi cinq questions pour vérifier que je comprends la cross-entropy. »
- « Voici ma courbe de loss; aide-moi à formuler des hypothèses. »
- « Ne me donne pas la réponse immédiatement : guide-moi dans le diagnostic. »

Le meilleur indicateur de progression reste ta capacité à expliquer le mécanisme sans l'outil.
