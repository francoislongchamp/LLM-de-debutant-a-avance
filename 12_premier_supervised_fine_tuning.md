# Module 12 — Premier Supervised Fine-Tuning

## Pourquoi ce module arrive ici

Le SFT est la première vraie modification d’un LLM du parcours. Il apprend au modèle à augmenter la probabilité des réponses fournies dans un dataset supervisé.

## Objectifs

Comprendre ce que SFT signifie token par token, utiliser un dataset conversationnel, savoir quelles positions contribuent à la loss, configurer un entraînement minimal et sauvegarder le résultat.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning — SFT

## Définitions concrètes

### SFT

**Définition concrète.** Supervised Fine-Tuning : entraînement sur des entrées avec sorties cibles connues.


**Exemple simple.** Question utilisateur + réponse assistant de référence.

### Instruction tuning

**Définition concrète.** SFT orienté vers la capacité à suivre des instructions variées.


**Exemple simple.** “Résume…”, “explique…”, “classe…”.

### Chat template

**Définition concrète.** Règle transformant des messages structurés en tokens attendus par le modèle.


**Exemple simple.** Rôles system/user/assistant deviennent des marqueurs spécifiques.

### Assistant-only loss

**Définition concrète.** Loss calculée uniquement sur les tokens de réponse assistant lorsque le template/dataset le permet.


**Exemple simple.** Les tokens user restent contexte mais ne sont pas des cibles de loss.

### Packing

**Définition concrète.** Regroupement de plusieurs petits exemples dans des séquences plus pleines pour réduire le padding.


**Exemple simple.** Trois conversations courtes peuvent partager un bloc de 1024 positions selon la stratégie.

## Intuition simple

Pendant un SFT, le prompt sert de contexte et la réponse de référence sert de démonstration. Le modèle n’apprend pas une règle symbolique “réponds bien”; il reçoit des gradients qui rendent les tokens de la réponse cible plus probables dans ce contexte.

## Ce qui se passe réellement sous le capot

1. Le dataset conversationnel est formaté avec le chat template.
2. Le tokenizer produit input IDs.
3. Les labels sont construits; selon la configuration, les tokens du prompt peuvent être masqués avec `-100`.
4. Le modèle calcule logits et cross-entropy.
5. Backward propage les gradients.
6. L’optimizer modifie les paramètres entraînables.
7. La validation mesure la loss et/ou des générations sur des exemples non entraînés.

## Exemple minimal à comprendre mentalement

Conversation :

```text
user: 2+2 ?
assistant: 4
```

Si on entraîne uniquement sur la réponse assistant, les tokens du user restent visibles dans `input_ids`, mais leurs labels deviennent ignorés :

```text
input  : [USER, 2, +, 2, ?, ASSISTANT, 4, EOS]
labels : [-100,-100,-100,-100,-100,-100, 4, EOS]
```

Le modèle utilise donc la question comme contexte tout en étant pénalisé seulement sur la réponse.

## Formules et notation utiles

La loss SFT reste généralement une cross-entropy causale :

\[
L=-\frac{1}{N_{cibles}}\sum_{t\in cibles}\log p_\theta(y_t|x,y_{<t})
\]

Le choix de `cibles` dépend du masking : séquence complète, completion seulement, assistant seulement, etc.

## Code minimal observable

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

train = load_dataset("json", data_files="data/train.jsonl", split="train")

args = SFTConfig(
    output_dir="outputs/sft",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    logging_steps=5,
    max_length=1024,
    assistant_only_loss=True,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=args,
    train_dataset=train,
)
trainer.train()
trainer.save_model("outputs/sft/final")
```

## Laboratoire guidé

1. Prépare 50–200 exemples pédagogiques propres.  
2. Avant training, génère 10 réponses baseline et sauvegarde-les.  
3. Lance un seul epoch sur un petit modèle.  
4. Sauvegarde logs et configuration.  
5. Régénère exactement les 10 prompts.  
6. Inspecte les labels d’un exemple tokenisé et vérifie quelles positions sont `-100`.  
7. Recommence avec full-sequence loss uniquement pour comprendre la différence, pas pour supposer qu’elle est meilleure.

## Ce que tu dois observer

- La loss peut baisser très vite sur un petit dataset.
- Une réponse mémorisée sur un prompt train ne prouve pas la généralisation.
- Le masking des labels change ce que le modèle est explicitement entraîné à prédire.
- Packing est une optimisation d’efficacité, pas une nouvelle fonction objective.

## À ne pas confondre

- SFT ≠ pré-entraînement : même loss possible, mais données et objectif pédagogique différents.
- Contexte visible ≠ tokens forcément inclus dans la loss.
- Un modèle instruct déjà post-entraîné ≠ base model.

## Erreurs fréquentes

- Fine-tuner sans baseline.
- Utiliser un chat template incorrect.
- Tronquer la réponse sans le remarquer.
- Mettre un learning rate très élevé sur un full FT juste parce qu’il fonctionnait pour LoRA.

## Exercices

1. Dessine input IDs et labels pour une conversation avec system/user/assistant.
2. Explique assistant-only loss avec tes mots.
3. Explique ce que packing change et ne change pas.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer ce que le SFT optimise token par token.
- Inspecter un exemple après chat template/tokenisation.
- Dire quels tokens participent à la loss.
- Lancer un SFT minimal et comparer avant/après sans confondre mémorisation et généralisation.

## Fiche mémo

SFT = augmenter la probabilité de sorties supervisées dans leurs contextes. Le point clé n’est pas la commande du trainer, mais **quels tokens deviennent des cibles**.

## Lien avec le module suivant

Le module 13 formalise la baseline et l’évaluation avant/après pour éviter de conclure à partir de quelques belles réponses.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
