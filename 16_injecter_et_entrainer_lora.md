# Module 16 — Injecter et entraîner LoRA

## Pourquoi ce module arrive ici

L’intuition mathématique de LoRA doit maintenant devenir observable dans un vrai modèle : où les modules sont injectés, quels paramètres restent gelés et ce qui est sauvegardé.

## Objectifs

Créer une `LoraConfig`, l’appliquer via PEFT/TRL, inspecter les noms de paramètres LoRA, lancer un petit SFT et sauvegarder l’adapter séparément.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning léger

## Définitions concrètes

### target_modules

**Définition concrète.** Modules linéaires dans lesquels LoRA est injecté.


**Exemple simple.** `q_proj`, `v_proj` ou `all-linear` selon stratégie/architecture.

### lora_dropout

**Définition concrète.** Dropout appliqué dans la branche LoRA pendant training.


**Exemple simple.** 0.05 signifie 5 % selon le mécanisme de dropout.

### modules_to_save

**Définition concrète.** Modules supplémentaires conservés entraînables/sauvegardés avec l’adapter.


**Exemple simple.** Utile dans certains scénarios pour une tête spécifique.

### PEFT model

**Définition concrète.** Wrapper/modèle contenant la base et les adapters.


**Exemple simple.** Expose `print_trainable_parameters()`.

### Adapter checkpoint

**Définition concrète.** Fichiers contenant principalement la configuration et les poids PEFT, pas nécessairement la base complète.


**Exemple simple.** Il faut connaître le modèle de base compatible.

## Intuition simple

L’injection LoRA ne “réduit” pas physiquement la matrice de base. Elle ajoute une petite branche parallèle à certaines projections. La sortie devient contribution de base + contribution LoRA.

## Ce qui se passe réellement sous le capot

1. Charger le modèle de base.
2. Créer `LoraConfig` avec rank, alpha, dropout et cibles.
3. PEFT remplace/wrappe les modules ciblés pour ajouter A/B.
4. Les poids de base sont gelés selon le setup.
5. Le trainer construit l’optimizer à partir des paramètres entraînables.
6. Le SFT calcule la même loss causale.
7. Seuls les paramètres LoRA reçoivent/emploient des gradients pour la mise à jour.
8. L’adapter est sauvegardé et peut être rechargé sur la base correspondante.

## Exemple minimal à comprendre mentalement

Si le modèle contient :

```text
layer.0.self_attn.q_proj.weight
```

après injection, on peut voir des paramètres similaires à :

```text
...q_proj.lora_A...
...q_proj.lora_B...
```

Le gros `q_proj.weight` de base reste présent; les petites matrices supplémentaires sont les paramètres d’adaptation.

## Code minimal observable

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

train = load_dataset("json", data_files="data/train.jsonl", split="train")

peft = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="outputs/lora",
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    assistant_only_loss=True,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=args,
    train_dataset=train,
    peft_config=peft,
)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model("outputs/lora/final")
```

## Laboratoire guidé

1. Lance uniquement la construction du trainer et imprime les paramètres entraînables avant de trainer.  
2. Liste tous les noms contenant `lora_`.  
3. Choisis un poids de base et confirme `requires_grad=False`.  
4. Choisis A/B et confirme `requires_grad=True`.  
5. Fais un petit training.  
6. Compare la taille du dossier adapter à celle du modèle de base.  
7. Recharge l’adapter sur la même base et vérifie une génération.

## Ce que tu dois observer

- Le pourcentage de paramètres entraînables chute fortement.
- Le checkpoint adapter est beaucoup plus petit qu’un checkpoint complet.
- Le modèle de base reste nécessaire pour utiliser un adapter séparé.
- Un LR adapté aux adapters peut être plus élevé que celui d’un full FT; ce n’est pas une règle absolue.

## À ne pas confondre

- `all-linear` ≠ tous les paramètres du modèle.
- Sauvegarder l’adapter ≠ sauvegarder forcément la base.
- `requires_grad=False` ≠ poids non utilisé dans le forward.

## Erreurs fréquentes

- Cibler des noms de modules qui n’existent pas dans l’architecture.
- Oublier de vérifier le nombre de paramètres entraînables.
- Comparer LoRA et full FT avec des hyperparamètres copiés sans adaptation.

## Exercices

1. Explique ce qui se passe dans le forward d’un module LoRA.
2. Trouve le pourcentage exact de paramètres entraînables.
3. Explique pourquoi le modèle de base peut être partagé par plusieurs adapters.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Injecter LoRA et identifier A/B.
- Vérifier paramètres gelés vs entraînables.
- Sauvegarder/recharger un adapter.
- Expliquer pourquoi le checkpoint PEFT est petit.

## Fiche mémo

LoRA devient concret lorsque tu peux montrer **les paramètres A/B**, leur `requires_grad`, leur nombre et leur checkpoint.

## Lien avec le module suivant

Le module 17 mesure précisément les paramètres entraînables et le budget qu’ils représentent.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
