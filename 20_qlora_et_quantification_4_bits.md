# Module 20 — QLoRA et quantification 4 bits

## Pourquoi ce module arrive ici

LoRA réduit les paramètres entraînables, mais les poids de base doivent toujours être chargés. QLoRA réduit aussi la mémoire de stockage de cette base en la chargeant quantifiée, tout en entraînant des adapters.

## Objectifs

Comprendre poids quantifiés, échelle/zéro au niveau conceptuel, NF4, compute dtype, double quantization et `prepare_model_for_kbit_training`. Réaliser un chargement 4-bit + LoRA et mesurer l’économie de mémoire.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning léger

## Définitions concrètes

### Quantification

**Définition concrète.** Approximation de valeurs continues avec un ensemble plus petit de niveaux représentables.


**Exemple simple.** Des poids 16-bit peuvent être représentés en 4-bit avec métadonnées d’échelle.

### 4-bit base

**Définition concrète.** Poids principaux stockés sous forme quantifiée pour réduire la mémoire.


**Exemple simple.** La base n’est pas entraînée directement comme une matrice FP32 ordinaire.

### NF4

**Définition concrète.** Format 4-bit conçu pour représenter efficacement des poids suivant une distribution proche de normale dans le contexte QLoRA.


**Exemple simple.** Disponible avec bitsandbytes.

### Compute dtype

**Définition concrète.** Précision utilisée pour certains calculs après déquantification/pendant opérations.


**Exemple simple.** BF16 peut être utilisé même si le stockage de base est 4-bit.

### Double quantization

**Définition concrète.** Quantification supplémentaire de certaines constantes de quantification.


**Exemple simple.** Réduit encore le stockage avec un compromis de complexité.

### QLoRA

**Définition concrète.** Entraînement LoRA sur un modèle de base chargé en faible précision quantifiée.


**Exemple simple.** Base 4-bit gelée + adapters entraînables.

## Intuition simple

Pense à un atlas lourd que tu compresses pour le garder en mémoire, puis tu ajoutes quelques feuilles de notes modifiables. QLoRA compresse la base et apprend surtout les “notes” LoRA. Les calculs utiles ne sont pas pour autant de simples multiplications naïves 4-bit de bout en bout.

## Ce qui se passe réellement sous le capot

1. Créer `BitsAndBytesConfig` 4-bit.
2. Charger le modèle avec cette configuration.
3. Préparer le modèle pour k-bit training.
4. Créer LoRA, souvent avec une couverture large des couches linéaires selon la stratégie.
5. Construire l’optimizer uniquement sur les adapters.
6. Pendant forward, les poids quantifiés sont utilisés via kernels/déquantification appropriés.
7. Backward met à jour les adapters, pas les poids de base quantifiés.

## Exemple minimal à comprendre mentalement

Ordre de grandeur brut pour une base 7B :

```text
BF16 weights : ~14 GB
4-bit raw    : ~3.5 GB
```

En pratique il faut ajouter métadonnées, buffers, adapters, activations et runtime. Le ratio réel de VRAM n’est donc pas exactement 4×.

## Formules et notation utiles

4 bits = 0,5 octet brut par valeur :

\[
7B\times0.5 \approx 3.5GB
\]

Mais ce calcul ne comprend pas les échelles de quantification ni les autres composantes du training.

## Code minimal observable

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    quantization_config=bnb,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()
```

## Laboratoire guidé

1. Charge la base en précision normale et mesure la mémoire après chargement.  
2. Libère-la proprement, puis charge en 4-bit et mesure.  
3. Ajoute LoRA et mesure à nouveau.  
4. Vérifie quels paramètres sont entraînables.  
5. Entraîne quelques steps et confirme que la loss diminue.  
6. Compare LoRA normal et QLoRA sur le même mini-benchmark.

## Ce que tu dois observer

- Le principal gain vient du stockage de la base.
- Les adapters restent petits et entraînables.
- Compute dtype et storage dtype sont des concepts différents.
- L’écosystème GPU/driver/bitsandbytes peut limiter les configurations disponibles.

## À ne pas confondre

- QLoRA ≠ “entraîner les poids 4-bit”.
- 4-bit storage ≠ tous les calculs 4-bit.
- Quantification ≠ LoRA.
- NF4 ≠ FP4.

## Erreurs fréquentes

- Supposer que 4-bit donne exactement 4× moins de VRAM totale.
- Oublier `prepare_model_for_kbit_training` dans un workflow qui le requiert.
- Comparer qualité LoRA/QLoRA avec des autres réglages différents.

## Exercices

1. Calcule la mémoire brute 4-bit d’une base 13B.
2. Explique storage dtype vs compute dtype.
3. Explique pourquoi la base quantifiée reste gelée dans QLoRA classique.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir QLoRA en une phrase exacte.
- Expliquer NF4 et compute dtype au niveau conceptuel.
- Charger 4-bit + adapters et vérifier trainable params.
- Mesurer l’économie réelle au lieu de la supposer.

## Fiche mémo

QLoRA = **base quantifiée pour la mémoire + LoRA pour l’apprentissage**.

## Lien avec le module suivant

Le module 21 retire les adapters et revient au cas où tous les poids sont modifiables : le full fine-tuning.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
