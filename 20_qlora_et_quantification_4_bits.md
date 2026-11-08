# Module 20 — Comprendre QLoRA et distinguer stockage et calcul

## 1. Ce que tu vas apprendre

LoRA réduit la partie entraînable, mais la base reste en mémoire. **QLoRA** combine une base quantifiée avec des adapters entraînables pour réduire aussi le stockage de cette base.

Tu vas quantifier quelques nombres à la main, distinguer NF4 et une quantification uniforme, puis lire un chargement 4 bits avec PEFT.

**Prérequis :** modules 5 et 15 à 19.

**Temps indicatif :** deux séances de 60 à 90 minutes.

**Matériel :** PyTorch sur CPU pour le laboratoire A. Le B utilise Transformers, PEFT, bitsandbytes et Accelerate sur un environnement matériel compatible ; le script proposé cible CUDA.

## 2. Quantifier signifie approximer

Une représentation à 4 bits permet de coder **16 niveaux**. Pour représenter des poids variés, il faut aussi des règles qui associent ces codes à des valeurs, souvent avec des échelles par groupe.

Exemple uniforme pédagogique : choisir les 16 niveaux `0/15, 1/15, ..., 15/15` pour des poids entre 0 et 1.

Un poids 0.20 donne le code 3 puis une reconstruction 0.20. Un poids 0.31 donne le code 5 puis environ 0.3333. La quantification introduit une erreur.

Ce mécanisme uniforme n’est **pas NF4**. Il sert à rendre visibles codes, reconstruction et échelle avant d’utiliser un format plus spécialisé.

## 3. Laboratoire A — Codes et reconstruction

Ce script autonome utilise seulement PyTorch, sans bitsandbytes.

```python
import torch

weights = torch.tensor([0.0, 0.20, 0.31, 0.80, 1.0])
levels = 15
codes = torch.round(weights * levels).clamp(0, levels).to(torch.uint8)
reconstructed = codes.float() / levels
print('Codes :', codes.tolist())
print('Reconstruits :', reconstructed.tolist())
print('Erreurs absolues :', (weights - reconstructed).abs().tolist())
print('Erreur maximale :', (weights - reconstructed).abs().max().item())
print('Octets reellement utilises par nos codes :', codes.numel() * codes.element_size())
print('Budget ideal a 4 bits :', weights.numel() * 0.5, 'octets avant arrondi et metadonnees')
```

Tu dois obtenir les codes **`[0,3,5,12,15]`** et une erreur maximale proche de **0.02333**.

Nos codes sont stockés dans un tenseur `uint8`, donc **un octet par code**, malgré leur plage 0 à 15. Ils ne sont pas compactés deux par octet. Pour cinq valeurs, un empaquetage en 4 bits demanderait au moins trois octets, plus les métadonnées nécessaires : `5×0.5` est un budget idéal, pas une allocation fractionnaire réellement possible.

## 4. Ce qu’ajoute QLoRA

| Terme | Sens concret |
| --- | --- |
| Base 4 bits | Certains poids stockés sous une représentation quantifiée |
| NF4 | Codebook non uniforme adapté à la distribution visée par QLoRA, notamment des poids proches d’une distribution normale |
| Compute dtype | Format utilisé pour les calculs concernés, par exemple BF16 ou FP16 |
| Double quantization | Quantification de certaines constantes de quantification pour réduire leur surcoût |
| Adapter LoRA | Petites matrices entraînables ajoutées aux projections ciblées |

La base classique QLoRA reste gelée. Les gradients servent à entraîner les adapters à travers le calcul qui utilise la base quantifiée. Les poids principaux ne deviennent pas une grande matrice 4 bits directement optimisée comme des FP32 ordinaires.

NF4 n’est pas FP4 et ne correspond pas aux 16 niveaux uniformes du laboratoire. Les kernels et les conversions permettent le calcul : tous les tenseurs ne sont pas en 4 bits.

## 5. Lire le budget de mémoire

Pour une base de 7 milliards de paramètres :

```text
Poids BF16 ideaux : 7e9 × 2   = 14 GB
Poids 4 bits bruts: 7e9 × 0.5 = 3.5 GB
```

Il faut ajouter échelles, métadonnées, parties non quantifiées, adapters, gradients/optimizer des adapters, activations, buffers et temporaires.

Un rapport 4× sur les poids idéaux n’est pas une promesse de rapport 4× sur le pic mémoire du training. La qualité peut aussi varier à cause de l’approximation de la base.

## 6. Laboratoire B — Charger une base 4 bits et faire un step LoRA

Ce bloc est autonome mais dépend d’un environnement CUDA/bitsandbytes compatible. Le premier chargement peut télécharger Qwen3-0.6B et consommer de la mémoire GPU. Sur une machine sans CUDA accessible, il affiche la limitation et ne charge pas le modèle.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if not torch.cuda.is_available():
    print('Laboratoire CUDA non lance : utiliser A sur CPU et un environnement compatible pour B.')
else:
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-0.6B', quantization_config=quantization,
        device_map={'': torch.cuda.current_device()},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, target_modules='all-linear',
        lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
    ))
    model.config.use_cache = False
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    inputs = tokenizer('Un batch regroupe plusieurs exemples.', return_tensors='pt').to('cuda')
    labels = inputs['input_ids'].clone()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(**inputs, labels=labels).loss
    loss.backward()
    print('Loss du step :', loss.item())
    print('Gradients adapters :', sum(p.grad is not None for p in model.parameters() if p.requires_grad))
    optimizer.step()
    model.save_pretrained('outputs/module20/adapter')
```

Le texte de ce test utilise une loss causale sur toute la séquence ; il ne démontre pas un SFT assistant-only. Un seul step confirme une mécanique, pas une amélioration du modèle.

`prepare_model_for_kbit_training` réalise des préparations liées au gel et aux dtypes ; les détails dépendent du modèle et des versions. L’ordre utilisé ici est : charger quantifié → préparer → injecter LoRA → construire l’optimizer.

Le `device_map` choisi vise un seul GPU. Ne transpose pas sans examen un placement automatique prévu pour l’inférence à un entraînement distribué.

## 7. Mesurer une comparaison réelle

Compare LoRA non quantifié et QLoRA avec même base/révision, données, cibles, rang, steps/tokens, batch et évaluation. Note séparément :

1. Mémoire après chargement.
2. Mémoire après injection.
3. Pic pendant un step complet, y compris optimizer.
4. Débit et durée, avec synchronisation GPU pour chronométrer.
5. Validation, benchmark et régressions.

Utilise des processus séparés pour comparer les allocations proprement. `empty_cache()` ne libère pas les tenseurs encore référencés. `memory_allocated` et `memory_reserved` ne représentent pas la même chose.

## 8. Expériences guidées

- Change seulement `bnb_4bit_use_double_quant` sur un matériel compatible et mesure le gain réel, sans l’inférer du nom.
- Inspecte les classes des projections quantifiées et les dtypes des adapters ; ne suppose pas que tout est 4 bits.
- Compare une entrée courte puis plus longue à même micro-batch. Identifie le coût des activations qui reste présent.
- Évalue la base quantifiée avant entraînement pour distinguer l’effet de quantification de celui de LoRA.

## 9. Exercices avec corrigés

### Exercice 1 — 13B en 4 bits

Quel stockage idéal brut pour 13 milliards de paramètres ?

<details>
<summary>Corrigé</summary>

`13e9×0.5 = 6.5e9` octets, soit **6.5 GB**, avant surcoûts et parties non quantifiées. Ce n’est pas le budget complet du training.

</details>

### Exercice 2 — Code ou stockage ?

Un tenseur uint8 contient uniquement des nombres de 0 à 15. Est-il automatiquement stocké en 4 bits ?

<details>
<summary>Corrigé</summary>

Non. Dans le laboratoire A, chaque élément occupe 8 bits. Il faudrait un format compacté pour stocker deux codes de 4 bits dans un octet.

</details>

### Exercice 3 — Compute dtype

Pourquoi une configuration peut-elle annoncer stockage 4 bits et calcul BF16 ?

<details>
<summary>Corrigé</summary>

Le stockage compact et les opérations numériques sont deux aspects différents. Les poids sont utilisés via les conversions et kernels appropriés, tandis que certaines opérations et activations utilisent BF16.

</details>

### Exercice 4 — Une comparaison manquante

QLoRA est moins bon après entraînement. Quelle mesure avant entraînement aide à isoler la cause ?

<details>
<summary>Corrigé</summary>

Évaluer la base quantifiée avant adaptation, en plus de la base non quantifiée. Cela permet de distinguer une différence introduite par la quantification d’une différence introduite par le training, sans les attribuer trop vite à une seule cause.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue niveaux, codes, reconstruction et compactage.
- [ ] Je sais pourquoi le laboratoire uniforme n’est pas NF4.
- [ ] Je définis QLoRA sans dire que tous les poids 4 bits sont optimisés.
- [ ] Je distingue storage dtype et compute dtype.
- [ ] Je connais les prérequis du bloc matériel et ne confonds pas saut et réussite.
- [ ] Je mesure mémoire et qualité avant/après quantification et adaptation.

## Fiche mémo

```text
QLoRA : base quantifiée et gelée + adapters entraînables.
4 bits de poids bruts ne décrivent pas tous les calculs ni toute la mémoire.
Mesurer la base seule, puis le step complet et la qualité.
```

Le [module 21](21_full_fine_tuning_principe.md) revient à l’entraînement direct de tous les poids.

## Références

[Transformers : bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes), [article QLoRA](https://arxiv.org/abs/2305.14314).
