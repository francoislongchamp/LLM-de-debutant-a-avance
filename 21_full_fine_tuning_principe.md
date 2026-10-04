# Module 21 — Full fine-tuning — principe

## Pourquoi ce module arrive ici

LoRA contraint l’adaptation à de petites branches. Le full fine-tuning autorise chaque paramètre du modèle à changer. C’est plus flexible, mais beaucoup plus coûteux et potentiellement plus destructeur.

## Objectifs

Comprendre précisément ce qui change en full FT, pourquoi la mémoire augmente, quand cette flexibilité est utile et quels risques apparaissent.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning complet

## Définitions concrètes

### Full fine-tuning

**Définition concrète.** Continuation de l’entraînement où l’ensemble ou presque des paramètres du modèle restent entraînables.


**Exemple simple.** Embeddings, attention, MLP, normes et tête peuvent changer.

### Catastrophic forgetting

**Définition concrète.** Dégradation de capacités précédemment acquises à cause de nouvelles mises à jour.


**Exemple simple.** Spécialisation forte qui fait régresser le général.

### Optimizer state

**Définition concrète.** État maintenu pour les paramètres entraînés.


**Exemple simple.** AdamW doit en maintenir pour presque tout le modèle en full FT.

### Checkpoint complet

**Définition concrète.** Sauvegarde des poids du modèle entier.


**Exemple simple.** Beaucoup plus volumineux qu’un adapter LoRA.

## Intuition simple

LoRA ajoute des corrections limitées; full FT donne une gomme et un crayon sur tout le modèle. Tu peux faire davantage, mais tu peux aussi effacer davantage.

## Ce qui se passe réellement sous le capot

1. Charger la base dans une précision d’entraînement adaptée.
2. Laisser les paramètres `requires_grad=True`.
3. Construire un optimizer sur tous les paramètres.
4. Forward conserve les activations nécessaires.
5. Backward calcule les gradients de toutes les parties.
6. Optimizer states existent à grande échelle.
7. Les checkpoints sauvegardent la totalité ou des shards.

## Exemple minimal à comprendre mentalement

Modèle 600M :

```text
LoRA trainable : quelques millions
Full FT        : ~600 millions
```

Même si les poids seuls tiennent sur la GPU, gradients + optimizer + activations peuvent dépasser largement la mémoire disponible.

## Formules et notation utiles

Un ordre de grandeur naïf des poids BF16 est `2 bytes/param`, mais full FT ajoute au minimum des gradients et des états d’optimizer dont la précision dépend de l’implémentation. Le budget réel doit être **mesuré** ou calculé composante par composante.

## Code minimal observable

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total, trainable)
assert total == trainable  # cas simple, sans gel explicite
```

## Laboratoire guidé

1. Charge un petit modèle compatible avec ta VRAM.  
2. Vérifie total=trainable.  
3. Fais un seul forward/backward et mesure peak VRAM.  
4. Compare au même modèle avec LoRA.  
5. Inspecte quels composants de mémoire augmentent.  
6. Ne cherche pas encore la meilleure qualité : l’objectif est de comprendre le coût.

## Ce que tu dois observer

- Le full FT peut devenir mémoire-limitant très vite.
- Le checkpoint est beaucoup plus grand.
- Un LR adapté à LoRA peut être excessif pour full FT.
- La flexibilité supplémentaire n’implique pas automatiquement une meilleure généralisation.

## À ne pas confondre

- Full FT ≠ pretraining from scratch : on part de poids pré-entraînés.
- Full FT ≠ continued pretraining : le terme décrit ce qui est entraîné, pas le type de données.
- Tous paramètres trainables ≠ tous changent beaucoup.

## Erreurs fréquentes

- Tester full FT sur un modèle trop gros avant d’avoir estimé la mémoire.
- Copier le LR LoRA.
- Ne pas conserver de benchmark général pour détecter forgetting.

## Exercices

1. Explique les composantes mémoire qui réapparaissent pour les poids de base.
2. Donne trois raisons de préférer LoRA malgré une infrastructure capable de full FT.
3. Donne deux cas où full FT peut être justifié.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Distinguer full FT, LoRA et pretraining from scratch.
- Expliquer la hausse mémoire.
- Expliquer le risque de forgetting.
- Vérifier que tous les paramètres voulus sont trainables.

## Fiche mémo

Full FT maximise la liberté de mise à jour, donc aussi le coût et le risque d’altérer les capacités existantes.

## Lien avec le module suivant

Le module 22 apprend à rendre un full FT petit modèle stable et mesurable.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
