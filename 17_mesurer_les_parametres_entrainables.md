# Module 17 — Mesurer les paramètres entraînables

## Pourquoi ce module arrive ici

Après avoir injecté LoRA, il faut quantifier ce que l’on a réellement changé. “Peu de paramètres” doit devenir un nombre, un pourcentage et un budget mémoire approximatif.

## Objectifs

Compter les paramètres totaux, entraînables et gelés; calculer leur proportion; relier ce nombre aux gradients, aux états de l’optimizer et à la taille d’un adapter.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning léger

## Définitions concrètes

### Paramètre total

**Définition concrète.** Tout élément appris présent dans le modèle chargé.


**Exemple simple.** La base + les adapters LoRA.

### Paramètre entraînable

**Définition concrète.** Paramètre avec gradient activé et inclus dans l’optimisation.


**Exemple simple.** Les matrices A/B de LoRA.

### Paramètre gelé

**Définition concrète.** Paramètre utilisé dans le forward mais non mis à jour.


**Exemple simple.** Les poids de la base.

### Ratio entraînable

**Définition concrète.** Fraction `trainable/total`.


**Exemple simple.** 6M sur 600M ≈1 %.

### Optimizer state

**Définition concrète.** État auxiliaire conservé par l’optimizer pour chaque paramètre optimisé.


**Exemple simple.** AdamW maintient typiquement des moments pour les paramètres entraînés.

## Intuition simple

LoRA est intéressant non parce que le modèle “devient petit”, mais parce que **la partie qui doit être optimisée devient petite**. La base continue à participer au calcul, mais on évite de stocker gradients et états d’optimizer pour tous ses poids.

## Ce qui se passe réellement sous le capot

1. Parcourir `model.parameters()`.
2. Sommer `numel()` pour le total.
3. Sommer seulement les paramètres `requires_grad=True`.
4. Calculer le ratio.
5. Inspecter les noms entraînables pour vérifier qu’ils correspondent bien aux adapters attendus.
6. Estimer mémoire de gradients/optimizer à partir du nombre entraînable et des dtypes/implémentations.

## Exemple minimal à comprendre mentalement

```text
Total      = 600 000 000
Trainable  =   6 000 000
Ratio      = 1 %
```

Même si 99 % des paramètres sont gelés, les 600M poids sont encore utilisés par le forward. La réduction concerne surtout la **mise à jour** et les états qui l’accompagnent.

## Formules et notation utiles

\[
ratio=100\times\frac{N_{trainable}}{N_{total}}
\]

Si les gradients des paramètres entraînables sont BF16, leur mémoire brute est approximativement `2×N_trainable` octets, avant autres états.

## Code minimal observable

```python
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total, trainable = count_params(trainer.model)
print(f"total      : {total:,}")
print(f"trainable  : {trainable:,}")
print(f"ratio      : {100*trainable/total:.4f}%")

for name, p in trainer.model.named_parameters():
    if p.requires_grad:
        print("TRAIN:", name, tuple(p.shape))
```

## Laboratoire guidé

1. Mesure le ratio pour r=4, 16 et 64.  
2. Exporte la liste des paramètres entraînables dans un fichier texte.  
3. Vérifie qu’aucun poids inattendu de la base n’est entraînable.  
4. Compare la taille des checkpoints adapters.  
5. Calcule paramètres LoRA théoriques et compare à la mesure réelle.

## Ce que tu dois observer

- Le ratio augmente à peu près linéairement avec r pour un ensemble de cibles fixe.
- Les biais ou modules_to_save peuvent modifier le compte.
- Le checkpoint peut contenir métadonnées en plus des tenseurs.

## À ne pas confondre

- Trainable ratio ≠ réduction identique de VRAM totale.
- Paramètre gelé ≠ paramètre retiré.
- Taille d’adapter ≠ nombre exact de paramètres × un seul dtype dans tous les formats.

## Erreurs fréquentes

- Faire confiance à une configuration sans inspecter `requires_grad`.
- Annoncer une économie mémoire en se basant uniquement sur le ratio de paramètres entraînables.

## Exercices

1. Si 12M paramètres sont entraînables sur 1,2B, calcule le ratio.
2. Explique quelles composantes mémoire diminuent lorsque la base est gelée.
3. Explique pourquoi les activations existent encore.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Compter total/trainable.
- Calculer le pourcentage.
- Relier paramètres entraînables aux gradients et optimizer states.
- Vérifier par noms que seules les bonnes parties sont entraînées.

## Fiche mémo

Mesurer évite les suppositions : un LoRA correctement configuré se voit dans les noms, `requires_grad`, le ratio et la taille du checkpoint.

## Lien avec le module suivant

Le module 18 étudie comment rank, alpha et dropout changent la capacité de l’adapter.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
