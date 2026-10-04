# Module 15 — PEFT et intuition de LoRA

## Pourquoi ce module arrive ici

Le full fine-tuning modifie toutes les grandes matrices du modèle. LoRA part de l’observation qu’une adaptation utile peut souvent être représentée par une mise à jour de faible rang, donc avec beaucoup moins de paramètres entraînables.

## Objectifs

Comprendre PEFT, matrice gelée, mise à jour ΔW, factorisation basse-rang A/B, rank et scaling. Calculer le nombre de paramètres LoRA d’une couche simple.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning léger

## Définitions concrètes

### PEFT

**Définition concrète.** Parameter-Efficient Fine-Tuning : famille de méthodes entraînant une petite partie/extension des paramètres.


**Exemple simple.** LoRA est une méthode PEFT.

### Poids gelé

**Définition concrète.** Paramètre conservé mais non mis à jour par l’optimizer.


**Exemple simple.** La matrice W du modèle de base reste fixe.

### LoRA

**Définition concrète.** Low-Rank Adaptation : ajoute une mise à jour factorisée à certaines matrices.


**Exemple simple.** `W_effective = W + ΔW` avec `ΔW = B A`.

### Rank r

**Définition concrète.** Dimension interne de la factorisation basse-rang.


**Exemple simple.** r=8 est beaucoup plus petit que hidden_size=4096.

### Alpha

**Définition concrète.** Facteur de mise à l’échelle de la contribution LoRA.


**Exemple simple.** Souvent utilisé avec `alpha/r` ou variantes selon configuration.

### Adapter

**Définition concrète.** Petit ensemble de paramètres ajoutés et sauvegardables séparément.


**Exemple simple.** Un même modèle de base peut charger différents adapters.

## Intuition simple

Au lieu de réécrire un énorme tableau `W`, LoRA apprend deux petits tableaux dont le produit représente la correction. C’est comme décrire une grande modification à partir d’un petit nombre de directions utiles.

## Ce qui se passe réellement sous le capot

1. La matrice de base W reste gelée.
2. On crée A de forme `[r, in]` et B de forme `[out, r]`.
3. Pendant le forward, la contribution LoRA est calculée et ajoutée à la projection de base.
4. Backward calcule les gradients de A et B seulement, sauf modules explicitement sauvegardés/entraînés.
5. L’optimizer maintient des états pour ces petits paramètres.
6. À l’inférence, l’adapter peut rester séparé ou être fusionné lorsque la méthode/format le permet.

## Exemple minimal à comprendre mentalement

Couche carrée 4096×4096 :

```text
Full matrix : 4096×4096 = 16 777 216 paramètres
```

LoRA rank 8 :

```text
A : 8×4096    = 32 768
B : 4096×8    = 32 768
Total LoRA    = 65 536
```

Ici, l’adaptation de cette couche utilise environ 256× moins de paramètres qu’une matrice complète.

## Formules et notation utiles

\[
W' = W + sBA
\]

avec souvent `s=α/r` (selon configuration). Nombre de paramètres LoRA d’une couche linéaire `in→out` :

\[
N_{LoRA}=r\times in + out\times r = r(in+out)
\]

## Code minimal observable

```python
def lora_params(in_features, out_features, r):
    return r * (in_features + out_features)

full = 4096 * 4096
lora = lora_params(4096, 4096, 8)
print("full:", full)
print("lora:", lora)
print("ratio:", full / lora)
```

## Laboratoire guidé

1. Calcule full vs LoRA pour ranks 4, 8, 16, 64 sur 4096×4096.  
2. Refais pour une matrice non carrée.  
3. Dessine W, A et B avec leurs formes.  
4. Explique pourquoi le produit `BA` retrouve la forme de W.  
5. Liste ce que LoRA réduit et ce qu’il ne supprime pas (activations, base weights nécessaires à l’inférence, etc.).

## Ce que tu dois observer

- Le coût en paramètres LoRA croît linéairement avec r.
- La matrice effective a la même forme que W même si A/B sont petites.
- LoRA réduit surtout le nombre de paramètres entraînables et les états associés.

## À ne pas confondre

- LoRA ≠ quantification.
- Adapter petit ≠ modèle de base inutile à l’inférence.
- Rank plus grand ≠ automatiquement meilleur.
- Gelé ≠ absent de la mémoire.

## Erreurs fréquentes

- Présenter LoRA seulement comme une astuce mémoire sans comprendre ΔW.
- Choisir r par habitude sans expérimentation.

## Exercices

1. Calcule les paramètres LoRA pour in=4096,out=11008,r=16.
2. Explique pourquoi BA est au plus de rang r.
3. Explique comment plusieurs adapters peuvent partager le même modèle de base.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Écrire `W’=W+sBA`.
- Calculer le nombre de paramètres LoRA.
- Expliquer ce qui est gelé et entraîné.
- Distinguer PEFT, LoRA et quantification.

## Fiche mémo

LoRA apprend une **correction basse-rang** de grandes matrices au lieu de modifier tous leurs éléments.

## Lien avec le module suivant

Le module 16 injecte réellement des adapters LoRA avec PEFT et vérifie quels paramètres reçoivent les gradients.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
