# Module 18 — Rank LoRA, alpha et dropout

## Pourquoi ce module arrive ici

Une configuration LoRA ne se résume pas à “r=16 parce que tout le monde le fait”. Rank, scaling et dropout déterminent capacité, amplitude et régularisation de la branche LoRA.

## Objectifs

Comprendre l’effet de r, alpha, scaling et dropout. Concevoir une expérience contrôlée comparant plusieurs ranks au lieu d’optimiser au hasard.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning léger

## Définitions concrètes

### Rank r

**Définition concrète.** Dimension interne de la factorisation BA.


**Exemple simple.** r=4 impose une correction plus contrainte que r=64.

### lora_alpha

**Définition concrète.** Hyperparamètre de mise à l’échelle.


**Exemple simple.** Avec scaling classique, la branche peut être multipliée par alpha/r.

### Scaling

**Définition concrète.** Facteur appliqué à la contribution LoRA.


**Exemple simple.** Il modifie l’amplitude effective de ΔW.

### lora_dropout

**Définition concrète.** Dropout sur le chemin LoRA pendant entraînement.


**Exemple simple.** Peut régulariser mais ajoute du bruit.

### rsLoRA

**Définition concrète.** Variante de scaling qui utilise notamment une dépendance différente au rank.


**Exemple simple.** Elle vise une meilleure stabilité pour certains ranks élevés.

## Intuition simple

Le rank détermine combien de “directions indépendantes” la correction peut combiner. Plus r est grand, plus l’adapter peut représenter des changements complexes — mais il coûte plus et peut aussi suradapter.

## Ce qui se passe réellement sous le capot

1. Choisir un ensemble de target modules fixe.
2. Changer seulement r pour mesurer son effet propre.
3. Calculer le nombre de paramètres pour chaque r.
4. Garder dataset, seed, steps, évaluation et autres hyperparamètres comparables.
5. Mesurer qualité, loss, VRAM, temps et taille du checkpoint.
6. Tester ensuite alpha/scaling séparément.
7. Tester dropout uniquement si nécessaire, sans changer simultanément tout le reste.

## Exemple minimal à comprendre mentalement

Pour une couche 4096→4096 :

```text
r=4  : 32 768 paramètres LoRA
r=16 : 131 072
r=64 : 524 288
```

Multiplier r par 4 multiplie ici les paramètres LoRA par 4.

## Formules et notation utiles

Paramètres :

\[
N=r(in+out)
\]

Scaling classique fréquent :

\[
\Delta W = \frac{\alpha}{r}BA
\]

Le détail de scaling dépend de la variante LoRA configurée.

## Code minimal observable

```python
def count_lora(in_f, out_f, r):
    return r*(in_f+out_f)

for r in [4, 8, 16, 32, 64]:
    print(r, count_lora(4096,4096,r))
```

## Laboratoire guidé

1. Prépare trois configs : r=4,16,64.  
2. Fixe exactement même dataset, nombre de steps, seed, target modules et benchmark.  
3. Mesure trainable params, peak VRAM, durée, validation et scores métier.  
4. Trace un tableau résultat.  
5. Choisis le plus petit rank donnant des résultats suffisants selon ton objectif, plutôt que le meilleur score brut sans contrainte.

## Ce que tu dois observer

- Rank plus élevé augmente capacité et coût.
- Une différence de score faible peut ne pas justifier un adapter 4× plus grand.
- L’effet d’alpha dépend du rank et de la stratégie de scaling.

## À ne pas confondre

- Rank ≠ nombre de couches LoRA.
- Alpha ≠ learning rate.
- Dropout ≠ quantification.

## Erreurs fréquentes

- Comparer des ranks avec des nombres de steps ou datasets différents.
- Conclure qu’un rank élevé est meilleur à partir de train loss uniquement.
- Modifier r et target_modules simultanément dans une expérience censée mesurer r.

## Exercices

1. Calcule les paramètres pour r=32 sur 4096→11008.
2. Explique pourquoi r=4096 ferait perdre une grande partie de l’intérêt “low-rank”.
3. Propose une expérience pour isoler l’effet du dropout.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer intuitivement le rank.
- Calculer son coût.
- Distinguer alpha et learning rate.
- Construire une comparaison contrôlée de ranks.

## Fiche mémo

Le bon rank est un compromis **capacité ↔ coût ↔ généralisation**, pas une constante universelle.

## Lien avec le module suivant

Le module 19 décide où placer cette capacité : les target modules.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
