# Module 32 — Rewards composites et normalisation

## Pourquoi ce module arrive ici

Une tâche réelle peut exiger exactitude, format et concision. Additionner naïvement plusieurs scores de différentes échelles peut donner un objectif dominé par le mauvais composant.

## Objectifs

Construire une reward composite, comprendre pondération, échelle, normalisation et conflits entre objectifs. Instrumenter chaque composante séparément.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Reinforcement learning

## Définitions concrètes

### Reward component

**Définition concrète.** Sous-score représentant un critère.


**Exemple simple.** correctness, format, longueur.

### Weight

**Définition concrète.** Coefficient contrôlant la contribution d’un composant.


**Exemple simple.** 0.8×correctness + 0.2×format.

### Scale

**Définition concrète.** Plage numérique naturelle d’un composant.


**Exemple simple.** Accuracy 0–1 vs score de longueur 0–100.

### Normalization

**Définition concrète.** Transformation mettant des composantes sur des échelles comparables ou standardisées.


**Exemple simple.** Min-max, z-score, ranking, selon contexte.

### Trade-off

**Définition concrète.** Compromis entre objectifs partiellement contradictoires.


**Exemple simple.** Être plus concis peut réduire l’explication.

## Intuition simple

Si tu additionnes “exactitude sur 1” et “nombre de caractères sur 1000”, la longueur écrase mathématiquement l’exactitude. Les poids n’ont de sens qu’en tenant compte des échelles.

## Ce qui se passe réellement sous le capot

1. Définir chaque composant séparément.
2. Mesurer sa distribution sur la policy initiale.
3. Ramener les composantes à des échelles maîtrisées si nécessaire.
4. Choisir des poids reflétant une priorité explicite.
5. Logger chaque composante ET la somme.
6. Vérifier les corrélations/conflits.
7. Tester des sorties adversariales pour voir ce qui maximise artificiellement le score.

## Exemple minimal à comprendre mentalement

Mauvais :

```text
reward = correctness (0..1) + length (0..500)
```

Le modèle maximise presque uniquement length.

Plus contrôlé :

```text
reward = 0.8 * correctness_01 + 0.2 * format_01
```

On peut ajouter une pénalité de longueur bornée séparément si nécessaire.

## Formules et notation utiles

\[
R=\sum_i w_i\tilde r_i
\]

où `\tilde r_i` sont des composantes ramenées à des échelles choisies. Les poids `w_i` expriment ensuite un compromis interprétable.

## Code minimal observable

```python
def total_reward(correct, format_ok):
    r_correct = 1.0 if correct else 0.0
    r_format = 1.0 if format_ok else 0.0
    return 0.8*r_correct + 0.2*r_format

print(total_reward(True, True))   # 1.0
print(total_reward(False, True))  # 0.2
```

## Laboratoire guidé

1. Définis deux composantes 0–1.  
2. Logge chacune séparément sur 100 générations.  
3. Crée quatre sorties : correcte+format, correcte+malformatée, fausse+format, fausse+malformatée.  
4. Vérifie l’ordre de récompense attendu.  
5. Change les poids et explique quel comportement tu encourages.

## Ce que tu dois observer

- Une récompense totale peut monter tandis que le critère principal baisse.
- La normalisation peut elle-même évoluer si elle dépend d’un batch/groupe.
- Les poids sont des choix de produit/objectif, pas des constantes scientifiques universelles.

## À ne pas confondre

- Pondération ≠ calibration.
- Normalisation ≠ garantie d’absence de reward hacking.
- Somme des rewards ≠ satisfaction de chaque contrainte.

## Erreurs fréquentes

- Ne logger que le total.
- Utiliser des composantes de plages très différentes.
- Ajouter trop de composantes sans pouvoir expliquer leur interaction.

## Exercices

1. Construis une matrice des quatre cas correct/format.
2. Propose un cas où concision et complétude entrent en conflit.
3. Explique comment détecter qu’un composant domine.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Construire une reward composite interprétable.
- Mesurer les distributions de chaque composante.
- Justifier les poids.
- Détecter un conflit entre objectifs.

## Fiche mémo

Une reward composite n’est utile que si ses **composantes restent observables** et ses compromis sont explicites.

## Lien avec le module suivant

Le module 33 montre comment une policy exploite les failles de ces récompenses : reward hacking.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
