# Module 55 — Catastrophic forgetting

## Pourquoi ce module arrive ici

Un modèle spécialisé peut gagner fortement sur le domaine tout en perdant des capacités générales. Il faut mesurer ce coût explicitement, surtout en full FT ou CPT étroit.

## Objectifs

Définir forgetting, distinguer forgetting et overfitting, construire un regression benchmark, mesurer gains/pertes et tester des stratégies de mitigation.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Évaluation

## Définitions concrètes

### Catastrophic forgetting

**Définition concrète.** Dégradation de compétences antérieures à la suite d’un nouvel entraînement.


**Exemple simple.** Math/general baissent après spécialisation domaine.

### Regression benchmark

**Définition concrète.** Jeu stable mesurant les capacités à préserver.


**Exemple simple.** Questions générales, format, raisonnement.

### Replay / data mixing

**Définition concrète.** Inclusion de données générales pendant la spécialisation.


**Exemple simple.** Mélanger un pourcentage de données générales.

### Regularization toward base

**Définition concrète.** Méthodes limitant la dérive par rapport au modèle initial.


**Exemple simple.** PEFT, KL-like constraints dans certains contextes, etc.

### Specialization trade-off

**Définition concrète.** Compromis entre performance domaine et conservation générale.


**Exemple simple.** +20 domaine, -2 général peut être acceptable; +20,-30 souvent non.

## Intuition simple

Un étudiant qui étudie exclusivement un micro-sujet pendant longtemps peut devenir très performant sur ce sujet tout en négligeant d’autres compétences. Le training peut faire un compromis similaire dans les poids.

## Ce qui se passe réellement sous le capot

1. Évaluer base sur domaine ET général.
2. Fine-tuner/CPT.
3. Réévaluer les mêmes suites.
4. Calculer delta par catégorie.
5. Identifier quelles capacités régressent.
6. Tester mitigation : LR plus faible, moins de steps, PEFT, mixture avec données générales, autre méthode.
7. Comparer le front gain domaine vs perte générale.

## Exemple minimal à comprendre mentalement

```text
              Base   Après
Domaine        45      88   +43
Général        82      75    -7
Raisonnement   76      60   -16
```

Dire seulement “domaine +43” cacherait une perte importante en raisonnement.

## Formules et notation utiles

Pour chaque capacité :

```text
delta = score_after - score_before
```

On peut définir des seuils de régression acceptables **avant** l’expérience afin de ne pas les déplacer opportunément après avoir vu les résultats.

## Code minimal observable

```python
before={"domain":45,"general":82,"reasoning":76}
after={"domain":88,"general":75,"reasoning":60}
for k in before:
    print(k, after[k]-before[k])
```

## Laboratoire guidé

1. Choisis au moins 2 catégories générales à préserver.  
2. Fixe un seuil maximal de régression.  
3. Compare base, LoRA et full FT sur domaine+général.  
4. Si full FT oublie plus, teste LR/steps réduits ou mélange de données générales.  
5. Trace un tableau gain domaine / perte générale / coût.

## Ce que tu dois observer

- Forgetting peut apparaître même si validation domaine est excellente.
- PEFT peut limiter certaines altérations mais ne garantit pas zéro régression.
- Mélanger des données générales consomme une partie du budget de training et modifie le compromis.

## À ne pas confondre

- Forgetting ≠ overfitting : on peut oublier sans surapprendre le train, et inversement.
- Régression sur benchmark ≠ preuve que les connaissances ont disparu de façon absolue; elle mesure le comportement sous ce protocole.

## Erreurs fréquentes

- Ne pas avoir de baseline générale.
- Accepter toute régression après coup parce que le score domaine est élevé.
- Comparer des méthodes avec des budgets de tokens différents sans le signaler.

## Exercices

1. Donne un exemple de forgetting sans overfitting évident.
2. Propose trois mitigations.
3. Définis un seuil d’acceptation pour un cas hypothétique.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Mesurer deltas par catégorie.
- Distinguer forgetting et overfitting.
- Tester une mitigation.
- Décider à partir d’un compromis explicite.

## Fiche mémo

Une spécialisation réussie maximise le gain utile **sous contrainte de régression acceptable**.

## Lien avec le module suivant

Les modules 56 à 62 sont des projets intégrateurs. Ils utilisent les compétences précédentes dans des expériences complètes et reproductibles.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
