# Module 31 — Construire une reward vérifiable

## Pourquoi ce module arrive ici

Un algorithme RL puissant ne corrige pas une reward mal définie. Pour apprendre proprement, il faut commencer par des tâches dont le succès peut être vérifié automatiquement et sans ambiguïté.

## Objectifs

Écrire des reward functions déterministes, séparer parsing et scoring, gérer réponses invalides et cas limites, créer des tests unitaires pour la reward.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Reinforcement learning

## Définitions concrètes

### Verifiable reward

**Définition concrète.** Score dont la correction peut être établie par une règle ou un oracle fiable.


**Exemple simple.** Exactitude d’un calcul ou conformité à un schéma.

### Parser

**Définition concrète.** Fonction extrayant la partie pertinente de la génération.


**Exemple simple.** Lire un entier ou un JSON.

### Oracle

**Définition concrète.** Mécanisme donnant la vérité de référence.


**Exemple simple.** Résultat calculé par Python pour un problème arithmétique.

### Sparse reward

**Définition concrète.** Reward prenant peu de valeurs, souvent 0/1.


**Exemple simple.** Correct=1, faux=0.

### Shaping

**Définition concrète.** Ajout de sous-signaux pour guider l’apprentissage.


**Exemple simple.** Petit score pour format valide, score principal pour exactitude.

## Intuition simple

Avant de noter un élève, définis un corrigé que deux correcteurs appliqueraient de la même façon. Une reward vérifiable doit produire le même score pour la même sortie, indépendamment de ton humeur.

## Ce qui se passe réellement sous le capot

1. Définir exactement la sortie attendue.
2. Écrire un parser robuste qui ne confond pas explication et réponse.
3. Écrire l’oracle ou récupérer la cible fiable.
4. Scorer correctement les cas valides.
5. Donner un comportement explicite aux erreurs de parsing.
6. Écrire des tests unitaires pour correct, incorrect, vide, malformé et cas limites.
7. Mesurer la distribution de reward sur la policy de départ.

## Exemple minimal à comprendre mentalement

Sortie requise : un JSON `{"answer": 306}`.

```text
{"answer":306}        → format ok, exact → 1
{"answer":308}        → format ok, faux  → 0
306                     → format faux      → 0
{"answer":"306"}      → dépend du contrat explicitement défini
```

La spécification doit décider le dernier cas avant le training.

## Formules et notation utiles

Reward binaire : `R∈{0,1}`. Taux de succès de départ :

\[
success=\frac{\sum_i R_i}{N}
\]

Si success≈0 sur des milliers d’échantillons, le signal peut être trop sparse pour apprendre efficacement sans adaptation de curriculum/shaping.

## Code minimal observable

```python
import json

def reward_json(text, expected):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return 0.0
    if set(obj) != {"answer"}:
        return 0.0
    return 1.0 if obj["answer"] == expected else 0.0

assert reward_json('{"answer": 306}', 306) == 1.0
assert reward_json('{"answer": 308}', 306) == 0.0
assert reward_json('306', 306) == 0.0
```

## Laboratoire guidé

1. Écris la spécification de sortie avant le code.  
2. Implémente parser + oracle + reward séparément.  
3. Ajoute au moins 20 tests unitaires.  
4. Génère 100 réponses de la policy de départ et affiche distribution de rewards.  
5. Inspecte manuellement 20 cas score 0 et 20 score 1 pour vérifier le scorer.

## Ce que tu dois observer

- Le parser fait partie du système de reward et peut introduire des bugs.
- Une reward trop stricte peut donner presque tous les scores à zéro.
- Une reward trop permissive peut valider des sorties incorrectes.

## À ne pas confondre

- Format valide ≠ réponse correcte.
- Reward shaping ≠ donner la réponse cible au modèle.
- Oracle fiable ≠ LLM judge nécessairement fiable.

## Erreurs fréquentes

- Changer la reward en plein run sans versionner.
- Ne pas tester les cas malformés.
- Autoriser involontairement plusieurs formats qui contournent le critère.

## Exercices

1. Écris une reward pour une classification parmi 4 labels.
2. Liste cinq cas limites pour un JSON parser.
3. Explique comment mesurer si la reward est trop sparse.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Écrire une reward déterministe et testée.
- Séparer parsing, oracle et scoring.
- Mesurer le taux de succès initial.
- Identifier les cas limites avant training.

## Fiche mémo

En RL, la reward est une spécification exécutable de ce qui compte. Une mauvaise spécification produit une bonne optimisation du mauvais objectif.

## Lien avec le module suivant

Le module 32 combine plusieurs signaux tout en évitant qu’un composant domine arbitrairement.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
