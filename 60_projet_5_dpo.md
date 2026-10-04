# Module 60 — Projet 5 — DPO

## Pourquoi ce module arrive ici

Ce projet vérifie si des préférences pairwise améliorent un comportement au-delà du SFT. L’accent est mis sur la qualité des paires et l’absence de raccourcis.

## Objectifs

Construire un dataset de préférences audité, entraîner DPO, mesurer pairwise win rate et vérifier que le gain de préférence n’abîme pas exactitude, diversité ou capacités générales.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 4 à 8 heures.  
**Niveau :** Projet intégrateur

## Définitions concrètes

### Preference objective

**Définition concrète.** Critère précis que les paires représentent.


**Exemple simple.** Concision sans perte d’exactitude.

### Preference holdout

**Définition concrète.** Paires non vues pour mesurer l’ordre choisi/rejected.


**Exemple simple.** Validation/test.

### Style shortcut

**Définition concrète.** Indice superficiel corrélé au chosen.


**Exemple simple.** Chosen toujours plus long.

### Win rate

**Définition concrète.** Fraction de comparaisons remportées selon un juge/protocole.


**Exemple simple.** DPO vs SFT en comparaison aveugle.

## Intuition simple

La difficulté n’est pas de lancer DPO; c’est de garantir que les paires enseignent **la préférence voulue** plutôt qu’un indice facile à exploiter.

## Ce qui se passe réellement sous le capot

1. Choisir un seul objectif préférence principal.
2. Créer chosen/rejected proches mais distinguables par ce critère.
3. Auditer longueur, format et mots-clés.
4. Séparer train/validation/test.
5. Mesurer modèle SFT sur préférence holdout.
6. Entraîner DPO avec beta/LR documentés.
7. Réévaluer préférence, benchmark général et statistiques de longueur.
8. Faire un pairwise blind si le critère est qualitatif.

## Exemple minimal à comprendre mentalement

Objectif : concision correcte. Une mauvaise paire serait chosen=100 mots corrects, rejected=5 mots faux : le modèle peut apprendre “long=bon”. Une meilleure paire garde les deux réponses factuellement correctes mais chosen est plus concise selon la rubrique.

## Formules et notation utiles

Rapporte au minimum :

```text
preference_accuracy_before/after
mean_length_before/after
general_score_before/after
```

Si preference accuracy monte mais longueur explose, inspecte un shortcut.

## Code minimal observable

```text
Preference audit checklist
--------------------------
[ ] same prompt/context
[ ] chosen truly preferred
[ ] rejected plausible
[ ] length distributions inspected
[ ] formatting artifacts inspected
[ ] reasons/rubric documented
[ ] holdout independent
```

## Laboratoire guidé

1. Crée 200 paires propres si possible, moins si tu veux seulement apprendre le pipeline.  
2. Annoter `reason/category` hors champs utilisés pour le training.  
3. Mesure statistiques chosen vs rejected.  
4. Baseline SFT.  
5. DPO court.  
6. Évalue preference holdout + general.  
7. Inspecte les 20 plus grands changements de probabilité/comportement.

## Ce que tu dois observer

- La qualité des rejected est cruciale.
- Le modèle peut apprendre des artefacts plus vite que le concept visé.
- Un DPO très fort peut homogénéiser le style.

## À ne pas confondre

- Win rate ≠ vérité factuelle.
- DPO ≠ reward model.
- Plus de marge chosen/rejected ≠ toujours meilleur modèle.

## Erreurs fréquentes

- Rejected absurdes.
- Chosen systématiquement plus long/court.
- Utiliser le même juge exact comme source des paires et seule validation.

## Exercices

1. Conçois trois paires “difficiles” pour une préférence claire.
2. Liste cinq shortcuts à mesurer.
3. Définis un critère d’arrêt si le général régresse.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Auditer les paires.
- Mesurer préférence avant/après.
- Mesurer les changements de style/longueur.
- Conserver un benchmark général.

## Fiche mémo

DPO est réussi lorsque la **préférence voulue**, et non un proxy superficiel, s’améliore.

## Lien avec le module suivant

Le projet 6 ajoute une boucle online avec reward vérifiable et GRPO ou autre policy optimization adaptée.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
