# Module 61 — Projet 6 — RL vérifiable

## Pourquoi ce module arrive ici

C’est le premier projet où la policy génère des réponses pendant l’entraînement et reçoit un reward calculé. La priorité absolue est de valider la reward avant de dépenser du compute.

## Objectifs

Construire une tâche automatiquement vérifiable, écrire une reward testée, mesurer la distribution initiale, lancer un petit GRPO et démontrer que reward et métrique indépendante progressent ensemble.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 6 à 12 heures pour un petit modèle et une tâche simple.  
**Niveau :** Projet intégrateur

## Définitions concrètes

### Task generator

**Définition concrète.** Procédure créant des prompts et vérités de référence.


**Exemple simple.** Petits problèmes arithmétiques générés avec seed.

### Reward test suite

**Définition concrète.** Tests unitaires du scorer.


**Exemple simple.** Correct/faux/malformé/cas limite.

### Initial success rate

**Définition concrète.** Taux de réussite de la policy avant RL.


**Exemple simple.** Évite reward entièrement nulle ou déjà saturée.

### Online samples

**Définition concrète.** Completions générées pendant training.


**Exemple simple.** Elles dépendent de la policy courante.

### Holdout task set

**Définition concrète.** Problèmes jamais utilisés dans les updates, pour évaluation.


**Exemple simple.** Même générateur mais seeds séparées.

## Intuition simple

Tu construis un petit environnement scientifique : le modèle essaie, le programme corrige, le modèle s’ajuste. Si le correcteur est faux, tout le reste est faux.

## Ce qui se passe réellement sous le capot

1. Choisir une tâche non dangereuse et automatiquement vérifiable.
2. Écrire générateur + oracle + parser + reward.
3. Écrire tests unitaires exhaustifs.
4. Créer train prompts et holdout prompts séparés par seed.
5. Mesurer success initial et distribution de reward.
6. Lancer un très petit run online.
7. Logger reward components, longueur, KL si disponible, success holdout.
8. Inspecter les plus hauts rewards pour hacking.
9. Augmenter l’échelle seulement après validation.

## Exemple minimal à comprendre mentalement

Tâche : additions de deux nombres, sortie JSON stricte.

```text
Prompt: 127 + 58
Oracle: 185
Expected output: {"answer":185}
```

Reward principale = exactitude. Reward format peut être séparée et faible, afin qu’un JSON valide mais faux ne gagne pas presque autant qu’une bonne réponse.

## Formules et notation utiles

Mesures :

```text
train reward mean
holdout exact accuracy
format accuracy
mean completion length
```

Le succès doit être démontré surtout par holdout exact accuracy, pas seulement reward train.

## Code minimal observable

```python
import json

def reward(text, expected):
    try:
        obj=json.loads(text)
    except Exception:
        return 0.0
    return 1.0 if obj == {"answer": expected} else 0.0

assert reward('{"answer":185}',185)==1.0
assert reward('{"answer":184}',185)==0.0
```

## Laboratoire guidé

1. Écris 30+ tests du reward.  
2. Génére 200 réponses baseline et inspecte 50 aléatoirement.  
3. Si success=0 %, simplifie la difficulté ou construis un curriculum avant un long run.  
4. Lance GRPO court.  
5. Évalue toutes les 50–100 updates sur holdout fixe.  
6. Cherche reward hacking.  
7. Compare au SFT sur quelques solutions si tu veux comprendre le compromis supervision vs RL.

## Ce que tu dois observer

- Reward train peut augmenter avant holdout.
- Un curriculum trop facile peut saturer.
- Le coût de génération peut dominer le coût du projet.
- Le modèle peut améliorer le format sans améliorer le raisonnement si le reward le permet.

## À ne pas confondre

- Reward train ≠ holdout accuracy.
- Online data ≠ dataset fixe.
- GRPO ≠ besoin obligatoire d’un reward model.

## Erreurs fréquentes

- Reward non testée.
- Aucun holdout.
- Inspecter seulement des exemples réussis.
- Faire une tâche complexe avant d’avoir validé le pipeline sur une tâche simple.

## Exercices

1. Conçois une autre tâche vérifiable non ambiguë.
2. Explique comment détecter une reward trop sparse.
3. Définis un curriculum de trois niveaux.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Tests reward complets.
- Baseline de reward/success.
- Run online court reproductible.
- Gain sur holdout et audit anti-hacking.

## Fiche mémo

Le RL vérifiable devient scientifique lorsque **l’oracle est fiable et le holdout indépendant**.

## Lien avec le module suivant

Le projet 7 assemble tout le parcours : tokenizer et modèle from scratch, puis SFT et post-training optionnel.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
