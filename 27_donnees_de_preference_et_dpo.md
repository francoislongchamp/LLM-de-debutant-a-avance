# Module 27 — Données de préférence et DPO

## Pourquoi ce module arrive ici

Le SFT dit “voici une réponse à reproduire”. Les données de préférence disent “parmi ces réponses, celle-ci est préférable”. Cela permet d’enseigner des critères relatifs : style, précision, format, utilité ou autres rubriques.

## Objectifs

Comprendre prompt/chosen/rejected, préférence pairwise, modèle de référence et intuition de DPO. Construire des paires de préférence cohérentes et éviter les préférences triviales ou biaisées.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Post-training — préférences

## Définitions concrètes

### Preference pair

**Définition concrète.** Deux réponses au même contexte avec une préférence explicite.


**Exemple simple.** chosen=A, rejected=B.

### Chosen

**Définition concrète.** Réponse préférée selon la politique d’annotation.


**Exemple simple.** Plus correcte et concise.

### Rejected

**Définition concrète.** Réponse moins préférée.


**Exemple simple.** Contient une erreur ou ne respecte pas le format.

### Pairwise preference

**Définition concrète.** Comparaison relative plutôt qu’un score absolu.


**Exemple simple.** A>B.

### Reference policy

**Définition concrète.** Modèle de référence servant à contrôler la dérive relative dans la formulation DPO.


**Exemple simple.** Souvent une copie/état de départ du modèle.

### DPO

**Définition concrète.** Direct Preference Optimization : méthode offline optimisant directement la préférence à partir de paires, sans boucle de reward model + PPO obligatoire.


**Exemple simple.** Le modèle augmente le rapport de probabilité relatif du chosen contre rejected.

## Intuition simple

SFT ressemble à montrer une seule bonne rédaction. DPO ressemble à montrer deux rédactions et dire laquelle tu préfères. L’information nouvelle est **la comparaison**.

## Ce qui se passe réellement sous le capot

1. Le même prompt/context est associé à chosen et rejected.
2. Le modèle calcule les log-probabilités des deux réponses.
3. Une référence fournit un point d’ancrage relatif.
4. La loss favorise le chosen relativement au rejected tout en tenant compte de la référence.
5. Backward met à jour la policy.
6. L’évaluation vérifie que la préférence apprise améliore les critères visés sans créer de régressions.

## Exemple minimal à comprendre mentalement

```json
{
  "prompt": "Réponds en une phrase : qu’est-ce que DNS ?",
  "chosen": "DNS associe notamment des noms de domaine à des informations comme des adresses IP.",
  "rejected": "DNS est un protocole qui chiffre toutes les connexions Internet."
}
```

Cette paire est facile : le rejected est factuellement faux. Des préférences de style nécessitent des critères d’annotation plus précis.

## Formules et notation utiles

Intuition simplifiée : DPO cherche à rendre le chosen relativement plus probable que le rejected par rapport à une politique de référence. Le paramètre `β` règle la force de cette préférence/du contrôle relatif selon la formulation utilisée. L’important ici est le **rapport de préférences**, pas de mémoriser immédiatement l’équation complète.

## Code minimal observable

```python
from datasets import Dataset

rows = [{
    "prompt": "Explique DNS en une phrase.",
    "chosen": "DNS permet notamment de résoudre des noms de domaine en données comme des adresses IP.",
    "rejected": "DNS est un système de chiffrement de disque.",
}]
print(Dataset.from_list(rows)[0])
```

## Laboratoire guidé

1. Crée 50 paires sur trois critères distincts : exactitude, format, concision.  
2. Pour chaque paire, ajoute hors training un champ `reason` expliquant le choix pour audit.  
3. Vérifie que chosen/rejected répondent au **même prompt**.  
4. Cherche les indices superficiels : chosen toujours plus long ? toujours ponctué différemment ?  
5. Construis 10 paires difficiles où les deux réponses sont plausibles.

## Ce que tu dois observer

- Des paires trop faciles peuvent enseigner des signaux superficiels.
- Si chosen est systématiquement plus long, le modèle peut apprendre longueur=préférence.
- La qualité de la règle d’annotation est centrale.

## À ne pas confondre

- DPO ≠ SFT sur le chosen uniquement.
- DPO ≠ PPO.
- Preference data ≠ reward scalar obligatoire.

## Erreurs fréquentes

- Créer rejected volontairement absurdes pour toutes les paires.
- Mélanger plusieurs critères sans les documenter.
- Laisser des artefacts de format révéler mécaniquement le chosen.

## Exercices

1. Transforme trois exemples SFT en vraies paires de préférence utiles.
2. Explique pourquoi “chosen plus long” peut devenir un raccourci.
3. Propose une rubrique d’annotation à 3 critères.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir chosen/rejected.
- Expliquer la différence SFT vs préférence.
- Expliquer l’intuition de la référence et de β.
- Auditer un dataset pour des raccourcis superficiels.

## Fiche mémo

DPO apprend un **ordre relatif** entre réponses, pas simplement une cible unique.

## Lien avec le module suivant

Le module 28 entraîne réellement une policy avec DPOTrainer et mesure les marges chosen/rejected.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
