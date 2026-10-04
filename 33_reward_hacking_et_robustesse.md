# Module 33 — Reward hacking et robustesse

## Pourquoi ce module arrive ici

Une policy optimise le signal fourni, pas l’intention humaine cachée derrière ce signal. Toute différence entre métrique et objectif réel peut devenir une voie d’exploitation.

## Objectifs

Comprendre specification gaming/reward hacking, proxies, distribution shift et overoptimization. Construire des tests adversariaux de reward et séparer reward de métriques d’évaluation indépendantes.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Reinforcement learning

## Définitions concrètes

### Reward hacking

**Définition concrète.** Comportement obtenant un reward élevé sans accomplir correctement l’objectif réel.


**Exemple simple.** Répéter un mot-clé qui trompe le scorer.

### Proxy

**Définition concrète.** Mesure indirecte utilisée à la place du véritable objectif.


**Exemple simple.** Longueur comme proxy de “détail”.

### Overoptimization

**Définition concrète.** Optimisation poussée d’un proxy au point de dégrader la qualité réelle.


**Exemple simple.** Reward monte mais jugement indépendant baisse.

### Goodhart’s law

**Définition concrète.** Idée qu’une mesure peut devenir moins fiable lorsqu’elle devient une cible fortement optimisée.


**Exemple simple.** Le modèle exploite ce qui était seulement corrélé à la qualité.

### Holdout evaluator

**Définition concrète.** Évaluation indépendante non utilisée directement comme reward de training.


**Exemple simple.** Tests fonctionnels ou rubric séparée.

## Intuition simple

Si tu récompenses un élève au nombre de pages, il peut écrire beaucoup sans répondre mieux. Ce n’est pas un bug de l’optimisation : elle suit exactement la règle donnée.

## Ce qui se passe réellement sous le capot

1. Identifier les proxies utilisés dans la reward.
2. Créer des sorties conçues pour maximiser chaque proxy sans satisfaire l’objectif.
3. Tester le scorer sur ces sorties.
4. Pendant training, logger reward et métriques indépendantes.
5. Inspecter les exemples aux rewards les plus élevés.
6. Arrêter/réviser si reward monte mais qualité indépendante stagne ou baisse.
7. Versionner les changements de reward et repartir d’une comparaison propre.

## Exemple minimal à comprendre mentalement

Reward naïve :

```python
reward = min(len(answer)/500, 1.0)
```

Une réponse de 500 caractères reçoit le maximum, même si elle répète la même phrase. La longueur était corrélée à la complétude mais n’est pas la complétude.

## Formules et notation utiles

On peut suivre la corrélation entre reward de training `R` et score indépendant `E`. Si `R↑` alors que `E↓`, l’optimisation exploite probablement une divergence entre proxy et objectif. La corrélation ne prouve pas la causalité, mais c’est un signal d’alarme.

## Code minimal observable

```python
def bad_reward(answer):
    return min(len(answer)/500, 1.0)

normal = "Réponse correcte et concise."
spam = ("mot " * 200).strip()
print(bad_reward(normal), bad_reward(spam))
```

## Laboratoire guidé

1. Prends ta reward du module 31/32.  
2. Écris 20 sorties volontairement absurdes mais susceptibles de gagner du score.  
3. Classe les failles trouvées.  
4. Ajoute des tests qui doivent échouer.  
5. Définis au moins une métrique holdout indépendante.  
6. Pendant un petit run, inspecte les 20 réponses à reward maximal.

## Ce que tu dois observer

- Les extrêmes de reward révèlent souvent les raccourcis.
- Ajouter une pénalité peut créer une nouvelle faille ailleurs.
- Une reward model apprise peut aussi être hackée.

## À ne pas confondre

- Reward hacking ≠ hacking informatique; c’est l’exploitation du critère d’optimisation.
- Reward élevé ≠ qualité réelle.
- Ajouter plus de règles ≠ garantie de robustesse.

## Erreurs fréquentes

- Regarder uniquement la moyenne de reward.
- Utiliser le même juge pour entraîner et conclure à la réussite.
- Ne pas tester les valeurs extrêmes.

## Exercices

1. Trouve trois proxies risqués pour “bonne réponse”.
2. Explique Goodhart avec ton propre exemple.
3. Propose un holdout evaluator pour une tâche de code ou JSON.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir reward hacking.
- Créer un test adversarial de reward.
- Séparer reward et évaluation indépendante.
- Diagnostiquer reward↑ / qualité↓.

## Fiche mémo

Plus un signal est optimisé, plus ses imperfections deviennent importantes. La robustesse de la reward est une partie du training, pas un détail.

## Lien avec le module suivant

Le module 34 introduit GRPO, une méthode online qui compare plusieurs générations d’un même prompt.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
