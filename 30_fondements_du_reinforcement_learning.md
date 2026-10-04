# Module 30 — Fondements du reinforcement learning

## Pourquoi ce module arrive ici

DPO est offline et fonctionne directement sur des paires. Le reinforcement learning introduit une boucle où la policy agit, reçoit une récompense et est optimisée à partir de trajectoires générées.

## Objectifs

Définir agent/policy, état/observation, action, trajectoire, reward, return, value et advantage. Relier ces concepts à un LLM sans anthropomorphiser le modèle.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Reinforcement learning

## Définitions concrètes

### Policy πθ

**Définition concrète.** Distribution qui choisit des actions selon une observation/contexte.


**Exemple simple.** Pour un LLM, distribution du prochain token ou de la séquence.

### Action

**Définition concrète.** Choix effectué par la policy.


**Exemple simple.** Un token, ou conceptuellement une réponse entière selon le niveau d’analyse.

### Trajectory

**Définition concrète.** Suite d’observations/actions produite pendant un épisode.


**Exemple simple.** Prompt puis tokens générés jusqu’à la fin.

### Reward

**Définition concrète.** Signal numérique fourni par l’environnement/évaluateur.


**Exemple simple.** 1 si une réponse vérifiable est correcte, 0 sinon.

### Return

**Définition concrète.** Somme pondérée des rewards futures dans le RL séquentiel.


**Exemple simple.** Si reward seulement à la fin, le return est souvent lié à ce reward terminal.

### Value

**Définition concrète.** Estimation du return attendu depuis un état.


**Exemple simple.** Un critic peut approximer cette valeur.

### Advantage

**Définition concrète.** Mesure de combien une action/trajectoire a fait mieux ou pire qu’une baseline attendue.


**Exemple simple.** Reward 0.9 comparé à baseline 0.5 → advantage positif.

## Intuition simple

SFT donne directement une réponse à imiter. RL donne plutôt une **note après l’action**. La policy doit découvrir quelles probabilités de génération augmentent les notes futures.

## Ce qui se passe réellement sous le capot

1. Échantillonner un prompt/état initial.
2. La policy génère une ou plusieurs actions/réponses.
3. Un environnement ou reward function calcule un score.
4. Construire un signal d’avantage ou objectif de policy à partir des rewards/baselines.
5. Calculer une loss RL.
6. Backward met à jour la policy.
7. Répéter avec de nouvelles générations produites par la policy courante dans les méthodes online.

## Exemple minimal à comprendre mentalement

Tâche vérifiable :

```text
Prompt: Donne le résultat de 17×18 sous forme d’un entier.
Réponse A: 306 → reward 1
Réponse B: 308 → reward 0
```

Le RL ne reçoit pas nécessairement une “réponse cible” à copier; il reçoit un signal indiquant quelle génération a réussi.

## Formules et notation utiles

Objectif conceptuel :

\[
\max_\theta\; \mathbb{E}_{y\sim\pi_\theta(.|x)}[R(x,y)]
\]

Le défi est que l’échantillonnage d’actions discrètes n’est pas dérivable directement de la même manière qu’une cible SFT; les méthodes de policy gradient construisent donc des estimateurs adaptés.

## Code minimal observable

```python
def reward(answer):
    return 1.0 if answer.strip() == "306" else 0.0

for a in ["306", "308", "306 "]:
    print(repr(a), reward(a))
```

## Laboratoire guidé

1. Choisis une tâche simple automatiquement vérifiable.  
2. Définis exactement observation, action et reward.  
3. Génére manuellement 10 réponses d’un modèle sans training.  
4. Score-les.  
5. Calcule le reward moyen et sa variance.  
6. Explique ce qui ferait un bon ou mauvais signal d’apprentissage.

## Ce que tu dois observer

- Un reward sparse 0/1 donne peu d’information lorsque presque tout échoue.
- Un reward facile à exploiter peut être optimisé sans accomplir l’intention réelle.
- Online signifie que les données changent avec la policy.

## À ne pas confondre

- Reward ≠ loss.
- Policy ≠ reward model.
- RL ≠ toute forme de post-training.
- Action token-level ≠ obligation de concevoir le reward token par token.

## Erreurs fréquentes

- Créer une reward ambiguë puis attribuer les échecs à l’algorithme.
- Oublier de mesurer la distribution des rewards.
- Anthropomorphiser l’agent au lieu d’analyser la policy statistiquement.

## Exercices

1. Définis state/action/reward pour une tâche de classification au format JSON.
2. Explique sparse vs dense reward.
3. Explique pourquoi le reward moyen peut changer à mesure que la policy change.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir policy/action/trajectory/reward/advantage.
- Relier ces termes à un LLM.
- Écrire une reward vérifiable simple.
- Expliquer la différence fondamentale avec SFT.

## Fiche mémo

RL optimise une **policy à partir des conséquences scorées de ses propres actions**, plutôt qu’en copiant seulement des cibles.

## Lien avec le module suivant

Le module 31 apprend à construire une reward vérifiable, composante la plus importante avant de choisir GRPO ou PPO.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
