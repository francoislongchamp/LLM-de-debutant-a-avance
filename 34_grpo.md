# Module 34 — GRPO

## Pourquoi ce module arrive ici

Après avoir construit une reward fiable, on peut optimiser une policy à partir de plusieurs générations. GRPO exploite des récompenses relatives au sein d’un groupe de réponses pour un même prompt.

## Objectifs

Comprendre groupe de completions, reward relative, advantage normalisé au groupe, policy ratio et rôle du contrôle de dérive. Lancer une petite expérience GRPO sur tâche vérifiable.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Reinforcement learning

## Définitions concrètes

### Group

**Définition concrète.** Ensemble de plusieurs réponses générées pour le même prompt.


**Exemple simple.** 8 completions pour un problème.

### Group-relative signal

**Définition concrète.** Comparaison d’une reward à la distribution du groupe.


**Exemple simple.** Une réponse au-dessus de la moyenne reçoit un avantage positif.

### Advantage

**Définition concrète.** Signal indiquant si la génération a fait mieux ou pire que la baseline/groupe.


**Exemple simple.** `A_i≈(r_i-mean(r))/std(r)` dans une intuition simplifiée.

### On-policy / online

**Définition concrète.** Les réponses d’entraînement sont générées par la policy courante ou proche.


**Exemple simple.** Le dataset de completions évolue pendant training.

### Policy ratio

**Définition concrète.** Rapport entre probabilité sous nouvelle et ancienne/reference policy utilisé dans les objectifs de policy optimization.


**Exemple simple.** Contrôle l’ampleur de la mise à jour.

## Intuition simple

Pour un même exercice, demande au modèle plusieurs tentatives. Au lieu de juger une réponse isolée, regarde lesquelles sont meilleures que leurs voisines. Ces différences fournissent le signal d’apprentissage.

## Ce qui se passe réellement sous le capot

1. Échantillonner un batch de prompts.
2. Générer plusieurs completions par prompt.
3. Calculer une ou plusieurs reward functions.
4. Transformer les rewards en signaux relatifs/advantages selon l’algorithme/configuration.
5. Calculer la loss de policy avec les log-probabilités des tokens générés.
6. Appliquer les mécanismes de clipping/KL/normalisation configurés.
7. Mettre à jour la policy.
8. Recommencer avec de nouvelles générations.

## Exemple minimal à comprendre mentalement

Pour un prompt, rewards :

```text
[0, 0, 1, 0.5]
mean = 0.375
```

Les réponses 1 et 0.5 sont au-dessus de la moyenne; les 0 sont en dessous. L’algorithme utilise une forme de comparaison relative plus précise selon son implémentation.

## Formules et notation utiles

Intuition d’avantage standardisé :

\[
A_i=\frac{r_i-\mu_r}{\sigma_r+\epsilon}
\]

Cela explique le terme “group relative”. Les variantes/configurations modernes peuvent utiliser d’autres détails; il faut lire la version de l’implémentation utilisée.

## Code minimal observable

```python
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/DeepMath-103K", split="train"),
    reward_funcs=accuracy_reward,
)
# Sur un vrai projet, commence petit et inspecte la reward avant trainer.train().
```

## Laboratoire guidé

1. Commence avec une tâche vérifiable que tu comprends.  
2. Mesure les rewards de la policy avant training.  
3. Utilise un petit nombre de prompts et completions pour inspecter manuellement les groupes.  
4. Lance un run court.  
5. Logge reward moyenne, taux de succès, longueur et métriques indépendantes.  
6. Inspecte les meilleurs/pire groupes.  
7. Vérifie qu’un gain de reward correspond bien à un gain réel.

## Ce que tu dois observer

- Si toutes les réponses d’un groupe ont le même reward, le signal relatif peut devenir faible selon la méthode.
- Plus de completions augmente le coût de génération.
- La reward doit être rapide et robuste car elle est appelée très souvent.

## À ne pas confondre

- GRPO ≠ DPO : GRPO génère/scorer online, DPO utilise des paires offline.
- Group reward ≠ majorité vote nécessairement.
- GRPO ≠ reward model obligatoire : des rewards vérifiables peuvent suffire.

## Erreurs fréquentes

- Lancer GRPO avant d’avoir testé la reward.
- Utiliser une tâche dont presque tous les groupes ont rewards identiques.
- Ne pas mesurer la longueur et autres comportements collatéraux.

## Exercices

1. Calcule mean et advantage intuitif pour [0,1,1,0].
2. Explique pourquoi 8 générations coûtent plus qu’une.
3. Compare le flux de données DPO vs GRPO.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer pourquoi plusieurs completions sont générées.
- Calculer un signal relatif simple.
- Distinguer GRPO de DPO.
- Lancer un mini-run et inspecter des groupes réels.

## Fiche mémo

GRPO apprend à partir de **générations online comparées relativement dans un groupe**, scorées par des rewards.

## Lien avec le module suivant

Le module 35 étudie PPO, historique important du RLHF et utile pour comprendre critic, advantage et clipping.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
