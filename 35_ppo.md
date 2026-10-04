# Module 35 — PPO

## Pourquoi ce module arrive ici

PPO est un algorithme majeur de policy optimization et un repère historique du RLHF. Même si d’autres méthodes sont souvent choisies aujourd’hui, comprendre PPO clarifie value model, advantage, ratio de policy et clipping.

## Objectifs

Comprendre actor/policy, critic/value, old policy, probability ratio, clipped objective et KL. Savoir expliquer le pipeline RLHF classique sans devoir immédiatement déployer une grande infrastructure PPO.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Reinforcement learning

## Définitions concrètes

### Actor / policy

**Définition concrète.** Modèle qui génère les actions.


**Exemple simple.** Le LLM à optimiser.

### Critic / value model

**Définition concrète.** Modèle/tête estimant le return attendu pour réduire la variance du signal.


**Exemple simple.** Prédit une valeur pour les états/tokens.

### Old policy

**Définition concrète.** Version de la policy ayant généré les trajectoires du batch courant.


**Exemple simple.** Sert à mesurer le ratio de changement.

### Probability ratio

**Définition concrète.** `π_new(a|s)/π_old(a|s)`.


**Exemple simple.** Ratio 1 = probabilité inchangée.

### Clipping

**Définition concrète.** Limitation de l’incitation à modifier trop fortement le ratio pendant une update.


**Exemple simple.** Empêche un avantage de pousser sans limite un changement local.

### PPO

**Définition concrète.** Proximal Policy Optimization : famille d’objectifs maintenant les updates relativement proches tout en maximisant les advantages.


**Exemple simple.** Utilisé historiquement dans plusieurs pipelines RLHF.

## Intuition simple

Si une réponse a eu une bonne récompense, on veut augmenter sa probabilité — mais pas transformer brutalement le modèle en une seule update. PPO met une “zone de prudence” autour du changement de probabilité.

## Ce qui se passe réellement sous le capot

1. Collecter des trajectoires avec la policy courante.
2. Calculer rewards.
3. Estimer values/returns et advantages.
4. Calculer le ratio nouvelle/ancienne policy pour les actions observées.
5. Appliquer l’objectif PPO avec clipping.
6. Entraîner aussi le critic/value selon sa loss.
7. Ajouter éventuellement contrôle KL vers une référence.
8. Répéter plusieurs mini-epochs sur le batch selon configuration, puis regénérer.

## Exemple minimal à comprendre mentalement

Supposons qu’un token d’une bonne trajectoire avait probabilité 0.20 sous old policy. Nouvelle policy propose 0.30 :

```text
ratio = 0.30/0.20 = 1.5
```

Avec une plage de clipping autour de 1, une partie de l’objectif empêche de continuer à récompenser sans limite ce grand déplacement pendant la même update.

## Formules et notation utiles

Objectif PPO simplifié :

\[
L^{clip}=\mathbb E[\min(r_tA_t,\;clip(r_t,1-\epsilon,1+\epsilon)A_t)]
\]

avec :

\[
r_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
\]

C’est une intuition de base; les implémentations LLM ajoutent souvent value loss, entropy/KL et nombreux détails.

## Code minimal observable

```python
def ratio(p_new, p_old):
    return p_new / p_old

print(ratio(0.30, 0.20))  # 1.5
print(ratio(0.18, 0.20))  # 0.9
```

## Laboratoire guidé

1. Calcule manuellement des ratios pour plusieurs probabilités.  
2. Dessine l’effet de `clip(r,0.8,1.2)` pour r=0.5,0.9,1.1,1.8.  
3. Simule des advantages positifs et négatifs et explique l’effet attendu.  
4. Étudie l’API PPO de la version de TRL ou framework que tu utilises uniquement après avoir compris ces quantités.  
5. Compare conceptuellement coût PPO (policy+value+generation) à DPO.

## Ce que tu dois observer

- PPO est plus complexe opérationnellement que DPO.
- Le critic est une source supplémentaire de mémoire, calcul et erreurs.
- Clipping limite une partie de l’incitation mais ne garantit pas l’absence de dérive.

## À ne pas confondre

- PPO clipping ≠ gradient clipping.
- Critic ≠ reward model, même s’ils produisent tous deux des scalaires.
- Old policy ≠ reference policy dans tous les sens; leurs rôles diffèrent.

## Erreurs fréquentes

- Mémoriser l’équation sans savoir ce que représente le ratio.
- Confondre reward, value et advantage.
- Déployer PPO à grande échelle avant une reward validée.

## Exercices

1. Calcule le ratio pour p_old=.1,p_new=.08.
2. Explique l’effet attendu d’un advantage négatif.
3. Compare reward model et value model en une phrase chacun.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Définir policy/value/reward/advantage.
- Calculer un probability ratio.
- Expliquer le clipping PPO.
- Distinguer PPO de DPO et GRPO au niveau du flux de données.

## Fiche mémo

PPO combine trajectoires online, advantages, critic et updates “proximales” pour éviter de trop déplacer la policy en une fois.

## Lien avec le module suivant

Le module 36 formalise une autre notion de proximité très utilisée : la divergence KL.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
