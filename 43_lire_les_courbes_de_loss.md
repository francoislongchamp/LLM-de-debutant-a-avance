# Module 43 — Lire les courbes de loss

## Pourquoi ce module arrive ici

Une loss qui descend est nécessaire mais pas suffisante. Sa forme, l’écart train/validation, les spikes, les plateaux et les changements liés au LR racontent ce qui arrive au training.

## Objectifs

Reconnaître baisse normale, plateau, divergence, spikes, overfitting et ruptures de pipeline. Corréler loss avec LR, grad norm, données et throughput.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — diagnostic

## Définitions concrètes

### Learning curve

**Définition concrète.** Évolution d’une métrique selon steps/tokens.


**Exemple simple.** Train loss vs tokens vus.

### Plateau

**Définition concrète.** Zone où la métrique n’améliore presque plus.


**Exemple simple.** Peut venir du LR, du modèle, des données ou simplement d’une convergence locale.

### Spike

**Définition concrète.** Hausse brève et importante de loss/grad norm.


**Exemple simple.** Batch difficile/corrompu ou instabilité.

### Divergence

**Définition concrète.** Détérioration persistante ou valeurs non finies.


**Exemple simple.** Loss qui explose.

### Smoothing

**Définition concrète.** Moyenne glissante pour visualiser une tendance.


**Exemple simple.** À utiliser sans cacher les spikes bruts.

## Intuition simple

Une courbe de loss est un électrocardiogramme du training : elle ne donne pas un diagnostic complet, mais des changements de forme orientent les vérifications à faire.

## Ce qui se passe réellement sous le capot

1. Logger loss brute et moyenne glissante.
2. Logger validation à intervalles réguliers.
3. Superposer ou corréler LR et grad norm.
4. Lors d’un spike, identifier les batch IDs/tokens si possible.
5. Lors d’un plateau, vérifier LR, données, capacité et validation.
6. Lors d’une divergence, stopper et inspecter le dernier état stable.

## Exemple minimal à comprendre mentalement

```text
step 0    9.7
100       7.3
500       5.4
1000      4.8
1500      4.7
2000      8.9  <- spike
2100      4.6
```

Un spike isolé suivi d’un retour n’a pas la même signification qu’une hausse persistante vers NaN.

## Formules et notation utiles

Moyenne glissante simple sur fenêtre k :

\[
\bar L_t=\frac1k\sum_{i=t-k+1}^{t}L_i
\]

Toujours conserver la série brute pour ne pas masquer les anomalies.

## Code minimal observable

```python
losses=[9.7,8.2,7.3,6.0,5.4,8.9,5.1]
window=3
for i in range(window-1,len(losses)):
    avg=sum(losses[i-window+1:i+1])/window
    print(i, avg)
```

## Laboratoire guidé

1. Sauvegarde loss/LR/grad_norm par step.  
2. Produit courbe brute et lissée.  
3. Ajoute val loss.  
4. Pour le plus gros spike, retrouve le batch si ton pipeline le permet.  
5. Fais une courte expérience avec LR trop élevé afin de reconnaître la divergence, puis jette ce checkpoint.  
6. Écris un diagnostic pour trois formes de courbe.

## Ce que tu dois observer

- Les batchs ont une difficulté variable, donc loss bruitée est normale.
- La validation peut se détériorer malgré train loss descendante.
- Un changement de pipeline de données peut créer une rupture de courbe.

## À ne pas confondre

- Plateau ≠ preuve que le modèle est “plein”.
- Spike ≠ divergence nécessairement.
- Smoothing ≠ nouvelle métrique.

## Erreurs fréquentes

- Ne logger que toutes les plusieurs milliers de steps et manquer les anomalies.
- Lisser tellement qu’on cache les spikes.
- Modifier scheduler en cours de run sans noter la rupture.

## Exercices

1. Décris le diagnostic d’un train loss↓ / val loss↑.
2. Décris le diagnostic d’un loss et grad norm qui deviennent NaN.
3. Explique pourquoi tokens vus est parfois un meilleur axe x que epochs.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Reconnaître plateau/spike/divergence/overfit.
- Corréler loss et LR/grad norm.
- Conserver données brutes et lissées.
- Proposer une investigation plutôt qu’un diagnostic magique.

## Fiche mémo

Une courbe ne répond pas seule, mais elle te dit **où regarder** dans les données, l’optimizer ou la stabilité.

## Lien avec le module suivant

Le module 44 convertit la cross-entropy en perplexité et explique pourquoi cette métrique est utile mais limitée.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
