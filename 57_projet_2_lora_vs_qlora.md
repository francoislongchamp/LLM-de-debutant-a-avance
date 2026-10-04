# Module 57 — Projet 2 — LoRA vs QLoRA

## Pourquoi ce module arrive ici

QLoRA est souvent présenté comme “presque pareil avec moins de VRAM”. Ce projet te force à mesurer cette affirmation sur ton hardware, ton modèle et ton dataset.

## Objectifs

Comparer LoRA et QLoRA en contrôlant dataset, rank, targets, steps et benchmark. Mesurer mémoire, temps, qualité et contraintes logicielles.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 3 à 6 heures en réutilisant le projet 1.  
**Niveau :** Projet intégrateur

## Définitions concrètes

### Treatment

**Définition concrète.** Condition expérimentale comparée.


**Exemple simple.** LoRA BF16 vs QLoRA 4-bit.

### Controlled variables

**Définition concrète.** Paramètres maintenus identiques pour isoler l’effet étudié.


**Exemple simple.** Dataset, rank, targets, seed, steps.

### Peak VRAM

**Définition concrète.** Pic mémoire pendant une étape représentative.


**Exemple simple.** Mesuré avec PyTorch/CUDA.

### Throughput

**Définition concrète.** Tokens ou exemples traités par seconde.


**Exemple simple.** Permet de quantifier le coût temps.

## Intuition simple

Si tu modifies quantification, rank, targets et LR à la fois, tu compares deux recettes, pas l’effet de QLoRA. Ce projet doit changer le moins de variables possible.

## Ce qui se passe réellement sous le capot

1. Copier la configuration du projet 1.
2. Définir condition A base non quantifiée + LoRA.
3. Définir condition B base 4-bit + LoRA.
4. Garder même budget de training et benchmark.
5. Mesurer peak VRAM après warmup de quelques steps.
6. Mesurer tokens/s et wall time.
7. Évaluer les deux checkpoints.
8. Répéter si les différences qualité sont petites/bruitées.

## Exemple minimal à comprendre mentalement

Tableau attendu :

```text
Metric            LoRA      QLoRA
Peak VRAM         12.1 GB   7.4 GB
Tokens/s          900       820
Domain score      84.2      83.8
General score     77.0      76.9
Adapter size      65 MB     65 MB
```

Ces chiffres sont fictifs : ton but est de produire les tiens.

## Formules et notation utiles

Gain mémoire relatif :

```text
saving = 1 - VRAM_QLoRA / VRAM_LoRA
```

Mais ne réduis pas la décision à ce nombre : ajoute qualité et temps.

## Code minimal observable

```python
def saving(normal, quant):
    return 100*(1-quant/normal)
print(saving(12.1, 7.4))
```

## Laboratoire guidé

1. Exécute A avec une instrumentation mémoire propre.  
2. Redémarre le processus avant B pour éviter des mesures CUDA polluées.  
3. Exécute B avec NF4/config documentée.  
4. Compare les logs de loss.  
5. Évalue le même benchmark.  
6. Écris une décision : dans quelles contraintes choisirais-tu chaque méthode ?

## Ce que tu dois observer

- QLoRA réduit principalement la mémoire de la base.
- Le débit peut être plus bas ou différent selon GPU/kernels.
- Les différences qualité peuvent être petites face à la variance du run.

## À ne pas confondre

- Économie VRAM ≠ accélération.
- Même adapter size ≠ même mémoire de base.
- QLoRA ≠ modèle final obligatoirement utilisé en 4-bit pour toujours.

## Erreurs fréquentes

- Mesurer QLoRA après un LoRA dans le même process sans nettoyer la mémoire.
- Comparer des learning rates différents sans le dire.
- Ne pas versionner bitsandbytes/driver/CUDA.

## Exercices

1. Calcule le saving du tableau fictif.
2. Explique pourquoi le checkpoint adapter peut avoir la même taille.
3. Décris un hardware où QLoRA est particulièrement utile.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Produire une comparaison contrôlée.
- Mesurer peak VRAM et throughput.
- Évaluer qualité avec le même benchmark.
- Écrire un choix conditionnel, pas “QLoRA toujours meilleur”.

## Fiche mémo

Le résultat attendu est une **carte de compromis** entre mémoire, vitesse et qualité.

## Lien avec le module suivant

Le projet 3 compare maintenant adaptation basse-rang et mise à jour complète.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
