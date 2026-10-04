# Module 47 — Dataset mixture et sampling

## Pourquoi ce module arrive ici

Après nettoyage, plusieurs sources restent. Leur simple concaténation laisse les plus grandes dominer. La mixture rend explicite la proportion de tokens que le modèle voit de chaque domaine/langue.

## Objectifs

Comprendre mixture weights, sampling probability, upsampling/downsampling, epochs implicites par source et température de sampling. Construire une mixture mesurable en tokens.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — données

## Définitions concrètes

### Mixture

**Définition concrète.** Distribution cible des sources pendant l’entraînement.


**Exemple simple.** 50 % web, 20 % code, 20 % docs, 10 % livres.

### Sampling weight

**Définition concrète.** Poids déterminant la probabilité de tirer une source.


**Exemple simple.** 0.2 pour code.

### Upsampling

**Définition concrète.** Voir une petite source plus souvent que sa proportion naturelle.


**Exemple simple.** Un corpus scientifique de 1 % peut fournir 10 % des tokens.

### Downsampling

**Définition concrète.** Réduire la fréquence d’une très grande source.


**Exemple simple.** Web 90 % naturel → 50 % training.

### Effective epochs

**Définition concrète.** Nombre moyen de fois où une source est revisitée sous un budget de tokens.


**Exemple simple.** Petite source fortement upsamplée peut être vue de nombreuses fois.

### Temperature sampling

**Définition concrète.** Transformation des proportions pour aplatir ou accentuer la distribution.


**Exemple simple.** Souvent utilisée pour langues/sources déséquilibrées, avec conventions variables.

## Intuition simple

La mixture est un budget d’attention pédagogique : sur 100 tokens d’entraînement, combien veux-tu consacrer à chaque type de contenu ? Une source dix fois plus petite peut être vue aussi souvent si tu l’upsamples, mais cela augmente le risque de répétition.

## Ce qui se passe réellement sous le capot

1. Mesurer tokens disponibles par source après nettoyage/dedup.
2. Définir proportions cibles en fonction des objectifs.
3. Calculer combien de tokens chaque source doit fournir sous le budget total.
4. Calculer effective epochs/repetition de chaque source.
5. Ajuster si une petite source serait répétée excessivement.
6. Échantillonner avec seed et logger source de chaque batch/shard.
7. Évaluer par domaines pour vérifier les conséquences.

## Exemple minimal à comprendre mentalement

Budget 1B tokens, mixture :

```text
web  50% → 500M
code 20% → 200M
docs 20% → 200M
book 10% → 100M
```

Si `docs` ne contient que 50M tokens uniques, atteindre 200M implique environ 4 passages moyens sur cette source.

## Formules et notation utiles

Tokens cibles source i :

\[
T_i=w_iT_{total}
\]

Effective passes approximatif :

\[
E_i=\frac{T_i}{T_{available,i}}
\]

Une valeur très élevée est un signal de répétition potentielle.

## Code minimal observable

```python
total=1_000_000_000
weights={"web":.5,"code":.2,"docs":.2,"books":.1}
available={"web":2_000_000_000,"code":300_000_000,"docs":50_000_000,"books":200_000_000}

for src,w in weights.items():
    target=total*w
    print(src, "target", int(target), "effective passes", target/available[src])
```

## Laboratoire guidé

1. Mesure tokens disponibles de tes sources.  
2. Propose une mixture.  
3. Calcule tokens cibles et effective passes.  
4. Identifie les sources trop répétées.  
5. Crée 100 000 tirages simulés et vérifie les proportions empiriques.  
6. Versionne mixture weights avec chaque run.

## Ce que tu dois observer

- La mixture après dédup peut différer fortement de la mixture brute.
- Upsampling d’une petite source augmente sa répétition.
- La mixture influence les capacités mais aussi le style et les langues dominantes.

## À ne pas confondre

- Mixture weight ≠ taille brute de source.
- Upsampling ≠ créer de nouvelles informations.
- Plus de source spécialisée ≠ bénéfice illimité.

## Erreurs fréquentes

- Définir des poids sans calculer effective passes.
- Changer la mixture sans changer l’identifiant du dataset/run.
- Mesurer seulement la loss globale au lieu de pertes/benchmarks par domaine.

## Exercices

1. Avec budget 500M et weight code 0.3, combien de tokens code ?
2. Si code disponible=50M, combien de passages moyens ?
3. Explique le risque de suréchantillonner 20× une petite source.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Calculer tokens par mixture weight.
- Calculer effective passes.
- Simuler et vérifier le sampling.
- Justifier les proportions par objectifs et benchmarks.

## Fiche mémo

La mixture transforme des corpus de tailles naturelles en une **distribution d’apprentissage choisie**.

## Lien avec le module suivant

Le module 48 passe à plusieurs GPU : d’abord le data parallel classique, avant le sharding FSDP.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
