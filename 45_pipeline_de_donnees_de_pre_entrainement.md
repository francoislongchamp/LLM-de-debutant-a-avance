# Module 45 — Pipeline de données de pré-entraînement

## Pourquoi ce module arrive ici

À grande échelle, le pipeline de données détermine une grande partie de ce que le modèle apprend. Le pré-entraînement nécessite extraction, filtrage, déduplication, mélange, tokenisation et sharding reproductibles.

## Objectifs

Concevoir un pipeline par étapes avec provenance, licence, qualité, langue, PII/sécurité, déduplication, mixture, tokenisation et shards. Produire un manifeste auditable.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — données

## Définitions concrètes

### Source

**Définition concrète.** Origine des documents.


**Exemple simple.** Corpus public/licencié, données internes autorisées.

### Extraction

**Définition concrète.** Conversion du format source en texte/structure exploitable.


**Exemple simple.** HTML→texte utile.

### Quality filtering

**Définition concrète.** Règles/modèles éliminant du contenu de faible qualité selon l’objectif.


**Exemple simple.** Pages vides, spam, corruption.

### Language ID

**Définition concrète.** Estimation de la langue d’un document.


**Exemple simple.** Permet de construire une mixture multilingue contrôlée.

### PII filtering

**Définition concrète.** Détection/traitement de données personnelles selon exigences juridiques et de gouvernance.


**Exemple simple.** Emails, identifiants ou autres données selon contexte.

### Shard

**Définition concrète.** Fichier contenant une fraction du dataset tokenisé pour streaming/parallelisme.


**Exemple simple.** train-00042.parquet.

### Manifest

**Définition concrète.** Métadonnées décrivant sources, versions, hashes, comptes et transformations.


**Exemple simple.** Permet audit/reproduction.

## Intuition simple

Le modèle ne voit jamais “Internet” ou “des documents” : il voit la sortie finale de ton pipeline. Chaque filtre et proportion est donc une décision de modélisation.

## Ce qui se passe réellement sous le capot

1. Inventorier les sources et droits d’utilisation.
2. Extraire texte + métadonnées stables.
3. Normaliser sans supprimer arbitrairement l’information utile.
4. Détecter langue/qualité et appliquer des filtres versionnés.
5. Traiter PII et exigences de gouvernance.
6. Dédupliquer exact/near-duplicate.
7. Définir la mixture et sampling.
8. Créer split holdout avant les transformations risquant la contamination.
9. Tokeniser avec version figée.
10. Écrire des shards et manifeste avec compte de tokens/hashes.

## Exemple minimal à comprendre mentalement

```text
raw/       1 000 000 docs
extract/     940 000
quality/     710 000
dedup/       640 000
mixture/     640 000 docs pondérés
train/test   séparation par source/document
tokenized/   2.4B tokens
shards/      256 fichiers
```

Chaque différence de compte doit être explicable.

## Formules et notation utiles

Taux de rétention :

\[
retention=\frac{N_{après}}{N_{avant}}
\]

Le nombre de documents seul ne suffit pas; suivre aussi bytes, caractères et tokens par source.

## Code minimal observable

```python
manifest = {
  "pipeline_version": "v1",
  "tokenizer": "tokenizer-sha256...",
  "sources": {"docs_a": 120000, "docs_b": 80000},
  "tokens": 250_000_000,
  "filters": ["lang_v2", "quality_v3", "dedup_v1"],
}
print(manifest)
```

## Laboratoire guidé

1. Prends un corpus de taille gérable.  
2. Crée des dossiers/étapes immuables raw→clean→dedup→tokenized.  
3. À chaque étape, écris un rapport de comptes et raisons de rejet.  
4. Calcule tokens par source après tokenisation.  
5. Écris un manifest JSON final avec hashes/configs.  
6. Reproduis le résultat depuis raw dans un nouveau dossier.

## Ce que tu dois observer

- Filtrer par documents et mesurer par documents peut cacher un changement massif de tokens.
- Le nettoyage peut modifier la mixture sans qu’on le veuille.
- La provenance doit survivre jusqu’au dataset final autant que possible.

## À ne pas confondre

- Nettoyage ≠ déduplication.
- Quality score ≠ vérité.
- Shard ≠ split train/test.

## Erreurs fréquentes

- Écraser les données intermédiaires sans version.
- Perdre les IDs/provenances après extraction.
- Ne suivre que le nombre de fichiers.

## Exercices

1. Dessine ton DAG de données.
2. Liste les métriques à enregistrer à chaque étape.
3. Explique pourquoi la séparation de test doit considérer les documents liés.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Décrire un pipeline complet.
- Produire des comptes par étape.
- Conserver provenance/config/hashes.
- Reproduire un dataset tokenisé.

## Fiche mémo

Le corpus final est un produit logiciel versionné, pas un dossier de textes assemblés manuellement.

## Lien avec le module suivant

Le module 46 approfondit l’étape qui évite de surpondérer les mêmes contenus : la déduplication.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
