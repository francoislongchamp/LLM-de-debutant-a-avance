# Module 9 — Construire un dataset de qualité

## Pourquoi ce module arrive ici

Le modèle n’apprend pas ce que tu “voulais dire”; il apprend les régularités présentes dans les exemples. Avant de parler de trainer, il faut donc savoir ce qu’un exemple enseigne réellement et reconnaître les données qui introduisent du bruit, des contradictions ou des biais de format.

## Objectifs

Savoir définir exemple, échantillon, schéma, qualité, diversité, couverture, duplication, contradiction et distribution. Construire un petit dataset SFT cohérent et justifier chaque champ.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning — données

## Définitions concrètes

### Exemple / sample

**Définition concrète.** Une unité d’entraînement logique.


**Exemple simple.** Une conversation user→assistant peut être un exemple.

### Schéma

**Définition concrète.** Structure attendue des colonnes/champs.


**Exemple simple.** `messages` contenant une liste de rôles et contenus.

### Qualité

**Définition concrète.** Degré auquel un exemple est correct, pertinent, clair et conforme au comportement voulu.


**Exemple simple.** Une réponse factuellement fausse est de mauvaise qualité même si son JSON est valide.

### Couverture

**Définition concrète.** Variété des situations que le dataset représente.


**Exemple simple.** Questions courtes, longues, ambiguës, cas limites.

### Distribution

**Définition concrète.** Fréquence relative des types d’exemples.


**Exemple simple.** 80 % de questions courtes peut pousser le modèle vers des réponses courtes.

### Déduplication

**Définition concrète.** Suppression/réduction d’exemples identiques ou quasi identiques.


**Exemple simple.** 100 copies du même Q/R surpondèrent artificiellement cette réponse.

### Contradiction

**Définition concrète.** Exemples enseignant des sorties incompatibles pour un contexte similaire.


**Exemple simple.** Même question avec deux réponses mutuellement exclusives sans contexte distinct.

## Intuition simple

Un dataset est le “manuel d’exemples” que le modèle voit. Si le manuel contient 100 fois une règle et une seule fois son exception, le signal statistique n’est pas équilibré. La quantité ne compense donc pas automatiquement une mauvaise sélection.

## Ce qui se passe réellement sous le capot

1. Définir d’abord le comportement cible : contenu, style, format, limites.
2. Définir un schéma stable.
3. Collecter/créer des exemples couvrant les cas courants et les cas limites.
4. Valider structure, longueur, champs manquants et encodage.
5. Contrôler exactitude, contradictions et duplications.
6. Inspecter les distributions : catégories, longueurs, langues, formats.
7. Versionner le dataset comme un artefact de code.

## Exemple minimal à comprendre mentalement

Exemple conversationnel simple :

```json
{"messages":[
  {"role":"user","content":"Qu’est-ce que HTTP ?"},
  {"role":"assistant","content":"HTTP est un protocole applicatif utilisé pour échanger des ressources entre un client et un serveur."}
]}
```

Ce seul exemple enseigne plusieurs choses à la fois : vocabulaire, contenu, longueur approximative, style affirmatif et format conversationnel.

## Formules et notation utiles

Pour inspecter une distribution de catégories :

\[
p(c)=\frac{N_c}{N_{total}}
\]

Une catégorie à 70 % du dataset aura beaucoup plus d’occasions d’influencer les gradients qu’une catégorie à 1 %, toutes choses égales par ailleurs.

## Code minimal observable

```python
import json
from collections import Counter

path = "data/train.jsonl"
rows=[]
with open(path, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        row=json.loads(line)
        assert "messages" in row, f"ligne {line_no}: messages absent"
        assert len(row["messages"]) >= 2
        rows.append(row)

lengths=[sum(len(m["content"]) for m in r["messages"]) for r in rows]
print("examples:", len(rows))
print("chars min/max:", min(lengths), max(lengths))
```

## Laboratoire guidé

1. Crée 30 exemples avec un objectif étroit et explicite.  
2. Ajoute une colonne ou métadonnée `category` hors des messages.  
3. Compte les catégories et longueurs.  
4. Introduis volontairement un doublon puis détecte-le.  
5. Introduis une contradiction et documente pourquoi elle est problématique.  
6. Écris une fiche `DATASET_CARD.md` décrivant source, licence, nettoyage et limites.

## Ce que tu dois observer

- Le format valide n’implique pas la qualité sémantique.
- Les distributions cachées deviennent visibles dès qu’on compte catégories et longueurs.
- Une petite quantité d’exemples excellents est utile pour apprendre le pipeline, mais ne garantit pas une spécialisation robuste.

## À ne pas confondre

- Dataset propre syntaxiquement ≠ dataset correct.
- Diversité ≠ bruit aléatoire.
- Plus de données ≠ meilleur résultat si les données sont contradictoires ou hors objectif.

## Erreurs fréquentes

- Mélanger plusieurs objectifs sans pouvoir mesurer leur proportion.
- Générer massivement des données synthétiques sans contrôle qualité.
- Ne pas conserver la provenance et la version du dataset.

## Exercices

1. Écris cinq critères mesurables de “bonne réponse” pour ton dataset.
2. Explique comment 100 doublons changent implicitement la pondération.
3. Propose trois cas limites qui devraient être présents dans un dataset d’explication technique.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer ce qu’un exemple enseigne au-delà de son contenu factuel.
- Décrire qualité, couverture, distribution et duplication.
- Valider un JSONL et produire des statistiques simples.
- Justifier pourquoi une donnée doit être incluse ou rejetée.

## Fiche mémo

Le dataset est une partie du modèle final : il détermine le signal que l’optimizer verra. La première optimisation est souvent d’améliorer les données, pas les hyperparamètres.

## Lien avec le module suivant

Le module 10 transforme ces données avec Hugging Face Datasets tout en gardant une trace de ce qui change.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
