# Module 26 — CPT vs SFT — savoir choisir

## Pourquoi ce module arrive ici

Beaucoup de projets fine-tunent alors qu’ils ont surtout besoin de RAG, ou font du CPT alors qu’ils veulent simplement un format de réponse. Il faut partir du type de manque observé.

## Objectifs

Diagnostiquer un besoin : connaissance externe dynamique, vocabulaire/distribution de domaine, comportement/instruction, préférence. Choisir RAG, CPT, SFT ou combinaison avec une hypothèse testable.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Adaptation de domaine

## Définitions concrètes

### Knowledge gap

**Définition concrète.** Information absente ou difficile à produire correctement.


**Exemple simple.** Le modèle ignore une procédure interne récente.

### Behavior gap

**Définition concrète.** Le modèle possède l’information mais ne répond pas dans le format/style attendu.


**Exemple simple.** Il connaît JSON mais ne respecte pas un schéma.

### Distribution shift

**Définition concrète.** Les textes visés diffèrent nettement des données générales.


**Exemple simple.** Langage spécialisé, syntaxe de code particulière.

### RAG

**Définition concrète.** Récupération de documents au moment de l’inférence pour fournir du contexte.


**Exemple simple.** Information qui change souvent et doit être citée.

### Post-training

**Définition concrète.** Étapes après le pré-entraînement de base, incluant SFT, préférence, RL, etc.


**Exemple simple.** Apprendre un comportement ou une préférence.

## Intuition simple

Avant de “traiter” le modèle, diagnostique le symptôme. Si l’information change chaque semaine, l’inscrire dans les poids par training peut être le mauvais outil. Si le problème est un style de réponse, faire lire 100 GB de texte brut peut être disproportionné.

## Ce qui se passe réellement sous le capot

1. Écrire des exemples précis d’échec du modèle de base.
2. Classer chaque échec : manque d’information, mauvaise utilisation, mauvaise distribution, mauvais comportement, mauvaise préférence.
3. Tester d’abord l’intervention la moins coûteuse répondant au problème.
4. Définir une baseline et un benchmark ciblé.
5. Appliquer une seule intervention principale.
6. Comparer et n’ajouter une seconde étape que si la mesure justifie sa nécessité.

## Exemple minimal à comprendre mentalement

Cas A : le modèle ne connaît pas les procédures internes mises à jour hier → RAG est souvent plus naturel.  
Cas B : il lit correctement les procédures mais refuse de produire le format YAML attendu → SFT peut être pertinent.  
Cas C : il peine sur un jargon et une distribution documentaire très spécialisée malgré contexte suffisant → CPT peut être étudié.  
Cas D : deux réponses sont valides mais on veut préférer systématiquement la plus claire → préférence/DPO peut être pertinent.

## Formules et notation utiles

Le choix est principalement expérimental, pas une formule. On peut toutefois comparer le **gain par coût** :

\[
Efficiency=\frac{\Delta score}{coût\;GPU\;ou\;tokens\;d’entraînement}
\]

C’est une aide à la décision, pas une métrique universelle.

## Code minimal observable

```python
decision = {
  "information_fraiche": "RAG",
  "comportement_format": "SFT",
  "distribution_domaine": "CPT à tester",
  "preference_reponses": "DPO/RL à tester",
}
for symptom, method in decision.items():
    print(symptom, "->", method)
```

## Laboratoire guidé

1. Prends 20 erreurs réelles du modèle de base.  
2. Classe chacune par type de gap.  
3. Choisis 5 erreurs représentatives.  
4. Écris une hypothèse : “si je fais SFT/CPT/RAG, alors telle métrique devrait s’améliorer”.  
5. Définis le test qui pourrait réfuter ton hypothèse.  
6. Ne combine pas tout avant d’avoir un résultat isolé.

## Ce que tu dois observer

- Plusieurs échecs ont des causes différentes.
- Le training n’est pas le seul moyen d’ajouter de l’information.
- Une combinaison CPT→SFT peut être logique, mais seulement si chaque étape répond à un manque identifié.

## À ne pas confondre

- RAG ≠ fine-tuning.
- CPT ≠ “charger des documents dans le modèle” de façon garantie et récupérable.
- SFT ≠ base de connaissances fiable.

## Erreurs fréquentes

- Choisir une méthode parce qu’elle est populaire.
- Empiler CPT+SFT+DPO sans baseline intermédiaire.
- Mesurer seulement la satisfaction subjective.

## Exercices

1. Pour cinq problèmes fictifs, choisis RAG/CPT/SFT et justifie.
2. Donne un exemple où CPT+SFT est cohérent.
3. Donne un exemple où RAG est préférable au training.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Diagnostiquer le type de gap.
- Choisir une méthode avec justification.
- Formuler une hypothèse mesurable.
- Éviter l’empilement d’étapes sans preuve.

## Fiche mémo

La meilleure méthode est celle qui corrige **le problème observé** avec le coût et le risque appropriés.

## Lien avec le module suivant

Le module 27 change de signal : au lieu d’une réponse cible unique, on apprend à partir de préférences `chosen > rejected`.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
