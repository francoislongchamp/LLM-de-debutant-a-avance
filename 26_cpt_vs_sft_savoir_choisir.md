# Module 26 — Choisir une intervention à partir des erreurs observées

## 1. Ce que tu vas apprendre

CPT, SFT, récupération documentaire et apprentissage de préférences répondent à des problèmes différents. Les empiler ne remplace pas un diagnostic.

Tu vas écrire des hypothèses testables, distinguer information manquante et mauvais comportement, puis choisir une première expérience avec une mesure qui pourrait réfuter ton idée.

**Prérequis :** modules 9 à 14 et 25.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python pour le tableau d’audit. Aucune récupération externe ni aucun entraînement n’est effectué par le laboratoire.

## 2. Décrire un échec sans nommer déjà le remède

« Il me faut un LoRA » est une solution supposée. « Sur 20 questions avec la procédure fournie, le modèle ne respecte pas le schéma de réponse dans 8 cas » est une observation.

Commence par conserver : entrée exacte, réponse, réponse attendue ou grille, source du contexte, catégorie d’échec et conditions de génération.

| Type de problème | Question de diagnostic |
| --- | --- |
| Information absente ou récente | L’information utile est-elle fournie et accessible ? |
| Contexte mal récupéré | Les bons passages sont-ils présents, complets et bien placés ? |
| Distribution spécialisée | Le modèle interprète-t-il correctement jargon, syntaxe et relations ? |
| Comportement/format | Réussit-il le fond mais ignore-t-il une consigne stable ? |
| Préférence | Plusieurs sorties sont-elles valides, mais certaines préférables selon une grille ? |

Une erreur peut avoir plusieurs causes. Un diagnostic est une hypothèse à tester, pas une étiquette produite automatiquement par le modèle.

## 3. Ce que chaque intervention change

| Intervention | Ce qui change principalement | Première mesure utile |
| --- | --- | --- |
| Prompt/exemples en contexte | Entrée au moment de l’inférence | Respect des consignes sur prompts réservés |
| RAG | Documents récupérés et fournis en contexte | Qualité de récupération puis réponse fondée sur ces documents |
| SFT | Paramètres adaptés à des sorties supervisées | Comportement et exactitude sur nouvelles instructions |
| CPT | Modélisation de la distribution documentaire | Loss domaine et tâches concrètes de domaine |
| Préférences/DPO | Ordre relatif entre réponses selon des annotations | Paires réservées et benchmark général |

**RAG**, *Retrieval-Augmented Generation*, ajoute une étape de récupération avant la génération. Les poids peuvent rester identiques. Il faut pourtant évaluer la récupération, le contexte produit et l’utilisation de ce contexte.

Une contrainte de décodage peut aider pour un schéma strict lorsque l’outil la prend en charge. Cela ne rend pas automatiquement correctes les valeurs produites.

## 4. Trois contre-tests utiles

### Test A — Fournir explicitement l’information

Si le modèle échoue sans la procédure récente mais réussit lorsqu’elle est fournie dans le prompt, une récupération documentaire peut être une première hypothèse utile.

S’il échoue même avec le bon passage, améliorer seulement le moteur de recherche ne suffit probablement pas. Vérifie instruction, lisibilité, longueur, conflit entre sources et compétence requise.

### Test B — Donner un exemple de format

Si un exemple en contexte suffit à produire le format attendu, teste cette solution avant un entraînement. Si le besoin reste fréquent et stable, un SFT peut être étudié avec un benchmark de format **et de contenu**.

### Test C — Mesurer le domaine sans confondre lecture et réponse

Une perte élevée sur des documents spécialisés peut motiver une étude CPT. Elle ne démontre pas seule que CPT résoudra des questions de raisonnement ou de restitution factuelle. Ajoute des tâches concrètes et un contrôle des capacités générales.

## 5. Laboratoire autonome — Faire l’inventaire des hypothèses

Les observations suivantes sont fictives. Le programme les regroupe ; il ne découvre pas leur cause et ne lance pas les interventions proposées.

```python
from collections import Counter

cases = [
    {'id': 'a', 'evidence': 'Reussit quand la procedure recente est fournie.',
     'hypothesis': 'information', 'first_test': 'contexte pertinent fourni puis recuperation'},
    {'id': 'b', 'evidence': 'Bonne valeur mais sortie en prose au lieu de JSON.',
     'hypothesis': 'format', 'first_test': 'consigne et exemple de schema sur validation'},
    {'id': 'c', 'evidence': 'Confond encore le jargon avec le glossaire dans le contexte.',
     'hypothesis': 'domaine', 'first_test': 'diagnostic de lecture puis pilote CPT ou SFT cible'},
    {'id': 'd', 'evidence': 'Deux reponses correctes ; une est plus claire selon la grille.',
     'hypothesis': 'preference', 'first_test': 'annotation de paires et accord des relecteurs'},
    {'id': 'e', 'evidence': 'Le passage recupere concerne une autre version de la procedure.',
     'hypothesis': 'recuperation', 'first_test': 'mesurer pertinence et version des sources'},
]
print('Types observes :', dict(Counter(case['hypothesis'] for case in cases)))
for case in cases:
    print(case['id'], '\n preuve :', case['evidence'], '\n premier test :', case['first_test'])

# Exemple de critère de décision fixé avant une expérience fictive.
required_content = 0.90
required_format = 0.95
pilots = [
    {'name': 'prompt', 'content': 0.91, 'format': 0.92, 'cost_units': 1},
    {'name': 'sft', 'content': 0.90, 'format': 0.97, 'cost_units': 4},
    {'name': 'cpt', 'content': 0.93, 'format': 0.88, 'cost_units': 8},
]
acceptable = [pilot for pilot in pilots
              if pilot['content'] >= required_content and pilot['format'] >= required_format]
print('Acceptables selon ces seuils fictifs :', [pilot['name'] for pilot in acceptable])
```

Le tableau doit retenir seulement **sft** selon ces seuils et ces scores **inventés**. Il ne constitue pas une recommandation mesurée pour ton projet.

Le pilote CPT a ici un meilleur score de contenu mais manque le seuil de format. Le prompt est moins coûteux mais n’atteint pas non plus la contrainte. Cela montre pourquoi il faut définir plusieurs exigences plutôt qu’un seul nombre de gain par coût.

## 6. Rédiger une hypothèse réfutable

Une fiche d’expérience pourrait dire :

> Sur des procédures fournies en contexte, un SFT de format devrait faire passer le respect du schéma de 70 % à au moins 95 %, sans réduire l’exactitude de plus de deux points sur la validation fixée.

Elle indique données disponibles, intervention, mesures et seuils. Un résultat qui échoue aux seuils remet l’hypothèse en question, même si quelques sorties sont jolies.

Les seuils doivent venir de l’usage et de la fiabilité de mesure ; 95 % sur vingt cas reste sensible à un exemple. Conserve les effectifs et les types d’erreurs.

## 7. Quand combiner des méthodes ?

CPT→SFT peut être cohérent si le CPT améliore une modélisation de domaine utile et que le SFT apporte un comportement encore manquant. RAG+SFT peut combiner information actualisable et comportement appris.

Compare des étapes intermédiaires : base, base+première intervention, puis combinaison. Si tu changes simultanément données, objectif, retrieval et décodage, l’origine du gain devient difficile à attribuer.

Le coût inclut calcul, annotation, intégration, maintenance et mises à jour. Une information changeant souvent peut rendre coûteux un entraînement répété ; un système de récupération a aussi ses propres coûts et erreurs.

## 8. Projet guidé — Auditer vingt erreurs réelles

1. Sélectionne vingt erreurs de la base sans ne garder que les cas favorables à ta méthode préférée.
2. Note les preuves disponibles et les causes possibles, avec une catégorie « incertain » si nécessaire.
3. Fais les contre-tests A/B/C sur quelques cas représentatifs.
4. Choisis une intervention principale et une mesure indépendante.
5. Écris ce qui réfuterait ton hypothèse.
6. Mesure une baseline, exécute le pilote et conserve les régressions.
7. Ajoute une seconde étape seulement si elle répond à un échec restant identifié.

Ton livrable est une décision argumentée, pas obligatoirement un modèle entraîné.

## 9. Exercices avec corrigés

### Exercice 1 — Information récente

Le modèle ignore une procédure modifiée hier et réussit quand le bon passage est fourni. Quel premier test envisager ?

<details>
<summary>Corrigé</summary>

Tester une récupération du bon passage, sa version et son utilisation dans la réponse. Cela soutient une hypothèse RAG, sans garantir que tout le système réussira.

</details>

### Exercice 2 — Format strict

Le contenu est correct mais le JSON est invalide. CPT est-il une conséquence nécessaire ?

<details>
<summary>Corrigé</summary>

Non. Tester consigne, exemples, validation ou décodage contraint si disponible, puis SFT de comportement si le besoin et les mesures le justifient. Le texte brut ne montre pas forcément le format attendu.

</details>

### Exercice 3 — Préférences

Deux réponses sont correctes mais l’une est plus claire. Pourquoi la grille d’annotation compte-t-elle ?

<details>
<summary>Corrigé</summary>

« Plus claire » doit être défini pour produire des paires cohérentes. Sinon les annotations peuvent refléter des goûts contradictoires ou un raccourci comme la longueur.

</details>

### Exercice 4 — Empilement

Tu appliques CPT, SFT et DPO puis le score monte. Peux-tu dire quelle étape a aidé ?

<details>
<summary>Corrigé</summary>

Pas sans comparaisons intermédiaires et conditions compatibles. Il faut mesurer les contributions et les coûts des étapes, avec leurs régressions éventuelles.

</details>

## 10. Validation — avant de continuer

- [ ] Je décris un échec observable avant de choisir une méthode.
- [ ] Je distingue absence d’information, récupération, domaine, format et préférence.
- [ ] Je peux proposer un contre-test avec contexte ou exemple fourni.
- [ ] J’écris une hypothèse avec critères de réussite et réfutation.
- [ ] Je distingue les scores fictifs du laboratoire de preuves pour mon projet.
- [ ] Je sais évaluer une combinaison par étapes.

## Fiche mémo

```text
Observer → formuler plusieurs causes → contre-tester → choisir un pilote → mesurer.
CPT, SFT et RAG ne sont pas des garanties d’information fiable.
La méthode répond au problème mesuré et aux contraintes d’usage.
```

Le [module 27](27_donnees_de_preference_et_dpo.md) passe des démonstrations uniques aux comparaisons entre réponses.
