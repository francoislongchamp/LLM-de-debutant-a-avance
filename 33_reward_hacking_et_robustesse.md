# Module 33 — Chercher les réponses qui trompent le reward

## 1. Ce que tu vas apprendre

Une policy peut recevoir une excellente note sans satisfaire l’intention du concepteur. Ce décalage est souvent appelé **reward hacking** ou *specification gaming*.

Tu vas fabriquer des contre-exemples, comparer un proxy à un critère indépendant et décider ce qu’il faut revoir avant de poursuivre l’optimisation.

**Prérequis :** modules 29 à 32.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python uniquement ; pas d’exécution de code généré.

## 2. Proxy et objectif réel

Un **proxy** est une mesure indirecte : longueur pour complétude, mots-clés pour pertinence, style assuré pour exactitude.

Ces indices peuvent être corrélés à de bonnes réponses dans un échantillon initial. Les optimiser fortement peut favoriser des sorties qui conservent l’indice et perdent la qualité.

Exemples :

| Proxy récompensé | Contre-exemple |
| --- | --- |
| Mentionner `306` | « La réponse n’est pas 306 » |
| Beaucoup de caractères | Répéter une phrase sans information |
| JSON valide | Objet correct syntaxiquement mais valeur fausse |
| Ton confiant | Affirmer une erreur avec assurance |

La référence à la loi de Goodhart décrit ce type de risque ; ce n’est pas une formule qui prouve qu’un proxy échouera dans chaque expérience.

## 3. Distinguer score de train et mesure indépendante

Une **évaluation indépendante** ne doit pas être simplement le même reward renommé. Elle peut être un autre test de réussite, un audit humain à l’aveugle ou une vérification fonctionnelle adaptée.

Elle reste imparfaite et peut devenir un outil de développement si tu la consultes sans cesse. Préserve donc aussi un test final et versionne les corrections de reward.

Reward en hausse et qualité indépendante en baisse signalent une divergence à examiner ; ils ne suffisent pas à identifier seuls la cause exacte.

## 4. Laboratoire autonome — Un scorer de mots-clés attaqué

La colonne `expected_success` contient ici une annotation déterminée à partir du contrat « entier seul, exactement 306 ». Ce sont des cas construits pour tester le proxy, pas une mesure d’un modèle réel.

```python

def proxy(answer):
    return float('306' in answer)


def strict_integer_contract(answer):
    return float(answer.strip() == '306')


cases = [
    {'answer': '306', 'expected_success': 1},
    {'answer': '308', 'expected_success': 0},
    {'answer': 'La reponse n est pas 306.', 'expected_success': 0},
    {'answer': '306 308 309', 'expected_success': 0},
    {'answer': 'x306x', 'expected_success': 0},
    {'answer': '306 ' , 'expected_success': 1},
]
for row in cases:
    observed = proxy(row['answer'])
    checked = strict_integer_contract(row['answer'])
    assert checked == row['expected_success']
    print(repr(row['answer']), 'proxy=', observed, 'contrat=', checked)

false_acceptances = [row['answer'] for row in cases
                     if proxy(row['answer']) == 1 and row['expected_success'] == 0]
print('Fausses acceptations du proxy :', len(false_acceptances))
print('Score moyen proxy :', sum(proxy(row['answer']) for row in cases) / len(cases))
print('Reussite selon le contrat :', sum(row['expected_success'] for row in cases) / len(cases))
```

Tu dois retrouver **3 fausses acceptations**, un score proxy de **5/6**, contre **2/6** de réussites selon le contrat.

Le correcteur strict résout ces cas pour ce format particulier. Il ne remplace pas le parser JSON du module 31 ni tous les tests d’une autre tâche.

Le scorer faible n’est pas « conscient de tricher ». La comparaison montre simplement un objectif numérique qui accepte des événements indésirables.

## 5. Construire un audit adversarial utile

Commence par écrire des familles de défauts : réponses multiples, bonnes valeurs dans une négation, texte parasite, type incorrect, répétitions, clés ambiguës et sorties tronquées.

Pour chaque cas, conserve : sortie brute, décision attendue, score reçu, règle violée et version du correcteur. Un test qui échoue devient une information de développement ; ajoute d’autres cas réservés pour ne pas juger la robustesse uniquement sur ceux utilisés pour réparer.

Inspecte aussi les meilleures sorties du run, pas seulement les échecs. Un exploit peut être concentré parmi les rewards maximaux et rester peu visible dans une moyenne globale.

## 6. Réviser sans déplacer silencieusement l’expérience

Si tu changes la reward au milieu d’un run, la courbe avant/après ne représente plus exactement le même objectif. Note le point de changement, garde l’ancienne version et rescrore des sorties archivées sous les deux conventions si nécessaire.

Pour une comparaison propre, on peut reprendre depuis un checkpoint identifié et lancer deux branches avec leurs règles explicites. Ne présente pas leur différence comme provenant uniquement d’un LR si la reward a aussi changé.

Ajouter une nouvelle pénalité peut créer un autre raccourci. Les composantes du module 32 doivent rester observables et les contraintes être testées.

## 7. Prolongement avec une policy

1. Mesure le taux initial de succès du reward et du critère indépendant sur les mêmes sorties.
2. Lance un petit entraînement avec la règle auditée.
3. Conserve des échantillons de différents niveaux de reward et de différentes longueurs.
4. Compare l’accord entre score optimisé et réussite réelle au fil des updates.
5. Si cet accord se dégrade, inspecte les nouvelles formes de réponse avant d’augmenter le compute.

Un reward appris peut aussi être exploité : la policy produit des réponses éloignées des exemples sur lesquels le juge était fiable.

## 8. Exercices guidés

- Crée dix textes contenant la bonne réponse sans respecter le contrat ; vérifie les fausses acceptations.
- Prends un bonus de longueur et construis une sortie répétitive qui atteint le maximum sans répondre.
- Répare une famille de cas et conserve une autre famille hors des modifications. Compare les deux taux d’erreur.
- Imagine une tâche de code : distingue tests de format, tests fonctionnels et isolation de l’exécution. Ce module ne propose pas de lancer du code généré dans ton environnement de travail.

## 9. Exercices avec corrigés

### Exercice 1 — Une note élevée

Un JSON valide reçoit 1, même si la réponse est fausse. Quel objectif a été optimisé ?

<details>
<summary>Corrigé</summary>

La conformité syntaxique ou structurelle définie par ce correcteur. Il manque un contrôle de contenu pour affirmer une réussite de la tâche.

</details>

### Exercice 2 — Reward et qualité divergent

Reward train monte, audit indépendant baisse. Faut-il immédiatement augmenter le nombre d’updates ?

<details>
<summary>Corrigé</summary>

Non. Il faut chercher les différences entre proxy et objectif, examiner les hauts rewards et les changements de distribution. Davantage d’optimisation peut accentuer le problème.

</details>

### Exercice 3 — Réparation et test

Pourquoi ne pas mesurer la robustesse uniquement sur les cas qui ont servi à corriger le parser ?

<details>
<summary>Corrigé</summary>

Ils sont devenus des données de développement. Il faut aussi des cas réservés et de nouvelles sorties pour vérifier que la correction dépasse ces exemples précis.

</details>

### Exercice 4 — Versionnement

Le score passe de 0.6 à 0.9 après un changement de reward. Peut-on annoncer un gain de réussite ?

<details>
<summary>Corrigé</summary>

Pas directement : les deux nombres peuvent noter des critères différents. Rescore selon une convention commune et utilise une mesure de réussite indépendante.

</details>

## 10. Validation — avant de continuer

- [ ] Je décris le proxy et l’objectif qu’il est censé représenter.
- [ ] Je retrouve les trois fausses acceptations du laboratoire.
- [ ] Je construis des tests adversariaux avec décision attendue.
- [ ] Je distingue réparation sur développement et test réservé.
- [ ] Je versionne une modification et conserve les sorties brutes.

## Fiche mémo

```text
Chercher ce qui maximise le score sans accomplir la tâche.
Auditer les extrêmes et comparer à un critère distinct.
Un correcteur réparé sur quelques exemples reste à tester ailleurs.
```

Le [module 34](34_grpo.md) utilise des groupes de réponses notées pour optimiser une policy.
