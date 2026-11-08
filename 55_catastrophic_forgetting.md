# Module 55 — Mesurer ce qu’une spécialisation fait perdre

## 1. Ce que tu vas apprendre

Une adaptation peut améliorer le domaine tout en dégradant des compétences antérieures. On parle d’**oubli catastrophique** lorsque cette dégradation est importante ; son ampleur se mesure sous un protocole défini.

Tu vas distinguer oubli et overfitting, observer un compromis contrôlé et comparer des stratégies de mitigation.

**Prérequis :** modules 13, 14, 25 et 54.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** PyTorch sur CPU pour un modèle à un paramètre.

## 2. Deux questions différentes

| Question | Mesure concernée |
| --- | --- |
| Le modèle s’adapte-t-il trop à son nouveau train ? | Généralisation au domaine : train/validation |
| Perd-il un comportement antérieur ? | Benchmark de régression avant/après |

On peut oublier une ancienne tâche sans surapprentissage manifeste du nouveau train. On peut aussi surapprendre le nouveau train sans constater une grande perte sur les capacités générales testées.

Une baisse de score ne démontre pas que toute une connaissance a disparu dans un sens absolu. Elle montre un comportement dégradé sous les entrées et réglages de la mesure ; template ou décodage peuvent aussi influencer ce comportement.

## 3. Un compromis impossible à cacher

Notre modèle prédit un seul scalaire w. L’ancienne tâche veut 1 ; la nouvelle veut −1. Une seule valeur constante ne peut satisfaire exactement ces deux objectifs contradictoires.

```text
Ancienne loss : (w−1)²
Nouvelle loss : (w+1)²
```

L’entraînement séquentiel pousse d’abord vers 1, puis vers −1. Une bonne nouvelle loss ne dit rien, à elle seule, de l’ancienne.

Le **replay** ajoute des données/termes de l’ancienne tâche pendant la spécialisation. Avec un poids ancien 0.25, l’objectif vaut `0.25(w−1)² + 0.75(w+1)²`, dont le minimum est w=−0.5.

Cela protège partiellement l’ancien objectif au prix d’une nouvelle loss moins basse. Cette miniature rend le compromis exact ; elle n’est pas un modèle de toutes les causes d’oubli dans un LLM.

## 4. Laboratoire autonome — Séquentiel ou mélange

```python
import torch


def train(start, old_weight, steps=40):
    w = torch.nn.Parameter(torch.tensor(float(start)))
    optimizer = torch.optim.SGD([w], lr=0.1)
    for _ in range(steps):
        old_loss = (w - 1).square()
        new_loss = (w + 1).square()
        loss = old_weight * old_loss + (1 - old_weight) * new_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return w.detach().item()


def evaluate(w):
    return {'w': w, 'old_loss': (w - 1)**2, 'new_loss': (w + 1)**2}


pretrained = train(start=0.0, old_weight=1.0)
sequential = train(start=pretrained, old_weight=0.0)
replay = train(start=pretrained, old_weight=0.25)
print('Avant specialisation :', evaluate(pretrained))
print('Nouvelle tache seule :', evaluate(sequential))
print('Avec melange ancien 25% :', evaluate(replay))
```

Le premier état approche **1**. Le séquentiel approche **−1**, avec une ancienne loss proche de **4**. Le mélange approche **−0.5**, avec ancienne loss proche de **2.25** et nouvelle proche de **0.25**.

Nous comparons le même nombre d’updates de spécialisation. Le mélange évalue deux termes par step : ce n’est pas une égalité de nombre d’exemples ou de compute à présenter sans précision.

## 5. Mesurer un modèle réel

Avant adaptation, fixe un benchmark domaine et plusieurs catégories à préserver. Garde prompts, contextes, template, décodage et scoring compatibles après adaptation.

Rapporte pour chaque capacité : score avant, après, delta, effectifs et coût. Fixe des seuils de perte acceptables avant de voir les résultats.

Exemple de décision hypothétique : domaine doit gagner au moins dix points et chaque catégorie générale perdre au plus deux points. Une moyenne générale seule peut masquer la perte d’une catégorie critique.

## 6. Mitigations à comparer, pas remèdes garantis

| Intervention | Hypothèse |
| --- | --- |
| Moins de steps ou LR adapté | Limiter une adaptation devenue excessive |
| Replay/mixture | Réexposer des compétences à préserver |
| PEFT/capacité contrainte | Restreindre certaines modifications |
| Régularisation vers un état | Limiter un déplacement selon la règle choisie |
| Meilleures données | Éviter contradictions et spécialisation inutilement étroite |

LoRA peut aussi produire des régressions. Le replay consomme un budget et peut réduire l’exposition au nouveau domaine. Une pénalité de proximité peut freiner un gain souhaité. Mesure le compromis avec les mêmes critères.

Si la régression vient d’un mauvais template de rechargement, changer la méthode d’entraînement ne traite pas la cause.

## 7. Suivre plusieurs checkpoints

Une capacité peut se dégrader avant la meilleure loss domaine. Compare plusieurs points d’adaptation et sélectionne sur validation selon les contraintes fixées.

Garde les poids de base pour revenir à un état connu. Un adapter désactivable facilite certaines comparaisons, mais ne prouve pas que son comportement activé conserve toutes les capacités.

Le test final doit estimer le système sélectionné ; si tu l’utilises continuellement pour corriger les régressions, reconnais son nouveau rôle de développement.

## 8. Expériences guidées

1. Fais varier le poids de replay 0,0.25,0.5,0.75. Prédis le minimum `w=2×old_weight−1` puis vérifie les losses.
2. Compare plusieurs nombres de steps et trace ancien/nouveau objectif sur le même axe.
3. Sur un petit modèle réel, compare adaptation légère et full FT avec protocole et tuning annoncés.
4. Examine cinq régressions brutes : distingue information absente, style, refus, format et erreur factuelle selon l’usage.

## 9. Exercices avec corrigés

### Exercice 1 — Minimum du mélange

Avec old_weight=0.5, quel minimum du modèle scalaire ?

<details>
<summary>Corrigé</summary>

w=0, avec une loss de 1 sur chaque tâche. La capacité constante ne peut satisfaire simultanément les cibles opposées.

</details>

### Exercice 2 — Oubli ou overfitting ?

Le nouveau domaine généralise bien, mais une ancienne capacité baisse. Est-ce exclu par l’absence d’overfitting ?

<details>
<summary>Corrigé</summary>

Non. Les deux questions sont différentes : on peut bien généraliser sur le nouveau domaine et régresser sur l’ancien.

</details>

### Exercice 3 — Seuil après coup

Pourquoi définir une perte acceptable seulement après les résultats affaiblit-il la décision ?

<details>
<summary>Corrigé</summary>

Cela permet de déplacer le critère pour accepter le modèle préféré. Un seuil préalable rend le compromis vérifiable ; toute révision doit être justifiée et versionnée.

</details>

### Exercice 4 — Replay

Une réduction d’oubli par replay est-elle gratuite ?

<details>
<summary>Corrigé</summary>

Non. Elle change les données exposées, le compute ou la part du budget dédiée au domaine. Il faut rapporter ces coûts et le nouveau compromis.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue oubli et généralisation au nouveau domaine.
- [ ] Je retrouve les trois états du modèle scalaire.
- [ ] Je mesure chaque capacité avec des conventions compatibles.
- [ ] Je fixe des seuils et rapporte les régressions par catégorie.
- [ ] Je teste une mitigation avec coût et hypothèse explicites.

## Fiche mémo

```text
Gagner sur le domaine sous contrainte de préserver les capacités utiles.
Une mitigation se juge par le compromis observé, pas par son nom.
```

Les [projets 56 à 62](56_projet_1_lora_specialise.md) assemblent maintenant ces compétences en expériences reproductibles.
