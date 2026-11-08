# Module 41 — Vérifier l’initialisation avant les premières updates

## 1. Ce que tu vas apprendre

Des poids aléatoires ne signifient pas des poids arbitrairement grands. Leur échelle et leur diversité influencent activations et gradients dès le premier batch.

Tu vas comparer trois initialisations sur un petit réseau, observer les normes et éviter d’interpréter une valeur finie comme une preuve de bon apprentissage.

**Prérequis :** modules 4, 7 et 40.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** PyTorch, CPU.

## 2. Échelle et symétrie

Une **distribution d’initialisation** définit comment les valeurs de départ sont générées. Une seed stabilise le tirage dans un environnement donné ; elle n’est pas un choix d’échelle.

Si des unités cachées ont exactement les mêmes poids et reçoivent les mêmes gradients, elles risquent de rester indifférenciées. Une initialisation aléatoire adaptée aide à briser cette symétrie.

Des poids trop grands peuvent saturer certaines activations, produire de grands logits ou des gradients mal conditionnés. Des poids trop petits peuvent aussi réduire la propagation du signal. « Le plus petit écart-type possible » n’est donc pas une règle d’initialisation.

Les règles Xavier, He et les conventions des Transformers répondent à des hypothèses différentes : activation, largeur, profondeur, résidus et normalisations comptent.

## 3. Ne pas interdire tous les zéros

Les biais sont souvent initialisés à zéro. Une norme peut commencer avec poids 1 et biais 0. La branche B de LoRA peut être nulle au départ tout en laissant A aléatoire et une base utile.

Le problème de symétrie ne se résume donc pas à « aucun zéro autorisé ». Notre laboratoire met **les deux matrices** d’un réseau sans biais à zéro pour rendre visible un cas qui bloque les gradients utiles.

## 4. Une référence pour la loss initiale

Sur V classes équiprobables, la cross-entropy d’une cible unique vaut `ln(V)`. Pour V=4 : environ **1.3863**.

Un modèle aléatoire n’est pas nécessairement exactement uniforme. S’il donne une forte probabilité à de mauvaises cibles, sa loss peut être plus grande. Une cross-entropy stable peut aussi rester finie pour de grands logits : « grand » n’implique pas automatiquement NaN.

## 5. Laboratoire autonome — Zéro, petite normale et grande normale

Nous utilisons Tanh pour observer un exemple de saturation. Les valeurs de sortie sont des scores de quatre classes ; ce réseau n’est pas un Transformer.

```python
import torch
from torch import nn

torch.manual_seed(42)
x = torch.randn(32, 16)
targets = torch.arange(32) % 4
reports = []

for name, scale in [('zero', 0.0), ('small', 0.02), ('large', 20.0)]:
    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(16, 32, bias=False), nn.Tanh(),
                          nn.Linear(32, 4, bias=False))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=scale)
    hidden = model[1](model[0](x))
    logits = model[2](hidden)
    loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()
    grad_norms = [parameter.grad.norm().item() for parameter in model.parameters()]
    report = {'name': name, 'loss': loss.item(), 'finite': torch.isfinite(loss).item(),
              'logits_abs_max': logits.abs().max().item(),
              'hidden_saturated_fraction': (hidden.abs() > 0.99).float().mean().item(),
              'gradient_norms': grad_norms}
    reports.append(report)
    print(report)
```

Pour les poids nuls, les logits et gradients des deux matrices doivent être nuls, tandis que la loss vaut environ **ln(4)**. Avoir une loss et un graphe valides ne suffit pas à recevoir un signal de mise à jour utile.

La petite normale donne des scores proches de zéro et des gradients non nuls. La grande normale sature une forte part des hidden states et produit des scores plus grands. Observe les nombres au lieu d’attendre obligatoirement un NaN.

## 6. Inspecter un Transformer réel de l’expérience

Commence par l’initialisation prévue par son implémentation. Liste quelques matrices, leur moyenne, écart-type, minimum/maximum et dtype. Compare les normes de logits et gradients après un vrai batch tokenisé.

Ne réinitialise pas aveuglément toutes les valeurs d’un modèle chargé : tu détruirais ses poids appris. Une expérience d’initialisation concerne un modèle construit pour démarrer from scratch.

Les paramètres de normalisation, biais et projections peuvent avoir des règles différentes. Un résumé global de tous les poids mélange ces rôles.

## 7. Finite checks et premiers steps

Contrôle successivement : entrées/labels, logits, loss, gradients et paramètres après l’update. La première étape non finie aide à localiser la cause.

Si tout est fini mais la loss ne descend pas sur un seul batch, vérifie aussi l’optimizer, le LR réellement utilisé et les labels. Une statistique d’initialisation inhabituelle n’est pas automatiquement l’unique cause.

Pour comparer des échelles, garde architecture, batch, loss, seed et optimizer fixes. Évalue plusieurs répétitions si une différence petite dépend fortement du tirage.

## 8. Expériences guidées

1. Essaie des écarts-types intermédiaires et représente saturation, loss et normes de gradients dans un tableau.
2. Remplace Tanh par ReLU. Explique pourquoi l’histoire de saturation et les règles d’échelle ne sont plus exactement les mêmes.
3. Sur le modèle du module 40, mesure la loss initiale et compare à ln(V), sans exiger une égalité exacte.
4. Fais quelques dizaines de steps sur un batch fixe ; logge les statistiques au début et à la fin.

## 9. Exercices avec corrigés

### Exercice 1 — Loss uniforme

V=100 classes avec une distribution uniforme : quelle loss approximative ?

<details>
<summary>Corrigé</summary>

`ln(100) ≈ 4.6052` nats par cible. Ce repère suppose une distribution uniforme et l’objectif de cible unique décrit.

</details>

### Exercice 2 — Finie mais bloquée

Pourquoi la configuration zéro a-t-elle une loss finie mais aucun gradient utile dans cette miniature ?

<details>
<summary>Corrigé</summary>

Les entrées de la dernière matrice sont nulles et la rétropropagation vers la première est multipliée par une dernière matrice nulle. Les deux matrices restent sans signal malgré une erreur de classification.

</details>

### Exercice 3 — LoRA

B nulle dans LoRA contredit-elle nécessairement cette observation ?

<details>
<summary>Corrigé</summary>

Non. Le montage diffère : A peut être aléatoire et B recevoir un premier gradient, sur une base déjà utile. Une seule matrice nulle n’équivaut pas à mettre toutes les matrices d’un réseau à zéro.

</details>

### Exercice 4 — Grands scores

Des logits de grande amplitude doivent-ils toujours faire échouer `cross_entropy` ?

<details>
<summary>Corrigé</summary>

Non. Son calcul stable peut rester fini. Il faut inspecter amplitude, confiance, gradients et dynamique réelle plutôt que déduire un NaN d’un seul seuil.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue seed, échelle et règle adaptée à l’architecture.
- [ ] Je retrouve loss ln(4) et gradients nuls du cas zéro.
- [ ] Je peux expliquer les limites de ce cas pédagogique.
- [ ] Je contrôle valeurs et gradients avant puis après une update.
- [ ] Je ne réinitialise pas un modèle préentraîné par erreur.

## Fiche mémo

```text
Statistiques des poids → forward réel → loss → gradients → premier step.
Fini ne signifie pas apprenable ; grand ne signifie pas automatiquement non fini.
```

Le [module 42](42_ecrire_une_boucle_de_pre_entrainement.md) assemble ces vérifications dans la boucle causale.
