# Module 37 — Planifier un pré-entraînement depuis des poids aléatoires

## 1. Ce que tu vas apprendre

Tu as déjà utilisé des miniatures aléatoires pour comprendre les trainers. Ici commence un parcours où tokenizer, corpus, architecture et optimisation sont conçus ensemble pour apprendre un modèle de langage de base.

Tu vas fixer des budgets, instancier une petite architecture et distinguer une expérience pédagogique d’un modèle utilisable comme assistant.

**Prérequis :** modules 1 à 8 et 25. La série 38 à 42 détaillera les composants ; le dimensionnement architectural sera approfondi au module 40.

**Temps indicatif :** 90 à 150 minutes pour la planification et le pilote, hors entraînement cible.

**Matériel :** PyTorch et Transformers sur CPU pour les configurations minuscules.

## 2. Que signifie from scratch ?

Le modèle démarre avec une initialisation numérique et sans les poids linguistiques d’une base préentraînée. `Model(config)` crée cet état ; `from_pretrained(...)` recharge un état appris ou sauvegardé.

Le tokenizer peut être nouveau ou choisi parmi des outils existants compatibles avec ce projet. Réutiliser un tokenizer ne donne pas les connaissances de son ancien modèle à de nouveaux embeddings.

Un **base model** appris sur du texte n’est pas automatiquement un assistant conversationnel. Les démonstrations SFT et autres étapes éventuelles restent des expériences séparées.

## 3. Les budgets à annoncer ensemble

| Budget | Question |
| --- | --- |
| Paramètres | Combien de poids et de paramètres optimisés ? |
| Données | Combien de documents et de tokens distincts, de quelles sources ? |
| Expositions | Combien de positions/cibles présentées, répétitions comprises ? |
| Contexte | Combien de positions par séquence ? |
| Calcul et temps | Quel débit mesuré, quel temps disponible ? |
| Mémoire et stockage | Le step complet et les checkpoints tiennent-ils ? |

Doubler la largeur peut augmenter plusieurs matrices approximativement au carré. Doubler la séquence accroît les activations et une attention dense peut avoir un coût quadratique en longueur. Les décisions ne sont pas indépendantes.

Une approximation dense souvent utilisée est `FLOPs ≈ 6×N×T` pour un training de Transformer, sous des conventions simplifiées. Attention, projections de vocabulaire, architecture, séquences et implémentations peuvent changer l’approximation. Ce n’est pas une promesse de durée ni une loi de dimensionnement suffisante.

## 4. Refaire un budget de tokens

Batch global de 16 séquences pleines de 512 positions : **8 192 positions d’entrée par update**.

Pour viser au moins 20M positions avec des updates pleines :

```text
ceil(20 000 000 / 8192) = 2442 updates
2442 × 8192 = 20 004 864 positions
```

Les cibles causales ne sont pas nécessairement toutes ces positions : dans des lignes séparées sans cible suivante, chaque ligne de L IDs fournit L−1 cibles. Padding et masques retirent encore certaines cibles.

Annonce si ton budget compte entrées, tokens non PAD ou cibles de loss. L’epoch seule décrit mal les comparaisons si le corpus est répété ou mélangé par sampling.

## 5. Laboratoire autonome — Créer trois tailles et compter

Ces modèles sont volontairement très petits : leur intérêt est la rapidité de vérification, pas leur performance linguistique.

```python
import math
import torch
from transformers import GPT2Config, GPT2LMHeadModel

torch.manual_seed(42)
counts = []
for hidden, layers in [(24, 1), (32, 2), (48, 2)]:
    config = GPT2Config(vocab_size=256, n_positions=64, n_embd=hidden,
                        n_layer=layers, n_head=4, bos_token_id=None,
                        eos_token_id=1, pad_token_id=0, use_cache=False)
    model = GPT2LMHeadModel(config)
    parameters = sum(p.numel() for p in model.parameters())
    counts.append(parameters)
    ids = torch.randint(2, 256, (1, 16))
    with torch.no_grad():
        output = model(input_ids=ids)
    print('H=', hidden, 'blocs=', layers, 'parametres=', parameters,
          'poids FP32 octets=', sum(p.numel() * p.element_size() for p in model.parameters()),
          'logits=', tuple(output.logits.shape))
    del model

positions_per_update = 16 * 512
updates = math.ceil(20_000_000 / positions_per_update)
print('Updates :', updates, 'positions :', updates * positions_per_update)
```

Les trois sorties doivent avoir la forme **[1,16,256]** et les nombres de paramètres augmenter avec ces configurations. Le budget final doit être **2442 updates** et **20 004 864 positions**.

Un forward réussi ne valide ni les gradients, ni la reprise, ni le tokenizer, ni l’apprentissage. Il faut encore franchir les étapes suivantes.

## 6. La porte avant un long run

Avant d’augmenter taille ou durée, exige :

1. Tokenizer figé, couverture et round-trip vérifiés.
2. Corpus et splits versionnés, sans recouvrement connu indésirable.
3. Batch correctement formé, loss et gradients finis.
4. Mémorisation possible d’un seul batch, dans un objectif compatible.
5. Validation réservée calculée avec la bonne pondération.
6. Sauvegarde puis reprise réellement testée.
7. Mémoire et débit du step complet mesurés avec marge.

Si un point échoue, le grand run reproduira plus cher un problème déjà observable. La série 38 à 52 construit ces vérifications.

## 7. Scaling laws : repères empiriques

Des relations empiriques relient paramètres, données, calcul et loss dans certains régimes. Elles aident à formuler un budget, pas à garantir un comportement de chat ni à choisir un modèle à partir d’une seule constante.

Les très petites expériences, données très répétées et architectures différentes peuvent s’écarter des régimes étudiés. Mesurer le débit et la courbe d’un pilote est plus concret qu’extrapoler aveuglément un tableau de grands modèles.

## 8. Expériences guidées

- Fixe largeur et profondeur puis double seulement le vocabulaire ; sépare le coût d’embedding et le total mesuré.
- Fixe les paramètres architecturaux et compare des forwards sur L=16,32,64. Mesure les temps sans promettre une relation idéale sur CPU.
- Rédige un budget en tokens et secondes à partir d’un débit **mesuré**, avec la convention du compteur.
- Définis ce que ton premier modèle doit démontrer : apprentissage du pipeline, et non qualité comparable à un grand assistant.

## 9. Exercices avec corrigés

### Exercice 1 — Deux départs

Quelle différence entre CPT et from scratch ?

<details>
<summary>Corrigé</summary>

CPT poursuit un objectif depuis des poids déjà appris ; from scratch construit les représentations à partir d’une initialisation. Les deux demandent des données et un protocole explicites.

</details>

### Exercice 2 — Budget exact

10 000 updates à 32×512 positions donnent combien de positions ?

<details>
<summary>Corrigé</summary>

**163 840 000 positions**. Il faut encore préciser padding, cibles et répétitions pour parler des tokens utiles ou distincts.

</details>

### Exercice 3 — Tokenizer réutilisé

Un tokenizer de qualité suffit-il à transmettre les connaissances de son modèle d’origine ?

<details>
<summary>Corrigé</summary>

Non. Il fournit une segmentation et des IDs ; de nouveaux embeddings aléatoires n’héritent pas des poids appris de l’ancien réseau.

</details>

### Exercice 4 — Choisir petit

Pourquoi un modèle de quelques dizaines de milliers de paramètres peut-il être utile ici ?

<details>
<summary>Corrigé</summary>

Il permet de vérifier rapidement les formes, pertes, gradients et sauvegardes. Sa limite linguistique ne l’empêche pas de révéler des erreurs de pipeline.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue initialisation, tokenizer et connaissances apprises.
- [ ] Je crée un modèle depuis une config et compte ses paramètres.
- [ ] Je retrouve le budget 2442 updates avec sa convention.
- [ ] Je sépare données distinctes, expositions et cibles.
- [ ] Je définis des critères avant de passer à l’échelle.

## Fiche mémo

```text
From scratch = système complet, pas seulement un constructeur de modèle.
Commencer assez petit pour tester tout le cycle.
Budgeter paramètres, données, expositions, contexte, mémoire et temps ensemble.
```

Le [module 38](38_entrainer_un_tokenizer_bpe.md) construit le tokenizer de cette chaîne.
