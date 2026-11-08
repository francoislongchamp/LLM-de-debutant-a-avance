# Module 19 — Choisir où injecter les corrections LoRA

## 1. Ce que tu vas apprendre

Le rang dit quelle correction chaque branche peut représenter. Les **target modules** disent dans quelles projections cette correction est autorisée.

Tu vas inspecter les couches avant injection, compter le coût de plusieurs couvertures et vérifier ensuite ce qui a réellement été adapté.

**Prérequis :** modules 4, 15 à 18.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** Python et PyTorch pour le laboratoire ; PEFT pour le prolongement d’injection.

## 2. Revenir au bloc Transformer

| Projections | Rôle général | Exemples de noms selon l’architecture |
| --- | --- | --- |
| Attention | Construire Q/K/V puis projeter leur combinaison | q_proj, k_proj, v_proj, o_proj |
| MLP | Transformer le vecteur de chaque position | gate_proj, up_proj, down_proj |
| Sortie vocabulaire | Produire les logits | lm_head |

Ces noms sont des conventions de certaines familles. D’autres modèles utilisent une matrice QKV fusionnée ou une classe de projection différente de `nn.Linear`.

Une liste copiée d’un modèle peut ne rien cibler dans un autre, ou cibler une couverture différente de celle imaginée. L’inspection précède la configuration.

## 3. Calculer un budget par module

Pour un ensemble de cibles :

\[
N_{LoRA,total} = \sum_i r(in_i+out_i)
\]

Le nombre de modules ne suffit pas. Une projection 4 → 12 coûte à r=2 : `2×(4+12)=32`, contre 16 pour une projection 4 → 4.

Une projection Q et une projection V n’ont pas toujours les mêmes dimensions, notamment avec certaines attentions groupées. Utilise les formes réelles.

### Exemple à un bloc

Notre bloc artificiel contient quatre projections d’attention 4 → 4, puis un MLP 4 → 12 → 4, sans gate séparée.

| Couverture, r=2 | Modules | Coût ajouté |
| --- | --- | --- |
| Q/V | 2 matrices 4→4 | 32 |
| Attention complète | 4 matrices 4→4 | 64 |
| Attention + MLP | Les précédentes + 4→12 et 12→4 | 128 |

« Même rang » ne signifie donc pas « même budget d’adaptation ».

## 4. Laboratoire autonome — Inspecter et sommer

Le réseau est une structure de démonstration ; nous n’effectuons pas de forward ni d’entraînement. Chaque bloc contient les matrices à inspecter.

```python
from torch import nn


def make_block():
    return nn.ModuleDict({
        'q_proj': nn.Linear(4, 4, bias=False),
        'k_proj': nn.Linear(4, 4, bias=False),
        'v_proj': nn.Linear(4, 4, bias=False),
        'o_proj': nn.Linear(4, 4, bias=False),
        'up_proj': nn.Linear(4, 12, bias=False),
        'down_proj': nn.Linear(12, 4, bias=False),
    })


model = nn.ModuleList([make_block() for _ in range(2)])
linears = {name: module for name, module in model.named_modules()
           if isinstance(module, nn.Linear)}
for name, module in linears.items():
    print(name, module.in_features, '->', module.out_features)

strategies = {
    'qv': {'q_proj', 'v_proj'},
    'attention': {'q_proj', 'k_proj', 'v_proj', 'o_proj'},
    'attention_mlp': {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'},
}
rank = 2
for strategy, suffixes in strategies.items():
    selected = {name: module for name, module in linears.items()
                if name.rsplit('.', 1)[-1] in suffixes}
    cost = sum(rank * (module.in_features + module.out_features)
               for module in selected.values())
    print(strategy, 'modules=', len(selected), 'parametres LoRA=', cost)
```

Avec deux blocs, tu dois retrouver : **4 modules/64 paramètres**, **8/128**, puis **12/256**.

Le script compte une intention de ciblage à l’aide de suffixes. Ce n’est pas encore la règle de correspondance complète de PEFT. Après injection, compare les noms et les A/B réelles à ce compte attendu.

## 5. Comprendre `all-linear`

Dans PEFT, `target_modules="all-linear"` vise les couches linéaires prises en charge, avec des règles adaptées aux classes de modèle ; la couche de sortie de certains modèles préentraînés est notamment exclue par le sélecteur.

Cela ne signifie pas « tous les paramètres deviennent entraînables ». Les matrices de base restent gelées dans le LoRA classique et seules les branches ou parties supplémentaires prévues sont optimisées.

Le traitement des projections fusionnées, poids partagés, couches quantifiées et architectures particulières demande de vérifier le résultat. Une inspection limitée à `isinstance(nn.Linear)` peut aussi manquer des classes linéaires spécialisées.

## 6. Choisir une comparaison honnête

Deux expériences possibles répondent à des questions différentes :

- **Rang fixe, couverture variable** : quel résultat obtient-on en donnant plus de lieux d’adaptation et souvent plus de paramètres ?
- **Budget approché fixe** : quel placement fonctionne mieux pour un nombre ajouté comparable, en ajustant éventuellement les rangs ?

Dans notre réseau artificiel à deux blocs, Q/V avec r=4 coûte 128 paramètres. Toute l’attention avec r=2 coûte aussi 128. Les corrections restent différentes : égaliser le nombre ne rend pas les architectures identiques.

Annonce ce que tu contrôles : données, steps/tokens, LR, scaling, seed, protocole d’évaluation et budget. Mesure qualité par catégorie, mémoire, temps et taille d’adapter.

## 7. Vérifier après injection

1. Liste les modules ciblés avant la transformation.
2. Construis la config et injecte une fois.
3. Liste les noms contenant A/B et leurs formes.
4. Somme leurs paramètres et examine tous les autres entraînables.
5. Fais un backward de contrôle et confirme les gradients attendus.
6. Sauvegarde cette liste avec l’expérience.

Si un nom de cible n’existe pas, corrige la configuration avant d’entraîner. Un trainer qui se construit ne prouve pas que la couverture correspond à ton intention.

## 8. Expériences guidées

1. Ajoute une `gate_proj` 4 → 12 dans chaque bloc. Recalcule le budget complet et explique pourquoi le précédent tableau n’est plus suffisant.
2. Fais varier les dimensions K/V pour imiter une couverture groupée. Recalcule Q/V à partir des formes réelles.
3. Sur le modèle du module 16, ajoute une deuxième projection et compare une seule cible à deux cibles, à rang fixe.
4. Sur un modèle réel déjà chargé, regroupe les noms de projections par suffixe puis compare les cibles disponibles à une liste trouvée dans un exemple externe.

## 9. Exercices avec corrigés

### Exercice 1 — Modules et matrices

32 blocs reçoivent chacun 7 branches LoRA. Combien de modules sont adaptés ? Combien de matrices A/B sont ajoutées ?

<details>
<summary>Corrigé</summary>

**224 modules** adaptés et **448 matrices A/B**, hors variantes ou parties supplémentaires. Le nombre de paramètres dépend encore de leurs dimensions et rangs.

</details>

### Exercice 2 — Coût de la miniature

Pourquoi attention+MLP coûte-t-il deux fois l’attention seule dans un bloc de notre exemple ?

<details>
<summary>Corrigé</summary>

L’attention coûte 64 paramètres à r=2. Les deux matrices MLP ajoutent chacune 32, soit 64 de plus : total 128. C’est un résultat de ces dimensions, pas un ratio universel.

</details>

### Exercice 3 — Budget égal

Comment comparer Q/V et toute l’attention à 128 paramètres sur les deux blocs ?

<details>
<summary>Corrigé</summary>

Q/V à r=4 et attention complète à r=2 donnent ici chacun 128 paramètres. Il faut aussi annoncer la règle d’alpha/scaling et les autres conditions.

</details>

### Exercice 4 — Sélecteur large

Pourquoi `all-linear` n’est-il pas du full FT ?

<details>
<summary>Corrigé</summary>

Il indique où ajouter des branches LoRA. Il ne rend pas toutes les grandes matrices de base directement entraînables. C’est le type de mise à jour, pas seulement le nombre de cibles, qui distingue les deux méthodes.

</details>

## 10. Validation — avant de continuer

- [ ] Je repère les projections et leurs dimensions réelles.
- [ ] Je calcule une somme de coûts, pas seulement un nombre de modules.
- [ ] Je retrouve 64/128/256 dans le laboratoire.
- [ ] Je distingue rang fixe et budget fixe.
- [ ] Je sais ce que `all-linear` veut dire et ce qu’il faut vérifier.
- [ ] Je compare l’intention de ciblage à l’injection réellement obtenue.

## Fiche mémo

```text
Inspecter les formes → choisir les cibles → calculer le budget → injecter → vérifier.
Même rang ne signifie pas même nombre de paramètres.
Une couverture plus large est une hypothèse d’amélioration à mesurer.
```

Le [module 20](20_qlora_et_quantification_4_bits.md) combine ces branches avec une base quantifiée.

## Référence

[PEFT : LoRA et sélecteurs de modules](https://huggingface.co/docs/peft/en/package_reference/lora).
