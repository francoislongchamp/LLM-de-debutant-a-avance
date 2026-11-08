# Module 40 — Dimensionner un petit Transformer cohérent

## 1. Ce que tu vas apprendre

La taille du vocabulaire ne suffit pas à définir le réseau. Largeur, profondeur, têtes, MLP, contexte et partage des poids déterminent ensemble ses formes et son coût.

Tu vas calculer les paramètres d’une petite architecture GPT-2, vérifier ce calcul par PyTorch et tester un forward.

**Prérequis :** modules 4, 5 et 37 à 39.

**Temps indicatif :** 90 à 150 minutes.

**Matériel :** PyTorch et Transformers, CPU, sans poids préentraînés.

## 2. Nommer les dimensions sans collision

| Symbole | Sens |
| --- | --- |
| V | Nombre d’IDs de vocabulaire |
| H | Largeur des hidden states |
| D | Nombre de blocs |
| P | Nombre de positions apprises pour ce GPT-2 |
| h | Nombre de têtes dans cette attention standard |
| M | Largeur intermédiaire du MLP |
| B,T | Batch et longueur de la séquence courante |

Pour ce GPT-2, H doit être divisible par h et la dimension par tête vaut H/h. Ce n’est pas une règle universelle pour toute architecture où `head_dim` peut être explicite et Q/K/V avoir des largeurs différentes.

La longueur courante T doit respecter le budget P. H décrit les nombres à chaque position, pas le nombre de tokens visibles.

## 3. Dessiner le réseau choisi

```text
IDs [B,T]
→ embedding tokens [V,H] + positions apprises [P,H]
→ D blocs : attention QKV/O + MLP + normes/résidus
→ norme finale
→ projection vocabulaire liée à l’embedding
→ logits [B,T,V]
```

Notre GPT-2 utilise des positions apprises, pas RoPE. Son MLP de travail a M=4H et deux couches affines. Une architecture à MLP gated demanderait un autre comptage.

Le **weight tying** partage ici l’embedding et le poids de sortie. Il faut compter les paramètres uniques, pas additionner deux noms qui pointent au même objet.

## 4. Compter un bloc

Dans cette configuration précise :

| Partie | Poids et biais |
| --- | --- |
| QKV et sortie attention | 4H² + 4H |
| MLP H→4H→H | 8H² + 5H |
| Deux LayerNorm | 4H |
| Total par bloc | **12H² + 13H** |

Avec embeddings et norme finale, le total unique est :

\[
N = VH + PH + D(12H^2+13H) + 2H
\]

Pour V=320, P=128, H=32 et D=2 : **39 808 paramètres**.

Cette formule correspond à notre GPT-2 avec tête liée, biais et MLP décrits ci-dessus. Ce n’est pas une formule universelle des Transformers.

## 5. Laboratoire autonome — Théorie et mesure

```python
import torch
from transformers import GPT2Config, GPT2LMHeadModel

vocab_size = 320
positions = 128
measurements = []
for hidden, depth, heads in [(24, 1, 4), (32, 2, 4), (48, 2, 4)]:
    assert hidden % heads == 0
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=vocab_size, n_positions=positions, n_embd=hidden,
        n_layer=depth, n_head=heads, n_inner=4 * hidden,
        tie_word_embeddings=True, bos_token_id=None, eos_token_id=2, pad_token_id=0,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
    ))
    theoretical = vocab_size * hidden + positions * hidden + depth * (12 * hidden**2 + 13 * hidden) + 2 * hidden
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert actual == theoretical
    model.eval()
    ids = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    shared = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    print('H=', hidden, 'D=', depth, 'par tete=', hidden // heads,
          'parametres=', actual, 'theorie=', theoretical,
          'poids lies=', shared, 'logits=', tuple(logits.shape))
    measurements.append(actual)
    del model
```

Tu dois retrouver **39 808** pour la deuxième configuration, une égalité théorie/mesure et des logits **[1,4,320]** pour les trois modèles.

Changer seulement le nombre de têtes à H fixe ne multiplie pas automatiquement les poids par ce nombre dans cette architecture : les mêmes dimensions globales sont réparties autrement.

## 6. Coût du contexte et coût des poids

Une attention dense conventionnelle traite des relations entre T positions et peut matérialiser des scores de forme `[B,h,T,T]`. Passer de T à 4T multiplie ce nombre de scores par 16.

Des kernels peuvent éviter de stocker toute cette matrice ; ils ne rendent pas tous les calculs indépendants de la longueur. Dans ce GPT-2 à positions apprises, augmenter P augmente aussi la table de positions, même si les autres matrices restent de même taille.

Un modèle qui passe sur T=4 ne valide donc pas sa mémoire sur T=128 avec backward et optimizer.

## 7. Expériences guidées

1. Double V et vérifie l’ajout VH lié à l’embedding. Compare le nombre de tokens d’un même texte avec le tokenizer choisi, pas seulement les poids.
2. Double D à H fixe, puis H à D fixe. Observe les termes linéaires et quadratiques de la formule.
3. Compare T=16,32,64 à architecture fixe ; mesure le step complet avec le protocole du module 50.
4. Désactive le tying dans une copie. Vérifie que la tête indépendante ajoute V×H paramètres pour ce cas sans biais de sortie.

## 8. Choisir la plus petite expérience informative

Fixe d’abord ce que tu dois observer : boucle correcte, reprise, effet du corpus ou comparaison architecturale. Un grand modèle qui rend chaque essai lent n’est pas forcément un meilleur outil d’apprentissage.

Conserve config complète, tokenizer/révision, nombre réel de paramètres, contexte de travail, précision et résultat du smoke test. Un nom comme « tiny » ne remplace aucun de ces nombres.

## 9. Exercices avec corrigés

### Exercice 1 — Dimension d’une tête

H=768, h=12 dans une attention standard partageant H : quelle dimension ?

<details>
<summary>Corrigé</summary>

**64**. Cette division s’applique au design précisé, pas automatiquement à toutes les architectures.

</details>

### Exercice 2 — Embedding

V=32 000 et H=768 : combien de valeurs d’entrée ?

<details>
<summary>Corrigé</summary>

**24 576 000** valeurs. La sortie peut être liée ou ajouter ses propres paramètres.

</details>

### Exercice 3 — Contexte

Pourquoi T multiplié par quatre peut-il coûter plus que quatre fois pour l’attention dense ?

<details>
<summary>Corrigé</summary>

Le nombre de relations entre positions est proportionnel à T², donc multiplié par 16. La mémoire réellement matérialisée dépend aussi des kernels.

</details>

### Exercice 4 — Formule exacte

Peut-on réutiliser 12H²+13H pour un bloc avec RMSNorm et MLP gated sans vérifier ?

<details>
<summary>Corrigé</summary>

Non. Le nombre de projections, leurs dimensions, les biais et les normes changent le compte. La formule est liée aux composants de notre GPT-2.

</details>

## 10. Validation — avant de continuer

- [ ] Je distingue V,H,D,P,h,M et B,T.
- [ ] Je retrouve le compte de la configuration à 39 808 paramètres.
- [ ] Je sais quelles hypothèses rendent la formule exacte.
- [ ] Je teste un forward et le partage de poids.
- [ ] Je distingue poids, relations d’attention et mémoire du step complet.

## Fiche mémo

```text
Choisir les composants → dessiner les formes → compter → instancier → mesurer.
Une architecture est un ensemble d’hypothèses, pas une seule taille annoncée.
```

Le [module 41](41_initialisation_aleatoire_et_stabilite.md) examine son état initial.
