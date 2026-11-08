# Module 1 — Suivre le trajet d’un texte dans un LLM

## 1. Ce que tu vas apprendre

Quand tu écris une question à un assistant, que se passe-t-il entre ton texte et les premiers caractères de la réponse ? Ce module construit la carte du trajet. Les modules 2 à 7 ouvriront chaque partie, puis le module 8 assemblera une boucle d’entraînement.

Nous étudions ici un **modèle de langage causal** : un réseau qui attribue des scores aux prochains tokens possibles à partir des tokens déjà présents. Cette famille sert notamment à générer du texte.

À la fin, tu devrais pouvoir :

- suivre le trajet messages → tokens → scores → prochain token ;
- distinguer un identifiant, un vecteur interne et une probabilité ;
- expliquer pourquoi une réponse se construit en plusieurs étapes ;
- distinguer utiliser un modèle et modifier ses paramètres ;
- observer les logits d’un vrai modèle sans l’entraîner.

**Prérequis :** savoir lancer un fichier Python et lire une liste. Pour le laboratoire réel, savoir importer des bibliothèques.

**Temps indicatif :** deux séances de 60 à 90 minutes, exercices compris.

**Matériel :** PyTorch pour le laboratoire miniature ; PyTorch et Transformers pour le laboratoire avec un modèle préentraîné. Le CPU suffit, avec plusieurs Go de RAM disponibles pour ce dernier.

Lis d’abord les sections 2 à 7. Les termes comme attention ou gradient sont présentés pour te repérer : tu n’as pas encore à en dériver les formules.

## 2. Le premier modèle mental : continuer une séquence

Prenons le texte :

```text
Le chat
```

Le modèle calcule des scores pour les tokens qui pourraient suivre. Il pourrait accorder une forte probabilité à ` dort`, une autre à ` mange`, et une très faible probabilité à un symbole sans rapport.

Après sélection de ` dort`, la séquence devient :

```text
Le chat dort
```

Elle sert à prédire le token suivant. Cette construction s’appelle **génération autorégressive** : chaque nouvelle prédiction dépend des tokens précédents, y compris ceux qui viennent d’être générés.

Un token n’est pas toujours un mot. ` dort` est une possibilité illustrative ; le découpage réel dépend du tokenizer. Un espace, un fragment de mot ou une ponctuation peuvent faire partie d’un token.

La génération causale standard ne choisit donc pas toute une phrase en une seule opération.

> **Pause compréhension.** Si le modèle choisit ` mange` au lieu de ` dort`, le calcul suivant reçoit-il exactement le même contexte ?
>
> Non : la séquence a changé. Les distributions suivantes peuvent donc changer aussi.

## 3. La carte du trajet

```text
Conversation : rôles et contenus des messages
        ↓ chat template : format attendu par le modèle de chat
Texte formaté / marqueurs de conversation
        ↓ tokenizer
Identifiants de tokens
        ↓ table d’embeddings
Vecteurs de départ
        ↓ blocs Transformer, avec information de position
Vecteurs contextualisés : hidden states
        ↓ LM head
Scores sur le vocabulaire : logits
        ↓ stratégie de décodage
Prochain token choisi
        ↓ ajout à la séquence
Recommencer jusqu’à une condition d’arrêt
```

Nous allons donner un sens concret à chaque transformation.

### 3.1 Messages, prompt et chat template

Le **prompt** est l’entrée qui conditionne la réponse. Pour un assistant, elle peut être une conversation structurée :

```text
system    : Tu expliques simplement les notions informatiques.
user      : Qu’est-ce que HTTP ?
```

Le **chat template** transforme ces rôles et contenus dans le format attendu par ce modèle. Un exemple fictif serait :

```text
<system>Tu expliques simplement les notions informatiques.</system>
<user>Qu’est-ce que HTTP ?</user>
<assistant>
```

Les marqueurs réels dépendent du modèle. Utilise son template plutôt que d’inventer ces marqueurs. Le préfixe final indique ici qu’une réponse assistant doit commencer.

Un modèle de base qui continue du texte et un modèle ajusté pour répondre en conversation peuvent partager une architecture sans attendre le même format d’entrée.

### 3.2 Tokens, vocabulaire et IDs

Le **tokenizer** transforme le texte en unités, puis en identifiants numériques. Son **vocabulaire** associe les unités à des IDs.

Exemple entièrement fictif :

| ID | Token |
| --- | --- |
| 0 | `<eos>` |
| 1 | `Le` |
| 2 | ` chat` |
| 3 | ` dort` |
| 4 | `.` |
| 5 | ` mange` |

`Le chat` devient alors `[1, 2]`. L’ID 2 est une adresse dans une table ; sa valeur numérique ne mesure pas le sens de « chat ».

Certains tokens ont une fonction structurelle : **BOS** pour un début de séquence, **EOS** pour une fin et **PAD** pour du remplissage. Tous les tokenizers ne les utilisent pas de la même manière.

### 3.3 Embeddings et positions

Une **table d’embeddings** associe un vecteur appris à chaque ID. Si la largeur choisie est 4, l’ID 2 sélectionne par exemple quatre nombres : `[0.2, -0.1, 0.7, 0.3]`.

Ces nombres sont illustratifs ; ils ne sont pas des probabilités. Ils constituent une représentation que le réseau apprend à exploiter.

L’ordre compte : « le chien suit le chat » et « le chat suit le chien » n’ont pas le même sens. Le réseau utilise donc une information de position. Elle peut être ajoutée aux représentations ou intervenir dans l’attention, par exemple avec RoPE. Tu verras cette distinction au module 4.

### 3.4 Transformer et hidden states

Les blocs Transformer transforment les vecteurs. L’**attention** fait circuler de l’information entre les positions autorisées. Le **MLP**, un réseau de couches linéaires et d’activations, transforme l’information à chaque position. Les résidus et normalisations facilitent le fonctionnement de cette pile.

Un **hidden state** est un vecteur interne à une étape du réseau. Après contextualisation, il dépend du préfixe lu. Dans un modèle causal, il n’utilise pas les tokens situés dans son futur.

L’embedding d’un token donné est le même à la consultation de sa ligne ; sa représentation après les blocs peut varier avec le contexte.

### 3.5 LM head et logits

Le **LM head** transforme un hidden state en un score par entrée du vocabulaire de sortie. Ces scores sont des **logits**.

Un score peut être négatif, dépasser 1, et la somme des scores n’a aucune raison de valoir 1. Un logit de 8 ne signifie pas « 80 % ».

Pour obtenir une distribution, on peut appliquer **softmax** à tous les logits du vocabulaire. Les probabilités obtenues sont positives et leur somme vaut 1, aux arrondis près.

## 4. Lire les formes sans se perdre

Un tenseur est un tableau de nombres. Sa **forme** indique le nombre de cases le long de chaque axe.

Nous utiliserons souvent ces lettres :

| Lettre | Signification | Petit exemple |
| --- | --- | --- |
| `B` | Nombre de séquences dans le batch | 2 |
| `L` | Longueur de chaque séquence après préparation | 3 |
| `H` | Largeur des vecteurs internes | 4 |
| `V` | Taille du vocabulaire de sortie | 6 |

```text
IDs           [B, L]     = [2, 3]
Hidden states [B, L, H]  = [2, 3, 4]
Logits        [B, L, V]  = [2, 3, 6]
```

`logits[0, -1]` sélectionne la première séquence, sa dernière position et tous les scores du vocabulaire. Pour une entrée unique non paddée, c’est la distribution de scores utilisée pour prédire le prochain token.

Le padding et les masques du module 3 préciseront ce qu’il faut faire lorsque certaines positions sont artificielles.

## 5. Comment choisit-on le prochain token ?

Supposons cette distribution sur trois candidats fictifs :

| Token | Probabilité |
| --- | --- |
| ` dort` | 0.60 |
| ` mange` | 0.30 |
| ` joue` | 0.10 |

- **Greedy** choisit le meilleur score : ici ` dort`.
- **Sampling** tire au hasard selon les probabilités : ` mange` reste possible.

Greedy donne le même choix pour les mêmes scores et la même règle de départage. Le meilleur token local ne garantit pas la meilleure séquence complète, ni une réponse vraie.

### Température

Pour une température `T > 0`, on peut calculer `softmax(logits / T)`.

- `T < 1` concentre davantage la distribution autour des meilleurs scores.
- `T > 1` l’aplatit.

Une température positive conserve le classement des scores. Elle ne donne pas de nouvelles connaissances au modèle et ne garantit pas l’exactitude. N’applique pas cette formule avec `T = 0` ; utilise directement greedy pour choisir le maximum.

### Top-k et top-p

**Top-k** conserve les `k` meilleurs candidats avant de tirer. **Top-p** conserve les candidats les plus probables jusqu’à atteindre un seuil cumulé, puis renormalise la distribution retenue.

Avec `[0.60, 0.30, 0.10]`, `top_k=2` conserve les deux premiers. `top_p=0.8` les conserve aussi, car `0.60 < 0.8` mais `0.60 + 0.30 = 0.90 ≥ 0.8`.

Top-p garde donc un nombre variable de candidats selon la distribution. Ces filtres modifient le décodage, pas les paramètres appris.

## 6. Pourquoi et quand la génération s’arrête

La boucle peut s’arrêter à un token de fin configuré, une séquence d’arrêt ou une limite de nouveaux tokens.

`max_new_tokens=20` limite les **nouveaux** tokens, pas la taille de toute la conversation. Atteindre cette limite peut couper une réponse avant sa fin naturelle.

Le **KV cache** conserve des clés et valeurs calculées dans l’attention pour réutiliser le travail des positions précédentes. Il accélère la génération au prix de mémoire supplémentaire. Il ne stocke pas de nouveaux poids appris et ne constitue pas une mémoire permanente d’une conversation à l’autre.

Le détail de ses tenseurs attendra les modules consacrés à la mémoire. Ici, retiens le compromis : davantage de tokens conservés peut demander davantage de mémoire.

## 7. Inférence, génération et entraînement

| Activité | Ce qui est calculé | Paramètres modifiés ? |
| --- | --- | --- |
| Inférence | Des sorties à partir d’entrées : logits, représentations ou scores | Non dans l’utilisation ordinaire |
| Génération | Une suite de nouveaux tokens ; c’est un cas d’inférence | Non |
| Entraînement | Prédictions, erreur, gradients et mises à jour | Oui lors des mises à jour |

La **loss** mesure l’erreur par rapport aux cibles. **Backward** calcule les gradients de cette erreur. L’**optimizer** utilise les gradients pour modifier les paramètres ; le **learning rate** règle l’échelle des corrections.

```text
Entrées → forward → logits → loss → backward → gradients → optimizer step
```

Calculer une loss sans appliquer de mise à jour peut aussi servir à évaluer un modèle. Le simple fait d’avoir produit un nombre appelé « loss » ne prouve pas qu’un entraînement a eu lieu.

Les modules 6, 7 et 8 te feront calculer puis observer ces étapes sur des exemples minuscules.

## 8. Laboratoire A — Voir scores, probabilités et choix

Ce bloc est autonome, nécessite seulement PyTorch et ne télécharge rien. Nous **fournissons** trois scores fictifs : il ne s’agit pas d’un réseau entraîné.

Avant de lancer le code, prédis le candidat greedy et l’effet de la température.

```python
import torch

candidates = [" dort", " mange", " joue"]
logits = torch.tensor([2.0, 1.0, 0.0])

for temperature in [0.5, 1.0, 2.0]:
    probs = torch.softmax(logits / temperature, dim=-1)
    print("Température :", temperature)
    for token, score, probability in zip(candidates, logits, probs):
        print(repr(token), "score=", score.item(), "prob=", round(probability.item(), 4))
    print("Somme :", probs.sum().item())

probs = torch.softmax(logits, dim=-1)
greedy_id = torch.argmax(probs).item()
print("Greedy :", repr(candidates[greedy_id]))

torch.manual_seed(42)
sampled_ids = torch.multinomial(probs, num_samples=12, replacement=True)
print("Tirages :", [candidates[index] for index in sampled_ids.tolist()])
```

À `T = 1`, les probabilités sont environ `[0.6652, 0.2447, 0.0900]`. Le maximum reste ` dort` pour les trois températures. Les tirages peuvent contenir plusieurs candidats ; douze tirages ne suffisent pas à retrouver exactement les proportions théoriques.

La seed aide à refaire une expérience dans le même environnement ; elle ne garantit pas des résultats identiques sur toutes les plateformes.

## 9. Laboratoire B — Inspecter un vrai modèle

Ce bloc est autonome. Utilise ton environnement du [module 0](00_METHODE_ET_ENVIRONNEMENT.md). Si nécessaire, installe les dépendances avec `python -m pip install torch transformers` ; Qwen3 nécessite une version de Transformers qui prend en charge cette architecture.

Le premier chargement de `Qwen/Qwen3-0.6B` peut télécharger le tokenizer et les poids. « 0.6B » représente un ordre de grandeur de centaines de millions de paramètres, pas 0.6 Go de RAM. L’exécution sur CPU peut prendre du temps.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

messages = [{"role": "user", "content": "Réponds en une phrase : qu’est-ce qu’un token ?"}]
# enable_thinking est une option propre au template de Qwen3 utilisé ici.
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
print("Entrée formatée :", repr(formatted))
inputs = tokenizer(formatted, add_special_tokens=False, return_tensors="pt")
print("Forme des IDs :", tuple(inputs["input_ids"].shape))

with torch.no_grad():
    outputs = model(**inputs, use_cache=False)
    next_logits = outputs.logits[0, -1]
    probabilities = torch.softmax(next_logits.float(), dim=-1)
    values, ids = torch.topk(probabilities, k=5)

    generated = model.generate(
        **inputs, max_new_tokens=12, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print("Forme des logits :", tuple(outputs.logits.shape))
for probability, token_id in zip(values.tolist(), ids.tolist()):
    print("id=", token_id, "texte=", repr(tokenizer.decode([token_id])),
          "prob=", round(probability, 5))

prompt_length = inputs["input_ids"].shape[1]
new_ids = generated[0, prompt_length:]
print("Nouveaux tokens seulement :", tokenizer.decode(new_ids, skip_special_tokens=True))
```

Le code affiche le template réel et sélectionne la dernière position du prompt pour examiner ses cinq meilleurs candidats. Il appelle ensuite `generate()` pour effectuer plusieurs étapes de génération.

`add_special_tokens=False` évite ici d’ajouter une seconde fois des marqueurs déjà présents dans le texte formaté. `model.eval()` règle le comportement des couches ; `torch.no_grad()` désactive l’enregistrement autograd pour ces calculs. Nous reprendrons cette distinction au module 7.

Ne cherche pas à reproduire un texte de sortie imposé : observe les formes et le trajet. Une limite de 12 tokens peut produire une réponse incomplète. Les cinq probabilités affichées ne couvrent pas forcément toute la masse du vocabulaire.

Le vocabulaire de sortie du modèle peut inclure des lignes réservées supplémentaires : sa taille n’est pas obligatoirement exactement `len(tokenizer)`. Lis la dernière dimension des logits et la configuration du modèle plutôt que de supposer leur égalité.

## 10. Expériences guidées

Conserve pour chaque essai le prompt, les réglages et une observation dans tes notes.

1. Dans le laboratoire A, remplace les scores par `[0.0, 0.0, 0.0]`. Prédis les probabilités : elles seront toutes proches de `1/3`, quelle que soit la température positive.
2. Augmente le nombre de tirages à 1 000. Compte les résultats et compare leurs fréquences aux probabilités, sans attendre une égalité parfaite.
3. Dans le laboratoire B, reformule seulement la question. Observe ce qui change dans le prompt formaté et les candidats.
4. Augmente seulement `max_new_tokens` à 24. Distingue une coupure par limite d’une fin naturelle.

Ne compare pas plusieurs changements simultanés si tu veux expliquer leur effet.

## 11. Exercices avec corrigés

### Exercice 1 — Expliquer le trajet

Quel élément transforme les IDs en vecteurs ? Quel élément produit les scores du vocabulaire ?

<details>
<summary>Corrigé</summary>

La table d’embeddings transforme les IDs en vecteurs de départ. Après les blocs du réseau, le LM head produit les logits du vocabulaire à partir des hidden states.

</details>

### Exercice 2 — Lire une forme

Des logits ont la forme `[2, 5, 100]`. Combien de séquences contient le batch ? Combien de scores décrit une position ? Que sélectionne `logits[0, -1]` ?

<details>
<summary>Corrigé</summary>

Deux séquences de cinq positions ; chaque position a 100 scores. `logits[0, -1]` sélectionne les 100 scores de la dernière position de la première séquence.

</details>

### Exercice 3 — Choisir un token

Les probabilités sont `[0.6, 0.3, 0.1]`. Le deuxième token est-il impossible avec sampling ? Quels candidats conserve top-p avec `p=0.8` ?

<details>
<summary>Corrigé</summary>

Le deuxième token a 30 % de probabilité avant tout filtrage. Top-p conserve les deux premiers : leur cumul vaut 0.9. Après renormalisation sur ces candidats, leurs probabilités deviennent environ 2/3 et 1/3.

</details>

### Exercice 4 — Repérer un entraînement

Un programme appelle le modèle sous `no_grad()`, choisit un token, puis l’ajoute au prompt. Les paramètres apprennent-ils cette conversation ?

<details>
<summary>Corrigé</summary>

Non. Le contexte change, mais aucune mise à jour des paramètres n’est effectuée. Il s’agit de génération. Entraîner demanderait une loss, des gradients et une mise à jour des paramètres.

</details>

### Exercice 5 — Comprendre le cache

Pourquoi le KV cache peut-il accélérer la génération tout en consommant de la mémoire ?

<details>
<summary>Corrigé</summary>

Il garde des résultats intermédiaires d’attention pour éviter de recalculer ceux des positions précédentes. Stocker ces tenseurs demande de la mémoire ; les conserver ne modifie pas les poids du réseau.

</details>

## 12. Validation — avant de continuer

- [ ] Je peux redessiner le trajet du texte jusqu’au prochain token.
- [ ] Je distingue un token, son ID, un embedding, un hidden state et un logit.
- [ ] Je sais lire `[B, L, V]` et sélectionner la dernière position d’une entrée non paddée.
- [ ] Je peux expliquer greedy, sampling, température, top-k et top-p avec trois candidats.
- [ ] Je sais pourquoi plusieurs tokens demandent plusieurs étapes de génération causale standard.
- [ ] Je distingue modification du contexte et modification des paramètres.
- [ ] Je peux commenter ce que les laboratoires montrent sans attribuer d’apprentissage au premier.

Si un terme bloque, reprends son exemple dans ce module. La dérivation de l’attention, RoPE et les détails d’AdamW peuvent attendre.

## Fiche mémo

```text
Messages → template → tokenizer → IDs → embeddings
→ Transformer avec positions → hidden states → LM head → logits
→ décodage → prochain token → ajout au contexte → recommencer

Inférence : utiliser des paramètres fixes.
Génération : produire de nouveaux tokens pendant l’inférence.
Entraînement : calculer une erreur et mettre à jour des paramètres.
```

Le [module 2](02_tokenisation_en_profondeur.md) ouvre le premier maillon : comment le texte devient des tokens et pourquoi ce découpage compte.

## Références

- [Hugging Face : chat templates](https://huggingface.co/docs/transformers/en/chat_templating) — formatage des conversations et ajout des marqueurs.
- [Hugging Face : génération de texte](https://huggingface.co/docs/transformers/en/llm_tutorial) — utilisation de `generate()` et préparation des entrées.

- [Fiche officielle Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — modèle et option `enable_thinking` utilisés dans le laboratoire réel.
