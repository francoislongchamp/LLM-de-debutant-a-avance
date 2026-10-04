# Module 3 — Fenêtre de contexte et séquences

## Pourquoi ce module arrive ici

Les tokens sont traités sous forme de séquences de longueur finie. Il faut comprendre ce que signifie réellement “contexte 32k”, comment plusieurs exemples deviennent un batch et pourquoi padding, truncation et masques existent.

## Objectifs

Définir sequence length, context window, padding, truncation, attention mask et causal mask. Lire la forme des tenseurs d’entrée et expliquer ce qui se passe lorsqu’un document dépasse la fenêtre de contexte.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fondations

## Définitions concrètes

### Longueur de séquence

**Définition concrète.** Nombre de positions de tokens dans une séquence.


**Exemple simple.** 17 token IDs → longueur 17.

### Fenêtre de contexte

**Définition concrète.** Nombre maximal de positions traitables dans une passe/configuration donnée.


**Exemple simple.** 4096 signifie 4096 tokens, pas 4096 mots.

### Padding

**Définition concrète.** Ajout de tokens artificiels pour rendre des séquences de même longueur dans un batch.


**Exemple simple.** [A,B] devient [A,B,PAD,PAD].

### Truncation

**Définition concrète.** Suppression de tokens excédant une limite choisie.


**Exemple simple.** 5000 tokens tronqués à 4096 perdent 904 positions.

### Attention mask

**Définition concrète.** Masque indiquant quelles positions sont généralement réelles ou padding.


**Exemple simple.** [1,1,0,0] peut marquer deux tokens utiles.

### Causal mask

**Définition concrète.** Masque empêchant une position de consulter les tokens futurs.


**Exemple simple.** La position 3 voit 1,2,3 mais pas 4.

## Intuition simple

Imagine une table possédant un nombre fixe de cases. Chaque token occupe une case. Le padding remplit les cases vides pour former un rectangle; la truncation coupe ce qui déborde; les masques indiquent quelles cases comptent et quelles relations sont autorisées.

## Ce qui se passe réellement sous le capot

1. Le tokenizer produit une liste d’IDs de longueur variable.
2. Pour un batch, les séquences sont souvent paddées jusqu’à la plus longue séquence du batch.
3. Les tenseurs deviennent `[batch_size, sequence_length]`.
4. L’attention mask évite de traiter le padding comme du contenu utile.
5. Le causal mask empêche l’accès au futur.
6. Si un document dépasse le budget, il faut le tronquer, le découper ou utiliser une stratégie/context window adaptée.

## Exemple minimal à comprendre mentalement

Deux séquences :

```text
A = [11, 12, 13, 14]
B = [21, 22]
```

Après padding :

```text
A = [11, 12, 13, 14]   mask=[1,1,1,1]
B = [21, 22,  0,  0]   mask=[1,1,0,0]
```

Le tenseur `input_ids` a la forme `[2,4]`. Les deux zéros sont des cases de remplissage, pas du contenu à apprendre.

## Formules et notation utiles

Avec `B` séquences paddées à `L`, le tenseur contient `B×L` positions. Exemple : `8×2048=16 384` positions. Si la moitié est du padding, beaucoup de calcul peut être gaspillé.

## Dimensions / formes à savoir lire

```text
input_ids      : [batch, sequence]
attention_mask : [batch, sequence]
logits         : [batch, sequence, vocab]
```

## Code minimal observable

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

texts = ["Bonjour.", "Bonjour, comment allez-vous aujourd’hui ?"]
batch = tok(texts, padding=True, return_tensors="pt")
print(batch["input_ids"])
print(batch["attention_mask"])
print(tuple(batch["input_ids"].shape))
```

## Laboratoire guidé

1. Teste trois phrases de tailles très différentes.  
2. Compare padding dynamique et `max_length=64`.  
3. Encode un long texte avec `truncation=True, max_length=32`.  
4. Calcule la fraction de padding d’un batch.  
5. Regroupe ensuite les exemples par longueur et compare cette fraction.

## Ce que tu dois observer

- Le padding suit souvent la séquence la plus longue du batch.
- Le padding peut gaspiller du calcul.
- Une fenêtre de contexte est un budget de tokens.
- La truncation peut supprimer exactement l’information importante.

## À ne pas confondre

- Attention mask ≠ causal mask.
- Fenêtre de contexte ≠ mémoire permanente.
- Longueur du prompt ≠ longueur totale : la génération consomme aussi du contexte.

## Erreurs fréquentes

- Tronquer silencieusement sans inspecter la partie supprimée.
- Paddder tout le dataset à la longueur maximale globale.
- Oublier que prompt et réponse partagent le budget total.

## Exercices

1. Prompt 3000 tokens + génération maximum 1500 : quel budget faut-il ?
2. Dessine un causal mask 4×4.
3. Explique pourquoi regrouper des séquences de longueur proche améliore l’efficacité.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer padding, truncation, attention mask et causal mask.
- Donner les formes principales d’un batch.
- Calculer la proportion de padding.
- Expliquer pourquoi 32k context ne signifie pas 32k mots.

## Fiche mémo

La fenêtre de contexte est un budget de **positions de tokens**. Les masques disent au modèle quelles positions compter ou autoriser.

## Lien avec le module suivant

Le module 4 ouvre le réseau lui-même : embeddings, attention, MLP, résidus et normalisation.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
