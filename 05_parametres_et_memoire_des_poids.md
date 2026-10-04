# Module 5 — Paramètres et mémoire des poids

## Pourquoi ce module arrive ici

Dire qu’un modèle fait “7B” n’est utile que si tu sais ce que sont ces 7 milliards de nombres et combien de mémoire ils consomment selon leur précision.

## Objectifs

Compter paramètres totaux/entraînables, convertir paramètres en mémoire de poids, comprendre dtype et distinguer mémoire d’inférence de mémoire d’entraînement.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fondations

## Définitions concrètes

### Paramètre

**Définition concrète.** Nombre appris stocké dans le réseau.


**Exemple simple.** Une matrice 100×200 contient 20 000 poids.

### Trainable parameter

**Définition concrète.** Paramètre autorisé à recevoir un gradient et à être optimisé.


**Exemple simple.** En LoRA, la base est généralement gelée.

### dtype

**Définition concrète.** Format numérique des valeurs.


**Exemple simple.** FP32≈4 octets, FP16/BF16≈2 octets.

### Quantification

**Définition concrète.** Représentation plus compacte des poids.


**Exemple simple.** 4 bits≈0,5 octet brut par poids avant surcoûts.

## Intuition simple

Un modèle est un ensemble gigantesque de tableaux de nombres. “600M” signifie environ 600 millions de valeurs apprises. `nombre de valeurs × octets` donne un premier ordre de grandeur des **poids seulement**.

## Ce qui se passe réellement sous le capot

1. `numel()` compte les éléments d’un tenseur.
2. La somme sur tous les paramètres donne le total.
3. `requires_grad` permet de compter les paramètres entraînables.
4. Le dtype donne la taille brute de chaque élément.
5. Le training complet ajoute gradients, optimizer states, activations et buffers.
6. PEFT réduit surtout les paramètres nécessitant gradients et états d’optimizer.

## Exemple minimal à comprendre mentalement

```text
Embedding : 10 000 × 512 = 5 120 000
Linear    :    512 × 512 =   262 144
Total partiel ≈ 5,38 M
```

En BF16 : `5,38M × 2 octets ≈ 10,76 MB` pour les poids de cet exemple. Ce n’est pas la mémoire totale du training.

## Formules et notation utiles

\[
M_{poids}\approx N_{params}\times \frac{bits}{8}
\]

Exemple 7B : BF16 ≈14 GB; FP32 ≈28 GB pour les poids bruts.

## Code minimal observable

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
bytes_now = sum(p.numel()*p.element_size() for p in model.parameters())
print(f"total={total:,}")
print(f"trainable={trainable:,}")
print(f"weights={bytes_now/1024**2:.1f} MiB")
```

## Laboratoire guidé

1. Mesure total, trainable et mémoire des poids.  
2. Trouve les dix plus gros tenseurs.  
3. Recalcule leur mémoire manuellement.  
4. Calcule les poids théoriques en FP32, BF16, 8-bit et 4-bit.  
5. Compare avec la VRAM observée et liste les raisons de l’écart.

## Ce que tu dois observer

- Taille sur disque, mémoire des poids et VRAM totale diffèrent.
- Le training peut conserver des états dans plusieurs précisions.
- Les grosses matrices dominent le nombre de paramètres.

## À ne pas confondre

- Paramètres ≠ taille exacte du fichier.
- Mémoire des poids ≠ mémoire totale de training.
- Quantification 4-bit ≠ tous les calculs en 4 bits.

## Erreurs fréquentes

- Dimensionner un full FT uniquement avec `2 bytes × params`.
- Oublier activations et optimizer states.

## Exercices

1. Calcule 1B paramètres en FP32/BF16/4-bit brut.
2. Calcule le nombre de paramètres d’une matrice 4096×11008.
3. Explique pourquoi LoRA réduit fortement certaines composantes mémoire mais pas toutes.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Calculer les paramètres d’une matrice.
- Convertir paramètres et dtype en ordre de grandeur mémoire.
- Distinguer total et trainable.
- Expliquer pourquoi l’inférence peut tenir alors que le full FT OOM.

## Fiche mémo

La mémoire des poids est seulement la première ligne du budget. Training = poids + gradients + optimizer + activations + temporaires.

## Lien avec le module suivant

Le module 6 explique comment la cross-entropy mesure l’erreur de prédiction.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
