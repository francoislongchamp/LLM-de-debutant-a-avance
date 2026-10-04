# Module 11 — Train, validation, test et contamination

## Pourquoi ce module arrive ici

Si tu mesures le modèle sur des exemples qu’il a vus pendant l’entraînement, tu mesures en partie sa mémoire. Il faut séparer ce qui sert à apprendre de ce qui sert à choisir des hyperparamètres et de ce qui sert à l’évaluation finale.

## Objectifs

Comprendre rôles de train/validation/test, data leakage, contamination, split groupé et effet du tuning répété sur le test. Savoir créer un split reproductible et vérifier les recouvrements.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fine-tuning — données

## Définitions concrètes

### Train set

**Définition concrète.** Données utilisées pour calculer les gradients et mettre à jour les paramètres.


**Exemple simple.** 80 % d’un corpus peut servir au train.

### Validation set

**Définition concrète.** Données non utilisées pour les gradients, consultées pour choix de configuration/early stopping.


**Exemple simple.** Comparer LR ou epochs sur validation.

### Test set

**Définition concrète.** Jeu réservé à l’estimation finale, idéalement peu consulté pendant le développement.


**Exemple simple.** Mesure finale après choix des hyperparamètres.

### Leakage

**Définition concrète.** Information du futur/évaluation qui fuit dans les données d’entraînement ou les features.


**Exemple simple.** Une réponse de benchmark copiée dans train.

### Contamination

**Définition concrète.** Présence directe ou quasi directe d’éléments d’évaluation dans le corpus d’apprentissage.


**Exemple simple.** Même question reformulée très légèrement.

### Group split

**Définition concrète.** Split qui garde ensemble des exemples liés par une même source/groupe.


**Exemple simple.** Tous les fragments d’un même document restent dans le même split.

## Intuition simple

Si tu étudies avec la feuille d’examen finale sous les yeux, un bon score ne mesure plus correctement ta capacité à généraliser. La validation sert de devoir d’entraînement; le test doit rester l’examen final.

## Ce qui se passe réellement sous le capot

1. Définir l’unité qui ne doit pas traverser les splits : document, utilisateur, conversation, source, problème.
2. Dédupliquer avant ou avec conscience des splits.
3. Créer train/validation/test avec seed et éventuellement group split.
4. Vérifier hashes, IDs ou similarités entre splits.
5. Utiliser validation pour décisions de training.
6. Garder test pour une mesure plus finale.
7. Si le test influence tes choix plusieurs fois, il devient de facto une validation supplémentaire.

## Exemple minimal à comprendre mentalement

Tu découpes un manuel en 1000 paragraphes au hasard. Des paragraphes adjacents du même chapitre peuvent se retrouver dans train et test. Même sans doublon exact, le test devient plus facile car le modèle a vu le contexte voisin. Un split **par document ou chapitre** peut être plus honnête.

## Formules et notation utiles

Split 80/10/10 sur 10 000 exemples :

```text
train = 8000
validation = 1000
test = 1000
```

Ces pourcentages sont des conventions, pas une loi. La qualité du split et la taille absolue comptent davantage.

## Code minimal observable

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="data/train.jsonl", split="train")
first = ds.train_test_split(test_size=0.2, seed=42)
second = first["test"].train_test_split(test_size=0.5, seed=42)

splits = {
    "train": first["train"],
    "validation": second["train"],
    "test": second["test"],
}
for k,v in splits.items():
    print(k, len(v))
```

## Laboratoire guidé

1. Crée un split 80/10/10 reproductible.  
2. Ajoute un ID stable à chaque exemple brut et vérifie qu’aucun ID ne traverse les splits.  
3. Si tes exemples proviennent de documents, refais un split par `document_id`.  
4. Compare les distributions de catégories dans les trois splits.  
5. Écris une règle claire indiquant quand le test peut être consulté.

## Ce que tu dois observer

- Un split aléatoire n’est pas toujours un split honnête.
- Les petits datasets peuvent avoir des validations très bruitées.
- La contamination peut être sémantique, pas seulement exacte.

## À ne pas confondre

- Validation ≠ test.
- Déduplication ≠ garantie d’absence de contamination.
- Même distribution ≠ mêmes exemples.

## Erreurs fréquentes

- Tuner les hyperparamètres directement sur le test.
- Séparer après avoir généré plusieurs variantes quasi identiques d’un même exemple.
- Comparer deux modèles sur des tests différents.

## Exercices

1. Donne un cas où un split par ligne est mauvais.
2. Explique pourquoi consulter le test à chaque expérience le transforme en validation.
3. Propose une stratégie de split pour des conversations multi-tours.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Expliquer le rôle distinct de train/validation/test.
- Donner un exemple de leakage et de contamination.
- Créer un split reproductible.
- Justifier l’unité de groupement utilisée pour ton propre dataset.

## Fiche mémo

Une évaluation crédible dépend autant de l’indépendance des données que de la métrique.

## Lien avec le module suivant

Avec un dataset propre et séparé, le module 12 réalise le premier SFT complet.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
