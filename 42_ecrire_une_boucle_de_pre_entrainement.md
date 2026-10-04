# Module 42 — Écrire une boucle de pré-entraînement

## Pourquoi ce module arrive ici

Tous les composants sont prêts : tokenizer, séquences et modèle aléatoire. Il faut maintenant relier batches de tokens, causal loss, optimizer, scheduler, validation et checkpointing.

## Objectifs

Écrire une boucle de pré-entraînement minimale, calculer tokens vus, gérer train/eval, accumulation, scheduler et sauvegarde. Être capable de reprendre conceptuellement un run.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Pré-entraînement — optimisation

## Définitions concrètes

### Token batch

**Définition concrète.** Tenseur d’IDs utilisé comme entrées/labels causaux.


**Exemple simple.** [B,L].

### Tokens per step

**Définition concrète.** Nombre approximatif de positions non-pad contribuant à une update.


**Exemple simple.** B×L×devices×accumulation si blocs pleins.

### Global step

**Définition concrète.** Compteur d’optimizer updates.


**Exemple simple.** Différent du nombre de micro-batches.

### Scheduler step

**Définition concrète.** Mise à jour du learning rate selon la politique.


**Exemple simple.** Souvent une fois par optimizer step.

### Validation interval

**Définition concrète.** Fréquence d’évaluation sans gradients.


**Exemple simple.** Tous les 500 steps.

## Intuition simple

Le pré-entraînement n’est pas mystérieux : c’est la même boucle PyTorch que le module 8, mais la tâche est maintenant “prédire chaque prochain token” sur énormément de séquences.

## Ce qui se passe réellement sous le capot

1. Prendre un batch de token IDs.
2. Construire/obtenir les labels causaux.
3. Forward et loss.
4. Diviser/normaliser si accumulation selon framework.
5. Backward.
6. Après N accumulations : clipping éventuel, optimizer step, scheduler step, zero grad.
7. Ajouter tokens traités au compteur.
8. À intervalles : validation et checkpoint.
9. Arrêter selon budget de steps/tokens.

## Exemple minimal à comprendre mentalement

Avec batch global 32 séquences de 512 tokens :

```text
≈16 384 token positions/update
```

Après 10 000 optimizer steps :

```text
≈163,84M token positions
```

Le vrai nombre utile doit tenir compte de padding/masques et éventuellement de tokens exclus.

## Formules et notation utiles

\[
TokensSeen\approx steps\times B_{global}\times L
\]

pour des blocs pleins. Suivre les tokens rend plus comparable un run lorsque batch/accumulation changent.

## Code minimal observable

```python
model.train()
optimizer.zero_grad()

for step, batch in enumerate(loader):
    ids = batch["input_ids"].to(device)
    out = model(input_ids=ids, labels=ids)
    loss = out.loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(step, loss.item())
```

## Laboratoire guidé

1. Fais d’abord 20 steps sur **un seul petit batch répété** : la loss doit pouvoir fortement baisser; c’est un test du pipeline, pas une vraie expérience.  
2. Ensuite utilise le vrai DataLoader et 500–2000 steps.  
3. Logge train loss, val loss, LR, grad norm, tokens/s, tokens vus.  
4. Sauvegarde un checkpoint et reprends-le.  
5. Génére quelques textes à intervalles fixes avec prompts identiques.

## Ce que tu dois observer

- Overfit un batch est un excellent test pour détecter un pipeline cassé.
- Une vraie validation doit utiliser des séquences non vues.
- Les premières générations seront incohérentes même si la loss commence à descendre.

## À ne pas confondre

- Overfit-one-batch test ≠ entraînement final.
- Global step ≠ nombre de batches si accumulation.
- Tokens seen ≠ tokens uniques du corpus.

## Erreurs fréquentes

- Lancer un long run avant de réussir l’overfit-one-batch.
- Ne pas sauvegarder optimizer/scheduler si on veut une vraie reprise.
- Générer avec des réglages différents à chaque checkpoint.

## Exercices

1. Calcule tokens vus pour 5000 steps, batch global 64, L=1024.
2. Explique pourquoi overfit-one-batch est utile.
3. Écris la différence entre reprendre seulement les poids et reprendre l’état complet.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Écrire une boucle causale minimale.
- Calculer tokens vus.
- Overfitter un batch volontairement.
- Sauvegarder/reprendre un run conceptuellement complet.

## Fiche mémo

Le pré-entraînement est la répétition à grande échelle d’une boucle simple; la difficulté vient de la fiabilité du système et du volume.

## Lien avec le module suivant

Le module 43 apprend à lire les courbes produites par cette boucle avant de gaspiller du compute.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
