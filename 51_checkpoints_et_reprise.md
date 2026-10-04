# Module 51 — Checkpoints et reprise

## Pourquoi ce module arrive ici

Un training long doit survivre à une panne, un redémarrage ou une préemption. Sauvegarder seulement `model.safetensors` ne suffit pas à reprendre exactement la dynamique d’optimisation.

## Objectifs

Distinguer model checkpoint et training state, savoir quoi sauvegarder pour reprendre, tester une restauration et comprendre checkpoints complets vs sharded/adapters.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 45 à 90 minutes, laboratoire compris.  
**Niveau :** Fiabilité du training

## Définitions concrètes

### Model state

**Définition concrète.** Valeurs des paramètres du modèle/adapters.


**Exemple simple.** Poids.

### Optimizer state

**Définition concrète.** Moments/états internes de l’optimizer.


**Exemple simple.** Nécessaires pour reprendre AdamW de façon cohérente.

### Scheduler state

**Définition concrète.** Position actuelle dans la courbe de learning rate.


**Exemple simple.** Reprendre au bon LR.

### RNG state

**Définition concrète.** État des générateurs pseudo-aléatoires.


**Exemple simple.** Affecte dropout, sampling, shuffling.

### Global step

**Définition concrète.** Nombre d’updates déjà réalisées.


**Exemple simple.** Utilisé pour logging/scheduler/sauvegardes.

### Resume

**Définition concrète.** Reconstruction de l’état de training au point sauvegardé.


**Exemple simple.** Plus qu’un simple chargement de poids.

## Intuition simple

Le modèle est la position d’une voiture sur la route. L’optimizer et le scheduler sont sa vitesse, son rapport et son itinéraire. Recharger uniquement la position ne recrée pas le même voyage.

## Ce qui se passe réellement sous le capot

1. À intervalle défini, synchroniser les ranks si nécessaire.
2. Sauvegarder model/adapters ou shards.
3. Sauvegarder optimizer et scheduler.
4. Sauvegarder scaler/mixed precision state si applicable.
5. Sauvegarder global step/epoch, sampler et RNG selon framework.
6. Écrire la configuration et version de code.
7. Tester réellement `resume_from_checkpoint` sur un run court.
8. Valider que LR/step/loss reprennent de manière cohérente.

## Exemple minimal à comprendre mentalement

Mauvaise reprise : charger les poids du step 10 000 mais redémarrer scheduler comme step 0. Le modèle est au milieu de l’entraînement, mais reçoit un LR correspondant au début. Cela peut changer complètement la trajectoire.

## Formules et notation utiles

La fréquence de sauvegarde est un compromis : si tu sauvegardes toutes les `K` updates, une panne peut faire perdre au maximum environ `K` updates depuis le dernier checkpoint valide, hors corruption/écriture en cours.

## Code minimal observable

```python
# Avec Trainer/TRL, la forme exacte dépend de la version :
trainer.train(resume_from_checkpoint="outputs/checkpoint-1000")

# Toujours tester la reprise sur un run court avant une expérience coûteuse.
```

## Laboratoire guidé

1. Lance 100 steps et sauvegarde au step 50.  
2. Note LR/loss/global step au checkpoint.  
3. Arrête volontairement le run.  
4. Reprends depuis le checkpoint.  
5. Vérifie global step et LR.  
6. Teste aussi le chargement “weights only” pour comprendre la différence.  
7. Pour FSDP, documente la procédure exacte de restauration des shards.

## Ce que tu dois observer

- Une reprise peut être fonctionnelle sans être bitwise identique selon stack/hardware.
- Les adapters ont des checkpoints différents des full models.
- Tester restore est aussi important que tester save.

## À ne pas confondre

- Model checkpoint ≠ full training state.
- Save model ≠ resume training automatiquement.
- Sharded checkpoint ≠ fichier incomplet; c’est un format distribué.

## Erreurs fréquentes

- Découvrir après une panne que les checkpoints ne se rechargent pas.
- Sauvegarder sans config/tokenizer.
- Écraser le dernier checkpoint valide lors d’une écriture interrompue.

## Exercices

1. Liste les états nécessaires pour reprendre AdamW + scheduler.
2. Explique weights-only vs resume.
3. Propose une politique de rotation de checkpoints.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Sauvegarder et reprendre un petit run.
- Vérifier step et LR.
- Expliquer model state vs training state.
- Documenter les checkpoints distribués/adapters.

## Fiche mémo

Un checkpoint utile est un artefact **restaurable**, pas seulement un fichier créé sans erreur.

## Lien avec le module suivant

Le module 52 organise les métriques afin de détecter les problèmes sans ouvrir manuellement des logs texte.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
