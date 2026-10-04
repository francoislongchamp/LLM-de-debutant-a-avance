# Module 59 — Projet 4 — Continued pretraining

## Pourquoi ce module arrive ici

Le CPT est facile à justifier verbalement (“le modèle doit apprendre le domaine”) mais difficile à prouver. Ce projet compare directement Base→SFT contre Base→CPT→SFT.

## Objectifs

Mesurer la valeur ajoutée d’un CPT de domaine avant SFT, en contrôlant le SFT final et en surveillant perplexité domaine, benchmark métier et forgetting général.

**Prérequis :** Les modules précédents.  
**Temps indicatif :** 6 à 12 heures pour une expérience pédagogique courte.  
**Niveau :** Projet intégrateur

## Définitions concrètes

### Control path

**Définition concrète.** Pipeline sans l’intervention étudiée.


**Exemple simple.** Base→SFT.

### Treatment path

**Définition concrète.** Pipeline avec intervention.


**Exemple simple.** Base→CPT→SFT.

### Domain holdout

**Définition concrète.** Texte du domaine non utilisé au CPT pour mesurer language modeling.


**Exemple simple.** Validation corpus.

### Downstream benchmark

**Définition concrète.** Tâche finale après SFT.


**Exemple simple.** Questions domaine.

### Intermediate checkpoint

**Définition concrète.** Modèle après CPT avant SFT.


**Exemple simple.** Permet d’isoler ce que CPT change.

## Intuition simple

Le CPT n’est utile au projet que si ses gains survivent et apportent quelque chose **après le même SFT**. Sinon, il ajoute du coût sans bénéfice démontré.

## Ce qui se passe réellement sous le capot

1. Évaluer base sur corpus domaine/général.
2. Faire CPT court et évaluer intermediate.
3. Dupliquer le même SFT : une branche depuis base, une depuis CPT.
4. Utiliser même dataset SFT et protocole.
5. Comparer les deux modèles finaux.
6. Mesurer généralisation et forgetting.
7. Calculer coût supplémentaire du CPT.

## Exemple minimal à comprendre mentalement

```text
                    Base→SFT   Base→CPT→SFT
Domain benchmark       72          81
General                80          78
CPT extra GPU-hours     0           6
```

Le projet doit décider si +9 domaine / -2 général / +6h est un bon compromis pour l’objectif.

## Formules et notation utiles

Valeur marginale du CPT :

```text
ΔCPT = score(CPT→SFT) - score(SFT-only)
```

Mesure aussi ce delta par catégories et non seulement globalement.

## Code minimal observable

```text
base
├── eval_base
├── sft_control ──> eval_final_control
└── cpt ──> eval_cpt ──> sft_same_data ──> eval_final_treatment
```

## Laboratoire guidé

1. Prépare corpus CPT et dataset SFT distincts.  
2. Vérifie contamination entre corpus holdout et train.  
3. Évalue base.  
4. Fais CPT et évalue.  
5. Fais les deux SFT avec config identique adaptée au point de départ.  
6. Compare finals sur le même test.  
7. Inspecte aussi des cas où CPT a empiré le comportement.

## Ce que tu dois observer

- La baisse de PPL domaine après CPT ne garantit pas un gain downstream.
- Un SFT fort peut masquer une partie des différences de CPT.
- CPT étroit peut créer forgetting avant même SFT.

## À ne pas confondre

- Gain intermediate ≠ gain final.
- CPT corpus ≠ SFT dataset.
- PPL domaine ≠ benchmark métier.

## Erreurs fréquentes

- Comparer Base→SFT avec Base→CPT→SFT mais changer aussi le dataset SFT.
- Ne pas conserver intermediate CPT.
- Ne pas calculer coût marginal.

## Exercices

1. Écris l’arbre expérimental de ce projet.
2. Définis trois métriques intermédiaires et finales.
3. Explique pourquoi même SFT final est nécessaire pour isoler CPT.

## Validation — avant de continuer

Tu dois pouvoir, sans simplement réciter le code :

- Produire control et treatment.
- Évaluer intermediate et final.
- Calculer gain marginal CPT.
- Inclure coût et forgetting dans la décision.

## Fiche mémo

Le CPT est justifié lorsqu’il apporte une valeur **mesurée** au pipeline final, pas seulement une loss domaine plus basse.

## Lien avec le module suivant

Le projet 5 prend un modèle SFT et teste l’effet de préférences DPO.

---

> **Règle de progression :** si un point de la validation n’est pas clair, refais le micro-exemple et le laboratoire. Le but est de construire un modèle mental solide, pas d’accumuler des commandes.
