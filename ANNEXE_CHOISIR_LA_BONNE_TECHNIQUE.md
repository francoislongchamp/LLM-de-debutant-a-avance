# Annexe — Choisir la bonne technique d'adaptation

Le choix ne doit pas commencer par « quelle technique est la plus avancée ? », mais par : **qu'est-ce que je veux modifier ?**

| Besoin principal | Technique à considérer d'abord | Ce qui change | Exemple simple |
|---|---|---|---|
| Donner accès à des informations externes qui changent | RAG / outils | Pas nécessairement les poids | Documentation interne mise à jour chaque semaine |
| Apprendre un format ou un comportement de réponse | SFT | Comportement conditionnel | Toujours répondre selon un schéma JSON |
| Faire un SFT avec peu de VRAM | LoRA | Petites matrices ajoutées | Adapter un 7B sans entraîner tous ses poids |
| Réduire encore la mémoire du modèle de base | QLoRA | Base quantifiée + LoRA | Base 4-bit gelée, adapters entraînés |
| Modifier profondément tous les poids | Full fine-tuning | Tous les paramètres | Adaptation importante avec beaucoup de données/compute |
| Faire apprendre la distribution d'un domaine à partir de texte brut | Continued pretraining | Modèle causal sur corpus de domaine | Documentation scientifique avant SFT |
| Apprendre « réponse A préférable à B » | DPO / méthodes de préférence | Préférences relatives | Style concis préféré à une réponse verbeuse |
| Apprendre un score humain via un modèle | Reward model | Fonction de score apprise | Classer plusieurs réponses |
| Optimiser une récompense calculable pendant génération | RL / GRPO/PPO selon cas | Policy par signal de reward | Réussite de tests unitaires |
| Créer architecture/tokenizer/base model lui-même | Pretraining from scratch | Tout | Petit LM de recherche |

## Arbre de décision simplifié

```text
L'information change souvent ?
├── Oui → RAG / outils d'abord
└── Non
    ↓
Le modèle sait-il déjà l'information mais répond mal ?
├── Oui → SFT / préférence
└── Non
    ↓
As-tu beaucoup de texte brut du domaine ?
├── Oui → considérer Continued Pretraining
└── Non → données supervisées / RAG

Besoin de modifier le comportement avec peu de ressources ?
├── Oui → LoRA / QLoRA
└── Non → comparer LoRA et Full FT, ne pas supposer que Full FT gagne

As-tu un signal de préférence pairwise ?
└── Oui → DPO est un bon premier candidat

As-tu une reward fiable, calculable pendant les générations ?
└── Oui → RL peut devenir pertinent
```

## Trois erreurs de choix fréquentes

### Fine-tuner pour injecter des faits volatils

Si le fait change demain, les poids deviennent déjà obsolètes. Une source externe peut être préférable.

### Utiliser RL pour une tâche supervisée simple

Si tu connais exactement la réponse cible, SFT est souvent plus direct et plus stable.

### Pré-entraîner from scratch parce que c'est « plus complet »

Un modèle existant possède déjà énormément de représentations utiles. Repartir de poids aléatoires est un choix de recherche ou d'architecture, pas une étape obligatoire.
