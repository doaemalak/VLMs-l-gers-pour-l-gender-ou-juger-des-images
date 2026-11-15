# Fiche du modèle

## Description

Deux petits modèles vision-langage (VLM), utilisés **principalement en inférence**,
en précision normale, 8 bits et 4 bits.

| Élément | VLM 1 | VLM 2 |
|---|---|---|
| Identifiant | | |
| Nombre de paramètres | | |
| Mémoire en précision normale | | |
| Mémoire en 8 bits | | |
| Mémoire en 4 bits | | |

## Usage prévu

- Option A : produire des légendes d'images pour entraîner une LoRA.
- Option B : évaluer des images générées, en comparaison avec un jugement humain.

## Usages non prévus

- Décisions importantes concernant des personnes à partir d'une image.
- Remplacement complet d'une évaluation humaine sans vérification.

## Performances

Résumer ici les mesures chiffrées (voir `results/`).

## Limites connues

- Légendes pouvant contenir des éléments absents de l'image.
- Effet possible de la quantification sur la qualité des réponses.
- À compléter avec les cas observés.
