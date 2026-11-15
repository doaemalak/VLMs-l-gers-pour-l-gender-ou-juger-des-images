# Petits modèles vision-langage : légender ou évaluer automatiquement des images

Projet 7 — IA générative visuelle.

Nous utilisons **deux petits VLM** (modèles vision-langage) pour **légender** des images
ou **évaluer** des images générées, et nous comparons chaque modèle en
**précision normale, 8 bits et 4 bits**. Tout fonctionne sur Google Colab ou Kaggle
avec **12 Go de VRAM au maximum**.

- **Option A** : créer des légendes pour entraîner une LoRA.
- **Option B** : comparer le jugement du VLM avec celui d'humains.

## Équipe

| Membre | Rôle principal |
|---|---|
| À compléter | Données |
| À compléter | Modèles et quantification |
| À compléter | Évaluation |
| À compléter | Démo et rapport |

## Organisation du dépôt

```
README.md      Présentation, installation, lancement, résultats
notebooks/     01_donnees, 02_baseline, 03_modeles, 04_evaluation, 05_demo
src/           Code réutilisable (données, modèles, légendes, jugement, mesures)
configs/       Paramètres des expériences (YAML)
data/          Script de téléchargement et fiche des données (pas d'images)
results/       Figures, tableaux et exemples générés
demo/          Application de démonstration et courte vidéo
report/        Rapport final (6 à 8 pages)
docs/          Analyse des besoins, licences, fiche du modèle, réflexion éthique
```

## Installation

Sur Google Colab ou Kaggle (GPU activé) :

```bash
git clone https://github.com/doaemalak/VLMs-l-gers-pour-l-gender-ou-juger-des-images.git
cd VLMs-l-gers-pour-l-gender-ou-juger-des-images
pip install -r requirements.txt
```

## Lancement

```bash
# 1. Télécharger les données et créer le sous-ensemble d'images
python data/download.py --config configs/data.yaml

# 2. Générer les légendes pour un modèle et une précision
python -m src.run_captions --config configs/vlm1.yaml --precision 4bit

# 3. Lancer la démonstration
python demo/app.py
```

Précisions disponibles : `normal`, `8bit`, `4bit`.

## Résultats principaux

À compléter au fil des semaines (tableaux et figures dans `results/`).

| Modèle | Précision | Mémoire GPU (Go) | Score | Remarques |
|---|---|---|---|---|
| VLM 1 | normale | | | |
| VLM 1 | 8 bits | | | |
| VLM 1 | 4 bits | | | |
| VLM 2 | normale | | | |
| VLM 2 | 8 bits | | | |
| VLM 2 | 4 bits | | | |

## Règles de travail

- Ne pas déposer de gros fichiers de données ou de modèles dans Git.
- Faire des commits réguliers avec des messages précis.
- Utiliser des issues pour répartir les tâches.
- Chaque membre contribue de manière visible.
