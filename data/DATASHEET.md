# Fiche des données

## Jeu principal : Flickr8k

| Élément | Valeur |
|---|---|
| Source | Kaggle : `adityajn105/flickr8k` |
| Contenu | Environ 8 000 images, chacune associée à cinq légendes |
| Utilisation dans le projet | Sous-ensemble de 500 images (voir `configs/data.yaml`) |
| Sélection | Tirage aléatoire avec graine fixe (`seed: 42`) |
| Préparation | Redimensionnement à la taille attendue par chaque VLM |
| Licence | Usage de recherche et d'enseignement, non commercial (voir `docs/LICENCES.md`) |

Les **mêmes images** sont utilisées pour comparer précision normale, 8 bits et 4 bits.

## Alternative légère : Wikimedia Commons

500 à 1 000 images libres, avec une légende de référence vérifiée manuellement.
Pour chaque image, conserver l'auteur, l'URL et la licence dans `data/wikimedia_licences.csv`.

## Option A : données de la LoRA (photos de chiens)

| Élément | Valeur |
|---|---|
| Origine | Les 500 images du sous-ensemble ci-dessus |
| Sélection | Image gardée si au moins 3 des 5 légendes humaines contiennent dog, dogs, puppy ou puppies |
| Nombre | 116 photos : 93 pour l'entraînement, 23 pour le test (graine 42) |
| Légendes d'entraînement | 3 versions : humaine (ref1), SmolVLM normal, SmolVLM 4 bits |
| Phrases de test | Légendes humaines (ref1) des 23 photos de test |
| Format | `data/lora/<source>/train/` + `metadata.jsonl` (créé par `src/lora_data.py`) |
| Pourquoi des chiens | Thème cohérent et fréquent, facile à juger, sans visages de personnes à reproduire |

## Ce qui n'est pas dans Git

Les images et les poids des modèles ne sont pas déposés. Lancer `python data/download.py`
pour récupérer les données.
