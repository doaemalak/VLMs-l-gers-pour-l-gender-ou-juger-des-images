# Fiche des données

## Jeu principal : Flickr8k

| Élément | Valeur |
|---|---|
| Source | Kaggle : `adityajn105/flickr8k` |
| Contenu | Environ 8 000 images, chacune associée à cinq légendes |
| Utilisation dans le projet | Sous-ensemble de 500 à 2 000 images (voir `configs/data.yaml`) |
| Sélection | Tirage aléatoire avec graine fixe (`seed: 42`) |
| Préparation | Redimensionnement à la taille attendue par chaque VLM |
| Licence | À vérifier et à indiquer ici (usage de recherche / enseignement) |

Les **mêmes images** sont utilisées pour comparer précision normale, 8 bits et 4 bits.

## Alternative légère : Wikimedia Commons

500 à 1 000 images libres, avec une légende de référence vérifiée manuellement.
Pour chaque image, conserver l'auteur, l'URL et la licence dans `data/wikimedia_licences.csv`.

## Option B : images générées et jugements humains

| Élément | À compléter |
|---|---|
| Origine des images générées | |
| Nombre d'images | |
| Nombre d'évaluateurs humains | |
| Consigne donnée aux évaluateurs | |
| Échelle de notation | |

## Ce qui n'est pas dans Git

Les images et les poids des modèles ne sont pas déposés. Lancer `python data/download.py`
pour récupérer les données.
