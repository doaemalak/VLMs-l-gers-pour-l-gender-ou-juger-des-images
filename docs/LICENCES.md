# Licences

Licences relevées sur les pages Hugging Face des modèles (novembre 2026).
À revérifier avant toute réutilisation en dehors du cadre du cours.

## Modèles utilisés

| Rôle | Modèle | Licence | Ce que ça implique pour nous |
|---|---|---|---|
| VLM 1 | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | **Qwen Research License** | Usage de recherche et d'enseignement autorisé ; **usage commercial non autorisé** sans accord. Compatible avec un projet étudiant. |
| VLM 2 | [HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) | **Apache 2.0** | Licence très permissive, y compris commerciale, en citant la source. |
| Générateur d'images | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | **CreativeML OpenRAIL-M** | Usage libre, mais avec une **liste d'usages interdits** (contenus illégaux, nuisibles, trompeurs, atteinte aux personnes…). Nos LoRA en sont des modifications : les mêmes restrictions s'appliquent à elles. |
| Juge automatique | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) | **MIT** (dépôt OpenAI CLIP) | Licence permissive. OpenAI précise que le modèle est destiné à la recherche. |

## Données

| Jeu | Licence | Remarques |
|---|---|---|
| Flickr8k (Kaggle `adityajn105/flickr8k`) | Usage de **recherche et d'enseignement**, non commercial | Les photos viennent de Flickr et restent la propriété de leurs auteurs. Elles ne sont **pas** déposées dans Git : le script `data/download.py` les télécharge. Seule exception : 4 photos en petite taille dans `results/lora/grille_generation.png`, à titre d'illustration pédagogique (colonne « vraie photo »). |

## Bibliothèques

transformers, diffusers, peft, accelerate (Apache 2.0) ; bitsandbytes (MIT) ;
PyTorch (BSD) ; Gradio (Apache 2.0) ; pycocoevalcap (BSD).

## Images générées

Les images produites à l'étape 9 (`results/lora/images/`) sont des créations de
Stable Diffusion avec nos LoRA. Elles sont utilisées uniquement pour l'évaluation
et la présentation du projet, dans le respect de la licence CreativeML OpenRAIL-M.
