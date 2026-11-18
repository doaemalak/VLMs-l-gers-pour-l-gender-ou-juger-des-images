# Fiche du modèle

## 1. Les deux VLM (utilisés en inférence uniquement)

Deux petits modèles vision-langage, **non réentraînés** (le sujet demande de ne pas
affiner complètement un VLM), testés en précision normale (float16), 8 bits et 4 bits (NF4).

| Élément | VLM 1 | VLM 2 |
|---|---|---|
| Identifiant | Qwen/Qwen2.5-VL-3B-Instruct | HuggingFaceTB/SmolVLM-Instruct |
| Nombre de paramètres | environ 3,8 milliards | environ 2,2 milliards |
| Taille d'image utilisée | 448 px | 384 px |
| Mémoire GPU max, précision normale | 7,05 Go | 6,14 Go |
| Mémoire GPU max, 8 bits | 4,48 Go | 4,20 Go |
| Mémoire GPU max, 4 bits | 4,42 Go | 2,68 Go |
| CIDEr (normal / 8 bits / 4 bits) | 0,844 / 0,667 / 0,746 | 0,859 / 0,894 / 0,917 |
| Temps par image (normal / 8 bits / 4 bits) | 0,90 / 3,75 / 1,32 s | 1,44 / 2,46 / 1,60 s |

Mesures sur 500 images Flickr8k, GPU T4 de Colab (voir `results/tableau_comparatif.csv`).

**Choix méthodologique :** seul le **modèle de langage** est quantifié ; l'encodeur
d'images reste en float16 (la multiplication 8 bits plantait sur l'encodeur d'images
du T4). Les chiffres « 8 bits » et « 4 bits » désignent donc une quantification partielle.

**Réglages :** consigne « Describe this image in one sentence. », 40 mots au maximum,
génération sans hasard (`do_sample=False`) pour des résultats reproductibles.

## 2. Les LoRA (option A)

| Élément | Valeur |
|---|---|
| Modèle de base | Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) |
| Données | 93 photos de chiens de Flickr8k (23 autres gardées pour le test) |
| 3 versions | légendes humaines (`humain`), SmolVLM normal (`vlm2_normal`), SmolVLM 4 bits (`vlm2_4bit`) |
| Partie adaptée | couches d'attention du U-Net (to_q, to_k, to_v, to_out) |
| Rang | 8 : 1,59 M paramètres entraînés sur 861 M (0,19 %) |
| Entraînement | 400 mises à jour × 4 images, taux d'apprentissage 1e-4, 512 × 512, graine 42 |
| Coût | 13 min et 4,45 Go de GPU max par LoRA (T4) |
| Fichier | `pytorch_lora_weights.safetensors` (quelques Mo, dans Google Drive, pas dans Git) |

**Performances (184 images, CLIP ViT-L/14) :**

| Condition | CLIP texte | Ressemblance aux vraies photos |
|---|---|---|
| Sans LoRA | 28,62 | 76,97 |
| LoRA humain | 28,53 | 79,38 |
| LoRA VLM 2 normal | 28,37 | 79,06 |
| LoRA VLM 2 4 bits | 28,52 | 78,62 |

Les 3 LoRA rapprochent les images du style des photos Flickr8k sans changer le
respect du texte. Les légendes du VLM normal sont aussi efficaces que les légendes
humaines ; en 4 bits, la baisse est légère (voir `results/lora/`).

## Usage prévu

- Légender automatiquement des images pour constituer un jeu d'entraînement (option A).
- Démonstration pédagogique de l'effet de la quantification (mémoire, vitesse, qualité).

## Usages non prévus

- Décisions importantes concernant des personnes à partir d'une image.
- Remplacement d'une vérification humaine des légendes ou des images, sans contrôle.
- Génération d'images de personnes réelles ou de contenus interdits par la licence
  CreativeML OpenRAIL-M.

## Limites connues

- **Hallucinations** : le VLM 1 a inventé un enfant « déguisé en Père Noël » puis
  « en super-héros » ; le VLM 2 a écrit « sandstorm » pour un peu de poussière.
- **Légendes pauvres** : le VLM oublie souvent les couleurs et les détails.
- **Effet de la quantification variable** : forte baisse pour le VLM 1 en 8 bits,
  aucune pour le VLM 2 (cause non établie avec certitude).
- **Stable Diffusion 1.5** : « black and white dog » donne des images en noir et
  blanc ; bandes noires sur le bord de certaines images (déjà présentes sans LoRA).
- **Petite évaluation** : 23 phrases de test, 2 graines, un seul entraînement par LoRA.
