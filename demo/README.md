# Démo

Application Gradio à deux onglets (lancer depuis `notebooks/05_demo.ipynb` sur Colab, GPU T4) :

1. **Légende ou score** : charger une image, choisir le modèle (VLM 1 / VLM 2),
   le niveau de quantification (précision normale, 8 bits, 4 bits) et le mode.
   L'application affiche la réponse, le temps de calcul et la mémoire GPU.
2. **Génération avec LoRA (option A)** : écrire une phrase et générer une image
   avec Stable Diffusion 1.5, sans LoRA ou avec l'une des 3 LoRA. Avec la même
   phrase et la même graine, seule la LoRA change.

Un seul modèle est gardé en mémoire à la fois, pour rester sous 12 Go.
Les LoRA sont lues dans `results/lora/` (Google Drive).

```bash
python demo/app.py --share
```

Ajouter ici une **courte vidéo** de démonstration (`demo/video_demo.mp4`, quelques Mo au maximum).
