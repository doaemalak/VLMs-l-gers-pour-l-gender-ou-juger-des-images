"""Option A, étape 9 : générer des images avec chaque LoRA et les évaluer.

4 conditions comparées, avec les MÊMES phrases et les MÊMES graines :
  sans_lora   : Stable Diffusion 1.5 d'origine (le point de départ)
  humain      : LoRA entraînée avec les légendes humaines
  vlm2_normal : LoRA entraînée avec les légendes de SmolVLM (précision normale)
  vlm2_4bit   : LoRA entraînée avec les légendes de SmolVLM (4 bits)

Deux mesures, calculées avec le modèle CLIP (qui place textes et images dans
un même espace : plus deux éléments se ressemblent, plus leur score est haut) :
  clip_texte : l'image générée correspond-elle à la phrase demandée ?
  clip_photo : l'image générée ressemble-t-elle à la VRAIE photo Flickr8k
               décrite par cette phrase ? (la LoRA a-t-elle appris le
               « style » des photos de chiens ?)
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

from src.utils import load_config, load_image

CONDITIONS = ["sans_lora", "humain", "vlm2_normal", "vlm2_4bit"]


def _paths(cfg):
    res = cfg["results_dir"]
    prompts = pd.read_csv(os.path.join(res, "prompts_test.csv"))
    return res, prompts, os.path.join(res, "images")


def _image_path(img_dir, cond, i, seed):
    return os.path.join(img_dir, cond, f"{i:02d}_s{seed}.jpg")


def generate(cfg_path: str = "configs/lora.yaml") -> None:
    """Génère 2 images par phrase de test pour chaque condition (reprise possible)."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Pas de GPU : Exécution → Modifier le type d'exécution → GPU T4.")
    cfg = load_config(cfg_path)
    g = cfg["generate"]
    res, prompts, img_dir = _paths(cfg)

    # Filtre de sécurité désactivé : il remplace parfois à tort une image par un carré
    # noir, ce qui fausserait les scores. Nos phrases sont des légendes de chiens
    # contrôlées (voir docs/ETHIQUE.md).
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["train"]["base_model"], torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    for cond in CONDITIONS:
        os.makedirs(os.path.join(img_dir, cond), exist_ok=True)
        todo = [(i, p, s) for i, p in enumerate(prompts["prompt"]) for s in g["seeds"]
                if not os.path.exists(_image_path(img_dir, cond, i, s))]
        if not todo:
            print(f"{cond} : images déjà générées, on passe")
            continue
        pipe.unload_lora_weights()  # on repart toujours du modèle d'origine
        if cond != "sans_lora":
            pipe.load_lora_weights(os.path.join(res, cond))
        for i, prompt, seed in tqdm(todo, desc=cond):
            gen = torch.Generator("cuda").manual_seed(seed)
            img = pipe(prompt, num_inference_steps=g["steps"], guidance_scale=g["guidance"],
                       generator=gen).images[0]
            img.save(_image_path(img_dir, cond, i, seed), quality=92)
    print("Mémoire GPU max (Go) :", round(torch.cuda.max_memory_allocated() / 1024**3, 3))


def _features(out) -> torch.Tensor:
    """Selon la version de transformers, CLIP renvoie un tenseur ou un objet."""
    if not torch.is_tensor(out):
        out = out.pooler_output
    return out / out.norm(dim=-1, keepdim=True)


@torch.no_grad()
def evaluate(cfg_path: str = "configs/lora.yaml") -> pd.DataFrame:
    """Calcule clip_texte et clip_photo pour chaque image générée."""
    from transformers import CLIPModel, CLIPProcessor

    cfg = load_config(cfg_path)
    res, prompts, img_dir = _paths(cfg)
    subset = pd.read_csv(cfg["subset_csv"]).set_index("image")
    name = cfg["generate"]["clip_model"]
    model = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained(name)

    def img_emb(img):
        px = proc(images=img, return_tensors="pt")["pixel_values"].to("cuda", torch.float16)
        return _features(model.get_image_features(pixel_values=px))

    rows = []
    for i, r in tqdm(prompts.iterrows(), total=len(prompts), desc="Scores CLIP"):
        t = proc(text=[r["prompt"]], padding=True, truncation=True, return_tensors="pt").to("cuda")
        txt = _features(model.get_text_features(**t))
        real = img_emb(load_image(subset.loc[r["image"], "path"]))
        for cond in CONDITIONS:
            for seed in cfg["generate"]["seeds"]:
                e = img_emb(Image.open(_image_path(img_dir, cond, i, seed)).convert("RGB"))
                rows.append({"prompt_id": i, "seed": seed, "condition": cond,
                             # CLIP score usuel : 100 x similarité cosinus, minimum 0
                             "clip_texte": max(0.0, 100 * (e @ txt.T).item()),
                             "clip_photo": 100 * (e @ real.T).item()})
    scores = pd.DataFrame(rows)
    scores.to_csv(os.path.join(res, "scores_generation.csv"), index=False)
    return scores


def summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Moyenne et écart-type par condition, dans l'ordre logique."""
    table = scores.groupby("condition")[["clip_texte", "clip_photo"]].agg(["mean", "std"]).round(2)
    return table.loc[CONDITIONS]


def paired_tests(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compare deux conditions image par image (même phrase, même graine).

    Test de Wilcoxon apparié : p < 0,05 = différence probablement réelle,
    p >= 0,05 = différence possiblement due au hasard.
    """
    wide = scores.pivot_table(index=["prompt_id", "seed"], columns="condition", values=metric)
    pairs = [("sans_lora", "humain"), ("sans_lora", "vlm2_normal"), ("sans_lora", "vlm2_4bit"),
             ("humain", "vlm2_normal"), ("humain", "vlm2_4bit"), ("vlm2_normal", "vlm2_4bit")]
    rows = []
    for a, b in pairs:
        diff = wide[b] - wide[a]
        rows.append({"comparaison": f"{b} - {a}", "écart moyen": round(diff.mean(), 3),
                     "p_value": round(wilcoxon(wide[a], wide[b]).pvalue, 4)})
    return pd.DataFrame(rows)


def grid(cfg_path: str = "configs/lora.yaml", n: int = 4, seed: int = 0) -> None:
    """Grille : une ligne par phrase ; la vraie photo puis les 4 conditions."""
    cfg = load_config(cfg_path)
    res, prompts, img_dir = _paths(cfg)
    subset = pd.read_csv(cfg["subset_csv"]).set_index("image")
    fig, axes = plt.subplots(n, 5, figsize=(17, 3.6 * n))
    for row, (i, r) in zip(axes, prompts.head(n).iterrows()):
        row[0].imshow(load_image(subset.loc[r["image"], "path"]))
        row[0].set_title("vraie photo", fontsize=10)
        row[0].set_ylabel(r["prompt"][:45], fontsize=8)
        for ax, cond in zip(row[1:], CONDITIONS):
            ax.imshow(Image.open(_image_path(img_dir, cond, i, seed)))
            ax.set_title(cond, fontsize=10)
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(res, "grille_generation.png"), bbox_inches="tight", dpi=80)
    plt.show()
