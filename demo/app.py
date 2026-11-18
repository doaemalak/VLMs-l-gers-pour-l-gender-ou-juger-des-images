"""Démo du projet 7.

Onglet 1 : charger une image pour obtenir une légende ou un score, avec choix
du modèle (VLM 1 / VLM 2) et du niveau de quantification.
Onglet 2 (option A) : générer une image de chien avec Stable Diffusion 1.5,
sans LoRA ou avec l'une des 3 LoRA entraînées (légendes humaines, VLM normal,
VLM 4 bits), pour comparer leur effet sur une même phrase et une même graine.

Un seul modèle est gardé en mémoire à la fois, pour rester sous 12 Go.

Usage :
    python demo/app.py            # local
    python demo/app.py --share    # lien public (Colab / Kaggle)
"""
import argparse
import os
import sys
import time

import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import gpu_peak_gb  # noqa: E402
from src.models import ask, load_vlm, unload  # noqa: E402
from src.utils import load_config  # noqa: E402

CONFIGS = {"VLM 1": "configs/vlm1.yaml", "VLM 2": "configs/vlm2.yaml"}
PRECISIONS = {"Précision normale": "normal", "8 bits": "8bit", "4 bits": "4bit"}
LORAS = {"Sans LoRA": None, "LoRA légendes humaines": "humain",
         "LoRA légendes VLM 2 normal": "vlm2_normal", "LoRA légendes VLM 2 4 bits": "vlm2_4bit"}
LORA_CFG = "configs/lora.yaml"

_current = {"key": None, "model": None, "processor": None}  # un seul modèle en mémoire


def _free() -> None:
    """Libère le modèle actuellement chargé (VLM ou Stable Diffusion)."""
    model = _current["model"]
    _current.update(key=None, model=None, processor=None)  # plus aucune référence...
    if model is not None:
        unload(model)  # ... la mémoire GPU peut donc vraiment être libérée


def get_vlm(cfg, precision):
    key = ("vlm", cfg["model_id"], precision)
    if _current["key"] != key:
        _free()
        _current["model"], _current["processor"] = load_vlm(cfg["model_id"], precision)
        _current["key"] = key
    return _current["model"], _current["processor"]


def get_pipe():
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    if _current["key"] != ("sd",):
        _free()
        cfg = load_config(LORA_CFG)
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg["train"]["base_model"], torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to("cuda")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        _current.update(key=("sd",), model=pipe, processor=None)
    return _current["model"]


def predict(image, model_name, precision_name, mode):
    if image is None:
        return "Veuillez charger une image.", ""
    cfg = load_config(CONFIGS[model_name])
    model, proc = get_vlm(cfg, PRECISIONS[precision_name])
    image.thumbnail((cfg["image_size"], cfg["image_size"]))
    prompt = cfg["caption_prompt"] if mode == "Légende" else cfg["judge_prompt"]
    t0 = time.perf_counter()
    answer = ask(model, proc, image, prompt, cfg.get("max_new_tokens", 40))
    info = f"{cfg['model_id']} · {precision_name} · {time.perf_counter() - t0:.2f} s · mémoire GPU max {gpu_peak_gb():.2f} Go"
    return answer, info


def generate(prompt, lora_name, seed):
    if not prompt.strip():
        return None, "Veuillez écrire une phrase."
    cfg = load_config(LORA_CFG)
    g = cfg["generate"]
    pipe = get_pipe()
    pipe.unload_lora_weights()  # on repart toujours du modèle d'origine
    source = LORAS[lora_name]
    if source:
        folder = os.path.join(cfg["results_dir"], source)
        if not os.path.exists(os.path.join(folder, "pytorch_lora_weights.safetensors")):
            return None, f"LoRA introuvable dans {folder} : montez Google Drive (voir notebook 05)."
        pipe.load_lora_weights(folder)
    t0 = time.perf_counter()
    gen = torch.Generator("cuda").manual_seed(int(seed))
    image = pipe(prompt, num_inference_steps=g["steps"], guidance_scale=g["guidance"], generator=gen).images[0]
    info = f"{lora_name} · graine {int(seed)} · {time.perf_counter() - t0:.1f} s · mémoire GPU max {gpu_peak_gb():.2f} Go"
    return image, info


with gr.Blocks(title="Petits VLM : légender ou évaluer") as app:
    gr.Markdown("# Petits modèles vision-langage\nProjet 7 — légender ou évaluer des images, "
                "et utiliser les légendes pour entraîner une LoRA (option A).")
    with gr.Tab("Légende ou score"):
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Image")
                model_name = gr.Radio(list(CONFIGS), value="VLM 2", label="Modèle")
                precision = gr.Radio(list(PRECISIONS), value="4 bits", label="Niveau de quantification")
                mode = gr.Radio(["Légende", "Score"], value="Légende", label="Mode")
                btn = gr.Button("Lancer", variant="primary")
            with gr.Column():
                out = gr.Textbox(label="Résultat", lines=4)
                info = gr.Textbox(label="Informations")
        btn.click(predict, [img, model_name, precision, mode], [out, info])

    with gr.Tab("Génération avec LoRA (option A)"):
        gr.Markdown("Même phrase et même graine : changez seulement la LoRA pour voir son effet.")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Phrase", value="A brown dog runs through the grass.")
                lora_name = gr.Radio(list(LORAS), value="Sans LoRA", label="LoRA")
                seed = gr.Number(value=0, precision=0, label="Graine")
                btn2 = gr.Button("Générer", variant="primary")
            with gr.Column():
                gen_img = gr.Image(label="Image générée")
                gen_info = gr.Textbox(label="Informations")
        btn2.click(generate, [prompt, lora_name, seed], [gen_img, gen_info])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    app.launch(share=p.parse_args().share)
