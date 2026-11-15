"""Démo : charger une image pour obtenir une légende ou un score,
avec choix du modèle et du niveau de quantification.

Usage :
    python demo/app.py            # local
    python demo/app.py --share    # lien public (Colab / Kaggle)
"""
import argparse
import os
import sys
import time

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import gpu_peak_gb  # noqa: E402
from src.models import ask, load_vlm, unload  # noqa: E402
from src.utils import load_config  # noqa: E402

CONFIGS = {"VLM 1": "configs/vlm1.yaml", "VLM 2": "configs/vlm2.yaml"}
PRECISIONS = {"Précision normale": "normal", "8 bits": "8bit", "4 bits": "4bit"}

_current = {"key": None, "model": None, "processor": None}  # un seul modèle en mémoire


def get_model(cfg, precision):
    key = (cfg["model_id"], precision)
    if _current["key"] != key:
        if _current["model"] is not None:
            unload(_current["model"])
        _current["model"], _current["processor"] = load_vlm(cfg["model_id"], precision)
        _current["key"] = key
    return _current["model"], _current["processor"]


def predict(image, model_name, precision_name, mode):
    if image is None:
        return "Veuillez charger une image.", ""
    cfg = load_config(CONFIGS[model_name])
    model, proc = get_model(cfg, PRECISIONS[precision_name])
    image.thumbnail((cfg["image_size"], cfg["image_size"]))
    prompt = cfg["caption_prompt"] if mode == "Légende" else cfg["judge_prompt"]
    t0 = time.perf_counter()
    answer = ask(model, proc, image, prompt, cfg.get("max_new_tokens", 40))
    info = f"{cfg['model_id']} · {precision_name} · {time.perf_counter() - t0:.2f} s · mémoire GPU max {gpu_peak_gb():.2f} Go"
    return answer, info


with gr.Blocks(title="Petits VLM : légender ou évaluer") as app:
    gr.Markdown("# Petits modèles vision-langage\nChargez une image pour obtenir une légende ou un score.")
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Image")
            model_name = gr.Radio(list(CONFIGS), value="VLM 1", label="Modèle")
            precision = gr.Radio(list(PRECISIONS), value="4 bits", label="Niveau de quantification")
            mode = gr.Radio(["Légende", "Score"], value="Légende", label="Mode")
            btn = gr.Button("Lancer", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Résultat", lines=4)
            info = gr.Textbox(label="Informations")
    btn.click(predict, [img, model_name, precision, mode], [out, info])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true")
    app.launch(share=p.parse_args().share)
