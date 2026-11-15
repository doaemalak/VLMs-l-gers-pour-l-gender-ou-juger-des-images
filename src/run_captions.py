"""Génère les légendes pour un VLM et une précision, puis enregistre les résultats.

Usage :
    python -m src.run_captions --config configs/vlm1.yaml --precision 4bit
"""
import argparse
import json
import os

from .caption import caption_dataset
from .metrics import caption_scores, gpu_peak_gb, model_size_gb
from .models import load_vlm, unload
from .utils import load_config, load_subset, set_seed


def run(cfg_path: str, precision: str, limit: int | None = None) -> dict:
    cfg = load_config(cfg_path)
    set_seed(42)
    subset = load_subset(cfg["subset_csv"])
    if limit:
        subset = subset.head(limit)

    model, processor = load_vlm(cfg["model_id"], precision)
    preds = caption_dataset(model, processor, subset, cfg)

    summary = {
        "model": cfg["name"],
        "model_id": cfg["model_id"],
        "precision": precision,
        "n_images": len(preds),
        "model_size_gb": round(model_size_gb(model), 3),
        "gpu_peak_gb": round(gpu_peak_gb(), 3),
        "sec_per_image": round(preds["seconds"].mean(), 3),
        **{k: round(v, 4) for k, v in caption_scores(preds, subset).items()},
    }
    unload(model)

    out_dir = os.path.join(cfg["output_dir"], "captions")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cfg['name']}_{precision}"
    preds.to_csv(os.path.join(out_dir, f"{tag}.csv"), index=False)
    with open(os.path.join(out_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--precision", default="normal", choices=["normal", "8bit", "4bit"])
    p.add_argument("--limit", type=int, default=None, help="nombre d'images (test rapide)")
    a = p.parse_args()
    run(a.config, a.precision, a.limit)
