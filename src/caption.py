"""Option A : légendage automatique des images."""
import time

import pandas as pd
from tqdm import tqdm

from .models import ask
from .utils import load_image


def caption_dataset(model, processor, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Génère une légende pour chaque image du DataFrame (colonne 'path')."""
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Légendes"):
        img = load_image(r["path"], cfg.get("image_size"))
        t0 = time.perf_counter()
        cap = ask(model, processor, img, cfg["caption_prompt"], cfg.get("max_new_tokens", 40))
        rows.append({"image": r["image"], "caption": cap, "seconds": time.perf_counter() - t0})
    return pd.DataFrame(rows)
