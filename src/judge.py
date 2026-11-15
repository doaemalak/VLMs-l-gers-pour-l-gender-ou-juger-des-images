"""Option B : le VLM juge des images (score), à comparer avec des humains."""
import re

import pandas as pd
from tqdm import tqdm

from .models import ask
from .utils import load_image


def parse_score(text: str) -> float | None:
    """Extrait le premier nombre de la réponse du VLM (None si absent)."""
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    return float(m.group().replace(",", ".")) if m else None


def judge_images(model, processor, paths: list[str], cfg: dict) -> pd.DataFrame:
    rows = []
    for p in tqdm(paths, desc="Jugements"):
        img = load_image(p, cfg.get("image_size"))
        answer = ask(model, processor, img, cfg["judge_prompt"], max_new_tokens=10)
        rows.append({"path": p, "answer": answer, "score": parse_score(answer)})
    return pd.DataFrame(rows)
