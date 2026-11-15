"""Fonctions utilitaires : configuration, graine, chargement des images."""
import random

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_image(path: str, size: int | None = None) -> Image.Image:
    """Charge une image en RGB et la redimensionne (côté le plus long = size)."""
    img = Image.open(path).convert("RGB")
    if size:
        img.thumbnail((size, size))
    return img


def load_subset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)
