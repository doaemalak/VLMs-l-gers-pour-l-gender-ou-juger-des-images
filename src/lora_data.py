"""Option A, étape 7 : préparer les données d'entraînement de la LoRA.

1. On sélectionne les photos de chiens parmi les 500 images (d'après les
   légendes humaines).
2. On les sépare en entraînement (80 %) et test (20 %).
3. Pour chaque source de légendes (humain, vlm2_normal, vlm2_4bit), on crée un
   dossier au format « imagefolder » de Hugging Face :
       data/lora/<source>/train/xxx.jpg
       data/lora/<source>/train/metadata.jsonl   ({"file_name": ..., "text": ...})
   C'est le format lu par le script d'entraînement LoRA de diffusers.
   Les images sont les MÊMES pour les 3 sources : seules les légendes changent.
"""
import argparse
import json
import os
import re
import shutil

import pandas as pd

from src.utils import load_config


def is_dog(row: pd.Series, keywords: list[str], min_refs: int) -> bool:
    """Vrai si au moins `min_refs` légendes humaines parlent d'un chien."""
    pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE)
    refs = [row[f"ref{i}"] for i in range(1, 6)]
    return sum(bool(pattern.search(str(r))) for r in refs) >= min_refs


def load_captions(source: str, value: str, df: pd.DataFrame, captions_dir: str) -> pd.Series:
    """Renvoie une légende par image (index = nom de l'image) pour une source."""
    if value in df.columns:  # légende humaine, ex. « ref1 »
        return df.set_index("image")[value]
    path = os.path.join(captions_dir, f"{value}.csv")
    return pd.read_csv(path).set_index("image")["caption"]


def clean(text: str) -> str:
    """Nettoyage léger : espaces en trop et « dog . » -> « dog. »."""
    text = re.sub(r"\s+([.,!?])", r"\1", str(text))
    return re.sub(r"\s+", " ", text).strip()


def prepare(cfg_path: str) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    df = pd.read_csv(cfg["subset_csv"])

    dogs = df[df.apply(is_dog, axis=1, keywords=cfg["keywords"], min_refs=cfg["min_refs"])]
    test = dogs.sample(frac=cfg["test_fraction"], random_state=cfg["seed"])
    train = dogs.drop(test.index)
    print(f"{len(dogs)} photos de chiens : {len(train)} entraînement, {len(test)} test")

    rows = []
    for source, value in cfg["sources"].items():
        caps = load_captions(source, value, df, cfg["captions_dir"])
        folder = os.path.join(cfg["out_dir"], source, "train")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "metadata.jsonl"), "w", encoding="utf-8") as f:
            for _, r in train.iterrows():
                shutil.copy(r["path"], os.path.join(folder, r["image"]))
                text = clean(caps[r["image"]])
                f.write(json.dumps({"file_name": r["image"], "text": text}, ensure_ascii=False) + "\n")
                rows.append({"image": r["image"], "source": source, "caption": text})
        print(f"  {source:12s} -> {folder}")

    # Récapitulatif : la même image avec ses 3 légendes côte à côte
    os.makedirs(cfg["results_dir"], exist_ok=True)
    recap = pd.DataFrame(rows).pivot(index="image", columns="source", values="caption")
    recap.to_csv(os.path.join(cfg["results_dir"], "legendes_entrainement.csv"))

    # Phrases de test (étape 9) : légende humaine des images jamais vues
    test_prompts = test[["image", "ref1"]].rename(columns={"ref1": "prompt"})
    test_prompts["prompt"] = test_prompts["prompt"].map(clean)
    test_prompts.to_csv(os.path.join(cfg["results_dir"], "prompts_test.csv"), index=False)
    return recap


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/lora.yaml")
    prepare(p.parse_args().config)
