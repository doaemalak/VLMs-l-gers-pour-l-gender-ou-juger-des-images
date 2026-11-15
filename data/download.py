"""Télécharge Flickr8k depuis Kaggle et crée un sous-ensemble reproductible.

Usage :
    python data/download.py --config configs/data.yaml

Les images ne sont PAS déposées dans Git (voir .gitignore).
Seul le fichier data/subset/subset.csv (liste des images + légendes) est produit.
"""
import argparse
import os
import shutil
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(cfg_path: str) -> None:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import kagglehub  # nécessite un compte Kaggle (ou fonctionne directement sur Kaggle)

    src = kagglehub.dataset_download(cfg["kaggle_id"])
    print("Téléchargé dans :", src)

    raw_dir = cfg["raw_dir"]
    if not os.path.exists(raw_dir):
        shutil.copytree(src, raw_dir)

    captions = pd.read_csv(os.path.join(raw_dir, "captions.txt"))  # colonnes : image, caption
    grouped = captions.groupby("image")["caption"].apply(list).reset_index()
    grouped = grouped[grouped["caption"].apply(len) == 5]

    subset = grouped.sample(n=cfg["n_images"], random_state=cfg["seed"]).reset_index(drop=True)
    subset["path"] = subset["image"].apply(lambda x: os.path.join(raw_dir, "Images", x))
    for i in range(5):
        subset[f"ref{i + 1}"] = subset["caption"].apply(lambda c, i=i: c[i])
    subset = subset.drop(columns=["caption"])

    os.makedirs(cfg["subset_dir"], exist_ok=True)
    out = os.path.join(cfg["subset_dir"], "subset.csv")
    subset.to_csv(out, index=False)
    print(f"{len(subset)} images sélectionnées -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    main(p.parse_args().config)
