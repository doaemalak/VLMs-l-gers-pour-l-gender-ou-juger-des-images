"""Mesures chiffrées : qualité des légendes, corrélation, mémoire GPU."""
import pandas as pd
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr


# ---------------------------------------------------------------- légendes
def caption_scores(pred: pd.DataFrame, subset: pd.DataFrame) -> dict:
    """Compare les légendes générées aux 5 légendes de référence (BLEU, ROUGE-L, CIDEr)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    merged = pred.merge(subset, on="image")
    norm = lambda s: str(s).lower().strip().rstrip(".")
    res = {r["image"]: [norm(r["caption"])] for _, r in merged.iterrows()}
    gts = {r["image"]: [norm(r[f"ref{i}"]) for i in range(1, 6)] for _, r in merged.iterrows()}

    bleu, _ = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return {"BLEU-4": bleu[3], "ROUGE-L": rouge, "CIDEr": cider}


# ---------------------------------------------------------------- option B
def correlation(vlm_scores, human_scores) -> dict:
    """Niveau d'accord entre les notes du VLM et celles des humains."""
    s = pd.DataFrame({"vlm": vlm_scores, "human": human_scores}).dropna()
    return {
        "n": len(s),
        "pearson": pearsonr(s.vlm, s.human)[0],
        "spearman": spearmanr(s.vlm, s.human)[0],
        "kendall": kendalltau(s.vlm, s.human)[0],
    }


# ---------------------------------------------------------------- mémoire
def gpu_peak_gb() -> float:
    """Mémoire GPU maximale utilisée depuis le dernier reset (en Go)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def model_size_gb(model) -> float:
    return model.get_memory_footprint() / 1024**3
