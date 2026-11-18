"""Option A, étape 8 : entraîner une LoRA de Stable Diffusion 1.5.

Principe (une « étape » d'entraînement) :
  1. on prend une photo de chien et sa légende ;
  2. le VAE compresse la photo en « latents » (une petite grille 64 x 64) ;
  3. on ajoute du bruit aléatoire aux latents ;
  4. le U-Net (le cœur de Stable Diffusion), guidé par la légende, doit deviner
     ce bruit ; l'erreur (loss) mesure à quel point il se trompe ;
  5. on corrige UNIQUEMENT les petits modules LoRA ajoutés au U-Net.
Le modèle d'origine reste gelé : seule la LoRA (quelques Mo) est enregistrée.
"""
import argparse
import json
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from src.utils import load_config, set_seed


def load_dataset(folder: str) -> list[tuple[str, str]]:
    """Lit metadata.jsonl et renvoie la liste (chemin de l'image, légende)."""
    with open(os.path.join(folder, "metadata.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    return [(os.path.join(folder, it["file_name"]), it["text"]) for it in items]


def train(cfg_path: str, source: str, max_steps: int | None = None) -> dict | None:
    """Entraîne la LoRA d'une source de légendes (humain, vlm2_normal, vlm2_4bit).

    max_steps : pour un test rapide ; les résultats vont alors dans results/lora/_test/.
    """
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    cfg = load_config(cfg_path)
    t = cfg["train"]
    test_mode = max_steps is not None
    steps = max_steps or t["max_steps"]
    out = os.path.join(cfg["results_dir"], "_test" if test_mode else "", source)
    if os.path.exists(os.path.join(out, "resume.json")):
        print(f"{source} : LoRA déjà entraînée, on passe")
        return None

    set_seed(t["seed"])  # même graine pour les 3 sources : même ordre d'images, même bruit
    device, base = "cuda", t["base_model"]

    # Les 4 morceaux de Stable Diffusion 1.5, tous gelés, en float16 (T4)
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=torch.float16).to(device)
    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    # Ajout de la LoRA sur les couches d'attention du U-Net (là où l'image « lit » le texte)
    unet.add_adapter(LoraConfig(r=t["rank"], lora_alpha=t["rank"], init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:  # la LoRA est entraînée en float32 (plus stable)
        p.data = p.data.float()
    n_params = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in unet.parameters())
    print(f"{source} : {n_params / 1e6:.2f} M paramètres entraînés sur {n_total / 1e6:.0f} M ({100 * n_params / n_total:.2f} %)")

    unet.enable_gradient_checkpointing()  # économise de la mémoire GPU
    unet.train()
    optimizer = torch.optim.AdamW(params, lr=t["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")  # évite les erreurs d'arrondi du float16

    res = t["resolution"]
    tf = transforms.Compose([
        transforms.Resize(res), transforms.CenterCrop(res),
        transforms.RandomHorizontalFlip(),  # effet miroir : un chien reste un chien
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    data = load_dataset(os.path.join(cfg["out_dir"], source, "train"))
    gen = torch.Generator().manual_seed(t["seed"])

    torch.cuda.reset_peak_memory_stats()
    start, order, losses, running = time.time(), [], [], 0.0
    for i in tqdm(range(steps * t["grad_accum"]), desc=f"LoRA {source}"):
        if not order:  # nouveau passage sur les images, dans un ordre mélangé
            order = torch.randperm(len(data), generator=gen).tolist()
        path, text = data[order.pop()]

        pixels = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device, torch.float16)
        ids = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
            hidden = text_encoder(ids)[0]
        noise = torch.randn_like(latents)
        timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        noisy = scheduler.add_noise(latents, noise, timestep)

        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy, timestep, hidden).sample
        loss = F.mse_loss(pred.float(), noise.float()) / t["grad_accum"]
        scaler.scale(loss).backward()
        running += loss.item()

        if i == 0 and any(p.grad is None for p in params):
            raise RuntimeError("La LoRA ne reçoit pas de gradient : l'entraînement ne servirait à rien.")

        if (i + 1) % t["grad_accum"] == 0:  # mise à jour après 4 images
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            losses.append(running)
            running = 0.0

    # Enregistrement de la LoRA (quelques Mo) au format lu par diffusers
    os.makedirs(out, exist_ok=True)
    lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(save_directory=out, unet_lora_layers=lora_state,
                                              safe_serialization=True)
    pd.DataFrame({"step": range(1, len(losses) + 1), "loss": losses}).to_csv(
        os.path.join(out, "loss.csv"), index=False)

    summary = {
        "source": source, "n_images": len(data), "steps": steps, "grad_accum": t["grad_accum"],
        "rank": t["rank"], "lora_params_M": round(n_params / 1e6, 2),
        "minutes": round((time.time() - start) / 60, 1),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "loss_last_50": round(float(pd.Series(losses[-50:]).mean()), 4),
    }
    with open(os.path.join(out, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(summary)

    del unet, vae, text_encoder, optimizer
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/lora.yaml")
    p.add_argument("--source", required=True, help="humain, vlm2_normal ou vlm2_4bit")
    p.add_argument("--max-steps", type=int, default=None, help="test rapide")
    a = p.parse_args()
    train(a.config, a.source, a.max_steps)
