"""Chargement d'un VLM en précision normale, 8 bits ou 4 bits."""
import gc

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

PRECISIONS = ("normal", "8bit", "4bit")


def load_vlm(model_id: str, precision: str = "normal"):
    """Retourne (model, processor) pour la précision demandée."""
    if precision not in PRECISIONS:
        raise ValueError(f"Précision inconnue : {precision} (choix : {PRECISIONS})")

    # float16 partout : le GPU T4 de Colab ne gère pas bien le bfloat16.
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    # Choix méthodologique : on quantifie le modèle de langage, mais on garde
    # l'encodeur d'images (« visual » pour Qwen, « vision_model » pour SmolVLM)
    # en float16. La multiplication 8 bits de bitsandbytes plante sur
    # l'encodeur d'images (erreur cublasLt sur T4).
    # Motifs regex : transformers compare depuis le DÉBUT du nom complet
    # (ex. « model.visual.blocks.0.mlp »), d'où le préfixe « .* ».
    skip = [r".*\.visual", r".*vision_model", "lm_head"]
    if precision == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=skip,
        )
    elif precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=skip,
        )

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, processor


def unload(model) -> None:
    """Libère la mémoire GPU avant de charger un autre modèle."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@torch.inference_mode()
def ask(model, processor, image, prompt: str, max_new_tokens: int = 40) -> str:
    """Pose une question (prompt) au VLM sur une image et renvoie sa réponse."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
