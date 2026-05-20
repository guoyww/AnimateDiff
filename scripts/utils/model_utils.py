# -------------------------
# scripts/utils/model_utils.py
# -------------------------
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path

from diffusers import DDIMScheduler, LMSDiscreteScheduler

def get_text_embeddings(text_encoder, tokenizer, prompt, negative_prompt="", device="cuda", dtype=torch.float32):
    """
    Retourne les embeddings textuels positifs et négatifs pour l'inference guidée.

    Args:
        text_encoder: modèle CLIPTextModel chargé
        tokenizer: tokenizer CLIP
        prompt: texte positif (str)
        negative_prompt: texte négatif (str)
        device: "cuda" ou "cpu"
        dtype: torch dtype (float32 ou float16)

    Returns:
        pos_embeds, neg_embeds: tensors shape [1, seq_len, hidden_dim]
    """

    # --- Tokenization ---
    pos_inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
    neg_inputs = tokenizer(negative_prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")

    # --- Passage dans le text encoder ---
    with torch.no_grad():
        pos_embeds = text_encoder(pos_inputs.input_ids.to(device))[0].to(dtype)
        neg_embeds = text_encoder(neg_inputs.input_ids.to(device))[0].to(dtype)

    return pos_embeds, neg_embeds

# -------------------------
# Chargement UNet
# -------------------------
def load_pretrained_unet(pretrained_path: str, device: str = "cuda", dtype=torch.float32):
    unet_path = Path(pretrained_path)
    unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", torch_dtype=dtype)
    unet.to(device)
    unet.eval()
    return unet

# -------------------------
# Chargement scheduler
# -------------------------
def load_scheduler(pretrained_path: str):
    scheduler_path = Path(pretrained_path)
    scheduler = LMSDiscreteScheduler.from_pretrained(scheduler_path, subfolder="scheduler")
    return scheduler


# -------------------------
# Chargement scheduler
# -------------------------

def load_DDIMScheduler(model_path, scheduler_type="DDIMScheduler"):
    """
    Charge le scheduler de diffusion depuis un modèle pré-entraîné.
    Renvoie un objet Scheduler, pas un tensor.
    """
    if scheduler_type == "DDIMScheduler":
        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif scheduler_type == "LMSDiscreteScheduler":
        scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
    else:
        raise ValueError(f"Scheduler inconnu: {scheduler_type}")

    return scheduler

# -------------------------
# Chargement Text Encoder + Tokenizer
# -------------------------
def load_text_encoder(pretrained_path: str, device: str = "cuda"):
    encoder_path = Path(pretrained_path)
    tokenizer = CLIPTokenizer.from_pretrained(encoder_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(encoder_path, subfolder="text_encoder")
    text_encoder.to(device)
    text_encoder.eval()
    return text_encoder, tokenizer
