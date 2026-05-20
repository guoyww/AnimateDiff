# n3r_utils.py
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torchvision.utils import save_image
import ffmpeg
import torch.nn as nn
import math
from tqdm import tqdm
from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
import os
import numpy as np
import yaml
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import CLIPTokenizer, CLIPTextModel
from einops import rearrange
from math import ceil
import copy
from torchvision.transforms import ToPILImage

LATENT_SCALE = 0.18215  # échelle typique pour SD/AnimateDiff


def adapt_embeddings_to_unet(pos_embeds, neg_embeds, target_dim):
    """Adapte automatiquement les embeddings texte pour correspondre au cross_attention_dim du UNet."""
    current_dim = pos_embeds.shape[-1]
    if current_dim == target_dim:
        return pos_embeds, neg_embeds
    # Troncature
    if current_dim > target_dim:
        pos_embeds = pos_embeds[..., :target_dim]
        neg_embeds = neg_embeds[..., :target_dim]
    # Padding
    elif current_dim < target_dim:
        pad = target_dim - current_dim
        pos_embeds = torch.nn.functional.pad(pos_embeds, (0, pad))
        neg_embeds = torch.nn.functional.pad(neg_embeds, (0, pad))
    return pos_embeds, neg_embeds

# -----n3r_utils----------- Ultra-safe embeddings pour UNet ----------------
# 🔹 Projection embeddings pour UNet
# Singleton pour la projection UNet

class UNetEmbeddingProjector(nn.Module):
    _singleton_instance = None

    def __init__(self, in_dim=1024, out_dim=768, device='cuda'):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim).to(device)

    @classmethod
    def get_instance(cls, in_dim=1024, out_dim=768, device='cuda'):
        if cls._singleton_instance is None:
            cls._singleton_instance = cls(in_dim, out_dim, device)
        return cls._singleton_instance

    def forward(self, embeds):
        if embeds.shape[-1] != 1024:
            raise ValueError(
                f"❌ Embeddings inattendus : dernier dim={embeds.shape[-1]}, attendu=1024"
            )
        return self.proj(embeds)


def prepare_embeddings_for_unet(embeds, device='cuda'):
    embeds = embeds.to(device)
    projector = UNetEmbeddingProjector.get_instance(device=device)
    projected = projector(embeds)
    if projected.shape[-1] != 768:
        raise ValueError(
            f"❌ UNet embeddings projetés ont mauvaise dimension : {projected.shape[-1]}, attendu=768"
        )
    return projected

# ---------------- Ultra-safe decode patchwise ----------------
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor


def generate_latents_safe_debug(unet, **kwargs):
    """
    Génération de latents avec UNet, FP16-safe et debug.
    Affiche min/max à chaque étape pour vérifier les latents.
    """

    # Arguments connus
    known_kwargs = [
        "scheduler", "input_latents", "embeddings", "motion_module",
        "guidance_scale", "device", "fp16", "steps", "debug"
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_kwargs}

    # Arguments obligatoires
    input_latents = filtered_kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' doit être fourni")

    device = filtered_kwargs.get("device", "cuda")
    fp16 = filtered_kwargs.get("fp16", True)
    debug = filtered_kwargs.get("debug", False)
    motion_module = filtered_kwargs.get("motion_module", None)
    scheduler = filtered_kwargs.get("scheduler", None)
    steps = filtered_kwargs.get("steps", 20)
    embeddings = filtered_kwargs.get("embeddings", None)
    guidance_scale = filtered_kwargs.get("guidance_scale", 4.0)

    # Latents initiaux
    latents = input_latents.clone().to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    is_video = latents.ndim == 5  # [B, C, F, H, W]
    steps_list = getattr(scheduler, "timesteps", range(steps))

    if debug:
        print(f"[DEBUG] Initial latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    # Désactiver Dynamo/Inductor pour debug
    torch._dynamo.reset()
    torch._dynamo.disable()

    for step_idx, t in enumerate(steps_list):

        # Motion module si présent
        if motion_module is not None:
            latents = motion_module(latents)
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

        # Préparer latents pour UNet
        if is_video:
            B, C, F, H, W = latents.shape
            latents_unet = latents.reshape(B*F, C, H, W)
        else:
            latents_unet = latents

        # FP16 si demandé
        latents_unet = latents_unet.half() if fp16 else latents_unet.float()

        # 🔹 Préparer embeddings (concat pour CF guidance)
        if isinstance(embeddings, tuple):
            pos_embeds, neg_embeds = embeddings
            encoder_states = torch.cat([neg_embeds.to(device), pos_embeds.to(device)], dim=0)
        else:
            encoder_states = embeddings

        # 🔹 Forward UNet
        try:
            unet_out = unet(latents_unet, t, encoder_hidden_states=encoder_states)

            # Détecter type de sortie UNet
            if isinstance(unet_out, dict):
                latents_out = unet_out["sample"]
            elif isinstance(unet_out, (tuple, list)):
                latents_out = unet_out[0]
            else:
                raise TypeError(f"Unexpected UNet output type: {type(unet_out)}")

        except Exception as e:
            print(f"⚠️ [UNet ERROR] step={step_idx} - {e}")
            continue

        # Reshape si vidéo
        if is_video:
            latents = latents_out.reshape(B, F, C, H, W)
        else:
            latents = latents_out

        # Clamp et nettoyage
        latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        if debug:
            print(f"[DEBUG Step {step_idx}] latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    if debug:
        print(f"[INFO] Finished latents generation, final min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    return latents


# ---------------- UTILS ----------------
def prepare_embeddings_for_unet_safe(pos_embeds, neg_embeds, target_dim=512, device="cuda"):
    """
    Prépare les embeddings pour UNet en tronquant ou projetant à target_dim.
    Concatène négatifs + positifs pour classifier-free guidance.
    LoRA peut rester sur 1024, UNet reçoit target_dim.
    """
    pos_embeds = pos_embeds.to(device)
    neg_embeds = neg_embeds.to(device)

    # Tronquer si plus grand que target_dim
    if pos_embeds.size(-1) > target_dim:
        pos_embeds = pos_embeds[..., :target_dim]
        neg_embeds = neg_embeds[..., :target_dim]

    # Concat CF guidance
    encoder_states = torch.cat([neg_embeds, pos_embeds], dim=0)
    return encoder_states


# ---------------- Latents generator wrapper ----------------
# ---------------- wrapper ultra-safe ----------------
def generate_latents_safe_wrapper(unet, **kwargs):
    """
    Wrapper ultra-safe pour générer les latents avec UNet.
    - Gère CUDA OOM
    - Accepte directement `encoder_states` pour UNet (projection déjà faite)
    """

    input_latents = kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' must be provided")

    encoder_states = kwargs.get("encoder_states", None)
    motion_module = kwargs.get("motion_module", None)
    guidance_scale = kwargs.get("guidance_scale", 1.0)
    scheduler = kwargs.get("scheduler", None)
    device = kwargs.get("device", "cuda")
    fp16 = kwargs.get("fp16", True)
    steps = kwargs.get("steps", 20)
    debug = kwargs.get("debug", False)

    filtered_kwargs = {
        "input_latents": input_latents,
        "scheduler": scheduler,
        "embeddings": encoder_states,  # juste passer encoder_states comme embeddings
        "motion_module": motion_module,
        "guidance_scale": guidance_scale,
        "device": device,
        "fp16": fp16,
        "steps": steps,
        "debug": debug
    }

    try:
        return generate_latents_safe_debug(unet, **filtered_kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ [SAFE WRAPPER] CUDA out of memory caught, returning input latents")
            return input_latents.clone()
        else:
            raise e


import torch

def generate_latents_safe_debug_v3(unet, **kwargs):
    """
    Génération de latents avec UNet, FP16-safe et debug.
    Corrige automatiquement la dimension des embeddings pour cross-attention UNet.
    Version vidéo stable : toutes les frames sont conservées.
    Affiche min/max à chaque étape pour détecter si les latents restent à zéro.
    """
    # Arguments connus
    known_kwargs = [
        "scheduler", "input_latents", "embeddings", "motion_module",
        "guidance_scale", "device", "fp16", "steps", "debug"
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_kwargs}

    # Arguments requis
    input_latents = filtered_kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' doit être fourni")

    device = filtered_kwargs.get("device", "cuda")
    fp16 = filtered_kwargs.get("fp16", True)
    debug = filtered_kwargs.get("debug", False)
    motion_module = filtered_kwargs.get("motion_module", None)
    scheduler = filtered_kwargs.get("scheduler", None)
    steps = filtered_kwargs.get("steps", 20)
    embeddings = filtered_kwargs.get("embeddings", None)

    # Latents initiaux
    latents = input_latents.clone().to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    is_video = latents.ndim == 5  # [B, C, F, H, W]
    steps_list = getattr(scheduler, "timesteps", range(steps))

    if debug:
        print(f"[DEBUG] Initial latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    # Désactiver Dynamo/Inductor pour debug
    torch._dynamo.reset()
    torch._dynamo.disable()

    # Récupération cross_attention_dim du UNet
    unet_cross_attention_dim = getattr(unet.config, "cross_attention_dim", 768)  # SD1.x = 768

    # Fonction interne pour adapter embeddings
    def adapt_embeddings(pos_embeds, neg_embeds, target_dim):
        current_dim = pos_embeds.shape[-1]
        if current_dim == target_dim:
            return pos_embeds, neg_embeds
        elif current_dim > target_dim:
            pos_embeds = pos_embeds[..., :target_dim]
            neg_embeds = neg_embeds[..., :target_dim]
        else:
            pad = target_dim - current_dim
            pos_embeds = torch.nn.functional.pad(pos_embeds, (0, pad))
            neg_embeds = torch.nn.functional.pad(neg_embeds, (0, pad))
        return pos_embeds, neg_embeds

    for step_idx, t in enumerate(steps_list):
        # Motion module si présent
        if motion_module is not None:
            latents = motion_module(latents)
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

        # Préparer latents pour UNet
        if is_video:
            B, C, F, H, W = latents.shape
            latents_unet = latents.reshape(B*F, C, H, W)
        else:
            latents_unet = latents

        # FP16 si demandé
        latents_unet = latents_unet.half() if fp16 else latents_unet.float()

        # Préparer embeddings pour CF guidance
        encoder_states = None
        if isinstance(embeddings, tuple) and len(embeddings) == 2:
            pos_embeds, neg_embeds = embeddings
            pos_embeds, neg_embeds = adapt_embeddings(pos_embeds.to(device), neg_embeds.to(device), unet_cross_attention_dim)
            encoder_states = torch.cat([neg_embeds, pos_embeds], dim=0)
        elif embeddings is not None:
            # Simple embeddings
            if embeddings.shape[-1] != unet_cross_attention_dim:
                linear_proj = torch.nn.Linear(embeddings.shape[-1], unet_cross_attention_dim).to(device)
                encoder_states = linear_proj(embeddings.to(device))
            else:
                encoder_states = embeddings.to(device)

        # 🔹 Forward UNet
        try:
            unet_out = unet(latents_unet, t, encoder_hidden_states=encoder_states)
            if isinstance(unet_out, dict):
                latents_out = unet_out["sample"]
            elif isinstance(unet_out, (tuple, list)):
                latents_out = unet_out[0]
            else:
                raise TypeError(f"Unexpected UNet output type: {type(unet_out)}")
        except Exception as e:
            print(f"⚠️ [UNet ERROR] step={step_idx} - {e}")
            continue

        # Reshape si vidéo
        if is_video:
            latents = latents_out.reshape(B, F, C, H, W)
        else:
            latents = latents_out

        # Nettoyage et clamp
        latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        if debug:
            print(f"[DEBUG Step {step_idx}] latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    if debug:
        print(f"[INFO] Finished latents generation, final min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    return latents




def generate_latents_safe_debug_v2(unet, **kwargs):
    """
    Génération de latents avec UNet, FP16-safe et debug.
    Version vidéo stable : toutes les frames sont conservées.
    Affiche min/max à chaque étape pour détecter si les latents restent à zéro.
    """

    import torch

    # Arguments connus
    known_kwargs = [
        "scheduler", "input_latents", "embeddings", "motion_module",
        "guidance_scale", "device", "fp16", "steps", "debug"
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_kwargs}

    # Required
    input_latents = filtered_kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' doit être fourni")

    device = filtered_kwargs.get("device", "cuda")
    fp16 = filtered_kwargs.get("fp16", True)
    debug = filtered_kwargs.get("debug", False)
    motion_module = filtered_kwargs.get("motion_module", None)
    scheduler = filtered_kwargs.get("scheduler", None)
    steps = filtered_kwargs.get("steps", 20)
    embeddings = filtered_kwargs.get("embeddings", None)

    # Latents initiaux
    latents = input_latents.clone().to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    is_video = latents.ndim == 5  # [B, C, F, H, W]
    steps_list = getattr(scheduler, "timesteps", range(steps))

    if debug:
        print(f"[DEBUG] Initial latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    # Désactiver Dynamo/Inductor pour debug
    torch._dynamo.reset()
    torch._dynamo.disable()

    for step_idx, t in enumerate(steps_list):

        # Motion module si présent
        if motion_module is not None:
            latents = motion_module(latents)
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

        # Préparer latents pour UNet
        if is_video:
            B, C, F, H, W = latents.shape
            latents_unet = latents.reshape(B*F, C, H, W)
        else:
            latents_unet = latents

        # FP16 si demandé
        latents_unet = latents_unet.half() if fp16 else latents_unet.float()

        # 🔹 Préparer embeddings
        if isinstance(embeddings, tuple):
            pos_embeds, neg_embeds = embeddings
            encoder_states = torch.cat([neg_embeds.to(device), pos_embeds.to(device)], dim=0)
        else:
            encoder_states = embeddings

        # 🔹 Forward UNet
        try:
            unet_out = unet(latents_unet, t, encoder_hidden_states=encoder_states)

            # Détecter type de sortie UNet
            if isinstance(unet_out, dict):
                latents_out = unet_out["sample"]
            elif isinstance(unet_out, (tuple, list)):
                latents_out = unet_out[0]  # prendre la première valeur (sample)
            else:
                raise TypeError(f"Unexpected UNet output type: {type(unet_out)}")

        except Exception as e:
            print(f"⚠️ [UNet ERROR] step={step_idx} - {e}")
            continue

        # Reshape si vidéo
        if is_video:
            latents = latents_out.reshape(B, F, C, H, W)
        else:
            latents = latents_out

        # Nettoyage et clamp
        latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        if debug:
            print(f"[DEBUG Step {step_idx}] latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    if debug:
        print(f"[INFO] Finished latents generation, final min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    return latents


# ---------------- Wrapper ultra-safe ----------------
# ici  plus d'erreur cross_attention_dim mais génération reste incorrect: ⚠ [UNet ERROR] step=3 - The size of tensor a (2048) must match the size of tensor b (4096) at non-singleton dimension 1

import torch
import torch.nn.functional as F

# ---------------- SAFE UNET FORWARD ----------------
def safe_forward_unet(unet, latents, t, encoder_hidden_states=None, device="cuda", fp16=True, debug=False):
    """
    Forward UNet sécurisé pour mini GPU.
    - Ajuste automatiquement H/W si nécessaire
    - Retourne toujours les latents (sample)
    """
    # Vérifie H/W selon sample_size du UNet
    target_H = getattr(unet.config, "sample_size", latents.shape[2]*8) // 8
    target_W = getattr(unet.config, "sample_size", latents.shape[3]*8) // 8
    if (latents.shape[2], latents.shape[3]) != (target_H, target_W):
        latents = F.interpolate(latents, size=(target_H, target_W),
                                mode="bilinear", align_corners=False)
        if debug:
            print(f"[DEBUG] Latents resized to {latents.shape[2:]}")

    # Mettre en fp16 si demandé
    if fp16:
        latents = latents.half()

    # Forward UNet
    out = unet(latents, t, encoder_hidden_states=encoder_hidden_states)
    if isinstance(out, dict):
        sample = out.get("sample", None)
        if sample is None:
            raise ValueError("UNet output dict does not contain 'sample'")
        return sample
    elif isinstance(out, (tuple, list)):
        return out[0]
    else:
        raise TypeError(f"Unexpected UNet output type: {type(out)}")


def run_diffusion_pipeline(unet, vae, scheduler, images, embeddings,
                           timesteps, guidance_scale=4.0, device="cuda",
                           fp16=True, debug=True):
    """
    Pipeline complet avec guidance scale :
    - Encode les images en latents
    - Forward UNet étape par étape (safe)
    - Applique le guidance scale correctement
    """
    # Encoder les images en latents
    latents = encode_images_to_latents_safe(images, vae, device=device)
    latents = latents.to(device=device, dtype=torch.float16 if fp16 else torch.float32)

    # Récupère les embeddings pos/neg
    pos_embeds, neg_embeds = embeddings
    # Duplique latents pour concat pos+neg
    latents_input = torch.cat([latents, latents], dim=0)
    # Concat embeddings
    encoder_hidden_states = torch.cat([pos_embeds.to(device), neg_embeds.to(device)], dim=0)

    for step, timestep in enumerate(timesteps):
        if debug:
            print(f"\n[INFO] Step {step}/{len(timesteps)-1} - timestep={timestep}")

        # 🔹 Forward UNet sécurisé
        noise_pred_all = safe_forward_unet(
            unet=unet,
            latents=latents_input,
            t=timestep,
            encoder_hidden_states=encoder_hidden_states,
            device=device
        )

        # 🔹 Appliquer guidance scale
        batch = latents.shape[0]
        # Formula: noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        noise_pred = noise_pred_all[batch:] * guidance_scale + noise_pred_all[:batch] * (1 - guidance_scale)

        # 🔹 Mise à jour des latents via scheduler
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

        # 🔹 Debug min/max pour NaN/Inf
        if debug:
            print(f"[DEBUG] Latents min/max: {latents.min().item():.4f}/{latents.max().item():.4f}")
            nan_count = torch.isnan(latents).sum().item()
            inf_count = torch.isinf(latents).sum().item()
            if nan_count or inf_count:
                print(f"⚠️ [DEBUG Step {step}] Found NaN: {nan_count}, Inf: {inf_count}")

    return latents


# ---------------- LATENTS GENERATION SAFE DEBUG ----------------

def generate_latents_safe_debug_v5(unet, **kwargs):
    """
    Génération de latents avec UNet, FP16-safe et debug.
    Supporte init_image_scale et creative_noise.
    Logs détaillés : shape, min/max, cross_attention_dim, NaN/Inf.
    Compatible mini GPU.
    """
    import torch
    import torch.nn.functional as F

    # ---------------- Filtrage kwargs ----------------
    known_kwargs = [
        "scheduler", "input_latents", "embeddings", "motion_module",
        "guidance_scale", "device", "fp16", "steps", "debug",
        "init_image_scale", "creative_noise"
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_kwargs}

    input_latents = filtered_kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' doit être fourni")

    device = filtered_kwargs.get("device", "cuda")
    fp16 = filtered_kwargs.get("fp16", True)
    debug = filtered_kwargs.get("debug", False)
    motion_module = filtered_kwargs.get("motion_module", None)
    scheduler = filtered_kwargs.get("scheduler", None)
    steps = filtered_kwargs.get("steps", 20)
    embeddings = filtered_kwargs.get("embeddings", None)

    init_image_scale = filtered_kwargs.get("init_image_scale", 1.0)
    creative_noise = filtered_kwargs.get("creative_noise", 0.0)

    latents = input_latents.clone().to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    original_latents = latents.clone()
    is_video = latents.ndim == 5  # [B,C,F,H,W]
    steps_list = getattr(scheduler, "timesteps", range(steps))

    if debug:
        print(f"[DEBUG] Initial latents shape: {latents.shape}")
        print(f"[DEBUG] Initial latents min/max: {latents.min().item():.4f}/{latents.max().item():.4f}")
        if motion_module:
            print(f"[DEBUG] Motion module detected: {motion_module}")
        if embeddings:
            if isinstance(embeddings, tuple):
                print(f"[DEBUG] Embeddings shapes - pos: {embeddings[0].shape}, neg: {embeddings[1].shape}")
            else:
                print(f"[DEBUG] Embeddings shape: {embeddings.shape}")

    # ---------------- Désactive Dynamo ----------------
    torch._dynamo.reset()
    torch._dynamo.disable()

    # ---------------- Boucle de steps ----------------
    for step_idx, t in enumerate(steps_list):
        if debug:
            print(f"\n[DEBUG] Step {step_idx}/{len(steps_list)} - timestep {t}")

        # ---------------- Motion Module ----------------
        if motion_module is not None:
            latents_before_motion = latents.clone()
            latents = motion_module(latents)
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
            if debug:
                diff = (latents - latents_before_motion).abs().max()
                print(f"[DEBUG] Motion module diff max abs: {diff:.4f}")

        # ---------------- Préparer latents pour UNet ----------------
        if is_video:
            B, C, F, H, W = latents.shape
            latents_unet = latents.reshape(B*F, C, H, W)
        else:
            latents_unet = latents

        latents_unet = latents_unet.half() if fp16 else latents_unet.float()

        # ---------------- Préparer embeddings ----------------
        if isinstance(embeddings, tuple):
            pos_embeds, neg_embeds = embeddings
            encoder_states = pos_embeds.to(device)  # ⚠ seulement pos_embeds
        else:
            encoder_states = embeddings

        if debug:
            print(f"[DEBUG] Latents input to UNet shape: {latents_unet.shape}")
            if embeddings: print(f"[DEBUG] Encoder states shape: {encoder_states.shape}")

        # ---------------- UNet SAFE FORWARD ----------------
        try:
            latents_out = safe_forward_unet(unet, latents_unet, t, encoder_hidden_states=encoder_states, device=device)
        except Exception as e:
            print(f"⚠️ [UNet ERROR] step={step_idx} - {e}")
            continue

        # ---------------- Reshape si vidéo ----------------
        if is_video:
            latents = latents_out.reshape(B, F, C, H, W)
        else:
            latents = latents_out

        # ---------------- Clamp & Nan/Inf ----------------
        latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        # ---------------- Ajout init_image_scale ----------------
        if init_image_scale > 0.0:
            # Redimension original_latents vers latents actuels
            resized_orig = F.interpolate(original_latents, size=latents.shape[-2:], mode='bilinear', align_corners=False)
            latents = init_image_scale * resized_orig + (1.0 - init_image_scale) * latents

        # ---------------- Ajout creative_noise ----------------
        if creative_noise > 0.0:
            noise = torch.randn_like(latents) * creative_noise
            latents = latents + noise

        if debug:
            nan_count = torch.isnan(latents).sum().item()
            inf_count = torch.isinf(latents).sum().item()
            print(f"[DEBUG Step {step_idx}] latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}, NaN={nan_count}, Inf={inf_count}")

    if debug:
        print(f"\n[INFO] Finished latents generation, final shape={latents.shape}, min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    return latents



# ---------------- LATENTS GENERATION SAFE DEBUG (avec init_image_scale & creative_noise) ----------------

def generate_latents_safe_debug_v4(unet, **kwargs):
    """
    Génération de latents avec UNet, FP16-safe et debug.
    Logs détaillés : shape, min/max, cross_attention_dim, NaN/Inf.
    Compatible mini GPU.
    """
    import torch
    import torch.nn.functional as F

    # ---------------- Filtrage kwargs ----------------
    known_kwargs = [
        "scheduler", "input_latents", "embeddings", "motion_module",
        "guidance_scale", "device", "fp16", "steps", "debug"
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_kwargs}

    input_latents = filtered_kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' doit être fourni")

    device = filtered_kwargs.get("device", "cuda")
    fp16 = filtered_kwargs.get("fp16", True)
    debug = filtered_kwargs.get("debug", False)
    motion_module = filtered_kwargs.get("motion_module", None)
    scheduler = filtered_kwargs.get("scheduler", None)
    steps = filtered_kwargs.get("steps", 20)
    embeddings = filtered_kwargs.get("embeddings", None)

    latents = input_latents.clone().to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    is_video = latents.ndim == 5  # [B,C,F,H,W]
    steps_list = getattr(scheduler, "timesteps", range(steps))

    if debug:
        print(f"[DEBUG] Initial latents shape: {latents.shape}")
        print(f"[DEBUG] Initial latents min/max: {latents.min().item():.4f}/{latents.max().item():.4f}")
        if motion_module:
            print(f"[DEBUG] Motion module detected: {motion_module}")
        if embeddings:
            if isinstance(embeddings, tuple):
                print(f"[DEBUG] Embeddings shapes - pos: {embeddings[0].shape}, neg: {embeddings[1].shape}")
            else:
                print(f"[DEBUG] Embeddings shape: {embeddings.shape}")

    # ---------------- Désactive Dynamo ----------------
    torch._dynamo.reset()
    torch._dynamo.disable()

    # ---------------- Boucle de steps ----------------
    for step_idx, t in enumerate(steps_list):
        if debug:
            print(f"\n[DEBUG] Step {step_idx}/{len(steps_list)} - timestep {t}")

        # ---------------- Motion Module ----------------
        if motion_module is not None:
            latents_before_motion = latents.clone()
            latents = motion_module(latents)
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
            if debug:
                diff = (latents - latents_before_motion).abs().max()
                print(f"[DEBUG] Motion module diff max abs: {diff:.4f}")

        # ---------------- Préparer latents pour UNet ----------------
        if is_video:
            B, C, F, H, W = latents.shape
            latents_unet = latents.reshape(B*F, C, H, W)
        else:
            latents_unet = latents

        latents_unet = latents_unet.half() if fp16 else latents_unet.float()

        # ---------------- Préparer embeddings ----------------
        if isinstance(embeddings, tuple):
            pos_embeds, neg_embeds = embeddings
            encoder_states = pos_embeds.to(device)  # ⚠ seulement pos_embeds
        else:
            encoder_states = embeddings

        if debug:
            print(f"[DEBUG] Latents input to UNet shape: {latents_unet.shape}")
            if embeddings: print(f"[DEBUG] Encoder states shape: {encoder_states.shape}")

        # ---------------- UNet SAFE FORWARD ----------------
        try:
            latents_out = safe_forward_unet(unet, latents_unet, t, encoder_hidden_states=encoder_states, device=device)
        except Exception as e:
            print(f"⚠️ [UNet ERROR] step={step_idx} - {e}")
            continue

        # ---------------- Reshape si vidéo ----------------
        if is_video:
            latents = latents_out.reshape(B, F, C, H, W)
        else:
            latents = latents_out

        # ---------------- Clamp & Nan/Inf ----------------
        latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        if debug:
            nan_count = torch.isnan(latents).sum().item()
            inf_count = torch.isinf(latents).sum().item()
            print(f"[DEBUG Step {step_idx}] latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}, NaN={nan_count}, Inf={inf_count}")

    if debug:
        print(f"\n[INFO] Finished latents generation, final shape={latents.shape}, min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    return latents


# ---------------- MINI GPU SAFE 320 ----------------
def generate_latents_mini_gpu_320(unet, **kwargs):
    """
    Wrapper ultra-safe pour GPU <4 Go et UNet miniSD 320-640-1280-1280.
    - Utilise generate_latents_safe_debug_v4
    - Forçage latents aux bonnes dimensions
    - Adaptation embeddings automatique
    - Ajout : init_image_scale & creative_noise
    """
    import torch
    import torch.nn.functional as F

    input_latents = kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' must be fourni")

    device = kwargs.get("device", "cuda")
    fp16 = kwargs.get("fp16", True)
    motion_module = kwargs.get("motion_module", None)
    scheduler = kwargs.get("scheduler", None)
    guidance_scale = kwargs.get("guidance_scale", 1.0)
    steps = kwargs.get("steps", 20)
    debug = kwargs.get("debug", False)

    # Nouveaux params
    init_image_scale = kwargs.get("init_image_scale", 0.0)
    creative_noise = kwargs.get("creative_noise", 0.0)

    # ---------------- FORCER LATENTS ----------------
    expected_channels = getattr(unet.config, "in_channels", 4)
    if input_latents.shape[1] != expected_channels:
        pad = expected_channels - input_latents.shape[1]
        if pad > 0:
            input_latents = F.pad(input_latents, (0,0,0,0,0,0,0,pad))
        else:
            input_latents = input_latents[:, :expected_channels]

    # Adapter H/W si nécessaire
    if input_latents.ndim == 4:  # B,C,H,W
        H, W = input_latents.shape[2:]
    elif input_latents.ndim == 5:  # B,C,T,H,W
        T = input_latents.shape[2]
        if T == 1:
            input_latents = input_latents.squeeze(2)
        H, W = input_latents.shape[2:]
    else:
        raise ValueError(f"[LATENT FIX] Shape inattendue: {input_latents.shape}")

    # Mini GPU → forcer petite taille latents compatible UNet
    target_size = min(getattr(unet.config, "sample_size", 320), 32)
    if (H != target_size) or (W != target_size):
        input_latents = F.interpolate(input_latents, size=(target_size, target_size), mode='nearest')

    kwargs["input_latents"] = input_latents
    kwargs["init_image_scale"] = init_image_scale
    kwargs["creative_noise"] = creative_noise

    # ---------------- Embeddings ----------------
    embeddings = kwargs.get("embeddings", None)
    if isinstance(embeddings, tuple) and len(embeddings) == 2:
        pos_embeds, neg_embeds = embeddings
        target_dim = getattr(unet.config, "cross_attention_dim", pos_embeds.shape[-1])
        pos_embeds, neg_embeds = adapt_embeddings_to_unet(pos_embeds, neg_embeds, target_dim)
        embeddings = (pos_embeds.to(device), neg_embeds.to(device))
    elif embeddings is not None:
        embeddings = embeddings.to(device)
    kwargs["embeddings"] = embeddings

    # ---------------- Génération safe ----------------
    if motion_module is not None and input_latents.ndim == 5:  # B,C,T,H,W
        B,C,T,H,W = input_latents.shape
        for t in range(T):
            frame_latents = input_latents[:, :, t, :, :]
            kwargs["input_latents"] = frame_latents
            frame_latents = generate_latents_safe_debug_v5(unet, **kwargs)
            input_latents[:, :, t, :, :] = frame_latents
        return input_latents
    else:
        return generate_latents_safe_debug_v5(unet, **kwargs)





def generate_latents_mini_gpu_320_v1(unet, **kwargs):
    """
    Wrapper ultra-safe pour GPU <4 Go et UNet miniSD 320-640-1280-1280.
    - Pas de duplication batch pour CF guidance
    - Motion module appliqué frame par frame
    - Forçage latents aux bonnes dimensions (miniGPU safe)
    - Adaptation automatique des embeddings au cross_attention_dim du UNet
    - Logs détaillés pour débogage
    - Gère CUDA OOM
    """
    input_latents = kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' must be fourni")

    device = kwargs.get("device", "cuda")
    fp16 = kwargs.get("fp16", True)
    motion_module = kwargs.get("motion_module", None)
    scheduler = kwargs.get("scheduler", None)
    guidance_scale = kwargs.get("guidance_scale", 1.0)
    steps = kwargs.get("steps", 20)
    debug = kwargs.get("debug", True)

    if debug: print(f"🔹 [DEBUG] input_latents.shape: {input_latents.shape}")

    # ---------------- FORCER LATENTS ----------------
    expected_channels = getattr(unet.config, "in_channels", 4)
    if input_latents.shape[1] != expected_channels:
        pad = expected_channels - input_latents.shape[1]
        if pad > 0:
            if debug: print(f"🔹 [DEBUG] Padding latents: +{pad} channels")
            input_latents = F.pad(input_latents, (0,0,0,0,0,0,0,pad))
        else:
            if debug: print(f"🔹 [DEBUG] Troncature latents: {input_latents.shape[1]} -> {expected_channels}")
            input_latents = input_latents[:, :expected_channels]

    # Adapter H/W si nécessaire
    if input_latents.ndim == 4:  # B,C,H,W
        H, W = input_latents.shape[2:]
    elif input_latents.ndim == 5:  # B,C,T,H,W
        T = input_latents.shape[2]
        if T == 1:
            input_latents = input_latents.squeeze(2)
        H, W = input_latents.shape[2:]
    else:
        raise ValueError(f"[LATENT FIX] Shape inattendue: {input_latents.shape}")

    if debug: print(f"🔹 [DEBUG] Latents H/W: {H}x{W}, channels: {input_latents.shape[1]}")

    # Mini GPU → forcer petite taille latents compatible UNet
    target_size = getattr(unet.config, "sample_size", 32)  # SD2-1.5 → 64
    if (H != target_size) or (W != target_size):
        if debug: print(f"🔹 [DEBUG] Interpolation latents {H}x{W} -> {target_size}x{target_size}")
        input_latents = F.interpolate(input_latents, size=(target_size, target_size), mode='nearest')

    kwargs["input_latents"] = input_latents

    # ---------------- Embeddings ----------------
    embeddings = kwargs.get("embeddings", None)
    if isinstance(embeddings, tuple) and len(embeddings) == 2:
        pos_embeds, neg_embeds = embeddings
        # Lecture du cross_attention_dim attendu par le UNet
        target_dim = getattr(unet.config, "cross_attention_dim", pos_embeds.shape[-1])
        if pos_embeds.shape[-1] != target_dim:
            if debug: print(f"🔄 Projection embeddings {pos_embeds.shape[-1]} -> {target_dim}")
            projection = torch.nn.Linear(pos_embeds.shape[-1], target_dim).to(pos_embeds.device).to(pos_embeds.dtype)
            pos_embeds = projection(pos_embeds)
            neg_embeds = projection(neg_embeds)
        if debug:
            print(f"🔹 [DEBUG] pos_embeds.shape: {pos_embeds.shape}, neg_embeds.shape: {neg_embeds.shape}")
        embeddings = (pos_embeds.to(device), neg_embeds.to(device))
    elif embeddings is not None:
        embeddings = embeddings.to(device)
    kwargs["embeddings"] = embeddings

    # ---------------- Filtrage arguments ----------------
    allowed_keys = [
        "input_latents",
        "scheduler",
        "embeddings",
        "motion_module",
        "guidance_scale",
        "device",
        "fp16",
        "steps",
        "debug",
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    # ---------------- Génération safe ----------------
    try:
        if motion_module is not None:
            latents = filtered_kwargs["input_latents"]
            if latents.ndim == 5:  # B,C,T,H,W
                B, C, T, H, W = latents.shape
                if debug: print(f"🔹 [DEBUG] Motion module présent, latents shape: {latents.shape}")
                for t in range(T):
                    frame_latents = latents[:, :, t, :, :]
                    filtered_kwargs["input_latents"] = frame_latents
                    frame_latents = generate_latents_safe_debug_v4(unet, **filtered_kwargs)
                    latents[:, :, t, :, :] = frame_latents
                return latents
            else:
                if debug: print("🔹 [DEBUG] Motion module absent ou latents 4D")
                return generate_latents_safe_debug_v4(unet, **filtered_kwargs)
        else:
            if debug: print("🔹 [DEBUG] Motion module absent, génération standard")
            return generate_latents_safe_debug_v4(unet, **filtered_kwargs)

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ [MINI GPU] CUDA OOM, retour des latents initiaux")
            return input_latents.clone()
        else:
            raise e





def generate_latents_mini_gpu_4go(unet, **kwargs):
    """
    Wrapper ultra-safe pour GPU <4 Go et UNet miniSD 320-640-1280-1280.
    - Pas de duplication batch pour CF guidance
    - Motion module appliqué frame par frame
    - Forçage latents aux bonnes dimensions (miniGPU safe)
    - Gère CUDA OOM
    """
    input_latents = kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' must be fourni")

    device = kwargs.get("device", "cuda")
    fp16 = kwargs.get("fp16", True)
    motion_module = kwargs.get("motion_module", None)
    scheduler = kwargs.get("scheduler", None)
    guidance_scale = kwargs.get("guidance_scale", 1.0)
    steps = kwargs.get("steps", 20)
    debug = kwargs.get("debug", True)

    # ---------------- FORCER LATENTS ----------------
    expected_channels = getattr(unet.config, "in_channels", 4)
    if input_latents.shape[1] != expected_channels:
        pad = expected_channels - input_latents.shape[1]
        if pad > 0:
            input_latents = F.pad(input_latents, (0,0,0,0,0,0,0,pad))
        else:
            input_latents = input_latents[:, :expected_channels]

    # Adapter H/W si nécessaire
    if input_latents.ndim == 4:  # B,C,H,W
        H, W = input_latents.shape[2:]
    elif input_latents.ndim == 5:  # B,C,T,H,W
        T = input_latents.shape[2]
        if T == 1:
            input_latents = input_latents.squeeze(2)
        H, W = input_latents.shape[2:]
    else:
        raise ValueError(f"[LATENT FIX] Shape inattendue: {input_latents.shape}")

    # Mini GPU → forcer petite taille latents compatible UNet 320-640-1280-1280
    target_size = min(unet.sample_size, 32)
    if (H != target_size) or (W != target_size):
        input_latents = F.interpolate(
            input_latents,
            size=(target_size, target_size),
            mode='nearest'
        )

    kwargs["input_latents"] = input_latents

    # ---------------- Embeddings ----------------
    embeddings = kwargs.get("embeddings", None)
    if isinstance(embeddings, tuple) and len(embeddings) == 2:
        pos_embeds, neg_embeds = embeddings
        embeddings = (pos_embeds.to(device), neg_embeds.to(device))
    elif embeddings is not None:
        embeddings = embeddings.to(device)
    kwargs["embeddings"] = embeddings

    # ---------------- Filtrage arguments ----------------
    allowed_keys = [
        "input_latents",
        "scheduler",
        "embeddings",
        "motion_module",
        "guidance_scale",
        "device",
        "fp16",
        "steps",
        "debug",
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    # ---------------- Génération safe ----------------
    try:
        if motion_module is not None:
            latents = filtered_kwargs["input_latents"]
            if latents.ndim == 5:  # B,C,T,H,W
                B, C, T, H, W = latents.shape
                for t in range(T):
                    frame_latents = latents[:, :, t, :, :]
                    filtered_kwargs["input_latents"] = frame_latents
                    frame_latents = generate_latents_safe_debug_v2(unet, **filtered_kwargs)
                    latents[:, :, t, :, :] = frame_latents
                return latents
            else:
                return generate_latents_safe_debug_v2(unet, **filtered_kwargs)
        else:
            return generate_latents_safe_debug_v2(unet, **filtered_kwargs)

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ [MINI GPU] CUDA OOM, retour des latents initiaux")
            return input_latents.clone()
        else:
            raise e


def generate_latents_mini_gpu(unet, **kwargs):
    """
    Wrapper ultra-safe pour GPU <4 Go.
    - Forçage des latents aux dimensions attendues par le UNet
    - Adaptation embeddings pour cross_attention_dim
    - Gère CUDA OOM
    """
    input_latents = kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' must be fourni")

    device = kwargs.get("device", "cuda")
    fp16 = kwargs.get("fp16", True)
    motion_module = kwargs.get("motion_module", None)  # peut être None
    scheduler = kwargs.get("scheduler", None)
    guidance_scale = kwargs.get("guidance_scale", 1.0)
    steps = kwargs.get("steps", 20)
    debug = kwargs.get("debug", True)

    # ---------------- FORCER LATENTS ----------------
    expected_channels = getattr(unet.config, "in_channels", 4)
    if input_latents.shape[1] != expected_channels:
        pad = expected_channels - input_latents.shape[1]
        if pad > 0:
            input_latents = F.pad(input_latents, (0,0,0,0,0,0,0,pad))
        else:
            input_latents = input_latents[:, :expected_channels]

    # Adapter H/W si nécessaire
    if input_latents.ndim == 4:  # B,C,H,W
        H, W = input_latents.shape[2:]
    elif input_latents.ndim == 5:  # B,C,T,H,W
        T = input_latents.shape[2]
        if T == 1:
            input_latents = input_latents.squeeze(2)
        H, W = input_latents.shape[2:]
    else:
        raise ValueError(f"[LATENT FIX] Shape inattendue: {input_latents.shape}")

    # Mini GPU → forcer petite taille latents
    target_size = min(getattr(unet, "sample_size", 32), 32)
    if (H != target_size) or (W != target_size):
        input_latents = F.interpolate(
            input_latents,
            size=(target_size, target_size),
            mode='nearest'
        )

    kwargs["input_latents"] = input_latents

    # ---------------- Embeddings ----------------
    embeddings = kwargs.get("embeddings", None)
    cross_dim = getattr(unet.config, "cross_attention_dim", 768)
    if isinstance(embeddings, tuple) and len(embeddings) == 2:
        pos_embeds, neg_embeds = embeddings
        # Pad / truncate embeddings pour correspondre au UNet
        if pos_embeds.shape[-1] != cross_dim:
            diff = cross_dim - pos_embeds.shape[-1]
            if diff > 0:
                pos_embeds = F.pad(pos_embeds, (0,diff))
                neg_embeds = F.pad(neg_embeds, (0,diff))
            else:
                pos_embeds = pos_embeds[..., :cross_dim]
                neg_embeds = neg_embeds[..., :cross_dim]
        embeddings = (pos_embeds.to(device), neg_embeds.to(device))
    elif embeddings is not None:
        embeddings = embeddings.to(device)
    kwargs["embeddings"] = embeddings

    # ---------------- Filtrage arguments ----------------
    allowed_keys = [
        "input_latents",
        "scheduler",
        "embeddings",
        "motion_module",
        "guidance_scale",
        "device",
        "fp16",
        "steps",
        "debug",
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    # ---------------- Génération safe ----------------
    try:
        # Motion module désactivé pour miniSD / VRAM ultra-light
        return generate_latents_safe_debug_v2(unet, **filtered_kwargs)

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ [MINI GPU] CUDA OOM, retour des latents initiaux")
            return input_latents.clone()
        else:
            raise e


def generate_latents_safe_miniGPU(unet, **kwargs):
    """
    Wrapper ultra-safe mini GPU pour UNet SD1.x / SD2.
    - Forcer latents ≥ 64x64
    - Motion module frame par frame
    - Projection embeddings si nécessaire
    - OOM safe
    """
    input_latents = kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' must be provided")

    device = kwargs.get("device", "cuda")
    fp16 = kwargs.get("fp16", True)
    debug = kwargs.get("debug", True)
    embeddings = kwargs.get("embeddings", None)
    motion_module = kwargs.get("motion_module", None)
    steps = kwargs.get("steps", 20)
    scheduler = kwargs.get("scheduler", None)
    guidance_scale = kwargs.get("guidance_scale", 7.5)

    # ---------------- Projection embeddings automatique ----------------
    unet_cross_dim = getattr(unet.config, "cross_attention_dim", 768)

    if isinstance(embeddings, tuple) and len(embeddings) == 2:
        pos_embeds, neg_embeds = embeddings
        if pos_embeds.shape[-1] != unet_cross_dim:
            linear_proj = torch.nn.Linear(pos_embeds.shape[-1], unet_cross_dim).to(
                device, dtype=torch.float16 if fp16 else torch.float32
            )
            pos_embeds = linear_proj(pos_embeds.to(device))
            neg_embeds = linear_proj(neg_embeds.to(device))
        embeddings = (pos_embeds, neg_embeds)
    elif embeddings is not None and embeddings.shape[-1] != unet_cross_dim:
        linear_proj = torch.nn.Linear(embeddings.shape[-1], unet_cross_dim).to(
            device, dtype=torch.float16 if fp16 else torch.float32
        )
        embeddings = linear_proj(embeddings.to(device))

    kwargs["embeddings"] = embeddings

    # ---------------- FORCER LATENTS CHANNEL ----------------
    expected_channels = getattr(unet.config, "in_channels", 4)
    if input_latents.shape[1] != expected_channels:
        if input_latents.shape[1] < expected_channels:
            pad = expected_channels - input_latents.shape[1]
            input_latents = F.pad(input_latents, (0,0,0,0,0,0,0,pad))
        else:
            input_latents = input_latents[:, :expected_channels]

    # ---------------- FORCER LATENTS H/W ----------------
    if input_latents.ndim == 4:  # B,C,H,W
        H, W = input_latents.shape[2:]
    elif input_latents.ndim == 5:  # B,C,T,H,W
        T = input_latents.shape[2]
        if T == 1:
            input_latents = input_latents.squeeze(2)
        H, W = input_latents.shape[2:]
    else:
        raise ValueError(f"[LATENT FIX] Shape inattendue: {input_latents.shape}")

    # SD1.x mini GPU → on force ≥ 64
    target_size = max(64, min(unet.sample_size, H, W))
    if (H != target_size) or (W != target_size):
        input_latents = F.interpolate(input_latents, size=(target_size, target_size), mode='nearest')

    kwargs["input_latents"] = input_latents

    # ---------------- Filtrage arguments ----------------
    allowed_keys = [
        "input_latents",
        "scheduler",
        "embeddings",
        "motion_module",
        "guidance_scale",
        "device",
        "fp16",
        "steps",
        "debug",
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    # ---------------- Appel safe debug frame par frame ----------------
    try:
        if motion_module is not None and input_latents.ndim == 5:  # B,C,T,H,W
            B, C, T, H, W = input_latents.shape
            latents_out = []
            for t in range(T):
                frame_latents = input_latents[:, :, t, :, :]
                filtered_kwargs["input_latents"] = frame_latents
                latents_frame = generate_latents_safe_debug_v2(unet, **filtered_kwargs)
                latents_out.append(latents_frame.unsqueeze(2))
            return torch.cat(latents_out, dim=2)
        else:
            return generate_latents_safe_debug_v2(unet, **filtered_kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ [SAFE WRAPPER] CUDA out of memory caught, returning input latents")
            return input_latents.clone()
        else:
            raise e

def generate_latents_safe_wrapper_final(unet, **kwargs):
    """
    Wrapper ultra-safe pour générer les latents avec UNet.
    - Gère CUDA OOM
    - Prépare embeddings pour classifier-free guidance
    - Applique motion module avant duplication
    - Force latents à correspondre à la taille UNet (H/W)
    """
    input_latents = kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' must be fourni")

    device = kwargs.get("device", "cuda")
    fp16 = kwargs.get("fp16", True)
    debug = kwargs.get("debug", True)
    embeddings = kwargs.get("embeddings", None)
    motion_module = kwargs.get("motion_module", None)
    steps = kwargs.get("steps", 20)
    scheduler = kwargs.get("scheduler", None)
    guidance_scale = kwargs.get("guidance_scale", 7.5)

    latents = input_latents.to(device)

    # ---------------- Motion module avant duplication ----------------
    if motion_module is not None:
        latents = motion_module(latents)

    # ---------------- Adapter H/W ----------------
    if latents.ndim == 4:  # B,C,H,W
        H, W = latents.shape[2:]
    elif latents.ndim == 5:  # B,C,T,H,W
        T = latents.shape[2]
        if T == 1:
            latents = latents.squeeze(2)
        H, W = latents.shape[2:]
    else:
        raise ValueError(f"[LATENT FIX] Shape inattendue: {latents.shape}")

    if (H != unet.sample_size) or (W != unet.sample_size):
        latents = torch.nn.functional.interpolate(
            latents,
            size=(unet.sample_size, unet.sample_size),
            mode='nearest'
        )

    # ---------------- Préparer embeddings ----------------
    unet_cross_dim = getattr(unet.config, "cross_attention_dim", 768)
    if isinstance(embeddings, tuple) and len(embeddings) == 2:
        pos_embeds, neg_embeds = embeddings
        # Projection automatique si nécessaire
        if pos_embeds.shape[-1] != unet_cross_dim:
            proj = torch.nn.Linear(pos_embeds.shape[-1], unet_cross_dim).to(device, dtype=torch.float16 if fp16 else torch.float32)
            pos_embeds = proj(pos_embeds.to(device))
            neg_embeds = proj(neg_embeds.to(device))
        embeddings = (pos_embeds, neg_embeds)
    elif embeddings is not None and embeddings.shape[-1] != unet_cross_dim:
        proj = torch.nn.Linear(embeddings.shape[-1], unet_cross_dim).to(device, dtype=torch.float16 if fp16 else torch.float32)
        embeddings = proj(embeddings.to(device))

    # ---------------- Duplication batch pour CFG ----------------
    if guidance_scale > 1.0:
        latents = torch.cat([latents, latents], dim=0)
        if isinstance(embeddings, tuple):
            pos_embeds, neg_embeds = embeddings
            embeddings = torch.cat([neg_embeds.to(device), pos_embeds.to(device)], dim=0)
        else:
            embeddings = embeddings.to(device)

    kwargs["input_latents"] = latents
    kwargs["embeddings"] = embeddings

    # ---------------- Filtrage arguments ----------------
    allowed_keys = [
        "input_latents",
        "scheduler",
        "embeddings",
        "motion_module",
        "guidance_scale",
        "device",
        "fp16",
        "steps",
        "debug",
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    # ---------------- Appel safe debug ----------------
    try:
        return generate_latents_safe_debug_v2(unet, **filtered_kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ [SAFE WRAPPER] CUDA out of memory caught, returning input latents")
            return latents.clone()
        else:
            raise e

# ici  erreur cross_attention_dim mais la génération reste correct:
def generate_latents_safe_wrapper_v2(unet, **kwargs):
    """
    Wrapper ultra-safe pour générer les latents avec UNet.
    - Gère CUDA OOM
    - Prépare embeddings tuple pour classifier-free guidance
    """

    input_latents = kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' must be provided")

    embeddings = kwargs.get("embeddings", None)

    # Si embeddings est un tuple de (pos, neg), concat pour CF guidance
    if isinstance(embeddings, tuple) and len(embeddings) == 2:
        pos_embeds, neg_embeds = embeddings
        kwargs["embeddings"] = torch.cat([neg_embeds.to(kwargs.get("device", "cuda")),
                                          pos_embeds.to(kwargs.get("device", "cuda"))], dim=0)

    # Arguments autorisés pour la version ultra-light
    allowed_keys = [
        "input_latents",
        "scheduler",
        "embeddings",
        "motion_module",
        "guidance_scale",
        "device",
        "fp16",
        "steps",
        "debug",
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    try:
        return generate_latents_safe_debug_v2(unet, **filtered_kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ [SAFE WRAPPER] CUDA out of memory caught, returning input latents")
            return input_latents.clone()
        else:
            raise e



def generate_latents_ultralight(
    unet,
    input_latents,
    scheduler,
    embeddings=None,
    motion_module=None,
    guidance_scale=1.0,
    device="cuda",
    fp16=True,
    steps=20,
    patch_size=32,
    debug=False
):
    """
    Génération de latents ultra-light, patch par patch pour GPU <4Go.
    - input_latents: torch.Tensor [B,C,H,W] ou [B,C,F,H,W] pour vidéo
    - motion_module: optionnel, appliqué à chaque step
    - embeddings: CLIP embeddings
    """
    latents = input_latents.clone().to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    is_video = latents.ndim == 5
    steps_list = getattr(scheduler, "timesteps", range(steps))

    for step_idx, t in enumerate(steps_list):
        if motion_module is not None:
            latents = motion_module(latents)
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

        # UNet patch par patch pour économiser la VRAM
        if is_video:
            B, C, F, H, W = latents.shape
            latents_patch = latents.permute(0,2,1,3,4).reshape(B*F, C, H, W)
        else:
            latents_patch = latents

        H, W = latents_patch.shape[-2:]
        output = torch.zeros_like(latents_patch, device=latents_patch.device)

        # Patch UNet
        stride = patch_size
        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                y1 = min(y0 + patch_size, H)
                x1 = min(x0 + patch_size, W)

                patch = latents_patch[:, :, y0:y1, x0:x1]
                patch = patch.half() if fp16 else patch.float()

                try:
                    with torch.no_grad():
                        patch_out = unet(patch, t, encoder_hidden_states=embeddings)["sample"]
                except Exception as e:
                    print(f"⚠ [UNet ERROR patch {y0},{x0}] {e}")
                    patch_out = patch.clone()

                # Clamp et cleanup
                patch_out = torch.nan_to_num(patch_out, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5,5)
                output[:, :, y0:y1, x0:x1] = patch_out
                del patch, patch_out
                torch.cuda.empty_cache()

        if is_video:
            latents = output.reshape(B, F, C, H, W).permute(0,2,1,3,4)
        else:
            latents = output

        if debug:
            print(f"[Step {step_idx}] latents min/max={latents.min():.4f}/{latents.max():.4f}")

    return latents





# --------------------------------------------------------------------------------------------------------


def decode_latents_ultrasafe_cpu(latents, vae, block_size=160, overlap=96, debug=True, threaded=True):
    """
    Decode latents safely on CPU in patches, optionally threaded.
    Supports 4D [B, C, H, W] and 5D [B, C, F, H, W] latents.
    Logs min/max for debugging.
    """
    import threading

    vae.to("cpu")
    latents = latents.to("cpu")
    is_5d = latents.dim() == 5
    if is_5d:
        B, C, F, H, W = latents.shape
    else:
        B, C, H, W = latents.shape
        F = 1
        latents = latents.unsqueeze(2)

    frames = [None] * F

    def decode_frame(f):
        frame_latents = latents[:, :, f]
        if debug:
            print(f"[DEBUG] Decoding frame {f} latents shape: {frame_latents.shape} min/max: {frame_latents.min().item():.4f}/{frame_latents.max().item():.4f}")

        H_idx = list(range(0, H, block_size - overlap))
        W_idx = list(range(0, W, block_size - overlap))
        decoded_frame = torch.zeros((B, 3, H, W), dtype=torch.float32)

        for i, h_start in enumerate(H_idx):
            h_end = min(h_start + block_size, H)
            for j, w_start in enumerate(W_idx):
                w_end = min(w_start + block_size, W)
                patch = frame_latents[:, :, h_start:h_end, w_start:w_end]
                with torch.no_grad():
                    decoded_patch = vae.decode(patch).sample
                decoded_frame[:, :, h_start:h_end, w_start:w_end] = decoded_patch

        frames[f] = decoded_frame

    if threaded:
        threads = []
        for f in range(F):
            t = threading.Thread(target=decode_frame, args=(f,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    else:
        for f in range(F):
            decode_frame(f)

    output = torch.stack(frames, dim=2) if is_5d else frames[0]
    if debug:
        print(f"[DEBUG] Final output shape: {output.shape} min/max: {output.min().item():.4f}/{output.max().item():.4f}")
    return output


# ---------------- Encodage images → latents safe ----------------
def encode_images_to_latents_safe(images, vae, device="cuda", dtype=torch.float16):
    # -- assure cohérence dtype/device --
    vae = vae.to(device=device, dtype=dtype)
    images_t = images.to(device=device, dtype=dtype)

    # sauvegarde dtype original pour restore
    original_dtype = next(vae.parameters()).dtype

    with torch.no_grad():
        latents = vae.encode(images_t).latent_dist.sample()

    latents = latents * LATENT_SCALE
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # normalisation pour UNet / motion module
    max_abs = latents.abs().max()
    if max_abs > 0:
        latents = latents / max_abs  # scale [-1,1]

    # reshape pour pipeline [B,C,T,H,W]
    if latents.ndim == 4:
        latents = latents.unsqueeze(2)

    print(
        "[SAFE ENCODE]",
        "min=", latents.min().item(),
        "max=", latents.max().item(),
        "std=", latents.std().item(),
        "dtype=", latents.dtype,
        "device=", latents.device,
        "shape=", latents.shape
    )

    return latents

def decode_latents_ultrasafe_blockwise_adapted(latents, vae, block_size=160, overlap=96, gamma=1.0, brightness=1.0, contrast=1.0, saturation=1.0):
    """
    Décodage ultrasafe des latents en PIL.Image
    Adapté pour accepter 4D [B,C,H,W] ou 5D [B,C,T,H,W].
    Si 5D, retourne une liste de frames PIL.
    """
    # Convert 5D → list de frames 4D
    if latents.ndim == 5:
        B,C,T,H,W = latents.shape
        frames_4d = [latents[:, :, t, :, :] for t in range(T)]
    elif latents.ndim == 4:
        frames_4d = [latents]
    else:
        raise ValueError(f"Latents doivent être 4D ou 5D, got {latents.shape}")

    pil_frames = []
    for fidx, latent_4d in enumerate(frames_4d):
        # On décode par blocs (supposons que decode_latents_ultrasafe_blockwise existe)
        frame_pil = decode_latents_ultrasafe_blockwise(latent_4d, vae, block_size=block_size, overlap=overlap,
                                                        gamma=gamma, brightness=brightness, contrast=contrast,
                                                        saturation=saturation)
        pil_frames.append(frame_pil)

    if len(pil_frames) == 1:
        return pil_frames[0]
    return pil_frames


# ---------------- Décodage latents → image ultrasafe (blockwise) ----------------
def decode_latents_ultrasafe_blockwise(latents, vae, block_size=160, overlap=96,
                                       gamma=1.0, brightness=1.0, contrast=1.0, saturation=1.0):
    """
    Décodage sécurisé des latents en image PIL, bloc par bloc pour réduire VRAM.
    Input:
        latents : [B,C,H,W] ou [B,C,1,H,W]
        vae : AutoencoderKL
    """
    if latents.ndim == 5:
        latents = latents.squeeze(2)  # [B,C,H,W]

    B, C, H, W = latents.shape
    device = next(vae.parameters()).device

    stride = block_size - overlap
    h_steps = max(1, (H - overlap + stride - 1) // stride)
    w_steps = max(1, (W - overlap + stride - 1) // stride)

    output = torch.zeros(B, C, H, W, device="cpu")
    weight_map = torch.zeros(B, 1, H, W, device="cpu")

    for i in range(h_steps):
        for j in range(w_steps):
            y0 = i * stride
            x0 = j * stride
            y1 = min(y0 + block_size, H)
            x1 = min(x0 + block_size, W)

            patch = latents[:, :, y0:y1, x0:x1].to(device)
            patch = torch.nan_to_num(patch, nan=0.0, posinf=5.0, neginf=-5.0)

            with torch.no_grad():
                patch_decoded = vae.decode(patch.to(vae.dtype)).sample
                patch_decoded = ((patch_decoded + 1)/2).clamp(0,1).cpu()

            # Weighted blend pour recouvrement
            h_patch, w_patch = patch_decoded.shape[2], patch_decoded.shape[3]
            mask = torch.ones(1, 1, h_patch, w_patch)
            output[:, :, y0:y0+h_patch, x0:x0+w_patch] += patch_decoded * mask
            weight_map[:, :, y0:y0+h_patch, x0:x0+w_patch] += mask

            torch.cuda.empty_cache()

    output = output / weight_map.clamp(min=1e-5)
    frame_pil = ToPILImage()(output[0].clamp(0,1))

    # Ajustements image
    if gamma != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(gamma)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    return frame_pil

# ---------------- Decode safe 4Go VRAM ----------------
def decode_latents_safe_vram(latents, vae, gamma=0.7, brightness=1.2, contrast=1.1, saturation=1.15):
    from torchvision.transforms import ToPILImage
    import torch
    from PIL import ImageEnhance

    latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

    # Si batch T multiple, on prend le 0 pour chaque frame
    if latents.ndim == 5:
        latents = latents[:, :, 0, :, :]

    latents = latents / LATENT_SCALE
    latents = latents.to(dtype=torch.float32, device="cpu")  # decode safe 4Go

    with torch.no_grad():
        image_tensor = vae.decode(latents).sample

    image_tensor = ((image_tensor + 1) / 2).clamp(0, 1)
    image_tensor = image_tensor.pow(1.0 / gamma)

    images = []
    to_pil = ToPILImage()
    for i in range(image_tensor.shape[0]):
        img = image_tensor[i]
        pil_img = to_pil(img.cpu())
        pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
        pil_img = ImageEnhance.Color(pil_img).enhance(saturation)
        images.append(pil_img)
    return images


def decode_latents_to_image_auto(latents, vae):
    """
    Décodage automatique du latent, quel que soit son format (3D, 4D, 5D),
    retourne un tenseur [B, C, H, W].
    """
    # S'assurer que latents est 4D : [B, C, H, W]
    if latents.ndim == 5:  # [B, C, T, H, W]
        latents = latents.squeeze(2)
    elif latents.ndim == 3:  # [C, H, W] -> [1, C, H, W]
        latents = latents.unsqueeze(0)
    elif latents.ndim != 4:
        raise ValueError(f"Latents avec dimension inattendue {latents.shape}")

    # Décodage avec VAE
    with torch.no_grad():
        images = vae.decode(latents / LATENT_SCALE).sample

    # Normalisation 0-1
    images = (images + 1.0) / 2.0
    images = images.clamp(0, 1)
    return images


# -------------------------
# scripts/utils/n3r_utils.py
# -------------------------

def patchify_latents(latents, tile_size=128, overlap=32):
    """
    Découpe les latents en patches pour traitement patch-based.
    Retourne une liste plate de patches + leurs positions.
    """
    _, C, H, W = latents.shape
    patches = []
    coords = []

    stride = tile_size - overlap
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y + tile_size, H)
            x1 = min(x + tile_size, W)
            y0 = y1 - tile_size if y1 - y < tile_size else y
            x0 = x1 - tile_size if x1 - x < tile_size else x

            patch = latents[:, :, y0:y1, x0:x1]
            patches.append(patch)           # ✅ Patch est un Tensor
            coords.append((y0, y1, x0, x1))  # ✅ Coordonnees patch

    return patches, coords



# -------------------------
# Dépatchification sécurisée
# -------------------------
def unpatchify_latents(patches, coords, full_shape, device=None, dtype=None):
    """
    Recompose les latents à partir de la liste de patches.
    """
    B, C, H, W = full_shape
    latents = torch.zeros(full_shape, device=device, dtype=dtype)

    for patch, (y0, y1, x0, x1) in zip(patches, coords):
        latents[:, :, y0:y1, x0:x1] = patch

    return latents
# -------------------------
# Génération d’un frame patch-based v3
# -------------------------
# -------------------------
# generate_frame_patched_v3
# -------------------------
def generate_frame_patched_v3(
    input_image,
    vae,
    unet,
    scheduler,
    pos_emb,
    neg_emb=None,
    tile_size=128,
    overlap=32,
    steps=12,
    guidance_scale=4.5,
    init_image_scale=0.75,
    creative_noise=0.0,
    device="cuda",
    dtype=torch.float16
):
    """
    Génère un frame patch-based safe pour CPU/GPU.
    """
    # -------------------------
    # Encode l'image en latents (toujours sur CPU pour VAE)
    # -------------------------
    with torch.no_grad():
        latents = vae.encode(input_image.to(torch.float32)).latent_dist.sample() * LATENT_SCALE
        latents = latents.to(device=device, dtype=dtype)

    # -------------------------
    # Patchify latents
    # -------------------------
    patches, patch_coords = patchify_latents(latents, tile_size=tile_size, overlap=overlap)

    # -------------------------
    # Scheduler timesteps
    # -------------------------
    for t in scheduler.timesteps[:steps]:
        t_tensor = torch.tensor([t], device=device, dtype=dtype)
        new_patches = []

        for patch in patches:
            patch = patch.to(device=device, dtype=dtype)

            # Flatten 5D -> 4D si nécessaire (batch x channels x H x W)
            if patch.dim() == 5:
                B,C,D,H,W = patch.shape
                patch = patch.view(B*C, D, H, W)

            # UNet forward
            noise_pred = unet(
                patch,
                timestep=t_tensor,
                encoder_hidden_states=pos_emb
            ).sample

            # Guidance si neg_emb fourni
            if neg_emb is not None:
                noise_pred_neg = unet(
                    patch,
                    timestep=t_tensor,
                    encoder_hidden_states=neg_emb
                ).sample
                noise_pred = noise_pred_neg + guidance_scale * (noise_pred - noise_pred_neg)

            # Creative noise
            if creative_noise > 0.0:
                noise_pred += creative_noise * torch.randn_like(noise_pred)

            new_patches.append(noise_pred)

            # Libération GPU intermédiaire
            del patch, noise_pred
            torch.cuda.empty_cache()

        patches = new_patches

    # -------------------------
    # Recomposition latents
    # -------------------------
    latents = unpatchify_latents(patches, patch_coords, latents.shape)

    # -------------------------
    # Décodage VAE
    # -------------------------
    with torch.no_grad():
        frame_tensor = vae.decode(latents.to(torch.float32)).sample
        frame_tensor = frame_tensor.to(device=device, dtype=dtype)

    return frame_tensor
# -------------------------
# Génération d’un frame patch-based v1
# -------------------------

def generate_frame_patched(
    input_image, vae, unet, scheduler,
    pos_emb, neg_emb=None,
    tile_size=128, overlap=32,
    steps=12,
    guidance_scale=4.5,
    init_image_scale=0.75,
    creative_noise=0.0,
    device="cuda",
    dtype=torch.float16
):
    """
    Génère un frame en utilisant la logique patch-based safe pour CPU/GPU.
    """
    # -------------------------
    # Encode l'image en latents (toujours sur CPU pour VAE)
    # -------------------------
    with torch.no_grad():
        latents = vae.encode(input_image.to(torch.float32)).latent_dist.sample() * LATENT_SCALE
        latents = latents.to(device=device, dtype=dtype)

    # -------------------------
    # Patchify
    # -------------------------
    patches, patch_coords = patchify_latents(latents, tile_size=tile_size, overlap=overlap)

    # -------------------------
    # Parcours des timesteps
    # -------------------------
    for t in scheduler.timesteps[:steps]:
        t_tensor = torch.tensor([t], device=device, dtype=dtype)

        new_patches = []
        for patch in patches:
            patch = patch.to(device=device, dtype=dtype)

            # UNet call : timestep obligatoire + embeddings
            noise_pred = unet(
                patch,
                timestep=t_tensor,
                encoder_hidden_states=pos_emb
            ).sample

            # Guidance si négatif fourni
            if neg_emb is not None:
                noise_pred_neg = unet(
                    patch,
                    timestep=t_tensor,
                    encoder_hidden_states=neg_emb
                ).sample
                noise_pred = noise_pred_neg + guidance_scale * (noise_pred - noise_pred_neg)

            new_patches.append(noise_pred)

        patches = new_patches

        # Optionnel : ajouter un peu de noise créatif
        if creative_noise > 0.0:
            for i in range(len(patches)):
                patches[i] += creative_noise * torch.randn_like(patches[i])

    # -------------------------
    # Unpatchify
    # -------------------------
    latents = unpatchify_latents(patches, patch_coords, latents.shape)

    # -------------------------
    # Décode en image
    # -------------------------
    with torch.no_grad():
        frame_tensor = vae.decode(latents.to(torch.float32)).sample
        frame_tensor = frame_tensor.to(dtype=dtype, device=device)

    return frame_tensor


# -------------------------
# Génération split_image_into_patches
# -------------------------

def split_image_into_patches(img, patch_size=128, overlap=16):
    """Découpe l'image en patches avec overlap."""
    _, C, H, W = img.shape
    stride = patch_size - overlap
    patches = []
    positions = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0, x0 = y, x
            y1, x1 = min(y+patch_size, H), min(x+patch_size, W)
            patch = img[:, :, y0:y1, x0:x1]
            patches.append(F.pad(patch, (0, patch_size-(x1-x0), 0, patch_size-(y1-y0))))
            positions.append((y0, y1, x0, x1))
    return torch.stack(patches), positions

def reassemble_patches(patches, positions, H, W, overlap=16):
    """Reconstitue l'image à partir des patches avec blending."""
    device = patches.device
    out = torch.zeros(1, 3, H, W, device=device)
    count = torch.zeros(1, 1, H, W, device=device)

    for patch, (y0, y1, x0, x1) in zip(patches, positions):
        h, w = y1-y0, x1-x0
        out[:, :, y0:y1, x0:x1] += patch[:, :, :h, :w]
        count[:, :, y0:y1, x0:x1] += 1
    return out / count




def generate_frame_with_tiling(
    input_image, vae, unet, scheduler, embeddings, motion_module=None,
    tile_size=128, overlap=16, fp16=True,
    guidance_scale=4.5, init_image_scale=0.75,
    creative_noise=0.0, steps=12
):
    """
    Génère une image à partir d'une input_image patchée, recomposée automatiquement.
    Optimisé pour MiniSD/TinySD afin de réduire la VRAM et éviter OOM.
    """

    device = input_image.device
    dtype = torch.float16 if fp16 else torch.float32
    _, C, H, W = input_image.shape

    # Calculer nombre de patches
    stride = tile_size - overlap
    y_positions = list(range(0, H, stride))
    x_positions = list(range(0, W, stride))

    # Préparer canvas final
    frame_latents = torch.zeros((1, C, H, W), device=device, dtype=dtype)
    weight_map = torch.zeros((1, 1, H, W), device=device, dtype=dtype)

    for y in y_positions:
        for x in x_positions:
            # Extraire patch
            y0, y1 = y, min(y + tile_size, H)
            x0, x1 = x, min(x + tile_size, W)
            patch = input_image[:, :, y0:y1, x0:x1].to(dtype)

            # --- Encoder en latents (VAE FP32-safe) ---
            with torch.no_grad():
                patch_latents = vae.encode(patch.float()).latent_dist.sample() * 0.18215
                patch_latents = patch_latents.to(dtype)

            # --- Motion module ---
            if motion_module is not None:
                patch_latents = motion_module(patch_latents)

            # --- Scheduler / UNet ---
            # Utiliser FP16 pour UNet si demandé
            patch_latents = patch_latents.half() if fp16 else patch_latents.float()
            for pos_embeds, neg_embeds in embeddings:
                for t in scheduler.timesteps:
                    with torch.no_grad():
                        noise_pred = unet(
                            patch_latents, timestep=t, encoder_hidden_states=pos_embeds
                        ).sample
                    # Guidance
                    patch_latents = patch_latents + guidance_scale * (noise_pred - patch_latents)

            # --- Décoder patch ---
            with torch.no_grad():
                patch_img = vae.decode(patch_latents.float()).sample  # toujours FP32 pour VAE
                patch_img = patch_img.to(dtype)

            # --- Ajouter au canvas final ---
            frame_latents[:, :, y0:y1, x0:x1] += patch_img
            weight_map[:, :, y0:y1, x0:x1] += 1.0

            # Libérer mémoire GPU
            torch.cuda.empty_cache()

    # Normaliser par superposition
    frame_latents /= weight_map
    return frame_latents


# -------------------------
# encode_image_latents reste FP32 pour stabilité
# -------------------------
def encode_image_latents_fp32(image_tensor, vae, scale=LATENT_SCALE):
    device = next(vae.parameters()).device
    img = image_tensor.to(device=device, dtype=next(vae.parameters()).dtype)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample() * scale
    return latents.unsqueeze(2)  # [B,C,1,H,W]


# -------------------------
# Génération patch par patch
# -------------------------
from scripts.modules.motion_module_tiny import MotionModuleTiny

def generate_frame_with_tiling_v1(
    input_image,
    vae,
    unet,
    motion_module,
    patch_size: int = 128,
    overlap: int = 16,
    fp16: bool = True,
    **kwargs
):
    """
    Découpe l'image en patches, encode, génère latents, puis recompose.
    FP16-safe pour Mini/Tiny SD.
    """
    # Convertir en FP16 si demandé
    dtype = torch.float16 if fp16 else torch.float32

    _, h, w = input_image.shape
    stride = patch_size - overlap
    frame_tensor = torch.zeros_like(input_image, dtype=dtype)

    # Eviter double passage de fp16
    kwargs = dict(kwargs)
    kwargs.pop("fp16", None)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2, x2 = min(y + patch_size, h), min(x + patch_size, w)
            patch = input_image[:, y1:y2, x1:x2]

            # Encoder le patch
            patch_latents = encode_image_latents_fp32(patch, vae, fp16=fp16)

            # Génération latents
            patch_latents = generate_latents_ai_5D_optimized(
                patch_latents,
                unet,
                motion_module,
                fp16=fp16,
                **kwargs
            )

            # Décodage patch
            patch_frame = vae.decode(patch_latents).sample.to(dtype)

            # Recomposer dans l'image finale
            frame_tensor[:, y1:y2, x1:x2] = patch_frame

    return frame_tensor


# -------------------------
# Génération et décodage sécurisée pour n3rHYBRID24
# -------------------------
def generate_and_decode(latent_frame, unet, scheduler, pos_embeds, neg_embeds,
                        motion_module, vae, device="cuda", dtype=torch.float32,
                        guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0,
                        seed=42, steps=35, tile_size=128, overlap=32, vae_offload=False):
    """
    Génère les latents pour un frame et les décode en image finale,
    avec gestion automatique des devices, FP16, offload et tiling.
    """
    import torch, time

    torch.manual_seed(seed)

    # -------------------------
    # Déplacer latents et embeddings sur le bon device et dtype
    # -------------------------
    latent_frame = latent_frame.to(device=device, dtype=dtype)
    pos_embeds = pos_embeds.to(device=device, dtype=dtype)
    neg_embeds = neg_embeds.to(device=device, dtype=dtype)

    # -------------------------
    # Génération avec UNet + Scheduler
    # -------------------------
    gen_start = time.time()
    batch_latents = generate_latents_ai_5D_optimized(
        latent_frame=latent_frame,
        scheduler=scheduler,
        pos_embeds=pos_embeds,
        neg_embeds=neg_embeds,
        unet=unet,
        motion_module=motion_module,
        device=device,
        dtype=dtype,
        guidance_scale=guidance_scale,
        init_image_scale=init_image_scale,
        creative_noise=creative_noise,
        seed=seed,
        steps=steps
    )
    gen_time = time.time() - gen_start


# -------------------------
# Encode tile safe FP32
# -------------------------
def encode_tile_safe_fp32(vae, tile_np, device="cuda", vae_offload=False):
    """
    Encode une tile numpy [C,H,W] en latent VAE [1,4,H/8,W/8]
    VRAM-safe, compatible FP32 VAE complet et offload
    """
    tile_tensor = torch.from_numpy(tile_np).unsqueeze(0).to(device=device, dtype=torch.float32)  # [1,3,H,W]
    with torch.no_grad():
        if vae_offload:
            vae.to(device)  # mettre VAE sur le même device que le tile
        latent = vae.encode(tile_tensor).latent_dist.sample() * LATENT_SCALE
        if vae_offload:
            vae.cpu()  # remettre VAE sur CPU pour économiser VRAM
            if device.startswith("cuda"):
                torch.cuda.synchronize()
    return latent

# -------------------------
# Merge tiles FP32
# -------------------------
def merge_tiles_fp32(tile_list, positions, H, W, latent_scale=1.0):
    """
    Fusionne les tiles latents [1,C,th,tw] en image complète [1,C,H,W].
    Supporte tiles de tailles différentes et bordures.
    """
    device = tile_list[0].device
    C = tile_list[0].shape[1]

    out = torch.zeros(1, C, H, W, dtype=tile_list[0].dtype, device=device)
    count = torch.zeros(1, C, H, W, dtype=tile_list[0].dtype, device=device)

    for tile, (y1, y2, x1, x2) in zip(tile_list, positions):
        _, c, th, tw = tile.shape
        h_len = y2 - y1
        w_len = x2 - x1
        th = min(th, h_len)
        tw = min(tw, w_len)
        out[:, :, y1:y1+th, x1:x1+tw] += tile[:, :, :th, :tw]
        count[:, :, y1:y1+th, x1:x1+tw] += 1.0

    count[count==0] = 1.0
    out = out / count
    return out

def encode_tile_safe_latent(vae, tile, device, LATENT_SCALE=0.18215):
    """
    Encode une tuile en latent FP32 et pad si nécessaire.
    tile: np.array (H,W,3) float32 0-1
    return: torch tensor (1,4,H_latent_max,W_latent_max)
    """
    tile_tensor = torch.tensor(tile).permute(2,0,1).unsqueeze(0).to(device)
    latent = vae.encode(tile_tensor).latent_dist.sample() * LATENT_SCALE
    # Vérifier H,W du latent
    H_lat, W_lat = latent.shape[2], latent.shape[3]
    H_max = (tile.shape[0] + 7)//8  # VAE scale
    W_max = (tile.shape[1] + 7)//8
    if H_lat != H_max or W_lat != W_max:
        padH = H_max - H_lat
        padW = W_max - W_lat
        latent = torch.nn.functional.pad(latent, (0,padW,0,padH))
    return latent

# --- Découper une image en tiles avec overlap ---
def tile_image_128(image, tile_size=128, overlap=16):
    """
    Découpe une image (H,W,C ou C,H,W) en tiles avec overlap.
    Retourne une liste de tiles (numpy arrays) et leurs positions (x1,y1,x2,y2).
    """
    # Assure shape [C,H,W]
    if image.ndim == 3 and image.shape[2] in [1,3]:
        # H,W,C -> C,H,W
        image = image.transpose(2,0,1)
    elif image.ndim != 3:
        raise ValueError(f"Image doit être 3D, shape={image.shape}")

    C,H,W = image.shape
    stride = tile_size - overlap
    tiles = []
    positions = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)
            tile = image[:, y1:y2, x1:x2]
            tiles.append(tile.astype(np.float32))   # reste numpy
            positions.append((x1, y1, x2, y2))
    return tiles, positions


# --- Normalisation d'une tile ---
def normalize_tile_128(img_array):
    """
    img_array: np.ndarray, shape [H,W,C] ou [C,H,W], valeurs 0-255
    Retour: torch.Tensor [1,3,H,W] float32, valeurs 0-1
    """
    if img_array.ndim == 3 and img_array.shape[2] == 3:  # HWC
        img_array = img_array.transpose(2,0,1)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
    return img_tensor




# --- Merge tiles pour reconstruire l'image ---
# -------------------------
# Merge tiles
# -------------------------
def merge_tiles(tile_list, positions, H, W):
    out = torch.zeros(1, 3, H, W, dtype=torch.float32)
    count = torch.zeros(1, 3, H, W, dtype=torch.float32)

    for tile, (y1, y2, x1, x2) in zip(tile_list, positions):
        _, c, th, tw = tile.shape
        out[:,:,y1:y2,x1:x2] += tile[:,:, :th, :tw]
        count[:,:,y1:y2,x1:x2] += 1.0

    count[count==0] = 1.0
    out = out / count
    return out


def save_frame(img_array, filename):
    img_array = np.clip(img_array, 0.0, 1.0)
    img_uint8 = (img_array * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Image.fromarray(img_uint8).save(filename)

def encode_image_latents(image_tensor, vae, scale=LATENT_SCALE):
    """Encode RGB -> latents 4 canaux"""
    device = next(vae.parameters()).device
    img = image_tensor.to(device=device, dtype=next(vae.parameters()).dtype)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample() * scale
    return latents  # [B,4,H/8,W/8]


def generate_latents_ai_5D_stable(
    latent_frame,         # [B,4,H,W] ou [1,4,H,W]
    pos_embeds,           # [1,77,768]
    neg_embeds,           # [1,77,768]
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=7.5,
    init_image_scale=0.7,
    creative_noise=0.03,
    seed=42,
    steps=40
):
    """
    Génération de latents ultra-stable avec :
      - réinjection du latent initial pour stabilité
      - creative_noise modéré
      - support batch [B,4,H,W] ou frame unique [1,4,H,W]
    Sortie : latents [B,4,H,W]
    """

    torch.manual_seed(seed)
    B = latent_frame.shape[0]

    latents = latent_frame.to(device=device, dtype=dtype)
    init_latents = latents.clone()

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:
        # 🔹 Motion module
        if motion_module is not None:
            latents = motion_module(latents)

        # 🔹 Creative noise
        if creative_noise > 0:
            latents = latents + torch.randn_like(latents) * creative_noise

        # 🔹 Classifier-Free Guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # 🔹 Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 🔹 Réinjection du latent initial (stabilité couleur / contenu)
        latents = latents + init_image_scale * (init_latents - latents)

        # 🔹 Sécurité NaN / inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = latents + torch.randn_like(latents) * 1e-3

        # 🔹 Debug
        mean_val = latents.abs().mean().item()
        if mean_val < 1e-5:
            print(f"⚠ Latent trop petit à timestep {t}, mean={mean_val:.6f}")

    return latents



def generate_latents_ai_5D_optimized(
    latent_frame,         # [1,4,H,W]
    pos_embeds,           # [1,77,768]
    neg_embeds,           # [1,77,768]
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=7.5,          # aligné robuste
    init_image_scale=2.0,        # aligné robuste
    creative_noise=0.0,
    seed=42,
    steps=40
):
    """
    Version équivalente à generate_latents_robuste mais pour une seule frame.
    Sortie: [1,4,H,W]
    """

    torch.manual_seed(seed)

    # ---- Setup ----
    latents = latent_frame.to(device=device, dtype=dtype)
    init_latents = latents.clone()

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:

        # 🔹 Motion module
        if motion_module is not None:
            latents = motion_module(latents)

        # 🔹 Creative noise (même endroit que robuste)
        if creative_noise > 0:
            latents = latents + torch.randn_like(latents) * creative_noise

        # 🔹 Classifier-Free Guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # 🔹 Scheduler step (IMPORTANT: batch normal, pas latents[:1])
        latents = scheduler.step(
            noise_pred,
            t,
            latents
        ).prev_sample

        # 🔹 Réinjection identique à robuste
        latents = latents + init_image_scale * (init_latents - latents)

        # 🔹 Sécurité NaN / inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"⚠ NaN/inf détecté à timestep {t}, correction légère")
            latents = latents.clone()
            latents = latents + torch.randn_like(latents) * 1e-3

        # 🔹 Debug stabilité
        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-5:
            print(f"⚠ Latent trop petit à timestep {t}, mean={mean_val:.6f}")

    return latents



# -------------------------
# 🔹 Génération des latents
# -------------------------

def generate_latents_4Go(
    latent_frame,          # [1,4,H,W]
    pos_embeds,            # [1,77,768]
    neg_embeds,            # [1,77,768]
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.0,
    init_image_scale=0.9,
    steps=20,
    seed=1234
):
    """
    Génération de latents optimisée pour AnimateDiff avec Classifier-Free Guidance.
    Corrige le mismatch batch entre UNet et embeddings.
    """
    torch.manual_seed(seed)

    # 🔹 Mettre le latent sur le device
    latents = latent_frame.to(device=device, dtype=dtype)

    batch_size = latents.shape[0]  # normalement 1

    # 🔹 Répliquer embeddings selon batch size
    neg_embeds = neg_embeds.repeat(batch_size, 1, 1)
    pos_embeds = pos_embeds.repeat(batch_size, 1, 1)

    # 🔹 Embeddings C.F.G
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)
    # batch = 2 * batch_size

    # 🔹 Scheduler
    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:

        # 🔹 Input latent doublé pour C.F.G
        latent_input = torch.cat([latents, latents], dim=0)

        # 🔹 Motion module si utilisé
        if motion_module is not None:
            latent_input = motion_module(latent_input)

        # 🔹 Blend avec latent original
        latent_input = latent_input * init_image_scale + torch.cat([latents, latents], dim=0) * (1 - init_image_scale)

        # 🔹 UNet forward
        noise_pred = unet(latent_input, t, encoder_hidden_states=embeds).sample

        # 🔹 Classifier-Free Guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # 🔹 Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # 🔹 Clamp sécurité fp16
        latents = latents.clamp(-1.5, 1.5)

    return latents

# -------------------------
# 🔹 Génération des texts
# -------------------------

def encode_text_embeddings(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: str = "",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    max_length: int = 77
):
    """
    Encode le texte en embeddings pour Classifier-Free Guidance.

    Args:
        tokenizer (CLIPTokenizer): tokenizer du modèle.
        text_encoder (CLIPTextModel): text encoder du modèle.
        prompt (str): texte positif.
        negative_prompt (str): texte négatif (CFG).
        device (str): "cuda" ou "cpu".
        dtype (torch.dtype): torch.float16 ou torch.float32.
        max_length (int): longueur max du tokenizer (77 pour SD).

    Returns:
        tuple: (pos_embeds, neg_embeds) chacun [1, max_length, 768]
    """
    # 🔹 Tokenisation texte positif
    pos_tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    # 🔹 Tokenisation texte négatif
    neg_tokens = tokenizer(
        negative_prompt if negative_prompt else "",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        # 🔹 Encodage en embeddings
        pos_embeds = text_encoder(**pos_tokens).last_hidden_state.to(dtype)
        neg_embeds = text_encoder(**neg_tokens).last_hidden_state.to(dtype)

    return pos_embeds, neg_embeds

def load_image_latent(image_path, vae, device="cuda", dtype=torch.float16, target_size=128):
    """
    Charge une image et la convertit en latent tensor pour le UNet.

    Args:
        image_path (str): Chemin vers l'image.
        vae (AutoencoderKL): VAE du modèle.
        device (str): "cuda" ou "cpu".
        dtype (torch.dtype): torch.float16 ou torch.float32.
        target_size (int): taille de l'image (carrée) pour le modèle.

    Returns:
        torch.Tensor: latent tensor [1, 4, H/8, W/8]
    """
    # 1️⃣ Charger et redimensionner
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W]

    # 2️⃣ Normalisation pour VAE
    img_tensor = img_tensor * 2.0 - 1.0  # [-1,1]

    # 3️⃣ Encoder via VAE (no_grad pour économiser mémoire)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()  # [1,4,H/8,W/8]

    return latent


# ------------------------------
# 🔹 Fonction latents optimisée
# ------------------------------

def generate_latents_ai_5D_light(
    latent_frame,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.0,
    init_image_scale=0.9,
    creative_noise=0.0,
    seed=1234,
    steps=25
):
    torch.manual_seed(seed)

    latent_frame = latent_frame.to(device=device, dtype=dtype)
    latents = latent_frame.repeat(2,1,1,1)  # batch=2 pour CFG une seule fois

    if creative_noise > 0.0:
        latents += torch.randn_like(latents) * creative_noise

    # embeddings
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:
        latent_input = latents

        if motion_module:
            latent_input = motion_module(latent_input)

        # blend avec image initiale
        latent_input = latent_input * init_image_scale + latent_frame.repeat(2,1,1,1) * (1 - init_image_scale)

        # UNet forward
        noise_pred = unet(latent_input, t, encoder_hidden_states=embeds).sample

        # guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # scheduler step
        latents_step = scheduler.step(noise_pred, t, latents[:1]).prev_sample

        # clamp sécurité
        latents_step = latents_step.clamp(-2.0, 2.0)

        # préparer pour prochaine étape (reduplication batch=2)
        latents = torch.cat([latents_step, latents_step], dim=0)

    return latents[:1]

# ------------------------------------------------------------
# generate_latents_ai_5D_optimized_test - Anomalie sortie Frame Noir
# ------------------------------------------------------------

def generate_latents_ai_5D_optimized_test(
    latent_frame,         # [1,4,H,W]
    pos_embeds,           # [1,77,768]
    neg_embeds,           # [1,77,768]
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.0,
    init_image_scale=0.9,
    creative_noise=0.0,
    seed=1234,
    steps=40
):

    torch.manual_seed(seed)

    latent_frame = latent_frame.to(device=device, dtype=dtype)

    if creative_noise > 0.0:
        latent_frame = latent_frame + torch.randn_like(latent_frame) * creative_noise

    # 🔹 Classifier-Free Guidance embeddings
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)
    # shape = [2,77,768]

    # 🔹 Duplicate latent for CFG
    latents = latent_frame.repeat(2, 1, 1, 1)  # [2,4,H,W]

    scheduler.set_timesteps(steps, device=device)

    for t in scheduler.timesteps:

        latent_input = latents

        # 🔹 Motion module (si utilisé)
        if motion_module is not None:
            latent_input = motion_module(latent_input)

        # 🔹 Blend avec image initiale (stabilité)
        latent_input = (
            latent_input * init_image_scale +
            latent_frame.repeat(2,1,1,1) * (1 - init_image_scale)
        )

        # 🔹 UNet forward
        noise_pred = unet(
            latent_input,
            t,
            encoder_hidden_states=embeds
        ).sample

        # 🔹 Guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # 🔹 Scheduler step
        latents = scheduler.step(
            noise_pred,
            t,
            latents[:1]  # on garde batch=1 pour le step
        ).prev_sample

        # 🔹 Clamp sécurité fp16
        latents = latents.clamp(-1.5, 1.5)

        # 🔹 Re-dupliquer pour prochaine itération
        latents = latents.repeat(2,1,1,1)

    # Retour final batch=1
    return latents[:1]


def generate_latents_ai_5D_batch(
    latent_frames,       # [B,C,T,H,W] tensor
    pos_embeds,          # embeddings positifs
    neg_embeds,          # embeddings négatifs
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.5,
    init_image_scale=0.85,
    creative_noise=0.0,
    seed=42,
    steps=35
):
    """
    Génération de latents animés pour plusieurs images / frames.

    latent_frames : [B,C,T,H,W]
    pos_embeds, neg_embeds : embeddings texte
    unet : modèle UNet
    scheduler : scheduler de diffusion
    motion_module : module de mouvement optionnel
    """

    torch.manual_seed(seed)

    B, C, T, H, W = latent_frames.shape
    latent_frames = latent_frames.to(device=device, dtype=dtype)

    # Ajouter bruit créatif si besoin
    if creative_noise > 0.0:
        noise = torch.randn_like(latent_frames) * creative_noise
        latent_frames = latent_frames + noise

    # Wrapper motion module
    if motion_module:
        def motion_wrapper(latents, timestep=None):
            try:
                return motion_module(latents)
            except TypeError:
                return motion_module(latents)
    else:
        motion_wrapper = lambda x, t=None: x

    # Scheduler timesteps
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    # Embeddings concaténés
    total_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
    total_batch = total_embeds.shape[0]

    # Boucle par frame
    for t_idx, t in enumerate(timesteps):
        # Itérer sur chaque timestep
        # latent_frames = [B,C,T,H,W] → on traite chaque frame
        for f_idx in range(T):
            latent_frame = latent_frames[:, :, f_idx, :, :]  # [B,C,H,W]

            # Motion module
            latent_frame = motion_wrapper(latent_frame, t)

            # Préparer batch pour guidance
            latent_input = latent_frame.repeat(total_batch, 1, 1, 1)
            latent_input = latent_input * init_image_scale + latent_frame.repeat(total_batch,1,1,1) * (1-init_image_scale)

            # UNet forward
            noise_pred = unet(latent_input, t, encoder_hidden_states=total_embeds).sample

            # Guidance scale
            if total_batch > 1:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            step_output = scheduler.step(noise_pred, t, latent_frame)
            latent_frame = step_output.prev_sample

            # Clamp sécurité fp16
            latent_frame = latent_frame.clamp(-1.5, 1.5)

            # Remettre la frame à sa place
            latent_frames[:, :, f_idx, :, :] = latent_frame

    return latent_frames

def generate_latents_ai_5D_optimized1(
    latent_frame,         # [B,C,H,W] tensor
    pos_embeds,           # embeddings positifs
    neg_embeds,           # embeddings négatifs
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.5,
    init_image_scale=0.85,
    creative_noise=0.0,
    seed=42,
    steps=35
):
    """
    Génération de latents animés 5D (B,C,T,H,W)
    - latent_frame : [B,C,H,W]
    - pos_embeds, neg_embeds : embeddings texte
    - unet : UNet
    - scheduler : scheduler de diffusion (DDIM/LMS)
    - motion_module : module de mouvement optionnel
    """
    torch.manual_seed(seed)

    # Assurer que le latent est sur le bon device
    latent_frame = latent_frame.to(device=device, dtype=dtype)

    # Ajouter bruit créatif
    if creative_noise > 0.0:
        noise = torch.randn_like(latent_frame) * creative_noise
        latent_frame = latent_frame + noise

    # Wrapper motion module
    if motion_module:
        def motion_wrapper(latents, timestep=None):
            try:
                return motion_module(latents)
            except TypeError:
                return motion_module(latents)
    else:
        motion_wrapper = lambda x, t=None: x

    # Scheduler timesteps
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    # Préparer batch pour guidance
    total_embeds = torch.cat([pos_embeds, neg_embeds], dim=0)
    batch_size = total_embeds.shape[0]

    # Répéter le latent pour correspondre au batch
    latent_model_input = latent_frame.repeat(batch_size, 1, 1, 1)

    # Boucle diffusion
    for i, t in enumerate(timesteps):
        latent_model_input = latent_model_input.to(device=device, dtype=dtype)

        # Motion module
        latent_model_input = motion_wrapper(latent_model_input, t)

        # Guidance input
        latent_model_input_in = latent_model_input.clone()
        latent_model_input_in = latent_model_input_in * init_image_scale + latent_frame.repeat(batch_size,1,1,1) * (1-init_image_scale)

        # UNet forward
        noise_pred = unet(latent_model_input_in, t, encoder_hidden_states=total_embeds).sample

        # Guidance scale
        if batch_size > 1:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred = noise_pred

        # Scheduler step
        step_output = scheduler.step(noise_pred, t, latent_model_input)
        latent_model_input = step_output.prev_sample

        # Clamp pour sécurité fp16
        latent_model_input = latent_model_input.clamp(-1.5, 1.5)

    return latent_model_input


# -------------------------
# generate_latents_ai_5D_optimized (fixed)
# -------------------------


def generate_latents_ai_5D_fixed(
    latent_frame,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.5,
    init_image_scale=0.85,
    creative_noise=0.0,
    seed=42,
    steps=35,
    clamp_max=1.0
):
    """
    Génération de latents animés 5D (B,C,H,W)
    """

    torch.manual_seed(seed)

    latent_model_input = latent_frame.clone()

    # Bruit créatif
    if creative_noise > 0.0:
        noise = torch.randn_like(latent_model_input) * creative_noise
        latent_model_input += noise

    # Motion module
    if motion_module:
        def motion_wrapper(latents, timestep=None):
            try:
                return motion_module(latents)
            except TypeError:
                return motion_module(latents)
    else:
        motion_wrapper = lambda x, t=None: x

    # Scheduler
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps

    for t in timesteps:
        latent_model_input = latent_model_input.to(device=device, dtype=dtype)

        # Motion module
        latent_model_input = motion_wrapper(latent_model_input, t)

        # Init image scale mix
        if init_image_scale < 1.0:
            latent_model_input = latent_model_input * init_image_scale + latent_frame * (1 - init_image_scale)

        # Guidance batch
        batch_size = pos_embeds.shape[0]
        latent_model_input_in = latent_model_input.repeat(batch_size * 2, 1, 1, 1)
        embeds = torch.cat([pos_embeds, neg_embeds], dim=0)

        # UNet
        noise_pred = unet(latent_model_input_in, t, encoder_hidden_states=embeds).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Scheduler step
        latent_model_input = scheduler.step(noise_pred, t, latent_model_input).prev_sample

        # Clamp
        latent_model_input = latent_model_input.clamp(-clamp_max, clamp_max)

    return latent_model_input

def generate_latents_ai_5D_256(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.75,
    creative_noise=0.1, seed=0, steps=20
):
    """
    Génère des latents animés optimisés pour AnimateDiff avec logs détaillés.
    Correctifs pour éviter fonds blancs et contours néons.
    """
    torch.manual_seed(seed)

    # Scheduler
    scheduler.set_timesteps(steps, device=device)

    # Déplacer latents et embeddings
    latents = latents.to(device=device, dtype=dtype)
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    # Ajout de bruit initial avec minimum 0.1 pour contraste
    noise = torch.randn_like(latents) * max(creative_noise, 0.1)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])
    print(f"[Init Noise] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}, Init_scale: {init_image_scale}, Noise: {creative_noise}")
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")
    print(f"Embeddings shape: {embeds.shape}, batch_size: {embeds.shape[0]}")

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for i, t in enumerate(scheduler.timesteps):

            # Motion module
            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
                print(f"[Step {i} Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Guidance
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step (clip_sample désactivé pour éviter fonds blancs)
            if "clip_sample" in scheduler.step.__code__.co_varnames:
                latents = scheduler.step(noise_pred, t, latents, clip_sample=False).prev_sample
            else:
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Logs avant clamp
            print(f"[Step {i} pre-clamp] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Clamp léger pour éviter overflow fp16
            LATENT_CLAMP = 1.5
            latents = latents.clamp(-LATENT_CLAMP, LATENT_CLAMP)

            # Logs après clamp
            print(f"[Step {i} post-clamp] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    # Amplification finale pour contraste avant VAE
    latents = latents * 1.2
    latents = latents.clamp(-LATENT_CLAMP, LATENT_CLAMP)

    return latents

def generate_latents_ai_5D_optimized_v1(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    """
    Génère des latents animés optimisés pour AnimateDiff avec logs détaillés.
    ✅ Modification : aucun clamp ou tanh sur les latents pour éviter fonds blancs et néons.
    """
    torch.manual_seed(seed)

    # Scheduler
    scheduler.set_timesteps(steps, device=device)

    # Déplacer latents et embeddings
    latents = latents.to(device=device, dtype=dtype)
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    # Ajout de bruit initial si demandé
    noise = torch.randn_like(latents) * creative_noise if creative_noise > 0 else torch.zeros_like(latents)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])
    print(f"[Init Noise] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}, Init_scale: {init_image_scale}, Noise: {creative_noise}")
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")
    print(f"Embeddings shape: {embeds.shape}, batch_size: {embeds.shape[0]}")

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for i, t in enumerate(scheduler.timesteps):

            # Motion module
            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
                print(f"[Step {i} Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Guidance
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Logs uniquement (pas de clamp)
            print(f"[Step {i}] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    return latents


def decode_latents_correct(latents, vae, latent_scale=0.18215):

    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Move + rescale latents
    latents = latents.to(device=vae_device, dtype=vae_dtype) / latent_scale

    with torch.no_grad():
        img = vae.decode(latents).sample

        # ✅ Conversion SD correcte
        img = (img / 2 + 0.5).clamp(0.0, 1.0)

    return img.cpu()


def decode_latents_correct_v1(latents, vae, latent_scale=0.18215):
    """
    Décodage correct des latents AnimateDiff.
    - latents: sortie UNet brute
    - vae: modèle VAE chargé
    - latent_scale: scale exact du VAE (souvent 0.18215 pour SD/miniSD)
    """
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Move latents sur le device et dtype du VAE
    if vae_device.type == "cpu":
        latents = latents.to(device=vae_device, dtype=torch.float32) / latent_scale
    else:
        latents = latents.to(device=vae_device, dtype=vae_dtype) / latent_scale

    with torch.no_grad():
        img = vae.decode(latents).sample
        img = img.clamp(0.0, 1.0)

    return img.cpu()


def decode_latents_safe_ai(latents, vae, tile_size=128, overlap=64):
    """
    Décodage sécurisé des latents pour AnimateDiff.
    - Compatible fp16/fp32 et VAE-offload.
    - Évite l’effet néon / fonds saturés.
    - Clamp léger pour prévenir overflow, sans écraser la dynamique.
    """
    vae_device = next(vae.parameters()).device

    # 1️⃣ Assurer le bon device et dtype
    dtype = torch.float16 if vae_device.type == "cuda" else torch.float32
    latents = latents.to(device=vae_device, dtype=dtype)

    # 2️⃣ Éviter NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=10.0, neginf=-10.0)

    # 3️⃣ Clamp large pour fp16 (prévention overflow)
    LATENT_CLAMP = 5.0
    latents = latents.clamp(-LATENT_CLAMP, LATENT_CLAMP)

    # 4️⃣ Décodage VAE
    with torch.no_grad():
        # Décode le latent
        frame_tensor = vae.decode(latents).sample

        # Clamp final entre 0 et 1 pour image
        frame_tensor = frame_tensor.clamp(0.0, 1.0)

    return frame_tensor.cpu()



def decode_latents_safe_torch(latent_frame, vae, tile_size=128, overlap=64, latent_scale=LATENT_SCALE):
    vae_device = next(vae.parameters()).device
    dtype = torch.float16 if vae_device.type == "cuda" else torch.float32
    latents = latent_frame.to(device=vae_device, dtype=dtype) / latent_scale

    # Éviter NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=10.0, neginf=-10.0)

    # Optionnel : clamp large pour fp16
    # latents = latents.clamp(-10.0, 10.0)

    with torch.no_grad():
        frame_tensor = vae.decode(latents).sample
        frame_tensor = frame_tensor.clamp(0.0, 1.0)

    return frame_tensor.cpu()


def decode_latents_safe_clamp(latent_frame, vae, tile_size=128, overlap=64, latent_scale=LATENT_SCALE):
    """
    Décodage sécurisé des latents en tensor float16/float32, compatible fp16/fp32 et VAE-offload.
    Découpe en tiles pour limiter l’usage VRAM.
    """
    vae_device = next(vae.parameters()).device

    # 1️⃣ Forcer dtype float16 si GPU, sinon float32 CPU
    dtype = torch.float16 if vae_device.type == "cuda" else torch.float32
    latents = latent_frame.to(device=vae_device, dtype=dtype) / latent_scale

    # 2️⃣ Éviter NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)

    # 3️⃣ Clamp pour éviter overflow fp16
    LATENT_CLAMP = 1.5
    latents = latents.clamp(-LATENT_CLAMP, LATENT_CLAMP)

    # 4️⃣ Décodage tile par tile si nécessaire (ici on simplifie pour petits latents)
    B, C, H, W = latents.shape
    with torch.no_grad():
        # Décode et force sortie entre 0 et 1
        frame_tensor = vae.decode(latents).sample
        frame_tensor = frame_tensor.clamp(0.0, 1.0)

    return frame_tensor.cpu()



def decode_latents_safe_ai1(latent_frame, vae, tile_size=128, overlap=64, latent_scale=LATENT_SCALE):
    """
    Décodage sécurisé des latents en tensor float32, compatible fp16/fp32 et VAE-offload.
    Découpe en tiles pour limiter l’usage VRAM.
    """
    vae_device = next(vae.parameters()).device
    latents = latent_frame.to(device=vae_device, dtype=torch.float32) / latent_scale  # float32 obligatoire pour VAE

    # Éviter NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)

    # Décodage tile par tile si nécessaire
    B, C, H, W = latents.shape
    frame_tensor = torch.zeros(B, 3, H*16, W*16, device=vae_device)  # dimension finale approximative
    # Pour AnimateDiff on peut simplifier avec une seule passe si pas très grand
    with torch.no_grad():
        frame_tensor = vae.decode(latents).sample.clamp(0, 1)

    return frame_tensor.cpu()

# -------------------------
# Encode image en latents
# -------------------------
def encode_image_latents(image_tensor, vae, scale=0.18215, dtype=torch.float16):
    """
    Encode une image en latents avec VAE.
    """
    vae_device = next(vae.parameters()).device
    img = image_tensor.to(device=vae_device, dtype=torch.float32 if vae_device.type=="cpu" else dtype)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample() * scale
    print(f"[Encode] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")
    return latents.unsqueeze(2)  # [B,C,1,H,W]
# -------------------------
# Génération latents animés optimisés
# -------------------------
def generate_latents_ai_5D_optimized_old(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    """
    Génère des latents animés optimisés pour AnimateDiff avec logs détaillés.
    """
    torch.manual_seed(seed)

    # Scheduler
    scheduler.set_timesteps(steps, device=device)

    # Déplacer latents et embeddings
    latents = latents.to(device=device, dtype=dtype)
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    # Ajout de bruit initial
    noise = torch.randn_like(latents) * creative_noise if creative_noise > 0 else torch.zeros_like(latents)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])
    print(f"[Init Noise] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance: {guidance_scale}, Init_scale: {init_image_scale}, Noise: {creative_noise}")
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")
    print(f"Embeddings shape: {embeds.shape}, batch_size: {embeds.shape[0]}")

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for i, t in enumerate(scheduler.timesteps):

            # Motion module
            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
                print(f"[Step {i} Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Guidance
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Logs avant clamp
            print(f"[Step {i} pre-clamp] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Clamp progressif pour éviter overflow
            #clamp_val = max(1.0, init_image_scale * 10)
            #latents = latents.clamp(-clamp_val, clamp_val)

            # Clamp fixe à ±1.0
            #CLAMP_MAX = 1.0
            #latents = latents.clamp(-CLAMP_MAX, CLAMP_MAX)

            # Calculer max absolu actuel
            #max_abs = latents.abs().max()
            #if max_abs > 1.0:
            #    latents = latents / max_abs  # ramène les latents dans [-1, 1]

            latents = torch.tanh(latents)  # comprime automatiquement dans [-1, 1]

            # Logs après clamp
            print(f"[Step {i} post-clamp] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    return latents

# -------------------------
# Décodage sécurisé
# -------------------------
def decode_latents_safe(latents, vae, device="cuda", tile_size=128, overlap=64):
    """
    Décodage sécurisé des latents en tensor float32 sur CPU.
    - Compatible avec vae_offload (CPU) et latents GPU
    """
    vae_device = next(vae.parameters()).device
    latents = latents.to(vae_device).float()

    # Nettoyage NaN/Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)
    print(f"[Decode start] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")

    # Décodage en tiles (pour VRAM limitée)
    frame_tensor = decode_latents_to_image_tiled128(
        latents,
        vae,
        tile_size=tile_size,
        overlap=overlap,
        device=vae_device
    ).clamp(0, 1)

    print(f"[Decode end] Frame tensor min: {frame_tensor.min():.4f}, max: {frame_tensor.max():.4f}")

    return frame_tensor.cpu()

def generate_latents_ai_5D_debug_opti(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    """
    Génère des latents animés optimisés pour AnimateDiff.
    """
    import torch

    torch.manual_seed(seed)

    # Configure le scheduler
    scheduler.set_timesteps(steps, device=device)

    # Déplace latents et embeddings sur device
    latents = latents.to(device=device, dtype=dtype)
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    # Ajout de bruit initial contrôlé par creative_noise
    if creative_noise > 0.0:
        noise = torch.randn_like(latents) * creative_noise
    else:
        noise = torch.zeros_like(latents)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])

    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance scale: {guidance_scale}, Init scale: {init_image_scale}, Creative noise: {creative_noise}")
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")
    print(f"Embeddings shape: {embeds.shape}, batch_size: {embeds.shape[0]}")

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for i, t in enumerate(scheduler.timesteps):
            # Motion module (optionnel)
            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
            print(f"[Step {i}] t: {t:.4f} | [Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Guidance
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Step scheduler et scaling progressif pour éviter la compression
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Scaling progressif pour garder amplitude cohérente
            clamp_val = max(1.0, init_image_scale * 10)
            latents = latents.clamp(-clamp_val, clamp_val)

            print(f"[Step {i} post-step] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    return latents

def generate_latents_ai_5D_brouillard(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    """
    Génère des latents animés 5D à partir d'une image de base avec logs.
    """

    torch.manual_seed(seed)
    print(f"🔥 Seed: {seed}, Steps: {steps}, Guidance scale: {guidance_scale}, Init scale: {init_image_scale}, Creative noise: {creative_noise}")

    # -------------------------------------
    # Scheduler et timesteps
    # -------------------------------------
    scheduler.set_timesteps(steps, device=device)
    print(f"Scheduler timesteps: {scheduler.timesteps}")

    # -------------------------------------
    # Initialisation des latents
    # -------------------------------------
    latents = latents.to(device=device, dtype=torch.float32)
    noise = torch.randn_like(latents) * (creative_noise if creative_noise > 0 else 1.0)
    latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0:1])
    print(f"[Init] Latents shape: {latents.shape}, min: {latents.min():.4f}, max: {latents.max():.4f}")

    # -------------------------------------
    # Concat embeddings une seule fois
    # -------------------------------------
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)
    batch_size = embeds.shape[0]
    print(f"Embeddings shape: {embeds.shape}, batch_size: {batch_size}")

    # -------------------------------------
    # Boucle principale
    # -------------------------------------
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):
        for step_idx, t in enumerate(scheduler.timesteps):
            print(f"\n[Step {step_idx}] t: {t}")

            # Motion module si disponible
            if motion_module is not None:
                if latents.dim() == 4:  # [B, C, H, W] -> ajouter F=1
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
                print(f"[Motion] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

            # Répéter latents pour correspondre au batch d'embeddings
            if latents.shape[0] != batch_size:
                repeats = batch_size // latents.shape[0]
                latent_model_input = latents.repeat(repeats, 1, 1, 1)
            else:
                latent_model_input = latents

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # UNet
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

            # CFG : split batch
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Step scheduler
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            latents = latents.clamp(-20, 20)  # pour stabilité

            print(f"[Step {step_idx} post-step] Latents min: {latents.min():.4f}, max: {latents.max():.4f}")

    return latents.to(dtype)

def generate_latents_ai_5D_testcreative(
    latents,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=5.0,
    init_image_scale=0.6,
    creative_noise=0.0,   # <-- AJOUTE ÇA
    seed=0,
    steps=12
):
    torch.manual_seed(seed)

    scheduler.set_timesteps(steps, device=device)

    latents = latents.to(device=device, dtype=dtype)

    # ---------------- IMG2IMG strength ----------------
    t_start = int(steps * init_image_scale)
    t_start = min(t_start, steps - 1)

    timesteps = scheduler.timesteps[t_start:]
    t_noise = timesteps[0:1]

    noise = torch.randn_like(latents)

    latents = scheduler.add_noise(
        latents,
        noise,
        t_noise
    )

    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):

        for t in timesteps:

            if motion_module is not None:
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)

            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)

            noise_pred = noise_uncond + guidance_scale * (
                noise_text - noise_uncond
            )

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents



def generate_latents_ai_5D_optitest(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85,
    creative_noise=0.0, seed=0, steps=20
):
    torch.manual_seed(seed)

    # 🔥 IMPORTANT : config réelle des steps
    scheduler.set_timesteps(steps, device=device)

    # Utilise dtype natif (fp16 si cuda)
    latents = latents.to(device=device, dtype=dtype)
    #latents = latents * scheduler.init_noise_sigma
    noise = torch.randn_like(latents)
    latents = scheduler.add_noise(
        latents,
        noise,
        scheduler.timesteps[0:1]
    )
    # -----------------------------------------------------
    # 🔥 concat embeddings UNE SEULE FOIS
    embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(device=device, dtype=dtype)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype):

        for t in scheduler.timesteps:

            if motion_module:
                #latents = motion_module.apply(latents, t)
                if motion_module is not None:
                    # Si latents = [B,C,H,W], on ajoute F=1
                    if latents.dim() == 4:
                        latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                    else:
                        latents = motion_module(latents)

            # CFG batching propre
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

            noise_uncond, noise_text = noise_pred.chunk(2)

            noise_pred = noise_uncond + guidance_scale * (
                noise_text - noise_uncond
            )

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


# -------------------------
# Décodage hybride safe VAE
# -------------------------
def decode_latents_hybrid(vae, latents, scale=LATENT_SCALE):
    """
    Décodage sécurisé des latents pour VAE fp32 + latents fp16.
    latents : Tensor GPU (fp16 ou fp32)
    vae : modèle VAE (fp32 ou offload)
    scale : facteur d'échelle des latents
    """
    # Sauvegarder dtype des latents
    lat_dtype = latents.dtype

    # Conversion en fp32 pour VAE
    latents_fp32 = latents.to(torch.float32)

    with torch.no_grad():
        decoded = vae.decode(latents_fp32 / scale).sample

    # Clamp pour éviter valeurs extrêmes
    decoded = decoded.clamp(-1, 1)

    # Retourner au dtype original pour GPU/fp16 si besoin
    return decoded.to(lat_dtype)


# --- PIPELINE PRINCIPALE ---
def generate_5D_video_auto(pretrained_model_path, config, device='cuda'):
    print("🔄 Chargement des modèles...")
    motion_module = MotionModuleTiny(device=device)
    scheduler = init_scheduler(config)  # ta fonction existante
    vae = load_vae(pretrained_model_path, device=device)

    total_frames = config['total_frames']
    fps = config['fps']
    H_src, W_src = config['image_size']  # résolution source

    # Génère les latents initiaux
    latents = torch.randn(1, 4, H_src//8, W_src//8, device=device, dtype=torch.float16)
    print(f"[INFO] Latents initiaux shape={latents.shape}")

    video_frames = []
    for t in range(total_frames):
        try:
            latents = motion_module.step(latents, t)
            frame = decode_latents_frame_auto(latents, vae, H_src, W_src)
            video_frames.append(frame)
        except Exception as e:
            print(f"⚠ Erreur frame {t:05d} → reset léger: {e}")
            continue

    save_video(video_frames, fps, output_path=config['output_path'])
    print(f"🎬 Vidéo générée : {config['output_path']}")

# -------------------------
# Load images utility
# -------------------------
def load_images(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        t = load_image_file(p, W, H, device, dtype)
        print(f"✅ Image chargée : {p}, shape={t.shape}, dtype={t.dtype}, device={t.device}")
        all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)

# -------------------------
# Mémoire GPU utils
# -------------------------
def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        print(f"[GPU MEM] {tag} → allocated={torch.cuda.memory_allocated()/1e6:.1f}MB, "
              f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB, "
              f"max_allocated={torch.cuda.max_memory_allocated()/1e6:.1f}MB")

def decode_latents_frame_auto(latents, vae, H_src, W_src):
    """
    Decode des latents VAE en images avec tiles 128x128, auto-adapté à la taille source.
    """
    device = vae.device
    print(f"[VAE] Decode → tile_size={tile_size}, overlap={overlap}, device={device}, latents.shape={latents.shape}")
    log_gpu_memory("avant decode VAE")

    # Assure batch 4D
    latents = latents.unsqueeze(0) if latents.dim() == 3 else latents

    # Décodage VAE en tiles
    with torch.no_grad():
        frame_tensor = decode_latents_to_image_tiled(
            latents,
            vae,
            tile_size=tile_size,
            overlap=overlap
        ).clamp(0,1)

        # Redimensionnement proportionnel à l'image source
        H_out, W_out = H_src, W_src
        if frame_tensor.shape[-2:] != (H_out, W_out):
            frame_tensor = torch.nn.functional.interpolate(
                frame_tensor,
                size=(H_out, W_out),
                mode='bicubic',
                align_corners=False
            )

    log_gpu_memory("après decode VAE")
    return frame_tensor.squeeze(0)
# -------------------------------------------------------------------
# ------------- PATCH Stable Diffusion --------
# ------------------------------------------------------------------

def generate_latents_ai_5D_std(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0, seed=0
):
    torch.manual_seed(seed)

    # Toujours float32 pour stabilité du scheduler
    latents = latents.to(device=device, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:

        # Sécurité NaN / Inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = torch.randn_like(latents) * 0.1

        # Si Motion Module est actif
        if motion_module:
            #latents = motion_module.apply(latents, t)
            if motion_module is not None:
                # Si latents = [B,C,H,W], on ajoute F=1
                if latents.dim() == 4:
                    latents = motion_module(latents.unsqueeze(2)).squeeze(2)
                else:
                    latents = motion_module(latents)
        # Motion Module fin

        # 🔥 Préparer batch pour guidance
        batch_size = pos_embeds.shape[0] + neg_embeds.shape[0]
        if latents.shape[0] != batch_size:
            # Répéter latents pour matcher embeddings
            repeats = batch_size // latents.shape[0]
            latent_model_input = latents.repeat(repeats, 1, 1, 1)
        else:
            latent_model_input = latents

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(dtype=dtype)

        embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        # Guidance : séparer le batch
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond.float() + guidance_scale * (
            noise_text.float() - noise_uncond.float()
        )

        # Step scheduler
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents.clamp(-20, 20)

    # Retour dtype original
    return latents.to(dtype)

# -------------------------------------------------------------------
# ------------- 5 D Original  --------
# ------------------------------------------------------------------

def generate_latents_ai_5D(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0, seed=0
):
    torch.manual_seed(seed)

    latents = latents.to(device=device, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:

        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = torch.randn_like(latents) * 0.1

        # 🔥 DUPLICATION POUR GUIDANCE
        latent_model_input = torch.cat([latents, latents], dim=0)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(dtype=dtype)

        embeds = torch.cat([neg_embeds, pos_embeds], dim=0).to(dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)

        noise_pred = noise_uncond.float() + guidance_scale * (
            noise_text.float() - noise_uncond.float()
        )

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents.clamp(-20, 20)

    return latents.to(dtype)


def generate_latents_ai_5D_v1(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0, seed=0
):
    torch.manual_seed(seed)
    latents = latents.to(device=device, dtype=torch.float32)  # scheduler en float32
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = torch.randn_like(latents) * 0.1

        # préparation input UNet 5D
        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input = latent_model_input.to(dtype=dtype)  # UNet en fp16 si demandé

        # concat embeddings pour guidance
        embeds = torch.cat([neg_embeds, pos_embeds])
        embeds = embeds.to(dtype=dtype)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample

        # guidance
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond.float() + guidance_scale * (noise_text.float() - noise_uncond.float())

        # scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # clamp anti-explosion
        latents = latents.clamp(-20, 20)

    return latents.to(dtype)


# -------------------------
# UNet 3D latents generation
# -------------------------
def generate_latents_3d(
    latents, pos_embeds, neg_embeds, unet, scheduler,
    motion_module=None, device="cuda", dtype=torch.float16,
    guidance_scale=4.5, init_image_scale=0.85, creative_noise=0.0, seed=0
):
    torch.manual_seed(seed)

    # Toujours float32 pour scheduler
    latents = latents.to(device=device, dtype=torch.float32)

    if init_image_scale < 1.0:
        noise = torch.randn_like(latents)
        latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0])
    else:
        latents = latents * scheduler.init_noise_sigma

    latents = latents.clamp(-10, 10)

    for t in scheduler.timesteps:

        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"⚠ NaN détecté à timestep {int(t)} → reset léger")
            latents = torch.randn_like(latents) * 0.1

        # 3D CFG concat
        latent_model_input = torch.cat([latents]*2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        embeds = torch.cat([neg_embeds, pos_embeds])

        if motion_module:
            latent_model_input = motion_module.apply(latent_model_input)

        with torch.no_grad():
            noise_pred = unet(latent_model_input.to(dtype), t, encoder_hidden_states=embeds).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond.float() + guidance_scale * (noise_text.float() - noise_uncond.float())

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents.clamp(-20,20)

        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-6:
            print(f"⚠ Latent instable à timestep {int(t)} → reset")
            latents = torch.randn_like(latents) * 0.05

    return latents.to(dtype)

# ---------------------------------------------------------
# Génération de latents AI compatible Fp16  et Fp32
# ---------------------------------------------------------

def generate_latents_ai(
    latents,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=4.5,
    init_image_scale=0.85,
    creative_noise=0.0,
    seed=0,
):
    torch.manual_seed(seed)

    use_fp16 = dtype == torch.float16 and device == "cuda"

    # ------------------------------------------------
    # Toujours garder les latents scheduler en float32
    # ------------------------------------------------
    latents = latents.to(device=device, dtype=torch.float32)

    # ------------------------------------------------
    # Initialisation correcte (image vs text mode)
    # ------------------------------------------------
    if init_image_scale < 1.0:
        noise = torch.randn_like(latents)
        latents = scheduler.add_noise(latents, noise, scheduler.timesteps[0])
    else:
        latents = latents * scheduler.init_noise_sigma

    latents = latents.clamp(-10, 10)

    for t in scheduler.timesteps:

        # ------------------------------------------------
        # Sécurité anti-NaN
        # ------------------------------------------------
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"⚠ NaN détecté à timestep {int(t)} → reset léger")
            latents = torch.randn_like(latents) * 0.1

        # ------------------------------------------------
        # CFG concat (plus stable et plus rapide)
        # ------------------------------------------------
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        embeds = torch.cat([neg_embeds, pos_embeds])

        # UNet en fp16 si activé
        model_input = latent_model_input.to(dtype if use_fp16 else torch.float32)

        with torch.no_grad():
            noise_pred = unet(
                model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)

        # ------------------------------------------------
        # Guidance toujours en float32 pour stabilité
        # ------------------------------------------------
        noise_pred = noise_uncond.float() + guidance_scale * (
            noise_text.float() - noise_uncond.float()
        )

        # ------------------------------------------------
        # Step scheduler en float32
        # ------------------------------------------------
        latents = scheduler.step(
            noise_pred,
            t,
            latents
        ).prev_sample

        latents = latents.clamp(-20, 20)

        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-6:
            print(f"⚠ Latent instable à timestep {int(t)} → reset")
            latents = torch.randn_like(latents) * 0.05

    # Retour au dtype demandé
    return latents.to(dtype if use_fp16 else torch.float32)



# ---------------------------------------------------------
# Génération de latents par bloc SAFE avec logs
# ---------------------------------------------------------
def generate_latents_2(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None,
                       device="cuda", dtype=torch.float16, guidance_scale=4.5, init_image_scale=0.85):
    """
    latents: [B, C, F, H, W]
    pos_embeds / neg_embeds: [B, L, D]
    """
    torch.manual_seed(42)
    B, C, F, H, W = latents.shape

    # ⚡ Assurer dtype/device compatibilité UNet
    unet_dtype = next(unet.parameters()).dtype
    latents = latents.to(device=device, dtype=unet_dtype)
    pos_embeds = pos_embeds.to(device=device, dtype=unet_dtype)
    if neg_embeds is not None:
        neg_embeds = neg_embeds.to(device=device, dtype=unet_dtype)

    motion_module = motion_module.to(device=device, dtype=unet_dtype) if motion_module else None

    # Reshape pour timesteps
    latents = latents.permute(0, 2, 1, 3, 4).reshape(B*F, C, H, W).contiguous()
    init_latents = latents.clone()

    for t in scheduler.timesteps:
        try:
            if motion_module:
                latents = motion_module(latents)

            # classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            embeds = torch.cat([neg_embeds, pos_embeds]) if neg_embeds is not None else pos_embeds

            # Vérification NaN avant UNet
            if torch.isnan(latents).any():
                print(f"❌ Warning: NaN detected in latents before UNet | t={t}")

            with torch.no_grad():
                noise_pred = unet(latent_model_input, timestep=t, encoder_hidden_states=embeds).sample

            if neg_embeds is not None:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # ⚡ conserver influence de l'image initiale
            latents = latents + init_image_scale * (init_latents - latents)

            # Logs min/max
            print(f"🔹 Step t={t} | latents min: {latents.min():.6f}, max: {latents.max():.6f}")

            # Stop si NaN
            if torch.isnan(latents).any():
                raise RuntimeError(f"NaN detected in latents after UNet at timestep {t}")

        except Exception as e:
            print(f"❌ Erreur UNet/scheduler à t={t}: {e}")
            # On peut remplacer par latents initiaux pour continuer
            latents = init_latents.clone()
            torch.cuda.empty_cache()

    # Reshape final
    latents = latents.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
    return latents


# ---------------------------------------------------------
# Tuilage sécurisé
# ---------------------------------------------------------
def decode_latents_to_image_tiled(latents, vae, tile_size=32, overlap=8):
    """
    Decode VAE en tuiles avec couverture complète garantie.
    - Aucun trou possible
    - Blending propre
    - Stable mathématiquement
    """

    device = vae.device
    latents = latents.to(device).float() / LATENT_SCALE

    B, C, H, W = latents.shape
    stride = tile_size - overlap

    # Dimensions image finale (scale factor VAE = 8)
    out_H = H * 8
    out_W = W * 8

    output = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output)

    # --- positions garanties ---
    y_positions = list(range(0, H - tile_size + 1, stride))
    x_positions = list(range(0, W - tile_size + 1, stride))

    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]

    if y_positions[-1] != H - tile_size:
        y_positions.append(H - tile_size)

    if x_positions[-1] != W - tile_size:
        x_positions.append(W - tile_size)

    for y in y_positions:
        for x in x_positions:

            y1 = y + tile_size
            x1 = x + tile_size

            tile = latents[:, :, y:y1, x:x1]

            with torch.no_grad():
                decoded = vae.decode(tile).sample

            decoded = (decoded / 2 + 0.5).clamp(0, 1)

            iy0 = y * 8
            ix0 = x * 8
            iy1 = y1 * 8
            ix1 = x1 * 8

            output[:, :, iy0:iy1, ix0:ix1] += decoded
            weight[:, :, iy0:iy1, ix0:ix1] += 1.0

    return output / weight.clamp(min=1e-6)
# -------------------------
# Génération tuilée 128x128 ultra safe VRAM
# -------------------------
def decode_latents_to_image_tiled_ori(latents, vae, tile_size=32, overlap=8):
    """
    Decode VAE en tuiles côté latent.
    Stable, sans toucher au scheduler.
    latents: [B, 4, H, W]
    """

    device = vae.device
    latents = latents.to(device).float() / LATENT_SCALE

    B, C, H, W = latents.shape
    stride = tile_size - overlap

    # Taille image finale (VAE scale factor = 8)
    out_H = H * 8
    out_W = W * 8

    output = torch.zeros(B, 3, out_H, out_W, device="cpu")
    weight = torch.zeros_like(output)

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            y1 = min(y + tile_size, H)
            x1 = min(x + tile_size, W)

            tile = latents[:, :, y:y1, x:x1]

            with torch.no_grad():
                decoded = vae.decode(tile).sample

            decoded = (decoded / 2 + 0.5).clamp(0, 1)

            # coordonnées en image space
            iy0 = y * 8
            ix0 = x * 8
            iy1 = y1 * 8
            ix1 = x1 * 8

            output[:, :, iy0:iy1, ix0:ix1] += decoded.cpu()
            weight[:, :, iy0:iy1, ix0:ix1] += 1

    return (output / weight.clamp(min=1e-6)).to(device)


# ---------------------------------------------------------
# Chargement image unique → [1,C,1,H,W]
# ---------------------------------------------------------
def load_input_image(image_path, W, H, device, dtype):
    img = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    img = preprocess(img).unsqueeze(0).unsqueeze(2)  # [1,C,1,H,W]
    return img.to(device=device, dtype=dtype)


# ---------------------------------------------------------
# Encode images → latents
# input: [B,C,T,H,W]
# output: [B,4,T,H_lat,W_lat]
# ---------------------------------------------------------
def encode_images(input_images, vae):
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    B, C, T, H, W = input_images.shape
    input_images = input_images.to(device=device, dtype=dtype)

    latents_list = []

    with torch.no_grad():
        for t in range(T):
            imgs_2d = input_images[:, :, t, :, :]  # [B,C,H,W]
            latent = vae.encode(imgs_2d).latent_dist.sample()
            latent = latent * LATENT_SCALE
            latents_list.append(latent)

    latents = torch.stack(latents_list, dim=2)  # [B,4,T,H_lat,W_lat]
    return latents

#---------------------------------------------------------
# VERSION ROBUSTE
# # Ajouter un peu de bruit créatif initial si demandé
# # Reshape pour traitement UNet [B*T, C, H, W]
#---------------------------------------------------------


from functools import wraps
import torch
from contextlib import contextmanager

# 🔹 Context manager pour désactiver xFormers temporairement
@contextmanager
def disable_xformers(unet):
    """
    Désactive memory-efficient xFormers attention si disponible.
    Utile pour éviter les erreurs FakeTensor pendant le safe encode/fallback.
    """
    orig = getattr(unet, "enable_xformers_memory_efficient_attention", None)
    if callable(orig):
        # Désactiver temporairement
        unet.enable_xformers_memory_efficient_attention(False)
    yield
    if callable(orig):
        # Réactiver
        unet.enable_xformers_memory_efficient_attention(True)


def generate_latents_safe_test(unet, **kwargs):
    """
    Génération de latents avec UNet, FP16-safe et debug.
    Affiche min/max à chaque étape pour détecter si les latents restent à zéro.
    """

    # Arguments connus
    known_kwargs = [
        "scheduler", "input_latents", "embeddings", "motion_module",
        "guidance_scale", "device", "fp16", "steps", "debug"
    ]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_kwargs}

    # Required
    input_latents = filtered_kwargs.get("input_latents")
    if input_latents is None:
        raise ValueError("⚠️ 'input_latents' doit être fourni")

    device = filtered_kwargs.get("device", "cuda")
    fp16 = filtered_kwargs.get("fp16", True)
    debug = filtered_kwargs.get("debug", False)
    motion_module = filtered_kwargs.get("motion_module", None)
    scheduler = filtered_kwargs.get("scheduler", None)
    steps = filtered_kwargs.get("steps", 20)
    embeddings = filtered_kwargs.get("embeddings", None)

    # Latents initiaux
    latents = input_latents.clone().to(device=device, dtype=torch.float16 if fp16 else torch.float32)
    is_video = latents.ndim == 5
    steps_list = getattr(scheduler, "timesteps", range(steps))

    if debug:
        print(f"[DEBUG] Initial latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    # Désactiver Dynamo/Inductor pour debug
    torch._dynamo.reset()
    torch._dynamo.disable()

    for step_idx, t in enumerate(steps_list):
        if motion_module is not None:
            latents = motion_module(latents)
            latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

        if is_video:
            B, C, F, H, W = latents.shape
            latents_unet = latents.permute(0, 2, 1, 3, 4).reshape(B*F, C, H, W)
        else:
            latents_unet = latents

        # FP16 si demandé
        latents_unet = latents_unet.half() if fp16 else latents_unet.float()

        # 🔹 Forward UNet
        try:
            unet_out = unet(latents_unet, t, encoder_hidden_states=embeddings)
            latents_out = unet_out["sample"]
        except Exception as e:
            print(f"⚠️ [UNet ERROR] step={step_idx} - {e}")
            continue

        # Reshape si vidéo
        if is_video:
            latents = latents_out.reshape(B, F, C, H, W).permute(0, 2, 1, 3, 4)
        else:
            latents = latents_out

        # Nettoyage et clamp
        latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        if debug:
            print(f"[DEBUG Step {step_idx}] latents min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    if debug:
        print(f"[INFO] Finished latents generation, final min/max={latents.min().item():.4f}/{latents.max().item():.4f}")

    return latents


# 🔹 Wrapper général sûr
def generate_latents_safe_wrapper_test(unet, **kwargs):
    try:
        return generate_latents_safe_test(unet, **kwargs)
    except TypeError as e:
        if "meta__efficient_attention_backward() got multiple values for argument 'scale'" in str(e):
            print("⚠️ [SAFE WRAPPER] Efficient attention error caught, returning input latents")
            return kwargs.get("input_latents").clone()
        else:
            raise e

# ---------------- SAFE WRAPPER ----------------


def generate_latents_robuste_safe(latents, pos_embeds, neg_embeds, unet, scheduler,
                                  motion_module=None, device="cuda", dtype=torch.float16,
                                  guidance_scale=7.5, init_image_scale=2.0,
                                  creative_noise=0.0, seed=42):

    B, C_orig, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)

    latents = latents.permute(0,2,1,3,4).reshape(B*T, C_orig, H, W).contiguous()
    init_latents = latents.clone()

    torch.manual_seed(seed)

    # 🔹 IMPORTANT pour DPMSolver
    scheduler._step_index = None

    for t_step in scheduler.timesteps:

        if motion_module:
            latents = motion_module(latents)

        if creative_noise > 0:
            latents = latents + torch.randn_like(latents) * creative_noise

        latent_model_input = torch.cat([latents, latents], dim=0)

        if latent_model_input.shape[1] == 1:
            latent_model_input = latent_model_input.repeat(1,4,1,1)

        embeds = torch.cat([neg_embeds, pos_embeds], dim=0)

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t_step,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        step_output = scheduler.step(
            model_output=noise_pred,
            timestep=t_step,
            sample=latents
        )

        latents = getattr(step_output, "prev_sample", step_output)

        latents = latents + init_image_scale * (init_latents - latents)

        latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)
        latents = latents.clamp(-5.0,5.0)

    C_final = latents.shape[1]
    latents = latents.reshape(B, T, C_final, H, W).permute(0,2,1,3,4).contiguous()

    return latents


#-------------------------------------------------------------------------------
# VERY STABLE
#-------------------------------------------------------------------------------
def generate_latents_robuste(latents, pos_embeds, neg_embeds, unet, scheduler,
                             motion_module=None, device="cuda", dtype=torch.float16,
                             guidance_scale=7.5, init_image_scale=2.0,
                             creative_noise=0.0, seed=42):
    """
    Génère des latents pour une frame ou un batch de frames avec robustesse.

    latents : [B,C,T,H,W] latents encodés et scalés
    pos_embeds / neg_embeds : embeddings textuels
    creative_noise : float, bruit supplémentaire pour variation frame à frame
    """
    torch.manual_seed(seed)
    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    init_latents = latents.clone()  # copie pour init_image_scale

    for t_step in scheduler.timesteps:
        # Motion module (optionnel)
        if motion_module is not None:
            latents = motion_module(latents)

        # Ajouter un peu de noise créatif si demandé
        if creative_noise > 0:
            latents = latents + torch.randn_like(latents) * creative_noise

        # Classifier-free guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0)
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t_step, encoder_hidden_states=embeds).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t_step, latents).prev_sample

        # Réinjection de l'image initiale
        latents = latents + init_image_scale * (init_latents - latents)

        # Vérification NaN / inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print(f"⚠ NaN/inf détecté à timestep {t_step}, réinitialisation avec petit bruit")
            latents = latents.clone()
            latents = latents + torch.randn_like(latents) * 1e-3

        # Log de debug
        mean_val = latents.abs().mean().item()
        std_val = latents.std().item()
        if math.isnan(mean_val) or mean_val < 1e-5:
            print(f"⚠ Latent trop petit à timestep {t_step}, mean={mean_val:.6f}")

    # Remettre la forme [B,C,T,H,W]
    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()
    return latents

#-------------------------------------------------------------------------------
# VERY STABLE - Parfait pour l'init de la video' - Version la plus adapter
#-------------------------------------------------------------------------------
def generate_latents_robuste_4D(
        latents,
        unet,
        scheduler,
        pos_embeds=None,
        neg_embeds=None,
        motion_module=None,
        device='cuda',
        dtype=torch.float16,
        guidance_scale=1.0,
        init_image_scale=0.85,
        creative_noise=0.0,
        steps=8,
        seed=None,
        gamma_boost=1.08,           # renforce couleurs ~10%
        detail_strength=0.03,        # amplification douce des détails
        adaptive_noise_strength=0.005 # bruit créatif local sur zones plates
    ):
    """
    Génération initiale de latents ultra-safe 4D pour AnimateDiff.
    Clamp, correction NaN/Inf et boost adaptatif de détails et couleurs.
    """
    import torch
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(seed)

    latents = latents.to(device=device, dtype=dtype)

    # Échelle initiale
    if init_image_scale != 1.0:
        latents = latents * init_image_scale

    for t in range(steps):
        with torch.no_grad():
            # ----- Dummy UNet step -----
            noise = torch.randn_like(latents) * 0.01
            latents = latents + noise

            # ----- Motion module si présent -----
            if motion_module is not None:
                latents, _ = motion_module(latents)

            # ----- Clamp strict et correction NaN/Inf -----
            latents = torch.clamp(latents, -1.0, 1.0)
            latents[torch.isnan(latents)] = 0.0
            latents[torch.isinf(latents)] = 0.0

            # ----- Amplification douce des détails -----
            mean = latents.mean(dim=[2,3], keepdim=True)
            detail = latents - mean
            latents = latents + detail_strength * torch.tanh(detail)

            # ----- Bruit créatif adaptatif local -----
            if adaptive_noise_strength > 0:
                low_contrast_mask = (latents.std(dim=[2,3], keepdim=True) < 0.1).float()
                latents = latents + low_contrast_mask * torch.randn_like(latents) * adaptive_noise_strength

            # ----- Bruit créatif global -----
            if creative_noise > 0.0:
                latents += torch.randn_like(latents) * creative_noise

    # ----- Gamma adaptatif pour les couleurs -----
    latents_norm = ((latents + 1.0) / 2.0).clamp(0,1)
    latents_norm = latents_norm ** gamma_boost
    latents = latents_norm * 2 - 1

    # ----- Assurer 4D -----
    if latents.ndim != 4:
        B, C, H, W = latents.shape[0], 4, latents.shape[-2], latents.shape[-1]
        latents = latents[:, :C, :, :] if latents.shape[1] >= 4 else F.pad(latents, (0,0,0,0,0,4-latents.shape[1]))

    return latents

def generate_latents_robuste_4D_net(
    latents,
    unet,
    scheduler,
    pos_embeds=None,
    neg_embeds=None,
    motion_module=None,
    device='cuda',
    dtype=torch.float16,
    guidance_scale=1.0,
    init_image_scale=0.85,
    creative_noise=0.0,
    steps=8,
    seed=None,
    clamp_percentile=99.5,
    smoothing_factor=0.05
):
    """
    Génération initiale de latents ultra-safe 4D pour AnimateDiff.
    - Clamp adaptatif par percentile
    - Correction NaN / Inf intelligente
    - Bruit créatif progressif
    - Motion module sécurisé
    - Option smoothing léger pour stabiliser latents
    """
    import torch
    import torch.nn.functional as F

    if seed is not None:
        torch.manual_seed(seed)

    latents = latents.to(device=device, dtype=dtype)
    if init_image_scale != 1.0:
        latents = latents * init_image_scale

    for t in range(steps):
        with torch.no_grad():
            # === Bruit UNet (placeholder) ===
            noise = torch.randn_like(latents) * 0.01
            latents = latents + noise

            # === Motion module ===
            if motion_module is not None:
                latents, _ = motion_module(latents)

            # === Correction NaN / Inf ===
            mask_invalid = torch.isnan(latents) | torch.isinf(latents)
            if mask_invalid.any():
                latents[mask_invalid] = latents[~mask_invalid].mean()

            # === Clamp adaptatif ===
            upper = torch.quantile(latents, clamp_percentile / 100.0)
            lower = torch.quantile(latents, 1 - clamp_percentile / 100.0)
            latents = latents.clamp(min=lower.item(), max=upper.item())

            # === Bruit créatif progressif ===
            if creative_noise > 0.0:
                progressive_noise = creative_noise * (1.0 - t / steps)
                latents += torch.randn_like(latents) * progressive_noise

            # === Smoothing léger pour stabilité ===
            if smoothing_factor > 0.0 and t > 0:
                latents = (1 - smoothing_factor) * latents + smoothing_factor * latents_prev

            latents_prev = latents.clone()

    # === Assurer 4D correct ===
    if latents.ndim != 4:
        B, H, W = latents.shape[0], latents.shape[-2], latents.shape[-1]
        C = min(4, latents.shape[1])
        latents = latents[:, :C, :, :] if latents.shape[1] >= 4 else F.pad(latents, (0,0,0,0,0,4-latents.shape[1]))

    return latents
            # Génération initiale robuste :
            #42	Classique, beaucoup de tests communautaires utilisent ce seed. #1234	Fidèle, stable, souvent utilisé pour des tests de cohérence.
            #5555	Fidélité à l’image initiale (ton choix actuel) #2026	Léger changement dans la texture ou la posture, subtil mais prévisible
            #9876	Variation un peu plus visible, garde la structure globale
def generate_latents_robuste_4D_v1(
        latents,
        unet,
        scheduler,
        pos_embeds=None,
        neg_embeds=None,
        motion_module=None,
        device='cuda',
        dtype=torch.float16,
        guidance_scale=1.0,
        init_image_scale=0.85,
        creative_noise=0.0,
        steps=8,
        seed=None
    ):
    """
    Génération initiale de latents ultra-safe 4D pour AnimateDiff.
    Clamp et correction NaN automatique à chaque step.
    """
    if seed is not None:
        torch.manual_seed(seed)

    latents = latents.to(device=device, dtype=dtype)

    # Échelle initiale si nécessaire
    if init_image_scale != 1.0:
        latents = latents * init_image_scale

    for t in range(steps):
        with torch.no_grad():
            # Dummy UNet step (ou remplacer par vrai step si nécessaire)
            noise = torch.randn_like(latents) * 0.01
            latents = latents + noise

            # Motion module si fourni
            if motion_module is not None:
                latents, _ = motion_module(latents)

            # Clamp strict et correction NaN
            latents = torch.clamp(latents, -1.0, 1.0)
            latents[torch.isnan(latents)] = 0.0
            latents[torch.isinf(latents)] = 0.0

            # Optionnel: petit bruit créatif
            if creative_noise > 0.0:
                latents += torch.randn_like(latents) * creative_noise

    # S'assurer que c'est bien 4D
    if latents.ndim != 4:
        B, C, H, W = latents.shape[0], 4, latents.shape[-2], latents.shape[-1]
        latents = latents[:, :C, :, :] if latents.shape[1] >= 4 else F.pad(latents, (0,0,0,0,0,4-latents.shape[1]))

    return latents
# --------------------- V1 ----------------------------------------
def generate_latents_robuste_v1(
    latents,
    pos_embeds,
    neg_embeds,
    unet,
    scheduler,
    motion_module=None,
    device="cuda",
    dtype=torch.float16,
    guidance_scale=7.5,
    init_image_scale=2.0,
    creative_noise=0.0,
    seed=42
):
    """
    Génère des latents animés à partir d'une séquence initiale.

    latents: [B, 4, T, H, W] (déjà encodés et scalés)
    pos_embeds / neg_embeds: embeddings texte pour guidance
    guidance_scale: poids de guidance classifier-free
    init_image_scale: poids de l'image initiale
    creative_noise: bruit aléatoire ajouté avant chaque étape pour variation
    """
    torch.manual_seed(seed)

    # Vérification dimensions
    if latents.ndim != 5 or latents.shape[1] != 4:
        raise ValueError(f"Latents attendus en [B,4,T,H,W], got {latents.shape}")

    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)

    # Ajouter un peu de bruit créatif initial si demandé
    if creative_noise > 0:
        latents = latents + torch.randn_like(latents) * creative_noise

    # Reshape pour traitement UNet [B*T, C, H, W]
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    init_latents = latents.clone()

    for t_idx, t in enumerate(scheduler.timesteps):

        # Appliquer motion module si présent
        if motion_module is not None:
            latents = motion_module(latents)

        # Classifier-free guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=embeds).sample

        noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Réappliquer init_image_scale pour garder influence image initiale
        latents = latents + init_image_scale * (init_latents - latents)

        # Log pour vérifier valeurs anormales
        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-5:
            print(f"⚠ Step {t_idx}/{len(scheduler.timesteps)}: mean latent {mean_val:.6f}, reset avec petit bruit")
            latents = init_latents + torch.randn_like(init_latents) * 0.01

    # Repasser en [B, C, T, H, W]
    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()

    return latents


# ---------------------------------------------------------
# Diffusion FONCTIONNE PARFAITEMENT
# images_latents: [B,4,T,H,W]
# ---------------------------------------------------------
def generate_latents(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None, device="cuda", dtype=torch.float16, guidance_scale=7.5, init_image_scale=2.0, seed=42):
    """
    latents: [B,4,T,H,W] (déjà encodés et scalés) init_image_scale: poids de l'image initiale
    """
    torch.manual_seed(seed)
    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    # ⚡ on garde une copie des latents initiaux
    init_latents = latents.clone()
    for t in scheduler.timesteps:
        if motion_module is not None:
            latents = motion_module(latents)

        # classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        embeds = torch.cat([neg_embeds, pos_embeds])

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # ⚡ appliquer init_image_scale pour garder l’influence de l’image initiale
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents + init_image_scale * (init_latents - latents)

    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()

    return latents

#---------------------------------------------------------
# -------------------------
# Génération de latents par bloc OK
# def generate_latents(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None, device="cuda", dtype=torch.float16, guidance_scale=7.5, init_image_scale=2.0, seed=42,
# -------------------------
def generate_latents_1(latents, pos_embeds, neg_embeds, unet, scheduler, motion_module=None, device="cuda", dtype=torch.float16, guidance_scale=4.5, init_image_scale=0.85):
    """
    latents: [B, C, F, H, W]
    pos_embeds / neg_embeds: [B, L, D]
    """
    """
    latents: [B,4,T,H,W] (déjà encodés et scalés) init_image_scale: poids de l'image initiale
    """
    torch.manual_seed(42)
    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    # ⚡ on garde une copie des latents initiaux
    init_latents = latents.clone()
    for t in scheduler.timesteps:
        if motion_module is not None:
            latents = motion_module(latents)

        # classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)
        embeds = torch.cat([neg_embeds, pos_embeds])

        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=embeds
            ).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # ⚡ appliquer init_image_scale pour garder l’influence de l’image initiale
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        latents = latents + init_image_scale * (init_latents - latents)

    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()

    return latents

# -------------------------
# Génération de latents tuilés (block_size + overlap)
# -------------------------
def generate_tiled(input_latents, pos_embeds, neg_embeds, unet, scheduler, motion_module,
                   device, dtype, guidance_scale=4.5, init_image_scale=0.85,
                   block_size=128, overlap=16):
    """
    input_latents: [B, C, F, H, W]
    Retourne latents [B, C, F, H, W]
    """
    B, C, F, H, W = input_latents.shape
    output_latents = torch.zeros_like(input_latents)

    h_blocks = math.ceil(H / (block_size - overlap))
    w_blocks = math.ceil(W / (block_size - overlap))

    for hi in range(h_blocks):
        for wi in range(w_blocks):
            h_start = hi * (block_size - overlap)
            w_start = wi * (block_size - overlap)
            h_end = min(h_start + block_size, H)
            w_end = min(w_start + block_size, W)
            h_start = max(h_end - block_size, 0)
            w_start = max(w_end - block_size, 0)

            block = input_latents[:, :, :, h_start:h_end, w_start:w_end]

            # Génération latents sur le bloc
            block_out = generate_latents_2(
                block, pos_embeds, neg_embeds, unet, scheduler, motion_module,
                device, dtype, guidance_scale, init_image_scale
            )

            output_latents[:, :, :, h_start:h_end, w_start:w_end] = block_out

    return output_latents


# ---------------------------------------------------------
# Decode latents → images
# latents: [B,4,T,H,W]
# ---------------------------------------------------------
def decode_latents(latents, vae):

    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)

    frames = []

    with torch.no_grad():
        for t in range(T):
            latent = latents[:, :, t, :, :]

            # 🔥 INVERSE DU SCALE (CORRECTION MAJEURE)
            latent = latent / LATENT_SCALE

            img = vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)

            frames.append(img.float())

    images = torch.stack(frames, dim=2)  # [B,3,T,H,W]
    return images


# ---------------------------------------------------------
# Sauvegarde vidéo
# ---------------------------------------------------------
def create_video_from_latents(latents, vae, output_dir, fps=12):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = decode_latents(latents, vae)

    images = images.squeeze(0)  # [3,T,H,W]
    images = images.permute(1,0,2,3)  # [T,3,H,W]

    for i, img in enumerate(images):
        save_image(img, output_dir / f"frame_{i:04d}.png")

    (
        ffmpeg
        .input(f"{output_dir}/frame_%04d.png", framerate=fps)
        .output(str(output_dir / "output.mp4"),
                vcodec="libx264",
                pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )

    print(f"🎬 Vidéo générée : {output_dir / 'output.mp4'}")


# ---------- NEW FONCTION ----------------------------------
# safe_load_unet, safe_load_vae, safe_load_scheduler, encode_images_to_latents, decode_latents_to_image, load_images, save_frames_as_video

# -------------------------
# Load models safe
# -------------------------

def safe_load_unet(model_path, device, fp16=True):
    folder = os.path.join(model_path, "unet")
    if os.path.exists(folder):
        model = UNet2DConditionModel.from_pretrained(folder)
        if fp16:
            model = model.half()  # réduit la VRAM de moitié
        return model.to(device)
    return None

#def safe_load_unet(model_path, device, fp16=False):
#    model = UNet2DConditionModel.from_pretrained(os.path.join(model_path,"unet"))
#    if fp16: model = model.half()
#    return model.to(device)

def safe_load_vae(model_path, device, fp16=False, offload=False):
    model = AutoencoderKL.from_pretrained(os.path.join(model_path,"vae"))
    model = model.to("cpu" if offload else device)
    if fp16: model = model.half()
    return model

def safe_load_scheduler(model_path):
    return DPMSolverMultistepScheduler.from_pretrained(os.path.join(model_path,"scheduler"))

# --------------------------------------
# Encode / Decode FP16 simple
# --------------------------------------
def encode_images_to_latents_simple(images, vae):
    device = vae.device
    dtype = next(vae.parameters()).dtype  # prend fp16 si le VAE est en FP16
    images = images.to(device=device, dtype=dtype)

    with torch.no_grad():
        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B*F, C, H, W)
            latents_2d = vae.encode(images_2d).latent_dist.sample() * LATENT_SCALE
            latent_shape = latents_2d.shape
            latents = latents_2d.view(B, F, latent_shape[1], latent_shape[2], latent_shape[3])
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
            latents = latents.unsqueeze(2)
    return latents


def decode_latents_to_image_safe(latents, vae):
    dtype = next(vae.parameters()).dtype
    latents = latents.to(vae.device).to(dtype) / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img

# ------------------------------
def encode_images_to_latents_half(images, vae):
    # récupère dtype réel du VAE
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    images = images.to(device=vae_device, dtype=vae_dtype)

    with torch.no_grad():

        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B * F, C, H, W)

            latents_2d = vae.encode(images_2d).latent_dist.sample()
            latents_2d = latents_2d * LATENT_SCALE

            latents = latents_2d.view(
                B, F,
                latents_2d.shape[1],
                latents_2d.shape[2],
                latents_2d.shape[3]
            )

            latents = latents.permute(0, 2, 1, 3, 4).contiguous()

        else:
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * LATENT_SCALE
            latents = latents.unsqueeze(2)

    return latents

def decode_latents_to_image_vae(latents, vae):

    # Récupère device + dtype réel du VAE
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Aligne le dtype sur celui du VAE
    latents = latents.to(device=vae_device, dtype=vae_dtype)

    latents = latents / LATENT_SCALE

    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)

    # On repasse en float32 pour sauvegarde PNG
    return img.float()
# -------------------------
# Encode / Decode
# -------------------------
def encode_images_to_latents_ori(images, vae):
    device = vae.device
    images = images.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B*F, C, H, W)
            latents_2d = vae.encode(images_2d).latent_dist.sample() * LATENT_SCALE
            latent_shape = latents_2d.shape
            latents = latents_2d.view(B, F, latent_shape[1], latent_shape[2], latent_shape[3])
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
            latents = latents.unsqueeze(2)
    return latents

def decode_latents_to_image_ori(latents, vae):
    latents = latents.to(vae.device).float() / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img


# --------------------------------------------------------
# | Mode        | VAE     | Images  | Latents | Résultat  |
# | ----------- | ------- | ------- | ------- | --------  |
# | fp32        | float32 | float32 | float32 | ✅        |
# | fp16        | float16 | float16 | float16 | ✅        |
# | offload CPU | float32 | float32 | float32 | ✅        |
# ------------------------- ci dessous:
# -------------------------
# Encode / Decode corrigé FP16 safe
# -------------------------
def encode_images_to_latents(images, vae):
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype  # on aligne avec le VAE
    images = images.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
    return latents

def decode_latents_to_image(latents, vae):
    # On force latents à avoir le même dtype et device que le VAE
    vae_dtype = next(vae.parameters()).dtype
    vae_device = next(vae.parameters()).device
    latents = latents.to(device=vae_device, dtype=vae_dtype) / LATENT_SCALE

    with torch.no_grad():
        img = vae.decode(latents).sample

    # Normalisation sûre vers 0-1
    img = (img / 2 + 0.5).clamp(0, 1)

    # Si FP16 → convertir en float32 pour torchvision save_image
    if img.dtype == torch.float16:
        img = img.float()

    return img

# NEW
#
#
# ---------------------------------------------------------
# Decode latents to image avec logs et sécurité
# ---------------------------------------------------------
def decode_latents_to_image_2(latents, vae, latent_scale=0.18215):
    """
    latents: [B, C, F, H, W] ou [B, C, 1, H, W] pour frame unique
    vae: VAE pour décodage
    """
    try:
        print(f"🔹 decode_latents_to_image_2 | input shape: {latents.shape}, dtype: {latents.dtype}, device: {latents.device}")

        # Si latents a une dimension de frame singleton, la squeeze
        if latents.shape[2] == 1:
            latents = latents.squeeze(2)
            print(f"🔹 Squeeze frame dimension → shape: {latents.shape}")

        # Assurer dtype et device compatible VAE
        vae_dtype = next(vae.parameters()).dtype
        vae_device = next(vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=vae_dtype) / latent_scale

        # Check NaN avant VAE
        print(f"🔹 Latents before VAE decode | min: {latents.min()}, max: {latents.max()}, dtype: {latents.dtype}")
        if torch.isnan(latents).any():
            print("❌ Warning: NaN detected in latents before VAE decode!")

        with torch.no_grad():
            img = vae.decode(latents).sample

        # Check NaN après décodage
        print(f"🔹 Image after VAE decode | min: {img.min()}, max: {img.max()}, dtype: {img.dtype}")
        if torch.isnan(img).any():
            print("❌ Warning: NaN detected in decoded image!")

        # Normalisation safe vers 0-1
        img = (img / 2 + 0.5).clamp(0, 1)
        print(f"🔹 Image final | min: {img.min()}, max: {img.max()}, dtype: {img.dtype}, shape: {img.shape}")

        # Conversion FP16 -> FP32 si nécessaire
        if img.dtype == torch.float16:
            img = img.float()

        return img

    except Exception as e:
        print(f"❌ Exception in decode_latents_to_image_2: {e}")
        # Retourne une image noire safe si VAE échoue
        B, C, H, W = latents.shape[:4]
        return torch.zeros(B, 3, H*8, W*8, device=latents.device)  # scale approx 8x pour SD VAE
# -------------------------
# Encode / Decode corrigé
# -------------------------
# -------------------------

def decode_latents_to_image_old(latents, vae):
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(device=vae_device, dtype=vae_dtype)
    latents = latents / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
    return img.float()  # on repasse en float32 pour PNG

# -------------------------
# Image utilities
# -------------------------
def load_image_file(path, W, H, device, dtype):
    img = Image.open(path).convert("RGB")
    img = img.resize((W,H), Image.LANCZOS)
    img_tensor = torch.tensor(np.array(img)).permute(2,0,1).to(device=device, dtype=dtype)/127.5 - 1.0
    return img_tensor

# -------------------------
# Image utilities
# -------------------------
def load_images_test(paths, W, H, device, dtype):

    all_tensors = []

    for p in paths:

        if p.lower().endswith(".gif"):

            img = Image.open(p)

            for f in ImageSequence.Iterator(img):

                t = torch.tensor(np.array(f)).float() / 127.5 - 1.0

                if t.ndim == 3:
                    t = t.permute(2,0,1)

                all_tensors.append(t)

            print(f"✅ GIF chargé : {p}")

        else:

            t = load_image_file(p, W, H, device="cpu", dtype=torch.float32)

            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2,0,1)

            all_tensors.append(t)

            print(f"✅ Image chargée : {p}")

    imgs = torch.stack(all_tensors, dim=0)

    print("IMAGE SHAPE:", imgs.shape)
    print("IMAGE MIN/MAX:", imgs.min().item(), imgs.max().item())

    return imgs.to(device=device, dtype=dtype)



def load_images(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        if p.lower().endswith(".gif"):
            img = Image.open(p)
            frames = [torch.tensor(np.array(f)).permute(2,0,1).to(device=device, dtype=dtype)/127.5 - 1.0
                      for f in ImageSequence.Iterator(img)]
            print(f"✅ GIF chargé : {p} avec {len(frames)} frames")
            all_tensors.extend(frames)
        else:
            t = load_image_file(p, W, H, device, dtype)
            print(f"✅ Image chargée : {p}")
            all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)



def load_images_s(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((W,H), Image.LANCZOS)
        t = torch.tensor(np.array(img)).permute(2,0,1).to(device=device,dtype=dtype)/127.5 - 1.0
        all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)

def load_images_all(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        if p.lower().endswith(".gif"):
            img = Image.open(p)
            frames = [torch.tensor(np.array(f)).permute(2,0,1).to(device=device, dtype=dtype)/127.5 - 1.0
                      for f in ImageSequence.Iterator(img)]
            print(f"✅ GIF chargé : {p} avec {len(frames)} frames")
            all_tensors.extend(frames)
        else:
            t = load_image_file(p, W, H, device, dtype)
            print(f"✅ Image chargée : {p}")
            all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)




# -------------------------
# Save video
# -------------------------
def save_frames_as_video_rmtmp(frames, output_path, fps=12):
    temp_dir = Path("temp_frames")
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")
    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)
# -------------------------
# Video utilities
# -------------------------
def save_frames_as_video(frames, output_path, fps=12):
    temp_dir = Path("temp_frames")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")

    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)
# -----------------------------
# -------- MOTION -------------
# -----------------------------
def default_motion_module(latents: torch.Tensor, frame_idx: int = 0, total_frames: int = 1) -> torch.Tensor:
    """
    Motion module par défaut avec effets aléatoires pour Tiny-SD
    - Translation aléatoire subtile
    - Oscillation douce
    - Zoom subtil (ne change pas la taille)
    - Bruit léger sur latents
    """
    # --- Zoom subtil ---
    zoom_factor = random.uniform(0.98, 1.02)  # ±2%
    latents = latents * zoom_factor  # scale des valeurs des latents seulement
    # (⚠️ Ne change pas H/W, safe pour le scheduler)

    B, C, H, W = latents.shape

    # --- Translation aléatoire subtile (pan/tilt) ---
    dx = random.randint(-1, 1)
    dy = random.randint(-1, 1)

    # --- Oscillation douce ---
    osc_amp = 1
    dx_osc = int(osc_amp * math.sin(2 * math.pi * frame_idx / max(total_frames,1)))
    dy_osc = int(osc_amp * math.cos(2 * math.pi * frame_idx / max(total_frames,1)))

    # Pan/tilt safe avec torch.roll
    latents = torch.roll(latents, shifts=(dy+dy_osc, dx+dx_osc), dims=(2,3))

    # --- Noise léger ---
    noise_sigma = 0.003
    latents = latents + torch.randn_like(latents) * noise_sigma

    return latents


def default_motion_module_test(latents: torch.Tensor, frame_idx: int = 0, total_frames: int = 1) -> torch.Tensor:
    # Zoom subtil
    zoom_factor = 0.01
    factor = 1.0 + zoom_factor * frame_idx / max(total_frames,1)
    B, C, H, W = latents.shape
    latents = F.interpolate(latents, scale_factor=factor, mode='bilinear', align_corners=False)
    latents = latents[:, :, :H, :W]  # recadrer si nécessaire

    # Oscillation subtile
    dx = int(1 * math.sin(2 * math.pi * frame_idx / max(total_frames,1)))
    dy = int(1 * math.cos(2 * math.pi * frame_idx / max(total_frames,1)))
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_x = torch.clamp(grid_x - dx, 0, W-1)
    grid_y = torch.clamp(grid_y - dy, 0, H-1)
    latents = latents[:, :, grid_y, grid_x]

    return latents

# -------------------------
# Model loaders
# -------------------------


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# -------------------------
# Upscale vidéo avec Real-ESRGAN
# -------------------------
def upscale_video_with_realesrgan(video_path, output_path, device="cuda", scale=2, fps=12):
    """
    Upscale une vidéo avec Real-ESRGAN (torch)
    - video_path : chemin vidéo d'entrée
    - output_path : chemin vidéo sortie
    - scale : facteur d'upscale (2, 4, etc.)
    - fps : framerate de sortie
    """
    try:
        from realesrgan import RealESRGAN
    except ImportError:
        print("❌ Module Real-ESRGAN non installé. pip install realesrgan")
        return

    import tempfile
    from PIL import Image
    import ffmpeg

    temp_dir = tempfile.mkdtemp()
    temp_out_dir = tempfile.mkdtemp()

    # 1️⃣ Extraire frames
    (
        ffmpeg
        .input(str(video_path))
        .output(f"{temp_dir}/frame_%05d.png")
        .overwrite_output()
        .run(quiet=True)
    )

    # 2️⃣ Charger modèle
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f"RealESRGAN_x{scale}.pth", download=True)  # si pas déjà présent

    # 3️⃣ Upscale frame par frame
    frame_paths = sorted(Path(temp_dir).glob("frame_*.png"))
    for idx, fpath in enumerate(frame_paths):
        img = Image.open(fpath).convert("RGB")
        upscaled = model.predict(img)
        upscaled.save(Path(temp_out_dir) / f"frame_{idx:05d}.png")

    # 4️⃣ Recomposer la vidéo
    (
        ffmpeg
        .input(f"{temp_out_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p", crf=18)
        .overwrite_output()
        .run(quiet=True)
    )

    # 5️⃣ Cleanup
    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_out_dir)
    print(f"✅ Vidéo finale Real-ESRGAN x{scale} générée : {output_path}")

# -------------------------
# Upscale cinématique (sans torch)
# -------------------------
def upscale_video_cinematic_smooth(input_video_path, output_video_path, scale=4, fps=12, interp_frames=1):
    """
    Upscale x4 avec interpolation de frames pour smooth cinematic style.
    - input_video_path : vidéo 128x128
    - output_video_path : vidéo finale upscalée
    - scale : facteur d'upscale (par défaut 4)
    - interp_frames : nombre de frames interpolées entre deux frames (smooth)
    """
    input_video_path = Path(input_video_path)
    temp_dir = input_video_path.parent / "temp_frames_smooth"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    # Extraction frames
    (
        ffmpeg
        .input(str(input_video_path))
        .output(str(temp_dir / "frame_%05d.png"))
        .overwrite_output()
        .run(quiet=True)
    )

    # Charger frames existantes
    frame_paths = sorted(temp_dir.glob("frame_*.png"))
    upscaled_frames = []

    for idx in range(len(frame_paths)):
        frame = Image.open(frame_paths[idx])
        W, H = frame.size
        frame_up = frame.resize((W*scale, H*scale), Image.LANCZOS)

        upscaled_frames.append(frame_up)

        # Interpolation linéaire entre frames pour smooth
        if idx < len(frame_paths) - 1 and interp_frames > 0:
            next_frame = Image.open(frame_paths[idx+1]).resize((W*scale, H*scale), Image.LANCZOS)
            for t in range(1, interp_frames+1):
                alpha = t / (interp_frames+1)
                interp_frame = Image.blend(frame_up, next_frame, alpha)
                upscaled_frames.append(interp_frame)

    # Ré-encodage vidéo
    temp_up_dir = input_video_path.parent / "temp_frames_upscaled_smooth"
    if temp_up_dir.exists():
        shutil.rmtree(temp_up_dir)
    temp_up_dir.mkdir()

    for i, f in enumerate(upscaled_frames):
        f.save(temp_up_dir / f"frame_{i:05d}.png")

    (
        ffmpeg
        .input(str(temp_up_dir / "frame_%05d.png"), framerate=fps*(interp_frames+1))
        .output(str(output_video_path), vcodec="libx264", pix_fmt="yuv420p", crf=18)
        .overwrite_output()
        .run(quiet=True)
    )

    shutil.rmtree(temp_dir)
    shutil.rmtree(temp_up_dir)
    print(f"✅ Vidéo finale x{scale} smooth cinematic générée : {output_video_path}")


#---------------------------------------------------------------------------------------
# -------------- Remove watermark fonction ---------------------------------------------
#---------------------------------------------------------------------------------------

def remove_watermark_auto_blur_v1(
    frame_pil,
    target_hex_list,
    tolerance=40,
    threshold=0.45,
    candidate_zones=None,
    blur_radius=10,
    feather_radius=10,
    show_mask=False
):
    """
    Version ULTRA INVISIBLE PRO
    - Détection couleur
    - Masque progressif (feather)
    - Fusion douce
    """

    import numpy as np
    import cv2
    from PIL import Image, ImageFilter

    img_np = np.array(frame_pil).astype(np.int16)
    H, W, _ = img_np.shape

    # Convertir hex en RGB
    target_colors = np.array(
        [[int(h[i:i+2], 16) for i in (1, 3, 5)] for h in target_hex_list],
        dtype=np.int16
    )

    if candidate_zones is None:
        candidate_zones = [(0, 0, W, H)]

    for idx, (x, y, w, h) in enumerate(candidate_zones):

        patch = img_np[y:y+h, x:x+w]
        mask_total = np.zeros((h, w), dtype=np.uint8)

        # Détection couleur
        for color in target_colors:
            dist = np.linalg.norm(patch - color, axis=2)
            mask_total += (dist <= tolerance).astype(np.uint8)

        ratio = mask_total.sum() / (w * h)

        if ratio >= threshold:

            # Masque binaire
            mask_binary = (mask_total > 0).astype(np.uint8) * 255

            # Dilatation légère pour couvrir bordures
            kernel = np.ones((3, 3), np.uint8)
            mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)

            # Feather (adoucissement progressif)
            mask_soft = cv2.GaussianBlur(mask_binary, (0, 0), feather_radius)

            # Normaliser masque 0..1
            mask_soft = mask_soft.astype(np.float32) / 255.0
            mask_soft = np.expand_dims(mask_soft, axis=2)

            # Extraire région
            region = frame_pil.crop((x, y, x+w, y+h))
            region_np = np.array(region).astype(np.float32)

            # Blur léger naturel
            region_blur = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            region_blur_np = np.array(region_blur).astype(np.float32)

            # Blend ultra doux
            blended = region_np * (1 - mask_soft) + region_blur_np * mask_soft
            blended = blended.astype(np.uint8)

            # Replacer dans image
            frame_pil.paste(Image.fromarray(blended), (x, y))

    return frame_pil

def remove_watermark_auto_blur_simple(frame_pil, target_hex_list, tolerance=15,
                               threshold=0.5, candidate_zones=None,
                               blur_radius=20, show_mask=False):
    """
    Détecte automatiquement les zones du watermark et les floute.

    - frame_pil : image PIL
    - target_hex_list : liste des couleurs du watermark en hex
    - tolerance : tolérance couleur
    - threshold : proportion minimale de pixels correspondants pour appliquer
    - candidate_zones : zones probables [(x,y,w,h), ...], sinon toute l'image
    - blur_radius : rayon du flou gaussien
    - show_mask : bool, afficher le masque de détection pour debug
    """
    import numpy as np
    import cv2
    from PIL import Image, ImageFilter

    img_np = np.array(frame_pil).astype(np.int16)
    H, W, _ = img_np.shape

    # Convertir hex en RGB
    target_colors = np.array([[int(h[i:i+2],16) for i in (1,3,5)] for h in target_hex_list], dtype=np.int16)

    if candidate_zones is None:
        candidate_zones = [(0, 0, W, H)]

    for idx, (x, y, w, h) in enumerate(candidate_zones):
        patch = img_np[y:y+h, x:x+w]
        mask_total = np.zeros((h, w), dtype=np.uint8)

        # Détection des pixels correspondant aux couleurs du watermark
        for c_idx, color in enumerate(target_colors):
            dist = np.linalg.norm(patch - color, axis=2)
            mask_total += (dist <= tolerance).astype(np.uint8)
            if show_mask:
                print(f"[DEBUG] Zone {idx}, Couleur {c_idx} match pixels: {(dist <= tolerance).sum()}")

        ratio = mask_total.sum() / (w*h)
        if show_mask:
            print(f"[DEBUG] Zone {idx} ratio total: {ratio:.3f} (seuil={threshold})")
            import matplotlib.pyplot as plt
            plt.imshow(mask_total, cmap='gray')
            plt.title(f"Zone {idx} mask")
            plt.show()

        if ratio >= threshold:
            # Créer un masque binaire pour OpenCV
            mask_uint8 = (mask_total > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                left = x + bx
                top = y + by
                right = left + bw
                bottom = top + bh

                # Extraire la région et flouter
                region = frame_pil.crop((left, top, right, bottom))
                region_blur = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                frame_pil.paste(region_blur, (left, top))

            if show_mask:
                print(f"[DEBUG] Zone {idx} floutée.")

    return frame_pil

def remove_watermark_white(frame_pil, bbox, padding=2):
    """
    Remplace le watermark par un rectangle blanc.
    bbox = (x, y, w, h)
    """
    if bbox is None:
        return frame_pil

    x, y, w, h = bbox

    # Ajouter un petit padding pour couvrir les bords
    x = max(0, x - padding)
    y = max(0, y - padding)
    w += padding * 2
    h += padding * 2

    img_w, img_h = frame_pil.size
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    # Créer rectangle blanc
    from PIL import ImageDraw
    draw = ImageDraw.Draw(frame_pil)
    draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))

    return frame_pil


def decode_latents_ultrasafe_blockwise_test(
    latents, vae,
    block_size=32, overlap=16,
    gamma=1.0, brightness=1.0,
    contrast=1.0, saturation=1.0,
    device="cuda", frame_counter=0, output_dir=Path("."),
    epsilon=1e-5  # valeur minimale pour éviter patches nuls
):
    """
    Décodage par blocs ultra-safe des latents en image PIL avec correction auto
    et debug min/max pour chaque patch et image finale.

    - latents: [B, 4, H, W] sur CPU ou GPU
    - vae: VAE déjà chargé (fp16 ou fp32)
    - block_size: taille des patches
    - overlap: chevauchement
    - gamma/brightness/contrast/saturation: correction finale
    - device: device du VAE (cuda ou cpu)
    - epsilon: valeur minimale pour patches nuls
    """
    vae_dtype = next(vae.parameters()).dtype
    B, C, H, W = latents.shape
    output_rgb = torch.zeros(B, 3, H * 8, W * 8, device=device, dtype=torch.float32)

    y_steps = list(range(0, H, block_size - overlap))
    x_steps = list(range(0, W, block_size - overlap))

    for y0 in y_steps:
        y1 = min(y0 + block_size, H)
        for x0 in x_steps:
            x1 = min(x0 + block_size, W)
            patch = latents[:, :, y0:y1, x0:x1].to(device=device, dtype=vae_dtype)

            # Debug avant VAE
            print(f"[DEBUG] patch avant VAE ({y0},{x0}): shape={patch.shape}, "
                  f"dtype={patch.dtype}, min={patch.min():.6f}, max={patch.max():.6f}")

            # ✅ Correction NaN / Inf et epsilon minimal
            patch = torch.nan_to_num(patch, nan=0.0, posinf=5.0, neginf=-5.0)
            if torch.all(patch == 0):
                patch += epsilon

            # log patch
            patch_idx = f"{y0}_{x0}"
            log_patch_stats(frame_idx=frame_counter, patch_idx=patch_idx, patch=patch, csv_path=output_dir / "patch_stats.csv")

            # Decode
            with torch.no_grad():
                patch_decoded = vae.decode(patch).sample  # [B, 3, h*8, w*8]
                patch_decoded = patch_decoded.to(torch.float32)

            # ✅ Recentrage automatique pour éviter frames sombres
            min_val = patch_decoded.min()
            max_val = patch_decoded.max()
            if max_val - min_val > 1e-6:
                patch_decoded = (patch_decoded - min_val) / (max_val - min_val)

            # Debug après VAE
            log_patch_stats(frame_idx=frame_counter, patch_idx=patch_idx+"_decoded", patch=patch_decoded, csv_path=output_dir / "patch_stats.csv")

            h_start, h_end = y0 * 8, y1 * 8
            w_start, w_end = x0 * 8, x1 * 8
            output_rgb[:, :, h_start:h_end, w_start:w_end] = patch_decoded

    # Debug avant clamp
    print(f"[DEBUG] output_rgb final avant clamp: shape={output_rgb.shape}, "
          f"min={output_rgb.min():.6f}, max={output_rgb.max():.6f}")

    output_rgb = output_rgb.clamp(0.0, 1.0)

    # Correction gamma / contraste / luminosité / saturation
    frame_pil_list = []
    for i in range(B):
        img = F.to_pil_image(output_rgb[i])
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        img = img.point(lambda x: (x / 255) ** (1 / gamma) * 255)
        frame_pil_list.append(img)

        # Debug post-correction
        pil_tensor = F.to_tensor(img)
        print(f"[DEBUG] frame {i} après PIL correction: shape={pil_tensor.shape}, "
              f"min={pil_tensor.min():.6f}, max={pil_tensor.max():.6f}")

    return frame_pil_list[0] if B == 1 else frame_pil_list



def encode_images_to_latents_safe(images, vae, device="cuda", epsilon=1e-5):
    """
    Encode des images en latents sûrs pour UNet.
    Retourne toujours un tensor [B, 4, H_latent, W_latent], dtype=vae.dtype.

    - epsilon : valeur minimale ajoutée aux latents nuls pour éviter frames noires
    """
    images_t = images.to(device=device, dtype=torch.float32)
    original_dtype = next(vae.parameters()).dtype

    vae = vae.to(device=device, dtype=torch.float32)  # safe pour l'encodage

    with torch.no_grad():
        latents = vae.encode(images_t).latent_dist.sample()

    print(f"[DEBUG encode] latents min/max après sample: {latents.min().item():.6f}/{latents.max().item():.6f}")

    # Scaling
    latents = latents * LATENT_SCALE
    print(f"[DEBUG encode] latents min/max après LATENT_SCALE: {latents.min().item():.6f}/{latents.max().item():.6f}")

    # Clamp NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # ✅ Remplir les latents entièrement nuls
    if torch.all(latents == 0):
        latents += epsilon

    # Normalisation pour éviter overflow
    max_abs = latents.abs().max()
    if max_abs > 0:
        latents = latents / max_abs

    # Conversion en dtype final du VAE
    latents = latents.to(original_dtype)

    # ---------------- FORCE 4 CANAUX ----------------
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents.repeat(1, 1, 4, 1, 1)

    print(f"[DEBUG encode] shape finale latents: {latents.shape}, min/max: {latents.min().item():.6f}/{latents.max().item():.6f}")
    return latents


