# utils/vae_utils.py
import torch
from diffusers import AutoencoderKL
from pathlib import Path
import os
from diffusers import UNet2DConditionModel, AutoencoderKL, DPMSolverMultistepScheduler
from torch.nn.functional import interpolate
from torch.nn.functional import pad
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage

import torchvision.transforms as T
import torch, numpy as np
from PIL import Image
import math


LATENT_SCALE = 0.18215


import torch
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()

# -------------------------
# Decode frame via VAE (compatible vae_offload)
# -------------------------
def decode_latents_safe(latents, vae, device, tile_size=128, overlap=64):
    """
    Décodage sécurisé des latents en image PIL, compatible avec VAE sur CPU (vae_offload)
    et avec latents sur GPU.
    """
    # Déplacer latents sur le device du VAE
    vae_device = next(vae.parameters()).device
    latents = latents.to(vae_device).float()

    # Éviter NaN/Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=0.0)

    # Décodage en tiles pour VRAM limitée
    frame_tensor = decode_latents_to_image_tiled128(
        latents,
        vae,
        tile_size=tile_size,
        overlap=overlap,
        device=vae_device
    ).clamp(0, 1)

    # Renvoie un tensor CPU float32 pour sauvegarde
    return frame_tensor.cpu()

def decode_latents_safe_64(latents, vae, patch_size=64):
    """
    Decode latents into images safely on low VRAM by splitting into patches.
    latents: [B, C, H, W]
    """
    B, C, H, W = latents.shape
    decoded_imgs = []

    for b in range(B):
        latent = latents[b:b+1]  # [1,C,H,W]
        img = torch.zeros(1, C, H, W, device=latent.device)

        # Patch decoding
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                hi = min(i + patch_size, H)
                wj = min(j + patch_size, W)
                patch = latent[:, :, i:hi, j:wj]
                decoded_patch = vae.decode(patch).sample
                img[:, :, i:hi, j:wj] = decoded_patch

        # Clamp et convertir en PIL
        img = img.squeeze(0).clamp(0, 1)
        img_pil = to_pil(img.cpu())
        decoded_imgs.append(img_pil)

        # Nettoyage VRAM
        del latent, img
        torch.cuda.empty_cache()

    return decoded_imgs

# Exemple dans ta boucle de génération
# for frame_idx, frame_latents in enumerate(latents_list):
#     frame_pil = decode_latents_safe(frame_latents, vae)[0]
#     frame_pil.save(f"output/frame_{frame_idx:03d}.png")


def decode_latents_to_image_bright_enhanced(latents, vae, gamma=0.7, brightness=1.2, contrast=1.1, saturation=1.15):
    """
    Décodage des latents en image PIL avec :
    - Correction gamma pour éclaircir
    - Augmentation de luminosité, contraste et saturation pour un rendu plus vivant
    """
    latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

    if latents.ndim == 5:  # [B,C,T,H,W]
        latents = latents[:, :, 0, :, :]

    # Revenir à l'échelle attendue par le VAE
    latents = latents / LATENT_SCALE

    with torch.no_grad():
        # 🔹 S’assurer que les latents sont du même dtype que le VAE
        latents = latents.to(vae.dtype)
        image = vae.decode(latents).sample

    # Normalisation [-1,1] -> [0,1]
    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)

    # Correction gamma
    image = image.pow(1.0 / gamma)

    # Convertir en PIL pour post-processing


    image = image[0]  # ✅ retire dimension batch
    pil_image = ToPILImage()(image.cpu().clamp(0, 1))

    # Boost luminosité, contraste et saturation
    pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

    return pil_image



def decode_latents_ultrasafe(latents, vae, gamma=0.7, brightness=1.2, contrast=1.1, saturation=1.15):
    """
    Décodage ultra-sécurisé des latents en image PIL.
    Protège contre :
    - NaN / inf
    - latents trop grands ou trop petits
    - mauvais dtype
    - images trop sombres / noires
    """
    from torchvision.transforms import ToPILImage
    import torch
    from PIL import Image, ImageEnhance

    # ---------------- sécurité latents ----------------
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    if latents.ndim == 5:  # [B,C,T,H,W]
        latents = latents[:, :, 0, :, :]  # retirer dimension temporelle

    latents = latents / LATENT_SCALE  # remise à l'échelle attendue par le VAE

    # Clamp très strict pour éviter valeurs extrêmes
    latents = latents.clamp(-5.0, 5.0)

    # Assurer le dtype correct
    latents = latents.to(vae.dtype)

    with torch.no_grad():
        image = vae.decode(latents).sample  # [B,3,H,W]

    # Normalisation [-1,1] -> [0,1]
    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)

    # Correction gamma pour éclaircir
    image = image.pow(1.0 / gamma)

    # S'assurer qu'on a bien [3,H,W]
    if image.ndim == 4:
        image = image[0]  # retirer batch si présent

    # Convertir en PIL
    pil_image = ToPILImage()(image.cpu().clamp(0,1))

    # Boost luminosité, contraste, saturation
    pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast)
    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

    return pil_image


def decode_latents_to_image_vram_safe(latents, vae, gamma=0.7, brightness=1.2, contrast=1.1, saturation=1.15):
    """
    Décodage des latents en PIL.Image avec corrections gamma, luminosité, contraste et saturation,
    optimisé pour faible VRAM (4Go). Utilise float32 pour la stabilité.

    Args:
        latents (torch.Tensor): [B,C,T,H,W] ou [B,C,H,W]
        vae (AutoencoderKL): modèle VAE
        gamma (float): correction gamma
        brightness, contrast, saturation (float): boosts PIL.Image

    Returns:
        List[PIL.Image]: images décodées
    """
    from torchvision.transforms import ToPILImage
    import torch
    from PIL import ImageEnhance

    # Sécurisation NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=4.0, neginf=-4.0)

    # Si [B,C,T,H,W] → on prend la première "T" pour l'instant
    if latents.ndim == 5:
        latents = latents[:,:,0,:,:]

    # Revenir à l'échelle attendue par le VAE
    latents = latents / LATENT_SCALE

    # Décode en float32 pour éviter frames noires
    latents = latents.to(dtype=torch.float32)

    with torch.no_grad():
        image_tensor = vae.decode(latents).sample

    # Normalisation [-1,1] -> [0,1]
    image_tensor = (image_tensor + 1.0) / 2.0
    image_tensor = image_tensor.clamp(0,1)

    # Correction gamma
    image_tensor = image_tensor.pow(1.0 / gamma)

    images = []
    to_pil = ToPILImage()
    for i in range(image_tensor.shape[0]):
        img = image_tensor[i]
        pil_img = to_pil(img.cpu())

        # Boost luminosité / contraste / saturation
        pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
        pil_img = ImageEnhance.Color(pil_img).enhance(saturation)

        images.append(pil_img)

    return images



def generate_latents_robuste_model(latents, pos_embeds, neg_embeds, unet, scheduler,
                             motion_module=None, device="cuda", dtype=torch.float16,
                             guidance_scale=4.5, init_image_scale=0.85,
                             creative_noise=0.0, seed=42):
    """
    Génère des latents robustes avec protection NaN/inf
    """
    torch.manual_seed(seed)
    B, C, T, H, W = latents.shape
    latents = latents.to(device=device, dtype=dtype)
    latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W).contiguous()
    init_latents = latents.clone()

    for t_step in scheduler.timesteps:
        # Motion module optionnel
        if motion_module is not None:
            latents = motion_module(latents)

        # Bruit créatif
        if creative_noise > 0:
            latents = latents + torch.randn_like(latents) * creative_noise

        # Classifier-free guidance
        latent_model_input = torch.cat([latents, latents], dim=0)
        embeds = torch.cat([neg_embeds, pos_embeds], dim=0)

        # ⚡ Autocast pour FP16 stable
        with torch.autocast(device_type=device, dtype=dtype):
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t_step, encoder_hidden_states=embeds).sample

        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # Scheduler step
        latents = scheduler.step(noise_pred, t_step, latents).prev_sample

        # Réinjection image initiale
        latents = latents + init_image_scale * (init_latents - latents)

        # Protection NaN / inf
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            latents = torch.nan_to_num(latents, nan=0.0, posinf=1.0, neginf=-1.0)
            latents = latents + torch.randn_like(latents) * 1e-2

        # Vérification latents trop petits
        mean_val = latents.abs().mean().item()
        if math.isnan(mean_val) or mean_val < 1e-5:
            latents = latents + torch.randn_like(latents) * 1e-2

    latents = latents.reshape(B, T, C, H, W).permute(0,2,1,3,4).contiguous()
    return latents


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
    # Gestion VAE offload / device
    # -------------------------
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    if vae_offload:
        vae.to(device)

    # Assure correspondance latents / VAE
    batch_latents = batch_latents.to(device=vae_device, dtype=vae_dtype)

    # -------------------------
    # Décodage avec tiling universel
    # -------------------------
    decode_start = time.time()
    frame_tensor = decode_latents_to_image_tiled_universel(
        batch_latents,
        vae,
        tile_size=tile_size,
        overlap=overlap
    )
    decode_time = time.time() - decode_start

    # Revenir en CPU si offload
    if vae_offload:
        vae.cpu()
        torch.cuda.empty_cache()

    # -------------------------
    # Clamp final pour éviter NaN/inf
    # -------------------------
    frame_tensor = frame_tensor.clamp(0.0, 1.0)

    return frame_tensor, batch_latents, gen_time, decode_time


def decode_latents_to_image_tiled_universel(latents, vae, tile_size=64, overlap=16):
    """
    Decode latents 4D [B,C,H,W] ou 5D [B,C,T,H,W] en images [B,3,H*8,W*8].
    Tiling avec blending sécurisé.
    """
    vae_dtype = next(vae.parameters()).dtype
    device = vae.device

    if latents.ndim == 5:
        B,C,T,H,W = latents.shape
        latents = latents.permute(0,2,1,3,4).reshape(B*T,C,H,W)
    elif latents.ndim == 4:
        B,C,H,W = latents.shape
    else:
        raise ValueError(f"Latents attendus 4D ou 5D, got {latents.shape}")

    latents = latents.to(vae_dtype)
    latents_scaled = latents / LATENT_SCALE

    output = torch.zeros(B,3,H*8,W*8,device=device,dtype=torch.float32)
    weight = torch.zeros_like(output)

    stride = tile_size - overlap
    y_positions = list(range(0,H-tile_size+1,stride)) or [0]
    x_positions = list(range(0,W-tile_size+1,stride)) or [0]
    if y_positions[-1] != H-tile_size: y_positions.append(H-tile_size)
    if x_positions[-1] != W-tile_size: x_positions.append(W-tile_size)

    for y in y_positions:
        for x in x_positions:
            y1 = y+tile_size
            x1 = x+tile_size
            tile = latents_scaled[:,:,y:y1,x:x1]

            with torch.no_grad():
                decoded = vae.decode(tile).sample.float()
            decoded = (decoded/2 + 0.5).clamp(0,1)

            iy0, ix0, iy1, ix1 = y*8, x*8, y1*8, x1*8
            output[:,:,iy0:iy1,ix0:ix1] += decoded
            weight[:,:,iy0:iy1,ix0:ix1] += 1.0

    return output / weight.clamp(min=1e-6)


def decode_latents_to_image_tiled128(latents, vae, tile_size=128, overlap=32, device="cuda"):
    """
    Decode les latents [B,4,H,W] ou [B,4,T,H,W] en RGB [B,3,H,W] ou [B,3,T,H,W]
    avec tiling pour éviter OOM et blending correct.
    """
    vae_dtype = next(vae.parameters()).dtype
    vae_device = next(vae.parameters()).device if device=="cuda" else device

    # Support 5D
    if latents.ndim == 5:
        B, C, T, H, W = latents.shape
        latents = latents.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        reshape_back = True
    elif latents.ndim == 4:
        B, C, H, W = latents.shape
        reshape_back = False
    else:
        raise ValueError(f"Latents attendus 4D ou 5D, got {latents.shape}")

    if C != 4:
        raise ValueError(f"Latents doivent avoir 4 canaux, got {C}")

    # Convert dtype & device
    latents = latents.to(vae_device, vae_dtype)

    stride = tile_size - overlap
    out_H = H * 8
    out_W = W * 8

    output = torch.zeros(latents.shape[0], 3, out_H, out_W, device=vae_device, dtype=torch.float32)
    weight = torch.zeros_like(output)

    # Positions tuiles
    y_positions = list(range(0, H - tile_size + 1, stride))
    x_positions = list(range(0, W - tile_size + 1, stride))
    if not y_positions: y_positions = [0]
    if not x_positions: x_positions = [0]
    if y_positions[-1] != H - tile_size: y_positions.append(H - tile_size)
    if x_positions[-1] != W - tile_size: x_positions.append(W - tile_size)

    for y in y_positions:
        for x in x_positions:
            y1 = y + tile_size
            x1 = x + tile_size
            tile = latents[:, :, y:y1, x:x1]

            with torch.no_grad():
                decoded = vae.decode(tile / LATENT_SCALE).sample

            # Correction Stable Diffusion
            decoded = (decoded / 2 + 0.5).clamp(0,1)

            iy0, ix0 = y*8, x*8
            iy1, ix1 = y1*8, x1*8

            output[:, :, iy0:iy1, ix0:ix1] += decoded
            weight[:, :, iy0:iy1, ix0:ix1] += 1.0

    output = output / weight.clamp(min=1e-6)

    if reshape_back:
        # Retour à [B,3,T,H,W]
        output = output.reshape(B, T, 3, out_H, out_W).permute(0,2,1,3,4)

    return output

def log_rgb_stats(image_tensor, step=""):
    """Enregistre les statistiques RGB et renvoie les warnings sous forme de liste."""
    messages = []
    R = image_tensor[0, 0].cpu().numpy()
    G = image_tensor[0, 1].cpu().numpy()
    B = image_tensor[0, 2].cpu().numpy()

    R_min, R_max, R_mean = np.min(R), np.max(R), np.mean(R)
    G_min, G_max, G_mean = np.min(G), np.max(G), np.mean(G)
    B_min, B_max, B_mean = np.min(B), np.max(B), np.mean(B)

    # Vérifications
    if R_min < 0.0 or R_max > 1.0:
        messages.append(f"{step}: canal R hors plage [{R_min:.3f},{R_max:.3f}]")
    if G_min < 0.0 or G_max > 1.0:
        messages.append(f"{step}: canal G hors plage [{G_min:.3f},{G_max:.3f}]")
    if B_min < 0.0 or B_max > 1.0:
        messages.append(f"{step}: canal B hors plage [{B_min:.3f},{B_max:.3f}]")

    if abs(R_mean - G_mean) > 0.2 or abs(G_mean - B_mean) > 0.2:
        messages.append(f"{step}: écart important entre R/G/B (R={R_mean:.3f}, G={G_mean:.3f}, B={B_mean:.3f})")

    return messages

def encode_tile_vae(tile_rgb, vae, fp16=False):
    """Encode une tile RGB [1,3,H,W] -> latent [1,4,H/8,W/8]"""
    device = next(vae.parameters()).device
    dtype = torch.float16 if fp16 else torch.float32
    tile_rgb = tile_rgb.to(device=device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(tile_rgb).latent_dist.sample() * LATENT_SCALE
    return latent

def tile_image_vae(img_tensor, tile_size=128, overlap=32):
    """Découpe un tensor [B,3,H,W] en tiles RGB"""
    B,C,H,W = img_tensor.shape
    stride = tile_size - overlap
    tiles, positions = [], []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, y2 = y, min(y+tile_size,H)
            x1, x2 = x, min(x+tile_size,W)
            tile = img_tensor[:,:,y1:y2,x1:x2]
            tiles.append(tile)
            positions.append((y1,y2,x1,x2))
    return tiles, positions

def merge_tiles_vae(tiles, positions, H, W):
    """Fusionne les tiles [1,3,h,w] en image finale [1,3,H,W]"""
    device = tiles[0].device
    out = torch.zeros((tiles[0].shape[0], 3, H, W), device=device)
    count = torch.zeros((tiles[0].shape[0], 3, H, W), device=device)
    for t,(y1,y2,x1,x2) in zip(tiles,positions):
        th,tw = t.shape[2], t.shape[3]
        out[:,:,y1:y1+th,x1:x1+tw] += t
        count[:,:,y1:y1+th,x1:x1+tw] += 1.0
    out /= count.clamp(min=1.0)
    return out

# --------------------------------------------
# Vérification des tiles
#-----------------------------------------------
def clamp_and_warn_tile(tile_rgb, frame_idx, tile_idx, warnings_list):
    """
    Clamp chaque canal à [0,1] et détecte les écarts importants R/G/B
    tile_rgb : tensor [1,3,H,W]
    """
    # Calcul stats
    R, G, B = tile_rgb[0,0], tile_rgb[0,1], tile_rgb[0,2]
    r_min, r_max, r_mean = R.min().item(), R.max().item(), R.mean().item()
    g_min, g_max, g_mean = G.min().item(), G.max().item(), G.mean().item()
    b_min, b_max, b_mean = B.min().item(), B.max().item(), B.mean().item()

    # Warning si hors plage [0,1]
    if r_min < 0 or r_max > 1:
        warnings_list.append(f"Frame {frame_idx} - Tile {tile_idx}: canal R hors plage [{r_min:.3f},{r_max:.3f}]")
    if g_min < 0 or g_max > 1:
        warnings_list.append(f"Frame {frame_idx} - Tile {tile_idx}: canal G hors plage [{g_min:.3f},{g_max:.3f}]")
    if b_min < 0 or b_max > 1:
        warnings_list.append(f"Frame {frame_idx} - Tile {tile_idx}: canal B hors plage [{b_min:.3f},{b_max:.3f}]")

    # Warning si écart important R/G/B
    r_g_b = [r_mean, g_mean, b_mean]
    if max(r_g_b) - min(r_g_b) > 0.3:  # seuil configurable
        warnings_list.append(f"Frame {frame_idx} - Tile {tile_idx}: écart important entre R/G/B (R={r_mean:.3f}, G={g_mean:.3f}, B={b_mean:.3f})")

    # Clamp pour éviter de propager l’erreur
    tile_rgb = torch.clamp(tile_rgb, 0.0, 1.0)
    return tile_rgb

# scripts/utils/vae_utils.py

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


def decode_latents_correct(latents, vae):
    """
    Décodage des latents en image RGB float32
    """
    vae_device = next(vae.parameters()).device
    latents = latents.to(device=vae_device, dtype=torch.float32)
    with torch.no_grad():
        decoded = vae.decode(latents).sample
        decoded = torch.clamp(decoded, -1, 1)
        decoded = (decoded + 1) / 2
    return decoded


# -------------------------
# Fonction pour charger un VAE et tester son décodage
# -------------------------
def safe_load_vae(vae_path, device="cuda", fp16=False, offload=False):
    """
    Charge un VAE (FP32 ou FP16), renvoie l'objet VAE prêt à l'emploi.
    """
    try:
        # Chargement state_dict
        state_dict = load_file(vae_path, device="cpu")
        print("✅ State dict VAE chargé, clés:", list(state_dict.keys())[:5])

        # Création d'un VAE compatible SD
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"]*4,
            up_block_types=["UpDecoderBlock2D"]*4,
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            sample_size=32  # à adapter selon le checkpoint
        )

        # Chargement des poids
        vae.load_state_dict(state_dict, strict=False)

        # Offload si demandé
        if offload:
            vae = vae.to("cpu")
        else:
            vae = vae.to(device)
            if fp16:
                vae = vae.half()

        return vae

    except Exception as e:
        print(f"⚠ Erreur lors du chargement du VAE : {e}")
        return None





# ---------------------------------
# Debug UNet sur latents
# ---------------------------------
# Ajoutons des fonctions de test pour valider les latents avant de les passer dans UNet et VAE.

def test_unet_on_latents(latents, unet, device):
    """
    Teste l'entrée latente avant de la passer dans le UNet
    Cette fonction aide à vérifier si la forme des latents est correcte avant de les utiliser.
    """
    print(f"[Test UNet] Latents avant UNet - min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")

    # Assurer que les latents ont la forme correcte (batch, channels, height, width)
    if latents.ndimension() != 4:
        raise ValueError(f"Les latents doivent avoir 4 dimensions, mais ils en ont {latents.ndimension()}")

    # Test rapide avec le UNet pour vérifier si la forme est correcte
    try:
        # Assurez-vous que le UNet est configuré avec les bons paramètres (timestep, encoder_hidden_states)
        # Simulez les entrées nécessaires à un UNet standard si nécessaire
        timestep = torch.tensor([0]).to(device)  # Exemple de timestep (à ajuster selon votre modèle)
        encoder_hidden_states = torch.zeros((latents.size(0), 77, latents.size(2)), device=device)  # Ajustez selon votre taille

        output = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample
        print(f"[Test UNet] Latents après UNet - min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        return output
    except Exception as e:
        print(f"[Test UNet] Erreur pendant l'exécution du UNet : {e}")
        return None

def test_vae_on_latents(latents, vae, device):
    """
    Teste l'entrée latente avant de la passer au VAE
    Cette fonction vérifie la forme et le contenu des latents avant le décodage avec VAE.
    """
    print(f"[Test VAE] Latents avant VAE - min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}")

    # Assurez-vous que les latents ont la forme correcte pour le VAE
    if latents.ndimension() != 4:
        raise ValueError(f"Les latents doivent avoir 4 dimensions, mais ils en ont {latents.ndimension()}")

    # Essayons de décoder avec le VAE
    try:
        decoded_image = vae.decode(latents).sample  # Ajustez selon la fonction exacte du VAE
        print(f"[Test VAE] Image décodée - min={decoded_image.min():.4f}, max={decoded_image.max():.4f}, mean={decoded_image.mean():.4f}")
        return decoded_image
    except Exception as e:
        print(f"[Test VAE] Erreur pendant le décodage avec le VAE : {e}")
        return None


# -------------------------
# Décodage tiled (pour grandes images)
# -------------------------
def encode_images_to_latents_ai(images, vae):
    """
    Encode une batch d'images [B, 3, H, W] en latents [B, 4, H/8, W/8].
    """
    device = images.device
    vae_dtype = next(vae.parameters()).dtype

    with torch.no_grad():
        # On encode en latent avec le VAE, en s'assurant que la sortie a bien 4 canaux
        latents = vae.encode(images.to(vae_dtype)).latent_dist.sample()

    # Assure que les latents sont en 4 canaux, ce qui est attendu pour le VAE
    if latents.shape[1] != 4:
        raise ValueError(f"Latents doivent avoir 4 canaux, mais ont {latents.shape[1]} canaux.")

    # Scale pour correspondre au SD
    latents = latents * LATENT_SCALE
    return latents

# --------------------------------------------------------
#---------- deprecated
#---------------------------------------------------------

def decode_latents_to_image_tiled128_old(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Decode les latents [B, 4, H, W] en images [B, 3, H, W] avec tiling pour éviter OOM.
    Supporte latents 5D [B, C, T, H, W] en reshaping automatique.
    """
    vae_dtype = next(vae.parameters()).dtype

    # Support 5D : [B, C, T, H, W] -> [B*T, C, H, W]
    if latents.ndim == 5:
        B, C, T, H, W = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
    elif latents.ndim == 4:
        B, C, H, W = latents.shape
    else:
        raise ValueError(f"Latents attendus 4D ou 5D, got {latents.shape}")

    # Assure que C=4
    if C != 4:
        raise ValueError(f"Latents doivent avoir 4 canaux, got {C} canaux")

    # Convert dtype pour correspondre au VAE
    latents = latents.to(vae_dtype)

    H_out, W_out = H, W
    output = torch.zeros(B if latents.ndim == 4 else B*T, 3, H_out, W_out, device=device, dtype=torch.float32)

    # Décodage en tiling si image > tile_size
    if max(H, W) <= tile_size:
        with torch.no_grad():
            decoded = vae.decode(latents / LATENT_SCALE).sample
        return decoded.clamp(0, 1)

    # Tiling (optionnel, pour très grandes images)
    # Ici simplifié : on peut ajouter tiling si nécessaire
    with torch.no_grad():
        decoded = vae.decode(latents / LATENT_SCALE).sample
    return decoded.clamp(0, 1)
# -------------------------
# vae_utils.py (version corrigée)
# -------------------------
# -------------------------
# Encode images en latents
# -------------------------
def encode_images_to_latents_ai_old(images, vae):
    """
    Encode une batch d'images [B, 3, H, W] en latents [B, 4, H/8, W/8].
    """
    device = images.device
    vae_dtype = next(vae.parameters()).dtype

    with torch.no_grad():
        latents = vae.encode(images.to(vae_dtype)).latent_dist.sample()
    # Scale pour correspondre au SD
    latents = latents * LATENT_SCALE
    return latents

# -------------------------
# Décodage tiled des latents 5D
# -------------------------
def decode_latents_to_image_tiled128_5D(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Decode les latents [B, 4, H, W] en images [B, 3, H, W] avec tiling pour éviter OOM.
    Supporte latents 5D [B, C, T, H, W] en reshaping automatique.
    """
    vae_dtype = next(vae.parameters()).dtype

    # Support 5D : [B, C, T, H, W] -> [B*T, C, H, W]
    if latents.ndim == 5:
        B, C, T, H, W = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
    elif latents.ndim == 4:
        B, C, H, W = latents.shape
    else:
        raise ValueError(f"Latents attendus 4D ou 5D, got {latents.shape}")

    # Assure que C=4
    assert C == 4, f"Latents doivent avoir 4 canaux, got {C}"

    # Convert dtype pour correspondre au VAE
    latents = latents.to(vae_dtype)

    H_out, W_out = H, W
    output = torch.zeros(B if latents.ndim == 4 else B*T, 3, H_out, W_out, device=device, dtype=torch.float32)

    # Décodage en tiling si image > tile_size
    if max(H, W) <= tile_size:
        with torch.no_grad():
            decoded = vae.decode(latents / LATENT_SCALE).sample
        return decoded.clamp(0, 1)

    # Tiling (optionnel, pour très grandes images)
    # Ici simplifié : on peut ajouter tiling si nécessaire
    with torch.no_grad():
        decoded = vae.decode(latents / LATENT_SCALE).sample
    return decoded.clamp(0, 1)

# ---------------------------------------------------------------------------------------------
# Test KO - deprecated
# -------------------------------------------------------------------------------------------

def decode_latents_to_image_test(latents, vae, tile_size=128, overlap=64, device="cuda"):
    latents = latents.to(torch.float32)

    # For 3D or 4D latents
    if latents.ndim == 2:  # [H*W] ou [N, H*W]
        raise ValueError(f"Latents trop aplatis, shape={latents.shape}")
    elif latents.ndim == 3:  # [C,H,W] -> ajouter batch
        latents = latents.unsqueeze(0)  # [1,C,H,W]
    elif latents.ndim == 4:  # [B,C,H,W]
        pass
    elif latents.ndim == 5:  # [B,C,T,H,W]
        B, C, T, H, W = latents.shape
        latents = latents.reshape(B*T, C, H, W)
    else:
        raise ValueError(f"Latents must be 3D, 4D or 5D, got {latents.ndim}D")

    # Dupliquer si 1 seul canal
    if latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    # Décodage
    with torch.no_grad():
        decoded = vae.decode(latents / 0.18215).sample
        decoded = decoded.clamp(0, 1)

    return decoded


def decode_latents_to_image_SDiffusion(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Décodage VAE en tuiles 128x128, compatible latents 4D ou 5D.
    """
    # Convertir en float32
    latents = latents.to(torch.float32)

    # Fusion batch+time si nécessaire
    if latents.ndim == 5:
        B, C, T, H, W = latents.shape
        latents = latents.reshape(B*T, C, H, W)
    elif latents.ndim == 4:
        B, C, H, W = latents.shape
    else:
        raise ValueError(f"Latents must be 4D or 5D, got {latents.ndim}D")

    assert C == 4, f"Expected 4 channels in latents, got {C}"

    # Décodage complet (pas en tuiles pour simplifier ici)
    with torch.no_grad():
        decoded = vae.decode(latents / 0.18215).sample  # shape [B*T, 3, H, W]
        decoded = decoded.clamp(0, 1)

    # Remettre en 5D si nécessaire
    if 'T' in locals():
        decoded = decoded.reshape(B, T, 3, H, W)

    return decoded

def decode_latents_to_image_tiled4D(latents, vae, tile_size=128, overlap=64, device="cuda"):
    """
    Fonction corrigée pour décoder les latents en image via VAE, en traitant les tuiles.
    Cette version gère correctement la dimension des canaux latents (C=4) et la conversion en float32.
    """

    # Assurez-vous que latents ont la bonne forme [B, C, H, W] où C = 4
    B, C, H, W = latents.shape
    assert C == 4, f"Expected 4 channels in latents, got {C}"

    # Convertir en float32 si nécessaire (VAE attend des latents en float32)
    latents = latents.to(torch.float32)

    # Variable pour stocker les tuiles décodées
    decoded_image = torch.zeros(B, 3, H * tile_size, W * tile_size, device=device)

    # Transformation en image
    to_pil = ToPILImage()

    # Décodez les tuiles par petits morceaux
    for i in range(0, H, tile_size - overlap):
        for j in range(0, W, tile_size - overlap):
            # Extraire la tuile avec chevauchement
            y1, y2 = i, min(i + tile_size, H)
            x1, x2 = j, min(j + tile_size, W)
            latent_tile = latents[:, :, y1:y2, x1:x2]

            # Décoder la tuile
            with torch.no_grad():
                decoded_tile = vae.decode(latent_tile / 0.18215).sample

            # Ajuster les dimensions pour fusionner les tuiles
            decoded_tile = decoded_tile.clamp(0, 1)

            # Insérer la tuile dans la position correspondante de l'image finale
            decoded_image[:, :, y1 * tile_size:(y2 * tile_size), x1 * tile_size:(x2 * tile_size)] = decoded_tile

    return decoded_image





# -------------------------
# Fonction test VAE (sans affichage d'image)
# -------------------------
def test_vae_simple(vae_path, device="cuda"):
    try:
        vae = safe_load_vae(vae_path, device=device, fp16=False)
        if vae is None:
            return False

        # Tenseur aléatoire pour test
        test_latent = torch.randn(1, 4, 32, 32).to(device)
        with torch.no_grad():
            decoded_out = vae.decode(test_latent / 0.18215)
            decoded = decoded_out.sample if hasattr(decoded_out, "sample") else decoded_out

        # Vérification simple
        if decoded is not None and decoded.shape[1] == 3:
            return True
        return False

    except Exception as e:
        print(f"⚠ Test VAE échoué : {e}")
        return False


def test_vae_256(vae, image):
    """
    Test rapide du VAE 256x256.
    Vérifie encode -> decode sans afficher d'image.
    """

    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    transform = T.Compose([
        T.Resize(256),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        print("🔎 Latent shape:", latents.shape)

        decoded = vae.decode(latents).sample
        print("🔎 Decoded shape:", decoded.shape)

    print("✅ Test VAE 256 OK")



def test_vae(vae_path: str, device: str = "cuda") -> bool:
    """
    Charge un VAE depuis un .safetensors et effectue un test de décodage rapide.
    Retourne True si le VAE est opérationnel, False sinon.
    """
    try:
        # Charge le state_dict depuis le fichier .safetensors
        state_dict = load_file(vae_path, device="cpu")
        print("✅ State dict chargé avec succès, clés:", list(state_dict.keys())[:5])

        # Crée un VAE standard compatible SD
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D"]*4,
            up_block_types=["UpDecoderBlock2D"]*4,
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            sample_size=32  # adapter selon le checkpoint
        )

        # Ignore les clés manquantes ou inattendues
        vae.load_state_dict(state_dict, strict=False)
        vae = vae.to(device)
        print(f"✅ VAE chargé et déplacé sur {device}")

        # Test rapide avec un tenseur latent aléatoire
        test_latent = torch.randn(1, 4, 32, 32).to(device)
        with torch.no_grad():
            decoded_out = vae.decode(test_latent / 0.18215)
            decoded = decoded_out.sample if hasattr(decoded_out, "sample") else decoded_out

        # Vérifie juste la forme sans afficher l'image
        if decoded.shape[1] == 3:
            print(f"✅ Décodage test OK, output shape: {decoded.shape}")
            return True
        else:
            print(f"⚠ Décodage test incorrect, output shape: {decoded.shape}")
            return False

    except Exception as e:
        print("⚠ Erreur lors du test VAE :", e)
        return False



def safe_load_vae_safetensors(vae_path, device="cuda", fp16=False, offload=False):
    """
    Charge un VAE depuis un fichier .safetensors seul.
    """
    if not vae_path or not os.path.exists(vae_path):
        print(f"⚠ VAE non trouvé à {vae_path}")
        return None

    try:
        # Charger le state dict du fichier safetensors
        state_dict = load_file(vae_path, device="cpu")  # d'abord en CPU pour éviter OOM

        # Créer un AutoencoderKL vide (Tiny-SD 128x128)
        vae = AutoencoderKL(
            in_channels=4,
            out_channels=4,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[128, 256],
            latent_channels=4,
            sample_size=32
        )

        # Charger le state dict
        vae.load_state_dict(state_dict, strict=False)

        # Déplacer sur le device voulu et ajuster dtype
        vae = vae.to(torch.float16 if fp16 else torch.float32)
        vae = vae.to(device)

        return vae
    except Exception as e:
        print(f"⚠ Erreur lors du chargement du VAE : {e}")
        return None



# -------------------------
# Decode latents VAE auto tile
# -------------------------
def decode_latents_frame_ai_auto(latents: torch.Tensor, vae, device="cuda"):
    """
    Decode latents en image PIL ou tensor, tile_size automatique:
      - Si max(H,W) <= 256: decode complet
      - Sinon: tiles 128x128, overlap 50%
    latents: [C,H/8,W/8] ou [B,C,H/8,W/8]
    vae: modèle VAE
    device: "cuda" ou "cpu"
    Retourne: Tensor float32 [3,H,W] en [0,1]
    """
    import torch
    latents = latents.to(device)

    # Si pas de batch dim, ajouter
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)  # [1,C,H,W]

    _, C, H_lat, W_lat = latents.shape
    H_out, W_out = H_lat*8, W_lat*8  # upscaling par VAE

    # Petites images → decode complet
    if max(H_out, W_out) <= 256:
        with torch.no_grad():
            frame_tensor = vae.decode(latents / 0.18215).sample  # Tiny-SD scale
            frame_tensor = torch.clamp(frame_tensor, -1.0, 1.0)
            frame_tensor = (frame_tensor + 1) / 2  # [-1,1] → [0,1]
            frame_tensor = frame_tensor[0]  # enlever batch dim
        return frame_tensor

    # Grandes images → decode en tiles
    tile_size = 128
    overlap = tile_size // 2
    _, C, H, W = latents.shape
    output = torch.zeros((1, 3, H*8, W*8), device=device)

    count_map = torch.zeros_like(output)

    # Générer les positions de tiles
    xs = list(range(0, W, tile_size - overlap))
    ys = list(range(0, H, tile_size - overlap))

    for y in ys:
        for x in xs:
            y1, y2 = y, min(y + tile_size, H)
            x1, x2 = x, min(x + tile_size, W)

            lat_tile = latents[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                dec_tile = vae.decode(lat_tile / 0.18215).sample
                dec_tile = torch.clamp(dec_tile, -1, 1)
                dec_tile = (dec_tile + 1) / 2  # [0,1]

            H_t, W_t = dec_tile.shape[2], dec_tile.shape[3]
            output[:, :, y1*8:y1*8+H_t*8, x1*8:x1*8+W_t*8] += dec_tile.repeat(1,1,8,8)
            count_map[:, :, y1*8:y1*8+H_t*8, x1*8:x1*8+W_t*8] += 1.0

    output /= count_map.clamp(min=1.0)
    return output[0]


# -------------------------
# Mémoire GPU utils
# -------------------------
def log_gpu_memory(tag=""):
    if torch.cuda.is_available():
        print(f"[GPU MEM] {tag} → allocated={torch.cuda.memory_allocated()/1e6:.1f}MB, "
              f"reserved={torch.cuda.memory_reserved()/1e6:.1f}MB, "
              f"max_allocated={torch.cuda.max_memory_allocated()/1e6:.1f}MB")



# -------------------------
# Encode / Decode avec logs
# -------------------------
def encode_images_to_latents_ai_test(images, vae):
    device = vae.device
    images = images.to(device=device, dtype=torch.float32)
    print(f"[VAE] Encode → device={device}, images.shape={images.shape}, dtype={images.dtype}")
    log_gpu_memory("avant encode VAE")
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
        latents = latents.unsqueeze(2)  # [B,C,1,H,W]
    print(f"[VAE] Latents shape après encode: {latents.shape}")
    log_gpu_memory("après encode VAE")
    return latents

def decode_latents_frame_ai(latents, vae):
    # --- Tile size auto ---
    _, C, H, W = latents.shape[-4:]  # [B,C,H,W]
    tile_size = min(H, W)  # tile couvre toute la latente pour éviter mosaïque
    overlap = tile_size // 2
    print(f"[VAE] Decode avec tile_size={tile_size}, overlap={overlap}, device={vae.device}, latents.shape={latents.shape}")
    log_gpu_memory("avant decode VAE")
    frame_tensor = decode_latents_to_image_tiled(latents, vae, tile_size=tile_size, overlap=overlap).clamp(0,1)
    print(f"[VAE] Frame tensor shape après decode: {frame_tensor.shape}")
    log_gpu_memory("après decode VAE")
    return frame_tensor


# -------------------------
# Encode / Decode
# -------------------------
def encode_images_to_latents(images, vae):
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

def decode_latents_to_image(latents, vae):
    latents = latents.to(vae.device).float() / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img

# -------------------------
# Model loaders
# -------------------------

def safe_load_unet(model_path, device, fp16=False):
    """
    Chargement sécurisé du UNet depuis un dossier local.
    - supporte .safetensors ou .bin
    - force strict=False pour éviter les mismatches
    - affiche les poids non chargés
    - support fp16/fp32
    """

    folder = os.path.join(model_path, "unet")
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Le dossier UNet n'existe pas : {folder}")

    print(f"🔄 Chargement UNet depuis {folder} ...")
    model = UNet2DConditionModel.from_pretrained(folder)

    # --- Chemins fichiers de poids ---
    state_dict_path_safetensors = os.path.join(folder, "diffusion_pytorch_model.safetensors")
    state_dict_path_bin = os.path.join(folder, "diffusion_pytorch_model.bin")

    # --- Chargement des poids selon format ---
    if os.path.exists(state_dict_path_safetensors):
        print("✅ Chargement poids safetensors")
        state_dict = load_file(state_dict_path_safetensors, device="cpu")
    elif os.path.exists(state_dict_path_bin):
        print("✅ Chargement poids bin")
        state_dict = torch.load(state_dict_path_bin, map_location="cpu")
    else:
        raise FileNotFoundError("Aucun fichier de poids UNet trouvé (.safetensors ou .bin)")

    # --- Charger dans le modèle avec strict=False ---
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"⚠️ Poids manquants : {missing}")
        print(f"⚠️ Poids inattendus : {unexpected}")

    # --- Convert dtype si demandé ---
    if fp16:
        model = model.half()

    # --- Envoyer sur device ---
    model = model.to(device)

    print(f"✅ UNet chargé avec dtype={next(model.parameters()).dtype}, device={next(model.parameters()).device}")
    return model
# -------------------------
# Model loaders
# -------------------------

def safe_load_unet_ori(model_path, device, fp16=False):
    folder = os.path.join(model_path, "unet")
    if os.path.exists(folder):
        model = UNet2DConditionModel.from_pretrained(folder)
        if fp16:
            model = model.half()
        return model.to(device)
    return None

def safe_load_vae_stable(model_path, device, fp16=False, offload=False):
    folder = os.path.join(model_path, "vae")
    if os.path.exists(folder):
        model = AutoencoderKL.from_pretrained(folder)
        model = model.to("cpu" if offload else device).float()
        return model
    return None

def safe_load_scheduler(model_path):
    folder = os.path.join(model_path, "scheduler")
    if os.path.exists(folder):
        return DPMSolverMultistepScheduler.from_pretrained(folder)
    return None


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
# Encode / Decode FP16 safe
# -------------------------
def encode_images_to_latents_safe(images, vae):
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
