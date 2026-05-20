import torch
from PIL import Image, ImageEnhance
from torchvision.transforms import ToTensor, ToPILImage
from pathlib import Path
from torchvision.transforms import functional as F
import math
import itertools
import numpy as np


from scripts.utils.vae_utils import safe_load_unet
from scripts.utils.n3r_utils import load_images_test
from scripts.modules.motion_ulta_lite_fix import MotionModuleUltraLiteFixComplete
from scripts.utils.n3r_utils import LATENT_SCALE
from scripts.utils.tools_utils import (
    prepare_frame_tensor,
    normalize_frame,
    tensor_to_pil,
    ensure_4_channels,
    save_frames_as_video_from_folder
)
#from scripts.n3rmodelSD import encode_images_to_latents_safe

# ------------------- DECODE ULTRA-SAFE GLOBAL -------------------
#gamma = 1.0  # 1.2
#brightness = 1.0 # 1.0
#contrast = 1.3 # 1.2
#saturation = 1.1 # 1.2



def test_parameter_grid_with_boost(
    latents_motion, vae, img_orig,
    gammas=[1.0,1.1,1.2,1.5],
    contrasts=[1.1,1.2,1.3,1.5],
    saturations=[1.0,1.1,1.2,1.5],
    brightnesses=[1.0],
    epsilon=1e-5,
    latent_scale_boost=5.71,  #[5.4, 5.5, 5.6, 5.7, 5.8]
    device="cuda"
):
    results = []

    # Convertir img_orig en tensor sur le device
    if isinstance(img_orig, list) or isinstance(img_orig, tuple):
        img_orig_tensor = img_orig[0]
    else:
        img_orig_tensor = img_orig
    if isinstance(img_orig_tensor, Image.Image):
        img_orig_tensor = torch.tensor(np.array(img_orig_tensor) / 255.0).permute(2,0,1)
    img_orig_tensor = img_orig_tensor.to(device=device, dtype=torch.float32)

    for gamma, contrast, saturation, brightness in itertools.product(gammas, contrasts, saturations, brightnesses):
        # Décodage blockwise avec boost et epsilon
        decoded_img = decode_latents_ultrasafe_blockwise(
            latents_motion, vae,
            gamma=gamma,
            contrast=contrast,
            saturation=saturation,
            brightness=brightness,
            device=device,
            epsilon=epsilon,
            latent_scale_boost=latent_scale_boost
        )

        # Si BATCH =1, assure tensor
        if isinstance(decoded_img, list):
            decoded_img = decoded_img[0]

        # Redimensionner le décodé pour matcher l'original
        decoded_img_resized = decoded_img.resize(
            (img_orig_tensor.shape[2], img_orig_tensor.shape[1]), Image.BICUBIC
        )

        # Comparer
        stats = compare_images_stats_v2(img_orig_tensor, decoded_img_resized, device=device)
        stats.update({
            "gamma": gamma,
            "contrast": contrast,
            "saturation": saturation,
            "brightness": brightness,
            "epsilon": epsilon,
            "latent_scale_boost": latent_scale_boost
        })
        results.append(stats)

    # Trier par meilleure fidélité (mean_diff_total)
    results_sorted = sorted(results, key=lambda x: x["mean_diff_total"])
    return results_sorted

def compare_images_stats_v2(img_orig, img_decoded, threshold=0.05, device="cuda"):
    """
    Compare deux images PIL ou tensors et retourne les écarts statistiques.
    - img_orig : tensor [C,H,W] ou [B,C,H,W] ou PIL
    - img_decoded : tensor ou PIL
    - threshold : seuil pour % de pixels différents
    - device : 'cuda' ou 'cpu'
    """

    # --- Convertir PIL en tensor float [0,1] ---
    if isinstance(img_orig, Image.Image):
        img_orig = torch.tensor(np.array(img_orig) / 255.0).permute(2,0,1)
    if isinstance(img_decoded, Image.Image):
        img_decoded = torch.tensor(np.array(img_decoded) / 255.0).permute(2,0,1)

    # --- Retirer batch si nécessaire ---
    if img_orig.ndim == 4:  # [B,C,H,W]
        img_orig = img_orig[0]
    if img_decoded.ndim == 4:  # [B,C,H,W]
        img_decoded = img_decoded[0]

    # --- Redimensionner img_decoded pour matcher img_orig ---
    C,H,W = img_orig.shape
    if img_decoded.shape[1:] != (H,W):  # shape [C,H_dec,W_dec]
        # torchvision functional resize attend [C,H,W] mais resize = (H,W)
        img_decoded = F.resize(img_decoded, size=(H,W), antialias=True)

    # --- Envoyer sur le device ---
    img_orig = img_orig.to(device=device, dtype=torch.float32)
    img_decoded = img_decoded.to(device=device, dtype=torch.float32)

    # --- Calcul des différences ---
    diff = torch.abs(img_orig - img_decoded)
    mean_diff_per_channel = diff.view(3,-1).mean(dim=1).cpu().numpy()
    mean_diff_total = diff.mean().item()
    percent_diff = 100 * (diff.max(dim=0)[0] > threshold).sum().item() / (diff.shape[1]*diff.shape[2])

    return {
        "mean_diff_r": mean_diff_per_channel[0],
        "mean_diff_g": mean_diff_per_channel[1],
        "mean_diff_b": mean_diff_per_channel[2],
        "mean_diff_total": mean_diff_total,
        "percent_diff_pixels": percent_diff
    }



def compare_images_stats(img_orig, img_decoded, threshold=0.05, device="cuda"):
    """Compare deux images PIL ou tensors et retourne les écarts statistiques"""
    if isinstance(img_orig, Image.Image):
        img_orig = torch.tensor(np.array(img_orig) / 255.0).permute(2,0,1).to(device)
    if isinstance(img_decoded, Image.Image):
        img_decoded = torch.tensor(np.array(img_decoded) / 255.0).permute(2,0,1).to(device)

    assert img_orig.shape == img_decoded.shape, "Les images doivent avoir la même taille"

    diff = torch.abs(img_orig - img_decoded)
    mean_diff_per_channel = diff.view(3,-1).mean(dim=1).cpu().numpy()
    mean_diff_total = diff.mean().item()
    percent_diff = 100 * (diff.max(dim=0)[0] > threshold).sum().item() / (diff.shape[1]*diff.shape[2])

    return {
        "mean_diff_r": mean_diff_per_channel[0],
        "mean_diff_g": mean_diff_per_channel[1],
        "mean_diff_b": mean_diff_per_channel[2],
        "mean_diff_total": mean_diff_total,
        "percent_diff_pixels": percent_diff
    }


def test_parameter_grid_extended(latents_motion, vae, img_orig,
                                 gammas=[1.0, 1.2, 1.5],
                                 contrasts=[1.0, 1.2, 1.5],
                                 saturations=[1.0, 1.2, 1.5],
                                 brightnesses=[1.0],
                                 device="cuda"):

    results = []

    # Convertir img_orig en tensor sur le device
    if isinstance(img_orig, list) or isinstance(img_orig, tuple):
        img_orig_tensor = img_orig[0]
    else:
        img_orig_tensor = img_orig
    if isinstance(img_orig_tensor, Image.Image):
        img_orig_tensor = torch.tensor(np.array(img_orig_tensor) / 255.0).permute(2,0,1)
    img_orig_tensor = img_orig_tensor.to(device=device, dtype=torch.float32)

    for gamma, contrast, saturation, brightness in itertools.product(gammas, contrasts, saturations, brightnesses):
        # Décodage blockwise
        decoded_img = decode_latents_ultrasafe_blockwise(
            latents_motion, vae,
            gamma=gamma,
            contrast=contrast,
            saturation=saturation,
            brightness=brightness,
            device=device
        )

        # Si BATCH =1, assure tensor
        if isinstance(decoded_img, list):
            decoded_img = decoded_img[0]

        # Redimensionner le décodé pour matcher l'original
        decoded_img_resized = decoded_img.resize(
            (img_orig_tensor.shape[2], img_orig_tensor.shape[1]), Image.BICUBIC
        )

        # Comparer
        stats = compare_images_stats(img_orig_tensor, decoded_img_resized, device=device)
        stats.update({
            "gamma": gamma,
            "contrast": contrast,
            "saturation": saturation,
            "brightness": brightness
        })
        results.append(stats)

    # Trier par meilleure fidélité (mean_diff_total)
    results_sorted = sorted(results, key=lambda x: x["mean_diff_total"])
    return results_sorted

# -------------------------------------------------------------
# Comparatif statistique intégré
# -------------------------------------------------------------


def compare_images_stats_v1(img_orig, img_decoded, threshold=0.05):
    """Compare deux images PIL ou tensors et retourne les écarts statistiques"""
    # Convertir en tensor CPU float [0,1]
    if isinstance(img_orig, Image.Image):
        img_orig = torch.tensor(np.array(img_orig) / 255.0).permute(2,0,1)
    if isinstance(img_decoded, Image.Image):
        img_decoded = torch.tensor(np.array(img_decoded) / 255.0).permute(2,0,1)

    # Forcer CPU pour éviter le RuntimeError
    img_orig = img_orig.cpu()
    img_decoded = img_decoded.cpu()

    # S’assurer que les tailles correspondent
    assert img_orig.shape == img_decoded.shape, "Les images doivent avoir la même taille"

    diff = torch.abs(img_orig - img_decoded)
    mean_diff_per_channel = diff.view(3,-1).mean(dim=1).numpy()
    mean_diff_total = diff.mean().item()
    percent_diff = 100 * (diff.max(dim=0)[0] > threshold).sum().item() / (diff.shape[1]*diff.shape[2])

    return {
        "mean_diff_r": mean_diff_per_channel[0],
        "mean_diff_g": mean_diff_per_channel[1],
        "mean_diff_b": mean_diff_per_channel[2],
        "mean_diff_total": mean_diff_total,
        "percent_diff_pixels": percent_diff
    }

def apply_adjustments(img_pil, gamma=1.0, brightness=1.0, contrast=1.0, saturation=1.0):
    img = ImageEnhance.Brightness(img_pil).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    if gamma != 1.0:
        img = img.point(lambda x: 255 * ((x/255) ** (1/gamma)))
    return img

def test_parameter_grid(latents_motion, vae, img_orig,
                        gammas=[1.0,1.1,1.2,1.5],
                        contrasts=[1.0,1.2,1.3,1.5],
                        saturations=[1.0,1.2,1.3,1.5],
                        brightnesses=[1.0, 1.1]):

    results = []
    for gamma, contrast, saturation, brightness in itertools.product(gammas, contrasts, saturations, brightnesses):
        decoded_img = decode_latents_ultrasafe_blockwise(
            latents_motion, vae,
            gamma=gamma,
            contrast=contrast,
            saturation=saturation,
            brightness=brightness
        )
        stats = compare_images_stats_v1(img_orig[0], decoded_img)
        stats.update({"gamma": gamma, "contrast": contrast, "saturation": saturation, "brightness": brightness})
        results.append(stats)

    # Trier par meilleure fidélité
    results_sorted = sorted(results, key=lambda x: x["mean_diff_total"])
    return results_sorted
# ------------------- DECODE ULTRA-SAFE BLOCKWISE -------------------
# --------------------------------------------------------------
# decode_latents_ultrasafe_blockwise.py
# --------------------------------------------------------------

import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def decode_latents_ultrasafe_blockwise_test(latents, vae, block_size=32, device='cuda', dtype=torch.float16):
    """
    Décodage des latents avec tiling et pondération cosinus pour éviter les artefacts de patch.

    Args:
        latents: Tensor [1, 4, H_latent, W_latent]
        vae: modèle VAE (decode)
        block_size: taille du patch latent (en latent space)
        device: 'cuda' ou 'cpu'
        dtype: torch.float16 pour accélérer sur GPU
    """

    B, C, H, W = latents.shape
    latents = latents.to(device, dtype=dtype)

    # Taille de sortie image
    out_h, out_w = H*8, W*8

    output_rgb = torch.zeros((B, 3, out_h, out_w), device=device, dtype=dtype)
    weight = torch.zeros_like(output_rgb)

    # Préparer la pondération cosinus
    yy = torch.linspace(-torch.pi/2, torch.pi/2, out_h, device=device)
    xx = torch.linspace(-torch.pi/2, torch.pi/2, out_w, device=device)
    wy = torch.cos(yy).clamp(min=0)
    wx = torch.cos(xx).clamp(min=0)
    weight_patch_full = torch.outer(wy, wx)[None, None, :, :]  # [1,1,H,W]

    # Découper les latents en patchs
    for y0 in range(0, H, block_size):
        y1 = min(y0 + block_size, H)
        for x0 in range(0, W, block_size):
            x1 = min(x0 + block_size, W)

            latent_patch = latents[:, :, y0:y1, x0:x1]

            # Décoder le patch (VAE decode)
            decoded = vae.decode(latent_patch).sample  # [B,3,H_patch*8,W_patch*8]
            decoded = decoded.to(dtype=dtype)

            # Pondération cosinus pour ce patch
            ih0, ih1 = y0*8, y1*8
            iw0, iw1 = x0*8, x1*8
            weight_patch = weight_patch_full[:, :, ih0:ih1, iw0:iw1]

            output_rgb[:, :, ih0:ih1, iw0:iw1] += decoded * weight_patch
            weight[:, :, ih0:ih1, iw0:iw1] += weight_patch

    # Normaliser
    output_rgb /= weight
    output_rgb = output_rgb.clamp(-1.0, 1.0)

    return output_rgb



# Version précédente fonctionnel ********************************
def decode_latents_ultrasafe_blockwise(
    latents, vae,
    block_size=32, overlap=16,
    gamma=1.2, brightness=1.0,
    contrast=1.2, saturation=1.3,
    device="cuda", frame_counter=0, output_dir=Path("."),
    epsilon=1e-6,
    latent_scale_boost=1.1  # boost léger pour récupérer les nuances
):
    """
    Décodage ultra-safe par blocs des latents en image PIL.
    Optimisé pour préserver les nuances de couleur et réduire l'effet "photocopie".
    """

    B, C, H, W = latents.shape
    latents = latents.to(device=device, dtype=torch.float32) * latent_scale_boost

    # Dimensions finales
    out_H = H * 8
    out_W = W * 8
    output_rgb = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output_rgb)

    stride = block_size - overlap

    # Calcul positions garanties pour full coverage
    y_positions = list(range(0, H - block_size + 1, stride)) or [0]
    x_positions = list(range(0, W - block_size + 1, stride)) or [0]

    if y_positions[-1] != H - block_size:
        y_positions.append(H - block_size)
    if x_positions[-1] != W - block_size:
        x_positions.append(W - block_size)

    for y in y_positions:
        for x in x_positions:
            y1 = y + block_size
            x1 = x + block_size

            patch = latents[:, :, y:y1, x:x1]

            # Sécurité : NaN / Inf / epsilon
            patch = torch.nan_to_num(patch, nan=0.0, posinf=5.0, neginf=-5.0)
            if torch.all(patch == 0):
                patch += epsilon

            # Décodage
            with torch.no_grad():
                decoded = vae.decode(patch).sample.to(torch.float32)

            # Intégration dans l'image finale
            iy0, ix0 = y*8, x*8
            iy1, ix1 = y1*8, x1*8
            output_rgb[:, :, iy0:iy1, ix0:ix1] += decoded
            weight[:, :, iy0:iy1, ix0:ix1] += 1.0

    # Moyenne pour blending final
    output_rgb = output_rgb / weight.clamp(min=1e-6)
    output_rgb = output_rgb.clamp(-1.0, 1.0)

    # Convertir en PIL et appliquer corrections gamma / contrast / saturation / brightness
    frame_pil_list = []
    for i in range(B):
        img = F.to_pil_image((output_rgb[i] + 1) / 2)  # [-1,1] -> [0,1]
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        if gamma != 1.0:
            img = img.point(lambda x: 255 * ((x / 255) ** (1 / gamma)))
        frame_pil_list.append(img)

    return frame_pil_list[0] if B == 1 else frame_pil_list


def encode_images_to_latents_nuanced(images, vae, device="cuda", latent_scale=LATENT_SCALE):
    """
    Encode une image en latents VAE tout en préservant le contraste et les nuances de couleur.
    - Utilise la moyenne de la distribution latente
    - Clamp minimal seulement pour sécurité
    - Force 4 canaux si nécessaire
    """

    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mean  # moyenne pour plus de stabilité

    # Appliquer le scaling mais garder la dynamique
    latents = latents * latent_scale

    # Sécurité NaN / Inf (mais pas normalisation globale)
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # Forcer 4 canaux si nécessaire (VAE attend souvent 4)
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents

# ---------------- Paramètres ----------------
device = "cuda"
image_path = "input/256/image_256x0.png"  # ton image de test
vae_path = "/mnt/62G/huggingface/vae/vae-ft-mse-840000-ema-pruned.safetensors"

# ---------------- Charger VAE ----------------
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16).to(device)
vae.enable_slicing()
vae.enable_tiling()

# ---------------- Charger Motion Module ----------------
motion_module = MotionModuleUltraLiteFixComplete(verbose=True)

# ---------------- Charger image ----------------
image = load_images_test([image_path], W=256, H=256, device=device, dtype=torch.float16)

# ---------------- Encoder ----------------
#latents = encode_images_to_latents_safe(image, vae, device=device, epsilon=1e-5)
latents = encode_images_to_latents_nuanced(image, vae, device=device, latent_scale=LATENT_SCALE)
latents = ensure_4_channels(latents)

# ---------------- Appliquer motion ----------------
latents_motion = motion_module(latents.clone())

# ---------------- Décoder ----------------
# latents_motion = ton tenseur [1, 4, F, H, W] ou [1, 4, H, W]
# vae = ton VAE chargé
# device = "cuda" ou "cpu"

# Paramètres boostés
gamma = 1.0  # 1.2
brightness = 1.0 # 1.0
contrast = 1.5 # 1.2
saturation = 1.5 # 1.2
upscale_factor = 2
frame_counter = 0  # pour debug/log si nécessaire

# Décodage ultra-safe (blockwise comme ton code)
frame_pil = decode_latents_ultrasafe_blockwise(
    latents_motion, vae,
    block_size=32, overlap=24,
    gamma=gamma,
    brightness=brightness,
    contrast=contrast,
    saturation=saturation,
    device=device,
    frame_counter=frame_counter,
    output_dir=Path("."),
    epsilon=1e-5,
    latent_scale_boost=5.71
)


#frame_pil =  decode_latents_ultrasafe_blockwise_test(latents_motion, vae, block_size=32, device='cuda', dtype=torch.float16)

# Upscale pour debug visuel
if upscale_factor > 1:
    frame_pil = frame_pil.resize(
        (frame_pil.width * upscale_factor, frame_pil.height * upscale_factor),
        Image.BICUBIC
    )

# Affichage rapide
frame_pil.show()


# -------------------------------------------------------------
# Exemple d'utilisation après ton décodage normal
# -------------------------------------------------------------
#results = test_parameter_grid(latents_motion, vae, image)


#print("Top 5 configurations les plus proches de l'image originale - test_parameter_grid :")
#for r in results[:5]:
#    print(r)

results = test_parameter_grid_with_boost(latents_motion, vae, image)


print("Top 5 configurations les plus proches de l'image originale - test_parameter_grid_with_boost:")
for r in results[:5]:
    print(r)
