# fx_utils.py
# ------------------- ENCODE -------------------

import torch
import os, math, threading
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageChops
import cv2
import numpy as np
import subprocess
from torchvision.transforms import functional as F
from .n3rMotionPose_tools import save_debug_mask, feather_inside_strict2, feather_outside_only_alpha2
import torch.nn.functional as Fu
import torch.nn.functional as TF
import torch.nn.functional as FF
import torch.nn.functional as F
LATENT_SCALE = 0.18215


def sanitize_coords(coords):
    valid = []
    if isinstance(coords, dict) and "center" in coords:
        coords = [coords["center"]]
    for p in coords:
        try:
            if len(p) != 2:
                print(f"⚠ Coordonnée ignorée car invalide: {p}")
                continue
            x, y = int(p[0]), int(p[1])
            valid.append([x, y])
        except (ValueError, TypeError):
            print(f"⚠ Coordonnée ignorée car invalide: {p}")
            continue
    return valid


def compress_highlights(frame_pil, threshold=235, strength=0.6):
    arr = np.array(frame_pil).astype("float32")

    # luminance approx
    lum = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

    mask = lum > threshold

    # compression douce
    factor = 1.0 - strength * ((lum - threshold) / (255 - threshold))
    factor = np.clip(factor, 0.6, 1.0)

    arr[mask] = arr[mask] * factor[mask][:, None]

    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def remove_white_noise(frame_pil, threshold=254, blur_radius=0.1):
    """
    Atténue les pixels trop blancs (artefacts) par lissage local.
    threshold : valeur RGB au-dessus de laquelle un pixel est considéré bruit
    blur_radius : rayon du lissage local
    """
    frame_rgb = frame_pil.convert("RGB")
    # créer un masque des pixels trop blancs
    mask = frame_rgb.point(lambda i: 255 if i > threshold else 0)
    mask = mask.convert("L")
    # flouter la zone bruyante
    blurred = frame_rgb.filter(ImageFilter.GaussianBlur(blur_radius))
    # fusionner uniquement sur le masque
    cleaned = Image.composite(blurred, frame_rgb, mask)
    return cleaned



def apply_post_processing_unreal_smooth(frame_pil,
                                        contrast=1.15,
                                        vibrance=1.05,
                                        edge_strength=1.5,
                                        simplify_radius=0.8,
                                        smooth_radius=0.05,
                                        sharpen_percent=70):
    # 1️⃣ Unreal (volume + bords)
    frame_pil = apply_post_processing_unreal_safe(
        frame_pil,
        contrast=contrast,
        vibrance=vibrance,
        edge_strength=edge_strength,
        simplify_radius=simplify_radius
    )

    # 2️⃣ Lissage adaptatif (smoothing léger, préserve contours)
    # On utilise un GaussianBlur très léger et on peut mélanger avec l'image originale pour contrôler le lissage
    frame_blur = frame_pil.filter(ImageFilter.GaussianBlur(radius=smooth_radius))
    frame_pil = Image.blend(frame_pil, frame_blur, alpha=0.35)  # alpha <1 pour ne pas tout écraser

    # 3️⃣ Adaptive / final tweaks
    frame_pil = apply_post_processing_adaptive(
        frame_pil,
        blur_radius=0.0,
        contrast=1.05,
        brightness=1.05,
        saturation=0.90,
        vibrance_base=1.0,
        vibrance_max=1.05,
        sharpen=True,
        sharpen_radius=1,
        sharpen_percent=sharpen_percent,
        sharpen_threshold=2
    )

    # 4️⃣ Clamp final pour éviter pixels blancs
    frame_pil = frame_pil.point(lambda i: max(0, min(255, int(i))))

    return frame_pil


def apply_post_processing_unreal_safe(
    frame_pil,
    blur_radius=0.01,          # 🔽 plus subtil
    contrast=1.08,             # 🔽 réduit
    brightness=1.02,
    saturation=0.98,
    vibrance=1.02,
    edge_boost=True,
    edge_strength=0.4,         # 🔥 énorme différence
    simplify_radius=0.4,       # 🔽 moins de blur destructif
    sharpen=True,
    sharpen_radius=0.8,
    sharpen_percent=60,        # 🔽 moins agressif
    sharpen_threshold=3
):
    """
    Version douce : rendu naturel, moins de pixelisation
    """
    from PIL import ImageFilter, ImageEnhance, ImageChops
    import numpy as np

    # 1️⃣ Micro-smooth (léger)
    if simplify_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=simplify_radius))

    # 2️⃣ Vibrance douce (continue, pas de seuil)
    if vibrance != 1.0:
        arr = np.array(frame_pil).astype(np.float32)
        mean = arr.mean(axis=2, keepdims=True)
        arr = mean + (arr - mean) * vibrance
        frame_pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # 3️⃣ Ajustements globaux doux
    frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    # 4️⃣ Edge boost subtil (blend au lieu de add)
    if edge_boost:
        gray = frame_pil.convert("L")
        edge = gray.filter(ImageFilter.FIND_EDGES)
        edge = ImageEnhance.Contrast(edge).enhance(1.2)

        edge_rgb = Image.merge("RGB", (edge, edge, edge))

        # 🔥 blend au lieu de add → beaucoup plus naturel
        frame_pil = Image.blend(frame_pil, edge_rgb, edge_strength)

    # 5️⃣ Sharpen léger (micro détails seulement)
    if sharpen:
        frame_pil = frame_pil.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold
        ))

    # 6️⃣ Micro blur final (anti pixel)
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return frame_pil


def apply_post_processing_unreal(frame_pil,
                                 blur_radius=0.02,
                                 contrast=1.2,
                                 brightness=1.05,
                                 saturation=1.0,
                                 vibrance=1.1,
                                 sharpen=True,
                                 sharpen_radius=2,
                                 sharpen_percent=150,
                                 sharpen_threshold=2,
                                 edge_boost=True,
                                 edge_strength=1.5,
                                 texture_simplify=True,
                                 simplify_radius=1.0):
    """
    Post-processing de type 'Unreal' pour donner plus de volume et styliser les images.
    """

    # ----------------- 1) Lissage / simplification textures -----------------
    if texture_simplify and simplify_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=simplify_radius))

    # ----------------- 2) Contraste / Luminosité / Saturation -----------------
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    # ----------------- 3) Vibrance (boost couleurs faibles) -----------------
    if vibrance != 1.0:
        frame_hsv = frame_pil.convert("HSV")
        h, s, v = frame_hsv.split()
        s = s.point(lambda i: min(255, int(i * vibrance) if i < 128 else i))
        frame_pil = Image.merge("HSV", (h, s, v)).convert("RGB")

    # ----------------- 4) Edge Enhance / Relief -----------------
    if edge_boost:
        # Edge enhance + high contrast overlay pour effet 'bordelands / unreal'
        edge = frame_pil.filter(ImageFilter.FIND_EDGES)
        edge = ImageEnhance.Contrast(edge).enhance(edge_strength)
        frame_pil = ImageChops.add(frame_pil, edge, scale=1.0, offset=0)

    # ----------------- 5) Sharp -----------------
    if sharpen:
        frame_pil = frame_pil.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold
        ))

    # ----------------- 6) Option blur final léger -----------------
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return frame_pil


# ---------------- Fusion N3R/VAE adaptative ----------------

def fuse_n3r_latents_adaptive(latents_frame, n3r_latents, latent_injection=0.7, clamp_val=1.0, creative_noise=0.0):
    # Assurer même taille
    if n3r_latents.shape != latents_frame.shape:
        n3r_latents = torch.nn.functional.interpolate(
            n3r_latents, size=latents_frame.shape[-2:], mode='bilinear', align_corners=False
        )
        # Ajuster le nombre de canaux si nécessaire
        if n3r_latents.shape[1] < latents_frame.shape[1]:
            extra = latents_frame[:, n3r_latents.shape[1]:, :, :].clone() * 0
            n3r_latents = torch.cat([n3r_latents, extra], dim=1)
        elif n3r_latents.shape[1] > latents_frame.shape[1]:
            n3r_latents = n3r_latents[:, :latents_frame.shape[1], :, :]

    n3r_latents = n3r_latents.clone()

    # Normalisation **canal par canal** (RGB uniquement)
    for c in range(min(3, n3r_latents.shape[1])):
        n3r_c = n3r_latents[:, c:c+1, :, :]
        frame_c = latents_frame[:, c:c+1, :, :]
        mean, std = n3r_c.mean(), n3r_c.std()
        n3r_c = (n3r_c - mean) / (std + 1e-6)
        n3r_c = n3r_c * frame_c.std() + frame_c.mean()
        n3r_latents[:, c:c+1, :, :] = n3r_c

    # Ajouter un bruit créatif léger si nécessaire
    if creative_noise > 0.0:
        noise = torch.randn_like(n3r_latents) * creative_noise
        n3r_latents += noise

    # Clamp stricte
    n3r_latents = torch.clamp(n3r_latents, -clamp_val, clamp_val)
    latents_frame = torch.clamp(latents_frame, -clamp_val, clamp_val)

    # Fusion finale
    fused_latents = latent_injection * latents_frame + (1 - latent_injection) * n3r_latents
    fused_latents = torch.clamp(fused_latents, -clamp_val, clamp_val)

    print(f"[N3R fusion frame] mean/std par canal: {fused_latents.mean(dim=(2,3))}, injection={latent_injection:.2f}")
    return fused_latents

def fuse_n3r_latents_adaptive_v1(latents_frame, n3r_latents, latent_injection=0.7, clamp_val=1.0, creative_noise=0.0):
    n3r_latents = n3r_latents.clone()

    # Normalisation **canal par canal**
    for c in range(3):  # RGB uniquement
        n3r_c = n3r_latents[:,c:c+1,:,:]
        frame_c = latents_frame[:,c:c+1,:,:]
        mean, std = n3r_c.mean(), n3r_c.std()
        n3r_c = (n3r_c - mean) / (std + 1e-6)
        n3r_c = n3r_c * frame_c.std() + frame_c.mean()
        n3r_latents[:,c:c+1,:,:] = n3r_c

    # Ajouter un bruit créatif léger si nécessaire
    if creative_noise > 0.0:
        noise = torch.randn_like(n3r_latents) * creative_noise
        n3r_latents += noise

    # Clamp stricte pour éviter débordement
    n3r_latents = torch.clamp(n3r_latents, -clamp_val, clamp_val)
    latents_frame = torch.clamp(latents_frame, -clamp_val, clamp_val)

    # Fusion finale
    fused_latents = latent_injection * latents_frame + (1 - latent_injection) * n3r_latents
    fused_latents = torch.clamp(fused_latents, -clamp_val, clamp_val)

    print(f"[N3R fusion frame] mean/std par canal: {fused_latents.mean(dim=(2,3))}, injection={latent_injection:.2f}")
    return fused_latents


def interpolate_param_fast(start_val, end_val, current_frame, total_frames, mode="linear", speed=2.0):
    """
    Interpolation accélérée pour faire varier les paramètres plus rapidement au début.
    speed > 1 → plus rapide, speed < 1 → plus lent
    """
    t = current_frame / max(total_frames-1, 1)
    t = min(1.0, t * speed)  # accélère la progression

    if mode == "linear":
        return start_val + (end_val - start_val) * t
    elif mode == "cosine":
        t = (1 - math.cos(math.pi * t)) / 2
        return start_val + (end_val - start_val) * t
    elif mode == "ease_in_out":
        t = t*t*(3 - 2*t)
        return start_val + (end_val - start_val) * t
    else:
        return start_val + (end_val - start_val) * t


def interpolate_param(start_val, end_val, current_frame, total_frames, mode="linear"):
    """
    Interpolation d'un paramètre entre start_val -> end_val sur total_frames.
    Modes disponibles: 'linear', 'cosine', 'ease_in_out'
    """
    t = current_frame / max(total_frames-1,1)
    if mode == "linear":
        return start_val + (end_val - start_val) * t
    elif mode == "cosine":
        # Cosine easing pour un départ/arrivée plus doux
        t = (1 - math.cos(math.pi * t)) / 2
        return start_val + (end_val - start_val) * t
    elif mode == "ease_in_out":
        t = t*t*(3 - 2*t)
        return start_val + (end_val - start_val) * t
    else:
        return start_val + (end_val - start_val) * t


def estimate_sharpness(image):
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)
    laplacian = np.abs(
        arr[:-2,1:-1] + arr[2:,1:-1] + arr[1:-1,:-2] + arr[1:-1,2:] - 4*arr[1:-1,1:-1]
    )
    return laplacian.mean()

def adaptive_post_process(image):
    sharpness = estimate_sharpness(image)

    # seuils empiriques (à ajuster)
    if sharpness < 8:
        # image floue → sharpen fort
        return apply_post_processing(
            image,
            blur_radius=0.02,
            contrast=1.1,
            brightness=1.05,
            saturation=0.9,
            sharpen=True,
            sharpen_radius=1,
            sharpen_percent=120,
            sharpen_threshold=2
        )

    elif sharpness > 15:
        # image déjà très nette → adoucir
        return apply_post_processing(
            image,
            blur_radius=0.15,
            contrast=1.05,
            brightness=1.02,
            saturation=0.85,
            sharpen=False
        )

    else:
        # équilibré
        return apply_post_processing(
            image,
            blur_radius=0.05,
            contrast=1.1,
            brightness=1.05,
            saturation=0.9,
            sharpen=True,
            sharpen_radius=1,
            sharpen_percent=80,
            sharpen_threshold=2
        )


def apply_post_processing_adaptive(
    frame_pil,
    blur_radius=0.05,
    contrast=1.15,
    brightness=1.05,
    saturation=0.85,
    vibrance_base=1.1,      # vibrance de base
    vibrance_max=1.3,       # max booster pour zones peu saturées
    sharpen=False,
    sharpen_radius=1,
    sharpen_percent=90,
    sharpen_threshold=2,
    clamp_r=True            # clamp adaptatif du canal rouge
):
    if frame_pil.mode != "RGB":
        frame_pil = frame_pil.convert("RGB")

    # ---------------- GaussianBlur léger ----------------
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # ---------------- Contrast / Brightness / Saturation ----------------
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    # ---------------- Vibrance adaptative ----------------
    try:
        frame_np = np.array(frame_pil).astype(np.float32)
        # calculer la saturation relative par pixel
        max_rgb = np.max(frame_np, axis=2)
        min_rgb = np.min(frame_np, axis=2)
        sat = max_rgb - min_rgb
        # vibrance: plus le pixel est peu saturé, plus on boost
        factor_map = vibrance_base + (vibrance_max - vibrance_base) * (1 - sat/255.0)
        factor_map = np.clip(factor_map, vibrance_base, vibrance_max)
        for c in range(3):
            frame_np[:,:,c] = np.clip(frame_np[:,:,c] * factor_map, 0, 255)
        frame_pil = Image.fromarray(frame_np.astype(np.uint8))
    except Exception as e:
        print(f"[WARNING] vibrance adaptative skipped: {e}")

    # ---------------- Clamp adaptatif du canal rouge ----------------
    if clamp_r:
        try:
            r, g, b = frame_pil.split()
            r_np = np.array(r).astype(np.float32)
            r_mean = r_np.mean()
            # si la moyenne est trop haute, réduire légèrement
            if r_mean > 180:
                factor = 180 / r_mean
                r_np = np.clip(r_np * factor, 0, 255)
            r = Image.fromarray(r_np.astype(np.uint8))
            frame_pil = Image.merge("RGB", (r, g, b))
        except Exception as e:
            print(f"[WARNING] clamp rouge skipped: {e}")

    # ---------------- Sharp / UnsharpMask ----------------
    if sharpen:
        try:
            frame_pil = frame_pil.filter(ImageFilter.UnsharpMask(
                radius=sharpen_radius,
                percent=sharpen_percent,
                threshold=sharpen_threshold
            ))
        except Exception as e:
            print(f"[WARNING] sharpening skipped: {e}")

    return frame_pil

def apply_post_processing(
    frame_pil,
    blur_radius=0.05,
    contrast=1.15,
    brightness=1.05,
    saturation=0.85,
    vibrance=1.0,   # valeurs raisonnables pour éviter doré
    sharpen=False,
    sharpen_radius=1,
    sharpen_percent=90,
    sharpen_threshold=2,
    clamp_r=True     # optionnel: clamp canal rouge pour éviter doré
):
    if frame_pil.mode != "RGB":
        frame_pil = frame_pil.convert("RGB")

    # GaussianBlur
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Contrast / Brightness / Saturation
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    # Vibrance: booster légèrement les couleurs peu saturées
    if vibrance != 1.0:
        try:
            frame_hsv = frame_pil.convert("HSV")
            h, s, v = frame_hsv.split()
            s = s.point(lambda i: min(255, int(i * vibrance) if i < 128 else i))
            frame_pil = Image.merge("HSV", (h, s, v)).convert("RGB")
        except Exception as e:
            print(f"[WARNING] vibrance skipped due to error: {e}")

    # Clamp canal rouge pour éviter les zones trop dorées
    if clamp_r:
        r, g, b = frame_pil.split()
        r = r.point(lambda i: min(230, i))  # clamp max à 230 (~90% du max)
        frame_pil = Image.merge("RGB", (r, g, b))

    # Sharp / UnsharpMask
    if sharpen:
        try:
            frame_pil = frame_pil.filter(ImageFilter.UnsharpMask(
                radius=sharpen_radius,
                percent=sharpen_percent,
                threshold=sharpen_threshold
            ))
        except Exception as e:
            print(f"[WARNING] sharpening skipped due to error: {e}")

    return frame_pil

def apply_post_processing_v1(frame_pil,
                          blur_radius=0.05,
                          contrast=1.15,
                          brightness=1.05,
                          saturation=0.85,
                          vibrance=1.1,  # <-- ajout vibrance
                          sharpen=False,
                          sharpen_radius=1,
                          sharpen_percent=90,
                          sharpen_threshold=2):
    # GaussianBlur
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Contrast / Brightness / Saturation
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    # Vibrance: booster les couleurs peu saturées
    if vibrance != 1.0:
        frame_hsv = frame_pil.convert("HSV")
        h, s, v = frame_hsv.split()
        s = s.point(lambda i: min(255, int(i * vibrance) if i < 128 else i))
        frame_pil = Image.merge("HSV", (h, s, v)).convert("RGB")

    # Sharp / UnsharpMask
    if sharpen:
        frame_pil = frame_pil.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold
        ))

    return frame_pil


def apply_post_processing_blur(frame_pil, blur_radius=0.2, contrast=1.0, brightness=1.0, saturation=1.0):
    """
    Appliquer des effets post-decode sur une frame PIL.

    Args:
        frame_pil (PIL.Image): L'image décodée depuis les latents
        blur_radius (float): Rayon du flou gaussien
        contrast (float): Facteur de contraste (1.0 = inchangé)
        brightness (float): Facteur de luminosité (1.0 = inchangé)
        saturation (float): Facteur de saturation (1.0 = inchangé)

    Returns:
        PIL.Image: Image modifiée
    """
    # GaussianBlur simple
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Ajustements optionnels (contrast, brightness, saturation)
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    return frame_pil


def encode_images_to_latents_hybrid_safe(images, vae, device="cuda", latent_scale=LATENT_SCALE):
    """
    Encodage hybride pour conserver la fidélité et la richesse de détails.
    - Utilise un échantillon de la distribution (plus vivant que mean)
    - Clamp léger pour éviter débordements extrêmes mais garder micro-contrastes
    - Assure 4 channels si nécessaire pour compatibilité UNet/N3R
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample()  # <-- sample() pour micro-variations

    latents = latents * latent_scale
    latents = torch.clamp(latents, -5.0, 5.0)  # Clamp léger, pas de nan_to_num

    # Assurer 4 channels si nécessaire
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents

#---------------------------------------------------------------------------------------------

def save_debug_mask_green(mask_np, H, W, debug_dir, frame_counter=0, prefix="debug"):
    """
    Sauvegarde un masque débogué avec une couleur verte.

    :param mask_np: Masque binaire (numpy array 2D) à visualiser
    :param H: Hauteur de l'image
    :param W: Largeur de l'image
    :param debug_dir: Répertoire où sauvegarder les images de débogage
    :param frame_counter: Compteur de frames, utilisé pour nommer les fichiers
    :param prefix: Préfixe du nom du fichier
    """
    os.makedirs(debug_dir, exist_ok=True)

    # Création de l'image RGB en utilisant la couleur verte (canal vert)
    dbg_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Assigner le masque à la composante verte (channel 1)
    dbg_rgb[..., 1] = mask_np.astype(np.uint8)

    # Construction du chemin de sauvegarde pour l'image
    debug_image_path = os.path.join(debug_dir, f"{prefix}_{frame_counter}.png")

    # Sauvegarde de l'image
    Image.fromarray(dbg_rgb).save(debug_image_path)

    print(f"[DEBUG] Debug image saved to: {debug_image_path}")
#-----------------------------------------------------------------------------------------

import cv2
import numpy as np
import torch

def create_eye_mask_from_coords(
    eye_coords_dict,
    size,
    image_size=None,
    device="cuda",
    expand=0.15,
    frame_idx=0,
    debug=False,
    debug_dir=None
):
    """
    Crée un masque pour les yeux à partir des coordonnées (dict).
    - eye_coords_dict : liste de dictionnaires [{'eyes': [(x1, y1), (x2, y2)]}, ...]
    - size : tuple (H, W) du masque final
    - expand : fraction pour agrandir légèrement l'ellipse autour de l'œil
    """
    B = len(eye_coords_dict)  # Nombre d'éléments dans le dictionnaire
    H, W = size  # Taille du masque final
    mask = torch.zeros(B, 1, H, W, device=device)  # Masque initialisé avec des zéros

    # Mise à l'échelle si image_size est défini
    if image_size is not None:
        img_H, img_W = image_size
        scale_x = W / img_W
        scale_y = H / img_H
    else:
        scale_x = 1.0
        scale_y = 1.0

    for b in range(B):
        # Extraire les coordonnées des yeux
        eyes = eye_coords_dict[b].get("eyes")
        if eyes is None:
            continue

        # Itérer sur les deux yeux (si existants)
        for eye in eyes:
            cx, cy = eye
            # Mise à l'échelle des coordonnées pour le masque
            cx = int(cx * scale_x)
            cy = int(cy * scale_y)

            # Validation des coordonnées (en dehors des limites de l'image)
            if not (0 <= cx < W and 0 <= cy < H):
                print(f"⚠ Coordonnée de l'œil invalide après mise à l'échelle (hors limites): ({cx}, {cy})")
                continue

            # Ajustement des coordonnées pour les maintenir dans les limites du masque
            cx = max(0, min(cx, W-1))
            cy = max(0, min(cy, H-1))

            # Rayon dynamique ajusté pour l'ellipse
            radius = int(max(H, W) * 0.05 * (1 + expand))  # Rayon relatif à l'image
            max_radius = 20  # Limite de la taille maximale du rayon
            radius = min(radius, max_radius)  # Application de la limite maximale

            # Création du masque local pour l'œil
            mask_np = np.zeros((H, W), dtype=np.uint8)
            cv2.ellipse(mask_np, (cx, cy), (radius, radius), 0, 0, 360, color=255, thickness=-1)

            # Addition du masque local au masque global
            mask[b, 0] = torch.max(mask[b, 0], torch.from_numpy(mask_np / 255.0).to(device))

            mask = feather_inside_strict2(mask, radius=2, sigma=1.0)

        # Débogage
        if debug and debug_dir is not None:
            save_debug_mask_green(mask_np, H, W, debug_dir, frame_counter=frame_idx, prefix="create_eye_mask_from_coords")

    return mask



def create_mouth_mask_from_coords(
    mouth_coords_dict,
    size=(512, 512),
    image_size=None,
    device="cuda",
    expand=0.15,
    frame_idx=0,
    debug=False,
    debug_dir=None
):
    H, W = size
    mask = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)

    # --- scaling coords ---
    if image_size is not None:
        img_H, img_W = image_size
        scale_x = W / img_W
        scale_y = H / img_H
    else:
        scale_x, scale_y = 1.0, 1.0

    total_points = 0

    # =========================================================
    # 1. BUILD RAW MASK (NO BLUR, NO FEATHER HERE)
    # =========================================================
    for k, value in mouth_coords_dict.items():

        if isinstance(value, dict):
            mouth_points = value.get("mouth", [])
        else:
            mouth_points = value if isinstance(value, list) else []

        if not mouth_points:
            continue

        for mouth in mouth_points:

            if mouth is None or len(mouth) != 2:
                continue

            try:
                cx = float(mouth[0]) * scale_x
                cy = float(mouth[1]) * scale_y
            except Exception:
                continue

            # clamp safety (évite points hors image)
            if not (0 <= cx < W and 0 <= cy < H):
                continue

            radius = max(8, int(min(H, W) * 0.01))

            mask_np = np.zeros((H, W), dtype=np.uint8)

            cv2.circle(
                mask_np,
                (int(cx), int(cy)),
                radius,
                255,
                thickness=-1
            )

            new_mask = torch.from_numpy(mask_np).to(device).float() / 255.0
            new_mask = new_mask.unsqueeze(0).unsqueeze(0)

            # 🔥 accumulation SAFE (pas de dépassement implicite)
            mask = torch.maximum(mask, new_mask)

            total_points += 1

            # debug ponctuel (optionnel)
            if debug:
                print(f"[MOUTH] point={cx:.1f},{cy:.1f} r={radius}")

    # =========================================================
    # 2. NORMALISATION ROBUSTE
    # =========================================================
    if total_points == 0:
        return mask  # masque vide stable

    mask = mask / (mask.max() + 1e-6)
    mask = mask.clamp(0.0, 1.0)

    # léger gating pour éviter bruit de fond
    mask = (mask > 0.1).float()

    mask = feather_inside_strict2( mask, radius=2, sigma=1.0 )

    # =========================================================
    # 4. DEBUG FINAL
    # =========================================================
    if debug:
        print(
            f"[DEBUG MOUTH MASK] "
            f"points={total_points} "
            f"min={mask.min().item():.4f} "
            f"max={mask.max().item():.4f}"
        )

    # Débogage
    if debug and debug_dir is not None:
        save_debug_mask_green(mask_np, H, W, debug_dir, frame_counter=frame_idx, prefix="create_mouth_mask_from_coords")

    return mask.clamp(0.0, 1.0)

def create_face_mask_from_coords(
    face_coords_dict,
    size=(512, 512),
    image_size=None,
    device="cuda",
    expand=0.15,
    frame_idx=0,
    debug=False,
    debug_dir=None
):
    H, W = size
    mask = torch.zeros(1, 1, H, W, device=device, dtype=torch.float32)

    # Si image_size est défini, appliquer une mise à l'échelle
    if image_size is not None:
        img_H, img_W = image_size
        scale_x = W / img_W
        scale_y = H / img_H
    else:
        scale_x = 1.0
        scale_y = 1.0

    # Debug: Affichage des coordonnées de face et des échelles
    print(f"[DEBUG] Image size: {image_size}, Mask size: {(H, W)}")
    print(f"[DEBUG] Scale X: {scale_x}, Scale Y: {scale_y}")

    for key, coords in face_coords_dict.items():
        if not coords:
            continue

        # coords peut être un dict (par exemple pour le nez)
        if isinstance(coords, dict) and "center" in coords:
            coords_list = [coords["center"]]
        else:
            coords_list = [tuple(p) for p in coords]

        # Filtrer uniquement les coordonnées valides
        valid_coords_list = []
        for p in coords_list:
            try:
                x = float(p[0]) * scale_x
                y = float(p[1]) * scale_y
                valid_coords_list.append([x, y])
                # Debug: Affichage des coordonnées après mise à l'échelle
                print(f"[DEBUG] Coordinate {p} -> Scaled: ({x}, {y})")
            except (ValueError, TypeError):
                print(f"⚠ Warning: coordonnée ignorée car invalide: {p}")

        if not valid_coords_list:
            continue

        pts_px = np.array(valid_coords_list, dtype=np.int32)

        x_min, y_min = pts_px.min(axis=0)
        x_max, y_max = pts_px.max(axis=0)

        # Si nous avons un seul point (comme pour la bouche), remplacez la largeur et la hauteur par une valeur par défaut
        if x_min == x_max:
            w = 30  # Largeur par défaut
        else:
            w = int((x_max - x_min) * (1 + expand))

        if y_min == y_max:
            h = 30  # Hauteur par défaut
        else:
            h = int((y_max - y_min) * (1 + expand))

        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        dx = pts_px[-1][0] - pts_px[0][0]
        dy = pts_px[-1][1] - pts_px[0][1]
        angle = int(np.degrees(np.arctan2(dy, dx)))

        # Debug: Affichage de la taille et de l'angle de l'ellipse
        print(f"[DEBUG] Mask key: {key}")
        print(f"[DEBUG] cx: {cx}, cy: {cy}, width: {w}, height: {h}, angle: {angle}")

        mask_np = np.zeros((H, W), dtype=np.uint8)

        # Masques pour les yeux et la bouche
        if key == "mouth" or key == "eyes":
            # Ajustement du rayon pour les yeux et la bouche
            max_radius = max(H, W) * 0.1  # Rayon plus grand pour les yeux et la bouche
            radius = min(int(max_radius * (1 + expand)), 40)  # Limiter la taille du rayon
            print(f"[DEBUG] Eye/Mouth mask radius: {radius}")  # Debug du rayon
            cv2.ellipse(mask_np, (cx, cy), (radius, radius), angle=angle, startAngle=0, endAngle=360, color=255, thickness=-1)
        else:
            # Rayon pour d'autres parties (comme le nez ou la tête)
            radius = max(w, h) // 2
            print(f"[DEBUG] Face mask radius: {radius}")  # Debug du rayon
            cv2.circle(mask_np, (cx, cy), radius, color=255, thickness=-1)

        mask += torch.from_numpy(mask_np / 255.0).to(device).unsqueeze(0).unsqueeze(0)

        mask = feather_outside_only_alpha2(mask, radius=3, sigma=1.2)

        # Debug
        if debug and debug_dir is not None:
            save_debug_mask_green(mask_np, H, W, debug_dir, frame_counter=frame_idx, prefix=f"create_{key}_mask")

    return mask.clamp(0, 1)
#----------------------------------------------------------------------------------------------------------

def log_latents_stats(latents, label="LATENTS"):
    with torch.no_grad():
        min_val = latents.min().item()
        max_val = latents.max().item()
        mean_val = latents.mean().item()
        std_val = latents.std().item()
        has_nan = torch.isnan(latents).any().item()

        print(f"[DEBUG] {label} | min={min_val:.4f} max={max_val:.4f} mean={mean_val:.4f} std={std_val:.4f} NaN={has_nan}")


def encode_images_to_latents_hybrid_pro(
    images,
    vae,
    eye_coords,     # [(x1, y1), (x2, y2), ...]
    mouth_coords,   # [(x, y), ...]
    ear_coords,     # [(x1, y1), (x2, y2)]
    nose_coords,    # {'center': (x, y), ...}
    device="cuda",
    latent_scale=1.0,
    creative_mode=None,   # 👈 NEW
    frame_idx=0,
    total_frames=1,
    debug=False,
    debug_dir=None
):
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)
    B, C, H, W = images.shape

    img_H, img_W = H, W

    # --- Encodage latents ---
    with torch.no_grad():
        dist = vae.encode(images).latent_dist
        latents = 0.88 * dist.mean + 0.12 * dist.sample()

    # 🔥 scaling OFFICIEL Stable Diffusion
    latents = latents * 0.18215

    # 🔧 ton réglage perso (OK après)
    latents = latents * latent_scale

    # --- Normalisation douce + soft clamp ---
    std = latents.std(dim=(1,2,3), keepdim=True).clamp(min=0.5, max=1.5)
    latents_norm = latents / std

    latents_centered = latents - latents.mean(dim=(1,2,3), keepdim=True)
    # --- Préparer les dicts pour les fonctions de masque ---
    eye_coords_dict = {0: {"eyes": [list(map(float, p)) for p in eye_coords]}}
    mouth_coords_dict = {0: {"mouth": [list(map(float, p)) for p in mouth_coords]}}
    nose_coords_dict = {0: {"nose": [list(map(float, p)) for p in nose_coords]}}

    face_coords_dict = {
        "eyes": [list(map(float, p)) for p in eye_coords],
        "mouth": [list(map(float, p)) for p in mouth_coords],
        "ears": [list(map(float, p)) for p in ear_coords],
        "nose": [list(map(float, p)) for p in nose_coords],
    }

    print("[DEBUG][FACE MASK INPUT TYPE]", type(face_coords_dict))

    for k, v in face_coords_dict.items():
        print(f"[DEBUG][FACE MASK KEY={k}] type={type(v)} value={v}")

    # --- Masques ---
    mask_face = create_face_mask_from_coords(face_coords_dict, size=(latents.shape[2], latents.shape[3]), image_size=(img_H, img_W), device=device, frame_idx=frame_idx, debug=debug, debug_dir=debug_dir)
    mask_eyes = create_eye_mask_from_coords(eye_coords_dict, size=(latents.shape[2], latents.shape[3]), image_size=(img_H, img_W), device=device, frame_idx=frame_idx, debug=debug, debug_dir=debug_dir)
    mask_mouth = create_mouth_mask_from_coords(mouth_coords_dict, size=(latents.shape[2], latents.shape[3]), image_size=(img_H, img_W), device=device, frame_idx=frame_idx, debug=debug, debug_dir=debug_dir)

    # --- Compatibilité channels ---
    mask_face = mask_face.expand(B, latents.shape[1], -1, -1)
    mask_eyes = mask_eyes.expand(B, latents.shape[1], -1, -1)
    mask_mouth = mask_mouth.expand(B, latents.shape[1], -1, -1)

    # Fusion des latents directement sans la notion de "remaining"
    mask_background = (1.0 - mask_face).clamp(0, 1)
    mask_background = torch.clamp(mask_background, 0, 1)

    # base propre  🔥 Important
    latents = latents_norm
    # signal de détail SAFE
    detail = latents_norm - latents_norm.mean(dim=(2,3), keepdim=True)
    # masques
    mask_detail = torch.clamp(mask_eyes + mask_mouth, 0, 1)
    mask_background = (1.0 - mask_face).clamp(0, 1)
    # 🔥 boost local UNIQUEMENT
    latents = latents + 0.25 * mask_eyes * detail
    latents = latents + 0.18 * mask_mouth * detail
    # 🔥 léger enrichissement fond
    latents = latents + 0.12 * mask_background * detail
    # 🔥 LOCK visage (clé)
    face_lock = 0.35
    latents = latents * (1 - face_lock * mask_face) + latents_norm * (face_lock * mask_face)
    std = latents.std(dim=(1,2,3), keepdim=True).clamp(min=0.5, max=1.5)
    latents = latents / std

    # =========================================================
    # 🎨 CREATIVE MODES + DEBUG
    # =========================================================
    if creative_mode is not None:

        t = frame_idx / max(total_frames, 1)
        t_tensor = torch.tensor(t, device=latents.device)

        log_latents_stats(latents, "BEFORE CREATIVE")

        if creative_mode == "cinematic":
            # Bruit léger et fluide à base de cosinus
            noise = torch.randn_like(latents) * 0.03  # Bruit de base léger
            # Modulateur cosinusoïdal pour introduire de la fluidité au bruit
            t = frame_idx / max(total_frames, 1)
            cos_mod = (torch.cos(torch.tensor(t * 3.14)) + 1) / 2  # Valeur entre 0 et 1
            # Appliquer un bruit léger, modulé par le cosinus pour donner une dynamique douce
            latents = latents + noise * cos_mod  # Le bruit varie en fonction du temps
            log_latents_stats(latents, "CINEMATIC")

        elif creative_mode == "cyber":
            # --- temps ---
            t = frame_idx / max(total_frames, 1)
            t_tensor = torch.tensor(t, device=latents.device)

            # --- base stable ---
            base = latents

            # --- signal de détail stable ---
            detail = latents_norm - latents_norm.mean(dim=(2,3), keepdim=True)

            # =========================================================
            # ⚡ 1. ÉNERGIE ÉLECTRIQUE (high-frequency boost contrôlé)
            # =========================================================
            hf_detail = detail - torch.nn.functional.avg_pool2d(detail, kernel_size=3, stride=1, padding=1)

            latents = latents + 0.18 * hf_detail

            # =========================================================
            # 🌐 2. PULSATION CYBER (néon / énergie)
            # =========================================================
            pulse = (torch.sin(t_tensor * 6.28) + 1) * 0.5  # 0 → 1

            latents = latents + pulse * 0.08 * detail

            # =========================================================
            # ⚡ 3. BRUIT "ÉLECTRIQUE" (pas gaussian pur)
            # =========================================================
            noise = torch.randn_like(latents)

            # structuration du bruit (donne un effet “signal électrique”)
            noise = noise - torch.roll(noise, shifts=1, dims=-1)

            latents = latents + noise * 0.02 * (0.5 + 0.5 * pulse)

            # =========================================================
            # 👁 4. ACCENT ZONES IMPORTANTES (yeux / bouche)
            # =========================================================
            mask_focus = torch.clamp(mask_eyes + mask_mouth, 0, 1)

            latents = latents + 0.15 * mask_focus * detail

            # =========================================================
            # 🧠 5. STABILISATION VISAGE (évite dérive)
            # =========================================================
            face_lock = 0.35
            latents = (
                latents * (1 - face_lock * mask_face) +
                base * (face_lock * mask_face)
            )

            log_latents_stats(latents, "CYBER")

        elif creative_mode == "pro":
            # --- 1. léger stabilisation globale ---
            latents = latents * 0.88 + latents_norm * 0.12
            # --- 2. signal de détail unique (référence stable) ---
            detail = latents_norm - latents_norm.mean(dim=(2,3), keepdim=True)
            # --- 3. masques combinés ---
            mask_face = mask_face
            mask_detail = torch.clamp(mask_eyes + mask_mouth, 0, 1)
            mask_bg = (1.0 - mask_face)

            # --- 4. modulation locale unifiée ---
            local_gain = (
                0.30 * mask_eyes +
                0.22 * mask_mouth +
                0.12 * mask_bg
            )
            latents = latents + local_gain * detail

            # --- 5. stabilisation visage (UNE seule fois) ---
            face_lock = 0.4
            latents = latents * (1 - face_lock * mask_face) + latents_norm * (face_lock * mask_face)

            # --- 6. normalisation finale légère ---
            latents = latents / (latents.std(dim=(1,2,3), keepdim=True).clamp(0.5, 1.5))
            log_latents_stats(latents, "QUALITY_PRO")

        elif creative_mode == "soft":
            noise = torch.randn_like(latents) * 0.03
            # Variation douce du flou avec modulation temporelle
            blur_strength = 0.5 + 0.5 * torch.sin(t_tensor * 3.14)
            latents = latents * (1 - 0.25 * mask_face) + latents_norm * 0.25 * mask_face
            latents = latents + noise * blur_strength
            log_latents_stats(latents, "SOFT")

        elif creative_mode == "anime":
            # Appliquer une amplification douce à l'aide d'une fonction sigmoïde
            mask_eyes_sigmoid = 1 / (1 + torch.exp(-8 * (mask_eyes - 0.5)))  # Douce transition pour les yeux
            mask_mouth_sigmoid = 1 / (1 + torch.exp(-8 * (mask_mouth - 0.5)))  # Douce transition pour la bouche
            # Amplification subtile
            latents = latents * (1 + 0.18 * mask_eyes_sigmoid)  # Légère réduction de l'intensité
            latents = latents * (1 + 0.12 * mask_mouth_sigmoid)  # Légère réduction pour la bouche
            # 🎨 ajout de détails directionnels
            latents = latents + 0.25 * mask_eyes * (latents_norm - latents)
            # 💫 micro-variation temporelle
            pulse = torch.sin(torch.tensor(t * 6.28)).to(latents.device) * 0.1
            latents = latents + pulse * mask_eyes
            # gain:
            feature_gain = 1 + 0.25 * mask_eyes + 0.15 * mask_mouth
            latents = latents * feature_gain
            log_latents_stats(latents, "ANIME")


        elif creative_mode == "hybrid":
            # Bruit léger et dynamique avec modulation cosinusoïdale
            noise = torch.randn_like(latents) * 0.03  # Bruit de base léger
            # Modulateur cosinusoïdal pour introduire de la fluidité au bruit
            t = frame_idx / max(total_frames, 1)
            cos_mod = (torch.cos(torch.tensor(t * 3.14)) + 1) / 2  # Valeur entre 0 et 1
            # Appliquer un bruit léger et modulé en fonction du temps
            latents = latents + noise * cos_mod  # Le bruit varie de manière fluide au fil du temps
            # Amplification de certaines zones, comme les yeux et la bouche
            mask_eyes_sigmoid = 1 / (1 + torch.exp(-8 * (mask_eyes - 0.5)))  # Douce transition pour les yeux
            mask_mouth_sigmoid = 1 / (1 + torch.exp(-8 * (mask_mouth - 0.5)))  # Douce transition pour la bouche
            # Amplification subtile dans les zones des yeux et de la bouche
            latents = latents * (1 + 0.15 * mask_eyes_sigmoid)
            latents = latents * (1 + 0.12 * mask_mouth_sigmoid)
            # Flou directionnel en fonction du temps
            blur_strength = 0.5 + 0.5 * torch.sin(t_tensor * 3.14)
            latents = latents * (1 - 0.25 * mask_face) + latents_norm * 0.25 * mask_face  # Flou plus fort autour du visage
            latents = latents + noise * blur_strength  # Appliquer le bruit et le flou
            # Déplacement léger des latents (effet de fluctuation)
            pulse = torch.sin(torch.tensor(t * 6.28)).to(latents.device) * 0.1
            latents = latents + pulse * mask_eyes  # Pulsation sur les yeux pour plus de dynamisme

            log_latents_stats(latents, "HYBRID")

        elif creative_mode == "smooth":
            # Appliquer un léger bruit au début et augmenter progressivement l'intensité
            noise = torch.randn_like(latents) * 0.02  # Bruit modéré au début
            t = frame_idx / max(total_frames, 1)
            cos_mod = (torch.cos(torch.tensor(t * 3.14)) + 1) / 2  # Modulation douce
            # Graduellement augmenter le bruit au fil du temps
            latents = latents + noise * cos_mod
            # Application progressive du flou et de la saturation sur le visage, doucement
            latents = latents * (1 - 0.25 * mask_face) + latents_norm * 0.25 * mask_face

            log_latents_stats(latents, "SMOOTH")

        elif creative_mode == "distorsion":

            # 1. Modulation cosinusoïdale rapide pour des variations extrêmes
            t = frame_idx / max(total_frames, 1)
            cos_mod = (torch.cos(torch.tensor(t * 8.0)) + 1) / 2  # Modulation très rapide
            # 2. Flou exponentiel et bruit sur les zones périphériques
            blur_strength = 1.5 + 0.5 * torch.sin(t_tensor * 4.0)  # Flou rapide et irrégulier
            latents = latents * (1 - 0.5 * mask_face) + latents_norm * 0.5 * mask_face  # Flou agressif sur les bords
            # 3. Distorsion géométrique violente
            shift_x = int(6 * torch.sin(t_tensor * 6).item())  # Décalage important dans la dimension X
            shift_y = int(6 * torch.cos(t_tensor * 6).item())  # Décalage important dans la dimension Y
            if shift_x != 0 or shift_y != 0:
                latents = torch.roll(latents, shifts=(shift_x, shift_y), dims=(2, 3))  # Déplacement violent des latents
            # 8. Effet de pulsation rapide
            pulse = torch.sin(torch.tensor(t * 12.0)).to(latents.device) * 0.2
            latents = latents + pulse * mask_eyes  # Effet de pulsation intense sur les yeux
            log_latents_stats(latents, "DISTORSION")

    # --- sécurité finale ---
    latents = latents.clamp(-3, 3)

    print("LATENTS RAW:", latents.min().item(), latents.max().item(), latents.std().item())
    log_latents_stats(latents, "FINAL CLAMP")

    # --- Assurer 4 channels si nécessaire ---
    if latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents

# Qualité de l'image supérieur !'
def encode_images_to_latents_hybrid(images, vae, device="cuda", latent_scale=LATENT_SCALE):
    """
    Encodage hybride STABLE :
    - mix mean + sample pour micro-variations contrôlées
    - normalisation douce (pas de clamp brutal)
    - préserve les couleurs
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        dist = vae.encode(images).latent_dist

        mean = dist.mean
        sample = dist.sample()

        # 🔥 Mélange contrôlé (clé !)
        latents = 0.85 * mean + 0.15 * sample

    latents = latents * latent_scale

    # 🔥 Normalisation douce (au lieu de clamp brutal)
    latents = latents / (latents.std(dim=(1,2,3), keepdim=True) + 1e-6)

    # Assurer 4 channels si nécessaire
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents


def encode_images_to_latents_safe(images, vae, device="cuda", latent_scale=0.18215):
    """
    Encode une image en latents VAE en gardant la stabilité et le contraste.
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mean  # moyenne pour stabilité

    latents = latents * latent_scale
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    return latents


def save_frames_as_video_from_folder(
    folder_path,
    out_path,
    pattern=None,
    fps=12,
    upscale_factor=1
):
    if pattern is None:
        pattern = "*.png"

    folder_path = Path(folder_path)
    images = sorted(folder_path.glob(pattern))
    if not images:
        raise ValueError(f"Aucune image trouvée dans {folder_path}")

    tmp_dir = folder_path / "_tmp_upscaled"
    tmp_dir.mkdir(exist_ok=True)

    # Upscale
    for idx,img_path in enumerate(images):
        img = Image.open(img_path)
        if upscale_factor != 1:
            img = img.resize((img.width*upscale_factor,img.height*upscale_factor), Image.BICUBIC)
        tmp_file = tmp_dir / f"frame_{idx:05d}.png"
        img.save(tmp_file)

    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(tmp_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path)
    ]

    print("⚡ Génération vidéo avec ffmpeg…")
    subprocess.run(cmd, check=True)
    print(f"🎬 Vidéo sauvegardée : {out_path}")

    # Nettoyage
    for f in tmp_dir.glob("*.png"):
        f.unlink()
    tmp_dir.rmdir()


def encode_images_to_latents_nuanced_v1(images, vae, device="cuda", latent_scale=LATENT_SCALE):
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




def encode_images_to_latents_nuanced(images, vae, unet, device="cuda", latent_scale=LATENT_SCALE):
    """
    Encode une image en latents VAE, en préservant nuances et contraste,
    et redimensionne dynamiquement pour correspondre à la taille attendue par le UNet.
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # Encoder l'image → latents
        latents = vae.encode(images).latent_dist.mean  # moyenne pour stabilité

    # Appliquer le scaling
    latents = latents * latent_scale

    # Sécurité NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # Forcer 4 canaux si nécessaire
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    # 🔹 Redimensionner dynamiquement pour correspondre au UNet
    # On regarde la taille attendue à partir de l'UNet (ou ses skip connections)
    # Supposons que le UNet a un module `config` avec "sample_size" ou attention_resolutions
    try:
        # Si UNet a `sample_size` ou autre attribut
        target_H = getattr(unet.config, "sample_size", latents.shape[2])
        target_W = getattr(unet.config, "sample_size", latents.shape[3])
    except AttributeError:
        # fallback : garder la taille actuelle
        target_H, target_W = latents.shape[2], latents.shape[3]

    # Interpolation bilinéaire pour adapter la taille
    if (latents.shape[2], latents.shape[3]) != (target_H, target_W):
        latents = TF.interpolate(latents, size=(target_H, target_W), mode="bilinear", align_corners=False)
        print(f"[DEBUG] Latents resized to ({target_H}, {target_W})")

    return latents

# ------------------- ENCODE -------------------

def encode_images_to_latents_target(images, vae, device="cuda", latent_scale=LATENT_SCALE, target_size=64):
    """
    Encode une image en latents VAE, compatible MiniSD/AnimateDiff ultra-light (~2Go VRAM)
    - garde la dynamique et contraste
    - clamp minimal pour sécurité
    - force 4 canaux
    - resize latents à target_size (MiniSD)
    """
    images = images.to(device=device, dtype=torch.float32)
    vae = vae.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mean  # moyenne pour plus de stabilité

    # Appliquer le scaling
    latents = latents * latent_scale

    # Sécurité NaN / Inf
    latents = torch.nan_to_num(latents, nan=0.0, posinf=5.0, neginf=-5.0)

    # Forcer 4 canaux si nécessaire
    if latents.ndim == 4 and latents.shape[1] == 1:
        latents = latents.repeat(1, 4, 1, 1)

    # Redimensionner à target_size x target_size
    if latents.shape[2] != target_size or latents.shape[3] != target_size:
        latents = torch.nn.functional.interpolate(
            latents, size=(target_size, target_size), mode="bilinear", align_corners=False
        )

    return latents



# ------------------- DECODE -------------------


def decode_latents_ultrasafe_blockwise(
    latents, vae,
    block_size=32, overlap=16,
    gamma=1.0, brightness=1.0,
    contrast=1.0, saturation=1.0,
    device="cuda", frame_counter=0, output_dir=Path("."),
    epsilon=1e-6,
    latent_scale_boost=1.0  # boost léger pour récupérer les nuances
):
    """
    Décodage ultra-safe par blocs des latents en image PIL.
    Optimisé pour préserver les nuances de couleur et réduire l'effet "photocopie".
    """

    # 🔹 Correctif : forcer le VAE sur le bon device et en float32
    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()

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
