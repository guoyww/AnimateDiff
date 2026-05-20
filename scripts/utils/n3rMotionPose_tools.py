#********************************************
# n3rOpenPose_utils.py
#********************************************
import torch
from diffusers import ControlNetModel
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import traceback
import torchvision.utils as vutils


#---- Dilation d'un mask ---- Version stable'
def dilate_mask(mask, kernel_size=5):
    """
    Appliquer une dilatation au masque pour étendre la zone où le bruit peut être appliqué,
    mais en évitant que le bruit se propage au-delà de la zone désirée.
    """
    # Assurez-vous que mask est un tensor 4D avant d'utiliser max_pool2d
    if mask.dim() == 2:  # Si mask est 2D
        mask = mask.unsqueeze(0).unsqueeze(0)  # Ajouter deux dimensions (batch et canaux)
    elif mask.dim() == 3:  # Si mask est 3D, ajoutez juste la dimension du batch
        mask = mask.unsqueeze(0)

    # Appliquer la dilatation
    dilated_mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    # Revenir à la forme originale (retirer les dimensions batch et canaux si nécessaire)
    dilated_mask = dilated_mask.squeeze(0).squeeze(0)  # Enlever les dimensions ajoutées (batch et canaux)

    return dilated_mask

def save_debug_mask_scale(
    mask: torch.Tensor,
    debug_dir: str,
    frame_counter: int,
    name: str = "mask",
    scale: int = 4,
    verbose: bool = True,
):
    try:
        os.makedirs(debug_dir, exist_ok=True)

        # =========================
        # SAFE NORMALIZATION
        # =========================
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)

        if mask.dim() == 4:
            m = mask[0, 0]
        elif mask.dim() == 3:
            m = mask[0]
        elif mask.dim() == 2:
            m = mask
        else:
            raise ValueError(f"[save_debug_mask] invalid shape: {mask.shape}")

        m = m.detach().cpu().numpy()

        if verbose:
            print(f"[DEBUG][{name.upper()} MASK] mean={m.mean():.6f} max={m.max():.6f}")

        img = (np.clip(m, 0, 1) * 255).astype(np.uint8)

        h, w = img.shape
        img = cv2.resize(
            img,
            (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST
        )

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        path = os.path.join(debug_dir, f"{name}_{frame_counter:05d}.png")
        cv2.imwrite(path, img)

        if verbose:
            print(f"[DEBUG] {name} saved: {path}")

        return path

    except Exception as e:
        print(f"[WARN] save_debug_mask failed ({e})")
        return None


def dilate_mask(mask, kernel_size=3):
    # Vérifier que le masque est de forme [B, C, H, W] et le redimensionner si nécessaire
    if mask.dim() == 3:  # [C, H, W], ajouter un batch fictif
        mask = mask.unsqueeze(0)  # Ajouter une dimension de batch
    elif mask.dim() == 2:  # [H, W], ajouter un batch et un canal fictifs
        mask = mask.unsqueeze(0).unsqueeze(0)  # Ajouter les dimensions [B, C, H, W]

    # Créer un noyau de dilatation (kernel_size x kernel_size)
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # Appliquer la dilatation via la convolution (padding pour éviter les effets de bord)
    dilated_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)

    # Convertir le résultat en un masque binaire (0 ou 1)
    dilated_mask = (dilated_mask > 0).float()

    return dilated_mask.squeeze(1)  # Retirer la dimension du canal, revenir à [B, H, W]

# Exemple d'application dans la fonction :
#valid_mask_dilated = dilate_mask(valid_mask, kernel_size=5)


#(4D : [B, C, H, W])
def dilate_mask_4D(mask, kernel_size=3):
    # Applique une dilatation (l'élément le plus proche dans la fenêtre est choisi)
    # Le kernel est une matrice de 1s de taille (kernel_size x kernel_size)
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # Dilatation via la convolution (padding pour éviter les effets de bord)
    dilated_mask = F.conv2d(mask.unsqueeze(1), kernel, padding=kernel_size//2)

    # Convertit en mask binaire (0 ou 1)
    dilated_mask = (dilated_mask > 0).float()

    return dilated_mask.squeeze(1)  # Retirer la dimension de canaux

# Exemple d'application dans la fonction :
#valid_mask_dilated = dilate_mask(valid_mask, kernel_size=5)


def feather_outside_only_alpha2(mask: torch.Tensor, radius: int = 2, sigma: float = 1.0):
    """
    Adoucit uniquement l'extérieur d'un masque (feathering glow externe)
    de manière stable, avec recadrage pour éviter les erreurs de dimensions.

    Args:
        mask: Tensor [B,1,H,W], valeurs 0..1
        radius: int, padding pour étendre le blur
        sigma: float, écart-type du blur gaussien

    Returns:
        Tensor [B,1,H,W] adouci à l'extérieur
    """
    B, C, H, W = mask.shape
    device = mask.device

    # Inverse du masque pour travailler sur l'extérieur
    mask_inv = 1.0 - mask

    # Padding pour ne pas perdre les bords
    mask_inv_pad = F.pad(mask_inv, (radius, radius, radius, radius), mode='reflect')

    # Création du kernel gaussien 2D
    kernel_size = radius * 2 + 1
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)

    # Convolution 2D pour blur
    blur = F.conv2d(mask_inv_pad, kernel, padding=0, groups=C)

    # Retirer le padding
    blur = blur[:, :, radius:radius+H, radius:radius+W]

    # ⚠️ Recadrage exact pour éviter les problèmes de dimension
    blur = F.interpolate(blur, size=(H, W), mode='bilinear', align_corners=False)

    # Re-inverser pour récupérer la zone originale avec bord adouci
    mask_feathered = 1.0 - blur

    # Clamp 0..1
    mask_feathered = mask_feathered.clamp(0.0, 1.0)

    return mask_feathered


def feather_inside_strict2(mask: torch.Tensor, radius: int = 2, blur_kernel: int = 3, sigma: float = 1.0):
    """
    Adoucit uniquement l'intérieur du masque (feathering interne strict)
    de manière stable, avec recadrage pour éviter les erreurs de dimensions.

    Args:
        mask: Tensor [B,1,H,W], valeurs 0..1
        radius: int, padding autour pour le blur
        blur_kernel: int, taille du kernel gaussien (impair)
        sigma: float, écart-type du blur gaussien

    Returns:
        Tensor [B,1,H,W] adouci à l'intérieur
    """
    B, C, H, W = mask.shape
    device = mask.device

    # Padding pour ne pas perdre les bords
    mask_pad = F.pad(mask, (radius, radius, radius, radius), mode='reflect')

    # Création du kernel gaussien 2D
    coords = torch.arange(blur_kernel, dtype=torch.float32, device=device) - blur_kernel // 2
    x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, blur_kernel, blur_kernel).repeat(C, 1, 1, 1)

    # Convolution 2D pour blur
    mask_blur = F.conv2d(mask_pad, kernel, padding=0, groups=C)

    # Retirer padding
    mask_blur = mask_blur[:, :, radius:radius+H, radius:radius+W]

    # ⚠️ Recadrage exact pour éviter les problèmes de dimension
    mask_blur = F.interpolate(mask_blur, size=(H, W), mode='bilinear', align_corners=False)

    # Clamp 0..1
    mask_blur = mask_blur.clamp(0.0, 1.0)

    return mask_blur


def smooth_noise(grid, frame, scale=0.05, time_scale=0.1):
    # grid: [B,H,W,2]
    x = grid[..., 0]
    y = grid[..., 1]

    noise = (
        torch.sin(x * scale + frame * time_scale) *
        torch.cos(y * scale * 1.3 + frame * time_scale * 0.8)
    )

    noise += (
        torch.sin((x + y) * scale * 0.7 + frame * time_scale * 1.5)
    ) * 0.5

    return noise


def feather_outside_only_alpha(mask, radius=5, sigma=2.0):
    """
    Ajoute une bande floue uniquement à l'extérieur du masque.

    mask: [B,1,H,W] (0 ou 1)
    radius: taille de la bande extérieure
    sigma: intensité du flou

    Retourne un masque avec transition douce extérieure uniquement.
    """

    B, C, H, W = mask.shape

    # -------------------- Dilatation contrôlée --------------------
    k = 2 * radius + 1

    # padding correct pour garder la taille
    pad = radius

    dilated = F.max_pool2d(
        mask,
        kernel_size=k,
        stride=1,
        padding=pad
    )

    # -------------------- Bande extérieure --------------------
    band = (dilated - mask).clamp(0, 1)

    # -------------------- Flou de la bande uniquement --------------------
    if sigma > 0:
        band = gaussian_blur_tensor(
            band,
            kernel_size=2 * int(2 * sigma) + 1,
            sigma=sigma
        )

    # -------------------- Reconstruction --------------------
    # IMPORTANT : on ne touche PAS l'intérieur
    out = mask + band * (1 - mask)

    return torch.clamp(out, 0, 1)



def debug_draw_openpose_skeleton(
    keypoints_tensor,
    debug_dir,
    frame_counter,
    pose_full_image=None,
    image_size=None
):
    os.makedirs(debug_dir, exist_ok=True)

    # =========================
    # SIZE RESOLUTION
    # =========================
    if pose_full_image is not None:
        B, C, H, W = pose_full_image.shape
    elif image_size is not None:
        H, W = image_size
    else:
        raise ValueError("Need pose_full_image or image_size")

    keypoints = keypoints_tensor[0].detach().cpu().numpy()

    pose_img = np.zeros((H, W, 3), dtype=np.uint8)  # ❗ RGB ONLY (important)

    def to_pixel(x, y):
        return int(x * W), int(y * H)

    """
    'right_shoulder': 2,
            'right_elbow': 3,
            'right_wrist': 4,
            'left_shoulder': 5,
            'left_elbow': 6,
            'left_wrist': 7,
    'mouth_left': 48,    # coin gauche des lèvres supérieures
            'mouth_right': 49,   # coin droit des lèvres supérieures
            'mouth_top': 50,     # index approximatif du haut de la bouche (à ajuster selon ton keypoints)
            'mouth_bottom': 51,  # index approximatif du bas de la bouche
    """

    COLORS = {
        "head": (255, 0, 255),  # 'nose': 0, 'right_ear': 16, 'left_ear': 17, 'mouth': 18,
        "eyes": (0, 0, 255),    # 'right_eye': 14, 'left_eye': 15,
        "nose": (128, 0, 128),
        "arms_r": (0, 200, 0),
        "arms_l": (0, 255, 0),
        "torso": (255, 0, 0),  # 'neck': 1,  'chin': 21, 'left_side_neck': 22, 'right_side_neck': 23, 'anchor_neck': 24,
        "hip": (255, 64, 64), #'right_hip': 8, 'left_hip': 11,
        "legs_r": (0, 255, 255),
        "legs_l": (0, 200, 255),
        "bouche": (0, 128, 255), # orange
        "nez": (0, 64, 255), # orange
        "cheveux": (255, 64, 255), # violet
        "front": (255, 128, 255), # violet clair
        "default": (200, 200, 200)
    }
    """
            'hair_root': 25,

            'hair_left': 26,
            'hair_right': 27,
            'hair_top': 28, # millieu centre
            'hair_top_left': 29,
            'hair_top_right': 30,

            'left_top_hair1': 31, # premier point
            'left_top_hair2': 32, # Point gauche au sommet, 2ème point centre gauche
            'left_top_hair3': 33, # point le plus eloigné

            'right_top_hair1': 34, # premier point
            'right_top_hair2': 35, # Point gauche au sommet, 2ème point centre droit
            'right_top_hair3': 36, # point le plus eloigné

            'top_hair1': 37,
            'top_hair2': 38,
            'top_hair3': 39,

            'front_left_1': 52, # front gauche 1
            'front_left_2': 53, # front gauche 2
            'front_m': 54, # front milleu
            'front_right_1': 55, # front droit 1
            'front_right_2': 56, # front droit 2

    """
    # =========================
    # POINTS
    # =========================
    for i, (x, y, conf) in enumerate(keypoints):
        if conf < 0.05:
            continue

        px, py = to_pixel(x, y)

        x1, x2 = max(px-3, 0), min(px+3, W)
        y1, y2 = max(py-3, 0), min(py+3, H)

        if i in [14,15]:
            color = COLORS["eyes"]
        elif i == 0:
            color = COLORS["nose"]
        elif i in [16,17]:
            color = COLORS["head"]
        elif i in [2,3,4,19]:
            color = COLORS["arms_r"]
        elif i in [5,6,7,20]:
            color = COLORS["arms_l"]
        elif i in [1,21,22,23,24]:
            color = COLORS["torso"]
        elif i in [8,11]:
            color = COLORS["hip"]
        elif i in [9,10]:
            color = COLORS["legs_r"]
        elif i in [12,13]:
            color = COLORS["legs_l"]
        elif i in [40,41,18]:
            color = COLORS["bouche"]
        elif i in [42,43,0]:
            color = COLORS["nez"]
        elif i in [26,27,28,29,30,31,32,33,34,35,36]:
            color = COLORS["nez"]
        else:
            color = COLORS["default"]

        cv2.circle(pose_img, (px, py), 3, color, -1)

    # =========================
    # SKELETON
    # =========================
    skeleton = [
        (0,1),(1,19),(19,2),(2,3),(3,4),
        (1,20),(20,5),(5,6),(6,7),
        (1,8),(1,11),(0,43),(0,42),(40,18),(18,41),
        (14,0),(15,0),(16,15),(17,14),
        (8,9),(9,10),(11,12),(12,13),
        (21,22),(21,23),(21,24),
        (28,34),(28,31),(31,32),(32,33),(34,35),(35,36),
        (54,52),(54,55),(52,53),(55,56)
    ]

    for i, j in skeleton:
        xi, yi, ci = keypoints[i]
        xj, yj, cj = keypoints[j]

        if ci < 0.05 or cj < 0.05:
            continue

        p1 = to_pixel(xi, yi)
        p2 = to_pixel(xj, yj)

        if i in [2,3,4,19]:
            color = COLORS["arms_r"]
        elif i in [5,6,7,20]:
            color = COLORS["arms_l"]
        elif i in [1,21,22,23,24]:
            color = COLORS["torso"]
        elif i in [8,11]:
            color = COLORS["hip"]
        elif i in [9,10]:
            color = COLORS["legs_r"]
        elif i in [12,13]:
            color = COLORS["legs_l"]
        elif i in [42,43,0]:
            color = COLORS["nez"]
        elif i in [40,41,18]:
            color = COLORS["bouche"]
        elif i in [52,53,54,55,56]:
            color = COLORS["front"]
        else:
            color = COLORS["default"]

        thickness = 3 if i in [1,8,11] else 2
        cv2.line(pose_img, p1, p2, color, thickness)

    save_path = f"{debug_dir}/skeleton_{frame_counter:05d}.png"
    cv2.imwrite(save_path, pose_img)

    print(f"[DEBUG] Skeleton saved: {save_path}")



# sigma correspond a la valeur du flou
def gaussian_blur_tensor(x, kernel_size=3, sigma=0.5):
    # x: [B,C,H,W]
    B, C, H, W = x.shape

    # Générer un kernel 1D
    def gauss1d(k, sigma):
        a = torch.arange(k).float() - (k - 1) / 2
        g = torch.exp(-(a**2)/(2*sigma**2))
        return g / g.sum()

    k = kernel_size
    g = gauss1d(k, sigma)
    kernel2d = g[:,None] * g[None,:]      # [k,k]
    kernel2d = kernel2d.to(x.device, dtype=x.dtype)
    kernel2d = kernel2d.expand(C, 1, k, k)  # [C,1,k,k] pour grouped conv
    pad = k // 2
    x = F.conv2d(x, kernel2d, padding=pad, groups=C)
    return x

# blur sur le coté du masque type photoshop
def feather_mask(mask, radius=3):
    """
    mask: [B,1,H,W] (0 ou 1)
    radius: épaisseur du bord en pixels
    """

    # approx distance via blur répété (rapide GPU)
    dist = mask.clone()

    for _ in range(radius):
        dist = F.max_pool2d(dist, kernel_size=3, stride=1, padding=1)

    # bande extérieure uniquement
    band = (dist - mask).clamp(0, 1)

    # normaliser pour faire un dégradé progressif
    band = band / (band.max().clamp(min=1e-6))

    # lisser légèrement (optionnel mais joli)
    band = gaussian_blur_tensor(band, kernel_size=5, sigma=1.0)

    # reconstruire
    mask = mask + band * (1 - mask)

    return torch.clamp(mask, 0, 1)

# blur sur le coté du masque type photoshop
def feather_mask_fast(mask, radius=3):
    k = 2 * radius + 1

    dist = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)

    band = (dist - mask).clamp(0, 1)
    band = gaussian_blur_tensor(band, kernel_size=5, sigma=1.0)

    return torch.clamp(mask + band * (1 - mask), 0, 1)


def feather_outside_only_stable(mask, radius=3, blur_kernel=5, sigma=1.0, debug=False):
    """
    mask: attendu [B,1,H,W] (0 ou 1)
    radius: largeur du dégradé extérieur (pixels)
    """

    # =========================================================
    # 0. SAFETY SHAPE (CRITICAL)
    # =========================================================
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [H,W] -> [1,1,H,W]
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)               # [B,H,W] -> [B,1,H,W]

    if mask.dim() != 4:
        raise ValueError(f"[feather] Invalid mask shape: {mask.shape}")

    B, C, H, W = mask.shape

    if C != 1:
        # On force 1 canal (important pour pooling / blur)
        mask = mask.mean(dim=1, keepdim=True)

    # Clamp sécurité
    mask = mask.clamp(0, 1)

    if debug:
        print(f"[FEATHER] input shape: {mask.shape}")

    # =========================================================
    # 1. DILATATION (couronne extérieure)
    # =========================================================
    k = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)

    # =========================================================
    # 2. BANDE EXTERNE
    # =========================================================
    band = (dilated - mask).clamp(0, 1)

    # 🔥 SAFETY (évite ton crash actuel)
    if band.dim() == 3:
        band = band.unsqueeze(1)

    if band.dim() != 4:
        raise ValueError(f"[feather] band invalid shape: {band.shape}")

    # =========================================================
    # 3. BLUR (soft edge)
    # =========================================================
    band = gaussian_blur_tensor(band, kernel_size=blur_kernel, sigma=sigma)

    # =========================================================
    # 4. NORMALISATION
    # =========================================================
    band_max = band.max().clamp(min=1e-6)
    band = band / band_max

    # =========================================================
    # 5. RECONSTRUCTION
    # =========================================================
    result = mask + band * (1 - mask)

    result = torch.clamp(result, 0, 1)

    # =========================================================
    # 6. DEBUG
    # =========================================================
    if debug:
        print(f"[FEATHER] band max={band_max.item():.6f}")
        print(f"[FEATHER] result mean={result.mean().item():.6f}")

    return result


def feather_outside_only(mask, radius=3, blur_kernel=5, sigma=1.0):
    """
    mask: [B,1,H,W] (0 ou 1)
    radius: largeur du dégradé extérieur (pixels)
    """

    # 1. dilatation contrôlée (couronne extérieure)
    k = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)

    # 2. isoler UNIQUEMENT l'extérieur
    band = (dilated - mask).clamp(0, 1)

    # 3. lisser cette bande (sans toucher l'intérieur)
    band = gaussian_blur_tensor(band, kernel_size=blur_kernel, sigma=sigma)

    # 4. normaliser pour un vrai dégradé
    band = band / (band.max().clamp(min=1e-6))

    # 5. reconstruction :
    # intérieur intact + dégradé extérieur uniquement
    result = mask + band * (1 - mask)

    return torch.clamp(result, 0, 1)


def feather_inside(mask, radius=5, blur_kernel=5, sigma=1.0):
    """
    mask: [B,1,H,W] (0 ou 1)
    radius: largeur de la bande à l'intérieur du mask
    """

    # 1. érosion (réduire le mask) pour créer la bande interne
    k = 2 * radius + 1
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=radius)  # erosion via max pooling sur négatif

    # 2. isoler la bande intérieure
    band = (mask - eroded).clamp(0, 1)

    # 3. lisser légèrement pour le feather
    band = gaussian_blur_tensor(band, kernel_size=blur_kernel, sigma=sigma)

    # 4. correction alpha pour éviter l’étirement
    band = band * band * (3 - 2 * band)  # smoothstep

    # 5. reconstruire le mask : intérieur net + dégradé à l’intérieur
    result = eroded + band

    return torch.clamp(result, 0, 1)


def feather_inside_strict(mask, radius=5, blur_kernel=5, sigma=1.0):
    """
    mask: [B,1,H,W] (0 ou 1)
    radius: largeur du feather à l'intérieur du mask
    """

    # 1️⃣ érosion : zone intérieure parfaitement nette
    k = 2 * radius + 1
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=radius)

    # 2️⃣ bande interne à flouter (feather)
    band = (mask - eroded).clamp(0, 1)

    # 3️⃣ flou uniquement sur la bande
    if band.max() > 0:  # éviter division par zéro
        band_blur = gaussian_blur_tensor(band, kernel_size=blur_kernel, sigma=sigma)
        # 4️⃣ correction alpha smoothstep pour éviter étirement
        band_blur = band_blur * band_blur * (3 - 2 * band_blur)
    else:
        band_blur = band

    # 5️⃣ reconstruction : intérieur net + bande floutée
    result = eroded + band_blur

    return torch.clamp(result, 0, 1)


def rotate_mask_around_torso_simple(mask, torso_points_px, angle, device="cuda"):
    """
    Rotate a mask around the torso center on X-axis only (horizontal rotation),
    corrected version with proper broadcasting.

    mask: [B, C, H, W]
    torso_points_px: [B, 2, N]
    angle: [B] tensor, rotation in radians
    device: device

    Returns:
        mask_rotated: [B, C, H, W]
    """


    B, C, H, W = mask.shape

    # Centre du torse
    torso_center = torso_points_px.mean(dim=2)  # [B, 2]
    cx = torso_center[:, 0].view(B, 1, 1)
    cy = torso_center[:, 1].view(B, 1, 1)

    # Coordonnées pixels
    xx = torch.arange(W, device=device).view(1, 1, W).float()  # [1,1,W]
    yy = torch.arange(H, device=device).view(1, H, 1).float()  # [1,H,1]

    # Décalage horizontal seulement
    x_shift = xx - cx
    y_shift = yy  # vertical inchangé

    # Rotation horizontale
    cos_a = torch.cos(angle).view(B, 1, 1)
    x_rot = cos_a * x_shift + cx  # rotation autour du centre X
    y_rot = y_shift

    # Broadcast pour stack
    x_norm = 2.0 * x_rot / (W - 1) - 1.0
    y_norm = 2.0 * y_rot / (H - 1) - 1.0
    # broadcast sur [B,H,W]
    x_norm = x_norm.expand(B, H, W)
    y_norm = y_norm.expand(B, H, W)

    grid = torch.stack((x_norm, y_norm), dim=-1)  # [B,H,W,2]

    # Rotation bilinéaire
    mask_rotated = F.grid_sample(
        mask, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    )

    return mask_rotated

def rotate_mask_around_visage(mask, torso_points_px, angle, H, W, device="cuda"):

    B, C, H_mask, W_mask = mask.shape

    # 🔥 utiliser H_mask / W_mask partout
    H, W = H_mask, W_mask

    # -------------------- centre du torse --------------------
    torso_center = torso_points_px.mean(dim=2)  # [B, 2]

    # -------------------- grid pixel --------------------
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    xx = xx.float().unsqueeze(0).expand(B, -1, -1)
    yy = yy.float().unsqueeze(0).expand(B, -1, -1)

    # -------------------- rotation autour centre --------------------
    cx = torso_center[:, 0].view(B, 1, 1)
    cy = torso_center[:, 1].view(B, 1, 1)

    x_shift = xx - cx
    y_shift = yy - cy

    cos_angle = torch.cos(angle).view(B, 1, 1)
    sin_angle = torch.sin(angle).view(B, 1, 1)

    x_rot = cos_angle * x_shift - sin_angle * y_shift
    y_rot = sin_angle * x_shift + cos_angle * y_shift

    x_final = x_rot + cx
    y_final = y_rot + cy

    # -------------------- NORMALISATION CORRECTE --------------------
    x_norm = 2.0 * x_final / (W - 1) - 1.0
    y_norm = 2.0 * y_final / (H - 1) - 1.0

    grid = torch.stack((x_norm, y_norm), dim=-1)

    # 🔥 FIX CRITIQUE : align_corners=False
    mask_rotated = F.grid_sample(
        mask,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    # 🔥 GARANTIE ABSOLUE
    assert mask_rotated.shape[2:] == (H, W), \
        f"Shape mismatch: {mask_rotated.shape} vs {(B,C,H,W)}"

    return mask_rotated





def save_debug_mask(mask: torch.Tensor, H: int, W: int, debug_dir: str, frame_counter: int, prefix: str = "mask", scale: int = 8):
    """
    Sauvegarde un masque pour debug.
    - mask: tensor [B,1,H,W]
    - H,W: dimensions originales
    - debug_dir: dossier de sortie
    - frame_counter: numéro de frame pour nommage
    - prefix: nom du fichier
    - scale: facteur d'agrandissement pour visualisation
    """
    if debug_dir is None:
        return

    os.makedirs(debug_dir, exist_ok=True)

    mask_np_debug = (mask[0,0].detach().cpu().numpy() * 255).astype(np.uint8)
    mask_debug = cv2.resize(mask_np_debug, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST)
    mask_debug_rgb = cv2.cvtColor(mask_debug, cv2.COLOR_GRAY2BGR)

    save_path = os.path.join(debug_dir, f"{prefix}_{frame_counter:05d}.png")
    cv2.imwrite(save_path, mask_debug_rgb)
    print(f"[DEBUG] {prefix} saved: {save_path}")


def save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="driven"):
    """
    Sauvegarde une carte d'impact montrant les différences entre latents et latents_in.

    Args:
        latents (torch.Tensor): Latents modifiés [B, C, H, W]
        latents_in (torch.Tensor): Latents originaux [B, C, H, W]
        debug_dir (str): Répertoire où sauvegarder l'image
        frame_counter (int): Index de la frame pour le nom de fichier
        prefix (str): Préfixe pour différencier le type d'impact map
    """
    if frame_counter % 2 == 0:
        return
    if debug_dir is None:
        return


    os.makedirs(debug_dir, exist_ok=True)

    # Calcul de l'impact map
    impact_map = torch.abs(latents - latents_in).mean(1, keepdim=True)
    impact_np = impact_map[0,0].detach().cpu().numpy()
    impact_np -= impact_np.min()
    if impact_np.max() > 0:
        impact_np /= impact_np.max()

    # Nom de fichier avec préfixe
    save_path = os.path.join(debug_dir, f"impact_map_{prefix}_{frame_counter:05d}.png")
    Image.fromarray((impact_np*255).astype(np.uint8)).save(save_path)
    print(f"[DEBUG] Impact map saved: {save_path}")


def feather_dynamic_vectorized(mask, delta_px, base_radius=3, sigma=1.5, scale=2.0):
    speed = torch.norm(delta_px, dim=-1, keepdim=True)  # [B,1,1]
    radius_dynamic = torch.clamp(base_radius + scale * speed, max=15.0)
    radius_int = radius_dynamic.round().long()  # converti en entier pour max_pool2d

    feathered_mask = torch.zeros_like(mask)
    B = mask.shape[0]

    for b in range(B):
        feathered_mask[b:b+1] = feather_outside_only_alpha(
            mask[b:b+1],
            radius=radius_int[b].item(),
            sigma=sigma
        )
    return feathered_mask


def compute_delta(latents_out, latent_ref, controlnet_scale, importance):
    delta = latents_out - latent_ref
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1.0, neginf=-1.0)
    # 🔥 adaptive blending ici
    delta = torch.tanh(delta) * 0.15 * importance
    return delta * controlnet_scale


# 🔹 Stabilise les latents pour éviter NaN ou valeurs extrêmes
#   Normalisation et clamp pour rester dans [-1.2,1.2].
def stabilize_latents_motion(latents):
    latents = torch.nan_to_num(latents)
    latents_max = latents.abs().amax(dim=(2,3), keepdim=True)
    latents = latents / (latents_max + 1e-6)
    latents = latents * 0.95
    return torch.clamp(latents, -1.2, 1.2)

# ---------------------- Ancienne fonction --------------------------------------------------------
# 🔹 Calcule le déplacement du torse par rapport à la frame précédente
#   Utilisé pour translater les latents afin de suivre le mouvement.

def compute_delta_torso(kp, latent_h, latent_w, scale=0.8):
    """
    Calcule le déplacement du torse en coordonnées latentes.
    Le centre du warp est aligné sur le torse du personnage.
    """

    # Extraire les épaules
    r_shoulder = get_point(kp, 2)  # [B,2]
    l_shoulder = get_point(kp, 5)

    # Centre du torse
    torso_center = (r_shoulder + l_shoulder) * 0.5  # [B,2]

    # Normaliser par rapport à l'image (0-1)
    # On suppose que kp est déjà normalisé sur H,W [0,1]
    torso_center_norm = torso_center.clone()

    # Calculer offset depuis le centre du latent
    center_offset_x = (torso_center_norm[:,0] - 0.5) * latent_w
    center_offset_y = (torso_center_norm[:,1] - 0.5) * latent_h

    delta_torso = torch.stack([center_offset_x, center_offset_y], dim=1) * scale

    # 🔒 Stabilisation pour éviter les jumps
    delta_torso = torch.tanh(delta_torso * 2.0) * 0.5

    return delta_torso


# 🔹 Recentre tous les keypoints par rapport au torse (entre épaules)
#   Cela évite que le personnage se déplace vers le coin haut-gauche.
def normalize_keypoints(kp_tensor):
    kp = kp_tensor.clone()
    r_shoulder = get_point(kp, 2)
    l_shoulder = get_point(kp, 5)
    torso_center = (r_shoulder + l_shoulder) * 0.5
    kp[...,0] = kp[...,0] - torso_center[:,0].unsqueeze(1)  # recentre X
    kp[...,1] = kp[...,1] - torso_center[:,1].unsqueeze(1)  # recentre Y
    return kp

# 🔹 Applique une translation sur les latents en utilisant un grid warp
#   Déplace visuellement le personnage selon le delta du torse.
def warp_latents(latents, delta_torso, H, W, device):

    B = latents.shape[0]

    dx = delta_torso[:, 0].reshape(B,1,1) * W
    dy = delta_torso[:, 1].reshape(B,1,1) * H

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).repeat(B,1,1,1)

    delta_grid = torch.cat([dx*2/W, dy*2/H], dim=-1).unsqueeze(2)
    grid = grid + delta_grid

    latents_warped = F.grid_sample(
        latents,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    return latents_warped, dx, dy, grid


def warp_latents_local(latents, delta, mask, center, H, W, device):

    B, C, _, _ = latents.shape

    # -------------------- Préparation --------------------

    # centre en pixels
    center_px = center * torch.tensor([W-1, H-1], device=device)
    center_px = center_px.view(B,1,1,2)

    # delta en pixels
    delta_px = delta * torch.tensor([W, H], device=device)
    delta_px = delta_px.view(B,1,1,2)

    # grille pixel
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=-1).float()
    grid = grid.unsqueeze(0).repeat(B,1,1,1)

    # masque
    mask_expand = mask.permute(0,2,3,1) ** 1.5

    # -------------------- 💥 warp pivot --------------------

    grid = grid - center_px
    grid = grid + delta_px * mask_expand
    grid = grid + center_px

    # -------------------- normalisation --------------------

    grid_norm = grid.clone()
    grid_norm[...,0] = 2.0 * grid[...,0] / (W-1) - 1.0
    grid_norm[...,1] = 2.0 * grid[...,1] / (H-1) - 1.0

    # -------------------- sampling --------------------

    latents_warped = F.grid_sample(
        latents,
        grid_norm,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )

    return latents_warped


#--------------------------------------------------------------------------------------------------------------

def save_debug_pose_image_with_skeleton(
    pose_tensor,
    keypoints_tensor,
    frame_counter,
    output_dir,
    cfg=None,
    prefix="openpose"
):
    """
    Sauvegarde une image de pose ET un squelette OpenPose pour contrôle visuel.

    Args:
        pose_tensor (torch.Tensor): [B,3,H,W] normalisé [-1,1] ou [C,H,W]
        keypoints_tensor (torch.Tensor): [B,18,3] (x,y,conf) normalisé [0,1]
        frame_counter (int): numéro de frame
        output_dir (str): dossier où sauvegarder
        cfg (dict, optional): peut contenir 'visual_debug' pour activer/désactiver
        prefix (str, optional): préfixe du fichier
    """
    print("[DEBUG] skeleton pipeline triggered")

    if cfg is not None and cfg.get("visual_debug") is False:
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------
    # 🔹 Convertir pose_tensor en image RGB [0,255]
    # ----------------------------
    pose_img = pose_tensor[0].detach().cpu()
    if pose_img.ndim == 3 and pose_img.shape[0] == 3:
        pose_img = pose_img.permute(1,2,0)  # H,W,C
    pose_img = ((pose_img + 1.0)/2.0 * 255).clamp(0,255).byte().numpy()

    # Sauvegarde simple de l'image de pose
    filename_pose = f"{prefix}_{frame_counter:05d}.png"
    path_pose = os.path.join(output_dir, filename_pose)
    cv2.imwrite(path_pose, cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Pose sauvegardée : {path_pose}")

    # ---------------------------
    # 🔹 Dessin du squelette via debug_draw_openpose_skeleton
    # ---------------------------
    if keypoints_tensor is not None:
        debug_draw_openpose_skeleton(
            pose_full_image=pose_tensor.unsqueeze(0) if pose_tensor.ndim==3 else pose_tensor,
            keypoints_tensor=keypoints_tensor,
            debug_dir=output_dir,
            frame_counter=frame_counter
        )
#----------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------     VENT                       -------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
def debug_save_mask_and_wind(
    mask,
    wind_delta,
    H,
    W,
    debug_dir,
    frame_counter,
    mask_prefix="torso_wind_mask_",
    wind_scale=200,
    upscale=8,              # 🔥 upscale réel
    draw_grid=True,
    overlay=True
):


    os.makedirs(debug_dir, exist_ok=True)

    # =========================================================
    # 🔹 1. NORMALIZE WIND DELTA (robuste)
    # =========================================================
    if isinstance(wind_delta, torch.Tensor):
        wd = wind_delta.detach().float().cpu()

        if wd.numel() == 2:
            dx, dy = wd.view(-1)
        else:
            wd = wd.view(-1)
            dx, dy = wd[-2], wd[-1]
    else:
        dx, dy = wind_delta

    dx, dy = float(dx), float(dy)

    # =========================================================
    # 🔹 2. UPSCALE MASK (VRAI upscale)
    # =========================================================
    mask_up = F.interpolate(
        mask,
        scale_factor=upscale,
        mode="bilinear",
        align_corners=False
    )

    mask_np = mask_up[0, 0].detach().cpu().numpy()
    H_up, W_up = mask_np.shape

    mask_vis = (mask_np * 255).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)

    # =========================================================
    # 🔹 3. CANVAS DEBUG
    # =========================================================
    canvas = np.zeros((H_up, W_up, 3), dtype=np.uint8)

    if overlay:
        canvas = cv2.addWeighted(mask_color, 0.6, canvas, 0.4, 0)

    # =========================================================
    # 🔹 4. DESSIN VENT
    # =========================================================
    center = (W_up // 2, H_up // 2)

    end_point = (
        int(center[0] + dx * wind_scale),
        int(center[1] + dy * wind_scale)
    )

    cv2.arrowedLine(
        canvas,
        center,
        end_point,
        color=(0, 255, 255),
        thickness=3,
        tipLength=0.25
    )

    # =========================================================
    # 🔹 5. VECTOR FIELD (optionnel 🔥 très utile)
    # =========================================================
    if draw_grid:
        step = max(20, W_up // 25)

        for y in range(0, H_up, step):
            for x in range(0, W_up, step):
                end = (
                    int(x + dx * wind_scale * 0.3),
                    int(y + dy * wind_scale * 0.3)
                )
                cv2.arrowedLine(
                    canvas,
                    (x, y),
                    end,
                    color=(255, 255, 255),
                    thickness=1,
                    tipLength=0.2
                )

    # =========================================================
    # 🔹 6. TEXTE DEBUG (🔥 très utile)
    # =========================================================
    cv2.putText(
        canvas,
        f"dx={dx:.4f} dy={dy:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"frame={frame_counter}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    # =========================================================
    # 🔹 7. SAVE
    # =========================================================
    save_path = os.path.join(debug_dir, f"{mask_prefix}{frame_counter:05d}.png")
    cv2.imwrite(save_path, canvas)

    print(f"[DEBUG] Wind+Mask saved: {save_path}")

def debug_save_mask_and_wind_simple(mask, wind_delta, H, W, debug_dir, frame_counter, mask_prefix="torso_wind_mask_", wind_scale=200):
    """
    Sauvegarde le masque et une icône vent pour debug.

    mask : torch tensor [B,1,H,W]
    wind_delta : torch tensor [1,1,1,2] ou [2]
    H, W : dimensions originales du masque
    debug_dir : dossier de sauvegarde
    frame_counter : numéro de la frame
    mask_prefix : préfixe du fichier masque
    wind_scale : échelle visuelle de la flèche vent
    """
    os.makedirs(debug_dir, exist_ok=True)

    # --- Sauvegarde masque ---
    save_debug_mask(mask, H, W, debug_dir, frame_counter, prefix=mask_prefix)

    # --- Image couleur vent ---
    wind_img = np.zeros((H*4, W*4, 3), dtype=np.uint8)

    # convertir wind_delta en numpy 1D si torch tensor
    if isinstance(wind_delta, torch.Tensor):
        wind_delta = wind_delta.detach().cpu().numpy().flatten()

    # position centrale
    pos = (W*2, H*2)
    end_point = (int(pos[0] + wind_delta[0]*wind_scale), int(pos[1] + wind_delta[1]*wind_scale))
    cv2.arrowedLine(wind_img, pos, end_point, color=(0,255,255), thickness=2, tipLength=0.3)

    # --- Sauvegarde vent ---
    save_path_wind = os.path.join(debug_dir, f"wind_icon_{frame_counter:05d}.png")
    cv2.imwrite(save_path_wind, wind_img)

    print(f"[DEBUG] Mask saved: {mask_prefix}{frame_counter:05d}, Wind icon saved: {save_path_wind}")
#----------------------------------------------------------------------------------------------------------------------------------

def draw_wind_icon(img, wind_delta, pos=(50,50), scale=100, color=(0,255,255), thickness=2):
    """
    Dessine un icône vent sur l'image de debug.

    img : np.array HxWx3 BGR
    wind_delta : torch tensor [dx, dy] ou [1,2], valeurs approximatives
    pos : tuple (x,y) centre du vent
    scale : multiplicateur pour agrandir la flèche
    color : couleur BGR
    thickness : épaisseur de la flèche
    """
    if isinstance(wind_delta, torch.Tensor):
        wind_delta = wind_delta.detach().cpu().numpy().flatten()

    start_point = pos
    end_point = (int(pos[0] + wind_delta[0]*scale), int(pos[1] + wind_delta[1]*scale))

    # flèche principale
    cv2.arrowedLine(img, start_point, end_point, color, thickness, tipLength=0.3)

    return img

#-------------------------------------------- Micro Boost & Micro moton ------------------------------------------------
def apply_micro_boost(
    latents,
    frame_counter,
    device,
    masks,
    keypoints,
    prev_keypoints=None,
    strength=1.0,
    debug=False,
    debug_dir=None
):
    t = torch.tensor(frame_counter / 6.0, device=device, dtype=latents.dtype)

    total = torch.zeros_like(latents)

    # -------------------- Motion strength --------------------
    if prev_keypoints is None:
        motion_strength = torch.tensor(0.0, device=device, dtype=latents.dtype)
    else:
        motion_strength = (keypoints[:, :, :2] - prev_keypoints[:, :, :2]).abs().mean()

        if torch.isnan(motion_strength):
            motion_strength = torch.tensor(0.0, device=device, dtype=latents.dtype)

        # clamp stable
        motion_strength = torch.clamp(motion_strength, 0.0, 0.05)

        # compression douce
        motion_strength = torch.log1p(motion_strength * 20.0)

        # scaling
        motion_strength = motion_strength * strength * 2.0

    if debug:
        print(f"[DEBUG][MICRO_BOOST]")
        print(f"  - strength: {strength}")
        print(f"  - motion_strength: {motion_strength.item():.6f}")
        print(f"  - frame_counter: {frame_counter}")

    zone_summaries = []
    weight_sum = 0.0

    # -------------------- Zones --------------------
    for zone_name, (mask, phase, amp) in masks.items():
        if mask is None:
            continue

        # oscillation toujours positive → évite annulation
        osc = 0.5 + 0.5 * torch.sin(t + phase)

        contrib = amp * mask * motion_strength * osc

        # pondération par importance réelle du masque
        weight = mask.mean().item() + 1e-6

        total += contrib * weight
        weight_sum += weight

        if debug:
            zone_summaries.append(
                (zone_name, float(amp), float(mask.mean().item()), float(contrib.mean().item()))
            )

    # normalisation intelligente (pas brutale)
    if weight_sum > 0:
        total = total / weight_sum

    # 🔥 gain final (clé)
    total = total * 3.0

    if debug:
        print("[DEBUG][MICRO_BOOST SUMMARY]")
        for name, amp, mmean, cmean in zone_summaries:
            print(f"  - {name}: amp={amp:.6f} mask={mmean:.6f} contrib={cmean:.8f}")

        print(f"[DEBUG][MICRO_BOOST] total mean: {total.mean().item():.8f}")
        print(f"[DEBUG][MICRO_BOOST] total max: {total.max().item():.8f}")

    return latents + total

def apply_micro_motion(
    latents: torch.Tensor,
    frame_counter: int,
    device,
    masks: dict,
    strength: float = 0.25,   # 🔥 NOUVEAU 0.3 - 0.6 → très réaliste (cinéma) - stable 0.25
    randomize: bool = True,
    debug=False
):
    """
    Micro motion avec contrôle global de l'intensité.
    strength : 0 = OFF, 1 = normal, >1 = amplifié
    """

    # 🔹 Clamp sécurité (évite explosion)
    strength = max(0.0, min(strength, 5.0))

    t = torch.tensor(frame_counter / 6.0, device=device)

    for zone_name, params in masks.items():
        if params is None:
            continue

        mask, phase, amplitude = params
        if mask is None:
            continue

        mask = mask.to(dtype=latents.dtype, device=device)

        # 🔹 Micro random plus stable (moins agressif)
        if randomize:
            noise = (torch.rand_like(mask) - 0.5) * 0.01
        else:
            noise = 0.0

        # 🔥 Application du strength AU BON ENDROIT
        delta = strength * amplitude * mask * torch.sin(t + phase + noise)

        latents = latents + delta

    if debug:
        print("[DEBUG] apply_micro_motion")
        print("  - delta mean px:", delta.abs().mean().item())
        print("  - delta max px:", delta.abs().max().item())

    return latents

