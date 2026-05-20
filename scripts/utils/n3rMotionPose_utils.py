#********************************************
# n3rOpenPose_utils.py
#********************************************
import torch
import math
import time
from diffusers import ControlNetModel

import torch.nn.functional as F
from .n3rcoords import pair, safe_xy, safe_update, norm, build_upper_body_inputs, animate_upper_body, reconstruct_hips
from .n3rControlNet import create_canny_control, control_to_latent, match_latent_size
from .tools_utils import ensure_4_channels, print_generation_params, sanitize_latents
from .n3rMotionPose_tools import gaussian_blur_tensor, debug_draw_openpose_skeleton, rotate_mask_around_torso_simple, rotate_mask_around_visage, save_impact_map, smooth_noise, feather_dynamic_vectorized, compute_delta, stabilize_latents_motion, save_debug_pose_image_with_skeleton, feather_inside_strict2, feather_outside_only_alpha2, apply_micro_motion, apply_micro_boost, dilate_mask, save_debug_mask_scale, feather_outside_only_stable, save_debug_mask

from .n3rMotionMouth import apply_mouth_smil, mouth_model
from .n3rMotionHair import apply_hair_motion_cycle
from .n3rMotionBreathing import apply_breathing, apply_breathing_real

from .n3rPoseModule import apply_actor_model, resolve_motion_model

from .n3rMotionPoseClass import Pose
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
import traceback
from torchvision.utils import save_image


"""
alpha_base = 0.92          # un peu plus inertie
freeze_threshold = 0.0022  # freeze plus actif
freeze_strength = 0.30     # freeze utile mais pas bloquant
micro_jitter = 0.00012     # encore plus subtil
time_scale = 0.45          # ralentit légèrement plus
max_velocity = 0.008       # 🔥 clé principale (cinematic cap)


"""

MOTION_PROFILES = {
    "stable": {
        "time_scale": 0.3,
        "camera_lock": 0.95,
        "max_velocity": 0.002,
        "rotation_gain": 0.7
    },

    "cinematic": {
        "time_scale": 0.4,
        "camera_lock": 0.85,
        "max_velocity": 0.005, #Max 0.03 ou 0.01
        "rotation_gain": 1.2
    },

    "warp": {
        "time_scale": 0.3,
        "camera_lock": 0.92,
        "max_velocity": 0.008,
        "rotation_gain": 1.2
    },

    "dynamic": {
        "time_scale": 0.45,
        "camera_lock": 0.95,
        "max_velocity": 0.003,
        "rotation_gain": 1.3
    },
    # pas d'animation des keypoints casi null'
    "locked": {
        "time_scale": 0.2,
        "camera_lock": 0.98,
        "max_velocity": 0.001,
        "rotation_gain": 0.001
    },

    # optional hint only (NOT authoritative)
    "acting": {
        "time_scale": 0.55,
        "camera_lock": 0.75,
        "max_velocity": 0.02,
        "rotation_gain": 1.3
    }
}

def border_fade_mask_rotate(
    H, W, device,
    fade_ratio=0.1,
    angle=0.0,
    inverse=False
):
    """
    Masque de bords avec pivot central + gestion du sens de rotation
    angle   : angle de référence (rad)
    inverse : si True → rotation inverse (compensation warp)
    """

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    # 🎯 Choix du sens
    angle_used = -angle if inverse else angle

    if angle_used != 0.0:
        cos_a = math.cos(angle_used)
        sin_a = math.sin(angle_used)

        x_rot = xx * cos_a - yy * sin_a
        y_rot = xx * sin_a + yy * cos_a

        xx, yy = x_rot, y_rot

    # Distance aux bords
    dx = torch.minimum(xx + 1, 1 - xx)
    dy = torch.minimum(yy + 1, 1 - yy)
    dist = torch.minimum(dx, dy)

    fade = fade_ratio

    mask = (1.0 - dist / fade).clamp(0, 1)

    # smoothstep
    mask = mask * mask * (3 - 2 * mask)

    return mask.unsqueeze(0).unsqueeze(0)


def border_fade_mask(H, W, device, fade_ratio=0.1):
    """
    fade_ratio = 0.1 → 10% des bords
    """
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing="ij"
    )

    # Distance aux bords (0 aux bords, 1 au centre)
    dx = torch.min(xx, 1 - xx)  # Distance horizontale aux bords
    dy = torch.min(yy, 1 - yy)  # Distance verticale aux bords

    dist = torch.min(dx, dy)  # Distance minimum aux bords

    # Largeur de transition (10%) -> 0.5 - (5%) -> 0.25 (2.5%)-> 0.125
    fade = fade_ratio * 0.1

    # Inverser la logique, donc 1 aux bords, 0 au centre
    mask = (1.0 - dist / fade).clamp(0, 1)

    # Lissage doux pour une transition fluide (optionnel)
    mask = mask * mask * (3 - 2 * mask)

    return mask.unsqueeze(0).unsqueeze(0)  # Ajout des dimensions batch et channel

#---------------------------------------------------------------------

def add_gaussian_noise(latent, valid_mask, noise_std=0.1, noise_strength=1.0, kernel_size=5):
    """
    Ajouter du bruit gaussien uniquement dans les zones invalides de l'image,
    avec une transition progressive sur les bords.

    :param latent: Tensor d'entrée de forme (B, C, H, W) représentant l'image.
    :param valid_mask: Tensor de masque valide (1 pour zones valides, 0 pour invalides).
    :param noise_std: Écart type du bruit de base.
    :param noise_strength: Facteur d'intensité du bruit, permettant d'ajuster la quantité de bruit.
    :param kernel_size: Taille du noyau pour dilater le masque.
    :return: Tensor latent avec bruit ajouté.
    """

    # Générer le bruit aléatoire gaussien de même forme que 'latent'
    noise = torch.randn_like(latent) * noise_std

    # Dilater le masque pour permettre une zone tampon
    dilated_mask = dilate_mask(valid_mask, kernel_size)

    # Appliquer le bruit uniquement dans les zones invalides (où valid_mask == 0)
    noise_applied = noise * (1 - dilated_mask)  # bruit dans les zones invalides

    # Ajouter le bruit progressif dans les zones invalides
    noisy_latent = latent + noise_applied * noise_strength

    return noisy_latent




def compensate_latent_shift_dev(
    latent,
    frame_counter,
    shift,
    global_angle_z=None,  # Angle axe Z
    prev_latent=None,
    max_shift_ratio=0.5,
    padding_mode="reflection",
    image_size=(1280, 896),
    debug=False,
    debug_dir=None
):

    if padding_mode == "reflect":
        padding_mode = "reflection"

    B, C, H, W = latent.shape
    device = latent.device

    # =========================================================
    # 1. SHIFT
    # =========================================================
    shift = shift.to(device).view(-1)
    shift_x = float(shift[0])
    shift_y = float(shift[1])

    shift_mag = math.sqrt(shift_x**2 + shift_y**2)

    # Smooth gain (stable)
    t = min(max(shift_mag * 10.0, 0.0), 1.0)
    t = t * t * (3 - 2 * t)  # smoothstep
    gain = 1.0 + 6.0 * t

    shift_x *= gain
    shift_y *= gain

    # Clamp spatial
    max_dx = W * max_shift_ratio
    max_dy = H * max_shift_ratio
    dx = max(-max_dx, min(max_dx, shift_x))
    dy = max(-max_dy, min(max_dy, shift_y))

    if debug:
        print(f"[SHIFT] in=({shift[0]:.4f},{shift[1]:.4f}) | gain={gain:.3f} | out=({dx:.4f},{dy:.4f})")

    # =========================================================
    # 2. ANGLE axe Z
    # =========================================================
    if global_angle_z is not None:
        if isinstance(global_angle_z, torch.Tensor):
            angle = float(global_angle_z.mean())
        else:
            angle = float(global_angle_z)
    else:
        angle = 0.0

    # Rotation gain
    rot_gain = 1.0 + 2.0 * min(max(shift_mag * 10.0, 0.0), 1.0)
    angle *= rot_gain

    # Clamp angle (~5.7°)
    max_angle = 0.1
    angle = max(-max_angle, min(max_angle, angle))

    if debug:
        print(f"[ANGLE] base={global_angle_z} | gain={rot_gain:.3f} | final={angle:.5f} rad")

    # Inverse rotation
    angle = -angle
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # =========================================================
    # 3. GRID
    # =========================================================
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    xx = xx.float()
    yy = yy.float()

    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    x = xx - cx
    y = yy - cy

    rx = x * cos_a - y * sin_a + cx
    ry = x * sin_a + y * cos_a + cy

    # Apply translation
    rx = rx - dx
    ry = ry - dy

    # Normalize
    grid_x = 2.0 * rx / (W - 1) - 1.0
    grid_y = 2.0 * ry / (H - 1) - 1.0

    grid = torch.stack((grid_x, grid_y), dim=-1)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # =========================================================
    # 4. WARP
    # =========================================================
    latent_warped = F.grid_sample(
        latent,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True
    )

    # Création du masque de validité : valeurs dans les limites de [-1, 1] pour chaque coordonnée
    # =========================================================
    # VALID MASK (SAFE VERSION)
    # =========================================================

    if global_angle_z is not None:
        angle_warp = float(global_angle_z)
        valid_mask = border_fade_mask_rotate( H, W, device, fade_ratio=0.1, angle=angle_warp, inverse=True )
    else:
        valid_mask = border_fade_mask(H, W, device, fade_ratio=0.05)
    if debug:
        save_debug_mask(valid_mask, H, W, debug_dir, frame_counter, prefix="compensate_mask1")

    # =========================================================
    # 5. Adapatation auto MASK
    # =========================================================

    shift_tensor = torch.tensor(shift, device=device)
    shift_magnitude = torch.norm(shift_tensor)  # Directement norme du vecteur

    if debug:
        print(f"[MASK] shift_magnitude={shift_magnitude:.4f}")

    # Dynamically adjust dilation size based on shift magnitude and image size
    kernel_size = max(3, int(shift_magnitude * max(H, W) * 0.5))
    kernel_size = min(kernel_size, 51)  # Limite la taille du noyau pour éviter une dilatation trop importante

    if debug:
        print(f"[MASK] kernel_size={kernel_size:.4f}")

    # Ajuster la taille du noyau en fonction de l'angle
    angle_factor = min(abs(float(global_angle_z)) * 200, 10) if global_angle_z is not None else 0
    kernel_size += int(angle_factor)

    if debug:
        print(f"[MASK] +Angle kernel_size={kernel_size:.4f}")

    oriH, oriW = image_size
    if oriH * oriW > 1280 * 896:  # Ajuster cette condition selon les tailles d'image courantes
        kernel_size += 2
        if debug:
            print(f"[MASK] +SIZE kernel_size={kernel_size:.4f}")

    if kernel_size % 2 == 0:
        kernel_size += 1


    # Applique le flou gaussien sur le masque dilaté
    # Paramètres à ajuster selon besoin sigma valeur du flou, blur_kernel longueur, radius valeur bande
    valid_mask = feather_outside_only_stable(valid_mask, radius=0, blur_kernel=kernel_size, sigma=0.005)


    # =========================================================
    # Ajuster la taille du valid_mask pour correspondre aux latents
    valid_mask = F.interpolate(valid_mask, size=(latent.shape[2], latent.shape[3]), mode="bilinear", align_corners=False)

    # Calcul du ratio de validité moyen
    valid_ratio = valid_mask.mean().item()

    if debug:
        print(f"[MASK] valid_ratio={valid_ratio:.4f}")
        save_debug_mask(valid_mask, H, W, debug_dir, frame_counter, prefix="compensate_mask5")

    # =========================================================
    # 6. BLEND (SOLUTION A)
    # =========================================================
    if prev_latent is not None:
        # Vérification des dimensions avant de faire le blending
        if latent_warped.shape != prev_latent.shape:
            latent_warped = F.interpolate(latent_warped, size=prev_latent.shape[2:], mode="bilinear", align_corners=False)
            if latent_warped.shape != prev_latent.shape:
                latent_warped = latent_warped[:, :, :prev_latent.shape[2], :prev_latent.shape[3]]
                if debug:
                    print(f"[BLEND] Redimensionnement de latent_warped à la taille de : {prev_latent.shape}")

        # Vérification finale des dimensions
        if latent_warped.shape != prev_latent.shape:
            raise ValueError(f"Les tailles de latent_warped {latent_warped.shape} et prev_latent {prev_latent.shape} ne correspondent toujours pas!")

        #Ajout de bruit zone vide
        prev_latent = add_gaussian_noise(prev_latent, valid_mask, noise_std=0.1)

        # Application du blending : plus marqué pour les bords, plus doux au centre

        latent_out = latent_warped * (1.0 - valid_mask) + prev_latent * valid_mask

        if debug:
            print("[BLEND] Using previous latent (temporal fix)")

    else:
        # fallback safe
        fill = F.avg_pool2d(latent_warped, kernel_size=5, stride=1, padding=2)

        #Ajout de bruit zone vide
        #latent_warped = add_gaussian_noise(latent_warped, valid_mask, noise_std=0.1)

        latent_out = latent_warped * valid_mask + fill * (1.0 - valid_mask)

        if debug:
            print("[BLEND] Using spatial fallback")

    # =========================================================
    # 7. DEBUG FINAL
    # =========================================================
    if debug:
        delta = (latent_out - latent).abs().mean().item()
        print(f"[OUTPUT] delta_mean={delta:.6f}")


    return latent_out


#------extract_keypoints_from_pose

def extract_keypoints_from_pose(
    pose_full_image=None,
    device="cuda",
    debug=False,
    debug_dir=None,
    frame_counter=None,
    image_size=(1280, 896)  # H, W fallback si pas d'image
):
    """
    Extraction MANUELLE des keypoints (COCO-like).
    Coordonnées normalisées [0,1].

    pose_full_image est optionnel et utilisé UNIQUEMENT pour le debug visuel.
    """

    # =========================================================
    # 🔹 SIZE HANDLING (IMPORTANT)
    # =========================================================
    if pose_full_image is not None:
        B, C, H, W = pose_full_image.shape
    else:
        H, W = image_size
        B = 1  # fallback batch

    # =========================================================
    # 🔥 KEYPOINTS TEMPLATE
    # =========================================================
    keypoints_template = [
        [422/W, 408/H, 1.0],
        [418/W, 490/H, 1.0],
        [562/W, 506/H, 1.0],
        [627/W, 896/H, 1.0],
        [488/W, 1040/H, 1.0],

        [275/W, 519/H, 1.0],
        [197/W, 944/H, 1.0],
        [431/W, 1087/H, 1.0],

        [308/W, 1129/H, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],

        [564/W, 1102/H, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],

        [374/W, 323/H, 1.0],
        [492/W, 355/H, 1.0],

        [529/W, 349/H, 1.0],
        [308/W, 282/H, 1.0],
        [406/W, 446/H, 1.0],

        [562/W, 506/H, 1.0],
        [275/W, 519/H, 1.0],

        [387/W, 480/H, 1.0],
        [307/W, 282/H, 1.0],
        [529/W, 349/H, 1.0],
        [422/W, 428/H, 1.0],


        [197/W, 405/H, 1.0], # Hair detected: [(197, 405), (300, 389), (403, 374)]  #'hair_root': 1000, 'hair_left': 1001, 'hair_right': 1002, 'hair_top': 1003,
        [300/W, 389/H, 1.0],
        [403/W, 374/H, 1.0],
        [629/W, 189/H, 1.0], #hair_top : (629, 189) hair_top_left : (540, 189) hair_top_right : (718, 189)
        [540/W, 189/H, 1.0],
        [718/W, 189/H, 1.0],

        [359/W, 432/H, 1.0], #left_top_hair1 OK
        [413/W, 425/H, 1.0], #left_top_hair2  KO
        [338/W, 288/H, 1.0], #left_top_hair3

        [499/W, 449/H, 1.0], # right_top_m bouche
        [491/W, 403/H, 1.0], # right_top_hair, right_top_hair1, right_top_hair2, right_top_hair3, right_hair, (center_x, center_y)]
        [434/W, 427/H, 1.0], # right_top_hair3 KO

        [0.0, 0.0, 0.0], # top_hair1
        [0.0, 0.0, 0.0], # top_hair2
        [0.0, 0.0, 0.0], # top_hair3


        [449/W, 449/H, 1.0], # bouche
        [359/W, 432/H, 1.0], #
        [432/W, 427/H, 1.0], # nez
        [413/W, 425/H, 1.0], # 43

        [0.0, 0.0, 0.0], # 44
        [0.0, 0.0, 0.0], # 45
        [0.0, 0.0, 0.0], # 46
        [0.0, 0.0, 0.0], # 47
        [0.0, 0.0, 0.0], # 48
        [0.0, 0.0, 0.0], # 49
        [0.0, 0.0, 0.0], # 50
        [0.0, 0.0, 0.0], # 51
        [0.0, 0.0, 0.0], # 52

        [0.0, 0.0, 0.0], # 53
        [0.0, 0.0, 0.0], # 54
        [0.0, 0.0, 0.0], # 55
        [0.0, 0.0, 0.0], # 56

        [0.0, 0.0, 0.0], # 57
        [0.0, 0.0, 0.0], # 58
        [0.0, 0.0, 0.0], # 59

        [0.0, 0.0, 0.0], # 60
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],

        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], #70

        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],

        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], #80


        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],

        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], #90




    ]

    # =========================================================
    # 🔹 TENSOR BUILD
    # =========================================================
    keypoints_np = np.array(keypoints_template, dtype=np.float32)
    keypoints_np = np.clip(keypoints_np, 0.0, 1.0)

    keypoints_np = np.expand_dims(keypoints_np, axis=0)
    keypoints_np = np.repeat(keypoints_np, B, axis=0)

    keypoints_tensor = torch.from_numpy(keypoints_np).to(device)

    # =========================================================
    # 🔹 DEBUG (OPTIONAL SAFE)
    # =========================================================
    if debug and debug_dir is not None and frame_counter is not None:
        if pose_full_image is not None:
            debug_draw_openpose_skeleton(
                pose_full_image=pose_full_image,
                keypoints_tensor=keypoints_tensor,
                debug_dir=debug_dir,
                frame_counter=frame_counter
            )
        else:
            # fallback debug sans image
            print(f"[DEBUG] keypoints frame {frame_counter} (no image provided)")

    return keypoints_tensor


def normalize_coord(coord):
    """
    Convert any coord format → (x, y) or None
    """

    if coord is None:
        return None

    # dict → essayer de récupérer center
    if isinstance(coord, dict):
        if "center" in coord:
            return normalize_coord(coord["center"])
        return None

    # list / tuple
    if isinstance(coord, (list, tuple)):
        # cas [x, y]
        if len(coord) == 2 and all(isinstance(v, (int, float)) for v in coord):
            return coord

        # cas [[x, y]]
        if len(coord) == 1:
            return normalize_coord(coord[0])

        # cas [[x1,y1],[x2,y2]] → prendre le centre
        if len(coord) == 2 and all(isinstance(v, (list, tuple)) for v in coord):
            x = (coord[0][0] + coord[1][0]) * 0.5
            y = (coord[0][1] + coord[1][1]) * 0.5
            return (x, y)

    return None




def update_keypoints_from_pose(
    current_keypoints,
    nose_coords,
    neck_coords,
    shoulders_coords,
    clavicules_coords,
    elbow_coords,
    wrists_coords,
    hips_coords,
    eye_coords,
    ear_coords,
    mouth_coords,
    hair_coords,
    image_size=(1280, 896),
    device="cuda",
    frame_counter=0,
    debug: bool = False,
    verbose: bool = False,
    debug_dir=None
):
    """
    Robust keypoint injection with fallback + reconstruction
    """

    # =========================================================
    # 🔹 VALIDATION FUNCTION
    # =========================================================
    def is_valid(coord, label):
        if coord is None:
            return False
        coord = normalize_coord(coord)

        if debug and coord is None:
            print(f"[WARN] {label} coord dropped (invalid format)")
        if coord is None:
            return False

        x, y = coord
        if x is None or y is None:
            print(f"[DEBUG] x or y is None: x={x}, y={y}")
            return False
        if not np.isfinite(x) or not np.isfinite(y):
            print(f"[DEBUG] Non-finite values: x={x}, y={y}")
            return False
        if x <= 0 or y <= 0:
            print(f"[DEBUG] Invalid coordinates: x={x}, y={y}")
            return False
        if x > W or y > H:
            print(f"[DEBUG] Out of bounds: x={x}, y={y}, W={W}, H={H}")
            return False
        return True

    H, W = image_size

    keypoints_np = current_keypoints.clone().detach().cpu().numpy()[0]

    # =========================================================
    # 🔹 SAFE NORMALIZATION
    # =========================================================
    left_shoulder, right_shoulder = pair(shoulders_coords, debug=debug)
    left_clavicle, right_clavicle = pair(clavicules_coords, debug=debug)
    left_elbow, right_elbow = pair(elbow_coords, debug=debug)
    left_wrist, right_wrist = pair(wrists_coords, debug=debug)
    left_hip, right_hip = pair(hips_coords, debug=debug)
    left_hip, right_hip, hips_reconstructed = reconstruct_hips( left_hip, right_hip, left_shoulder, right_shoulder, image_size, image_size, debug=debug )

    #💇 Hair detected: [(197, 405), (300, 389), (403, 374)]  #'hair_root': 25, 'hair_left': 26, 'hair_right': 27, 'hair_top': 28,

    hair_left, hair_top_left, mouth_left_ext, nose_left, hair_top, hair_top_right, mouth_right_ext, right_top_hair1, right_top_hair2, right_top_hair3, left_top_hair1, left_top_hair2, left_top_hair3, nose_right, hair_right, hair_root, front_m, front_left1, front_left2, front_right1, front_right2  = pair(hair_coords, debug=debug)

    if verbose:
        print(f"Position de hair_root en pixels : {hair_root}")
    hair_center = np.mean([hair_left, hair_right], axis=0)
    # vecteur tête → cheveux -> Pose.get_head_up()
    head_up = np.array(hair_center) - np.array(left_clavicle)
    head_up = head_up / (np.linalg.norm(head_up) + 1e-6)

    # ============ Bouche =============

    # Vérifier si des coordonnées de la bouche ont été détectées
    if mouth_coords:
        if debug:
            # Afficher les coordonnées pour chaque partie de la bouche
            for key, value in mouth_coords.items():
                print(f"{key}: {value}")


        # Extraire les coordonnées du centre de la bouche
        mouth_center = mouth_coords.get('mouth_center')
        mouth_left = mouth_coords.get('mouth_left')
        mouth_right = mouth_coords.get('mouth_right')
        mouth_top_mid_r3 = mouth_coords.get('mouth_top_mid_r3')
        mouth_top_mid_r2 = mouth_coords.get('mouth_top_mid_r2')
        mouth_top_mid_r1 = mouth_coords.get('mouth_top_mid_r1')
        mouth_top_mid = mouth_coords.get('mouth_top_mid')
        mouth_top_mid_l1 = mouth_coords.get('mouth_top_mid_l1')
        mouth_top_mid_l2 = mouth_coords.get('mouth_top_mid_l2')
        mouth_top_mid_l3 = mouth_coords.get('mouth_top_mid_l3')

        mouth_bot_mid_r3 = mouth_coords.get('mouth_bot_mid_r3')
        mouth_bot_mid_r2 = mouth_coords.get('mouth_bot_mid_r2')
        mouth_bot_mid_r1 = mouth_coords.get('mouth_bot_mid_r1')
        mouth_bot_mid = mouth_coords.get('mouth_bot_mid')
        mouth_bot_mid_l1 = mouth_coords.get('mouth_bot_mid_l1')
        mouth_bot_mid_l2 = mouth_coords.get('mouth_bot_mid_l2')
        mouth_bot_mid_l3 = mouth_coords.get('mouth_bot_mid_l3')
        if verbose:
            print(f"Centre de la bouche: {mouth_center}")
    else:
        print("⚠️ Aucune bouche détectée.")


    # =========================================================
    # 🔥 KNEE ANKLE - GENOU ET CHEVILLE
    # =========================================================
    left_knee = (0,0)
    right_knee = (0,0)
    left_ankle = (0,0)
    left_ankle = (0,0)

    left_eye, right_eye, left_iris1, left_iris2, left_iris3, left_iris4, right_iris1, right_iris2, right_iris3, right_iris4 = pair(eye_coords, debug=debug)
    left_ear, right_ear  = pair(ear_coords, debug=debug)

    # =========================================================
    # 🔥 HIP - KNEE - ANKLE  RECONSTRUCTION (CRITICAL)
    # =========================================================
    if (left_hip is None or right_hip is None or
        left_hip == (0,0) or right_hip == (0,0)):

        if left_shoulder and right_shoulder:
            cx = (left_shoulder[0] + right_shoulder[0]) * 0.5
            cy = (left_shoulder[1] + right_shoulder[1]) * 0.5

            offset_y = H * 0.18
            left_hip  = (cx - 70, cy + offset_y)
            right_hip = (cx + 70, cy + offset_y)


            if debug:
                print("🦿 HIP RECONSTRUCTED")

    if (left_knee is None or right_knee is None or
        left_knee == (0,0) or right_knee == (0,0)):

        if left_hip and right_hip:
            cy = (left_hip[1] + right_hip[1]) * 0.5  # Calcul du centre

            offset_y2 = H * 0.12
            left_knee  = (left_hip[0], cy + offset_y2)
            right_knee = (right_hip[0], cy + offset_y2)

            if debug:
                print("🦿 KNEE RECONSTRUCTED BY HIP")

        elif left_shoulder and right_shoulder:
            cx = (left_shoulder[0] + right_shoulder[0]) * 0.5 # Calcul du centre
            cy = (left_shoulder[1] + right_shoulder[1]) * 0.5 # Calcul du centre

            offset_y2 = H * 0.26
            left_knee  = (cx - 90, cy + offset_y2)
            right_knee = (cx + 90, cy + offset_y2)

            if debug:
                print("🦿 KNEE RECONSTRUCTED BY SHOULDER")

    if (left_ankle is None or left_ankle is None or
        left_ankle == (0,0) or left_ankle == (0,0)):

        if left_hip and right_hip:
            cx = (left_hip[0] + right_hip[0]) * 0.5
            cy = (left_hip[1] + right_hip[1]) * 0.5

            offset_y2 = H * 0.1
            left_ankle  = (cx - 100, cy - offset_y2)
            right_ankle = (cx + 100, cy - offset_y2)

            if debug:
                print("🦿 ANKLE RECONSTRUCTED BY HIP")

        elif left_shoulder and right_shoulder:
            cx = (left_shoulder[0] + right_shoulder[0]) * 0.5
            cy = (left_shoulder[1] + right_shoulder[1]) * 0.5

            offset_y3 = H * 0.36
            left_ankle = (cx - 100, cy + offset_y3)
            right_ankle = (cx + 100, cy + offset_y3)

            if debug:
                print("🦿 ANKLE RECONSTRUCTED BY SHOULDER")
    # =========================================================
    # =========================================================
    # 🔹 NECK SAFE
    # =========================================================
    if isinstance(neck_coords, dict):
        neck_map = {
            "neck": neck_coords.get("center"),
            "chin": neck_coords.get("chin"),
            "left_side_neck": neck_coords.get("left"),
            "right_side_neck": neck_coords.get("right"),
            "anchor": neck_coords.get("anchor"),
        }
    else:
        neck_map = {"neck": neck_coords}



    # =========================================================
    # 🔹 SAFE UPDATE WRAPPER
    # =========================================================
    def safe_apply(idx, coord, label):
        coord = normalize_coord(coord)

        if is_valid(coord, label):
            safe_update(idx, coord, keypoints_np, W=W, H=H, label=label, debug=debug)
        else:
            if debug:
                print(f"[SKIP] {label} invalid → keep previous")

    # =========================================================
    # 🔹 APPLY UPDATES
    # =========================================================

    updates = {
        0: ("nose", nose_coords),
        1: ("neck", neck_map.get("neck")),

        2: ("right_shoulder", right_shoulder),
        3: ("right_elbow", right_elbow),
        4: ("right_wrist", right_wrist),

        5: ("left_shoulder", left_shoulder),
        6: ("left_elbow", left_elbow),
        7: ("left_wrist", left_wrist),

        8: ("right_hip", right_hip),
        9: ("right_knee", right_knee),
        10: ("right_ankle", right_ankle),

        11: ("left_hip", left_hip),
        12: ("left_knee", left_knee),
        13: ("left_ankle", left_ankle),

        14: ("right_eye", right_eye),
        15: ("left_eye", left_eye),

        16: ("right_ear", right_ear),
        17: ("left_ear", left_ear),

        18: ("mouth", mouth_center),

        19: ("right_clavicle", right_clavicle),
        20: ("left_clavicle", left_clavicle),

        21: ("chin", neck_map.get("chin")),
        22: ("left_side_neck", neck_map.get("left_side_neck")),
        23: ("right_side_neck", neck_map.get("right_side_neck")),
        24: ("anchor", neck_map.get("anchor")),


        25: ("hair_root", hair_root), #OK
        26: ("hair_left", hair_left), #OK
        27: ("hair_right", hair_right), #OK
        28: ("hair_top", hair_top), #OK
        29: ("hair_top_left", hair_top_left), #OK
        30: ("hair_top_right", hair_top_right), #OK

        31: ("left_top_hair1", left_top_hair1),
        32: ("left_top_hair2", left_top_hair2),
        33: ("left_top_hair3", left_top_hair3), #OK

        34: ("right_top_hair1", right_top_hair1),
        35: ("right_top_hair2", right_top_hair2), #OK
        36: ("right_top_hair3", right_top_hair3),

        #37: ("top_hair1", top_hair1), #OK
        #38: ("top_hair2", top_hair2), #OK
        #39: ("top_hair3", top_hair3), #OK

        38: ("mouth_left_ext", mouth_left_ext),
        39: ("mouth_right_ext", mouth_right_ext),

        40: ("mouth_left", mouth_left),
        41: ("mouth_right", mouth_right),

        42: ("nose_left", nose_left),
        43: ("nose_right", nose_right),


        52: ("front_left_1", front_left1), #OK
        53: ("front_left_2", front_left2), #OK
        54: ("front_m", front_m), #OK
        55: ("front_right_1", front_right1), #OK
        56: ("front_right_2", front_right2), #OK


        70: ("mouth_top_mid_r3", mouth_top_mid_r3), #OK
        71: ("mouth_top_mid_r2", mouth_top_mid_r2), #OK
        72: ("mouth_top_mid_r1", mouth_top_mid_r1), #OK
        73: ("mouth_top_mid", mouth_top_mid), #OK
        74: ("mouth_top_mid_l1", mouth_top_mid_l1), #OK
        75: ("mouth_top_mid_l2", mouth_top_mid_l2), #OK
        76: ("mouth_top_mid_l3", mouth_top_mid_l3), #OK

        77: ("mouth_bot_mid_r3", mouth_bot_mid_r3), #OK
        78: ("mouth_bot_mid_r2", mouth_bot_mid_r2), #OK
        79: ("mouth_bot_mid_r1", mouth_bot_mid_r1), #OK
        80: ("mouth_bot_mid", mouth_bot_mid), #OK
        81: ("mouth_bot_mid_l1", mouth_bot_mid_l1), #OK
        82: ("mouth_bot_mid_l2", mouth_bot_mid_l2), #OK
        83: ("mouth_bot_mid_l3", mouth_bot_mid_l3), #OK




    }

    for idx, (label, coord) in updates.items():
        safe_apply(idx, coord, label)

    # =========================================================
    # 🔹 BACK TO TENSOR
    # =========================================================
    keypoints_np = np.expand_dims(keypoints_np, axis=0)
    keypoints_np = np.repeat(keypoints_np, current_keypoints.shape[0], axis=0)

    keypoints_tensor = torch.from_numpy(keypoints_np).to(device)

    # =========================================================
    # 🔹 DEBUG
    # =========================================================
    if debug and debug_dir is not None:
        debug_draw_openpose_skeleton(
            keypoints_tensor=keypoints_tensor,
            debug_dir=debug_dir,
            frame_counter=frame_counter,
            image_size=image_size
        )

    return keypoints_tensor


#----------------------------------------------------------------------------------------------------------------

def resize_pose(pose_tile, H_latent, W_latent):
    target_h = H_latent * 8
    target_w = W_latent * 8

    if pose_tile.shape[-2:] != (target_h, target_w):
        return F.interpolate(
            pose_tile,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
    return pose_tile


def prepare_inputs(latent_tile, pose_tile, cf_embeds, device):
    pos_embeds, neg_embeds = cf_embeds

    latent_fp32 = latent_tile.to(device=device, dtype=torch.float32)
    pose_fp32 = pose_tile.to(device=device, dtype=torch.float32)

    pos_fp32 = pos_embeds.to(device=device, dtype=torch.float32)
    neg_fp32 = neg_embeds.to(device=device, dtype=torch.float32) if neg_embeds is not None else None

    return latent_fp32, pose_fp32, pos_fp32, neg_fp32


def add_noise(latent, scheduler, t, noise_strength=0.5):
    noise = torch.randn_like(latent) * noise_strength
    latent_noisy = scheduler.add_noise(latent, noise, t)
    return torch.clamp(latent_noisy, -20, 20)


def apply_cfg(latent_input, pos_embeds, neg_embeds, guidance_scale):
    if neg_embeds is not None:
        latent_input = torch.cat([latent_input] * 2)
        embeds = torch.cat([neg_embeds, pos_embeds])
        return latent_input, embeds, True
    return latent_input, pos_embeds, False


def compute_noise_pred(unet, controlnet, latent_input, t, embeds, pose):
    down_samples, mid_sample = controlnet(
        latent_input,
        t,
        encoder_hidden_states=embeds,
        controlnet_cond=pose,
        return_dict=False
    )

    noise_pred = unet(
        latent_input,
        t,
        encoder_hidden_states=embeds,
        down_block_additional_residuals=down_samples,
        mid_block_additional_residual=mid_sample,
        return_dict=False
    )[0]

    return torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)


def merge_cfg(noise_pred, guidance_scale, use_cfg):
    if use_cfg:
        noise_uncond, noise_text = noise_pred.chunk(2)
        return noise_uncond + guidance_scale * (noise_text - noise_uncond)
    return noise_pred


def compute_adaptive_importance(latent):
    with torch.no_grad():
        blurred = F.avg_pool2d(latent, kernel_size=3, stride=1, padding=1)
        high_freq = torch.abs(latent - blurred)
        importance = high_freq.mean(dim=1, keepdim=True)
        # normalisation douce
        importance = importance / (importance.mean() + 1e-6)
        # compression pour éviter extrêmes
        importance = torch.sqrt(importance)
        return torch.clamp(importance, 0.7, 1.3)



def controlnet_tile_fn(
    latent_tile,
    pose_tile,
    frame_counter,
    unet,
    controlnet,
    scheduler,
    cf_embeds,
    current_guidance_scale,
    controlnet_scale,
    device,
    target_dtype,
    **kwargs
):

    B, C, H_latent, W_latent = latent_tile.shape

    # 1️⃣ Resize pose
    pose_resized = resize_pose(pose_tile, H_latent, W_latent)
    # 2️⃣ Inputs
    latent_fp32, pose_fp32, pos_embeds, neg_embeds = prepare_inputs(
        latent_tile, pose_resized, cf_embeds, device
    )
    # 3️⃣ Timestep
    t = scheduler.timesteps[min(frame_counter, len(scheduler.timesteps) - 1)]

    # =========================================================
    # 4️⃣ Noise
    # =========================================================
    latent_noisy = add_noise(latent_fp32, scheduler, t)

    latent_input = scheduler.scale_model_input(latent_noisy, t)

    # =========================================================
    # 5️⃣ CFG
    # =========================================================
    latent_input, embeds, use_cfg = apply_cfg(
        latent_input, pos_embeds, neg_embeds, current_guidance_scale
    )

    latent_input = latent_input.to(target_dtype)
    embeds = embeds.to(target_dtype)
    pose_fp32 = pose_fp32.to(target_dtype)

    # =========================================================
    # 6️⃣ UNet + ControlNet
    # =========================================================
    noise_pred = compute_noise_pred(
        unet, controlnet, latent_input, t, embeds, pose_fp32
    )

    noise_pred = merge_cfg(noise_pred, current_guidance_scale, use_cfg)

    # =========================================================
    # 7️⃣ Scheduler step
    # =========================================================
    latents_out = scheduler.step(noise_pred, t, latent_noisy).prev_sample

    # =========================================================
    # 🔥 8️⃣ Adaptive blending (clé)
    # =========================================================
    importance = compute_adaptive_importance(latent_fp32)

    delta = compute_delta( latents_out, latent_fp32, controlnet_scale, importance )

    latents_final = latent_fp32 + delta

    return latents_final.to(target_dtype)

#---------------------------------------------------------------------------------------------------------------------------------------------
def log_frame_error(img_path, error: Exception, verbose: bool = True):
    print(f"\n[FRAME ERROR] {img_path}")
    print(f"Type d'erreur : {type(error).__name__}")
    print(f"Message d'erreur : {error}")

    if verbose:
        print("Traceback complet :")
        traceback.print_exc()


def prepare_controlnet(
    controlnet,
    freeze: bool = True,
    enable_slicing: bool = True,
    device=None,
    dtype=None,
    verbose: bool = True
):
    """
    Prépare un ControlNet :
    - eval mode
    - freeze des poids
    - attention slicing (si dispo)
    - move device / dtype
    - init pose_sequence

    Returns:
        controlnet, pose_sequence (None par défaut)
    """

    # ---- eval mode
    controlnet.eval()
    if verbose:
        print("✅ ControlNet en mode eval")

    # ---- freeze
    if freeze:
        for p in controlnet.parameters():
            p.requires_grad = False
        if verbose:
            print("✅ Paramètres gelés")

    # ---- attention slicing
    if enable_slicing:
        fn = getattr(controlnet, "enable_attention_slicing", None)
        if callable(fn):
            fn()
            if verbose:
                print("✅ Attention slicing activé")
        else:
            if verbose:
                print("⚠ enable_attention_slicing non disponible")

    # ---- device / dtype
    if device is not None or dtype is not None:
        controlnet = controlnet.to(device=device, dtype=dtype)
        if verbose:
            print(f"✅ Déplacé sur {device} / {dtype}")

    # ---- init pose
    pose_sequence = None

    return controlnet, pose_sequence

def fix_pose_sequence(
    pose_sequence: torch.Tensor,
    total_frames: int,
    device=None,
    dtype=None,
    verbose: bool = True
) -> torch.Tensor:
    """
    Ajuste une séquence de poses au bon nombre de frames avec interpolation.

    Args:
        pose_sequence: Tensor (F, C, H, W)
        total_frames: nombre de frames cible
        device: device cible (optionnel)
        dtype: dtype cible (optionnel)
        verbose: afficher logs

    Returns:
        Tensor (F, C, H, W)
    """
    print(f"🎞 fix_pose_sequence - Frames JSON: {pose_sequence.shape[0]}")
    print(f"🎞 fix_pose_sequence - Frames attendues: {total_frames}")

    if pose_sequence.shape[0] != total_frames:
        if verbose:
            print("⚠ Ajustement du nombre de frames OpenPose")

        # (F, C, H, W) → (1, C, F, H, W)
        pose_sequence = pose_sequence.permute(1, 0, 2, 3).unsqueeze(0)

        pose_sequence = F.interpolate(
            pose_sequence,
            size=(total_frames, pose_sequence.shape[-2], pose_sequence.shape[-1]),
            mode='trilinear',
            align_corners=False
        )

        # retour → (F, C, H, W)
        pose_sequence = pose_sequence.squeeze(0).permute(1, 0, 2, 3)

    # Fix device + dtype
    if device is not None or dtype is not None:
        pose_sequence = pose_sequence.to(device=device, dtype=dtype)

    if verbose:
        print(
            "✅ PoseSequence final:",
            pose_sequence.shape,
            pose_sequence.device,
            pose_sequence.dtype
        )

    return pose_sequence

def tensor_to_pil(tensor):
    """
    Convertit un tensor torch [C,H,W] ou [H,W] en PIL.Image RGB.
    """
    if tensor.dim() == 3:
        C, H, W = tensor.shape
        if C == 1:
            array = tensor[0].cpu().numpy()  # [H,W]
            pil_img = Image.fromarray(array).convert("RGB")
        elif C == 3:
            array = tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
            pil_img = Image.fromarray(array)
        else:
            raise ValueError(f"Tensor avec {C} canaux non supporté")
    elif tensor.dim() == 2:
        pil_img = Image.fromarray(tensor.cpu().numpy()).convert("RGB")
    else:
        raise ValueError(f"Tensor shape non supportée: {tensor.shape}")
    return pil_img



def save_debug_pose_image(pose_tensor, frame_counter, output_dir, cfg=None, prefix="openpose"):
    """
    Sauvegarde une image de pose pour contrôle visuel.

    pose_tensor : torch.Tensor [C,H,W] ou [H,W]
    frame_counter : int, numéro de frame
    output_dir : str, dossier où sauvegarder
    cfg : dict ou None, peut contenir paramètre 'visual_debug' pour activer/désactiver
    prefix : str, préfixe du fichier
    """

    # Vérifie si le debug visuel est activé
    if cfg is not None and cfg.get("visual_debug") is False:
        return

    # Convertir tensor en uint8 [0,255]
    pose_img = (pose_tensor * 255).clamp(0, 255).byte()

    # Fonction interne pour gérer tous les formats [C,H,W], [H,W]
    def tensor_to_pil(tensor):
        if tensor.dim() == 3:
            C, H, W = tensor.shape
            if C == 1:
                array = tensor[0].cpu().numpy()  # [H,W]
                pil_img = Image.fromarray(array).convert("RGB")
            elif C == 3:
                array = tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,C]
                pil_img = Image.fromarray(array)
            else:
                raise ValueError(f"Tensor avec {C} canaux non supporté")
        elif tensor.dim() == 2:
            pil_img = Image.fromarray(tensor.cpu().numpy()).convert("RGB")
        else:
            # Si la tensor a une forme inattendue, on essaie de la "squeezer"
            tensor = tensor.squeeze()
            if tensor.dim() in [2, 3]:
                return tensor_to_pil(tensor)
            raise ValueError(f"Tensor shape non supportée: {tensor.shape}")
        return pil_img

    pil_pose = tensor_to_pil(pose_img)

    # Création du dossier si nécessaire
    os.makedirs(output_dir, exist_ok=True)

    # Nom du fichier : openpose_00001.png
    filename = f"{prefix}_{frame_counter:05d}.png"
    path = os.path.join(output_dir, filename)

    pil_pose.save(path)
    print(f"[DEBUG] Pose sauvegardée : {path}")


def debug_pose_visual(pose_tensor, frame_counter, cfg=None, title="Pose Debug"):
    """
    Affiche la pose détectée pour vérification visuelle.

    Args:
        pose_tensor (torch.Tensor): Tensor BCHW ou CHW (1,3,H,W ou 3,H,W)
        frame_counter (int): numéro de la frame
        cfg (dict, optional): configuration, active si cfg.get("debug_pose_visual", False) est True
        title (str): titre pour l'affichage
    """
    if cfg is None or not cfg.get("debug_pose_visual", False):
        return

    # S'assurer que le tensor est BCHW
    if pose_tensor.ndim == 3:  # CHW -> BCHW
        pose_tensor = pose_tensor.unsqueeze(0)

    pose_tensor = pose_tensor[0]  # retirer batch

    # Limiter à 3 canaux
    if pose_tensor.shape[0] > 3:
        pose_tensor = pose_tensor[:3, :, :]

    # CHW -> HWC pour PIL
    pose_np = pose_tensor.permute(1, 2, 0).cpu().numpy()
    pose_np = (pose_np - pose_np.min()) / (pose_np.max() - pose_np.min() + 1e-8) * 255.0
    pose_np = pose_np.astype("uint8")
    img = Image.fromarray(pose_np)

    # Affichage rapide avec matplotlib
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{title} - Frame {frame_counter}")
    plt.show(block=False)
    plt.pause(0.1)  # court délai pour rafraîchir
    plt.close()


#------------- JSON TO POSE SEQUENCE --------------------

def convert_json_to_pose_sequence(anim_data, H=512, W=512, device="cuda", dtype=torch.float16, debug=False, output_dir=None):
    """
    Convertit un JSON d'animation OpenPose en tensor ControlNet, avec centrage et scaling automatique.
    Output : [num_frames, 3, H, W], dtype et device configurables.
    """
    frames = anim_data.get("animation", [])
    pose_images = []

    # --- Détecter le bounding box global des keypoints ---
    all_x = []
    all_y = []
    for frame in frames:
        for kp in frame.get("keypoints", []):
            all_x.append(kp["x"])
            all_y.append(kp["y"])

    if len(all_x) == 0 or len(all_y) == 0:
        raise ValueError("Aucun keypoint trouvé dans le JSON.")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Scale et translation pour centrer et remplir le canvas
    scale_x = (W - 20) / (max_x - min_x + 1e-6)  # marge 10px
    scale_y = (H - 20) / (max_y - min_y + 1e-6)
    scale = min(scale_x, scale_y)

    offset_x = (W - (max_x - min_x) * scale) / 2 - min_x * scale
    offset_y = (H - (max_y - min_y) * scale) / 2 - min_y * scale

    for idx, frame in enumerate(frames):
        keypoints = frame.get("keypoints", [])
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # --- Dessin des points ---
        for kp in keypoints:
            x = int(kp["x"] * scale + offset_x)
            y = int(kp["y"] * scale + offset_y)
            conf = kp.get("confidence", 1.0)
            if conf > 0.3:
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

        # --- Dessin des connexions ---
        skeleton = [
            (0, 1),  # tête → torse
            (1, 2),  # torse → bras gauche
            (1, 3),  # torse → bras droit
            (1, 4),  # torse → jambe gauche
            (1, 5),  # torse → jambe droite
        ]
        for a, b in skeleton:
            if a < len(keypoints) and b < len(keypoints):
                x1 = int(keypoints[a]["x"] * scale + offset_x)
                y1 = int(keypoints[a]["y"] * scale + offset_y)
                x2 = int(keypoints[b]["x"] * scale + offset_x)
                y2 = int(keypoints[b]["y"] * scale + offset_y)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        img = torch.from_numpy(canvas).float() / 255.0
        img = img.permute(2, 0, 1)  # C,H,W
        pose_images.append(img)

        # Debug
        if debug and output_dir is not None:
            cv2.imwrite(f"{output_dir}/debug_pose_{idx:03d}.png", canvas)

    pose_sequence = torch.stack(pose_images).to(device=device, dtype=dtype)
    pose_sequence = pose_sequence * 2.0 - 1.0  # [-1,1]

    if debug:
        print(f"[JSON->POSE] shape: {pose_sequence.shape}")
        print(f"[JSON->POSE] min/max: {pose_sequence.min().item()} / {pose_sequence.max().item()}")

    return pose_sequence

def convert_json_to_pose_sequence_debug(anim_data, H=512, W=512, original_w=512, original_h=512,
                                  device="cuda", dtype=torch.float16, debug=False, output_dir=None):
    """
    Convertit un JSON d'animation OpenPose simplifié en tensor utilisable par ControlNet.

    Args:
        anim_data: dict JSON avec "animation" -> frames -> keypoints
        H, W: résolution finale du canvas
        original_w, original_h: résolution originale des keypoints
        device: "cuda" ou "cpu"
        dtype: torch dtype (ex: torch.float16)
        debug: bool, sauvegarde les images pour visualisation
        output_dir: chemin pour debug images (optionnel)

    Returns:
        pose_sequence: tensor [num_frames, 3, H, W] (RGB type)
    """
    frames = anim_data.get("animation", [])
    pose_images = []

    for idx, frame in enumerate(frames):
        keypoints = frame.get("keypoints", [])

        # Image noire
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # --- Dessin des points ---
        for kp in keypoints:
            # remapping keypoints vers la résolution finale
            x = int(kp["x"] * W / original_w)
            y = int(kp["y"] * H / original_h)
            conf = kp.get("confidence", 1.0)

            if conf > 0.3:
                cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

        # --- Dessin des connexions (squelette simple) ---
        skeleton = [
            (0, 1),  # tête → torse
            (1, 2),  # torse → bras gauche
            (1, 3),  # torse → bras droit
            (1, 4),  # torse → jambe gauche
            (1, 5),  # torse → jambe droite
        ]

        for a, b in skeleton:
            if a < len(keypoints) and b < len(keypoints):
                x1 = int(keypoints[a]["x"] * W / original_w)
                y1 = int(keypoints[a]["y"] * H / original_h)
                x2 = int(keypoints[b]["x"] * W / original_w)
                y2 = int(keypoints[b]["y"] * H / original_h)
                cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # --- Conversion en tensor ---
        img = torch.from_numpy(canvas).float() / 255.0  # [H, W, C]
        img = img.permute(2, 0, 1)  # → [C, H, W]

        pose_images.append(img)

        # --- Debug save ---
        if debug and output_dir is not None:
            debug_path = f"{output_dir}/debug_pose_{idx:03d}.png"
            cv2.imwrite(debug_path, (canvas).astype(np.uint8))

    # --- Stack frames + normalisation [-1,1] ---
    pose_sequence = torch.stack(pose_images).to(device=device, dtype=dtype)
    pose_sequence = pose_sequence * 2.0 - 1.0  # [0,1] → [-1,1]

    if debug:
        print(f"[JSON->POSE] shape: {pose_sequence.shape}")
        print(f"[JSON->POSE] min/max: {pose_sequence.min().item()} / {pose_sequence.max().item()}")

    return pose_sequence



def build_control_latent_debug(input_pil, vae, device="cuda", latent_scale=0.18215):

    print("\n================ CONTROL LATENT DEBUG ================")

    # 1. Canny
    control = create_canny_control(input_pil)

    print("[STEP 1] RAW CONTROL")
    print(" shape:", control.shape)
    print(" dtype:", control.dtype)
    print(" min/max:", control.min().item(), control.max().item())

    # 2. 1 → 3 channels
    if control.shape[1] == 1:
        control = control.repeat(1, 3, 1, 1)

    # 3. Normalize PROPERLY (CRUCIAL)
    control = control.clamp(0, 1)          # sécurité
    control = control * 2.0 - 1.0          # [-1,1]

    print("[STEP 2] NORMALIZED")
    print(" min/max:", control.min().item(), control.max().item())

    # 4. Move to device FP32
    control = control.to(device=device, dtype=torch.float32)

    print("[STEP 3] DEVICE")
    print(" device:", control.device)
    print(" dtype:", control.dtype)

    # 5. Sync VAE
    print("[STEP 4] VAE STATE")
    print(" vae dtype:", next(vae.parameters()).dtype)
    print(" vae device:", next(vae.parameters()).device)

    # 🔥 FORCER cohérence VAE
    vae = vae.to(device=device, dtype=torch.float32)

    # 6. Encode SAFE (no autocast)
    with torch.no_grad():
        try:
            latent_dist = vae.encode(control).latent_dist
            latent = latent_dist.sample()
        except Exception as e:
            print("❌ VAE ENCODE CRASH:", e)
            raise

    print("[STEP 5] LATENT RAW")
    print(" min/max:", latent.min().item(), latent.max().item())
    print(" NaN:", torch.isnan(latent).sum().item())

    # 🚨 CHECK NaN
    if torch.isnan(latent).any():
        print("⚠️ NaN DETECTED → applying fallback")

        # fallback 1: zero latent
        latent = torch.zeros_like(latent)

        # fallback 2 (optionnel):
        # latent = torch.randn_like(latent) * 0.1

    # 7. Scale (SD standard)
    latent = latent * latent_scale

    print("[STEP 6] SCALED LATENT")
    print(" min/max:", latent.min().item(), latent.max().item())

    # 8. Back to FP16
    latent = latent.to(dtype=torch.float16)

    print("[FINAL]")
    print(" dtype:", latent.dtype)
    print(" device:", latent.device)
    print("=====================================================\n")

    return latent

# ---------------- Control -> Latent sécurisé ----------------
def control_to_latent_safe(control_tensor, vae, device="cuda", LATENT_SCALE=1.0):
    # 🔥 FORCE VAE EN FP32
    vae = vae.to(device=device, dtype=torch.float32)

    control_tensor = control_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        latent = vae.encode(control_tensor).latent_dist.sample()

    return latent * LATENT_SCALE

def process_latents_streamed(control_latent, mini_latents=None, mini_weight=0.5, device="cuda"):
    """
    Fusionne ControlNet / mini-latents frame par frame, patch par patch
    pour réduire l'empreinte VRAM.
    """
    # On garde tout en float16 tant que possible
    control_latent = control_latent.to(device=device, dtype=torch.float16)

    if mini_latents is not None:
        mini_latents = mini_latents.to(device=device, dtype=torch.float16)

    # Initialisation finale du tensor latents en float16
    latents = control_latent.clone()

    # Si mini_latents existe, on fait un mix patch par patch
    if mini_latents is not None:
        B, C, H, W = latents.shape
        patch_size = 16  # petit patch pour limiter la VRAM
        for y in range(0, H, patch_size):
            y1 = min(y + patch_size, H)
            for x in range(0, W, patch_size):
                x1 = min(x + patch_size, W)

                # Sélection patch
                patch_main = latents[:, :, y:y1, x:x1]
                patch_mini = mini_latents[:, :, y:y1, x:x1]

                # Mix float16 → float16 pour VRAM
                patch_main = (1 - mini_weight) * patch_main + mini_weight * patch_mini

                # Écriture patch back
                latents[:, :, y:y1, x:x1] = patch_main

                # Nettoyage immédiat pour libérer VRAM
                del patch_main, patch_mini
                torch.cuda.empty_cache()

    return latents


def match_latent_size(latents_main, *tensors):
    """
    Interpole tous les tensors pour correspondre à la taille HxW de latents_main.
    """
    matched = []
    for t in tensors:
        if t.shape[2:] != latents_main.shape[2:]:
            t = F.interpolate(t, size=latents_main.shape[2:], mode='bilinear', align_corners=False)
        matched.append(t)
    return matched if len(matched) > 1 else matched[0]

def gaussian_blend_mask(H, W, overlap):
    """Crée un masque gaussien pour fusionner les tiles avec overlap."""

    y = np.linspace(-1,1,H)
    x = np.linspace(-1,1,W)
    xv, yv = np.meshgrid(x,y)
    mask = np.exp(-(xv**2 + yv**2) / 0.5)  # ajuste le sigma si nécessaire
    mask = torch.tensor(mask, dtype=torch.float32)
    return mask


#---------------------------------------------------------

# 🔹 Récupère les coordonnées (x,y) d’un keypoint spécifique dans le batch
# 🔹 Récupère les coordonnées (x,y) d’un keypoint spécifique dans le batch
def get_point(kp_tensor, idx):
    return kp_tensor[:, idx, :2]  # [B,2]

# 🔹 Applique la différence entre latents avant/après OpenPose
#   Permet de conserver l’impact du pose controlnet.
def apply_openpose_delta(latents, latents_before, latents_after, mask):
    if latents_before is not None and latents_after is not None:
        delta = latents_after - latents_before
        delta = torch.clamp(delta, -0.15, 0.15)
        latents = latents + delta * mask * 0.5
    return latents

# -------------------- Fonction utilitaire --------------------
def compute_torso_angle(keypoints):
    """
    Calcule l'angle du torse selon les épaules (radians).
    """
    right_shoulder = get_point(keypoints, 2)
    left_shoulder = get_point(keypoints, 5)
    vec = right_shoulder - left_shoulder
    angle = torch.atan2(vec[:,1], vec[:,0])  # [B]
    torso_center = (right_shoulder + left_shoulder) * 0.5
    return angle, torso_center


# -------------------- Fonction principale -----------------------------------------------------------------

def apply_torso_warp(
    latents,
    pose,
    mask_torso,
    grid,
    H,
    W,
    device,
    prev_delta_px=None,
    strength=0.2,   # 🔥 NEW: 0.3
    debug=False,
    debug_dir=None
):
    B = latents.shape[0]

    # -------------------- Centre du torse --------------------
    # Points clés pour le torse : épaules et hanches
    points_idx = [2, 5, 8, 11]  # 'right_shoulder', 'left_shoulder', 'right_hip', 'left_hip'
    pts = torch.stack([pose.get_point(i) for i in points_idx], dim=1)  # [B, 4, 2]

    # Calcul du centre du torse
    torso_center = pts.mean(dim=1)  # [B, 2]
    torso_center_px = torso_center * torch.tensor([W - 1, H - 1], device=device)  # [B, 2]
    torso_center_px = torso_center_px.view(B, 1, 1, 2)  # [B, 1, 1, 2]

    # -------------------- Delta torse --------------------
    delta_px = pose.delta.clone()  # delta initial
    delta_px[..., 0] *= W  # mise à l'échelle en fonction de la largeur
    delta_px[..., 1] *= H  # mise à l'échelle en fonction de la hauteur
    delta_px = delta_px.view(B, 1, 1, 2)  # [B, 1, 1, 2]
    print("delta_px max:", delta_px.abs().max().item())

    # -------------------- Lissage temporel --------------------
    if prev_delta_px is not None:
        alpha = 0.7  # facteur de lissage
        delta_px = alpha * delta_px + (1 - alpha) * prev_delta_px

    # -------------------- Strength control --------------------
    # Appliquer une transformation contrôlée par la force (strength)
    delta_px = torch.tanh(delta_px) * 5.0 * strength

    # -------------------- Feather dynamique --------------------
    mask_torso = feather_dynamic_vectorized(
        mask_torso,
        delta_px,
        base_radius=3,
        sigma=1.5,
        scale=2.0
    )

    mask_expand = mask_torso.permute(0, 2, 3, 1)  # Permuter pour correspondre aux dimensions attendues

    # -------------------- Déformation non-linéaire (IMPORTANT) --------------------
    # Calcul du vecteur d'offset (écart par rapport au centre du torse)
    offset = grid - torso_center_px
    distance = torch.norm(offset, dim=-1, keepdim=True)  # Calcul de la distance euclidienne
    falloff = torch.exp(-distance / (0.35 * W))  # Fonction d'atténuation basée sur la distance

    # -------------------- Warp torse --------------------
    # Appliquer le warp basé sur delta_px et la force
    warp = delta_px * strength
    warp = torch.tanh(warp / 5.0) * 5.0  # Clamp sécuritaire pour éviter les déformations extrêmes
    grid_torso = grid + warp * mask_expand * falloff  # Appliquer l'atténuation

    # -------------------- Normalisation --------------------
    # Normaliser les coordonnées du grid pour être dans l'intervalle [-1, 1] pour grid_sample
    grid_torso[..., 0] = 2.0 * grid_torso[..., 0] / (W - 1) - 1.0
    grid_torso[..., 1] = 2.0 * grid_torso[..., 1] / (H - 1) - 1.0

    # -------------------- Sampling --------------------
    # Appliquer la transformation de grille aux latents
    latents_out = F.grid_sample(latents, grid_torso, align_corners=True)

    # -------------------- Debug --------------------
    if debug:
        print("[DEBUG][TORSO] strength:", strength)
        print("[DEBUG][TORSO] delta_px mean:", delta_px.abs().mean().item())
        print("[DEBUG][TORSO] delta_px max:", delta_px.abs().max().item())
        print("[DEBUG][TORSO] falloff mean:", falloff.mean().item())
        print("[DEBUG][TORSO] mask mean:", mask_expand.mean().item())

    # -------------------- Return --------------------
    return latents_out, delta_px


# version corriger
import torch
import torch.nn.functional as F
import math
import time
import os

def apply_global_pose(
    latents,
    pose,
    prev_pose=None,
    H=None,
    W=None,
    device="cuda",
    strength=0.4,
    clamp=True,
    debug=True,
    debug_dir=None
):

    B, C, H_lat, W_lat = latents.shape
    device = latents.device

    t0 = time.time()

    # =========================================================
    # 🔹 Identity fallback
    # =========================================================
    if prev_pose is None:
        yy, xx = torch.meshgrid(
            torch.arange(H_lat, device=device),
            torch.arange(W_lat, device=device),
            indexing="ij"
        )
        grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)
        global_angle = torch.zeros((B, 1, 1), device=device)
        return latents, torch.zeros((B, 1, 1, 2), device=device), grid, None, torch.zeros((B, 1, 1, 2), device=device), global_angle

    # =========================================================
    # 🔹 Key joints
    # =========================================================
    joints = [
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    idx = [pose.FACIAL_POINT_IDX[j] for j in joints]

    pts = torch.stack([pose.get_point(i) for i in idx], dim=1)  # [B, 12, 2]
    prev_pts = torch.stack([prev_pose.get_point(i) for i in idx], 1)

    # =========================================================
    # 🔹 Global center motion (translation)
    # =========================================================
    center = pts.mean(1)
    prev_center = prev_pts.mean(1)
    delta = center - prev_center

    # =========================================================
    # 🔹 Spine axis (NEW)
    # =========================================================
    spine_top = (pts[:, 0] + pts[:, 1]) * 0.5  # shoulders
    spine_bottom = (pts[:, 2] + pts[:, 3]) * 0.5  # hips

    spine_vec = spine_top - spine_bottom
    prev_spine_vec = (prev_pts[:, 0] + prev_pts[:, 1]) * 0.5 - (prev_pts[:, 2] + prev_pts[:, 3]) * 0.5

    spine_len = torch.norm(spine_vec, dim=-1, keepdim=True).clamp(1e-6)
    prev_len = torch.norm(prev_spine_vec, dim=-1, keepdim=True).clamp(1e-6)

    spine_dir = spine_vec / spine_len
    prev_dir = prev_spine_vec / prev_len

    # angle via dot/cross stable
    dot = (spine_dir * prev_dir).sum(-1, keepdim=True).clamp(-1, 1)
    angle = torch.acos(dot)

    # signed approx
    cross = spine_dir[..., 0] * prev_dir[..., 1] - spine_dir[..., 1] * prev_dir[..., 0]
    angle = angle * torch.sign(cross).unsqueeze(-1)

    # =========================================================
    # 🔹 Temporal smoothing (IMPORTANT)
    # =========================================================
    delta = 0.8 * delta + 0.2 * getattr(pose, "_vel", torch.zeros_like(delta))
    pose._vel = delta.detach()

    # =========================================================
    # 🔹 Pixel conversion
    # =========================================================
    delta_px = torch.zeros_like(delta)
    delta_px[..., 0] = delta[..., 0] * W_lat
    delta_px[..., 1] = delta[..., 1] * H_lat

    if clamp:
        delta_px = torch.tanh(delta_px / 4.0) * 4.0

    delta_px = delta_px.view(B, 1, 1, 2)

    # =========================================================
    # 🔹 Base grid
    # =========================================================
    yy, xx = torch.meshgrid(
        torch.arange(H_lat, device=device),
        torch.arange(W_lat, device=device),
        indexing="ij"
    )

    grid = torch.stack((xx, yy), -1).float().unsqueeze(0).repeat(B, 1, 1, 1)

    # =========================================================
    # 🔹 CENTER-BASED ROTATION (stable)
    # =========================================================
    center_px = center.clone()
    center_px[..., 0] *= W_lat
    center_px[..., 1] *= H_lat
    center_px = center_px.view(B, 1, 1, 2)

    theta = angle * strength * 0.6

    cos_t = torch.cos(theta).view(B, 1, 1, 1)
    sin_t = torch.sin(theta).view(B, 1, 1, 1)

    x = grid[..., 0:1] - center_px[..., 0:1]
    y = grid[..., 1:2] - center_px[..., 1:2]

    rot_x = x * cos_t - y * sin_t
    rot_y = x * sin_t + y * cos_t

    grid = torch.cat([rot_x + center_px[..., 0:1],
                      rot_y + center_px[..., 1:2]], dim=-1)

    # =========================================================
    # 🔹 SPINE AXIS DEFORMATION (NEW CORE FEATURE)
    # =========================================================
    spine_mid = (spine_top + spine_bottom) * 0.5
    spine_mid_px = spine_mid.clone()
    spine_mid_px[..., 0] *= W_lat
    spine_mid_px[..., 1] *= H_lat
    spine_mid_px = spine_mid_px.view(B, 1, 1, 2)

    spine_dir_px = spine_dir.clone()
    spine_dir_px = spine_dir_px.view(B, 1, 1, 2)

    # projection of pixel onto spine axis
    rel = grid - spine_mid_px
    proj = (rel * spine_dir_px).sum(-1, keepdim=True) * spine_dir_px

    ortho = rel - proj

    # squash/stretch VERY subtle
    stretch = 1.0 + torch.tanh(delta_px.norm(dim=-1, keepdim=True)) * 0.03
    squash = 1.0 - torch.tanh(delta_px.norm(dim=-1, keepdim=True)) * 0.02

    grid = spine_mid_px + proj * stretch + ortho * squash

    # =========================================================
    # 🔹 FINAL MOTION
    # =========================================================
    global_gain = 1.6

    grid = grid + delta_px * strength * global_gain

    # =========================================================
    # 🔥 REAL WARP SHIFT (CRITICAL FIX)
    # =========================================================
    base_grid = torch.stack((xx, yy), -1).float().unsqueeze(0).repeat(B, 1, 1, 1)

    warp_delta = grid - base_grid  # (B,H,W,2)

    # moyenne globale → shift réel
    warp_shift = warp_delta.mean(dim=(1, 2), keepdim=True)  # (B,1,1,2)

    # clamp sécurité (évite spikes)
    warp_shift = torch.clamp(warp_shift, -10.0, 10.0)

    # =========================================================
    # 🔹 Normalize for grid_sample
    # =========================================================
    grid_norm = grid.clone()
    grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (W_lat - 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (H_lat - 1) - 1.0

    # =========================================================
    # 🔹 Apply warp
    # =========================================================
    latents_out = F.grid_sample(latents, grid_norm, align_corners=True)

    # =========================================================
    # 🔹 DEBUG
    # =========================================================
    if debug:
        print("\n[DEBUG][GLOBAL] ======= apply_global_pose ==============")
        print(f"Time: {time.time()-t0:.4f}s")
        print(f"Strength: {strength}")
        print(f"Angle spine: {angle.mean().item():.5f}")
        print(f"Delta px mean: {delta_px.abs().mean().item():.5f}")

        for i, j in enumerate(joints):
            mv = (pts[:, i] - prev_pts[:, i]) * torch.tensor([W_lat, H_lat], device=device)
            print(f"{j}: {mv.squeeze().tolist()}")

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            img = latents_out[0].detach().float().cpu()
            img = img[:3] if img.shape[0] > 3 else img

            torchvision.utils.save_image(
                (img + 1) / 2,
                os.path.join(debug_dir, "global_pose_debug.png")
            )

    # =========================================================
    # 🔹 Return values
    # =========================================================
    return latents_out, delta_px, grid, grid_norm, warp_shift, angle




#------------------------------------------------------------------------------------------
def calibrate_amplitude(mask, base_amp=0.002, max_amp=0.005):
    """
    Calibre automatiquement l'amplitude d'un micro-boost en fonction de la taille du masque.

    Args:
        mask (torch.Tensor): masque binaire [B,H,W] ou [H,W], valeurs 0-1
        base_amp (float): amplitude minimale
        max_amp (float): amplitude maximale

    Returns:
        float: amplitude calibrée
    """
    # Calculer proportion de pixels activés
    mask_area_ratio = mask.mean().item()  # entre 0 et 1

    # Interpolation linéaire
    amplitude = base_amp + (max_amp - base_amp) * mask_area_ratio

    return amplitude



def apply_face_warp(
    latents,
    pose,
    mask_face,
    grid,
    H,
    W,
    frame_counter,
    device=None,
    debug=False,
    debug_dir=None,
    smooth=0.85,
    prev_grid=None,
    strength=0.3,
    paused=False  # Paramètre pour activer/désactiver l'effet
):
    if device is None:
        device = latents.device

    B, C, H_lat, W_lat = latents.shape
    latents_in = latents.clone()

    # =========================
    # Facial points (toujours calculés)
    # =========================
    facial_points = pose.estimate_facial_points_full(smooth=smooth)
    pose.set_prev_facial_points(facial_points)

    # Si la fonction est en pause, on renvoie les valeurs de base sans modifier latents
    if paused:
        return latents, torch.zeros((B, H_lat, W_lat, 2), device=device), facial_points

    # =========================
    # Time
    # =========================
    t = frame_counter / 10.0
    t = torch.tensor(t, device=device)

    # =========================
    # 🔥 BASE MOTION (boosté volontairement)
    # =========================
    face_delta = torch.zeros((B, H_lat, W_lat, 2), device=device)

    # Mouvement global du visage : oscillations sinusoïdales pour un mouvement naturel
    dx = 0.03 * torch.sin(t * 2.0)
    dy = 0.04 * torch.sin(t * 1.7)

    face_delta[..., 0] += strength * dx
    face_delta[..., 1] += strength * dy

    # =========================
    # 🔥 WIND MICRO MOTION (structure corrigée)
    # =========================
    mask = mask_face
    if mask.ndim == 4:
        mask = mask.squeeze(1)

    mask = mask.unsqueeze(-1)  # [B, H, W, 1]

    # Animation du vent sur le visage avec un effet plus réaliste
    wind1 = torch.sin(t * 1.5)
    wind2 = torch.cos(t * 1.1)

    face_delta[..., 0] += strength * mask[..., 0] * 0.02 * wind1
    face_delta[..., 1] += strength * mask[..., 0] * 0.015 * wind2

    if debug:
        print("FACE DELTA MEAN:", face_delta.abs().mean().item())

    # =========================
    # 🔥 FACE CENTER (nose)
    # =========================
    # Le centre du visage est situé au niveau du nez (point 'nose')
    face_center = pose.get_point(0)
    face_center_px = face_center * torch.tensor([W_lat - 1, H_lat - 1], device=device)
    face_center_px = face_center_px.view(B, 1, 1, 2)

    # =========================
    # GRID WARP
    # =========================
    grid_face = grid - face_center_px
    grid_face = grid_face + face_delta
    grid_face = grid_face + face_center_px

    # =========================
    # 🔥 TEMPORAL SMOOTHING
    # =========================
    # Appliquer un lissage temporel pour rendre les transitions plus douces
    if prev_grid is not None:
        alpha = 0.7
        grid_face = alpha * prev_grid + (1.0 - alpha) * grid_face

    # =========================
    # NORMALIZATION GRID_SAMPLE
    # =========================
    grid_face = grid_face.clone()

    grid_face[..., 0] = 2.0 * grid_face[..., 0] / (W_lat - 1) - 1.0
    grid_face[..., 1] = 2.0 * grid_face[..., 1] / (H_lat - 1) - 1.0

    # =========================
    # WARP
    # =========================
    latents_out = F.grid_sample(
        latents,
        grid_face,
        align_corners=True,
        mode="bilinear",
        padding_mode="reflection"
    )

    # =========================
    # Debugging information
    # =========================
    if debug:
        print("[DEBUG] Face warp applied OK")
        print("  grid mean:", grid_face.abs().mean().item())

        # Visualisation du visage pour le débogage
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            img = latents_out[0].detach().float().cpu()
            img = img[:3] if img.shape[0] > 3 else img
            torchvision.utils.save_image(
                (img + 1) / 2,
                os.path.join(debug_dir, f"face_warp_debug_{frame_counter}.png")
            )

    return latents_out, face_delta, facial_points

#--------------------------------------------------------------------




#-------------------------------------------------test -----------------------------------------
def normalize_mask(mask):
    m = mask.mean().clamp(1e-6, 0.2)
    mask = mask / (m + 1e-6)
    mask = torch.clamp(mask, 0.0, 3.0)
    return mask
    """
    Ultra PRO 2.0 motion pipeline:
    - Global Pose + Stabilisation avancée par keypoints
    - Torso Warp + Breathing dynamique
    - Face Warp stateful + temporal smoothing
    - Mouth & Corner micro-expressions
    - Eyes micro-motion
    - Hair Motion Cycle with temporal buffer
    - Micro-boost per zone
    - Full debug and timings
    """
def should_freeze(frame_idx, frame_pause):
    return (frame_pause is not None) and (frame_idx % frame_pause == 0)

def get_breathing_mode(frame_counter, freeze):
    if freeze:
        return "soft", 0.9
    else:
        return "real", 1.2


def time_sin(frame_counter, freq=2.0, device="cuda"):
    t = torch.tensor(frame_counter / 10.0, device=device)
    return torch.sin(t * freq)


def get_time(frame_counter, fps=10.0, device=None):
    return torch.tensor(frame_counter / fps, device=device, dtype=torch.float32)





def apply_pose_world(
    latents_base,
    latents_world,
    mask_torso,
    mask_torso_exp,
    grid,
    timings,
    pose,
    prev_pose,
    state,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None,
    extra_keypoints=None,  # Nouveaux points clés supplémentaires
    angle_smoothing_factor=0.3,  # Facteur pour lisser les rotations
    z_rotation_angle=None,  # Nouvel argument pour l'angle de rotation sur l'axe Z
    y_rotation_angle=None,  # Nouvel argument pour l'angle de rotation sur l'axe Y
    x_rotation_angle=None   # Nouvel argument pour l'angle de rotation sur l'axe X
):
    B, C, H, W = latents_world.shape

    # =========================
    # 🔹 Global pose & stabilisation avancée
    # =========================
    if frame_counter > 1:
        start = time.time()
        latents_world, global_delta, grid_raw, grid_global, warp_shift, global_angle = apply_global_pose(
            latents_world, pose, prev_pose, H, W, device=device, strength=2.0, debug=debug, debug_dir=debug_dir
        )
        print("[DEBUG] WARP Shift: ")
        dx = warp_shift[0, 0, 0, 0].item()
        dy = warp_shift[0, 0, 0, 1].item()
        print(f"[SHIFT FLOAT] dx={dx:.4f}, dy={dy:.4f}")

        # Mise à jour du décalage global
        warp_shift_norm = warp_shift.view(1, 2) / torch.tensor([W, H], device=warp_shift.device, dtype=warp_shift.dtype)
        state["global_shift"] = 0.6 * state["global_shift"] + 0.4 * warp_shift_norm

        # Mise à jour des angles avec lissage temporel
        if "global_angle_x" not in state:
            state["global_angle_x"] = 0.0
        if "global_angle_y" not in state:
            state["global_angle_y"] = 0.0
        if "global_angle_z" not in state:
            state["global_angle_z"] = 0.0

        # Calcule les nouvelles rotations sur chaque axe X, Y, Z
        if prev_pose is not None:
            # Calculate rotation differences between the current pose and the previous pose
            ls_idx = pose.FACIAL_POINT_IDX["left_shoulder"]
            rs_idx = pose.FACIAL_POINT_IDX["right_shoulder"]

            ls = pose.keypoints[:, ls_idx, :2]
            rs = pose.keypoints[:, rs_idx, :2]

            ls_prev = prev_pose.keypoints[:, ls_idx, :2]
            rs_prev = prev_pose.keypoints[:, rs_idx, :2]

            v = rs - ls
            v_prev = rs_prev - ls_prev

            # Calculer l'angle de rotation autour de l'axe Z (plan XY)
            angle_z = torch.atan2(v[:, 1], v[:, 0])  # Calcul de l'angle de rotation sur Z
            angle_z_prev = torch.atan2(v_prev[:, 1], v_prev[:, 0])
            delta_angle_z = angle_z - angle_z_prev
            delta_angle_z = torch.atan2(torch.sin(delta_angle_z), torch.cos(delta_angle_z))  # Wrap the angle
            delta_angle_z = torch.clamp(delta_angle_z, -0.2, 0.2)

            # Calcul des différences d'angles pour Y et X
            # Angle pour Y (pitch) et X (yaw) basé sur les épaules et hanches
            # À ajuster en fonction de votre modèle spécifique
            # Calcul de delta_angle_y (Lacet, rotation autour de Y)
            # Prenons les épaules comme exemple pour déterminer le mouvement du tronc

            # Indices des épaules (ajustez selon votre modèle)
            ls_idx = pose.FACIAL_POINT_IDX["left_shoulder"]
            rs_idx = pose.FACIAL_POINT_IDX["right_shoulder"]

            # Positions des épaules dans le plan 2D (X, Y)
            ls = pose.keypoints[:, ls_idx, :2]
            rs = pose.keypoints[:, rs_idx, :2]

            ls_prev = prev_pose.keypoints[:, ls_idx, :2]
            rs_prev = prev_pose.keypoints[:, rs_idx, :2]

            # Calculer le vecteur directionnel entre les épaules
            v = rs - ls
            v_prev = rs_prev - ls_prev

            # Calculer l'angle de rotation autour de l'axe Y
            angle_y = torch.atan2(v[:, 1], v[:, 0])  # L'angle dans le plan XY
            angle_y_prev = torch.atan2(v_prev[:, 1], v_prev[:, 0])
            delta_angle_y = angle_y - angle_y_prev  # Différence d'angle entre l'angle actuel et le précédent
            delta_angle_y = torch.atan2(torch.sin(delta_angle_y), torch.cos(delta_angle_y))  # "Wrapped" pour éviter des sauts brusques
            delta_angle_y = torch.clamp(delta_angle_y, -0.2, 0.2)  # Limiter l'angle pour éviter des rotations trop larges


            # Calcul de delta_angle_x (Tangage, rotation autour de X)
            # Prenons les hanches et les épaules comme exemple pour déterminer le mouvement du tronc

            # Indices des hanches (ajustez selon votre modèle)
            lh_idx = pose.FACIAL_POINT_IDX["left_hip"]
            rh_idx = pose.FACIAL_POINT_IDX["right_hip"]

            # Positions des hanches dans le plan 2D (X, Y)
            lh = pose.keypoints[:, lh_idx, :2]
            rh = pose.keypoints[:, rh_idx, :2]

            lh_prev = prev_pose.keypoints[:, lh_idx, :2]
            rh_prev = prev_pose.keypoints[:, rh_idx, :2]

            # Calculer le vecteur directionnel entre les hanches
            v_hips = rh - lh
            v_prev = rh_prev - lh_prev

            # Calculer l'angle de rotation autour de l'axe X (tangage)
            angle_x = torch.atan2(v_hips[:, 1], v_hips[:, 0])  # L'angle dans le plan XY (entre les hanches)
            angle_x_prev = torch.atan2(v_prev[:, 1], v_prev[:, 0])
            delta_angle_x = angle_x - angle_x_prev  # Différence d'angle entre l'angle actuel et le précédent
            delta_angle_x = torch.atan2(torch.sin(delta_angle_x), torch.cos(delta_angle_x))  # "Wrapped" pour éviter des sauts brusques
            delta_angle_x = torch.clamp(delta_angle_x, -0.2, 0.2)  # Limiter l'angle pour éviter des rotations trop larges

            # Mises à jour des états des angles
            state["global_angle_x"] = (1 - angle_smoothing_factor) * state["global_angle_x"] + angle_smoothing_factor * delta_angle_x
            state["global_angle_y"] = (1 - angle_smoothing_factor) * state["global_angle_y"] + angle_smoothing_factor * delta_angle_y
            state["global_angle_z"] = (1 - angle_smoothing_factor) * state["global_angle_z"] + angle_smoothing_factor * delta_angle_z

            # Récupérer les valeurs finales des angles
            x_rotation_angle = state["global_angle_x"]
            y_rotation_angle = state["global_angle_y"]
            z_rotation_angle = state["global_angle_z"]


            # ============================
            # 🔹 Rotation sur l'axe Z
            # ============================
            rotation_matrix_z = torch.tensor([
                [torch.cos(z_rotation_angle.detach().clone()), -torch.sin(z_rotation_angle.detach().clone())],
                [torch.sin(z_rotation_angle.detach().clone()), torch.cos(z_rotation_angle.detach().clone())]
            ], device=device)

            # ============================
            # 🔹 Rotation sur l'axe Y (pitch)
            # ============================
            rotation_matrix_y = torch.tensor([
                [torch.cos(y_rotation_angle.detach().clone()), 0, torch.sin(y_rotation_angle.detach().clone())],
                [0, 1, 0],
                [-torch.sin(y_rotation_angle.detach().clone()), 0, torch.cos(y_rotation_angle.detach().clone())]
            ], device=device)

            # ============================
            # 🔹 Rotation sur l'axe X (yaw)
            # ============================
            rotation_matrix_x = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(x_rotation_angle.detach().clone()), -torch.sin(x_rotation_angle.detach().clone())],
                [0, torch.sin(x_rotation_angle.detach().clone()), torch.cos(x_rotation_angle.detach().clone())]
            ], device=device)


            # Appliquer la rotation aux keypoints pour chaque axe
            for i in range(B):
                for j in range(len(pose.keypoints[i])):
                    keypoint = pose.keypoints[i, j, :2]

                    # Appliquer la rotation autour de Z
                    rotated_keypoint_z = torch.matmul(rotation_matrix_z, keypoint)

                    # Appliquer la rotation autour de Y
                    rotated_keypoint_y = torch.matmul(rotation_matrix_y, torch.cat([rotated_keypoint_z, torch.zeros(1, device=device)]).unsqueeze(0).T).squeeze(1)

                    # Appliquer la rotation autour de X
                    # Correction : enlever la concaténation d'un 0 supplémentaire et s'assurer que le vecteur a bien 3 dimensions.
                    rotated_keypoint_x = torch.matmul(rotation_matrix_x, rotated_keypoint_y.unsqueeze(0).T).squeeze(1)

                    # Mise à jour du keypoint
                    pose.keypoints[i, j, :2] = rotated_keypoint_x[:2]  # Mise à jour des coordonnées X, Y après rotation

        timings["GLOBAL"] = time.time() - start

        if debug and frame_counter % 4 == 0:
            print("[DEBUG] GLOBAL WARP REPORT")
            print("  - delta mean px:", global_delta.abs().mean().item())
            print("  - delta max px:", global_delta.abs().max().item())
            save_impact_map(latents_world, latents_base, debug_dir, frame_counter, prefix="torso_global")

    # =========================
    # 🔹 Torso - application du mouvement du torse
    # =========================
    if should_freeze(frame_counter, 1):  # Pause traitement
        t = torch.tensor(frame_counter / 10.0, device=device)
        delta_px = pose.delta.clone()
        delta_px[..., 0] *= W
        delta_px[..., 1] *= H
        delta_px = delta_px.view(B, 1, 1, 2)

        start = time.time()
        latents_before = latents_world.clone()
        latents_torso, delta_px = apply_torso_warp(latents_world, pose, mask_torso, grid, H, W, device=device, debug=debug, debug_dir=debug_dir)
        breath_strength = 0.2 + 0.1 * torch.sin(t)

        latents_world = latents_before * (1.0 - breath_strength * mask_torso_exp) + latents_torso * (breath_strength * mask_torso_exp)
        timings["TORSO+BREATH"] = time.time() - start
        if debug:
            save_impact_map(latents_world, latents_before, debug_dir, frame_counter, prefix="torso_warp")

    # =========================
    # 🔹 BREATHING
    # =========================
    start = time.time()
    freeze = should_freeze(frame_counter, 1)
    mode, mode_strength = get_breathing_mode(frame_counter, freeze)
    latents_before = latents_world
    latents_breath = apply_breathing(latents_world, pose, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir)

    # Temporal modulation
    t = frame_counter / 10.0
    breath_strength = (0.2 + 0.2 * math.sin(t)) * mode_strength
    mask = mask_torso_exp ** 1.5  # Smoothing falloff

    # Residual injection
    latents_world = latents_before + ((latents_breath - latents_before) * mask * breath_strength)
    timings["breathing"] = time.time() - start
    if debug:
        print(f"[DEBUG] Breathing applied ({mode})")

    return latents_world, state, delta_px



def apply_pose_driven_motion_ultra2(
    latents,
    state,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    mouth=True,
    hair=True,
    debug_dir=None
):
    timings = {}
    B, C, H, W = latents.shape
    device = latents.device
    latents = latents.float()
    latents_base = latents.clone()
    latents_world = latents.clone()
    dx, dy = 0, 0  # warp return
    # =========================
    # 🔹 Pose et deltas
    # =========================
    start = time.time()
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W)
    prev_pose = Pose(prev_keypoints.to(device)) if prev_keypoints is not None else None
    timings["Pose"] = time.time() - start



    # =========================
    # 🔹 Animation Upper Body
    # =========================
    try:
        # On récupère les coordonnées brutes depuis keypoints
        upper_body_inputs = {
            "nose": keypoints[:, pose.FACIAL_POINT_IDX["nose"], :2],
            "neck": keypoints[:, pose.FACIAL_POINT_IDX["neck"], :2],
            "right_shoulder": keypoints[:, pose.FACIAL_POINT_IDX["right_shoulder"], :2],
            "right_elbow": keypoints[:, pose.FACIAL_POINT_IDX["right_elbow"], :2],
            "right_wrist": keypoints[:, pose.FACIAL_POINT_IDX["right_wrist"], :2],
            "left_shoulder": keypoints[:, pose.FACIAL_POINT_IDX["left_shoulder"], :2],
            "left_elbow": keypoints[:, pose.FACIAL_POINT_IDX["left_elbow"], :2],
            "left_wrist": keypoints[:, pose.FACIAL_POINT_IDX["left_wrist"], :2],
            "right_clavicle": keypoints[:, pose.FACIAL_POINT_IDX.get("right_clavicle", 19), :2],
            "left_clavicle": keypoints[:, pose.FACIAL_POINT_IDX.get("left_clavicle", 20), :2],
        }

        # Mise à jour via animate_upper_body
        pose_copy = Pose(pose.keypoints.clone())
        updated_upper_body = animate_upper_body(
            pose=pose_copy,
            inputs=upper_body_inputs,
            mode="smooth",
            strength=0.35, debug=debug
        )
        n = min(pose.keypoints.shape[1], updated_upper_body.shape[1])
        pose.keypoints[:, :n] = updated_upper_body[:, :n]

        if debug:
            print(f"[DEBUG] Upper body animated, first shoulder delta:",
                (pose.keypoints[0, pose.FACIAL_POINT_IDX['left_shoulder'], :2] -
                keypoints[0, pose.FACIAL_POINT_IDX['left_shoulder'], :2]))

    except Exception as e:
        print("[WARN] Upper body animation failed:", e)

    # =========================
    # 🔹 Global compensation
    # =========================
    global_shift = torch.zeros((B,1,1,2), device=device)
    if prev_pose is not None:
        c1 = pose.get_center()[..., :2]
        c0 = prev_pose.get_center()[..., :2]
        delta = c1 - c0
        delta = torch.clamp(delta, -5.0, 5.0)
        global_shift = delta.view(B,1,1,2)

    # =========================
    # 🔹 Grid
    # =========================
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B,1,1,1)


    # =========================
    # 🔹 Masks
    # =========================
    mask_face  = torch.clamp(pose.create_face_mask(H,W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter),0,1).float()
    mask_mouth, _ = pose.create_mouth_mask(H,W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    mask_mouth = torch.clamp(mask_mouth,0,1).float()
    mask_mouth_corners, _ = pose.create_mouth_corners_mask(H,W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter)
    mask_mouth_corners = torch.clamp(mask_mouth_corners,0,1).float()
    mask_torso = torch.clamp(pose.create_upper_body_mask(H,W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter),0,1).float()
    mask_hair = torch.clamp(pose.create_hair_mask(H,W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter),0,1).float()
    mask_left_eye = torch.clamp(pose.create_left_eye_mask(H,W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter),0,1).float()
    mask_right_eye = torch.clamp(pose.create_right_eye_mask(H,W, debug=debug, debug_dir=debug_dir, frame_counter=frame_counter),0,1).float()

    mask_torso_exp = mask_torso * (1.0 - mask_face)
    mask_hair_exp = mask_hair * (1.0 - mask_face)
    mask_face_exp = mask_face
    mask_mouth_exp = mask_mouth

    print("mask_hair_exp mean:", mask_hair_exp.mean().item())
    print("mask_face_exp mean:", mask_face_exp.mean().item())
    print("mask_mouth_exp mean:", mask_mouth_exp.mean().item())

    # 🔥 NOUVEAU MASQUE DÉCOR
    mask_decor = pose.create_decor_mask(H, W, mask_face, mask_torso, mask_hair, debug=debug, debug_dir=debug_dir)  # doit devenir
    mask_decor = torch.clamp(mask_decor, 0, 1).float()

    #================== PARTI WORD ========================================
    latents_world, state, delta_px = apply_pose_world(latents_base=latents_base, latents_world=latents_world, mask_torso=mask_torso, mask_torso_exp=mask_torso_exp, grid=grid, timings=timings, pose=pose, prev_pose=prev_pose, state=state, keypoints=keypoints, prev_keypoints=prev_keypoints, frame_counter=frame_counter, device=device, breathing=breathing, debug=debug, debug_dir=debug_dir )

    #================== PARTI LOCAL ========================================

    latents_local = latents_world.clone()
    # =========================
    # 🔹 Face + temporal smoothing
    # =========================
    if not hasattr(apply_pose_driven_motion_ultra2,"prev_face_grid"):
        apply_pose_driven_motion_ultra2.prev_face_grid = [None]*B
    start = time.time()
    paused = (frame_counter % 10 == 0)
    latents_local, face_delta, facial_points = apply_face_warp(
        latents_local, pose, mask_face, grid, H, W, frame_counter,
        device=device, debug=debug, debug_dir=debug_dir, smooth=0.85,
        prev_grid=apply_pose_driven_motion_ultra2.prev_face_grid[0] if B==1 else None, paused=paused
    )
    apply_pose_driven_motion_ultra2.prev_face_grid[0] = grid.clone() if B==1 else None

    face_mix = feather_inside_strict2(mask_face_exp, radius=6, blur_kernel=5, sigma=1.5)

    face_strength = 0.9
    latents_local = (
        latents_world * (1 - face_strength * face_mix) +
        latents_local * (face_strength * face_mix)
    )

    face_strength_mouth = 0.9
    latents_local = (
        latents_world * (1 - face_strength_mouth * mask_mouth_exp) +
        latents_local * (face_strength_mouth * mask_mouth_exp)
    )
    timings["FACE"] = time.time() - start

    # ===================================
    # 🔹 Mouth & micro-expressions - OK
    # ==================================
    #if should_freeze(frame_counter, 2): # Pause traitement
    if mouth:
        start = time.time()
        #latents_local, mouth_delta, _ = apply_mouth_smil( latents_local, pose, mask_mouth, grid, H, W, frame_counter, device=device, debug=debug, debug_dir=debug_dir, smooth=0.85, strength=2.0, npy=False )

        latents_local, mouth_delta, _ = apply_mouth_smil( latents, pose, mask_mouth, grid, frame_counter, mouth_model, H=None, W=None, device=device, debug=debug, debug_dir=debug_dir, smooth=0.85, strength=2.0, npy=False )
        print("[MOUTH DELTA MEAN]:", mouth_delta.abs().mean().item())

        # Broadcasting correct pour la bouche
        mask_mouth_corners_broadcast = mask_mouth_corners.repeat(1, C, 1, 1)

        phase = time_sin(frame_counter, device=latents_local.device)
        latents_local += 0.002 * mask_mouth_corners_broadcast * phase

        # Broadcasting correct pour les yeux
        mask_left_eye_broadcast  = mask_left_eye.repeat(1, C, 1, 1)
        mask_right_eye_broadcast = mask_right_eye.repeat(1, C, 1, 1)


        t = time_sin(frame_counter, freq=3.0, device=latents_world.device)
        eye_motion = 0.1 * (mask_left_eye_broadcast * t +
                        mask_right_eye_broadcast * time_sin(frame_counter, freq=3.0, device=latents_world.device))

        eye_motion = eye_motion * mask_face_exp
        latents_local += eye_motion

        timings["MOUTH+EYES"] = time.time() - start

    # ==============================
    # 🔹 Hair motion cycle - OK
    # ==============================
    #if should_freeze(frame_counter, 2): # Pause traitement
    #if frame_counter > 10:
    if hair:
        if not hasattr(apply_pose_driven_motion_ultra2,"prev_hair_fields"):
            apply_pose_driven_motion_ultra2.prev_hair_fields = [None]*B
        start = time.time()
        latents_before = latents_local.clone()
        latents_hair, hair_delta = apply_hair_motion_cycle(
            latents_local, mask_hair, grid, H, W, frame_counter, device, delta_px,
            prev_hair_field=apply_pose_driven_motion_ultra2.prev_hair_fields[0] if B==1 else None, target="hair",
            debug=debug, debug_dir=debug_dir
        )
        latents_local = latents_hair * mask_hair_exp + latents_before * (1.0 - mask_hair_exp)
        print("HAIR DELTA MEAN:", hair_delta.abs().mean().item())
        apply_pose_driven_motion_ultra2.prev_hair_fields[0] = hair_delta
        timings["HAIR"] = time.time() - start


    # ===========================
    # 🔹 Decor motion cycle - OK
    # ===========================
    #if should_freeze(frame_counter, 10): # Pause traitement
    if frame_counter > 10:
        if not hasattr(apply_pose_driven_motion_ultra2,"prev_decor_fields"):
            apply_pose_driven_motion_ultra2.prev_decor_fields = [None]*B
        start = time.time()
        latents_before = latents_local.clone()

        latents_decor, decor_delta = apply_hair_motion_cycle(
            latents_local, mask_decor, grid, H, W, frame_counter, device, delta_px,
            prev_hair_field=apply_pose_driven_motion_ultra2.prev_decor_fields[0] if B==1 else None, target="decor",
            debug=debug, debug_dir=debug_dir
        )
        latents_local = latents_decor * mask_decor + latents_before * (1.0 - mask_decor)
        print("DECOR DELTA MEAN:", decor_delta.abs().mean().item())
        apply_pose_driven_motion_ultra2.prev_decor_fields[0] = decor_delta
        timings["DECOR"] = time.time() - start

    # =========================
    # 🔹 Micro boost global
    # =========================

    MICRO_GAIN = 2.0   # contrôle global unique

    masks = {
        "torso": (mask_torso_exp, 0.05, calibrate_amplitude(mask_torso_exp, 0.002, 0.006)),
        "hair": (mask_hair_exp, 0.20, calibrate_amplitude(mask_hair_exp, 0.002, 0.006)),
        "face": (mask_face_exp, 0.15, calibrate_amplitude(mask_face_exp, 0.002, 0.006)),
        "left_eye": (mask_left_eye, 0.6, calibrate_amplitude(mask_left_eye, 0.0008, 0.0025)),
        "right_eye": (mask_right_eye, 0.6, calibrate_amplitude(mask_right_eye, 0.0008, 0.0025)),
        "mouth": (mask_mouth_exp, 0.25, calibrate_amplitude(mask_mouth_exp, 0.01, 0.08)),
        "mouth_corners": (mask_mouth_corners, 0.2, calibrate_amplitude(mask_mouth_corners, 0.005, 0.03)),
        "decor": (mask_decor, 0.02, calibrate_amplitude(mask_decor, 0.0005, 0.0015))
    }

    start = time.time()

    # =========================
    # PREPROCESS MASKS (clean)
    # =========================
    for k, (mask, phase, amp) in masks.items():
        if mask is None:
            continue

        mask = normalize_mask(mask)

        # stabilisation douce (OK)
        mask = torch.sqrt(mask.clamp(0, 1))

        masks[k] = (mask, phase, amp)

    # =========================
    # BASE LATENTS MIX
    # =========================
    latents_mix = 0.7 * latents_local + 0.3 * latents_world

    # =========================
    # MICRO BOOST CORE (1 seule source)
    # =========================
    latents = apply_micro_boost( latents_mix, frame_counter, device, masks, keypoints, prev_keypoints, strength=1.0, debug=debug )

    # =========================
    # SECONDARY SINUS BOOST (optionnel mais propre)
    # =========================
    for key, (mask, speed, amplitude) in masks.items():
        if mask is None:
            continue

        # sécurité dimension
        if mask.ndim == 5:
            mask = mask.squeeze(2)

        mask_exp = mask.repeat(1, C, 1, 1)

        # UNIQUE scaling propre
        t = get_time(frame_counter, device=latents.device)
        signal = torch.sin(t * speed) * torch.exp(-0.1 * t)

        latents = latents + MICRO_GAIN * amplitude * mask_exp * signal

    # =========================
    # MICRO MOTION FINAL (very light)
    # =========================
    latents = apply_micro_motion( latents, frame_counter, device, masks, strength=0.05, randomize=True, debug=debug )

    timings["MICRO_BOOST"] = time.time() - start

    # =========================
    # 🔹 DECOR MASK (post-process)
    # =========================
    decor_strength = 0.25  # 🔥 réglable
    decor_mix = 0.2  # conserve un peu du mouvement
    decor_mask_soft = mask_decor * 0.8  # 🔥 réduit impact

    latents = latents * (1.0 - decor_strength * decor_mask_soft) + \
            (latents_world * (1.0 - decor_mix) + latents * decor_mix) * (decor_strength * decor_mask_soft)

    # =========================
    # 🔹 DEBUG FINAL
    # =========================
    if debug:
        save_impact_map(latents, latents_base, debug_dir, frame_counter, prefix="final")
        print("[DEBUG] Ultra2 Full motion pipeline applied")
        print("[DEBUG] Timings per step:", timings)

    return latents, state

    """
    Ultra PRO motion pipeline:
    - Global Pose + Stabilisation
    - Torso Warp + Breathing
    - Face Warp (stateful)
    - Mouth & Corners Warp
    - Hair Motion Cycle
    - Eyes micro-motion
    - Micro-boost per zone
    - Full debug and timing outputs
    """


def apply_pose_driven_motion_stable(
    latents,
    keypoints,
    prev_keypoints=None,
    frame_counter=0,
    device="cuda",
    breathing=True,
    debug=False,
    debug_dir=None
):
    """
    Pipeline motion PRO (stable + vivant + isolé) :
    - Global Pose
    - Torso Warp
    - Face Warp (stateful)
    - Hair Motion (alt normal/extreme)
    - Breathing (torso only)
    - Stabilisation (face protected)
    - Micro-boost par zone pour éviter le rendu statique
    """
    # dictionnaire pour stocker les temps
    timings = {}
    # =========================
    # 🔹 SETUP
    # =========================
    B, C, H, W = latents.shape
    device = latents.device
    latents = latents.float()
    latents_in = latents.clone()

    # =========================
    # 🔹 POSE (NOW CONSISTENT)
    # =========================
    start = time.time()
    pose = Pose(keypoints.to(device))
    pose.compute_torso_delta(latent_h=H, latent_w=W)

    prev_pose = Pose(prev_keypoints.to(device)) if prev_keypoints is not None else None
    timings["Pose"] = time.time() - start

    # =========================
    # 🔥 GLOBAL COMPENSATION
    # =========================


    global_shift = None

    if prev_pose is not None:
        # centers en pixels directement (IMPORTANT)
        c1 = pose.get_center()
        c0 = prev_pose.get_center()

        c1 = pose.get_center()[..., :2]
        c0 = prev_pose.get_center()[..., :2]

        global_shift = (c1 - c0).to(device)
        global_shift = global_shift.view(B, 1, 1, 2)

    # =========================
    # 🔹 Grid
    # =========================
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0).repeat(B, 1, 1, 1)

    # =========================
    # 🔹 Masks (CLAMP SAFE)
    # =========================
    mask_face  = torch.clamp(pose.create_face_mask(H, W, debug=debug, debug_dir=debug_dir), 0, 1).float()
    mask_mouth, mouth_points = pose.create_mouth_mask(H, W, debug=debug, debug_dir=debug_dir)
    mask_mouth = torch.clamp(mask_mouth, 0, 1).float()
    mask_mouth_corners, corners_points_batch = pose.create_mouth_corners_mask(H, W, debug=debug, debug_dir=debug_dir)
    mask_mouth_corners = torch.clamp(mask_mouth_corners, 0, 1).float()

    mask_torso = torch.clamp(pose.create_upper_body_mask(H, W, debug=debug, debug_dir=debug_dir), 0, 1).float()
    mask_hair  = torch.clamp(pose.create_hair_mask(H, W, debug=debug, debug_dir=debug_dir), 0, 1).float()
    mask_left_eye = pose.create_left_eye_mask(H, W, debug=debug, debug_dir=debug_dir)
    mask_left_eye = torch.clamp(mask_left_eye, 0, 1).float()
    mask_right_eye = pose.create_right_eye_mask(H, W, debug=debug, debug_dir=debug_dir)
    mask_right_eye = torch.clamp(mask_right_eye, 0, 1).float()


    mask_face_exp  = mask_face
    mask_mouth_exp  = mask_mouth
    mask_torso_exp = mask_torso * (1.0 - mask_face_exp)
    mask_hair_exp  = mask_hair  * (1.0 - mask_face_exp)
    mask_right_eye_exp = mask_right_eye
    mask_left_eye_exp = mask_left_eye

    # =========================
    # 🔹 GLOBAL POSE
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_global, global_delta = apply_global_pose(
        latents=latents, pose=pose, prev_pose=prev_pose, H=H, W=W, device=device,
        debug=debug,
        debug_dir=debug_dir
    )
    if debug:
        print("GLOBAL delta mean:", global_delta.abs().mean())



    latents = latents_global * (1.0 - mask_face_exp) + latents_before * mask_face_exp
    timings["GLOBAL"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="torso_global")
        print("[DEBUG] Global pose applied")

    # =========================
    # 🔹 TORSO
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_torso, delta_px = apply_torso_warp(
        latents=latents, pose=pose, mask_torso=mask_torso, grid=grid, H=H, W=W, device=device,
        debug=debug,
        debug_dir=debug_dir
    )
    latents = latents_torso * mask_torso_exp + latents_before * (1.0 - mask_torso_exp)
    timings["TORSO"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="torso_warp")
        print("[DEBUG] Torso warp applied")

    # ========================
    # FIX FACE + BOUCHE
    # ========================
    grid_base = grid

    global_shift = global_shift if global_shift is not None else torch.zeros((B,1,1,2), device=device)
    grid_base = grid_base + global_shift
    # =========================
    # 🔹 FACE (STATEFUL)
    # =========================
    face_grid = grid_base
    start = time.time()
    latents, face_delta, facial_points = apply_face_warp(
        latents=latents, pose=pose, mask_face=mask_face, grid=face_grid, H=H, W=W, frame_counter=frame_counter, device=device, debug=debug, debug_dir=debug_dir,
        smooth=0.85
    )
    timings["face_warp"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="face_warp")
        print("[DEBUG] Face warp applied")
    # =========================
    # 🔹 BOUCHE (STATEFUL)
    # =========================
    mouth_grid = grid_base
    start = time.time()
    latents, mouth_delta, _ = apply_mouth_smil(
        latents=latents, pose=pose, mask_mouth=mask_mouth, grid=mouth_grid, H=H, W=W, frame_counter=frame_counter, device=device, debug=debug, debug_dir=debug_dir,
        smooth=0.85
    )
    timings["mouth_warp"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="mouth_warp")
        print("[DEBUG] Mouth warp applied")


    # =========================
    # 🔹 HAIR (ALTERNANCE CINÉMA)
    # =========================
    # Définir un dictionnaire global ou un buffer par batch
    if not hasattr(apply_pose_driven_motion, "prev_hair_fields"):
        apply_pose_driven_motion.prev_hair_fields = [None] * B

    # =========================
    # 🔹 HAIR (ALTERNANCE CINÉMA)
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_hair, hair_delta = apply_hair_motion_cycle(
        latents=latents, mask_hair=mask_hair, grid=grid, H=H, W=W, frame_counter=frame_counter, device=device, delta_px=delta_px,
        prev_hair_field=apply_pose_driven_motion.prev_hair_fields[0] if B==1 else None,
        debug=debug,
        debug_dir=debug_dir
    )
    latents = latents_hair * mask_hair_exp + latents_before * (1.0 - mask_hair_exp)
    # Stocker pour la prochaine frame
    apply_pose_driven_motion.prev_hair_fields[0] = hair_delta
    timings["HAIR"] = time.time() - start
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="hair")
        print("[DEBUG] Hair motion applied")

    # =========================
    # 🔹 BREATHING (TORSO ONLY)
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_breath = apply_breathing_real( latents, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir )
    t = torch.tensor(frame_counter / 10.0, device=latents.device)
    breath_strength = 0.2 + 0.1 * torch.sin(t)
    latents = (
        latents_before * (1.0 - breath_strength * mask_torso_exp)
        + latents_breath * (breath_strength * mask_torso_exp)
    )
    timings["breathing"] = time.time() - start
    if debug:
        print("[DEBUG] Breathing applied")

    # =========================
    # 🔹 STABILISATION
    # =========================
    start = time.time()
    latents_before = latents.clone()
    latents_stab = stabilize_latents_motion(latents)
    latents = latents_stab * (1.0 - mask_face_exp) + latents_before * mask_face_exp
    timings["stabilisation"] = time.time() - start

    if debug:
        print("[DEBUG] Stabilization applied")

    # =========================
    # 🔹 MICRO BOOST GLOBAL PAR ZONE
    # =========================
    masks = {
        "torso": (mask_torso_exp, 0.1, calibrate_amplitude(mask_torso_exp, base_amp=0.002, max_amp=0.004)),
        "hair":  (mask_hair_exp,  0.2, calibrate_amplitude(mask_hair_exp, base_amp=0.003, max_amp=0.0035)),
        "face":  (mask_face_exp,  0.3, calibrate_amplitude(mask_face_exp, base_amp=0.002, max_amp=0.006)),
        "mouth":  (mask_mouth_exp,  0.3, calibrate_amplitude(mask_mouth_exp, base_amp=0.003, max_amp=0.008)),
        # Yeux (clignements / micro-mouvements)
        "left_eye":  (mask_left_eye,  0.5, calibrate_amplitude(mask_left_eye, 0.0015, 0.004)),
        "right_eye": (mask_right_eye, 0.6, calibrate_amplitude(mask_right_eye, 0.0015, 0.004)),
        # Coins de bouche pour sourire subtil
        "mouth_corners": (mask_mouth_corners, 0.3, calibrate_amplitude(mask_mouth_corners, 0.002, 0.006)),
    }

    start = time.time()
    latents = apply_micro_boost(latents, frame_counter, device, masks, keypoints, prev_keypoints)
    latents = apply_micro_motion(latents, frame_counter, device, masks, randomize = True)

    timings["micro_boost"] = time.time() - start

    # =========================
    # 🔹 DEBUG FINAL
    # =========================
    if debug:
        save_impact_map(latents, latents_in, debug_dir, frame_counter, prefix="final")
        print("[DEBUG] Full motion pipeline applied")
        print("[DEBUG] Timings per step:", timings)

    return latents

#-------Gestion de l'animation -----------------------------------------------------------------------------------
def to_float(x):
    if torch.is_tensor(x):
        return x.item()
    return float(x)

def compute_time(frame_idx, frame_pause, base_dt=0.03):
    if frame_pause is None:
        return frame_idx * base_dt

    frozen_blocks = frame_idx // frame_pause
    intra = frame_idx % frame_pause

    # ralentissement progressif dans le bloc
    decay = intra / frame_pause
    return (frozen_blocks + decay * 0.3) * base_dt
# --- En DEV: synchroniser ce freeze intelligent avec ton apply_global_pose pour éviter les conflits entre keypoints et warp global.


# =========================================================
# MAIN ENTRY
# =========================================================

def ensure_kp3(kp):
    # kp: (B,N,2) or (B,N,3)
    if kp.shape[-1] == 2:
        conf = torch.ones_like(kp[..., :1])
        kp = torch.cat([kp, conf], dim=-1)
    return kp



def safe_kp(kp):
    """Sécurise les keypoints en gérant les NaN et en forçant une forme (B, N, 2)."""
    if kp is None:
        return None

    if not torch.is_tensor(kp):
        return None

    kp = kp.clone()
    kp = torch.nan_to_num(kp, nan=0.0)

    # force (B,N,2)
    if kp.dim() == 3:
        kp = kp[..., :2]

    return kp

def apply_rotation(kp, center, angle_x, angle_y, angle_z, device):
    """Applique une rotation 3D autour des axes X, Y et Z."""
    cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
    cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
    cos_z, sin_z = math.cos(angle_z), math.sin(angle_z)

    rot_x = torch.tensor(
        [[1, 0, 0],
         [0, cos_x, -sin_x],
         [0, sin_x, cos_x]],
        device=device, dtype=kp.dtype
    )
    rot_y = torch.tensor(
        [[cos_y, 0, sin_y],
         [0, 1, 0],
         [-sin_y, 0, cos_y]],
        device=device, dtype=kp.dtype
    )
    rot_z = torch.tensor(
        [[cos_z, -sin_z, 0],
         [sin_z, cos_z, 0],
         [0, 0, 1]],
        device=device, dtype=kp.dtype
    )
    rotation_matrix = torch.matmul(rot_z, torch.matmul(rot_y, rot_x))  # Rotation combinée (Z -> Y -> X)

    # Appliquer la rotation sur les coordonnées
    xy = kp[..., :2] - center
    xy_3d = torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)  # Passer de 2D à 3D (ajouter un zéro pour Z)
    kp[..., :2] = torch.einsum('bnc,cd->bnd', xy_3d, rotation_matrix[..., :2]) + center
    return kp
# ============= FONCTION PRINCIPAL ============================

def update_sequence_from_keypoints_batch(
    sequence,
    frame_idx,
    prev_keypoints=None,
    state=None,
    profile=None,
    time_scale=0.1,  # 0.2
    max_velocity=0.05,
    camera_lock=0.9,
    debug=False,
    debug_dir=None,
    image_size=(1280, 896)
):
    # =========================================================
    # 0. STATE SAFE INIT
    # =========================================================
    if state is None:
        state = {}

    # Vérification explicite de la présence de prev_keypoints et de sequence
    ref_kp = safe_kp(prev_keypoints) if prev_keypoints is not None else safe_kp(sequence[0]) if len(sequence) > 0 else None

    if ref_kp is not None:
        B, N, _ = ref_kp.shape
        state.setdefault("kp_prev", ref_kp.clone())
        state.setdefault("velocity", torch.zeros((B, N, 2), device=ref_kp.device))
        state.setdefault("angular_vel", torch.zeros((B, N, 2), device=ref_kp.device))
        state.setdefault("anchor", ref_kp[..., :2].mean(dim=1, keepdim=True))
    else:
        state.setdefault("kp_prev", None)
        state.setdefault("velocity", None)
        state.setdefault("angular_vel", None)
        state.setdefault("anchor", torch.zeros((1, 1, 2)))

    state.setdefault("angle", 0.0)
    state.setdefault("rotation", 0.0)
    state.setdefault("initialized", True)
    state.setdefault("global_angle_x", 0.0)
    state.setdefault("global_angle_y", 0.0)
    state.setdefault("global_angle_z", 0.0)


    # =========================================================
    # 1. PROFILE SAFE
    # =========================================================
    motion_model = profile or resolve_motion_model(frame_idx)
    p = MOTION_PROFILES.get(motion_model, {})

    time_scale = p.get("time_scale", time_scale)
    camera_lock = p.get("camera_lock", camera_lock)
    max_velocity = p.get("max_velocity", max_velocity)
    rotation_gain = p.get("rotation_gain", 1.0)

    # =========================================================
    # 2. TIME WARP
    # =========================================================
    scaled = frame_idx * time_scale
    i0 = max(0, min(int(scaled), len(sequence) - 1))
    i1 = min(i0 + 1, len(sequence) - 1)
    t = scaled - i0
    kp = (1 - t) * sequence[i0] + t * sequence[i1]
    kp = torch.nan_to_num(kp, nan=0.0)

    if kp.dim() == 3:
        kp = kp[..., :2]

    B, N, _ = kp.shape

    # =========================================================
    # GLOBAL SHIFT (ROBUST - BODY BASED)
    # =========================================================
    shift = torch.zeros((1, 2), device=kp.device)
    kp_prev = state.get("kp_prev", None)

    if kp_prev is not None and torch.is_tensor(kp_prev):
        body_ids = [2, 5, 8, 11]  # shoulders + hips
        body_ids = [i for i in body_ids if i < kp.shape[1]]
        if len(body_ids) > 0:
            kp_body = kp[:, body_ids, :2]
            kp_prev_body = kp_prev[:, body_ids, :2]
            center_now = kp_body.mean(dim=1)
            center_prev = kp_prev_body.mean(dim=1)
            shift_raw = center_now - center_prev
            shift_raw = shift_raw.mean(dim=0, keepdim=True)
            if "global_shift" not in state or not torch.is_tensor(state["global_shift"]):
                state["global_shift"] = torch.zeros_like(shift_raw)
            shift = state["global_shift"] * 0.7 + shift_raw * 0.3
            shift = torch.clamp(shift, -0.2, 0.2)
            state["global_shift"] = shift

    # =========================================================
    # DRIFT SAFE INIT (CLEAN)
    # =========================================================
    if "drift" not in state or not torch.is_tensor(state["drift"]):
        state["drift"] = torch.zeros((1,1,2), device=kp.device)
    # =========================================================
    # DRIFT SAFE (VECTOR EMA STABLE)
    # =========================================================
    drift = torch.zeros((1, 1, 2), device=kp.device)
    if kp_prev is not None and torch.is_tensor(kp_prev):
        kp_prev_xy = kp_prev[..., :2]
        kp_xy = kp[:, :min(kp_prev_xy.shape[1], kp.shape[1]), :]
        center_now = kp_xy.mean(dim=1, keepdim=True)
        center_prev = kp_prev_xy.mean(dim=1, keepdim=True)
        drift_raw = center_now - center_prev
        drift = state["drift"] * 0.9 + drift_raw * 0.1
        drift = torch.clamp(drift, -0.05, 0.05)
        state["drift"] = drift

    # =========================================================
    # CAMERA STABILIZATION
    # =========================================================
    anchor = state.get("anchor", None)
    if anchor is not None and torch.is_tensor(anchor):
        if anchor.dim() == 2:
            anchor = anchor.unsqueeze(1)
        anchor_xy = anchor[..., :2]
        kp[..., :2] = kp[..., :2] * camera_lock + anchor_xy * (1 - camera_lock)

    # =========================================================
    # 6. ACTOR MODEL SAFE CALL
    # =========================================================
    try:
        kp, new_state = apply_actor_model(
            kp, state, frame_idx=frame_idx, profile=motion_model
        )
    except Exception as e:
        print(f"[⚠ ACTOR FALLBACK] {e}")
        new_state = state

    # =========================================================
    # ROTATION SAFE (X, Y, Z)
    # =========================================================
    if kp_prev is not None:
        center = kp[..., :2].mean(dim=1, keepdim=True)
        angle_global_x = state.get("global_angle_x", 0.0)
        angle_global_y = state.get("global_angle_y", 0.0)
        angle_global_z = state.get("global_angle_z", 0.0)
        angle_fake = 0.02 * math.sin(frame_idx * 0.05)

        angle_x = 0.5 * angle_global_x + 0.5 * angle_fake
        angle_y = 0.5 * angle_global_y + 0.5 * angle_fake
        angle_z = 0.5 * angle_global_z + 0.5 * angle_fake

        kp = apply_rotation(kp, center, angle_x, angle_y, angle_z, kp.device)

    # =========================================================
    # PHYSICS LIMIT
    # =========================================================
    if kp_prev is not None:
        delta = kp[..., :2] - kp_prev[..., :2]
        speed = torch.norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(max_velocity / torch.where(speed < 1e-6, torch.ones_like(speed), speed), 0.1, 1.0)
        kp[..., :2] = kp_prev[..., :2] + delta * scale

    # =========================================================
    # POST ROTATION GAIN
    # =========================================================
    if kp_prev is not None:
        upper_ids = [2, 3, 4, 5, 6, 7, 8, 11]
        upper_ids = [i for i in upper_ids if i < kp.shape[1]]
        rotation_gain = min(rotation_gain, 0.7)
        kp[:, upper_ids, :2] = (
            kp_prev[:, upper_ids, :2] +
            (kp[:, upper_ids, :2] - kp_prev[:, upper_ids, :2]) * rotation_gain
        )

    # =========================================================
    # DEBUG FINAL
    # =========================================================
    if debug:
        motion = (kp[..., :2] - kp_prev[..., :2]).abs().mean() if kp_prev is not None else 0.0
        print("\n[🎬 MOTION ENGINE V6/V7/V8/V9]")
        print(f"frame: {frame_idx}, Motion model: {motion_model}, motion_mean: {motion:.6f}")

        if frame_idx % 2 == 0 and debug_dir:
            debug_draw_openpose_skeleton(keypoints_tensor=ensure_kp3(kp), debug_dir=debug_dir, frame_counter=frame_idx, image_size=image_size)

    # =========================================================
    # STATE UPDATE SAFE
    # =========================================================
    new_state = new_state or {}
    new_state["kp_prev"] = kp.clone()
    new_state["anchor"] = new_state.get("anchor", kp.mean(dim=1, keepdim=True))
    new_state["global_shift"] = state.get("global_shift", None)
    new_state["global_angle_x"] = state.get("global_angle_x", None)
    new_state["global_angle_y"] = state.get("global_angle_y", None)
    new_state["global_angle_z"] = state.get("global_angle_z", None)

    if debug:
        print(f"[DEBUG][STATE UPDATE SAFE] global_shift: {state.get('global_shift', None)}")
        print(f"[DEBUG][STATE UPDATE SAFE] global_angle_x: {state.get('global_angle_x', None)}")
        print(f"[DEBUG][STATE UPDATE SAFE] global_angle_y: {state.get('global_angle_y', None)}")
        print(f"[DEBUG][STATE UPDATE SAFE] global_angle_z: {state.get('global_angle_z', None)}")


    return kp, new_state




def update_sequence_from_keypoints_batch_stable(
    sequence,
    frame_idx,
    prev_keypoints=None,
    alpha_base=0.90,
    freeze_threshold=0.0025,
    freeze_strength=0.15,
    micro_jitter=0.00025,
    time_scale=1.00,          # 🔥 contrôle vitesse globale
    max_velocity=0.009,       # 🔥 clamp px/frame (IMPORTANT)
    debug=False,
    debug_dir=None,
    image_size=(1280, 896)
):

    # =========================================================
    # 1. TIME WARP (interpolation propre)
    # =========================================================
    scaled_idx = frame_idx * time_scale
    i0 = int(scaled_idx)
    i1 = min(i0 + 1, len(sequence) - 1)
    t = scaled_idx - i0

    kp_raw = (1 - t) * sequence[i0] + t * sequence[i1]
    B, N, _ = kp_raw.shape

    # =========================================================
    # 2. MOTION ENERGY (RAW ONLY)
    # =========================================================
    if prev_keypoints is not None:
        motion_energy = (kp_raw - prev_keypoints).abs().mean()
    else:
        motion_energy = torch.tensor(1.0, device=kp_raw.device)

    # =========================================================
    # 3. FREEZE GATE
    # =========================================================
    freeze_gate = torch.ones_like(motion_energy)

    if motion_energy < freeze_threshold:
        freeze_gate = motion_energy / freeze_threshold
        freeze_gate = freeze_strength + (1 - freeze_strength) * freeze_gate

    freeze_gate = torch.clamp(freeze_gate, 0.0, 1.0)

    # =========================================================
    # 4. ADAPTIVE SMOOTHING (IMPORTANT)
    # =========================================================
    alpha = alpha_base + (1.0 - alpha_base) * freeze_gate

    kp = kp_raw
    if prev_keypoints is not None:
        kp = alpha * kp + (1 - alpha) * prev_keypoints

    # =========================================================
    # 5. VELOCITY CLAMP (CRITICAL FIX)
    # =========================================================
    if prev_keypoints is not None:
        delta = kp[..., :2] - prev_keypoints[..., :2]

        speed = torch.norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(max_velocity / (speed + 1e-6), max=1.0)

        kp[..., :2] = prev_keypoints[..., :2] + delta * scale

    # =========================================================
    # 8. KINETIC CHAIN PROPAGATION (PRO+)
    # =========================================================
    if prev_keypoints is not None:

        def propagate(parent, child, stiffness=0.15, delay=0.6):
            parent_vel = kp[:, parent, :2] - prev_keypoints[:, parent, :2]
            child_vel = kp[:, child, :2] - prev_keypoints[:, child, :2]

            # propagation retardée
            propagated = parent_vel * delay

            # mélange physique
            new_child_vel = (
                child_vel * (1 - stiffness) +
                propagated * stiffness
            )

            kp[:, child, :2] = prev_keypoints[:, child, :2] + new_child_vel

        # 🔹 chaîne bras
        propagate(2, 3, stiffness=0.25, delay=0.7)  # shoulder → elbow
        propagate(3, 4, stiffness=0.35, delay=0.8)  # elbow → wrist

        propagate(5, 6, stiffness=0.25, delay=0.7)
        propagate(6, 7, stiffness=0.35, delay=0.8)

        # 🔹 chaîne corps
        propagate(1, 8, stiffness=0.2, delay=0.6)   # neck → hip
        propagate(1, 11, stiffness=0.2, delay=0.6)

    # =========================================================
    # 6. MICRO MOTION (ONLY WHEN FROZEN)
    # =========================================================
    if freeze_gate < 0.6:
        jitter = torch.randn_like(kp[..., :2]) * micro_jitter
        kp[..., :2] += jitter * (1.0 - freeze_gate)

    # =========================================================
    # 7. FINAL DAMPING (VERY SOFT)
    # =========================================================
    if prev_keypoints is not None:
        delta = kp[..., :2] - prev_keypoints[..., :2]
        kp[..., :2] -= delta * (1 - freeze_gate) * 0.25

    # =========================================================
    # 8. CLAMP SAFE SPACE
    # =========================================================
    kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)

    # =========================================================
    # 9. DEBUG
    # =========================================================
    if debug:
        print("\n[DEBUG][SEQ PRO]")
        print(f"frame: {frame_idx}")
        print(f"motion_energy: {motion_energy.item():.6f}")
        print(f"freeze_gate: {freeze_gate.item():.4f}")
        print(f"alpha: {alpha:.4f}")

        if debug_dir is not None:
            debug_draw_openpose_skeleton(
                keypoints_tensor=kp,
                debug_dir=debug_dir,
                frame_counter=frame_idx,
                image_size=image_size
            )

    return kp



def update_pose_from_keypoints_batch(
    keypoints_tensor,
    state=None,
    frame_idx=0,
    smooth=0.8,
    motion_scale=0.01,
    debug=False
):
    kp = keypoints_tensor.clone()
    B, N, _ = kp.shape

    # =========================================================
    # INIT STATE
    # =========================================================
    if state is None:
        state = {
            "kp_prev": kp.clone(),
            "anchor": kp.mean(dim=1)
        }

    prev_kp = state["kp_prev"]
    anchor  = state["anchor"]

    # =========================================================
    # 1. TEMPORAL SMOOTHING (inertia)
    # =========================================================
    kp = smooth * kp + (1 - smooth) * prev_kp

    # =========================================================
    # 2. GLOBAL STABILIZATION (anti drift)
    # =========================================================
    kp[..., :2] = 0.7 * kp[..., :2] + 0.3 * anchor.unsqueeze(1)

    # =========================================================
    # 3. LIGHT NATURAL MOTION
    # =========================================================
    t = frame_idx * 0.05

    offset_x = motion_scale * torch.sin(torch.tensor(t, device=kp.device))
    offset_y = motion_scale * torch.cos(torch.tensor(t * 0.7, device=kp.device))

    kp[..., 0] += offset_x
    kp[..., 1] += offset_y

    # =========================================================
    # 4. MICRO JITTER (very subtle)
    # =========================================================
    kp[..., :2] += 0.001 * torch.randn_like(kp[..., :2])

    # =========================================================
    # 5. CLAMP (safety)
    # =========================================================
    kp[..., :2] = torch.clamp(kp[..., :2], -1.2, 1.2)

    # =========================================================
    # UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp.clone()
    state["anchor"]  = kp.mean(dim=1)

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        motion = (kp - prev_kp).abs().mean()
        print("\n[DEBUG SIMPLE]")
        print(f"motion: {motion.item():.6f}")
        print(f"anchor: {state['anchor'].mean().item():.6f}")

    return kp, state

def update_pose_sequence_from_keypoints_batch_stable(
    keypoints_tensor,
    prev_keypoints=None,
    frame_idx=0,
    alpha=0.9,
    add_motion=True,
    motion_scale=0.4,   # 🔥 NOUVEAU: contrôle global vitesse
    debug=False
):

    kp = keypoints_tensor.clone()
    B, N, _ = kp.shape

    # =========================================================
    # 🔹 1. TEMPORAL SMOOTHING (plus fort)
    # =========================================================
    if prev_keypoints is not None:
        kp = alpha * kp + (1 - alpha) * prev_keypoints

    if not add_motion:
        return kp

    t = frame_idx * 0.03   # 🔥 RALENTI x3

    # =========================================================
    # 🔹 2. GROUPS
    # =========================================================
    pelvis_ids   = [11, 8]
    spine_ids    = [1, 2, 5, 8]
    shoulder_ids = [2, 5]
    head_id      = 0
    limb_ids     = list(range(min(N, 25)))

    # =========================================================
    # 🔹 3. GLOBAL MOTION (VERY SLOW drift)
    # =========================================================
    slow_sway = 0.003 * math.sin(t * 0.6)
    drift_x   = 0.001 * math.sin(t * 0.15)
    drift_y   = 0.001 * math.cos(t * 0.13)

    global_offset = torch.zeros_like(kp[:, :, :2])
    global_offset[..., 0] += slow_sway + drift_x
    global_offset[..., 1] += drift_y

    kp[..., :2] += global_offset * motion_scale

    # =========================================================
    # 🔹 4. BREATHING (ultra slow)
    # =========================================================
    breath = 0.004 * math.sin(t * 0.7)  # 🔥 beaucoup plus lent

    kp[:, spine_ids, 1] += breath
    kp[:, shoulder_ids, 1] += breath * 0.6

    # =========================================================
    # 🔹 5. SPINE WAVE (low frequency)
    # =========================================================
    spine_wave = 0.002 * math.sin(t * 0.8)

    kp[:, spine_ids, 1] += spine_wave
    kp[:, spine_ids, 0] += spine_wave * 0.2

    # =========================================================
    # 🔹 6. HEAD MOTION (very slow inertia)
    # =========================================================
    head_x = 0.0025 * math.sin(t * 0.9)
    head_y = 0.0020 * math.cos(t * 0.85)

    kp[:, head_id, 0] += head_x
    kp[:, head_id, 1] += head_y

    # inertia head (très réduit)
    kp[:, head_id] += (kp[:, 2] - kp[:, 5]) * 0.03

    # =========================================================
    # 🔹 7. LIMBS (quasi static)
    # =========================================================
    limb_noise = 0.0006 * torch.randn_like(kp[:, limb_ids, :2])
    kp[:, limb_ids, :2] += limb_noise

    # =========================================================
    # 🔹 8. MICRO STABILIZATION
    # =========================================================
    kp[..., :2] += torch.randn_like(kp[..., :2]) * 0.0004

    # =========================================================
    # 🔹 9. CLAMP
    # =========================================================
    kp[..., :2] = torch.clamp(kp[..., :2], -1.2, 1.2)

    # =========================================================
    # 🔹 DEBUG
    # =========================================================
    if debug:
        motion_strength = (kp - keypoints_tensor).abs().mean()
        print(f"[DEBUG] motion: {motion_strength.item():.6f}")
        print(f"[DEBUG] motion_scale: {motion_scale}")
        print(f"[DEBUG] freq t: {t:.4f}")

    return kp

