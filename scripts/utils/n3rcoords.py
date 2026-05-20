#n3rcoords.py

import torch
import numpy as np
from .n3rMotionPoseClass import Pose
import math

def lerp(a, b, t):
    return a * (1 - t) + b * t

def prepare_pose_tensor(pose_full, device, target_dtype):
    """
    Convertit un tensor image en format BCHW, corrige les channels,
    normalise en [-1, 1], et envoie sur le bon device/dtype.

    Args:
        pose_full (torch.Tensor): input tensor (HWC, CHW ou BCHW)
        device (torch.device): device cible
        target_dtype (torch.dtype): dtype cible

    Returns:
        torch.Tensor: tensor prêt à l'emploi (B,3,H,W)
    """

    # --- Format BCHW ---
    if pose_full.ndim == 3:
        if pose_full.shape[0] in [1, 3]:  # CHW
            pose_full = pose_full.unsqueeze(0)
        else:  # HWC
            pose_full = pose_full.permute(2, 0, 1).unsqueeze(0)

    # --- Fix channels ---
    if pose_full.shape[1] > 3:
        pose_full = pose_full[:, :3]
    elif pose_full.shape[1] == 1:
        pose_full = pose_full.repeat(1, 3, 1, 1)

    # --- Normalisation [-1,1] ---
    pose_full = (pose_full - 0.5) * 2.0
    pose_full = torch.clamp(pose_full, -1.0, 1.0)

    # --- Device + dtype ---
    pose_full = pose_full.to(device=device, dtype=target_dtype)

    # --- Debug ---
    print(f"[DEBUG] Pose full {pose_full.shape} dtype={pose_full.dtype}")

    return pose_full

# --- Vérifier que les listes ne sont pas vides avant l'appel ---
def has_valid_coords_v1(face_coords_dict):
    for k, coords in face_coords_dict.items():
        if coords and all(isinstance(c, (list, tuple)) and len(c) == 2 for c in coords):
            return True
        return False

def has_valid_coords(face_coords_dict):
    """
    Vérifie si au moins une des listes de coordonnées contient des points valides.
    """
    if face_coords_dict is None:
        return False
    for coords in face_coords_dict.values():
        if coords and all(isinstance(c, (list, tuple)) and len(c) == 2 for c in coords):
            return True
    return False


def sanitize_coords(coords):
    """
    Filtre les coordonnées pour s'assurer qu'elles sont valides.
    Si les coordonnées sont valides (deux valeurs numériques), elles sont ajoutées à la liste valide.

    Args:
        coords (list/dict/None): Liste ou dictionnaire de coordonnées.

    Returns:
        list: Liste des coordonnées valides.
    """
    valid = []

    if coords is None:
        print("⚠️ Coordonnées None reçues, retour d'une liste vide")
        return valid

    # Si c'est un dictionnaire, on itère sur ses valeurs
    if isinstance(coords, dict):
        for key, value in coords.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    x, y = float(value[0]), float(value[1])
                    valid.append([x, y])
                except (ValueError, TypeError):
                    print(f"⚠️ Coordonnée ignorée pour {key} : {value}")
            else:
                print(f"⚠️ Coordonnée ignorée pour {key} : {value}")
    elif isinstance(coords, list):
        # Si c'est une liste, on applique la logique sur chaque élément
        for p in coords:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                print(f"⚠️ Coordonnée ignorée car invalide: {p}")
                continue
            try:
                x, y = float(p[0]), float(p[1])
                valid.append([x, y])
            except (ValueError, TypeError):
                print(f"⚠️ Coordonnée ignorée car invalide: {p}")
                continue
    else:
        print(f"⚠️ Type de coordonnées non supporté: {type(coords)}")

    return valid



def process_coords(coords, label="coords"):
    """
    Convertit les coordonnées en flottants et les affiche pour débogage.
    Args:
        coords (list/dict): Liste ou dictionnaire de coordonnées.
        label (str): Label pour l'affichage.

    Returns:
        list: Liste des coordonnées sous forme de [[x, y]].
    """
    if coords is not None:
        print(f"{label}: {coords}")
    else:
        return None

    if isinstance(coords, dict):
        # Si c'est un dictionnaire, on récupère les valeurs (qui sont des tuples ou des listes avec 2 éléments)
        coords = list(coords.values())

    # S'assure que chaque élément de coords est une paire de coordonnées (x, y)
    return [[float(x), float(y)] for x, y in coords]


def prepare_face_coords(
    eye_coords,
    mouth_coords,
    ear_coords,
    nose_coords,
    process_coords
):
    """
    Prépare les coordonnées du visage, les nettoie et les normalise.
    Les coordonnées sont traitées pour être prêtes à être utilisées ailleurs.

    Patch minimal : si une liste est vide, toutes les valeurs deviennent None.
    """
    if eye_coords is not None and mouth_coords is not None and ear_coords is not None and nose_coords is not None and process_coords is not None:
        print("🟢 Debug coords:")

        # --- Normalisation initiale ---
        process_coords(eye_coords, "eye_coords")
        process_coords(mouth_coords, "mouth_coords")
        process_coords(ear_coords, "ear_coords")

        print(f"nose_coords: {nose_coords}")

        # --- Sanitize ---
        eye_coords_list = sanitize_coords(eye_coords)
        mouth_coords_list = sanitize_coords(mouth_coords)
        ear_coords_list = sanitize_coords(ear_coords)
        nose_coords_list = sanitize_coords(nose_coords)

        print(f"eye_coords sanitize_coords: {eye_coords_list}")
        print(f"mouth_coords sanitize_coords: {mouth_coords_list}")
        print(f"ear_coords sanitize_coords: {ear_coords_list}")
        print(f"nose_coords sanitize_coords: {nose_coords_list}")

        # --- Patch minimal : si une liste est vide, on met tout à None ---
        if any(len(lst) == 0 for lst in [eye_coords_list, mouth_coords_list, ear_coords_list, nose_coords_list]):
            print("⚠️ Une des listes de coordonnées est vide, retour de None pour toutes")
            return None, None, None, None, None, None

        # --- Nez dict (conservé pour ailleurs) ---
        nose_coords_dict = nose_coords if nose_coords else None

        # --- Dict global ---
        face_coords_dict = {
            "eyes": eye_coords_list,
            "mouth": mouth_coords_list,
            "ears": ear_coords_list,
            "nose": nose_coords_list,
        }

        print(f"Face coords dict: {face_coords_dict}")

        return face_coords_dict, nose_coords_dict, eye_coords_list, mouth_coords_list, ear_coords_list, nose_coords_list
    else:
        return None, None, None, None, None, None

# =========================
# 🔹 NORMALISATION INPUTS
# =========================
def pair(coords, debug=False):
    if coords:
        out = [safe_xy(c, debug=debug) for c in coords]
        if debug:
            print(f"[pair] input={coords} → output={out}")
        return out
    if debug:
        print("[pair] coords=None → fallback (0,0)")
    return [(0, 0), (0, 0)]

# =========================
# 🔹 SAFE UTILS
# =========================
def safe_xy(coord, debug=False):
    if coord is None:
        if debug:
            print("[safe_xy] None → (0,0)")
        return (0, 0)

    if isinstance(coord, list) and len(coord) == 1:
        val = tuple(coord[0])
        if debug:
            print(f"[safe_xy] list[1] {coord} → {val}")
        return val

    if isinstance(coord, (list, tuple)) and len(coord) == 2:
        val = tuple(coord)
        if debug:
            print(f"[safe_xy] tuple/list {coord} → {val}")
        return val

    if debug:
        print(f"[safe_xy] format inconnu {coord} → (0,0)")
    return (0, 0)

def norm(coord):
    if isinstance(coord, (list, tuple)) and len(coord) == 1:
        return coord[0]
    return coord


def safe_update(idx, coord, keypoints_np, W, H, label="", debug=False):
    # =========================
    # 🔹 NORMALISATION ROBUSTE
    # =========================
    if coord is None:
        x, y = 0, 0
    elif isinstance(coord, dict):
        # sécurité si jamais MediaPipe ou autre passe un dict
        x, y = 0, 0
    elif isinstance(coord, (list, tuple)):

        # cas [[x,y]]
        if len(coord) == 1 and isinstance(coord[0], (list, tuple)):
            coord = coord[0]

        if len(coord) == 2:
            x, y = coord
        else:
            x, y = 0, 0
    else:
        x, y = 0, 0

    old_x, old_y = keypoints_np[idx, 0] * W, keypoints_np[idx, 1] * H

    # =========================
    # 🔹 LOGIQUE SAFE IDENTIQUE V1
    # =========================
    if x == 0 and y == 0:
        if old_x != 0 or old_y != 0:
            if debug:
                print(f"[safe_update] {label} fallback → garde ({old_x:.1f},{old_y:.1f})")
        else:
            if debug:
                print(f"[safe_update] ⚠ {label} ignoré (aucune valeur valide)")
        return

    # clamp sécurité (optionnel mais utile)
    x = max(0, min(x, W))
    y = max(0, min(y, H))

    if debug:
        print(f"[safe_update] {label}: ({old_x:.1f},{old_y:.1f}) → ({x:.1f},{y:.1f})")

    keypoints_np[idx, 0] = x / W
    keypoints_np[idx, 1] = y / H
    keypoints_np[idx, 2] = 1.0

#--------------------------------- Moteur d'animation Class ---------------------------

UPPER_BODY_MAP = {
    "nose": 0,
    "neck": 1,

    "right_shoulder": 2,
    "right_elbow": 3,
    "right_wrist": 4,

    "left_shoulder": 5,
    "left_elbow": 6,
    "left_wrist": 7,

    "right_clavicle": 8,
    "left_clavicle": 9,

    "chin": 10,
    "left_side_neck": 11,
    "right_side_neck": 12,
    "anchor": 13,

    "right_eye": 14,
    "left_eye": 15,
    "right_ear": 16,
    "left_ear": 17,
    "mouth": 18,

    "hips_center": 19,
    # =========================
    # 🔥 VIRTUAL / CINEMATIC POINTS
    # =========================
    'hair_root': 25,
    'hair_left': 26,
    'hair_right': 27,
    'hair_top': 28,
    'hair_top_left': 29,
    'hair_top_right': 30,

    'front_left_1': 52, # front gauche 1
    'front_left_2': 53, # front gauche 2
    'front_m': 54, # front milleu
    'front_right_1': 55, # front droit 1
    'front_right_2': 56, # front droit 2
}

class UpperBodySkeleton:
    """
    Mini skeleton pour le haut du corps uniquement
    Index convention:
        0 - nose
        1 - neck
        2 - right_shoulder
        3 - right_elbow
        4 - right_wrist
        5 - left_shoulder
        6 - left_elbow
        7 - left_wrist
        8 - right_clavicle
        9 - left_clavicle
        10 - chin
        11 - left_side_neck
        12 - right_side_neck
        13 - anchor
        14 - right_eye
        15 - left_eye
        16 - right_ear
        17 - left_ear
        18 - mouth
        19 - hips_center  # uniquement pour ancrage portrait
    """
    def __init__(self):
        self.joints = np.zeros((20, 2), dtype=np.float32)
        self.velocity = np.zeros((20, 2), dtype=np.float32)

class UpperBodyClip:
    def __init__(self):
        self.keyframes = []

    def add_keyframe(self, frame_id, pose):
        """
        pose = (20,2) np.array
        """
        self.keyframes.append(Keyframe(frame_id, pose))
        self.keyframes.sort(key=lambda x: x.frame_id)

def evaluate_upper_body_clip(clip, frame):
    if len(clip.keyframes) == 0:
        return None

    kf0, kf1 = None, None
    for i in range(len(clip.keyframes) - 1):
        if clip.keyframes[i].frame_id <= frame <= clip.keyframes[i + 1].frame_id:
            kf0 = clip.keyframes[i]
            kf1 = clip.keyframes[i + 1]
            break

    if kf0 is None:
        return clip.keyframes[0].pose

    t = (frame - kf0.frame_id) / (kf1.frame_id - kf0.frame_id + 1e-6)
    return lerp(kf0.pose, kf1.pose, t)



def build_upper_body_inputs(
    nose_coords,
    neck_coords,
    shoulders_coords,
    clavicules_coords,
    elbow_coords,
    wrists_coords,
    hips_coords,
    eye_coords,
    ear_coords,
    mouth_coords
):
    left_shoulder, right_shoulder = pair(shoulders_coords)
    left_clavicle, right_clavicle = pair(clavicules_coords)
    left_elbow, right_elbow = pair(elbow_coords)
    left_wrist, right_wrist = pair(wrists_coords)
    left_hip, right_hip = pair(hips_coords)
    left_eye, right_eye = pair(eye_coords)
    left_ear, right_ear = pair(ear_coords)

    neck_map = (
        neck_coords if isinstance(neck_coords, dict)
        else {"center": neck_coords}
    )

    return {
        "nose": nose_coords,
        "neck": norm(neck_map.get("center")),

        "right_shoulder": right_shoulder,
        "left_shoulder": left_shoulder,

        "right_elbow": right_elbow,
        "left_elbow": left_elbow,

        "right_wrist": right_wrist,
        "left_wrist": left_wrist,

        "right_clavicle": right_clavicle,
        "left_clavicle": left_clavicle,

        "chin": neck_map.get("chin"),
        "left_side_neck": neck_map.get("left"),
        "right_side_neck": neck_map.get("right"),
        "anchor": neck_map.get("anchor"),

        "right_eye": right_eye,
        "left_eye": left_eye,
        "right_ear": right_ear,
        "left_ear": left_ear,
        "mouth": mouth_coords,

        "hips_center": None if hips_coords is None else np.mean(hips_coords, axis=0)
    }


def animate_upper_body(
    pose: Pose,
    inputs: dict,
    mapping: dict = None,
    mode: str = "smooth",
    strength: float = 0.35,
    debug: bool = True  # <-- nouveau paramètre pour activer/désactiver les logs
):
    """
    Met à jour les keypoints du haut du corps à partir des inputs détectés.
    Affiche les positions avant/après si debug=True.
    """

    if mapping is None:
        mapping = pose.FACIAL_POINT_IDX

    # Cloner les keypoints pour ne pas modifier directement l'original
    keypoints = pose.keypoints.clone()  # shape: (1, N, 2)

    # -----------------------------
    # Update des points du haut du corps
    # -----------------------------
    for name, idx in mapping.items():
        if name in inputs:
            new_coord = inputs[name]  # tensor ou list
            if not isinstance(new_coord, torch.Tensor):
                new_coord = torch.tensor(new_coord, device=keypoints.device, dtype=keypoints.dtype)

            old_coord = keypoints[0, idx, :2].clone()

            if mode == "smooth":
                keypoints[0, idx, :2] = old_coord * (1 - strength) + new_coord * strength
            elif mode == "instant":
                keypoints[0, idx, :2] = new_coord
            else:
                raise ValueError(f"Mode '{mode}' non supporté, choisir 'smooth' ou 'instant'")

            if debug:
                print(f"[DEBUG] {name}: old={old_coord.cpu().numpy()} → new={keypoints[0, idx, :2].cpu().numpy()}")

    # -----------------------------
    # Mise à jour interne
    # -----------------------------
    pose._prev_keypoints = keypoints.clone()
    pose.keypoints = keypoints.clone()

    return keypoints



def generate_pose_sequence_keypoints(
    base_keypoints,
    num_frames=16,
    fps=10.0,
    breathing_strength=1.0,
    sway_strength=0.5,
    device="cuda",
    debug=False
):

    B, K, _ = base_keypoints.shape
    base_keypoints = base_keypoints.to(device)

    seq = []

    IDX = {
        "nose": 0,
        "neck": 1,
        "r_shoulder": 2,
        "l_shoulder": 5,
        "r_clav": 19,
        "l_clav": 20,
        "r_eye": 14,
        "l_eye": 15,
        "mouth": 18,
    }

    for f in range(num_frames):

        # =========================================================
        # 🔒 FRAME 0 = ANCHOR (CRUCIAL)
        # =========================================================
        if f == 0:
            kp = base_keypoints.clone()

            # sécurité
            kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)
            seq.append(kp)
            continue  # 🔥 skip tout le reste

        # =========================================================
        # 🔹 TEMPS
        # =========================================================
        t = f / fps
        phase = 2 * math.pi * (f / num_frames)

        kp = base_keypoints.clone()

        # =========================================================
        # 🫁 BREATHING
        # =========================================================
        breath = breathing_strength * (
            0.6 * math.sin(phase) +
            0.4 * math.sin(phase * 0.5 + 1.3)
        )
        breath *= 0.015

        kp[:, IDX["neck"], 1] -= breath * 0.6
        kp[:, IDX["r_shoulder"], 1] -= breath
        kp[:, IDX["l_shoulder"], 1] -= breath

        kp[:, IDX["r_shoulder"], 0] += breath * 0.4
        kp[:, IDX["l_shoulder"], 0] -= breath * 0.4

        # clavicle lag
        lag = math.sin(phase - 0.4) * 0.01
        kp[:, IDX["r_clav"], 1] -= lag
        kp[:, IDX["l_clav"], 1] -= lag

        # =========================================================
        # 💓 HEARTBEAT
        # =========================================================
        heartbeat = 0.002 * math.sin(t * 6.0)
        kp[:, IDX["neck"], 1] += heartbeat
        kp[:, IDX["nose"], 1] += heartbeat * 0.5

        # =========================================================
        # ⚖️ SWAY
        # =========================================================
        sway_x = sway_strength * 0.01 * math.sin(phase * 0.7)
        sway_y = sway_strength * 0.005 * math.cos(phase * 0.5)

        kp[:, :, 0] += sway_x
        kp[:, :, 1] += sway_y

        # =========================================================
        # ⚠️ ASYMMETRY
        # =========================================================
        asym = 0.003 * math.sin(phase * 1.3 + 2.0)

        kp[:, IDX["r_shoulder"], 1] += asym
        kp[:, IDX["l_shoulder"], 1] -= asym

        # =========================================================
        # 👁 FACE SYNC
        # =========================================================
        face_phase = math.sin(phase * 1.2)

        kp[:, IDX["nose"], 1] += face_phase * 0.002
        kp[:, IDX["mouth"], 1] += face_phase * 0.003

        kp[:, IDX["r_eye"], 0] += 0.001 * math.sin(t * 3.0)
        kp[:, IDX["l_eye"], 0] -= 0.001 * math.sin(t * 3.0)

        # =========================================================
        # 🔒 CLAMP
        # =========================================================
        kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)
        kp[..., 2] = base_keypoints[..., 2]

        seq.append(kp)

    if debug:
        print(f"[PoseSeq KP] frames: {num_frames}")
        print("[PoseSeq KP] frame0 == base:",
              torch.allclose(seq[0], base_keypoints))

    return seq




def reconstruct_hips(
    left_hip,
    right_hip,
    left_shoulder,
    right_shoulder,
    image_size=(1280, 896),
    hip_drop_ratio=0.65,     # ↓ hauteur dépend largeur épaules
    hip_width_ratio=1.20,    # ↔ hanches plus larges
    debug=False
):
    """
    Stable proportional hip reconstruction:
    - hip height ∝ shoulder width
    - hip width  ∝ shoulder width
    """

    H, W = image_size

    def valid(c):
        if c is None:
            return False
        x, y = c
        return np.isfinite(x) and np.isfinite(y)

    hips_reconstructed = False

    # =========================================================
    # 1. fallback condition
    # =========================================================
    if (not valid(left_hip)) or (not valid(right_hip)):

        if valid(left_shoulder) and valid(right_shoulder):

            ls = np.array(left_shoulder, dtype=np.float32)
            rs = np.array(right_shoulder, dtype=np.float32)

            shoulder_center = (ls + rs) * 0.5
            shoulder_vec = rs - ls

            shoulder_width = np.linalg.norm(shoulder_vec)
            if shoulder_width < 1e-6:
                shoulder_width = W * 0.1

            x_axis = shoulder_vec / shoulder_width

            # =================================================
            # 2. HEIGHT = function of shoulder width
            # =================================================
            hip_drop = shoulder_width * hip_drop_ratio

            hip_center = shoulder_center + np.array([0, hip_drop], dtype=np.float32)

            # =================================================
            # 3. WIDTH = function of shoulder width
            # =================================================
            half_width = shoulder_width * hip_width_ratio * 0.5

            left_hip  = hip_center - x_axis * half_width
            right_hip = hip_center + x_axis * half_width

            hips_reconstructed = True

            if debug:
                print("🦿 [HIP RECONSTRUCTION] proportional model applied")

    return left_hip, right_hip, hips_reconstructed

