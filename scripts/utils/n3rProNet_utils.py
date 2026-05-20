# n3rProNet_utils.py
#-------------------------------------------------------------------------------
from .tools_utils import ensure_4_channels, sanitize_latents, log_debug
from .n3rProPost_Prod import soft_tone_map, post_denoise_light, adjust_color_temperature, IntelligentGlowPro, AdaptivePostProcessor, neutralize_color_cast, apply_post_processing_unreal_cinematic, apply_post_processing_sketch
from .n3rProDenoising import denoise_latents, denoising_model, optimizer, criterion, show_latents, save_denoising_model, load_denoising_model, debug_tensor, train_denoiser
from .n3rProTemporal import TemporalResidualNet, TemporalLoss, save_temporal_model, weights_temporal_init, load_temporal_model
from .n3rStyleClass import  weights_init, StyleInjector, StyleLoss, save_style_model, load_style_model
from .n3rApparenceModel import  apply_appearance, appearance_model, optimizer_apparence, criterion_apparence
from .n3rCreativeModel import  apply_creative, creative_model, optimizer_creative, criterion_creative
from .n3rArtModel import  apply_art, art_model, optimizer_art, criterion_art
from .n3r_EMA import motion_aware_ema_fusion, motion_aware_ema_low_high, compute_high_freq_energy
from .n3r_latent_utils import export_latents_for_comfy_meta, export_latents_file, build_latent_metadata, load_latent, generate_sequence_id

from torch.optim import Adam
import torch
import math
import uuid
import time
from datetime import datetime
import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
import torch.nn.functional as F
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import mediapipe as mp
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from collections import deque



def scale_mouth_coords_to_latents(mouth_coords, img_H, img_W, lat_H, lat_W):
    """
    Convertit des coordonnées de bouche dans l'image originale
    vers l'espace latent correspondant.

    mouth_coords : liste de tuples [(x, y), ...] ou dictionnaire {key: (x, y), ...} ou None
    img_H, img_W : dimensions de l'image d'origine
    lat_H, lat_W : dimensions du latent
    """
    # 🔥 FIX : gérer None ou liste vide
    if not mouth_coords:
        return None

    # Si mouth_coords est un dictionnaire, récupère les valeurs (les coordonnées)
    if isinstance(mouth_coords, dict):
        mouth_coords = list(mouth_coords.values())

    # Calcule les facteurs d'échelle pour x et y
    scale_x = lat_W / img_W
    scale_y = lat_H / img_H

    # Retourne les coordonnées transformées dans l'espace latent
    return [(int(x * scale_x), int(y * scale_y)) for x, y in mouth_coords]
#----------------------------------------------------------------------------

def get_hips_coords_safe(image_pil, pose_model=None, H=None, W=None):
    """
    Détecte les coudes (left/right) de manière sécurisée.
    Si MediaPipe ne détecte pas les hanches.
    Retourne les coordonnées normalisées [0,1] pour x et y.
    """
    try:
        coords = get_hips_coords_pixels(image_pil, pose_model, H, W)
        if coords is None:
            print("⚠ Aucune hanche détecté.")
        return coords
    except Exception as e:
        print(f"[Hips detection ERROR] {e}")
        return None


def get_hips_coords_pixels(image_pil, pose_model=None, H=None, W=None):
    """
    Détecte les hanches de manière robuste :
    1️⃣ Priorité : Pose via pose_model si fourni
    2️⃣ Fallback proportionnel via épaules
    """

    img_width, img_height = image_pil.size
    if W is None: W = img_width
    if H is None: H = img_height

    image = np.array(image_pil.convert("RGB"))

    # =========================
    # 🔹 1. POSE (prioritaire)
    # =========================
    if pose_model is not None:
        try:
            results = pose_model.process(image)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                LEFT_HIP = 23
                RIGHT_HIP = 24

                left_hip = (int(round(lm[LEFT_HIP].x * W)),
                            int(round(lm[LEFT_HIP].y * H)))
                right_hip = (int(round(lm[RIGHT_HIP].x * W)),
                             int(round(lm[RIGHT_HIP].y * H)))

                print(f"🦿📍 Hips detected (Pose): left={left_hip}, right={right_hip}")
                return [left_hip, right_hip]

        except Exception as e:
            print(f"[Pose ERROR] {e}")

    # =========================
    # 🔹 2. FALLBACK via épaules
    # =========================
    shoulders = get_shoulders_coords(image_pil, pose_model, face_mesh=None)
    if shoulders is None:
        print("⚠ 🦿📍 Hips hanches non détectées, fallback impossible")
        return None

    left_shoulder, right_shoulder = shoulders
    vertical_offset = int(0.4 * H)

    left_hip = (int(round(left_shoulder[0])),
                int(round(left_shoulder[1] + vertical_offset)))
    right_hip = (int(round(right_shoulder[0])),
                 int(round(right_shoulder[1] + vertical_offset)))

    print(f"⚠ 🦿📍 Aucune hanche détectée, fallback proportionnel utilisé: left={left_hip}, right={right_hip}")
    return [left_hip, right_hip]



#----------------------------------------------------------------------------------
def get_elbows_coords_safe(image_pil, H=None, W=None):
    """
    Détecte les coudes (left/right) de manière sécurisée.
    Si MediaPipe ne détecte pas les coudes, utilise un fallback proportionnel
    basé sur épaules et cou.
    Retourne les coordonnées normalisées [0,1] pour x et y.
    """
    try:
        coords = get_elbows_coords_pixels(image_pil, H, W)
        if coords is None:
            print("⚠ Aucun coude détecté, fallback proportionnel utilisé")
        print(f"🦾 Elbows detected/estimated: {coords}")
        return coords
    except Exception as e:
        print(f"[Elbow detection ERROR] {e}")
        return None

def get_elbows_coords_pixels(image_pil, H=None, W=None):
    """
    Détecte les coudes en pixels via MediaPipe Pose.
    Si aucun coude n'est détecté, fallback proportionnel est utilisé.

    Args:
        image_pil (PIL.Image): image d'entrée
        H, W (int, optional): dimensions pour normaliser. Si None, prend image.size

    Returns:
        list[[x, y], [x, y]] : [left_elbow, right_elbow] en pixels
    """

    mp_pose = mp.solutions.pose

    img_width, img_height = image_pil.size
    if W is None: W = img_width
    if H is None: H = img_height

    image = np.array(image_pil.convert("RGB"))

    try:
        with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
            results = pose.process(image)

        fallback = True
        # fallback proportionnel
        left_elbow = [int(0.2 * W), int(0.7 * H)]
        right_elbow = [int(0.7 * W), int(0.7 * H)]

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
            RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            NOSE = mp_pose.PoseLandmark.NOSE.value

            left_shoulder = np.array([lm[LEFT_SHOULDER].x * W, lm[LEFT_SHOULDER].y * H])
            right_shoulder = np.array([lm[RIGHT_SHOULDER].x * W, lm[RIGHT_SHOULDER].y * H])
            nose = np.array([lm[NOSE].x * W, lm[NOSE].y * H])

            def estimate_elbow(shoulder, opposite_shoulder):
                torso_h = max(abs(shoulder[1] - nose[1]), 1)
                shoulder_w = max(abs(shoulder[0] - opposite_shoulder[0]), 1)
                x_offset = 0.15 * shoulder_w * np.sign(opposite_shoulder[0] - shoulder[0])
                y_offset = 0.35 * torso_h
                return [shoulder[0] + x_offset, shoulder[1] + y_offset]

            # visibilité
            left_elbow = ([lm[LEFT_ELBOW].x * W, lm[LEFT_ELBOW].y * H]
                          if lm[LEFT_ELBOW].visibility > 0.5
                          else estimate_elbow(left_shoulder, right_shoulder))
            right_elbow = ([lm[RIGHT_ELBOW].x * W, lm[RIGHT_ELBOW].y * H]
                           if lm[RIGHT_ELBOW].visibility > 0.5
                           else estimate_elbow(right_shoulder, left_shoulder))

            # convertir en int pixels
            left_elbow = [int(c) for c in left_elbow]
            right_elbow = [int(c) for c in right_elbow]
            fallback = False

        if fallback:
            print("⚠ Aucun coude détecté, fallback proportionnel utilisé")

        return [left_elbow, right_elbow]

    except Exception as e:
        print(f"[Elbow detection ERROR] {e}")
        return [[int(0.2*W), int(0.7*H)], [int(0.7*W), int(0.7*H)]]



#-----------------------------------------------------------------------------

def get_neck_coords_safe(image_pil, face_mesh, H=None, W=None):
    """
    Détecte les coordonnées approximatives du cou de manière sécurisée.
    Renvoie None si aucun visage n'est détecté ou en cas d'erreur.
    """
    try:
        coords = get_neck_coords_full(image_pil, face_mesh, H, W)
        if coords is None:
            print("⚠️ Aucun visage détecté ou cou non détecté")
            return None
        print(f"🦵 Neck detected: {coords}")
        return coords
    except Exception as e:
        print(f"[Neck detection ERROR] {e}")
        return None


def get_neck_coords_full(image_pil, face_mesh, H=None, W=None):

    image = np.array(image_pil.convert("RGB"))
    h, w, _ = image.shape

    if H is None: H = h
    if W is None: W = w

    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    CHIN       = 152
    LEFT_JAW   = 234
    RIGHT_JAW  = 454
    NOSE_BASE  = 1

    def get_point(idx):
        return int(lm[idx].x * W), int(lm[idx].y * H)

    chin  = get_point(CHIN)
    left  = get_point(LEFT_JAW)
    right = get_point(RIGHT_JAW)
    nose  = get_point(NOSE_BASE)

    # centre du cou
    center_x = (left[0] + right[0]) // 2
    neck_y = int(chin[1] + 0.2 * (chin[1] - nose[1]))

    center = (center_x, neck_y)

    return {
        "center": center,
        "chin": chin,
        "left": left,
        "right": right,
        "anchor": nose  # utile pour stabilisation
    }


def get_nose_coords_safe(image_pil, face_mesh):
    """
    Détecte les coordonnées du nez de manière sécurisée avec le FaceMesh singleton.
    Renvoie None si aucun visage n'est détecté ou en cas d'erreur.
    """
    if face_mesh is None:
        print("[Nose detection ERROR] face_mesh is None")
        return None

    try:
        coords = get_nose_coords_full(image_pil, face_mesh)
        if coords is None:
            print("⚠️ Aucun visage détecté ou nez non détecté")
            return None
        print(f"👃 Nose detected: {coords['center']}")
        return coords
    except Exception as e:
        print(f"[Nose detection ERROR] {e}")
        return None

def get_nose_coords_full(image_pil, face_mesh, debug=True):


    image = np.array(image_pil.convert("RGB"))
    h, w, _ = image.shape
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        if debug:
            print("[NoseDetection] Aucun visage détecté !")
        return None

    face_landmarks = results.multi_face_landmarks[0]

    # Points clés du nez
    NOSE_TIP_POINTS = [1, 2, 98, 327]
    NOSE_TOP_POINTS = [168, 6, 197]
    NOSE_LEFT_POINTS = [98, 97, 2]
    NOSE_RIGHT_POINTS = [327, 326, 2]
    NOSE_BOTTOM_POINTS = [2, 5]

    def get_point(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def average_points(indices, name=""):
        try:
            pts = np.array([get_point(i) for i in indices])
            return np.mean(pts, axis=0)
        except Exception as e:
            if debug:
                print(f"[NoseDetection ERROR] Impossible de calculer {name}: {e}")
            raise

    try:
        tip = average_points(NOSE_TIP_POINTS, "tip")
        top = average_points(NOSE_TOP_POINTS, "top")
        left = average_points(NOSE_LEFT_POINTS, "left")
        right = average_points(NOSE_RIGHT_POINTS, "right")
        bottom = average_points(NOSE_BOTTOM_POINTS, "bottom")
    except Exception:
        if debug:
            print("[NoseDetection] Fallback proportionnel activé")
        # Fallback proportionnel basé sur les yeux et la bouche
        eyes = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark[33:263]]  # approx
        eyes = np.array(eyes)
        tip = np.mean(eyes, axis=0)
        top = tip - np.array([0, 20])
        left = tip - np.array([10, 0])
        right = tip + np.array([10, 0])
        bottom = tip + np.array([0, 20])

    # Centre horizontal
    center_x = (left[0] + right[0]) * 0.5
    # Centre vertical pondéré
    center_y = (top[1] * 0.3 + tip[1] * 0.45 + bottom[1] * 0.25)

    center = (int(center_x), int(center_y))

    if debug:
        print(f"[NoseDetection] tip={tip}, top={top}, left={left}, right={right}, bottom={bottom}")
        print(f"[NoseDetection] center={center}")

    return {
        "center": center,
        "tip": tuple(tip.astype(int)),
        "top": tuple(top.astype(int)),
        "left": tuple(left.astype(int)),
        "right": tuple(right.astype(int)),
        "bottom": tuple(bottom.astype(int))
    }



def get_mouth_coords_safe(image_pil, face_mesh, H=None, W=None):
    """
    Détecte les coordonnées de la bouche de manière sécurisée.
    Renvoie None si aucun visage n'est détecté ou en cas d'erreur.
    """
    try:
        coords = get_mouth_coords(image_pil, face_mesh)
        if coords is None:
            print("⚠️ Aucun visage détecté ou bouche non détectée")
            return None
        print(f"👄 Mouth detected: {coords}")
        return coords
    except Exception as e:
        print(f"[Mouth detection ERROR] {e}")
        return None


def get_mouth_coords(image_pil, face_mesh):
    """
    Détecte les coordonnées de la bouche avec MediaPipe FaceMesh (mode tracking).

    Args:
        image_pil (PIL.Image): image d'entrée
        face_mesh: instance persistante MediaPipe FaceMesh

    Returns:
        dict: Dictionnaire des coordonnées des points de la bouche.
    """

    # Convertir l'image PIL en format compatible avec MediaPipe
    image = np.array(image_pil.convert("RGB"))
    h, w, _ = image.shape

    # 🔥 Utilisation de l'instance existante
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        print("⚠️ No face detected (mouth)")
        return None

    face_landmarks = results.multi_face_landmarks[0]

    # 🔹 Indices bouche (outer lips)
    MOUTH_OUTER = [61, 291, 0, 17, 37, 267, 78, 308]

    MOUTH_LEFT = [61]
    MOUTH_RIGHT = [291]

    MOUTH_TOP_MID_R3 = [270]
    MOUTH_TOP_MID_R2 = [269]
    MOUTH_TOP_MID_R1 = [267]
    MOUTH_TOP_MID = [0]
    MOUTH_TOP_MID_L1 = [37]
    MOUTH_TOP_MID_L2 = [39]
    MOUTH_TOP_MID_L3 = [40]

    MOUTH_BOT_MID_R3 = [321]
    MOUTH_BOT_MID_R2 = [405]
    MOUTH_BOT_MID_R1 = [314]
    MOUTH_BOT_MID = [17]
    MOUTH_BOT_MID_L1 = [84]
    MOUTH_BOT_MID_L2 = [181]
    MOUTH_BOT_MID_L3 = [91]

    def get_center(indices):
        """Retourne le centre de la bouche pour un ensemble d'indices."""
        xs, ys = [], []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    # Calcul des coordonnées pour chaque partie de la bouche
    mouth_coords = {}

    mouth_coords['mouth_center'] = get_center(MOUTH_OUTER)

    mouth_coords['mouth_left'] = get_center(MOUTH_LEFT)
    mouth_coords['mouth_right'] = get_center(MOUTH_RIGHT)

    mouth_coords['mouth_top_mid_r3'] = get_center(MOUTH_TOP_MID_R3)
    mouth_coords['mouth_top_mid_r2'] = get_center(MOUTH_TOP_MID_R2)
    mouth_coords['mouth_top_mid_r1'] = get_center(MOUTH_TOP_MID_R1)
    mouth_coords['mouth_top_mid'] = get_center(MOUTH_TOP_MID)
    mouth_coords['mouth_top_mid_l1'] = get_center(MOUTH_TOP_MID_L1)
    mouth_coords['mouth_top_mid_l2'] = get_center(MOUTH_TOP_MID_L2)
    mouth_coords['mouth_top_mid_l3'] = get_center(MOUTH_TOP_MID_L3)

    mouth_coords['mouth_bot_mid_r3'] = get_center(MOUTH_BOT_MID_R3)
    mouth_coords['mouth_bot_mid_r2'] = get_center(MOUTH_BOT_MID_R2)
    mouth_coords['mouth_bot_mid_r1'] = get_center(MOUTH_BOT_MID_R1)
    mouth_coords['mouth_bot_mid'] = get_center(MOUTH_BOT_MID)
    mouth_coords['mouth_bot_mid_l1'] = get_center(MOUTH_BOT_MID_L1)
    mouth_coords['mouth_bot_mid_l2'] = get_center(MOUTH_BOT_MID_L2)
    mouth_coords['mouth_bot_mid_l3'] = get_center(MOUTH_BOT_MID_L3)

    return mouth_coords

#--------------------------------------------------------------------------------
def get_wrists_coords(image_pil, pose_model, H=None, W=None):
    """
    Récupère les coordonnées des poignets avec MediaPipe Pose de manière robuste.
    Fallback proportionnel si non détectés.

    Args:
        image_pil (PIL.Image): image d'entrée
        pose_model: MediaPipe Pose déjà initialisé
        H, W: dimensions pour normaliser (pixels)

    Returns:
        list[(x, y)] ou None: [left_wrist, right_wrist]
    """

    if pose_model is None:
        print("⚠️ Pose model non initialisé")
        return None

    try:
        image = np.array(image_pil.convert("RGB"))
        img_h, img_w, _ = image.shape
        if H is None: H = img_h
        if W is None: W = img_w

        results = pose_model.process(image)
        if not results.pose_landmarks:
            print("⚠️ Aucun landmark détecté pour les poignets")
            return None

        lm = results.pose_landmarks.landmark
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        left_lm = lm[LEFT_WRIST]
        right_lm = lm[RIGHT_WRIST]

        # Vérification de visibilité
        if left_lm.visibility < 0.5 or right_lm.visibility < 0.5:
            print("⚠️ Poignets peu visibles, fallback proportionnel utilisé")
            # fallback proportionnel approximatif : aligné sur coudes
            # ici juste retour None, tu peux adapter avec offset
            return None

        left_wrist = (int(left_lm.x * W), int(left_lm.y * H))
        right_wrist = (int(right_lm.x * W), int(right_lm.y * H))

        print(f"🖐️ Wrists detected: {[left_wrist, right_wrist]}")
        return [left_wrist, right_wrist]

    except Exception as e:
        print(f"[Wrist detection ERROR] {e}")
        return None


def get_wrists_coords_safe(image_pil, pose_model, face_mesh, H=None, W=None):
    """
    Détecte les poignets avec MediaPipe Pose de manière sécurisée.
    Retourne None si aucun poignet n'est détecté.
    """
    try:
        coords = get_wrists_coords_pixels(image_pil, pose_model, face_mesh, H, W)
        if coords is None:
            print("⚠ Aucun poignet / Wrists détecté")
            return None
        print(f"✋ Wrists detected: {coords}")
        return coords
    except Exception as e:
        print(f"[Wrists detection ERROR] {e}")
        return None


def get_wrists_coords_pixels(image_pil, pose_model, face_mesh, H=None, W=None):
    """
    Retourne les coordonnées des poignets en pixels.
    Fallback proportionnel si MediaPipe échoue.
    """


    # --- Taille de l'image ---
    img_width, img_height = image_pil.size
    if W is None: W = img_width
    if H is None: H = img_height

    # --- Essaye MediaPipe Pose ---
    coords = get_wrists_coords(image_pil, H, W)
    if coords is not None:
        return coords

    print("⚠ Aucun poignet détecté, fallback proportionnel utilisé")

    # --- Fallback proportionnel depuis les coudes ---
    elbows = get_elbows_coords_pixels(image_pil, H, W)
    shoulders = get_shoulders_coords(image_pil, pose_model, face_mesh, H, W)
    if elbows is None or shoulders is None:
        return None

    left_wrist = (elbows[0][0], elbows[0][1] + 0.2 * H)
    right_wrist = (elbows[1][0], elbows[1][1] + 0.2 * H)

    return [left_wrist, right_wrist]


#--------------------------------------------------------------------------------

def scale_eye_coords_to_latents(eye_coords, img_H, img_W, lat_H, lat_W):
    """
    Convertit coords image -> latent space
    """

    # 🔥 FIX : gérer None ou liste vide
    if not eye_coords:
        return None

    scale_x = lat_W / img_W
    scale_y = lat_H / img_H

    return [(int(x * scale_x), int(y * scale_y)) for x, y in eye_coords]


def get_eye_coords_safe(image_pil, face_mesh, H=None, W=None):
    try:
        coords = get_eye_coords(image_pil, face_mesh)
        if coords is None:
            print("⚠️ Aucun visage détecté")
            return None
        print(f"👁 Eyes detected: {coords}")
        return coords
    except Exception as e:
        print(f"[Eye detection ERROR] {e}")
        return None


def get_eye_coords(image_pil, face_mesh):
    """
    Détecte les coordonnées des yeux avec MediaPipe (mode tracking).
    """

    image = np.array(image_pil.convert("RGB"))
    h, w, _ = image.shape

    # 🔥 utiliser l'instance EXISTANTE
    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]


    LEFT_IRIS1 = [474]
    LEFT_IRIS2 = [475]
    LEFT_IRIS3 = [476]
    LEFT_IRIS4 = [477]

    RIGHT_IRIS1 = [469]
    RIGHT_IRIS2 = [470]
    RIGHT_IRIS3 = [471]
    RIGHT_IRIS4 = [472]

    def get_center(indices):
        xs, ys = [], []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    left_eye = get_center(LEFT_IRIS)
    right_eye = get_center(RIGHT_IRIS)

    left_iris1 = get_center(LEFT_IRIS1)
    left_iris2 = get_center(LEFT_IRIS2)
    left_iris3 = get_center(LEFT_IRIS3)
    left_iris4 = get_center(LEFT_IRIS4)

    right_iris1 = get_center(RIGHT_IRIS1)
    right_iris2 = get_center(RIGHT_IRIS2)
    right_iris3 = get_center(RIGHT_IRIS3)
    right_iris4 = get_center(RIGHT_IRIS4)

    # Left eye indices list
    # LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
    # Right eye indices list
    # RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]


    return [left_eye, right_eye, left_iris1, left_iris2, left_iris3, left_iris4, right_iris1, right_iris2, right_iris3, right_iris4]


def get_elbows_coords_safe_test(image_pil, pose_model):
    try:
        coords = get_elbows_coords(image_pil, pose_model)
        if coords is None:
            print("⚠️ Aucun coude détecté")
            return None
        print(f"🦾 Elbows detected: {coords}")
        return coords
    except Exception as e:
        print(f"[Elbow detection ERROR] {e}")
        return None

def get_shoulders_coords(image_pil, pose_model, face_mesh=None):
    """
    Détecte les épaules de manière robuste :
    1. Pose (prioritaire)
    2. Fallback via clavicules (si dispo)
    """


    if pose_model is None:
        print("⚠️ Pose model non initialisé")
        return None

    image = np.array(image_pil.convert("RGB"))
    h, w, _ = image.shape

    # =========================
    # 🔹 1. POSE (prioritaire)
    # =========================
    try:
        results = pose_model.process(image)

        if results.pose_landmarks:
            l = results.pose_landmarks.landmark[11]
            r = results.pose_landmarks.landmark[12]

            if l.visibility > 0.5 and r.visibility > 0.5:
                left = (int(l.x * w), int(l.y * h))
                right = (int(r.x * w), int(r.y * h))

                print(f"🦾 Shoulders detected (Pose): {[left, right]}")
                return [left, right]

    except Exception as e:
        print(f"[Pose ERROR] {e}")

    # =========================
    # 🔹 2. FALLBACK CLAVICULES
    # =========================
    try:
        clav_coords = get_clavicules_coords_safe(
            image_pil, pose_model, face_mesh, H=h, W=w
        )

        if clav_coords is not None:
            left_clav, right_clav = clav_coords

            # 👉 estimation douce (pas de gros offsets)
            dx = right_clav[0] - left_clav[0]
            dy = right_clav[1] - left_clav[1]

            # 👉 petit décalage vers l’extérieur + léger drop vertical
            shoulder_ratio = 0.25   # ⚠️ beaucoup plus faible que ton ancien 0.7
            vertical_offset = 0.02 * h

            left = (
                int(left_clav[0] - shoulder_ratio * dx),
                int(left_clav[1] - shoulder_ratio * dy + vertical_offset)
            )

            right = (
                int(right_clav[0] + shoulder_ratio * dx),
                int(right_clav[1] + shoulder_ratio * dy + vertical_offset)
            )

            print(f"🦾 Shoulders estimated (clavicles fallback): {[left, right]}")
            return [left, right]

        else:
            print("⚠️ Clavicules indisponibles → fallback impossible")

    except Exception as e:
        print(f"[Clavicle fallback ERROR] {e}")

    # =========================
    # ❌ Échec total
    # =========================
    print("⚠️ Impossible de détecter les épaules (Pose + fallback)")
    return None


# ------------------------------
# 1️⃣ Fonction "safe" (appels externes)
# ------------------------------
def get_shoulders_coords_safe(image_pil, pose_model=None, face_mesh=None, H=None, W=None):
    """
    Détecte les épaules via Pose ou fallback FaceMesh.
    Retourne None si aucune détection possible.
    """
    try:
        coords = get_shoulders_coords(image_pil, pose_model, face_mesh, H, W)
        if coords is None:
            print("⚠️ Aucune détection d'épaules")
            return None
        print(f"🦾 Shoulders detected: {coords}")
        return coords
    except Exception as e:
        print(f"[Shoulder detection ERROR] {e}")
        return None


# ------------------------------
# 2️⃣ Fonction principale (pose + fallback)
# ------------------------------
def get_shoulders_coords(image_pil, pose_model=None, face_mesh=None, H=None, W=None):
    """
    Détecte les épaules via MediaPipe Pose (prioritaire) ou FaceMesh (fallback).

    Args:
        image_pil (PIL.Image)
        pose_model : instance de MediaPipe Pose déjà initialisée
        face_mesh  : instance de MediaPipe FaceMesh déjà initialisée
        H, W       : dimensions de l'image pour normaliser

    Returns:
        list[(x, y)] ou None
    """
    image = np.array(image_pil.convert("RGB"))
    img_h, img_w, _ = image.shape
    if H is None: H = img_h
    if W is None: W = img_w

    # --- 1️⃣ Pose (prioritaire) ---
    if pose_model is not None:
        try:
            results = pose_model.process(image)
            if results.pose_landmarks:
                LEFT_SHOULDER = 11
                RIGHT_SHOULDER = 12
                l = results.pose_landmarks.landmark[LEFT_SHOULDER]
                r = results.pose_landmarks.landmark[RIGHT_SHOULDER]

                if l.visibility >= 0.5 and r.visibility >= 0.5:
                    left_coords = (int(l.x * W), int(l.y * H))
                    right_coords = (int(r.x * W), int(r.y * H))
                    return [left_coords, right_coords]
                else:
                    print("⚠️ Épaules détectées mais peu visibles via Pose")
        except Exception as e:
            print(f"[Pose detection ERROR] {e}")

    # --- 2️⃣ Fallback FaceMesh ---
    if face_mesh is not None:
        try:
            results = face_mesh.process(image)
            if not results.multi_face_landmarks:
                return None
            lm = results.multi_face_landmarks[0].landmark

            LEFT_JAW = 234
            RIGHT_JAW = 454
            CHIN = 152

            def jaw_to_shoulder(idx):
                x = lm[idx].x * W
                y = lm[idx].y * H
                y += 1.2 * (lm[CHIN].y * H - y)  # projection vers torse
                return int(x), int(y)

            left_coords = jaw_to_shoulder(LEFT_JAW)
            right_coords = jaw_to_shoulder(RIGHT_JAW)
            return [left_coords, right_coords]

        except Exception as e:
            print(f"[FaceMesh fallback ERROR] {e}")

    return None

def get_shoulders_coords_old(image_pil, H=None, W=None):
    """
    Approxime les coordonnées des épaules à partir des clavicules
    avec un léger décalage proportionnel et correction de l'inclinaison du torse.

    Args:
        image_pil (PIL.Image): image d'entrée
        H, W: dimensions optionnelles pour normaliser

    Returns:
        list[(x, y)] ou None: gauche et droite des épaules
    """

    # --- Récupère les clavicules ---
    clav_coords = get_clavicules_coords(image_pil, H, W)
    if clav_coords is None:
        print("⚠️ Impossible de détecter les clavicules pour calculer les épaules")
        return None

    left_clav, right_clav = clav_coords

    # --- Taille de l'image pour proportion ---
    img_width, img_height = image_pil.size
    if W is None: W = img_width
    if H is None: H = img_height

    # --- Distance et angle ---
    dx = right_clav[0] - left_clav[0]
    dy = right_clav[1] - left_clav[1]
    angle = np.arctan2(dy, dx)  # angle du torse (rad)

    # --- Ratios proportionnels ---
    shoulder_x_ratio_left = -0.7   # déplacement relatif gauche
    shoulder_x_ratio_right = 0.7   # déplacement relatif droite
    shoulder_y_ratio = 0.03        # léger décalage vertical proportionnel à la hauteur de l'image

    # --- Décalage vectoriel selon angle du torse ---
    def offset(x_ratio, y_ratio, dx, dy, H):
        # décale proportionnellement sur l'axe du torse et verticalement
        x_offset = x_ratio * dx
        y_offset = x_ratio * dy + y_ratio * H
        return x_offset, y_offset

    left_offset = offset(shoulder_x_ratio_left, shoulder_y_ratio, dx, dy, H)
    right_offset = offset(shoulder_x_ratio_right, shoulder_y_ratio, dx, dy, H)

    left_shoulder = (left_clav[0] + left_offset[0], left_clav[1] + left_offset[1])
    right_shoulder = (right_clav[0] + right_offset[0], right_clav[1] + right_offset[1])

    return [left_shoulder, right_shoulder]
#-------------------------------------- oreilles ----------



def release_mediapipe_models():
    if hasattr(init_mediapipe_models, "pose_model"):
        init_mediapipe_models.pose_model.close()

    if hasattr(init_mediapipe_models, "face_mesh"):
        init_mediapipe_models.face_mesh.close()

    print("🧹 MediaPipe models released")

def get_face_mesh():
    """
    Retourne une instance persistante de MediaPipe FaceMesh (mode vidéo).
    Évite de recréer l'objet à chaque frame (gain énorme CPU/GPU).
    """
    if not hasattr(get_face_mesh, "_instance"):
        mp_face_mesh = mp.solutions.face_mesh

        get_face_mesh._instance = mp_face_mesh.FaceMesh(
            static_image_mode=False,   # 🔥 tracking vidéo
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ FaceMesh initialisé (mode vidéo)")

    return get_face_mesh._instance

def release_face_mesh():
    """
    Libère proprement FaceMesh (optionnel mais recommandé en fin de run)
    """
    if hasattr(get_face_mesh, "_instance"):
        get_face_mesh._instance.close()
        del get_face_mesh._instance
        print("🧹 FaceMesh libéré")
# --------- detection des cheveux  ---------------------------


def get_hair_coords_safe(image_pil, face_mesh, H=None, W=None):
    try:
        coords = get_hair_coords(image_pil, face_mesh)

        if coords is None:
            print("⚠️ Aucun visage détecté (hair)")
            return None

        print(f"💇 Hair detected: {coords}")
        return coords

    except Exception as e:
        print(f"[Hair detection ERROR] {e}")
        return None


def get_hair_coords(image_pil, face_mesh):
    image = np.array(image_pil.convert("RGB"))
    h, w, _ = image.shape

    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    """
            'hair_root': 25,

            'hair_left': 26,
            'hair_right': 27, # bouche gauche 'mouth_left': 48
            'hair_top': 28, # nez gauche - nose_left'  -> 'nose': 0,
            'hair_top_left': 29,
            'hair_top_right': 30,

            'left_top_hair1': 31,
            'left_top_hair2': 32, # bouche droite 'mouth_right': 49,
            'left_top_hair3': 33,

            'right_top_hair1': 34, # nez droite - nose_right
            'right_top_hair2': 35,
            'right_top_hair3': 36,

            'top_hair1': 37,
            'top_hair2': 38,
            'top_hair3': 39,

            'mouth_left': 40,    # coin gauche des lèvres supérieures
            'mouth_right': 41,   # coin droit des lèvres supérieures

            'nose_left': 42,    # coin gauche du nez
            'nose_right': 43,   # coin droit du nez
    """
    # =========================
    # HAIR PROXY POINTS (MediaPipe FaceMesh)
    # =========================
    HAIR_LEFT = [251, 389]  # Points sur le côté gauche des cheveux
    HAIR_RIGHT = [21, 162]  # Points sur le côté droit des cheveux
    HAIR_TOP = [10]  # Point central au sommet
    HAIR_LEFT_TOP = [338, 297, 332, 284, 251]  # Points vers le sommet des cheveux (gauche)
    HAIR_RIGHT_TOP = [109, 67, 103, 51, 21]  # Points vers le sommet des cheveux (droit)


    # Points pour les coins de la bouche MOUTH_OUTER = [61, 291, 0, 17, 37, 267, 78, 308]
    M_L = [287]  # Point gauche point coins de la bouche OK
    M_R = [43]  # Point droite point coins de la bouche OK

    # Points sur le nez  NOSE_TIP    = 1 NOSE_TOP    = 168 NOSE_LEFT   = 98 NOSE_RIGHT  = 327
    NOZE_LEFT = [98]  # Point gauche au sommet, 1er point NOZE
    NOZE_RIGHT = [327]  # Point droit au sommet, 3ème point NOZE (290 narine)

    HAIR_RIGHT_TOP1 = [109]  # Point droit au sommet, 2ème point
    HAIR_RIGHT_TOP2 = [103]  # Point droit au sommet, 2ème point centre droit
    HAIR_RIGHT_TOP3 = [21]  # Point droit au sommet, 2ème point

    HAIR_LEFT_TOP1 = [338]  # Point gauche au sommet, 1ème point
    HAIR_LEFT_TOP2 = [332]  # Point gauche au sommet, 2ème point centre gauche
    HAIR_LEFT_TOP3 = [251]  # Point gauche au sommet, 3ème point


    FRONT_299  = [299] # point anexe cheveux long milleu
    FRONT_104  = [104] # point anexe cheveux long milleu
    FRONT_TOP0 = [151] # point anexe cheveux long milleu
    FRONT_162 = [162] # point anexe cheveux long droite
    FRONT_389 = [389] # point anexe cheveux long gauche

    def get_center(indices):
        xs, ys = [], []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    left_hair = get_center(HAIR_LEFT)
    right_hair = get_center(HAIR_RIGHT)
    hair_top = get_center(HAIR_TOP) #OK
    # Calculer le centre du sommet des cheveux
    left_top_hair = get_center(HAIR_LEFT_TOP) #OK
    right_top_hair = get_center(HAIR_RIGHT_TOP)


    right_top_hair1 = get_center(HAIR_RIGHT_TOP1) #OK
    right_top_hair2 = get_center(HAIR_RIGHT_TOP2) #OK
    right_top_hair3 = get_center(HAIR_RIGHT_TOP3) #OK

    left_top_hair1 = get_center(HAIR_LEFT_TOP1) #OK
    left_top_hair2 = get_center(HAIR_LEFT_TOP2) #OK
    left_top_hair3 = get_center(HAIR_LEFT_TOP3) #OK


    front_left2 = get_center(FRONT_389) #OK 389 point exterme
    front_left1 = get_center(FRONT_299) # 299
    front_m = get_center(FRONT_TOP0) #OK # point anexe Front milleu 151
    front_right1 = get_center(FRONT_104) # 104
    front_right2 = get_center(FRONT_162) #OK 162 point exterme


    # Calcul des points pour le nez et les coins de la bouche
    mouth_left = get_center(M_L) #OK
    mouth_right = get_center(M_R) #OK
    nose_left = get_center(NOZE_LEFT) #OK
    nose_right = get_center(NOZE_RIGHT) #OK


    # Calculer le centre du front (base des cheveux)
    center_x = (left_hair[0] + right_hair[0]) // 2
    center_y = (left_hair[1] + right_hair[1]) // 2


    return [left_hair, left_top_hair, mouth_left, nose_left, hair_top, right_top_hair, mouth_right, right_top_hair1, right_top_hair2, right_top_hair3, left_top_hair1, left_top_hair2, left_top_hair3, nose_right, right_hair, (center_x, center_y), front_m, front_left1, front_left2, front_right1, front_right2]


# ---------- detection des oreilles ----------

def get_ear_coords_safe(image_pil, face_mesh, H=None, W=None):
    try:
        coords = get_ear_coords(image_pil, face_mesh)

        if coords is None:
            print("⚠️ Aucun visage détecté (ears)")
            return None

        print(f"👂 Ears detected: {coords}")
        return coords

    except Exception as e:
        print(f"[Ear detection ERROR] {e}")
        return None


def get_ear_coords(image_pil, face_mesh):

    image = np.array(image_pil.convert("RGB"))
    h, w, _ = image.shape

    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    LEFT_EAR  = [234]
    RIGHT_EAR = [454]

    def get_center(indices):
        xs, ys = [], []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    left_ear = get_center(LEFT_EAR)
    right_ear = get_center(RIGHT_EAR)

    return [left_ear, right_ear]


def init_mediapipe_models():

    if not hasattr(init_mediapipe_models, "_initialized"):

        mp_pose = mp.solutions.pose

        init_mediapipe_models.pose_model = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )

        init_mediapipe_models.face_mesh = get_face_mesh()

        init_mediapipe_models._initialized = True
        print("✅ MediaPipe fully initialized")

    return init_mediapipe_models.pose_model, init_mediapipe_models.face_mesh

def get_clavicules_coords_safe(image_pil, pose_model, face_mesh, H=None, W=None):
    """
    Détecte les coordonnées des épaules avec MediaPipe Pose de manière sécurisée.
    Retourne None si aucun torse détecté ou modèles non initialisés.
    """
    if pose_model is None:
        print("⚠️ Pose model non initialisé")
        return None
    if face_mesh is None:
        print("⚠️ FaceMesh model non initialisé")
        return None

    try:
        coords = get_clavicules_coords(image_pil, pose_model, face_mesh, H, W)
        if coords is None:
            print("⚠️ Aucune détection clavicules")
            return None
        print(f"🦾 clavicules detected: {coords}")
        return coords
    except Exception as e:
        print(f"[clavicules detection ERROR] {e}")
        return None


def get_clavicules_coords(image_pil, pose_model, face_mesh, H=None, W=None):
    """
    Détecte les clavicules avec Pose (prioritaire) + fallback FaceMesh.
    Version optimisée tracking (aucune re-init).
    """

    image = np.array(image_pil.convert("RGB"))
    h, w, _ = image.shape
    if H is None: H = h
    if W is None: W = w

    # =========================
    # 🔹 1. POSE (prioritaire)
    # =========================
    if pose_model is not None:
        try:
            results = pose_model.process(image)

            if results.pose_landmarks:
                LEFT_SHOULDER = 11
                RIGHT_SHOULDER = 12

                l = results.pose_landmarks.landmark[LEFT_SHOULDER]
                r = results.pose_landmarks.landmark[RIGHT_SHOULDER]

                left_x, left_y = l.x * W, l.y * H
                right_x, right_y = r.x * W, r.y * H

                # élargissement naturel
                shoulder_span = right_x - left_x
                left_x  -= 0.15 * shoulder_span
                right_x += 0.15 * shoulder_span

                return [
                    (int(left_x), int(left_y)),
                    (int(right_x), int(right_y))
                ]
        except Exception as e:
            print(f"[Pose detection ERROR] {e}")

    # =========================
    # 🔹 2. FALLBACK FACEMESH
    # =========================
    if face_mesh is not None:
        try:
            results = face_mesh.process(image)

            if not results.multi_face_landmarks:
                print("⚠️ No face detected (clavicles fallback)")
                return None

            lm = results.multi_face_landmarks[0].landmark

            LEFT_JAW  = 234
            RIGHT_JAW = 454
            CHIN      = 152

            def jaw_to_shoulder(idx):
                x = lm[idx].x * W
                y = lm[idx].y * H

                # projection vers torse
                chin_y = lm[CHIN].y * H
                y += 1.2 * (chin_y - y)

                return x, y

            left_x, left_y = jaw_to_shoulder(LEFT_JAW)
            right_x, right_y = jaw_to_shoulder(RIGHT_JAW)

            shoulder_span = right_x - left_x
            left_x  -= 0.15 * shoulder_span
            right_x += 0.15 * shoulder_span

            return [
                (int(left_x), int(left_y)),
                (int(right_x), int(right_y))
            ]

        except Exception as e:
            print(f"[FaceMesh fallback ERROR] {e}")

    # Aucun modèle dispo
    return None



def apply_glow_froid_iris(latents, eye_coords, iris_radius_ratio=0.08, strength=0.25, blur_kernel=5):
    """
    Applique un glow froid ciblé sur l'iris des yeux dans les latents [B,C,H,W].

    Args:
        latents (torch.Tensor): Latents [B,C,H,W].
        eye_coords (list of tuples): Coordonnées yeux [(x1,y1),(x2,y2)].
        iris_radius_ratio (float): Ratio de rayon de l'iris par rapport à la plus petite dimension H/W.
        strength (float): Intensité du glow (0.0 à 1.0).
        blur_kernel (int): Taille du noyau pour un léger flou gaussien.

    Returns:
        torch.Tensor: Latents avec glow appliqué sur les iris.
    """
    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ Créer un masque radial pour chaque œil
    mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    min_dim = min(H, W)
    iris_radius = iris_radius_ratio * min_dim

    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    for x_eye, y_eye in eye_coords:
        dist = torch.sqrt((xx - x_eye)**2 + (yy - y_eye)**2)
        eye_mask = torch.exp(-(dist**2) / (2 * iris_radius**2))
        mask += eye_mask.unsqueeze(0)  # broadcast batch dimension

    # Clamp à 1 pour éviter dépassement si 2 yeux se chevauchent
    mask = mask.clamp(0.0, 1.0)

    # 2️⃣ Appliquer léger blur pour adoucir les bords
    if blur_kernel > 1:
        kernel = torch.ones((C, 1, blur_kernel, blur_kernel), device=device, dtype=dtype)
        kernel = kernel / kernel.sum()
        mask = F.conv2d(mask.repeat(1, C, 1, 1), kernel, padding=blur_kernel//2, groups=C)

    # 3️⃣ Créer glow gaussien via convolution légère
    sigma = blur_kernel / 3.0
    glow_kernel = torch.exp(-((torch.arange(-blur_kernel//2+1, blur_kernel//2+2, device=device).view(-1,1))**2)/ (2*sigma**2))
    glow_kernel = glow_kernel / glow_kernel.sum()
    glow_kernel = glow_kernel.view(1,1,blur_kernel,1).repeat(C,1,1,1)
    glow = F.conv2d(latents, glow_kernel, padding=(blur_kernel//2,0), groups=C)
    glow = F.conv2d(glow, glow_kernel.transpose(2,3), padding=(0,blur_kernel//2), groups=C)  # convolution 2D approximative

    # 4️⃣ Fusion glow sur iris seulement
    latents_out = latents * (1 - mask) + glow * mask * strength
    latents_out = latents_out.clamp(-1.0, 1.0)

    return latents_out

def apply_intelligent_glow_froid_latents(latents, strength=0.2, blur_kernel=7):
    """
    Applique un effet "glow froid" directement sur des latents [B, C, H, W].

    Args:
        latents (torch.Tensor): Latents [B,C,H,W].
        strength (float): Intensité du glow (0.0 à 1.0).
        blur_kernel (int): Taille du noyau pour le flou gaussien (doit être impair).

    Returns:
        torch.Tensor: Latents avec glow appliqué.
    """
    if latents.ndim != 4:
        raise ValueError("Latents doivent être de shape [B, C, H, W]")

    B, C, H, W = latents.shape

    # 🔹 Création noyau gaussien 2D
    def gaussian_kernel(kernel_size, sigma, channels):
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=latents.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        return kernel

    sigma = blur_kernel / 3.0
    kernel = gaussian_kernel(blur_kernel, sigma, C).to(latents.device, latents.dtype)

    padding = blur_kernel // 2
    # 🔹 Appliquer convolution pour obtenir le glow
    glow = F.conv2d(latents, kernel, padding=padding, groups=C)

    # 🔹 Fusion latents original + glow
    latents_out = latents * (1 - strength) + glow * strength

    # 🔹 Clamp pour stabilité
    latents_out = latents_out.clamp(-1.0, 1.0)

    return latents_out


# Appplication effect sur les iris yeux:
def apply_glow_froid_iris(latents, eye_coords, iris_radius_ratio=0.08, strength=0.2, blur_kernel=7):
    """
    Applique un glow froid uniquement sur l'iris des yeux dans les latents [B,C,H,W].

    Args:
        latents (torch.Tensor): Latents SD [B,C,H,W]
        eye_coords (list of tuples): Coordonnées des yeux [(x1,y1),(x2,y2)]
        iris_radius_ratio (float): proportion de H/W pour rayon iris
        strength (float): intensité du glow
        blur_kernel (int): taille du kernel gaussien (impair)

    Returns:
        torch.Tensor: latents avec glow sur iris
    """
    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ Créer masque radial pour l’iris
    iris_mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    for i, (x, y) in enumerate(eye_coords):
        rx = int(W * iris_radius_ratio)
        ry = int(H * iris_radius_ratio)
        # coordonnées grille
        Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        dist2 = ((X - x)**2) / (rx**2) + ((Y - y)**2) / (ry**2)
        iris_mask[0, 0] += (dist2 <= 1).float()
    iris_mask = iris_mask.clamp(0, 1)  # éviter >1 si deux yeux se chevauchent

    # 2️⃣ Créer kernel gaussien 2D
    sigma = blur_kernel / 3
    ax = torch.arange(-blur_kernel // 2 + 1., blur_kernel // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel_2d = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, blur_kernel, blur_kernel).repeat(C, 1, 1, 1)  # [C,1,kH,kW]

    # 3️⃣ Appliquer convolution channel-wise
    glow = F.conv2d(latents * iris_mask, kernel, padding=blur_kernel // 2, groups=C)

    # 4️⃣ Fusion glow sur iris uniquement
    latents_out = latents * (1 - iris_mask) + glow * iris_mask * strength
    latents_out = latents_out.clamp(-1.0, 1.0)

    return latents_out


#----------- Rendu HD ------------------------------
# version optimized
#--------------------------------------------------
def apply_pro_net_volumetrique(
    latents,
    coords_v=None,
    n3r_pro_net=None,
    n3r_pro_strength=0.3,
    sanitize_fn=None,
    volume_strength=0.03,     # relief global doux
    shadow_strength=0.06,     # ombres (clé anime)
    highlight_strength=0.01, # lumière très contrôlée
    iris_light=0.02,          # glow iris doux
    iris_radius_ratio=0.04,
    mask_blur_kernel=13,
    debug=False
):
    """
    Version anime volumétrique :
    - low frequency uniquement (pas de bruit)
    - ombres renforcées (style anime)
    - lumière douce (jamais cramée)
    - iris propre (pas de sharp)
    """

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ ProNet (léger)
    if n3r_pro_net is not None:
        with torch.no_grad():
            latents_prot = apply_n3r_pro_net(
                latents,
                model=n3r_pro_net,
                strength=n3r_pro_strength,
                sanitize_fn=sanitize_fn
            ).to(dtype)
    else:
        latents_prot = latents

    # 2️⃣ LOW FREQUENCY = base anime propre
    smooth = F.avg_pool2d(latents_prot, 5, stride=1, padding=2)

    # 3️⃣ Volume (relief global, sans bruit)
    volume = (latents_prot - smooth) * volume_strength

    # 4️⃣ Ombres (effet 3D anime)
    shadows = torch.relu(smooth - latents_prot) * shadow_strength

    # 5️⃣ Lumière douce (jamais cramée)
    highlights = torch.relu(latents_prot - smooth) * highlight_strength

    latents_3D = latents_prot + volume + shadows + highlights

    # 6️⃣ Iris (optionnel, ultra clean)
    if coords_v:
        Y, X = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        iris_mask = torch.zeros((1,1,H,W), device=device, dtype=dtype)

        for x, y in coords_v:
            rx = max(1, int(W * iris_radius_ratio))
            ry = max(1, int(H * iris_radius_ratio))
            dist = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
            iris_mask[0,0] += (dist <= 1).float()

        iris_mask = iris_mask.clamp(0,1)

        # flou large = pas de contour paupière
        if mask_blur_kernel > 1:
            iris_mask = F.avg_pool2d(
                iris_mask,
                kernel_size=mask_blur_kernel,
                stride=1,
                padding=mask_blur_kernel // 2
            ).clamp(0,1)

        # glow très propre basé sur low-freq
        iris_glow = torch.relu(latents_prot - smooth) * iris_light

        latents_3D = latents_3D * (1 - iris_mask) + (latents_3D + iris_glow) * iris_mask

    # 7️⃣ Clamp final (sécurité)
    latents_out = latents_3D.clamp(-1.0, 1.0)

    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(latents_prot[0,0].detach().cpu(), cmap='gray'); plt.title("ProNet")
        plt.subplot(1,3,2); plt.imshow(smooth[0,0].detach().cpu(), cmap='gray'); plt.title("Low-Freq (Anime)")
        if coords_v:
            plt.subplot(1,3,3); plt.imshow(iris_mask[0,0].detach().cpu(), cmap='Reds'); plt.title("Iris Mask")
        plt.tight_layout(); plt.show()
    print("🎨 Anime volumetric applied")

    return latents_out

def apply_pro_net_volumetrique_high(
    latents,
    coords_v=None,                # optionnel pour iris
    n3r_pro_net=None,
    n3r_pro_strength=0.5,
    sanitize_fn=None,
    glow_strength=0.01,           # glow très doux
    volume_strength=0.03,         # amplification du relief général
    blur_kernel=3,
    mask_blur_kernel=5,
    contrast=1.05,
    shadows_boost=0.05,           # léger boost des ombres
    highlights_boost=0.005,        # léger boost des zones lumineuses
    debug=False
):
    """
    Amplification 3D douce pour latents : relief, volumes et glow subtil sur iris.
    """

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ ProNet inference
    if n3r_pro_net is not None:
        with torch.no_grad():
            latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn).to(dtype)
    else:
        latents_prot = latents.clone()

    # 2️⃣ Calcul du high-frequency (texture / relief)
    if blur_kernel > 1:
        sigma = blur_kernel / 3
        ax = torch.arange(-blur_kernel//2+1., blur_kernel//2+1., device=device, dtype=dtype)
        g1d = torch.exp(-(ax**2)/(2*sigma**2))
        g1d /= g1d.sum()
        kx = g1d.view(1,1,1,blur_kernel).repeat(C,1,1,1)
        ky = g1d.view(1,1,blur_kernel,1).repeat(C,1,1,1)
        blurred = F.conv2d(F.conv2d(latents_prot, kx, padding=(0, blur_kernel//2), groups=C),
                           ky, padding=(blur_kernel//2,0), groups=C)
        high_freq = latents_prot - blurred
    else:
        high_freq = torch.zeros_like(latents_prot)

    # 3️⃣ Amplification du relief général
    relief_map = high_freq * volume_strength

    # 4️⃣ Amplification subtile des ombres et lumières
    shadows = torch.clamp(latents_prot, min=-1.0, max=0.0) * shadows_boost
    highlights = torch.clamp(latents_prot, min=0.0, max=1.0) * highlights_boost

    latents_3D = latents_prot + relief_map + shadows + highlights

    # 5️⃣ Glow léger sur iris si coords_v fournis
    if coords_v:
        Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        iris_mask = torch.zeros((1,1,H,W), device=device, dtype=dtype)
        for x, y in coords_v:
            rx = max(1, int(W * 0.05))
            ry = max(1, int(H * 0.05))
            dist2 = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
            iris_mask[0,0] += (dist2 <= 1).float()
        iris_mask = iris_mask.clamp(0,1)
        if mask_blur_kernel > 1:
            sigma = mask_blur_kernel / 3
            ax = torch.arange(-mask_blur_kernel//2+1., mask_blur_kernel//2+1., device=device, dtype=dtype)
            g1d = torch.exp(-(ax**2)/(2*sigma**2))
            g1d /= g1d.sum()
            kx = g1d.view(1,1,1,mask_blur_kernel)
            ky = g1d.view(1,1,mask_blur_kernel,1)
            iris_mask = F.conv2d(F.conv2d(iris_mask, kx, padding=(0, mask_blur_kernel//2)),
                                 ky, padding=(mask_blur_kernel//2,0)).clamp(0,1)
        latents_3D = latents_3D * (1 - iris_mask) + (latents_3D + glow_strength*high_freq) * iris_mask

    # 6️⃣ Contraste léger sur l’ensemble
    latents_mean = latents_3D.mean(dim=[2,3], keepdim=True)
    latents_3D = (latents_3D - latents_mean) * contrast + latents_mean

    # 7️⃣ Clamp final
    latents_out = latents_3D.clamp(-1.0,1.0)

    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(latents_prot[0,0].detach().cpu(), cmap='gray'); plt.title("ProNet")
        plt.subplot(1,3,2); plt.imshow(high_freq[0,0].detach().cpu(), cmap='gray'); plt.title("High-Freq / Relief")
        if coords_v:
            plt.subplot(1,3,3); plt.imshow(iris_mask[0,0].detach().cpu(), cmap='Reds', alpha=0.5); plt.title("Iris Mask Glow")
        plt.tight_layout(); plt.show()
        print("👁 Relief 3D + glow iris appliqué")

    return latents_out

def apply_pro_net_volumetrique_natural(
    latents,
    coords_v,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    glow_strength=0.03,        # glow très doux
    blur_kernel=3,
    iris_radius_ratio=0.05,
    mask_blur_kernel=5,
    gamma=1.0,                 # quasi pas de gamma
    contrast=1.05,
    debug=False
):
    """
    ProNet volumétrique HD minimal + glow iris très subtil.
    Effet léger, contours doux, détails iris subtils.
    """


    if not coords_v:
        with torch.no_grad():
            return apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ ProNet inference
    with torch.no_grad():
        latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn).to(dtype)

    # 2️⃣ Masque iris vectorisé
    Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    iris_mask = torch.zeros((1,1,H,W), device=device, dtype=dtype)
    for x, y in coords_v:
        rx = max(1, int(W * iris_radius_ratio))
        ry = max(1, int(H * iris_radius_ratio))
        dist2 = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
        iris_mask[0,0] += (dist2 <= 1).float()
    iris_mask = iris_mask.clamp(0,1)

    # 3️⃣ Flou du masque léger pour contours très doux
    if mask_blur_kernel > 1:
        sigma = mask_blur_kernel / 3
        ax = torch.arange(-mask_blur_kernel//2+1., mask_blur_kernel//2+1., device=device, dtype=dtype)
        g1d = torch.exp(-(ax**2)/(2*sigma**2))
        g1d /= g1d.sum()
        kx = g1d.view(1,1,1,mask_blur_kernel)
        ky = g1d.view(1,1,mask_blur_kernel,1)
        iris_mask = F.conv2d(F.conv2d(iris_mask, kx, padding=(0, mask_blur_kernel//2)),
                             ky, padding=(mask_blur_kernel//2,0)).clamp(0,1)

    # 4️⃣ High-frequency léger pour glow iris
    if blur_kernel > 1:
        sigma = blur_kernel / 3
        ax = torch.arange(-blur_kernel//2+1., blur_kernel//2+1., device=device, dtype=dtype)
        g1d = torch.exp(-(ax**2)/(2*sigma**2))
        g1d /= g1d.sum()
        kx = g1d.view(1,1,1,blur_kernel).repeat(C,1,1,1)
        ky = g1d.view(1,1,blur_kernel,1).repeat(C,1,1,1)
        blurred = F.conv2d(F.conv2d(latents_prot, kx, padding=(0, blur_kernel//2), groups=C),
                           ky, padding=(blur_kernel//2,0), groups=C)
        high_freq = latents_prot - blurred
    else:
        high_freq = torch.zeros_like(latents_prot)

    # 5️⃣ Glow très subtil sur iris
    latents_out = latents_prot + glow_strength * high_freq * iris_mask

    # 6️⃣ Contraste léger sur iris uniquement
    iris_pixels = latents_out * iris_mask
    iris_pixels = (iris_pixels - iris_pixels.mean()) * contrast + iris_pixels.mean()
    latents_out = latents_out * (1 - iris_mask) + iris_pixels

    # 7️⃣ Clamp final
    latents_out = latents_out.clamp(-1.0,1.0)

    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(latents_prot[0,0].detach().cpu(), cmap='gray'); plt.title("ProNet")
        plt.subplot(1,3,2); plt.imshow(high_freq[0,0].detach().cpu(), cmap='gray'); plt.title("High-Freq")
        plt.subplot(1,3,3); plt.imshow(iris_mask[0,0].detach().cpu(), cmap='Reds', alpha=0.5); plt.title("Mask Iris")
        plt.tight_layout(); plt.show()
        print("👁 Glow iris très subtil appliqué")

    return latents_out

def apply_pro_net_volumetrique_clean(
    latents,
    coords_v,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    glow_strength=0.05,        # glow doux
    blur_kernel=3,            # détails HD
    iris_radius_ratio=0.06,
    mask_blur_kernel=5,       # flou doux du masque
    gamma=0.85,               # léger boost gamma
    contrast=1.1,             # léger boost contraste
    debug=False
):
    """
    ProNet volumétrique HD optimisé + glow iris doux + gamma/contrast
    """

    if not coords_v:
        with torch.no_grad():
            return apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ ProNet avec no_grad pour économiser VRAM
    with torch.no_grad():
        latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn).to(dtype)

    # 2️⃣ Masque iris vectorisé
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    iris_mask = torch.zeros((1,1,H,W), device=device, dtype=dtype)
    for x, y in coords_v:
        rx = max(1, int(W * iris_radius_ratio))
        ry = max(1, int(H * iris_radius_ratio))
        dist2 = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
        iris_mask[0,0] += (dist2 <= 1).float()
    iris_mask = iris_mask.clamp(0,1)

    # 3️⃣ Flou léger du masque (conv 1D séparables)
    if mask_blur_kernel > 1:
        sigma = mask_blur_kernel / 3
        ax = torch.arange(-mask_blur_kernel//2 + 1., mask_blur_kernel//2 + 1., device=device, dtype=dtype)
        gauss_1d = torch.exp(-(ax**2)/(2*sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_x = gauss_1d.view(1,1,1,mask_blur_kernel)
        kernel_y = gauss_1d.view(1,1,mask_blur_kernel,1)
        iris_mask = F.conv2d(F.conv2d(iris_mask, kernel_x, padding=(0, mask_blur_kernel//2)),
                             kernel_y, padding=(mask_blur_kernel//2,0))
        iris_mask = iris_mask.clamp(0,1)

    # 4️⃣ High-frequency details (blur separable)
    if blur_kernel > 1:
        sigma = blur_kernel / 3
        ax = torch.arange(-blur_kernel//2 + 1., blur_kernel//2 + 1., device=device, dtype=dtype)
        gauss_1d = torch.exp(-(ax**2)/(2*sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_x = gauss_1d.view(1,1,1,blur_kernel).repeat(C,1,1,1)
        kernel_y = gauss_1d.view(1,1,blur_kernel,1).repeat(C,1,1,1)
        blurred = F.conv2d(F.conv2d(latents_prot, kernel_x, padding=(0, blur_kernel//2), groups=C),
                           kernel_y, padding=(blur_kernel//2,0), groups=C)
        high_freq = latents_prot - blurred
    else:
        high_freq = torch.zeros_like(latents_prot)

    # 5️⃣ Glow adaptatif sur iris (soft)
    glow = torch.tanh(glow_strength * high_freq * iris_mask)
    latents_out = latents_prot + glow

    # 6️⃣ Boost gamma & contrast sur iris uniquement
    iris_pixels = latents_out * iris_mask
    iris_pixels = torch.sign(iris_pixels) * torch.pow(torch.abs(iris_pixels), gamma)  # gamma soft
    iris_pixels = (iris_pixels - iris_pixels.mean()) * contrast + iris_pixels.mean()   # contrast soft
    latents_out = latents_out * (1 - iris_mask) + iris_pixels

    # 7️⃣ Clamp final pour éviter extrêmes
    latents_out = latents_out.clamp(-1.0, 1.0)

    # 8️⃣ Debug
    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(latents_prot[0,0].detach().cpu(), cmap='gray'); plt.title("ProNet")
        plt.subplot(1,3,2); plt.imshow(high_freq[0,0].detach().cpu(), cmap='gray'); plt.title("High-Freq")
        plt.subplot(1,3,3); plt.imshow(iris_mask[0,0].detach().cpu(), cmap='Reds', alpha=0.5); plt.title("Mask Iris")
        plt.tight_layout(); plt.show()
        print("👁 DEBUG HD soft + gamma/contrast appliqué")

    return latents_out

def apply_pro_net_volumetrique_ice(
    latents,
    coords_v,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    glow_strength=0.1,       # glow plus doux
    blur_kernel=3,           # détails HD
    iris_radius_ratio=0.08,
    mask_blur_kernel=3,      # flou doux du masque
    debug=False
):
    """
    ProNet volumétrique HD optimisé + glow iris doux
    """

    if not coords_v:
        with torch.no_grad():
            return apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ ProNet avec no_grad pour économiser GPU
    with torch.no_grad():
        latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn).to(dtype)

    # 2️⃣ Masque iris vectorisé
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    iris_mask = torch.zeros((1,1,H,W), device=device, dtype=dtype)
    for x, y in coords_v:
        rx = max(1, int(W * iris_radius_ratio))
        ry = max(1, int(H * iris_radius_ratio))
        dist2 = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
        iris_mask[0,0] += (dist2 <= 1).float()
    iris_mask = iris_mask.clamp(0,1)

    # 3️⃣ Flou léger du masque avec conv séparables 1D
    if mask_blur_kernel > 1:
        sigma = mask_blur_kernel / 3
        ax = torch.arange(-mask_blur_kernel//2 + 1., mask_blur_kernel//2 + 1., device=device, dtype=dtype)
        gauss_1d = torch.exp(-(ax**2)/(2*sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_x = gauss_1d.view(1,1,1,mask_blur_kernel)
        kernel_y = gauss_1d.view(1,1,mask_blur_kernel,1)
        iris_mask = F.conv2d(F.conv2d(iris_mask, kernel_x, padding=(0, mask_blur_kernel//2)),
                             kernel_y, padding=(mask_blur_kernel//2,0))
        iris_mask = iris_mask.clamp(0,1)

    # 4️⃣ High-frequency details (blur separable)
    if blur_kernel > 1:
        sigma = blur_kernel / 3
        ax = torch.arange(-blur_kernel//2 + 1., blur_kernel//2 + 1., device=device, dtype=dtype)
        gauss_1d = torch.exp(-(ax**2)/(2*sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_x = gauss_1d.view(1,1,1,blur_kernel).repeat(C,1,1,1)
        kernel_y = gauss_1d.view(1,1,blur_kernel,1).repeat(C,1,1,1)
        blurred = F.conv2d(F.conv2d(latents_prot, kernel_x, padding=(0, blur_kernel//2), groups=C),
                           kernel_y, padding=(blur_kernel//2,0), groups=C)
        high_freq = latents_prot - blurred
    else:
        high_freq = torch.zeros_like(latents_prot)

    # 5️⃣ Glow adaptatif sur iris (effet plus doux et contrôlé)
    latents_out = latents_prot + torch.tanh(glow_strength * high_freq * iris_mask)
    latents_out = latents_out.clamp(-1.0,1.0)

    # 6️⃣ Debug
    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(latents_prot[0,0].detach().cpu(), cmap='gray'); plt.title("ProNet")
        plt.subplot(1,3,2); plt.imshow(high_freq[0,0].detach().cpu(), cmap='gray'); plt.title("High-Freq")
        plt.subplot(1,3,3); plt.imshow(iris_mask[0,0].detach().cpu(), cmap='Reds', alpha=0.5); plt.title("Mask Iris")
        plt.tight_layout(); plt.show()
        print("👁 DEBUG HD soft appliqué")

    return latents_out

def apply_pro_net_volumetrique_glow(
    latents,
    coords_v,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    glow_strength=0.2,
    blur_kernel=3,          # plus petit = détails plus nets
    iris_radius_ratio=0.08,
    mask_blur_kernel=1,     # très léger flou du masque
    debug=False
):
    """
    ProNet volumétrique HD + glow iris avec contours adoucis mais plus net
    """

    if not coords_v:
        return apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ ProNet
    latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn).to(dtype)

    # 2️⃣ Masque iris
    iris_mask = torch.zeros((B,1,H,W), device=device, dtype=dtype)
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    for x, y in coords_v:
        rx = max(1, int(W * iris_radius_ratio))
        ry = max(1, int(H * iris_radius_ratio))
        dist2 = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
        iris_mask[0,0] += (dist2 <= 1).float()
    iris_mask = iris_mask.clamp(0,1)

    # 3️⃣ Léger flou du masque seulement
    if mask_blur_kernel > 1:
        sigma = mask_blur_kernel / 3
        ax = torch.arange(-mask_blur_kernel//2 + 1., mask_blur_kernel//2 + 1., device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        mask_kernel = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        mask_kernel = mask_kernel / mask_kernel.sum()
        mask_kernel = mask_kernel.view(1,1,mask_blur_kernel,mask_blur_kernel)
        iris_mask = F.conv2d(iris_mask, mask_kernel, padding=mask_blur_kernel//2)
        iris_mask = iris_mask.clamp(0,1)

    # 4️⃣ Détails HD (high-frequency)
    if blur_kernel > 1:
        sigma = blur_kernel / 3
        ax = torch.arange(-blur_kernel//2 + 1., blur_kernel//2 + 1., device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel_2d = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.view(1,1,blur_kernel,blur_kernel).repeat(C,1,1,1).to(dtype)
        blurred = F.conv2d(latents_prot, kernel, padding=blur_kernel//2, groups=C)
        high_freq = latents_prot - blurred
    else:
        high_freq = latents_prot - latents_prot  # pas de flou → pas de high_freq

    # 5️⃣ Glow adaptatif seulement sur iris
    latents_out = latents_prot + glow_strength * high_freq * iris_mask
    latents_out = latents_out.clamp(-1.0,1.0)

    # 6️⃣ Debug
    if debug:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.imshow(latents_prot[0,0].detach().cpu(), cmap='gray'); plt.title("ProNet")
        plt.subplot(1,3,2); plt.imshow(high_freq[0,0].detach().cpu(), cmap='gray'); plt.title("High-Freq")
        plt.subplot(1,3,3); plt.imshow(iris_mask[0,0].detach().cpu(), cmap='Reds', alpha=0.5); plt.title("Mask Iris")
        plt.tight_layout(); plt.show()
        print("👁 DEBUG HD sharp appliqué")

    return latents_out

def apply_pro_net_volumetrique_good(
    latents,
    coords_v,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    glow_strength=0.2,
    blur_kernel=7,
    iris_radius_ratio=0.08,
    debug=False
):
    """
    Applique ProNet et un effet "HDR / détail" sur les iris des yeux,
    compatible FP16 / latents interpolés.

    Args:
        latents (torch.Tensor): [B,C,H,W] Latents à traiter.
        coords_v (list of tuples): Coordonnées yeux [(x1,y1),(x2,y2)].
        n3r_pro_net: modèle ProNet
        n3r_pro_strength (float): force ProNet
        sanitize_fn: fonction de nettoyage latents
        glow_strength (float): intensité du glow / amplification
        blur_kernel (int): taille du kernel pour flou
        iris_radius_ratio (float): proportion de H/W pour rayon iris
        debug (bool): visualisation mask + latents

    Returns:
        torch.Tensor: latents avec effet HDR sur iris uniquement
    """
    if not coords_v:
        # Aucun yeux détectés → ProNet seul
        return apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ Appliquer ProNet
    latents_prot = apply_n3r_pro_net(
        latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn
    )

    # 2️⃣ Créer masque iris
    iris_mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    for x, y in coords_v:
        rx = int(W * iris_radius_ratio)
        ry = int(H * iris_radius_ratio)
        dist2 = ((X - x)**2)/(rx**2) + ((Y - y)**2)/(ry**2)
        iris_mask[0, 0] += (dist2 <= 1).float()
    iris_mask = iris_mask.clamp(0, 1)

    # 3️⃣ Kernel gaussien, même dtype que latents (FP16 ok)
    sigma = blur_kernel / 3
    ax = torch.arange(-blur_kernel // 2 + 1., blur_kernel // 2 + 1., device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel_2d = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, blur_kernel, blur_kernel).repeat(C, 1, 1, 1)

    # 4️⃣ Convolution channel-wise → amplification détails iris
    glow = F.conv2d(latents_prot * iris_mask, kernel, padding=blur_kernel // 2, groups=C)

    # 5️⃣ Fusion ProNet + iris glow
    latents_out = latents_prot * (1 - iris_mask) + glow * iris_mask * glow_strength
    latents_out = latents_out.clamp(-1.0, 1.0)

    # ---------------- DEBUG ----------------
    if debug:
        lat_vis = latents[0, 0].detach().cpu()
        prot_vis = latents_prot[0, 0].detach().cpu()
        glow_vis = glow[0, 0].detach().cpu()
        mask_vis = iris_mask[0, 0].detach().cpu()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(lat_vis, cmap='gray')
        plt.title("Latent original")
        plt.subplot(1, 4, 2)
        plt.imshow(prot_vis, cmap='gray')
        plt.title("ProNet")
        plt.subplot(1, 4, 3)
        plt.imshow(glow_vis, cmap='gray')
        plt.title("HDR / Glow Iris")
        plt.subplot(1, 4, 4)
        plt.imshow(lat_vis, cmap='gray', alpha=0.7)
        plt.imshow(mask_vis, cmap='Reds', alpha=0.4)
        plt.title("Mask Iris")
        plt.tight_layout()
        plt.show()
        print("👁 DEBUG activé → vérifie position / taille iris")

    return latents_out

#----- Amplification des détails des yeux - version optimisé
def apply_pro_net_with_eyes_v2(
    latents,
    eye_coords,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    iris_radius_ratio=0.04,
    mask_blur_kernel=13,
    shade_strength=0.04,
    light_strength=0.025,
    blend_strength=0.6,
    preserve_detail=0.5,
    train=False,
    frame_counter=0,
    ema_prev_latents=None,
    ema_alpha=0.3,
    debug=False
):
    """
    Eyes enhancement module compatible Stable Diffusion latent space.

    Logique:
    1. Détection de bruit (EMA + high freq)
    2. Option training dynamique (future extension model-based)
    3. ProNet base enhancement
    4. Iris mask soft spatial weighting
    5. Detail injection (high freq preservation)
    6. EMA temporal smoothing
    """

    if latents is None:
        raise ValueError("latents is None")

    device = latents.device

    # =========================================================
    # 🔹 1. Noise estimation (comme ton denoise pipeline)
    # =========================================================
    if ema_prev_latents is not None:
        diff = latents - ema_prev_latents
        noise_level = diff.std()
        is_noisy = noise_level > 0.08
    else:
        high_freq = latents - F.avg_pool2d(latents, 3, 1, 1)
        noise_level = high_freq.std()
        is_noisy = True

    if debug:
        print(f"[Eyes] noise_level={noise_level:.5f}, is_noisy={is_noisy}")

    # =========================================================
    # 🔹 2. ProNet base enhancement (global latent refinement)
    # =========================================================
    with torch.no_grad():
        latents_base = n3r_pro_net(
            latents,
            strength=n3r_pro_strength
        )

    latents_base = sanitize_fn(latents_base)

    # =========================================================
    # 🔹 3. Iris mask (soft spatial attention)
    # =========================================================
    B, C, H, W = latents.shape

    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    iris_mask = torch.zeros((1, 1, H, W), device=device, dtype=latents.dtype)

    for (x, y) in eye_coords:
        rx = max(1, int(W * iris_radius_ratio))
        ry = max(1, int(H * iris_radius_ratio))

        dist = ((X - x) ** 2) / (rx ** 2 + 1e-6) + ((Y - y) ** 2) / (ry ** 2 + 1e-6)
        iris_mask += (dist <= 1).float()

    iris_mask = iris_mask.clamp(0, 1)

    if mask_blur_kernel > 1:
        iris_mask = F.avg_pool2d(
            iris_mask,
            kernel_size=mask_blur_kernel,
            stride=1,
            padding=mask_blur_kernel // 2
        ).clamp(0, 1)

    # =========================================================
    # 🔹 4. Low / High frequency separation
    # =========================================================
    smooth = F.avg_pool2d(latents_base, 5, 1, 2)

    shadow = (smooth - latents_base) * shade_strength
    light = F.relu(latents_base - smooth) * light_strength

    iris_effect = shadow + light

    # =========================================================
    # 🔹 5. Non-linear stabilization (avoid oversharpen)
    # =========================================================
    iris_effect = torch.tanh(iris_effect * 2.0) * 0.5

    # =========================================================
    # 🔹 6. Detail preservation (eyes realism)
    # =========================================================
    high_freq = latents_base - smooth
    iris_effect = (
        iris_effect * (1.0 - preserve_detail)
        + high_freq * preserve_detail * 0.35
    )

    # =========================================================
    # 🔹 7. Adaptive blending mask
    # =========================================================
    blend_mask = iris_mask * blend_strength

    if is_noisy:
        blend_mask = blend_mask * 1.2  # stronger correction on noisy frames

    latents_out = (
        latents_base * (1.0 - blend_mask)
        + (latents_base + iris_effect) * blend_mask
    )

    # =========================================================
    # 🔹 8. Safe normalization (Stable Diffusion friendly)
    # =========================================================
    latents_out = torch.tanh(latents_out).clamp(-1.0, 1.0)
    latents_out = sanitize_fn(latents_out)

    # =========================================================
    # 🔹 9. Adaptive injection (same philosophy as denoise pipeline)
    # =========================================================
    if is_noisy:
        strength = min(0.05 + 0.25 * noise_level, 0.8)
    else:
        strength = min(0.05 + 0.1 * noise_level, 0.2)

    latents = latents + strength * (latents_out - latents)

    if debug:
        print(f"[Eyes] strength={strength:.4f}")

    # =========================================================
    # 🔹 10. Temporal EMA smoothing
    # =========================================================
    if ema_prev_latents is not None:
        ema_prev_latents = ema_prev_latents.to(device)
        latents = ema_alpha * latents + (1 - ema_alpha) * ema_prev_latents

    # =========================================================
    # 🔹 11. Final diagnostics
    # =========================================================
    if debug:
        hf = latents - F.avg_pool2d(latents, 3, 1, 1)
        print(f"[Eyes] final_std={latents.std():.5f} hf_std={hf.std():.5f}")

    return latents

def apply_pro_net_with_eyes(
    latents,
    eye_coords,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    iris_radius_ratio=0.04,
    mask_blur_kernel=13,
    shade_strength=0.04,
    light_strength=0.025,
    blend_strength=0.6,         # 🔥 nouveau
    preserve_detail=0.5         # 🔥 nouveau
):


    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # =========================================================
    # 🔹 1. ProNet (base)
    # =========================================================
    with torch.no_grad():
        latents_prot = apply_n3r_pro_net(
            latents,
            model=n3r_pro_net,
            strength=n3r_pro_strength,
            sanitize_fn=sanitize_fn
        ).to(dtype)

    if not eye_coords:
        return latents_prot

    # =========================================================
    # 🔹 2. Iris mask (soft ellipse)
    # =========================================================
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    iris_mask = torch.zeros((1,1,H,W), device=device, dtype=dtype)

    for x, y in eye_coords:
        rx = max(1, int(W * iris_radius_ratio))
        ry = max(1, int(H * iris_radius_ratio))

        dist = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
        iris_mask[0,0] += (dist <= 1).float()

    iris_mask = iris_mask.clamp(0,1)

    if mask_blur_kernel > 1:
        iris_mask = F.avg_pool2d(
            iris_mask,
            kernel_size=mask_blur_kernel,
            stride=1,
            padding=mask_blur_kernel // 2
        ).clamp(0,1)

    # =========================================================
    # 🔹 3. Base smooth (low freq)
    # =========================================================
    smooth = F.avg_pool2d(latents_prot, 5, stride=1, padding=2)

    # =========================================================
    # 🔹 4. Volume (shadow + light)
    # =========================================================
    shadow = (smooth - latents_prot) * shade_strength
    light  = torch.relu(latents_prot - smooth) * light_strength

    iris_effect = shadow + light

    # =========================================================
    # 🔹 5. 🔥 NON-LINEAR COMPRESSION (clé)
    # évite saturation / écrasement
    # =========================================================
    iris_effect = torch.tanh(iris_effect * 2.0) * 0.5

    # =========================================================
    # 🔹 6. 🔥 DETAIL PRESERVATION
    # garde les hautes fréquences originales
    # =========================================================
    high_freq = latents_prot - smooth
    iris_effect = iris_effect * (1.0 - preserve_detail) + high_freq * preserve_detail * 0.3

    # =========================================================
    # 🔹 7. 🔥 BLEND ADAPTATIF (AU LIEU DE ADD)
    # =========================================================
    blend_mask = iris_mask * blend_strength

    latents_out = (
        latents_prot * (1.0 - blend_mask) +
        (latents_prot + iris_effect) * blend_mask
    )

    # =========================================================
    # 🔹 8. 🔥 SOFT GATING (évite zones plates)
    # =========================================================
    contrast = torch.abs(high_freq)
    gate = torch.sigmoid(contrast * 10.0)

    latents_out = latents_prot * (1 - gate) + latents_out * gate

    print("👁 Fusion iris perceptuelle (soft blend, no crushing)")

    return latents_out.clamp(-1.0, 1.0)

def apply_pro_net_with_eyes_boost(
    latents,
    eye_coords,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    iris_radius_ratio=0.04,
    mask_blur_kernel=13,
    shade_strength=0.04,   # profondeur (ombres)
    light_strength=0.025   # lumière douce
):
    """
    Version anime :
    - aucun sharp
    - aucun bruit
    - volume doux (ombre + lumière)
    - iris lisse et propre
    """

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ ProNet (léger uniquement)
    with torch.no_grad():
        latents_prot = apply_n3r_pro_net(
            latents,
            model=n3r_pro_net,
            strength=n3r_pro_strength,
            sanitize_fn=sanitize_fn
        ).to(dtype)

    if not eye_coords:
        return latents_prot

    # 2️⃣ Masque iris (ellipse douce)
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    iris_mask = torch.zeros((1,1,H,W), device=device, dtype=dtype)

    for x, y in eye_coords:
        rx = max(1, int(W * iris_radius_ratio))
        ry = max(1, int(H * iris_radius_ratio))

        dist = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
        iris_mask[0,0] += (dist <= 1).float()

    iris_mask = iris_mask.clamp(0,1)

    # 3️⃣ Flou très large → style anime
    if mask_blur_kernel > 1:
        iris_mask = F.avg_pool2d(
            iris_mask,
            kernel_size=mask_blur_kernel,
            stride=1,
            padding=mask_blur_kernel // 2
        ).clamp(0,1)

    # 4️⃣ Base lissée (anime = surfaces propres)
    smooth = F.avg_pool2d(latents_prot, 5, stride=1, padding=2)

    # 5️⃣ Volume (ombres)
    shadow = (smooth - latents_prot) * shade_strength

    # 6️⃣ Lumière douce (pas de brûlé)
    light = torch.relu(latents_prot - smooth) * light_strength

    # 7️⃣ Fusion anime (très stable)
    iris_effect = shadow + light

    latents_out = latents_prot + iris_effect * iris_mask

    print("👁 HDR détails appliqué sur iris avec contours adoucis")

    return latents_out.clamp(-1.0, 1.0)



#---------- Mask bouche -------
def apply_pro_net_with_mouth(
    latents,
    mouth_coords,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    detail_strength=0.35,
    blur_kernel=5,
    mouth_radius_ratio=0.08,
    mask_blur_kernel=7,
    fusion_strength=0.35,   # 🔥 NOUVEAU (clé)
    preserve_base=0.6       # 🔥 NOUVEAU (anti écrasement)
):


    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # =========================================================
    # 🔹 1. ProNet (base HD)
    # =========================================================
    latents_prot = apply_n3r_pro_net(
        latents,
        model=n3r_pro_net,
        strength=n3r_pro_strength,
        sanitize_fn=sanitize_fn
    ).to(dtype)

    if not mouth_coords:
        return latents_prot

    # =========================================================
    # 🔹 2. MASK (inchangé mais plus smooth)
    # =========================================================
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    mouth_mask = torch.zeros((B,1,H,W), device=device, dtype=dtype)

    for x, y in mouth_coords:
        rx = int(W * mouth_radius_ratio)
        ry = int(H * mouth_radius_ratio)

        dist = ((X - x)**2)/(rx**2+1e-6) + ((Y - y)**2)/(ry**2+1e-6)
        mouth_mask += (dist <= 1).float()

    mouth_mask = mouth_mask.clamp(0,1)

    if mask_blur_kernel > 1:
        mouth_mask = F.avg_pool2d(
            mouth_mask,
            kernel_size=mask_blur_kernel,
            stride=1,
            padding=mask_blur_kernel//2
        ).clamp(0,1)

    # =========================================================
    # 🔹 3. EXTRACTION DETAILS (HF uniquement)
    # =========================================================
    blurred = F.avg_pool2d(latents_prot, blur_kernel, stride=1, padding=blur_kernel//2)
    details = latents_prot - blurred

    # 👉 compression (évite HDR violent)
    details = torch.tanh(details * 2.0) * 0.5

    # =========================================================
    # 🔹 4. GATE ADAPTATIF
    # =========================================================
    detail_energy = details.abs().mean(dim=1, keepdim=True)
    gate = torch.sigmoid(detail_energy * 10.0)  # focus zones utiles

    # =========================================================
    # 🔹 5. RESIDUAL FUSION (non destructive)
    # =========================================================
    weight = mouth_mask * gate * fusion_strength

    latents_out = latents + details * weight

    # =========================================================
    # 🔹 6. PRESERVE BASE (anti flicker)
    # =========================================================
    latents_out = latents * preserve_base + latents_out * (1 - preserve_base)

    print("👄 Fusion bouche HDR (soft + motion preserved)")

    return latents_out.clamp(-1.0, 1.0)

def apply_pro_net_with_mouth_clean(
    latents,
    mouth_coords,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    detail_strength=0.35,
    blur_kernel=5,
    mouth_radius_ratio=0.08,
    mask_blur_kernel=5,
    blend_strength=0.5,        # 🔥 nouveau
    preserve_motion=0.6        # 🔥 nouveau
):

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # =========================================================
    # 🔹 1. ProNet
    # =========================================================
    latents_prot = apply_n3r_pro_net(
        latents,
        model=n3r_pro_net,
        strength=n3r_pro_strength,
        sanitize_fn=sanitize_fn
    ).to(dtype)

    if not mouth_coords:
        return latents_prot

    # =========================================================
    # 🔹 2. Masque bouche (ellipse)
    # =========================================================
    mouth_mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    for x, y in mouth_coords:
        rx = int(W * mouth_radius_ratio)
        ry = int(H * mouth_radius_ratio)

        dist = ((X - x)**2)/(rx**2 + 1e-6) + ((Y - y)**2)/(ry**2 + 1e-6)
        mouth_mask[0,0] += (dist <= 1).float()

    mouth_mask = mouth_mask.clamp(0,1)

    # blur mask
    if mask_blur_kernel > 1:
        mouth_mask = F.avg_pool2d(
            mouth_mask,
            kernel_size=mask_blur_kernel,
            stride=1,
            padding=mask_blur_kernel // 2
        ).clamp(0,1)

    # =========================================================
    # 🔹 3. Low / High freq split
    # =========================================================
    blurred = F.avg_pool2d(latents_prot, blur_kernel, stride=1, padding=blur_kernel // 2)
    details = latents_prot - blurred

    # =========================================================
    # 🔹 4. 🔥 NON-LINEAR DETAIL COMPRESSION
    # évite effet "over sharpen"
    # =========================================================
    details = torch.tanh(details * 2.5) * 0.5

    # =========================================================
    # 🔹 5. 🔥 MOTION PRESERVATION
    # protège les zones en mouvement
    # =========================================================
    motion_map = torch.abs(details)
    motion_gate = torch.sigmoid(motion_map * 12.0)

    # =========================================================
    # 🔹 6. HDR effect (soft)
    # =========================================================
    enhanced = latents_prot + details * detail_strength

    # =========================================================
    # 🔹 7. 🔥 ADAPTIVE BLEND (clé)
    # =========================================================
    blend_mask = mouth_mask * blend_strength

    latents_blend = (
        latents_prot * (1 - blend_mask) +
        enhanced * blend_mask
    )

    # =========================================================
    # 🔹 8. 🔥 GATING FINAL (anti écrasement animation)
    # =========================================================
    latents_out = (
        latents_prot * (1 - motion_gate * preserve_motion) +
        latents_blend * (motion_gate * preserve_motion)
    )

    # =========================================================
    # 🔹 9. Clamp
    # =========================================================
    latents_out = torch.clamp(latents_out, -1.0, 1.0)

    print("👄 Fusion bouche HDR (soft + motion preserved)")

    return latents_out

def apply_pro_net_with_mouth_boost(
    latents,
    mouth_coords,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    detail_strength=0.35,       # intensité HDR
    blur_kernel=5,              # kernel pour détails
    mouth_radius_ratio=0.08,    # taille du masque relatif à l'image
    mask_blur_kernel=5           # flou du masque pour adoucir les contours
):
    """
    ProNet optimisé + amplification HDR des détails sur la bouche
    avec fusion douce pour éviter halo sur les contours.
    """

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ Appliquer ProNet standard
    latents_prot = apply_n3r_pro_net(
        latents,
        model=n3r_pro_net,
        strength=n3r_pro_strength,
        sanitize_fn=sanitize_fn
    ).to(dtype)

    # 2️⃣ Si pas de bouche détectée → fallback
    if not mouth_coords:
        return latents_prot

    # 3️⃣ Création masque bouche
    mouth_mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    for x, y in mouth_coords:
        rx = int(W * mouth_radius_ratio)
        ry = int(H * mouth_radius_ratio)
        dist = ((X - x)**2) / (rx**2 + 1e-6) + ((Y - y)**2) / (ry**2 + 1e-6)
        mouth_mask[0, 0] += (dist <= 1).float()

    mouth_mask = mouth_mask.clamp(0, 1)

    # 4️⃣ Flouter le masque pour adoucir les contours
    if mask_blur_kernel > 1:
        sigma = mask_blur_kernel / 3
        ax = torch.arange(-mask_blur_kernel // 2 + 1., mask_blur_kernel // 2 + 1., device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        mask_kernel_2d = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        mask_kernel_2d /= mask_kernel_2d.sum()
        mask_kernel = mask_kernel_2d.view(1, 1, mask_blur_kernel, mask_blur_kernel)
        mouth_mask = F.conv2d(mouth_mask, mask_kernel, padding=mask_blur_kernel // 2)
        mouth_mask = mouth_mask.clamp(0, 1)

    # 5️⃣ Blur pour récupérer les détails (high-frequency)
    sigma = blur_kernel / 3
    ax = torch.arange(-blur_kernel // 2 + 1., blur_kernel // 2 + 1., device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel_2d = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel_2d /= kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, blur_kernel, blur_kernel).repeat(C, 1, 1, 1).to(dtype)
    blurred = F.conv2d(latents_prot, kernel, padding=blur_kernel // 2, groups=C)
    details = latents_prot - blurred

    # 6️⃣ Amplification HDR adaptative selon le masque flou
    detail_strength_map = detail_strength * mouth_mask
    enhanced = latents_prot + details * detail_strength_map

    # 7️⃣ Fusion douce
    latents_out = latents_prot * (1 - mouth_mask) + enhanced * mouth_mask

    # 8️⃣ Clamp final pour sécurité
    latents_out = torch.clamp(latents_out, -1.0, 1.0)

    print("👄 HDR détails appliqué sur la bouche avec contours adoucis")

    return latents_out

#------------ Stable version mais un peu fort ----
def apply_pro_net_with_eyes_boost(
    latents,
    eye_coords,
    n3r_pro_net,
    n3r_pro_strength,
    sanitize_fn,
    detail_strength=0.35,     # intensité HDR
    blur_kernel=5,            # plus petit = plus précis
    iris_radius_ratio=0.06    # plus petit = cible mieux iris
):
    """
    ProNet + amplification HDR des détails sur l’iris (pas glow).
    """

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ ProNet
    latents_prot = apply_n3r_pro_net(
        latents,
        model=n3r_pro_net,
        strength=n3r_pro_strength,
        sanitize_fn=sanitize_fn
    )

    # 🔒 sécurité dtype (évite ton erreur Half/Float)
    latents_prot = latents_prot.to(dtype)

    # 2️⃣ Si pas d’yeux → fallback
    if not eye_coords:
        return latents_prot

    # 3️⃣ Création masque IRIS (ellipse fine)
    iris_mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    Y, X = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    for x, y in eye_coords:
        rx = int(W * iris_radius_ratio)
        ry = int(H * iris_radius_ratio)

        dist = ((X - x)**2) / (rx**2 + 1e-6) + ((Y - y)**2) / (ry**2 + 1e-6)
        iris_mask[0, 0] += (dist <= 1).float()

    iris_mask = iris_mask.clamp(0, 1)

    # 4️⃣ Kernel GAUSSIEN (corrigé)
    sigma = blur_kernel / 3

    ax = torch.arange(-blur_kernel // 2 + 1., blur_kernel // 2 + 1., device=device, dtype=dtype)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')

    kernel_2d = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel_2d = kernel_2d / kernel_2d.sum()

    kernel = kernel_2d.view(1, 1, blur_kernel, blur_kernel).repeat(C, 1, 1, 1)

    # 🔒 même dtype que latents
    kernel = kernel.to(dtype)

    # 5️⃣ Blur = base low-frequency
    blurred = F.conv2d(
        latents_prot,
        kernel,
        padding=blur_kernel // 2,
        groups=C
    )

    # 6️⃣ Détails (high-frequency)
    details = latents_prot - blurred

    # 7️⃣ Amplification HDR
    enhanced = latents_prot + detail_strength * details

    # 8️⃣ Fusion UNIQUEMENT sur iris
    latents_out = latents_prot * (1 - iris_mask) + enhanced * iris_mask

    # 9️⃣ Clamp sécurité
    latents_out = latents_out.clamp(-1.0, 1.0)

    print("👁 HDR détails appliqué sur iris")

    return latents_out

def apply_pro_net_with_eyes_test(latents, eye_coords, n3r_pro_net, n3r_pro_strength, sanitize_fn,
                           glow_strength=0.2, blur_kernel=7, iris_radius_ratio=0.08):
    """
    Applique ProNet et un glow froid uniquement sur l’iris des yeux.

    Args:
        latents (torch.Tensor): [B,C,H,W] Latents à traiter.
        eye_coords (list of tuples): Coordonnées yeux [(x1,y1),(x2,y2)]
        n3r_pro_net: modèle ProNet
        n3r_pro_strength (float): force ProNet
        sanitize_fn: fonction de nettoyage latents
        glow_strength (float): intensité du glow
        blur_kernel (int): kernel pour flou gaussien
        iris_radius_ratio (float): proportion de H/W pour rayon iris

    Returns:
        torch.Tensor: latents avec glow sur iris uniquement
    """
    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    # 1️⃣ Application ProNet
    latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)

    # 2️⃣ Glow froid uniquement sur l’iris
    if eye_coords:
        iris_mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)
        for x, y in eye_coords:
            rx = int(W * iris_radius_ratio)
            ry = int(H * iris_radius_ratio)
            Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
            dist2 = ((X - x)**2) / (rx**2) + ((Y - y)**2) / (ry**2)
            iris_mask[0, 0] += (dist2 <= 1).float()
        iris_mask = iris_mask.clamp(0, 1)

        # Kernel gaussien 2D
        sigma = blur_kernel / 3
        ax = torch.arange(-blur_kernel // 2 + 1., blur_kernel // 2 + 1., device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel_2d = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel_2d = kernel_2d / kernel_2d.sum()
        kernel = kernel_2d.view(1, 1, blur_kernel, blur_kernel).repeat(C, 1, 1, 1)

        # Convolution channel-wise pour glow
        glow = F.conv2d(latents_prot * iris_mask, kernel, padding=blur_kernel // 2, groups=C)

        # Fusion uniquement sur l’iris
        latents_out = latents_prot * (1 - iris_mask) + glow * iris_mask * glow_strength
        latents_out = latents_out.clamp(-1.0, 1.0)
        print("👁 Glow froid appliqué sur iris uniquement")
    else:
        # fallback si pas d’yeux détectés
        latents_out = latents_prot

    return latents_out


def apply_pro_net_with_eye_glow(latents, eye_coords, n3r_pro_net, n3r_pro_strength, sanitize_fn, glow_strength=0.2, blur_kernel=7):
    """
    Applique ProNet et un glow froid uniquement sur les yeux.

    Args:
        latents (torch.Tensor): [B,C,H,W] Latents à traiter.
        eye_coords (list of tuples): Coordonnées yeux [(x1,y1),(x2,y2)]
        n3r_pro_net: modèle ProNet
        n3r_pro_strength (float): force ProNet
        sanitize_fn: fonction de nettoyage latents
        glow_strength (float): intensité du glow
        blur_kernel (int): kernel pour le flou

    Returns:
        torch.Tensor: latents avec glow sur yeux uniquement
    """
    # 1️⃣ Appliquer ProNet
    latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)

    # 2️⃣ Glow froid sur latents ProNet
    glow_latents = apply_intelligent_glow_froid_latents(latents_prot, strength=glow_strength, blur_kernel=blur_kernel)


    # 3️⃣ Fusion glow uniquement sur les yeux
    if eye_coords:
        eye_radius = int(min(latents.shape[-2:]) * 0.15)
        eye_mask = create_eye_mask(latents, eye_coords, eye_radius)
        if eye_mask is not None:
            eye_mask = eye_mask.to(latents.device).float()
            if eye_mask.ndim == 3:  # [B,H,W] -> [B,1,H,W]
                eye_mask = eye_mask.unsqueeze(1)
            latents = latents * (1 - eye_mask) + glow_latents * eye_mask
            print("👁 Glow froid appliqué uniquement sur yeux")
        else:
            latents = glow_latents  # fallback
    else:
        latents = glow_latents  # pas d’yeux détectés → glow global

    return latents

# Application effect en dehors de yeux:
def apply_pro_net_with_out_eyes(latents, eye_coords, n3r_pro_net, n3r_pro_strength, sanitize_fn):
    # 1️⃣ Application du ProNet
    latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)

    # 2️⃣ Application du glow froid intelligent en dehors des yeux sur le ProNet
    latents_prot = apply_intelligent_glow_froid_out(latents_prot)

    # 3️⃣ Fusion avec le masque yeux si détecté
    if eye_coords:
        print("Eye coords:", eye_coords)
        eye_radius = int(min(latents.shape[-2:]) * 0.15)  # augmenter légèrement pour protection
        eye_mask = create_eye_mask(latents, eye_coords, eye_radius)

        if eye_mask is not None:
            eye_mask = eye_mask.to(latents.device)
            # protection yeux + ProNet + glow
            latents = latents * eye_mask + latents_prot * (1 - eye_mask)
            print("👁 protection yeux appliquée avec glow froid")
        else:
            # si le masque échoue, on applique ProNet + glow sur tout
            latents = latents_prot
    else:
        # pas d’yeux détectés → ProNet + glow global
        latents = latents_prot

    return latents


def apply_pro_net_with_eye_simple(latents, eye_coords, n3r_pro_net, n3r_pro_strength, sanitize_fn):
    latents_prot = apply_n3r_pro_net(latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_fn)
    if eye_coords:
        print("Eye coords:", eye_coords)
        eye_radius = int(min(latents.shape[-2:]) * 0.15)  # un peu plus large valeur 0.12 ou 0.15
        eye_mask = create_eye_mask(latents, eye_coords, eye_radius)
        if eye_mask is not None:
            eye_mask = eye_mask.to(latents.device)
            latents = latents * eye_mask + latents_prot * (1 - eye_mask)
            print("👁 protection yeux appliquée (main frames)")
        else:
            latents = latents_prot
    else:
        latents = latents_prot
    return latents

def tensor_to_pil(tensor):
    """
    tensor: [1,3,H,W] ou [3,H,W] dans [-1,1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = (tensor.clamp(-1,1) + 1) / 2
    return to_pil_image(tensor.cpu())

try:

    MP_FACE_MESH = mp_face_mesh
except Exception:
    MP_FACE_MESH = None
    print("⚠ mediapipe non disponible → fallback sans yeux")


def get_coords_safe(image, H, W):
    coords = get_coords(image)

    if coords:
        print(f"👁 Eyes detected: {coords}")
        return coords

    print("⚠ fallback eye coords used")

    # 🔥 adapté portrait vertical (ton cas 536x960)
    return [
        (int(H * 0.32), int(W * 0.38)),
        (int(H * 0.32), int(W * 0.62))
    ]

# --------------------------------------------------
# 🔥 Détection yeux (version clean sans cv2)
# --------------------------------------------------
def get_coords(image):
    """
    Retourne [(y_left, x_left), (y_right, x_right)]
    Compatible PIL ou numpy
    """
    if MP_FACE_MESH is None:
        return []

    # Conversion propre
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image

    if img is None or img.ndim != 3:
        return []

    h, w, _ = img.shape

    with MP_FACE_MESH.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(img)  # ✅ déjà RGB → pas besoin de cv2

        if not results.multi_face_landmarks:
            return []

        lm = results.multi_face_landmarks[0].landmark

        # 🔥 Points clés yeux (stables)
        left_eye_pts  = [33, 133]
        right_eye_pts = [362, 263]

        left_eye = np.mean([(lm[i].y * h, lm[i].x * w) for i in left_eye_pts], axis=0)
        right_eye = np.mean([(lm[i].y * h, lm[i].x * w) for i in right_eye_pts], axis=0)

        return [
            (int(left_eye[0]), int(left_eye[1])),
            (int(right_eye[0]), int(right_eye[1]))
        ]

# --------------------------------------------------
# 🔥 Création mask yeux (latents)
# --------------------------------------------------

def create_volumetrique_mask(latents, coords, radius_ratio=0.15, only=False, in_radius_ratio=0.08, debug=False):
    """
    Crée un masque pour les yeux ou uniquement pour l’iris.

    Args:
        latents (torch.Tensor): [B,C,H,W] Latents
        coords (list of tuples): [(x1,y1),(x2,y2)] coordonnées yeux
        radius_ratio (float): proportion H/W pour rayon
        only (bool): True → masque uniquement iris, False → masque œil entier
        in_radius_ratio (float): proportion H/W pour rayon iris si only=True
        debug (bool): Si True, affiche le masque

    Returns:
        torch.Tensor: [B,1,H,W] masque float (0=hors masque, 1=masque)
    """
    #if not coords or latents.ndim != 4:
    if coords is None or len(coords) == 0 or latents.ndim != 4:
        return None

    B, C, H, W = latents.shape
    device, dtype = latents.device, latents.dtype

    mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

    for x, y in coords:
        r = int(min(H, W) * (radius_ratio if only else in_radius_ratio))
        Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        dist2 = (X - x)**2 + (Y - y)**2
        mask[0, 0] += (dist2 <= r**2).float()

    mask = mask.clamp(0, 1)

    if debug:
        # Affiche le masque superposé à un latents converti en image pour vérification
        lat_vis = latents[0, 0].detach().cpu()  # canal 0
        plt.figure(figsize=(6,6))
        plt.imshow(lat_vis, cmap='gray', alpha=0.7)
        plt.imshow(mask[0,0].cpu(), cmap='Reds', alpha=0.3)
        plt.title("Debug Eye/Iris Mask")
        plt.show()

    return mask

def create_eye_mask(latents, eye_coords, eye_radius=8, falloff=4):
    """
    Soft mask gaussien → transitions naturelles
    """
    if eye_coords is None or len(eye_coords) == 0:
        return None

    B, C, H, W = latents.shape
    mask = torch.zeros((1, 1, H, W), device=latents.device)

    for y_c, x_c in eye_coords:
        y_lat = int(y_c / 8)
        x_lat = int(x_c / 8)

        for y in range(H):
            for x in range(W):
                dist = ((y - y_lat)**2 + (x - x_lat)**2)**0.5
                value = max(0, 1 - dist / (eye_radius + falloff))
                mask[0, 0, y, x] = torch.maximum(mask[0, 0, y, x], torch.tensor(value, device=latents.device))

    return mask.repeat(B, C, 1, 1)

def create_mouth_mask(latents, mouth_coords, mouth_radius=8, falloff=4):
    """
    Crée un masque gaussien doux autour des coordonnées de la bouche.
    Args:
        latents (torch.Tensor): tenseur des latents [B, C, H, W]
        mouth_coords (list[(y,x)]): centre(s) de la bouche en coords latents
        mouth_radius (int): rayon principal du masque
        falloff (int): transition douce
    Returns:
        torch.Tensor: masque [B, C, H, W] avec valeurs entre 0 et 1
    """
    if mouth_coords is None or len(mouth_coords) == 0:
        return None

    B, C, H, W = latents.shape
    mask = torch.zeros((1, 1, H, W), device=latents.device)

    for y_c, x_c in mouth_coords:
        # 🔹 Si coords image -> latents, elles doivent déjà être scalées
        y_lat = int(y_c)
        x_lat = int(x_c)

        # 🔹 Création d'une grille
        yy, xx = torch.meshgrid(torch.arange(H, device=latents.device),
                                torch.arange(W, device=latents.device),
                                indexing='ij')
        dist = ((yy - y_lat)**2 + (xx - x_lat)**2).sqrt()
        value = torch.clamp(1 - dist / (mouth_radius + falloff), min=0.0)
        mask[0, 0] = torch.maximum(mask[0, 0], value)

    return mask.repeat(B, C, 1, 1)


def detect_eyes_auto(frame_pil):
    """Retourne les coordonnées (y,x) approximatives des yeux"""
    img = np.array(frame_pil)
    h, w, _ = img.shape
    with MP_FACE_MESH.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not results.multi_face_landmarks:
            return []
        lm = results.multi_face_landmarks[0].landmark
        left_eye = np.mean([(lm[i].y*h, lm[i].x*w) for i in [33, 133]], axis=0)
        right_eye = np.mean([(lm[i].y*h, lm[i].x*w) for i in [362, 263]], axis=0)
        return [(int(left_eye[0]), int(left_eye[1])), (int(right_eye[0]), int(right_eye[1]))]


# ************************************************************************************************************************


temporal_model = TemporalResidualNet(
    channels=4,
    hidden=64,
    num_blocks=4
).to(device="cuda")

optimizer_temporal = torch.optim.AdamW(
    temporal_model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

temporal_model.apply(weights_temporal_init)

criterion_temporal = TemporalLoss()

#-----------------------------------------------------------------------



# Exemple d'initialisation du StyleInjector
latent_channels = 4  # correspond au nombre de canaux de tes latents (ex: 4)
hidden = 64          # taille interne, peut être ajustée
num_blocks = 4       # nombre de ResidualBlocks
prompt_dim = 768     # dimension de ton embedding de prompt

# Instanciation du modèle
style_model = StyleInjector(
    latent_channels=latent_channels,
    hidden=hidden,
    num_blocks=num_blocks,
    prompt_dim=prompt_dim
).to(device="cuda")


# Initialisation des poids
style_model.apply(weights_init)


# --- Critère pour StyleInjector ---
criterion_style = StyleLoss(reduction="mean")  # ou "sum" si tu préfères

# --- Hyperparamètres ---
learning_rate = 1e-4  # à ajuster selon ton entraînement
weight_decay = 1e-5   # régularisation optionnelle

# --- Création de l'optimizer ---
optimizer_style = torch.optim.Adam(
    style_model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    weight_decay=weight_decay
)


def apply_style_injection(
    latents,
    style_prompt_embedding,
    style_model,
    optimizer=None,
    criterion=None,
    train=False,
    new_image=False,
    frame_counter=0,
    max_epochs_up=5,
    model_path="models/style_injector_latest.pt",
    debug=True,
    ema=False,
    ema_prev_latents=None,
    ema_alpha=0.2
):
    device = latents.device
    style_model.to(device)
    latents_train = latents.clone().detach().to(device)
    B, C, H, W = latents_train.shape




    # ----- Préparer le prompt -----
    if style_prompt_embedding is None:
        style_embedding_train = torch.zeros(B, style_model.prompt_proj.in_features, device=device)
    else:
        style_embedding_train = style_prompt_embedding.clone().detach().to(device)
        if style_embedding_train.dim() == 3:  # (B, seq_len, dim)
            style_embedding_train = style_embedding_train[:, 0, :]  # CLS token

    model_exists = os.path.exists(model_path)

    # =====================================================
    # TRAIN MODE
    # =====================================================
    if train and optimizer is not None and criterion is not None:
        style_model.train()
        max_epochs = max(1, max_epochs_up)

        early_stop_threshold = 0.25  # seuil d'early stop
        epoch_done = 0  # nombre réel d'epochs effectués

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            with torch.enable_grad():
                pred_latents = style_model(latents_train, style_embedding_train)
                loss = criterion(pred_latents, latents_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(style_model.parameters(), 1.0)
            optimizer.step()

            epoch_done = epoch + 1

            if debug:
                print(f"[StyleInjection] Epoch {epoch_done}/{max_epochs} | Loss={loss.item():.6f}")

            # 🔥 EARLY STOP
            if loss.item() < early_stop_threshold:
                if debug:
                    print(f"🟢 Early stop triggered (loss={loss.item():.4f})")
                break

        save_style_model(style_model, optimizer=optimizer, epoch=epoch_done, loss=loss.item())
    else:
        style_model.eval()
        if model_exists:
            style_model, checkpoint = load_style_model(type(style_model), optimizer=optimizer)
            if debug:
                print(f"[StyleInjection] Loaded checkpoint | Last loss={checkpoint.get('loss')}")

    # =====================================================
    # FORWARD FINAL
    # =====================================================
    with torch.no_grad():
        stylized_latents = style_model(latents_train, style_embedding_train)


    # =====================================================
    # NEW IMAGE HANDLING (reset EMA avant motion)
    # =====================================================
    if new_image:
        print("🟢 nouvelle image → reset latents et EMA")
        stylized_latents = latents.clone()
        ema_prev_latents = stylized_latents.clone()
        # On retourne stylized_latents normal, pas de blending
        return stylized_latents

    # =====================================================
    # ADAPTIVE TEMPORAL BLENDING (REMPLACE EMA BRUTE)
    # =====================================================
    if ema and ema_prev_latents is not None and not new_image:

        if debug:
            print("🔥 [apply_style_injection] EMA.")

        # -----------------------------
        # Device safety
        # -----------------------------
        if ema_prev_latents.device != stylized_latents.device:
            ema_prev_latents = ema_prev_latents.to(stylized_latents.device)

        # -----------------------------
        # Motion stats
        # -----------------------------
        motion = (stylized_latents - ema_prev_latents).abs().mean()
        lmotion = (stylized_latents - latents).abs().mean().clamp(0, 1)

        # -----------------------------
        # Adaptive alpha
        # -----------------------------
        alpha = ema_alpha * (1.0 + 2.0 * lmotion)
        alpha = alpha.clamp(0.02, 0.12)

        # -----------------------------
        # Blend
        # -----------------------------
        hf = compute_high_freq_energy(latents_train)

        stylized_latents = motion_aware_ema_fusion(
            out=stylized_latents,
            ema_prev_latents=ema_prev_latents,
            hf=hf,
            debug=True
        )

        diff = (stylized_latents - ema_prev_latents).abs().mean().item()
        print(f"[StyleInjection EMA drift] {diff:.6f}")

        # -----------------------------
        # Logs (debug only)
        # -----------------------------
        if debug:
            print(
                f"[StyleInjection][EMA] "
                f"motion={motion:.4f} | "
                f"lmotion={lmotion:.4f} | "
                f"alpha={alpha:.4f}"
            )
    # =====================================================
    # DEBUG STATS
    # =====================================================
    if debug:
        final_std = stylized_latents.std()
        final_hf = (stylized_latents - F.avg_pool2d(stylized_latents, kernel_size=3, stride=1, padding=1)).std()
        print(f"[StyleInjection FINAL] std={final_std:.4f} | high_freq={final_hf:.4f}")

    return stylized_latents

# =========================================================
# TEMPORAL APPLY PIPELINE
# =========================================================

def apply_temporal_consistency(
    prev_latents,
    current_latents,
    temporal_model,
    optimizer=None,
    criterion=None,
    train=False,
    new_image=False,
    frame_counter=0,
    max_epochs_up=10,
    model_path="models/temporal_latest.pt",
    debug=False,
    ema_prev_latents=None,
    ema_alpha=0.3,
    ema=False
):
    """
    Applique une cohérence temporelle entre prev/current latents.

    Pipeline:
    1. Détection du mouvement temporel
    2. Adaptation dynamique du training
    3. Prediction du latent suivant
    4. Injection temporelle douce
    5. EMA smoothing

    Args:
        prev_latents (Tensor):
            Latents frame N-1

        current_latents (Tensor):
            Latents frame N

        temporal_model (nn.Module):
            Modèle temporel

        optimizer:
            Optimizer PyTorch

        criterion:
            Fonction de perte

        train (bool):
            Active entraînement dynamique

        frame_counter (int):
            Numéro de frame

        max_epochs_up (int):
            Epochs max d'entraînement

        model_path (str):
            Checkpoint path

        debug (bool):
            Affichage debug

        ema_prev_latents (Tensor):
            EMA latents précédente

        ema_alpha (float):
            EMA smoothing

    Returns:
        Tensor:
            Latents temporellement stabilisés
    """

    device = current_latents.device
    print(f"[Temporal] device={device}")

    # =====================================================
    # TEMPORAL MOTION DETECTION
    # =====================================================
    if (prev_latents.device != current_latents.device):
        print(f"[Temporal] prev_latents :device={device} ")

        prev_latents= prev_latents.clone().detach().to(device)


    temporal_delta = current_latents - prev_latents

    motion_level = temporal_delta.abs().mean()

    high_freq_motion = (
        temporal_delta -
        F.avg_pool2d(
            temporal_delta,
            kernel_size=3,
            stride=1,
            padding=1
        )
    )

    motion_noise = high_freq_motion.std()

    is_motion_heavy = motion_noise > 0.15

    print(
        f"[Temporal] "
        f"motion_level={motion_level:.4f} | "
        f"motion_noise={motion_noise:.4f} | "
        f"is_motion_heavy={is_motion_heavy}"
    )

    # =====================================================
    # PREPARE LATENTS
    # =====================================================
    prev_train = prev_latents.clone().detach().to(device)
    current_train = current_latents.clone().detach().to(device)


    prev_train.requires_grad_(False)
    current_train.requires_grad_(False)

    model_exists = os.path.exists(model_path)

    # =====================================================
    # TRAIN MODE
    # =====================================================

    if train:

        print("[Temporal] Dynamic temporal training")

        temporal_model.train()

        temporal_model.to(device)

        # -------------------------------------------------
        # DYNAMIC EPOCHS
        # -------------------------------------------------

        if is_motion_heavy:

            if frame_counter == 0:
                max_epochs = max_epochs_up
            else:
                max_epochs = max_epochs_up + 15

        else:

            min_epochs = 1
            max_epochs_cap = max_epochs_up

            max_epochs = int(
                min_epochs +
                (max_epochs_cap - min_epochs)
                * motion_noise.item()
            )

        max_epochs = max(1, max_epochs)

        print(f"[Temporal] max_epochs={max_epochs}")

        # -------------------------------------------------
        # SKIP TRAIN
        # -------------------------------------------------

        if max_epochs == 1:

            print("[Temporal] Training skipped")

            train = False

        # -------------------------------------------------
        # ENABLE GRAD
        # -------------------------------------------------

        for param in temporal_model.parameters():
            param.requires_grad = True

        # -------------------------------------------------
        # TRAIN LOOP
        # -------------------------------------------------

        if train:
            temporal_model.train()
            early_stop_threshold = 0.025  # exemple, à ajuster selon tes observations
            patience = 2  # nombre d'epochs supplémentaires avant d'arrêter
            patience_counter = 0

            for epoch in range(max_epochs):
                optimizer.zero_grad()

                with torch.enable_grad():
                    pred_next = temporal_model(prev_train, current_train)
                    target = current_train
                    loss = criterion(pred_next, target)

                print(
                    f"[Temporal DEBUG] "
                    f"pred_grad={pred_next.requires_grad} | "
                    f"loss_grad={loss.requires_grad} | "
                    f"grad_fn={type(loss.grad_fn).__name__}"
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), max_norm=1.0)
                optimizer.step()

                print(f"[Temporal] Epoch [{epoch+1}/{max_epochs}] Loss={loss.item():.6f}")

                if debug:
                    print(f"[Temporal DEBUG] pred min={pred_next.min():.4f} max={pred_next.max():.4f}")

                # 🔥 EARLY STOP AVEC PATIENCE
                if loss.item() < early_stop_threshold:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"🟢 Early stop triggered at epoch {epoch+1} (loss={loss.item():.4f})")
                        break
                else:
                    patience_counter = 0  # reset si loss remonte

            save_temporal_model(temporal_model, optimizer=optimizer, epoch=epoch+1, loss=loss.item())

    # =====================================================
    # EVAL MODE
    # =====================================================

    else:

        print("[Temporal] Eval mode")

        if model_exists:

            temporal_model, checkpoint = load_temporal_model(
                type(temporal_model),
                optimizer=optimizer
            )

            print(
                f"[Temporal] "
                f"Last loss={checkpoint.get('loss')}"
            )

        else:

            print(
                "[WARN] Temporal model not found, "
                "using untrained model."
            )

        temporal_model.eval()

    # =====================================================
    # FINAL INFERENCE
    # =====================================================

    with torch.no_grad():

        pred_next = temporal_model(
            prev_train,
            current_train
        )

    # =====================================================
    # SANITIZE
    # =====================================================

    pred_next = torch.tanh(pred_next)
    pred_next = pred_next.clamp(-1.0, 1.0)
    pred_next = sanitize_latents(pred_next)

    # =====================================================
    # ADAPTIVE TEMPORAL INJECTION
    # =====================================================

    if is_motion_heavy:

        strength = min(
            0.05 + 0.35 * motion_noise.item() ** 0.5,
            0.85
        )

    else:

        strength = min(
            max(0.03, 0.1 * motion_noise.item()),
            0.25
        )

    print(
        f"[Temporal] "
        f"strength={strength:.4f}"
    )

    temporal_latents = (
        current_latents +
        strength * pred_next
    )

    # =====================================================
    # EMA SMOOTHING
    # =====================================================

    if ema and ema_prev_latents is not None and new_image == False:
        print(f"🔥 [Temporal] EMA.")

        if ema_prev_latents.device != temporal_latents.device:

            ema_prev_latents = ema_prev_latents.to(
                temporal_latents.device
            )

        temporal_latents = (
            ema_alpha * temporal_latents +
            (1.0 - ema_alpha) * ema_prev_latents
        )

        print("[Temporal] EMA smoothing applied")

    # =====================================================
    # FINAL DEBUG
    # =====================================================

    final_std = temporal_latents.std()

    final_hf = (
        temporal_latents -
        F.avg_pool2d(
            temporal_latents,
            kernel_size=3,
            stride=1,
            padding=1
        )
    ).std()

    print(
        f"[Temporal FINAL] "
        f"std={final_std:.4f} | "
        f"high_freq={final_hf:.4f}"
    )

    return temporal_latents

# ***********************************************NOTE *************************************************************************
#GlobalStateTracker
#DriftController
#AnchorLoss module
#Style model = injecte une transformation créative contrôlée par prompt - Style Injection Network
# “Video diffusion system with controllable temporal motion + adaptive denoising + prompt-driven stylistic rendering”
# ton style model doit être un “injecteur léger”, pas un second générateur
# ************************************************************************************************************************

class EMADeltaRebound:
    """
    Détecte une frame bruitée si la variation par rapport à la dernière EMA dépasse le dernier pic bruité.
    """

    def __init__(self):
        self.last_noisy_level = None  # Niveau de delta de la dernière frame bruitée

    def compute_delta_std(self, latents, ema_prev_latents):
        """
        Calcule l'écart-type de la différence entre la frame actuelle et l'EMA précédente.
        """
        delta = latents - ema_prev_latents
        return delta.std().item()

    def update(self, latents, ema_prev_latents):
        """
        Args:
            latents (torch.Tensor): Latents de la frame actuelle.
            ema_prev_latents (torch.Tensor): EMA des latents précédents.

        Returns:
            noise_level (float): Niveau de delta actuel.
            is_noisy (bool): True si la frame est plus bruyante que le dernier pic.
        """
        # Assurer que les deux tensors sont sur le même device
        if ema_prev_latents.device != latents.device:
            ema_prev_latents = ema_prev_latents.to(latents.device)

        noise_level = self.compute_delta_std(latents, ema_prev_latents)

        # Détection basée sur le rebond
        if self.last_noisy_level is None:
            is_noisy = True  # Première frame considérée bruitée
        else:
            is_noisy = noise_level > self.last_noisy_level

        # Mise à jour systématique du dernier niveau high_freq
        self.last_noisy_level = noise_level


        return noise_level, is_noisy

# Initialisation (une seule fois)
hf_rebound = EMADeltaRebound()

# Version very parfaite final !
#        ┌─────────────────────────────┐
#        │      Frame courante         │
#        └─────────────┬──────────────┘
#                      │
#                      ▼
#         ┌─────────────────────────┐
#         │  EMA_prev_latents existe? │
#         └───────┬───────────────┘
#                 │ Oui
#                 ▼
#       ┌─────────────────────┐
#       │  Calcul high-freq   │
#       │  & Noise_level      │
#       └───────┬─────────────┘
#               │
#               ▼
#           ┌─────────────┐
#           │ is_noisy?   │
#           └─────┬───────┘
#          Oui     Non
#           │       │
#           ▼       ▼
#  ┌─────────────────────┐        ┌─────────────────────────────┐
#  │  Frames bruitées     │       │ Frames calmes (non bruitées) │
#  │  → max_epochs_up     │       │  → max_epochs minimal ou 0  │
#  │  → injection forte   │       │  → injection douce          │
#  └─────────┬───────────┘        └─────────────┬────────────────┘
#            │                                 │
#            ▼                                 ▼
#      ┌─────────────┐                  ┌─────────────┐
#      │ Entraînement │                  │ Pas d'entraînement
#      │ dynamique    │                  │ ou injection minimale
#      └─────┬───────┘                  └─────┬───────┘
#            │                                │
#            ▼                                ▼
#    ┌──────────────────┐            ┌─────────────────┐
#    │  Denoising       │            │  Denoising      │
#    │  latents_out     │            │  latents_out    │
#    └─────────┬────────┘            └─────────┬──────┘
#              │                               │
#              ▼                               ▼
#      ┌─────────────────────┐         ┌─────────────────────────┐
#      │ Injection adaptative │         │ Injection adaptative │
#      │ strength ↑ si bruit  │         │ strength ↓ si calme  │
#      └─────────┬───────────┘         └─────────┬───────────────┘
#                │                               │
#                ▼                               ▼
#         ┌─────────────┐                 ┌─────────────┐
#         │  EMA smooth │                 │  EMA smooth │
#         │  latents    │                 │  latents    │
#         └─────────────┘                 └─────────────┘
#                │                               │
#                ▼                               ▼
#           Latents finaux après denoising


def dynamic_max_epochs(loss, threshold=0.03, max_epochs=25):
    if loss < threshold:
        return 1
    return max_epochs

def apply_denoising(
    latents,
    denoising_model,
    optimizer=None,
    criterion=None,
    train=False,
    frame_counter=0,
    max_epochs_up=10,
    model_path="models/denoise_latest.pt",
    debug=False,
    ema_prev_latents=None
):
    """
    Applique un denoising autoencoder sur les latents avec entraînement optionnel et dynamique.

    Logique dynamique:
    1. Detection du bruit via EMA (high_freq + std)
    2. Frames bruitées → entraînement plus long + injection forte
    3. Frames calmes → entraînement minimal ou nul + injection douce
    4. EMA finale pour lisser les transitions

    Args:
        latents (torch.Tensor): Latents d'entrée.
        denoising_model (torch.nn.Module): Le modèle de denoising.
        optimizer (torch.optim.Optimizer, optional): Optimizer pour l'entraînement.
        criterion (callable, optional): Fonction de perte.
        train (bool): Mode entraînement si True.
        frame_counter (int): Numéro de frame actuel.
        max_epochs (int): Nombre d'epochs pour l'entraînement.
        model_path (str): Chemin pour sauvegarde/chargement du modèle.
        debug (bool): Activer affichage debug.
        denoise (bool): Appliquer le denoising si True.
    Returns:
        torch.Tensor: Latents après denoising/injection.
    """
    is_noisy = False
    if ema_prev_latents is not None:
        # delta entre la frame courante et l'EMA précédente
        # Test noise frames
        noise_level, is_noisy = hf_rebound.update(latents, ema_prev_latents)
        print(f"Noise_level EMA - high_freq: [{noise_level}], is_noisy: [{is_noisy}],  on start...🔥 ")
    else:
        is_noisy = True
        # detection du bruit
        noise_level = latents.std()
        print(f"Noise_level - latents.std: [{noise_level}], on start...🔥 ")
        high_freq = latents - torch.nn.functional.avg_pool2d(latents, kernel_size=3, stride=1, padding=1)
        noise_level = high_freq.std()
        print(f"Noise_level - high_freq: [{noise_level}], on start...🔥 , is_noisy: [{is_noisy}]")

    latents_train = latents.clone().detach().to("cuda").requires_grad_(True)
    model_exists = os.path.exists(model_path)

    # ----- Entraînement -----
    if train:
        print(f"Mode Denoise - Entraînement dynamique")
        denoising_model.train()
        denoising_model.to("cuda")

        # Ajustement du nombre d'epochs selon bruit
        if is_noisy:
            if frame_counter == 0:
                max_epochs = max_epochs_up
            else:
                max_epochs = max_epochs_up + 20
        else:
            min_epochs = 1
            max_epochs_cap = max_epochs_up
            max_epochs = int(min_epochs + (max_epochs_cap - min_epochs) * noise_level / 2.0)

        if max_epochs == 1:
            # Pas d'entraînement si très peu d'epochs
            train = False

        # Activation du gradient
        for param in denoising_model.parameters():
            param.requires_grad = True

        # Early stop threshold (à ajuster selon ton dataset)
        early_stop_threshold = 0.23

        epoch = 0  # <-- initialisation
        loss = None
        latents_out = latents_train.clone()  # valeur par défaut : rien n'est modifié

        if train:
            for epoch in range(max_epochs):
                optimizer.zero_grad()
                with torch.enable_grad():
                    latents_out = denoising_model(latents_train)
                    loss = criterion(latents_out, latents_train)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoising_model.parameters(), max_norm=1.0)
                optimizer.step()

                print(f"[DENoise] Epoch {epoch+1}/{max_epochs} | Loss={loss.item():.6f}")

                # 🔥 Early stop
                if loss.item() < early_stop_threshold:
                    print(f"🟢 Early stop triggered at epoch {epoch+1} (loss={loss.item():.4f})")
                    break

        save_denoising_model(
            denoising_model,
            optimizer=optimizer,
            epoch=epoch + 1,  # pour refléter la dernière epoch réelle
            loss=loss.item() if loss is not None else 0.0
        )

    # ----- Évaluation / Chargement -----
    else:
        latents_out = latents_train.clone()  # valeur par défaut : rien n'est modifié
        print(f"Mode simple - Denoise sans Entraînement")
        if model_exists:
            denoising_model, checkpoint = load_denoising_model(
                type(denoising_model),  # récupère la classe de ton modèle MyDenoiser,
                optimizer=optimizer
            )
            print("Last saved loss:", checkpoint.get("loss"))
        else:
            print("[WARN] Denoising model not found, using untrained model.")

        denoising_model.eval()
        latents_out, loss = denoise_latents( latents, denoising_model, optimizer=None, criterion=criterion, device="cuda", train=False )

    if isinstance(latents_out, tuple):
        latents_out = latents_out[0]

    # 🔹 Limiter les valeurs extrêmes et sanitize il n'est utils ici'
    #latents_out = torch.tanh(latents_out).clamp(-1.0, 1.0)
    #latents_out = sanitize_latents(latents_out)

    # 🔹 Injection douce adaptative
    if loss is not None:
        if is_noisy:
            #strength = min(max(0.05, 0.15 * noise_level), 0.4)  # plus agressif si bruit réel
            # Boost pour bruit intense
            strength = min(0.05 + 0.3 * noise_level**0.5, 0.8)
        else:
            strength = min(max(0.05, 0.1 * noise_level), 0.2)   # sinon injection plus douce

        print(f"Strength [{strength}], Noise_level: [{noise_level}], Loss: [{loss}]")
        latents = latents + strength * latents_out
    else:
        latents = 0.9 * latents + 0.1 * latents_out


    # Dection final pour vérification
    noise_level = latents.std()
    print(f"Noise_level - latents.std: [{noise_level}], after denoising... 🔥 ")
    high_freq = latents - torch.nn.functional.avg_pool2d(latents, kernel_size=3, stride=1, padding=1)
    noise_level = high_freq.std()
    print(f"Noise_level - high_freq: [{noise_level}], after denoising... 🔥 ")


    return latents


def update_ema(latents, ema_prev_latents, alpha=0.5, device="cuda"):
    if ema_prev_latents is None:
        return latents

    return (
        alpha * latents.detach().to(device)
        + (1 - alpha) * ema_prev_latents.detach().to(device)
    )


def get_ema_style_prev(latents, train_on_image, ema_prev_latents=None, debug=False):
    if train_on_image:
        if debug:
            print("[StyleEMA] image training → no temporal coherence 🔥")
        return latents

    if ema_prev_latents is not None:
        if debug:
            print(f"[StyleEMA] video → using ema_prev_latents {ema_prev_latents.shape} 🔥")
        return ema_prev_latents

    if debug:
        print("[StyleEMA] video → fallback to latents 🔥")
    return latents

"""
stage	          rôle
Denoise	          nettoyer image brute
Temporal	      stabiliser vidéo
Style	          définir identité
Appearance	      éclairage / finition render (appearance = correction photométrique + micro-contrast + tone mapping) -> Appearance (photometric refinement ONLY)
"""

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DECODE LATENTS FONCTION PRINCIPAL
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def decode_latents_ultrasafe_blockwise_ultranatural(
    latents,
    vae,
    block_size=32, overlap=16,
    device="cuda",
    new_image=False,
    frame_counter=0,
    latent_scale_boost=1.0,
    use_hann=True,
    sharpen_mode="both2",              # None, "tanh", "edges", "both", "both2"
    sharpen_strength=0.015,
    sharpen_edges_strength=0.02,
    gamma_boost=1.00,                  # légèrement plus de punch naturel
    scale=8,
    denoise=False, temporal_consistency=False, style_injection=False, appearance=False, creative=False, art=False,
    micro_boost=False,
    filtre_vae=True,
    suffix="00",
    export_latents=False,
    export_latents_comfy=True,
    output_dir: Path = None,
    pos_embeds_list=None,
    train=False,                       # Paramètre ajouté pour gérer l'entraînement
    train_on_image=True,
    max_epochs_up=10,
    ema_prev_latents=None,
    latents_sample=None, # Sample exemple
    ema_global=None,
    ema_micro=None,
    debug=False
):

    vae.eval()  # pas besoin de caster tout le VAE
    B, C, H, W = latents.shape


    if frame_counter == 0:
        denoise=True
        temporal_consistency=True
        style_injection=True
        appearance=True
        creative=True
        art=True
    else:
        denoise=True #OK
        temporal_consistency=True #OK
        style_injection=True # A verifier (motion_aware_ema_fusion) ema=False by Default
        appearance=True # OK (Pas de EMA)
        creative=True # OK (Pas de EMA)
        art=True # OK (Pas de EMA)


    # 1. Cloner les latents pour traitement
    raw_latents = latents.clone()
    latents_out = raw_latents.clone()

    # 2. NEW_IMAGE reset
    if new_image:
        print("🔥 NEW IMAGE RESET EMA")
        ema_prev_latents = raw_latents.clone().to(device).detach()

    # 3. EMA / Scene Change
    if ema_prev_latents is not None:
        ema_prev_latents = ema_prev_latents.to(device)
        scene_delta = (raw_latents - ema_prev_latents).abs().mean().item()
        is_scene_change = scene_delta > 0.45
        if is_scene_change:
            print(f"🟠 Scene change (delta={scene_delta:.4f}) → EMA RESET")
            ema_prev_latents = raw_latents.clone().to(device).detach()
    else:
        is_scene_change = False

    # 4. Calcul motion_noise
    low = F.avg_pool2d(raw_latents, 3, 1, 1)
    high_freq = raw_latents - low
    motion_noise = high_freq.abs().mean() / (raw_latents.abs().mean() + 1e-6)
    print(f"🔥 [decode_latents_ultrasafe] motion_noise={motion_noise:.3f}")
    ema_alpha = 0.001 + 0.005 * (1.0 - motion_noise)
    ema_alpha = ema_alpha.clamp(0.005, 0.01)
    print(f"🔥 [decode_latents_ultrasafe] EMA ema_alpha={ema_alpha:.3f}")
    # =====================================================
    # MODEL AND TRAINNIG
    # =====================================================
    #style_prompt_embedding = torch.randn(B, prompt_dim, device=device, dtype=latents.dtype) # version random
    style_prompt_embedding = torch.zeros( B, prompt_dim, device=device, dtype=latents.dtype )
    # Exemple avec le premier prompt positif
    if pos_embeds_list is not None:
        style_prompt_embedding = pos_embeds_list[0].to(device).to(dtype=latents.dtype)  # forme [B, prompt_dim]

    #Rendu perceptuel (ce qui manque)

    #C’est là que vivent :
    #lighting #volumetrics #cinematic feel #texture richness #atmospheric depth
    # 🔥 Appearance model (créativité visuelle)

    # =====================================================
    # 1. DENOISE (optionnel, propre input cleanup)
    # =====================================================
    if denoise and ema_prev_latents is not None:
        # Créer un latents indépendant pour l'entraînement
        latents = apply_denoising( latents=latents_out, denoising_model=denoising_model, optimizer=optimizer, criterion=criterion, train=train, frame_counter=frame_counter,
                            max_epochs_up=max_epochs_up, model_path="models/denoise_latest.pt", debug=False, ema_prev_latents=ema_prev_latents)

    # =====================================================
    # 2. CREATIVE (STRUCTURE / TEXTURE / STOCHASTIC FIELD)
    # =====================================================
    # New Fonction : in DEV
    if creative and latents_sample is not None:
        #apprend une représentation de style stable en training compact (style_score + embedding) sert de “cerveau de style”
        latents = apply_creative(
            latents=latents, latents_sample=latents_sample,
            style_prompt_embedding=style_prompt_embedding, creative_model=creative_model, optimizer=optimizer_creative, criterion=criterion_creative, train=True, device="cuda", strength=1.00, debug=True,
            new_image=new_image, frame_counter=frame_counter, max_epochs_up=max_epochs_up, model_path="models/creative_model_latest.pt")

    if art:
        # ⚠️ IMPORTANT: art DOIT être déterministe en inference
        # train_art = train and train_on_image
        #applique des transformations locales contrôle texture / structure / détails effets artistiques explicites sert de “renderer créatif”
        latents = apply_art(
            latents=latents,
            style_prompt_embedding=style_prompt_embedding, art_model=art_model, optimizer=optimizer_art, criterion=criterion_art, train=False, device="cuda", strength=0.50, debug=True, new_image=new_image, frame_counter=frame_counter, max_epochs_up=max_epochs_up, model_path="models/art_model_latest.pt")

    # =====================================================
    # 3. STYLE INJECTION (OPTIONNEL middle layer)
    # =====================================================
    if style_injection and ema_prev_latents is not None:

        # Appliquer le StyleInjector
        latents = apply_style_injection(
                latents=latents,
                style_prompt_embedding=style_prompt_embedding,  # ton embedding prompt
                style_model=style_model,
                optimizer=optimizer_style,  # optionnel si fine-tuning dynamique
                criterion=criterion_style,  # StyleLoss
                train=train,  # si tu veux fine-tuner à la volée
                new_image=new_image,
                frame_counter=frame_counter,
                max_epochs_up=max_epochs_up,
                model_path="models/style_injector_latest.pt",
                debug=True,
                ema_prev_latents=ema_prev_latents,
                ema_alpha=ema_alpha #ema_alpha=0.2 + 0.3 * motion_noise  # tu peux adapter la force EMA
        )

    # =====================================================
    # 4. APPEARANCE (PHOTOMETRIC RENDERING LAYER)
    # =====================================================

    if appearance and latents_sample is not None:
        #photométrie contraste / gamma / exposure micro-structure visuelle agit comme un rendering layer
        latents = apply_appearance(
            latents=latents,
            style_prompt_embedding=style_prompt_embedding, appearance_model=appearance_model, optimizer=optimizer_apparence, criterion=criterion_apparence, train=False, device="cuda", strength=0.1, debug=True, new_image=new_image, frame_counter=frame_counter, max_epochs_up=max_epochs_up, model_path="models/appearance_model_latest.pt", latents_sample=latents_sample)


    # =====================================================
    # 5. TEMPORAL CONSISTENCY (OPTIONAL smoothing layer)
    # =====================================================
    if temporal_consistency and ema_prev_latents is not None:

        latents = apply_temporal_consistency( prev_latents=ema_prev_latents, current_latents=latents, temporal_model=temporal_model, optimizer=optimizer_temporal, criterion=criterion_temporal,
        train=train, new_image=new_image, frame_counter=frame_counter, max_epochs_up=max_epochs_up, model_path="models/temporal_latest.pt", debug=False, ema_prev_latents=ema_prev_latents, ema_alpha = ema_alpha)

    # -------------------------
    # 6. UPDATE EMA (CRUCIAL)
    # -------------------------
    ema_prev_latents, ema_global, ema_micro = motion_aware_ema_low_high(
            latents=latents,
            ema_prev_latents=ema_prev_latents,
            ema_global=ema_global,
            ema_micro=ema_micro,
            motion_noise=motion_noise,
            train=train,
            debug=True,
        )

    # Export Standard
    if export_latents:
        export_latents_file(latents, suffix, frame_counter, output_dir)

    # Export for ConfyUI
    if export_latents_comfy:
        sequence_id = generate_sequence_id()
        print(f"🔥 [decode_latents_ultrasafe] sequence_id={sequence_id} ")
        #save_latents_for_comfy(latents, frame_counter, output_dir)
        metadata = build_latent_metadata( latents=latents, frame_counter=frame_counter, sequence_id=sequence_id, scale=scale, denoise=denoise, temporal_consistency=temporal_consistency, style_injection=style_injection, appearance=appearance, creative=creative, sharpen_mode=sharpen_mode, gamma_boost=gamma_boost, train=train, ema_prev_latents=ema_prev_latents )

        export_latents_for_comfy_meta( latents, suffix, frame_counter, output_dir, metadata)


    # ⚡ latents en float16 pour réduire VRAM, multiplication par scale
    latents = latents.to(device=device, dtype=torch.float16) * latent_scale_boost

    out_H, out_W = H * scale, W * scale
    output_rgb = torch.zeros(B, 3, out_H, out_W, device=device, dtype=torch.float32)
    weight = torch.zeros_like(output_rgb)

    stride = block_size - overlap
    y_positions = list(range(0, H, stride))
    x_positions = list(range(0, W, stride))

    # ---------------- Feather ----------------
    def create_feather(h, w):
        if use_hann:
            wy = torch.hann_window(h, device=device, dtype=torch.float32)
            wx = torch.hann_window(w, device=device, dtype=torch.float32)
            return (wy[:, None] * wx[None, :]).clamp(min=1e-3)
        else:
            y = torch.linspace(0, 1, h, device=device, dtype=torch.float32)
            x = torch.linspace(0, 1, w, device=device, dtype=torch.float32)
            wy = 1 - torch.abs(y - 0.5) * 2
            wx = 1 - torch.abs(x - 0.5) * 2
            return (wy[:, None] * wx[None, :]).clamp(min=1e-3)

    # ---------------- Decode patch par patch ----------------
    for y in y_positions:
        for x in x_positions:
            y1 = min(y + block_size, H)
            x1 = min(x + block_size, W)

            patch = latents[:, :, y:y1, x:x1]
            patch = torch.nan_to_num(patch, nan=0.0)

            with torch.no_grad():
                # ⚡ Convertir temporairement patch en float32 pour compatibilité VAE
                decoded = vae.decode(patch.to(torch.float32)).sample
                decoded = decoded.to(torch.float32)

            fh, fw = decoded.shape[2], decoded.shape[3]
            feather = create_feather(fh, fw).unsqueeze(0).unsqueeze(0)

            iy0, ix0 = y*scale, x*scale
            iy1, ix1 = iy0 + fh, ix0 + fw

            output_rgb[:, :, iy0:iy1, ix0:ix1] += decoded * feather
            weight[:, :, iy0:iy1, ix0:ix1] += feather

            # ⚡ Libération VRAM patch
            del patch, decoded, feather
            torch.cuda.empty_cache()

    # ---------------- Normalisation ----------------
    weight = torch.clamp(weight, min=1e-3)
    output_rgb = (output_rgb / weight).clamp(-1.0, 1.0)

    # ---------------- Sharp adaptatif ----------------
    if sharpen_mode is not None:
        if sharpen_mode in ["tanh", "both", "both2"]:
            mean = output_rgb.mean(dim=[2,3], keepdim=True)
            detail = output_rgb - mean
            local_std = detail.std(dim=[2,3], keepdim=True) + 1e-6
            adapt_strength = sharpen_strength / (1 + 5*(1-local_std))
            output_rgb = output_rgb + adapt_strength * torch.tanh(detail)

        if sharpen_mode in ["edges", "both"]:
            B, C, H, W = output_rgb.shape
            kernel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=output_rgb.dtype)
            kernel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=output_rgb.dtype)
            kernel_x = kernel_x.view(1,1,3,3).repeat(C,1,1,1)
            kernel_y = kernel_y.view(1,1,3,3).repeat(C,1,1,1)

            grad_x = F.conv2d(output_rgb, kernel_x, padding=1, groups=C)
            grad_y = F.conv2d(output_rgb, kernel_y, padding=1, groups=C)
            edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            edges = edges / (edges.mean(dim=[2,3], keepdim=True) + 1e-6)
            edge_mask = torch.sigmoid(6.0 * (edges - 0.7))
            output_rgb = output_rgb + sharpen_edges_strength * edges * edge_mask

        if sharpen_mode in ["both2"]:
            B, C, H, W = output_rgb.shape

            kernel_x = torch.tensor( [[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=output_rgb.dtype )
            kernel_y = torch.tensor( [[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=output_rgb.dtype )

            kernel_x = kernel_x.view(1,1,3,3)
            kernel_y = kernel_y.view(1,1,3,3)

            # -----------------------------
            # Luminance-only sharpening
            # -----------------------------
            luma = (
                0.299 * output_rgb[:,0:1] +
                0.587 * output_rgb[:,1:2] +
                0.114 * output_rgb[:,2:3]
            )
            grad_x = F.conv2d(luma, kernel_x, padding=1)
            grad_y = F.conv2d(luma, kernel_y, padding=1)
            edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            # normalize
            edges = edges / (edges.mean(dim=[2,3], keepdim=True) + 1e-6)
            # stabilize
            edges = torch.clamp(edges, 0, 2.5)
            # soft edge selection
            edge_mask = torch.sigmoid(4.0 * (edges - 0.5))
            # final sharpen
            output_rgb = output_rgb + (
                sharpen_edges_strength * edges * edge_mask
            )

        output_rgb = output_rgb.clamp(-1.0, 1.0)

    # ---------------- Gamma adaptatif ----------------
    if gamma_boost > 1.0:
        output_rgb_gamma = ((output_rgb + 1) / 2.0).clamp(0,1)
        luminance = output_rgb_gamma.mean(dim=1, keepdim=True)
        adapt_gamma = gamma_boost * (1.0 + 0.1*(0.5-luminance))
        output_rgb_gamma = output_rgb_gamma ** adapt_gamma
        output_rgb = output_rgb_gamma * 2 - 1

    # ---------------- Micro-boost couleur ----------------
    if micro_boost:
        mean_c = output_rgb.mean(dim=[2,3], keepdim=True)
        color_boost = torch.sigmoid(5.0*(output_rgb - mean_c)) * 0.03
        output_rgb = (output_rgb + color_boost).clamp(-1.0, 1.0)
    else:
        mean_c = output_rgb.mean(dim=[2,3], keepdim=True)
        color_delta = output_rgb - mean_c
        color_boost = torch.tanh(5.0 * color_delta) * 0.015
        output_rgb = (output_rgb + color_boost).clamp(-1.0, 1.0)

    if filtre_vae:
        output_rgb[:,0] *= 0.97   # réduire rouge
        output_rgb[:,2] *= 1.01   # léger boost bleu

    # ---------------- Conversion PIL ----------------
    frames = [to_pil_image((output_rgb[i] + 1) / 2) for i in range(B)]

    return frames[0] if B == 1 else frames, ema_prev_latents



def decode_latents_ultrasafe_blockwise_sharp(
    latents, vae,
    block_size=32, overlap=16,
    device="cuda",
    frame_counter=0,
    latent_scale_boost=1.0,
    use_hann=True,
    sharpen_mode="both",              # None, "tanh", "edges", "both"
    sharpen_strength=0.04,
    sharpen_edges_strength=0.05
):

    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()

    B, C, H, W = latents.shape
    latents = latents.to(device=device, dtype=torch.float32) * latent_scale_boost

    out_H, out_W = H * 8, W * 8
    output_rgb = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output_rgb)

    stride = block_size - overlap
    y_positions = list(range(0, H, stride))
    x_positions = list(range(0, W, stride))

    # ---------------- Feather ----------------
    def create_feather(h, w):
        if use_hann:
            wy = torch.hann_window(h, device=device)
            wx = torch.hann_window(w, device=device)
            return (wy[:, None] * wx[None, :]).clamp(min=1e-3)
        else:
            y = torch.linspace(0, 1, h, device=device)
            x = torch.linspace(0, 1, w, device=device)
            wy = 1 - torch.abs(y - 0.5) * 2
            wx = 1 - torch.abs(x - 0.5) * 2
            return (wy[:, None] * wx[None, :]).clamp(min=1e-3)

    # ---------------- Decode ----------------
    for y in y_positions:
        for x in x_positions:
            y1 = min(y + block_size, H)
            x1 = min(x + block_size, W)

            patch = latents[:, :, y:y1, x:x1]
            patch = torch.nan_to_num(patch, nan=0.0)

            with torch.no_grad():
                decoded = vae.decode(patch).sample.to(torch.float32)

            fh, fw = decoded.shape[2], decoded.shape[3]

            feather = create_feather(fh, fw)
            feather = feather.unsqueeze(0).unsqueeze(0)

            iy0, ix0 = y*8, x*8
            iy1, ix1 = iy0 + fh, ix0 + fw

            output_rgb[:, :, iy0:iy1, ix0:ix1] += decoded * feather
            weight[:, :, iy0:iy1, ix0:ix1] += feather

    # ---------------- Normalisation ----------------
    weight = torch.clamp(weight, min=1e-3)
    output_rgb = (output_rgb / weight).clamp(-1.0, 1.0)

    # =========================================================
    # 🔥 SHARPEN SECTION
    # =========================================================

    if sharpen_mode is not None:

        # ---- 1. Tanh sharpen (détails globaux)
        if sharpen_mode in ["tanh", "both"]:
            mean = output_rgb.mean(dim=[2,3], keepdim=True)
            detail = output_rgb - mean
            output_rgb = output_rgb + sharpen_strength * torch.tanh(detail)

        # ---- Edge sharpen PRO (anti plastique)
        if sharpen_mode in ["edges", "both"]:
            B, C, H, W = output_rgb.shape

            kernel_x = torch.tensor(
                [[-1,0,1],[-2,0,2],[-1,0,1]],
                device=device,
                dtype=output_rgb.dtype
            )

            kernel_y = torch.tensor(
                [[-1,-2,-1],[0,0,0],[1,2,1]],
                device=device,
                dtype=output_rgb.dtype
            )

            kernel_x = kernel_x.view(1,1,3,3).repeat(C,1,1,1)
            kernel_y = kernel_y.view(1,1,3,3).repeat(C,1,1,1)

            grad_x = F.conv2d(output_rgb, kernel_x, padding=1, groups=C)
            grad_y = F.conv2d(output_rgb, kernel_y, padding=1, groups=C)

            edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

            # 🔥 NORMALISATION douce (pas globale)
            edges = edges / (edges.mean(dim=[2,3], keepdim=True) + 1e-6)

            # 🔥 MASQUE BEAUCOUP plus sélectif (clé)
            edge_mask = torch.sigmoid(6.0 * (edges - 0.7))

            # 🔥 DIRECTION du contraste (pas ajout brut)
            sign = torch.sign(output_rgb)

            output_rgb = output_rgb + sharpen_edges_strength * edge_mask * sign * edges * 0.5



        output_rgb = output_rgb.clamp(-1.0, 1.0)

    # ---------------- To PIL ----------------
    # Ajouter gamma boost ici
    gamma = 1.10
    output_rgb_gamma = ((output_rgb + 1.0) / 2.0).clamp(0,1)
    output_rgb_gamma = output_rgb_gamma ** gamma
    output_rgb_gamma = output_rgb_gamma * 2.0 - 1.0
    output_rgb = output_rgb_gamma

    frames = [to_pil_image((output_rgb[i] + 1) / 2) for i in range(B)]
    return frames[0] if B == 1 else frames


def decode_latents_ultrasafe_blockwise_plastique(
    latents, vae,
    block_size=32, overlap=16,
    device="cuda",
    frame_counter=0,
    latent_scale_boost=1.0,
    use_hann=True,
    sharpen_mode="both",              # None, "tanh", "edges", "both"
    sharpen_strength=0.04,
    sharpen_edges_strength=0.05
):

    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()

    B, C, H, W = latents.shape
    latents = latents.to(device=device, dtype=torch.float32) * latent_scale_boost

    out_H, out_W = H * 8, W * 8
    output_rgb = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output_rgb)

    stride = block_size - overlap
    y_positions = list(range(0, H, stride))
    x_positions = list(range(0, W, stride))

    # ---------------- Feather ----------------
    def create_feather(h, w):
        if use_hann:
            wy = torch.hann_window(h, device=device)
            wx = torch.hann_window(w, device=device)
            return (wy[:, None] * wx[None, :]).clamp(min=1e-3)
        else:
            y = torch.linspace(0, 1, h, device=device)
            x = torch.linspace(0, 1, w, device=device)
            wy = 1 - torch.abs(y - 0.5) * 2
            wx = 1 - torch.abs(x - 0.5) * 2
            return (wy[:, None] * wx[None, :]).clamp(min=1e-3)

    # ---------------- Decode ----------------
    for y in y_positions:
        for x in x_positions:
            y1 = min(y + block_size, H)
            x1 = min(x + block_size, W)

            patch = latents[:, :, y:y1, x:x1]
            patch = torch.nan_to_num(patch, nan=0.0)

            with torch.no_grad():
                decoded = vae.decode(patch).sample.to(torch.float32)

            fh, fw = decoded.shape[2], decoded.shape[3]

            feather = create_feather(fh, fw)
            feather = feather.unsqueeze(0).unsqueeze(0)

            iy0, ix0 = y*8, x*8
            iy1, ix1 = iy0 + fh, ix0 + fw

            output_rgb[:, :, iy0:iy1, ix0:ix1] += decoded * feather
            weight[:, :, iy0:iy1, ix0:ix1] += feather

    # ---------------- Normalisation ----------------
    weight = torch.clamp(weight, min=1e-3)
    output_rgb = (output_rgb / weight).clamp(-1.0, 1.0)

    # =========================================================
    # 🔥 SHARPEN SECTION
    # =========================================================

    if sharpen_mode is not None:

        # ---- 1. Tanh sharpen (détails globaux)
        if sharpen_mode in ["tanh", "both"]:
            mean = output_rgb.mean(dim=[2,3], keepdim=True)
            detail = output_rgb - mean
            output_rgb = output_rgb + sharpen_strength * torch.tanh(detail)

        # ---- 2. Edge sharpen (version PRO stable)
        if sharpen_mode in ["edges", "both"]:
            B, C, H, W = output_rgb.shape

            kernel_x = torch.tensor(
                [[-1,0,1],[-2,0,2],[-1,0,1]],
                device=device,
                dtype=output_rgb.dtype
            )

            kernel_y = torch.tensor(
                [[-1,-2,-1],[0,0,0],[1,2,1]],
                device=device,
                dtype=output_rgb.dtype
            )

            kernel_x = kernel_x.view(1,1,3,3).repeat(C,1,1,1)
            kernel_y = kernel_y.view(1,1,3,3).repeat(C,1,1,1)

            grad_x = F.conv2d(output_rgb, kernel_x, padding=1, groups=C)
            grad_y = F.conv2d(output_rgb, kernel_y, padding=1, groups=C)

            edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

            # 🔥 NORMALISATION LOCALE (clé stabilité)
            edges = edges / (edges.mean(dim=[2,3], keepdim=True) + 1e-6)

            # 🔥 MASQUE edges (évite bruit dans zones plates)
            edge_mask = torch.sigmoid(4.0 * (edges - 0.5))

            # 🔥 Sharpen intelligent
            output_rgb = output_rgb + sharpen_edges_strength * edges * edge_mask

        output_rgb = output_rgb.clamp(-1.0, 1.0)

    # ---------------- To PIL ----------------
    frames = [to_pil_image((output_rgb[i] + 1) / 2) for i in range(B)]
    return frames[0] if B == 1 else frames


def decode_latents_ultrasafe_blockwise_pro(
    latents, vae,
    block_size=32, overlap=16,
    device="cuda",
    frame_counter=0,
    latent_scale_boost=1.0,
    use_hann=True
):

    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()

    B, C, H, W = latents.shape
    latents = latents.to(device=device, dtype=torch.float32) * latent_scale_boost

    out_H, out_W = H * 8, W * 8
    output_rgb = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output_rgb)

    stride = block_size - overlap
    y_positions = list(range(0, H, stride))
    x_positions = list(range(0, W, stride))

    # 🔥 Fenêtre de blending PRO (Hann = ultra stable)
    def create_feather(h, w):
        if use_hann:
            wy = torch.hann_window(h, device=device)
            wx = torch.hann_window(w, device=device)
            return (wy[:, None] * wx[None, :]).clamp(min=1e-3)
        else:
            y = torch.linspace(0, 1, h, device=device)
            x = torch.linspace(0, 1, w, device=device)
            wy = 1 - torch.abs(y - 0.5) * 2
            wx = 1 - torch.abs(x - 0.5) * 2
            return (wy[:, None] * wx[None, :]).clamp(min=1e-3)

    for y in y_positions:
        for x in x_positions:
            y1 = min(y + block_size, H)
            x1 = min(x + block_size, W)

            patch = latents[:, :, y:y1, x:x1]
            patch = torch.nan_to_num(patch, nan=0.0)

            with torch.no_grad():
                decoded = vae.decode(patch).sample.to(torch.float32)

            fh, fw = decoded.shape[2], decoded.shape[3]

            # 🔥 feather dynamique (corrige bord image)
            feather = create_feather(fh, fw)
            feather = feather.unsqueeze(0).unsqueeze(0)

            iy0, ix0 = y*8, x*8
            iy1, ix1 = iy0 + fh, ix0 + fw

            output_rgb[:, :, iy0:iy1, ix0:ix1] += decoded * feather
            weight[:, :, iy0:iy1, ix0:ix1] += feather

    # 🔥 sécurité critique (évite artefacts)
    weight = torch.clamp(weight, min=1e-3)

    output_rgb = (output_rgb / weight).clamp(-1.0, 1.0)

    frames = [to_pil_image((output_rgb[i] + 1) / 2) for i in range(B)]
    return frames[0] if B == 1 else frames


# Decode latents par blockwise - ultrasafe :
def decode_latents_ultrasafe_blockwise(latents, vae,
                                       block_size=32, overlap=16,
                                       device="cuda",
                                       frame_counter=0,
                                       latent_scale_boost=1.0):
    """
    Décodage ultra-safe par blocs des latents en image PIL.
    Paramètres conservés uniquement : block_size, overlap, device, frame_counter, latent_scale_boost
    """

    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()

    B, C, H, W = latents.shape
    latents = latents.to(device=device, dtype=torch.float32) * latent_scale_boost

    out_H, out_W = H * 8, W * 8
    output_rgb = torch.zeros(B, 3, out_H, out_W, device=device)
    weight = torch.zeros_like(output_rgb)

    stride = block_size - overlap
    y_positions = list(range(0, H, stride))
    x_positions = list(range(0, W, stride))

    for y in y_positions:
        for x in x_positions:
            y1 = min(y + block_size, H)
            x1 = min(x + block_size, W)
            patch = latents[:, :, y:y1, x:x1]
            patch = torch.nan_to_num(patch, nan=0.0)

            with torch.no_grad():
                decoded = vae.decode(patch).sample.to(torch.float32)

            iy0, ix0 = y*8, x*8
            iy1, ix1 = iy0 + decoded.shape[2], ix0 + decoded.shape[3]
            output_rgb[:, :, iy0:iy1, ix0:ix1] += decoded
            weight[:, :, iy0:iy1, ix0:ix1] += 1.0

    output_rgb = (output_rgb / weight.clamp(min=1e-6)).clamp(-1.0, 1.0)

    frames = [to_pil_image((output_rgb[i] + 1) / 2) for i in range(B)]
    return frames[0] if B == 1 else frames



def apply_intelligent_glow_froid(
    frame_pil,
    strength=0.18,
    edge_weight=0.6,
    luminance_weight=0.8,
    blur_radius=1.2
):


    # ---------------- Base ----------------
    if frame_pil.mode != "RGB":
        frame_pil = frame_pil.convert("RGB")

    arr = np.array(frame_pil).astype(np.float32) / 255.0

    # ---------------- Luminance mask ----------------
    # luminance perceptuelle
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # masque doux (favorise les zones claires)
    lum_mask = np.clip((lum - 0.6) / 0.4, 0, 1)
    lum_mask = np.power(lum_mask, 1.5)  # douceur

    # ---------------- Edge mask ----------------
    gray = (lum * 255).astype(np.uint8)
    edge = Image.fromarray(gray).filter(ImageFilter.FIND_EDGES)
    edge = np.array(edge).astype(np.float32) / 255.0

    # adoucir les edges (évite bruit)
    edge = np.clip(edge * 1.2, 0, 1)
    edge = np.power(edge, 1.3)

    # ---------------- Fusion intelligente ----------------
    combined_mask = (
        luminance_weight * lum_mask +
        edge_weight * edge
    )

    combined_mask = np.clip(combined_mask, 0, 1)

    # ---------------- Glow ----------------
    # blur image pour glow
    glow_img = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    glow_arr = np.array(glow_img).astype(np.float32) / 255.0

    # appliquer glow uniquement où mask actif
    result = arr + (glow_arr - arr) * combined_mask[..., None] * strength

    result = np.clip(result, 0, 1)

    return Image.fromarray((result * 255).astype(np.uint8))


def smooth_edges(frame_pil, strength=0.4, blur_radius=1.2):

    # 1️⃣ edges
    edges = frame_pil.convert("L").filter(ImageFilter.FIND_EDGES)

    # 2️⃣ normalisation du masque
    edges_np = np.array(edges).astype(np.float32) / 255.0
    edges_np = np.clip(edges_np * 2.0, 0, 1)  # renforce zones edges

    # 3️⃣ blur global (source)
    blurred = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 4️⃣ blend intelligent
    orig = np.array(frame_pil).astype(np.float32)
    blur = np.array(blurred).astype(np.float32)

    mask = edges_np[..., None] * strength

    result = orig * (1 - mask) + blur * mask

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))






def apply_intelligent_glow(frame_pil,
                           glow_strength=0.22,
                           blur_radius=1.2,
                           luminance_threshold=0.7,
                           edge_strength=1.2,
                           detail_preservation=0.85):
    """
    Glow intelligent :
    - basé sur luminance + edges
    - évite effet flou global
    - boost détails lumineux uniquement
    """

    # -----------------------
    # 1️⃣ Base numpy
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32) / 255.0

    # -----------------------
    # 2️⃣ Luminance mask
    # -----------------------
    gray = frame_pil.convert("L")
    lum = np.array(gray).astype(np.float32) / 255.0

    lum_mask = np.clip((lum - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)

    # -----------------------
    # 3️⃣ Edge mask (important 🔥)
    # -----------------------
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageEnhance.Contrast(edges).enhance(edge_strength)

    edge_arr = np.array(edges).astype(np.float32) / 255.0

    # 🔥 combinaison intelligente
    combined_mask = lum_mask * edge_arr

    # -----------------------
    # 4️⃣ Glow blur
    # -----------------------
    blurred = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_arr = np.array(blurred).astype(np.float32) / 255.0

    # -----------------------
    # 5️⃣ Application du glow
    # -----------------------
    for c in range(3):
        arr[..., c] = arr[..., c] + glow_strength * combined_mask * blurred_arr[..., c]

    arr = np.clip(arr, 0, 1)

    # -----------------------
    # 6️⃣ Reconstruction
    # -----------------------
    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # 7️⃣ Préservation détails
    # -----------------------
    img = Image.blend(frame_pil, img, 1 - detail_preservation)

    # -----------------------
    # 8️⃣ Micro sharpen
    # -----------------------
    img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=25, threshold=2))

    return img


def apply_chromatic_soft_glow(frame_pil,
                              glow_strength=0.25,
                              exposure=1.05,
                              blur_radius=2.0,
                              luminance_threshold=0.8,
                              color_saturation=1.05,
                              sharpen=True):
    """
    Soft Glow chromatique localisé :
    - Glow appliqué sur pixels clairs selon leur canal (R/G/B)
    - Zones sombres préservées
    - Détails conservés
    """


    arr = np.array(frame_pil).astype(np.float32) / 255.0
    arr = np.clip(arr * exposure, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # Masque par canal
    # -----------------------
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    mask_r = np.clip((r - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)
    mask_g = np.clip((g - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)
    mask_b = np.clip((b - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)

    # -----------------------
    # Glow par canal
    # -----------------------
    bright = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bright_arr = np.array(bright).astype(np.float32) / 255.0

    # Mélange selon masque couleur
    arr[...,0] = np.clip(arr[...,0] + glow_strength * mask_r * bright_arr[...,0], 0, 1)
    arr[...,1] = np.clip(arr[...,1] + glow_strength * mask_g * bright_arr[...,1], 0, 1)
    arr[...,2] = np.clip(arr[...,2] + glow_strength * mask_b * bright_arr[...,2], 0, 1)

    img = Image.fromarray((arr*255).astype(np.uint8))

    # -----------------------
    # Saturation douce
    # -----------------------
    img = ImageEnhance.Color(img).enhance(color_saturation)

    # -----------------------
    # Micro sharpen subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=30, threshold=2))

    return img


def apply_localized_soft_glow(frame_pil,
                              glow_strength=0.25,
                              exposure=1.05,
                              blur_radius=2.0,
                              luminance_threshold=0.6,
                              color_saturation=1.05,
                              sharpen=True):
    """
    Filtre 'Soft Glow Localisé':
    - Glow appliqué seulement sur les zones lumineuses
    - Effet subtil, préserve les zones sombres
    - Maintien des détails
    """

    # -----------------------
    # 1️⃣ Convertir en float + exposure
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32) / 255.0
    arr = np.clip(arr * exposure, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # 2️⃣ Masque de luminosité
    # -----------------------
    gray = img.convert("L")
    lum_arr = np.array(gray).astype(np.float32) / 255.0
    mask = np.clip((lum_arr - luminance_threshold) / (1.0 - luminance_threshold), 0, 1)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    # -----------------------
    # 3️⃣ Glow léger
    # -----------------------
    bright = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    glow_img = ImageChops.screen(img, bright)
    # Appliquer glow uniquement là où mask > 0
    glow_img = Image.composite(glow_img, img, mask_img)
    img = Image.blend(img, glow_img, glow_strength)

    # -----------------------
    # 4️⃣ Saturation douce
    # -----------------------
    img = ImageEnhance.Color(img).enhance(color_saturation)

    # -----------------------
    # 5️⃣ Micro sharpen subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=30, threshold=2))

    return img


def apply_soft_glow(frame_pil,
                    glow_strength=0.25,
                    exposure=1.05,
                    blur_radius=2.0,
                    color_saturation=1.05,
                    sharpen=True):
    """
    Filtre 'Soft Glow' :
    - Surexposition douce sur les zones claires
    - Glow léger et subtil
    - Maintien des détails et textures
    """

    # -----------------------
    # 1️⃣ Convertir en float + exposure léger
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32) / 255.0
    arr = np.clip(arr * exposure, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # 2️⃣ Glow subtil (Light Bloom)
    # -----------------------
    bright = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    img = ImageChops.screen(img, bright)
    img = Image.blend(img, bright, glow_strength)

    # -----------------------
    # 3️⃣ Saturation douce
    # -----------------------
    img = ImageEnhance.Color(img).enhance(color_saturation)

    # -----------------------
    # 4️⃣ Micro sharpen subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=30, threshold=2))

    return img





def save_frame_verbose(frame: Image.Image, output_dir: Path, frame_counter: int, suffix: str = "00", psave: bool = True):
    """
    Sauvegarde une frame avec suffixe et affiche un message si verbose=True

    Args:
        frame (Image.Image): Image PIL à sauvegarder
        output_dir (Path): Dossier de sortie
        frame_counter (int): Numéro de frame
        suffix (str): Suffixe pour différencier les étapes
        verbose (bool): Affiche le message si True
    """
    file_path = output_dir / f"frame_{frame_counter:05d}_{suffix}.png"

    if psave:
        print(f"[SAVE Frame {frame_counter:03d}_{suffix}] -> {file_path}")
        frame.save(file_path)
    return file_path



#---------------------------------------------
# version optimiser - soft et net
#--------------------------------------------
def apply_n3r_pro_net(latents, model=None, strength=0.15, sanitize_fn=None):
    """
    Version ultra-subtile de ProNet :
    - réduit fortement le bruit
    - amplification très douce
    - très rapide GPU
    - évite halos / sur-détails
    """
    detail_smoothing=0.7
    clamp_range=1.0

    if model is None or strength <= 0:
        return latents

    try:
        dtype = next(model.parameters()).dtype
        latents = latents.to(dtype)

        # 🔹 Inference optimisée (no grad = VRAM ↓)
        with torch.no_grad():
            refined = model(latents)

        # 🔹 Detail map
        detail = refined - latents

        # 🔹 🔥 Suppression du bruit AVANT amplification
        # mélange entre brut et lissé
        if detail_smoothing > 0:
            smooth = F.avg_pool2d(detail, kernel_size=3, stride=1, padding=1)
            detail = (1 - detail_smoothing) * detail + detail_smoothing * smooth

        # 🔹 🔥 Limiteur de détails extrêmes (très important)
        detail = torch.tanh(detail)

        # 🔹 🔥 Injection très douce
        latents_out = latents + strength * detail

        # 🔹 Sanitize léger
        if sanitize_fn:
            latents_out = sanitize_fn(latents_out)

        # 🔹 Clamp sécurisé
        return latents_out.clamp(-clamp_range, clamp_range)

    except Exception as e:
        print(f"[N3RProNet ERROR] {e}")
        return latents

#---------------------------------------------
# version optimiser - soft
#--------------------------------------------

def apply_n3r_pro_net_soft(latents, model=None, strength=0.05, sanitize_fn=None):
    """
    N3R ProNet simplifié et optimisé :
    - amplification de détail très douce
    - lissage du détail pour réduire le bruit
    - contrôle mémoire via no_grad()
    """

    if model is None or strength <= 0:
        return latents

    try:
        device = latents.device
        dtype = next(model.parameters()).dtype

        latents = latents.to(dtype)

        # 🔹 Inference sans gradient pour économiser VRAM
        with torch.no_grad():
            refined = model(latents)

        # 🔹 Calcul de la carte de détail
        detail = refined - latents

        # 🔹 Lissage pour limiter le bruit
        #    avg_pool plus léger ou conv 3x3 si besoin
        detail = F.avg_pool2d(detail, kernel_size=3, stride=1, padding=1)

        # 🔹 Injection contrôlée et réduite
        latents = latents + strength * detail

        # 🔹 Optionnel : fonction de sanitation très légère
        if sanitize_fn:
            latents = sanitize_fn(latents)

        # 🔹 Clamp final pour éviter les extrêmes
        return latents.clamp(-1.0, 1.0)

    except Exception as e:
        print(f"[N3RProNet ERROR] {e}")
        return latents

#---------------------------------------------
# version optimiser - boost eclat
#--------------------------------------------

def apply_n3r_pro_net_boost(latents, model=None, strength=0.3, sanitize_fn=None):
    if model is None or strength <= 0:
        return latents

    try:
        latents = latents.to(next(model.parameters()).dtype)
        refined = model(latents)

        # 🔥 différence (detail map)
        detail = refined - latents

        # 🔥 SMOOTH du détail (clé !!!)
        detail = F.avg_pool2d(detail, kernel_size=3, stride=1, padding=1)

        # 🔥 injection contrôlée
        latents = latents + strength * detail

        if sanitize_fn:
            latents = sanitize_fn(latents)

        return latents

    except Exception as e:
        print(f"[N3RProNet ERROR] {e}")
        return latents



# Initialiser le glow --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
glow_processor = IntelligentGlowPro(strength=0.18)
adaptive_processor = AdaptivePostProcessor(blur_radius=0.025, denoise_strength=0.03, detail_strength=0.5, contrast_strength=1.22, vibrance_strength=0.22, shadow_lift=0.25, shadow_threshold=0.35)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Post Process
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def full_frame_postprocess(
    frame_pil: Image.Image,
    output_dir: Path,
    frame_counter: int,
    target_temp: int = 7800,
    reference_temp: int = 6500,
    temp_strength: float = 0.15,   # 🔥 légèrement réduit (moins bleu)
    blur_radius: float = 0.025,    # 🔥 un peu moins de blur global
    contrast: float = 1.08,        # 🔥 évite sur-contraste cumulé
    sharpen_percent: int = 90,
    psave: bool = True,
    unreal: bool = False,
    cartoon: bool = False
) -> Image.Image:

    # ---------------- 1️⃣ Input ----------------
    save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="01", psave=psave)

    # ---------------- 2️⃣ Température ----------------
    frame_pil = adjust_color_temperature(
        frame_pil,
        target_temp=target_temp,
        reference_temp=reference_temp,
        strength=temp_strength
    )
    save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="02", psave=psave)

    # ---------------- 3️⃣ Neutralisation (adoucie) ----------------
    frame_pil = neutralize_color_cast(frame_pil, strength=0.6)  # 🔥 clé
    save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="03", psave=psave)

    # ---------------- 4️⃣ Tone mapping (plus doux) ----------------
    frame_pil = soft_tone_map(frame_pil)
    save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="04", psave=psave)

    # ---------------- 5️⃣ Adaptive (nettoyage + micro boost) ----------------
    frame_pil = adaptive_processor.process(frame_pil, frame_counter=frame_counter)
    save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="05", psave=psave)

    # ---------------- 6️⃣ Stylisation ----------------
    if frame_counter < 2:
        if unreal:
            frame_pil = apply_post_processing_unreal_cinematic(frame_pil)
            frame_pil = smooth_edges(frame_pil, strength=0.30, blur_radius=0.8)  # 🔥 moins destructif
            save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="06", psave=psave)

        elif cartoon:
            frame_pil = apply_post_processing_sketch(frame_pil)
            save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="07", psave=psave)

    # ---------------- 7️⃣ Glow intelligent (rééquilibré) ----------------
    # 🔥 strength = moins agressif edge_weight=0.6 # 🔥 edge_weight = priorise edges luminance_weight=0.8 # 🔥 luminance_weight = glow sur zones lumineuses
    frame_pil = glow_processor(frame_pil, frame_counter)
    # ---------------- 8️⃣ Mini post-denoise léger ----------------
    # 🔹 Légère réduction de micro-bruit sur frames initiales
    if frame_counter < 2:
        frame_pil = post_denoise_light(frame_pil, radius=0.5, strength=0.25)

    # 🔥 micro contraste FINAL (après glow → très important)
    frame_pil = ImageEnhance.Contrast(frame_pil).enhance(1.04)

    save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="09", psave=psave)

    return frame_pil
