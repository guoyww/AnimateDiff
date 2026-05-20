#n3rPoseModule.py
# ✔ energy field
# ✔ intent vector (mémoire directionnelle)
# ✔ angular velocity (physique)
# ✔ inertia / damping
# ✔ pivot body (centre de rotation)
# ✔ head lag (cinematic delay)
# ✔ gaze system (attention model)
# ✔ body segmentation (upper / hips / head)
#cinematic_motion_graph_v3, update_motion_state

import torch
import torch.nn.functional as F
import inspect
import math
import traceback
from .n3rMotionPoseClass import Pose

# =========================================================
# UTILS
# =========================================================

def ensure_B1(x, B, device):
    if x is None:
        return torch.zeros((B,1), device=device)

    if not torch.is_tensor(x):
        return torch.full((B,1), float(x), device=device)

    if x.numel() != B:
        return torch.zeros((B,1), device=device)

    return x.view(B,1)


def to_B1(x, B, device):
    if x is None:
        return torch.zeros((B,1), device=device)
    if not torch.is_tensor(x):
        return torch.full((B,1), float(x), device=device)
    if x.dim() == 0:
        return x.view(1,1).expand(B,1)
    if x.dim() == 1:
        return x.view(B,1)
    if x.dim() == 2:
        return x[:, :1]
    return torch.zeros((B,1), device=device)


# =========================================================
# ACTOR FUNCTION MAP (EXTENSIBLE SYSTEM)
# =========================================================

def update_motion_state(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,41,42,43,52,53,54,55,56),
    body_ids=(2,3,4,5,6,7,8,9,10,11,12,13),
    anchor_smooth=0.08,
    velocity_smooth=0.65,
    drift_smooth=0.95,
    head_lock=True,
    head_lock_strength=0.95,
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # INIT
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "velocity": torch.zeros_like(kp[..., :2]),
            "anchor": kp[:, body_ids, :2].mean(dim=1, keepdim=True),
            "drift": torch.zeros((B,1,2), device=device)
        }

    kp_prev = state["kp_prev"]

    if kp_prev.shape != kp.shape:
        kp_prev = kp.clone()

    # =========================================================
    # 1. VELOCITY (TRUE INERTIA CORE)
    # =========================================================
    raw_velocity = kp[..., :2] - kp_prev[..., :2]

    velocity = (
        state["velocity"] * velocity_smooth +
        raw_velocity * (1.0 - velocity_smooth)
    )

    state["velocity"] = velocity

    # =========================================================
    # 2. ANCHOR (VERY SLOW CAMERA MEMORY)
    # =========================================================
    anchor_prev = state["anchor"]
    anchor_now = kp[:, body_ids, :2].mean(dim=1, keepdim=True)
    anchor = ( anchor_prev * (1.0 - anchor_smooth) + anchor_now * anchor_smooth )
    state["anchor"] = anchor

    # =========================================================
    # 3. DRIFT (REAL LOW-PASS FILTER - SAFE) - stabilisation caméra
    # =========================================================
    drift_raw = anchor_now - anchor_prev  # (B,1,2)

    # init safe
    if "drift" not in state or not torch.is_tensor(state["drift"]):
        state["drift"] = torch.zeros_like(drift_raw)

    # sécurité shape (très important dans ton pipeline)
    if state["drift"].shape != drift_raw.shape:
        state["drift"] = torch.zeros_like(drift_raw)

    # EMA drift (low-pass réel)
    drift = ( state["drift"] * drift_smooth + drift_raw * (1.0 - drift_smooth) )

    state["drift"] = drift

    # =========================================================
    # 4. RECONSTRUCT MOTION FROM PREVIOUS STATE
    # =========================================================
    kp_out = kp_prev.clone()

    # apply inertia motion
    kp_out[..., :2] = kp_prev[..., :2] + velocity

    # apply drift compensation (camera stabilization)
    kp_out[..., :2] = kp_out[..., :2] - drift

    # =========================================================
    # 5. BLEND WITH CURRENT INPUT (CRUCIAL)
    # =========================================================
    # gives responsiveness without instability
    kp_out[..., :2] = ( kp_out[..., :2] * 0.7 + kp[..., :2] * 0.3 )

    # =========================================================
    # 6. HEAD LOCK (STABLE BUT NOT FROZEN)
    # =========================================================
    if head_lock:
        kp_out[:, head_ids, :2] = (
            kp_prev[:, head_ids, :2] * head_lock_strength +
            kp_out[:, head_ids, :2] * (1.0 - head_lock_strength)
        )

    # =========================================================
    # 7. FINAL DAMPING (ANTI-JITTER)
    # =========================================================
    kp_out[..., :2] = kp_prev[..., :2] + (
        kp_out[..., :2] - kp_prev[..., :2]
    ) * 0.9

    # =========================================================
    # 8. CLEAN
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 9. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎯 MOTION STATE STABLE] - velocity, drift, reconstruction, blend, head lock, damping")
        print(f"velocity: {velocity.norm(dim=-1).mean().item():.6f}")
        print(f"drift: {drift.norm(dim=-1).mean().item():.6f}")
        print(f"anchor: {anchor.mean().item():.6f}")

    return kp_out, state

ACTOR_LABELS = {
    "base": "[BASE MOTION]",
    "v6": "[V6 CINEMATIC]",
    "v7": "[V7 PHYSICS LAYER]",
    "v8": "[V8 ACTING SYSTEM]",
    "v9": "[V9 FULL ACTOR SYSTEM]",
    "pause": "[V10 PAUSE INTER]",
    "v14": "[V14 ROTATION Z]",
    "v12": "[V12 SMALL ROTATION]",
}


ACTOR_MODEL_SCHEDULE_test = [
    (0, "pause"),
    (1, "v12"),
    (15, "v14"),

]

ACTOR_MODEL_SCHEDULE = [
    (0, "pause"),
    (12,  "base"),
    (15,  "v9"),
    (19, "pause"),
    (25, "v14"), #OK
    (30,  "v12"), #OK
    (35, "pause"),
    (40, "v7"), #OK
    (45, "v8"), #OK
    (50, "v6"), # a verifier
]

# =========================================================
# ACTOR MODEL RESOLUTION (DIRECTOR SYSTEM)
# =========================================================

MOTION_MODEL_SCHEDULE = [
    (0,  "locked"),
    (15, "stable"),
    (20, "cinematic"),
    (30, "locked"),
    (35, "dynamic"),
    (50, "warp"),
]

def resolve_motion_model(frame_idx: int) -> str:
    """
    Motion system:
    controls physics behavior over time (camera + velocity + warp).
    """
    model = "locked"

    for threshold, name in MOTION_MODEL_SCHEDULE:
        if frame_idx >= threshold:
            model = name

    return model



"""
Motion Graph v5 – Cinematic Rig System”
spine inertiel
head lock caméra optionnel
breathing séparé propre
full skeleton graph (pas seulement bras)
compatible diffusion latents (anti-blur garanti)

“Cinematic Motion Graph Engine”
rotation du buste (pseudo 3D)
inertie physique (spring system)
shoulders follow pelvis rotation
breathing sync multi-phase
walking cycle blending propre

ou

⚙️ “Production Stable ControlNet Layer”
zéro drift garanti
compatible batch inference
stable sur longues séquences (200+ frames)
"""


def compute_torso_rotation_delta(kp, prev_vec=None, eps=1e-6):
    """
    Stable torso rotation estimation with robustness to missing hips.
    Returns:
        angle: (B, 1, 1)
        torso_vec: (B, 2)
    """

    # =========================================================
    # 1. EXTRACT KEYPOINTS
    # =========================================================
    l_sh = kp[:, 5, :2]
    r_sh = kp[:, 6, :2]
    l_hp = kp[:, 11, :2]
    r_hp = kp[:, 12, :2]

    # =========================================================
    # 2. BUILD VECTORS
    # =========================================================
    shoulder_vec = r_sh - l_sh
    hip_vec = r_hp - l_hp

    # robust validity check (not just finite)
    def is_valid(v):
        return (
            torch.isfinite(v).all(dim=-1, keepdim=True) &
            (torch.norm(v, dim=-1, keepdim=True) > 1e-4)
        )

    shoulder_ok = is_valid(shoulder_vec)
    hip_ok = is_valid(hip_vec)

    # =========================================================
    # 3. SMART FUSION (IMPORTANT)
    # =========================================================
    # cases:
    # - both valid → average
    # - only shoulder → fallback
    # - only hip → rare but handle
    # - none → zero (safe)
    torso_vec = torch.zeros_like(shoulder_vec)

    both = shoulder_ok & hip_ok
    torso_vec = torch.where(both, (shoulder_vec + hip_vec) * 0.5, torso_vec)
    torso_vec = torch.where(shoulder_ok & ~hip_ok, shoulder_vec, torso_vec)
    torso_vec = torch.where(~shoulder_ok & hip_ok, hip_vec, torso_vec)

    # =========================================================
    # 4. NORMALIZE SAFELY
    # =========================================================
    norm = torch.norm(torso_vec, dim=-1, keepdim=True)
    torso_vec = torso_vec / (norm + eps)

    # =========================================================
    # 5. INIT CASE
    # =========================================================
    if prev_vec is None:
        zero_angle = torch.zeros((kp.shape[0], 1, 1), device=kp.device)
        return zero_angle, torso_vec

    prev_vec = F.normalize(prev_vec, dim=-1)

    # =========================================================
    # 6. ANGLE COMPUTATION
    # =========================================================
    cross = torso_vec[..., 0] * prev_vec[..., 1] - torso_vec[..., 1] * prev_vec[..., 0]
    dot = (torso_vec * prev_vec).sum(dim=-1)

    angle = torch.atan2(cross, dot)

    # =========================================================
    # 7. STABILITY SAFETY
    # =========================================================
    angle = torch.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)

    # clamp violent jumps (important for video stability)
    angle = torch.clamp(angle, -1.0, 1.0)

    return angle.unsqueeze(-1).unsqueeze(-1), torso_vec

#---------------------------------------------------------------------





def rotate_points_around_pivot(points, pivot, angle, name="ROT", debug=None):
    """
    2D rotation (stable, batch-safe) + debug
    """

    s = torch.sin(angle)
    c = torch.cos(angle)
    if debug:
        print(f"[{name}] angle: {angle}")
        print(f"[{name}] pivot: {pivot}")
        print(f"[{name}] points shape: {points.shape}")

    # translate
    p = points - pivot
    if debug:
        print(f"[{name}] pre-translate sample: {p[0] if p.ndim > 1 else p}")

    x = p[..., 0]
    y = p[..., 1]

    x_new = x * c - y * s
    y_new = x * s + y * c

    out = torch.stack([x_new, y_new], dim=-1) + pivot
    if debug:
        print(f"[{name}] post-rotate sample: {out[0] if out.ndim > 1 else out}")

    return out

# ---------------------------------------------------------
# utils debug
# ---------------------------------------------------------
def _dbg(debug, *args):
    if debug:
        print(*args)




# ---------------------------------------------------------
# MAIN MOTION GRAPH (FIXED)
# ---------------------------------------------------------



def fish_eye_distortion(kp, pivot, strength=1.5, max_radius=1.0):
    """
    Applique une distorsion fish-eye uniquement sur l'axe X des points autour du pivot,
    tout en maintenant l'axe Y inchangé.

    kp : [B, N, 2] - Keypoints à distordre (x, y)
    pivot : [B, 1, 2] - Le centre de distorsion (généralement le centre de la caméra)
    strength : Intensité de l'effet fish-eye
    max_radius : Rayon maximal à partir du pivot pour appliquer la distorsion

    Retourne les points distordus.
    """
    B, N, _ = kp.shape
    kp_centered = kp - pivot  # Centrer les points par rapport au pivot

    # Calculer la distance de chaque point par rapport au centre
    dist = torch.norm(kp_centered, dim=-1, keepdim=True)  # [B, N, 1]
    dist = torch.clamp(dist, max=max_radius)  # Limiter la distance au rayon max

    # Appliquer la distorsion fish-eye uniquement sur l'axe X
    factor = 1 + strength * dist ** 2  # Facteur de distorsion basé sur la distance radiale

    # Appliquer la distorsion uniquement à l'axe X (garder Y inchangé)
    kp_centered[:, :, 0] = kp_centered[:, :, 0] * factor[:, :, 0]  # Appliquer sur X seulement

    # Revenir aux coordonnées d'origine (ajuster X, mais laisser Y inchangé)
    kp_out = kp_centered + pivot  # Recentrer les points distordus

    return kp_out


def cinematic_motion_graph_v12(
    kp,
    state,
    head_ids=(0, 1, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43, 52, 53, 54, 55, 56),
    upper_ids=(5, 6, 7, 8, 9, 10),
    hip_ids=(11, 12),
    debug=False,
    rotation_strength=0.5,  # Augmenter légèrement la force de la rotation Z
    motion_sensitivity=4.0,
    rotation_damping=0.85,  # Réduire le damping pour moins de perte d'angle au fil du temps
    inertia=0.85,
    head_lag=0.7, #OK
    max_deg=1.0, #"OK"
    apply_head_lag=True
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # SAFE INIT (NO SHAPE DRIFT EVER)
    # =========================================================
    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())

    state["energy"] = to_B1(state.get("energy", 0.0), B, device=device)
    state["inertia"] = state.get("inertia", 0.75)
    state["intent_vec"] = state.get("intent_vec", None)
    state.setdefault("angle", torch.zeros((B, 1), device=device))
    state.setdefault("angle_vel", torch.zeros((B, 1), device=device))

    # =========================================================
    # 1. MOTION ENERGY (STABLE)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion_vec = delta.mean(dim=1, keepdim=True)  # (B,1,2)

    energy_raw = torch.norm(motion_vec, dim=-1, keepdim=True)  # (B,1,1)
    energy_raw = energy_raw[:, :, 0]  # → (B,1)

    state["energy"] = torch.clamp(
        state["energy"] * state["inertia"] + energy_raw * (1.0 - state["inertia"]),
        min=0.0,
        max=1.0
    )
    energy = state["energy"]  # (B,1)

    # =========================================================
    # 2. INTENT VECTOR (SAFE)
    # =========================================================
    if state["intent_vec"] is None:
        state["intent_vec"] = motion_vec
    else:
        alpha_intent = 0.85 + 0.1 * state["energy"]
        state["intent_vec"] = state["intent_vec"] * alpha_intent + motion_vec * (1.0 - alpha_intent)

    intent = state["intent_vec"]  # (B,1,2)

    # =========================================================
    # 3. UPDATE ROTATION ANGLES (ONLY UPDATE GLOBAL ANGLES)
    # =========================================================
    intent_angle = torch.atan2(
        intent[..., 1],
        intent[..., 0] + 1e-6
    )

    # -----------------------------
    # ENERGY GAIN (soft, non explosif)
    # -----------------------------
    energy = state["energy"].clamp(0.0, 1.0)

    gain = 0.25 + 0.9 * energy   # 🔥 réduit fortement

    target_angle = intent_angle * gain

    # -----------------------------
    # HARD LIMIT (safe cinematic range)
    # -----------------------------
    max_angle = math.radians(max_deg)  # Utilisation de max_deg ici
    target_angle = torch.clamp(target_angle, -max_angle, max_angle)

    # -----------------------------
    # ANGLE SMOOTHING (IMPORTANT FIX)
    # -----------------------------
    prev_angle = state["angle"]

    # EMA smoothing (critical stability)
    alpha = 0.95  # Smooth more to avoid large first frame jumps
    angle = prev_angle * alpha + target_angle * (1.0 - alpha)

    # -----------------------------
    # ANGULAR VELOCITY CONTROLLED
    # -----------------------------
    angle_vel = angle - prev_angle

    # clamp velocity (anti-jerk)
    angle_vel = torch.clamp(angle_vel, -0.05, 0.05)

    state["angle_vel"] = angle_vel
    state["angle"] = angle

    # -----------------------------
    # ROTATION DAMPING PER FRAME
    # -----------------------------
    angle = angle * rotation_damping

    # =========================================================
    # UPDATE GLOBAL ANGLES (ONLY UPDATE STATE ANGLES)
    # =========================================================
    state["global_angle_x"] += angle.mean().item() * rotation_strength
    state["global_angle_y"] += angle.mean().item() * rotation_strength
    state["global_angle_z"] += angle.mean().item() * rotation_strength

    # =========================================================
    # APPLY HEAD LAG (OPTIONAL)
    # =========================================================
    if apply_head_lag:
        kp_out = kp.clone()
        kp_out[:, head_ids, :2] = (
            kp_prev[:, head_ids, :2] * head_lag +
            kp_out[:, head_ids, :2] * (1.0 - head_lag)
        )
    else:
        kp_out = kp.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 V12 ADVANCED MOVEMENT]")
        print("energy:", energy.mean().item())
        print("intent:", intent.norm(dim=-1).mean().item())
        print("angle:", angle.mean().item())
        print("motion:", energy_raw.mean().item())

    # =========================================================
    # 9. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # Return updated keypoints and state
    return kp_out, state




def cinematic_motion_graph_v14(
    kp,
    state,

    head_ids=(0,1,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,41,42,43,52,53,54,55,56),
    upper_ids=(5,6,7,8,9,10),
    hip_ids=(11,12),

    # =========================
    # TOGGLES
    # =========================
    enable_rotation=True,
    enable_inertia=True,
    enable_handheld=True,
    enable_breathing=True,
    enable_overshoot=True,

    # =========================
    # PARAMETERS
    # =========================
    max_deg=1.0,
    rotation_strength=0.5,
    motion_sensitivity=2.5,

    inertia=0.80,
    damping=0.90,

    overshoot_strength=0.25,
    handheld_strength=0.0025,
    breathing_strength=0.0015,

    head_lag=0.65,
    lock_y=True,

    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # INIT STATE
    # =========================================================
    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())

    angle = ensure_B1(state.get("angle"), B, device)
    angular_vel = ensure_B1(state.get("angular_vel"), B, device)
    prev_target = ensure_B1(state.get("prev_target"), B, device)

    t = state.get("time", 0.0) + 1.0
    state["time"] = t

    # =========================================================
    # 1. MOTION ENERGY (clean + stable)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]

    motion = torch.norm(delta, dim=-1)

    w = torch.ones_like(motion)
    w[:, upper_ids] *= 1.5
    w[:, hip_ids] *= 1.2

    motion_energy = (motion * w).mean(dim=1, keepdim=True)
    motion_gain = torch.tanh(motion_energy * motion_sensitivity)

    # =========================================================
    # 2. INTENT (STABLE VERSION 🔥)
    # =========================================================
    motion_vec = (kp[:, upper_ids, :2] - kp_prev[:, upper_ids, :2]).mean(dim=1)

    intent_angle = torch.atan2(
        motion_vec[..., 1],
        motion_vec[..., 0] + 1e-6
    ).unsqueeze(-1)

    target_angle = intent_angle * rotation_strength * motion_gain

    # clamp direct (important)
    max_angle = math.radians(max_deg)
    target_angle = torch.clamp(target_angle, -max_angle, max_angle)

    # =========================================================
    # 3. OVERSHOOT (cinematic effect)
    # =========================================================
    if enable_overshoot:
        delta_target = target_angle - prev_target
        target_angle = target_angle + delta_target * overshoot_strength

    state["prev_target"] = target_angle

    # =========================================================
    # 4. PHYSICS ROTATION (REALISTIC 🔥)
    # =========================================================
    if enable_rotation:

        angular_acc = target_angle - angle

        if enable_inertia:
            angular_vel = angular_vel * inertia + angular_acc * (1 - inertia)
        else:
            angular_vel = angular_acc

        angular_vel = angular_vel * damping
        angle = angle + angular_vel

        angle = torch.clamp(angle, -max_angle, max_angle)

    # =========================================================
    # 5. PIVOT (stable body center)
    # =========================================================
    upper = kp[:, upper_ids, :2].mean(dim=1, keepdim=True)
    hips  = kp[:, hip_ids, :2].mean(dim=1, keepdim=True)

    pivot = upper * 0.35 + hips * 0.65
    pivot = torch.nan_to_num(pivot)

    # =========================================================
    # 6. APPLY ROTATION (FIXED)
    # =========================================================
    kp_out = kp.clone()

    if enable_rotation:

        angle = torch.nan_to_num(angle)
        angle = torch.clamp(angle, -math.radians(max_deg), math.radians(max_deg))
        if debug:
            print("ANGLE SHAPE:", angle.shape, "NUMEL:", angle.numel())

        angle_safe = angle.view(B,1,1)
        pivot_safe = pivot.view(B,1,2)


        kp_out[:, upper_ids, :2] = Pose.rotate_points_around_z_axis(
            kp[:, upper_ids, :2],
            pivot_safe,
            angle_safe
        )

    # =========================================================
    # 7. HANDHELD CAMERA (REALISTIC MICRO SHAKE)
    # =========================================================
    if enable_handheld:
        t_tensor = torch.tensor(t, device=device)

        noise_x = (
            torch.sin(t_tensor * 0.6) +
            torch.sin(t_tensor * 1.3) * 0.5
        ) * handheld_strength

        kp_out[..., 0] += noise_x

        if not lock_y:
            noise_y = torch.cos(t_tensor * 0.9) * handheld_strength
            kp_out[..., 1] += noise_y

    # =========================================================
    # 8. BREATHING
    # =========================================================
    if enable_breathing and not lock_y:
        t_tensor = torch.tensor(t, device=device)
        breath = torch.sin(t_tensor * 0.2) * breathing_strength
        kp_out[:, upper_ids, 1] += breath

    # =========================================================
    # 9. HEAD LAG
    # =========================================================
    kp_out[:, head_ids, :2] = (
        kp_prev[:, head_ids, :2] * head_lag +
        kp_out[:, head_ids, :2] * (1 - head_lag)
    )

    # =========================================================
    # 10. GLOBAL SMOOTH (anti jitter)
    # =========================================================
    alpha = 0.80 + 0.10 * (1.0 - motion_gain)

    kp_out[..., :2] = kp_prev[..., :2] + (
        kp_out[..., :2] - kp_prev[..., :2]
    ) * alpha

    # =========================================================
    # 11. LOCK Y (clean)
    # =========================================================
    if lock_y:
        kp_out[..., 1] = kp_prev[..., 1]

    # =========================================================
    # CLEAN
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2])
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()
    state["angle"] = angle
    state["angular_vel"] = angular_vel

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 V14 CINEMATIC CAMERA]")
        print("motion:", motion_energy.mean().item())
        print("target:", target_angle.mean().item())
        print("angle:", angle.mean().item())
        print("vel:", angular_vel.mean().item())

    return kp_out, state



def cinematic_motion_graph_v10_pause(
    kp,
    state,
    head_ids=(0, 1, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40, 41, 42, 43, 52, 53, 54, 55, 56),
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. SAFE INIT (NO SHAPE DRIFT EVER)
    # =========================================================
    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())

    state["energy"] = to_B1(state.get("energy", 0.0), B, device=device)
    state["inertia"] = state.get("inertia", 0.75)
    state["intent_vec"] = state.get("intent_vec", None)
    state.setdefault("angle", torch.zeros((B, 1), device=device))
    state.setdefault("angle_vel", torch.zeros((B, 1), device=device))

    # =========================================================
    # 1. MOTION ENERGY (STABLE)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion_vec = delta.mean(dim=1, keepdim=True)  # (B, 1, 2)

    energy_raw = torch.norm(motion_vec, dim=-1, keepdim=True)  # (B, 1, 1)
    energy_raw = energy_raw[:, :, 0]  # → (B, 1)

    # Appliquer une régulation de l'énergie plus douce et dynamique
    state["energy"] = torch.clamp(
        state["energy"] * state["inertia"] + energy_raw * (1.0 - state["inertia"]),
        min=0.0,
        max=1.0
    )

    energy = state["energy"]  # (B, 1)

    # =========================================================
    # 2. INTENT VECTOR (NO MOVEMENT)
    # =========================================================
    # Ne pas appliquer de changement sur le vecteur d'intention
    # On garde simplement la position précédente
    state["intent_vec"] = torch.zeros_like(motion_vec)

    # L'intention est nulle, donc pas de changement de direction
    intent = state["intent_vec"]  # (B, 1, 2)

    # =========================================================
    # 3. NO ROTATION (NO MOVEMENT)
    # =========================================================
    # Nous n'appliquons aucune rotation ici
    # Le vecteur d'intention étant nul, aucune rotation ne se produit

    target_angle = torch.zeros_like(energy)  # Pas de changement d'angle

    # =========================================================
    # 4. PIVOT (NO MOVEMENT)
    # =========================================================
    # Le pivot reste fixe, basé sur la dernière position
    upper = kp[:, 5:11, :2].mean(dim=1, keepdim=True)
    lower = kp[:, 11:13, :2].mean(dim=1, keepdim=True)

    pivot = upper * 0.35 + lower * 0.65

    # Ne pas ajuster le pivot, le garder stable
    state.setdefault("pivot_prev", pivot)

    # =========================================================
    # 5. APPLY NO ROTATION (SIMPLE PAUSE)
    # =========================================================
    # Ici, on ne modifie pas les keypoints
    kp_out = kp.clone()

    # Les keypoints ne changent pas. Le processus de pose ne fait qu'appliquer la stabilité
    kp_out = kp_prev.clone()

    # =========================================================
    # 6. HEAD STABILITY - **AUCUN MOUVEMENT DE TÊTE**
    # =========================================================
    # Cette section est désormais complètement désactivée
    # kp_out[:, head_ids, :2] = kp_prev[:, head_ids, :2]
    # Le mouvement de la tête est totalement désactivé.

    # =========================================================
    # 7. CLEAN OUTPUT
    # =========================================================
    # On s'assure que les keypoints sont dans la plage [0, 1]
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 8. UPDATE STATE
    # =========================================================
    # Conserver la position de l'état sans changer la position
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 V10 PAUSE TRANSITION]")
        print("energy:", energy.mean().item())
        print("intent:", intent.norm(dim=-1).mean().item())
        print("pivot:", state["pivot_prev"][0, 0].tolist())

    return kp_out, state



def actor_system_v9(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,41,42,43,52,53,54,55,56),
    body_ids=(2,3,4,5,6,7,8,9,10,11,12,13),
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. SAFE INIT (CRASH FIX GLOBAL)
    # =========================================================
    if state is None:
        state = {}

    def init_tensor(key, shape):
        if key not in state or not torch.is_tensor(state[key]):
            state[key] = torch.zeros(shape, device=device)

    init_tensor("kp_prev", (B, N, 2))
    init_tensor("energy", (B, 1, 1))
    init_tensor("emotion", (B, 1, 1))
    init_tensor("intent_vec", (B, 1, 2))
    init_tensor("attention", (B, 1, 2))
    init_tensor("angular_vel", (B, 1, 1))
    init_tensor("angle", (B, 1, 1))
    init_tensor("angle_vel", (B, 1, 1))

    state.setdefault("gaze_target", None)
    state.setdefault("gaze_jitter", 0.001)
    state.setdefault("inertia", 0.9)

    kp_prev = state["kp_prev"]

    # =========================================================
    # 1. MOTION ENERGY (SAFE)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion_vec = delta.mean(dim=1, keepdim=True)

    energy_raw = torch.norm(motion_vec, dim=-1, keepdim=True)

    #state["energy"] = ( state["energy"] * state["inertia"] + energy_raw * (1.0 - state["inertia"]) )
    # Normalisation de l'énergie avec une régulation
    state["energy"] = torch.clamp(
        state["energy"] * state["inertia"] + energy_raw * (1.0 - state["inertia"]),
        min=0.0,
        max=1.0
    )

    # =========================================================
    # 2. INTENT VECTOR
    # =========================================================
    #state["intent_vec"] = ( state["intent_vec"] * 0.85 + motion_vec * 0.15 )
    # Moduler l'intensité de l'intention en fonction de l'énergie ou d'autres facteurs
    state["intent_vec"] = state["intent_vec"] * 0.85 + motion_vec * (0.15 * state["energy"])

    intent = state["intent_vec"]

    # =========================================================
    # 3. EMOTION
    # =========================================================
    emotion_target = torch.clamp(state["energy"] * 3.0, 0.0, 1.0)

    state["emotion"] = (
        state["emotion"] * 0.92 +
        emotion_target * 0.08
    )

    # =========================================================
    # 4. GAZE ENGINE (SAFE)
    # =========================================================
    eye_center = kp[:, [1,2], :2].mean(dim=1, keepdim=True)

    if state["gaze_target"] is None:
        state["gaze_target"] = eye_center.clone()

    drift = intent * 0.03

    state["gaze_target"] = (
        state["gaze_target"] * 0.9 +
        (eye_center + drift) * 0.1
    )

    gaze_dir = state["gaze_target"] - eye_center
    gaze_dir = gaze_dir + torch.randn_like(gaze_dir) * state["gaze_jitter"]

    # =========================================================
    # 5. FACE MOTION (SAFE INDEX CHECK)
    # =========================================================
    emotion = state["emotion"]

    mouth_open = emotion * 0.02

    if N > 14:
        kp[:, 14, 1] = kp[:, 14, 1] + mouth_open.squeeze(-1)

    kp[:, 1:3, :2] = kp[:, 1:3, :2] + gaze_dir * 0.15 * emotion

    # =========================================================
    # 6. BODY ROTATION (STABLE CINEMATIC VERSION)
    # =========================================================

    intent_angle = torch.atan2(
        intent[..., 1],
        intent[..., 0] + 1e-6
    )

    # -----------------------------
    # ENERGY GAIN (soft, non explosif)
    # -----------------------------
    energy = state["energy"].clamp(0.0, 1.0)

    gain = 0.25 + 0.9 * energy   # 🔥 réduit fortement

    target_angle = intent_angle * gain

    # -----------------------------
    # HARD LIMIT (safe cinematic range)
    # -----------------------------
    max_angle = math.radians(18)  # 🔥 réduit de 25 → 18
    target_angle = torch.clamp(target_angle, -max_angle, max_angle)

    # -----------------------------
    # ANGLE SMOOTHING (IMPORTANT FIX)
    # -----------------------------
    prev_angle = state["angle"]

    # EMA smoothing (critical stability)
    alpha = 0.85
    angle = prev_angle * alpha + target_angle * (1.0 - alpha)

    # -----------------------------
    # ANGULAR VELOCITY CONTROLLED
    # -----------------------------
    angle_vel = angle - prev_angle

    # clamp velocity (anti-jerk)
    angle_vel = torch.clamp(angle_vel, -0.04, 0.04)

    state["angle_vel"] = angle_vel
    state["angle"] = angle

    # -----------------------------
    # ROTATION DAMPING PER FRAME
    # -----------------------------
    rotation_damping = 0.92
    angle = angle * rotation_damping

    # =========================================================
    # APPLY ROTATION
    # =========================================================
    # Liste des indices des points à faire pivoter
    # Création de l'instance de la classe Pose
    kp_out = kp.clone()

    # Créer une instance de la classe Pose avec les keypoints
    pose = Pose(kp_out)  # on passe les keypoints ici

    # Calculer l'angle de rotation (par exemple)
    angle = torch.tensor(0.1)  # Remplacer par l'angle réel en radians

    # Appliquer la rotation
    pose.apply_rotation(angle)

    # Les keypoints après rotation sont maintenant stockés dans pose.keypoints

    # =========================================================
    # 7. MICRO NOISE (SAFE)
    # =========================================================
    #noise = torch.randn_like(kp[..., :2]) * 0.0012
    #kp[..., :2] = kp[..., :2] + noise * state["energy"]

    # Ajuster l'échelle du bruit en fonction de l'énergie, avec une fonction logarithmique
    noise_scale = torch.log1p(state["energy"]) * 0.0012  # Logarithmique pour plus de contrôle
    noise = torch.randn_like(kp[..., :2]) * noise_scale
    kp[..., :2] = kp[..., :2] + noise

    # =========================================================
    # 8. HEAD STABILITY
    # =========================================================
    kp[:, head_ids, :2] = (
        kp[:, head_ids, :2] * 0.965 +
        kp_prev[:, head_ids, :2] * 0.035
    )

    # =========================================================
    # 9. CLEAN OUTPUT
    # =========================================================
    kp[..., :2] = torch.nan_to_num(kp[..., :2], nan=0.0)
    kp[..., :2] = torch.clamp(kp[..., :2], 0.0, 1.0)

    # =========================================================
    # 10. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎭 V9 SAFE ACTOR]")
        print(f"energy: {state['energy'].mean().item():.6f}")
        print(f"emotion: {state['emotion'].mean().item():.6f}")
        print(f"intent_norm: {intent.norm(dim=-1).mean().item():.6f}")
        print(f"gaze: {state['gaze_target'][0,0].tolist()}")

    return kp, state



def cinematic_motion_graph_v8(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,41,42,43,52,53,54,55,56),
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. SAFE INIT (NO SHAPE DRIFT EVER)
    # =========================================================

    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())

    state["energy"] = to_B1(state.get("energy", 0.0), B, device=device)
    state["inertia"] = state.get("inertia", 0.75)
    state["intent_vec"] = state.get("intent_vec", None)
    state.setdefault("angle", torch.zeros((B,1), device=device))
    state.setdefault("angle_vel", torch.zeros((B,1), device=device))

    # =========================================================
    # 1. MOTION ENERGY (STABLE)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]

    motion_vec = delta.mean(dim=1, keepdim=True)  # (B,1,2)

    energy_raw = torch.norm(motion_vec, dim=-1, keepdim=True)  # (B,1,1)
    energy_raw = energy_raw[:, :, 0]  # → (B,1)

    # Appliquer une régulation de l'énergie plus douce et dynamique
    state["energy"] = torch.clamp(
        state["energy"] * state["inertia"] + energy_raw * (1.0 - state["inertia"]),
        min=0.0,
        max=1.0
    )

    energy = state["energy"]  # (B,1)

    # =========================================================
    # 2. INTENT VECTOR (SAFE)
    # =========================================================
    if state["intent_vec"] is None:
        state["intent_vec"] = motion_vec
    else:
        alpha_intent = 0.85 + 0.1 * state["energy"]  # Augmenter l'alpha si l'énergie est élevée
        state["intent_vec"] = state["intent_vec"] * alpha_intent + motion_vec * (1.0 - alpha_intent)

    intent = state["intent_vec"]  # (B,1,2)

    # =========================================================
    # 3. ROTATION (REDUCED + STABLE)
    # =========================================================

    raw_angle = torch.atan2(
        intent[..., 1],
        intent[..., 0] + 1e-6
    )  # (B,1)

    # -----------------------------
    # ENERGY GAIN (FORTEMENT RÉDUIT)
    # -----------------------------
    gain = 0.2 + energy * 1.2   # 🔥 réduit drastiquement (avant 2.5)

    target_angle = raw_angle * gain

    # -----------------------------
    # HARD LIMIT (PLUS STRICT)
    # -----------------------------
    max_angle = 0.1  # ~5° → ajusté pour éviter des rotations trop larges
    target_angle = torch.clamp(target_angle, -max_angle, max_angle)

    # -----------------------------
    # ANGLE SMOOTHING (ESSENTIEL)
    # -----------------------------
    prev_angle = state.get("angle", torch.zeros_like(target_angle))

    alpha = 0.95  # Diminution de l'alpha pour des rotations plus douces
    angle = prev_angle * alpha + target_angle * (1 - alpha)

    # -----------------------------
    # LIMIT ANGULAR SPEED (CRUCIAL FIX)
    # -----------------------------
    angle_vel = angle - prev_angle
    angle_vel = torch.clamp(angle_vel, -0.02, 0.02)  # Réduction de la vitesse angulaire pour plus de stabilité

    # Intégration douce
    angle = prev_angle + angle_vel

    # optional damping
    angle = angle * 0.95  # Réduction plus importante de l'amortissement

    state["angle"] = angle
    state["angle_vel"] = angle_vel

    # =========================================================
    # 4. PIVOT (SAFE CENTER OF MASS)
    # =========================================================
    upper = kp[:, 5:11, :2].mean(dim=1, keepdim=True)
    lower = kp[:, 11:13, :2].mean(dim=1, keepdim=True)

    # Ajuster le pivot pour éviter des distorsions
    pivot = upper * 0.4 + lower * 0.6

    invalid = ~torch.isfinite(pivot).all(dim=-1, keepdim=True)
    pivot = torch.where(invalid, upper, pivot)

    # =========================================================
    # 5. ROTATION APPLY (SAFE)
    # =========================================================
    kp_out = kp.clone()

    # Créer une instance de la classe Pose avec les keypoints
    pose = Pose(kp_out)  # on passe les keypoints ici

    # Calculer l'angle de rotation
    angle = angle.view(B, 1)  # Formater l'angle en [B, 1] si nécessaire

    # Appliquer la rotation via la méthode de la classe Pose
    pose.apply_rotation(angle)

    # Les keypoints après rotation sont maintenant dans pose.keypoints
    kp_out = pose.keypoints

    # =========================================================
    # 6. HEAD STABILITY
    # =========================================================
    kp_out[:, head_ids, :2] = (
        kp_out[:, head_ids, :2] * 0.98 +  # Amortissement réduit
        kp_prev[:, head_ids, :2] * 0.02
    )

    # =========================================================
    # 7. CLEAN OUTPUT
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 8. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 V8 SAFE ACTING SYSTEM]")
        print("energy:", energy.mean().item())
        print("intent:", intent.norm(dim=-1).mean().item())
        print("angle:", angle.mean().item())
        print("motion:", energy_raw.mean().item())
        print("pivot:", pivot[0,0].tolist())

    return kp_out, state



def cinematic_motion_graph_v7(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,41,42,43,52,53,54,55,56),
    upper_ids=(5,6,7,8,9,10),
    hip_ids=(11,12),
    rotation_strength=0.5,
    motion_sensitivity=4.0,
    damping=0.90,
    inertia=0.85,
    head_lag=0.7,
    max_deg=5.0,
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. SAFE STATE INIT (UNCHANGED BUT HARDENED)
    # =========================================================

    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())

    angle = to_B1(state.get("angle", 0.0), B, device=device)
    angular_vel = to_B1(state.get("angular_vel", 0.0), B, device=device)

    state["angle"] = angle
    state["angular_vel"] = angular_vel
    state.setdefault("torso_vec_prev", None)

    # =========================================================
    # 1. MOTION ENERGY (STRUCTURED FIX)
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion = torch.norm(delta, dim=-1)  # (B,N)

    # body weighting (minimal but critical fix)
    w = torch.ones_like(motion)

    torso_ids = list(range(5, 11))

    w[:, torso_ids] *= 1.5
    w[:, hip_ids] *= 1.2
    w[:, [0,1]] *= 0.8

    motion_energy = (motion * w).mean(dim=1, keepdim=True)
    motion_gain = torch.tanh(motion_energy * motion_sensitivity)

    # =========================================================
    # 3. PIVOT (MORE STABLE BODY CENTER)
    # =========================================================
    upper_center = kp[:, torso_ids, :2].mean(dim=1, keepdim=True)
    hip_center = kp[:, hip_ids, :2].mean(dim=1, keepdim=True)

    #pivot = upper_center * 0.4 + hip_center * 0.6
    pivot = upper_center * 0.3 + hip_center * 0.7  # Le pivot devient plus centré sur les hanches, réduisant l'impact de la rotation

    valid = torch.isfinite(pivot).all(dim=-1, keepdim=True)
    shoulder_center = kp[:, [5,6], :2].mean(dim=1, keepdim=True)

    pivot = torch.where(valid, pivot, shoulder_center)

    # =========================================================
    # 4. ROTATION (UNCHANGED BUT STABILIZED INPUT)
    # =========================================================
    kp_out = kp.clone()

    strength = rotation_strength * (1.0 + motion_gain)

    angle_final = torch.clamp(
        state["angle"] * strength,
        -max_deg,
        max_deg
    ).view(B,1,1)

    kp_out[:, upper_ids, :2] = rotate_points_around_pivot(
        kp[:, upper_ids, :2],
        pivot,
        angle_final
    )

    # =========================================================
    # 5. HEAD LAG (UNCHANGED)
    # =========================================================
    kp_out[:, head_ids, :2] = (
        kp_prev[:, head_ids, :2] * head_lag +
        kp_out[:, head_ids, :2] * (1.0 - head_lag)
    )

    # =========================================================
    # 6. GLOBAL STABILITY (SLIGHTLY ADAPTIVE FIX)
    # =========================================================
    #alpha = 0.88 + 0.08 * (1.0 - motion_gain)
    alpha = 0.85 + 0.05 * (1.0 - motion_gain)  # Rend le mouvement plus stable et moins réactif

    kp_out[..., :2] = kp_prev[..., :2] + (
        kp_out[..., :2] - kp_prev[..., :2]
    ) * alpha

    # =========================================================
    # 7. CLEAN OUTPUT
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 8. STATE UPDATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # DEBUG (UNCHANGED)
    # =========================================================
    if debug:
        print("\n[🎬 V7 SAFE FIXED]")
        print("motion:", motion_energy.mean().item())
        print("gain:", motion_gain.mean().item())
        print("angle:", state["angle"].mean().item())
        print("vel:", state["angular_vel"].mean().item())
        print("pivot:", pivot[0,0].tolist())

    return kp_out, state



def cinematic_motion_graph_v6(
    kp,
    state,
    head_ids=(0,1,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,41,42,43,52,53,54,55,56),
    body_ids=(2,3,4,5,6,7,8,9,10,11,12,13),
    hip_ids=(11,12),
    rotation_strength=0.5,
    rotation_smooth=0.3,
    motion_sensitivity=3.0,   # 🔥 BOOST IMPORTANT valeur conseillé 4.0
    head_lock=0.95,
    head_lag=0.7,
    max_rotation_deg=20.0,
    debug=False
):
    B, N, _ = kp.shape
    device = kp.device

    # =========================================================
    # 0. INIT STATE SAFE
    # =========================================================
    if state is None:
        return kp, {
            "kp_prev": kp.clone(),
            "angle": torch.zeros((B,1), device=device),
            "torso_vec_prev": None
        }

    kp_prev = state.get("kp_prev", kp.clone())
    state.setdefault("angle", torch.zeros((B,1), device=device))
    state.setdefault("torso_vec_prev", None)

    # =========================================================
    # 1. REAL MOTION ENERGY
    # =========================================================
    delta = kp[..., :2] - kp_prev[..., :2]
    motion_per_joint = torch.norm(delta, dim=-1, keepdim=True)
    motion_score = (motion_per_joint.mean(dim=1, keepdim=True) +
                    motion_per_joint.max(dim=1, keepdim=True).values) * 0.5
    motion_gain = torch.tanh(motion_score * motion_sensitivity)

    # =========================================================
    # 2. TORSO ANGLE (stable + smoothing)
    # =========================================================
    angle_raw, torso_vec = compute_torso_rotation_delta(
        kp,
        state["torso_vec_prev"]
    )
    state["torso_vec_prev"] = torso_vec
    angle_raw = angle_raw.view(B, 1).clamp(-1.2, 1.2)

    state["angle"] = state["angle"] * (1 - rotation_smooth) + angle_raw * rotation_smooth
    angle = state["angle"]


    # =========================================================
    # 4. ROTATION (using Pose class)
    # =========================================================
    pose = Pose(kp.clone())  # Créer une instance de la classe Pose avec les keypoints actuels

    # Calcul de la rotation (angle ajusté par la force de rotation)
    strength = rotation_strength * (1.0 + motion_gain)

    max_rotation_deg_dynamic = max_rotation_deg * torch.clamp(motion_gain, 0.5, 1.0)
    angle_final = torch.clamp(angle * strength, -max_rotation_deg_dynamic, max_rotation_deg_dynamic)

    # Appliquer la rotation avec la méthode de la classe Pose
    pose.apply_rotation(angle_final)

    # Les keypoints après rotation sont stockés dans pose.keypoints
    kp_out = pose.keypoints

    # =========================================================
    # 5. HEAD LAG (UNCHANGED)
    # =========================================================
    kp_out[:, head_ids, :2] = (
        kp_prev[:, head_ids, :2] * head_lag +
        kp_out[:, head_ids, :2] * (1.0 - head_lag)
    )

    # =========================================================
    # 6. FULL BODY SOFT DAMPING
    # =========================================================
    kp_out[..., :2] = kp_prev[..., :2] + (kp_out[..., :2] - kp_prev[..., :2]) * 0.95
    # =========================================================
    # 6. GLOBAL STABILITY (SLIGHTLY ADAPTIVE FIX)
    # =========================================================
    #alpha = 0.88 + 0.08 * (1.0 - motion_gain)
    alpha = 0.85 + 0.05 * (1.0 - motion_gain)  # Rend le mouvement plus stable et moins réactif

    kp_out[..., :2] = kp_prev[..., :2] + (
        kp_out[..., :2] - kp_prev[..., :2]
    ) * alpha

    # =========================================================
    # 7. CLEAN OUTPUT
    # =========================================================
    kp_out[..., :2] = torch.nan_to_num(kp_out[..., :2], nan=0.0)
    kp_out[..., :2] = torch.clamp(kp_out[..., :2], 0.0, 1.0)

    # =========================================================
    # 8. UPDATE STATE
    # =========================================================
    state["kp_prev"] = kp_out.clone()

    # =========================================================
    # 9. DEBUG
    # =========================================================
    if debug:
        print("\n[🎬 CINEMA V6 SAFE]")
        print("motion_score:", motion_score.mean().item())
        print("motion_gain:", motion_gain.mean().item())
        print("angle:", angle.mean().item())
        print("angle_final:", angle_final.mean().item())

    return kp_out, state


def apply_actor_model(kp, state, frame_idx, profile=None, debug=True):

    # =========================
    # SAFE MODEL RESOLUTION
    # =========================
    try:
        rmodel = resolve_actor_model(frame_idx)
    except Exception as e:
        if debug:
            print(f"[⚠ resolve_actor_model failed] {e}")
        rmodel = "v7"

    fn = ACTOR_PIPELINE.get(rmodel, update_motion_state)
    label = ACTOR_LABELS.get(rmodel, "[UNKNOWN MODEL]")

    # =========================
    # DEBUG HEADER
    # =========================
    if debug:
        print("\n[🎬 ACTOR PIPELINE] =====================================")
        print(f"frame_idx : {frame_idx}")
        print(f"model     : {rmodel}")
        print(f"profile   : {profile}")
        print(f"layer     : {label}")
        print("========================================================")

    # =========================
    # SAFE STATE INIT
    # =========================
    if state is None:
        state = {}

    kp_prev = state.get("kp_prev", kp.clone())
    state.setdefault("global_angle_x", 0.0)
    state.setdefault("global_angle_y", 0.0)
    state.setdefault("global_angle_z", 0.0)
    state.setdefault("kp_prev", kp_prev)

    # FIX: velocity always valid shape (B,N,2)
    state.setdefault(
        "velocity",
        torch.zeros_like(kp[..., :2])
    )

    # =========================
    # SAFE ARG INSPECTION
    # (évite crash frame_idx / debug / velocity mismatch)
    # =========================
    try:
        sig = inspect.signature(fn)
        kwargs = {}

        if "frame_idx" in sig.parameters:
            kwargs["frame_idx"] = frame_idx
        if "debug" in sig.parameters:
            kwargs["debug"] = debug
        if "profile" in sig.parameters:
            kwargs["profile"] = profile

        result = fn(kp, state, **kwargs)

    except TypeError as e:
        if debug:
            print(f"[⚠ ACTOR SIGNATURE MISMATCH] {e}")
            print("[↩ fallback update_motion_state]")
        result = update_motion_state(kp, state)

    except Exception as e:
        if debug:
            print(f"[⚠ ACTOR ERROR] model={rmodel} -> fallback base motion")
            print(f"error: {e}")
            traceback.print_exc()
        result = update_motion_state(kp, state)

    # =========================
    # NORMALIZE OUTPUT
    # =========================
    if isinstance(result, tuple):
        kp_out, new_state = result
    else:
        kp_out, new_state = result, state

    global_angle_x = None
    global_angle_y = None
    global_angle_z = None

    # cas 1 : pipeline renvoie 3 valeurs
    if isinstance(result, tuple) and len(result) == 3:
        kp_out, new_state, global_angle = result

    # cas 2 : angle déjà dans state
    if isinstance(new_state, dict):
        if "global_angle_x" in new_state:
            global_angle_x = new_state["global_angle_x"]
        if "global_angle_y" in new_state:
            global_angle_y = new_state["global_angle_y"]
        if "global_angle_z" in new_state:
            global_angle_z = new_state["global_angle_z"]

    # cas 3 : si global_angle_x, global_angle_y, global_angle_z sont tous absents
    if global_angle_x is None or global_angle_y is None or global_angle_z is None:
        # Calculer les angles de rotation ici, par exemple à partir de la pose actuelle et précédente
        global_angle_x, global_angle_y, global_angle_z = compute_rotation_angles(kp, state, kp_prev)

    # =========================
    # Apply Temporal Smoothing (EMA) for Each Angle
    # =========================
    if global_angle_x is not None:
        prev_x = state.get("global_angle_x", 0.0)
        smooth_x = 0.9 * prev_x + 0.1 * global_angle_x
        new_state["global_angle_x"] = smooth_x

    if global_angle_y is not None:
        prev_y = state.get("global_angle_y", 0.0)
        smooth_y = 0.9 * prev_y + 0.1 * global_angle_y
        new_state["global_angle_y"] = smooth_y

    if global_angle_z is not None:
        prev_z = state.get("global_angle_z", 0.0)
        smooth_z = 0.9 * prev_z + 0.1 * global_angle_z
        new_state["global_angle_z"] = smooth_z

    # =========================
    # SAFE STATE UPDATE (IMPORTANT FIX)
    # =========================
    new_state["kp_prev"] = kp_out.clone()

    # FIX: avoid velocity/angle crash in downstream v6/v7
    if "velocity" not in new_state:
        new_state["velocity"] = torch.zeros_like(kp_out[..., :2])

    if "angle" not in new_state:
        new_state["angle"] = 0.0

    if "global_angle_x" not in new_state:
        new_state["global_angle_x"] = state.get("global_angle_x", 0.0)

    if "global_angle_y" not in new_state:
        new_state["global_angle_y"] = state.get("global_angle_y", 0.0)

    if "global_angle_z" not in new_state:
        new_state["global_angle_z"] = state.get("global_angle_z", 0.0)

    # =========================
    # DEBUG MOTION
    # =========================
    if kp_prev is not None:
        delta = (kp_out[..., :2] - kp_prev[..., :2]).abs().mean().item()
        print(f"[DEBUG] motion_delta_mean: {delta:.6f}")

    return kp_out, new_state



def resolve_actor_model(frame_idx: int) -> str:
    """
    Director timeline system:
    progressive acting complexity over time.
    """
    model = "v9"

    for threshold, name in ACTOR_MODEL_SCHEDULE:
        if frame_idx >= threshold:
            model = name

    return model


ACTOR_PIPELINE = {
    "base": update_motion_state,
    "v6": cinematic_motion_graph_v6,
    "v7": cinematic_motion_graph_v7,
    "v8": cinematic_motion_graph_v8,
    "v9": actor_system_v9,
    "pause": cinematic_motion_graph_v10_pause,
    "v14": cinematic_motion_graph_v14,
    "v12": cinematic_motion_graph_v12,
}

