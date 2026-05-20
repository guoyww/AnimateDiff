#n3rMotionBreathing.py
import torch
import torch.nn.functional as F
import os
import math
import numpy as np
from .n3rMotionPose_tools import save_impact_map
from .n3rMotionPoseClass import Pose


def apply_breathing(
    latents_world,
    pose,
    mask_torso_exp,
    frame_counter,
    breathing,
    debug=False,
    debug_dir=None,
):
    #if mode == "real":
    if frame_counter % 2 == 0:
        return apply_breathing_real(latents_world, pose, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir)
    elif frame_counter % 3 == 0:
        return apply_breathing_simple_anime(latents_world, pose, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir)
    else:
        return apply_breathing_soft(latents_world, pose, mask_torso_exp, frame_counter, breathing, debug=debug, debug_dir=debug_dir)


def apply_breathing_xy(latents, previous_latent, frame_counter, breathing=True):
    """
    Applique une respiration réaliste sur les latents : décalage vertical centré.
    """
    if previous_latent is not None and breathing:
        B, C, H, W = latents.shape
        breath = 0.004 * math.sin(frame_counter * 0.15)  # amplitude réduite
        # créer un shift vertical uniforme
        yy, xx = torch.meshgrid(torch.linspace(-1,1,H,device=latents.device),
                                torch.linspace(-1,1,W,device=latents.device), indexing='ij')
        grid = torch.stack((xx, yy + breath, ), dim=-1)  # on ajoute seulement sur y
        grid = grid.unsqueeze(0).repeat(B,1,1,1)  # batch
        latents = torch.nn.functional.grid_sample(latents, grid, align_corners=True)
    return latents

def apply_breathing_mask(latents, previous_latent, frame_counter, breathing=True):
    """
    Applique une respiration réaliste sur les latents : décalage vertical uniquement.
    """
    if previous_latent is not None and breathing:
        B, C, H, W = latents.shape
        device = latents.device

        # amplitude respiration
        breath = 0.003 * math.sin(frame_counter * 0.1)  # amplitude réduite

        # grille pixels avec meshgrid
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )  # yy, xx → [H, W]

        # ajouter dimension batch et stack
        grid = torch.stack((xx, yy + breath), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).expand(B, H, W, 2)   # [B, H, W, 2]

        # appliquer le décalage vertical uniquement
        latents = F.grid_sample(latents, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    return latents




#🫁 Breathing (FIX OK)
# version ultra
def apply_breathing_soft(
    latents,
    pose,
    mask_torso,
    frame_counter,
    breathing=True,
    amplitude=2.0,
    heartbeat_strength=0.6,
    inertia=0.85,
    face_sync=0.35,
    debug=False,
    debug_dir=None,
):
    if not breathing:
        return latents

    B, C, H, W = latents.shape
    device = latents.device

    # =========================================================
    # 1. TIME SIGNALS
    # =========================================================
    t = frame_counter * 0.1

    # respiration (slow)
    breath = amplitude * 0.18 * (
        0.65 * math.sin(t) +
        0.35 * math.sin(t * 0.5 + 1.7)
    )
    # heartbeat (fast micro jitter)
    heartbeat = heartbeat_strength * 0.06 * math.sin(t * 6.0)

    # combined chest signal
    chest_signal = breath + heartbeat

    # =========================================================
    # 2. GRID
    # =========================================================
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    xx = xx.unsqueeze(0).expand(B, H, W)
    yy = yy.unsqueeze(0).expand(B, H, W)

    mask = mask_torso.permute(0, 2, 3, 1)

    # =========================================================
    # 3. POSE CONTEXT (clavicles + ribs anchor)
    # =========================================================
    chest_center = torch.zeros((B, 2), device=device)
    clavicle_vec = torch.zeros((B, 2), device=device)

    face_center = None
    mouth_center = None

    if pose is not None:
        try:
            fp = pose.estimate_facial_points_full(smooth=0.85)

            # -------------------------
            # CHEST ANCHOR (sternum)
            # -------------------------
            if "neck" in fp and "left_shoulder" in fp and "right_shoulder" in fp:
                ls = fp["left_shoulder"]
                rs = fp["right_shoulder"]
                neck = fp["neck"]

                clavicle_vec = rs - ls
                shoulders_mid = (ls + rs) * 0.5

                chest_center = 0.35 * neck + 0.65 * shoulders_mid

            # -------------------------
            # FACE CENTER (for sync)
            # -------------------------
            if "nose" in fp:
                face_center = fp["nose"]

            if "mouth_center" in fp:
                mouth_center = fp["mouth_center"]

        except Exception:
            pass

    # =========================================================
    # 4. SHOULDERS INERTIA MODEL (lag simulation)
    # =========================================================
    # simulate delayed response of shoulders
    inertia_factor = inertia

    clavicle_motion = clavicle_vec * (1.0 - inertia_factor)

    # =========================================================
    # 5. RIBS EXPANSION FIELD (anisotropic breathing)
    # =========================================================
    dx = xx - chest_center[:, None, None, 0]
    dy = yy - chest_center[:, None, None, 1]

    dist2 = dx * dx + dy * dy

    # ribs influence (broad area)
    ribs = torch.exp(-dist2 / (2 * 0.18 * 0.18))

    # vertical expansion stronger (rib cage lift)
    rib_x = 0.8
    rib_y = 1.25

    # clavicle coupling (slight horizontal opening)
    clavicle_x = clavicle_motion[:, 0].view(B, 1, 1) * 0.15

    # =========================================================
    # 6. MICRO HEART JITTER (global subtle pulse)
    # =========================================================
    heart_jitter = heartbeat * 0.6

    # =========================================================
    # 7. FINAL BREATH FIELD
    # =========================================================
    base = chest_signal * mask[..., 0] * ribs

    delta_x = base * rib_x + clavicle_x + heart_jitter
    delta_y = base * rib_y + heart_jitter

    # =========================================================
    # 8. FACE SYNC (very subtle)
    # =========================================================
    if face_center is not None:
        face_breath = breath * face_sync * mask[..., 0]

        fx = xx - face_center[:, None, None, 0]
        fy = yy - face_center[:, None, None, 1]

        face_radial = torch.exp(-(fx*fx + fy*fy) / (2 * 0.10 * 0.10))

        delta_x += face_breath * 0.15 * face_radial
        delta_y += face_breath * 0.25 * face_radial

    # =========================================================
    # 9. GRID WARP
    # =========================================================
    grid = torch.stack((xx + delta_x, yy + delta_y), dim=-1)

    latents_out = F.grid_sample(latents, grid, align_corners=True)

    # =========================================================
    # 10. DEBUG
    # =========================================================
    if debug:
        delta_mag = torch.sqrt(delta_x**2 + delta_y**2)
        print("🫁 Soft breath mean:", delta_mag.mean().item())
        print("💓 heartbeat contrib:", float(heartbeat))

    return latents_out

# Version real !
def apply_breathing_real(
    latents,
    pose,
    mask_torso,
    frame_counter,
    breathing=True,
    amplitude=4.5,
    asymmetry=0.12,   # 🔥 micro naturel gauche/droite
    vertical_bias=0.20, # 🔥 respiration légèrement plus verticale
    debug=False,
    debug_dir=None,
):
    if not breathing:
        return latents

    B, C, H, W = latents.shape
    device = latents.device

    # =========================================================
    # 1. BREATH SIGNAL (smooth + organic)
    # =========================================================
    phase = frame_counter * 0.1

    breath = amplitude * 0.05 * (
        0.65 * math.sin(phase + 1.2) +
        0.35 * math.sin(phase * 0.5 + 2.7)
    )

    # =========================================================
    # 2. GRID (normalized coords)
    # =========================================================
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    xx = xx.unsqueeze(0).expand(B, H, W)
    yy = yy.unsqueeze(0).expand(B, H, W)

    mask = mask_torso.permute(0, 2, 3, 1)

    # =========================================================
    # 3. STERNUM / CHEST CENTER ESTIMATION
    # =========================================================
    chest_center = None

    if pose is not None:
        try:
            fp = pose.estimate_facial_points_full(smooth=0.8)

            if "neck" in fp and "left_shoulder" in fp and "right_shoulder" in fp:
                neck = fp["neck"]
                ls = fp["left_shoulder"]
                rs = fp["right_shoulder"]

                shoulders_mid = (ls + rs) * 0.5

                # sternum approx (slightly below neck toward shoulders)
                chest_center = shoulders_mid * 0.7 + neck * 0.3

            elif "neck" in fp:
                chest_center = fp["neck"]

        except Exception:
            chest_center = None

    if chest_center is None:
        chest_center = torch.zeros((B, 2), device=device)

    # =========================================================
    # 4. BREATH FIELD (RADIAL EXPANSION)
    # =========================================================
    # distance from chest center
    dx = xx - chest_center[:, None, None, 0]
    dy = yy - chest_center[:, None, None, 1]

    dist2 = dx * dx + dy * dy

    # gaussian falloff (chest influence)
    sigma = 0.12
    falloff = torch.exp(-dist2 / (2 * sigma * sigma))

    # =========================================================
    # 5. ANATOMICAL ANISOTROPY (vertical > horizontal)
    # =========================================================
    # vertical expansion slightly stronger (natural breathing)
    scale_x = 0.85
    scale_y = 1.15 + vertical_bias

    # =========================================================
    # 6. MICRO ASYMMETRY (natural human imperfection)
    # =========================================================
    # smooth left/right imbalance
    asym = torch.tanh(xx * 1.5) * asymmetry  # -0.08 → +0.08

    # =========================================================
    # 7. FINAL BREATH DISPLACEMENT FIELD
    # =========================================================
    base = breath * mask[..., 0] * falloff

    delta_x = base * scale_x * asym
    delta_y = base * scale_y

    grid = torch.stack((xx + delta_x, yy + delta_y), dim=-1)

    latents_out = F.grid_sample(latents, grid, align_corners=True)

    # =========================================================
    # 8. DEBUG
    # =========================================================
    if debug:
        delta_mag = torch.sqrt(delta_x**2 + delta_y**2)
        print("🫁 Real Breathing mean:", delta_mag.mean().item())
        print("🫁 Real Breathing max:", delta_mag.max().item())

    return latents_out

# version anime:
def apply_breathing_simple_anime(
    latents,
    pose,
    mask_torso,
    frame_counter,
    breathing=True,
    amplitude=1.5, # 1.5
    debug=False,
    debug_dir=None,
):
    if not breathing:
        return latents

    B, C, H, W = latents.shape
    device = latents.device

    phase = frame_counter * 0.1

    breath = amplitude * 0.06 * (
        0.6 * math.sin(phase + 1.0) +
        0.4 * math.sin(phase * 0.5 + 2.0)
    )

    # =========================================================
    # GRID
    # =========================================================
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )

    xx = xx.unsqueeze(0).expand(B, H, W)
    yy = yy.unsqueeze(0).expand(B, H, W)

    mask = mask_torso.permute(0, 2, 3, 1)

    # =========================================================
    # BODY ORIENTATION (ROBUST + FALLBACK SAFE)
    # =========================================================
    dir_x = torch.zeros(B, device=device)
    dir_y = torch.ones(B, device=device)

    if pose is not None:
        try:
            fp = pose.estimate_facial_points_full(smooth=0.8)

            ctx = None
            if hasattr(pose, "compute_pose_context"):
                ctx = pose.compute_pose_context(fp)

            # =========================
            # PRIORITY 1: CONTEXT (BEST)
            # =========================
            if ctx is not None and "shoulder_vec" in ctx:
                vec = ctx["shoulder_vec"]  # [B,2]
                norm = torch.norm(vec, dim=-1, keepdim=True) + 1e-6
                body_dir = vec / norm

                dir_x = body_dir[:, 0]
                dir_y = body_dir[:, 1]

            # =========================
            # PRIORITY 2: FALLBACK (eyes/neck)
            # =========================
            elif "neck" in fp and "nose" in fp:
                neck = fp["neck"]
                nose = fp["nose"]

                body_vec = nose - neck
                norm = torch.norm(body_vec, dim=-1, keepdim=True) + 1e-6
                body_dir = body_vec / norm

                dir_x = body_dir[:, 0]
                dir_y = body_dir[:, 1]

        except Exception as e:
            if debug:
                print(f"[BREATH] fallback triggered: {e}")

    # =========================================================
    # SAFE RESHAPE
    # =========================================================
    dir_x = dir_x.view(B, 1, 1)
    dir_y = dir_y.view(B, 1, 1)

    # =========================================================
    # BREATHING FIELD (PROFILE-AWARE)
    # =========================================================
    base = breath * mask[..., 0]

    # 🔥 IMPORTANT: breathing should NOT fully follow direction
    # we mix vertical + orientation instead of replacing it
    delta_x = base * (0.25 * dir_x)
    delta_y = base * (1.0 + 0.35 * dir_y)

    grid = torch.stack((xx + delta_x, yy + delta_y), dim=-1)

    latents_out = F.grid_sample(latents, grid, align_corners=True)

    # =========================================================
    # DEBUG
    # =========================================================
    if debug:
        global_delta = torch.stack((delta_x, delta_y), dim=-1)
        print("🫁 Simple Breathing - mean:", global_delta.abs().mean().item())
        print("🫁 Simple Breathing - max:", global_delta.abs().max().item())

    return latents_out


