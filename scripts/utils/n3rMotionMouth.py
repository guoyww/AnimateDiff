#n3rMotionMouth.py
import numpy as np
import os
import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from .tools_utils import ensure_4_channels, log_debug, sanitize_latents
from .n3r_EMA import motion_aware_ema_fusion, compute_high_freq_energy


from .n3rMotionPose_tools import save_impact_map
from .n3rMotionPoseClass import Pose


""""
        38: ("mouth_left_ext", mouth_left_ext),
        39: ("mouth_right_ext", mouth_right_ext),

        40: ("mouth_left", mouth_left),
        41: ("mouth_right", mouth_right),
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

        mouth_points_idx = [ 40, 41, 70,71,72,73,74,75,76, 77,78,79,80,81,82,83 ]

"""

class MouthMotionModel(nn.Module):

    def __init__(self, in_channels=4, hidden=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )

        self.landmark_proj = nn.Linear(32, hidden)

        self.fusion = nn.Conv2d(hidden * 2, hidden, 1)

        # séparation des intentions
        self.motion_head = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.flow_head = nn.Conv2d(hidden, 2, 3, padding=1)
        self.gate_head = nn.Conv2d(hidden, 1, 1)

        #self.register_buffer("prev_flow", None)
        self.prev_flow = None

    def forward(self, x, landmarks):

        h = self.encoder(x)

        l = self.landmark_proj(landmarks)
        l = l[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])

        h = self.fusion(torch.cat([h, l], dim=1))

        motion = self.motion_head(h)

        gate = torch.sigmoid(self.gate_head(motion))

        flow = torch.tanh(self.flow_head(motion))

        flow = flow * gate

        # temporal smoothing inside model
        if self.prev_flow is not None:
            flow = 0.7 * flow + 0.3 * self.prev_flow

        self.prev_flow = flow.detach()

        return {
            "flow": flow,
            "gate": gate
        }



# instance par défaut pour ton pipeline
mouth_model = MouthMotionModel().cuda()



def build_mask(idx_list, pose, W, H, device, base_scale=0.08):

    pts = torch.stack([pose.get_point(i) for i in idx_list], dim=1)  # [B,N,2]

    pts_px = pts * torch.tensor(
        [W - 1, H - 1],
        device=device,
        dtype=pts.dtype
    )

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    grid_xy = torch.stack([xx, yy], dim=-1).float()  # [H,W,2]
    grid_xy = grid_xy.unsqueeze(0)  # [1,H,W,2]

    # reshape pour broadcast propre
    pts_px = pts_px.unsqueeze(2).unsqueeze(2)  # [B,N,1,1,2]

    dist = torch.norm(grid_xy.unsqueeze(1) - pts_px, dim=-1)  # [B,N,H,W]

    min_dist = dist.min(dim=1).values  # [B,H,W]

    mask = torch.exp(-min_dist * base_scale)

    return mask.unsqueeze(-1)



def apply_mouth_smil_dev_test(
    latents,
    pose,
    mask_mouth,
    grid,
    frame_counter,
    mouth_model,
    H=None,
    W=None,
    device=None,
    debug=False,
    debug_dir=None,
    smooth=0.75,
    strength=2.0,
    motion_scale=0.4,
    speed=1.0,
    naturel=True,
    npy=False
):
    if (frame_counter < 20):
        #sourire humain subtil lèvres crédibles presque zéro jitter stabilité inter-frame élevée très peu d’artefacts de warp
        latents_local, mouth_delta, mouth_points = apply_mouth_smil_pro( latents, pose, mask_mouth, grid, frame_counter, mouth_model, device=device, debug=debug,
                                                                    debug_dir=debug_dir, smooth=smooth, strength=strength, motion_scale=0.4, speed=1.0, naturel=True, npy=False)
    else:
        #animation stylisée streamer/avatar rendu plus expressif
        latents_local, mouth_delta, mouth_points = apply_mouth_smil_pro( latents, pose, mask_mouth, grid, frame_counter, mouth_model, device=device, debug=debug,
                                                                    debug_dir=debug_dir, smooth=smooth, strength=strength, motion_scale=0.55, speed=0.7, naturel=False, npy=False)
    return latents_local, mouth_delta, mouth_points

#==================================================================================================================
"""
strength=0.82
motion_scale=0.32
scale_flow=0.028
speed=0.42
naturel=True
"""
#sourire humain subtil lèvres crédibles presque zéro jitter stabilité inter-frame élevée très peu d’artefacts de warp

#ou ----------------------------------------------------------------------------------------------------------------
"""
naturel=False
motion_scale=0.55
speed=0.7
"""
#animation stylisée streamer/avatar rendu plus expressif
#=====================================================================================================================


def apply_mouth_smil(
    latents,
    pose,
    mask_mouth,
    grid,
    frame_counter,
    mouth_model,
    H=None,
    W=None,
    device=None,
    debug=False,
    debug_dir=None,
    naturel_dyn=False,
    smooth=0.75,
    strength=2.0,
    motion_scale=0.4,
    speed=1.0,
    naturel=True,
    npy=False
):
    """
    Wrapper dynamique autour de apply_mouth_smil_pro()

    PHASES:
    0-20   : sourire naturel stable
    20-45  : transition progressive
    45+    : version expressive / streamer

    Plug & Play:
    - même signature
    - mêmes outputs
    - aucun changement requis ailleurs
    """

    # =====================================================
    # INIT
    # =====================================================

    if device is None:
        device = latents.device

    # =====================================================
    # PHASE BLEND
    # =====================================================

    # 0 -> naturel
    # 1 -> stylized

    transition_start = 20
    transition_end   = 45

    if frame_counter <= transition_start:
        style_blend = 0.0

    elif frame_counter >= transition_end:
        style_blend = 1.0

    else:
        t = (
            (frame_counter - transition_start)
            / float(transition_end - transition_start)
        )

        # smoothstep
        style_blend = t * t * (3.0 - 2.0 * t)

    # =====================================================
    # INTERPOLATION PARAMS
    # =====================================================

    # Naturel -> Stylized

    motion_scale_dyn = (
        0.40 * (1.0 - style_blend)
        + 0.55 * style_blend
    )

    speed_dyn = (
        1.00 * (1.0 - style_blend)
        + 0.70 * style_blend
    )

    # naturel devient False progressivement
    if frame_counter <= transition_start:
        naturel_dyn = True
    elif frame_counter >= transition_end:
        naturel_dyn = False

    # =====================================================
    # OPTIONAL MICRO OSCILLATION
    # =====================================================

    # ajoute vie subtile sans casser stabilité
    micro_motion = 1.0 + 0.04 * math.sin(frame_counter * 0.12)

    motion_scale_dyn *= micro_motion

    # =====================================================
    # DEBUG
    # =====================================================

    print(f"""
    [MOUTH WRAPPER]
    frame            : {frame_counter}
    style_blend      : {style_blend:.3f}

    motion_scale_dyn : {motion_scale_dyn:.3f}
    speed_dyn        : {speed_dyn:.3f}
    strength         : {strength:.3f}

    naturel_dyn      : {naturel_dyn}
    """)

    # =====================================================
    # APPLY
    # =====================================================

    latents_local, mouth_delta, mouth_points = apply_mouth_smil_pro(
        latents=latents,
        pose=pose,
        mask_mouth=mask_mouth,
        grid=grid,
        frame_counter=frame_counter,
        mouth_model=mouth_model,
        device=device,
        debug=debug,
        debug_dir=debug_dir,
        smooth=smooth,
        strength=strength,
        motion_scale=motion_scale_dyn,
        speed=speed_dyn,
        naturel=naturel_dyn,
        npy=npy
    )

    return latents_local, mouth_delta, mouth_points




def apply_mouth_smil_pro(
    latents,
    pose,
    mask_mouth,
    grid,
    frame_counter,
    mouth_model,
    device=None,
    debug=False,
    debug_dir=None,
    smooth=0.75,
    strength=0.85,
    motion_scale=0.4,
    scale_flow=0.035,
    speed=1.0,
    naturel=True,
    mouth_scale=True, # Réduction du bruit sur image bouche grand format
    npy=False
):

    if device is None:
        device = latents.device

    B, C, H, W = latents.shape
    latents_in = latents.clone()

    # =====================================================
    # TIME
    # =====================================================
    time_t = frame_counter * speed

    # =====================================================
    # LANDMARKS
    # =====================================================
    mouth_points_idx = [ 40, 41, 70,71,72,73,74,75,76, 77,78,79,80,81,82,83 ]

    mouth_points = torch.stack(
        [pose.get_point(i) for i in mouth_points_idx],
        dim=1
    )

    mouth_center = mouth_points.mean(dim=1)

    scale_tensor = torch.tensor(
        [W - 1, H - 1],
        device=device,
        dtype=latents.dtype
    )

    mouth_center_px = (mouth_center * scale_tensor).view(B, 1, 1, 2)

    # =====================================================
    # DELTA
    # =====================================================
    if mouth_model is None:
        print("[MOTION MOUTH] ANALYTIC DELTA ✅")

        delta, _ = Pose.compute_mouth_delta( pose=pose, mask_mouth=mask_mouth, H=H, W=W, frame_counter=frame_counter, device=device, smooth=smooth, strength=strength, debug=debug )

        # assume BCHW -> convert to BHWC
        if delta.shape[1] == 2:
            delta = delta.permute(0, 2, 3, 1).contiguous()

        motion_gate = torch.ones((B, H, W, 1), device=device)

    else:
        print("[MOTION MOUTH] NEURAL DELTA ✅")

        landmarks = mouth_points.reshape(B, -1)

        pred = mouth_model(latents, landmarks)

        # BCHW -> BHWC
        delta = pred["flow"].permute(0, 2, 3, 1).contiguous()
        motion_gate = pred["gate"].permute(0, 2, 3, 1).contiguous()

        # scale flow -> scale_flow
        delta[..., 0] *= W * scale_flow
        delta[..., 1] *= H * scale_flow


    # =====================================================
    # MASK PROCESSING
    # =====================================================
    if mask_mouth.dim() == 3:
        mask = mask_mouth.unsqueeze(1)
    else:
        mask = mask_mouth

    mask = mask.float()

    # anisotropic dilation (good mouth behavior)
    # =====================================================
    # WIDE SOFT MOUTH FIELD
    # =====================================================

    mask_h = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    mask_v = F.avg_pool2d(mask_h, kernel_size=3, stride=1, padding=1)

    mask_soft = F.interpolate(
        mask_v,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    # BOOST STRUCTURE (important)
    # diffusion douce
    mask_soft = torch.pow(mask_soft, 0.7)

    # amplification
    mask_soft = torch.clamp(mask_soft * 4.0, 0.0, 1.0)
    # =====================================================
    # BCHW -> BHWC
    mask_soft = mask_soft.permute(0, 2, 3, 1).contiguous()

    print("[MASK MAX]", mask_soft.max().item())
    print("[MASK MEAN]", mask_soft.mean().item())

    # =====================================================
    # INERTIA - V2
    # =====================================================
    if not hasattr(apply_mouth_smil_pro, "prev_delta"):
        apply_mouth_smil_pro.prev_delta = torch.zeros_like(delta)

    alpha = 0.55 * speed

    # clamp sécurité
    alpha = max(0.05, min(alpha, 0.95))

    smoothed = (
        alpha * delta
        + (1 - alpha) * apply_mouth_smil_pro.prev_delta
    )

    apply_mouth_smil_pro.prev_delta = smoothed.detach()

    delta = smoothed

    # =======================================================================================
    # MOUTH SCALE NORMALIZATION ( reduction du bruit sur zone important exemple grande bouche)
    # =======================================================================================
    if mouth_scale:
        mouth_min = mouth_points.min(dim=1).values
        mouth_max = mouth_points.max(dim=1).values
        mouth_size = torch.norm(mouth_max - mouth_min, dim=-1, keepdim=True)
        # valeur de référence
        ref_size = 0.18
        # normalisation
        mouth_scale_norm = ref_size / (mouth_size + 1e-6)
        # clamp sécurité
        mouth_scale_norm = mouth_scale_norm.clamp(0.65, 1.35)
        mouth_scale_norm = mouth_scale_norm.view(B, 1, 1, 1)

    # =====================================================
    # TEMPORAL (APPLY BEFORE MASK)
    # =====================================================
    t = torch.tensor(time_t / 10.0, device=delta.device, dtype=delta.dtype)

    sin_t = torch.sin(t)
    cos_t = torch.cos(t * 0.8)

    temporal_vec = torch.stack([
        torch.full_like(delta[..., 0], sin_t),
        torch.full_like(delta[..., 1], cos_t)
    ], dim=-1)

    # Occilation constantes
    temporal_weight = 0.6 + 0.4 * torch.sin(
        torch.tensor(time_t * 0.25, device=delta.device, dtype=delta.dtype)
    )

    if mouth_scale:
        # grandes bouches => moins de turbulence
        noise_scale = mouth_scale_norm.view(B,1,1,1)

        delta = delta + (
            temporal_vec
            * temporal_weight
            * 0.08
            * noise_scale
            * (1.0 + 0.08 * torch.tanh(delta.abs()))
        )
    else:
        delta = delta + ( temporal_vec * temporal_weight * 0.25 * (1.0 + 0.2 * torch.tanh(delta.abs())) )

    # =====================================================
    # TEMPORAL DYNAMICS (SAFE)
    # =====================================================

    if not hasattr(apply_mouth_smil_pro, "prev_motion"):
        apply_mouth_smil_pro.prev_motion = torch.zeros_like(delta)

    motion_change = delta - apply_mouth_smil_pro.prev_motion

    motion_energy = torch.norm( motion_change, dim=-1, keepdim=True )

    if mouth_scale:
        # grandes bouches => moins de turbulence
        motion_boost = 1.0 + torch.clamp( motion_energy * 0.02 * mouth_scale_norm, 0.0, 0.04 )
    else:
        motion_boost = 1.0 + torch.clamp( motion_energy * 0.05, 0.0, 0.12 )

    delta = delta * motion_boost

    apply_mouth_smil_pro.prev_motion = delta.detach()

    # =====================================================
    # DEBUG: MOTION ANALYSIS CORE
    # =====================================================

    if not hasattr(apply_mouth_smil_pro, "debug_prev"):
        apply_mouth_smil_pro.debug_prev = torch.zeros_like(delta)

    delta_change = (delta - apply_mouth_smil_pro.debug_prev).abs().mean().item()
    delta_norm = torch.norm(delta, dim=-1).mean().item()
    motion_inertia = (apply_mouth_smil_pro.prev_delta - delta).abs().mean().item()
    temporal_energy = (temporal_vec.abs().mean().item() * float(temporal_weight))
    mask_energy = mask_soft.mean().item()
    gate_energy = motion_gate.mean().item() if motion_gate is not None else 0.0

    print(f"""
    [MOUTH DEBUG]
    frame: {frame_counter}
    time: {time_t}
    delta_mean: {delta.abs().mean().item():.6f}
    delta_max : {delta.abs().max().item():.6f}

    delta_change (t vs t-1): {delta_change:.6f}  <-- IMPORTANT
    delta_norm: {delta_norm:.6f}

    inertia_effect: {motion_inertia:.6f}
    temporal_energy: {temporal_energy:.6f}

    mask_energy: {mask_energy:.3f}
    gate_energy: {gate_energy:.3f}
    """)

    motion_alive = delta_change / (delta_norm + 1e-6)

    print(f"[MOTION ALIVE SCORE] {motion_alive:.6f}")

    apply_mouth_smil_pro.debug_prev = delta.detach().clone()

    # =====================================================
    # CONSTRAINTS (MASK + GATE)
    # =====================================================
    #delta = delta * mask_soft * motion_gate

    combined_gate = mask_soft * (0.7 + 0.3 * motion_gate)
    delta = delta * combined_gate

    if delta.abs().mean().item() < 1e-4:
        print("[WARNING] delta almost zero → motion collapsed by mask/gate/inertia")

    # =====================================================
    # SCALING (ONLY ONCE)
    # =====================================================
    delta = delta * strength

    # =====================================================
    # REGION MASKS (CORE / CORNER / ANCHOR)
    # =====================================================

    core_idx = list(range(70, 84))
    corner_idx = [40, 41]
    anchor_idx = [38, 39]

    core_mask   = build_mask(core_idx, pose, W, H, device,  0.10)
    corner_mask = build_mask(corner_idx, pose, W, H, device, 0.06)
    anchor_mask = build_mask(anchor_idx, pose, W, H, device, 0.04)

    # =====================================================
    # Deblocage des coins
    # =====================================================
    corner_mask = corner_mask * (0.9 + 0.1 * torch.sin(
        torch.tensor(time_t * 0.3, device=device, dtype=delta.dtype)
    ))

    # =====================================================
    # COMPUTE DELTA
    # =====================================================
    core_delta   = delta * core_mask
    corner_delta = delta * corner_mask
    anchor_delta = delta * anchor_mask

    # inertia séparée
    if not hasattr(apply_mouth_smil_pro, "corner_prev"):
        apply_mouth_smil_pro.corner_prev = torch.zeros_like(delta)
    if not hasattr(apply_mouth_smil_pro, "anchor_prev"):
        apply_mouth_smil_pro.anchor_prev = torch.zeros_like(delta)

    corner_delta = 0.85 * corner_delta + 0.15 * apply_mouth_smil_pro.corner_prev
    anchor_delta = 0.95 * anchor_delta + 0.05 * apply_mouth_smil_pro.anchor_prev

    apply_mouth_smil_pro.corner_prev = corner_delta.detach()
    apply_mouth_smil_pro.anchor_prev = anchor_delta.detach()

    # recomposition hiérarchique
    delta = delta * (1 - core_mask - corner_mask - anchor_mask) + core_delta + corner_delta + anchor_delta

    # =====================================================
    # UPDATE INERTIA BUFFER
    # =====================================================
    apply_mouth_smil_pro.prev_delta = delta.detach()

    # =====================================================
    # GRID WARP
    # =====================================================
    base_grid = grid.clone()

    # ensure grid BHWC
    if base_grid.shape[-1] != 2:
        base_grid = base_grid.permute(0, 2, 3, 1).contiguous()

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    yy = yy.unsqueeze(0).unsqueeze(-1)
    xx = xx.unsqueeze(0).unsqueeze(-1)


    # =====================================================
    # LIP ARTICULATION FIELD
    # =====================================================

    # séparation verticale lèvres
    upper_field = torch.exp(-((yy + 0.08) ** 2) * 40.0)
    lower_field = torch.exp(-((yy - 0.08) ** 2) * 40.0)

    # dynamique opposée
    lip_field_y = (lower_field - upper_field)

    # léger asymétrique horizontal
    lip_field_x = torch.sin(xx * 3.14) * 0.15

    articulation = torch.cat([
        lip_field_x,
        lip_field_y
    ], dim=-1)

    # modulation temporelle
    articulation_strength = 0.06 + 0.03 * torch.sin(
        torch.tensor(time_t * 0.25, device=device)
    )

    delta = delta + articulation * articulation_strength * mask_soft

    # =====================================================
    # RIGIDITY FIELD (CRITICAL FOR SHARPNESS)
    # =====================================================

    # centre bouche = mobile
    center_weight = torch.exp(-(xx**2) * 6.0)
    # lèvres hautes plus rigides
    upper_rigid = torch.exp(-((yy + 0.15) ** 2) * 60.0)
    # lèvres basses mobiles
    lower_mobile = torch.exp(-((yy - 0.10) ** 2) * 30.0)
    # coins rigides
    corner_rigid = 1.0 - torch.exp(-(xx**2) * 2.5)

    # rigidité finale
    rigidity = (
        0.25
        + 0.75 * center_weight * lower_mobile
    )

    rigidity = rigidity * (1.0 - 0.6 * upper_rigid)
    rigidity = rigidity * (1.0 - 0.5 * corner_rigid)

    rigidity = rigidity.clamp(0.1, 1.0)

    delta = delta * rigidity

    # =====================================================
    # LIP COMPRESSION FIELD
    # =====================================================
    # centre horizontal bouche
    center_x = torch.exp(-(xx**2) * 10.0)

    # ligne fermeture lèvres
    lip_line = torch.exp(-(yy**2) * 80.0)

    compression = center_x * lip_line

    upper_part = torch.clamp(-yy, 0.0, 1.0)
    lower_part = torch.clamp(yy, 0.0, 1.0)

    if naturel:
        compress_y = ( -upper_part * 0.015 + lower_part * 0.05 ) * compression
    else:
        compress_y = ( -upper_part * 0.04 + lower_part * 0.16 ) * compression

    print( "[COMPRESS_Y]", compress_y.mean().item(), compress_y.min().item(), compress_y.max().item() )

    compression_field = torch.cat([
        torch.zeros_like(compress_y),
        compress_y
    ], dim=-1)

    # =====================================================
    # Prev Compression - New code
    # =====================================================
    if not hasattr(apply_mouth_smil_pro, "prev_compression"):
        apply_mouth_smil_pro.prev_compression = compression_field

    compression_field = (
        0.92 * apply_mouth_smil_pro.prev_compression
        + 0.08 * compression_field
    )

    apply_mouth_smil_pro.prev_compression = compression_field.detach()

    # APPLY AFTER SMOOTHING
    delta = delta + compression_field * mask_soft
    # =====================================================
    # LOG
    # =====================================================
    lip_opening = delta[...,1].mean().item()

    upper_motion = delta[...,1][yy.squeeze(-1) < 0].abs().mean().item()
    lower_motion = delta[...,1][yy.squeeze(-1) > 0].abs().mean().item()

    print(f"[LIP OPENING] {lip_opening:.6f}")
    print(f"[UPPER LIP MOTION] {upper_motion:.6f}")
    print(f"[LOWER LIP MOTION] {lower_motion:.6f}")

    # =====================================================
    # AMPLITUDE DU MOUVEMENT
    # =====================================================
    delta = delta * motion_scale

    # =====================================================
    # FLOW LIMITER - new code ! (Optionnelle !)
    # =====================================================

    if mouth_scale:
        mouth_factor = (1.0 / mouth_scale_norm).clamp(1.0, 1.5)
        flow_limit_x = W * 0.020 * mouth_factor
        flow_limit_y = H * 0.020 * mouth_factor

    else:
        flow_limit_x = W * 0.028
        flow_limit_y = H * 0.028

    delta[..., 0] = delta[..., 0].clamp( -flow_limit_x, flow_limit_x )
    delta[..., 1] = delta[..., 1].clamp( -flow_limit_y, flow_limit_y )

    # =====================================================
    # GRID
    # =====================================================
    grid_mouth = base_grid + delta

    # normalize
    grid_norm = grid_mouth.clone()
    grid_norm[..., 0] = 2.0 * grid_norm[..., 0] / (W - 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid_norm[..., 1] / (H - 1) - 1.0

    grid_norm = torch.clamp(grid_norm, -1.2, 1.2)

    # =====================================================
    # GRID SAMPLE
    # =====================================================

    if grid_norm.shape[-1] != 2:
        grid_norm = grid_norm.permute(0, 2, 3, 1).contiguous()

    grid_shift = (grid_mouth - base_grid).abs().mean().item()

    print(f"[GRID DEBUG] shift mean: {grid_shift:.6f}")

    if grid_shift < 1e-4:
        print("[WARNING] grid is static → no visible deformation")

    latents_out = F.grid_sample( latents, grid_norm, mode='bilinear', padding_mode='reflection', align_corners=True )
    # =====================================================
    # BLENDING (SAFE)
    # =====================================================
    blend = mask_soft.permute(0, 3, 1, 2).contiguous()

    if blend.shape[-2:] != latents.shape[-2:]:
        blend = F.interpolate(
            blend,
            size=latents.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    latents_out = latents * (1.0 - blend) + latents_out * blend

    # =====================================================
    # DEBUG
    # =====================================================
    delta_mean = delta.abs().mean().item()
    delta_max = delta.abs().max().item()

    print("[MOTION MOUTH] delta mean:", delta_mean)
    print("[MOTION MOUTH] delta max :", delta_max)

    if debug and debug_dir is not None:
        try:
            os.makedirs(debug_dir, exist_ok=True)

            save_impact_map( latents_out, latents_in, debug_dir, frame_counter, prefix="mouth" )

            if npy:
                np.save( os.path.join(debug_dir, f"mouth_delta_{frame_counter:05d}.npy"), delta.detach().cpu().numpy() )

        except Exception as e:
            print("[WARN] debug failed:", e)

    return latents_out, delta, mouth_points


# =========================================================
# OPTIONAL DEBUG UTILS
# =========================================================

def _safe_stat(x):
    return {
        "mean": x.mean().item(),
        "std": x.std().item(),
        "max": x.max().item(),
        "min": x.min().item(),
    }


def _log_tensor(name, x):
    s = _safe_stat(x)

    print(
        f"[{name}] "
        f"mean={s['mean']:.6f} "
        f"std={s['std']:.6f} "
        f"min={s['min']:.6f} "
        f"max={s['max']:.6f}"
    )


def reset_mouth_state():
    """
    Hard reset persistent temporal states.
    Useful between videos.
    """

    attrs = [
        "velocity",
        "displacement",
        "prev_delta",
        "motion_energy",
        "frame_index",
    ]

    for a in attrs:
        if hasattr(apply_mouth_smil, a):
            delattr(apply_mouth_smil, a)

    print("[MOUTH] temporal state reset")


# =========================================================
# MAIN
# =========================================================

def apply_mouth_smil_dev(
    latents,
    pose,
    mask_mouth,
    grid,
    frame_counter,

    mouth_model=None,

    H=None,
    W=None,
    device=None,
    smooth=0.90, # old
    # =====================================================
    # PHYSICS
    # =====================================================

    strength=0.30,

    inertia=0.92,
    spring_k=0.015,

    force_gain=0.035,

    displacement_decay=0.995,

    max_velocity=2.50,
    max_displacement=12.0,

    # =====================================================
    # STABILITY
    # =====================================================

    temporal_smoothing=0.15,

    adaptive_motion=True,
    adaptive_strength=True,

    velocity_damping_edge=0.85,

    # =====================================================
    # MASK
    # =====================================================

    mask_power=0.75,
    blend_strength=0.35,

    # =====================================================
    # DEBUG
    # =====================================================

    debug=False,
    debug_dir=None,
    save_npy=False,
    npy=False
):
    """
    =========================================================
    STABLE PHYSICS-BASED LATENT MOUTH MOTION
    =========================================================

    FEATURES
    --------
    - physically coherent motion
    - velocity integration
    - spring restoration
    - temporal stability
    - drift prevention
    - adaptive motion control
    - stable long sequence behavior
    - edge damping
    - safe latent blending

    DYNAMICS
    --------
        force -> velocity -> displacement -> grid

    NO:
    ----
    - oscillation hacks
    - EMA grid drift
    - fake sinus motion
    - unstable accumulations
    """

    # =====================================================
    # DEVICE
    # =====================================================

    if device is None:
        device = latents.device

    B, C, H_lat, W_lat = latents.shape

    if H is None:
        H = H_lat

    if W is None:
        W = W_lat

    latents_in = latents.clone()

    # =====================================================
    # GRID FORMAT
    # =====================================================

    base_grid = grid.clone()

    if base_grid.shape[-1] != 2:
        base_grid = (
            base_grid
            .permute(0, 2, 3, 1)
            .contiguous()
        )

    # =====================================================
    # LANDMARKS
    # =====================================================

    mouth_points_idx = [
        40, 41,
        70, 71, 72, 73, 74, 75, 76,
        77, 78, 79, 80, 81, 82, 83
    ]

    mouth_points = torch.stack(
        [pose.get_point(i) for i in mouth_points_idx],
        dim=1
    )

    # =====================================================
    # MOTION TARGET
    # =====================================================

    if mouth_model is None:

        print("[MOUTH] analytic motion")

        target_motion, _ = Pose.compute_mouth_delta(
            pose=pose,
            mask_mouth=mask_mouth,
            H=H,
            W=W,
            frame_counter=frame_counter,
            device=device,
            smooth=0.90,
            strength=1.0,
            debug=debug
        )

        # BCHW -> BHWC
        if target_motion.shape[1] == 2:
            target_motion = (
                target_motion
                .permute(0, 2, 3, 1)
                .contiguous()
            )

        motion_gate = torch.ones(
            (B, H, W, 1),
            device=device,
            dtype=target_motion.dtype
        )

    else:

        print("[MOUTH] neural motion")

        landmarks = mouth_points.reshape(B, -1)

        pred = mouth_model(latents, landmarks)

        target_motion = (
            pred["flow"]
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        motion_gate = (
            pred["gate"]
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        target_motion[..., 0] *= W * 0.025
        target_motion[..., 1] *= H * 0.025

    # =====================================================
    # TEMPORAL SMOOTHING
    # =====================================================

    if not hasattr(apply_mouth_smil, "prev_delta"):

        apply_mouth_smil.prev_delta = (
            torch.zeros_like(target_motion)
        )

    prev_delta = apply_mouth_smil.prev_delta

    target_motion = (
        prev_delta * temporal_smoothing
        + target_motion * (1.0 - temporal_smoothing)
    )

    apply_mouth_smil.prev_delta = (
        target_motion.detach()
    )

    # =====================================================
    # MASK
    # =====================================================

    if mask_mouth.dim() == 3:
        mask = mask_mouth.unsqueeze(1)
    else:
        mask = mask_mouth

    mask = mask.float()

    # =====================================================
    # SOFT MOUTH FIELD
    # =====================================================

    mask_h = F.max_pool2d(
        mask,
        kernel_size=(3, 7),
        stride=1,
        padding=(1, 3)
    )

    mask_v = F.avg_pool2d(
        mask_h,
        kernel_size=(3, 5),
        stride=1,
        padding=(1, 2)
    )

    mask_soft = F.interpolate(
        mask_v,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    mask_soft = torch.pow(mask_soft, mask_power)

    mask_soft = torch.clamp(
        mask_soft * 1.75,
        0.0,
        1.0
    )

    # =====================================================
    # EDGE DAMPING
    # =====================================================

    edge_mask = (
        1.0 - mask_soft
    ) * velocity_damping_edge + mask_soft

    # BCHW -> BHWC
    mask_soft = (
        mask_soft
        .permute(0, 2, 3, 1)
        .contiguous()
    )

    edge_mask = (
        edge_mask
        .permute(0, 2, 3, 1)
        .contiguous()
    )

    # =====================================================
    # MOTION GATE
    # =====================================================

    combined_gate = (
        mask_soft
        * (0.8 + 0.2 * motion_gate)
    )

    target_motion = (
        target_motion
        * combined_gate
    )

    # =====================================================
    # TEMPORAL STATE INIT
    # =====================================================

    if not hasattr(apply_mouth_smil, "velocity"):

        apply_mouth_smil.velocity = (
            torch.zeros_like(target_motion)
        )

    if not hasattr(apply_mouth_smil, "displacement"):

        apply_mouth_smil.displacement = (
            torch.zeros_like(target_motion)
        )

    velocity = apply_mouth_smil.velocity
    displacement = apply_mouth_smil.displacement

    # =====================================================
    # ADAPTIVE MOTION
    # =====================================================

    motion_energy = (
        torch.norm(
            target_motion,
            dim=-1,
            keepdim=True
        )
    )

    if adaptive_motion:

        adaptive_force = (
            0.5
            + torch.clamp(
                motion_energy * 12.0,
                0.0,
                1.5
            )
        )

    else:

        adaptive_force = 1.0

    # =====================================================
    # FORCE
    # =====================================================

    force = (
        target_motion
        * force_gain
        * adaptive_force
    )

    # =====================================================
    # SPRING RESTORATION
    # =====================================================

    spring_force = (
        displacement
        * spring_k
    )

    force = force - spring_force

    # =====================================================
    # VELOCITY INTEGRATION
    # =====================================================

    velocity = (
        velocity * inertia
        + force
    )

    # =====================================================
    # EDGE DAMPING
    # =====================================================

    velocity = velocity * edge_mask

    # =====================================================
    # VELOCITY CLAMP
    # =====================================================

    velocity = torch.clamp(
        velocity,
        -max_velocity,
        max_velocity
    )

    # =====================================================
    # POSITION INTEGRATION
    # =====================================================

    displacement = (
        displacement
        + velocity
    )

    # =====================================================
    # DECAY
    # =====================================================

    displacement = (
        displacement
        * displacement_decay
    )

    # =====================================================
    # DISPLACEMENT CLAMP
    # =====================================================

    displacement = torch.clamp(
        displacement,
        -max_displacement,
        max_displacement
    )

    # =====================================================
    # ADAPTIVE STRENGTH
    # =====================================================

    if adaptive_strength:

        local_strength = (
            strength
            * (
                0.75
                + torch.clamp(
                    motion_energy * 8.0,
                    0.0,
                    0.5
                )
            )
        )

    else:

        local_strength = strength

    displacement = (
        displacement
        * local_strength
    )

    # =====================================================
    # SAVE STATE
    # =====================================================

    apply_mouth_smil.velocity = (
        velocity.detach()
    )

    apply_mouth_smil.displacement = (
        displacement.detach()
    )

    # =====================================================
    # FINAL GRID
    # =====================================================

    grid_mouth = (
        base_grid
        + displacement
    )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    grid_norm = grid_mouth.clone()

    grid_norm[..., 0] = (
        2.0 * grid_norm[..., 0] / (W - 1)
        - 1.0
    )

    grid_norm[..., 1] = (
        2.0 * grid_norm[..., 1] / (H - 1)
        - 1.0
    )

    grid_norm = torch.clamp(
        grid_norm,
        -1.05,
        1.05
    )

    # =====================================================
    # GRID SAMPLE
    # =====================================================

    latents_warped = F.grid_sample(
        latents,
        grid_norm,
        mode="bicubic",
        padding_mode="border",
        align_corners=True
    )

    # =====================================================
    # SAFE BLENDING
    # =====================================================

    blend = (
        mask_soft
        .permute(0, 3, 1, 2)
        .contiguous()
    )

    if blend.shape[-2:] != latents.shape[-2:]:

        blend = F.interpolate(
            blend,
            size=latents.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    blend = blend * blend_strength

    latents_out = (
        latents * (1.0 - blend)
        + latents_warped * blend
    )

    # =====================================================
    # STABILIZATION PASS
    # =====================================================

    residual = (
        latents_out - latents_in
    )

    residual = torch.clamp(
        residual,
        -3.0,
        3.0
    )

    latents_out = latents_in + residual

    # =====================================================
    # DEBUG LOGS
    # =====================================================

    if debug:

        print("\n=================================================")
        print(f"[MOUTH FRAME {frame_counter}]")
        print("=================================================")

        _log_tensor("target_motion", target_motion)
        _log_tensor("velocity", velocity)
        _log_tensor("displacement", displacement)

        print(
            f"[energy] "
            f"{motion_energy.mean().item():.6f}"
        )

        print(
            f"[mask] "
            f"mean={mask_soft.mean().item():.6f} "
            f"max={mask_soft.max().item():.6f}"
        )

        shift = (
            displacement
            .abs()
            .mean()
            .item()
        )

        print(f"[grid_shift] {shift:.6f}")

        if shift < 1e-5:
            print("[WARN] motion nearly frozen")

        print("=================================================\n")

    # =====================================================
    # DEBUG SAVE
    # =====================================================

    if debug and debug_dir is not None:

        try:

            os.makedirs(
                debug_dir,
                exist_ok=True
            )

            if save_npy:

                np.save(
                    os.path.join(
                        debug_dir,
                        f"mouth_displacement_{frame_counter:05d}.npy"
                    ),
                    displacement
                    .detach()
                    .cpu()
                    .numpy()
                )

                np.save(
                    os.path.join(
                        debug_dir,
                        f"mouth_velocity_{frame_counter:05d}.npy"
                    ),
                    velocity
                    .detach()
                    .cpu()
                    .numpy()
                )

        except Exception as e:

            print("[WARN] debug save failed:", e)

    # =====================================================
    # RETURN
    # =====================================================

    return (
        latents_out,
        displacement,
        mouth_points
    )


