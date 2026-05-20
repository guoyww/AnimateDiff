# n3rMotionHair_refactored.py
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from .n3rMotionPose_tools import ( smooth_noise, debug_save_mask_and_wind )

# ============================================================
# CACHE GLOBAL
# ============================================================
_FALLOFF_CACHE = {}
# ============================================================
# PROFILS
# ============================================================

@dataclass
class HairMotionProfile:
    name: str

    noise_x_amp: float
    noise_y_amp: float

    wind_base: float
    wind_var1: float
    wind_var2: float
    wind_var3: float

    gravity: float

    inertia: float

    spring_amp: float
    spring_freq: float

    micro_noise: float

    torso_motion_amp: float
    torso_wind_amp: float
    torso_gravity_amp: float

    falloff_power: float

    strength_clamp: float = 5.0


# ============================================================
# PROFILS PRESETS
# ============================================================

HAIR_PROFILES = {

    "cinema": HairMotionProfile( name="cinema", noise_x_amp=0.03, noise_y_amp=0.05, wind_base=0.02, wind_var1=0.01, wind_var2=0.005, wind_var3=0.001, gravity=0.004, inertia=0.45, spring_amp=0.003, spring_freq=2.5, micro_noise=0.0005, torso_motion_amp=0.10, torso_wind_amp=0.10, torso_gravity_amp=0.05, falloff_power=2.0 ),

    "3d": HairMotionProfile( name="3d", noise_x_amp=0.06, noise_y_amp=0.10, wind_base=0.04, wind_var1=0.02, wind_var2=0.01, wind_var3=0.001, gravity=0.008, inertia=0.7, spring_amp=0.006, spring_freq=0.005, micro_noise=0.001, torso_motion_amp=0.0012, torso_wind_amp=0.1, torso_gravity_amp=0.05, falloff_power=2.5 ),

    "vent": HairMotionProfile( name="vent", noise_x_amp=0.10, noise_y_amp=0.14, wind_base=0.08, wind_var1=0.04, wind_var2=0.02, wind_var3=0.003, gravity=0.012, inertia=0.6, spring_amp=0.010, spring_freq=0.05, micro_noise=0.001, torso_motion_amp=0.14, torso_wind_amp=0.10, torso_gravity_amp=0.02, falloff_power=2.8 ),

    "tornade": HairMotionProfile( name="tornade", noise_x_amp=0.04, noise_y_amp=0.07, wind_base=0.03, wind_var1=0.02, wind_var2=0.010, wind_var3=0.006, gravity=0.015, inertia=0.5, spring_amp=0.010, spring_freq=0.08, micro_noise=0.0015, torso_motion_amp=0.18, torso_wind_amp=0.3, torso_gravity_amp=0.15, falloff_power=3.0 ),

    "extreme": HairMotionProfile(
    name="extreme",

    # --- turbulence (plus élevée et équilibrée)
    noise_x_amp=0.08,
    noise_y_amp=0.12,

    # --- vent plus instable (pas juste plus fort)
    wind_base=0.07,
    wind_var1=0.04,
    wind_var2=0.02,
    wind_var3=0.01,

    # --- gravité modérée mais constante (effet “tirage”)
    gravity=0.010,

    # --- faible mémoire = chaos dynamique
    inertia=0.35,

    # --- ressort actif (micro oscillations visibles)
    spring_amp=0.008,
    spring_freq=4.5,

    # --- micro instabilité plus présente
    micro_noise=0.0012,

    # --- interaction corps forte mais pas dominante
    torso_motion_amp=0.22,
    torso_wind_amp=0.18,
    torso_gravity_amp=0.08,

    # --- structure spatiale plus rigide aux racines
    falloff_power=3.2
    ),

    "brise": HairMotionProfile( name="brise", noise_x_amp=0.008, noise_y_amp=0.012, wind_base=0.015, wind_var1=0.004, wind_var2=0.002, wind_var3=0.001, gravity=0.003, inertia=0.88, spring_amp=0.0012, spring_freq=2.0, micro_noise=0.00015, torso_motion_amp=0.06, torso_wind_amp=0.05, torso_gravity_amp=0.02, falloff_power=1.8 ),

    "decor": HairMotionProfile( name="decor", noise_x_amp=0.015, noise_y_amp=0.025, wind_base=0.010, wind_var1=0.005, wind_var2=0.003, wind_var3=0.001, gravity=0.002, inertia=0.92, spring_amp=0.001, spring_freq=2.0, micro_noise=0.0002, torso_motion_amp=0.08, torso_wind_amp=0.04, torso_gravity_amp=0.02, falloff_power=2.0 )
}


# ============================================================
# UTILS
# ============================================================

def get_falloff(
    H: int,
    power: float,
    device
):
    """
    Cache du falloff vertical.
    """

    key = (H, power, str(device))

    if key not in _FALLOFF_CACHE:

        yy = torch.linspace(
            0,
            1,
            H,
            device=device
        ).view(1, H, 1, 1)

        _FALLOFF_CACHE[key] = yy ** power

    return _FALLOFF_CACHE[key]


def multi_noise(
    grid,
    t,
    scales=(0.05, 0.15, 0.3),
    weights=(1.0, 0.5, 0.25)
):
    """
    Turbulence multi-échelle.
    """

    out = torch.zeros_like(smooth_noise(grid, t, scale=scales[0]))

    for scale, weight in zip(scales, weights):
        out += weight * smooth_noise(
            grid,
            t,
            scale=scale
        )

    return out


def build_dynamic_wind(
    profile: HairMotionProfile,
    t1,
    t2,
    B,
    H,
    W,
    device,
    strength
):
    """
    Vent procédural cohérent.
    """

    angle = (
        0.5 * torch.sin(t2)
        + 0.3 * torch.cos(t1)
    )

    wind_dir = torch.stack([
        torch.cos(angle),
        0.5 * torch.sin(angle)
    ], dim=-1).view(1, 1, 1, 2)

    wind_strength = (
        profile.wind_base
        + profile.wind_var1 * torch.sin(t1)
        + profile.wind_var2 * torch.sin(t2)
        + profile.wind_var3 * torch.cos(t1 * 1.3)
    )

    wind_strength *= strength

    wind_delta = wind_dir * wind_strength

    #return wind_delta.expand(B, H, W, 2)
    return wind_delta.expand(B, H, W, 2).clone()


def temporal_micro_noise(
    grid,
    t,
    amplitude
):
    """
    Bruit temporel cohérent.
    Remplace rand_like().
    """

    noise_x = smooth_noise(
        grid,
        t * 0.15,
        scale=0.4
    )

    noise_y = smooth_noise(
        grid,
        t * 0.15 + 100,
        scale=0.4
    )

    out = torch.stack([
        noise_x,
        noise_y
    ], dim=-1)

    return out * amplitude


# ============================================================
# MOTEUR PRINCIPAL
# ============================================================

def apply_hair_motion( latents, mask_hair, grid, H, W, frame_counter, device, profile: HairMotionProfile, delta_px=None, prev_hair_field=None, strength=1.0, debug=False, debug_dir=None ):

    B = latents.shape[0]

    strength = float(
        max(
            0.0,
            min(
                strength,
                profile.strength_clamp
            )
        )
    )

    # ========================================================
    # TEMPS
    # ========================================================

    t = torch.as_tensor(
        frame_counter,
        device=device,
        dtype=torch.float32
    )

    t1 = t / 15.0
    t2 = t / 60.0

    # ========================================================
    # BRUIT PROCEDURAL
    # ========================================================

    noise_x = multi_noise(grid, t)

    noise_y = multi_noise(
        grid,
        t + 123,
        scales=(0.08, 0.2, 0.4)
    )

    # ========================================================
    # DELTA FIELD
    # ========================================================

    hair_delta_field = torch.zeros(
        (B, H, W, 2),
        device=device
    )

    hair_delta_field[..., 0] = (
        profile.noise_x_amp
        * noise_x
    )

    hair_delta_field[..., 1] = (
        profile.noise_y_amp
        * noise_y
    )

    # ========================================================
    # WIND
    # ========================================================

    wind_delta = build_dynamic_wind( profile, t1, t2, B, H, W, device, strength )

    # ========================================================
    # GRAVITY
    # ========================================================

    gravity_delta = torch.zeros_like(
        hair_delta_field
    )

    gravity_delta[..., 1] = (
        profile.gravity
        * strength
    )

    # ========================================================
    # TORSO INFLUENCE
    # ========================================================

    if delta_px is not None:

        speed = torch.norm(
            delta_px,
            dim=-1,
            keepdim=True
        ).view(B, 1, 1, 1)

        speed = torch.clamp(speed, 0.0, 0.15)

        hair_delta_field *= (
            1.0
            + profile.torso_motion_amp * speed
        )

        wind_delta *= (
            1.0
            + profile.torso_wind_amp * speed
        )

        gravity_delta *= (
            1.0
            + profile.torso_gravity_amp * speed
        )

    # ========================================================
    # INERTIA
    # ========================================================

    if prev_hair_field is not None:

        hair_delta_field = (
            profile.inertia * prev_hair_field
            + (1.0 - profile.inertia)
            * hair_delta_field
        )

    # ========================================================
    # SPRING
    # ========================================================

    spring = (
        profile.spring_amp
        * torch.sin(
            t * 0.5
            + grid[..., 1:2]
            * profile.spring_freq
        )
    )

    hair_delta_field[..., 1:2] += spring

    # ========================================================
    # MICRO TURBULENCE
    # ========================================================

    hair_delta_field += temporal_micro_noise(
        grid,
        t,
        profile.micro_noise
    )

    # ========================================================
    # STRENGTH GLOBAL
    # ========================================================

    hair_delta_field *= strength

    # ========================================================
    # FALLOFF
    # ========================================================

    mask_hair_expand = mask_hair.permute( 0, 2, 3, 1 )

    falloff = get_falloff( H, profile.falloff_power, device )

    mask_hair_expand *= falloff

    # ========================================================
    # APPLICATION
    # ========================================================

    grid_hair = grid.clone()

    grid_hair += (
        hair_delta_field
        * mask_hair_expand
    )

    grid_hair += (
        wind_delta
        * mask_hair_expand
    )

    grid_hair += (
        gravity_delta
        * mask_hair_expand
    )

    # ========================================================
    # NORMALISATION
    # ========================================================

    grid_hair[..., 0] = (
        2.0 * grid_hair[..., 0]
        / (W - 1)
        - 1.0
    )

    grid_hair[..., 1] = (
        2.0 * grid_hair[..., 1]
        / (H - 1)
        - 1.0
    )

    # ========================================================
    # CLAMP SECURITE
    # ========================================================
    grid_hair = torch.clamp( grid_hair, -1.2, 1.2 )

    # ========================================================
    # GRID SAMPLE
    # ========================================================

    latents_out = F.grid_sample(
        latents,
        grid_hair,
        align_corners=True,
        #padding_mode='border'
        padding_mode='reflection'
    )

    # ========================================================
    # DEBUG
    # ========================================================

    if debug:

        mean_delta = hair_delta_field.abs().mean().item()
        max_delta = hair_delta_field.abs().max().item()

        print(
            f"[DEBUG][HAIR] "
            f"profile={profile.name} | "
            f"strength={strength:.2f} | "
            f"mean_delta={mean_delta:.6f} | "
            f"max_delta={max_delta:.6f}"
        )

        if debug_dir is not None and frame_counter % 6 == 0:
            debug_save_mask_and_wind( mask=mask_hair, wind_delta=wind_delta, H=H, W=W, debug_dir=debug_dir, frame_counter=frame_counter )

    return latents_out, hair_delta_field


# ============================================================
# WRAPPERS
# ============================================================

def apply_hair_motion_cinema(*args, **kwargs):
    return apply_hair_motion( *args, profile=HAIR_PROFILES["cinema"], **kwargs )


def apply_hair_motion_3D(*args, **kwargs):
    return apply_hair_motion( *args, profile=HAIR_PROFILES["3d"], **kwargs )


def apply_hair_motion_vent(*args, **kwargs):
    return apply_hair_motion( *args, profile=HAIR_PROFILES["vent"], **kwargs )


def apply_hair_motion_extreme(*args, **kwargs):
    return apply_hair_motion( *args, profile=HAIR_PROFILES["extreme"], **kwargs )

def apply_hair_motion_brise(*args, **kwargs):
    return apply_hair_motion( *args, profile=HAIR_PROFILES["brise"], **kwargs )


# ============================================================
# CYCLE
# ============================================================

def apply_hair_motion_cycle( latents, mask_hair, grid, H, W, frame_counter, device, delta_px=None, prev_hair_field=None, target="hair", debug=False, debug_dir=None ):

    if target == "hair":
        profiles = [ HAIR_PROFILES["vent"], HAIR_PROFILES["3d"], HAIR_PROFILES["brise"] , HAIR_PROFILES["cinema"]]
    else:
        profiles = [ HAIR_PROFILES["decor"]]

    profile = profiles[ frame_counter % len(profiles) ]


    strength = 1.5
    TARGET_STRENGTH = {
        "hair": 1.5,
        "decor": 0.35,
    }

    strength *= TARGET_STRENGTH[target]

    print(
        f"[DEBUG]✅ [HAIR] "
        f"target={target} | "
        f"profile={profile.name} | "
        f"strength={strength:.2f}"
    )

    return apply_hair_motion(
        latents=latents,
        mask_hair=mask_hair,
        grid=grid,
        H=H,
        W=W,
        frame_counter=frame_counter,
        device=device,
        profile=profile,
        delta_px=delta_px,
        prev_hair_field=prev_hair_field,
        strength=strength,
        debug=debug,
        debug_dir=debug_dir
    )
