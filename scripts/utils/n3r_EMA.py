import torch
import torch.nn.functional as F


# =========================================================
# Compute high freq
# =========================================================

def compute_high_freq_energy(latents, kernel_size=3, normalize=True, per_channel=False):

    latents = latents.float()

    blur = F.avg_pool2d(
        latents,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )

    high_freq = latents - blur

    hf_energy = torch.sqrt(high_freq.pow(2).mean(dim=(2, 3)) + 1e-8)

    if normalize:
        base = torch.sqrt(latents.pow(2).mean(dim=(2, 3)) + 1e-8)
        hf_energy = hf_energy / (base + 1e-6)

    if per_channel:
        return hf_energy

    return hf_energy.mean(dim=1)


def motion_aware_ema_fusion_v1(
    out,
    ema_prev_latents,
    hf,
    debug=False
):
    """
    Motion-aware EMA fusion with multi-frequency decomposition.
    Stable for temporal latent pipelines (AnimateDiff-like).

    Args:
        out: current latents [B, C, H, W]
        ema_prev_latents: previous EMA latents [B, C, H, W]
        hf: high-frequency energy tensor or scalar map
        debug: print diagnostics

    Returns:
        fused latents
    """

    prev = ema_prev_latents.to(out.device)

    # =====================================================
    # MULTI-FREQUENCY SPLIT
    # =====================================================

    out_low  = F.avg_pool2d(out, 3, 1, 1)
    prev_low = F.avg_pool2d(prev, 3, 1, 1)

    out_high  = out - out_low
    prev_high = prev - prev_low

    # =====================================================
    # MOTION ESTIMATION
    # =====================================================

    motion_global = hf.mean().item()

    micro_motion = (
        out_high.abs().mean() /
        (out.abs().mean() + 1e-6)
    ).item()

    motion = 0.7 * motion_global + 0.3 * micro_motion

    # =====================================================
    # LOW FREQUENCY EMA
    # =====================================================

    alpha_low = 0.04 + 0.08 * motion
    alpha_low = max(0.04, min(alpha_low, 0.14))

    # =====================================================
    # HIGH FREQUENCY EMA
    # =====================================================

    alpha_high = 0.15 + 0.35 * motion
    alpha_high = max(0.15, min(alpha_high, 0.55))

    # =====================================================
    # SOFT HIGH-FREQ ALIGNMENT
    # =====================================================

    out_high = 0.7 * out_high + 0.3 * prev_high

    # =====================================================
    # RECOMPOSITION
    # =====================================================

    out = (
        alpha_low * out_low + (1 - alpha_low) * prev_low +
        alpha_high * out_high + (1 - alpha_high) * prev_high
    )

    # =====================================================
    # DEBUG
    # =====================================================

    if debug:
        print(
            f"[EMA FUSION] "
            f"low={alpha_low:.4f} | "
            f"high={alpha_high:.4f} | "
            f"motion={motion:.4f}"
        )

    return out



def motion_aware_ema_fusion(
    out,
    ema_prev_latents,
    hf,
    debug=False
):
    """
    Motion-aware EMA fusion with structure/texture-aware motion gating.
    Stable for AnimateDiff-like latent pipelines.
    """

    prev = ema_prev_latents.to(out.device)

    # =====================================================
    # MULTI-FREQUENCY SPLIT
    # =====================================================

    out_low  = F.avg_pool2d(out, 3, 1, 1)
    prev_low = F.avg_pool2d(prev, 3, 1, 1)

    out_high  = out - out_low
    prev_high = prev - prev_low

    # =====================================================
    # MOTION ESTIMATION (IMPROVED)
    # =====================================================

    motion_global = hf.mean().item()

    micro_motion = (
        out_high.abs().mean() /
        (out.abs().mean() + 1e-6)
    ).item()

    # =====================================================
    # 🔥 NEW: STRUCTURE vs TEXTURE MOTION SEPARATION
    # =====================================================

    structure_motion = (out_low - prev_low).abs().mean().item()
    texture_motion   = (out_high - prev_high).abs().mean().item()

    # ratio = “coherent motion confidence”
    motion_confidence = structure_motion / (texture_motion + 1e-6)
    motion_confidence = max(0.2, min(motion_confidence, 3.0))
    motion_confidence = motion_confidence / 3.0

    # =====================================================
    # COMBINED MOTION
    # =====================================================

    motion = (
        0.5 * motion_global +
        0.3 * micro_motion +
        0.2 * motion_confidence
    )

    # =====================================================
    # LOW FREQUENCY EMA (structure stability)
    # =====================================================

    alpha_low = 0.04 + 0.08 * motion * (0.5 + 0.5 * motion_confidence)
    alpha_low = max(0.04, min(alpha_low, 0.14))

    # =====================================================
    # HIGH FREQUENCY EMA (texture + detail motion)
    # =====================================================

    alpha_high = 0.15 + 0.35 * motion * (1.0 - 0.4 * motion_confidence)
    alpha_high = max(0.15, min(alpha_high, 0.55))

    # =====================================================
    # SOFT HIGH-FREQ ALIGNMENT
    # =====================================================

    out_high = 0.7 * out_high + 0.3 * prev_high

    # =====================================================
    # RECOMPOSITION
    # =====================================================

    out = (
        alpha_low * out_low + (1 - alpha_low) * prev_low +
        alpha_high * out_high + (1 - alpha_high) * prev_high
    )

    # =====================================================
    # DEBUG
    # =====================================================

    if debug:
        print(
            f"[🔥 EMA FUSION v2] "
            f"low={alpha_low:.4f} | "
            f"high={alpha_high:.4f} | "
            f"motion={motion:.4f} | "
            f"conf={motion_confidence:.3f}"
        )

    return out




#-----------------------------------------------------
# EMA Globale SAFE
#-----------------------------------------------------

def motion_aware_ema_low_high_v1(
    latents,
    ema_prev_latents,
    ema_global,
    ema_micro,
    motion_noise,
    train=False,
    debug=True,
):
    """
    Motion-aware EMA fusion with low/high frequency decomposition (SAFE version)
    """

    if ema_prev_latents is None or train:
        if debug:
            print("🔥 [EMA SAFE] None (init or train mode)")
        return latents, ema_prev_latents, ema_global, ema_micro

    device = latents.device
    prev = ema_prev_latents.to(device)

    # =====================================================
    # MOTION ADAPTIVE COEFFICIENTS (NO torch.tensor WRAP)
    # =====================================================

    alpha_global = 0.02 + 0.06 * motion_noise
    alpha_global = max(0.02, min(alpha_global, 0.08))

    alpha_micro = 0.10 + 0.25 * (1.0 - motion_noise)
    alpha_micro = max(0.10, min(alpha_micro, 0.35))

    if debug:
        print(f"🔥 EMA global={alpha_global:.3f} | micro={alpha_micro:.3f}")

    # =====================================================
    # FREQUENCY DECOMPOSITION
    # =====================================================

    global_component = F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
    micro_component  = latents - global_component

    prev_global = F.avg_pool2d(prev, kernel_size=5, stride=1, padding=2)
    prev_micro  = prev - prev_global

    # =====================================================
    # INIT SAFETY
    # =====================================================

    if ema_global is None:
        ema_global = global_component.clone()
    if ema_micro is None:
        ema_micro = micro_component.clone()

    # =====================================================
    # EMA UPDATE
    # =====================================================

    ema_global = alpha_global * global_component + (1.0 - alpha_global) * ema_global
    ema_micro  = alpha_micro  * micro_component  + (1.0 - alpha_micro)  * ema_micro

    # =====================================================
    # RECOMBINATION
    # =====================================================

    ema_prev_latents = ema_global + ema_micro

    return ema_prev_latents, ema_global, ema_micro



#-----------------------------------------------------
# EMA Globale SAFE (FINAL)
#-----------------------------------------------------

def motion_aware_ema_low_high_v2(
    latents,
    ema_prev_latents,
    ema_global,
    ema_micro,
    motion_noise,
    train=False,
    debug=True,
):
    """
    Motion-aware EMA fusion with low/high frequency decomposition.
    Stable, lightweight, and micro-motion aware (Lite+ final version).
    """

    # =====================================================
    # INIT / TRAIN SAFETY
    # =====================================================

    if ema_prev_latents is None or train:
        if debug:
            print("🔥 [EMA SAFE] init / train mode → bypass")
        return latents, ema_prev_latents, ema_global, ema_micro

    device = latents.device
    prev = ema_prev_latents.to(device)

    # =====================================================
    # MOTION ADAPTIVE COEFFICIENTS
    # =====================================================

    # global stability (low frequency)
    alpha_global = 0.02 + 0.06 * motion_noise
    alpha_global = max(0.02, min(alpha_global, 0.08))

    # micro detail retention (high frequency)
    alpha_micro = 0.10 + 0.25 * (1.0 - motion_noise)
    alpha_micro = max(0.10, min(alpha_micro, 0.35))

    if debug:
        print(f"🔥 EMA global={alpha_global:.3f} | micro={alpha_micro:.3f}")

    # =====================================================
    # FREQUENCY DECOMPOSITION
    # =====================================================

    global_component = F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
    micro_component  = latents - global_component

    prev_global = F.avg_pool2d(prev, kernel_size=5, stride=1, padding=2)
    prev_micro  = prev - prev_global

    # =====================================================
    # INIT SAFETY (EMA STATE)
    # =====================================================

    if ema_global is None:
        ema_global = global_component.clone()
    if ema_micro is None:
        ema_micro = micro_component.clone()

    # =====================================================
    # MICRO-MOTION PRESERVATION (IMPORTANT FIX)
    # =====================================================

    # preserves small motion that was previously flattened
    motion_boost = 1.0 + (1.2 * (1.0 - motion_noise))
    motion_boost = max(0.8, min(motion_boost, 1.8))

    residual_motion = (micro_component - prev_micro) * 0.35
    adapted_micro = micro_component + motion_boost * residual_motion

    # soft stabilization to avoid drift
    adapted_micro = 0.85 * adapted_micro + 0.15 * prev_micro

    # =====================================================
    # EMA UPDATE
    # =====================================================

    ema_global = alpha_global * global_component + (1.0 - alpha_global) * ema_global
    ema_micro  = alpha_micro  * adapted_micro     + (1.0 - alpha_micro)  * ema_micro

    # =====================================================
    # RECOMPOSITION
    # =====================================================

    ema_prev_latents = ema_global + ema_micro

    # =====================================================
    # DEBUG
    # =====================================================

    if debug:
        print(
            f"[EMA FINAL] "
            f"low={alpha_global:.3f} | "
            f"high={alpha_micro:.3f} | "
            f"motion={motion_noise:.3f}"
        )

    return ema_prev_latents, ema_global, ema_micro


#-----------------------------------------------------
# EMA Globale SAFE (FINAL VERSION)
#-----------------------------------------------------

def motion_aware_ema_low_high(
    latents,
    ema_prev_latents,
    ema_global,
    ema_micro,
    motion_noise,
    train=False,
    debug=True,
):
    """
    Motion-aware EMA fusion with low/high frequency decomposition (FINAL SAFE VERSION)

    - stable first frame
    - no torch.tensor warnings
    - consistent low/high decomposition
    - safe EMA updates for 3–4GB GPU pipelines
    """

    # =====================================================
    # INIT / TRAIN SAFETY
    # =====================================================

    if ema_prev_latents is None or train:
        if debug:
            print("🔥 [EMA SAFE] init or train mode → bypass EMA")

        return latents, ema_prev_latents, ema_global, ema_micro

    device = latents.device
    prev = ema_prev_latents.to(device)

    # =====================================================
    # MOTION COEFFICIENTS (STABLE SCALING)
    # =====================================================

    alpha_global = 0.02 + 0.06 * motion_noise
    alpha_global = max(0.02, min(alpha_global, 0.08))

    alpha_micro = 0.10 + 0.25 * (1.0 - motion_noise)
    alpha_micro = max(0.10, min(alpha_micro, 0.35))

    if debug:
        print(f"🔥 EMA global={alpha_global:.3f} | micro={alpha_micro:.3f}")

    # =====================================================
    # FREQUENCY DECOMPOSITION (CURRENT + PREVIOUS)
    # =====================================================

    global_component = F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
    micro_component  = latents - global_component

    prev_global = F.avg_pool2d(prev, kernel_size=5, stride=1, padding=2)
    prev_micro  = prev - prev_global

    # =====================================================
    # EMA STATE INITIALIZATION (CRITICAL STABILITY FIX)
    # =====================================================

    if ema_global is None:
        ema_global = global_component.clone()

    if ema_micro is None:
        ema_micro = micro_component.clone()

    # =====================================================
    # EMA UPDATE (SEPARATED FREQUENCIES)
    # =====================================================

    ema_global = (
        alpha_global * global_component +
        (1.0 - alpha_global) * ema_global
    )

    ema_micro = (
        alpha_micro * micro_component +
        (1.0 - alpha_micro) * ema_micro
    )

    # =====================================================
    # RECOMPOSITION
    # =====================================================

    ema_prev_latents = ema_global + ema_micro

    # =====================================================
    # DEBUG OUTPUT (SAFE)
    # =====================================================

    if debug:
        print(
            f"[EMA FINAL] "
            f"low={alpha_global:.3f} | "
            f"high={alpha_micro:.3f} | "
            f"motion={motion_noise:.3f}"
        )

    return ema_prev_latents, ema_global, ema_micro
