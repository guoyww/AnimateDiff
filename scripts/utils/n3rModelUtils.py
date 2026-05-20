# n3rModelUtils.py
import os, math
import torch
from torchvision.transforms import functional as F
from scripts.utils.tools_utils import update_n3r_memory, inject_external_embeddings

def fuse_n3r_latents_adaptive_new(latents_frame, n3r_latents, frame_counter=None, total_frames=None,
                              latent_injection_start=0.9, latent_injection_end=0.55,
                              clamp_val=1.0, creative_noise=0.0):
    """
    Fusion adaptative des latents N3R avec les latents du frame courant.
    - latents_frame : tensor [B,C,H,W] du frame courant
    - n3r_latents   : tensor [B,C,H,W] N3R latents à fusionner
    - frame_counter, total_frames : pour injection dynamique
    - latent_injection_start / end : plage d'injection dynamique
    - creative_noise : ajout de bruit léger
    """
    # ---------------- Resize si nécessaire ----------------
    if n3r_latents.shape != latents_frame.shape:
        n3r_latents = torch.nn.functional.interpolate(
            n3r_latents, size=latents_frame.shape[-2:], mode='bilinear', align_corners=False
        )
        # Ajuster le nombre de canaux
        if n3r_latents.shape[1] < latents_frame.shape[1]:
            extra = latents_frame[:, n3r_latents.shape[1]:, :, :].clone() * 0
            n3r_latents = torch.cat([n3r_latents, extra], dim=1)
        elif n3r_latents.shape[1] > latents_frame.shape[1]:
            n3r_latents = n3r_latents[:, :latents_frame.shape[1], :, :]

    n3r_latents = n3r_latents.clone()

    # ---------------- Normalisation sûre par canal ----------------
    for c in range(min(3, n3r_latents.shape[1])):
        n3r_c = n3r_latents[:, c:c+1, :, :]
        frame_c = latents_frame[:, c:c+1, :, :]
        mean, std = n3r_c.mean(), n3r_c.std()
        if std.item() == 0:  # sécurité division par zéro
            std = 1.0
        n3r_c = (n3r_c - mean) / std
        frame_std = frame_c.std()
        frame_mean = frame_c.mean()
        if frame_std.item() == 0:
            frame_std = 1.0
        n3r_c = n3r_c * frame_std + frame_mean
        n3r_latents[:, c:c+1, :, :] = n3r_c

    # ---------------- Bruit créatif léger ----------------
    if creative_noise > 0.0:
        noise = torch.randn_like(n3r_latents) * creative_noise
        n3r_latents += noise

    # ---------------- Clamp ----------------
    n3r_latents = torch.clamp(n3r_latents, -clamp_val, clamp_val)
    latents_frame = torch.clamp(latents_frame, -clamp_val, clamp_val)

    # ---------------- Injection dynamique ----------------
    if frame_counter is not None and total_frames is not None:
        alpha = 0.5 - 0.5 * math.cos(math.pi * frame_counter / max(total_frames - 1, 1))
        latent_injection = latent_injection_start + (latent_injection_end - latent_injection_start) * alpha
        latent_injection = min(max(latent_injection, 0.0), 1.0)
    else:
        latent_injection = latent_injection_start

    fused_latents = latent_injection * latents_frame + (1 - latent_injection) * n3r_latents
    fused_latents = torch.clamp(fused_latents, -clamp_val, clamp_val)

    return fused_latents


def generate_n3r_coords(H, W, N_samples, seed, frame_counter, device):
    """Génère des coordonnées normalisées + jitter + bruit."""
    ys = torch.linspace(-1.0, 1.0, H, device=device)
    xs = torch.linspace(-1.0, 1.0, W, device=device)
    ss = torch.arange(N_samples, device=device, dtype=torch.float32)
    ys, xs, ss = torch.meshgrid(ys, xs, ss, indexing='ij')
    coords = torch.stack([xs, ys, ss], dim=-1).reshape(-1, 3)

    # jitter léger
    noise_scale = 0.01 + 0.02 * math.sin(frame_counter * 0.1)
    torch.manual_seed(seed + frame_counter)
    coords = coords + (torch.rand_like(coords) - 0.5) * 0.02
    coords = coords + torch.randn_like(coords) * noise_scale
    coords = torch.nan_to_num(coords)

    return coords


def process_n3r_latents(n3r_model, coords, H, W, target_H, target_W):
    """Forward N3R et interpolation vers HxW cible + ajout canal alpha."""
    n3r_latents_raw = n3r_model(coords, H, W)[:, :3]
    expected = H * W * n3r_model.N_samples
    if n3r_latents_raw.shape[0] != expected:
        raise RuntimeError(f"N3R reshape mismatch: {n3r_latents_raw.shape[0]} vs {expected}")

    n3r_latents = n3r_latents_raw.view(H, W, n3r_model.N_samples, 3).mean(dim=2)
    n3r_latents = n3r_latents.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    if n3r_latents.shape[1] == 3:
        n3r_latents = torch.cat([n3r_latents, torch.zeros_like(n3r_latents[:, :1])], dim=1)

    if n3r_latents.shape[-2:] != (target_H, target_W):
        n3r_latents = F.interpolate(n3r_latents, size=(target_H, target_W), mode='bilinear', align_corners=False)

    return n3r_latents


def fuse_with_memory(n3r_latents, memory_dict, cf_embeds, frame_counter):
    """Fusionne N3R latents avec la mémoire et calcule alpha adaptatif."""
    memory_alpha = 0.1 + 0.1 * math.sin(frame_counter * 0.05)
    pos_emb, neg_emb = cf_embeds
    key_embed = pos_emb - 0.5 * neg_emb
    fused_latents = update_n3r_memory(memory_dict, key_embed, n3r_latents, memory_alpha=memory_alpha)

    if fused_latents.shape[-2:] != n3r_latents.shape[-2:]:
        fused_latents = F.interpolate(fused_latents, size=n3r_latents.shape[-2:], mode='bilinear', align_corners=False)

    similarity = torch.cosine_similarity(n3r_latents.flatten(), fused_latents.flatten(), dim=0)
    adaptive_alpha = 0.1 + 0.2 * (1 - similarity)
    fused_latents = (1 - adaptive_alpha) * fused_latents + adaptive_alpha * n3r_latents

    return fused_latents


def inject_external(fused_latents, external_latent, frame_counter, device):
    """Injection du latent externe avec poids dynamique."""
    if external_latent is None or external_latent.shape != fused_latents.shape:
        raise RuntimeError("External latent absent ou dimensions incorrectes")

    dynamic_weight = 0.08 * (0.6 + 0.4 * math.sin(frame_counter * 0.1))
    external_embeddings = [{"key": "knx_neg", "latent": external_latent, "weight": dynamic_weight, "type": "negative"}]

    if external_embeddings:
        fused_latents = 0.9 * fused_latents + 0.1 * inject_external_embeddings(fused_latents, external_embeddings, device)

    return fused_latents
