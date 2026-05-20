# n3rApparenceModel.py
#petit renderer photométrique différentiable + module de style conditionné texte + adaptation temporelle EMA
# exposure → gain global # gamma → non-linéarité luminance # contrast → variance locale # micro → détail haute fréquence
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .tools_utils import ensure_4_channels, log_debug, sanitize_latents
from .n3r_EMA import motion_aware_ema_fusion, compute_high_freq_energy


# ===============================================================================
# pipeline de rendu temporel + style + régularisation perceptuelle + feedback EMA
# ===============================================================================



def stabilize_latents(latents, target_std=1.0):

    std = latents.std(dim=(1,2,3), keepdim=True)
    scale = target_std / (std + 1e-6)
    scale = torch.clamp(scale, 0.7, 1.3)

    return latents * scale

# =========================================================
# SAVE / LOAD
# =========================================================

def save_appearance_model(model, optimizer=None, epoch=None, loss=None,
                          path="models/appearance_model.pt",
                          latest_path="models/appearance_model_latest.pt"):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": getattr(model, "config", {}),
        "model_version": getattr(model, "version", "v2"),
        "timestamp": datetime.datetime.now().isoformat()
    }

    if optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if loss is not None:
        checkpoint["loss"] = float(loss)

    torch.save(checkpoint, path)
    torch.save(checkpoint, latest_path)

    print(f"[INFO] Saved AppearanceModel -> {path}")


# =========================================================
# LOAD ( V2 COMPATIBLE)
# =========================================================
def load_appearance_model(model_class,
                          path="models/appearance_model_latest.pt",
                          optimizer=None,
                          device="cuda"):

    if not os.path.exists(path):
        print("[WARN] No checkpoint found.")
        return model_class().to(device), None

    checkpoint = torch.load(path, map_location=device)

    model = model_class(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    if optimizer and "optimizer_state" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"[WARN] optimizer not loaded: {e}")

    print(
        f"[INFO] Loaded AppearanceModel | "
        f"version={checkpoint.get('model_version')} | "
        f"epoch={checkpoint.get('epoch')} | "
        f"loss={checkpoint.get('loss')} | "
        f"time={checkpoint.get('timestamp')}"
    )

    return model, checkpoint


# =========================================================
# MODEL -on peut rendre ton micro-detail content-aware sans casser la stabilité, sans transformer ton model en usine à features.
# =========================================================

# upgrades (optionnel)
# Sans casser ton système, la première amélioration logique serait :
# 👉 remplacer global pooling par multi-scale pooling
# Ex :
# 1x1 (global)
# 2x2 (semi-local)
# 4x4 (structure)
# Et concat → head
# Mais ça reste une évolution, pas une correction.

import torch
import torch.nn as nn
import torch.nn.functional as F


class AppearanceModel(nn.Module):
    """
    V2.1 — Photometric Renderer + Micro Content Gate (lightweight)
    """

    def __init__(self, in_channels=4, base_channels=32, prompt_dim=768):

        super().__init__()

        self.config = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "prompt_dim": prompt_dim
        }

        self.version = "v2.1_micro_gate"

        # -------------------------------------------------
        # ENCODER
        # -------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------
        self.prompt_proj = nn.Linear(prompt_dim, base_channels)

        # -------------------------------------------------
        # FUSION
        # -------------------------------------------------
        self.fusion = nn.Conv2d(base_channels * 2, base_channels, 1)

        # -------------------------------------------------
        # GLOBAL POOLING
        # -------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)

        hidden = base_channels

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

        # -------------------------------------------------
        # GLOBAL RENDER PARAMETERS
        # -------------------------------------------------
        self.exposure = nn.Linear(hidden, 1)
        self.gamma    = nn.Linear(hidden, 1)
        self.contrast = nn.Linear(hidden, 1)
        self.micro    = nn.Linear(hidden, 1)

        # -------------------------------------------------
        # MICRO CONTENT GATE (NEW)
        # -------------------------------------------------
        self.micro_gate = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x, prompt_emb):

        # -------------------------------------------------
        # ENCODE
        # -------------------------------------------------
        h = self.encoder(x)

        # -------------------------------------------------
        # PROMPT INJECTION
        # -------------------------------------------------
        if prompt_emb.dim() == 3:
            prompt_emb = prompt_emb[:, 0, :]

        p = self.prompt_proj(prompt_emb)
        p = p[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])

        # fusion
        h = self.fusion(torch.cat([h, p], dim=1))

        # -------------------------------------------------
        # GLOBAL FEATURES
        # -------------------------------------------------
        g = self.pool(h).squeeze(-1).squeeze(-1)
        g = self.head(g)

        # -------------------------------------------------
        # MICRO GATE (CONTENT-AWARE MAP)
        # -------------------------------------------------
        micro_gate = torch.sigmoid(self.micro_gate(h))

        # optional mild smoothing (VERY safe)
        micro_gate = F.avg_pool2d(micro_gate, kernel_size=3, stride=1, padding=1)

        return {
            "exposure": self.exposure(g),
            "gamma": self.gamma(g),
            "contrast": self.contrast(g),
            "micro": self.micro(g),
            "micro_gate": micro_gate
        }


# instance par défaut pour ton pipeline
appearance_model = AppearanceModel().cuda()


optimizer_apparence = optim.AdamW(
    appearance_model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.99),
    weight_decay=1e-5
)

criterion_apparence = torch.nn.MSELoss()

# =========================================================
# APPLY FUNCTION
# =========================================================
def extract_latents(x):
    if x is None:
        return None
    if isinstance(x, dict):
        if "latents" in x:
            return x["latents"]
        return next(iter(x.values()))
    return x

def appearance_debug(pred, x, out, hf=None, prefix="[Appearance DEBUG]"):
    exposure = torch.tanh(pred["exposure"]).mean().item()
    gamma    = torch.tanh(pred["gamma"]).mean().item()
    contrast = torch.tanh(pred["contrast"]).mean().item()
    micro    = torch.tanh(pred["micro"]).mean().item()

    print(f"\n{prefix}")
    print(f"params:")
    print(f"  exposure : {exposure:.4f}")
    print(f"  gamma    : {gamma:.4f}")
    print(f"  contrast : {contrast:.4f}")
    print(f"  micro    : {micro:.4f}")

    print(f"image stats:")
    print(f"  mean     : {out.mean().item():.4f}")
    print(f"  std      : {out.std().item():.4f}")
    print(f"  min/max  : {out.min().item():.4f} / {out.max().item():.4f}")

    if hf is not None:
        print(f"  hf energy : {hf.mean().item():.6f}")



def stabilize_latents(latents, target_std=1.0, clamp_value=3.0, eps=1e-6, mode="adaptive"):
    """
    Stabilisation des latents pour renderers photométriques.

    Args:
        latents: Tensor [B, C, H, W]
        target_std: cible de variance globale
        clamp_value: limite pour éviter explosion dynamique
        eps: stabilité numérique
        mode:
            - "adaptive" (recommandé)
            - "hard"
            - "tanh"
    """

    x = latents

    if mode == "tanh":
        # ultra stable mais plus "compressif"
        return torch.tanh(x)

    # =====================================================
    # 1. normalisation globale
    # =====================================================
    mean = x.mean(dim=(1,2,3), keepdim=True)
    std  = x.std(dim=(1,2,3), keepdim=True)

    x = (x - mean) / (std + eps)

    # =====================================================
    # 2. re-scaling contrôlé
    # =====================================================
    if mode == "adaptive":
        # préserve un peu de dynamique originale
        scale = target_std
        x = x * scale

    elif mode == "hard":
        # force stricte stabilité
        x = x * target_std

    # =====================================================
    # 3. clamp sécurité (anti explosion HF)
    # =====================================================
    x = torch.clamp(x, -clamp_value, clamp_value)

    return x



def apply_appearance(
    latents,
    style_prompt_embedding,
    appearance_model,
    optimizer=None,
    criterion=None,
    train=False,
    strength=0.1,
    device="cuda",
    frame_counter=0,
    max_epochs_up=12,
    model_path="models/appearance_model_latest.pt",
    latents_sample=None,
    new_image=False,
    debug=False
):

    device = latents.device
    appearance_model.to(device)

    x = latents.to(device)
    x0 = x.detach()

    # =====================================================
    # LATENT CHECK (NO NORMALIZATION)
    # =====================================================
    print("[LATENT CHECK]")
    print("std :", x0.std().item())

    model_exists = os.path.exists(model_path)

    # =====================================================
    # TRAIN
    # =====================================================
    if train and optimizer is not None and criterion is not None:

        appearance_model.train()
        max_epochs = max(1, max_epochs_up)

        for epoch in range(max_epochs):

            optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(True):

                pred = appearance_model(x0, style_prompt_embedding)

                exposure = pred["exposure"].view(-1,1,1,1)
                gamma    = pred["gamma"].view(-1,1,1,1)
                contrast = pred["contrast"].view(-1,1,1,1)
                micro    = pred["micro"].view(-1,1,1,1)

                # =================================================
                # FORWARD PIPELINE
                # =================================================
                out = x0

                # exposure
                out = out * (1.0 + 0.4 * torch.tanh(exposure))

                # gamma (stable + smooth gain)
                g = torch.tanh(gamma)
                out = torch.sign(out) * (torch.abs(out) ** (1.0 + 0.25 * g))
                out = out * (1.0 + 0.10 * g)

                # contrast
                m = out.mean(dim=(2,3), keepdim=True)
                out = (out - m) * (1.0 + 0.6 * torch.tanh(contrast)) + m

                # =================================================
                # MICRO DETAIL (IMPROVED VERSION)
                # =================================================

                blur = F.avg_pool2d(out, 3, 1, 1)
                detail = out - blur

                # structure component (edges)
                edge = torch.tanh(detail)

                # smooth texture band-pass (less noisy than previous version)
                texture = F.avg_pool2d(detail, 3, 1, 1) - F.avg_pool2d(blur, 3, 1, 1)

                # adaptive gating based on local energy
                hf = compute_high_freq_energy(out)
                #micro_gate = torch.tanh(micro) / (1.0 + 2.0 * hf)

                micro_gate = pred["micro_gate"]  # <-- IMPORTANT (spatial, learned)

                # refined combination
                detail_refined = 0.8 * edge + 0.2 * torch.tanh(texture)

                out = out + 0.15 * micro_gate * detail_refined

                # =================================================
                # LOSS
                # =================================================
                target = x0.detach()

                loss_id = F.l1_loss(out, target)

                loss_struct = F.l1_loss(
                    out - F.avg_pool2d(out, 3, 1, 1),
                    target - F.avg_pool2d(target, 3, 1, 1)
                )

                loss_detail = F.l1_loss(
                    compute_high_freq_energy(out),
                    compute_high_freq_energy(target)
                )

                loss_energy = 0.002 * out.pow(2).mean()

                loss_contrast = F.mse_loss(
                    out.std(dim=(2,3)),
                    target.std(dim=(2,3))
                )

                loss_param_reg = (
                    pred["exposure"].pow(2).mean() +
                    pred["gamma"].pow(2).mean() +
                    pred["contrast"].pow(2).mean() +
                    pred["micro"].pow(2).mean()
                ) * 0.01

                loss = (
                    0.25 * loss_id +
                    0.15 * loss_struct +
                    0.20 * loss_detail +
                    0.02 * loss_energy +
                    0.10 * loss_contrast +
                    loss_param_reg
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(appearance_model.parameters(), 1.0)
            optimizer.step()

            if debug:
                print(f"[Appearance TRAIN] Epoch {epoch+1}/{max_epochs} | Loss={loss.item():.6f}")

            if (frame_counter % 10 == 0) and (epoch == max_epochs - 1):
                save_appearance_model(
                    appearance_model,
                    optimizer=optimizer,
                    epoch=frame_counter,
                    loss=loss.item(),
                    path=model_path
                )

    # =====================================================
    # LOAD
    # =====================================================
    else:

        appearance_model.eval()

        if model_exists:
            appearance_model, _ = load_appearance_model(
                type(appearance_model),
                path=model_path,
                optimizer=optimizer,
                device=device
            )

    # =====================================================
    # INFERENCE
    # =====================================================
    with torch.no_grad():

        pred = appearance_model(x0, style_prompt_embedding)

        exposure = 0.12 * torch.tanh(pred["exposure"])
        gamma    = 0.10 * torch.tanh(pred["gamma"])
        contrast = 0.25 * torch.tanh(pred["contrast"])
        micro    = 0.20 * torch.tanh(pred["micro"])

        out = x0

        out = out * (1.0 + exposure)

        g = gamma
        out = torch.sign(out) * (torch.abs(out) ** (1.0 + g))
        out = out * (1.0 + 0.10 * g)

        m = out.mean(dim=(2,3), keepdim=True)
        out = (out - m) * (1.0 + 1.5 * contrast) + m

        # =================================================
        # MICRO DETAIL (INFERENCE STABLE)
        # =================================================

        blur = F.avg_pool2d(out, 3, 1, 1)
        detail = out - blur

        edge = torch.tanh(detail)
        texture = F.avg_pool2d(detail, 3, 1, 1) - F.avg_pool2d(blur, 3, 1, 1)

        hf = compute_high_freq_energy(out)
        #micro_gate = torch.tanh(micro) / (1.0 + 2.0 * hf)

        micro_gate = pred["micro_gate"]  # <-- IMPORTANT (spatial, learned)

        detail_refined = 0.8 * edge + 0.2 * torch.tanh(texture)

        out = out + 0.15 * micro_gate * detail_refined

    # =====================================================
    # STRENGTH BLENDING
    # =====================================================
    hf = compute_high_freq_energy(x0)

    strength_map = strength / (1.0 + 2.5 * hf)
    strength_map = strength_map.clamp(0.01, strength)

    out = x0 + strength_map * (out - x0)

    if debug:
        appearance_debug(pred=pred, x=x0, out=out, hf=hf)

    return out





