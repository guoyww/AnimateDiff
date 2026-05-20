# n3rCreativeModel.py
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from .tools_utils import ensure_4_channels, log_debug, sanitize_latents
from .n3r_EMA import motion_aware_ema_fusion
from .n3r_latent_utils import compute_high_freq_energy

# =========================================================
# SAVE / LOAD
# =========================================================

def save_creative_model(
    model,
    optimizer=None,
    epoch=None,
    loss=None,

    path="models/creative_model.pt",
    latest_path="models/creative_model_latest.pt",

    # =====================================================
    # CREATIVE METADATA
    # =====================================================

    style_name=None,
    prompt_signature=None,
    hf_mean=None,
    training_mode="latent_style_transfer",

    # =====================================================
    # STYLE LATENT
    # =====================================================

    style_latent=None,
    style_embedding=None,

    # =====================================================
    # TEMPORAL
    # =====================================================

    ema_state=None,

    # =====================================================
    # DEBUG / CUSTOM
    # =====================================================

    extra_metadata=None,

    # =====================================================
    # SAVE OPTIONS
    # =====================================================

    save_latest=True,
    verbose=True
):

    import os
    import torch
    import datetime

    # =====================================================
    # CREATE DIR
    # =====================================================

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    # =====================================================
    # STYLE LATENT STATS
    # =====================================================

    style_latent_stats = None

    if style_latent is not None:

        try:

            style_latent_stats = {

                "shape": list(style_latent.shape),

                "mean": float(
                    style_latent.mean().item()
                ),

                "std": float(
                    style_latent.std().item()
                ),

                "min": float(
                    style_latent.min().item()
                ),

                "max": float(
                    style_latent.max().item()
                ),

                "dtype": str(style_latent.dtype),

                "device": str(style_latent.device),
            }

        except Exception as e:

            print(
                f"[WARN] style_latent stats failed: {e}"
            )

    # =====================================================
    # STYLE EMBEDDING STATS
    # =====================================================

    style_embedding_stats = None

    if style_embedding is not None:

        try:

            style_embedding_stats = {

                "shape": list(style_embedding.shape),

                "mean": float(
                    style_embedding.mean().item()
                ),

                "std": float(
                    style_embedding.std().item()
                ),
            }

        except Exception as e:

            print(
                f"[WARN] style_embedding stats failed: {e}"
            )

    # =====================================================
    # CHECKPOINT
    # =====================================================

    checkpoint = {

        # =================================================
        # MODEL
        # =================================================

        "model_state": model.state_dict(),

        "model_config": getattr(
            model,
            "config",
            {}
        ),

        "model_version": getattr(
            model,
            "version",
            "v2_style_latent"
        ),

        # =================================================
        # TRAINING
        # =================================================

        "epoch": epoch,

        "loss": (
            float(loss)
            if loss is not None
            else None
        ),

        # =================================================
        # CREATIVE METADATA
        # =================================================

        "style_name": style_name,

        "prompt_signature": prompt_signature,

        "hf_mean": hf_mean,

        "training_mode": training_mode,

        # =================================================
        # STYLE LEARNING
        # =================================================

        "style_latent_stats": style_latent_stats,

        "style_embedding_stats": style_embedding_stats,

        # =================================================
        # TEMPORAL
        # =================================================

        "ema_state": ema_state,

        # =================================================
        # DEBUG / CUSTOM
        # =================================================

        "extra_metadata": (
            extra_metadata or {}
        ),

        # =================================================
        # TIMESTAMP
        # =================================================

        "timestamp": (
            datetime.datetime.now().isoformat()
        )
    }

    # =====================================================
    # OPTIMIZER
    # =====================================================

    if optimizer is not None:

        try:

            checkpoint["optimizer_state"] = (
                optimizer.state_dict()
            )

        except Exception as e:

            print(
                f"[WARN] optimizer save failed: {e}"
            )

    # =====================================================
    # SAVE MAIN
    # =====================================================

    torch.save(
        checkpoint,
        path
    )

    # =====================================================
    # SAVE LATEST
    # =====================================================

    if save_latest:

        torch.save(
            checkpoint,
            latest_path
        )

    # =====================================================
    # LOG
    # =====================================================

    if verbose:

        print(
            f"[INFO] Saved CreativeModel | "
            f"version={checkpoint['model_version']} | "
            f"epoch={epoch} | "
            f"loss={loss} | "
            f"path={path}"
        )

        if style_name is not None:

            print(
                f"[Creative] style={style_name}"
            )

        if hf_mean is not None:

            print(
                f"[Creative] hf_mean={hf_mean:.4f}"
            )

        if style_latent_stats is not None:

            print(
                f"[Creative] style_latent "
                f"shape={style_latent_stats['shape']} | "
                f"mean={style_latent_stats['mean']:.4f} | "
                f"std={style_latent_stats['std']:.4f}"
            )

        if style_embedding_stats is not None:

            print(
                f"[Creative] style_embedding "
                f"shape={style_embedding_stats['shape']} | "
                f"std={style_embedding_stats['std']:.4f}"
            )

    return checkpoint

# =========================================================
# LOAD ( V1 COMPATIBLE)
# =========================================================
def load_creative_model(
    model_class,
    path="models/creative_model_latest.pt",
    optimizer=None,
    device="cuda",

    # compatibility
    strict=False,

    # restore extras
    load_optimizer=True,
    load_ema=True,
    load_style_stats=True,

    # debug
    verbose=True
):

    import os
    import torch

    # =====================================================
    # CHECKPOINT EXISTS
    # =====================================================

    if not os.path.exists(path):

        print("[WARN] No creative checkpoint found.")

        model = model_class().to(device)

        return model, None

    # =====================================================
    # LOAD CHECKPOINT
    # =====================================================

    checkpoint = torch.load(
        path,
        map_location=device
    )

    # =====================================================
    # MODEL BUILD
    # =====================================================

    model_config = checkpoint.get("model_config", {})

    try:

        model = model_class(**model_config).to(device)

    except Exception as e:

        print(f"[WARN] model_config incompatible: {e}")
        print("[INFO] fallback to empty init")

        model = model_class().to(device)

    # =====================================================
    # LOAD WEIGHTS
    # =====================================================

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state"],
        strict=strict
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    if (
        optimizer is not None
        and load_optimizer
        and "optimizer_state" in checkpoint
    ):

        try:

            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

        except Exception as e:

            print(f"[WARN] optimizer not loaded: {e}")

    # =====================================================
    # EMA
    # =====================================================

    ema_state = None

    if load_ema:

        ema_state = checkpoint.get("ema_state", None)

    # =====================================================
    # STYLE STATS (NEW)
    # =====================================================

    style_latent_stats = None
    style_embedding_stats = None

    if load_style_stats:

        style_latent_stats = checkpoint.get(
            "style_latent_stats",
            None
        )

        style_embedding_stats = checkpoint.get(
            "style_embedding_stats",
            None
        )

    # =====================================================
    # METADATA
    # =====================================================

    model_version = checkpoint.get("model_version", "unknown")
    style_name = checkpoint.get("style_name", None)
    prompt_signature = checkpoint.get("prompt_signature", None)
    hf_mean = checkpoint.get("hf_mean", None)
    training_mode = checkpoint.get("training_mode", None)

    # =====================================================
    # LOG
    # =====================================================

    if verbose:

        print(
            f"[INFO] Loaded CreativeModel | "
            f"version={model_version} | "
            f"epoch={checkpoint.get('epoch')} | "
            f"loss={checkpoint.get('loss')} | "
            f"time={checkpoint.get('timestamp')}"
        )

        if style_name is not None:
            print(f"[Creative] style={style_name}")

        if prompt_signature is not None:
            print(f"[Creative] prompt='{prompt_signature}'")

        if hf_mean is not None:
            print(f"[Creative] hf_mean={hf_mean:.4f}")

        if training_mode is not None:
            print(f"[Creative] mode={training_mode}")

        # =================================================
        # STYLE DEBUG
        # =================================================

        if style_latent_stats is not None:
            print(
                f"[Creative] style_latent "
                f"shape={style_latent_stats.get('shape')} | "
                f"mean={style_latent_stats.get('mean', 0):.4f} | "
                f"std={style_latent_stats.get('std', 0):.4f}"
            )

        if style_embedding_stats is not None:
            print(
                f"[Creative] style_embedding "
                f"shape={style_embedding_stats.get('shape')} | "
                f"std={style_embedding_stats.get('std', 0):.4f}"
            )

        # =================================================
        # COMPAT WARNINGS
        # =================================================

        if len(missing_keys) > 0:
            print(f"[WARN] Missing keys: {len(missing_keys)}")

        if len(unexpected_keys) > 0:
            print(f"[WARN] Unexpected keys: {len(unexpected_keys)}")

    # =====================================================
    # RETURN
    # =====================================================

    checkpoint["_ema_state_loaded"] = ema_state
    checkpoint["_style_latent_stats"] = style_latent_stats
    checkpoint["_style_embedding_stats"] = style_embedding_stats

    return model, checkpoint


# =========================================================
# MODEL CreativeDecoratorModel
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CreativeDecoratorModel(nn.Module):

    def __init__(
        self,
        in_channels=4,
        base_channels=24,
        prompt_dim=768,
        style_dim=128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.style_dim = style_dim

        # =====================================================
        # LATENT ENCODER
        # =====================================================

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
        )

        # =====================================================
        # PROMPT PROJECT
        # =====================================================

        self.prompt_proj = nn.Linear(prompt_dim, base_channels)

        # =====================================================
        # STYLE ENCODER
        # =====================================================

        self.style_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.style_proj = nn.Linear(base_channels, style_dim)

        # =====================================================
        # STYLE -> FEATURE
        # =====================================================

        self.style_to_feat = nn.Linear(style_dim, base_channels)

        # learnable gates (important pour stabilité)
        self.prompt_gate = nn.Parameter(torch.tensor(0.5))
        self.style_gate = nn.Parameter(torch.tensor(0.5))

        # =====================================================
        # FUSION
        # =====================================================

        self.fusion = nn.Conv2d(base_channels * 3, base_channels, 1)

        # =====================================================
        # GLOBAL REPRESENTATION
        # =====================================================

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, base_channels),
            nn.SiLU(),
        )

        # STYLE OUTPUT (main target)
        self.style_score = nn.Linear(base_channels, 1)

    # =========================================================
    # STYLE ENCODING
    # =========================================================

    def encode_style(self, style_latent):
        s = self.style_encoder(style_latent)
        s = s.flatten(1)
        s = self.style_proj(s)
        return F.normalize(s, dim=-1)

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(self, x, prompt_emb, style_latent=None):

        B, C, H, W = x.shape

        # ----------------------------
        # latent encoding
        # ----------------------------

        h = self.encoder(x)

        # ----------------------------
        # prompt normalize
        # ----------------------------

        if prompt_emb.dim() == 3:
            prompt_emb = prompt_emb.mean(dim=1)

        prompt_emb = torch.tanh(prompt_emb)
        prompt_emb = prompt_emb / (prompt_emb.std(dim=-1, keepdim=True) + 1e-6)

        p = self.prompt_proj(prompt_emb)
        p = p[:, :, None, None].expand(B, -1, H, W)

        # ----------------------------
        # style encoding
        # ----------------------------

        if style_latent is not None:
            s_emb = self.encode_style(style_latent)
            s = self.style_to_feat(s_emb)
            s = s[:, :, None, None].expand(B, -1, H, W)
        else:
            s = torch.zeros_like(h)

        # ----------------------------
        # fusion
        # ----------------------------

        fused = self.fusion(torch.cat([h, p, s], dim=1))

        h = h + (
            self.prompt_gate * p +
            self.style_gate * s +
            0.5 * fused
        )

        # ----------------------------
        # global decision
        # ----------------------------

        g = self.pool(h).flatten(1)
        g = self.head(g)

        return {
            "style_score": self.style_score(g),
            "style_embedding": s_emb if style_latent is not None else None
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

creative_model = CreativeDecoratorModel().to(device)

optimizer_creative = torch.optim.AdamW(
    creative_model.parameters(),
    lr=1e-4,  # plus stable que 2e-4
    betas=(0.9, 0.99),
    weight_decay=1e-5
)

criterion_creative = torch.nn.L1Loss()

# (optionnel mais fortement recommandé)
from copy import deepcopy
ema_creative_model = deepcopy(creative_model)
ema_decay = 0.999

# =========================================================
# APPLY FUNCTION
# =========================================================
def apply_creative(
    latents,
    latents_sample,
    style_prompt_embedding,
    creative_model,
    optimizer=None,
    criterion=None,
    train=False,
    strength=0.10,
    device="cuda",
    frame_counter=0,
    max_epochs_up=6,
    model_path="models/creative_model_latest.pt",
    new_image=False,
    debug=True,
    stable=False #Stable ou Créatif
):

    import os
    import torch
    import torch.nn.functional as F

    device = latents.device
    creative_model.to(device)

    x = latents.to(device)
    x0 = x.detach()

    print(f"[Creative FINAL] device={device}")

    # =====================================================
    # STYLE LATENT
    # =====================================================

    style_latent = None
    if latents_sample is not None:

        style_latent = (
            latents_sample["samples"]
            if isinstance(latents_sample, dict)
            else latents_sample
        ).to(device)

        if debug:
            print(f"[Creative FINAL] style shape={tuple(style_latent.shape)}")
            print(f"[Creative FINAL] mean={style_latent.mean().item():.4f} std={style_latent.std().item():.4f}")

    # =====================================================
    # PROMPT CLEANING
    # =====================================================

    if style_prompt_embedding.dim() == 3:
        prompt = style_prompt_embedding.mean(dim=1)
    else:
        prompt = style_prompt_embedding

    prompt = torch.tanh(prompt)
    prompt = prompt / (prompt.std(dim=-1, keepdim=True) + 1e-6)

    # =====================================================
    # HIGH FREQUENCY ENERGY
    # =====================================================

    hf = compute_high_freq_energy(x0)

    if debug:
        print(f"[Creative FINAL] hf={hf.mean().item():.4f}")

    model_exists = os.path.exists(model_path)

    # =====================================================
    # MODEL LOAD (if inference)
    # =====================================================

    if (not train) and model_exists:
        creative_model, _ = load_creative_model(
            type(creative_model),
            path=model_path,
            optimizer=optimizer,
            device=device
        )

    # =====================================================
    # TRAINING
    # =====================================================

    if train and optimizer and criterion:

        creative_model.train()

        for epoch in range(max_epochs_up):

            optimizer.zero_grad()
            with torch.enable_grad():

                pred = creative_model(
                    x0,
                    prompt,
                    style_latent=style_latent
                )

                style_score = pred["style_score"]

                # -----------------------------
                # SAFE STYLE TARGET
                # -----------------------------
                if style_latent is not None:
                    target_style = style_latent.mean(dim=(2,3))
                    target_style = target_style.mean(dim=1, keepdim=True)
                    target_style = torch.tanh(target_style)
                else:
                    target_style = torch.zeros_like(style_score)

                loss_recon = F.l1_loss(style_score, target_style)

                loss_detail = 0.02 * F.l1_loss(
                    compute_high_freq_energy(x0),
                    compute_high_freq_energy(x0)
                )

                loss = loss_recon + loss_detail

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                creative_model.parameters(),
                0.8
            )

            optimizer.step()

            print(f"[Creative FINAL] Epoch {epoch+1}/{max_epochs_up} | Loss={loss.item():.6f}")

        save_creative_model(
            creative_model,
            optimizer=optimizer,
            epoch=frame_counter,
            loss=loss.item(),
            path=model_path
        )

    # =====================================================
    # INFERENCE
    # =====================================================

    creative_model.eval()

    with torch.no_grad():

        pred = creative_model(
            x,
            prompt,
            style_latent=style_latent
        )

        if debug:
            print(
                f"[Creative FINAL] style_score={pred['style_score'].mean().item():.4f}"
            )

    # =====================================================
    # STYLE GUIDED LATENT MODIFICATION
    # =====================================================
    # -----------------------------
    # STYLE STRENGTH
    # -----------------------------
    style_strength = torch.tanh(pred["style_score"] * 1.5)
    style_strength = style_strength * 2.0
    style_strength = (style_strength - 0.5) * 2.0
    style_strength = style_strength * 2.5
    style_strength = torch.clamp(style_strength, -2.0, 2.0)

    # -----------------------------
    # HF MAP
    # -----------------------------
    hf_map = compute_high_freq_energy(x)
    hf_map = hf_map / (hf_map.mean() + 1e-6)
    hf_map = hf_map ** 1.3

    #strength_map = strength * (0.4 + 1.2 * hf_map)
    strength_map = strength * (0.8 - 0.6 * hf_map)
    strength_map = strength_map.clamp(0.2 * strength, strength)

    # -----------------------------
    # IMPORTANT FIX: better direction
    # -----------------------------
    with torch.no_grad():

        # feature space (24 channels)
        features = creative_model.encoder(x)
        features = torch.tanh(features)

        # compress spatially but keep structure
        spatial = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)

        # high frequency emphasis
        hf = features - spatial

        direction = torch.zeros_like(x)

        direction = F.conv2d(
            features,
            weight=torch.ones(4, features.shape[1], 1, 1, device=x.device) / features.shape[1],
            bias=None
        )


    alpha = strength_map * torch.sigmoid(style_strength)
    style_map = torch.tanh(direction)
    out = x * (1.0 - alpha) + style_map * alpha


    # =====================================================
    # CLEAN STABILITY
    # =====================================================

    out = torch.nan_to_num(
        out,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0
    )

    return out
