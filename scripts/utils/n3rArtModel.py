# n3rArtModel.py
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from .tools_utils import ensure_4_channels, log_debug, sanitize_latents
from .n3r_EMA import compute_high_freq_energy

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

def stabilize_latents(latents, target_std=1.0):

    std = latents.std(dim=(1,2,3), keepdim=True)
    scale = target_std / (std + 1e-6)
    scale = torch.clamp(scale, 0.7, 1.3)

    return latents * scale

# =========================================================
# SAVE / LOAD
# =========================================================

def save_art_model(
    model,
    optimizer=None,
    epoch=None,
    loss=None,
    path="models/art_model.pt",
    latest_path="models/art_model_latest.pt",

    # art metadata
    style_name=None,
    prompt_signature=None,
    hf_mean=None,
    training_mode="semantic_decorator",

    # optional temporal state
    ema_state=None,

    # extra debug infos
    extra_metadata=None
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

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
            "v1_art"
        ),

        # =================================================
        # TRAINING
        # =================================================

        "epoch": epoch,
        "loss": float(loss) if loss is not None else None,

        # =================================================
        # CREATIVE METADATA
        # =================================================

        "style_name": style_name,
        "prompt_signature": prompt_signature,
        "hf_mean": hf_mean,
        "training_mode": training_mode,

        # =================================================
        # TEMPORAL
        # =================================================

        "ema_state": ema_state,

        # =================================================
        # DEBUG / CUSTOM
        # =================================================

        "extra_metadata": extra_metadata or {},

        # =================================================
        # TIMESTAMP
        # =================================================

        "timestamp": datetime.datetime.now().isoformat()
    }

    # =====================================================
    # OPTIMIZER
    # =====================================================

    if optimizer is not None:

        checkpoint["optimizer_state"] = (
            optimizer.state_dict()
        )

    # =====================================================
    # SAVE
    # =====================================================

    torch.save(checkpoint, path)

    torch.save(checkpoint, latest_path)

    # =====================================================
    # LOG
    # =====================================================

    print(
        f"[INFO] Saved ArtModel | "
        f"version={checkpoint['model_version']} | "
        f"epoch={epoch} | "
        f"loss={loss} | "
        f"path={path}"
    )

    if style_name is not None:

        print(
            f"[Art] style={style_name}"
        )

    if hf_mean is not None:

        print(
            f"[Art] hf_mean={hf_mean:.4f}"
        )

# =========================================================
# LOAD ( V1 COMPATIBLE)
# =========================================================
def load_art_model(
    model_class,
    path="models/art_model_latest.pt",
    optimizer=None,
    device="cuda",

    # compatibility
    strict=False,

    # restore extras
    load_optimizer=True,
    load_ema=True,

    # debug
    verbose=True
):

    # =====================================================
    # CHECKPOINT EXISTS
    # =====================================================

    if not os.path.exists(path):

        print(
            "[WARN] No art checkpoint found."
        )

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
    # BUILD MODEL
    # =====================================================

    model_config = checkpoint.get(
        "model_config",
        {}
    )

    model = model_class(
        **model_config
    ).to(device)

    # =====================================================
    # LOAD WEIGHTS
    # =====================================================

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model_state"],
        strict=strict
    )

    # =====================================================
    # LOAD OPTIMIZER
    # =====================================================

    if (
        optimizer is not None
        and
        load_optimizer
        and
        "optimizer_state" in checkpoint
    ):

        try:

            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )

        except Exception as e:

            print(
                f"[WARN] optimizer not loaded: {e}"
            )

    # =====================================================
    # EXTRAS
    # =====================================================

    ema_state = None

    if load_ema:

        ema_state = checkpoint.get(
            "ema_state",
            None
        )

    # =====================================================
    # METADATA
    # =====================================================

    model_version = checkpoint.get(
        "model_version",
        "unknown"
    )

    style_name = checkpoint.get(
        "style_name",
        None
    )

    prompt_signature = checkpoint.get(
        "prompt_signature",
        None
    )

    hf_mean = checkpoint.get(
        "hf_mean",
        None
    )

    training_mode = checkpoint.get(
        "training_mode",
        None
    )

    # =====================================================
    # LOG
    # =====================================================

    if verbose:

        print(
            f"[INFO] Loaded ArtModel | "
            f"version={model_version} | "
            f"epoch={checkpoint.get('epoch')} | "
            f"loss={checkpoint.get('loss')} | "
            f"time={checkpoint.get('timestamp')}"
        )

        if style_name is not None:

            print(
                f"[Art] style={style_name}"
            )

        if prompt_signature is not None:

            print(
                f"[Art] prompt='{prompt_signature}'"
            )

        if hf_mean is not None:

            print(
                f"[Art] hf_mean={hf_mean:.4f}"
            )

        if training_mode is not None:

            print(
                f"[Art] mode={training_mode}"
            )

        # partial compatibility infos

        if len(missing_keys) > 0:

            print(
                f"[WARN] Missing keys: "
                f"{len(missing_keys)}"
            )

        if len(unexpected_keys) > 0:

            print(
                f"[WARN] Unexpected keys: "
                f"{len(unexpected_keys)}"
            )

    # =====================================================
    # RETURN
    # =====================================================

    checkpoint["_ema_state_loaded"] = ema_state

    return model, checkpoint


# =========================================================
# MODEL ArtDecoratorModelLitePlus
# =========================================================
class ArtDecoratorModel_Safe(nn.Module):
    """
    SAFE Art latent decorator
    - NO attention (OOM-proof)
    - multi-scale perception via dilated convs
    - stable for 3–4GB GPUs
    - compatible diffusion latents pipelines
    """

    def __init__(
        self,
        in_channels=4,
        base_channels=24,
        prompt_dim=768
    ):
        super().__init__()

        self.version = "v1_art_safe"

        # =====================================================
        # LATENT ENCODER
        # =====================================================
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # =====================================================
        # PROMPT PROJECTOR
        # =====================================================
        self.prompt_proj = nn.Linear(prompt_dim, base_channels)

        # =====================================================
        # FUSION LAYER
        # =====================================================
        self.fusion = nn.Conv2d(base_channels * 2, base_channels, 1)

        # =====================================================
        # SAFE MULTI-SCALE CONTEXT (NO ATTENTION)
        # =====================================================

        # local structure
        self.context_low = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            groups=base_channels
        )

        # medium receptive field
        self.context_mid = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=base_channels
        )

        # large receptive field
        self.context_high = nn.Conv2d(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=base_channels
        )

        # =====================================================
        # GLOBAL REPRESENTATION
        # =====================================================
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
            nn.SiLU(),
        )

        # =====================================================
        # CREATIVE OUTPUT HEADS
        # =====================================================
        self.structure = nn.Linear(base_channels, 1)
        self.texture   = nn.Linear(base_channels, 1)
        self.style     = nn.Linear(base_channels, 1)
        self.chaos     = nn.Linear(base_channels, 1)
        self.rhythm    = nn.Linear(base_channels, 1)

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(self, x, prompt_emb):

        # -------------------------------------------------
        # latent encoding
        # -------------------------------------------------
        h = self.encoder(x)

        # -------------------------------------------------
        # prompt handling (safe reduction)
        # -------------------------------------------------
        if prompt_emb.dim() == 3:
            prompt_emb = prompt_emb.mean(dim=1)

        p = self.prompt_proj(prompt_emb)

        # reshape spatial conditioning
        p = p[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])

        # mild boost (safe)
        p = p * 1.5

        # -------------------------------------------------
        # fusion
        # -------------------------------------------------
        h = self.fusion(torch.cat([h, p], dim=1))

        # =====================================================
        # SAFE MULTI-SCALE CONTEXT (NO ATTENTION)
        # =====================================================

        low  = self.context_low(h)
        mid  = self.context_mid(h)
        high = self.context_high(h)

        # weighted residual mixing
        h = h + 0.06 * low + 0.04 * mid + 0.02 * high

        # =====================================================
        # GLOBAL POOLING
        # =====================================================
        g = self.pool(h).squeeze(-1).squeeze(-1)

        g = self.head(g)

        # =====================================================
        # CREATIVE OUTPUTS
        # =====================================================
        return {
            "structure": self.structure(g),
            "texture": self.texture(g),
            "style": self.style(g),
            "chaos": self.chaos(g),
            "rhythm": self.rhythm(g),
        }


class ArtDecoratorModel(nn.Module):
    """
    Art latent decorator - Lite+ version
    Balanced for stability + micro-detail enhancement
    """

    def __init__(
        self,
        in_channels=4,
        base_channels=24,
        prompt_dim=768
    ):
        super().__init__()

        self.config = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "prompt_dim": prompt_dim
        }

        self.version = "v1_art_lite_plus"

        # =====================================================
        # LATENT ENCODER (stable + normalized)
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
        # PROMPT PROJECTION
        # =====================================================

        self.prompt_proj = nn.Linear(prompt_dim, base_channels)

        # =====================================================
        # FUSION (residual-safe)
        # =====================================================

        self.fusion = nn.Conv2d(base_channels * 2, base_channels, 1)

        # =====================================================
        # MICRO ATTENTION (cheap spatial gating)
        # =====================================================

        self.micro_attn = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, groups=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid()
        )

        # =====================================================
        # GLOBAL HEAD
        # =====================================================

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Linear(base_channels, base_channels * 2),
            nn.SiLU(),
            nn.Linear(base_channels * 2, base_channels),
            nn.SiLU(),
        )

        # =====================================================
        # CREATIVE OUTPUT HEADS
        # =====================================================

        self.structure = nn.Linear(base_channels, 1)
        self.texture   = nn.Linear(base_channels, 1)
        self.style     = nn.Linear(base_channels, 1)
        self.chaos     = nn.Linear(base_channels, 1)
        self.rhythm    = nn.Linear(base_channels, 1)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x, prompt_emb):

        # =====================================================
        # ENCODING
        # =====================================================

        h = self.encoder(x)

        # =====================================================
        # PROMPT NORMALIZATION
        # =====================================================

        if prompt_emb.dim() == 3:
            prompt_emb = prompt_emb.mean(dim=1)

        prompt_emb = torch.tanh(prompt_emb)
        prompt_emb = prompt_emb / (prompt_emb.std(dim=-1, keepdim=True) + 1e-6)

        # =====================================================
        # PROMPT PROJECT
        # =====================================================

        p = self.prompt_proj(prompt_emb)

        p = p[:, :, None, None].expand(
            -1, -1, h.shape[2], h.shape[3]
        )

        # léger boost signal
        p = p * 1.5

        # =====================================================
        # FUSION (RESIDUAL SAFE)
        # =====================================================

        fused = self.fusion(torch.cat([h, p], dim=1))
        h = h + 0.5 * fused

        # =====================================================
        # MICRO ATTENTION (VERY IMPORTANT)
        # =====================================================

        attn = self.micro_attn(h)
        h = h * (0.75 + 0.25 * attn)

        # =====================================================
        # EDGE ENHANCEMENT (LOW COST)
        # =====================================================

        edges = h - F.avg_pool2d(h, 3, 1, 1)
        edge_strength = edges.abs().mean(dim=1, keepdim=True)

        h = h + 0.05 * edge_strength * edges

        # =====================================================
        # GLOBAL REPRESENTATION
        # =====================================================

        g = self.pool(h).squeeze(-1).squeeze(-1)
        g = self.head(g)

        # =====================================================
        # CREATIVE OUTPUT
        # =====================================================

        return {
            "structure": self.structure(g),
            "texture": self.texture(g),
            "style": self.style(g),
            "chaos": self.chaos(g),
            "rhythm": self.rhythm(g)
        }
# =========================================================
# MODEL ArtDecoratorModel
# =========================================================
class ArtDecoratorModel_v1(nn.Module):
    """
    Art semantic latent decorator
    """

    def __init__(
        self,
        in_channels=4,
        base_channels=32,
        prompt_dim=768
    ):

        super().__init__()

        self.config = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "prompt_dim": prompt_dim
        }

        self.version = "v1_art"

        # -------------------------------------------------
        # LATENT ENCODER
        # -------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),

            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        # -------------------------------------------------
        # PROMPT ENCODER
        # -------------------------------------------------

        self.prompt_proj = nn.Linear(prompt_dim, base_channels)

        # -------------------------------------------------
        # FUSION
        # -------------------------------------------------

        self.fusion = nn.Conv2d(
            base_channels * 2,
            base_channels,
            1
        )

        # -------------------------------------------------
        # GLOBAL
        # -------------------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(1)
        """
        self.head = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.SiLU()
        )
        """
        self.head = nn.Sequential(

            nn.Linear(base_channels, base_channels * 4),
            nn.SiLU(),

            nn.Linear(base_channels * 4, base_channels * 2),
            nn.SiLU(),

            nn.Linear(base_channels * 2, base_channels),
            nn.SiLU(),
        )

        # -------------------------------------------------
        # CREATIVE HEADS
        # -------------------------------------------------

        self.structure = nn.Linear(base_channels, 1)
        self.texture   = nn.Linear(base_channels, 1)
        self.style     = nn.Linear(base_channels, 1)
        self.chaos     = nn.Linear(base_channels, 1)
        self.rhythm    = nn.Linear(base_channels, 1)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x, prompt_emb):

        # =====================================================
        # LATENT ENCODER
        # =====================================================

        h = self.encoder(x)

        # =====================================================
        # PROMPT PROCESSING (IMPORTANT FIX)
        # =====================================================

        if prompt_emb.dim() == 3:

            # SAFE + INFORMATIVE POOLING
            prompt_emb = prompt_emb.mean(dim=1)

            print(
                "[Prompt pooled]",
                prompt_emb.mean().item(),
                prompt_emb.std().item(),
                prompt_emb.min().item(),
                prompt_emb.max().item()
            )

        # =====================================================
        # PROMPT PROJECTION
        # =====================================================

        p = self.prompt_proj(prompt_emb)

        # reshape spatial conditioning
        p = p[:, :, None, None].expand(
            -1,
            -1,
            h.shape[2],
            h.shape[3]
        )

        # =====================================================
        # STRENGTHEN PROMPT SIGNAL (CRITICAL)
        # =====================================================

        p = p * 2.0  # boost signal (important for non-collapse)

        # optional light stochasticity (helps avoid dead fusion)
        p = p * (1.0 + 0.05 * torch.randn_like(p))

        # =====================================================
        # FUSION
        # =====================================================

        h = self.fusion(torch.cat([h, p], dim=1))

        # =====================================================
        # DEBUG (SAFE, INFORMATIVE)
        # =====================================================

        print(
            "p_norm:", p.norm().item(),
            "h_std:", h.std().item()
        )

        # =====================================================
        # GLOBAL POOLING
        # =====================================================

        g = self.pool(h).squeeze(-1).squeeze(-1)

        # =====================================================
        # SHARED REPRESENTATION HEAD
        # =====================================================

        g = self.head(g)

        # =====================================================
        # CREATIVE HEADS OUTPUT
        # =====================================================

        return {
            "structure": self.structure(g),
            "texture": self.texture(g),
            "style": self.style(g),
            "chaos": self.chaos(g),
            "rhythm": self.rhythm(g)
        }


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

art_model = ArtDecoratorModel().to(device)

optimizer_art = optim.AdamW(

    art_model.parameters(),

    lr=2e-4,

    betas=(0.9, 0.99),

    weight_decay=1e-5
)

criterion_art = torch.nn.L1Loss()

# =========================================================
# APPLY FUNCTION
# =========================================================*

def apply_art(
    latents,
    style_prompt_embedding,
    art_model,
    optimizer=None,
    criterion=None,
    train=False,
    strength=0.50,
    device="cuda",
    frame_counter=0,
    max_epochs_up=6,
    model_path="models/art_model_latest.pt",
    new_image=False,
    debug=False
):


    device = latents.device
    art_model.to(device)

    x = latents.to(device)
    x0 = x.detach()

    print(f"[Art STABLE] device={device}")

    # =====================================================
    # SAFE PROMPT NORMALIZATION
    # =====================================================

    if style_prompt_embedding.dim() == 3:
        prompt = style_prompt_embedding.mean(dim=1)
    else:
        prompt = style_prompt_embedding

    prompt = torch.tanh(prompt)
    prompt = prompt / (
        prompt.std(dim=-1, keepdim=True) + 1e-6
    )

    # =====================================================
    # HIGH FREQUENCY ENERGY
    # =====================================================

    hf = compute_high_freq_energy(x0)

    print(
        f"[Art STABLE] hf={hf.mean().item():.4f}"
    )

    model_exists = os.path.exists(model_path)

    # =====================================================
    # INTERNAL CREATIVE PASS
    # =====================================================

    def art_pass(inp, pred_dict):

        out = inp

        # =================================================
        # SAFE PARAMS
        # =================================================

        structure = torch.tanh(
            pred_dict["structure"]
        ).view(-1,1,1,1)

        texture = torch.tanh(
            pred_dict["texture"]
        ).view(-1,1,1,1)

        style = torch.tanh(
            pred_dict["style"]
        ).view(-1,1,1,1)

        chaos = torch.tanh(
            pred_dict["chaos"]
        ).view(-1,1,1,1)

        rhythm = torch.tanh(
            pred_dict["rhythm"]
        ).view(-1,1,1,1)

        # =================================================
        # SPATIAL ATTENTION MAP
        # =================================================

        attention = torch.mean(
            torch.abs(out),
            dim=1,
            keepdim=True
        )

        attention = attention / (
            attention.amax(
                dim=(2,3),
                keepdim=True
            ) + 1e-6
        )

        attention = torch.pow(attention, 0.7)

        # =================================================
        # STRUCTURE
        # =================================================

        low = F.avg_pool2d(out, 5, 1, 2)

        out = out + 0.06 * structure * (low - out)

        # =================================================
        # MULTI SCALE DETAIL
        # =================================================

        blur3 = F.avg_pool2d(out, 3, 1, 1)
        blur5 = F.avg_pool2d(out, 5, 1, 2)

        detail_small = out - blur3
        detail_large = blur3 - blur5

        detail = (
            0.7 * detail_small +
            0.3 * detail_large
        )

        # =================================================
        # EDGE MASK
        # =================================================

        edge_energy = torch.abs(detail)

        edge_mask = (
            edge_energy >
            edge_energy.mean(
                dim=(2,3),
                keepdim=True
            )
        ).float()

        edge_mask = F.avg_pool2d(
            edge_mask,
            3,
            1,
            1
        )

        # =================================================
        # TEXTURE INJECTION
        # =================================================

        out = (
            out +
            0.08 *
            texture *
            detail *
            attention
        )

        # =================================================
        # EDGE PRESERVATION
        # =================================================

        out = (
            out +
            0.03 *
            edge_mask *
            detail
        )

        # =================================================
        # MICRO CONTRAST
        # =================================================

        micro = (
            out -
            F.avg_pool2d(out, 7, 1, 3)
        )

        out = (
            out +
            0.015 *
            micro *
            attention
        )

        # =================================================
        # STYLE MAP
        # =================================================

        style_map = torch.sin(out * math.pi)

        out = (
            out +
            0.04 *
            style *
            style_map *
            attention
        )

        # =================================================
        # CHAOS (ATTENTION GUIDED)
        # =================================================

        noise = torch.randn_like(out)

        noise = F.avg_pool2d(
            noise,
            3,
            1,
            1
        )

        out = (
            out +
            0.01 *
            chaos *
            noise *
            attention *
            0.5
        )

        # =================================================
        # RHYTHM
        # =================================================

        wave = torch.sin(out * 4.0)

        out = (
            out +
            0.02 *
            rhythm *
            wave *
            attention
        )

        return (
            out,
            structure,
            texture,
            style,
            chaos,
            rhythm,
            attention,
            detail,
            edge_mask
        )

    # =====================================================
    # TRAIN
    # =====================================================

    if train and optimizer and criterion:

        art_model.train()

        max_epochs = max(
            1,
            max_epochs_up
        )

        print("[Art STABLE] Training")

        for epoch in range(max_epochs):

            optimizer.zero_grad()

            with torch.enable_grad():

                pred = art_model(
                    x0,
                    prompt
                )

                (
                    out,
                    structure,
                    texture,
                    style,
                    chaos,
                    rhythm,
                    attention,
                    detail,
                    edge_mask
                ) = art_pass(x0, pred)

                # =========================================
                # LOSSES
                # =========================================

                loss_id = (
                    0.05 *
                    F.l1_loss(out, x0)
                )

                loss_detail = (
                    0.02 *
                    F.l1_loss(
                        compute_high_freq_energy(out),
                        compute_high_freq_energy(x0)
                    )
                )

                loss_energy = (
                    0.001 *
                    out.pow(2).mean()
                )

                loss_stability = (
                    0.01 *
                    (out - x0).pow(2).mean()
                )

                # edge preservation loss
                edge_out = (
                    out -
                    F.avg_pool2d(out, 3, 1, 1)
                )

                edge_x0 = (
                    x0 -
                    F.avg_pool2d(x0, 3, 1, 1)
                )

                loss_edges = (
                    0.01 *
                    F.l1_loss(
                        edge_out,
                        edge_x0
                    )
                )

                loss = (
                    loss_id +
                    loss_detail +
                    loss_energy +
                    loss_stability +
                    loss_edges
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                art_model.parameters(),
                0.8
            )

            optimizer.step()

            print( f"[Art STABLE] " f"Epoch {epoch+1}/{max_epochs} | " f"Loss={loss.item():.6f}" )

            if (
                frame_counter % 10 == 0
            ) and (
                epoch == max_epochs - 1
            ):

                save_art_model( art_model, optimizer=optimizer, epoch=frame_counter, loss=loss.item(), path=model_path )

    # =====================================================
    # LOAD
    # =====================================================

    else:

        art_model.eval()

        if model_exists:

            art_model, _ = load_art_model(
                type(art_model),
                path=model_path,
                optimizer=optimizer,
                device=device
            )

    # =====================================================
    # INFERENCE
    # =====================================================

    with torch.no_grad():

        pred = art_model(x, prompt)

        (
            out,
            structure,
            texture,
            style,
            chaos,
            rhythm,
            attention,
            detail,
            edge_mask
        ) = art_pass(x, pred)

        print(
            f"[Art STABLE] "
            f"struct={structure.mean().item():.4f} | "
            f"texture={texture.mean().item():.4f} | "
            f"style={style.mean().item():.4f} | "
            f"chaos={chaos.mean().item():.4f} | "
            f"rhythm={rhythm.mean().item():.4f}"
        )

        print(
            f"[Art STABLE] "
            f"attention={attention.mean().item():.4f} | "
            f"detail={detail.abs().mean().item():.4f} | "
            f"edge={edge_mask.mean().item():.4f}"
        )

    # =====================================================
    # STRENGTH CONTROL
    # =====================================================

    hf = compute_high_freq_energy(x)

    strength_map = (
        strength /
        (1.0 + 4.0 * hf)
    )

    strength_map = strength_map.clamp(
        0.01,
        strength
    )

    out = (
        x +
        strength_map *
        (out - x)
    )

    # =====================================================
    # FINAL STABILIZATION
    # =====================================================

    out = torch.nan_to_num(
        out,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0
    )


    return out

