import os
import datetime
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .n3r_EMA import motion_aware_ema_fusion, compute_high_freq_energy

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# SEED
# =========================================================

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =========================================================
# SAVE / LOAD
# =========================================================

def save_style_model(
    model,
    optimizer=None,
    epoch=None,
    loss=None,
    path="models/style_injector.pt",
    latest_path="models/style_injector_latest.pt"
):
    """
    Sauvegarde du modèle StyleInjector.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "timestamp": datetime.datetime.now().isoformat()
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if loss is not None:
        checkpoint["loss"] = loss

    torch.save(checkpoint, path)
    torch.save(checkpoint, latest_path)

    print(f"[INFO] StyleInjector saved -> {path}")


def load_style_model(
    model_class,
    path="models/style_injector_latest.pt",
    optimizer=None,
    device=device
):
    """
    Chargement du modèle StyleInjector.
    """

    if not os.path.exists(path):
        print("[WARN] No checkpoint found.")
        model = model_class().to(device)
        return model, None

    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("model_config", {})

    model = model_class(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(
        f"[INFO] Loaded StyleInjector | "
        f"epoch={checkpoint.get('epoch')} | "
        f"loss={checkpoint.get('loss')} | "
        f"time={checkpoint.get('timestamp')}"
    )

    return model, checkpoint

# =========================================================
# WEIGHT INIT
# =========================================================

def weights_init(m):
    """
    Initialisation des poids.
    """

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)

        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            init.ones_(m.weight)

        if m.bias is not None:
            init.zeros_(m.bias)

# =========================================================
# LATENT SANITIZE
# =========================================================

def sanitize_latents(latents):
    """
    Normalisation par sample.
    """

    latents = latents.float()

    mean = latents.mean(dim=(1, 2, 3), keepdim=True)
    std = latents.std(dim=(1, 2, 3), keepdim=True)

    latents = (latents - mean) / (std + 1e-6)
    latents = latents.clamp(-4.0, 4.0)

    return latents

# =========================================================
# STYLE LOSS
# =========================================================

class StyleLoss(nn.Module):
    """
    Loss principale pour l'entraînement.
    """

    def __init__(
        self,
        l1_weight=1.0,
        reduction="mean"
    ):
        super().__init__()

        self.l1 = nn.L1Loss(reduction=reduction)

        self.l1_weight = l1_weight

    def forward(
        self,
        pred_latents,
        target_latents
    ):
        """
        Args:
            pred_latents: latents prédits
            target_latents: latents cibles stylisés
        """

        l1_loss = self.l1(pred_latents, target_latents)

        total_loss = self.l1_weight * l1_loss

        return total_loss

# =========================================================
# RESIDUAL BLOCK
# =========================================================

class ResidualBlock(nn.Module):
    """
    Bloc résiduel stable.
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(8, channels),
            nn.SiLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        return x + self.block(x)
# =========================================================
# CHANNEL ATTENTION (VRAM SAFE)
# =========================================================
class ChannelAttention(nn.Module):
    """
    Attention légère sur les canaux, VRAM-friendly.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        y = self.avg_pool(x).view(B, C)       # (B, C)
        y = self.fc(y).view(B, C, 1, 1)       # (B, C, 1, 1)
        return x * y


# =========================================================
# STYLE INJECTOR AVEC ATTENTION CANAUX
# =========================================================
class StyleInjector(nn.Module):
    """
    Injecteur de style latent avec option attention VRAM-friendly.
    """

    def __init__(
        self,
        latent_channels=4,
        hidden=64,
        num_blocks=4,
        prompt_dim=768,
        use_attention=True,   # <-- option pour activer/désactiver
    ):
        super().__init__()

        self.config = {
            "latent_channels": latent_channels,
            "hidden": hidden,
            "num_blocks": num_blocks,
            "prompt_dim": prompt_dim,
            "use_attention": use_attention
        }

        self.latent_channels = latent_channels
        self.hidden = hidden
        self.prompt_dim = prompt_dim
        self.num_blocks = num_blocks
        self.use_attention = use_attention

        # =====================================================
        # Prompt projection
        # =====================================================
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, hidden),
            nn.SiLU()
        )

        # =====================================================
        # Input projection
        # =====================================================
        self.input_proj = nn.Sequential(
            nn.Conv2d(
                latent_channels + hidden,
                hidden,
                kernel_size=3,
                padding=1
            ),
            nn.GroupNorm(8, hidden),
            nn.SiLU()
        )

        # =====================================================
        # Attention canaux optionnelle
        # =====================================================
        if self.use_attention:
            self.attn = ChannelAttention(hidden)

        # =====================================================
        # Residual backbone
        # =====================================================
        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden) for _ in range(num_blocks)]
        )

        # =====================================================
        # Output projection
        # =====================================================
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(
                hidden,
                latent_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, latents, style_prompt_embedding):
        """
        Args:
            latents: (B, C, H, W)
            style_prompt_embedding: (B, prompt_dim)
        Returns:
            latents stylisés
        """
        B, C, H, W = latents.shape

        # -----------------------------------------------------
        # Prompt -> feature map
        # -----------------------------------------------------
        style_feat = self.prompt_proj(style_prompt_embedding)
        style_feat = style_feat.view(B, self.hidden, 1, 1)
        style_feat = style_feat.expand(-1, -1, H, W)

        # -----------------------------------------------------
        # Concat latent + style
        # -----------------------------------------------------
        x = torch.cat([latents, style_feat], dim=1)

        # -----------------------------------------------------
        # Input projection
        # -----------------------------------------------------
        x = self.input_proj(x)

        # -----------------------------------------------------
        # Attention canaux optionnelle
        # -----------------------------------------------------
        if self.use_attention:
            x = self.attn(x)

        # -----------------------------------------------------
        # Backbone résiduel
        # -----------------------------------------------------
        x = self.resblocks(x)

        # -----------------------------------------------------
        # Output projection + injection résiduelle
        # -----------------------------------------------------
        delta = self.output_proj(x)
        out_latents = latents + delta

        return out_latents
# =========================================================
# STYLE INJECTOR
# =========================================================

class StyleInjector_v1(nn.Module):
    """
    Injecteur de style latent.
    """

    def __init__(
        self,
        latent_channels=4,
        hidden=64,
        num_blocks=4,
        prompt_dim=768
    ):
        super().__init__()

        self.config = {
            "latent_channels": latent_channels,
            "hidden": hidden,
            "num_blocks": num_blocks,
            "prompt_dim": prompt_dim
        }

        self.latent_channels = latent_channels
        self.hidden = hidden
        self.prompt_dim = prompt_dim
        self.num_blocks = num_blocks

        # =====================================================
        # Prompt projection
        # =====================================================

        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, hidden),
            nn.SiLU()
        )

        # =====================================================
        # Input projection
        # FIX IMPORTANT :
        # plus de création dynamique dans forward()
        # =====================================================

        self.input_proj = nn.Sequential(
            nn.Conv2d(
                latent_channels + hidden,
                hidden,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(8, hidden),
            nn.SiLU()
        )

        # =====================================================
        # Residual backbone
        # =====================================================

        self.resblocks = nn.Sequential(
            *[
                ResidualBlock(hidden)
                for _ in range(num_blocks)
            ]
        )

        # =====================================================
        # Output projection
        # =====================================================

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),

            nn.Conv2d(
                hidden,
                latent_channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(
        self,
        latents,
        style_prompt_embedding
    ):
        """
        Args:
            latents: (B, C, H, W)
            style_prompt_embedding: (B, prompt_dim)

        Returns:
            Tensor stylisé
        """

        B, C, H, W = latents.shape

        # =====================================================
        # Prompt -> feature map
        # =====================================================

        style_feat = self.prompt_proj(style_prompt_embedding)

        style_feat = style_feat.view(
            B,
            self.hidden,
            1,
            1
        )

        style_feat = style_feat.expand(
            -1,
            -1,
            H,
            W
        )

        # =====================================================
        # Concat latent + style
        # =====================================================

        x = torch.cat(
            [latents, style_feat],
            dim=1
        )

        # =====================================================
        # Backbone
        # =====================================================

        x = self.input_proj(x)

        x = self.resblocks(x)

        delta = self.output_proj(x)

        # =====================================================
        # Residual injection
        # =====================================================

        out_latents = latents + delta

        return out_latents

# =========================================================
# TRAIN STEP
# =========================================================

def style_train_step(
    model,
    optimizer,
    criterion,
    latents,
    style_prompt_embedding,
    target_latents,
    device=device,
    grad_clip=1.0,
    debug=False
):
    """
    Une étape d'entraînement.
    """

    model.train()

    latents = sanitize_latents(latents).to(device)

    target_latents = sanitize_latents(
        target_latents
    ).to(device)

    style_prompt_embedding = (
        style_prompt_embedding.to(device)
    )

    optimizer.zero_grad()

    pred_latents = model(
        latents,
        style_prompt_embedding
    )

    loss = criterion(
        pred_latents,
        target_latents
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=grad_clip
    )

    optimizer.step()

    if debug:
        print(f"[DEBUG] Loss={loss.item():.6f}")

    return pred_latents.detach(), loss.item()

# =========================================================
# TRAIN LOOP
# =========================================================

def train_style_model(
    model,
    optimizer,
    criterion,
    style_dataset,
    epochs=10,
    device=device,
    save_every=1
):
    """
    Boucle d'entraînement principale.

    Dataset format:
    {
        "latents": Tensor,
        "prompt": Tensor,
        "target": Tensor
    }
    """

    for epoch in range(epochs):

        epoch_loss = 0.0

        for step, batch in enumerate(style_dataset):

            latents = batch["latents"]

            prompt_embedding = batch["prompt"]

            target_latents = batch["target"]

            _, loss = style_train_step(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                latents=latents,
                style_prompt_embedding=prompt_embedding,
                target_latents=target_latents,
                device=device
            )

            epoch_loss += loss

            print(
                f"[Epoch {epoch+1}] "
                f"[Step {step+1}] "
                f"Loss={loss:.6f}"
            )

        avg_loss = epoch_loss / len(style_dataset)

        print(
            f"\n[Epoch {epoch+1}] "
            f"Average Loss={avg_loss:.6f}\n"
        )

        if (epoch + 1) % save_every == 0:

            save_style_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss
            )

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    B, C, H, W = 2, 4, 64, 64

    latents = torch.randn(B, C, H, W)

    prompt_embedding = torch.randn(B, 768)

    model = StyleInjector(
        latent_channels=C,
        hidden=64,
        num_blocks=4,
        prompt_dim=768
    ).to(device)

    model.apply(weights_init)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    criterion = StyleLoss()

    out = model(
        latents.to(device),
        prompt_embedding.to(device)
    )

    print("Input shape :", latents.shape)
    print("Output shape:", out.shape)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

