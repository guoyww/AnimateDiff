# ------------------------------------------------------------------
# vae_config.py - utilitaires pour VAE / détection type et infos
# ------------------------------------------------------------------
import torch
from diffusers import AutoencoderKL

def load_vae(vae_path, device="cpu", dtype=torch.float16):
    """
    Charge un VAE depuis un fichier .safetensors ou .ckpt et active slicing/tiling.
    Retourne le VAE et ses informations de compatibilité.
    """
    print(f"📦 Chargement VAE : {vae_path}")

    vae = AutoencoderKL.from_single_file(
        vae_path,
        torch_dtype=dtype
    ).to(device)

    vae.enable_slicing()
    vae.enable_tiling()

    # Détection du type
    vae_type, latent_channels, scaling_factor = detect_vae_type(vae)
    print("🧠 Détection VAE")
    print(f"   type : {vae_type}")
    print(f"   latent_channels : {latent_channels}")
    print(f"   scaling_factor : {scaling_factor}")

    return vae, vae_type, latent_channels, scaling_factor


def detect_vae_type(vae):
    """
    Détecte le type de VAE chargé en se basant sur scaling_factor et latent_channels.
    Retourne : type VAE (str), latent_channels (int), scaling_factor (float)
    """
    latent_channels = getattr(vae.config, "latent_channels", None)
    scaling_factor = getattr(vae.config, "scaling_factor", None)

    # fallback classique pour SD1/SD2
    if scaling_factor is None:
        scaling_factor = 0.18215

    if abs(scaling_factor - 0.18215) < 1e-4:
        vae_type = "SD1 / SD2 compatible"
    elif abs(scaling_factor - 0.13025) < 1e-4:
        vae_type = "SDXL compatible"
    else:
        vae_type = "VAE custom"

    return vae_type, latent_channels, scaling_factor


def vae_summary(vae):
    """
    Affiche un résumé complet du VAE pour debug.
    """
    vae_type, latent_channels, scaling_factor = detect_vae_type(vae)
    print("─────────────────────────────")
    print("📌 VAE SUMMARY")
    print(f" type           : {vae_type}")
    print(f" latent_channels: {latent_channels}")
    print(f" scaling_factor : {scaling_factor}")
    print("─────────────────────────────")
