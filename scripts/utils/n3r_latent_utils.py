import time
import uuid
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

#mini node custom ComfyUI :
def load_latent(path, map_location="cpu", debug=True):
    data = torch.load(path, map_location=map_location)

    samples = data["samples"]
    metadata = data.get("metadata", {})

    if debug:
        print("\n📦 Loaded latent file")
        print(f"📁 path: {path}")

        print("\n🧠 Samples info")
        print(f"shape   : {tuple(samples.shape)}")
        print(f"dtype   : {samples.dtype}")
        print(f"device  : {samples.device}")
        print(f"mean    : {samples.mean().item():.6f}")
        print(f"std     : {samples.std().item():.6f}")
        print(f"min/max : {samples.min().item():.6f} / {samples.max().item():.6f}")

        print("\n📝 Metadata")
        if metadata:
            for k, v in metadata.items():
                print(f"{k}: {v}")
        else:
            print("No metadata found")

        print()

    return {
        "samples": samples,
        "metadata": metadata,
    }


def export_latents_file(latents, suffix, frame_counter, output_dir="exports"):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"latents_{frame_counter:05d}_{suffix}.pt"

    torch.save(
        {
            "latents": latents.detach().float().cpu(),
            "frame": frame_counter,
            "shape": list(latents.shape),
        },
        save_path
    )

    print(f"💾 Latents exported: {save_path}")



"""

{
    "samples": latents,

    # dimensions image réelles
    "width": 896,
    "height": 1280,

    # infos latent
    "latent_shape": [1,4,160,112],
    "latent_scale_factor": 8,

    # modèle
    "model": "SDXL",
    "vae": "sdxl_vae",
    "dtype": "float32",

    # génération
    "seed": seed,
    "steps": steps,
    "cfg": cfg,
    "sampler": sampler_name,
    "scheduler": scheduler_name,

    # animation
    "frame": frame_counter,
    "fps": 24,

    # ton moteur
    "engine": "AnimateHybrid",
    "motion_noise": float(motion_noise),
    "temporal_strength": float(temporal_strength),

    # debug
    "timestamp": time.time(),
}
"""



def generate_sequence_id(prefix="ah"):
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:8]

    return f"{prefix}_{date_str}_{short_uuid}"

def build_latent_metadata(
    latents,
    frame_counter=0,
    sequence_id="AnimateHybrid",
    scale=8,
    fps=24,

    # pipeline
    denoise=True,
    temporal_consistency=True,
    style_injection=True,
    appearance=True,
    creative=True,

    # render
    sharpen_mode="both",
    gamma_boost=1.0,

    # training
    train=False,

    # temporal
    ema_prev_latents=None,

    # misc
    engine="AnimateHybrid",
):
    B, C, H, W = latents.shape

    return {

        # ------------------------------------------------
        # latent info
        # ------------------------------------------------
        "latent_shape": [B, C, H, W],
        "channels": C,

        # image size
        "width": W * scale,
        "height": H * scale,
        "latent_scale_factor": scale,

        "latent_mean": float(latents.mean().item()),
        "latent_std": float(latents.std().item()),
        "is_video": True,


        "format_version": 1,

        # frame
        "frame_index": frame_counter,
        "sequence_id": sequence_id,
        "fps": fps,

        # dtype/device
        "dtype": str(latents.dtype),
        "device": str(latents.device),

        # ------------------------------------------------
        # pipeline modules
        # ------------------------------------------------
        "denoise": denoise,
        "temporal_consistency": temporal_consistency,
        "style_injection": style_injection,
        "appearance": appearance,
        "creative": creative,

        # ------------------------------------------------
        # render settings
        # ------------------------------------------------
        "sharpen_mode": sharpen_mode,
        "gamma_boost": gamma_boost,

        # ------------------------------------------------
        # temporal
        # ------------------------------------------------
        "has_ema": ema_prev_latents is not None,

        # ------------------------------------------------
        # train/debug
        # ------------------------------------------------
        "train": train,

        # ------------------------------------------------
        # engine
        # ------------------------------------------------
        "engine": engine,

        "vae_scaling_factor": 0.18215,
        "base_model": "SD1.5",

        # ------------------------------------------------
        # timestamp
        # ------------------------------------------------
        "timestamp": time.time(),
    }

def save_latents_for_comfy(latents, suffix, frame_counter, output_dir="exports"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "samples": latents.detach().float().cpu(),
    }

    path = output_dir / f"latent_{frame_counter:05d}_{suffix}.latent.pt"

    torch.save(data, path)

    print(f"💾 Comfy Latent exported: {path}")

def export_latents_for_comfy_meta(
    latents, suffix,
    frame_counter,
    output_dir="exports",
    metadata=None,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "samples": latents.detach().float().cpu(),
        "metadata": metadata or {},
    }

    path = output_dir / f"latent_{frame_counter:05d}_{suffix}.latent.pt"

    torch.save(data, path)

    print(f"💾 Latents exported: {path}")


# =========================================================
# Compute high freq
# =========================================================

def compute_high_freq_energy(
    latents,
    kernel_size=3,
    normalize=True,
    per_channel=False,
    return_map=False
):

    latents = latents.float()

    # ----------------------------------------------------
    # blur
    # ----------------------------------------------------

    blur = F.avg_pool2d(
        latents,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2
    )

    # ----------------------------------------------------
    # high frequencies
    # ----------------------------------------------------

    high_freq = latents - blur

    # ----------------------------------------------------
    # spatial energy map
    # ----------------------------------------------------

    hf_map = torch.sqrt(
        high_freq.pow(2) + 1e-8
    )

    # ----------------------------------------------------
    # pooled energy
    # ----------------------------------------------------

    hf_energy = hf_map.mean(dim=(2, 3))

    # ----------------------------------------------------
    # normalization
    # ----------------------------------------------------

    if normalize:

        base = torch.sqrt(
            latents.pow(2).mean(dim=(2,3)) + 1e-8
        )

        hf_energy = hf_energy / (base + 1e-6)

    # ----------------------------------------------------
    # outputs
    # ----------------------------------------------------

    if return_map:
        return hf_map

    if per_channel:
        return hf_energy

    return hf_energy.mean(dim=1)
