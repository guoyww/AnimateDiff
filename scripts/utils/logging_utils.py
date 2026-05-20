# utils/logging_utils.py
import os
import csv
import torch


def debug_latents(tag, latents):
    print(
        f"[DEBUG] {tag} "
        f"min={latents.min().item():.4f}, "
        f"max={latents.max().item():.4f}, "
        f"NaN={torch.isnan(latents).any().item()}"
    )

def log_latent_stats(frame_idx, latents, csv_path="latent_stats.csv"):
    """
    Écrit les stats latentes dans un CSV.
    Args:
        frame_idx (int): numéro de la frame
        latents (torch.Tensor): tensor latent
        csv_path (str or Path): chemin vers le CSV
    """
    min_val = float(latents.min())
    max_val = float(latents.max())
    mean_val = float(latents.mean())
    std_val = float(latents.std())

    # Écrire l'en-tête si le fichier n'existe pas
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["frame", "min", "max", "mean", "std"])
        writer.writerow([frame_idx, min_val, max_val, mean_val, std_val])


def log_patch_stats(frame_idx, patch_idx, patch, csv_path="patch_stats.csv"):
    """
    Écrit les stats de chaque patch VAE dans un CSV.
    Args:
        frame_idx (int): numéro de la frame
        patch_idx (str): identifiant du patch (ex: "0_0")
        patch (torch.Tensor): patch latent ou décodé
        csv_path (str or Path): chemin vers le CSV
    """
    min_val = float(patch.min())
    max_val = float(patch.max())
    mean_val = float(patch.mean())
    std_val = float(patch.std())

    shape_str = "x".join(map(str, patch.shape))
    dtype_str = str(patch.dtype)
    device_str = str(patch.device)
    any_nan = int(torch.isnan(patch).any())
    any_inf = int(torch.isinf(patch).any())

    # Mémoire GPU (si sur CUDA)
    if patch.is_cuda:
        mem_alloc = torch.cuda.memory_allocated()
        mem_reserved = torch.cuda.memory_reserved()
    else:
        mem_alloc = 0
        mem_reserved = 0

    # Écrire l'en-tête si le fichier n'existe pas
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "frame", "patch", "shape", "dtype", "device",
                "min", "max", "mean", "std", "NaN", "Inf",
                "gpu_alloc_bytes", "gpu_reserved_bytes"
            ])
        writer.writerow([
            frame_idx, patch_idx, shape_str, dtype_str, device_str,
            min_val, max_val, mean_val, std_val,
            any_nan, any_inf, mem_alloc, mem_reserved
        ])
