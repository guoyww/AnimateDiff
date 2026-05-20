import torch

def ensure_valid(latents, eps=1e-3):
    """
    Remplace NaN/inf et ajoute un petit bruit si le latent est trop faible.
    """
    latents = torch.nan_to_num(latents, nan=eps, posinf=eps, neginf=-eps)
    if latents.abs().mean() < eps:
        latents += torch.randn_like(latents) * eps
    return latents.clamp(-3.0, 3.0)



