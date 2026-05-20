#n3rControlNet.py ------------------------------------------------
#**** Ensemble des outils pour n3rControlNet
#------------------------------------------------------------------

import torch, numpy as np, cv2, gc

def create_canny_control(image_pil, low=100, high=200, device='cuda', dtype=torch.float16):
    """
    Génère un tenseur de contrôle à partir d'une image PIL via Canny.
    Logs détaillés pour debug.
    """
    print("[Canny] Conversion PIL -> grayscale")
    img = np.array(image_pil.convert("L"), dtype=np.float32) / 255.0
    print(f"[Canny] Image shape: {img.shape}, min: {img.min():.6f}, max: {img.max():.6f}")

    # Canny edges
    edges = cv2.Canny((img * 255).astype(np.uint8), low, high).astype(np.float32) / 255.0
    print(f"[Canny] Edges computed, min: {edges.min():.6f}, max: {edges.max():.6f}")

    # Convertir en tensor directement sur device
    edges_tensor = torch.tensor(edges, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    print(f"[Canny] Tensor shape: {edges_tensor.shape}, dtype: {edges_tensor.dtype}, device: {edges_tensor.device}")

    # Clamp pour éviter 0 ou 1 exacts
    edges_tensor = edges_tensor.clamp(1e-5, 1-1e-5)

    return edges_tensor


def control_to_latent(control_tensor, vae, device='cuda', LATENT_SCALE=1.0):
    # Si 1 canal, dupliquer pour obtenir 3 canaux
    if control_tensor.shape[1] == 1:
        control_tensor = control_tensor.repeat(1, 3, 1, 1)

    # Assurer le type float16 pour économiser la VRAM
    control_tensor = control_tensor.to(device=device, dtype=vae.dtype)  # <- correction ici
    print(f"[Control->Latent] Converted tensor dtype: {control_tensor.dtype}, device: {control_tensor.device}")

    # Encode VAE
    with torch.no_grad():  # économise un peu de VRAM
        latent = vae.encode(control_tensor).latent_dist.sample()

    print(f"[Control->Latent] Latent shape: {latent.shape}, min: {latent.min()}, max: {latent.max()}")

    return latent * LATENT_SCALE


import torch
import torch.nn.functional as F

def match_latent_size(latents, control_latent, control_weight_map):
    """
    Redimensionne control_latent et control_weight_map pour correspondre
    à la taille de latents, et ajuste dtype/device.

    Args:
        latents (torch.Tensor): tensor cible [B, C, H, W]
        control_latent (torch.Tensor): tensor ControlNet [B, Cc, Hc, Wc]
        control_weight_map (torch.Tensor): tensor poids [B, 1, Hc, Wc]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tensors redimensionnés et alignés
    """
    target_size = latents.shape[-2:]

    # Redimensionner si nécessaire
    if control_latent.shape[-2:] != target_size:
        control_latent = F.interpolate(
            control_latent, size=target_size,
            mode='bilinear', align_corners=False
        )

    if control_weight_map.shape[-2:] != target_size:
        control_weight_map = F.interpolate(
            control_weight_map, size=target_size,
            mode='bilinear', align_corners=False
        )

    # Assurer dtype et device identiques à latents
    control_latent = control_latent.to(dtype=latents.dtype, device=latents.device)
    control_weight_map = control_weight_map.to(dtype=latents.dtype, device=latents.device)

    return control_latent, control_weight_map
