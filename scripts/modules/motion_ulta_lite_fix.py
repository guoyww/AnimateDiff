# scripts/modules/motion_ulta_lite_fix.py
import torch
import torch.nn as nn

class MotionModuleUltraLiteFixComplete(nn.Module):
    """
    Module motion ultra-léger pour corriger les frames mortes et ajouter un petit bruit créatif.
    """
    def __init__(self, creative_noise=0.07, verbose=True):
        super().__init__()
        self.creative_noise = creative_noise
        self.verbose = verbose

    def forward(self, latents):
        """
        Applique le motion module aux latents.
        latents: torch.Tensor [B, C, H, W] ou [B, C, F, H, W]
        """
        original_dim = latents.dim()
        if original_dim == 4:
            latents = latents.unsqueeze(2)  # [B, C, 1, H, W]

        B, C, F, H, W = latents.shape

        if self.verbose:
            print(f"[SAFE DEBUG] Latents avant motion: shape={latents.shape}, "
                  f"min={latents.min():.6f}, max={latents.max():.6f}, "
                  f"mean={latents.mean():.6f}, std={latents.std():.6f}")

        # Correction des frames nulles
        for f in range(F):
            frame_latents = latents[:, :, f, :, :]
            if (frame_latents.abs() < 1e-6).all():
                if f == 0:
                    # Première frame → petit bruit
                    latents[:, :, f, :, :] = frame_latents + torch.randn_like(frame_latents) * (self.creative_noise * 0.1)
                else:
                    # Interpolation depuis la frame précédente + petit bruit
                    latents[:, :, f, :, :] = latents[:, :, f-1, :, :] + torch.randn_like(frame_latents) * (self.creative_noise * 0.05)
                if self.verbose:
                    print(f"[SAFE DEBUG] Frame {f} morte remplacée par bruit (première frame)" if f==0 else f"[SAFE DEBUG] Frame {f} corrigée depuis frame précédente")

        # Ajouter un petit bruit créatif sur toutes les frames
        latents = latents + torch.randn_like(latents) * (self.creative_noise * 0.5)

        if self.verbose:
            print(f"[SAFE DEBUG] Latents après motion: min={latents.min():.6f}, "
                  f"max={latents.max():.6f}, mean={latents.mean():.6f}, "
                  f"std={latents.std():.6f}")

        # Stats par frame
        for f in range(F):
            frame_latents = latents[:, :, f, :, :]
            print(f"[SAFE TRACE] Frame {f}: min={frame_latents.min():.6f}, "
                  f"max={frame_latents.max():.6f}, mean={frame_latents.mean():.6f}, "
                  f"std={frame_latents.std():.6f}")

        if original_dim == 4:
            latents = latents.squeeze(2)

        return latents
