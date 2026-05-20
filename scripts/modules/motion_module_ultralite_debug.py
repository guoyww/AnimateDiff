# motion_module_ultralite_debug.py
import torch
import torch.nn as nn

class MotionModuleUltraLiteDebug(nn.Module):
    def __init__(self, strength: float = 0.01):
        super().__init__()
        self.strength = strength

    def forward(self, latents):
        # Forcer 5D : [B, C, F, H, W]
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)  # ajoute F=1

        B, C, F, H, W = latents.shape
        device = latents.device

        # Très léger mouvement
        motion = torch.linspace(0, 1, F, device=device).view(1, 1, F, 1, 1)
        latents = latents + motion * self.strength

        # Debug frame par frame
        for f in range(F):
            frame_latents = latents[:, :, f, :, :]
            print(f"[DEBUG TRACE] Frame {f}: latents min={frame_latents.min().item():.6f}, max={frame_latents.max().item():.6f}")

        # Repasser à 4D si nécessaire (optionnel, selon pipeline)
        if F == 1:
            latents = latents.squeeze(2)

        return latents

class MotionModule(MotionModuleUltraLiteDebug):
    pass
