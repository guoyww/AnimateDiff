import torch
import torch.nn as nn

class MotionModuleTiny(nn.Module):
    def __init__(self, strength: float = 0.05):
        super().__init__()
        self.strength = strength

    def forward(self, latents):
        """
        latents: [B, C, F, H, W]
        Applique une légère dérive temporelle pour simuler du mouvement.
        """
        if latents.dim() != 5:
            return latents

        B, C, F, H, W = latents.shape

        # Création d'un léger décalage temporel progressif
        motion = torch.linspace(0, 1, F, device=latents.device).view(1, 1, F, 1, 1)

        # Applique une petite variation
        latents = latents + motion * self.strength

        return latents


class MotionModule(nn.Module):
    def __init__(self, strength: float = 0.05):
        super().__init__()
        self.strength = strength

    def forward(self, latents):
        """
        latents: [B, C, F, H, W]
        Applique une légère dérive temporelle pour simuler du mouvement.
        """
        if latents.dim() != 5:
            return latents

        B, C, F, H, W = latents.shape

        # Création d'un léger décalage temporel progressif
        motion = torch.linspace(0, 1, F, device=latents.device).view(1, 1, F, 1, 1)

        # Applique une petite variation
        latents = latents + motion * self.strength

        return latents
