import torch
import torch.nn as nn
import math


class MotionModuleTiny(nn.Module):
    def __init__(
        self,
        strength: float = 0.03,
        hair_bias: float = 0.7,
        wave_speed: float = 1.5,
        wave_amplitude: float = 1.0,
    ):
        super().__init__()
        self.strength = strength
        self.hair_bias = hair_bias
        self.wave_speed = wave_speed
        self.wave_amplitude = wave_amplitude

    def forward(self, latents):
        """
        latents: [B, C, F, H, W]
        Simule un léger mouvement de cheveux naturel.
        """

        if latents.dim() != 5:
            return latents

        B, C, F, H, W = latents.shape
        device = latents.device

        # -------------------------
        # 1️⃣ Masque vertical (plus fort en haut)
        # -------------------------
        y = torch.linspace(1.0, 0.0, H, device=device).view(1, 1, 1, H, 1)
        vertical_mask = y ** self.hair_bias  # accentuation haut image

        # -------------------------
        # 2️⃣ Oscillation temporelle sinusoïdale
        # -------------------------
        t = torch.linspace(0, 2 * math.pi, F, device=device)
        wave = torch.sin(t * self.wave_speed) * self.wave_amplitude
        wave = wave.view(1, 1, F, 1, 1)

        # -------------------------
        # 3️⃣ Bruit cohérent léger
        # -------------------------
        noise = torch.randn_like(latents) * 0.5

        # -------------------------
        # 4️⃣ Combinaison
        # -------------------------
        motion = noise * wave * vertical_mask * self.strength

        return latents + motion


# Version générique (alias)
class MotionModule(MotionModuleTiny):
    pass
