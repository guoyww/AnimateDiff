import torch
import torch.nn as nn
import math
from torchvision.transforms import functional as TF

class MotionModuleMasked(nn.Module):
    def __init__(
        self,
        strength_person: float = 0.03,      # mouvement personnage (cheveux / corps)
        strength_bg: float = 0.008,         # mouvement décor
        hair_bias: float = 0.7,
        wave_speed: float = 1.5,
        wave_amplitude: float = 1.0,
        decor_wave_speed: float = 0.3,
        decor_wave_amplitude: float = 0.02
    ):
        super().__init__()
        self.strength_person = strength_person
        self.strength_bg = strength_bg
        self.hair_bias = hair_bias
        self.wave_speed = wave_speed
        self.wave_amplitude = wave_amplitude
        self.decor_wave_speed = decor_wave_speed
        self.decor_wave_amplitude = decor_wave_amplitude

    def create_person_mask(self, input_image_latent):
        """
        Crée un masque grossier du personnage à partir du latent.
        Ici simple méthode basée sur la luminosité pour séparer sujet / fond.
        """
        # input_image_latent: [B, C, H, W] en float
        gray = input_image_latent.mean(dim=1, keepdim=True)  # [B,1,H,W]
        mask = (gray > gray.mean(dim=[2,3], keepdim=True) * 0.9).float()  # sujet plus clair que fond moyen
        return mask  # 1 = personnage, 0 = fond

    def forward(self, latents, input_image_latent=None):
        """
        latents: [B, C, F, H, W]
        input_image_latent: [B, C, H, W] image d'entrée pour créer le masque personnage
        """
        if latents.dim() != 5:
            return latents

        B, C, F, H, W = latents.shape
        device = latents.device

        if input_image_latent is not None:
            person_mask = self.create_person_mask(input_image_latent).to(device)  # [B,1,H,W]
            person_mask = person_mask.unsqueeze(2).expand(-1, -1, F, -1, -1)        # [B,1,F,H,W]
            bg_mask = 1.0 - person_mask
            bg_mask = bg_mask.repeat(1,C,1,1,1)                                      # canaux
            person_mask = person_mask.repeat(1,C,1,1,1)
        else:
            # Pas de mask fourni -> mouvement uniforme
            person_mask = torch.ones_like(latents)
            bg_mask = torch.ones_like(latents)

        # -------------------------
        # Mouvement personnage
        # -------------------------
        y = torch.linspace(1.0, 0.0, H, device=device).view(1,1,1,H,1)
        hair_mask = y ** self.hair_bias
        t = torch.linspace(0, 2*math.pi, F, device=device)
        wave_person = torch.sin(t * self.wave_speed).view(1,1,F,1,1) * self.wave_amplitude

        noise = torch.randn(B, C, F, H, W, device=device) * 0.1
        person_motion = noise * wave_person * hair_mask * self.strength_person * person_mask

        # -------------------------
        # Mouvement décor
        # -------------------------
        decor_wave = torch.sin(t * self.decor_wave_speed).view(1,1,F,1,1) * self.decor_wave_amplitude
        decor_motion = noise * decor_wave * self.strength_bg * bg_mask

        return latents + person_motion + decor_motion

# Alias pour compatibilité
MotionModule = MotionModuleMasked
