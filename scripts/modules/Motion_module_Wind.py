import torch
import torch.nn as nn
import math

class MotionModuleEnhanced(nn.Module):
    """
    Motion module simple et cohérent, prêt pour animation VRAM-light (~2Go)
    - Propagation depuis la frame précédente
    - Oscillation sinus + bruit pour mouvement naturel
    """
    def __init__(
        self,
        strength: float = 0.03,         # mouvement global du personnage/décor
        wave_speed: float = 1.5,
        wave_amplitude: float = 0.7,
        decor_strength: float = 0.05,
        decor_wave_speed: float = 0.8,
        decor_wave_amplitude: float = 0.3,
        camera_shift: float = 0.05
    ):
        super().__init__()
        self.strength = strength
        self.wave_speed = wave_speed
        self.wave_amplitude = wave_amplitude
        self.decor_strength = decor_strength
        self.decor_wave_speed = decor_wave_speed
        self.decor_wave_amplitude = decor_wave_amplitude
        self.camera_shift = camera_shift

    def forward(self, latents, previous_latent=None):
        """
        latents: [B, C, F, H, W]  (F=1 si une frame)
        previous_latent: [B, C, F, H, W] ou None
        """
        if latents.dim() != 5:
            return latents

        B, C, F, H, W = latents.shape
        device = latents.device

        # Temps pour oscillation
        t = torch.linspace(0, 2 * math.pi, F, device=device).view(1, 1, F, 1, 1)

        # Oscillation personnage / décor
        wave_person = torch.sin(t * self.wave_speed) * self.wave_amplitude
        wave_decor = torch.sin(t * self.decor_wave_speed) * self.decor_wave_amplitude

        # Bruit léger
        noise = torch.randn_like(latents) * 0.05

        motion_person = noise * wave_person * self.strength
        motion_decor = noise * wave_decor * self.decor_strength

        # Déplacement global (caméra)
        shift_x = torch.sin(t * 0.3) * self.camera_shift
        shift_y = torch.cos(t * 0.3) * self.camera_shift
        motion_camera = torch.zeros_like(latents)
        motion_camera = motion_camera.roll(int(H * shift_y.mean().item()), dims=3)
        motion_camera = motion_camera.roll(int(W * shift_x.mean().item()), dims=4)

        # Propagation depuis frame précédente pour continuité
        if previous_latent is not None:
            latents = previous_latent + (latents - previous_latent) * self.strength

        # Combinaison finale
        latents = latents + motion_person + motion_decor + motion_camera
        latents = torch.clamp(latents, -1.0, 1.0)

        return latents

# Alias pour compatibilité avec ton script existant
MotionModule = MotionModuleEnhanced
