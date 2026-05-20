import torch
import torch.nn as nn
import math

class MotionModuleEnhanced(nn.Module):
    def __init__(
        self,
        strength: float = 0.01,          # mouvement personnage
        hair_bias: float = 0.7,           # accentuation cheveux
        wave_speed: float = 1.5,
        wave_amplitude: float = 1.0,
        decor_strength: float = 0.1,     # décor subtil 0.03
        decor_wave_speed: float = 0.7,      # wave 0.5
        decor_wave_amplitude: float = 0.2,  # 0.2
        camera_shift: float = 0.7      # effet caméra léger 0.02
    ):
        super().__init__()
        self.strength = strength
        self.hair_bias = hair_bias
        self.wave_speed = wave_speed
        self.wave_amplitude = wave_amplitude
        self.decor_strength = decor_strength
        self.decor_wave_speed = decor_wave_speed
        self.decor_wave_amplitude = decor_wave_amplitude
        self.camera_shift = camera_shift

    def forward(self, latents):
        if latents.dim() != 5:
            return latents

        B, C, F, H, W = latents.shape
        device = latents.device

        # -------------------------
        # 1️⃣ Masque vertical (plus fort en haut)
        # -------------------------
        y = torch.linspace(1.0, 0.0, H, device=device).view(1, 1, 1, H, 1)
        vertical_mask = y ** self.hair_bias

        # -------------------------
        # 2️⃣ Oscillation temporelle pour personnage
        # -------------------------
        t = torch.linspace(0, 2 * math.pi, F, device=device).view(1, 1, F, 1, 1)
        wave_person = torch.sin(t * self.wave_speed) * self.wave_amplitude

        # -------------------------
        # 3️⃣ Oscillation temporelle pour décor subtil
        # -------------------------
        wave_decor = torch.sin(t * self.decor_wave_speed) * self.decor_wave_amplitude

        # -------------------------
        # 4️⃣ Bruit léger
        # -------------------------
        noise = torch.randn_like(latents) * 0.3

        # -------------------------
        # 5️⃣ Combinaison des mouvements
        # -------------------------
        motion_person = noise * wave_person * vertical_mask * self.strength
        motion_decor = noise * wave_decor * self.decor_strength

        # -------------------------
        # 6️⃣ Effet caméra simulé (décalage global très léger)
        # -------------------------
        shift_x = torch.sin(t * 0.3) * self.camera_shift
        shift_y = torch.cos(t * 0.3) * self.camera_shift
        motion_camera = torch.zeros_like(latents)
        motion_camera = motion_camera.roll(int(H * shift_y.mean().item()), dims=3)
        motion_camera = motion_camera.roll(int(W * shift_x.mean().item()), dims=4)

        # -------------------------
        # 7️⃣ Retour latents combinés
        # -------------------------
        return latents + motion_person + motion_decor + motion_camera

# Alias pour compatibilité
MotionModule = MotionModuleEnhanced
