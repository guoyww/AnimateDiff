# scripts/modules/motion_module_cam3.py
import torch
import torch.nn as nn

class MotionModuleCam(nn.Module):  # ⚠ même nom que l’ancien module
    def __init__(self, strength: float = 0.15,
                 prompt_injection_alpha: float = 0.2,
                 prompt_injection_every_n_frames: int = 5):
        super().__init__()
        self.strength = strength
        self.prompt_injection_alpha = prompt_injection_alpha
        self.prompt_injection_every_n_frames = prompt_injection_every_n_frames

    def forward(self, latents, prompt_latents=None):
        if latents.dim() != 5:
            return latents

        B, C, F, H, W = latents.shape

        t = torch.linspace(0, 2*torch.pi, F, device=latents.device).view(1, 1, F, 1, 1)
        motion = torch.sin(t) * self.strength
        latents = latents + motion

        if prompt_latents is not None:
            for f in range(F):
                if f % self.prompt_injection_every_n_frames == 0:
                    latents[:, :, f, :, :] = (
                        latents[:, :, f, :, :] * (1 - self.prompt_injection_alpha) +
                        prompt_latents[:, :, f, :, :] * self.prompt_injection_alpha
                    )

        latents = latents + 0.02 * torch.randn_like(latents)
        return latents
