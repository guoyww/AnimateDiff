import torch
import torch.nn as nn

class MotionModuleCam(nn.Module):
    def __init__(self, strength: float = 0.05,
                 prompt_injection_alpha: float = 0.3,
                 prompt_injection_every_n_frames: int = 5):
        super().__init__()
        self.strength = strength
        self.prompt_injection_alpha = prompt_injection_alpha
        self.prompt_injection_every_n_frames = prompt_injection_every_n_frames

    def forward(self, latents, prompt_latents=None):
        """
        latents: [B, C, F, H, W] - sortie N3R
        prompt_latents: [B, C, F, H, W] - latents générés depuis le prompt seul
        """
        if latents.dim() != 5:
            return latents

        B, C, F, H, W = latents.shape

        # ---------------------------
        # 1️⃣ Appliquer une dérive temporelle progressive
        # ---------------------------
        motion = torch.linspace(0, 1, F, device=latents.device).view(1, 1, F, 1, 1)
        latents = latents + motion * self.strength

        # ---------------------------
        # 2️⃣ Ignorer les clés mémoire à 0 (si applicable)
        # ---------------------------
        # On suppose que la 4ème dimension des canaux contient des clés à 0.0
        if C >= 4:
            mask = torch.ones(C, device=latents.device)
            mask[latents[0,:,0,0,0] == 0.0] = 0.0
            latents = latents * mask.view(1, -1, 1, 1, 1)

        # ---------------------------
        # 3️⃣ Injection du prompt toutes les N frames
        # ---------------------------
        if prompt_latents is not None:
            for f in range(F):
                if f % self.prompt_injection_every_n_frames == 0:
                    latents[:, :, f, :, :] = (
                        latents[:, :, f, :, :] * (1 - self.prompt_injection_alpha) +
                        prompt_latents[:, :, f, :, :] * self.prompt_injection_alpha
                    )

        return latents
