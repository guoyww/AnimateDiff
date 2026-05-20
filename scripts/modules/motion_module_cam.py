import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionModuleCam(nn.Module):
    """
    Motion module caméra pour N3R :
    - Applique une rotation progressive (face -> profil)
    - Translation et zoom
    - Compatible latents [B, C, F, H, W]
    """
    def __init__(self, rot_strength: float = 30.0, tx_strength: float = 20.0,
                 ty_strength: float = 0.0, zoom_strength: float = 0.9):
        """
        Args:
            rot_strength: rotation max en degrés
            tx_strength: translation max X en pixels
            ty_strength: translation max Y en pixels
            zoom_strength: zoom final (1.0 = pas de zoom)
        """
        super().__init__()
        self.rot_strength = rot_strength
        self.tx_strength = tx_strength
        self.ty_strength = ty_strength
        self.zoom_strength = zoom_strength

    def forward(self, latents):
        """
        latents: [B, C, F, H, W]
        """
        if latents.dim() != 5:
            return latents  # fallback

        B, C, F, H, W = latents.shape

        for f in range(F):
            # progression normalisée frame / total frames
            t = f / max(F-1, 1)

            # rotation progressive (0° -> rot_strength)
            angle = t * self.rot_strength

            # translation progressive
            tx = t * self.tx_strength
            ty = t * self.ty_strength

            # zoom progressif (1.0 -> zoom_strength)
            zoom = 1.0 - t * (1.0 - self.zoom_strength)

            # transformation de la frame
            latents[:,:,f] = self.transform_frame(latents[:,:,f], angle, tx, ty, zoom)

        return latents

    @staticmethod
    def transform_frame(frame, angle, tx, ty, zoom):
        """
        Applique rotation + translation + zoom sur une frame [B,C,H,W]
        """
        B, C, H, W = frame.shape
        device = frame.device
        dtype = frame.dtype

        angle_rad = math.radians(angle)
        cos = math.cos(angle_rad) / zoom
        sin = math.sin(angle_rad) / zoom

        # matrice affine [B,2,3]
        theta = torch.tensor([
            [cos, -sin, 2*tx/W],
            [sin,  cos, 2*ty/H]
        ], device=device, dtype=dtype).unsqueeze(0).repeat(B,1,1)

        # grid et sampling
        grid = F.affine_grid(theta, frame.size(), align_corners=False)
        transformed = F.grid_sample(frame, grid, align_corners=False, padding_mode='border')

        return transformed
