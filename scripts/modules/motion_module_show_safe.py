# motion_module_show.py
import torch
import matplotlib.pyplot as plt
from scripts.modules.motion_ulta_lite_fix import MotionModuleUltraLiteFixComplete

class MotionModuleSafePatch:
    """
    Version patchée de MotionModuleSafe pour :
    - Éviter les erreurs de dimension avec UNet (addition au lieu de concat)
    - Injecter un bruit minimal si frames trop faibles
    - Affichage debug optionnel
    - Lissage optionnel pour réduire le flou/flickering
    """
    def __init__(self, verbose=False, min_threshold=1e-3, noise_scale=1e-2, smoothing_alpha=0.3):
        self.motion_module = MotionModuleUltraLiteFixComplete(verbose=False)
        self.verbose = verbose
        self.min_threshold = min_threshold
        self.noise_scale = noise_scale
        self.smoothing_alpha = smoothing_alpha
        self.previous_latents = None

    def ensure_valid_latents(self, latents):
        """Injecte du bruit si frames trop faibles"""
        B, C, F, H, W = latents.shape
        for f in range(F):
            frame_abs_max = latents[:, :, f, :, :].abs().max()
            if frame_abs_max < self.min_threshold:
                latents[:, :, f, :, :] += torch.randn_like(latents[:, :, f, :, :]) * self.noise_scale
                if self.verbose:
                    print(f"[SAFE DEBUG] Frame {f} trop faible ({frame_abs_max:.6f}) → bruit injecté")
        return latents

    def smooth_latents(self, latents):
        """Lissage simple entre frames pour réduire le flou et flickering"""
        if self.previous_latents is None:
            self.previous_latents = latents.clone()
            return latents
        latents = (1 - self.smoothing_alpha) * self.previous_latents + self.smoothing_alpha * latents
        self.previous_latents = latents.clone()
        return latents

    def show_latents(self, latents, title="Latents"):
        """Affiche les latents de manière propre"""
        if latents.abs().max() < self.min_threshold:
            if self.verbose:
                print("[SAFE DEBUG] Frames trop faibles pour affichage")
            return
        F = latents.shape[2]
        fig, axes = plt.subplots(1, F, figsize=(3*F, 3))
        if F == 1:
            axes = [axes]
        for f in range(F):
            img = latents[0, :3, f, :, :].permute(1,2,0).clamp(-1,1)
            img = (img + 1) / 2.0
            axes[f].imshow(img.detach().cpu())
            axes[f].axis('off')
            axes[f].set_title(f"Frame {f}")
        fig.suptitle(title)
        plt.show()

    def __call__(self, latents, init_image_scale_override=None, apply_smoothing=True):
        latents = self.ensure_valid_latents(latents)

        # Override temporaire init_image_scale si demandé
        if init_image_scale_override is not None:
            original_scale = getattr(self.motion_module, "init_image_scale", 1.0)
            self.motion_module.init_image_scale = init_image_scale_override

        # Application du motion module
        latents_after = self.motion_module(latents)

        # Patch dimensionnel : addition au lieu de concat
        if latents_after.shape != latents.shape:
            if latents_after.shape[1] == 2 * latents.shape[1]:
                latents_after = latents + latents_after[:, latents.shape[1]:, :, :, :]
                if self.verbose:
                    print(f"[SAFE PATCH] Fusion add : {latents.shape} ← {latents_after.shape}")

        # Restauration scale original
        if init_image_scale_override is not None:
            self.motion_module.init_image_scale = original_scale

        # Lissage pour réduire flou / flickering
        if apply_smoothing:
            latents_after = self.smooth_latents(latents_after)

        # Affichage debug
        if latents_after.abs().max() > self.min_threshold:
            self.show_latents(latents_after, title="Après Motion Module")
        elif self.verbose:
            print("[SAFE DEBUG] Motion module appliqué mais frames trop faibles pour affichage")

        return latents_after
