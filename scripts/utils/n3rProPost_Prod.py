#n3rProPost_Prod

# soft_tone_map, post_denoise_light, adjust_color_temperature, IntelligentGlowPro, AdaptivePostProcessor, neutralize_color_cast, apply_post_processing_unreal_cinematic, apply_post_processing_sketch

from PIL import Image, ImageFilter, ImageChops, ImageEnhance
import numpy as np
import math


# Version de test:
def apply_post_processing_minimal(
    frame_pil,
    blur_radius=0.05,
    contrast=1.15,
    vibrance_base=1.0,
    vibrance_max=1.25,
    sharpen=False,
    sharpen_radius=1,
    sharpen_percent=90,
    sharpen_threshold=2,
    clamp_r=True
):


    if frame_pil.mode != "RGB":
        frame_pil = frame_pil.convert("RGB")

    # ---------------- 1. Blur léger ----------------
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # ---------------- 2. Contraste ----------------
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)

    # ---------------- 3. Vibrance adaptative ----------------
    try:
        frame_np = np.array(frame_pil).astype(np.float32)

        max_rgb = np.max(frame_np, axis=2)
        min_rgb = np.min(frame_np, axis=2)
        sat = max_rgb - min_rgb

        factor_map = vibrance_base + (vibrance_max - vibrance_base) * (1 - sat / 255.0)
        factor_map = np.clip(factor_map, vibrance_base, vibrance_max)

        frame_np *= factor_map[..., None]
        frame_np = np.clip(frame_np, 0, 255)

        frame_pil = Image.fromarray(frame_np.astype(np.uint8))

    except Exception as e:
        print(f"[WARNING] vibrance skipped: {e}")

    # ---------------- 4. Clamp rouge ----------------
    if clamp_r:
        try:
            arr = np.array(frame_pil).astype(np.float32)
            r_mean = arr[..., 0].mean()

            if r_mean > 180:
                factor = 180 / r_mean
                arr[..., 0] *= factor

            frame_pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        except Exception as e:
            print(f"[WARNING] clamp rouge skipped: {e}")

    # ---------------- 5. Sharpen ----------------
    if sharpen:
        frame_pil = frame_pil.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold
        ))

    return frame_pil

# Version de test:
def apply_cinematic_neon_glow(frame_pil,
                              glow_strength=0.25,
                              edge_strength=0.15,
                              color_saturation=1.15,
                              exposure=1.05,
                              contrast=1.25,
                              blur_radius=0.4,
                              sharpen=True):
    """
    Filtre original 'Cinematic Neon Glow':
    - Glow subtil autour des zones claires
    - Couleurs saturées style néon / cinématographique
    - Bords légèrement lumineux type sketch
    """


    # -----------------------
    # 1️⃣ Convertir en float
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32) / 255.0

    # -----------------------
    # 2️⃣ Exposure léger
    # -----------------------
    arr *= exposure
    arr = np.clip(arr, 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))

    # -----------------------
    # 3️⃣ Glow subtil (Light Bloom)
    # -----------------------
    bright = img.filter(ImageFilter.GaussianBlur(radius=5))
    img = ImageChops.screen(img, bright)  # effet lumineux
    img = Image.blend(img, bright, glow_strength)

    # -----------------------
    # 4️⃣ Edge sketch léger
    # -----------------------
    gray = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.0))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageChops.invert(edges)
    edges_rgb = Image.merge("RGB", (edges, edges, edges))
    img = ImageChops.blend(img, edges_rgb, edge_strength)

    # -----------------------
    # 5️⃣ Saturation & Contraste
    # -----------------------
    img = ImageEnhance.Color(img).enhance(color_saturation)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # -----------------------
    # 6️⃣ Micro blur anti-pixel
    # -----------------------
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # -----------------------
    # 7️⃣ Sharpen subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=40, threshold=2))

    return img

# Version de test :
def apply_post_processing_drawing(frame_pil,
                                  edge_strength=0.7,
                                  color_levels=48,
                                  saturation=0.95,
                                  contrast=1.10,
                                  sharpen=True):
    """
    Post-processing dessin type line-art.
    Simplifie les couleurs, ajoute des contours au crayon blanc,
    supprime les points noirs et garde un rendu net.
    """

    # -----------------------
    # 1️⃣ Color simplification douce
    # -----------------------
    arr = np.array(frame_pil).astype(np.float32)
    levels = color_levels
    arr = np.round(arr / (256 / levels)) * (256 / levels)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # -----------------------
    # 2️⃣ Edge detection propre
    # -----------------------
    gray = frame_pil.convert("L").filter(ImageFilter.GaussianBlur(radius=0.6))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.8))
    edges = edges.filter(ImageFilter.MedianFilter(size=3))  # supprime points isolés
    edges = ImageEnhance.Contrast(edges).enhance(1.4)
    edges = edges.point(lambda x: 0 if x < 15 else int(x * 1.2))
    edges = ImageChops.invert(edges)
    edge_rgb = Image.merge("RGB", (edges, edges, edges))

    # -----------------------
    # 3️⃣ Fusion douce contours
    # -----------------------
    img_edges = ImageChops.multiply(img, edge_rgb)
    img = Image.blend(img, img_edges, edge_strength * 0.85)

    # -----------------------
    # 4️⃣ Color / Contrast / Sharpen
    # -----------------------
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=3))

    return img

def apply_post_processing_sketch(frame_pil, edge_strength=0.2, blur_radius=0.3, sharpen=True,
                                           contrast_boost=1.6,   # +60% contraste
                                           exposure=0.80):       # -20% brillance
    """
    Effet dessin subtil / croquis clair ajusté :
    - Contours légèrement visibles (blancs doux)
    - +40% contraste, -10% brillance
    - Lisse les pixels isolés
    - Ne dénature pas les couleurs de base
    """


    # -----------------------
    # 1️⃣ Edge detection doux
    # -----------------------
    gray = frame_pil.convert("L").filter(ImageFilter.GaussianBlur(radius=0.5))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.MedianFilter(size=3))   # supprime points isolés
    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.6))  # lissage
    edges = ImageEnhance.Contrast(edges).enhance(1.2)
    edges = ImageChops.invert(edges)
    edge_rgb = Image.merge("RGB", (edges, edges, edges))

    # -----------------------
    # 2️⃣ Fusion douce des edges
    # -----------------------
    img = ImageChops.blend(frame_pil, edge_rgb, edge_strength)

    # -----------------------
    # 3️⃣ Exposure / Brillance
    # -----------------------
    img = ImageEnhance.Brightness(img).enhance(exposure)

    # -----------------------
    # 4️⃣ Contraste
    # -----------------------
    img = ImageEnhance.Contrast(img).enhance(contrast_boost)

    # -----------------------
    # 5️⃣ Blur léger anti-pixel
    # -----------------------
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # -----------------------
    # 6️⃣ Sharp subtil
    # -----------------------
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=40, threshold=2))

    return img


def apply_post_processing_unreal_cinematic(
    frame_pil,
    exposure=1.0,
    vibrance=1.02,
    edge_strength=0.25,
    sharpen=True,
    brightness_adj=0.90,   # 🔻 -5%
    contrast_adj=1.65      # 🔺 +65%
):


    # 🔥 1. Base (sans toucher contraste global)
    arr = np.array(frame_pil).astype(np.float32) / 255.0
    arr *= exposure

    # Vibrance douce
    mean_c = arr.mean(axis=2, keepdims=True)
    arr = mean_c + (arr - mean_c) * vibrance
    arr = np.clip(arr, 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8))

    # =========================
    # ✏️ EDGE CRAYON BLANC
    # =========================
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)

    edges = edges.filter(ImageFilter.GaussianBlur(radius=0.8))
    edges = ImageChops.invert(edges)
    edges = ImageEnhance.Contrast(edges).enhance(1.2)

    edge_rgb = Image.merge("RGB", (edges, edges, edges))

    # Screen = effet lumineux propre
    img_edges = ImageChops.screen(img, edge_rgb)

    # Blend final contrôlé
    img = Image.blend(frame_pil, img_edges, edge_strength)

    # =========================
    # 🔥 AJUSTEMENTS DEMANDÉS
    # =========================
    img = ImageEnhance.Brightness(img).enhance(brightness_adj)
    img = ImageEnhance.Contrast(img).enhance(contrast_adj)

    # =========================
    # 🔧 Sharpen doux
    # =========================
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=0.5,
            percent=30,
            threshold=3
        ))

    # 🔥 micro lissage final
    img = img.filter(ImageFilter.GaussianBlur(radius=0.25))

    return img


def neutralize_color_cast(img, strength=0.45, warm_bias=0.015, green_bias=-0.07):
    """
    Neutralise la dominante de couleur tout en corrigeant un excès de vert.

    Args:
        img (PIL.Image): image à corriger
        strength (float): intensité de neutralisation (0.0 = off, 1.0 = full)
        warm_bias (float): réchauffe légèrement (rouge+/bleu-)
        green_bias (float): ajuste le vert (-0.07 = moins 7%)
    """

    arr = np.array(img).astype(np.float32)

    mean = arr.mean(axis=(0,1))
    gray = mean.mean()

    gain = gray / (mean + 1e-6)
    gain = 1.0 + (gain - 1.0) * strength

    arr[..., 0] *= gain[0] * (1 + warm_bias)  # rouge +
    arr[..., 1] *= gain[1] * (1 + green_bias) # vert corrigé
    arr[..., 2] *= gain[2] * (1 - warm_bias)  # bleu -

    arr = np.clip(arr, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def neutralize_color_cast_clean(img, strength=0.6, warm_bias=0.02):

    arr = np.array(img).astype(np.float32)

    mean = arr.mean(axis=(0,1))
    gray = mean.mean()

    gain = gray / (mean + 1e-6)
    gain = 1.0 + (gain - 1.0) * strength

    arr[..., 0] *= gain[0] * (1 + warm_bias)  # 🔥 léger rouge +
    arr[..., 1] *= gain[1]
    arr[..., 2] *= gain[2] * (1 - warm_bias)  # 🔥 léger bleu -

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def neutralize_color_cast_str(img, strength=0.6):

    arr = np.array(img).astype(np.float32)

    mean = arr.mean(axis=(0,1))
    gray = mean.mean()

    gain = gray / (mean + 1e-6)

    # 🔥 interpolation (clé)
    gain = 1.0 + (gain - 1.0) * strength

    arr[..., 0] *= gain[0]
    arr[..., 1] *= gain[1]
    arr[..., 2] *= gain[2]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def neutralize_color_cast_simple(img):
    arr = np.array(img).astype(np.float32)
    mean = arr.mean(axis=(0,1))

    # cible gris neutre
    gray = mean.mean()

    gain = gray / (mean + 1e-6)

    arr[..., 0] *= gain[0]
    arr[..., 1] *= gain[1]
    arr[..., 2] *= gain[2]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


"""
adaptive_processor = AdaptivePostProcessor(
    blur_radius=0.025,          # micro-blur léger
    denoise_strength=0.03,      # denoise très léger
    detail_strength=0.5,        # boost détails
    contrast_strength=1.08,     # léger contraste global
    vibrance_strength=0.22,     # micro vibrance
    shadow_lift=0.25,           # ajustement shadow
    shadow_threshold=0.35       # seuil mask shadows
)
"""
class AdaptivePostProcessor:
    def __init__(
        self,
        blur_radius=0.01,
        denoise_strength=0.03,
        detail_strength=0.5,
        contrast_strength=1.22,
        vibrance_strength=0.25,
        shadow_lift=0.25,
        shadow_threshold=0.35,
    ):
        self.blur_radius = blur_radius
        self.denoise_strength = denoise_strength
        self.detail_strength = detail_strength
        self.contrast_strength = contrast_strength
        self.vibrance_strength = vibrance_strength
        self.shadow_lift = shadow_lift
        self.shadow_threshold = shadow_threshold

        # Buffers pour caches calculs intensifs
        self.prev_frame_shape = None
        self.shadow_mask = None
        self.mid_mask = None
        self.luma_cache = None  # 🔹 ajouter cache luma

    def _prepare_masks(self, arr):
        """Pré-calcule les masques globaux une seule fois si forme de frame identique"""
        H, W, _ = arr.shape
        if self.prev_frame_shape != (H, W):
            # recalculer luma
            self.luma_cache = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

            # Shadows mask
            self.shadow_mask = np.clip((self.shadow_threshold - self.luma_cache) / self.shadow_threshold, 0, 1) ** 2.0

            # Midtones mask pour dominante globale
            mid_mask = np.clip((self.luma_cache - 0.15) / 0.6, 0, 1) * np.clip((0.9 - self.luma_cache) / 0.6, 0, 1)
            self.mid_mask = mid_mask / (np.max(mid_mask) + 1e-6)

            self.prev_frame_shape = (H, W)

        return self.luma_cache  # 🔹 toujours retourner luma

    def process(self, frame_pil, frame_counter=0):
        if frame_pil.mode != "RGB":
            frame_pil = frame_pil.convert("RGB")

        # ---------------- 1️⃣ MICRO BLUR ----------------
        if frame_counter < 2 and self.blur_radius > 0:
            frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        arr = np.array(frame_pil).astype(np.float32) / 255.0

        # ---------------- 2️⃣ DENOISE ----------------
        if self.denoise_strength > 0:
            mean = np.mean(arr, axis=(0, 1))
            arr = arr * (1.0 - self.denoise_strength) + mean * self.denoise_strength

        # ---------------- 3️⃣ LOCAL CONTRAST ----------------
        mean_lum = np.mean(arr, axis=2, keepdims=True)
        arr = mean_lum + self.contrast_strength * (arr - mean_lum)

        # ---------------- 4️⃣ DETAIL BOOST ----------------
        blurred = np.zeros_like(arr)
        for c in range(3):
            channel = Image.fromarray((arr[..., c] * 255).astype(np.uint8))
            blurred[..., c] = np.array(channel.filter(ImageFilter.GaussianBlur(radius=0.6))).astype(np.float32) / 255.0
        arr = arr + self.detail_strength * (arr - blurred)

        # ---------------- 5️⃣ VIBRANCE ----------------
        max_rgb = np.max(arr, axis=2)
        min_rgb = np.min(arr, axis=2)
        sat = np.clip(max_rgb - min_rgb, 0, 1)
        arr *= (1.0 + self.vibrance_strength * (1.0 - sat))[..., None]

        # ---------------- 6️⃣ Masks et corrections globales ----------------
        luma = self._prepare_masks(arr)

        # Dominante globale midtones
        mean_color = np.sum(arr * self.mid_mask[..., None], axis=(0, 1))
        norm = np.sum(self.mid_mask) + 1e-6
        mean_color /= norm
        neutral = np.mean(arr, axis=(0,1))
        tint_direction = (mean_color - neutral) * 0.6

        black_protect = np.clip(luma / 0.10, 0, 1) ** 2.0
        arr = arr + tint_direction * self.shadow_mask[..., None] * black_protect[..., None] * 0.25

        # Anchor léger pour noirs
        anchor = np.clip(0.07 - luma, 0, 0.07) / 0.07
        arr = arr * (1.0 - 0.02 * anchor[..., None])

        # ---------------- 7️⃣ Final touche ----------------
        arr = np.clip(arr, 0, 1)
        arr *= 0.90  # exposure léger
        arr = np.power(arr, 1.03) ** 1.01  # gamma doux

        return Image.fromarray((arr * 255).astype(np.uint8))


# ------------------------- Glow intelligent pro avec mémoire -------------------------
class IntelligentGlowPro:
    def __init__(self, strength=0.18, edge_weight=0.6, luminance_weight=0.8, blur_radius=1.2):
        self.strength = strength
        self.edge_weight = edge_weight
        self.luminance_weight = luminance_weight
        self.blur_radius = blur_radius
        self.global_lum_mask = None  # Pour stocker l'effet luminance pré-calculé

    def __call__(self, frame_pil: Image.Image, frame_counter: int):
        if frame_pil.mode != "RGB":
            frame_pil = frame_pil.convert("RGB")

        arr = np.array(frame_pil).astype(np.float32) / 255.0

        # ---------------- Luminance ----------------
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        # ---------------- Pré-calcul du mask sur les 2 premières frames ----------------
        if frame_counter < 2 or self.global_lum_mask is None:
            lum_mask = np.clip((lum - 0.6) / 0.4, 0, 1)
            lum_mask = np.power(lum_mask, 1.5)

            # ---------------- Edge ----------------
            gray = (lum * 255).astype(np.uint8)
            edge = Image.fromarray(gray).filter(ImageFilter.FIND_EDGES)
            edge = np.array(edge).astype(np.float32) / 255.0
            edge = np.clip(edge * 1.2, 0, 1)
            edge = np.power(edge, 1.3)

            # ---------------- Mask combiné ----------------
            self.global_lum_mask = np.clip(self.luminance_weight * lum_mask + self.edge_weight * edge, 0, 1)

        combined_mask = self.global_lum_mask

        # ---------------- Glow ----------------
        if frame_counter < 2:
            glow_img = frame_pil.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
            glow_arr = np.array(glow_img).astype(np.float32) / 255.0
            glow_lum = 0.299 * glow_arr[:, :, 0] + 0.587 * glow_arr[:, :, 1] + 0.114 * glow_arr[:, :, 2]
            # Stocker glow_lum pour usage futur si besoin
            self.glow_lum = glow_lum
        else:
            glow_lum = self.glow_lum  # Réutilisation du glow des 2 premières frames

        # ---------------- Appliquer glow seulement sur la luminance ----------------
        result = arr.copy()
        for c in range(3):
            result[:, :, c] = arr[:, :, c] + (glow_lum - lum) * combined_mask * self.strength

        result = np.clip(result, 0, 1)
        return Image.fromarray((result * 255).astype(np.uint8))


def kelvin_to_rgb_photo(temp):
    """
    Approximation réaliste Kelvin → RGB (inspiré photographie)
    """
    temp = temp / 100.0

    # Rouge
    if temp <= 66:
        r = 255
    else:
        r = temp - 60
        r = 329.698727446 * (r ** -0.1332047592)

    # Vert
    if temp <= 66:
        g = temp
        g = 99.4708025861 * math.log(g) - 161.1195681661
    else:
        g = temp - 60
        g = 288.1221695283 * (g ** -0.0755148492)

    # Bleu
    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = temp - 10
        b = 138.5177312231 * math.log(b) - 305.0447927307

    return (
        max(0, min(255, r)) / 255.0,
        max(0, min(255, g)) / 255.0,
        max(0, min(255, b)) / 255.0
    )



def kelvin_to_rgb(temperature):
    """
    Convertit la température en kelvins en valeurs RGB approximatives.
    Température en Kelvin (ex: 6500K pour une lumière blanche neutre).
    """
    temp = temperature / 100.0

    if temp <= 66:
        r = 255
        g = temp
        g = 99.4708025861 * np.log(g) - 161.1195681661
        if temp <= 19:
            b = 0
        else:
            b = temp - 10
            b = 138.5177312231 * np.log(b) - 305.0447927307
    else:
        r = temp - 60
        r = 329.698727446 * (r ** -0.1332047592)
        g = temp - 60
        g = 288.1221695283 * (g ** -0.0755148492)
        b = 255

    return np.clip([r, g, b], 0, 255)


def post_denoise_light(frame_pil: Image.Image, radius: float = 0.5, strength: float = 0.25) -> Image.Image:
    """
    Applique un léger denoising sur l'image PIL pour réduire micro-bruit/banding.

    Args:
        frame_pil (PIL.Image.Image): Image d'entrée.
        radius (float): Rayon du filtre flou léger.
        strength (float): Poids du blending (0 = pas de denoise, 1 = full denoise).

    Returns:
        PIL.Image.Image: Image post-denoise légère.
    """
    # 🔹 Filtre flou très léger
    blurred = frame_pil.filter(ImageFilter.GaussianBlur(radius=radius))

    # 🔹 Blending subtil pour garder détails
    frame_pil = ImageChops.blend(frame_pil, blurred, strength)
    return frame_pil



#--------------------------------------------------------------------------
# Ajustement des couleurs
#--------------------------------------------------------------------------

def adjust_color_temperature(
    image,
    target_temp=7800,
    reference_temp=6500,
    strength=0.5,
    adaptive=True,
    max_gain=2.0,
    neutral_zone=0.08,
    debug=False
):

    img = np.array(image).astype(np.float32) / 255.0

    # =====================================================
    # Kelvin gains
    # =====================================================

    r1, g1, b1 = kelvin_to_rgb(reference_temp)
    r2, g2, b2 = kelvin_to_rgb(target_temp)

    base_gain = np.array([
        r2 / max(r1, 1e-6),
        g2 / max(g1, 1e-6),
        b2 / max(b1, 1e-6)
    ], dtype=np.float32)

    # =====================================================
    # image statistics
    # =====================================================

    mean_rgb = img.reshape(-1, 3).mean(axis=0)
    mean_rgb = np.maximum(mean_rgb, 1e-6)

    r, g, b = mean_rgb

    # =====================================================
    # continuous warmth estimation
    # =====================================================

    # score perceptuel plus robuste
    warmth_score = (
        (r * 0.67 + g * 0.33) - b
    )

    # normalisation
    thermal_bias = np.tanh(warmth_score * 2.5)

    # =====================================================
    # profile classification
    # =====================================================

    if thermal_bias < -neutral_zone:
        profile = "cold"

    elif thermal_bias > neutral_zone:
        profile = "warm"

    else:
        profile = "mixed"

    # =====================================================
    # adaptive modulation
    # =====================================================

    wb_ratio = mean_rgb / mean_rgb.mean()
    imbalance = np.std(wb_ratio)

    if adaptive:

        # image mixte → correction plus douce
        if profile == "mixed":

            adaptive_factor = 1.0 + imbalance * 0.35
            strength_local = strength * 0.55

        # image froide
        elif profile == "cold":

            coldness = abs(thermal_bias)

            adaptive_factor = 1.0 + imbalance * (0.5 + coldness)
            strength_local = strength * (0.7 + coldness * 0.4)

        # image chaude
        else:

            warmness = abs(thermal_bias)

            adaptive_factor = 1.0 + imbalance * (0.7 + warmness)
            strength_local = strength * (0.8 + warmness * 0.5)

    else:

        adaptive_factor = 1.0
        strength_local = strength

    # =====================================================
    # smooth gain interpolation
    # =====================================================

    final_gain = (
        (1.0 - strength_local)
        + strength_local * base_gain * adaptive_factor
    )

    # =====================================================
    # intelligent asymmetric clamp
    # =====================================================

    if profile == "mixed":

        final_gain = np.clip(final_gain, 0.92, 1.08)

    elif profile == "cold":

        final_gain = np.clip(final_gain, 0.85, max_gain)

    else:

        final_gain = np.clip(final_gain, 1 / max_gain, 1.15)

    # =====================================================
    # luminance protection
    # =====================================================

    luminance_before = (
        img[..., 0] * 0.2126 +
        img[..., 1] * 0.7152 +
        img[..., 2] * 0.0722
    )

    img *= final_gain

    luminance_after = (
        img[..., 0] * 0.2126 +
        img[..., 1] * 0.7152 +
        img[..., 2] * 0.0722
    )

    # compensation légère d'exposition
    exposure_fix = (
        luminance_before.mean() /
        max(luminance_after.mean(), 1e-6)
    )

    img *= exposure_fix

    img = np.clip(img, 0, 1)

    # =====================================================
    # debug
    # =====================================================

    if debug:

        print("\n=== TEMP DEBUG V2 ===")
        print("profile:", profile)
        print("thermal_bias:", round(float(thermal_bias), 4))
        print("warmth_score:", round(float(warmth_score), 4))
        print("imbalance:", round(float(imbalance), 4))
        print("adaptive_factor:", round(float(adaptive_factor), 4))
        print("strength_local:", round(float(strength_local), 4))
        print("final_gain:", final_gain)
        print("=====================\n")

    return Image.fromarray(
        (img * 255).astype(np.uint8)
    )

def adjust_color_temperature_v2(
    image,
    target_temp=7800,
    reference_temp=6500,
    strength=0.5,
    adaptive=True,
    max_gain=2.0,
    debug=False
):

    img = np.array(image).astype(np.float32) / 255.0

    r1, g1, b1 = kelvin_to_rgb(reference_temp)
    r2, g2, b2 = kelvin_to_rgb(target_temp)

    base_gain = np.array([r2 / r1, g2 / g1, b2 / b1])

    mean_rgb = img.reshape(-1, 3).mean(axis=0)
    mean_rgb = np.maximum(mean_rgb, 1e-6)

    wb_ratio = mean_rgb / mean_rgb.mean()

    # =====================================================
    # improved temperature direction detection
    # =====================================================

    warmth_score = (mean_rgb[0] * 0.6 + mean_rgb[1] * 0.3) / (mean_rgb[2] + 1e-6)

    # cold image (needs warm correction)
    if warmth_score < 0.95:
        direction = "cold_to_warm"
    else:
        direction = "warm_to_cold"

    # =====================================================
    # adaptive factor (ASYMMETRIC FIX)
    # =====================================================

    imbalance = np.std(wb_ratio)

    if adaptive:
        if direction == "cold_to_warm":
            # doux + progressif
            print("Image trop froide, ajustement vers une température plus chaude.")
            adaptive_factor = 1.0 + 0.6 * imbalance
            strength_local = strength * 0.7
        else:
            print("Image trop chaude, ajustement vers une température plus froide.")
            # plus agressif mais clampé
            adaptive_factor = 1.0 + 1.2 * imbalance
            strength_local = strength * 1.3
    else:
        adaptive_factor = 1.0
        strength_local = strength

    # =====================================================
    # gain interpolation (stable)
    # =====================================================

    final_gain = (1 - strength_local) + strength_local * base_gain * adaptive_factor

    # =====================================================
    # asymmetric clamp (IMPORTANT FIX)
    # =====================================================

    if direction == "cold_to_warm":
        # éviter surchauffe
        final_gain = np.clip(final_gain, 0.85, max_gain)
    else:
        # éviter washout froid
        final_gain = np.clip(final_gain, 1 / max_gain, 1.15)

    # =====================================================
    # apply
    # =====================================================

    img *= final_gain
    img = np.clip(img, 0, 1)

    if debug:
        print("=== TEMP DEBUG ===")
        print("direction:", direction)
        print("warmth_score:", warmth_score)
        print("adaptive_factor:", adaptive_factor)
        print("strength_local:", strength_local)
        print("final_gain:", final_gain)
        print("==================")

    return Image.fromarray((img * 255).astype(np.uint8))

def adjust_color_temperature_v1(
    image,
    target_temp=7800,
    reference_temp=6500,
    strength=0.5,
    adaptive=True,
    max_gain=2.0,
    debug=False
):
    img = np.array(image).astype(np.float32) / 255.0

    # --- 1. Gains température (comme ton code)
    r1, g1, b1 = kelvin_to_rgb(reference_temp)
    r2, g2, b2 = kelvin_to_rgb(target_temp)

    base_gain = np.array([
        r2 / r1,
        g2 / g1,
        b2 / b1
    ])

    # --- 2. Estimation rapide du WB actuel (gray-world simplifié)
    if adaptive:
        mean_rgb = img.reshape(-1, 3).mean(axis=0)
        mean_rgb = np.maximum(mean_rgb, 1e-6)

        # normalisation sur G
        wb_ratio = mean_rgb / mean_rgb[1]

        # mesure du déséquilibre
        imbalance = np.std(wb_ratio)

        # facteur adaptatif doux (évite overcorrection)
        adaptive_factor = 1.0 + min(1.0, imbalance * 2.0)

        # --- 3. Détermination automatique du type de température
        if np.mean(wb_ratio[0]) < 1:
            # Image plus froide, vers plus de chaleur
            print("Image trop froide, ajustement vers une température plus chaude.")
            target_temp = min(target_temp * (1 + strength), 10000)
            print(f"Image target_temp:  {target_temp} ")
            print("==============================================================")
        elif np.mean(wb_ratio[0]) > 1:
            # Image plus chaude, vers plus de froid
            print("Image trop chaude, ajustement vers une température plus froide.")
            target_temp = max(target_temp * (1 - strength), 3000)
            print(f"Image target_temp:  {target_temp} ")
            print("==============================================================")

    else:
        adaptive_factor = 1.0

    # --- 4. Interpolation (ta logique conservée 💡)
    final_gain = (1 - strength) + strength * base_gain * adaptive_factor

    # --- 5. Clamp sécurité (très important en pratique)
    final_gain = np.clip(final_gain, 1 / max_gain, max_gain)

    # --- 6. Application
    img *= final_gain

    img = np.clip(img, 0, 1)

    if debug:
        print("=== DEBUG TEMP ===")
        print(f"mean_rgb: {mean_rgb if adaptive else 'disabled'}")
        print(f"base_gain: {base_gain}")
        print(f"adaptive_factor: {adaptive_factor}")
        print(f"final_gain: {final_gain}")
        print("==================")

    return Image.fromarray((img * 255).astype(np.uint8))

# Exemple d'appel
#image = Image.open("exemple_image.jpg")
#adjusted_image = adjust_color_temperature(image, adaptive=True, debug=True)
#adjusted_image.show()

def adjust_color_temperature_basic(image, target_temp=10000, reference_temp=6500, strength=0.5):


    img = np.array(image).astype(np.float32) / 255.0

    r1, g1, b1 = kelvin_to_rgb(reference_temp)
    r2, g2, b2 = kelvin_to_rgb(target_temp)

    # 🔥 interpolation (clé)
    r_gain = (1 - strength) + strength * (r2 / r1)
    g_gain = (1 - strength) + strength * (g2 / g1)
    b_gain = (1 - strength) + strength * (b2 / b1)

    img[..., 0] *= r_gain
    img[..., 1] *= g_gain
    img[..., 2] *= b_gain

    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def soft_tone_map(img):

    arr = np.array(img).astype(np.float32) / 255.0

    # 🔥 contraste léger (au lieu de compression)
    mean = arr.mean(axis=(0,1), keepdims=True)
    arr = (arr - mean) * 1.1 + mean

    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
