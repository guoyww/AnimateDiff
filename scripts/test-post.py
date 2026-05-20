# test_post_processing.py
from PIL import Image, ImageEnhance, ImageFilter
import argparse

def apply_post_processing(frame_pil,
                          blur_radius=0.1,
                          contrast=1.3,
                          brightness=1.1,
                          saturation=0.6,
                          sharpen=True,
                          sharpen_radius=1,
                          sharpen_percent=80,
                          sharpen_threshold=3):
    """Appliquer des effets post-decode sur une frame PIL avec blur, contraste, luminosité, saturation et sharpen."""
    # GaussianBlur
    if blur_radius > 0:
        frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Ajustements
    if contrast != 1.0:
        frame_pil = ImageEnhance.Contrast(frame_pil).enhance(contrast)
    if brightness != 1.0:
        frame_pil = ImageEnhance.Brightness(frame_pil).enhance(brightness)
    if saturation != 1.0:
        frame_pil = ImageEnhance.Color(frame_pil).enhance(saturation)

    # UnsharpMask
    if sharpen:
        frame_pil = frame_pil.filter(ImageFilter.UnsharpMask(
            radius=sharpen_radius,
            percent=sharpen_percent,
            threshold=sharpen_threshold
        ))

    return frame_pil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester le post-processing d'une image PIL")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers l'image d'entrée")
    parser.add_argument("--output", type=str, default="output.png", help="Chemin de l'image sortie")
    parser.add_argument("--blur", type=float, default=0.2, help="Rayon du blur")
    parser.add_argument("--contrast", type=float, default=1.5, help="Facteur de contraste")
    parser.add_argument("--brightness", type=float, default=1.0, help="Facteur de luminosité")
    parser.add_argument("--saturation", type=float, default=1.05, help="Facteur de saturation")
    parser.add_argument("--sharpen", action="store_true", help="Activer le sharpen")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    processed = apply_post_processing(
        img,
        blur_radius=args.blur,
        contrast=args.contrast,
        brightness=args.brightness,
        saturation=args.saturation,
        sharpen=args.sharpen
    )
    processed.save(args.output)
    print(f"✅ Image traitée sauvegardée : {args.output}")
