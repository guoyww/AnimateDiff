# --------------------------------------------------------------
# nr3perfect - INTERPOLATION fast movie - Optimal (video support)
# --------------------------------------------------------------

import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime
import os
import math
import cv2
from PIL import Image, ImageFilter
from torchvision.transforms import ToPILImage
from transformers import CLIPTokenizerFast, CLIPTextModel
from PIL import ImageDraw, ImageFont
from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import (
    safe_load_unet,
    safe_load_scheduler,
    safe_load_vae_stable
)
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import (
    generate_latents_robuste,
    load_image_file,
    decode_latents_to_image_auto
)

LATENT_SCALE = 0.18215


# --------------------------------------------------------------
# STOP THREAD
# --------------------------------------------------------------

import threading
import numpy as np
stop_generation = False

def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²":
        stop_generation = True

threading.Thread(target=wait_for_stop, daemon=True).start()

# --------------------------------------------------------------
# UTILS
# --------------------------------------------------------------
# ---------------- TRACKING GLOBALS ----------------
tracker = None
tracking_initialized = False

# ---------------- WATERMARK ----------------
import matplotlib.pyplot as plt


# ---------------- WATERMARK OPTIMISÉ ----------------
def remove_watermark_with_cached_mask(
    frame_pil,
    target_hex_list,
    candidate_zones,
    tolerance=26,
    threshold=0.4,
    blur_radius=10,
    feather_radius=8,
    overlay_text=None,
    text_opacity=0.6,
    text_scale=0.7,
    text_color=(255, 255, 255),
    cached_mask=None,
    force_recalc=False
):
    """
    Retire le watermark avec masque mis en cache pour accélérer le traitement.
    - cached_mask: liste de tuples (x, y, w, h, mask_soft) pour réutilisation
    - force_recalc: recalculer le masque même si cached_mask existe
    """
    img_np = np.array(frame_pil).astype(np.int16)
    H, W, _ = img_np.shape

    if cached_mask is None or force_recalc:
        mask_store = []

        target_colors = np.array(
            [[int(h[i:i+2], 16) for i in (1, 3, 5)] for h in target_hex_list],
            dtype=np.int16
        )

        if candidate_zones is None:
            candidate_zones = [(0, 0, W, H)]

        for (x, y, w, h) in candidate_zones:
            patch = img_np[y:y+h, x:x+w]

            mask_total = np.zeros((h, w), dtype=np.uint8)
            for color in target_colors:
                dist = np.linalg.norm(patch - color, axis=2)
                mask_total += (dist <= tolerance).astype(np.uint8)

            ratio = mask_total.sum() / (w * h)
            if ratio >= threshold:
                mask_binary = mask_total * 255
                mask_binary = cv2.GaussianBlur(mask_binary, (0, 0), feather_radius)
                mask_binary = mask_binary.astype(np.float32) / 255.0
            else:
                mask_binary = np.zeros((h, w), dtype=np.float32)

            mask_store.append((x, y, w, h, mask_binary))

        cached_mask = mask_store

    # ---- APPLICATION DU MASK ----
    for (x, y, w, h, mask_soft) in cached_mask:
        mask_soft_exp = np.expand_dims(mask_soft, axis=2).astype(np.float32)

        region = img_np[y:y+h, x:x+w].astype(np.float32)
        region_blur = cv2.GaussianBlur(region, (0, 0), blur_radius)

        blended = region * (1 - mask_soft_exp) + region_blur * mask_soft_exp
        img_np[y:y+h, x:x+w] = blended.astype(np.uint8)

    blended_img = Image.fromarray(img_np.astype(np.uint8))

    # ---- TEXT OVERLAY ----
    if overlay_text is not None:
        draw_layer = Image.new("RGBA", blended_img.size, (0,0,0,0))
        draw = ImageDraw.Draw(draw_layer)

        for (x, y, w, h, _) in cached_mask:
            try:
                font_size = int(h * text_scale)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0,0), overlay_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            tx = x + (w - text_w)//2
            ty = y + (h - text_h)//2

            draw.text(
                (tx, ty),
                overlay_text,
                font=font,
                fill=(*text_color, int(255 * text_opacity))
            )

        blended_img = Image.alpha_composite(blended_img.convert("RGBA"), draw_layer).convert("RGB")

    return blended_img, cached_mask

#---------------------------------------------------------------------------------------
def normalize_frame(frame_tensor):
    if frame_tensor.min() < 0:
        frame_tensor = (frame_tensor + 1.0) / 2.0
    return frame_tensor.clamp(0, 1)

def compute_overlap(W, H, block_size, max_overlap_ratio=0.6):
    overlap = int(block_size * max_overlap_ratio)
    overlap = min(overlap, min(W, H) // 4)
    return overlap

def load_images(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        t = load_image_file(p, W, H, device, dtype)
        print(f"✅ Image chargée : {p}")
        all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)


def load_images_from_pil(pil_images, W, H, device, dtype, preproc_fn=None):
    """
    Charge une liste de PIL.Images et retourne un tensor 4D [B, C, H, W].
    - preproc_fn : fonction optionnelle à appliquer après resize et avant conversion tensor
    """
    all_tensors = []

    for idx, img_pil in enumerate(pil_images):
        # Redimensionner
        img_resized = img_pil.resize((W, H), Image.LANCZOS)

        # Appliquer un pré-traitement optionnel (ex: flou watermark)
        if preproc_fn is not None:
            img_resized = preproc_fn(img_resized)

        # Convertir en numpy 0..1
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        # Convertir en tensor torch [C, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device=device, dtype=dtype)
        # Normaliser [-1, 1]
        img_tensor = img_tensor * 2 - 1
        all_tensors.append(img_tensor)
        print(f"✅ Image chargée et préparée : {idx}")

    return torch.stack(all_tensors, dim=0)


def extract_frames_from_video(video_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = output_dir / f"video_frame_{idx:05d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        idx += 1

    cap.release()
    print(f"🎬 {idx} frames extraites depuis la vidéo.")
    return frame_paths

def save_frames_as_video_from_folder(folder_path, output_path, fps=12):
    import ffmpeg
    folder_path = Path(folder_path)

    frame_files = sorted(folder_path.glob("frame_*.png"))
    if not frame_files:
        print("❌ Aucun frame trouvé.")
        return

    pattern = str(folder_path / "frame_*.png")

    (
        ffmpeg.input(pattern, framerate=fps, pattern_type='glob')
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )

def encode_images_to_latents(images, vae):
    images = images.to(device=vae.device, dtype=torch.float32)
    with torch.inference_mode():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * LATENT_SCALE
        latents = latents.unsqueeze(2)
    return latents

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

def main(args):

    cfg = load_config(args.config)

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    fps = cfg.get("fps", 12)
    upscale_factor = cfg.get("upscale_factor", 2)
    transition_frames = cfg.get("transition_frames", 8)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    rm_watermark = cfg.get("rm_watermark", True)

    steps = cfg.get("steps", 50)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise", 0.0)

    # ----------------------------------------------------------
    # INPUT IMAGE / VIDEO
    # ----------------------------------------------------------

    if args.input_video:
        print("🎥 Mode vidéo activé")
        temp_dir = Path("./temp_video_frames")
        input_paths = extract_frames_from_video(args.input_video, temp_dir)
    else:
        input_paths = cfg.get("input_images") or [cfg.get("input_image")]

    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    total_frames = len(input_paths) * max(len(prompts), 1) * num_fraps_per_image
    if rm_watermark:
        print(f"🎞 Remove Water Active")

    print(f"🎞 Frames estimées : {total_frames}")
    print("⏹ Appuyez sur '²' + Entrée pour stopper.")

    # ----------------------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------------------

    unet = safe_load_unet(args.pretrained_model_path, device, fp16=args.fp16)
    vae = safe_load_vae_stable(args.pretrained_model_path, device, fp16=args.fp16, offload=args.vae_offload)
    scheduler = safe_load_scheduler(args.pretrained_model_path)
    scheduler.set_timesteps(steps, device=device)

    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None

    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path, "text_encoder")).to(device)
    if args.fp16:
        text_encoder = text_encoder.half()

    # ----------------------------------------------------------
    # PROMPT EMBEDDINGS
    # ----------------------------------------------------------

    embeddings = []
    for prompt_item in prompts:
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item, list) else str(prompt_item)
        neg_text = " ".join(negative_prompts) if isinstance(negative_prompts, list) else str(negative_prompts)

        text_inputs = tokenizer(prompt_text, padding="max_length", truncation=True,
                                max_length=tokenizer.model_max_length, return_tensors="pt")
        neg_inputs = tokenizer(neg_text, padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt")

        with torch.no_grad():
            pos_embeds = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
            neg_embeds = text_encoder(neg_inputs.input_ids.to(device)).last_hidden_state

        embeddings.append((pos_embeds.to(dtype), neg_embeds.to(dtype)))

    # ----------------------------------------------------------
    # OUTPUT
    # ----------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/fastperfect_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    to_pil = ToPILImage()
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    # ----------------------------------------------------------
    # MAIN LOOP
    # ----------------------------------------------------------
    # Variables pour watermark mis en cache
    watermark_recalc_every = 10
    last_mask = None
    frame_index = 0
    #-----------------------------------------------------------
    for img_path in input_paths:

        if stop_generation:
            break

        #----------- fonction remove watermark auto
        if rm_watermark:
            colors = ["#EADBDE", "#FDF8F4", "#FFFAF9", "#F9EDEB", "#C5BABA",
                    "#FBF8FA", "#FAF8F9", "#FBFAF5", "#ECE3E2", "#6D6C6B",
                    "#F9F3EF", "#DDCAC6", "#DDCAC6", "#F3EBEB", "#FDF8F6",
                    "#FFFAFF", "#FDFCFB", "#FAF8F8", "#F9F4F7", "#222021"]
            candidate_zones = [
                (21, 366, 123, 28),
                (421, 572, 125, 28)
            ]

            # Chargement PIL
            pil_img = Image.open(img_path).convert("RGB")

            # Recalcul du watermark tous les watermark_recalc_every frames
            force_recalc = (frame_index % watermark_recalc_every == 0)

            pil_img, last_mask = remove_watermark_with_cached_mask(
                pil_img,
                target_hex_list=colors,
                candidate_zones=candidate_zones,
                overlay_text="N3ORAY",
                text_opacity=0.9,
                text_scale=4.0,
                cached_mask=last_mask,
                force_recalc=force_recalc
            )

            # Convertir en tensor
            input_image = load_images_from_pil(
                [pil_img],
                W=cfg["W"],
                H=cfg["H"],
                device=device,
                dtype=dtype
            )
        else:
            input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)

        input_latents_single = encode_images_to_latents(input_image, vae)
        input_latents = input_latents_single.repeat(1, 1, num_fraps_per_image, 1, 1)

        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):

                if stop_generation:
                    break

                if f == 0:
                    frame_tensor = (input_image.squeeze(0) + 1.0) / 2.0
                else:
                    latents_frame = input_latents[:, :, f:f+1, :, :].clone()
                    try:
                        latents_frame = generate_latents_robuste(
                            latents_frame,
                            pos_embeds,
                            neg_embeds,
                            unet,
                            scheduler,
                            motion_module=motion_module,
                            device=device,
                            dtype=dtype,
                            guidance_scale=guidance_scale,
                            init_image_scale=init_image_scale,
                            creative_noise=creative_noise,
                            seed=frame_counter
                        )
                    except:
                        pass

                    frame_tensor = decode_latents_to_image_auto(latents_frame, vae)
                    del latents_frame

                frame_tensor = normalize_frame(frame_tensor)
                if frame_tensor.ndim == 4:
                    frame_tensor = frame_tensor.squeeze(0)

                frame_pil = to_pil(frame_tensor.cpu()).filter(ImageFilter.GaussianBlur(radius=0.2))

                # -----------------------------------------------------------------------------
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize(
                        (frame_pil.width * upscale_factor, frame_pil.height * upscale_factor),
                        resample=Image.BICUBIC
                    )

                frame_path = output_dir / f"frame_{frame_counter:05d}.png"
                frame_pil.save(frame_path)

                del frame_tensor, frame_pil
                #torch.cuda.empty_cache()

                frame_counter += 1
                pbar.update(1)

                # Nettoyage VRAM toutes les 20 frames
                if frame_counter % 20 == 0:
                    torch.cuda.empty_cache()
            frame_index += 1
    pbar.close()

    save_frames_as_video_from_folder(output_dir, out_video, fps=fps)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé proprement.")

# --------------------------------------------------------------
# ARGPARSE
# --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    parser.add_argument("--input-video", type=str, default=None)
    args = parser.parse_args()

    main(args)
