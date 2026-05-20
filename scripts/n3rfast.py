import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime
import os
import math
import shutil
from PIL import Image
from PIL import ImageFilter
from torchvision.transforms import ToPILImage

from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import (
    safe_load_unet,
    safe_load_scheduler,
    safe_load_vae_stable,
    decode_latents_to_image_tiled
)
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_robuste, load_image_file

LATENT_SCALE = 0.18215


def normalize_frame(frame_tensor):
    if frame_tensor.min() < 0:
        frame_tensor = (frame_tensor + 1.0) / 2.0
    return frame_tensor.clamp(0,1)

# -------------------------
def compute_overlap(W, H, block_size, max_overlap_ratio=0.6):
    overlap = int(block_size * max_overlap_ratio)
    overlap = min(overlap, min(W,H)//4)
    return overlap

# -------------------------
def load_images(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        t = load_image_file(p, W, H, device, dtype)
        print(f"✅ Image chargée : {p}")
        all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)

# -------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# -------------------------
def save_frames_as_video(frames, output_path, fps=12):
    import ffmpeg
    temp_dir = Path("temp_frames")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    for idx, frame in enumerate(frames):
        frame.save(temp_dir / f"frame_{idx:05d}.png")
    (
        ffmpeg.input(f"{temp_dir}/frame_%05d.png", framerate=fps)
        .output(str(output_path), vcodec="libx264", pix_fmt="yuv420p")
        .overwrite_output()
        .run(quiet=True)
    )
    shutil.rmtree(temp_dir)

# -------------------------
def encode_images_to_latents(images, vae):
    images = images.to(device=vae.device, dtype=torch.float32)
    with torch.inference_mode():
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * LATENT_SCALE
        latents = latents.unsqueeze(2)
    return latents

# -------------------------
def main(args):
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    fps = cfg.get("fps", 12)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    steps = cfg.get("steps", 50)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise", 0.0)

    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)
    estimated_seconds = total_frames / fps
    print("📌 Paramètres de génération :")
    print(f"  fps                  : {fps}")
    print(f"  num_fraps_per_image : {num_fraps_per_image}")
    print(f"  steps                : {steps}")
    print(f"  guidance_scale       : {guidance_scale}")
    print(f"  init_image_scale     : {init_image_scale}")
    print(f"  creative_noise       : {creative_noise}")
    print(f"⏱ Durée totale estimée de la vidéo : {estimated_seconds:.1f}s")

    # -------------------------
    unet = safe_load_unet(args.pretrained_model_path, device, fp16=args.fp16)
    try:
        unet.enable_xformers_memory_efficient_attention()
        unet.set_attention_slice("max")
        vae.enable_slicing()
        vae.enable_tiling()
        print("✅ xFormers memory efficient attention activé")
    except Exception:
        print("⚠ xFormers non disponible")

    vae = safe_load_vae_stable(args.pretrained_model_path, device, fp16=args.fp16, offload=args.vae_offload)
    scheduler = safe_load_scheduler(args.pretrained_model_path)

    if not vae:
        print("❌ VAE manquant.")
        return
    if not scheduler:
        print("❌ Scheduler manquant.")
        return
    scheduler.set_timesteps(steps, device=device)
    if not unet:
        print("❌ UNet manquant.")
        return

    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path, "text_encoder")).to(device)
    if args.fp16:
        text_encoder = text_encoder.half()

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/fast_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    to_pil = ToPILImage()
    frames_for_video = []
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    # -------------------------
    for img_idx, img_path in enumerate(input_paths):

        input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        input_latents = encode_images_to_latents(input_image, vae)
        input_latents = input_latents.expand(-1, -1, num_fraps_per_image, -1, -1).clone()

        block_size = cfg.get("block_size", 64)
        overlap = compute_overlap(cfg["W"], cfg["H"], block_size, max_overlap_ratio=0.6)

        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                # ------------------------- Première frame
                #if frame_counter == 0:
                if f == 0:
                    # Première frame = image input originale
                    frame_tensor = input_image.squeeze(0)
                    # Si le tenseur est normalisé [-1,1], le ramener à [0,1]
                    frame_tensor = (frame_tensor + 1.0) / 2.0
                    frame_tensor = frame_tensor.clamp(0,1)
                    print(f"ℹ️ Frame {frame_counter:05d} = image input originale")
                    latents_frame = None
                else:
                    latents_frame = input_latents[:, :, f:f+1, :, :].clone()

                if latents_frame is not None: # new code
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
                    except Exception as e:
                        print(f"⚠ Erreur génération frame {frame_counter:05d}, reset avec petit bruit: {e}")
                        latents_frame = input_latents[:, :, f:f+1, :, :] + torch.randn_like(input_latents[:, :, f:f+1, :, :]) * 0.05
                        latents_frame = latents_frame.to(dtype=dtype)

                # Clamp et decode tuilé
                if latents_frame is not None:
                    latents_frame = latents_frame.squeeze(2).clamp(-3.0, 3.0)
                #frame_tensor = decode_latents_to_image_tiled(latents_frame, vae, tile_size=32, overlap=16).clamp(0,1)

                #------------------- NEW CODE -------------------------------------------
                block_size = cfg.get("block_size", 64)
                overlap = compute_overlap(cfg["W"], cfg["H"], block_size, max_overlap_ratio=0.6) # 0.6 ou 0.65 Max

                if latents_frame is not None:
                    frame_tensor = decode_latents_to_image_tiled(
                        latents_frame, vae,
                        tile_size=block_size,
                        overlap=overlap
                    ).clamp(0,1)

                    frame_tensor = normalize_frame(frame_tensor) # A test ok
                #------------------------------------------------------------------------


                if frame_tensor.ndim == 4 and frame_tensor.shape[0] == 1:
                    frame_tensor = frame_tensor.squeeze(0)

                frame_pil = to_pil(frame_tensor.cpu())
                frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=0.2)) # Ajout flou sur le bord pour lisser l'overlap'
                # ---------------- UPSCALE ----------------
                upscale_factor = cfg.get("upscale_factor", 2)  # ajouter dans config si besoin
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize(
                        (frame_pil.width * upscale_factor, frame_pil.height * upscale_factor),
                        resample=Image.BICUBIC
                    )
                # ----------------------------------------

                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")

                frames_for_video.append(frame_pil)
                frame_counter += 1
                pbar.update(1)

                if latents_frame is not None:
                    mean_lat = latents_frame.abs().mean().item()
                    if math.isnan(mean_lat) or mean_lat < 1e-5:
                        print(f"⚠ Frame {frame_counter:05d} contient NaN ou latent trop petit, réinitialisation")

                del latents_frame, frame_tensor
                torch.cuda.empty_cache()

    pbar.close()
    save_frames_as_video(frames_for_video, out_video, fps=fps)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé proprement.")

# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
