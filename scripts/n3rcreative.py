import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime
import os
import math
import shutil
from PIL import Image, ImageSequence
import numpy as np
import ffmpeg
from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae_stable, safe_load_unet, safe_load_scheduler
from scripts.utils.vae_utils import encode_images_to_latents, decode_latents_to_image_tiled
from scripts.utils.motion_utils import load_motion_module, apply_motion_module
from scripts.utils.safe_latent import ensure_valid
from scripts.utils.video_utils import save_frames_as_video
from scripts.utils.n3r_utils import load_image_file, generate_latents

LATENT_SCALE = 0.18215  # Tiny-SD 128x128

# -------------------------
# Image utilities
# -------------------------
def load_images(paths, W, H, device, dtype):
    all_tensors = []
    for p in paths:
        if p.lower().endswith(".gif"):
            img = Image.open(p)
            frames = [torch.tensor(np.array(f)).permute(2,0,1).to(device=device, dtype=dtype)/127.5 - 1.0
                      for f in ImageSequence.Iterator(img)]
            print(f"✅ GIF chargé : {p} avec {len(frames)} frames")
            all_tensors.extend(frames)
        else:
            t = load_image_file(p, W, H, device, dtype)
            print(f"✅ Image chargée : {p}")
            all_tensors.append(t)
    return torch.stack(all_tensors, dim=0)

# -------------------------
# Main
# -------------------------
def main(args):

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    # Paramètres principaux
    fps = cfg.get("fps", 12)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    steps = cfg.get("steps", 35)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)

    # Paramètres créatifs
    creative_mode = cfg.get("creative_mode", False)
    creative_scale_min = cfg.get("creative_scale_min", 0.2)
    creative_scale_max = cfg.get("creative_scale_max", 0.8)
    creative_noise = cfg.get("creative_noise", 0.0)

    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)
    estimated_seconds = total_frames / fps

    print("📌 Paramètres de génération :")
    print(f"  fps                  : {fps}")
    print(f"  num_fraps_per_image  : {num_fraps_per_image}")
    print(f"  steps                : {steps}")
    print(f"  guidance_scale        : {guidance_scale}")
    print(f"  init_image_scale      : {init_image_scale}")
    print(f"  creative_noise        : {creative_noise}")
    print(f"⏱ Durée totale estimée de la vidéo : {estimated_seconds:.1f}s")

    # -------------------------
    # Load models
    # -------------------------
    unet = safe_load_unet(args.pretrained_model_path, device, fp16=args.fp16)
    vae = safe_load_vae_stable(args.pretrained_model_path, device, fp16=args.fp16, offload=args.vae_offload)
    scheduler = safe_load_scheduler(args.pretrained_model_path)
    if not unet or not vae or not scheduler:
        print("❌ UNet, VAE ou Scheduler manquant.")
        return

    # Motion module
    motion_path = cfg.get("motion_module")
    motion_module = load_motion_module(motion_path, device=device) if motion_path else None
    if not callable(motion_module):
        from scripts.utils.motion_utils import default_motion_module
        motion_module = default_motion_module

    # Tokenizer / Text Encoder
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path, "text_encoder")).to(device)
    if args.fp16:
        text_encoder = text_encoder.half()

    # Préparer embeddings
    embeddings = []
    for prompt_item in prompts:
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item, list) else str(prompt_item)
        neg_text = " ".join(negative_prompts) if isinstance(negative_prompts, list) else str(negative_prompts)
        text_inputs = tokenizer(prompt_text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        neg_inputs = tokenizer(neg_text, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
        with torch.no_grad():
            pos_embeds = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
            neg_embeds = text_encoder(neg_inputs.input_ids.to(device)).last_hidden_state
        embeddings.append((pos_embeds.to(dtype), neg_embeds.to(dtype)))

    # -------------------------
    # Output
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/creative_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()

    frames_for_video = []
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    # -------------------------
    # Generation loop
    # -------------------------
    for img_path in input_paths:

        input_image = load_images([img_path],
                                  W=cfg["W"],
                                  H=cfg["H"],
                                  device=device,
                                  dtype=dtype)

        input_latents = encode_images_to_latents(input_image, vae)
        # On s'assure que les latents ont bien une dimension F=1 si image unique
        if input_latents.dim() == 4:
            input_latents = input_latents.unsqueeze(2)

        B, C, F, H_lat, W_lat = input_latents.shape

        # On génère chaque frame
        for pos_embeds, neg_embeds in embeddings:
            for f_idx in range(F):
                scheduler.set_timesteps(steps, device=device)

                latents_frame = input_latents[:, :, f_idx:f_idx+1, :, :].clone()
                # Appliquer bruit créatif si activé
                if creative_mode and creative_noise > 0.0:
                    latents_frame += torch.randn_like(latents_frame) * creative_noise

                # guidance dynamique
                dynamic_scale = guidance_scale
                if creative_mode:
                    dynamic_scale *= creative_scale_min + (creative_scale_max - creative_scale_min) * (f_idx / F)

                # init_image_scale progressif
                progressive_init_scale = init_image_scale * (1 - f_idx / max(F-1,1))

                # Génération du latent
                latents_frame = generate_latents(
                    latents=latents_frame,
                    pos_embeds=pos_embeds,
                    neg_embeds=neg_embeds,
                    unet=unet,
                    scheduler=scheduler,
                    motion_module=motion_module,
                    device=device,
                    dtype=dtype,
                    guidance_scale=dynamic_scale,
                    init_image_scale=progressive_init_scale
                )

                # Clamp et check NaN
                if torch.isnan(latents_frame).any():
                    print(f"⚠ Frame {frame_counter:05d} contient NaN, réinitialisation avec petit bruit")
                    latents_frame = input_latents[:, :, f_idx:f_idx+1, :, :].clone()
                    latents_frame += torch.randn_like(latents_frame) * max(creative_noise, 0.01)
                latents_frame = latents_frame.clamp(-3.0, 3.0).squeeze(2).to(torch.float32)

                mean_lat = latents_frame.abs().mean().item()
                if mean_lat < 0.01:
                    print(f"⚠ Frame {frame_counter:05d} a latent moyen trop faible ({mean_lat:.6f}), ajout de bruit minimal")
                    latents_frame += torch.randn_like(latents_frame) * max(creative_noise, 0.01)

                # Décodage tuilé
                frame_tensor = decode_latents_to_image_tiled(latents_frame, vae, tile_size=32, overlap=16).clamp(0,1)
                if frame_tensor.ndim == 4 and frame_tensor.shape[0] == 1:
                    frame_tensor = frame_tensor.squeeze(0)

                frame_pil = to_pil(frame_tensor.cpu())
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frames_for_video.append(frame_pil)

                frame_counter += 1
                pbar.update(1)
                print(f"Frame {frame_counter:05d} | mean abs(latent) = {mean_lat:.6f}")

    pbar.close()

    # -------------------------
    # Sauvegarde vidéo
    # -------------------------
    save_frames_as_video(frames_for_video, out_video, fps=fps)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé proprement.")

# -------------------------
# Entrée
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
