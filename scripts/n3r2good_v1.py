import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from datetime import datetime
from transformers import CLIPTokenizerFast, CLIPTextModel
import os
import math
import ffmpeg


from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae, safe_load_unet, safe_load_scheduler
from scripts.utils.vae_utils import encode_images_to_latents, decode_latents_to_image_tiled
from scripts.utils.motion_utils import load_motion_module, apply_motion_module
from scripts.utils.safe_latent import ensure_valid
from scripts.utils.video_utils import save_frames_as_video, upscale_video
from scripts.utils.n3r_utils import load_image_file, generate_latents

LATENT_SCALE = 0.18215  # Tiny-SD 128x128

# -------------------------
# Main pipeline
# -------------------------

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
# Encode / Decode
# -------------------------
def encode_images_to_latents(images, vae):
    device = vae.device
    images = images.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if images.dim() == 5:  # [B,C,F,H,W]
            B, C, F, H, W = images.shape
            images_2d = images.view(B*F, C, H, W)
            latents_2d = vae.encode(images_2d).latent_dist.sample() * LATENT_SCALE
            latent_shape = latents_2d.shape
            latents = latents_2d.view(B, F, latent_shape[1], latent_shape[2], latent_shape[3])
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
        else:
            latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
            latents = latents.unsqueeze(2)
    return latents

def decode_latents_to_image(latents, vae):
    latents = latents.to(vae.device).float() / LATENT_SCALE
    with torch.no_grad():
        img = vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0,1)
    return img

# -------------------------
# Video utilities
# -------------------------
def save_frames_as_video(frames, output_path, fps=12):
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
# Main ultra safe VRAM
# -------------------------
def main(args):

    cfg = load_config(args.config)
    print("DEBUG: num_fraps_per_image =", cfg.get("num_fraps_per_image"))
    print("DEBUG: full cfg keys =", list(cfg.keys()))
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    # New option creatif
    creative_mode = cfg.get("creative_mode", False)
    creative_scale_min = cfg.get("creative_scale_min", 0.2)
    creative_scale_max = cfg.get("creative_scale_max", 0.8)
    creative_noise = cfg.get("creative_noise", 0.0)
    # PATCH ------------------------------------------------------
    #num_fraps_per_image = int(cfg.get("num_fraps_per_image", 20))
    #print("DEBUG: num_fraps_per_image final =", num_fraps_per_image)
    # ----------------------------------------------------------------
    fps = cfg.get("fps", 12)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    steps = cfg.get("steps", 35)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)

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
    print(f"⏱️ Durée totale estimée de la vidéo : {estimated_seconds:.1f}s")

    # -------------------------
    # Load models
    # -------------------------
    unet = safe_load_unet(args.pretrained_model_path, device, fp16=args.fp16)
    vae = safe_load_vae(args.pretrained_model_path, device, fp16=args.fp16, offload=args.vae_offload)
    scheduler = safe_load_scheduler(args.pretrained_model_path)
    if not unet or not vae or not scheduler:
        print("❌ UNet, VAE ou Scheduler manquant.")
        return

    # ---------------------- Motion Module param ---------------------------------------------------------

    #motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else default_motion_module


    motion_path = cfg.get("motion_module")

    if motion_path:
        motion_module = load_motion_module(motion_path, device=device)

        # Injection paramètres dynamiques
        if hasattr(motion_module, "strength"):
            motion_module.strength = cfg.get("motion_strength", 0.03)

        if hasattr(motion_module, "hair_bias"):
            motion_module.hair_bias = cfg.get("motion_hair_bias", 0.7)

        if hasattr(motion_module, "wave_speed"):
            motion_module.wave_speed = cfg.get("motion_wave_speed", 1.5)

        if hasattr(motion_module, "wave_amplitude"):
            motion_module.wave_amplitude = cfg.get("motion_wave_amplitude", 1.0)
    else:
        motion_module = None

    # -----------------------------------------------------------------------------------------------------


    if not callable(motion_module):
        motion_module = default_motion_module

    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path, "text_encoder")).to(device)
    if args.fp16:
        text_encoder = text_encoder.half()

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()

    frames_for_video = []
    frame_counter = 0

    total_frames = len(input_paths) * num_fraps_per_image * max(len(embeddings), 1)
    pbar = tqdm(total=total_frames, ncols=120)

    # -------------------------
    # Generation loop ultra-safe
    # -------------------------
    for img_path in input_paths:

        input_image = load_images([img_path],
                                W=cfg["W"],
                                H=cfg["H"],
                                device=device,
                                dtype=dtype)

        input_latents = encode_images_to_latents(input_image, vae)
        input_latents = input_latents.expand(-1, -1, num_fraps_per_image, -1, -1).clone()
        input_latents += torch.randn_like(input_latents) * 0.01  # bruit initial

        for pos_embeds, neg_embeds in embeddings:

            B, C, F, H, W = input_latents.shape

            for f in range(F):

                scheduler.set_timesteps(steps, device=device)

                if f == 0:
                    latents_frame = input_latents[:, :, f:f+1, :, :]
                else:
                    # ------------------ GENERATION -------------------------
                    if creative_mode:
                        # guidance dynamique par frame
                        dynamic_scale = guidance_scale * (creative_scale_min + (creative_scale_max - creative_scale_min) * (f / F))
                    else:
                        dynamic_scale = guidance_scale

                    latents_frame = generate_latents(
                        latents=input_latents[:, :, f:f+1, :, :],
                        pos_embeds=pos_embeds,
                        neg_embeds=neg_embeds,
                        unet=unet,
                        scheduler=scheduler,
                        motion_module=motion_module,
                        device=device,
                        dtype=dtype,
                        guidance_scale=dynamic_scale,
                        init_image_scale=init_image_scale * (1 - f / F)
                    )

                    # ajout de bruit créatif léger
                    if creative_mode and creative_noise > 0:
                        latents_frame += torch.randn_like(latents_frame) * creative_noise
                    # ----------------------------------------------------------------------------------------------------------------------
                # -------------------------
                # Vérification et correction des latents
                # -------------------------
                mean_latent = latents_frame.abs().mean().item()
                if mean_latent < 0.05 or math.isnan(mean_latent):
                    # Relance frame avec bruit contrôlé
                    latents_frame = input_latents[:, :, f:f+1, :, :].clone()
                    latents_frame += torch.randn_like(latents_frame) * 0.05
                    print(f"⚠ Frame {frame_counter:05d} relancée, mean_latent={latents_frame.abs().mean().item():.6f}")

                # Clamp pour éviter valeurs extrêmes
                latents_frame = latents_frame.squeeze(2).to(torch.float32).clamp(-3.0, 3.0)

                # Decode VAE tuilé
                frame_tensor = decode_latents_to_image_tiled(
                    latents_frame,
                    vae,
                    tile_size=32, #16 24 32
                    overlap=16 # valeur mini 4 , 8 conseillé , 16 max
                ).clamp(0, 1)

                if frame_tensor.ndim == 4 and frame_tensor.shape[0] == 1:
                    frame_tensor = frame_tensor.squeeze(0)

                frame_pil = to_pil(frame_tensor.cpu())
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")

                frames_for_video.append(frame_pil)
                frame_counter += 1
                pbar.update(1)

                # --- log latent moyen ---
                print(f"Frame {frame_counter:05d} | mean abs(latent) = {latents_frame.abs().mean().item():.6f}")

    pbar.close()

    # --- sauvegarde vidéo ---
    save_frames_as_video(frames_for_video, out_video, fps=fps)
    print(f"🎬 Vidéo générée : {out_video}")

    # --- upscale final ---
    upscaled_video = output_dir / f"output_{timestamp}_x2.mp4"
    upscale_video(out_video, upscaled_video, scale_factor=2)
    print(f"🎬 Vidéo générée X2 : {out_video}")

    print("✅ Pipeline terminé proprement.")

# Entrée
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
