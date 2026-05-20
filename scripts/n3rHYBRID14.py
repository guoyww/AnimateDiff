# -------------------------
# n3rHYBRID14_ULTRA_STABLE.py
# -------------------------

import os, time, csv
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from PIL import Image
import cv2

from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae_safetensors, safe_load_unet, safe_load_scheduler, safe_load_vae_stable
from scripts.utils.vae_utils import decode_latents_to_image_tiled
from scripts.utils.vae_utils import test_vae_256
from scripts.utils.model_utils import load_pretrained_unet, get_text_embeddings, load_DDIMScheduler
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_images, decode_latents_correct, generate_latents_ai_5D_optimized
from transformers import CLIPTextModel, CLIPTokenizer

LATENT_SCALE = 0.18215
CLAMP_MAX = 1.0
torch.backends.cuda.matmul.allow_tf32 = True

def save_frame(img_array, filename):
    img_array = np.clip(img_array, 0.0, 1.0)
    img_uint8 = (img_array * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Image.fromarray(img_uint8).save(filename)

def encode_image_latents(image_tensor, vae, scale=LATENT_SCALE):
    device = next(vae.parameters()).device
    img = image_tensor.to(device=device, dtype=next(vae.parameters()).dtype)
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample() * scale
    return latents.unsqueeze(2)

def main(args):
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    fps = cfg.get("fps", 12)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    steps = cfg.get("steps", 35)
    seed = cfg.get("seed",42)
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
    print(f"  seed                : {seed}")
    print(f"  guidance_scale       : {guidance_scale}")
    print(f"  init_image_scale     : {init_image_scale}")
    print(f"  creative_noise       : {creative_noise}")
    print(f"⏱ Durée totale estimée de la vidéo : {estimated_seconds:.1f}s")

    # -------------------------
    # Load models
    # -------------------------
    unet = safe_load_unet(args.pretrained_model_path, device, fp16=args.fp16)
    vae = safe_load_vae_stable(args.pretrained_model_path, device, fp16=args.fp16, offload=args.vae_offload)
    scheduler = safe_load_scheduler(args.pretrained_model_path)
    if not vae :
        print("❌ VAE manquant.")
        return

    if not scheduler:
        print("❌ Scheduler manquant.")
        return

    if not unet :
        print("❌ UNet manquant.")
        return

    test_vae_256(vae, Image.open("scripts/utils/logo.png").convert("RGB"))

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

    # -------------------------
    # UNet + Scheduler
    # -------------------------
    #unet = load_pretrained_unet(args.pretrained_model_path, device=device, dtype=dtype)
    unet.eval()
    try: unet.enable_xformers_memory_efficient_attention()
    except: pass
    #scheduler = load_DDIMScheduler(args.pretrained_model_path)

    motion_module = load_motion_module(cfg.get("motion_module"), device=device) \
                    if cfg.get("motion_module") else None

    # -------------------------
    # OUTPUT
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/hybrid14_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"; debug_dir.mkdir(exist_ok=True)
    csv_file = output_dir / "generation_log.csv"

    video = None
    frame_counter = 0

    with open(csv_file,"w",newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame","latent_min","latent_max","gen_time","decode_time"])

        for img_path in input_paths:
            input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
            input_latents = encode_image_latents(input_image, vae)
            input_latents = input_latents.expand(-1,-1,num_fraps_per_image,-1,-1).clone()

            # --- Découpage en segments pour éviter OOM ---
            segment_size = 4
            num_segments = (num_fraps_per_image + segment_size - 1) // segment_size

            for seg_idx in range(num_segments):
                start_idx = seg_idx * segment_size
                end_idx = min(start_idx + segment_size, num_fraps_per_image)

                for f_idx in range(start_idx, end_idx):
                    torch.cuda.empty_cache()
                    latent_frame = input_latents[:,:,f_idx:f_idx+1,:,:].squeeze(2)
                    frame_seed = seed + f_idx

                    # --- Génération latent ---
                    gen_start = time.time()
                    latent_frame = generate_latents_ai_5D_optimized(
                        latent_frame=latent_frame,
                        scheduler=scheduler,
                        pos_embeds=pos_embeds,
                        neg_embeds=neg_embeds,
                        unet=unet,
                        motion_module=motion_module,
                        device=device,
                        dtype=dtype,
                        guidance_scale=guidance_scale,
                        init_image_scale=init_image_scale,
                        creative_noise=creative_noise,
                        seed=frame_seed,
                        steps=steps
                    )
                    gen_time = time.time() - gen_start

                    # --- Décodage VAE ---
                    decode_start = time.time()
                    if args.vae_offload:
                        vae.to(device)
                    print("Latent stats:",
                          float(latent_frame.min()),
                          float(latent_frame.max()),
                          float(latent_frame.mean()))
                    # important
                    #latent_frame = latent_frame / LATENT_SCALE
                    frame_tensor = decode_latents_correct(latent_frame, vae)
                    decode_time = time.time() - decode_start
                    if args.vae_offload:
                        vae.cpu(); torch.cuda.empty_cache()

                    # --- Conversion finale ---

                    # conversion correcte SD
                    #frame_tensor = (frame_tensor / 2 + 0.5).clamp(0, 1)
                    frame_tensor = frame_tensor.clamp(0.0,1.0)
                    print("After decode:",
                          float(frame_tensor.min()),
                          float(frame_tensor.max()))
                    frame_array = frame_tensor[0].permute(1,2,0).cpu().numpy()  # (H,W,C)
                    save_frame(frame_array, debug_dir/f"frame_{frame_counter:05d}.png")

                    if video is None:
                        h,w = frame_array.shape[:2]
                        video_path = output_dir/"animation.mp4"
                        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

                    video.write(cv2.cvtColor((frame_array*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    writer.writerow([frame_counter,float(latent_frame.min()),float(latent_frame.max()),round(gen_time,4),round(decode_time,4)])

                    del latent_frame, frame_tensor
                    torch.cuda.empty_cache()
                    frame_counter += 1

    if video: video.release()
    print("✅ Génération ultra-stable terminée.")

# -------------------------
# Entrée
# -------------------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
