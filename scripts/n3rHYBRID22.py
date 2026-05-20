import os, time, csv
from pathlib import Path
from datetime import datetime
import torch, numpy as np
from PIL import Image
import cv2

from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae_stable, safe_load_unet, safe_load_scheduler, clamp_and_warn_tile, tile_image_vae, merge_tiles_vae, log_rgb_stats
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_images, generate_latents_ai_5D_stable

LATENT_SCALE = 0.18215
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# -------------------------
# Fonctions utilitaires
# -------------------------
def save_frame(img_array, filename):
    img_array = np.clip(img_array, 0.0, 1.0)
    img_uint8 = (img_array * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    Image.fromarray(img_uint8).save(filename)


def encode_images_to_latents(images, vae):
    """Encode images -> latents [B,4,1,H,W] float32 pour stabilité couleur"""
    vae_device = next(vae.parameters()).device
    images = images.to(device=vae_device, dtype=torch.float32)
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * LATENT_SCALE
    latents = latents.unsqueeze(2)  # [B,C,1,H,W]
    return latents

def decode_latents_to_image(latents, vae):
    """Decode latents 4D ou 5D -> RGB [0,1]"""
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    if latents.ndim == 5:
        B,C,T,H,W = latents.shape
        latents = latents.permute(0,2,1,3,4).reshape(B*T,C,H,W)
    latents = latents.to(device=vae_device, dtype=vae_dtype)
    latents = latents / LATENT_SCALE
    with torch.no_grad():
        images = vae.decode(latents).sample
    return images.clamp(0,1)



# -------------------------
# MAIN
# -------------------------
def main(args):
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    fps = cfg.get("fps",12)
    num_fraps_per_image = cfg.get("num_fraps_per_image",12)
    steps = cfg.get("steps",10)
    seed = cfg.get("seed",42)
    guidance_scale = cfg.get("guidance_scale",7.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise",0.03)
    tile_size = cfg.get("tile_size",128)
    tile_overlap = cfg.get("tile_overlap",32)

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
    unet_dtype = next(unet.parameters()).dtype
    dtype = unet_dtype  # dtype pipeline

    vae = safe_load_vae_stable(args.pretrained_model_path, device, fp16=False, offload=args.vae_offload)
    vae = vae.float()
    scheduler = safe_load_scheduler(args.pretrained_model_path)

    if not vae or not unet or not scheduler:
        print("❌ Un ou plusieurs modèles manquent.")
        return

    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None

    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to(device)
    if args.fp16:
        text_encoder = text_encoder.half()

    # -------------------------
    # Encode prompts
    # -------------------------
    embeddings = []
    for prompt_item in prompts:
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item,list) else str(prompt_item)
        neg_text = " ".join(negative_prompts) if isinstance(negative_prompts,list) else str(negative_prompts)

        text_inputs = tokenizer(prompt_text, padding="max_length", truncation=True,
                                max_length=tokenizer.model_max_length, return_tensors="pt")
        neg_inputs = tokenizer(neg_text, padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt")

        with torch.no_grad():
            pos_embeds = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
            neg_embeds = text_encoder(neg_inputs.input_ids.to(device)).last_hidden_state

        # Adapter dtype UNet
        pos_embeds = pos_embeds.to(device=device, dtype=dtype)
        neg_embeds = neg_embeds.to(device=device, dtype=dtype)

        embeddings.append((pos_embeds, neg_embeds))

    unet.eval()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except:
        pass

    # -------------------------
    # OUTPUT
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/hybrid22_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_frames"
    debug_dir.mkdir(exist_ok=True)
    csv_file = output_dir / "generation_log.csv"

    video = None
    frame_counter = 0

    with open(csv_file,"w",newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["frame","latent_min","latent_max","gen_time","decode_time","warnings"])

        for img_path in input_paths:
            input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
            tiles, positions = tile_image_vae(input_image, tile_size, tile_overlap)

            for f_idx in range(num_fraps_per_image):
                decoded_tiles = []
                all_warnings = []

                for tile_idx, tile_rgb in enumerate(tiles):
                    tile_rgb = clamp_and_warn_tile(tile_rgb, frame_counter, tile_idx, all_warnings)

                    # ---- Encode tile -> latents float32
                    tile_latent = encode_images_to_latents(tile_rgb, vae)

                    # ---- Convert pour UNet (fp16 si --fp16)
                    tile_latent = tile_latent.to(device=device, dtype=dtype)

                    # ---- Squeeze 5D -> 4D pour UNet
                    if tile_latent.ndim == 5 and tile_latent.shape[2] == 1:
                        tile_latent = tile_latent.squeeze(2)  # -> [B,C,H,W]

                    # ---- Génération latents
                    gen_start = time.time()
                    pos_embeds, neg_embeds = embeddings[0]

                    batch_latents = generate_latents_ai_5D_stable(
                        latent_frame=tile_latent,
                        scheduler=scheduler,
                        pos_embeds=pos_embeds,
                        neg_embeds=neg_embeds,
                        unet=unet,
                        motion_module=motion_module,
                        device=device,
                        dtype=dtype,
                        guidance_scale=guidance_scale,
                        creative_noise=creative_noise,
                        seed=seed + f_idx,
                        steps=steps
                    )
                    gen_time = time.time() - gen_start

                    # ---- Décodage VAE
                    decode_start = time.time()
                    if args.vae_offload:
                        vae.to(device)

                    print("Latent stats:",
                          float(batch_latents.min()),
                          float(batch_latents.max()),
                          float(batch_latents.mean()))
                    decoded_tile = decode_latents_to_image(batch_latents, vae)
                    decode_time = time.time() - decode_start
                    if args.vae_offload:
                        vae.cpu()
                        torch.cuda.empty_cache()

                    decoded_tile = clamp_and_warn_tile(decoded_tile, frame_counter, tile_idx, all_warnings)

                    # ---- Log RGB stats par tile
                    tile_warnings = log_rgb_stats(decoded_tile, step=f"frame{frame_counter}_tile{tile_idx}")
                    all_warnings.extend(tile_warnings)

                    decoded_tiles.append(decoded_tile)

                # ---- Fusion tiles
                final_frame = merge_tiles_vae(decoded_tiles, positions, H=input_image.shape[2], W=input_image.shape[3])
                final_frame = clamp_and_warn_tile(final_frame, frame_counter, "final", all_warnings)

                # ---- Log RGB stats frame finale
                frame_warnings = log_rgb_stats(final_frame, step=f"frame{frame_counter}_final")
                all_warnings.extend(frame_warnings)

                #frame_array = final_frame[0].clamp(0,1).permute(1,2,0).cpu().numpy()

                final_frame = final_frame.clamp(0.0,1.0)
                print("After decode:",
                      float(final_frame.min()),
                      float(final_frame.max()))
                frame_array = final_frame[0].permute(1,2,0).cpu().numpy()  # (H,W,C)


                save_frame(frame_array, debug_dir / f"frame_{frame_counter:05d}.png")

                if video is None:
                    h,w = frame_array.shape[:2]
                    video_path = output_dir / "animation.mp4"
                    video = cv2.VideoWriter(str(video_path),
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps,
                                            (w,h))
                video.write(cv2.cvtColor((frame_array*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                writer.writerow([
                    frame_counter,
                    float(batch_latents.min()),
                    float(batch_latents.max()),
                    round(gen_time,4),
                    round(decode_time,4),
                    "; ".join(all_warnings)
                ])
                frame_counter += 1

    if video:
        video.release()
    print("✅ Génération 128x128 VRAM-safe avec correction couleur et logging RGB terminée.")

# -------------------------
# Entrée
# -------------------------
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path",type=str,required=True)
    parser.add_argument("--config",type=str,required=True)
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument("--fp16",action="store_true")
    parser.add_argument("--vae-offload",action="store_true")
    args = parser.parse_args()
    main(args)
