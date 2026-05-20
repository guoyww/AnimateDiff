# --------------------------------------------------------------
# nr3perfect - INTERPOLATION fast movie - Multi-Model n3oray (VAE séparé)
# --------------------------------------------------------------
import os
import torch
import argparse
from pathlib import Path
from datetime import datetime
import math
from tqdm import tqdm
from PIL import Image, ImageFilter
from torchvision.transforms import ToPILImage
import threading

from diffusers import AutoencoderKL
from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_unet, safe_load_scheduler
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_robuste, load_image_file, decode_latents_to_image_auto
from safetensors.torch import load_file

LATENT_SCALE = 0.18215
stop_generation = False

# ---------------- Thread pour stopper la génération ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²":
        stop_generation = True

threading.Thread(target=wait_for_stop, daemon=True).start()

# ---------------- Fonctions utilitaires ----------------
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

def save_frames_as_video_from_folder(folder_path, output_path, fps=12):
    import ffmpeg
    folder_path = Path(folder_path)
    frame_files = sorted(folder_path.glob("frame_*.png"))
    if not frame_files:
        print("❌ Aucun frame trouvé dans le dossier")
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
        latents = latents.unsqueeze(2)  # [B, C, 1, H/8, W/8]
    return latents

# ---------------- MAIN ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)

    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 else torch.float32

    fps = cfg.get("fps", 12)
    upscale_factor = cfg.get("upscale_factor", 2)
    transition_frames = cfg.get("transition_frames", 8)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 12)
    steps = cfg.get("steps", 50)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise", 0.0)
    vae_path = cfg.get("vae_path", "/mnt/62G/huggingface/vae/vae-ft-mse-840000-ema-pruned.safetensors")
    # Nom du modèle choisi
    n3_model_name = args.n3_model  # exemple: "cyber_skin"

    # Chemin du modèle depuis le YAML
    n3_model_path = cfg["n3oray_models"].get(n3_model_name)
    if n3_model_path is None:
        raise ValueError(f"Le modèle N3 '{n3_model_name}' n'est pas défini dans le YAML")
    print(f"✅ Chargement du modèle N3 '{n3_model_name}' depuis : {n3_model_path}")

    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    total_frames = (
        len(input_paths) * num_fraps_per_image * max(len(prompts), 1)
        + max(len(input_paths) - 1, 0) * transition_frames
    )
    print(f"🎞 Frames totales estimées : {total_frames}")
    print("⏹ Touche '²' pour arrêter la génération et création de la vidéo directement...")

    # ---------------- LOAD MODELS ----------------
    # Créer UNET vide avec la config standard
    unet = safe_load_unet(args.pretrained_model_path, device=device, fp16=args.fp16)

    # Charger le safetensors correspondant au N3 choisi
    state_dict = load_file(n3_model_path, device=device)
    unet.load_state_dict(state_dict, strict=False)
    print(f"✅ UNET N3 '{n3_model_name}' chargé correctement")

    scheduler = safe_load_scheduler(args.pretrained_model_path)
    scheduler.set_timesteps(steps, device=device)
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None

    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path, "text_encoder")).to(device)
    if args.fp16:
        text_encoder = text_encoder.half()

    # ---------------- LOAD SEPARATE VAE ----------------
    device = "cuda"

    # 1️⃣ Crée le modèle VAE vide correspondant
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"]*4,
        up_block_types=["UpDecoderBlock2D"]*4,
        block_out_channels=[128, 256, 512, 512],
        latent_channels=4,
        layers_per_block=2,
        sample_size=256
    )

    # 2️⃣ Charge les poids safetensors via safetensors
    state_dict = load_file(vae_path, device=device)
    vae.load_state_dict(state_dict, strict=False)

    #vae = vae.to(device)
    offload=True
    vae = vae.to("cpu" if offload else device).float()
    #vae.enable_tiling()
    vae.enable_slicing()

    print(f"✅ VAE safetensors chargé depuis : {vae_path}")

    # ---------------- PROMPT EMBEDDINGS ----------------
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

    # ---------------- OUTPUT ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/fastperfect_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    to_pil = ToPILImage()
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)
    previous_latent_single = None
    stop_generation = False

    # ================= MAIN LOOP =================
    for img_idx, img_path in enumerate(input_paths):
        if stop_generation:
            break

        input_image = load_images([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        input_latents_single = encode_images_to_latents(input_image, vae)
        input_latents = input_latents_single.repeat(1, 1, num_fraps_per_image, 1, 1)
        current_latent_single = input_latents_single.clone()
        block_size = cfg.get("block_size", 64)
        overlap = cfg.get("overlap", compute_overlap(cfg["W"], cfg["H"], block_size))

        # --- Transition latente ---
        if previous_latent_single is not None and transition_frames > 0:
            for t in range(transition_frames):
                if stop_generation:
                    print("⏹ Arrêt demandé, création de la vidéo...")
                    break
                alpha = 0.5 - 0.5 * math.cos(math.pi * t / (transition_frames - 1))
                latent_interp = ((1 - alpha) * previous_latent_single + alpha * current_latent_single).squeeze(2).clamp(-3.0, 3.0)
                frame_tensor = decode_latents_to_image_auto(latent_interp, vae)
                frame_tensor = normalize_frame(frame_tensor)
                if frame_tensor.ndim == 4:
                    frame_tensor = frame_tensor.squeeze(0)
                frame_pil = to_pil(frame_tensor.cpu()).filter(ImageFilter.GaussianBlur(radius=0.2))
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize((frame_pil.width * upscale_factor, frame_pil.height * upscale_factor), Image.BICUBIC)
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        # --- Boucle principale fraps/prompts (sans mélange dynamique de styles) ---
        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                if f == 0:
                    frame_tensor = (input_image.squeeze(0) + 1.0) / 2.0
                    frame_tensor = frame_tensor.clamp(0, 1)
                else:
                    latents_frame = input_latents[:, :, f:f+1, :, :].clone()
                    latents_frame = generate_latents_robuste(
                        latents_frame, pos_embeds, neg_embeds,
                        unet, scheduler, motion_module, device, dtype,
                        guidance_scale, init_image_scale, creative_noise, seed=frame_counter
                    )
                    frame_tensor = decode_latents_to_image_auto(latents_frame, vae)
                    frame_tensor = normalize_frame(frame_tensor)
                    if frame_tensor.ndim == 4:
                        frame_tensor = frame_tensor.squeeze(0)

                frame_pil = to_pil(frame_tensor.cpu()).filter(ImageFilter.GaussianBlur(radius=0.2))
                if upscale_factor > 1:
                    frame_pil = frame_pil.resize((frame_pil.width * upscale_factor, frame_pil.height * upscale_factor), Image.BICUBIC)
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")

                del frame_tensor, frame_pil
                if f != 0:
                    del latents_frame
                frame_counter += 1
                if frame_counter % 10 == 0:
                    torch.cuda.empty_cache()
                pbar.update(1)

        previous_latent_single = current_latent_single.clone()

    pbar.close()
    save_frames_as_video_from_folder(output_dir, out_video, fps=fps)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé proprement.")

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    parser.add_argument("--n3_model", type=str, default="cyberpunk_style_v3")
    args = parser.parse_args()
    main(args)
