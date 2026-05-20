# --------------------------------------------------------------
# n3rmodelFast.py - AnimateDiff ultra-light ~2Go VRAM
# --------------------------------------------------------------
import os, math, threading
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from PIL import ImageFilter
from torchvision.transforms import ToPILImage
import argparse

from diffusers import PNDMScheduler
from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.lora_utils import apply_lora_smart
from scripts.utils.vae_config import load_vae
from scripts.utils.tools_utils import ensure_4_channels
from scripts.utils.config_loader import load_config
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_safe_miniGPU, generate_latents_mini_gpu, load_images_test, generate_latents_mini_gpu_320, run_diffusion_pipeline
from scripts.utils.fx_utils import encode_images_to_latents_nuanced, decode_latents_ultrasafe_blockwise, save_frames_as_video_from_folder, encode_images_to_latents_safe
from scripts.utils.vae_utils import safe_load_unet

LATENT_SCALE = 0.18215
stop_generation = False


# ---------------- DEBUG UTILS ----------------
def log_debug(message, level="INFO", verbose=True):
    """
    Affiche le message si verbose=True.
    level: "INFO", "DEBUG", "WARNING"
    """
    if verbose:
        print(f"[{level}] {message}")

# ---------------- Thread stop ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²":
        stop_generation = True
threading.Thread(target=wait_for_stop, daemon=True).start()

# ---------------- Utilitaires ----------------

def compute_overlap(W, H, block_size, max_overlap_ratio=0.6):
    overlap = int(block_size * max_overlap_ratio)
    return min(overlap, min(W,H)//4)

def apply_motion_safe(latents, motion_module, threshold=1e-2):
    if latents.abs().max() < threshold:
        return latents, False
    return motion_module(latents), True

# ---------------- MAIN ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    use_mini_gpu = cfg.get("use_mini_gpu", True) # True for <2 Go VRAM - False for <4 Go VRAM
    verbose = cfg.get("verbose", False) # True or False
    # ---------------- Injection contrôlée du latent original ----------------
    # latent_injection = 0.0 -> pas d'influence de l'image originale
    # latent_injection = 1.0 -> on garde totalement le latent d'origine
    latent_injection = max(0.0, min(1.0, cfg.get("latent_injection", 0.7))) # implication de l'image original dans le resultat final'
    final_latent_scale = cfg.get("final_latent_scale", 1/8) #1/2  de l'image,  1/4 de l'image, 0.125 pour 1/8, etc.


    fps = cfg.get("fps", 12)
    upscale_factor = cfg.get("upscale_factor", 1)
    transition_frames = cfg.get("transition_frames", 4)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 2)
    steps = max(cfg.get("steps", 16), 4)
    guidance_scale = cfg.get("guidance_scale", 4.5)
    init_image_scale = cfg.get("init_image_scale", 0.85)
    creative_noise = cfg.get("creative_noise", 0.0)
    latent_scale_boost = cfg.get("latent_scale_boost", 5.71)


    print("📌 Paramètres de génération :")
    print(f"  fps                  : {fps}")
    print(f"  upscale_factor       : {upscale_factor}")
    print(f"  num_fraps_per_image  : {num_fraps_per_image}")
    print(f"  steps                : {steps}")
    print(f"  guidance_scale       : {guidance_scale}")
    print(f"  init_image_scale     : {init_image_scale}")
    print(f"  creative_noise       : {creative_noise}")
    print(f"  latent_scale_boost   : {latent_scale_boost}")

    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps, device=device)

    # ---------------- UNET ----------------
    unet = safe_load_unet(args.pretrained_model_path, device=device, fp16=True)
    if hasattr(unet, "enable_attention_slicing"): unet.enable_attention_slicing()
    if hasattr(unet, "enable_xformers_memory_efficient_attention"):
        try: unet.enable_xformers_memory_efficient_attention(True)
        except: pass

    # ---------------- LoRA ----------------
    unet_cross_attention_dim = getattr(unet.config, "cross_attention_dim", 768)
    n3oray_models = cfg.get("n3oray_models")
    if n3oray_models:
        for model_name, lora_path in n3oray_models.items():
            applied = apply_lora_smart(unet, lora_path, alpha=0.5, device=device, verbose=verbose)
            if not applied:
                print(f"⚠ LoRA '{model_name}' ignorée (incompatible UNet)")
    else:
        print("⚠ Aucun modèle LoRA n'est configuré, étape ignorée.")
    # ---------------- Motion module ----------------
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None
    #motion_module = None
    if motion_module is not None:
        log_debug(f"motion_module type: {type(motion_module)}", level="INFO", verbose=cfg.get("verbose", True))


    # ---------------- Tokenizer / Text encoder ----------------
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to(device).to(dtype)

    # ---------------- VAE ----------------
    vae_path = cfg.get("vae_path")
    vae, vae_type, latent_channels, LATENT_SCALE = load_vae(vae_path, device=device, dtype=dtype)

    # ---------------- Embeddings ----------------
    # ---------------- Embeddings ----------------
    embeddings = []
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])

    # Récupération du cross_attention_dim attendu par le UNet
    unet_cross_attention_dim = getattr(unet.config, "cross_attention_dim", 1024)

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

        # ---------------- CORRECTION DES DIMENSIONS ----------------
        current_dim = pos_embeds.shape[-1]
        if current_dim != unet_cross_attention_dim:
            # Projection linéaire 768 -> 1024
            projection = torch.nn.Linear(current_dim, unet_cross_attention_dim).to(device).to(dtype)
            pos_embeds = projection(pos_embeds)
            neg_embeds = projection(neg_embeds)

        embeddings.append((pos_embeds, neg_embeds))

    print(f"✅ Embeddings adaptées à UNet cross_attention_dim={unet_cross_attention_dim}")

    # ---------------- DEBUG DIMENSIONS ----------------
    print("\n🔍 Vérification des dimensions avant génération")
    for i, (pos, neg) in enumerate(embeddings):
        print(f"Embedding {i}: pos {pos.shape}, neg {neg.shape}")
        if pos.shape[-1] != unet_cross_attention_dim:
            log_debug(f"Attention : pos_embedding dim {pos.shape[-1]} != UNet {unet_cross_attention_dim}", level="WARNING", verbose=verbose)
        if neg.shape[-1] != unet_cross_attention_dim:
            print(f"⚠ Attention : neg_embedding dim {neg.shape[-1]} != UNet {unet_cross_attention_dim}")

    print(f"UNet cross_attention_dim attendu : {unet_cross_attention_dim}")
    print("✅ Toutes les dimensions semblent correctes\n")


    # ---------------- Input images ----------------
    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/modelFast_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    print(f"📌 fps: {fps}, frames/image: {num_fraps_per_image}, steps: {steps}, guidance_scale: {guidance_scale}")

    block_size = cfg.get("block_size", 64)
    overlap = compute_overlap(cfg["W"], cfg["H"], block_size)

    previous_latent_single = None
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    #to_pil = ToPILImage()

    for img_idx, img_path in enumerate(input_paths):
        if stop_generation: break

        # Charger et normaliser l'image
        input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
        input_image = ensure_4_channels(input_image)

        # Encoder l'image en latents
        current_latent_single = encode_images_to_latents_safe(input_image, vae, device=device, latent_scale=LATENT_SCALE)

        log_debug(f"Latents shape after encoding: {current_latent_single.shape}", level="DEBUG", verbose=verbose)


        # ---------------- Ajuster la taille des latents pour UNet ----------------
        # UNet.sample_size correspond à la taille d'entrée attendue par le modèle (ex: 320 ou 512)
        target_H = getattr(unet.config, "sample_size", cfg["H"]) // 8
        target_W = getattr(unet.config, "sample_size", cfg["W"]) // 8

        # Interpolation bilinéaire pour correspondre à UNet
        current_latent_single = torch.nn.functional.interpolate(
            current_latent_single, size=(target_H, target_W), mode='bilinear', align_corners=False
        )

        # Assurer 4 channels (sécurité)
        current_latent_single = ensure_4_channels(current_latent_single)

        log_debug(f"DEBUG latents shape after interpolation: {current_latent_single.shape}", level="DEBUG", verbose=verbose)

        # ---------------- Transition frames ----------------
        if previous_latent_single is not None and transition_frames > 0:
            for t in range(transition_frames):
                if stop_generation: break
                alpha = 0.5 - 0.5*math.cos(math.pi*t/max(transition_frames-1,1))
                latent_interp = (1-alpha)*previous_latent_single + alpha*current_latent_single
                if motion_module:
                    latent_interp, _ = apply_motion_safe(latent_interp, motion_module)

                # ⚡ Forcer la taille finale (VRAM-safe), exactement comme les autres latents
                final_latent_H = int(cfg["H"] * final_latent_scale)
                final_latent_W = int(cfg["W"] * final_latent_scale)
                if latent_interp.shape[-2:] != (final_latent_H, final_latent_W):
                    latent_interp = torch.nn.functional.interpolate(
                        latent_interp,
                        size=(final_latent_H, final_latent_W),
                        mode='bilinear',
                        align_corners=False
                    )
                frame_pil = decode_latents_ultrasafe_blockwise(latent_interp, vae,
                                                               block_size=block_size, overlap=overlap,
                                                               gamma=1.0, brightness=1.0,
                                                               contrast=1.5, saturation=1.3,
                                                               device=device, frame_counter=frame_counter,
                                                               latent_scale_boost=latent_scale_boost)
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)

        # ---------------- Frames principales ----------------
        for pos_embeds, neg_embeds in embeddings:
            for f in range(num_fraps_per_image):
                if stop_generation: break
                if f == 0:
                    # Frame initiale = image d'entrée
                    frame_tensor = torch.clamp((input_image.squeeze(0)+1)/2, 0, 1)

                    # Upscale proportionnel à final_latent_scale
                    # On multiplie par final_latent_scale > 0 pour correspondre à la taille des latents
                    # Si final_latent_scale <1 → on agrandit légèrement, si >1 → on réduit légèrement
                    upscale_H = int(frame_tensor.shape[-2] * final_latent_scale * 8)  # 8 = facteur latent->image
                    upscale_W = int(frame_tensor.shape[-1] * final_latent_scale * 8)
                    frame_tensor = torch.nn.functional.interpolate(
                        frame_tensor.unsqueeze(0),
                        size=(upscale_H, upscale_W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                    frame_pil = to_pil_image(frame_tensor)
                else:
                    latents_frame = current_latent_single.clone()

                    # ---------------- Redimension latents selon final_latent_scale avant UNet ----------------
                    target_H = int(latents_frame.shape[-2] * final_latent_scale)
                    target_W = int(latents_frame.shape[-1] * final_latent_scale)
                    if latents_frame.shape[-2:] != (target_H, target_W):
                        latents_frame = torch.nn.functional.interpolate(
                            latents_frame,
                            size=(target_H, target_W),
                            mode='bilinear',
                            align_corners=False
                        )

                    # Assurer 4 canaux
                    latents_frame = ensure_4_channels(latents_frame)

                    # ---------------- Génération latents ----------------
                    cf_embeds = (pos_embeds.to(device), neg_embeds.to(device))
                    if use_mini_gpu:
                        latents = generate_latents_mini_gpu_320(
                            unet=unet,
                            scheduler=scheduler,
                            input_latents=latents_frame,
                            embeddings=cf_embeds,
                            motion_module=motion_module,
                            guidance_scale=guidance_scale,
                            device=device,
                            fp16=True,
                            steps=steps,
                            debug=verbose,
                            init_image_scale=init_image_scale,   # <-- ajouté
                            creative_noise=creative_noise        # <-- ajouté
                        )
                        # Injection contrôlée du latent d'entrée depuis le cfg
                        if latent_injection > 0.0:
                            latents = latent_injection * current_latent_single + (1 - latent_injection) * latents
                    else:
                        latents_input = latents_frame[:, :3, :, :]
                        latents = run_diffusion_pipeline(
                            unet=unet,
                            vae=vae,
                            scheduler=scheduler,
                            images=latents_input,
                            embeddings=cf_embeds,
                            timesteps=scheduler.timesteps,
                            device=device
                        )

                    # ---------------- Motion Module ----------------
                    if motion_module:
                        latents, _ = apply_motion_safe(latents, motion_module)

                    # ---------------- Interpolation vers latents finaux (VRAM-safe) ----------------
                    final_latent_H = int(cfg["H"] * final_latent_scale)
                    final_latent_W = int(cfg["W"] * final_latent_scale)
                    if latents.shape[-2:] != (final_latent_H, final_latent_W):
                        latents = torch.nn.functional.interpolate(
                            latents,
                            size=(final_latent_H, final_latent_W),
                            mode='bilinear',
                            align_corners=False
                        )

                    # ---------------- Décodage bloc par bloc ----------------
                    frame_pil = decode_latents_ultrasafe_blockwise(
                        latents, vae,
                        block_size=block_size,
                        overlap=overlap,
                        gamma=1.0,
                        brightness=1.0,
                        contrast=1.5,
                        saturation=1.3,
                        device=device,
                        frame_counter=frame_counter,
                        latent_scale_boost=latent_scale_boost
                    )

                    del latents
                    torch.cuda.empty_cache()

                    # Appliquer un flou léger sur toute l'image
                    frame_pil = frame_pil.filter(ImageFilter.GaussianBlur(radius=0.2))

                # ---------------- Sauvegarde ----------------
                frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                frame_counter += 1
                pbar.update(1)


        previous_latent_single = current_latent_single

    pbar.close()
    save_frames_as_video_from_folder(output_dir, out_video, fps=fps, upscale_factor=2)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé avec motion module safe.")

# ---------------- ENTRY ----------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--vae-offload", action="store_true")
    args = parser.parse_args()
    main(args)
