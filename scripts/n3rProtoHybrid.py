# --------------------------------------------------------------
# n3rmodelSD_final.py - AnimateDiff ultra-light ~2Go VRAM
# Prompt / Input → N3RModelOptimized → MotionModule → UNet → LoRA → VAE → Image / Vidéo
#Avec use_mini_gpu et generate_latents_mini_gpu_320 → ~2,1 Go VRAM, ultra léger ✅

#Avec use_n3r_model et N3RModelOptimized → ~3,6 Go VRAM, un peu plus gourmand mais toujours raisonnable ✅
# --------------------------------------------------------------
import os, math, threading
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from PIL import ImageFilter
import argparse

from diffusers import PNDMScheduler
from transformers import CLIPTokenizerFast, CLIPTextModel

from scripts.utils.lora_utils import apply_lora_smart
from scripts.utils.vae_config import load_vae
from scripts.utils.tools_utils import ensure_4_channels
from scripts.utils.config_loader import load_config
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_safe_miniGPU, generate_latents_mini_gpu, load_images_test, generate_latents_mini_gpu_320, run_diffusion_pipeline, generate_latents_robuste_4D
from scripts.utils.fx_utils import encode_images_to_latents_nuanced, decode_latents_ultrasafe_blockwise, save_frames_as_video_from_folder, encode_images_to_latents_safe, apply_post_processing
from scripts.utils.vae_utils import safe_load_unet
from scripts.utils.n3rModelFast4Go import N3RModelFast4GB, N3RModelLazyCPU, N3RModelOptimized

LATENT_SCALE = 0.18215
stop_generation = False


# ---------------- Fusion N3R/VAE adaptative ----------------

def fuse_n3r_latents_adaptive(latents_frame, n3r_latents, latent_injection=0.7, clamp_val=1.0, creative_noise=0.0):
    n3r_latents = n3r_latents.clone()

    # Normalisation **canal par canal**
    for c in range(4):
        n3r_c = n3r_latents[:,c:c+1,:,:]
        frame_c = latents_frame[:,c:c+1,:,:]
        mean, std = n3r_c.mean(), n3r_c.std()
        n3r_c = (n3r_c - mean) / (std + 1e-6)  # centre / std
        n3r_c = n3r_c * frame_c.std() + frame_c.mean()
        n3r_latents[:,c:c+1,:,:] = n3r_c

    # Ajouter un bruit créatif léger si nécessaire
    if creative_noise > 0.0:
        noise = torch.randn_like(n3r_latents) * creative_noise
        n3r_latents += noise

    # Clamp stricte pour éviter débordement
    n3r_latents = torch.clamp(n3r_latents, -clamp_val, clamp_val)
    latents_frame = torch.clamp(latents_frame, -clamp_val, clamp_val)

    # Fusion finale
    fused_latents = latent_injection * latents_frame + (1 - latent_injection) * n3r_latents
    fused_latents = torch.clamp(fused_latents, -clamp_val, clamp_val)

    print(f"[N3R fusion frame] mean/std par canal: {fused_latents.mean(dim=(2,3))}, injection={latent_injection:.2f}")
    return fused_latents



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

def apply_motion_safe(latents, motion_module, threshold=1e-3):
    if latents.abs().max() < threshold:
        return latents, False
    return motion_module(latents), True

def adapt_embeddings_to_unet(pos_embeds, neg_embeds, target_dim):
    """Adapte automatiquement les embeddings texte pour correspondre au cross_attention_dim du UNet."""
    current_dim = pos_embeds.shape[-1]
    if current_dim == target_dim:
        return pos_embeds, neg_embeds
    # Troncature
    if current_dim > target_dim:
        pos_embeds = pos_embeds[..., :target_dim]
        neg_embeds = neg_embeds[..., :target_dim]
    # Padding
    elif current_dim < target_dim:
        pad = target_dim - current_dim
        pos_embeds = torch.nn.functional.pad(pos_embeds, (0, pad))
        neg_embeds = torch.nn.functional.pad(neg_embeds, (0, pad))
    return pos_embeds, neg_embeds

# ---------------- MAIN ----------------
# ---------------- MAIN FIABLE ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    use_mini_gpu = cfg.get("use_mini_gpu", True)
    verbose = cfg.get("verbose", False)
    latent_injection = max(0.0, min(1.0, cfg.get("latent_injection", 0.7)))
    final_latent_scale = cfg.get("final_latent_scale", 1/8) # 1/8 speed, 1/4 moyen, 1/2 low
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
    print(f"  final_latent_scale   : {final_latent_scale}")

    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps, device=device)

    # ---------------- UNET ----------------
    unet = safe_load_unet(args.pretrained_model_path, device=device, fp16=True)
    if hasattr(unet, "enable_attention_slicing"): unet.enable_attention_slicing()
    if hasattr(unet, "enable_xformers_memory_efficient_attention"):
        try: unet.enable_xformers_memory_efficient_attention(True)
        except: pass

    # ---------------- LoRA ----------------
    n3oray_models = cfg.get("n3oray_models")
    if n3oray_models:
        for model_name, lora_path in n3oray_models.items():
            applied = apply_lora_smart(unet, lora_path, alpha=0.5, device=device, verbose=verbose)
            if not applied: print(f"⚠ LoRA '{model_name}' ignorée (incompatible UNet)")
    else:
        print("⚠ Aucun modèle LoRA configuré, étape ignorée.")

    # ---------------- Motion module ----------------
    motion_module = load_motion_module(cfg.get("motion_module"), device=device) if cfg.get("motion_module") else None
    if motion_module and verbose:
        print(f"[INFO] motion_module type: {type(motion_module)}")

    # ---------------- Tokenizer / Text encoder ----------------
    tokenizer = CLIPTokenizerFast.from_pretrained(os.path.join(args.pretrained_model_path,"tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to(device).to(dtype)

    # ---------------- VAE ----------------
    vae_path = cfg.get("vae_path")
    vae, vae_type, latent_channels, LATENT_SCALE = load_vae(vae_path, device=device, dtype=dtype)

    # ---------------- Embeddings ----------------
    embeddings = []
    prompts = cfg.get("prompt", [])
    negative_prompts = cfg.get("n_prompt", [])
    unet_cross_attention_dim = getattr(unet.config, "cross_attention_dim", 1024)

    # --- Projection adaptative
    text_inputs_sample = tokenizer("test", padding="max_length", truncation=True,
                                max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        sample_embeds = text_encoder(text_inputs_sample.input_ids.to(device)).last_hidden_state
    current_dim = sample_embeds.shape[-1]
    projection = None
    if current_dim != unet_cross_attention_dim:
        projection = torch.nn.Linear(current_dim, unet_cross_attention_dim).to(device).to(dtype)

    # --- Pré-calcul des embeddings
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

        if projection is not None:
            pos_embeds = projection(pos_embeds)
            neg_embeds = projection(neg_embeds)

        embeddings.append((pos_embeds, neg_embeds))

    # ---------------- N3RModelOptimized ----------------
    use_n3r_model = cfg.get("use_n3r_model", False)
    n3r_model = None
    if use_n3r_model:
        n3r_model = N3RModelOptimized(
            L_low=cfg.get("n3r_L_low",3),
            L_high=cfg.get("n3r_L_high",6),
            N_samples=cfg.get("n3r_N_samples",32),
            tile_size=cfg.get("n3r_tile_size",64),
            cpu_offload=cfg.get("n3r_cpu_offload",True)
        ).to(device)
        n3r_model.eval()
        print(f"✅ N3RModelOptimized initialisé sur {device}")

    # ---------------- Input images ----------------
    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/ProtoHybrid_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"
    block_size = cfg.get("block_size", 160)
    overlap = compute_overlap(cfg["W"], cfg["H"], block_size)

    previous_latent_single = None
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    # ---------------- Frames principales VRAM-safe ----------------
    previous_latent_single = None
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    for img_idx, img_path in enumerate(input_paths):
        if stop_generation: break
        try:
            # Charger et encoder l'image sur GPU, puis déplacer sur CPU pour stockage
            input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
            input_image = ensure_4_channels(input_image)
            current_latent_single = encode_images_to_latents_safe(input_image, vae, device=device, latent_scale=LATENT_SCALE)
            current_latent_single = torch.nn.functional.interpolate(
                current_latent_single, size=(cfg["H"]//8, cfg["W"]//8),
                mode='bilinear', align_corners=False
            )

            # ------------------- NOUVEAU -------------------
            # Génération robuste des latents initiaux (optionnel, steps réduits pour VRAM)
            try:
                current_latent_single = generate_latents_robuste_4D(
                    latents=current_latent_single.to(device),  # <-- nom correct
                    pos_embeds=None,  # pas encore de prompt, ou embeddings neutres
                    neg_embeds=None,
                    unet=unet,
                    scheduler=scheduler,
                    motion_module=None,
                    device=device,
                    dtype=dtype,
                    guidance_scale=1.5,   # plus safe pour init
                    init_image_scale=0.85,  # pas 1.0 pour ne pas amplifier NaN
                    creative_noise=0.0,   # pour init, on peut désactiver
                    seed=42
                )
            except Exception as e:
                print(f"[Robuste INIT ERROR] {e}")
            # -----------------------------------------------
            current_latent_single = ensure_4_channels(current_latent_single)
            # Déplacer sur CPU dès que possible
            current_latent_single = current_latent_single.to('cpu')
            del input_image
            torch.cuda.empty_cache()

            # ---------------- Transition frames ----------------
            if previous_latent_single is not None and transition_frames > 0:
                for t in range(transition_frames):
                    if stop_generation: break
                    alpha = 0.5 - 0.5*math.cos(math.pi*t/max(transition_frames-1,1))

                    # Interpolation sur GPU pour calcul
                    with torch.no_grad():
                        latent_interp = (1-alpha)*previous_latent_single.to(device) + alpha*current_latent_single.to(device)
                        latent_interp = torch.clamp(latent_interp, -1.0, 1.0).contiguous()

                        if motion_module:
                            latent_interp, _ = apply_motion_safe(latent_interp, motion_module)

                        # Resize pour VAE
                        final_H, final_W = int(cfg["H"]*final_latent_scale), int(cfg["W"]*final_latent_scale)
                        if latent_interp.shape[-2:] != (final_H, final_W):
                            latent_interp = torch.nn.functional.interpolate(
                                latent_interp, size=(final_H, final_W),
                                mode='bilinear', align_corners=False
                            ).contiguous()

                        # Décodage en streaming vers CPU
                        frame_pil = decode_latents_ultrasafe_blockwise(
                            latent_interp, vae,
                            block_size=block_size, overlap=overlap,
                            gamma=1.0, brightness=1.0,
                            contrast=1.5, saturation=1.3,
                            device=device,  # GPU uniquement pour blocs, frame final vers CPU
                            frame_counter=frame_counter,
                            latent_scale_boost=latent_scale_boost
                        )

                        frame_pil = apply_post_processing(frame_pil, blur_radius=0.2)
                        frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                        frame_counter += 1
                        pbar.update(1)

                    del latent_interp
                    torch.cuda.empty_cache()

            # ---------------- Frames principales ----------------
            for pos_embeds, neg_embeds in embeddings:
                for f in range(num_fraps_per_image):
                    if stop_generation: break

                    with torch.no_grad():
                        # Latent de base sur GPU pour calcul
                        latents_frame = current_latent_single.to(device)
                        cf_embeds = (pos_embeds.to(device), neg_embeds.to(device))
                        latents = latents_frame.clone()
                        n3r_latents = None

                        # --- N3R ---
                        if use_n3r_model:
                            try:
                                H, W = cfg["H"], cfg["W"]
                                ys, xs, ss = torch.meshgrid(
                                    torch.arange(H, device=device),
                                    torch.arange(W, device=device),
                                    torch.arange(n3r_model.N_samples, device=device),
                                    indexing='ij'
                                )
                                coords = torch.stack([xs, ys, ss.float()], dim=-1).reshape(-1,3).float()
                                n3r_latents_raw = n3r_model(coords, H, W)[:, :3]
                                n3r_latents = n3r_latents_raw.view(H, W, n3r_model.N_samples, 3).mean(dim=2)
                                n3r_latents = n3r_latents.permute(2,0,1).unsqueeze(0)
                                if n3r_latents.shape[1] == 3:
                                    n3r_latents = torch.cat([n3r_latents, torch.zeros_like(n3r_latents[:, :1, :, :])], dim=1)
                                # Resize et clamp
                                target_H, target_W = latents.shape[-2], latents.shape[-1]
                                if n3r_latents.shape[-2:] != (target_H, target_W):
                                    n3r_latents = torch.nn.functional.interpolate(
                                        n3r_latents, size=(target_H, target_W),
                                        mode='bilinear', align_corners=False
                                    ).contiguous()
                                n3r_latents = torch.clamp(n3r_latents, -1.0, 1.0)
                                latents = fuse_n3r_latents_adaptive(latents, n3r_latents, latent_injection=latent_injection)
                            except Exception as e:
                                print(f"[N3R ERROR] {e}")

                        # --- Mini GPU diffusion ---
                        elif use_mini_gpu:
                            latents = generate_latents_mini_gpu_320(
                                unet=unet, scheduler=scheduler,
                                input_latents=latents_frame, embeddings=cf_embeds,
                                motion_module=motion_module, guidance_scale=guidance_scale,
                                device=device, fp16=True, steps=steps,
                                debug=verbose, init_image_scale=init_image_scale,
                                creative_noise=creative_noise
                            )
                            if latent_injection > 0:
                                # ⚡ Adapter la taille avant fusion
                                if latents.shape[-2:] != latents_frame.shape[-2:]:
                                    latents = torch.nn.functional.interpolate(
                                        latents,
                                        size=latents_frame.shape[-2:],
                                        mode='bilinear',
                                        align_corners=False
                                    ).contiguous()
                                latents = latent_injection*latents_frame + (1-latent_injection)*latents

                        # Motion module
                        if motion_module:
                            latents, _ = apply_motion_safe(latents, motion_module)

                        # Clamp et resize final
                        latents = torch.clamp(latents, -1.0, 1.0)
                        final_H, final_W = int(cfg["H"]*final_latent_scale), int(cfg["W"]*final_latent_scale)
                        if latents.shape[-2:] != (final_H, final_W):
                            latents = torch.nn.functional.interpolate(latents, size=(final_H, final_W),
                                                                    mode='bilinear', align_corners=False).contiguous()

                        # Décodage streaming vers CPU
                        frame_pil = decode_latents_ultrasafe_blockwise(
                            latents, vae,
                            block_size=block_size, overlap=overlap,
                            gamma=1.0, brightness=1.0,
                            contrast=1.5, saturation=1.3,
                            device=device,
                            frame_counter=frame_counter,
                            latent_scale_boost=latent_scale_boost
                        )
                        frame_pil = apply_post_processing(frame_pil, blur_radius=0.05, contrast=1.15, brightness=1.05, saturation=0.85, sharpen=True, sharpen_radius=1, sharpen_percent=90, sharpen_threshold=2)
                        frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                        frame_counter += 1
                        pbar.update(1)

                        # Nettoyage VRAM
                        del latents, latents_frame, cf_embeds, n3r_latents
                        torch.cuda.empty_cache()

            previous_latent_single = current_latent_single  # reste sur CPU

        except Exception as e:
            print(f"[FRAME ERROR] {img_path} : {e}")
            continue

    pbar.close()
    save_frames_as_video_from_folder(output_dir, out_video, fps=fps, upscale_factor=upscale_factor)
    print(f"🎬 Vidéo générée : {out_video}")
    print("✅ Pipeline terminé avec motion module safe et N3RModelOptimized.")

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
