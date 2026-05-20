# --------------------------------------------------------------
# n3rProBoost.py - AnimateDiff ultra-light ~2Go VRAM
# Prompt / Input → N3RModelOptimized → MotionModule → UNet → LoRA → VAE → Image / Vidéo
#Avec use_mini_gpu et generate_latents_mini_gpu_320 → ~2,1 Go VRAM, ultra léger ✅ Avec use_n3r_model et N3RModelOptimized → ~3,6 Go VRAM
# --------------------------------------------------------------
import os, math, threading, random
import traceback
import hashlib
import torch
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageFilter
import argparse
from diffusers import PNDMScheduler
from transformers import CLIPTokenizerFast, CLIPTextModel
from scripts.utils.lora_utils import apply_lora_smart
from scripts.utils.vae_config import load_vae
from scripts.utils.n3rModelUtils import generate_n3r_coords, process_n3r_latents, fuse_with_memory, inject_external, fuse_n3r_latents_adaptive_new
from scripts.utils.tools_utils import ensure_4_channels, print_generation_params, sanitize_latents, stabilize_latents_advanced, log_debug, compute_overlap, get_interpolated_embeddings, save_memory, load_memory, load_external_embedding_as_latent, inject_external_embeddings, update_n3r_memory, compute_weighted_params, adapt_embeddings_to_unet, get_dynamic_latent_injection, save_input_frame
from scripts.utils.config_loader import load_config
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_images_test, generate_latents_mini_gpu_320, run_diffusion_pipeline, generate_latents_robuste_4D
from scripts.utils.fx_utils import encode_images_to_latents_nuanced, decode_latents_ultrasafe_blockwise, adaptive_post_process, save_frames_as_video_from_folder, encode_images_to_latents_safe, apply_post_processing_adaptive, encode_images_to_latents_hybrid, interpolate_param_fast, fuse_n3r_latents_adaptive, adaptive_post_process, remove_white_noise, apply_post_processing

from scripts.utils.vae_utils import safe_load_unet
from scripts.utils.n3rModelFast4Go import N3RModelFast4GB, N3RModelLazyCPU, N3RModelOptimized
from scripts.utils.n3rProNet import N3RProNet
from scripts.utils.n3rProNet_utils import apply_n3r_pro_net, soft_tone_map

LATENT_SCALE = 0.18215
stop_generation = False

# Variation de l'interpolation' Valeurs de départ (fidèles à l'image)-----------------------interpolate_param_fast ---
#init_image_scale_start = 0.95 #guidance_scale_start   = 1.5 #creative_noise_start   = 0.0
# ---------------- Thread stop ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²":
        stop_generation = True
threading.Thread(target=wait_for_stop, daemon=True).start()

# ---------------- Utilitaires ----------------
def apply_motion_safe(latents, motion_module, threshold=1e-3):
    if latents.abs().max() < threshold:
        return latents, False
    return motion_module(latents), True

# ---------------- MAIN FIABLE ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    use_mini_gpu = cfg.get("use_mini_gpu", True)
    verbose = cfg.get("verbose", False)
    latent_injection = float(cfg.get("latent_injection", 0.75))
    latent_injection = min(max(latent_injection, 0.5), 0.9)  # plage sûre
    final_latent_scale = cfg.get("final_latent_scale", 1/8) # 1/8 speed, 1/4 moyen, 1/2 low
    fps = cfg.get("fps", 12)
    upscale_factor = cfg.get("upscale_factor", 1)
    transition_frames = cfg.get("transition_frames", 4)
    num_fraps_per_image = cfg.get("num_fraps_per_image", 2)
    steps = max(cfg.get("steps", 16), 4)
    guidance_scale = cfg.get("guidance_scale", 6.5) # 0.15 peut de créativité 4.5 moderé
    guidance_scale_end = cfg.get("guidance_scale_end", 7.0) # 0.15 peut de créativité 4.5 moderé
    init_image_scale = cfg.get("init_image_scale", 0.75) # 0.85 ou 0.95 proche de l'init' (0.75)
    init_image_scale_end = cfg.get("init_image_scale_end", 0.9) # 0.85 ou 0.95 proche de l'init'
    creative_noise = cfg.get("creative_noise", 0.0)
    creative_noise_end = cfg.get("creative_noise_end", 0.08)
    latent_scale_boost = cfg.get("latent_scale_boost", 1.0)
    frames_per_prompt = cfg.get("frames_per_prompt", 10)  # nombre de frames par prompt
    contrast = cfg.get("contrast", 1.15)  # Post Traitement constrat
    saturation = cfg.get("saturation", 1.00)  # Post Traitement saturation
    blur_radius = cfg.get("blur_radius", 0.03)  # Post Traitement blur
    sharpen_percent = cfg.get("sharpen_percent", 90)  #Post Traitement sharpen
    H, W = cfg.get("H", 512), cfg.get("W", 512)
    block_size = min(256, H//2, W//2)  # block_size auto selon résolution
    use_n3r_model = cfg.get("use_n3r_model", False)

    # Seed aléatoire
    seed = torch.randint(0, 100000, (1,)).item()
    params = { 'use_mini_gpu': use_mini_gpu,  'fps': fps, 'upscale_factor': upscale_factor, 'num_fraps_per_image': num_fraps_per_image, 'steps': steps, 'guidance_scale': guidance_scale, 'guidance_scale_end': guidance_scale_end, 'init_image_scale': init_image_scale, 'init_image_scale_end': init_image_scale_end, 'creative_noise': creative_noise, 'creative_noise_end': creative_noise_end, 'latent_scale_boost': latent_scale_boost, 'final_latent_scale': final_latent_scale, 'seed': seed, 'latent_injection': latent_injection, 'transition_frames': transition_frames, 'block_size': block_size, 'use_n3r_model': use_n3r_model }
    print_generation_params(params)


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
    #iniy external_latent
    external_latent = None
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

    # --- Pré-calcul des embeddings pour interpolation
    pos_embeds_list = []
    neg_embeds_list = []

    # Si prompts et n_prompts sont des listes de listes ou chaînes
    for i, prompt_item in enumerate(prompts):
        # Texte positif
        prompt_text = " ".join(prompt_item) if isinstance(prompt_item, list) else str(prompt_item)
        # Texte négatif correspondant
        neg_text_item = negative_prompts[i] if i < len(negative_prompts) else negative_prompts[0]
        neg_text = " ".join(neg_text_item) if isinstance(neg_text_item, list) else str(neg_text_item)

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

        # Ajouter à la liste complète
        pos_embeds_list.append(pos_embeds)
        neg_embeds_list.append(neg_embeds)

    # ---------------- N3RModelOptimized ----------------
    n3r_model = None
    if use_n3r_model:
        n3r_model = N3RModelOptimized(
            L_low=cfg.get("n3r_L_low",3), # 3 ou 4 # plutôt que 3, un peu plus de finesse
            L_high=cfg.get("n3r_L_high",6), # garde structure globale
            N_samples=cfg.get("n3r_N_samples",32), # plus de samples pour un rendu détaillé 48
            tile_size=cfg.get("n3r_tile_size",64), # inchangé pour VRAM raisonnable
            cpu_offload=cfg.get("n3r_cpu_offload",True)
        ).to(device)
        n3r_model.eval()
        print(f"✅ N3RModelOptimized initialisé sur {device}")

        # ------------------- Initialisation mémoire -------------------
        output_dir_m = Path("./outputs")
        memory_file = output_dir_m / "n3r_memory"
        memory_dict = load_memory(memory_file)

    # Configurable depuis ton fichier cfg
    use_n3r_pro_net = cfg.get("use_n3r_pro_net", True)
    n3r_pro_strength = cfg.get("n3r_pro_strength", 0.3)

    n3r_pro_net = None
    if use_n3r_pro_net:
        n3r_pro_net = N3RProNet(channels=4).to(device).to(dtype)
        n3r_pro_net.eval()
        print("✅ N3RProNet activé")


    # ---------------- Input images ----------------
    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/ProBoost{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    overlap = compute_overlap(cfg["W"], cfg["H"], block_size)

    previous_latent_single = None
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    # ---------------- Frames principales avec interpolation prompts ----------------
    external_embeddings = None

    # Charger latent externe avant la génération
    external_path = "/mnt/62G/huggingface/cyber-fp16/pt/KnxCOmiXNeg.safetensors"
    external_latent = load_external_embedding_as_latent(
        external_path, (1, 4, cfg["H"]//8, cfg["W"]//8)
    ).to(device)
    #------------------------------------------------------------------------------
    for img_idx, img_path in enumerate(input_paths):
        if stop_generation: break
        try:
            # Paramètres interpolés
            current_init_image_scale, current_creative_noise, current_guidance_scale = compute_weighted_params( frame_counter, total_frames, init_start=0.85, init_end=0.5,noise_start=0.0, noise_end=0.08, guidance_start=3.5, guidance_end=4.5, mode="cosine" )
            print(f"[Frame {frame_counter:03d}] " f"init_image_scale={current_init_image_scale:.3f}, " f"guidance_scale={current_guidance_scale:.3f}, " f"creative_noise={current_creative_noise:.3f}")

            # Charger et encoder l'image sur GPU
            input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
            input_image = ensure_4_channels(input_image)
            frame_counter = save_input_frame( input_image, output_dir, frame_counter, pbar=pbar, blur_radius=blur_radius, contrast=contrast, saturation=saturation, apply_post=False )

            current_latent_single = encode_images_to_latents_hybrid(input_image, vae, device=device, latent_scale=LATENT_SCALE)
            current_latent_single = torch.nn.functional.interpolate(
                current_latent_single, size=(cfg["H"]//8, cfg["W"]//8),
                #current_latent_single, size=(cfg["H"]//6, cfg["W"]//6),
                mode='bilinear', align_corners=False
            )

            # 🔥 FIX NaN / stabilité
            current_latent_single = sanitize_latents(current_latent_single)

            # Génération initiale robuste :
            #42	Classique, beaucoup de tests communautaires utilisent ce seed. #1234	Fidèle, stable, souvent utilisé pour des tests de cohérence.
            #5555	Fidélité à l’image initiale (ton choix actuel) #2026	Léger changement dans la texture ou la posture, subtil mais prévisible
            #9876	Variation un peu plus visible, garde la structure globale
            pos_embeds, neg_embeds = get_interpolated_embeddings( frame_counter, frames_per_prompt, pos_embeds_list, neg_embeds_list, device )
            try:
                current_latent_single = generate_latents_robuste_4D(
                    latents=current_latent_single.to(device),
                    pos_embeds=pos_embeds,
                    neg_embeds=neg_embeds,
                    unet=unet,
                    scheduler=scheduler,
                    motion_module=None,
                    device=device,
                    dtype=dtype,
                    guidance_scale=current_guidance_scale,  #guidance_scale: 1.5      # un peu plus strict pour que le chat ressorte
                    init_image_scale=current_init_image_scale, #init_image_scale: 0.85  # presque tout le signal de l'image d'origine
                    creative_noise=current_creative_noise, # creative_noise: 0.08    # moins de liberté, plus de cohérence
                    seed=seed  # 42, 1234, 2026, 5555
                )

                # 🔥 FIX NaN / stabilité
                current_latent_single = sanitize_latents(current_latent_single)
            except Exception as e:
                print(f"[Robuste INIT ERROR] {e}")

            current_latent_single = ensure_4_channels(current_latent_single)
            current_latent_single = current_latent_single.to('cpu')
            del input_image
            torch.cuda.empty_cache()

            # ---------------- Transition frames ----------------
            if previous_latent_single is not None and transition_frames > 0:
                for t in range(transition_frames):
                    if stop_generation: break
                    alpha = 0.5 - 0.5*math.cos(math.pi*t/max(transition_frames-1,1))
                    with torch.no_grad():
                        # --- Fusion adaptative avec diminution progressive de l'influence de la frame précédente
                        injection_start = 0.8  # influence initiale de l'ancienne frame
                        injection_end   = 0.1  # influence finale
                        denom = max(transition_frames-1, 1)
                        injection_alpha = injection_start * (1 - t/denom) + injection_end * (t/denom)

                        latent_interp = injection_alpha * previous_latent_single.to(device) + (1 - injection_alpha) * current_latent_single.to(device)
                        # 🔥 FIX NaN / stabilité
                        latent_interp = sanitize_latents(latent_interp)

                        if motion_module:
                            latent_interp, _ = apply_motion_safe(latent_interp, motion_module)


                        # Application de n3r_pro_net
                        latent_interp = apply_n3r_pro_net( latent_interp, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_latents )
                        # Décodage streaming
                        latent_interp = latent_interp / LATENT_SCALE  # “rescale” avant décodage
                        # contrast=1.5, saturation=1.3, latent_scale_boost  #  Recommmander 1.0
                        frame_pil = decode_latents_ultrasafe_blockwise( latent_interp, vae, block_size=block_size, overlap=overlap, gamma=1.0, brightness=1.0, contrast=1.0, saturation=1.0, device=device, frame_counter=frame_counter, latent_scale_boost=latent_scale_boost )
                        #frame_pil = apply_post_processing_adaptive(frame_pil, blur_radius=blur_radius, contrast=contrast, brightness=1.05, saturation=saturation, vibrance_base=1.0, vibrance_max=1.1, sharpen=True, sharpen_radius=1, sharpen_percent=sharpen_percent, sharpen_threshold=2)


                        frame_pil = soft_tone_map(frame_pil)
                        frame_pil = apply_post_processing_adaptive(
                            frame_pil,
                            blur_radius=0.02,          # ↓ plus léger (évite wash)
                            contrast=1.03,             # 🔥 très important → quasi neutre
                            brightness=1.0,            # ne jamais toucher sauf besoin
                            saturation=0.93,           # 🔥 clé → évite saturation globale
                            vibrance_base=0.98,        # 🔥 baisse globale
                            vibrance_max=1.02,         # limite haute très faible
                            sharpen=True,
                            sharpen_radius=0.6,        # 🔥 plus fin
                            sharpen_percent=60,        # 🔥 beaucoup moins agressif
                            sharpen_threshold=3        # évite bruit
                        )

                        # save
                        print(f"[ init SAVE Frame {frame_counter:03d}]")
                        frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                        frame_counter += 1
                        pbar.update(1)

                    del latent_interp
                    torch.cuda.empty_cache()

            # ---------------- Frames principales ----------------

            for f in range(num_fraps_per_image):
                if stop_generation: break
                with torch.no_grad():
                    latents_frame = current_latent_single.to(device)

                    # --- Interpolation des embeddings prompts ---
                    cf_embeds = get_interpolated_embeddings( frame_counter, frames_per_prompt, pos_embeds_list, neg_embeds_list, device )

                    # --- N3R ou mini GPU diffusion ---
                    n3r_latents = None
                    latents = latents_frame.clone()

                    # 🔥 FIX NaN / stabilité
                    latents = sanitize_latents(latents)

                    # ---------------- N3R avec mémoire latente conditionnée ----------------
                    use_n3r_this_frame = use_n3r_model and (frame_counter % random.choice([4,5,6]) == 0)
                    # ------------------- Bloc N3R par frame -------------------
                    if use_n3r_this_frame:
                        try:
                            H, W = latents.shape[-2], latents.shape[-1]
                            N_samples = n3r_model.N_samples
                            coords = generate_n3r_coords(H, W, N_samples, seed, frame_counter, device)
                            n3r_latents = process_n3r_latents(n3r_model, coords, H, W, H, W)
                            fused_latents = fuse_with_memory(n3r_latents, memory_dict, cf_embeds, frame_counter)
                            fused_latents = inject_external(fused_latents, external_latent, frame_counter, device)
                            #latent_injection = get_dynamic_latent_injection(frame_counter, total_frames, start=0.90, end=0.55)
                            #latents = fuse_n3r_latents_adaptive(latents, fused_latents, latent_injection=latent_injection)
                            latents = fuse_n3r_latents_adaptive_new(latents, fused_latents, frame_counter=frame_counter, total_frames=total_frames, latent_injection_start=0.90, latent_injection_end=0.55)
                            latents = sanitize_latents(latents)
                        except Exception as e:
                            print(f"[N3R ERROR] {e}")

                        # Sauvegarde mémoire périodique
                        if frame_counter % 30 == 0:
                            save_memory(memory_dict, memory_file)

                    elif use_mini_gpu:
                        latents = generate_latents_mini_gpu_320(
                            unet=unet, scheduler=scheduler,
                            input_latents=latents_frame, embeddings=cf_embeds,
                            motion_module=motion_module, guidance_scale=current_guidance_scale,
                            device=device, fp16=True, steps=steps,
                            debug=verbose, init_image_scale=current_init_image_scale,
                            creative_noise=current_creative_noise
                        )
                        if latent_injection > 0:
                            if latents.shape[-2:] != latents_frame.shape[-2:]:
                                latents = torch.nn.functional.interpolate(
                                    latents,
                                    size=latents_frame.shape[-2:],
                                    mode='bilinear', align_corners=False
                                ).contiguous()
                            latents = latent_injection*latents_frame + (1-latent_injection)*latents

                    # --- Motion module propre et safe ---
                    if motion_module is not None:
                        if previous_latent_single is not None:
                            latents_seq = torch.stack([
                                previous_latent_single.to(device),
                                latents,
                                latents + 0.01 * torch.randn_like(latents)
                            ], dim=2)  # [B,C,F,H,W]
                        else:
                            latents_seq = latents.unsqueeze(2).repeat(1, 1, 3, 1, 1)

                        # 🔥 sécurisation AVANT motion
                        latents_seq = sanitize_latents(latents_seq)

                        # 🔥 motion safe (une seule fois)
                        latents_seq, applied = apply_motion_safe(latents_seq, motion_module)

                        if applied:
                            latents = latents_seq[:, :, 1, :, :]
                        else:
                            latents = latents

                        latents = sanitize_latents(latents)

                    # 🔥 AUCUN blending → juste update mémoire
                    previous_latent_single = latents.detach().cpu()
                    # Application de n3r_pro_net
                    latents = apply_n3r_pro_net( latents, model=n3r_pro_net, strength=n3r_pro_strength, sanitize_fn=sanitize_latents )
                     # Clamp et resize final 🔥 FIX NaN / stabilité  🔥 nettoyage final intelligent (LE point clé)
                    latents = latents / LATENT_SCALE
                    # Decode
                    frame_pil = decode_latents_ultrasafe_blockwise( latents, vae, block_size=block_size, overlap=overlap, gamma=1.0, brightness=1.0, contrast=1.0, saturation=1.0, device=device, frame_counter=frame_counter, latent_scale_boost=latent_scale_boost )
                    #frame_pil = apply_post_processing_adaptive(frame_pil, blur_radius=blur_radius, contrast=contrast, brightness=1.00, saturation=saturation, vibrance_base=1.1, vibrance_max=1.2, sharpen=True, sharpen_radius=1, sharpen_percent=sharpen_percent, sharpen_threshold=2)

                    frame_pil = soft_tone_map(frame_pil)
                    frame_pil = apply_post_processing_adaptive(
                        frame_pil,
                        blur_radius=0.02,          # ↓ plus léger (évite wash)
                        contrast=1.03,             # 🔥 très important → quasi neutre
                        brightness=1.0,            # ne jamais toucher sauf besoin
                        saturation=0.93,           # 🔥 clé → évite saturation globale
                        vibrance_base=0.98,        # 🔥 baisse globale
                        vibrance_max=1.02,         # limite haute très faible
                        sharpen=True,
                        sharpen_radius=0.6,        # 🔥 plus fin
                        sharpen_percent=60,        # 🔥 beaucoup moins agressif
                        sharpen_threshold=3        # évite bruit
                    )

                    frame_pil.save(output_dir / f"frame_{frame_counter:05d}.png")
                    frame_counter += 1
                    pbar.update(1)

                    # Nettoyage VRAM
                    del latents, latents_frame, cf_embeds, n3r_latents
                    torch.cuda.empty_cache()

            previous_latent_single = current_latent_single

        except Exception as e:
            print(f"\n[FRAME ERROR] {img_path}")
            print(f"Type d'erreur : {type(e).__name__}")
            print(f"Message d'erreur : {e}")
            print("Traceback complet :")
            traceback.print_exc()
            continue

    pbar.close()
    save_frames_as_video_from_folder(output_dir, out_video, fps=fps, upscale_factor=upscale_factor)
    print(f"🎬 Vidéo générée : {out_video}")

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
