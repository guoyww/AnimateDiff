# ----------------------------------------------------------------------------------------
# n3rRealControl.py - AnimateDiff stables, ProNet + HDR ultra-light ~2Go VRAM - pipeline 4D
# Prompt / Input → N3RModelOptimized → MotionModule → UNet → LoRA → VAE → Image / Vidéo
#Avec use_mini_gpu et generate_latents_mini_gpu_320 → ~2,1 Go VRAM ✅ Avec use_n3r_model et N3RModelOptimized → ~3,6 Go VRAM
# Image input ↓ OpenPose → skeleton (frame t) ↓ ControlNet (condition pose) ↓ UNet (avec pos/neg embeds) ↓ Latents 4D (animés) ↓ N3RProNet (détails + iris + sharpen) ↓ Decode blockwise ↓ Frames animées
#use_n3r_model: True # Amélioration de la qualité use_n3r_pro_net: False # Augmentation des détails à mettre au point use_mini_gpu: False # Génération rapide use_openpose: False # Pas encore au point
# ----------------------------------------------------------------------------------------
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

import math, threading, random, json, traceback, hashlib, pickle, argparse
from pathlib import Path
from datetime import datetime
import traceback
from functools import partial
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image, ImageFilter
from diffusers import PNDMScheduler
from transformers import CLIPTokenizerFast, CLIPTextModel
from scripts.utils.lora_utils import apply_lora_smart
from scripts.utils.vae_config import load_vae
from scripts.utils.n3rModelUtils import generate_n3r_coords, process_n3r_latents, fuse_with_memory, inject_external, fuse_n3r_latents_adaptive_new
from scripts.utils.tools_utils import ensure_4_channels, print_generation_params, sanitize_latents, stabilize_latents_advanced, log_debug, compute_overlap, get_interpolated_embeddings, save_memory, load_memory, load_external_embedding_as_latent, inject_external_embeddings, update_n3r_memory, compute_weighted_params, adapt_embeddings_to_unet, get_dynamic_latent_injection, save_input_frame, apply_motion_safe, encode_prompts_batch
from scripts.utils.config_loader import load_config
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import load_images_test, generate_latents_mini_gpu_320, run_diffusion_pipeline, generate_latents_robuste_4D
from scripts.utils.fx_utils import encode_images_to_latents_nuanced, adaptive_post_process, save_frames_as_video_from_folder, encode_images_to_latents_safe, encode_images_to_latents_hybrid, interpolate_param_fast, fuse_n3r_latents_adaptive, adaptive_post_process, remove_white_noise

from scripts.utils.vae_utils import safe_load_unet
from scripts.utils.n3rModelFast4Go import N3RModelFast4GB, N3RModelLazyCPU, N3RModelOptimized
from scripts.utils.n3rProNet import N3RProNet
from scripts.utils.n3rProNet_utils import apply_n3r_pro_net, save_frame_verbose, full_frame_postprocess, decode_latents_ultrasafe_blockwise, get_eye_coords_safe, create_volumetrique_mask, create_eye_mask, tensor_to_pil, apply_pro_net_volumetrique, apply_pro_net_with_eyes, get_eye_coords_safe, scale_eye_coords_to_latents, get_coords, get_coords_safe, decode_latents_ultrasafe_blockwise_pro, decode_latents_ultrasafe_blockwise_sharp, decode_latents_ultrasafe_blockwise_natural, decode_latents_ultrasafe_blockwise_ultranatural, create_mouth_mask, get_mouth_coords_safe, scale_mouth_coords_to_latents, apply_pro_net_with_mouth
from scripts.utils.n3rControlNet import create_canny_control, control_to_latent, match_latent_size
from scripts.utils.n3rOpenPose_utils import generate_pose_sequence, apply_controlnet_openpose_step, load_controlnet_openpose, load_controlnet_openpose_local, match_latent_size, control_to_latent_safe, build_control_latent_debug, convert_json_to_pose_sequence, debug_pose_visual, save_debug_pose_image, fix_pose_sequence, prepare_controlnet, log_frame_error, apply_controlnet_openpose_step_ultrasafe, apply_openpose_tilewise, controlnet_tile_fn, apply_openpose_tilewise_safe, apply_breathing_latents, apply_upper_body_motion, extract_keypoints_from_pose, apply_pose_driven_motion, save_debug_pose_image_with_skeleton, update_pose_sequence_from_keypoints_batch

LATENT_SCALE = 0.18215
stop_generation = False
# ---------------- Thread stop ----------------
def wait_for_stop():
    global stop_generation
    inp = input("Appuyez sur '²' + Entrée pour arrêter : ")
    if inp.lower() == "²":
        stop_generation = True
threading.Thread(target=wait_for_stop, daemon=True).start()

# ---------------- MAIN FIABLE ----------------
def main(args):
    global stop_generation
    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    # Configurable depuis ton fichier cfg
    use_mini_gpu = cfg.get("use_mini_gpu", True)
    verbose, psave = cfg.get("verbose", False), cfg.get("psave", False)
    latent_injection = float(cfg.get("latent_injection", 0.75))
    latent_injection = min(max(latent_injection, 0.5), 0.9)  # plage sûre
    final_latent_scale = cfg.get("final_latent_scale", 1/8) # 1/8 speed, 1/4 moyen, 1/2 low
    fps, upscale_factor = cfg.get("fps", 12), cfg.get("upscale_factor", 1)
    transition_frames, num_fraps_per_image = cfg.get("transition_frames", 4), cfg.get("num_fraps_per_image", 2)
    steps = max(cfg.get("steps", 16), 4)
    guidance_scale = cfg.get("guidance_scale", 6.5) # 0.15 peut de créativité 4.5 moderé
    guidance_scale_end = cfg.get("guidance_scale_end", 7.0) # 0.15 peut de créativité 4.5 moderé
    init_image_scale = cfg.get("init_image_scale", 0.75) # 0.85 ou 0.95 proche de l'init' (0.75)
    init_image_scale_end = cfg.get("init_image_scale_end", 0.9) # 0.85 ou 0.95 proche de l'init'
    creative_noise, creative_noise_end = cfg.get("creative_noise", 0.0), cfg.get("creative_noise_end", 0.08)
    latent_scale_boost = cfg.get("latent_scale_boost", 1.0)
    frames_per_prompt = cfg.get("frames_per_prompt", 20)  # nombre de frames par prompt
    contrast, blur_radius, sharpen_percent = cfg.get("contrast", 1.15), cfg.get("blur_radius", 0.03), cfg.get("sharpen_percent", 90)  # Post Traitement
    H, W = cfg.get("H", 512), cfg.get("W", 512)
    block_size = cfg.get("block_size", 64)  # block_size auto selon résolution 64
    overlap = compute_overlap(cfg["W"], cfg["H"], block_size) #overlap = 32 #32
    print(f"Dimention : overlap :", overlap)

    use_n3r_model, use_n3r_pro_net  = cfg.get("use_n3r_model", False), cfg.get("use_n3r_pro_net", True)
    use_openpose = cfg.get("use_openpose", True)
    controlnet_scale = cfg.get("controlnet_scale", 1.0) # typiquement 0.5 → 1.0
    control_strength = cfg.get("control_strength", 1.5)
    n3r_pro_strength = cfg.get("n3r_pro_strength", 0.001) # 0.1, 0.2, 0.3
    target_temp, reference_temp = 7800, 6500 #target_temp = 8000 reference_temp = 6000  (Froid)
    facteur = cfg.get("facteur", 8) # 8, 6, 4

    # Seed aléatoire
    seed = torch.randint(0, 100000, (1,)).item()
    params = { 'use_mini_gpu': use_mini_gpu,  'fps': fps, 'upscale_factor': upscale_factor, 'num_fraps_per_image': num_fraps_per_image, 'steps': steps, 'guidance_scale': guidance_scale, 'guidance_scale_end': guidance_scale_end, 'init_image_scale': init_image_scale, 'init_image_scale_end': init_image_scale_end, 'creative_noise': creative_noise, 'creative_noise_end': creative_noise_end, 'latent_scale_boost': latent_scale_boost, 'final_latent_scale': final_latent_scale, 'seed': seed, 'latent_injection': latent_injection, 'transition_frames': transition_frames, 'block_size': block_size, 'use_n3r_model': use_n3r_model }
    print_generation_params(params)

    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012,
                              beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps, device=device)

    # 🔥 FIX CRITIQUE GLOBAL
    if hasattr(scheduler, "alphas_cumprod"):
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.cpu()

    if hasattr(scheduler, "final_alpha_cumprod"):
        scheduler.final_alpha_cumprod = scheduler.final_alpha_cumprod.cpu()

    if hasattr(scheduler, "timesteps") and isinstance(scheduler.timesteps, torch.Tensor):
        scheduler.timesteps = scheduler.timesteps.cpu()

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
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_path,"text_encoder")).to("cpu").to(dtype)
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
        sample_embeds = text_encoder(text_inputs_sample.input_ids.to("cpu")).last_hidden_state
    current_dim = sample_embeds.shape[-1]
    projection = None
    if current_dim != unet_cross_attention_dim:
        projection = torch.nn.Linear(current_dim, unet_cross_attention_dim).to(device).to(dtype)

    # --- Pré-calcul des embeddings pour interpolation
    # Appel de la fonction - encode_prompts_batch
    pos_embeds_list, neg_embeds_list = encode_prompts_batch( prompts=prompts, negative_prompts=negative_prompts, tokenizer=tokenizer, text_encoder=text_encoder, device="cpu", projection=None)
    print(f"Pos embeds shape: {pos_embeds_list[0].shape} | Neg embeds shape: {neg_embeds_list[0].shape}")

    # ---------- Input image -----------------------------------
    input_paths = cfg.get("input_images") or [cfg.get("input_image")]
    total_frames = len(input_paths) * num_fraps_per_image * max(len(prompts), 1)

    # ---------------- Input  ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./outputs/RealControl{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_video = output_dir / f"output_{timestamp}.mp4"

    # ---------------- load_controlnet_openpose ----------------
    if use_openpose:
        controlnet = load_controlnet_openpose_local( device=device, dtype=torch.float16, use_fp16=True, debug=True )
        controlnet, pose_sequence = prepare_controlnet( controlnet, device=device, dtype=dtype )

        try:
            base_dir = Path(__file__).resolve().parent
            json_file = base_dir / "json" / "anim2.json"

            with open(json_file, "r") as f:
                anim_data = json.load(f)

            print(f"✅ JSON chargé : {json_file}")
            pose_sequence = convert_json_to_pose_sequence( anim_data, H=cfg["H"], W=cfg["W"], device=device, dtype=dtype, debug=True, output_dir=output_dir)

            if pose_sequence is None:
                print("❌ Aucun pose_sequence → OpenPose désactivé")
                use_openpose = False
            else:
                # 🔥 Fix interpolation
                pose_sequence = fix_pose_sequence( pose_sequence, total_frames=total_frames, device=device, dtype=dtype )

        except Exception as e:
            print(f"[Load Json animation INIT ERROR] {e}")

    # ---------------- N3RModelOptimized ----------------
    n3r_model = None
    if use_n3r_model:
        n3r_model = N3RModelOptimized(
            L_low=cfg.get("n3r_L_low",3), L_high=cfg.get("n3r_L_high",6), N_samples=cfg.get("n3r_N_samples",32), tile_size=cfg.get("n3r_tile_size",64),
            cpu_offload=cfg.get("n3r_cpu_offload",True)
        ).to(device)
        n3r_model.eval()
        print(f"✅ N3RModelOptimized initialisé sur {device}")

        # ------------------- Initialisation mémoire -------------------
        output_dir_m = Path("./outputs")
        memory_file = output_dir_m / "n3r_memory"
        memory_dict = load_memory(memory_file)
    # ---------------- n3r_pro_net ----------------
    n3r_pro_net = None
    if use_n3r_pro_net:
        n3r_pro_net = N3RProNet(channels=4).to(device).to(dtype)
        n3r_pro_net.eval()
        print("✅ N3RProNet activé")

    previous_latent_single = None
    frame_counter = 0
    pbar = tqdm(total=total_frames, ncols=120)

    # ---------------- Frames principales avec interpolation prompts ----------------
    external_embeddings = None

    # Charger latent externe avant la génération
    external_path = "/mnt/62G/huggingface/cyber-fp16/pt/KnxCOmiXNeg.safetensors"
    external_latent = load_external_embedding_as_latent(
        external_path, (1, 4, cfg["H"]//facteur, cfg["W"]//facteur)
    ).to(device)

    for img_idx, img_path in enumerate(input_paths):
        if stop_generation: break
        try:
            # Paramètres interpolés
            current_init_image_scale, current_creative_noise, current_guidance_scale = compute_weighted_params( frame_counter, total_frames, init_start=init_image_scale, init_end=init_image_scale_end,noise_start=creative_noise, noise_end=creative_noise_end, guidance_start=guidance_scale, guidance_end=guidance_scale_end, mode="cosine" )
            print(f"[Frame {frame_counter:03d}] " f"init_image_scale={current_init_image_scale:.3f}, " f"guidance_scale={current_guidance_scale:.3f}, " f"creative_noise={current_creative_noise:.3f}")

            # Charger et encoder l'image sur GPU
            input_image = load_images_test([img_path], W=cfg["W"], H=cfg["H"], device=device, dtype=dtype)
            # ---------------- Pose sequence ---------------------------------------------
            start_pose = input_image.to(device=device, dtype=dtype) # start_pose = tensor 4D BCHW directement
            # Pose sequence
            if use_openpose and pose_sequence is None:
                pose_sequence = generate_pose_sequence(base_pose=start_pose, num_frames=total_frames, device=device, dtype=dtype, debug=True)
            # 🔥 Détection yeux (une seule fois par image)
            input_pil = tensor_to_pil(input_image)  # à créer si tu ne l'as pas

            # 🔥 n3rControl - encode Canny en sécurité------------------------------------------------------------------------------------
            base_control_latent = build_control_latent_debug(
                input_pil,
                vae,
                device="cuda",
                latent_scale=LATENT_SCALE
            )
            base_control_latent = sanitize_latents(base_control_latent)
            base_control_latent = torch.clamp(torch.nan_to_num(base_control_latent), -1.0, 1.0)

            control_latent = base_control_latent + 0.01 * torch.randn_like(base_control_latent, dtype=torch.float16, device="cuda")
            control_latent = sanitize_latents(control_latent)
            # -----------------------------------------------------------------------------------------
            # coordonner masque eye et masque volumetrique
            eye_coords = get_eye_coords_safe(input_pil)
            mouth_coords = get_mouth_coords_safe(input_pil)
            # coordonner globale
            coords_v = get_coords_safe( input_pil, H=cfg["H"], W=cfg["W"] )
            input_image = ensure_4_channels(input_image)
            if frame_counter > 0:
                initframe = frame_counter+transition_frames
            else:
                initframe = frame_counter
            save_input_frame( input_image, output_dir, initframe, pbar=pbar, blur_radius=blur_radius, contrast=contrast, saturation=1.0, apply_post=False )

            current_latent_single = encode_images_to_latents_hybrid(input_image, vae, device=device, latent_scale=LATENT_SCALE)
            current_latent_single = torch.nn.functional.interpolate(
                current_latent_single, size=(cfg["H"]//facteur, cfg["W"]//facteur),
                mode='bilinear', align_corners=False
            )

            # 🔥 FIX NaN / stabilité
            current_latent_single = sanitize_latents(current_latent_single)
            # Génération initiale robuste :
            pos_embeds, neg_embeds = get_interpolated_embeddings( frame_counter, frames_per_prompt, pos_embeds_list, neg_embeds_list, device, debug=False)
            try:
                current_latent_single = generate_latents_robuste_4D(
                    latents=current_latent_single.to(device),
                    pos_embeds=pos_embeds, neg_embeds=neg_embeds, unet=unet, scheduler=scheduler,
                    motion_module=None, device=device, dtype=dtype, guidance_scale=current_guidance_scale,
                    init_image_scale=current_init_image_scale, #init_image_scale: 0.85  # presque tout le signal de l'image d'origine
                    creative_noise=current_creative_noise, seed=seed  # 42, 1234, 2026, 5555
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
                        injection_start = 0.9  # influence initiale de l'ancienne frame
                        injection_end   = 0.1  # influence finale
                        denom = max(transition_frames-1, 1)
                        injection_alpha = injection_start * (1 - t/denom) + injection_end * (t/denom)

                        latent_interp = injection_alpha * previous_latent_single.to(device) + (1 - injection_alpha) * current_latent_single.to(device)
                        # 🔥 FIX NaN / stabilité
                        latent_interp = sanitize_latents(latent_interp)

                        if motion_module:
                            latent_interp, _ = apply_motion_safe(latent_interp, motion_module)

                        # Application de n3r_pro_net - réutilisé pour toutes les frames - creation des masques
                        eye_coords_latent = scale_eye_coords_to_latents( eye_coords, img_H=cfg["H"], img_W=cfg["W"], lat_H=latent_interp.shape[-2], lat_W=latent_interp.shape[-1] )
                        mouth_coords_latent = scale_mouth_coords_to_latents( mouth_coords, img_H=cfg["H"], img_W=cfg["W"], lat_H=latent_interp.shape[-2], lat_W=latent_interp.shape[-1] )
                        if eye_coords_latent:
                            eye_mask = create_eye_mask(latent_interp, eye_coords_latent)
                        if mouth_coords_latent:
                            mouth_mask = create_mouth_mask(latent_interp, mouth_coords_latent)
                        volume_mask = create_volumetrique_mask(latent_interp, coords_v, debug=False)
                        # Application du ProNet tout en protégeant les yeux
                        if use_n3r_pro_net:
                            latents = apply_pro_net_volumetrique(latent_interp, coords_v, n3r_pro_net, n3r_pro_strength, sanitize_latents, debug=False)
                            eye_coords_latent = scale_eye_coords_to_latents( eye_coords, img_H=cfg["H"], img_W=cfg["W"], lat_H=latents.shape[-2], lat_W=latents.shape[-1] )
                            mouth_coords_latent = scale_mouth_coords_to_latents( mouth_coords, img_H=cfg["H"], img_W=cfg["W"], lat_H=latents.shape[-2], lat_W=latents.shape[-1] )
                            if eye_coords_latent:
                                latents = apply_pro_net_with_eyes(latents, eye_coords_latent, n3r_pro_net, n3r_pro_strength, sanitize_fn=sanitize_latents)
                            if mouth_coords_latent:
                                latents = apply_pro_net_with_mouth(latents, mouth_coords_latent, n3r_pro_net, n3r_pro_strength, sanitize_fn=sanitize_latents)

                        # Décodage streaming
                        latent_interp = latent_interp / LATENT_SCALE  # “rescale” avant décodage
                        print(f"Dimention frame inter: Shape de latent_interp :", latent_interp.shape)
                        frame_pil = decode_latents_ultrasafe_blockwise_ultranatural( latent_interp, vae, block_size=block_size, overlap=overlap, device=device, frame_counter=frame_counter, latent_scale_boost=latent_scale_boost )

                        #Post Traitement
                        frame_pil = full_frame_postprocess( frame_pil, output_dir, frame_counter, target_temp=target_temp, reference_temp=reference_temp, blur_radius=blur_radius, contrast=contrast, sharpen_percent=sharpen_percent, psave=psave )
                        save_frame_verbose(frame_pil, output_dir, frame_counter-1, suffix="0i", psave=True)
                        frame_counter += 1
                        pbar.update(1)

                    del latent_interp
                    torch.cuda.empty_cache()

            # ---------------- Frames principales ----------------
            prev_keypoints = None # motion Open Pose
            current_keypoints = None # init None
            for f in range(num_fraps_per_image):
                if stop_generation:
                    break
                with torch.no_grad():
                    latents_frame = current_latent_single.to(device)
                    print(f"Dimention inital: Shape de latents_frame :", latents_frame.shape)

                    # --- Interpolation des embeddings prompts ---
                    cf_embeds = get_interpolated_embeddings(frame_counter, frames_per_prompt, pos_embeds_list, neg_embeds_list, device, debug=False)
                    latents = sanitize_latents(latents_frame.clone())  # 🔥 FIX NaN / stabilité
                    # --- volume mask ---
                    volume_mask = create_volumetrique_mask(latents, coords_v, debug=False)
                    control_weight_map = 0.05 + 0.25 * volume_mask**1.5
                    control_latent = sanitize_latents(base_control_latent + 0.005 * torch.randn_like(base_control_latent))
                    control_latent, control_weight_map = match_latent_size(latents, control_latent, control_weight_map)

                    # ---------------- N3R avec mémoire latente conditionnée ----------------
                    #use_n3r_this_frame = math.sin(frame_counter * 0.2) > 0.7
                    use_n3r_this_frame = use_n3r_model and (frame_counter % random.choice([4,5,6]) == 0)
                    #control_strength = 0.05 * (1 - frame_counter / total_frames) + 0.02
                    controlnet_scale = 1.0 + 0.1 * torch.sin(torch.tensor(frame_counter * 0.1))
                    print(f"[DEBUG] 🧠 Pose control_strength ={control_strength:.4f}")
                    print(f"[DEBUG] 🧠 Pose controlnet_scale ={controlnet_scale:.4f}")


                    if use_n3r_this_frame and use_n3r_model:
                        print(f"[DEBUG] N3R active: {use_n3r_this_frame}")
                        try:
                            H, W = latents.shape[-2], latents.shape[-1]
                            coords = generate_n3r_coords(H, W, n3r_model.N_samples, seed, frame_counter, device)
                            n3r_latents = process_n3r_latents(n3r_model, coords, H, W, H, W)
                            fused_latents = fuse_with_memory(n3r_latents, memory_dict, cf_embeds, frame_counter)
                            external_weight = 0.2 * (1 - frame_counter / total_frames)
                            fused_latents = (1 - external_weight) * fused_latents + external_weight * external_latent
                            latents = fuse_n3r_latents_adaptive_new(latents, fused_latents, frame_counter,
                                                                    total_frames=total_frames,
                                                                    latent_injection_start=0.90, latent_injection_end=0.55)
                            latents = sanitize_latents(latents)
                        except Exception as e:
                            print(f"[N3R ERROR] {e}")
                            traceback.print_exc()

                        if frame_counter % 30 == 0:
                            save_memory(memory_dict, memory_file)

                    # ---------------- Mini-GPU diffusion ----------------
                    elif use_mini_gpu:
                        latents = generate_latents_mini_gpu_320(
                            unet=unet, scheduler=scheduler,
                            input_latents=latents_frame, embeddings=cf_embeds,
                            motion_module=motion_module, guidance_scale=current_guidance_scale, device=device, fp16=True, steps=steps,
                            debug=verbose, init_image_scale=current_init_image_scale, creative_noise=current_creative_noise
                        )
                        # ControlNet injection :
                        control_latent, control_weight_map = match_latent_size(latents, control_latent, control_weight_map)
                        print(f"[DEBUG] latents: {latents.shape}, control_latent: {control_latent.shape}, control_weight_map: {control_weight_map.shape}")
                        latents = latents + control_strength * control_weight_map * control_latent
                        latents = sanitize_latents(latents)

                        if latent_injection > 0:
                            if latents.shape[-2:] != latents_frame.shape[-2:]:
                                latents = torch.nn.functional.interpolate( latents, size=latents_frame.shape[-2:], mode='bilinear', align_corners=False ).contiguous()
                            latents = latent_injection*latents_frame + (1-latent_injection)*latents


                    # ---------------- ControlNet OpenPose ----------------------------------------------------------
                    # 🔹 Si motion_module est None, injecter un léger bruit temporel
                    if use_openpose:
                        try:
                            # 🔥 dtype cible réel (UNet)
                            target_dtype = next(unet.parameters()).dtype
                            # ------------------- Load pose frame -------------------
                            pose_full = pose_sequence[frame_counter % pose_sequence.shape[0]]
                            # Format BCHW et channels fix
                            if pose_full.ndim == 3:
                                if pose_full.shape[0] in [1, 3]:  # C,H,W
                                    pose_full = pose_full.unsqueeze(0)
                                else:  # H,W,C
                                    pose_full = pose_full.permute(2, 0, 1).unsqueeze(0)

                            # → channels fix
                            if pose_full.shape[1] > 3:
                                pose_full = pose_full[:, :3]
                            elif pose_full.shape[1] == 1:
                                pose_full = pose_full.repeat(1, 3, 1, 1)

                            # → normalize [-1,1]
                            pose_full = (pose_full - 0.5) * 2.0
                            pose_full = torch.clamp(pose_full, -1.0, 1.0)

                            # → device + dtype
                            pose_full = pose_full.to(device=device, dtype=target_dtype)

                            print(f"[DEBUG] Pose full {pose_full.shape} dtype={pose_full.dtype}")

                            # 🔹 ===== 2. BUILD LATENT-SPACE POSE (CRUCIAL) =====
                            latent_h, latent_w = latents.shape[2], latents.shape[3]
                            pose_latent_full = F.interpolate( pose_full, size=(latent_h, latent_w), mode='bilinear', align_corners=False ).to(device=device, dtype=target_dtype) # ou bilinear
                            #pose_latent_full = F.interpolate( pose_full, size=(latent_h, latent_w), mode='nearest').to(device=device, dtype=target_dtype)
                            print(f"[DEBUG] Pose latent {pose_latent_full.shape} dtype={pose_latent_full.dtype}")

                            # ------------------- Backup latents -------------------
                            latents_before_openpose = latents.clone()
                            print(f"[DEBUG] Latents avant OpenPose min={latents.min().item():.4f}, max={latents.max().item():.4f}")


                            # ------------------- Tile-wise OpenPose -------------------
                            tile_fn_partial = partial( controlnet_tile_fn, frame_counter=frame_counter, unet=unet, controlnet=controlnet, scheduler=scheduler, cf_embeds=cf_embeds, current_guidance_scale=current_guidance_scale, controlnet_scale=controlnet_scale, device=device, target_dtype=target_dtype, )

                            # 🔹 Appel correct de apply_openpose_tilewise
                            latents = apply_openpose_tilewise_safe( latents, pose_latent_full, tile_fn_partial, block_size=block_size, overlap=overlap, device=device, debug=True, debug_dir=output_dir,frame_idx=frame_counter)

                            # BOOST
                            latents_after = latents.clone()  # sortie OpenPose


                            #new code fonction --------------------------------------------------------------------------------------

                            # Extraire keypoints + debug visuel
                            # 🔹 Extraction / update des keypoints
                            current_keypoints = extract_keypoints_from_pose( pose_full, debug=True, debug_dir=output_dir, frame_counter=frame_counter)
                            """
                            if current_keypoints is None:
                                current_keypoints = extract_keypoints_from_pose(
                                    pose_full, debug=True, debug_dir=output_dir, frame_counter=frame_counter
                                )
                            """
                            # Mettre à jour les keypoints éventuellement modifiés
                            current_keypoints = update_pose_sequence_from_keypoints_batch( keypoints_tensor=current_keypoints, prev_keypoints=prev_keypoints, frame_idx=frame_counter, alpha=0.5, add_motion=True, debug=True )

                            # 🔹 Appliquer le mouvement du haut du corps / OpenPose
                            latents = apply_pose_driven_motion(
                                latents=latents,
                                previous_latent=previous_latent_single,
                                latents_before_openpose=latents_before_openpose if 'latents_before_openpose' in locals() else None,
                                latents_after_openpose=latents_after if 'latents_after' in locals() else None,
                                keypoints=current_keypoints,
                                prev_keypoints=prev_keypoints,
                                frame_counter=frame_counter,
                                device=device,
                                breathing=True,
                                debug=True,
                                debug_dir=output_dir
                            )

                            # 🔹 Nettoyage / stabilisation
                            latents = sanitize_latents(latents)

                            # 🔹 Stocker les keypoints pour la frame suivante
                            prev_keypoints = current_keypoints.clone()

                            #-------------------------------------------------------------------------------------------------------------
                            delta = latents_after - latents_before_openpose
                            # 🔥 masque pose plus agressif mais propre
                            pose_strength = pose_latent_full.abs().mean(dim=1, keepdim=True)
                            pose_mask = torch.clamp(pose_strength * 4.0, 0.0, 1.0)

                            # 🔥 énergie de mouvement locale
                            motion_energy = delta.abs().mean(dim=1, keepdim=True)

                            # 🔥 amplification NON linéaire (clé)
                            motion_boost = 1.0 + torch.pow(torch.clamp(motion_energy * 3.0, 0.0, 1.0), 0.7)

                            # Fusion intelligente
                            latents = latents + 1.2 * delta * pose_mask * motion_boost
                            # ------------------- Cleanup -------------------
                            latents = torch.nan_to_num(latents)
                            latents = sanitize_latents(latents)
                            latents_max = latents.abs().amax(dim=(2,3), keepdim=True)
                            latents = latents / (latents_max + 1e-6) * 1.05
                            latents = torch.clamp(latents, -1.3, 1.3)

                            diff = (latents - latents_before_openpose).abs().mean()
                            print(f"[DEBUG] OpenPose impact: {diff.item():.6f}")
                            print(f"[DEBUG] Latents après OpenPose min={latents.min().item():.4f}, max={latents.max().item():.4f}")

                        except Exception:
                            print("[ERROR] ControlNet OpenPose failed:")
                            traceback.print_exc()
                            latents = latents_before_openpose.clone()

                        # 🔹 debug image
                        #save_debug_pose_image(pose_full, frame_counter, output_dir, cfg, prefix="openpose")
                        save_debug_pose_image_with_skeleton( pose_tensor=pose_full, keypoints_tensor=current_keypoints, frame_counter=frame_counter, output_dir=output_dir, cfg=cfg, prefix="openpose" )


                    # ---------------- Motion module --------------------------------------------------------------------------------------------------
                    if motion_module is not None:
                        latents_seq = latents.unsqueeze(2).repeat(1, 1, 3, 1, 1) if previous_latent_single is None \
                                    else torch.stack([previous_latent_single.to(device), latents, latents + 0.003 * torch.randn_like(latents)], dim=2)
                        latents_seq = sanitize_latents(latents_seq)
                        latents_seq, applied = apply_motion_safe(latents_seq, motion_module)
                        latents = latents_seq[:, :, 1, :, :] if applied else latents
                        latents = sanitize_latents(latents)
                        print(f"[DEBUG] Après Motion module min={latents.min().item():.4f}, max={latents.max().item():.4f}, NaN={torch.isnan(latents).any().item()}")

                    # ---------------- ProNet yeux ----------------
                    if use_n3r_pro_net:
                        latents = apply_pro_net_volumetrique(latents, coords_v, n3r_pro_net, n3r_pro_strength, sanitize_latents, debug=False)
                        print(f"[DEBUG] Après ProNet volumetrique min={latents.min().item():.4f}, max={latents.max().item():.4f}, NaN={torch.isnan(latents).any().item()}")

                        eye_coords_latent = scale_eye_coords_to_latents(eye_coords, img_H=cfg["H"], img_W=cfg["W"],
                                                                        lat_H=latents.shape[-2], lat_W=latents.shape[-1])
                        if eye_coords_latent:
                            latents = apply_pro_net_with_eyes(latents, eye_coords_latent, n3r_pro_net, n3r_pro_strength,
                                                            sanitize_fn=sanitize_latents)
                            print(f"[DEBUG] Après ProNet yeux min={latents.min().item():.4f}, max={latents.max().item():.4f}, NaN={torch.isnan(latents).any().item()}")

                        mouth_coords_latent = scale_mouth_coords_to_latents(eye_coords, img_H=cfg["H"], img_W=cfg["W"],
                                                                        lat_H=latents.shape[-2], lat_W=latents.shape[-1])
                        if mouth_coords_latent:
                            latents = apply_pro_net_with_mouth(latents, mouth_coords_latent, n3r_pro_net, n3r_pro_strength,
                                                            sanitize_fn=sanitize_latents)
                            print(f"[DEBUG] Après ProNet Bouche min={latents.min().item():.4f}, max={latents.max().item():.4f}, NaN={torch.isnan(latents).any().item()}")
                        latents = sanitize_latents(latents)
                    # ---------------- Décodage final ----------------
                    # 🔥 SANITY AVANT DECODE
                    latents = latents / LATENT_SCALE
                    print(f"Dimention : Shape de latents :", latents.shape)
                    frame_pil = decode_latents_ultrasafe_blockwise_ultranatural(latents, vae, block_size=block_size, overlap=overlap, device=device,
                        frame_counter=frame_counter, latent_scale_boost=latent_scale_boost, scale=facteur
                    )
                    frame_pil = full_frame_postprocess(frame_pil, output_dir, frame_counter, target_temp=target_temp, reference_temp=reference_temp,
                                                    blur_radius=blur_radius, contrast=contrast, sharpen_percent=sharpen_percent, psave=psave)
                    save_frame_verbose(frame_pil, output_dir, frame_counter, suffix="0f", psave=True)

                    previous_latent_single = latents.detach().cpu()
                    frame_counter += 1
                    pbar.update(1)
                    for var in ["latents", "latents_frame", "cf_embeds", "n3r_latents"]:
                        if var in locals():
                            del locals()[var]
                    torch.cuda.empty_cache()

            previous_latent_single = current_latent_single

        except Exception as e:
            log_frame_error(img_path, e)
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
