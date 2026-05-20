# n3rUnet5D_auto_tile128.py

import torch
from scripts.utils import log_gpu_memory
from scripts.modules.motion_module_tiny import MotionModuleTiny
from scripts.vae import decode_latents_to_image_tiled
from scripts.utils.config_loader import load_config
from scripts.utils.vae_utils import safe_load_vae, safe_load_unet, safe_load_scheduler
from scripts.utils.vae_utils import decode_latents_to_image_tiled, decode_latents_frame_auto, generate_5D_video_auto
from scripts.utils.motion_utils import load_motion_module
from scripts.utils.n3r_utils import generate_latents_ai_5D, load_image_file, generate_5D_video_auto
from scripts.utils.n3r_utils import decode_latents_frame_auto, generate_5D_video_auto, log_gpu_memory

tile_size = 128
overlap = 64





# --- UTILISATION ---
if __name__ == "__main__":
    import yaml
    config_path = "configs/prompts/1_animate/256.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generate_5D_video_auto(
        pretrained_model_path="/mnt/62G/huggingface/miniSD",
        config=config,
        device='cuda'
    )
