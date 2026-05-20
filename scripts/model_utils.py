# scripts/model_utils.py
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel

def load_tokenizer(model_path):
    return CLIPTokenizer.from_pretrained(model_path)

def load_text_encoder(model_path):
    return CLIPTextModel.from_pretrained(model_path)

def load_vae(model_path, offload=False):
    vae = AutoencoderKL.from_pretrained(model_path)
    if offload:
        vae.to("cpu")
    return vae

def load_unet(model_path, inference_config):
    return UNet2DConditionModel.from_pretrained(model_path)

def generate_latents(prompt, unet, tokenizer, text_encoder, L, H, W, key_frames, inter_frames, seed, fp16, device):
    torch.manual_seed(seed)
    B, C = 1, 4
    dtype = torch.float16 if fp16 else torch.float32
    return torch.randn(B, C, L, H//8, W//8, device=device, dtype=dtype)

def decode_frames(video_tensor, out_gif, out_mp4):
    print(f"Décodage et sauvegarde de la vidéo ({out_gif}, {out_mp4})")
