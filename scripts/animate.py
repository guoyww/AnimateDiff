import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

from tokenizers import Tokenizer
from diffusers import AutoencoderKL, HeunDiscreteScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

import shutil

import math
from pathlib import Path


def process_samples(samples, pipeline, n_prompt, prompt, model_config, savedir,gif_name, guidance_scale=7.5, init_image=None):
    print(f"current seed: {torch.initial_seed()}")
    print(f"sampling {prompt} ...")
    print(f"init_image {init_image}")
    sample = pipeline(
        prompt,
        negative_prompt=n_prompt,
        num_inference_steps=model_config.steps,
        guidance_scale=guidance_scale,
        width=args.W,
        height=args.H,
        video_length=args.L,   
        init_image=init_image     
    ).videos
    samples.append(sample)
    save_videos_grid(sample, f"{savedir}/sample/{gif_name}.gif")


def main_single(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    gif_name = "town_side"
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config = OmegaConf.load(args.config)
    samples = []
    print(f"pre-trained model {args.pretrained_model_path}")
    init_image=args.init_image
        
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        print(f"config key {config_key}")
        if args.prompt is None:
            prompt = model_config.get("prompt",  args.inference_config)[0]
        else:
            prompt = args.prompt
        n_prompt = model_config.get("n_prompt",  args.inference_config)[0]
        
        # init_image   = model_config.init_image if hasattr(model_config, 'init_image') else None
        motion_modules = model_config.motion_module
        motion_modules = (
            [motion_modules]
            if isinstance(motion_modules, str)
            else list(motion_modules)
        )
        for motion_module in motion_modules:
            inference_config = OmegaConf.load(
                model_config.get("inference_config", args.inference_config)
            )

            ### >>> create validation pipeline >>> ###
            tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_path, subfolder="tokenizer"
            )
            text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_path, subfolder="text_encoder"
            )
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_path, subfolder="vae"
            )
            unet = UNet3DConditionModel.from_pretrained_2d(
                args.pretrained_model_path,
                subfolder="unet",
                unet_additional_kwargs=OmegaConf.to_container(
                    inference_config.unet_additional_kwargs
                ),
            )

            if is_xformers_available():
                print("using xformers")
                unet.enable_xformers_memory_efficient_attention()
            else:
                assert False

            animated_pipeline = AnimationPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=HeunDiscreteScheduler()
            ).to("cuda")

            # pipeline.enable_xformers_memory_efficient_attention()
            # pipeline.enable_cpu_offload()

            pipeline = load_weights(
                animated_pipeline,
                # motion module
                motion_module_path=motion_module,
                motion_module_lora_configs=model_config.get(
                    "motion_module_lora_configs", []
                ),
                # image layers
                dreambooth_model_path=model_config.get("dreambooth_path", ""),
                lora_model_path=model_config.get("lora_model_path", ""),
                lora_alpha=model_config.get("lora_alpha", 0.8),
            ).to("cuda")
            
            print(f"init_image: {init_image}")
            process_samples(
                    init_image=init_image,
                    samples=samples,
                    pipeline=pipeline,
                    n_prompt=n_prompt,
                    prompt=prompt,
                    model_config=model_config,
                    savedir=savedir,
                    guidance_scale=args.guidance_scale,
                    gif_name=gif_name
            )

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.mp4", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")

    if init_image is not None:
        shutil.copy(init_image, f"{savedir}/init_image.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="models/StableDiffusion",
    )
    parser.add_argument(
        "--inference_config", type=str, default="configs/inference/inference-v2.yaml"
    )
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--init_image", type=str)
    
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main_single(args)
