from collections import defaultdict
from typing import Any

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

from .controlnet_processors import CONTROLNET_PROCESSORS


def get_video_info(cap):
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return height, width, fps, frame_count


class ControlnetModule:
    def __init__(self, config):
        self.config = config
        self.video_length = self.config['video_length']
        self.img_w = self.config['img_w']
        self.img_h = self.config['img_h']
        self.do_cfg = self.config['guidance_scale'] > 1.0
        self.num_inference_steps = config['steps']
        self.guess_mode = config['guess_mode']
        self.device = config['device']

        controlnet_info = CONTROLNET_PROCESSORS[self.config['controlnet_processor']]

        if ('controlnet_processor_path' not in config) or not len(config['controlnet_processor_path']):
            config['controlnet_processor_path'] = controlnet_info['controlnet']

        controlnet = ControlNetModel.from_pretrained(
            controlnet_info['controlnet'], torch_dtype=torch.float16)
        
        if controlnet_info['is_custom']:
            self.processor = controlnet_info['processor'](
                **controlnet_info['processor_params'])
        else:
            self.processor = controlnet_info['processor'].from_pretrained(
                'lllyasviel/Annotators')
            
        self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config['controlnet_pipeline'], #"runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            torch_dtype=torch.float16
        )

        del self.controlnet_pipe.vae
        del self.controlnet_pipe.unet
        del self.controlnet_pipe.feature_extractor

        self.controlnet_pipe.to(self.device)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        orig_height, orig_width, fps, frames_count = get_video_info(cap)
        print('| --- START VIDEO PROCESSING --- |')
        print(f'| HxW: {orig_height}x{orig_width} | FPS: {fps} | FRAMES COUNT: {frames_count} |')

        get_each = self.config.get('get_each', 1)
        processed_images = []

        for frame_index in tqdm(range(self.config['video_length'] * get_each)):
            ret, image = cap.read()
            if not ret or image is None:
                break
            
            if frame_index % get_each != 0:
                continue
            
            image = cv2.resize(image, (self.img_w, self.img_h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            condition_image = self.processor(Image.fromarray(image))
            processed_images.append(condition_image)

        return processed_images

    def generate_control_blocks(self, processed_images, prompt, negative_prompt, seed):
        print('| --- EXTRACT CONTROLNET FEATURES --- |')

        shape = (1, 4, self.video_length, self.img_h // 8, self.img_w // 8)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        control_latents = torch.randn(
            shape, 
            generator=generator, 
            device=self.device, 
            dtype=torch.float16
        )

        prompt_embeds = self.controlnet_pipe._encode_prompt(
                    prompt,
                    self.device,
                    1,
                    self.do_cfg,
                    negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    lora_scale=None,
                )

        self.controlnet_pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.controlnet_pipe.scheduler.timesteps

        control_blocks = []
        for t in tqdm(timesteps):
            down_block_samples = []
            mid_block_samples = []

            for img_index, image in enumerate(processed_images):
                latents = control_latents[:, :, img_index, :, :]
                image = self.controlnet_pipe.control_image_processor.preprocess(
                    image, 
                    height=self.img_h,
                    width=self.img_w
                ).to(dtype=torch.float32)
                
                image = image.repeat_interleave(1, dim=0)
                image = image.to(device=self.device, dtype=torch.float16)

                if self.do_cfg and not self.guess_mode:
                    image = torch.cat([image] * 2)

                latent_model_input = torch.cat([latents] * 2) if self.do_cfg else latents
                latent_model_input = self.controlnet_pipe.scheduler.scale_model_input(latent_model_input, t)

                if self.guess_mode and self.do_cfg:
                    control_model_input = latents
                    control_model_input = self.controlnet_pipe.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                down_block_res_samples, mid_block_res_sample = self.controlnet_pipe.controlnet(
                    control_model_input.to(self.device),
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds.to(self.device),
                    controlnet_cond=image,
                    conditioning_scale=1.0,
                    guess_mode=self.guess_mode,
                    return_dict=False,
                )

                if self.guess_mode and self.do_cfg:
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                down_block_samples.append([x.detach().cpu() for x in down_block_res_samples])
                mid_block_samples.append(mid_block_res_sample.detach().cpu())
                
            control_blocks.append({
                'down_block_samples': down_block_samples,
                'mid_block_samples': mid_block_samples,
            })

        return control_blocks

    def resort_features(self, control_blocks):
        mid_blocks = []
        down_blocks = []

        for c_block in control_blocks:
            d_blocks = defaultdict(list)
            for image_weights in c_block['down_block_samples']:
                for b_index, block_weights in enumerate(image_weights):
                    d_blocks[b_index] += block_weights.unsqueeze(0)
            
            down_block = []
            for _, value in d_blocks.items():
                down_block.append(torch.stack(value).permute(1, 2, 0, 3, 4))
            
            mid_block = torch.stack(c_block['mid_block_samples']).permute(1, 2, 0, 3, 4)

            down_blocks.append(down_block)
            mid_blocks.append(mid_block)

        return down_blocks, mid_blocks
    
    def __call__(self, video_path, prompt, negative_prompt, generator):
        processed_images = self.process_video(video_path)
        control_blocks = self.generate_control_blocks(
            processed_images, prompt, negative_prompt, generator)
        down_features, mid_features = self.resort_features(control_blocks)
        return down_features, mid_features
