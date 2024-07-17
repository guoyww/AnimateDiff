
import os
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import DDIMScheduler, EulerDiscreteScheduler, PNDMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid, load_weights, auto_download, MOTION_MODULES, BACKUP_DREAMBOOTH_MODELS
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
import pdb


sample_idx = 0
scheduler_dict = {
    "DDIM": DDIMScheduler,
    "Euler": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

PRETRAINED_SD = "runwayml/stable-diffusion-v1-5"

default_motion_module = "v3_sd15_mm.ckpt"
default_inference_config = "configs/inference/inference-v3.yaml"
default_dreambooth_model = "realisticVisionV60B1_v51VAE.safetensors"
default_prompt = "b&w photo of 42 y.o man in black clothes, bald, face, half body, body, high detailed skin, skin pores, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
default_n_prompt = "semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
default_seed = 8893659352891878017

device = "cuda" if torch.cuda.is_available() else "cpu"


class AnimateController:
    def __init__(self):
        # config dirs
        self.basedir = os.getcwd()
        self.stable_diffusion_dir = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.savedir = os.path.join(self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list = [PRETRAINED_SD]
        self.motion_module_list = MOTION_MODULES
        self.personalized_model_list = BACKUP_DREAMBOOTH_MODELS
        
        # config models
        self.pipeline = None
        # self.lora_model_state_dict = {}
        
        self.refresh_stable_diffusion()
        self.refresh_personalized_model()
        
        # default setting
        self.update_pipeline(
            stable_diffusion_dropdown=PRETRAINED_SD,
            motion_module_dropdown=default_motion_module,
            base_model_dropdown=default_dreambooth_model,
            sampler_dropdown="DDIM",
        )

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = [PRETRAINED_SD] + glob(os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = BACKUP_DREAMBOOTH_MODELS + [os.path.basename(p) for p in personalized_model_list if os.path.basename(p) not in BACKUP_DREAMBOOTH_MODELS]

    # for dropdown update
    def update_pipeline(
        self,
        stable_diffusion_dropdown,
        motion_module_dropdown,
        base_model_dropdown="",
        lora_model_dropdown="none",
        lora_alpha_dropdown="0.6",
        sampler_dropdown="DDIM",
    ):
        if "v2" in motion_module_dropdown:
            inference_config = "configs/inference/inference-v2.yaml"
        elif "v3" in motion_module_dropdown:
            inference_config = "configs/inference/inference-v3.yaml"
        else:
            inference_config = "configs/inference/inference-v1.yaml"

        unet = UNet3DConditionModel.from_pretrained_2d(
            stable_diffusion_dropdown, subfolder="unet", 
            unet_additional_kwargs=OmegaConf.load(inference_config).unet_additional_kwargs
        )
        if is_xformers_available() and torch.cuda.is_available():
            unet.enable_xformers_memory_efficient_attention()

        noise_scheduler_cls = scheduler_dict[sampler_dropdown]
        noise_scheduler_kwargs = OmegaConf.load(inference_config).noise_scheduler_kwargs
        if noise_scheduler_cls == EulerDiscreteScheduler:
            noise_scheduler_kwargs.pop("steps_offset")
            noise_scheduler_kwargs.pop("clip_sample")
        elif noise_scheduler_cls == PNDMScheduler:
            noise_scheduler_kwargs.pop("clip_sample")

        pipeline = AnimationPipeline(
            unet=unet,
            vae=AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae"), 
            text_encoder=CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder"), 
            tokenizer=CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer"), 
            scheduler=noise_scheduler_cls(**noise_scheduler_kwargs),
        )

        pipeline = load_weights(
            pipeline,
            motion_module_path=os.path.join(self.motion_module_dir, motion_module_dropdown),
            dreambooth_model_path=os.path.join(self.personalized_model_dir, base_model_dropdown) if base_model_dropdown != "" else "",
            lora_model_path=os.path.join(self.personalized_model_dir, lora_model_dropdown) if lora_model_dropdown != "none" else "",
            lora_alpha=float(lora_alpha_dropdown),
        )

        pipeline.to(device)
        self.pipeline = pipeline
        print("done.")

        return gr.Dropdown.update()

    def update_pipeline_alpha(
        self,
        stable_diffusion_dropdown,
        motion_module_dropdown,
        base_model_dropdown="",
        lora_model_dropdown="none",
        lora_alpha_dropdown="0.6",
        sampler_dropdown="DDIM",
    ):
        if lora_model_dropdown == "none":
            return gr.Slider.update()

        self.update_pipeline(
            stable_diffusion_dropdown=stable_diffusion_dropdown,
            motion_module_dropdown=motion_module_dropdown,
            base_model_dropdown=base_model_dropdown,
            lora_model_dropdown=lora_model_dropdown,
            lora_alpha_dropdown=lora_alpha_dropdown,
            sampler_dropdown=sampler_dropdown,
        )

        return gr.Slider.update()


    @torch.no_grad()
    def animate(
        self,
        prompt_textbox,
        negative_prompt_textbox,
        sampler_dropdown,
        sample_step_slider,
        width_slider,
        length_slider,
        height_slider,
        cfg_scale_slider,
        seed_textbox,
    ):
        if int(seed_textbox) != -1:
            torch.manual_seed(int(seed_textbox))
        else:
            torch.seed()
        seed = torch.initial_seed()
        
        sample = self.pipeline(
            prompt_textbox,
            negative_prompt = negative_prompt_textbox,
            num_inference_steps = sample_step_slider,
            guidance_scale = cfg_scale_slider,
            width = width_slider,
            height = height_slider,
            video_length = length_slider,
        ).videos

        save_sample_path = os.path.join(self.savedir_sample, f"{sample_idx}.mp4")
        save_videos_grid(sample, save_sample_path)
    
        sample_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "sampler": sampler_dropdown,
            "num_inference_steps": sample_step_slider,
            "guidance_scale": cfg_scale_slider,
            "width": width_slider,
            "height": height_slider,
            "video_length": length_slider,
            "seed": seed
        }

        json_str = json.dumps(sample_config, indent=4)
        with open(os.path.join(self.savedir, "logs.json"), "a") as f:
            f.write(json_str)
            f.write("\n\n")
            
        return gr.Video.update(value=save_sample_path)
        

controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning
            Yuwei Guo, Ceyuan Yang✝, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai (✝Corresponding Author)<br>
            [Paper](https://arxiv.org/abs/2307.04725) | [Webpage](https://animatediff.github.io/) | [Github](https://github.com/guoyww/animatediff/)
            """
        )
        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 1. Model Checkpoints
                """
            )
            with gr.Row():
                stable_diffusion_dropdown = gr.Dropdown(
                    label="Pretrained Model Path",
                    choices=controller.stable_diffusion_list,
                    value=PRETRAINED_SD,
                    interactive=True,
                )
                
            with gr.Row():
                motion_module_dropdown = gr.Dropdown(
                    label="Select motion module",
                    choices=controller.motion_module_list,
                    value=default_motion_module,
                    interactive=True,
                )
                                
                base_model_dropdown = gr.Dropdown(
                    label="Select base Dreambooth model (required)",
                    choices=controller.personalized_model_list,
                    value=default_dreambooth_model,
                    interactive=True,
                )
                
                lora_model_dropdown = gr.Dropdown(
                    label="Select LoRA model (optional)",
                    choices=["none"] + controller.personalized_model_list,
                    value="none",
                    interactive=True,
                )
                
                lora_alpha_dropdown = gr.Dropdown(
                    label="LoRA alpha", 
                    choices=["0.", "0.2", "0.4", "0.6", "0.8", "1.0"],
                    value="0.6",
                    interactive=True,
                )
                
                personalized_refresh_button = gr.Button(value="\U0001F503", elem_classes="toolbutton")
                def update_personalized_model():
                    controller.refresh_stable_diffusion()
                    controller.refresh_personalized_model()
                    return [
                        gr.Dropdown.update(choices=controller.stable_diffusion_list),
                        gr.Dropdown.update(choices=controller.personalized_model_list),
                        gr.Dropdown.update(choices=["none"] + controller.personalized_model_list)
                    ]
                personalized_refresh_button.click(fn=update_personalized_model, inputs=[], outputs=[stable_diffusion_dropdown, base_model_dropdown, lora_model_dropdown])

        with gr.Column(variant="panel"):
            gr.Markdown(
                """
                ### 2. Configs for AnimateDiff.
                """
            )
            prompt_textbox = gr.Textbox(label="Prompt", lines=2, value=default_prompt)
            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value=default_n_prompt)

            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Row():
                        sampler_dropdown = gr.Dropdown(label="Sampling method", choices=list(scheduler_dict.keys()), value=list(scheduler_dict.keys())[0])
                        sample_step_slider = gr.Slider(label="Sampling steps", value=25, minimum=10, maximum=100, step=1)
                        
                    width_slider = gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=64)
                    height_slider = gr.Slider(label="Height", value=512, minimum=256, maximum=1024, step=64)
                    length_slider = gr.Slider(label="Animation length (default: 16)", value=16, minimum=8, maximum=24, step=1)
                    cfg_scale_slider = gr.Slider(label="CFG Scale", value=8.0, minimum=0, maximum=20)
                    
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed (-1 for random seed)", value=default_seed)
                        seed_button = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e8)), inputs=[], outputs=[seed_textbox])
            
                    generate_button = gr.Button(value="Generate", variant='primary')
                    
                result_video = gr.Video(label="Generated Animation", interactive=False)

            # update method
            stable_diffusion_dropdown.change(fn=controller.update_pipeline, inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[stable_diffusion_dropdown])
            motion_module_dropdown.change(fn=controller.update_pipeline,    inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[motion_module_dropdown])
            base_model_dropdown.change(fn=controller.update_pipeline,       inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[base_model_dropdown])
            lora_model_dropdown.change(fn=controller.update_pipeline,       inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[lora_model_dropdown])
            lora_alpha_dropdown.change(fn=controller.update_pipeline_alpha, inputs=[stable_diffusion_dropdown, motion_module_dropdown, base_model_dropdown, lora_model_dropdown, lora_alpha_dropdown, sampler_dropdown], outputs=[lora_alpha_dropdown])

            generate_button.click(
                fn=controller.animate,
                inputs=[
                    prompt_textbox, 
                    negative_prompt_textbox, 
                    sampler_dropdown, 
                    sample_step_slider, 
                    width_slider, 
                    length_slider, 
                    height_slider, 
                    cfg_scale_slider, 
                    seed_textbox,
                ],
                outputs=[result_video]
            )
            
    return demo


if __name__ == "__main__":
    demo = ui()
    demo.launch(share=True)
