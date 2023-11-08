import os
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers import EulerDiscreteScheduler, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint


pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
inference_config_path = "configs/inference/inference.yaml"

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

examples = [
    # 1-ToonYou
    [
        "toonyou_beta3.safetensors", 
        "mm_sd_v14.ckpt", 
        "masterpiece, best quality, 1girl, solo, cherry blossoms, hanami, pink flower, white flower, spring season, wisteria, petals, flower, plum blossoms, outdoors, falling petals, white hair, black eyes",
        "worst quality, low quality, nsfw, logo",
        512, 512, "13204175718326964000"
    ],
    # 2-Lyriel
    [
        "lyriel_v16.safetensors", 
        "mm_sd_v15.ckpt", 
        "A forbidden castle high up in the mountains, pixel art, intricate details2, hdr, intricate details, hyperdetailed5, natural skin texture, hyperrealism, soft light, sharp, game art, key visual, surreal",
        "3d, cartoon, anime, sketches, worst quality, low quality, normal quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, bad anatomy, girl, loli, young, large breasts, red eyes, muscular",
        512, 512, "6681501646976930000"
    ],
    # 3-RCNZ
    [
        "rcnzCartoon3d_v10.safetensors", 
        "mm_sd_v14.ckpt", 
        "Jane Eyre with headphones, natural skin texture,4mm,k textures, soft cinematic light, adobe lightroom, photolab, hdr, intricate, elegant, highly detailed, sharp focus, cinematic look, soothing tones, insane details, intricate details, hyperdetailed, low contrast, soft cinematic light, dim colors, exposure blend, hdr, faded",
        "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
        512, 512, "2416282124261060"
    ],
    # 4-MajicMix
    [
        "majicmixRealistic_v5Preview.safetensors", 
        "mm_sd_v14.ckpt", 
        "1girl, offshoulder, light smile, shiny skin best quality, masterpiece, photorealistic",
        "bad hand, worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, watermark, moles",
        512, 512, "7132772652786303"
    ],
    # 5-RealisticVision
    [
        "realisticVisionV40_v20Novae.safetensors", 
        "mm_sd_v15.ckpt", 
        "photo of coastline, rocks, storm weather, wind, waves, lightning, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
        "blur, haze, deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",
        512, 512, "1490157606650685400"
    ]
]

# clean unrelated ckpts
ckpts = [
    "realisticVisionV40_v20Novae.safetensors",
    "majicmixRealistic_v5Preview.safetensors",
    "rcnzCartoon3d_v10.safetensors",
    "lyriel_v16.safetensors",
    "toonyou_beta3.safetensors"
]

for path in glob(os.path.join("models", "DreamBooth_LoRA", "*.safetensors")):
    for ckpt in ckpts:
        if path.endswith(ckpt): break
    else:
        print(f"### Cleaning {path} ...")
        os.system(f"rm -rf {path}")

# os.system(f"rm -rf {os.path.join('models', 'DreamBooth_LoRA', '*.safetensors')}")

# os.system(f"bash download_bashscripts/1-ToonYou.sh")
# os.system(f"bash download_bashscripts/2-Lyriel.sh")
# os.system(f"bash download_bashscripts/3-RcnzCartoon.sh")
# os.system(f"bash download_bashscripts/4-MajicMix.sh")
# os.system(f"bash download_bashscripts/5-RealisticVision.sh")

# clean Grdio cache
print(f"### Cleaning cached examples ...")
os.system(f"rm -rf gradio_cached_examples/")


class AnimateController:
    def __init__(self):
        
        # config dirs
        self.basedir                = os.getcwd()
        self.stable_diffusion_dir   = os.path.join(self.basedir, "models", "StableDiffusion")
        self.motion_module_dir      = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(self.basedir, "models", "DreamBooth_LoRA")
        self.savedir                = os.path.join(self.basedir, "samples")
        os.makedirs(self.savedir, exist_ok=True)

        self.base_model_list    = []
        self.motion_module_list = []
        
        self.selected_base_model    = None
        self.selected_motion_module = None
        
        self.refresh_motion_module()
        self.refresh_personalized_model()
        
        # config models
        self.inference_config      = OmegaConf.load(inference_config_path)

        self.tokenizer             = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        self.text_encoder          = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").cuda()
        self.vae                   = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").cuda()
        self.unet                  = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()
        
        self.update_base_model(self.base_model_list[0])
        self.update_motion_module(self.motion_module_list[0])
        
        
    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        base_model_list = glob(os.path.join(self.personalized_model_dir, "*.safetensors"))
        self.base_model_list = [os.path.basename(p) for p in base_model_list]


    def update_base_model(self, base_model_dropdown):
        self.selected_base_model = base_model_dropdown
        
        base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
        base_model_state_dict = {}
        with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
            for key in f.keys(): base_model_state_dict[key] = f.get_tensor(key)
                
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_model_state_dict, self.vae.config)
        self.vae.load_state_dict(converted_vae_checkpoint)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
        self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        self.selected_motion_module = motion_module_dropdown
        
        motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
        motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
        _, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0
        return gr.Dropdown.update()
    
    
    def animate(
        self,
        base_model_dropdown,
        motion_module_dropdown,
        prompt_textbox,
        negative_prompt_textbox,
        width_slider,
        height_slider,
        seed_textbox,
    ):
        if self.selected_base_model != base_model_dropdown: self.update_base_model(base_model_dropdown)
        if self.selected_motion_module != motion_module_dropdown: self.update_motion_module(motion_module_dropdown)
        
        if is_xformers_available(): self.unet.enable_xformers_memory_efficient_attention()

        pipeline = AnimationPipeline(
            vae=self.vae, text_encoder=self.text_encoder, tokenizer=self.tokenizer, unet=self.unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs))
        ).to("cuda")
        
        if int(seed_textbox) > 0: seed = int(seed_textbox)
        else: seed = random.randint(1, 1e16)
        torch.manual_seed(int(seed))
        
        assert seed == torch.initial_seed()
        print(f"### seed: {seed}")
        
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        
        sample = pipeline(
            prompt_textbox,
            negative_prompt     = negative_prompt_textbox,
            num_inference_steps = 25,
            guidance_scale      = 8.,
            width               = width_slider,
            height              = height_slider,
            video_length        = 16,
            generator           = generator,
        ).videos

        save_sample_path = os.path.join(self.savedir, f"sample.mp4")
        save_videos_grid(sample, save_sample_path)
    
        json_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "width": width_slider,
            "height": height_slider,
            "seed": seed,
            "base_model": base_model_dropdown,
            "motion_module": motion_module_dropdown,
        }
        return gr.Video.update(value=save_sample_path), gr.Json.update(value=json_config)
        

controller = AnimateController()

quick_start_en = """
            ### Quick Start
            1. Select desired `Base DreamBooth Model`.
            2. Select `Motion Module` from `mm_sd_v14.ckpt` and `mm_sd_v15.ckpt`. We recommend trying both of them for the best results.
            3. Provide `Prompt` and `Negative Prompt` for each model. You are encouraged to refer to each model's webpage on CivitAI to learn how to write prompts for them. Below are the DreamBooth models in this demo. Click to visit their homepage.
                - [`toonyou_beta3.safetensors`](https://civitai.com/models/30240?modelVersionId=78775)
                - [`lyriel_v16.safetensors`](https://civitai.com/models/22922/lyriel)
                - [`rcnzCartoon3d_v10.safetensors`](https://civitai.com/models/66347?modelVersionId=71009)
                - [`majicmixRealistic_v5Preview.safetensors`](https://civitai.com/models/43331?modelVersionId=79068)
                - [`realisticVisionV20_v20.safetensors`](https://civitai.com/models/4201?modelVersionId=29460)
            4. Click `Generate`, wait for ~1 min, and enjoy.
            """

quick_start_cn = """
            ### 快速开始
            1. 选择所需的`基础DreamBooth模型`。
            2. 从mm_sd_v14.ckpt和mm_sd_v15.ckpt中选择Motion模块。我们建议尝试两者取得最佳效果。
            3. 为每个模型提供`提示`和`负提示`。鼓励您参考CivitAI上的每个模型的网页，以了解如何为它们编写提示。以下是此演示中的DreamBooth模型，点击访问他们的主页：
                - [`toonyou_beta3.safetensors`](https://civitai.com/models/30240?modelVersionId=78775)
                - [`lyriel_v16.safetensors`](https://civitai.com/models/22922/lyriel)
                - [`rcnzCartoon3d_v10.safetensors`](https://civitai.com/models/66347?modelVersionId=71009)
                - [`majicmixRealistic_v5Preview.safetensors`](https://civitai.com/models/43331?modelVersionId=79068)
                - [`realisticVisionV20_v20.safetensors`](https://civitai.com/models/4201?modelVersionId=29460)
            4. 点击`生成`按钮，等待约1分钟，然后享受生成的结果。
            """

def change_language(lang):
    if lang == '中文':
        lang = gr.Button(value='English')
        quick_start_controller = gr.Markdown(value=quick_start_cn)
        base_model_dropdown = gr.Dropdown(label="基础DreamBooth模型")
        motion_module_dropdown = gr.Dropdown(label="Motion模块")
        prompt_textbox = gr.Textbox(label="提示")
        negative_prompt_textbox = gr.Textbox(label="负提示")
        width_slider = gr.Slider(label="宽度")
        height_slider = gr.Slider(label="高度")
        seed_textbox = gr.Textbox(label="种子")
        generate_button = gr.Button(value="生成")
        result_video = gr.Video(label="已生成的动画")
        json_config = gr.JSON(label="配置")
        advance_settings = gr.Accordion(label="高级设置")
    elif lang == 'English':
        lang = gr.Button(value='中文')
        quick_start_controller = gr.Markdown(value=quick_start_en)
        base_model_dropdown = gr.Dropdown(label="Base DreamBooth Model")
        motion_module_dropdown = gr.Dropdown(label="Motion Module")
        prompt_textbox = gr.Textbox(label="Prompt")
        negative_prompt_textbox = gr.Textbox(label="Negative Prompt")
        width_slider = gr.Slider(label="Width")
        height_slider = gr.Slider(label="Height")
        seed_textbox = gr.Textbox(label="Seed")
        generate_button = gr.Button(value="Generate")
        result_video = gr.Video(label="Generated Animation")
        json_config = gr.JSON(label="Config")
        advance_settings = gr.Accordion(label="Advance")

    return [lang, quick_start_controller, base_model_dropdown,
            motion_module_dropdown, prompt_textbox, negative_prompt_textbox, width_slider, height_slider,
            seed_textbox, generate_button, result_video, json_config, advance_settings]


def ui():
    with gr.Blocks(css=css) as demo:
        with gr.Row():
            with gr.Column(scale=20):
                gr.Markdown(
                    """
                    # AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning
                    Yuwei Guo, Ceyuan Yang*, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, Bo Dai (*Corresponding Author)<br>
                    [Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://animatediff.github.io/) | [Github](https://github.com/guoyww/animatediff/)
                    """
                )
            with gr.Column(scale=1, min_width=100):
                lang_btn = gr.Button(value="中文")

        quick_start_controller = gr.Markdown(quick_start_en)

        with gr.Row():
            with gr.Column():
                base_model_dropdown     = gr.Dropdown( label="Base DreamBooth Model", choices=controller.base_model_list,    value=controller.base_model_list[0],    interactive=True )
                motion_module_dropdown  = gr.Dropdown( label="Motion Module",  choices=controller.motion_module_list, value=controller.motion_module_list[0], interactive=True )

                base_model_dropdown.change(fn=controller.update_base_model,       inputs=[base_model_dropdown],    outputs=[base_model_dropdown])
                motion_module_dropdown.change(fn=controller.update_motion_module, inputs=[motion_module_dropdown], outputs=[motion_module_dropdown])

                prompt_textbox          = gr.Textbox( label="Prompt",          lines=3 )
                negative_prompt_textbox = gr.Textbox( label="Negative Prompt", lines=3, value="worst quality, low quality, nsfw, logo")

                with gr.Accordion("Advance", open=False) as advance_settings:
                    with gr.Row():
                        width_slider  = gr.Slider(  label="Width",  value=512, minimum=256, maximum=1024, step=64 )
                        height_slider = gr.Slider(  label="Height", value=512, minimum=256, maximum=1024, step=64 )
                    with gr.Row():
                        seed_textbox = gr.Textbox( label="Seed",  value=-1)
                        seed_button  = gr.Button(value="\U0001F3B2", elem_classes="toolbutton")
                        seed_button.click(fn=lambda: gr.Textbox.update(value=random.randint(1, 1e16)), inputs=[], outputs=[seed_textbox])

                generate_button = gr.Button( value="Generate", variant='primary' )

            with gr.Column():
                result_video = gr.Video( label="Generated Animation", interactive=False )
                json_config  = gr.Json( label="Config", value=None )

            inputs  = [base_model_dropdown, motion_module_dropdown, prompt_textbox, negative_prompt_textbox, width_slider, height_slider, seed_textbox]
            outputs = [result_video, json_config]
            
            generate_button.click( fn=controller.animate, inputs=inputs, outputs=outputs )
                
        gr.Examples( fn=controller.animate, examples=examples, inputs=inputs, outputs=outputs, cache_examples=True )
        lang_btn.click(change_language, inputs=lang_btn, outputs=[lang_btn, quick_start_controller, base_model_dropdown, motion_module_dropdown,
                    prompt_textbox, negative_prompt_textbox, width_slider, height_slider,
                    seed_textbox, generate_button, result_video, json_config, advance_settings])
        
    return demo


if __name__ == "__main__":
    demo = ui()
    queue_max_size = int(os.environ.get("queue_max_size", 16))
    queue_concurrency_count = int(os.environ.get("queue_concurrency_count", 2))
    max_threads = int(os.environ.get("max_threads", 20))
    api_open = bool(os.environ.get("api_open", False))
    demo.queue(max_size=queue_max_size, concurrency_count=queue_concurrency_count, api_open=api_open)
    demo.launch(max_threads=max_threads)
