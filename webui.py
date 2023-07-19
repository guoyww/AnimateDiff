import argparse
import gradio as gr
import copy
import os
import glob

import utils as help

sep = '\\' if help.is_windows() else '/'
all_motion_model_opts = ["mm_sd_v14.ckpt", "mm_sd_v15.ckpt"]

def get_available_motion_models():
    motion_model_opts_path = os.path.join(os.getcwd(), os.path.join("models", "Motion_Module"))
    motion_model_opts = sorted([ckpt for ckpt in glob.glob(os.path.join(motion_model_opts_path, f"*.ckpt"))])
    return motion_model_opts

def get_available_sd_models():
    sd_model_opts_path = os.path.join(os.getcwd(), os.path.join("models", "StableDiffusion"))
    sd_model_opts = sorted([safetensor.split(sep)[-1] for safetensor in glob.glob(os.path.join(sd_model_opts_path, f"*.safetensors"))])
    return sd_model_opts

def get_available_db_models():
    db_model_opts_path = os.path.join(os.getcwd(), os.path.join("models", "DreamBooth_LoRA"))
    db_model_opts = sorted([safetensor.split(sep)[-1] for safetensor in glob.glob(os.path.join(db_model_opts_path, f"*.safetensors"))])
    return db_model_opts

def get_db_config():
    prompt_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "prompts")
    return sorted([(prompt_yaml.split(sep)[-1]) for prompt_yaml in glob.glob(os.path.join(prompt_configs_path, f"*.yaml"))])

def get_sd_config():
    inference_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "inference")
    return sorted([(inference_yaml.split(sep)[-1]) for inference_yaml in glob.glob(os.path.join(inference_configs_path, f"*.yaml"))])

def set_motion_model(menu_opt: gr.SelectData):
    model_name = menu_opt.value
    motion_model_opts_path = os.path.join(os.getcwd(), os.path.join("models", "Motion_Module"))
    motion_model_opts = sorted([ckpt for ckpt in glob.glob(os.path.join(motion_model_opts_path, f"*.ckpt"))])
    motion_model_map = {"mm_sd_v14.ckpt": "1RqkQuGPaCO5sGZ6V6KZ-jUWmsRu48Kdq",
                        "mm_sd_v15.ckpt": "1ql0g_Ys4UCz2RnokYlBjyOYPbttbIpbu"}
    if not os.path.join(motion_model_opts_path, model_name) in motion_model_opts: # download
        help.download_from_drive_gdown(motion_model_map[model_name], os.path.join(motion_model_opts_path, model_name))
    return gr.update(value=os.path.join(motion_model_opts_path, model_name)) # model path

def set_sd_model(menu_opt: gr.SelectData):
    model_name = menu_opt.value
    sd_model_opts_path = os.path.join(os.getcwd(), os.path.join("models", "StableDiffusion"))
    return gr.update(value=os.path.join(sd_model_opts_path, model_name)), gr.update(value=os.path.join(sd_model_opts_path, model_name)) # sd path, pretrained path

def set_db_model(menu_opt: gr.SelectData):
    model_name = menu_opt.value
    db_model_opts_path = os.path.join(os.getcwd(), os.path.join("models", "DreamBooth_LoRA"))
    return gr.update(value=os.path.join(db_model_opts_path, model_name)) # db path

def update_available_sd_models():
    sd_model_opts_path = os.path.join(os.getcwd(), os.path.join("models", "StableDiffusion"))
    sd_model_opts = sorted([safetensor.split(sep)[-1] for safetensor in glob.glob(os.path.join(sd_model_opts_path, f"*.safetensors"))])
    return gr.update(choices=sd_model_opts)

def update_available_db_models():
    db_model_opts_path = os.path.join(os.getcwd(), os.path.join("models", "DreamBooth_LoRA"))
    db_model_opts = sorted([safetensor.split(sep)[-1] for safetensor in glob.glob(os.path.join(db_model_opts_path, f"*.safetensors"))])
    return gr.update(choices=db_model_opts)

def update_sd_config():
    inference_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "inference")
    return gr.update(choices=sorted([(inference_yaml.split(sep)[-1]) for inference_yaml in glob.glob(os.path.join(inference_configs_path, f"*.yaml"))]))

def update_db_config():
    prompt_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "prompts")
    return gr.update(choices=sorted([(prompt_yaml.split(sep)[-1]) for prompt_yaml in glob.glob(os.path.join(prompt_configs_path, f"*.yaml"))]))

def load_db_config(filename: gr.SelectData):
    filename = filename.value

    global prompt_config_dict
    prompt_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "prompts")
    # populate the dictionary
    prompt_config_dict = help.yaml_to_dict(os.path.join(prompt_configs_path, f"{filename}"))

    name_only = list(prompt_config_dict.keys())[0]
    help.verbose_print(f"Config Key Name:\t{name_only}")

    # return populated UI components
    config_name = name_only

    motion_model_path = list(prompt_config_dict[name_only]["motion_module"])[0]


    base_path = str(prompt_config_dict[name_only]["base"])
    db_path = str(prompt_config_dict[name_only]["path"])
    steps = int(prompt_config_dict[name_only]["steps"])
    guidance_scale = float(prompt_config_dict[name_only]["guidance_scale"])
    lora_alpha = float(prompt_config_dict[name_only]["lora_alpha"]) if "lora_alpha" in prompt_config_dict[name_only] else 1.0

    seed_list = list(prompt_config_dict[name_only]["seed"])
    prompt_list = list(prompt_config_dict[name_only]["prompt"])
    n_prompt_list = list(prompt_config_dict[name_only]["n_prompt"])

    seed1 = str(seed_list[0]) if len(seed_list) > 0 else "-1"
    prompt1 = str(prompt_list[0]) if len(prompt_list) > 0 else ""
    n_prompt1 = str(n_prompt_list[0]) if len(n_prompt_list) > 0 else ""
    seed2 = str(seed_list[1]) if len(seed_list) > 1 else "-1"
    prompt2 = str(prompt_list[1]) if len(prompt_list) > 1 else ""
    n_prompt2 = str(n_prompt_list[1]) if len(n_prompt_list) > 1 else ""
    seed3 = str(seed_list[2]) if len(seed_list) > 2 else "-1"
    prompt3 = str(prompt_list[2]) if len(prompt_list) > 2 else ""
    n_prompt3 = str(n_prompt_list[2]) if len(n_prompt_list) > 2 else ""
    seed4 = str(seed_list[3]) if len(seed_list) > 3 else "-1"
    prompt4 = str(prompt_list[3]) if len(prompt_list) > 3 else ""
    n_prompt4 = str(n_prompt_list[3]) if len(n_prompt_list) > 3 else ""
    help.verbose_print(f"Done Loading Prompt Config!")

    motion_model_dropdown = gr.update(value=motion_model_path.split(sep)[-1])
    sd_model_dropdown = gr.update(value=base_path.split(sep)[-1])
    db_model_dropdown = gr.update(value=db_path.split(sep)[-1])
    pretrained_model_path = gr.update(value=base_path)

    return config_name, motion_model_path, base_path, db_path, steps, guidance_scale, lora_alpha, \
           seed1, prompt1, n_prompt1, seed2, prompt2, n_prompt2, seed3, prompt3, n_prompt3, seed4, prompt4, n_prompt4, \
           motion_model_dropdown, sd_model_dropdown, db_model_dropdown, pretrained_model_path

def save_db_config(filename):
    global prompt_config_dict
    prompt_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "prompts")
    help.dict_to_yaml(copy.deepcopy(prompt_config_dict), os.path.join(prompt_configs_path, f"{filename}.yaml"))
    help.verbose_print(f"Done Creating NEW Prompt Config!")

def save_prompt_dict(config_name, motion_model_path, base_path, db_path, steps, guidance_scale, lora_alpha,
           seed1, prompt1, n_prompt1, seed2, prompt2, n_prompt2, seed3, prompt3, n_prompt3, seed4, prompt4, n_prompt4):
    global prompt_config_dict
    prompt_config_dict[config_name] = {}
    prompt_config_dict[config_name]["base"] = base_path
    prompt_config_dict[config_name]["path"] = db_path

    prompt_config_dict[config_name]["motion_module"] = []
    prompt_config_dict[config_name]["motion_module"].append(motion_model_path)

    prompt_config_dict[config_name]["seed"] = [0] * 4
    prompt_config_dict[config_name]["steps"] = steps
    prompt_config_dict[config_name]["guidance_scale"] = guidance_scale
    prompt_config_dict[config_name]["lora_alpha"] = lora_alpha


    prompt_config_dict[config_name]["prompt"] = [""]*4
    prompt_config_dict[config_name]["n_prompt"] = [""]*4

    prompt_config_dict[config_name]["seed"][0] = int(seed1)
    prompt_config_dict[config_name]["prompt"][0] = prompt1
    prompt_config_dict[config_name]["n_prompt"][0] = n_prompt1
    prompt_config_dict[config_name]["seed"][1] = int(seed2)
    prompt_config_dict[config_name]["prompt"][1] = prompt2
    prompt_config_dict[config_name]["n_prompt"][1] = n_prompt2
    prompt_config_dict[config_name]["seed"][2] = int(seed3)
    prompt_config_dict[config_name]["prompt"][2] = prompt3
    prompt_config_dict[config_name]["n_prompt"][2] = n_prompt3
    prompt_config_dict[config_name]["seed"][3] = int(seed4)
    prompt_config_dict[config_name]["prompt"][3] = prompt4
    prompt_config_dict[config_name]["n_prompt"][3] = n_prompt4

    prompt_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "prompts")
    help.dict_to_yaml(copy.deepcopy(prompt_config_dict), os.path.join(prompt_configs_path, f"{config_name}.yaml"))
    help.verbose_print(f"Done Updating Prompt Config!")

def animate(pretrained_model_path, frame_count, width, height, inference_yaml_select, prompt_yaml_select):
    prompt_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "prompts")
    inference_configs_path = os.path.join(os.path.join(os.getcwd(), "configs"), "inference")

    command_str = f"python -m scripts.animate --config {os.path.join(prompt_configs_path, prompt_yaml_select)}"
    if pretrained_model_path is not None and len(pretrained_model_path) > 0:
        command_str += f" --pretrained_model_path {pretrained_model_path}"
    command_str += f" --L {frame_count}"
    command_str += f" --W {width}"
    command_str += f" --H {height}"
    if inference_yaml_select is not None and len(inference_yaml_select) > 0:
        command_str += f" --inference_config {os.path.join(inference_configs_path, inference_yaml_select)}"

    help.verbose_print(f"Running Command:\t{command_str}")
    for line in help.execute(command_str.split(" ")):
        help.verbose_print(line)
    help.verbose_print(f"Done Generating!")

def build_ui():
    with gr.Blocks() as demo:
        with gr.Tab("Model Selection & Setup"):
            with gr.Row():
                motion_model_dropdown = gr.Dropdown(interactive=True, label="Select Motion Model", info="Downloads model if not present", choices=all_motion_model_opts)
                sd_model_dropdown = gr.Dropdown(interactive=True, label="Select Stable Diffusion Model", info="At user/s discretion to download", choices=get_available_sd_models())
                db_model_dropdown = gr.Dropdown(interactive=True, label="Select LoRA/Dreambooth Model", info="At user/s discretion to download", choices=get_available_db_models())
            with gr.Row():
                pretrained_model_path = gr.Textbox(info="Pretrained Model Path", interactive=True, show_label=False)
            with gr.Row():
                frame_count = gr.Slider(info="Total Frames", minimum=0, maximum=1000, step=1, value=16, show_label=False)
                width = gr.Slider(info="Width", minimum=0, maximum=4096, step=1, value=512, show_label=False)
                height = gr.Slider(info="Height", minimum=0, maximum=4096, step=1, value=512, show_label=False)
            with gr.Row():
                inference_yaml_select = gr.Dropdown(info='YAML Select', interactive=True, choices=get_sd_config(), show_label=False)
            animate_button = gr.Button(value="Generate", variant='primary')

        with gr.Tab("LoRA/Dreambooth Prompt Config"):
            with gr.Row():
                config_save = gr.Button(value="Apply & Save Settings", variant='primary')
                create_prompt_yaml = gr.Button(value="Create Prompt Config", variant='secondary')
            with gr.Row():
                prompt_yaml_select = gr.Dropdown(info='YAML Select', interactive=True, choices=get_db_config(), show_label=False)
                config_name = gr.Textbox(info="Config Name", interactive=True, show_label=False)
                motion_model_path = gr.Textbox(info="Motion Model Path", interactive=True, show_label=False)
            with gr.Row():
                base_path = gr.Textbox(info="Base Model Path", interactive=True, show_label=False)
                db_path = gr.Textbox(info="LoRA/Dreambooth Path", interactive=True, show_label=False)
            with gr.Row():
                steps = gr.Slider(info="Steps", minimum=0, maximum=1000, step=1, value=25, show_label=False)
            with gr.Row():
                guidance_scale = gr.Slider(info="Guidance Scale", minimum=0.0, maximum=100.0, step=0.05, value=6.5, show_label=False)
                lora_alpha = gr.Slider(info="LoRA Alpha", minimum=0.0, maximum=1.0, step=0.025, value=1.0, show_label=False)
            with gr.Accordion("Prompt 1", visible=True, open=False):
                with gr.Column():
                    seed1 = gr.Textbox(info="Seed", interactive=True, show_label=False)
                    prompt1 = gr.Textbox(info="Prompt", interactive=True, show_label=False)
                    n_prompt1 = gr.Textbox(info="Negative Prompt", interactive=True, show_label=False)
            with gr.Accordion("Prompt 2", visible=True, open=False):
                with gr.Column():
                    seed2 = gr.Textbox(info="Seed", interactive=True, show_label=False)
                    prompt2 = gr.Textbox(info="Prompt", interactive=True, show_label=False)
                    n_prompt2 = gr.Textbox(info="Negative Prompt", interactive=True, show_label=False)
            with gr.Accordion("Prompt 3", visible=True, open=False):
                with gr.Column():
                    seed3 = gr.Textbox(info="Seed", interactive=True, show_label=False)
                    prompt3 = gr.Textbox(info="Prompt", interactive=True, show_label=False)
                    n_prompt3 = gr.Textbox(info="Negative Prompt", interactive=True, show_label=False)
            with gr.Accordion("Prompt 4", visible=True, open=False):
                with gr.Column():
                    seed4 = gr.Textbox(info="Seed", interactive=True, show_label=False)
                    prompt4 = gr.Textbox(info="Prompt", interactive=True, show_label=False)
                    n_prompt4 = gr.Textbox(info="Negative Prompt", interactive=True, show_label=False)

        motion_model_dropdown.select(fn=set_motion_model, inputs=[], outputs=[motion_model_path])
        sd_model_dropdown.select(fn=set_sd_model, inputs=[], outputs=[base_path, pretrained_model_path])
        db_model_dropdown.select(fn=set_db_model, inputs=[], outputs=[db_path])
        prompt_yaml_select.select(fn=load_db_config, inputs=[],
                                  outputs=[config_name, motion_model_path, base_path, db_path, steps, guidance_scale, lora_alpha,
                                  seed1, prompt1, n_prompt1, seed2, prompt2, n_prompt2, seed3, prompt3, n_prompt3,
                                  seed4, prompt4, n_prompt4, motion_model_dropdown, sd_model_dropdown,
                                  db_model_dropdown, pretrained_model_path]).then(
                                  fn=update_db_config, inputs=[], outputs=[prompt_yaml_select])
        create_prompt_yaml.click(fn=save_db_config, inputs=[config_name], outputs=[])
        config_save.click(fn=save_prompt_dict, inputs=[config_name, motion_model_path, base_path, db_path, steps, guidance_scale, lora_alpha,
           seed1, prompt1, n_prompt1, seed2, prompt2, n_prompt2, seed3, prompt3, n_prompt3, seed4, prompt4, n_prompt4],
                          outputs=[])
        animate_button.click(fn=animate, inputs=[pretrained_model_path, frame_count, width, height, inference_yaml_select, prompt_yaml_select], outputs=[])
    return demo

def UI(**kwargs):
    # Show the interface
    launch_kwargs = {}
    if not kwargs.get('username', None) == '':
        launch_kwargs['auth'] = (
            kwargs.get('username', None),
            kwargs.get('password', None),
        )
    if kwargs.get('server_port', 0) > 0:
        launch_kwargs['server_port'] = kwargs.get('server_port', 0)
    if kwargs.get('share', True):
        launch_kwargs['share'] = True

    print(launch_kwargs)
    demo.queue().launch(**launch_kwargs)

if __name__ == "__main__":
    # init client & server connection
    HOST = "127.0.0.1"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Share live gradio link',
    )

    args = parser.parse_args()
    demo = build_ui()

    global prompt_config_dict
    prompt_config_dict = {}

    help.verbose_print(f"Motion models available to use:\t{get_available_motion_models()}")
    help.verbose_print(f"Stable Diffusion models available to use:\t{get_available_sd_models()}")
    help.verbose_print(f"LoRA/Dreambooth models available to use:\t{get_available_db_models()}")

    UI(
        username=args.username,
        password=args.password,
        server_port=args.server_port,
        share=args.share,
    )
