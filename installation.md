

## Installation

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
`pip install transformers diffusers xformers imageio decord gdown einops omegaconf safetensors gradio wandb`
`pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl`

### How to 
- use .env 
- check https://education.civitai.com/beginners-guide-to-animatediff/

### Colab

```
# !rm AnimateDiff/StableDiffusion
# !git lfs install
# !git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/
```

```
# !wget https://civitai.com/api/download/models/159987 -P models/Motion_Module/ --content-disposition --no-check-certificate
# 
# !wget https://civitai.com/api/download/models/130072 -P models/Motion_Module/ --content-disposition --no-check-certificate 
```

### v2 required models
```
# !bash download_bashscripts/9-MotionModuleLoRA.sh
# !bash download_bashscripts/5-RealisticVision.sh
# !bash download_bashscripts/0-MotionModule.sh
# https://civitai.com/models/144354/temporaldiff-motion-module
# !wget https://civitai.com/api/download/models/160418 -P models/Motion_Module/ --content-disposition --no-check-certificate
# gem motion
# !wget https://civitai.com/api/download/models/126394 -P models/Motion_Module/ --content-disposition --no-check-certificate
# !wget https://civitai.com/api/download/models/48171 -P models/DreamBooth_LoRA/  --content-disposition --no-check-certificate
# !gdown 1h-yMX6HfR4ChljyiEPG1Ts9pSbI9sYfy -O models/Motion_Module/
```

### Install
```
# !rm -rf AnimateDiff
# !git clone https://github.com/hunzai/AnimateDiff.git
%cd AnimateDiff
!pwd
!git checkout updated
!git pull origin updated
```

`#!pip install -r requirements.txt`
