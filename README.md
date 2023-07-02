# AnimateDiff

This repository is the official implementation of [AnimateDiff].

**[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning]()**
</br>
Yuwei Guo,
Ceyuan Yang,
Anyi Rao,
Yaohui Wang,
Yu Qiao,
Dahua Lin,
Bo Dai 
<!-- [Humphrey Shi](https://www.humphreyshi.com) -->
</br>

[Paper]() | [Project](https://animatediff.github.io/)

## Setup
Install the required packages:
```
pip install -r requirements.txt
```

## Inference

1. Download personalized Stable Diffusion (LoRA or DreamBooth are currently supported) from [CivitAI](https://civitai.com/) or Huggingface;

2. Download our motion module's pretrained weights from [this link](); 

3. Edit config files in 
```
configs/prompts/lora.yaml
```

4. Execute animation generation:
```
python -m scripts.animate --prompt configs/prompts/lora.yaml
```
