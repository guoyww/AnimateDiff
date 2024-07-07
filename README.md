## *** Updated by [damian0815](https://github.com/damian0815) for SD2 (v-prediction support)

see [configs/training/sd21/training-vpred-adapt.yaml](configs/training/sd21/training-vpred-adapt.yaml) for training reference.

for inference:
```bash
python -m scripts.animate --config configs/prompts/sd2/sd21-1-T2V.yaml --pretrained-model-path "<path/to/sd21/diffusers>"
```

note: the difference between v1, v2, and v3 seems to just be the unet parameters `use_inflated_groupnorm`, `motion_module_mid_block`, and `temporal_position_encoding_max_len` (ref [configs/training/sd21/training-vpred-adapt.yaml](configs/training/sd21/training-vpred-adapt.yaml)):

```bash
  # v1:
  # use_inflated_groupnorm: false
  # motion_module_mid_block: false
  # temporal_position_encoding_max_len: 24

  # v2:
  # use_inflated_groupnorm: true
  # motion_module_mid_block: true
  # temporal_position_encoding_max_len: 32
  
  # v3:
  # use_inflated_groupnorm:     true
  # motion_module_mid_block: false
  # temporal_position_encoding_max_len: 32
```

# AnimateDiff

This repository is the official implementation of [AnimateDiff](https://arxiv.org/abs/2307.04725) [ICLR2024 Spotlight].
It is a plug-and-play module turning most community models into animation generators, without the need of additional training.

**[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725)** 
</br>
[Yuwei Guo](https://guoyww.github.io/),
[Ceyuan Yang*](https://ceyuan.me/),
[Anyi Rao](https://anyirao.com/),
[Zhengyang Liang](https://maxleung99.github.io/),
[Yaohui Wang](https://wyhsirius.github.io/),
[Yu Qiao](https://scholar.google.com.hk/citations?user=gFtI-8QAAAAJ),
[Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/),
[Dahua Lin](http://dahua.site),
[Bo Dai](https://daibo.info)
(*Corresponding Author)

<!-- [Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://animatediff.github.io/) -->
[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2307.04725)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://animatediff.github.io/)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/Masbfca/AnimateDiff)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/guoyww/AnimateDiff)

We developed four versions of AnimateDiff: `v1`, `v2` and `v3` for [Stable Diffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5); `sdxl-beta` for [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

## Next
- [ ] Update to latest diffusers version
- [ ] Update Gradio demo
- [ ] Release training scripts
- [x] Release AnimateDiff v3 and SparseCtrl

## Gallery
We show some results in the [GALLERY](./__assets__/docs/gallery.md).
Some of them are contributed by the community.

## Preparations

Note: see [ANIMATEDIFF](__assets__/docs/animatediff.md) for detailed setup.

### Setup repository and conda environment

```
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff

conda env create -f environment.yaml
conda activate animatediff
```

### Download Stable Diffusion V1.5

```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/StableDiffusion/
```

### Prepare Community Models

Manually download the community `.safetensors` models from [CivitAI](https://civitai.com), and save them to `models/DreamBooth_LoRA`. We recommand [RealisticVision V5.1](https://civitai.com/models/4201?modelVersionId=130072) and [ToonYou Beta6](https://civitai.com/models/30240?modelVersionId=125771).

### Prepare AnimateDiff Modules

Manually download the AnimateDiff modules. The download links can be found in each version's model zoo, as provided in the following. Save the modules to `models/Motion_Module`.


##  [2023.12] AnimateDiff v3 and SparseCtrl

In this version, we did the image model finetuning through **Domain Adapter LoRA** for more flexiblity at inference time.

Additionally, we implement two (RGB image/scribble) [SparseCtrl](https://arxiv.org/abs/2311.16933) Encoders, which can take abitary number of condition maps to control the generation process.

- **Explanation:** Domain Adapter is a LoRA module trained on static frames of the training video dataset. This process is done before training the motion module, and helps the motion module focus on motion modeling, as shown in the figure below. At inference, By adjusting the LoRA scale of the Domain Adapter, some visual attributes of the training video, e.g., the watermarks, can be removed. To utilize the SparseCtrl encoder, it's necessary to use a full Domain Adapter in the pipeline.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="__assets__/figs/adapter_explain.png" style="width:60%">


Technical details of SparseCtrl can be found in this research paper:

**[SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models](https://arxiv.org/abs/2311.16933)**
</br>
[Yuwei Guo](https://guoyww.github.io/),
[Ceyuan Yang*](https://ceyuan.me/),
[Anyi Rao](https://anyirao.com/),
[Maneesh Agrawala](https://graphics.stanford.edu/~maneesh/),
[Dahua Lin](http://dahua.site),
[Bo Dai](https://daibo.info)
(*Corresponding Author)

[![arXiv](https://img.shields.io/badge/arXiv-2311.16933-b31b1b.svg)](https://arxiv.org/abs/2311.16933)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://guoyww.github.io/projects/SparseCtrl/)


<details open>
<summary>AnimateDiff v3 Model Zoo</summary>

  | Name                          | HuggingFace                                                                                | Type                | Storage Space | Description                        |
  |-------------------------------|--------------------------------------------------------------------------------------------|---------------------|---------------|------------------------------------|
  | `v3_adapter_sd_v15.ckpt`      | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_adapter.ckpt)           | Domain Adapter      | 97.4 MB       |                                    |
  | `v3_sd15_mm.ckpt.ckpt`           | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_mm.ckpt)              | Motion Module       | 1.56 GB       |                                    |
  | `v3_sd15_sparsectrl_scribble.ckpt` | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_sparsectrl_scribble.ckpt)    | SparseCtrl Encoder  | 1.86 GB       | scribble condition  |
  | `v3_sd15_sparsectrl_rgb.ckpt`    | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_sparsectrl_rgb.ckpt)       | SparseCtrl Encoder  | 1.85 GB       | RGB image condition |
</details>

### Quick Demos

<table class="center">
    <tr style="line-height: 0">
    <td width=25% style="border: none; text-align: center">Input (by RealisticVision)</td>
    <td width=25% style="border: none; text-align: center">Animation</td>
    <td width=25% style="border: none; text-align: center">Input</td>
    <td width=25% style="border: none; text-align: center">Animation</td>
    </tr>
    <tr>
    <td width=25% style="border: none"><img src="__assets__/demos/image/RealisticVision_firework.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__assets__/animations/v3/animation_fireworks.gif" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__assets__/demos/image/RealisticVision_sunset.png" style="width:100%"></td>
    <td width=25% style="border: none"><img src="__assets__/animations/v3/animation_sunset.gif" style="width:100%"></td>
    </tr>
</table>

<table class="center">
    <tr style="line-height: 0">
    <td width=25% style="border: none; text-align: center">Input Scribble</td>
    <td width=25% style="border: none; text-align: center">Output</td>
    <td width=25% style="border: none; text-align: center">Input Scribbles</td>
    <td width=25% style="border: none; text-align: center">Output</td>
    </tr>
    <tr>
      <td width=25% style="border: none"><img src="__assets__/demos/scribble/scribble_1.png" style="width:100%"></td>
      <td width=25% style="border: none"><img src="__assets__/animations/v3/sketch_boy.gif" style="width:100%"></td>
      <td width=25% style="border: none"><img src="__assets__/demos/scribble/scribble_2_readme.png" style="width:100%"></td>
      <td width=25% style="border: none"><img src="__assets__/animations/v3/sketch_city.gif" style="width:100%"></td>
    </tr>
</table>


### Inference

Here we provide three demo inference scripts. The corresponding AnimateDiff modules and community models need to be downloaded in advance. Put motion module in `models/Motion_Module`; put SparseCtrl encoders in `models/SparseCtrl`.
```
# under general T2V setting
python -m scripts.animate --config configs/prompts/v3/v3-1-T2V.yaml

# image animation (on RealisticVision)
python -m scripts.animate --config configs/prompts/v3/v3-2-animation-RealisticVision.yaml

# sketch-to-animation and storyboarding (on RealisticVision)
python -m scripts.animate --config configs/prompts/v3/v3-3-sketch-RealisticVision.yaml
```

### Limitations
1. Small fickering is noticable. To be solved in future versions;
2. To stay compatible with comunity models, there is no specific optimizations for general T2V, leading to limited visual quality under this setting;
3. **(Style Alignment) For usage such as image animation/interpolation, it's recommanded to use images generated by the same community model.**


## [2023.11] AnimateDiff SDXL-Beta

Release the Motion Module (beta version) on SDXL, available at [Google Drive](https://drive.google.com/file/d/1EK_D9hDOPfJdK4z8YDB8JYvPracNx2SX/view?usp=share_link
) / [HuggingFace](https://huggingface.co/guoyww/animatediff/blob/main/mm_sdxl_v10_beta.ckpt
) / [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules). High resolution videos (i.e., 1024x1024x16 frames with various aspect ratios) could be produced **with/without** personalized models. Inference usually requires ~13GB VRAM and tuned hyperparameters (e.g., #sampling steps), depending on the chosen personalized models.

Checkout to the branch [sdxl](https://github.com/guoyww/AnimateDiff/tree/sdxl) for more details of the inference. More checkpoints with better-quality would be available soon. Stay tuned. Examples below are manually downsampled for fast loading.

<details open>
<summary>AnimateDiff SDXL-Beta Model Zoo</summary>

  | Name                          | HuggingFace             | Type                | Storage Space |
  |-------------------------------|-----------------------------------------------------------------------------------|---------------------|---------------|
  | `mm_sdxl_v10_beta.ckpt`       | [Link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sdxl_v10_beta.ckpt) | Motion Module       | 950 MB        |
</details>

<table class="center">
    <tr style="line-height: 0">
    <td width=52% style="border: none; text-align: center">Original SDXL</td>
    <td width=30% style="border: none; text-align: center">Community SDXL</td>
    <td width=18% style="border: none; text-align: center">Community SDXL</td>
    </tr>
    <tr>
    <td width=52% style="border: none"><img src="__assets__/animations/motion_xl/01.gif" style="width:100%"></td>
    <td width=30% style="border: none"><img src="__assets__/animations/motion_xl/02.gif" style="width:100%"></td>
    <td width=18% style="border: none"><img src="__assets__/animations/motion_xl/03.gif" style="width:100%"></td>
    </tr>
</table>



## [2023.09] AnimateDiff v2

In this version, the motion module is trained upon larger resolution and batch size.
We observe this significantly helps improve the sample quality.

Moreover, we support **MotionLoRA** for eight basic camera movements.

<details open>
<summary>AnimateDiff v2 Model Zoo</summary>

  | Name                                 | HuggingFace                                                                                      | Type          | Parameter | Storage Space |
  |--------------------------------------|--------------------------------------------------------------------------------------------------|---------------|-----------|---------------|
  | mm_sd_v15_v2.ckpt                    | [Link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)                    | Motion Module | 453 M     | 1.7 GB        |
  | v2_lora_ZoomIn.ckpt                  | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_ZoomIn.ckpt)                  | MotionLoRA    | 19 M      | 74 MB         |
  | v2_lora_ZoomOut.ckpt                 | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_ZoomOut.ckpt)                 | MotionLoRA    | 19 M      | 74 MB         |
  | v2_lora_PanLeft.ckpt                 | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_PanLeft.ckpt)                 | MotionLoRA    | 19 M      | 74 MB         |
  | v2_lora_PanRight.ckpt                | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_PanRight.ckpt)                | MotionLoRA    | 19 M      | 74 MB         |
  | v2_lora_TiltUp.ckpt                  | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_TiltUp.ckpt)                  | MotionLoRA    | 19 M      | 74 MB         |
  | v2_lora_TiltDown.ckpt                | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_TiltDown.ckpt)                | MotionLoRA    | 19 M      | 74 MB         |
  | v2_lora_RollingClockwise.ckpt        | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_RollingClockwise.ckpt)        | MotionLoRA    | 19 M      | 74 MB         |
  | v2_lora_RollingAnticlockwise.ckpt    | [Link](https://huggingface.co/guoyww/animatediff/blob/main/v2_lora_RollingAnticlockwise.ckpt)    | MotionLoRA    | 19 M      | 74 MB         |

</details>


- Release **MotionLoRA** and its model zoo, **enabling camera movement controls**! Please download the MotionLoRA models (**74 MB per model**, available at [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing) / [HuggingFace](https://huggingface.co/guoyww/animatediff) / [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules) ) and save them to the `models/MotionLoRA` folder. Example:
  ```
  python -m scripts.animate --config configs/prompts/v2/5-RealisticVision-MotionLoRA.yaml
  ```
    <table class="center">
      <tr style="line-height: 0">
      <td colspan="2" style="border: none; text-align: center">Zoom In</td>
      <td colspan="2" style="border: none; text-align: center">Zoom Out</td>
      <td colspan="2" style="border: none; text-align: center">Zoom Pan Left</td>
      <td colspan="2" style="border: none; text-align: center">Zoom Pan Right</td>
      </tr>
      <tr>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/01.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/02.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/02.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/01.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/03.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/04.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/04.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/03.gif"></td>
      </tr>
      <tr style="line-height: 0">
      <td colspan="2" style="border: none; text-align: center">Tilt Up</td>
      <td colspan="2" style="border: none; text-align: center">Tilt Down</td>
      <td colspan="2" style="border: none; text-align: center">Rolling Anti-Clockwise</td>
      <td colspan="2" style="border: none; text-align: center">Rolling Clockwise</td>
      </tr>
      <tr>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/05.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/05.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/06.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/06.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/07.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/07.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_01/08.gif"></td>
      <td style="border: none"><img src="__assets__/animations/motion_lora/model_02/08.gif"></td>
      </tr>
  </table>

- New Motion Module release! `mm_sd_v15_v2.ckpt` was trained on larger resolution & batch size, and gains noticeable quality improvements. Check it out at [Google Drive](https://drive.google.com/drive/folders/1EqLC65eR1-W-sGD0Im7fkED6c8GkiNFI?usp=sharing) / [HuggingFace](https://huggingface.co/guoyww/animatediff) / [CivitAI](https://civitai.com/models/108836/animatediff-motion-modules) and use it with `configs/inference/inference-v2.yaml`. Example:
  ```
  python -m scripts.animate --config configs/prompts/v2/5-RealisticVision.yaml
  ```
  Here is a qualitative comparison between `mm_sd_v15.ckpt` (left) and `mm_sd_v15_v2.ckpt` (right):
  <table class="center">
      <tr>
      <td><img src="__assets__/animations/compare/old_0.gif"></td>
      <td><img src="__assets__/animations/compare/new_0.gif"></td>
      <td><img src="__assets__/animations/compare/old_1.gif"></td>
      <td><img src="__assets__/animations/compare/new_1.gif"></td>
      <td><img src="__assets__/animations/compare/old_2.gif"></td>
      <td><img src="__assets__/animations/compare/new_2.gif"></td>
      <td><img src="__assets__/animations/compare/old_3.gif"></td>
      <td><img src="__assets__/animations/compare/new_3.gif"></td>
      </tr>
  </table>


## [2023.07] AnimateDiff v1

<details open>
<summary>AnimateDiff v1 Model Zoo</summary>

  | Name            | HuggingFace                                                                  | Parameter | Storage Space |
  |-----------------|------------------------------------------------------------------------------|-----------|---------------|
  | mm_sd_v14.ckpt  | [Link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v14.ckpt)   | 417 M     | 1.6 GB        |
  | mm_sd_v15.ckpt  | [Link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15.ckpt)   | 417 M     | 1.6 GB        |

</details>

### Quick Demos
<table class="center">
    <tr>
    <td><img src="__assets__/animations/model_01/01.gif"></td>
    <td><img src="__assets__/animations/model_01/02.gif"></td>
    <td><img src="__assets__/animations/model_01/03.gif"></td>
    <td><img src="__assets__/animations/model_01/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/30240/toonyou">ToonYou</a></p>

<table>
    <tr>
    <td><img src="__assets__/animations/model_03/01.gif"></td>
    <td><img src="__assets__/animations/model_03/02.gif"></td>
    <td><img src="__assets__/animations/model_03/03.gif"></td>
    <td><img src="__assets__/animations/model_03/04.gif"></td>
    </tr>
</table>
<p style="margin-left: 2em; margin-top: -1em">Model：<a href="https://civitai.com/models/4201/realistic-vision-v20">Realistic Vision V2.0</a></p>


### Inference

Here we provide several demo inference scripts. The corresponding AnimateDiff modules and community models need to be downloaded in advance. See [ANIMATEDIFF](__assets__/docs/animatediff.md) for detailed setup.

```
python -m scripts.animate --config configs/prompts/1-ToonYou.yaml
python -m scripts.animate --config configs/prompts/3-RcnzCartoon.yaml
```


## Community Contributions

User Interface developed by community: 
  - A1111 Extension [sd-webui-animatediff](https://github.com/continue-revolution/sd-webui-animatediff) (by [@continue-revolution](https://github.com/continue-revolution))
  - ComfyUI Extension [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) (by [@Kosinkadink](https://github.com/Kosinkadink))
  - Google Colab: [Colab](https://colab.research.google.com/github/camenduru/AnimateDiff-colab/blob/main/AnimateDiff_colab.ipynb) (by [@camenduru](https://github.com/camenduru))

## Gradio Demo

We created a Gradio demo to make AnimateDiff easier to use. To launch the demo, please run the following commands:

```
conda activate animatediff
python app.py
```

By default, the demo will run at `localhost:7860`.
<br><img src="__assets__/figs/gradio.jpg" style="width: 50em; margin-top: 1em">


## Common Issues
<details>
<summary>Installation</summary>

Please ensure the installation of [xformer](https://github.com/facebookresearch/xformers) that is applied to reduce the inference memory.
</details>


<details>
<summary>Various resolution or number of frames</summary>
Currently, we recommend users to generate animation with 16 frames and 512 resolution that are aligned with our training settings. Notably, various resolution/frames may affect the quality more or less. 
</details>


<details>
<summary>How to use it without any coding</summary>

1) Get lora models: train lora model with [A1111](https://github.com/continue-revolution/sd-webui-animatediff) based on a collection of your own favorite images (e.g., tutorials [English](https://www.youtube.com/watch?v=mfaqqL5yOO4), [Japanese](https://www.youtube.com/watch?v=N1tXVR9lplM), [Chinese](https://www.bilibili.com/video/BV1fs4y1x7p2/)) 
or download Lora models from [Civitai](https://civitai.com/).

2) Animate lora models: using gradio interface or A1111 
(e.g., tutorials [English](https://github.com/continue-revolution/sd-webui-animatediff), [Japanese](https://www.youtube.com/watch?v=zss3xbtvOWw), [Chinese](https://941ai.com/sd-animatediff-webui-1203.html)) 

3) Be creative togther with other techniques, such as, super resolution, frame interpolation, music generation, etc.
</details>


<details>
<summary>Animating a given image</summary>

We totally agree that animating a given image is an appealing feature, which we would try to support officially in future. For now, you may enjoy other efforts from the [talesofai](https://github.com/talesofai/AnimateDiff).  
</details>

<details>
<summary>Contributions from community</summary>
Contributions are always welcome!! The <code>dev</code> branch is for community contributions. As for the main branch, we would like to align it with the original technical report :)
</details>

## Training and inference
Please refer to [ANIMATEDIFF](./__assets__/docs/animatediff.md) for the detailed setup.

<!-- ## Gallery -->
<!-- We collect several generated results in [GALLERY](./__assets__/docs/gallery.md). -->

## BibTeX
```
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Liang, Zhengyang and Wang, Yaohui and Qiao, Yu and Agrawala, Maneesh and Lin, Dahua and Dai, Bo},
  journal={International Conference on Learning Representations},
  year={2024}
}

@article{guo2023sparsectrl,
  title={SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Agrawala, Maneesh and Lin, Dahua and Dai, Bo},
  journal={arXiv preprint arXiv:2311.16933},
  year={2023}
}
```

## Disclaimer
This project is released for academic use. We disclaim responsibility for user-generated content. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards. 
Please be advised that our only official website is https://github.com/guoyww/AnimateDiff, and all the other websites are NOT associated with us at AnimateDiff. 


## Contact Us
**Yuwei Guo**: [guoyuwei@pjlab.org.cn](mailto:guoyuwei@pjlab.org.cn)  
**Ceyuan Yang**: [yangceyuan@pjlab.org.cn](mailto:yangceyuan@pjlab.org.cn)  
**Bo Dai**: [daibo@pjlab.org.cn](mailto:daibo@pjlab.org.cn)

## Acknowledgements
Codebase built upon [Tune-a-Video](https://github.com/showlab/Tune-A-Video).
