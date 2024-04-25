# Run AnimateDiff on AscendNPU

## Prepare Environment


1. Clone this repository and Install package
```shell
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff

conda env create -f environment.yaml
conda activate animatediff
```
2. Install Ascend Extension for PyTorch

You can follow this [guide](https://www.hiascend.com/document/detail/en/ModelZoo/pytorchframework/ptes/ptes_00001.html) to download and install the Ascend NPU Firmware, Ascend NPU Driver, and CANN. Afterwards, you need to install additional Python packages.
```shell
pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu  # For X86
pip3 install torch==2.1.0  # For Aarch64
pip3 install accelerate==0.28.0 diffusers==0.11.1 decorator==5.1.1 scipy==1.12.0 attrs==23.2.0  torchvision==0.16.0 transformers==4.25.1
```
After installing the above Python packages,
You can follow this [README](https://github.com/Ascend/pytorch/blob/master/README.md) to install the torch_npu environment.
Then you can use AnimateDiff on Ascend NPU.

## Prepare Checkpoints
You can follow this [README](animatediff.md) to prepare your checkpoints for inference, training and finetune.

## Training/Finetune AnimateDiff on AscendNPU

***Note: AscendNPU does not support xformers acceleration, so the option 'enable_xformers_memory_efficient_attention' in the yaml file under 'training/v1/' directory needs to be changed to <font color=red>False</font>. I have integrated torch_npu flash attention and other acceleration methods into project that can speed up the training process.***

If you want to train animatediff on ascendnpu, you only add 'source' command on your shell scripts.

As shown below:
```shell
# Firstly, add environment variables to the system via the 'source' command.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/v1/image_finetune.yaml
```

## Inference AnimateDiff on AscendNPU

If you want to inference animatediff on ascendnpu, you only add 'source' command on your shell scripts.

As shown below:
```shell
# Firstly, add environment variables to the system via the 'source' command.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m scripts.animate --config configs/prompts/1-ToonYou.yaml
python -m scripts.animate --config configs/prompts/2-Lyriel.yaml
python -m scripts.animate --config configs/prompts/3-RcnzCartoon.yaml
python -m scripts.animate --config configs/prompts/4-MajicMix.yaml
python -m scripts.animate --config configs/prompts/5-RealisticVision.yaml
python -m scripts.animate --config configs/prompts/6-Tusun.yaml
python -m scripts.animate --config configs/prompts/7-FilmVelvia.yaml
python -m scripts.animate --config configs/prompts/8-GhibliBackground.yaml
```