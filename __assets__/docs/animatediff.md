## Steps for Training

### Dataset
Before training, download the videos files and the `.csv` annotations of [WebVid10M](https://maxbain.com/webvid-dataset/) to the local mechine.
Note that our examplar training script requires all the videos to be saved in a single folder. You may change this by modifying `animatediff/data/dataset.py`.

### Configuration
After dataset preparations, update the below data paths in the config `.yaml` files in `configs/training/` folder:
```
train_data:
  csv_path: [Replace with .csv Annotation File Path]
  video_folder: [Replace with Video Folder Path]
  sample_size: 256
```
Other training parameters (lr, epochs, validation settings, etc.) are also included in the config files.

### Training
To finetune the unet's image layers
```
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/v1/image_finetune.yaml
```

To train motion modules
```
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/v1/training.yaml
```
