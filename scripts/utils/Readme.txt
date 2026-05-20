"All fonction utils:"

Sample run:
python -m scripts.n3rRealControl \
                      --pretrained-model-path "/mnt/62G/huggingface/miniSD" \
                      --config "configs/prompts/0_n3r/512-c.yaml" \
                      --device "cuda" \
                      --vae-offload \
                      --fp16
Appuyez sur '²' + Entrée pour arrêter : Dimention : overlap : 16
📌 Paramètres de génération :
Paramètre                Valeur   Paramètre                Valeur
fps                          12   use_mini_gpu                  0
num_fraps_per_image          10   upscale_factor                1
guidance_scale              7.0   steps                        50
guidance_scale_end          7.5   init_image_scale            0.9
creative_noise            0.002   init_image_scale_end       0.75
creative_noise_end         0.01   latent_scale_boost         0.85
final_latent_scale         0.85   seed                      36989
transition_frames             1   latent_injection            0.5
use_n3r_model                 0   block_size                   32
🔄 Chargement UNet depuis /mnt/62G/huggingface/miniSD/unet ...
✅ Chargement poids safetensors
✅ UNet chargé avec dtype=torch.float16, device=cuda:0
⚠ Aucun modèle LoRA configuré, étape ignorée.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [00:00<00:00, 23043.69it/s]
CLIPTextModel LOAD REPORT from: /mnt/62G/huggingface/miniSD/text_encoder
Key                                | Status     | Details
-----------------------------------+------------+--------
text_model.embeddings.position_ids | UNEXPECTED |        

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
📦 Chargement VAE : /mnt/62G/huggingface/miniSD-fp16/vae/diffusion_pytorch_model.safetensors
/home/n3oray/.pyenv/versions/animatediff/lib/python3.11/site-packages/huggingface_hub/utils/_validators.py:206: UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored in `hf_hub_download`. Downloading to a local directory does not use symlinks anymore.
  warnings.warn(
🧠 Détection VAE
   type : SD1 / SD2 compatible
   latent_channels : 4
   scaling_factor : 0.18215
Pos embeds shape: torch.Size([1, 77, 768]) | Neg embeds shape: torch.Size([1, 77, 768])
Chargement ControlNet OpenPose depuis dossier local : /mnt/62G/huggingface/sd-controlnet-openpose
device   : cuda
dtype    : torch.float16
🧠 ControlNet prêt
   params : 361.3M
   dtype  : torch.float16
   device : cuda:0
✅ ControlNet en mode eval
✅ Paramètres gelés
⚠ enable_attention_slicing non disponible
✅ Déplacé sur cuda / torch.float16
✅ JSON chargé : /mnt/62G/AnimateDiff/scripts/json/anim2.json
[JSON->POSE] shape: torch.Size([10, 3, 1280, 896])
[JSON->POSE] min/max: -1.0 / 1.0
🎞 fix_pose_sequence - Frames JSON: 10
🎞 fix_pose_sequence - Frames attendues: 10
✅ PoseSequence final: torch.Size([10, 3, 1280, 896]) cuda:0 torch.float16
✅ N3RProNet activé
  0%|                                                                                            | 0/10 [00:00<?, ?it/s]WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1775831041.181812  980177 gl_context_egl.cc:85] Successfully initialized EGL. Major : 1 Minor: 5
I0000 00:00:1775831041.184503  980309 gl_context.cc:357] GL version: 3.2 (OpenGL ES 3.2 Mesa 25.3.5-arch1.2), renderer: Mesa Intel(R) UHD Graphics 630 (CFL GT2)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
I0000 00:00:1775831041.190918  980177 gl_context_egl.cc:85] Successfully initialized EGL. Major : 1 Minor: 5
I0000 00:00:1775831041.192474  980324 gl_context.cc:357] GL version: 3.2 (OpenGL ES 3.2 Mesa 25.3.5-arch1.2), renderer: Mesa Intel(R) UHD Graphics 630 (CFL GT2)
✅ FaceMesh initialisé (mode vidéo)
✅ MediaPipe fully initialized
[Frame 000] init_image_scale=0.900, guidance_scale=7.343, creative_noise=0.002
W0000 00:00:1775831041.194913  980314 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
✅ Image chargée : input/768-1536/2.png


Sample run:
python -m scripts.n3rHYBRID10 \
                         --pretrained-model-path "/mnt/62G/huggingface/miniSD" \
                         --config configs/prompts/2_animate/128.yaml \
                         --device cuda
📌 Paramètres : fps=12, frames/image=12, steps=12, seed=1234
⏱ Durée totale estimée : 5.0s
🔄 Chargement tokenizer et text_encoder
✅ Text encoder OK
✅ State dict VAE chargé, clés: ['decoder.conv_in.bias', 'decoder.conv_in.weight', 'decoder.conv_out.bias', 'decoder.conv_out.weight', 'decoder.mid.attn_1.k.bias']
🔎 Latent shape: torch.Size([1, 4, 32, 32])
🔎 Decoded shape: torch.Size([1, 3, 256, 256])
✅ Test VAE 256 OK
✅ VAE OK
✅ UNet + Scheduler OK
✅ Motion module (Python) loaded and instantiated: scripts/modules/motion_module_tiny.py
✅ Image chargée : input/image_128x0.png
✅ Image chargée : input/image_128x1.png
✅ Image chargée : input/image_128x2.png
✅ Image chargée : input/image_128x3.png
✅ Image chargée : input/image_128x4.png
✅ Génération terminée.



python -m scripts.n3rHYBRID11 \
                         --pretrained-model-path "/mnt/62G/huggingface/miniSD" \
                         --config configs/prompts/2_animate/256_quality.yaml \
                         --device cuda \
                         --vae-offload \
                         --fp16

