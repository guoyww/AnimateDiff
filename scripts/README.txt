n3rRealControl.py - Description et Options

Le script n3rRealControl.py permet de générer des vidéos et images animées à partir de modèles de diffusion avec des ajustements avancés. Il offre une flexibilité maximale via des options de personnalisation pour la qualité, la vitesse, et l’utilisation de la VRAM.

Principales Options :
use_mini_gpu : Utiliser un GPU à faible VRAM pour des générations rapides (2 Go VRAM).
verbose : Activer les logs détaillés pour le débogage.
latent_injection : Contrôler l'injection de latents pour ajuster la qualité.
fps et upscale_factor : Définir la fréquence d'images et la mise à l'échelle.
steps : Nombre d'étapes pour la diffusion, affecte la qualité.
guidance_scale : Intégrer la guidance de l'algorithme pour mieux contrôler la génération.
use_n3r_model et use_n3r_pro_net : Activer des modèles pour améliorer la qualité et les détails.
use_openpose : Utiliser OpenPose pour contrôler les poses humaines.
controlnet_scale et control_strength : Ajuster l’intensité de ControlNet pour une meilleure gestion des poses.
Fonctionnalités supplémentaires :
Motion Module : Applique des animations basées sur des données de mouvement.
LoRA : Charge des modèles LoRA pour personnaliser les résultats.
VAE & Tokenizer : Gère les embeddings texte pour une meilleure cohérence avec les prompts.

#-----------------------------------------------------------------------------------------------------
n3rRealControl.py - Description and Options

The n3rRealControl.py script enables the generation of animated videos and images from diffusion models with advanced adjustments. It provides maximum flexibility through customization options for quality, speed, and VRAM usage.

Key Options:
use_mini_gpu: Use a low-VRAM GPU for fast generation (~2GB VRAM).
verbose: Enable detailed logging for debugging purposes.
latent_injection: Control latent injection to adjust quality.
fps and upscale_factor: Define frames per second and upscaling factor.
steps: Number of steps for diffusion, affecting quality.
guidance_scale: Integrate algorithm guidance to better control the generation.
use_n3r_model and use_n3r_pro_net: Activate models to enhance quality and details.
use_openpose: Use OpenPose to control human poses.
controlnet_scale and control_strength: Adjust ControlNet intensity for better pose management.
Additional Features:
Motion Module: Apply animations based on motion data.
LoRA: Load LoRA models to customize results.
VAE & Tokenizer: Manage text embeddings for better consistency with prompts.


python -m scripts.n3rRealControl \
                      --pretrained-model-path "/**********/huggingface/miniSD" \
                      --config "configs/prompts/0_n3r/512-c.yaml" \
                      --device "cuda" \
                      --vae-offload \
                      --fp16
