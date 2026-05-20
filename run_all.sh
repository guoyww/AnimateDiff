#!/bin/bash
set -e

MODEL_PATH="/mnt/62G/huggingface/miniSD"
MODEL_PATHFP16="/mnt/62G/huggingface/miniSD-fp16"

DEVICE="cuda"

CONFIGS128=("configs/prompts/0_animate/128.yaml")
CONFIGS128P=("configs/prompts/2_animate/128p.yaml")
CONFIGS256P=("configs/prompts/1_animate/256p.yaml")
CONFIGS256SP=("configs/prompts/2_animate/256_speed.yaml")
CONFIGS256=("configs/prompts/2_animate/256.yaml")
CONFIGS512=("configs/prompts/2_animate/512.yaml")
CONFIGS512x640=("configs/prompts/2_animate/640x512.yaml")
CONFIGSCYBER=("configs/prompts/2_animate/cyber.yaml")
CONFIGSHD=("configs/prompts/2_animate/512-c.yaml")
CONFIGSHD2=("configs/prompts/2_animate/960.yaml")

SCRIPTS0=("n3rperfect") #512 OK 640 OK
SCRIPTS1=("n3rfast") #512 OK 640 OK
SCRIPTS2=("n3rfastmovie") #512 OK 640 OK
SCRIPTS3=("n3rfastinterpol") #512 OK 640 OK
SCRIPTS4=("n3rspeed") # 128 256 512 ok
SCRIPTS5=("n3rcreative") # 128 256 512 ok
SCRIPTS6=("n3rHYBRID14") # 128 256 512 ok
SCRIPTS7=("n3rHYBRID21") # 128 256 512 ok
SCRIPTS8=("n3rHYBRID26") #128 ok 256 ko
SCRIPTS9=("n3rProtoBoost") #512 KO
SCRIPTS10=("n3rmodelSD") # #512 OK 640 OK 960 OK
SCRIPTS11=("n3rProBoost") # #512 OK 640 OK 960 OK


run_batch () {
    local -n CONFIG_ARRAY=$1
    local -n SCRIPT_ARRAY=$2
    local -n MODEL_ARRAY=$3

    for CONFIG_PATH in "${CONFIG_ARRAY[@]}"; do
        echo "========================================"
        echo "Config : $CONFIG_PATH"
        echo "========================================"

        for script in "${SCRIPT_ARRAY[@]}"; do
            echo "Lancement : $script"
            for model in "${MODEL_ARRAY[@]}"; do
                echo "Model : $model"

                python -m scripts.$script \
                    --pretrained-model-path "$model" \
                    --config "$CONFIG_PATH" \
                    --device "$DEVICE" \
                    --vae-offload \
                    --fp16

                echo "$script terminé."

                # Nettoyage CUDA
                python - <<EOF
import torch
torch.cuda.empty_cache()
EOF

            nvidia-smi
            echo "========================================"
        done
    done
}


# CONFIGS2 → SCRIPTS2 256 OK
#run_batch CONFIGS2 SCRIPTS1

# CONFIGS2 → SCRIPTS2 512 OK

run_batch CONFIGSCYBER SCRIPTS10 MODEL_PATHFP16
# run_batch CONFIGS512 SCRIPTS10 MODEL_PATHFP16
# run_batch CONFIGS512x640 SCRIPTS10 MODEL_PATHFP16
# run_batch CONFIGS128 SCRIPTS10 MODEL_PATHFP16
# run_batch CONFIGS256SP SCRIPTS10 MODEL_PATHFP16


#run_batch CONFIGS512 SCRIPTS0 MODEL_PATH # KO RuntimeError: Input type (c10::Half) and bias type (float) should be the same
#run_batch CONFIGS512x640 SCRIPTS0 MODEL_PATH #Latent trop petit à timestep 959, mean=nan
#run_batch CONFIGS128 SCRIPTS0 MODEL_PATH # RuntimeError: Input type (c10::Half) and bias type (float) should be the same
#run_batch CONFIGS256SP SCRIPTS0 MODEL_PATH #RuntimeError: Input type (c10::Half) and bias type (float) should be the same

#run_batch CONFIGS512 SCRIPTS1 MODEL_PATH #OK
#run_batch CONFIGS512x640 SCRIPTS1 MODEL_PATH #OK
#run_batch CONFIGS128 SCRIPTS1 MODEL_PATH # OK
#run_batch CONFIGS256SP SCRIPTS1 MODEL_PATH #"OK"


#run_batch CONFIGS512 SCRIPTS2 MODEL_PATH #OK
#run_batch CONFIGS512x640 SCRIPTS2 MODEL_PATH #OK
#run_batch CONFIGS128 SCRIPTS2 MODEL_PATH #OK
#run_batch CONFIGS256SP SCRIPTS2 MODEL_PATH #OK


# CONFIGS2 → SCRIPTS2 256
#run_batch CONFIGS256 SCRIPTS2 MODEL_PATH

# CONFIGS3 → SCRIPTS1 (le "sinon") 512
#run_batch CONFIGS1 SCRIPTS3 MODEL_PATH

echo "Tous les scripts ont été exécutés."
