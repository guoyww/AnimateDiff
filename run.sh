#!/bin/bash
set -e

MODEL_PATH="/mnt/62G/huggingface/miniSD"
DEVICE="cuda"

clear
echo "========================================"
echo "        N3R LAUNCHER INTERACTIF        "
echo "========================================"
echo ""

# -------------------------
# Température initiale
# -------------------------
TEMP_BEFORE=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
echo "🌡️ Température GPU actuelle : ${TEMP_BEFORE}°C"
echo ""

# -------------------------
# MENU CONFIG
# -------------------------
echo "Choisir la configuration :"
select CONFIG_CHOICE in \
    "128" \
    "128p" \
    "256" \
    "256p" \
    "512" \
    "512x640" \
    "640" \
    "Quitter"
do
    case $CONFIG_CHOICE in
        128)   CONFIG="configs/prompts/0_animate/128.yaml"; break ;;
        128p)  CONFIG="configs/prompts/2_animate/128p.yaml"; break ;;
        256)   CONFIG="configs/prompts/2_animate/256.yaml"; break ;;
        256p)  CONFIG="configs/prompts/1_animate/256p.yaml"; break ;;
        512)   CONFIG="configs/prompts/2_animate/512.yaml"; break ;;
        512x640)   CONFIG="configs/prompts/2_animate/640x512.yaml"; break ;;
        640)   CONFIG="configs/prompts/2_animate/640.yaml"; break ;;
        Quitter) exit 0 ;;
        *) echo "Choix invalide." ;;
    esac
done

echo ""
echo "Choisir le script :"
select SCRIPT_CHOICE in \
    "n3rfast" \
    "n3rspeed" \
    "n3rcreative" \
    "n3rHYBRID21" \
    "n3rHYBRID22" \
    "n3rHYBRID26" \
    "n3rHYBRID14" \
    "Quitter"
do
    case $SCRIPT_CHOICE in
        Quitter) exit 0 ;;
        *)
            SCRIPT="$SCRIPT_CHOICE"
            break
            ;;
    esac
done

# -------------------------
# Vérification compatibilité
# -------------------------
if [[ "$SCRIPT" == "n3rHYBRID26" && "$CONFIG_CHOICE" == "256" ]]; then
    echo "❌ n3rHYBRID26 ne supporte pas 256."
    exit 1
fi

if [[ "$SCRIPT" == "n3rHYBRID14" && "$CONFIG_CHOICE" == "512" ]]; then
    echo "❌ n3rHYBRID14 ne supporte pas 512."
    exit 1
fi

# -------------------------
# Résumé
# -------------------------
echo ""
echo "========================================"
echo "Config : $CONFIG"
echo "Script : $SCRIPT"
echo "========================================"
echo ""

read -p "Confirmer l'exécution ? (y/n) : " CONFIRM
if [[ "$CONFIRM" != "y" ]]; then
    echo "Annulé."
    exit 0
fi

# -------------------------
# Exécution avec timing
# -------------------------
echo ""
echo "🚀 Lancement..."
echo ""

START_TIME=$(date +%s)

python -m scripts.$SCRIPT \
    --pretrained-model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --vae-offload \
    --fp16

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

MIN=$((DURATION / 60))
SEC=$((DURATION % 60))

echo ""
echo "========================================"
echo "✅ Exécution terminée."
echo "⏱️ Temps total : ${MIN}m ${SEC}s"
echo "========================================"

# -------------------------
# Nettoyage CUDA
# -------------------------
python - <<EOF
import torch
torch.cuda.empty_cache()
EOF

# -------------------------
# Température finale
# -------------------------
TEMP_AFTER=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)

echo ""
echo "🌡️ Température GPU après run : ${TEMP_AFTER}°C"

DELTA=$((TEMP_AFTER - TEMP_BEFORE))
echo "📈 Variation température : ${DELTA}°C"
echo ""

echo "📊 Etat GPU :"
nvidia-smi

echo ""
echo "========================================"
echo "Fin du programme."
echo "========================================"
