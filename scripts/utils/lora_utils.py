# ------------------------------------------------------------------
# lora_utils_smart_device.py - Chargement intelligent des LoRA avec device
# ------------------------------------------------------------------
import torch
from safetensors.torch import load_file


def detect_unet_type(unet):
    """
    Détecte le type de UNet selon cross_attention_dim
    """
    dim = getattr(getattr(unet, "config", None), "cross_attention_dim", None)

    if dim == 768:
        model_type = "SD1.x compatible"
    elif dim == 1024:
        model_type = "SD2.x compatible"
    elif dim == 2048:
        model_type = "SDXL compatible"
    else:
        model_type = "UNet custom"

    return model_type, dim


def apply_lora(unet, lora_path, alpha=0.8, device=None, verbose=True):
    """
    Applique un modèle LoRA / n3oray sur UNet
    """

    device = device or next(unet.parameters()).device

    # ---------------- Détection UNet ----------------
    model_type, cross_dim = detect_unet_type(unet)

    print("🧠 Détection modèle UNet")
    print(f"   type : {model_type}")
    print(f"   cross_attention_dim : {cross_dim}")

    print(f"📌 Chargement LoRA : {lora_path}")

    # Charger LoRA sur CPU
    lora_state = load_file(lora_path, device="cpu")

    unet_state = dict(unet.named_parameters())

    applied = 0
    skipped = 0
    missing = 0

    for name, lora_param in lora_state.items():

        if name not in unet_state:
            missing += 1
            continue

        param = unet_state[name]

        # vérification dimension
        if param.shape != lora_param.shape:

            if verbose:
                print(
                    f"[LoRA SKIP] {name} "
                    f"{tuple(lora_param.shape)} != {tuple(param.shape)}"
                )

            skipped += 1
            continue

        lora_param = lora_param.to(device=device, dtype=param.dtype)

        # mélange des poids
        param.data.mul_(1 - alpha).add_(lora_param, alpha=alpha)

        applied += 1

    print("✅ LoRA résumé")
    print(f"   couches appliquées : {applied}")
    print(f"   couches ignorées   : {skipped}")
    print(f"   couches absentes   : {missing}")

    return unet


def apply_lora_smart(unet, lora_path, alpha=0.8, device=None, verbose=True):
    device = device or next(unet.parameters()).device
    model_type, cross_dim = detect_unet_type(unet)

    if verbose:
        print("🧠 Détection modèle UNet")
        print(f"   type : {model_type}")
        print(f"   cross_attention_dim : {cross_dim}")
        print(f"📌 Chargement LoRA : {lora_path}")

    lora_state = load_file(lora_path, device="cpu")
    unet_state = dict(unet.named_parameters())

    # Affichage filtré
    if verbose:
        for k, v in lora_state.items():
            if "up_blocks" in k and "attn1.to_q.weight" in k:
                print(k, v.shape)

    # Compatibilité : intersection avec le UNet
    compatible_keys = [k for k in lora_state if k in unet_state and unet_state[k].shape == lora_state[k].shape]
    if not compatible_keys:
        print(f"⚠ LoRA '{lora_path}' incompatible avec ce UNet, chargement annulé.")
        return unet

    # Application
    applied = 0
    skipped = 0
    missing = 0
    for k, lora_param in lora_state.items():
        if k not in unet_state:
            missing += 1
            continue
        param = unet_state[k]
        if param.shape != lora_param.shape:
            skipped += 1
            continue
        lora_param = lora_param.to(device=device, dtype=param.dtype)
        with torch.no_grad():
            param.copy_(param*(1-alpha) + lora_param*alpha)
        applied += 1

    if verbose:
        print("✅ LoRA résumé")
        print(f"   couches appliquées : {applied}")
        print(f"   couches ignorées   : {skipped}")
        print(f"   couches absentes   : {missing}")

    return unet


def list_lora_parameters(lora_path):
    """
    Liste les paramètres contenus dans un LoRA
    """
    lora_state = load_file(lora_path, device="cpu")
    print(f"📌 Paramètres dans {lora_path}")
    for k, v in lora_state.items():
        print(f"{k} -> {tuple(v.shape)}")
    return list(lora_state.keys())
