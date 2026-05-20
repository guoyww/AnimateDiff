# scripts/utils/config_loader.py
import yaml

def load_config(path):
    cfg_main = yaml.safe_load(open(path))

    inference_cfg_path = cfg_main.get("inference_config")
    if inference_cfg_path:
        cfg_infer = yaml.safe_load(open(inference_cfg_path))
        # Merge : priorité au YAML principal
        for k, v in cfg_infer.items():
            if k not in cfg_main:
                cfg_main[k] = v
    return cfg_main
