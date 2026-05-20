For n3rRealControl

python -m scripts.n3rRealControl \
                      --pretrained-model-path "/mnt/62G/huggingface/miniSD" \
                      --config "configs/prompts/0_n3r/512-c.yaml" \
                      --device "cuda" \
                      --vae-offload \
                      --fp16
