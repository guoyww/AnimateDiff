import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialisation des poids
def weights_temporal_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

# =========================================================
# SAVE / LOAD
# =========================================================

def save_temporal_model(
    model,
    optimizer=None,
    epoch=None,
    loss=None,
    path="models/temporal.pt",
    latest_path="models/temporal_latest.pt"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "timestamp": datetime.datetime.now().isoformat()
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if loss is not None:
        checkpoint["loss"] = loss

    torch.save(checkpoint, path)
    torch.save(checkpoint, latest_path)

    print(f"[INFO] Model saved -> {path}")


def load_temporal_model(
    model_class,
    path="models/temporal_latest.pt",
    optimizer=None,
    device=device
):

    if not os.path.exists(path):
        print("[WARN] No checkpoint found.")
        return model_class().to(device), None

    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("model_config", {})

    model = model_class(**config).to(device)

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(
        f"[INFO] Loaded checkpoint | "
        f"epoch={checkpoint.get('epoch')} | "
        f"loss={checkpoint.get('loss')} | "
        f"time={checkpoint.get('timestamp')}"
    )

    return model, checkpoint


# =========================================================
# DEBUG
# =========================================================

def print_latents_stats(name, x):

    print(
        f"{name} | "
        f"shape={tuple(x.shape)} | "
        f"min={x.min():.4f} | "
        f"max={x.max():.4f} | "
        f"mean={x.mean():.4f} | "
        f"std={x.std():.4f}"
    )

    if torch.isnan(x).any():
        print(f"[WARN] NaN detected in {name}")

    if torch.isinf(x).any():
        print(f"[WARN] Inf detected in {name}")


# =========================================================
# NORMALIZATION
# =========================================================

def sanitize_latents(latents):

    latents = latents.float()

    latents = (latents - latents.mean()) / (latents.std() + 1e-6)

    latents = latents.clamp(-4.0, 4.0)

    return latents


# =========================================================
# WEIGHTS INIT
# =========================================================

def weights_init(m):

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):

        init.kaiming_normal_(m.weight, nonlinearity="linear")

        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):

        init.xavier_normal_(m.weight)

        if m.bias is not None:
            init.zeros_(m.bias)


# =========================================================
# RESIDUAL BLOCK
# =========================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),

            nn.Conv2d(channels, channels, 3, padding=1)
        )

        self.act = nn.SiLU()

    def forward(self, x):

        residual = x

        x = self.block(x)

        x = x + residual

        x = self.act(x)

        return x


# =========================================================
# TEMPORAL MODEL
# =========================================================

class TemporalResidualNet(nn.Module):

    def __init__(
        self,
        channels=4,
        hidden=64,
        num_blocks=4
    ):

        super().__init__()

        self.config = {
            "channels": channels,
            "hidden": hidden,
            "num_blocks": num_blocks
        }

        self.input_proj = nn.Sequential(

            nn.Conv2d(channels * 2, hidden, 3, padding=1),
            nn.SiLU()
        )

        self.resblocks = nn.Sequential(
            *[ResidualBlock(hidden) for _ in range(num_blocks)]
        )

        self.output_proj = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, prev_latents, current_latents):

        x = torch.cat(
            [prev_latents, current_latents],
            dim=1
        )

        x = self.input_proj(x)

        x = self.resblocks(x)

        delta = self.output_proj(x)

        # residual prediction
        pred_next = current_latents + delta

        return pred_next


# =========================================================
# LOSS
# =========================================================

class TemporalLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(self, pred, target):

        return self.l1(pred, target)


# =========================================================
# TRAIN STEP
# =========================================================

def temporal_train_step(
    model,
    optimizer,
    criterion,
    prev_latents,
    current_latents,
    next_latents,
    device=device,
    debug=False
):

    model.train()

    prev_latents = sanitize_latents(prev_latents).to(device)
    current_latents = sanitize_latents(current_latents).to(device)
    next_latents = sanitize_latents(next_latents).to(device)

    if debug:
        print_latents_stats("prev", prev_latents)
        print_latents_stats("current", current_latents)
        print_latents_stats("next", next_latents)

    optimizer.zero_grad()

    pred_next = model(
        prev_latents,
        current_latents
    )

    loss = criterion(
        pred_next,
        next_latents
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    return pred_next.detach(), loss.item()


# =========================================================
# EVAL STEP
# =========================================================

@torch.no_grad()
def temporal_eval_step(
    model,
    prev_latents,
    current_latents,
    device=device
):

    model.eval()

    prev_latents = sanitize_latents(prev_latents).to(device)
    current_latents = sanitize_latents(current_latents).to(device)

    pred_next = model(
        prev_latents,
        current_latents
    )

    return pred_next


# =========================================================
# TRAIN LOOP
# =========================================================

def train_model(
    model,
    optimizer,
    criterion,
    temporal_dataset,
    epochs=10,
    device=device,
    save_every=1
):

    for epoch in range(epochs):

        epoch_loss = 0.0

        for step, batch in enumerate(temporal_dataset):

            prev_latents = batch["prev"]
            current_latents = batch["current"]
            next_latents = batch["next"]

            _, loss = temporal_train_step(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                prev_latents=prev_latents,
                current_latents=current_latents,
                next_latents=next_latents,
                device=device
            )

            epoch_loss += loss

            print(
                f"[Epoch {epoch+1}] "
                f"[Step {step+1}] "
                f"Loss={loss:.6f}"
            )

        avg_loss = epoch_loss / len(temporal_dataset)

        print(
            f"\n[Epoch {epoch+1}] "
            f"Average Loss={avg_loss:.6f}\n"
        )

        if (epoch + 1) % save_every == 0:

            save_temporal_model(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=avg_loss
            )


# =========================================================
# FAKE TEMPORAL DATASET
# =========================================================

def create_fake_temporal_dataset(
    num_samples=32,
    shape=(1, 4, 160, 112)
):

    dataset = []

    for _ in range(num_samples):

        prev = torch.randn(shape)

        current = prev + torch.randn(shape) * 0.05

        next_frame = current + torch.randn(shape) * 0.05

        dataset.append({
            "prev": prev,
            "current": current,
            "next": next_frame
        })

    return dataset


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    model = TemporalResidualNet(
        channels=4,
        hidden=64,
        num_blocks=4
    ).to(device)

    model.apply(weights_init)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    criterion = TemporalLoss()

    temporal_dataset = create_fake_temporal_dataset()

    train_model(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        temporal_dataset=temporal_dataset,
        epochs=10,
        device=device
    )

    print("[DONE] Training completed.")
