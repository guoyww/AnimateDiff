import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import traceback
from .tools_utils import sanitize_latents_for_train
import matplotlib.pyplot as plt

from torch.optim import Adam
import os
import datetime

def save_denoising_model(model, optimizer=None, epoch=None, loss=None,
                         path="models/denoise.pt", latest_path="models/denoise_latest.pt"):
    """
    Sauvegarde le modèle denoising en version stable et dernière version.
    Ajoute le timestamp directement dans le checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": getattr(model, "config", None),
        "timestamp": datetime.datetime.now().isoformat()
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if loss is not None:
        checkpoint["loss"] = loss

    # Sauvegarde principale (stable)
    torch.save(checkpoint, path)

    # Sauvegarde du dernier checkpoint
    torch.save(checkpoint, latest_path)

    print(f"[INFO] Model saved to {path} (latest: {latest_path}), timestamp: {checkpoint['timestamp']}")

def load_denoising_model(model_class, path="models/denoise_latest.pt", optimizer=None):
    """
    Charge le dernier modèle denoising.
    Si aucun checkpoint n'existe, renvoie un modèle non entraîné.
    """
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location="cuda")
        config = checkpoint.get("model_config") or {}
        model = model_class(**config) if config else model_class()
        model.load_state_dict(checkpoint["model_state"])

        if optimizer is not None and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"[INFO] Loaded latest model, timestamp: {checkpoint.get('timestamp')}, loss: {checkpoint.get('loss')}")
        return model, checkpoint
    else:
        print("[WARN] No latest model found, using untrained model.")
        return model_class(), None


def save_denoising_model_old(model, optimizer=None, epoch=None, loss=None, path="models/denoise.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if loss is not None:
        checkpoint["loss"] = loss

    torch.save(checkpoint, path)
    print(f"[INFO] Denoising model saved to {path}")


def load_denoising_model_old(model_class, path="models/denoise.pt", optimizer=None, device="cuda"):
    checkpoint = torch.load(path, map_location=device)

    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # par défaut en mode évaluation

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    print(f"[INFO] Denoising model loaded from {path}")
    return model, checkpoint

def show_latents(latents, decoded_latents, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(latents[0, 0].cpu().detach().numpy(), cmap='gray')
    axes[0].set_title(f"Original Latents - Epoch {epoch}")
    axes[1].imshow(decoded_latents[0, 0].cpu().detach().numpy(), cmap='gray')
    axes[1].set_title(f"Decoded Latents - Epoch {epoch}")
    plt.show()

# Classe du modèle DenoisingAutoencoder

# ----------------------
# U-Net léger pour débruitage
# ----------------------
class DenoiseUNet(nn.Module):
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()

        # Encodeur
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Décodeur
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, in_channels, 3, padding=1),
            nn.Tanh()  # pour limiter la sortie entre -1 et 1
        )

    def forward(self, x):
        # Encodeur
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Décodeur avec skip connections
        d3 = self.dec3(b)
        d3 = torch.cat([d3, e2], dim=1)  # skip connection

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)  # skip connection

        out = self.dec1(d2)
        return out
# Modèle simple
class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class SimpleAE_Optimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # limite la sortie à [-1,1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class SimpleAE32(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodeur : légère montée en filtres pour block_size=32
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # downscale par 2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # downscale par 2
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # garde spatial size
            nn.ReLU(),
        )

        # Décodeur : remonte à la taille d'origine
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SimpleAEIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        # Encodeur avec plus de canaux et stride pour downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # [H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1), # [H/4, W/4]
            nn.ReLU()
        )
        # Décodeur symétrique avec ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [H/2, W/2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),   # [H, W]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class IntermediateAE(nn.Module):
    def __init__(self):
        super().__init__()

        # Encodeur : légèrement plus large que SimpleAE mais pas trop profond
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # downsample x2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # downsample x2
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),  # garder la résolution
            nn.ReLU()
        )

        # Décodeur : symétrique à l'encodeur pour restaurer amplitude
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),  # résolution inchangée
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # upsample x2
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),  # upsample x2
            nn.Tanh()  # contraint les sorties à [-1,1], ce qui améliore le Loss
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DenoisingAE32(nn.Module):
    def __init__(self):
        super(DenoisingAE32, self).__init__()

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),  # 4→32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32→64
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 64→128
            nn.ReLU(),
        )

        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Normalisation des sorties entre -1 et 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def print_latents(latents, debug=True):
    if debug:
        print(f"Latents max: {latents.max():.4f}, min: {latents.min():.4f}, Latents mean: {latents.mean():.4f}, std: {latents.std():.4f}")
        if torch.isnan(latents).any():
            print("WARNING: NaN detected in latents")
        if torch.isinf(latents).any():
            print("WARNING: Inf detected in latents")

# Initialisation des poids
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

# Créer le modèle de débruitage avec 4 canaux en entrée
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# *****************************************************************************************************************************
denoising_model = DenoiseUNet().to(device)
denoising_model.apply(weights_init)

# Optimiseur et fonction de perte
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-5, weight_decay=1e-5) # apprentissage faible
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4, weight_decay=1e-5) # apprentissage moyen
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-3, weight_decay=0) # apprentissage elevé
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-3, weight_decay=1e-4)
#optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4, weight_decay=1e-5)
#criterion = nn.MSELoss()  # Fonction de perte pour la reconstruction d'image
criterion = nn.L1Loss()  # Essayer la L1Loss
#criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss()  # Fonction de perte alternative

optimizer = optim.Adam(denoising_model.parameters(), lr=1e-4, weight_decay=1e-5)
#criterion = nn.MSELoss()  # ou SmoothL1Loss si tu veux plus de robustesse

# Fonction de débruitage et d'entraînement intégré avec contrôle de `Loss requires_grad`
def sanitize_latents_for_train_grad(latents, debug=True):
    # Normalisation simple
    return latents.clamp(-1.0, 1.0)  # renvoie un tensor PyTorch, non détaché

def train_denoiser(
    latents,
    model,
    optimizer,
    criterion,
    max_epochs,
    device,
    debug=False
):

    model.train().to(device)

    latents_train = latents.to(device)

    final_loss = None
    final_pred = None

    for epoch in range(max_epochs):

        optimizer.zero_grad(set_to_none=True)

        with torch.enable_grad():

            pred, loss = denoise_latents(
                latents_train,
                model,
                optimizer=None,
                criterion=criterion,
                device=device,
                train=True
            )

            if loss is None:
                print(f"Epoch [{epoch+1}/{max_epochs}] Loss: None")
                continue

            if not torch.isfinite(loss):
                print("[WARN] Non finite loss")
                continue

            final_loss = loss
            final_pred = pred

            print(f"Epoch {epoch+1}: {loss.item():.4f}")

            if debug:
                show_latents(latents_train, pred, epoch+1)

            loss.backward()

            optimizer.step()

    return model, final_pred, final_loss

def train_denoiser_v1(latents, model, optimizer, criterion, max_epochs, device, debug=False):

    model.train().to(device)
    latents = latents.to(device)

    for epoch in range(max_epochs):

        optimizer.zero_grad(set_to_none=True)

        pred = model(latents)

        loss = criterion(pred, latents)

        if not torch.isfinite(loss):
            print("[WARN] invalid loss")
            continue

        print(f"Epoch {epoch+1}: {loss.item():.4f}")

        if debug:
            show_latents(latents, pred, epoch)

        loss.backward()
        optimizer.step()

    return model, loss


def debug_tensor(name, tensor, debug=False):
    if debug:
        if tensor is None:
            print(f"[DEBUG] {name}: None")
            return

        print(
            f"[DEBUG] {name} | "
            f"shape={tuple(tensor.shape)} | "
            f"dtype={tensor.dtype} | "
            f"device={tensor.device} | "
            f"requires_grad={tensor.requires_grad} | "
            f"is_leaf={tensor.is_leaf} | "
            f"grad_fn={tensor.grad_fn}"
        )

def denoise_latents(
    latents,
    denoising_model,
    optimizer=None,
    criterion=None,
    device="cuda",
    train=True,
    debug=True
):

    if latents is None:
        raise ValueError("Latents is None")

    debug_tensor("latents_input_before", latents)

    latents = (latents - latents.mean()) / (latents.std() + 1e-5)
    latents = latents.clamp(-1.0, 1.0)

    debug_tensor("latents_after_norm", latents)

    latents = sanitize_latents_for_train_grad(latents, debug=True)

    debug_tensor("latents_after_sanitize", latents)

    latents = latents.to(device=device, dtype=torch.float32)

    debug_tensor("latents_after_to", latents)

    denoising_model = denoising_model.to(device)

    if train:
        denoising_model.train()
    else:
        denoising_model.eval()

    # DEBUG paramètres modèle
    first_param = next(denoising_model.parameters())

    print(
        f"[DEBUG MODEL] requires_grad={first_param.requires_grad} | "
        f"is_leaf={first_param.is_leaf} | "
        f"grad_fn={first_param.grad_fn}"
    )

    decoded_latents = denoising_model(latents)

    debug_tensor("decoded_latents", decoded_latents)

    loss_val = None

    if criterion is not None:
        loss_val = criterion(decoded_latents, latents)

        debug_tensor("loss_val", loss_val)

        if debug:
            print(f"[DENoise] Loss: {loss_val.item():.6f}")

    return decoded_latents, loss_val


def denoise_latents_v1(latents, denoising_model, optimizer=None, criterion=None, device="cuda", train=True, debug=True):

    if latents is None:
        raise ValueError("Latents is None")

    print_latents(latents, debug=True)

    latents = (latents - latents.mean()) / (latents.std() + 1e-5)
    latents = latents.clamp(-1.0, 1.0)

    latents = sanitize_latents_for_train_grad(latents, debug=True)

    latents = latents.to(device=device, dtype=torch.float32)
    denoising_model = denoising_model.to(device)

    if train:
        denoising_model.train()
    else:
        denoising_model.eval()

    # ❌ SUPPRESSION CRITIQUE DU no_grad
    decoded_latents = denoising_model(latents)

    loss_val = None
    if criterion is not None:
        loss_val = criterion(decoded_latents, latents)

        if debug:
            print(f"[DENoise] Latents max: {latents.max():.4f}, min: {latents.min():.4f}")
            print(f"[DENoise] Decoded max: {decoded_latents.max():.4f}, min: {decoded_latents.min():.4f}")
            print(f"[DENoise] Loss: {loss_val.item():.4f}")

    # ⚠️ IMPORTANT : ne pas toucher gradients ici
    if train and optimizer is not None:
        pass  # training doit être dans train_denoiser()

    return decoded_latents, loss_val



# Fonction principale pour entraîner et tester avec plus de contrôle
def train_model(num_epochs, latents_train, denoising_model, optimizer, criterion, device="cuda"):
    """
    Fonction pour entraîner le modèle avec les latents et afficher les pertes.
    """
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Vérification de la validité des latents avant de procéder au débruitage
        if latents_train is None:
            print("Latents is None, skipping this epoch.")
            continue

        # Denoising des latents
        decoded_latents, loss = denoise_latents(latents_train, denoising_model, optimizer, criterion, device, train=True)

        # Affichage de la perte
        if loss is not None:
            print(f"Loss: {loss:.4f}")
        else:
            print("Loss was not computed due to an error.")

# Exemple d'utilisation avec des données aléatoires
if __name__ == "__main__":
    # Dimensions de latents (exemple)
    latents_train = torch.randn(1, 4, 160, 112)  # Exemple de tensor de latents (format [batch, channels, height, width])

    # Entraîner le modèle pendant 10 époques
    train_model(num_epochs=10, latents_train=latents_train, denoising_model=denoising_model, optimizer=optimizer, criterion=criterion, device=device)
