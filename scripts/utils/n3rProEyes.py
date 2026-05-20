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

def save_eyes_model(model, optimizer=None, epoch=None, loss=None,
                         path="models/eyes.pt", latest_path="models/eyes_latest.pt"):
    """
    Sauvegarde le modèle eyes en version stable et dernière version.
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

def load_eyes_model(model_class, path="models/eyes_latest.pt", optimizer=None):
    """
    Charge le dernier modèle eyes.
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




def show_latents(latents, decoded_latents, epoch):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(latents[0, 0].cpu().detach().numpy(), cmap='gray')
    axes[0].set_title(f"Original Latents - Epoch {epoch}")
    axes[1].imshow(decoded_latents[0, 0].cpu().detach().numpy(), cmap='gray')
    axes[1].set_title(f"Decoded Latents - Epoch {epoch}")
    plt.show()

# ----------------------
# Classe du modèle Eyes
# ----------------------
class EyeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(pred, target)

class EyeRefiner(nn.Module):
    def __init__(self, in_channels=4, base=32):
        super().__init__()

        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, 1, 1),
            nn.GroupNorm(8, base),
            nn.SiLU()
        )

        self.e2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, 2, 1),
            nn.GroupNorm(8, base*2),
            nn.SiLU()
        )

        self.e3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, 2, 1),
            nn.GroupNorm(8, base*4),
            nn.SiLU()
        )

        self.mid = nn.Sequential(
            nn.Conv2d(base*4, base*4, 3, 1, 1),
            nn.GroupNorm(8, base*4),
            nn.SiLU()
        )

        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, 4, 2, 1),
            nn.GroupNorm(8, base*2),
            nn.SiLU()
        )

        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base*2, base, 4, 2, 1),
            nn.GroupNorm(8, base),
            nn.SiLU()
        )

        self.out = nn.Conv2d(base, in_channels, 3, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        m = self.mid(e3)

        d2 = self.d2(m) + e2
        d1 = self.d1(d2) + e1

        return x + self.out(d1)


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
eyes_model = EyesUNet().to(device)
eyes_model.apply(weights_init)

# Optimiseur et fonction de perte
#optimizer = optim.Adam(eyes_model.parameters(), lr=1e-5, weight_decay=1e-5) # apprentissage faible
#optimizer = optim.Adam(eyes_model.parameters(), lr=1e-4, weight_decay=1e-5) # apprentissage moyen
#optimizer = optim.Adam(eyes_model.parameters(), lr=1e-3, weight_decay=0) # apprentissage elevé
#optimizer = optim.Adam(eyes_model.parameters(), lr=1e-3, weight_decay=1e-4)
#optimizer = optim.Adam(eyes_model.parameters(), lr=1e-4, weight_decay=1e-5)
#criterion = nn.MSELoss()  # Fonction de perte pour la reconstruction d'image
criterion = nn.L1Loss()  # Essayer la L1Loss
#criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss()  # Fonction de perte alternative

optimizer = optim.Adam(eyes_model.parameters(), lr=1e-4, weight_decay=1e-5)
#criterion = nn.MSELoss()  # ou SmoothL1Loss si tu veux plus de robustesse

# Fonction de débruitage et d'entraînement intégré avec contrôle de `Loss requires_grad`
def sanitize_latents_for_train_grad(latents, debug=True):
    # Normalisation simple
    return latents.clamp(-1.0, 1.0)  # renvoie un tensor PyTorch, non détaché

def eyes_latents_vao_load(latents, eyes_model, optimizer=None, criterion=None, device="cuda", train=True):
    """
    Eyes latents avec entraînement possible, training-safe même si le VAE est offloadé.
    """
    import torch

    if latents is None:
        raise ValueError("Latents is None, cannot proceed with eyes.")

    # Normaliser les latents
    latents = latents.clamp(-1.0, 1.0)

    # Mettre latents sur le device
    latents = latents.to(device=device, dtype=torch.float32)

    if train:
        # Mode entraînement
        eyes_model.train()
        # Forcer les paramètres à require_grad
        for param in eyes_model.parameters():
            param.requires_grad_(True)

        optimizer.zero_grad()

        # Forward avec graph pour grad_fn
        decoded_latents = eyes_model(latents)
        if criterion is not None:
            loss = criterion(decoded_latents, latents)
        else:
            # Loss fictive pour éviter None
            loss = torch.mean(decoded_latents ** 2)

        # Backward et update
        loss.backward()
        if optimizer is not None:
            optimizer.step()

        print(f"[EyesHDR TRAIN] Latents max: {latents.max():.4f}, min: {latents.min():.4f}")
        print(f"[EyesHDR TRAIN] Decoded max: {decoded_latents.max():.4f}, min: {decoded_latents.min():.4f}")
        print(f"[EyesHDR TRAIN] Loss: {loss.item():.4f}")

        return decoded_latents, loss.item()

    else:
        # Mode évaluation, no grad
        eyes_model.eval()
        with torch.no_grad():
            decoded_latents = eyes_model(latents)

        print(f"[EyesHDR EVAL] Latents max: {latents.max():.4f}, min: {latents.min():.4f}")
        print(f"[EyesHDR EVAL] Decoded max: {decoded_latents.max():.4f}, min: {decoded_latents.min():.4f}")

        return decoded_latents, None

def extract_eye_patch(latents, x, y, size=32):
    B, C, H, W = latents.shape

    x1 = max(0, x - size)
    x2 = min(W, x + size)
    y1 = max(0, y - size)
    y2 = min(H, y + size)

    return latents[:, :, y1:y2, x1:x2], (x1, y1, x2, y2)

def eyes_latents(latents, eyes_model, optimizer=None, criterion=None, device="cuda", train=True, debug=True):
    """
    Eyes latents de façon training-safe sans backward, compatible --vae-offload.
    Si train=True, le modèle peut s'adapter progressivement via une mise à jour heuristique.
    """

    if latents is None:
        raise ValueError("Latents is None, cannot proceed.")

    # Normalisation simple des latents
    print_latents(latents, debug=True)
    latents = (latents - latents.mean()) / (latents.std() + 1e-5)
    latents = latents.clamp(-1.0, 1.0)
    print_latents(latents, debug=True)

    # Utilisation avant de passer au modèle ou dans le processus d'entraînement

    latents = sanitize_latents_for_train_grad(latents, debug=True)


    # Forcer le modèle et les latents sur le même device
    latents = latents.to(device=device, dtype=torch.float32)
    eyes_model = eyes_model.to(device=device)

    # Mode entraînement ou évaluation
    eyes_model.train() if train else eyes_model.eval()

    # Pas de require_grad sur les latents
    latents.requires_grad_(False)

    # Décodage sans graphe pour éviter les erreurs de grad_fn
    with torch.no_grad():
        decoded_latents = eyes_model(latents)

    # Calcul de la perte juste pour suivi (si criterion fourni)
    loss_val = None
    if criterion is not None:
        loss_val = criterion(decoded_latents, latents)
        if debug:
            print(f"[EyesHDR] Latents max: {latents.max():.4f}, min: {latents.min():.4f}")
            print(f"[EyesHDR] Decoded max: {decoded_latents.max():.4f}, min: {decoded_latents.min():.4f}")
            #print(f"[EyesHDR] Loss: {loss_val.item():.4f}")

    # Mise à jour heuristique des paramètres si training
    if train and optimizer is not None:
        # Exemple de mise à jour simple : move légèrement chaque param vers zéro
        for param in eyes_model.parameters():
            if param.grad is not None:
                param.grad.zero_()
            # update léger, proportionnel à paramètre (pas de vrai gradient)
            param.data -= 1e-4 * param.data.sign()
        optimizer.step()

    return decoded_latents, loss_val.item() if loss_val is not None else None


# Fonction principale pour entraîner et tester avec plus de contrôle
def train_step(model, optimizer, criterion, x):
    model.train()

    x = x.to(next(model.parameters()).device)

    pred = model(x)
    loss = criterion(pred, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train_model(num_epochs, latents_train, eyes_model, optimizer, criterion, device="cuda"):
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
        decoded_latents, loss = eyes_latents(latents_train, eyes_model, optimizer, criterion, device, train=True)

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
    train_model(num_epochs=10, latents_train=latents_train, eyes_model=eyes_model, optimizer=optimizer, criterion=criterion, device=device)
