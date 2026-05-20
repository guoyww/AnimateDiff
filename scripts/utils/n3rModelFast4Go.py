
import torch
import torch.nn as nn
import torch.nn.functional as F

class N3RModelOptimized(nn.Module):
    def __init__(self, L_low=3, L_high=6, N_samples=6, tile_size=64, cpu_offload=True):
        super().__init__()
        self.L_low = L_low
        self.L_high = L_high
        self.N_samples = N_samples
        self.tile_size = tile_size
        self.cpu_offload = cpu_offload

        input_dim = 3 + 2 * 3 * L_high  # Max freq
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # RGB + density
        ).half()  # FP16

    def positional_encoding(self, x):
        """
        Coord [-1,1], mix L_low et L_high pour un rendu plus naturel
        """
        x = x.half()
        enc = [x]
        # basse fréquence (structure)
        for i in range(self.L_low):
            for fn in [torch.sin, torch.cos]:
                enc.append(fn((2.0 ** i) * x))
        # haute fréquence (détails)
        for i in range(self.L_low, self.L_high):
            for fn in [torch.sin, torch.cos]:
                enc.append(fn((2.0 ** i) * x) * 0.2)  # réduire l'amplitude pour limiter le néon
        return torch.cat(enc, dim=-1)

    def normalize_coords(self, coords, H, W):
        coords = coords.clone()
        coords[:, 0] = coords[:, 0] / (W-1) * 2 - 1
        coords[:, 1] = coords[:, 1] / (H-1) * 2 - 1
        return coords

    def forward_tile(self, coords_tile):
        mlp_device = next(self.mlp.parameters()).device
        x_enc = self.positional_encoding(coords_tile.to(mlp_device))
        out = self.mlp(x_enc)
        return out

    def forward(self, coords, H, W):
        """
        coords : tensor (H*W*N_samples, 3) sur le device final (cuda)
        H, W   : dimensions de l'image
        """
        device = coords.device
        ts = self.tile_size

        # Choix du device de sortie
        output_device = torch.device("cpu") if self.cpu_offload else device
        output = torch.zeros((H*W*self.N_samples, 4), device=output_device, dtype=torch.float16)

        for y in range(0, H, ts):
            for x in range(0, W, ts):
                y_end = min(y + ts, H)
                x_end = min(x + ts, W)

                # meshgrid vectorisé
                y_range = torch.arange(y, y_end, device=device)
                x_range = torch.arange(x, x_end, device=device)
                s_range = torch.arange(self.N_samples, device=device)
                yy, xx, s = torch.meshgrid(y_range, x_range, s_range, indexing='ij')
                idx_tile = (yy * W * self.N_samples + xx * self.N_samples + s).reshape(-1)

                coords_tile = coords[idx_tile]

                # forward tile sur CPU si offload
                mlp_device = torch.device("cpu") if self.cpu_offload else device
                output_tile = self.forward_tile(coords_tile.to(mlp_device))

                # assignation sur le même device que output
                output[idx_tile.to(output_device)] = output_tile.to(output_device)

        # remettre sur le device final si CPU offload
        if self.cpu_offload:
            output = output.to(device)

        return output


# -------------------------
# n3rModelLazyCPU.py
# -------------------------

class N3RModelLazyCPU(nn.Module):
    def __init__(self, L=6, N_samples=6, tile_size=64, cpu_offload=True):
        """
        L          : nombre de fréquences pour positional encoding
        N_samples  : échantillons par pixel
        tile_size  : taille de chaque tile
        cpu_offload: True pour décharger le MLP sur CPU
        """
        super().__init__()
        self.L = L
        self.N_samples = N_samples
        self.tile_size = tile_size
        self.cpu_offload = cpu_offload

        input_dim = 3 + 2 * 3 * self.L
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # RGB + density
        ).half()  # FP16

        if self.cpu_offload:
            self.mlp = self.mlp.to("cpu")

    def positional_encoding(self, x):
        x = x.half()
        enc = [x]
        for i in range(self.L):
            for fn in [torch.sin, torch.cos]:
                enc.append(fn((2.0 ** i) * x))
        return torch.cat(enc, dim=-1)

    def forward_tile(self, coords_tile):
        """
        Forward d'un tile avec CPU-offload safe
        """
        device_orig = coords_tile.device

        # On envoie tout sur le même device que le MLP
        mlp_device = next(self.mlp.parameters()).device
        coords_tile = coords_tile.to(mlp_device).half()

        x_enc = self.positional_encoding(coords_tile)
        out = self.mlp(x_enc)

        # Si MLP est sur CPU, remettre le output sur le device original (GPU)
        if mlp_device.type == "cpu" and device_orig.type != "cpu":
            out = out.to(device_orig)

        return out

    def forward(self, coords, H, W):
        """
        coords : (H*W*N_samples, 3)
        """
        device = coords.device
        output = torch.zeros((H*W*self.N_samples, 4), device=device, dtype=torch.float16)

        ts = self.tile_size

        for y in range(0, H, ts):
            for x in range(0, W, ts):
                y_end = min(y + ts, H)
                x_end = min(x + ts, W)

                # Meshgrid vectorisé pour tile
                y_range = torch.arange(y, y_end, device=device)
                x_range = torch.arange(x, x_end, device=device)
                s_range = torch.arange(self.N_samples, device=device)
                yy, xx, ss = torch.meshgrid(y_range, x_range, s_range, indexing='ij')
                idx_tile = (yy * W * self.N_samples + xx * self.N_samples + ss).reshape(-1)

                coords_tile = coords[idx_tile]
                output_tile = self.forward_tile(coords_tile)

                output[idx_tile] = output_tile

        return output

class N3RModelFast4GB(nn.Module):
    def __init__(self, L=6, N_samples=16, tile_size=64, cpu_offload=False):
        """
        L         : nombre de fréquences pour le positional encoding
        N_samples : échantillons par pixel
        tile_size : taille des tiles pour le forward
        cpu_offload: True pour décharger les tiles sur CPU pour VRAM limitée
        """
        super().__init__()
        self.L = L
        self.N_samples = N_samples
        self.tile_size = tile_size
        self.cpu_offload = cpu_offload

        # Dimensions d'entrée : 3 coords + 2*3*L (sin/cos)
        input_dim = 3 + 2 * 3 * self.L
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # RGB + density
        ).half()  # FP16 partout

    def positional_encoding(self, x):
        """Encode input coordinates avec sin/cos."""
        x = x.half()
        enc = [x]
        for i in range(self.L):
            factor = 2.0 ** i
            enc.append(torch.sin(factor * x))
            enc.append(torch.cos(factor * x))
        return torch.cat(enc, dim=-1)

    def forward_tile(self, coords):
        """Forward pass d’un tile avec autocast FP16."""
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            x_enc = self.positional_encoding(coords)
            out = self.mlp(x_enc)
        return out

    def forward(self, coords, H, W):
        """
        coords : tensor (H*W*N_samples, 3)
        H, W   : dimensions de l'image
        """
        device = coords.device
        output = torch.zeros((H*W*self.N_samples, 4), device=device, dtype=torch.float16)

        ts = self.tile_size
        for y in range(0, H, ts):
            for x in range(0, W, ts):
                y_end = min(y + ts, H)
                x_end = min(x + ts, W)

                # Meshgrid vectorisé
                yy, xx = torch.meshgrid(
                    torch.arange(y, y_end, device=device),
                    torch.arange(x, x_end, device=device),
                    indexing='ij'
                )
                idx_tile = (yy[..., None] * W * self.N_samples + xx[..., None] * self.N_samples +
                            torch.arange(self.N_samples, device=device)).reshape(-1)

                coords_tile = coords[idx_tile]
                output_tile = self.forward_tile(coords_tile)

                # CPU offload sélectif
                if self.cpu_offload:
                    output[idx_tile.cpu()] = output_tile.cpu()
                else:
                    output[idx_tile] = output_tile

        return output
