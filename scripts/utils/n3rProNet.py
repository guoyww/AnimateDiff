import torch
import torch.nn as nn
import torch.nn.functional as F

class N3RProNet(nn.Module):
    def __init__(self, channels=4):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, channels, 1)

        self.act = nn.SiLU()

    def forward(self, x):
        residual = x

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x + residual
