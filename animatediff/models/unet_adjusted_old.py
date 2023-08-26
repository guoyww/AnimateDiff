
import torch.nn as nn
class UNetAdjusted(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetAdjusted, self).__init__()

        # Downsampling
        self.down1 = self.conv_block(in_channels, 64, stride=2)
        self.down2 = self.conv_block(64, 128, stride=2)
        self.down3 = self.conv_block(128, 256, stride=2)
        self.down4 = self.conv_block(256, 512, stride=2)
        self.down5 = self.conv_block(512, 1024, stride=2)
        
        # Upsampling
        self.up5 = self.conv_trans_block(1024, 512)
        self.up4 = self.conv_trans_block(512, 256)
        self.up3 = self.conv_trans_block(256, 128)
        self.up2 = self.conv_trans_block(128, 64)
        self.up1 = self.conv_trans_block(64, out_channels, final_layer=True)

    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        
        # Upsampling with skip connections
        u5 = self.up5(d5)
        u4 = self.up4(u5 + d4)  # skip connection
        u3 = self.up3(u4 + d3)  # skip connection
        u2 = self.up2(u3 + d2)  # skip connection
        u1 = self.up1(u2 + d1)  # skip connection
        
        return u1

    def conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def conv_trans_block(self, in_channels, out_channels, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)


