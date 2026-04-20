# unet_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv2d -> BatchNorm -> ReLU) * 2
    Added BatchNorm for stability and smoothness.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Added Batch Norm
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Added Batch Norm
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128)):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(features[2], features[2] * 2)

        # Decoder 
        # REPLACED ConvTranspose2d with Upsample + Conv to fix checkerboard artifacts
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[2] * 2, features[2], kernel_size=1)
        )
        self.dec3 = DoubleConv(features[2] * 2, features[2])

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[2], features[1], kernel_size=1)
        )
        self.dec2 = DoubleConv(features[1] * 2, features[1])

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[1], features[0], kernel_size=1)
        )
        self.dec1 = DoubleConv(features[0] * 2, features[0])

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        
        # Bottleneck
        x4 = self.bottleneck(self.pool3(x3))

        # ----- Decoder -----
        # Level 3
        up_x = self.up3(x4)
        # Handle padding/odd shapes: resize up_x to match x3 exactly
        if up_x.shape != x3.shape:
             up_x = F.interpolate(up_x, size=x3.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x3, up_x], dim=1)
        x = self.dec3(x)

        # Level 2
        up_x = self.up2(x)
        if up_x.shape != x2.shape:
             up_x = F.interpolate(up_x, size=x2.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x2, up_x], dim=1)
        x = self.dec2(x)

        # Level 1
        up_x = self.up1(x)
        if up_x.shape != x1.shape:
             up_x = F.interpolate(up_x, size=x1.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x1, up_x], dim=1)
        x = self.dec1(x)

        return self.final_conv(x)