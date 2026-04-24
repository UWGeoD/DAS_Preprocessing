"""
UNetV2 — paper-inspired architecture for DAS denoising.

Key additions over UNet:
  - 4 encoder levels (32→64→128→256, bottleneck 512) for larger receptive field
  - SPP (Spatial Pyramid Pooling) at bottleneck: captures multi-scale temporal context
    (where in the window is the vehicle event?)
  - Channel attention on every skip connection: forces the model to locate sparse
    vehicle signal rather than memorize per-sample noise patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ChannelAttention(nn.Module):
    """
    SE-Net style channel attention applied to encoder skip connections.

    Squeezes spatial dims → learns channel importance weights → scales the skip.
    Forces the model to selectively pass through channels that carry vehicle
    signal, suppressing channels dominated by noise.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SPPBottleneck(nn.Module):
    """
    Spatial Pyramid Pooling adapted for DAS data.

    DAS bottleneck feature maps are ~(2, 625) — very wide, very short.
    We pool along the time axis only (height=1) to avoid collapsing a
    spatial dimension that is already small.

    Scales (1,1), (1,4), (1,8) capture:
      - Global context: is there any vehicle in this window at all?
      - 4-segment: roughly which quarter of the window has the event?
      - 8-segment: finer temporal localization

    Each scale is upsampled back and concatenated with the original, then
    projected to the original channel count.
    """
    def __init__(self, channels):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, s)) for s in [1, 4, 8]
        ])
        # original + 3 pooled branches → 4 × channels → channels
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        branches = [x]
        for pool in self.pools:
            branches.append(
                F.interpolate(pool(x), size=(h, w), mode='bilinear', align_corners=True)
            )
        return self.proj(torch.cat(branches, dim=1))


class UNetV2(nn.Module):
    """
    4-level UNet with SPP bottleneck and channel attention on skip connections.

    Encoder:  in → 32 → 64 → 128 → 256, each followed by MaxPool2d(2)
    Bottleneck: 256 → 512 → SPP(512)
    Decoder:  attention(skip) + upsample + concat + DoubleConv, ×4
    Output:   Conv(32 → out_channels, 1×1)
    """
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()

        # Encoder
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc.append(DoubleConv(ch, f))
            self.pool.append(nn.MaxPool2d(2))
            ch = f

        # Bottleneck
        bot_ch = features[-1] * 2   # 512
        self.bottleneck = DoubleConv(features[-1], bot_ch)
        self.spp        = SPPBottleneck(bot_ch)
        self.drop       = nn.Dropout2d(p=0.2)

        # Decoder
        self.up   = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.dec  = nn.ModuleList()

        ch = bot_ch
        for f in reversed(features):
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch, f, kernel_size=1, bias=False),
            ))
            self.attn.append(ChannelAttention(f))
            self.dec.append(DoubleConv(f * 2, f))
            ch = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        for enc, pool in zip(self.enc, self.pool):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.drop(self.spp(self.bottleneck(x)))

        # Decoder
        for up, attn, dec, skip in zip(self.up, self.attn, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            skip = attn(skip)
            x = dec(torch.cat([skip, x], dim=1))

        return self.final_conv(x)
